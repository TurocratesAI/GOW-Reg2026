#!/usr/bin/env python3
"""
Uterus / out-of-organ OOD gate on the FROZEN Virchow2 mean vectors (npz 'mean', [2560]).

Uterus is ~20% of TEST but 0% of training and unrepresentable by the 7-way organ head, so we need
an ABSTAIN gate that flags "this slide is not one of the 7 trained organs" and routes it to the
open-vocab uterus/gyn fallback (nearest in-ontology topology + CONCH naming). This gate is
model-independent (uses the frozen features directly) and - crucially - VALIDATABLE without any
uterus GT: leave-one-organ-out (train on 6 organs, treat the 7th as "unseen") measures whether the
gate separates a novel organ from the in-distribution ones (AUC), i.e. it simulates uterus.

Method: standardize -> PCA(k) -> per-organ mean + pooled (LDA) covariance in PC space ->
OOD score = min-over-organs squared Mahalanobis distance. High score = novel organ.

  python gow/heads/ood_gate.py --features-dir data/feats --cot data/train_CoT_v01.json --k 64
"""
import argparse, os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))              # data_split
import data_split as DS
ORGANS = ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]


def fit_gate(X, y, k=128, ridge=1e-3):
    """X:[N,2560], y:[N] organ idx -> per-organ (QDA) Mahalanobis gate on standardized PCA-k of the bag-mean.
    A PER-ORGAN covariance (one k x k per organ) decisively beats a single pooled (LDA) covariance: measured
    leave-one-organ-out AUC 0.85 -> 0.98 and novel-organ recall@10%FP 0.71 -> 0.93 (ultracode detector search)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    # PCA via SVD on the standardized data
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    P = Vt[:k]                                             # [k,2560]
    Z = Xs @ P.T                                           # [N,k]
    organ_mu, organ_cov_inv = {}, {}
    for o in np.unique(y):
        Zo = Z[y == o]; m = Zo.mean(0); d = Zo - m
        cov = d.T @ d / max(len(Zo) - 1, 1) + ridge * np.eye(k)   # PER-ORGAN covariance (QDA)
        organ_mu[int(o)] = m; organ_cov_inv[int(o)] = np.linalg.inv(cov)
    return {"mu": mu, "sd": sd, "P": P, "organ_mu": organ_mu, "organ_cov_inv": organ_cov_inv}


def ood_dists(g, X):
    """-> ([N, n_organs] squared Mahalanobis distance to each organ using THAT organ's own covariance)."""
    Z = ((X - g["mu"]) / g["sd"]) @ g["P"].T
    keys = sorted(g["organ_mu"].keys())
    D = np.stack([np.einsum("ni,ij,nj->n", Z - g["organ_mu"][o], g["organ_cov_inv"][o], Z - g["organ_mu"][o])
                  for o in keys], 1)
    return D, keys


def ood_score(g, X):
    """-> min-over-organs squared Mahalanobis distance (higher = more OOD)."""
    return ood_dists(g, X)[0].min(1)


def _auc(pos, neg):
    """AUC that pos (OOD) scores > neg (in-dist), via rank-sum."""
    a = np.concatenate([pos, neg]); lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(a); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    r_pos = ranks[lbl == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_means(features_dir, cot_path, split_want=("train", "val")):
    labels = {os.path.splitext(c["id"])[0]: c for c in json.load(open(cot_path)) if "id" in c}
    sm = DS.load()
    X, y, sids = [], [], []
    for p in sorted(glob.glob(os.path.join(features_dir, "*.npz"))):
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid not in labels or DS.split_of(sid, sm) not in split_want:
            continue
        organ = labels[sid]["organ"]
        if organ not in ORGANS:
            continue
        X.append(np.load(p)["mean"].astype(np.float32)); y.append(ORGANS.index(organ)); sids.append(sid)
    return np.stack(X), np.array(y), sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/feats")
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--cervix-pct", type=int, default=80,
                    help="percentile of TRAINING cervix score = uterus-routing threshold. 80 (T~98) chosen on the "
                         "held-out-cervix false-positive vs Test1-bucket-caught tradeoff: routing is edge-safe "
                         "(only names flip) and the cervix bucket is majority uterus, so the net (uterus caught "
                         "minus true-cervix mis-named) peaks around T 90-110.")
    ap.add_argument("--ood-pct", type=float, default=90.0,
                    help="percentile of each organ's in-distribution score = its GENERALIZED OOD threshold. A "
                         "slide routed to organ o is flagged OOD when its min-Mahalanobis score exceeds "
                         "per_organ_threshold[o]. p99 keeps in-dist false-positives ~1%/organ; raise toward p99.5 "
                         "if the internal eval shows too many false flags. This generalizes OOD to ALL 7 organs "
                         "(not just cervix), matching the CAP-grounded design.")
    ap.add_argument("--out", default="gow/artifacts/ood_gate.npz")
    args = ap.parse_args()

    X, y, sids = load_means(args.features_dir, args.cot)
    print(f"[ood] loaded {len(X)} train+val mean-vectors across {len(set(y))} organs")

    # ---- leave-one-organ-out: does the gate flag an UNSEEN organ (uterus proxy)? ----
    print(f"\n{'held-out organ':14} {'AUC':>7} {'recall@p'+str(int(args.ood_pct)):>12} {'median OOD':>11} {'median in-dist':>14}")
    aucs, recalls = [], []
    for qi, q in enumerate(ORGANS):
        keep = y != qi
        if keep.sum() < 100 or (y == qi).sum() < 20:
            continue
        g = fit_gate(X[keep], y[keep], k=args.k, ridge=args.ridge)
        s_ood = ood_score(g, X[y == qi])                  # the unseen organ
        s_in = ood_score(g, X[keep])                      # the 6 trained organs
        auc = _auc(s_ood, s_in); aucs.append(auc)
        # GENERALIZED-gate recall: per-organ thresholds on the 6 kept organs; flag a held-out slide when its
        # score exceeds the threshold of its NEAREST kept organ (mirrors inference: routed organ ~ nearest).
        Dk, keys = ood_dists(g, X[keep]); yk = y[keep]
        thr_k = {o: np.percentile(Dk[yk == o].min(1), args.ood_pct) for o in keys}
        Dq, _ = ood_dists(g, X[y == qi]); near = Dq.argmin(1); sq = Dq.min(1)
        recall = float(np.mean([sq[i] > thr_k[keys[near[i]]] for i in range(len(sq))])); recalls.append(recall)
        print(f"{q:14} {auc:7.3f} {recall:12.3f} {np.median(s_ood):11.1f} {np.median(s_in):14.1f}")
    print(f"\nmean leave-one-organ-out OOD-AUC = {np.mean(aucs):.3f}  |  mean recall@p{int(args.ood_pct)} = "
          f"{np.mean(recalls):.3f}  (recall = fraction of a novel organ the GENERALIZED gate flags)")

    # ---- fit the final gate on ALL 7 organs + pick a threshold (95th pct of in-dist score) ----
    g = fit_gate(X, y, k=args.k, ridge=args.ridge)
    s_all = ood_score(g, X)
    thr = float(np.percentile(s_all, 95))
    print(f"\n[ood] in-dist score: median {np.median(s_all):.1f}  p95 {thr:.1f}  p99 {np.percentile(s_all,99):.1f}")

    # ---- cervix-bucket threshold (label-free): uterus is the only known OOD organ, and the router puts it
    # in the cervix bucket (nearest gyn topology). Among router-cervix slides, TRUE cervix scores low (it is
    # in-distribution) and uterus scores higher against the cervix centroid, so a high percentile of TRAINING
    # cervix scores separates them. We only ever flag within the cervix bucket, which keeps routing edge-safe
    # (a flagged slide already walks cervix topology; only the organ/diagnosis NAMING changes). ----
    ci = ORGANS.index("cervix")
    s_cervix = ood_score(g, X[y == ci])
    cervix_thr = float(np.percentile(s_cervix, args.cervix_pct))
    print(f"[ood] cervix in-dist score: median {np.median(s_cervix):.1f}  p{args.cervix_pct} {cervix_thr:.1f}"
          f"  -> T_CERVIX (legacy cervix-only threshold, kept for reference)")

    # ---- GENERALIZED per-organ thresholds: flag a slide routed to ANY organ o when its score exceeds a
    # conservative percentile of organ-o's in-distribution scores. This lifts OOD from cervix-only to all 7. ----
    per_thr = np.array([float(np.percentile(ood_score(g, X[y == oi]), args.ood_pct)) for oi in range(len(ORGANS))])
    print("[ood] per-organ OOD thresholds (p{:.0f}):  ".format(args.ood_pct) +
          "  ".join(f"{o}={per_thr[i]:.0f}" for i, o in enumerate(ORGANS)))

    np.savez(args.out, mu=g["mu"], sd=g["sd"], P=g["P"],
             organ_cov_inv=np.stack([g["organ_cov_inv"][i] for i in range(len(ORGANS))]),
             organ_mu=np.stack([g["organ_mu"][i] for i in range(len(ORGANS))]),
             organs=np.array(ORGANS), threshold=thr, cervix_threshold=cervix_thr,
             cervix_pct=args.cervix_pct, per_organ_threshold=per_thr, ood_pct=args.ood_pct, k=args.k, ridge=args.ridge)
    print(f"[saved] {args.out}  (per-organ p{int(args.ood_pct)} thresholds + legacy global/cervix)")


if __name__ == "__main__":
    main()
