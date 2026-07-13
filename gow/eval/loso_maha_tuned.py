#!/usr/bin/env python3
"""
Tuned min-over-organs squared-Mahalanobis OOD detector, LOSO-evaluated.

Detector = current gate (standardize -> PCA-k -> per-organ centroids -> min squared Mahalanobis
of the bag-MEAN), TUNED. Sweep k in {16,32,64,128}, ridge in {1e-3,1e-2,1e-1}, and
covariance = single global pooled (LDA) vs per-organ (QDA). LOSO protocol per the task spec:
 for each held-out organ q, FIT on the OTHER 6 organs, score held-out-q as NOVEL(+) and the 6
 in-dist organs as NEG; AUC via rank-sum; recall at 5/10/15% in-dist false-flag budgets
 (threshold = (100-FP)th pct of in-dist scores); per-organ recall at the 10% budget.
"""
import os, sys, json, glob, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # data_split
sys.path.insert(0, os.path.join(HERE, "..", "heads"))   # ood_gate helpers
from ood_gate import load_means, _auc, ORGANS

FP_BUDGETS = [5.0, 10.0, 15.0]
KS = [16, 32, 64, 128]
RIDGES = [1e-3, 1e-2, 1e-1]
COVS = ["global", "per_organ"]


def fold_prep(Xtr, ytr, Xq, kmax=128):
    """Standardize on the 6 kept organs, PCA-kmax, project kept + novel. Returns projections and
    per-organ scatter matrices in the kmax PC space (slice columns to get any k<=kmax)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd
    cen = Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xs - cen, full_matrices=False)
    P = Vt[:kmax]                                        # [kmax, D]
    Ztr = (Xs - cen) @ P.T                               # NOTE: center consistent for both
    Zq = ((Xq - mu) / sd - cen) @ P.T
    organs = sorted(np.unique(ytr).tolist())
    organ_mu = {o: Ztr[ytr == o].mean(0) for o in organs}
    scat = {o: (Ztr[ytr == o] - organ_mu[o]).T @ (Ztr[ytr == o] - organ_mu[o]) for o in organs}
    ncls = {o: int((ytr == o).sum()) for o in organs}
    return Ztr, Zq, ytr, organ_mu, scat, ncls, organs


def maha_scores(Z, organ_mu, cov_inv, organs, k):
    """min-over-organs squared Mahalanobis in the first-k PC subspace. cov_inv is either one [k,k]
    (global) or a dict o->[k,k] (per-organ)."""
    Zk = Z[:, :k]
    ds = []
    for o in organs:
        d = Zk - organ_mu[o][:k]
        ci = cov_inv if not isinstance(cov_inv, dict) else cov_inv[o]
        ds.append(np.einsum("ni,ij,nj->n", d, ci, d))
    return np.stack(ds, 1).min(1)


def recall_at(s_novel, s_in, fp):
    thr = np.percentile(s_in, 100.0 - fp)
    return float(np.mean(s_novel > thr))


def main():
    X, y, sids = load_means("data/feats", "data/train_CoT_v01.json", split_want=("train", "val"))
    print(f"[loso] loaded {len(X)} train+val bag-means; per-organ counts: "
          + ", ".join(f"{ORGANS[o]}={int((y==o).sum())}" for o in range(len(ORGANS))))

    # results[config] = {"auc":[per fold], "r5":[], "r10":[], "r15":[], "per_organ_r10":{organ:val}}
    results = {}
    kmax = max(KS)
    for qi, q in enumerate(ORGANS):
        keep = y != qi
        Ztr, Zq, ytr, organ_mu, scat, ncls, organs = fold_prep(X[keep], y[keep], X[y == qi], kmax)
        Ntr = len(Ztr)
        for k, ridge, cov in itertools.product(KS, RIDGES, COVS):
            if cov == "global":
                within = np.zeros((k, k))
                for o in organs:
                    within += scat[o][:k, :k]
                C = within / max(Ntr - len(organs), 1) + ridge * np.eye(k)
                cov_inv = np.linalg.inv(C)
            else:  # per-organ QDA
                cov_inv = {}
                for o in organs:
                    C = scat[o][:k, :k] / max(ncls[o] - 1, 1) + ridge * np.eye(k)
                    cov_inv[o] = np.linalg.inv(C)
            s_novel = maha_scores(Zq, organ_mu, cov_inv, organs, k)
            s_in = maha_scores(Ztr, organ_mu, cov_inv, organs, k)
            key = (k, ridge, cov)
            r = results.setdefault(key, {"auc": [], "r5": [], "r10": [], "r15": [], "per_organ_r10": {}})
            r["auc"].append(_auc(s_novel, s_in))
            r["r5"].append(recall_at(s_novel, s_in, 5.0))
            r10 = recall_at(s_novel, s_in, 10.0)
            r["r10"].append(r10)
            r["r15"].append(recall_at(s_novel, s_in, 15.0))
            r["per_organ_r10"][q] = r10

    # rank configs by mean recall@10 (primary), tiebreak AUC
    rows = []
    for key, r in results.items():
        rows.append((key, float(np.mean(r["auc"])), float(np.mean(r["r5"])),
                     float(np.mean(r["r10"])), float(np.mean(r["r15"])), r["per_organ_r10"]))
    rows.sort(key=lambda t: (t[3], t[1]), reverse=True)

    print(f"\n{'k':>4} {'ridge':>7} {'cov':>10} {'AUC':>7} {'r@5':>7} {'r@10':>7} {'r@15':>7}")
    for key, auc, r5, r10, r15, _ in rows:
        k, ridge, cov = key
        print(f"{k:4d} {ridge:7.0e} {cov:>10} {auc:7.3f} {r5:7.3f} {r10:7.3f} {r15:7.3f}")

    best = rows[0]
    key, auc, r5, r10, r15, per_org = best
    k, ridge, cov = key
    print(f"\n[BEST by mean recall@10] k={k} ridge={ridge:g} cov={cov}")
    print(f"  AUC={auc:.4f}  recall@5={r5:.4f}  recall@10={r10:.4f}  recall@15={r15:.4f}")
    print("  per-organ recall@10 (held-out organ = NOVEL):")
    for o in ORGANS:
        print(f"    {o:10} {per_org[o]:.3f}")

    out = {"best": {"k": k, "ridge": ridge, "cov": cov, "auc": auc, "r5": r5, "r10": r10,
                    "r15": r15, "per_organ_r10": {o: per_org[o] for o in ORGANS}},
           "all": [{"k": kk, "ridge": rr, "cov": cc, "auc": a, "r5": x5, "r10": x10, "r15": x15}
                   for (kk, rr, cc), a, x5, x10, x15, _ in rows]}
    with open(os.path.join(HERE, "loso_maha_tuned_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[saved] {os.path.join(HERE, 'loso_maha_tuned_result.json')}")


if __name__ == "__main__":
    main()
