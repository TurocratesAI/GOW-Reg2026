#!/usr/bin/env python3
"""Robustness check for the BEST tuned config (k=128, ridge=1e-3, per-organ cov):
(1) cache bag-means to scratchpad npz for fast reruns,
(2) re-confirm the protocol (in-sample-threshold) LOSO numbers,
(3) HELD-OUT-in-dist variant: split each in-dist organ 70/30, fit centroids+cov on 70%, calibrate
    the FP threshold on the untouched 30% (out-of-sample) -> honest false-flag calibration.
Reports both so the optimism of the in-sample protocol is quantified."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "heads"))
from ood_gate import load_means, _auc, ORGANS

CACHE = os.path.join(HERE, "means_cache.npz")
K, RIDGE = 128, 1e-3
FPS = [5.0, 10.0, 15.0]


def get_means():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["X"], d["y"]
    X, y, sids = load_means("data/feats", "data/train_CoT_v01.json", split_want=("train", "val"))
    np.savez(CACHE, X=X, y=y)
    return X, y


def prep(Xtr, k=128):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - mu) / sd
    cen = Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xs - cen, full_matrices=False)
    P = Vt[:k]
    return mu, sd, cen, P


def project(X, mu, sd, cen, P):
    return ((X - mu) / sd - cen) @ P.T


def fit_per_organ(Z, y, organs, ridge):
    omu, cinv = {}, {}
    for o in organs:
        Zo = Z[y == o]
        omu[o] = Zo.mean(0)
        d = Zo - omu[o]
        C = d.T @ d / max(len(Zo) - 1, 1) + ridge * np.eye(Z.shape[1])
        cinv[o] = np.linalg.inv(C)
    return omu, cinv


def score(Z, omu, cinv, organs):
    return np.stack([np.einsum("ni,ij,nj->n", Z - omu[o], cinv[o], Z - omu[o]) for o in organs], 1).min(1)


def recall_at(s_novel, s_in_cal, fp):
    thr = np.percentile(s_in_cal, 100.0 - fp)
    return float(np.mean(s_novel > thr))


def main():
    X, y = get_means()
    rng = np.random.RandomState(0)
    proto = {f: [] for f in FPS}
    heldout = {f: [] for f in FPS}
    aucs = []
    print(f"[robust] best config k={K} ridge={RIDGE:g} per-organ cov")
    print(f"{'held-out':10} {'AUC':>7} | protocol r@5/10/15         | heldout-cal r@5/10/15")
    for qi, q in enumerate(ORGANS):
        keep = y != qi
        Xk, yk = X[keep], y[keep]
        organs = sorted(np.unique(yk).tolist())
        # 70/30 split of in-dist by organ
        tr_idx, cal_idx = [], []
        for o in organs:
            idx = np.where(yk == o)[0]
            rng.shuffle(idx)
            c = int(0.3 * len(idx))
            cal_idx.append(idx[:c]); tr_idx.append(idx[c:])
        tr_idx = np.concatenate(tr_idx); cal_idx = np.concatenate(cal_idx)

        # PROTOCOL (in-sample): fit on all 6, threshold on same points
        mu, sd, cen, P = prep(Xk, K)
        Zk = project(Xk, mu, sd, cen, P); Zq = project(X[y == qi], mu, sd, cen, P)
        omu, cinv = fit_per_organ(Zk, yk, organs, RIDGE)
        s_in = score(Zk, omu, cinv, organs); s_novel = score(Zq, omu, cinv, organs)
        auc = _auc(s_novel, s_in); aucs.append(auc)
        for f in FPS:
            proto[f].append(recall_at(s_novel, s_in, f))

        # HELD-OUT-cal: fit centroids+cov on 70%, calibrate FP threshold on untouched 30%
        mu2, sd2, cen2, P2 = prep(Xk[tr_idx], K)
        Ztr = project(Xk[tr_idx], mu2, sd2, cen2, P2)
        Zcal = project(Xk[cal_idx], mu2, sd2, cen2, P2)
        Zq2 = project(X[y == qi], mu2, sd2, cen2, P2)
        omu2, cinv2 = fit_per_organ(Ztr, yk[tr_idx], organs, RIDGE)
        s_cal = score(Zcal, omu2, cinv2, organs); s_nov2 = score(Zq2, omu2, cinv2, organs)
        for f in FPS:
            heldout[f].append(recall_at(s_nov2, s_cal, f))
        print(f"{q:10} {auc:7.3f} | "
              + "/".join(f"{proto[f][-1]:.2f}" for f in FPS)
              + "               | "
              + "/".join(f"{heldout[f][-1]:.2f}" for f in FPS))

    print(f"\nmean AUC = {np.mean(aucs):.3f}")
    print("PROTOCOL (in-sample threshold): "
          + "  ".join(f"r@{int(f)}={np.mean(proto[f]):.3f}" for f in FPS))
    print("HELD-OUT-cal (out-of-sample threshold): "
          + "  ".join(f"r@{int(f)}={np.mean(heldout[f]):.3f}" for f in FPS))


if __name__ == "__main__":
    main()
