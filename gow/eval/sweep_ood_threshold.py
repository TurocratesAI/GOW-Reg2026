#!/usr/bin/env python3
"""
Sweep the OOD gate's per-organ Mahalanobis threshold (scaled) to trade in-distribution false-positive rate
against novel-organ recall. In-dist scores come from the held-out challenge slides; the novel-organ scores are
the 10 public TCGA slides (recorded routed-organ + score from bench_tcga). Prints, per scale, the in-dist FP%
and the TCGA recall so we can pick the precision-tuned operating point that ships in the container.

  python gow/eval/sweep_ood_threshold.py --device cuda:0
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for s in ("walker", "heads"):
    sys.path.insert(0, os.path.join(ROOT, s))
sys.path.insert(0, ROOT)
import gow_walker as W, gow_model, data_split as DS, ood_route as OR
from train_heads import TextEmb

# (routed organ, min-Mahalanobis score, true organ) for the 10 public TCGA novel-organ slides (from bench_tcga)
TCGA = [("cervix", 840, "brain"), ("lung", 2213, "kidney"), ("stomach", 1382, "thyroid"),
        ("lung", 2001, "liver"), ("lung", 1229, "pancreas"), ("lung", 1185, "skin"),
        ("breast", 1131, "ovary"), ("cervix", 989, "head and neck"), ("lung", 2658, "adrenal"),
        ("stomach", 296, "esophagus")]
ORGANS = ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default=os.path.join(ROOT, "artifacts/gow_heads_v2.pt"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    model = gow_model.build().to(dev); model.load_state_dict(torch.load(a.heads, map_location=dev)); model.eval()
    gate = OR.get_gate()
    ss = DS.load()
    cot = {os.path.splitext(c["id"])[0] for c in json.load(open(os.path.join(ROOT, "..", "data/train_CoT_v01.json"))) if "id" in c}
    bags = [b for b in sorted(glob.glob(os.path.join(ROOT, "..", "data/feats/*.npz")))
            if os.path.splitext(os.path.basename(b))[0] in cot and DS.split_of(b, ss) == "test"]
    print(f"[sweep] scoring {len(bags)} held-out in-dist slides ...")
    ind = []  # (routed_organ, score)
    for npz in bags:
        H = torch.from_numpy(np.load(npz)["X"].astype("float32")).to(dev)
        with torch.inference_mode():
            o_logits, _ = model.organ(H)
        routed = ORGANS[int(o_logits.argmax())]
        ind.append((routed, gate.score(H.float().mean(0).cpu().numpy())))
    thr = gate.organ_thr  # per-organ base (p90) threshold
    print(f"[sweep] per-organ base thresholds: " + "  ".join(f"{o}={thr[o]:.0f}" for o in ORGANS))
    print(f"\n{'scale':>6} {'in-dist FP%':>11} {'TCGA recall':>12}   dropped TCGA organs")
    for scale in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]:
        fp = np.mean([s > thr[o] * scale for o, s in ind])
        caught, dropped = 0, []
        for routed, score, true in TCGA:
            if score > thr[routed] * scale:
                caught += 1
            else:
                dropped.append(true)
        print(f"{scale:6.2f} {fp*100:10.1f}% {caught:>7}/10     {', '.join(dropped) if dropped else '-'}")


if __name__ == "__main__":
    main()
