#!/usr/bin/env python3
"""
Measure the in-distribution cost of the generalized OOD gate: on held-out (in-dist) cases, compare the named
organ with the gate ON vs OFF. A false OOD flag opens naming to CAP and can mis-name a trained organ; this
quantifies how often that happens and the organ-accuracy delta, so we can decide whether to tighten/soften it.

  python gow/eval/measure_ood_fp.py --heads gow/artifacts/gow_heads_v2.pt --n 250 --device cuda:0
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for s in ("walker", "heads"):
    sys.path.insert(0, os.path.join(ROOT, s))
sys.path.insert(0, ROOT)
import gow_walker as W, gow_model, data_split as DS
from train_heads import TextEmb
from eval_real import predict_chain


def named_organ(chain):
    return next((s["answer"] for s in chain if s["question"].lower().startswith("what is the organ")), "")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default=os.path.join(ROOT, "artifacts/gow_heads_v2.pt"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(os.path.join(ROOT, "artifacts/text_emb.npz"))
    model = gow_model.build().to(dev); model.load_state_dict(torch.load(a.heads, map_location=dev)); model.eval()
    cot = {os.path.splitext(c["id"])[0]: c for c in json.load(open(os.path.join(ROOT, "..", "data/train_CoT_v01.json"))) if "id" in c}
    ss = DS.load()
    bags = [b for b in sorted(glob.glob(os.path.join(ROOT, "..", "data/feats/*.npz")))
            if os.path.splitext(os.path.basename(b))[0] in cot and DS.split_of(b, ss) == "test"][: a.n]
    print(f"[fp] {len(bags)} held-out in-dist cases; gate ON vs OFF ...")
    on_hit = off_hit = flagged = flag_lost = 0
    for npz in bags:
        sid = os.path.splitext(os.path.basename(npz))[0]; gt = cot[sid]["organ"].strip().lower()
        H = torch.from_numpy(np.load(npz)["X"].astype("float32")).to(dev)
        _, ch_off = predict_chain(model, H, T, QSURF, AV, META, text, dev, use_ood=False)
        _, ch_on = predict_chain(model, H, T, QSURF, AV, META, text, dev, use_ood=True)
        n_off, n_on = named_organ(ch_off).strip().lower(), named_organ(ch_on).strip().lower()
        off_ok = gt in n_off or n_off in gt
        on_ok = gt in n_on or n_on in gt
        off_hit += off_ok; on_hit += on_ok
        if n_off != n_on:
            flagged += 1
            if off_ok and not on_ok:
                flag_lost += 1
    n = len(bags)
    print(f"\n[fp] organ-name accuracy  OFF={off_hit}/{n} ({off_hit/n:.3f})   ON={on_hit}/{n} ({on_hit/n:.3f})")
    print(f"[fp] gate flipped the name on {flagged}/{n} ({flagged/n:.1%}); of those, {flag_lost} went correct->wrong")
    print(f"[fp] net in-dist organ-accuracy cost of the generalized gate = {(off_hit-on_hit)/n:+.3f}")


if __name__ == "__main__":
    main()
