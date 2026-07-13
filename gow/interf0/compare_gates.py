#!/usr/bin/env python3
"""
Head-to-head: CONCH margin vs GrandQC tissue-fraction as the interf0 gate, across MULTIPLE slides.
The question: does GrandQC give ONE threshold that works on every slide (fixing CONCH's slide-dependence)?

  python gow/interf0/compare_gates.py --wsis data/PIT_455153.tiff data/PIT_286556.tiff --n 40 --device cpu
"""
import argparse, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))
sys.path.insert(0, HERE)
import wsi_io
from respond import Responder
from grandqc_roi import GrandQCGate
from validate_gate import sample_coords


def collect(path, n, conch, gqc):
    p, tmp = wsi_io.resolve_tiled(path, "data/_tmpviz")
    tis, bg = sample_coords(p, n)
    rows = {"tissue": {"conch": [], "gqc": []}, "glass": {"conch": [], "gqc": []}}
    for lab, coords in (("tissue", tis), ("glass", bg)):
        for t in wsi_io.read_tiles(p, coords, 224, 224):
            im = Image.fromarray(t)
            rows[lab]["conch"].append(conch._conch_margin(im, tta=False))  # background-ish is HIGH
            rows[lab]["gqc"].append(gqc.tissue_fraction(im))               # tissue is HIGH
    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    return {k: {s: np.array(v) for s, v in d.items()} for k, d in rows.items()}


def sweep(data, signal, lo, hi, tissue_high):
    """best single threshold across ALL slides; tissue_high=True if tissue has the HIGHER value."""
    best = (0, None)
    for thr in np.linspace(lo, hi, 61):
        accs = []
        for d in data:
            tv, gv = d["tissue"][signal], d["glass"][signal]
            if tissue_high:
                t_acc = (tv >= thr).mean(); g_acc = (gv < thr).mean()
            else:
                t_acc = (tv <= thr).mean(); g_acc = (gv > thr).mean()
            accs.append(0.5 * (t_acc + g_acc))
        m = np.mean(accs)
        if m > best[0]:
            best = (m, thr, accs)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsis", nargs="+", default=["data/PIT_455153.tiff", "data/PIT_286556.tiff"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    print("[compare] loading CONCH + GrandQC...")
    conch = Responder(device=a.device, use_grandqc=False)      # raw CONCH margin only, for the comparison
    gqc = GrandQCGate(device=a.device)

    data = []
    for w in a.wsis:
        d = collect(w, a.n, conch, gqc)
        data.append(d)
        print(f"  {os.path.basename(w):22} tissue: conch-margin {np.median(d['tissue']['conch']):+.3f}  gqc-frac {np.median(d['tissue']['gqc']):.3f}   "
              f"| glass: conch-margin {np.median(d['glass']['conch']):+.3f}  gqc-frac {np.median(d['glass']['gqc']):.3f}")

    print("\n================ SINGLE-THRESHOLD, ALL SLIDES ================")
    cm, ct, ca = sweep(data, "conch", -0.12, 0.35, tissue_high=False)
    gm, gt, ga = sweep(data, "gqc", 0.05, 0.95, tissue_high=True)
    print(f"  CONCH margin   : best thr {ct:+.3f}  balanced {cm:.3f}   per-slide {[round(x,3) for x in ca]}")
    print(f"  GrandQC frac   : best thr {gt:.3f}   balanced {gm:.3f}   per-slide {[round(x,3) for x in ga]}")
    print("\n  (a single threshold that stays high on EVERY slide = the robust, slide-independent gate)")


if __name__ == "__main__":
    main()
