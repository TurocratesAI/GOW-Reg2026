#!/usr/bin/env python3
"""
Validate the interf0 CONCH tissue/background gate on REAL crops synthesized from a slide.

We can't download the challenge's ROI bundle, so we build both sides from an official WSI (exactly
the plan's approach): sample tiles from stained tissue regions (positives) and from blank glass /
background regions (negatives), then check the gate labels each correctly. This is the one thing the
whole of Metric-B depends on.

  python gow/interf0/validate_gate.py --wsi data/PIT_455153.tiff --n 40 --device cpu
"""
import argparse, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))       # wsi_io
sys.path.insert(0, HERE)                                       # respond
import wsi_io
from respond import Responder


def sample_coords(path, n, tile=224, thumb_long=640):
    """Split a thumbnail into SOLID-tissue vs CLEAN-glass grid cells; return level-0 coords for each.
    Background = genuinely blank (bright + desaturated), not faint-tissue edges, to mirror real ROIs."""
    W, H = wsi_io.dims(path)
    ds = max(1.0, max(W, H) / thumb_long)
    thumb = wsi_io.fast_thumbnail(path, ds)                    # [h,w,3] uint8
    hsv = np.asarray(Image.fromarray(thumb).convert("HSV"), np.int16)
    sat, val = hsv[..., 1], hsv[..., 2]
    tissue_px = (val < 220) & (sat > 25)                       # H&E colored & not blank-white
    glass_px = (val > 232) & (sat < 12)                        # bright + desaturated = clean glass
    cell = max(1, int(round(tile / ds)))                      # thumb pixels per 224 level-0 tile
    gh, gw = thumb.shape[0] // cell, thumb.shape[1] // cell
    tis, bg = [], []
    for gy in range(gh):
        for gx in range(gw):
            ys, xs = slice(gy*cell, (gy+1)*cell), slice(gx*cell, (gx+1)*cell)
            tf, gf = tissue_px[ys, xs].mean(), glass_px[ys, xs].mean()
            x0, y0 = int(gx * cell * ds), int(gy * cell * ds)
            if x0 + tile >= W or y0 + tile >= H:
                continue
            if tf > 0.55:
                tis.append((x0, y0))
            elif gf > 0.95:                                    # almost entirely clean glass
                bg.append((x0, y0))
    rng = np.random.default_rng(0)
    def pick(lst):
        return [lst[i] for i in rng.permutation(len(lst))[:n]] if lst else []
    return pick(tis), pick(bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="data/PIT_455153.tiff")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", default="", help="optional dir to save a few sampled crops")
    a = ap.parse_args()

    path, tmp = wsi_io.resolve_tiled(a.wsi, "data/_tmpviz")
    tis, bg = sample_coords(path, a.n)
    print(f"[gate] {os.path.basename(a.wsi)} -> sampled {len(tis)} tissue + {len(bg)} background crops")
    if not tis or not bg:
        print("[gate] not enough of one class sampled; try another slide"); return

    tis_tiles = wsi_io.read_tiles(path, tis, 224, 224)
    bg_tiles = wsi_io.read_tiles(path, bg, 224, 224)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        for i, t in enumerate(tis_tiles[:6]): Image.fromarray(t).save(f"{a.dump}/tissue_{i}.png")
        for i, t in enumerate(bg_tiles[:6]): Image.fromarray(t).save(f"{a.dump}/bg_{i}.png")

    print("[gate] loading responder (GrandQC primary + CONCH backup)...")
    from collections import Counter
    r = Responder(device=a.device)

    def eval_side(tiles, want):
        ok, by = 0, Counter()
        for t in tiles:
            lab, info = r.classify(Image.fromarray(t), tta=False)
            ok += int(lab == want); by[info["by"]] += 1
        return ok, len(tiles), by

    t_ok, t_n, t_by = eval_side(tis_tiles, "tissue")
    b_ok, b_n, b_by = eval_side(bg_tiles, "background")
    print("\n================ FINAL RESPONDER GATE ================")
    print(f"  tissue crops : {t_ok}/{t_n} -> tissue      (recall {t_ok/t_n:.3f})   decided by {dict(t_by)}")
    print(f"  glass  crops : {b_ok}/{b_n} -> background  (recall {b_ok/b_n:.3f})   decided by {dict(b_by)}")
    print(f"  balanced acc : {0.5*(t_ok/t_n + b_ok/b_n):.3f}")
    # show the two deterministic answers the responder emits
    from respond import TISSUE_ANSWER, BG_ANSWER
    print(f'\n  tissue -> "{TISSUE_ANSWER[:70]}..."')
    print(f'  glass  -> "{BG_ANSWER[:70]}..."')
    print("\n  every B1/B2/B3 point is downstream of the balanced acc above.")
    if tmp and os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main()
