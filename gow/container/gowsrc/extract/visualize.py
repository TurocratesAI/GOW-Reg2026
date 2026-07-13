#!/usr/bin/env python3
"""Show the pipeline on one slide: GrandQC clean-tissue mask overlay + sampled patches + Virchow2.
  python gow/extract/visualize.py --wsi s3://.../train/X.tiff --out data/viz/X --device cuda:1
"""
import argparse, os, sys, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as E, grandqc_mask as G, wsi_io

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"

ap = argparse.ArgumentParser()
ap.add_argument("--wsi", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device
os.makedirs(os.path.dirname(a.out), exist_ok=True)
os.makedirs("data/_tmpviz", exist_ok=True)

local, is_tmp = E.localize(a.wsi, "data/_tmpviz")
try:
    W, H = wsi_io.dims(local)
    mask = G.clean_tissue_mask(local, dev, TW, AW)                 # GrandQC clean tissue (tissue ∩ NORMAL)

    # (1) mask overlay on a display thumbnail
    ds = max(1, max(W, H) // 1200)
    thumb = wsi_io.fast_thumbnail(local, ds)
    th, tw = thumb.shape[:2]
    mask_up = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)) > 127
    ov = thumb.copy()
    ov[mask_up] = (0.45 * ov[mask_up] + 0.55 * np.array([40, 220, 60], np.uint8)).astype(np.uint8)
    Image.fromarray(ov).save(a.out + "_mask.png")

    # (2) sampled patches (the actual tiles kept) -> montage
    coords = E.cap_tiles(E._grid_from_mask(mask, W, H, 224, 0.25), 6000)   # reuse the mask
    n = len(coords)
    k = min(64, n)
    sample = [coords[i] for i in np.linspace(0, n - 1, k).astype(int)]
    patches = wsi_io.read_tiles(local, sample, 224, 224)
    cols = 8; rows = (k + cols - 1) // cols; cell = 128
    canvas = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    for i, p in enumerate(patches):
        canvas.paste(Image.fromarray(p).resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
    canvas.save(a.out + "_patches.png")

    # (3) confirm: patches -> Virchow2 -> embedding
    m = E.load_virchow2(None, dev)
    emb = E.embed_batch(m, np.stack(patches), dev)
    print(f"{os.path.basename(local)}: tissue_tiles={n}  sampled={k}  "
          f"patches[{k},224,224,3] -> Virchow2 -> emb {emb.shape}  <- THIS is the bag (per-tile 2560-d)")
finally:
    if is_tmp and os.path.exists(local):
        os.remove(local)
