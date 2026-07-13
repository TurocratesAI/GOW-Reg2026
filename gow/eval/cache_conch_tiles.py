#!/usr/bin/env python3
"""Cache per-tile CONCH-image embeddings for the OOD-validation slides so the MI-Zero naming can be iterated
without re-reading WSIs. Writes data/tcga_ood/conch_cache/<organ>.npz {emb[N,512]}."""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for s in ("extract", "heads"):
    sys.path.insert(0, os.path.join(ROOT, s))
import extract_features as EF, wsi_io, conch_image_id as CI

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"


def main():
    import openslide
    from types import SimpleNamespace
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/tcga_ood")
    ap.add_argument("--n-tiles", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    out = os.path.join(a.dir, "conch_cache"); os.makedirs(out, exist_ok=True)
    os.makedirs("/tmp/gowcache", exist_ok=True)
    eargs = SimpleNamespace(qc="grandqc", grandqc_no_artifact=True, grandqc_tissue=TW, grandqc_artifact=AW,
                            tissue_frac=0.25, max_tiles=4000, batch_size=64, readers=8, tmpdir="/tmp/gowcache", dry_run=True)
    print("[cache] loading CONCH ...")
    cmodel, preprocess, tok = CI.load_conch(dev)
    picks = {p["organ"].replace(" ", "_"): p for p in json.load(open(os.path.join(a.dir, "manifest.json")))}
    for key, p in picks.items():
        slide = os.path.join(a.dir, key + ".svs")
        dst = os.path.join(out, key + ".npz")
        if not os.path.exists(slide) or os.path.exists(dst):
            continue
        _, coords, meta = EF.extract_slide(slide, None, dev, eargs)         # dry-run -> coords only
        coords = np.asarray(coords)
        if len(coords) == 0:
            print(f"  {key}: no tissue"); continue
        resolved, tmp = wsi_io.resolve_tiled(slide, "/tmp/gowcache")
        s = openslide.OpenSlide(resolved)
        try:
            mpp = float(s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5) or 0.5)
        except Exception:
            mpp = 0.5
        s.close()
        read_size = int(round(CI.TILE_PX * CI.TARGET_MPP / mpp))
        idx = np.linspace(0, len(coords) - 1, min(a.n_tiles, len(coords))).astype(int)
        tiles = np.stack(wsi_io.read_tiles(resolved, coords[idx].astype(int), read_size, CI.TILE_PX, 8))
        emb = CI._encode_tiles(tiles, cmodel, preprocess, dev)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        np.savez(dst, emb=emb.astype("float16"), organ=p["organ"])
        print(f"  {key}: cached {emb.shape} tiles")
    print("[cache] done")


if __name__ == "__main__":
    main()
