#!/usr/bin/env python3
"""
Figure 1: combined pipeline figure grounded in real imagery.

Top row = real panels from one WSI: thumbnail -> GrandQC tissue mask -> sampled 20x patches -> ROI crop.
Bottom row = the processing pipeline as clean nodes: Virchow2 -> patch bag -> (organ router + OOD detector)
-> per-question pooler -> answer head (CONCH text space) -> ontology walker -> CAP report, with an ROI ->
grounding branch. Restrained clinical palette; the OOD branch is the one accented element.

  python gow/eval/make_pipeline_fig.py --wsi data/PIT_455153.tiff --device cpu
"""
import argparse, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))
import extract_features as E, grandqc_mask as G, wsi_io

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"
INK, BLUE_F, BLUE_E, GRAY_F, GRAY_E, ACCENT = "#1F2933", "#E7EEF6", "#3B6EA5", "#F3F4F6", "#9AA5B1", "#C1553B"


def make_panels(wsi, dev):
    local, is_tmp = E.localize(wsi, "data/_tmpviz")
    try:
        W, H = wsi_io.dims(local)
        mask = G.clean_tissue_mask(local, dev, TW, AW)
        ds = max(1, max(W, H) // 900)
        thumb = wsi_io.fast_thumbnail(local, ds)
        th, tw = thumb.shape[:2]
        mask_up = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)) > 127
        overlay = thumb.copy()
        overlay[mask_up] = (0.5 * overlay[mask_up] + 0.5 * np.array([40, 200, 90], np.uint8)).astype(np.uint8)
        coords = E.cap_tiles(E._grid_from_mask(mask, W, H, 224, 0.25), 6000)
        k = min(16, len(coords))
        sample = [coords[i] for i in np.linspace(0, len(coords) - 1, k).astype(int)]
        tiles = wsi_io.read_tiles(local, sample, 224, 224)
        cell, cols = 96, 4
        rows = (k + cols - 1) // cols
        mont = Image.new("RGB", (cols * cell, rows * cell), (250, 250, 250))
        for i, p in enumerate(tiles):
            mont.paste(Image.fromarray(p).resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
        roi = Image.fromarray(tiles[len(tiles) // 2]).resize((256, 256))
        return thumb, overlay, np.asarray(mont), np.asarray(roi)
    finally:
        if is_tmp and os.path.exists(local):
            os.remove(local)


def build(thumb, overlay, mont, roi, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({"font.family": "sans-serif"})
    fig = plt.figure(figsize=(10.2, 4.8)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 102); ax.set_ylim(-3, 46); ax.axis("off")

    def img(im, x, y, w, h, label):
        ax.imshow(im, extent=(x, x + w, y, y + h), aspect="auto", zorder=2)
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.9, zorder=3))
        ax.text(x + w / 2, y - 1.6, label, ha="center", va="top", fontsize=8, color=INK)

    def node(x, y, w, h, t, kind="op", fs=8.2):
        fc, ec = (BLUE_F, BLUE_E) if kind == "data" else (ACCENT + "22", ACCENT) if kind == "ood" else (GRAY_F, GRAY_E)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.7", fc=fc, ec=ec, lw=1.1, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, zorder=3)

    def arr(x1, y1, x2, y2, style="-", color=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color, ls=style, zorder=1))

    # top row: real imagery  thumbnail -> mask -> patches -> ROI
    ty, tw2 = 30, 15
    xs = [3, 22, 41, 62]
    img(thumb, xs[0], ty, tw2, 13, "thumbnail")
    img(overlay, xs[1], ty, tw2, 13, "tissue mask")
    img(mont, xs[2], ty, 17, 13, "20x patches")
    img(roi, xs[3], ty, 13, 13, "ROI")
    arr(xs[0] + tw2, ty + 6.5, xs[1], ty + 6.5)
    arr(xs[1] + tw2, ty + 6.5, xs[2], ty + 6.5)

    # bottom row: pipeline nodes
    by = 8
    node(2, by, 12, 8, "Virchow2\nencoder", "op")
    node(17, by, 12, 8, "patch bag\n[N, 2560]", "data")
    node(32, by, 13, 8, "organ router\n+ OOD detector", "ood")
    node(48, by, 12, 8, "per-question\npooler", "op")
    node(63, by, 13, 8, "answer head\n(CONCH space)", "op")
    node(79, by, 11, 8, "ontology\nwalker", "op")
    node(92, by, 9, 8, "CAP\nreport", "data")
    for a, b in [(14, 17), (29, 32), (45, 48), (60, 63), (76, 79), (90, 92)]:
        arr(a, by + 4, b, by + 4)
    # patches -> encoder (imagery into pipeline)
    arr(xs[2] + 8, ty, 8, by + 8, color=INK)
    # ROI -> grounding branch (single clean vertical arrow into the branch box)
    node(57, 17.5, 24, 6, "ROI  ->  tissue/background gate  ->  grounding response", "op", fs=7.2)
    arr(xs[3] + 6.5, ty, xs[3] + 6.5, 23.5)
    # OOD routing (accent, dashed): the OOD detector bypasses to the walker's gyn topology, arced below the row
    ax.add_patch(FancyArrowPatch((38, by), (84, by), connectionstyle="arc3,rad=-0.13",
                 arrowstyle="-|>", mutation_scale=11, lw=1.2, color=ACCENT, ls="--", zorder=1))

    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, dpi=300)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="data/PIT_455153.tiff")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/figures/pipeline.pdf")
    a = ap.parse_args()
    os.makedirs("data/_tmpviz", exist_ok=True)
    thumb, overlay, mont, roi = make_panels(a.wsi, a.device)
    build(thumb, overlay, mont, roi, a.out)


if __name__ == "__main__":
    main()
