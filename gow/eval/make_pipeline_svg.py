#!/usr/bin/env python3
"""
Figure 1 as a hand-authored vector diagram (SVG -> PDF via cairosvg / CLI), publication quality.
Panels are cached under paper/figures/_panels so layout can be iterated without re-running GrandQC.

  python gow/eval/make_pipeline_svg.py --wsi data/PIT_455153.tiff --device cpu [--refresh]
"""
import argparse, base64, io, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_pipeline_fig import make_panels

INK, SUB, DATA_F, DATA_E, OP_F, OP_E, ACC_F, ACC_E = "#1F2933", "#5B6670", "#E7EEF6", "#3B6EA5", "#F4F5F7", "#AEB6C0", "#F6E7E1", "#C1553B"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
CACHE = "paper/figures/_panels"


def b64(arr, maxdim=360):
    im = Image.fromarray(arr)
    if max(im.size) > maxdim:
        sc = maxdim / max(im.size); im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def get_imgs(wsi, dev, refresh):
    keys = ["thumb", "mask", "patch", "roi"]
    if os.path.isdir(CACHE) and all(os.path.exists(f"{CACHE}/{k}.png") for k in keys) and not refresh:
        return {k: b64(np.asarray(Image.open(f"{CACHE}/{k}.png").convert("RGB"))) for k in keys}
    os.makedirs(CACHE, exist_ok=True)
    thumb, overlay, mont, roi = make_panels(wsi, dev)
    for k, v in zip(keys, [thumb, overlay, mont, roi]):
        Image.fromarray(v).save(f"{CACHE}/{k}.png")
    return {k: b64(np.asarray(Image.fromarray(v))) for k, v in zip(keys, [thumb, overlay, mont, roi])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default="data/PIT_455153.tiff")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paper/figures/pipeline.pdf")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    os.makedirs("data/_tmpviz", exist_ok=True)
    imgs = get_imgs(a.wsi, a.device, a.refresh)

    W, Hc = 1180, 512
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hc}" viewBox="0 0 {W} {Hc}" font-family="{FONT}">',
         f'<rect width="{W}" height="{Hc}" fill="white"/>', '<defs>',
         f'<marker id="ah" markerWidth="10" markerHeight="10" refX="7.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{INK}"/></marker>',
         f'<marker id="aha" markerWidth="10" markerHeight="10" refX="7.5" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{ACC_E}"/></marker>',
         '</defs>']

    def img(k, x, y, w, h, cap, par="xMidYMid slice"):
        s.append(f'<clipPath id="clip_{k}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/></clipPath>')
        s.append(f'<rect x="{x+2}" y="{y+3}" width="{w}" height="{h}" rx="7" fill="#000" opacity="0.10"/>')
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="#FCFCFD" clip-path="url(#clip_{k})"/>')
        s.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="{par}" '
                 f'clip-path="url(#clip_{k})" href="data:image/png;base64,{imgs[k]}"/>')
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" stroke="{INK}" stroke-width="1.1"/>')
        s.append(f'<text x="{x+w/2}" y="{y+h+17}" text-anchor="middle" font-size="12.5" fill="{SUB}">{cap}</text>')

    def node(x, y, w, h, lines, kind, fs=14.5):
        f, e = {"data": (DATA_F, DATA_E), "op": (OP_F, OP_E), "ood": (ACC_F, ACC_E)}[kind]
        s.append(f'<rect x="{x+2}" y="{y+3}" width="{w}" height="{h}" rx="9" fill="#000" opacity="0.07"/>')
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{f}" stroke="{e}" stroke-width="1.5"/>')
        cy = y + h / 2 - (len(lines) - 1) * 8.5
        for i, ln in enumerate(lines):
            s.append(f'<text x="{x+w/2}" y="{cy+i*17+5}" text-anchor="middle" font-size="{fs}" fill="{INK}">{ln}</text>')

    def line(x1, y1, x2, y2, acc=False, dash=False):
        c, m = (ACC_E, "aha") if acc else (INK, "ah")
        d = 'stroke-dasharray="6,4"' if dash else ''
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.7" marker-end="url(#{m})" {d}/>')

    def path(d, acc=False, dash=False):
        c, m = (ACC_E, "aha") if acc else (INK, "ah")
        da = 'stroke-dasharray="6,4"' if dash else ''
        s.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.7" marker-end="url(#{m})" {da}/>')

    ARR = "→"
    # --- top band: real imagery (complete slide via "meet") ---
    s.append(f'<text x="44" y="26" font-size="12" fill="{SUB}" letter-spacing="1.6" font-weight="600">FROM SLIDE TO EVIDENCE</text>')
    iy, ih = 40, 116
    img("thumb", 44, iy, 150, ih, "thumbnail", par="xMidYMid meet")
    img("mask", 214, iy, 150, ih, "tissue mask", par="xMidYMid meet")
    img("patch", 384, iy, 168, ih, "20x patches")
    img("roi", 792, iy, 116, ih, "region of interest")
    line(194, iy + ih / 2, 214, iy + ih / 2)
    line(364, iy + ih / 2, 384, iy + ih / 2)

    # --- bottom band: pipeline nodes ---
    ny, nh = 340, 66
    node(44, ny, 126, nh, ["Virchow2", "encoder"], "op")
    node(196, ny, 120, nh, ["patch bag", "[N, 2560]"], "data")
    node(342, ny, 150, nh, ["organ router", "+ OOD detector"], "ood")
    node(516, ny, 130, nh, ["per-question", "pooler"], "op")
    node(670, ny, 158, nh, ["answer head", "(CONCH space)"], "op")
    node(852, ny, 120, nh, ["ontology", "walker"], "op")
    node(996, ny, 110, nh, ["CAP", "report"], "data")
    for x1, x2 in [(170, 196), (316, 342), (492, 516), (646, 670), (828, 852), (972, 996)]:
        line(x1, ny + nh / 2, x2, ny + nh / 2)

    # patches -> encoder: smooth S-curve from the panel bottom into the TOP of the Virchow2 box (arrowhead points down)
    path(f"M402,{iy+ih} C 402,248 100,248 100,{ny}")
    # ROI -> grounding branch: start BELOW the caption; box is wide with a smaller font so text fits
    node(700, 250, 300, 42, [f"ROI  {ARR}  tissue/background gate  {ARR}  response"], "op", fs=12.5)
    line(850, iy + ih + 30, 850, 250)
    # OOD routing arc (accent, dashed): OOD detector bypasses to the walker gyn topology
    path(f"M417,{ny+nh} C 480,474 850,474 912,{ny+nh}", acc=True, dash=True)
    s.append(f'<text x="664" y="490" text-anchor="middle" font-size="13.5" fill="{ACC_E}" font-style="italic">out-of-distribution routing: any unseen organ keeps topology, opens naming</text>')

    s.append('</svg>')
    svg = "\n".join(s)
    svg_path = a.out.replace(".pdf", ".svg")
    open(svg_path, "w").write(svg)
    try:
        import cairosvg
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=a.out)
    except ImportError:
        import subprocess
        subprocess.run(["cairosvg", svg_path, "-o", a.out], check=True)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
