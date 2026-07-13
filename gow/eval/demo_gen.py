#!/usr/bin/env python3
"""
Generate the data for the pathologist demo: run GOW on one WSI, capture the chain + report + the
per-question QCPooler attention, and render a thumbnail plus one attention heatmap per answered question.
Writes gow/artifacts/demo_data.json (base64 PNGs + chain), consumed by the HTML interface.

  python gow/eval/demo_gen.py --wsi <path.tiff> --device cuda:0
"""
import argparse, base64, io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))
sys.path.insert(0, os.path.join(HERE, "..", "walker"))
sys.path.insert(0, os.path.join(HERE, "..", "heads"))
sys.path.insert(0, os.path.join(HERE, ".."))
import gow_walker as W, gow_model
import extract_features as EF, wsi_io
from train_heads import TextEmb, ORGANS, MAX_CAND
import ood_route as OR

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"


def b64png(arr):
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def heatmap_overlay(thumb, coords, attn, ds, tile_px=224):
    """Paint per-tile attention onto the thumbnail grid, blur, colormap, alpha-composite over the thumbnail."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.cm as cm
    from scipy.ndimage import gaussian_filter
    th, tw = thumb.shape[:2]
    grid = np.zeros((th, tw), np.float32)
    cell = max(1, int(round(tile_px / ds)))
    a = attn / (attn.max() + 1e-8)                                  # per-question normalize -> brightest tile = 1
    for (x, y), w in zip(coords, a):
        cx, cy = int(x / ds), int(y / ds)
        grid[cy:cy + cell, cx:cx + cell] = np.maximum(grid[cy:cy + cell, cx:cx + cell], w)
    grid = gaussian_filter(grid, sigma=cell * 0.6)
    grid /= (grid.max() + 1e-8)
    rgba = cm.get_cmap("inferno")(grid)                             # [th,tw,4]
    heat = (rgba[..., :3] * 255).astype(np.uint8)
    alpha = (np.clip(grid * 1.3, 0, 1) ** 0.8)[..., None] * 0.72    # translucent where attention is high
    out = (thumb * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True)
    ap.add_argument("--heads", default="gow/artifacts/gow_heads_final.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="gow/artifacts/demo_data.json")
    a = ap.parse_args()
    import torch
    from types import SimpleNamespace
    dev = a.device

    print("[demo] loading Virchow2 + heads ...")
    virchow2 = EF.load_virchow2(None, dev)
    model = gow_model.build().to(dev); model.load_state_dict(torch.load(a.heads, map_location=dev)); model.eval()
    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb("gow/artifacts/text_emb.npz")
    eargs = SimpleNamespace(qc="grandqc", grandqc_no_artifact=True, grandqc_tissue=TW, grandqc_artifact=AW,
                            tissue_frac=0.25, max_tiles=6000, batch_size=64, readers=8, tmpdir="/tmp/gowdemo", dry_run=False)
    os.makedirs("/tmp/gowdemo", exist_ok=True)

    print("[demo] embedding slide ...")
    X, coords, meta = EF.extract_slide(a.wsi, virchow2, dev, eargs)
    coords = np.asarray(coords)
    H = torch.from_numpy(X.astype("float32")).to(dev)

    # thumbnail + downsample factor to map level-0 coords -> thumbnail pixels
    resolved, tmp = wsi_io.resolve_tiled(a.wsi, "/tmp/gowdemo")
    Wd, Hd = wsi_io.dims(resolved)
    ds = max(1, max(Wd, Hd) / 1100)
    thumb = wsi_io.fast_thumbnail(resolved, ds)
    if tmp and os.path.exists(tmp): os.remove(tmp)

    with torch.inference_mode():
        o_logits, z = model.organ(H)
    org_vec = torch.softmax(o_logits, -1)
    organ = ORGANS[int(o_logits.argmax())]
    is_ood = False
    gate = OR.get_gate()
    if gate is not None and gate.should_route_ood(organ, H.float().mean(0).cpu().numpy(), org_vec):
        is_ood = True; organ = OR.OOD_ROUTE_ORGAN

    steps = []                                                      # (question_surface, answer, attn np[N])

    def afn(org, cq):
        d = AV.get(org, {}).get(cq, {})
        cands = [x for x, _ in sorted(d.items(), key=lambda kv: -kv[1])][:MAX_CAND]
        if is_ood:
            if cq == W.ORGAN_Q: cands = OR.OOD_ORGANS + cands
            elif cq in W.DIAG_QS or "histologic type" in cq: cands = OR.OOD_DX + cands
            cands = cands[:MAX_CAND]
        if not cands:
            return ""
        cemb = torch.from_numpy(np.stack([text(x) for x in cands])).to(dev)
        qemb = torch.from_numpy(text(cq)).to(dev)
        with torch.inference_mode():
            logits, attn = model.answer(H, z, org_vec, qemb, cemb)
        ans = cands[int(logits.argmax())]
        steps.append((QSURF.get(cq, {}).get("surface", cq), ans, attn.detach().cpu().numpy()))
        return ans

    first_q = META.get(organ, {}).get("first_question_canon", W.ORGAN_Q)
    co = META.get(organ, {}).get("co_roots", [])
    chain, ans = W.walk(organ, afn, T, QSURF, first_q, use_renderer=True, co_roots=co)
    report = next((s["answer"] for s in chain if "final pathology report" in s["question"].lower()), "")

    print(f"[demo] organ={organ}  ood={is_ood}  answered nodes={len(steps)}  tiles={len(coords)}")
    out_steps = []
    seen = set()
    for q, ansv, attn in steps:
        if q in seen or "final pathology report" in q.lower():
            continue
        seen.add(q)
        heat = heatmap_overlay(thumb, coords, attn, ds)
        out_steps.append({"question": q, "answer": ansv, "heatmap": b64png(heat)})

    data = {"organ": organ, "is_ood": is_ood, "n_tiles": int(len(coords)),
            "thumbnail": b64png(thumb), "report": report.replace("\\n", "\n"),
            "steps": out_steps}
    json.dump(data, open(a.out, "w"))
    print(f"[demo] wrote {a.out}  ({len(out_steps)} question overlays, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
