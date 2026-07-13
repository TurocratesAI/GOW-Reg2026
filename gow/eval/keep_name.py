#!/usr/bin/env python3
"""
KEEP (Astaxanthin/KEEP, MIT-licensed, ungated) zero-shot organ naming head-to-head vs CONCH, using the SAME
recipe (organ name + CAP diagnoses x 22 CONCH templates -> averaged, mean-centered prototype; mean-pooled tile
cosine). KEEP is a public pretrained VLM used at inference only (compliant). Reports open-vocab + constrained
accuracy on the same 8 real novel organs. Caches KEEP tile embeddings so the recipe can be re-iterated.

  python gow/eval/keep_name.py --dir data/tcga_ood --device cuda:1
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for s in ("extract", "heads"):
    sys.path.insert(0, os.path.join(ROOT, s))
import extract_features as EF, wsi_io, conch_image_id as CI
import ood_route as OR

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"


def load_keep(dev):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from torchvision import transforms
    import transformers.modeling_utils as _mu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # KEEP predates newer transformers' tied-weights finalization; give it an empty map so the check is a no-op.
    if hasattr(_mu.PreTrainedModel, "mark_tied_weights_as_initialized"):
        _orig = _mu.PreTrainedModel.mark_tied_weights_as_initialized
        def _patched(self, *aa, **kk):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
            return _orig(self, *aa, **kk)
        _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched
    model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return model, tok, tf


def keep_image(model, tf, tiles_uint8, dev, bs=64):
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(tiles_uint8), bs):
        batch = torch.stack([tf(Image.fromarray(t)) for t in tiles_uint8[i:i + bs]]).to(dev)
        with torch.inference_mode():
            f = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def keep_text(model, tok, dev, names, templates):
    import torch
    protos = []
    for n in names:
        prompts = [t.format(n) for t in templates]
        ti = tok(prompts, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        with torch.inference_mode():
            e = torch.nn.functional.normalize(model.encode_text(ti).float(), dim=-1)
        v = e.mean(0); protos.append((v / v.norm().clamp_min(1e-8)).cpu().numpy())
    return np.stack(protos)


def build_protos(model, tok, dev, sites, center=True):
    protos = []
    for s in sites:
        prompts = [s] + OR.cap_diagnoses(s)[:40]
        e = keep_text(model, tok, dev, prompts, CI.TEMPLATES)
        v = e.mean(0); protos.append(v / (np.linalg.norm(v) + 1e-8))
    P = np.stack(protos)
    if center:
        P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return P


def main():
    import openslide
    from types import SimpleNamespace
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/tcga_ood")
    ap.add_argument("--n-tiles", type=int, default=300)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    cache = os.path.join(a.dir, "keep_cache"); os.makedirs(cache, exist_ok=True)
    os.makedirs("/tmp/gowkeep", exist_ok=True)
    eargs = SimpleNamespace(qc="grandqc", grandqc_no_artifact=True, grandqc_tissue=TW, grandqc_artifact=AW,
                            tissue_frac=0.25, max_tiles=4000, batch_size=64, readers=8, tmpdir="/tmp/gowkeep", dry_run=True)
    print("[keep] loading KEEP ...")
    model, tok, tf = load_keep(dev)
    picks = {p["organ"].replace(" ", "_"): p for p in json.load(open(os.path.join(a.dir, "manifest.json")))}
    embs, known = [], []
    for key, p in picks.items():
        slide = os.path.join(a.dir, key + ".svs")
        dst = os.path.join(cache, key + ".npz")
        if os.path.exists(dst):
            d = np.load(dst); embs.append(d["emb"].astype("float32")); known.append(str(d["organ"])); continue
        if not os.path.exists(slide):
            continue
        _, coords, _ = EF.extract_slide(slide, None, dev, eargs)
        coords = np.asarray(coords)
        if len(coords) == 0:
            continue
        resolved, tmp = wsi_io.resolve_tiled(slide, "/tmp/gowkeep")
        s = openslide.OpenSlide(resolved)
        try:
            mpp = float(s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5) or 0.5)
        except Exception:
            mpp = 0.5
        s.close()
        rs = int(round(CI.TILE_PX * CI.TARGET_MPP / mpp))
        idx = np.linspace(0, len(coords) - 1, min(a.n_tiles, len(coords))).astype(int)
        tiles = np.stack(wsi_io.read_tiles(resolved, coords[idx].astype(int), rs, CI.TILE_PX, 8))
        e = keep_image(model, tf, tiles, dev)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        np.savez(dst, emb=e.astype("float16"), organ=p["organ"])
        embs.append(e); known.append(p["organ"]); print(f"  {key}: KEEP-encoded {e.shape}")

    embs = [e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8) for e in embs]

    def to_site(o):
        for s in CI.CLEAN_SITES:
            if o.split()[0].lower() in s.lower() or s.lower() in o.lower():
                return s
        return o.title()
    constrained = sorted({to_site(o) for o in known})
    for label, sites in [("open-vocab (47 sites)", CI.CLEAN_SITES), (f"constrained ({len(constrained)})", constrained)]:
        P = build_protos(model, tok, dev, sites)
        hit = 0
        rows = []
        for e, o in zip(embs, known):
            pred = sites[int((e @ P.T).mean(0).argmax())]
            ok = o.split()[0].lower() in pred.lower() or pred.lower() in o.lower()
            hit += int(ok); rows.append((o, pred, ok))
        print(f"\n[KEEP {label}]  accuracy = {hit}/{len(known)}")
        for o, pred, ok in rows:
            print(f"    {o:12} -> {pred:20} {'<-hit' if ok else ''}")


if __name__ == "__main__":
    main()
