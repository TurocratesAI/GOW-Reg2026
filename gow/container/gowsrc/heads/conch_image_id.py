#!/usr/bin/env python3
"""
CONCH-image zero-shot organ + diagnosis identification for the OOD path (COMPLIANT: a public pretrained model at
inference, no training). When the OOD gate flags a slide as not-one-of-the-7-organs, the Virchow2 answer head is
unreliable across the 78-way CAP set, so we encode a sample of the slide's tiles with CONCH's IMAGE tower and do
proper CONCH zero-shot: cosine-match the pooled image embedding against CONCH TEXT embeddings of CLEAN anatomical
site names built with a prompt-template ensemble, then the chosen site's CAP diagnoses. CONCH is CLIP-style so
image and (templated) text embeddings share one space.

  python gow/heads/conch_image_id.py --wsi data/PIT_286556.tiff --coords data/feats/PIT_286556.npz --device cuda:0
"""
import argparse, os, sys, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))
sys.path.insert(0, os.path.join(HERE, ".."))
import wsi_io
import ood_route as OR

CONCH_CKPT = "/home/swapnil/master/screening_v1_container/weights/conch/conch_v1.pt"
TILE_PX, TARGET_MPP = 224, 0.5

# Clean anatomical site names (better zero-shot prompts than CAP protocol titles). Each maps into the CAP library
# by fuzzy word-overlap (ood_route.cap_diagnoses), e.g. "Breast" -> the breast CAP protocols' diagnoses.
CLEAN_SITES = ["Breast", "Prostate", "Colon", "Rectum", "Stomach", "Urinary bladder", "Lung", "Cervix", "Uterus",
               "Endometrium", "Ovary", "Fallopian tube", "Kidney", "Liver", "Pancreas", "Gallbladder", "Bile duct",
               "Esophagus", "Small intestine", "Appendix", "Anus", "Thyroid", "Adrenal gland", "Skin", "Melanoma",
               "Brain", "Central nervous system", "Lymph node", "Bone", "Soft tissue", "Testis", "Penis", "Ureter",
               "Urethra", "Vagina", "Vulva", "Salivary gland", "Larynx", "Oral cavity", "Nasal cavity", "Nasopharynx",
               "Pharynx", "Thymus", "Pituitary gland", "Placenta", "Peritoneum", "Eye"]
# The official CONCH zero-shot 22-template ensemble (from mahmoodlab/CONCH prompts). Measured: swapping our 5
# hand-written templates for these lifts constrained 8-way organ ID from 3/8 to 6/8 - the single biggest lever.
TEMPLATES = [
    "{}.", "a histopathological image showing {}.", "a photomicrograph showing {}.",
    "a histopathological image of {}.", "a photomicrograph of {}.", "a histopathological photograph of {}.",
    "an image of {}.", "a histopathological photograph showing {}.", "an image showing {}.", "shows {}.",
    "an example of {}.", "presence of {}.", "{} is shown.", "{} is present.", "this is {}.",
    "an H&E stained image of {}.", "there is {}.", "an H&E stained image showing {}.",
    "an H&E image showing {}.", "an H&E image of {}.", "{}, H&E stain.", "{}, H&E.",
]


def load_conch(dev):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=CONCH_CKPT, device=dev)
    model.eval()
    return model, preprocess, get_tokenizer()


def _tokenize(tok, texts):
    import torch.nn.functional as F
    t = tok(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
            truncation=True, padding="max_length", return_tensors="pt")
    return F.pad(t["input_ids"], (0, 1), value=tok.pad_token_id)


def encode_text(model, tok, names, dev, templates=("{}",)):
    """Per name: encode each template, average over templates -> L2-normalized [len(names), 512]."""
    import torch
    protos = []
    for n in names:
        toks = _tokenize(tok, [t.format(n) for t in templates]).to(dev)
        with torch.inference_mode():
            e = torch.nn.functional.normalize(model.encode_text(toks).float(), dim=-1)
        v = e.mean(0)
        protos.append((v / v.norm().clamp_min(1e-8)).cpu().numpy())
    return np.stack(protos)


def _encode_tiles(tiles_uint8, model, preprocess, dev, bs=64):
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(tiles_uint8), bs):
        batch = torch.stack([preprocess(Image.fromarray(t)) for t in tiles_uint8[i:i + bs]]).to(dev)
        with torch.inference_mode():
            e = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def slide_embedding(wsi_resolved, coords, mpp, model, preprocess, dev, n_sample=96, readers=8):
    read_size = int(round(TILE_PX * TARGET_MPP / mpp))
    idx = np.linspace(0, len(coords) - 1, min(n_sample, len(coords))).astype(int)
    tiles = np.stack(wsi_io.read_tiles(wsi_resolved, coords[idx].astype(int), read_size, TILE_PX, readers))
    v = _encode_tiles(tiles, model, preprocess, dev).mean(0)
    return v / (np.linalg.norm(v) + 1e-8)


@functools.lru_cache(maxsize=1)
def _site_protos(model_id, tok_id, dev):
    # cached per (model,tok,dev); recomputed only once. keys are ids so the lru works with unhashable objects.
    return encode_text(_MODEL, _TOK, CLEAN_SITES, dev, TEMPLATES)


_MODEL = _TOK = None


def identify(slide_emb, model, tok, dev, topk=3):
    """CONCH-image zero-shot: pooled image emb -> best clean site (templated prompts, mean-centered to kill the
    'soft tissue' attractor) -> that site's CAP diagnosis."""
    global _MODEL, _TOK
    _MODEL, _TOK = model, tok
    site_protos = _site_protos(id(model), id(tok), dev)
    centered = site_protos - site_protos.mean(0)             # remove common-mode bias (attractor prompts)
    centered = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)
    s = centered @ slide_emb
    order = s.argsort()[::-1]
    site = CLEAN_SITES[order[0]]
    dxs = OR.cap_diagnoses(site) or OR.OOD_DX
    dx = ""
    if dxs:
        dproto = encode_text(model, tok, dxs, dev, TEMPLATES)
        dx = dxs[int((dproto @ slide_emb).argmax())]
    return site, dx, [(CLEAN_SITES[i], float(s[i])) for i in order[:topk]]


def main():
    import openslide
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    coords = np.load(a.coords)["coords"]
    resolved, tmp = wsi_io.resolve_tiled(a.wsi, "/tmp/gowconch")
    s = openslide.OpenSlide(resolved)
    try:
        mpp = float(s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5) or 0.5)
    except Exception:
        mpp = 0.5
    s.close()
    print(f"[conch-id] loading CONCH; sampling tiles from {os.path.basename(a.wsi)} (mpp={mpp:.3f}) ...")
    model, preprocess, tok = load_conch(a.device)
    v = slide_embedding(resolved, coords, mpp, model, preprocess, a.device)
    site, dx, top = identify(v, model, tok, a.device)
    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    print(f"[conch-id] site={site!r}  dx={dx!r}")
    print("[conch-id] top sites:", [(o, round(sc, 3)) for o, sc in top])


if __name__ == "__main__":
    main()
