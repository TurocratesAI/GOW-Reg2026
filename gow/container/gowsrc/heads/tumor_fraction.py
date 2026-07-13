#!/usr/bin/env python3
"""
Tumor area-fraction via CONCH zero-shot per-tile scoring - a ROBUST replacement for the CLIP-scorer
guessing a "20%" percentage string (which scored 0.18 acc on tumor-volume, 0.07 on Gleason-pattern-4-%).

Adapted from master/screening-v1/wsiqc/tumor_conch.py (CONCH tumor-vs-normal cosine per tile, organ-aware
prompts). Here we reduce the per-tile p_tumor over a slide's tissue tiles to a single AREA FRACTION, then
map it to the nearest CoT percentage bucket via a calibration fit on OFFICIAL CoT tumor-volume labels only
(CONCH is a public model, the mapping is trained on official data -> rules-clean; no private grader).

Two uses:
  1. tumor-volume / Gleason-%-4 report fields  -> calibrated(fraction) bucket (this file).
  2. a malignancy PRIOR for the #1-diagnosis node (high fraction contradicts "No tumor present").

  frac = TumorFraction(device).fraction(wsi_path, coords, organ)   # coords = the bag's tile coords (npz)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))          # wsi_io
CKPT = os.environ.get("GOW_CONCH", "/home/swapnil/master/screening_v1_container/weights/conch/conch_v1.pt")

TUMOR_PROMPTS = ["tumor", "invasive carcinoma", "malignant cells", "H&E image of invasive tumor tissue",
                 "histopathology showing malignant neoplasm", "cancerous tissue with nuclear atypia"]
NORMAL_PROMPTS = ["normal tissue", "benign tissue", "stroma", "H&E image of normal tissue",
                  "histopathology showing benign epithelium", "fibrous connective tissue", "non-neoplastic tissue"]
ORGAN_PROMPTS = {
    "prostate": ["prostatic adenocarcinoma", "Gleason pattern prostate cancer", "cribriform prostate carcinoma"],
    "breast": ["invasive ductal carcinoma of breast", "invasive lobular carcinoma", "breast carcinoma"],
    "colon": ["colorectal adenocarcinoma", "colonic carcinoma with glandular differentiation", "malignant colonic epithelium"],
    "stomach": ["gastric adenocarcinoma", "malignant gastric epithelium"],
    "lung": ["lung adenocarcinoma", "squamous cell carcinoma of lung"],
    "bladder": ["urothelial carcinoma", "invasive bladder carcinoma"],
    "cervix": ["cervical squamous cell carcinoma", "cervical adenocarcinoma"],
}


class TumorFraction:
    def __init__(self, device="cpu", temp=0.1):
        import torch, torch.nn.functional as F
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        self.torch, self.F, self.dev, self.temp = torch, F, device, temp
        self.model, self.preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=CKPT, device=device)
        self.model.eval(); self.tok = get_tokenizer()
        self._text_cache = {}

    def _tokenize(self, texts):
        t = self.tok(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                     truncation=True, padding="max_length", return_tensors="pt")
        return self.F.pad(t["input_ids"], (0, 1), value=self.tok.pad_token_id)

    def _text(self, prompts):
        toks = self._tokenize(prompts).to(self.dev)
        with self.torch.inference_mode():
            e = self.model.encode_text(toks).float()
        e = e / e.norm(dim=-1, keepdim=True)
        return e.mean(0)                                          # ensemble -> one centroid

    def _text_pair(self, organ):
        if organ not in self._text_cache:
            tp = self._text(TUMOR_PROMPTS + ORGAN_PROMPTS.get(organ, []))
            npr = self._text(NORMAL_PROMPTS)
            self._text_cache[organ] = self.torch.stack([tp, npr])  # (2, D)
        return self._text_cache[organ]

    def fraction(self, wsi_path, coords, organ=None, thr=0.5, batch=64, max_tiles=2000):
        """-> (tumor area fraction in [0,1], n tiles scored). p_tumor per tile via CONCH; mean(p>thr)."""
        import wsi_io
        from PIL import Image
        path, tmp = wsi_io.resolve_tiled(wsi_path, "data/_tmpviz")
        ps = []
        try:
            if len(coords) > max_tiles:                          # subsample for speed (area fraction is stable)
                coords = [coords[i] for i in np.random.default_rng(0).permutation(len(coords))[:max_tiles]]
            tiles = wsi_io.read_tiles(path, coords, 224, 224)
            text = self._text_pair(organ)
            for i in range(0, len(tiles), batch):
                ims = self.torch.stack([self.preprocess(Image.fromarray(t)) for t in tiles[i:i+batch]]).to(self.dev)
                with self.torch.inference_mode():
                    try: ie = self.model.encode_image(ims, proj_contrast=True, normalize=True)
                    except TypeError: ie = self.model.encode_image(ims)
                    ie = ie / ie.norm(dim=-1, keepdim=True)
                    sims = ie.float() @ text.T                   # (B,2) [tumor, normal]
                    p = self.F.softmax(sims / self.temp, -1)[:, 0].cpu().numpy()
                ps.append(p)
        finally:
            if tmp and os.path.exists(tmp):                      # remove the libvips-converted temp even on error
                os.remove(tmp)
        p = np.concatenate(ps) if ps else np.array([0.0])
        return float((p >= thr).mean()), len(p)


def fit_bucketing(fracs, gt_pcts):
    """Monotonic map CONCH area-fraction -> the discrete CoT percentage buckets (isotonic on official labels).
    fracs: [n] in [0,1]; gt_pcts: [n] ints (e.g. 10,20,...). Returns a callable frac->bucket string."""
    from sklearn.isotonic import IsotonicRegression
    buckets = sorted(set(int(g) for g in gt_pcts))
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(fracs), np.asarray(gt_pcts, float))
    def predict(frac):
        y = float(iso.predict([frac])[0])
        return f"{min(buckets, key=lambda b: abs(b - y))}%"
    return predict


if __name__ == "__main__":
    print("TumorFraction: adapt screening CONCH tumor-vs-normal -> area fraction. "
          "Calibrate fit_bucketing on official prostate CoT tumor-volume labels (needs WSI pixels).")
