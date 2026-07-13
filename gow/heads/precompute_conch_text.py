#!/usr/bin/env python3
"""Precompute CONCH text embeddings for the ontology -> gow/artifacts/text_emb.npz.

Gives the answer/grade heads real semantics (open-vocab: rare types + gyn/mesenchymal OOD naming).
Encodes every canonical question + every answer surface string that train_heads looks up.
Works around CONCH's tokenizer using the removed transformers `batch_encode_plus`.
  python gow/heads/precompute_conch_text.py --device cuda:1
"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

CKPT = "/home/swapnil/master/screening_v1_container/weights/conch/conch_v1.pt"
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
dev = a.device

os.environ.setdefault("HF_HUB_OFFLINE", "1")
model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=CKPT, device=dev)
tok = get_tokenizer()


def tokenize(texts):                                          # conch.tokenize, transformers-5 compatible
    t = tok(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
            truncation=True, padding="max_length", return_tensors="pt")
    return F.pad(t["input_ids"], (0, 1), value=tok.pad_token_id)


def encode(strings, bs=256):
    out = []
    for i in range(0, len(strings), bs):
        toks = tokenize(strings[i:i + bs]).to(dev)
        with torch.inference_mode():
            out.append(model.encode_text(toks).float().cpu().numpy())
    return np.concatenate(out)


# Open-vocab OOD naming (uterus is 20% of test, absent from training) - gyn/mesenchymal entities.
OOD_VOCAB = ["Leiomyoma", "Leiomyosarcoma", "Endometrioid carcinoma", "Serous carcinoma",
             "Endometrial stromal sarcoma", "Adenomyosis", "Endometrial hyperplasia",
             "Carcinosarcoma", "Clear cell carcinoma", "Uterus", "Uterine corpus",
             "Endometrium", "Myometrium", "Smooth muscle tumor of uncertain malignant potential"]

# The CAP protocol library (cap_organs.json: 78 protocols) so the generalized OOD open-vocab path can name ANY
# organ + histologic type, not just the gyn set. CLEAN before embedding: drop TNM-staging phrases and form-filler
# ("No evidence of primary tumor", "Cannot be assessed", "Other (specify)", "Specify percent") so text_emb.npz
# holds only real, matchable naming terms. Keep organ surfaces + real diagnoses/procedures/grades.
import re as _re
_JUNK = _re.compile(r"no evidence|cannot be|primary tumor|regional lymph|distant metast|^yp?[tnm][0-4x]|see comment|"
                    r"not identified|indeterminate|^specify|no residual|not applicable|other \(specify\)|"
                    r"^specify percent|^percent|not reported|^see |^n/?a$|^none$|^present$|^absent$", _re.I)


def _keep(t):
    return bool(t) and 2 < len(t) < 200 and not _JUNK.search(t.strip())


CAP = json.load(open(os.path.join(ART, "cap_organs.json")))
cap_terms, _dropped = set(), 0
for _o, _v in CAP.items():
    if not isinstance(_v, dict):
        continue
    cap_terms.add(_v.get("surface", _o))                     # organ/site surface always kept
    for _field in ("procedures", "diagnoses", "grades"):
        for _t in (_v.get(_field) or []):
            if _keep(_t):
                cap_terms.add(_t)
            else:
                _dropped += 1
    if _v.get("grade_system"):
        cap_terms.add(_v["grade_system"])
print(f"CAP library: {len(CAP)} protocols -> {len(cap_terms)} clean terms to embed (dropped {_dropped} junk/staging)")

Q = json.load(open(os.path.join(ART, "questions.json")))
AV = json.load(open(os.path.join(ART, "answer_vocab.json")))
keys = set(Q.keys()) | set(OOD_VOCAB) | cap_terms             # questions + gyn OOD + the full CAP library
for organ in AV:
    for cq in AV[organ]:
        keys.update(AV[organ][cq].keys())                     # answer surface strings
keys = sorted(k for k in keys if k and len(k) < 400)          # drop empty + the long final-report answers

emb = encode(keys)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
out = os.path.join(ART, "text_emb.npz")
np.savez(out, keys=np.array(keys, dtype=object), emb=emb.astype(np.float32))
print(f"encoded {len(keys)} strings -> {out}  emb {emb.shape}")
