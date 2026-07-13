#!/usr/bin/env python3
"""
OOD routing for interf1: detect an out-of-7-organ slide and emit a SAFE chain instead of the
force-argmax mis-route.

Design (from two ultracode reviews):
  * gate = Mahalanobis on the frozen Virchow2 bag-mean vs the 7 trained organs (gow/artifacts/ood_gate.npz,
    leave-one-organ-out AUC 0.88). Flag OOD iff score > threshold. Tuned for PRECISION (a false flag on a
    SEEN organ costs -0.06..-0.13), so we use a conservative margin above the p95 fit threshold.
  * on OOD: walk the CERVIX (gyn) topology - every emitted edge stays an EXISTING closed-ontology string
    (no false-positive edges); NEVER route to prostate/breast (10-12 organ-specific FP edges/case).
  * NAME the organ + diagnosis via CONCH open-vocab over the gyn/uterine OOD_VOCAB, so the report reads
    "Uterine corpus, ...; Endometrioid carcinoma" instead of a cervix diagnosis. Answers touch only
    MESS + report, never the edge set.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_NPZ = os.path.join(HERE, "..", "artifacts", "ood_gate.npz")
CAP_PATH = os.path.join(HERE, "..", "artifacts", "cap_organs.json")

# gyn/uterine open-vocab naming set (the known ~20% OOD). Must be present in text_emb.npz (CONCH-embedded).
OOD_ORGANS = ["Uterine corpus", "Uterus", "Endometrium"]
OOD_DX = ["Leiomyoma", "Leiomyosarcoma", "Endometrioid carcinoma", "Serous carcinoma",
          "Clear cell carcinoma", "Carcinosarcoma", "Endometrial stromal sarcoma",
          "Endometrial hyperplasia", "Adenomyosis",
          "Smooth muscle tumor of uncertain malignant potential"]
OOD_ROUTE_ORGAN = "cervix"                                    # nearest safe gyn topology (legacy)
CONF_THR = 0.90                                               # organ-router max-softmax below this = "unsure"
GATE_MARGIN = 1.25                                            # Mahalanobis threshold = 1.25 * fit-p95 (precision)
# Generalized per-organ gate operating point. The per-organ p90 fit thresholds flag ~14% of in-distribution
# slides (measured on held-out): far too loose, because a false flag opens naming to the 78-protocol CAP
# open-vocab and its attractors mis-name the trained organ. Scaling the per-organ Mahalanobis threshold x2.5
# (a threshold sweep against held-out in-dist + public TCGA novel organs) drops in-dist FP to ~2% while still
# catching uterus (the dominant known OOD; UCEC scores 893/1146 vs cervix thr 584 = 1.5-2x margin) and 9/10
# TCGA novel organs (only esophagus, which routes to the adjacent-GI stomach, is missed). Precision-first: on
# this challenge a false OOD flag on a seen organ costs more than a missed borderline novel organ.
GEN_GATE_SCALE = 2.5


# ------------------------------------------------- CAP open-vocab naming (the full CAP protocol library)
# A CAP protocol IS a walker ontology (data element -> allowed options), so cap_organs.json gives 78 ready-made
# option sets. A flagged OOD slide keeps its nearest-trained TOPOLOGY (edges stay in-ontology) but its organ +
# diagnosis are named open-vocab from the CAP library, so we can name ANY organ, not just the gyn set. Every
# term below is CONCH-embedded in text_emb.npz (precompute_conch_text.py embeds the whole CAP library).
_CAP = None


def _cap():
    global _CAP
    if _CAP is None:
        _CAP = json.load(open(CAP_PATH)) if os.path.exists(CAP_PATH) else {}
    return _CAP


def cap_organ_names():
    """All CAP protocol organ/site surface names -> open-vocab organ candidates for a flagged OOD slide."""
    return [v.get("surface", k) for k, v in _cap().items() if isinstance(v, dict) and v.get("surface", k)]


import re
_DX_JUNK = re.compile(r"no evidence|cannot be|primary tumor|regional lymph|distant metast|^yp?T[0-4]|see comment|"
                      r"not identified|indeterminate|^specify|no residual|not applicable|other \(specify\)", re.I)


def _clean_dx(dxs):
    """Drop TNM-staging / form-filler phrases (~17% of the CAP 'diagnoses' field) so candidates are real dx."""
    return [d for d in dxs if d and not _DX_JUNK.search(d)]


# Clean site name -> CAP protocol surface, for organs whose name shares NO word with the protocol title (so the
# fuzzy match below silently fails and falls back to the wrong-organ union). Found by the ultracode audit.
SITE_TO_CAP = {
    "brain": "Central Nervous System", "central nervous system": "Central Nervous System",
    "liver": "Hepatocellular Carcinoma", "lymph node": "Precursor and Mature Lymphoid Malignancies",
    "melanoma": "Invasive Melanoma of the Skin", "skin": "Invasive Melanoma of the Skin",
    "bile duct": "Intrahepatic Bile Ducts", "adrenal gland": "Adrenal Gland",
    "salivary gland": "Major Salivary Glands", "pituitary gland": "Pituitary Neuroendocrine Tumor",
    "placenta": "Trophoblast", "nasal cavity": "Nasal Cavity and Paranasal Sinuses", "eye": "Uveal Melanoma",
}


def cap_diagnoses(organ_surface=None, cap_max=80):
    """Diagnoses of the CAP protocol(s) matching organ_surface (site->CAP map, then exact, then fuzzy on a shared
    word); if None/unmatched, a bounded union of all CAP diagnoses. Bounded + junk-filtered for precise candidates."""
    lib = _cap()
    if organ_surface:
        mapped = SITE_TO_CAP.get(organ_surface.lower())
        if mapped:
            for k, v in lib.items():
                if isinstance(v, dict) and v.get("surface", k) == mapped and (v.get("diagnoses") or []):
                    return _clean_dx(v["diagnoses"])[:cap_max]
        toks = {w for w in organ_surface.lower().split() if len(w) > 3}
        best = []
        for k, v in lib.items():
            if not isinstance(v, dict) or not (v.get("diagnoses") or []):
                continue
            surf = v.get("surface", k).lower()
            if surf == organ_surface.lower():
                return _clean_dx(v["diagnoses"])[:cap_max]
            if toks and (toks & set(surf.split())):
                best += [d for d in _clean_dx(v["diagnoses"]) if d not in best]
        if best:
            return best[:cap_max]
    seen, out = set(), []
    for v in lib.values():
        for d in (_clean_dx(v.get("diagnoses") or []) if isinstance(v, dict) else []):
            if d not in seen:
                seen.add(d); out.append(d)
                if len(out) >= cap_max:
                    return out
    return out


class OODGate:
    def __init__(self, path=GATE_NPZ, margin=GATE_MARGIN):
        d = np.load(path, allow_pickle=True)
        self.mu = d["mu"].astype(np.float32); self.sd = d["sd"].astype(np.float32)
        self.P = d["P"].astype(np.float32)
        self.organ_mu = d["organ_mu"].astype(np.float32)      # [7, k]
        # per-organ (QDA) covariance: one k x k inverse-covariance per organ (min-over-organs Mahalanobis).
        if "organ_cov_inv" in d.files:
            self.organ_cov_inv = d["organ_cov_inv"].astype(np.float32)     # [7, k, k]
        else:                                                 # legacy pooled-covariance npz -> broadcast to 7
            self.organ_cov_inv = np.repeat(d["cov_inv"].astype(np.float32)[None], len(self.organ_mu), 0)
        self.thr = float(d["threshold"]) * margin             # legacy global threshold (kept for is_ood)
        # cervix-bucket threshold: the operating threshold for uterus routing. Uterus is the only known OOD
        # organ and the router places it in the cervix bucket, so we only decide OOD within that bucket. This
        # keeps routing EDGE-SAFE: a flagged slide already walks cervix topology, so flipping it to the OOD
        # naming path changes only the organ/diagnosis answers (MESS + report), never the reasoning edges.
        self.cervix_thr = float(d["cervix_threshold"]) if "cervix_threshold" in d.files else self.thr
        # GENERALIZED gate: one conservative threshold per organ, so ANY of the 7 organs (not just cervix) can
        # be flagged OOD. On a flag the slide keeps its routed-organ topology; only the organ/diagnosis naming
        # opens to CAP open-vocab, so a false flag never changes reasoning edges.
        self.organs = [str(o) for o in d["organs"]] if "organs" in d.files else \
            ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]
        if "per_organ_threshold" in d.files:
            pt = d["per_organ_threshold"].astype(np.float32)
            self.organ_thr = {o: float(pt[i]) * GEN_GATE_SCALE for i, o in enumerate(self.organs)}
        else:
            self.organ_thr = {}                               # legacy npz -> cervix-only fallback below

    def score(self, mean_vec):
        """min-over-7-organs squared Mahalanobis (each organ uses its OWN covariance) of a mean-vector [2560]."""
        z = ((mean_vec.astype(np.float32) - self.mu) / self.sd) @ self.P.T   # [k]
        dists = [float((z - m) @ ci @ (z - m)) for m, ci in zip(self.organ_mu, self.organ_cov_inv)]
        return min(dists)

    def is_ood(self, mean_vec):
        """Legacy global flag (score above the in-dist p95 x margin). Prefer should_route_ood."""
        return self.score(mean_vec) > self.thr

    def should_route_ood(self, raw_organ, mean_vec, org_vec=None):
        """Flag a slide routed to ANY of the 7 organs as OOD when its min-Mahalanobis score exceeds that
        organ's conservative in-distribution threshold. This generalizes the old cervix-only gate to every
        organ (the CAP-grounded design: the detector was always general, only the gate was cervix-restricted).
        Edge-safe: a flagged slide KEEPS its routed-organ topology and only the organ/diagnosis NAMING opens to
        the CAP open-vocab, so a false flag can only mis-name, never change a reasoning edge. Legacy npz without
        per-organ thresholds falls back to the old cervix-bucket rule."""
        if self.organ_thr:
            thr = self.organ_thr.get(raw_organ)
            return thr is not None and self.score(mean_vec) > thr
        if raw_organ != OOD_ROUTE_ORGAN:                      # legacy fallback (cervix-only)
            return False
        return self.score(mean_vec) > self.cervix_thr


_GATE = None


def get_gate(margin=GATE_MARGIN):
    global _GATE
    if _GATE is None and os.path.exists(GATE_NPZ):
        _GATE = OODGate(margin=margin)
    return _GATE
