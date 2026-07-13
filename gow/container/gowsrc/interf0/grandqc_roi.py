#!/usr/bin/env python3
"""
GrandQC as an ROI-level tissue/background gate for interf0.

GrandQC's tissue-detection head (smp UnetPlusPlus, EffNet-B0) is trained EXACTLY to separate tissue
from glass/background on H&E - including pale tissue (fat/mucin/necrosis) that Otsu wrongly rejects.
Its artifact head knows fold/pen/darkspot/focus/edge. That is a purpose-built discriminator, far more
robust than a CONCH zero-shot margin, so we use it as the PRIMARY gate signal and let CONCH be the
semantic backup for the ambiguous middle.

Reuses the model loading + tiled inference from gow/extract/grandqc_mask.py.
  tissue frac in [0,1]: high = tissue, low = background.
"""
import os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extract"))       # grandqc_mask, wsi_io
import grandqc_mask as G

TISSUE_W = os.environ.get("GOW_GRANDQC_TISSUE", "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth")
ARTIFACT_W = os.environ.get("GOW_GRANDQC_ARTIFACT", "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth")


class GrandQCGate:
    def __init__(self, device="cpu", use_artifact=False):
        self.device = device
        self.tmodel = G._load("unetpp", TISSUE_W, 2, device)      # class 0 = tissue, 1 = background
        self.amodel = G._load("unet", ARTIFACT_W, 8, device) if use_artifact else None

    def tissue_fraction(self, pil):
        """Fraction of the ROI GrandQC calls tissue (class 0). High -> tissue.

        NOTE (resolution): the tissue head is trained at MPP10 and clean_tissue_mask feeds it a MPP10 thumbnail.
        Here the ROI is fed at its native resolution because the challenge ROI MPP is not provided; on the
        sample crops the tissue/glass call (color/whiteness dominated) is balanced 1.000 at native res. If the
        real ROI MPP is learned to be much finer than MPP10, resample here (mirror G._thumb(path, G.MPP_TISSUE))
        before predict; validate against gow/interf0/compare_gates.py so it does not regress."""
        arr = np.asarray(pil.convert("RGB"), np.uint8)
        pred = G._predict(G._jpeg(arr), self.tmodel, self.device)  # matches the JPEG-Q80 training dist
        return float((pred == 0).mean())

    def normal_fraction(self, pil):
        """Fraction the artifact head calls NORMAL (non-fold/pen/blur). Only if use_artifact=True."""
        if self.amodel is None:
            return 1.0
        arr = np.asarray(pil.convert("RGB"), np.uint8)
        pred = G._predict(G._jpeg(arr), self.amodel, self.device)
        return float((pred == G.NORMAL_CLASS).mean())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    g = GrandQCGate(device=a.device)
    print(f"{a.roi} -> GrandQC tissue fraction {g.tissue_fraction(Image.open(a.roi)):.3f}")
