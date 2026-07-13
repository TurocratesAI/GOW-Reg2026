"""
Container config: put our gow/ source on sys.path and resolve weight paths.

Weights load from /opt/ml/model/<...> inside the container (the mounted model.tar.gz), and fall back to
the dev locations when those are absent - so the SAME code runs under do_test_run (Docker) and locally.
Import this FIRST in the interface modules (it sets env vars the gow modules read at import time).
"""
import os, sys, time
from pathlib import Path

# Record container-boot time (monotonic) so extract_slide can budget the WHOLE job -- model load + extract --
# against the 5-min platform kill, not just the extract step. Set here because gowcfg is imported first.
os.environ.setdefault("GOW_PROC_START", repr(time.monotonic()))

# --- our gow source tree (container: /opt/app/gow ; dev: reg2026/gow) ---
_env = os.environ.get("GOW_SRC")
if _env and Path(_env).exists():
    GOW = Path(_env)
elif Path("/opt/app/gow").exists():
    GOW = Path("/opt/app/gow")
else:
    GOW = Path(__file__).resolve().parent.parent            # .../reg2026/gow
for _sub in ("extract", "heads", "walker", "interf0"):
    _p = str(GOW / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(GOW) not in sys.path:
    sys.path.insert(0, str(GOW))

# --- weights: /opt/ml/model first, else dev ---
MODEL = Path(os.environ.get("MODEL_PATH", "/opt/ml/model"))


def _w(rel, dev):
    p = MODEL / rel
    return str(p) if p.exists() else dev


ARTIFACTS = str(GOW / "artifacts")
TEXT_EMB = _w("artifacts/text_emb.npz", str(GOW / "artifacts" / "text_emb.npz"))
GOW_HEADS = _w("gow_heads.pt", str(GOW / "artifacts" / "gow_heads_final.pt"))
CONCH = _w("conch/conch_v1.pt", "/home/swapnil/master/screening_v1_container/weights/conch/conch_v1.pt")
GRANDQC_TISSUE = _w("grandqc/tissue_detection_mpp10.pth", "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth")
GRANDQC_ARTIFACT = _w("grandqc/grandqc_artifact_mpp15.pth", "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth")
VIRCHOW2 = None                                             # load via timm hf-hub from the (bundled) HF cache
_hf = MODEL / "hf_cache"                                    # container: point HF at the bundled Virchow2 cache
if _hf.exists():
    os.environ["HF_HOME"] = str(_hf)

# --- export env for the gow modules that read weight paths at import ---
os.environ["GOW_CONCH"] = CONCH
os.environ["GOW_GRANDQC_TISSUE"] = GRANDQC_TISSUE
os.environ["GOW_GRANDQC_ARTIFACT"] = GRANDQC_ARTIFACT
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
