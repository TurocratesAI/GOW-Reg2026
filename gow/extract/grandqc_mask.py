"""Self-contained GrandQC clean-tissue mask for the GOW extractor.

clean tissue = tissue(smp.UnetPlusPlus, EffNet-B0, @MPP10, class0)
             ∩ NORMAL(smp.Unet, EffNet-B0, @MPP1.5, class1)   # drops fold/pen/darkspot/focus/edge

Reuses the logic of the working integration `qc/wsiqc/grandqc.py`; weights are local:
  tissue   : qc/wsiqc/models/tissue_detection_mpp10.pth
  artifact : qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth
GrandQC runs on downsampled thumbnails (not per-tile) -> cheap, CPU-viable.
"""
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ENC = "timm-efficientnet-b0"
TILE = 512
IMN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMN_STD = np.array([0.229, 0.224, 0.225], np.float32)
MPP_TISSUE, MPP_ARTIFACT, DEFAULT_MPP = 10.0, 1.5, 0.5
NORMAL_CLASS = 1
_CACHE: dict = {}


def _slide_mpp(slide):
    try:
        m = float(slide.properties.get("openslide.mpp-x"))
        return m if 0.1 < m < 2.0 else DEFAULT_MPP
    except (TypeError, ValueError):
        return DEFAULT_MPP


def _thumb(path, target_mpp, max_dim=10000):
    import wsi_io                                             # fast threaded overview (NOT get_thumbnail)
    W, H = wsi_io.dims(path)
    ds = max(target_mpp / DEFAULT_MPP, max(W, H) / max_dim)  # cap long side -> avoid cv2 65500px/pixel limits on huge slides
    return wsi_io.fast_thumbnail(path, downsample=ds)


def _jpeg(arr, q=80):
    import cv2  # GrandQC was trained on JPEG-Q80 thumbnails; match the distribution
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return arr if not ok else cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _load(arch, weights_path, classes, device):
    import torch
    import segmentation_models_pytorch as smp
    key = (arch, weights_path, str(device))
    if key in _CACHE:
        return _CACHE[key]
    ctor = smp.UnetPlusPlus if arch == "unetpp" else smp.Unet
    model = ctor(encoder_name=ENC, encoder_weights=None, classes=classes, activation=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)  # trusted local file
    if isinstance(state, torch.nn.Module):
        model = state
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval().to(device)
    _CACHE[key] = model
    return model


def _predict(arr, model, device, tile=TILE):
    import torch
    H, W = arr.shape[:2]
    ph, pw = -H % tile, -W % tile
    a = np.pad(arr, ((0, ph), (0, pw), (0, 0)), constant_values=255)
    Hp, Wp = a.shape[:2]
    out = np.zeros((Hp, Wp), np.uint8)
    with torch.inference_mode():
        for y in range(0, Hp, tile):
            for x in range(0, Wp, tile):
                patch = a[y:y + tile, x:x + tile].astype(np.float32) / 255.0
                t = ((patch - IMN_MEAN) / IMN_STD).transpose(2, 0, 1)
                t = torch.from_numpy(np.ascontiguousarray(t)).unsqueeze(0).to(device)
                out[y:y + tile, x:x + tile] = model(t).squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    return out[:H, :W]


def clean_tissue_mask(path, device, tissue_w, artifact_w=None):
    """Return mask01 float [h,w] on the finer grid. tissue ∩ NORMAL if artifact_w given, else tissue only."""
    tmodel = _load("unetpp", tissue_w, 2, device)
    tissue = (_predict(_jpeg(_thumb(path, MPP_TISSUE)), tmodel, device) == 0)  # class 0 = tissue
    if artifact_w is None:
        return tissue.astype(np.float32)
    amodel = _load("unet", artifact_w, 8, device)
    normal = (_predict(_jpeg(_thumb(path, MPP_ARTIFACT)), amodel, device) == NORMAL_CLASS)  # class 1 = NORMAL
    th, tw = normal.shape
    tissue_up = np.asarray(
        Image.fromarray((tissue * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)) > 127
    return (tissue_up & normal).astype(np.float32)
