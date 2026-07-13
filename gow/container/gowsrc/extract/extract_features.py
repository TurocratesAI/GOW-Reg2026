#!/usr/bin/env python3
"""
GOW feature extractor:  WSI  ->  tissue tiles (224px @ 20x/0.5MPP)  ->  Virchow2  ->  cached bag.

Per slide writes  <out>/<stem>.npz  with:
    X      : [N, 2560] float16   # Virchow2 embedding = concat(CLS[1280], mean(patch_tokens)[1280])
    coords : [N, 2]    int32     # level-0 (x, y) top-left of each tile
    mean   : [2560]    float16   # mean-pooled slide vector (cheap global; ABMIL trained later)
    meta   : mpp, read_size, tile_px, n_tiles, source

Design notes (REG2026 specifics):
  * Test slides are single-level 20x generic-TIFF with NO usable magnification metadata
    (bogus resolution tag) -> we ASSUME 0.5 MPP unless a sane openslide.mpp-x is present.
  * Virchow2 (ViT-H/14) wants 224x224 @ 0.5 MPP, ImageNet norm; embedding drops the 4 register tokens.
  * Frozen encoder: no grad. Disk-frugal: --delete-wsi removes the slide after embedding.
  * Shard across GPUs by launching one process per GPU with --device cuda:k --shard k/2.
  * --dry-run does tissue+tiling only (no torch/timm/weights) to validate coverage.

Run (in the gleason venv which has torch/timm/safetensors):
    gleason/.venv/bin/python gow/extract/extract_features.py --wsi-dir data --out-dir data/feats --device cuda:0
Requires: openslide-python, numpy, Pillow  (+ torch, timm  unless --dry-run).
"""
import argparse, os, sys, glob, json, time, subprocess
import numpy as np
from PIL import Image
import wsi_io                                               # fast threaded WSI IO (same dir)

Image.MAX_IMAGE_PIXELS = None
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
TARGET_MPP = 0.5          # 20x
TILE_PX = 224             # Virchow2 input
N_PREFIX = 5              # Virchow2: 1 CLS + 4 register tokens -> patch tokens start at index 5

# FALLBACK extract-relative budget, used only if GOW_PROC_START (boot time) is unavailable, e.g. standalone CLI.
# The shipped deadline is the boot-anchored TOTAL_BUDGET_S below. Env-overridable.
EXTRACT_BUDGET_S = float(os.environ.get("GOW_EXTRACT_BUDGET_S", "210"))
# TOTAL wall-clock budget measured from PROCESS start (container boot; GOW_PROC_START set in gowcfg), i.e. it
# includes the model-load time, not just extract. On Grand Challenge a cold 2.5GB Virchow2 load off the network
# mount can take 30-60s; budgeting only the extract (as EXTRACT_BUDGET_S did) let load + a huge slide tip past
# the 5-min kill on the slowest cases. Anchoring the deadline at boot + 210s makes every case exit ~230s
# regardless of load speed -- and it ONLY shortens the few slides that would otherwise time out (a real partial
# bag instead of zero); fast/normal slides finish well before it and are unaffected. Env-overridable.
TOTAL_BUDGET_S = float(os.environ.get("GOW_TOTAL_BUDGET_S", "210"))


def localize(path, tmpdir):
    """s3://... -> stream to a tempfile (returns local_path, is_temp). Local paths pass through."""
    if path.startswith("s3://"):
        os.makedirs(tmpdir, exist_ok=True)
        dst = os.path.join(tmpdir, os.path.basename(path))
        subprocess.run(["aws", "s3", "cp", "--quiet", path, dst], check=True)
        return dst, True
    return path, False


# ---------------------------------------------------------------- tissue mask
def _otsu(gray_u8: np.ndarray) -> int:
    """Otsu threshold on a uint8 array (returns level 0..255)."""
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    sum_all = np.dot(np.arange(256), hist)
    wB = 0.0; sumB = 0.0; best = 0.0; level = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= best:
            best = between; level = i
    return level


def _grid_from_mask(mask01: np.ndarray, W: int, H: int, read_size: int, tissue_frac: float):
    """Aggregate any binary mask (any resolution) to the (ny,nx) tile grid -> kept tile coords."""
    nx, ny = W // read_size, H // read_size
    if nx < 1 or ny < 1:
        return np.zeros((0, 2), np.int32)
    cell = Image.fromarray((mask01 * 255).astype(np.uint8)).resize((nx, ny), Image.BILINEAR)
    frac = np.asarray(cell, np.float32) / 255.0
    gy, gx = np.where(frac >= tissue_frac)
    return np.stack([gx * read_size, gy * read_size], 1).astype(np.int32)


def otsu_mask(path, W, H, thumb_max: int = 2048) -> np.ndarray:
    """Simple chroma-Otsu tissue mask (no weights), on a fast threaded overview."""
    ds = max(1, int(max(W, H) / thumb_max))
    a = wsi_io.fast_thumbnail(path, ds).astype(np.int16)
    chroma = (a.max(2) - a.min(2)).astype(np.uint8)          # colour on H&E; ~0 on white glass
    return ((chroma > _otsu(chroma)) & (a.max(2) < 240)).astype(np.float32)


def tissue_grid(path, W, H, read_size: int, args, device):
    """Return level-0 (x,y) coords of tissue tiles, using the selected QC backend."""
    if args.qc == "grandqc":
        from grandqc_mask import clean_tissue_mask           # same-dir module; loaded lazily
        art = None if args.grandqc_no_artifact else args.grandqc_artifact
        mask01 = clean_tissue_mask(path, device, args.grandqc_tissue, art)
    else:
        mask01 = otsu_mask(path, W, H)
    return _grid_from_mask(mask01, W, H, read_size, args.tissue_frac)


def cap_tiles(coords: np.ndarray, max_tiles: int, seed: int = 0) -> np.ndarray:
    """Spatial-coverage-preserving cap: stride-subsample the grid so no region is dropped wholesale."""
    if max_tiles <= 0 or len(coords) <= max_tiles:
        return coords
    step = int(np.ceil(len(coords) / max_tiles))
    return coords[::step][:max_tiles]


# ---------------------------------------------------------------- encoder
def load_virchow2(weights: str | None, device: str):
    import torch, timm
    from timm.layers import SwiGLUPacked
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2", pretrained=(weights is None),
        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
    )
    if weights is not None:
        from safetensors.torch import load_file
        sd = load_file(weights) if weights.endswith(".safetensors") else torch.load(weights, map_location="cpu")
        model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model


def embed_batch(model, tiles_u8: np.ndarray, device: str):
    """tiles_u8: [B,224,224,3] uint8 -> [B,2560] float32 (on cpu)."""
    import torch
    x = torch.from_numpy(tiles_u8).to(device).float().div_(255.0)
    x = (x - torch.tensor(IMAGENET_MEAN, device=device)) / torch.tensor(IMAGENET_STD, device=device)
    x = x.permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        out = model(x)                                   # [B, 261, 1280]
        emb = torch.cat([out[:, 0], out[:, N_PREFIX:].mean(1)], dim=-1)  # [B, 2560]
    return emb.float().cpu().numpy()


# ---------------------------------------------------------------- per-slide
def read_mpp(slide) -> float:
    try:
        mpp = float(slide.properties.get("openslide.mpp-x", "nan"))
        if 0.1 < mpp < 2.0:
            return mpp
    except Exception:
        pass
    return TARGET_MPP  # anonymized REG2026 slides carry no/ bogus MPP -> assume 20x


def extract_slide(path, model, device, args):
    import openslide
    # Anchor the deadline at PROCESS start (boot), so slow model-load is counted against the budget too. Falls
    # back to now-relative EXTRACT_BUDGET_S if boot time is unavailable (e.g. standalone CLI use).
    proc_start = float(os.environ.get("GOW_PROC_START") or 0.0)
    if proc_start > 0.0:
        wsi_io.DEADLINE = proc_start + TOTAL_BUDGET_S
    else:
        budget = getattr(args, "extract_budget_s", EXTRACT_BUDGET_S)
        wsi_io.DEADLINE = time.monotonic() + budget          # shared cutoff read by fast_thumbnail + resolve_tiled
    local, is_tmp = localize(path, args.tmpdir)              # stream s3:// to a tempfile if needed
    resolved, converted = local, None
    try:
        resolved, converted = wsi_io.resolve_tiled(local, args.tmpdir)   # striped (PIT_03) -> tiled via vips
        # Open through the multi-backend chain (openslide -> tiffslide -> ...): identical to a direct
        # openslide open on normal/transcoded slides (returns the openslide handle), but also reads a format
        # vips could not transcode instead of raising here. mpp/W/H are read the same way from any backend.
        s = wsi_io._open(resolved); mpp = read_mpp(s); W, H = s.level_dimensions[0]; s.close()
        read_size = int(round(TILE_PX * TARGET_MPP / mpp))   # region px at level 0 -> resized to 224
        coords = cap_tiles(tissue_grid(resolved, W, H, read_size, args, device), args.max_tiles)
        fallback = False
        if len(coords) == 0 and args.qc == "grandqc":        # GrandQC found no tissue -> permissive Otsu (don't lose the slide)
            coords = cap_tiles(_grid_from_mask(otsu_mask(resolved, W, H), W, H, read_size, args.tissue_frac), args.max_tiles)
            fallback = True
        n = len(coords)
        meta = {"mpp": mpp, "read_size": read_size, "n_tiles": n, "fallback": fallback}
        if n == 0 or args.dry_run:
            return None, coords, meta
        X = np.empty((n, 2560), np.float32)
        bs = args.batch_size
        embedded = 0
        for start in range(0, n, bs):
            if time.monotonic() > wsi_io.DEADLINE:           # cooperative cutoff at a Python boundary (no signal)
                meta["truncated"] = True                     # real partial bag (<= max_tiles); the cap is UNCHANGED
                break
            chunk = coords[start:start + bs]
            tiles = np.stack(wsi_io.read_tiles(resolved, chunk, read_size, TILE_PX, args.readers))
            X[start:start + len(chunk)] = embed_batch(model, tiles, device)
            embedded = start + len(chunk)
        if embedded == 0:                                    # ran out of budget before any tile -> fallback chain
            meta["n_tiles"] = 0
            return None, coords[:0], meta
        meta["n_tiles"] = embedded                           # happy path: embedded == n -> X[:n]/coords[:n] identical
        return X[:embedded], coords[:embedded], meta
    finally:
        wsi_io.DEADLINE = None                               # disarm so the next case starts unbounded
        if converted and os.path.exists(converted):
            os.remove(converted)                             # drop the vips-tiled temp
        if is_tmp and os.path.exists(local):
            os.remove(local)                                 # disk-frugal: drop the WSI after embedding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi"); ap.add_argument("--wsi-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", default=None, help="local Virchow2 safetensors (else hf-hub cache)")
    ap.add_argument("--max-tiles", type=int, default=20000)
    ap.add_argument("--tissue-frac", type=float, default=0.25)
    ap.add_argument("--qc", choices=["otsu", "grandqc"], default="otsu",
                    help="tissue/QC backend: simple Otsu mask, or GrandQC clean-tissue (artifact-filtered)")
    ap.add_argument("--grandqc-tissue", default="/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth")
    ap.add_argument("--grandqc-artifact", default="/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth")
    ap.add_argument("--grandqc-no-artifact", action="store_true", help="tissue-only GrandQC (skip artifact filtering)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--readers", type=int, default=16, help="parallel tile-read threads")
    ap.add_argument("--shard", default="0/1", help="i/N: process slides where index%N==i")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delete-wsi", action="store_true", help="embed-then-delete (disk-frugal)")
    ap.add_argument("--dry-run", action="store_true", help="tissue+tiling only; no torch/weights")
    ap.add_argument("--wsi-list", help="file of WSI paths/URIs (local or s3://...), one per line")
    ap.add_argument("--tmpdir", default="data/_tmp", help="scratch dir for streamed s3 slides")
    args = ap.parse_args()

    if args.wsi:
        paths = [args.wsi]
    elif args.wsi_list:
        paths = [ln.strip() for ln in open(args.wsi_list) if ln.strip()]
    else:
        paths = sorted(glob.glob(os.path.join(args.wsi_dir, "*.tif*")))
    i, N = (int(v) for v in args.shard.split("/"))
    paths = [p for k, p in enumerate(paths) if k % N == i]
    os.makedirs(args.out_dir, exist_ok=True)

    model = None if args.dry_run else load_virchow2(args.weights, args.device)
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out) and not args.overwrite and not args.dry_run:
            print(f"[skip] {stem}"); continue
        t0 = time.time()
        try:
            X, coords, meta = extract_slide(p, model, args.device, args)
        except Exception as e:
            print(f"[FAIL] {stem}: {e}"); continue
        dt = time.time() - t0
        print(f"[{'dry' if args.dry_run else 'ok'}] {stem}  tiles={meta['n_tiles']:>6}"
              f"{' FB' if meta.get('fallback') else '   '}  mpp={meta['mpp']:.3f} read={meta['read_size']}  {dt:.1f}s")
        if args.dry_run:
            continue
        if X is None:                                        # 0 tissue tiles -> skip, don't crash the worker
            print(f"[empty] {stem}"); continue
        np.savez_compressed(out, X=X.astype(np.float16), coords=coords.astype(np.int32),
                            mean=X.mean(0).astype(np.float16), meta=json.dumps(meta))
        if args.delete_wsi and not p.startswith("s3://"):
            os.remove(p)


if __name__ == "__main__":
    main()
