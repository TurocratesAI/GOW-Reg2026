"""Fast, format-robust WSI IO.

Two jobs, both bounded so no slide can wedge the container:
  1. OPEN ANY WSI the platform can serve -- multi-backend chain (openslide -> tiffslide -> cucim /
     wsidicom if importable). openslide stays PRIMARY so the fast tiled-JPEG path is byte-identical to
     before; the fallbacks only catch formats openslide refuses (striped / JPEG2000 / odd-compression
     TIFFs, DICOM-WSI).
  2. Build the low-res overview WITHOUT an O(area) full-res scan on pathological slides. For a normal
     small-tile slide (whether single-level like debug PIT_03 or 10-level-pyramid like a real test slide)
     the overview is the SAME threaded block-read of level 0 as before -- byte-identical mask/tiles/bag.
     Only when the level-0 atomic decode unit is huge (one tile or strip covering a large fraction of the
     plane -> every sub-region read_region re-decodes the whole unit -> the classic multi-minute hang) do
     we diverge: read an existing pyramid level, or a single whole-level decode, or a bounded strided
     sample -- never the whole plane block-by-block.

Key measured facts on REG2026 slides: openslide.get_thumbnail(2048) = ~88s (reads the whole slide
inefficiently), but per-thread read_region does ~2770 tiles/s on small-tile slides. So we build the
overview ourselves from threaded block-reads, and read tiles with per-thread handles.

A cooperative DEADLINE (time.monotonic seconds, set by extract_slide) is checked at Python loop
boundaries here so the loops cannot silently blow past the platform kill. It CANNOT interrupt a single
uninterruptible native read_region -- that residual worst case is bounded by the giant-unit handling
below plus (recommended) the process-level hard wall-clock documented in inference.py.
"""
import math
import os
import subprocess
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
_TLS = threading.local()

# Cooperative wall-clock deadline (time.monotonic() seconds) set by extract_slide; None = unbounded.
DEADLINE = None

# A level-0 page whose atomic decode unit (one tile, or width*rowsperstrip for a strip) exceeds this many
# pixels is "giant": a sub-region read_region forces a full-unit decode, so building the overview by
# per-block reads becomes O(area * n_blocks) = the multi-minute wedge. 2048*2048 = 4.19M px; normal WSI
# tiles (<=1024px) and modest strips stay well under it, so the normal block-read path is unchanged.
_GIANT_UNIT_PX = 2048 * 2048
# Largest level-0 area we will decode in ONE pass (via tifffile, which decodes each tile/strip EXACTLY
# once -- unlike openslide.read_region on a whole-plane region, which re-decodes the giant unit per internal
# 512-cell as its tile cache thrashes) when a giant slide has no pyramid. 7e8 px -> ~5GB peak RSS, safe on
# the 16 GiB box; 20000x20000 (4e8) qualifies, 60000x60000 (3.6e9) does not and takes the strided path.
_WHOLE_READ_PX = 700_000_000
_MAX_GIANT_BLOCKS = 64            # strided-sample cap for a giant slide too large to whole-decode


# ------------------------------------------------------------------ backends
def _suspicious(handle):
    """True if openslide likely probed a label/macro instead of the slide (level-0 long side < 2048),
    so we should fall through to a backend that sees the real series."""
    try:
        return max(handle.level_dimensions[0]) < 2048
    except Exception:
        return True


class _CuImageSlide:
    """Thin openslide-compatible adapter over cucim.CuImage (only used if cucim is importable)."""
    def __init__(self, path):
        from cucim import CuImage
        self._im = CuImage(path)
        res = self._im.resolutions
        self._lc = int(res["level_count"])
        self.level_dimensions = [tuple(int(v) for v in d) for d in res["level_dimensions"]]
        self.level_downsamples = [float(v) for v in res["level_downsamples"]]
        self.properties = {}

    @property
    def level_count(self):
        return self._lc

    def read_region(self, location, level, size):
        arr = np.asarray(self._im.read_region(location=list(location), size=list(size), level=level))
        return Image.fromarray(arr)

    def close(self):
        pass


class _WsiDicomSlide:
    """Thin openslide-compatible adapter over wsidicom (only used if wsidicom is importable). DICOM-WSI is
    a multi-file set in a directory; accept a member file or its parent dir. Defensive across API versions;
    any failure raises and _open falls through."""
    def __init__(self, path):
        from wsidicom import WsiDicom
        d = path if os.path.isdir(path) else os.path.dirname(path) or "."
        self.s = WsiDicom.open(d)
        levels = list(self.s.levels)
        self.level_dimensions = [(int(lv.size.width), int(lv.size.height)) for lv in levels]
        base = self.level_dimensions[0][0]
        self.level_downsamples = [base / wh[0] for wh in self.level_dimensions]
        self.properties = {}

    @property
    def level_count(self):
        return len(self.level_dimensions)

    def read_region(self, location, level, size):
        reg = self.s.read_region(location=tuple(location), level=level, size=tuple(size))
        return reg if isinstance(reg, Image.Image) else Image.fromarray(np.asarray(reg))

    def close(self):
        try:
            self.s.close()
        except Exception:
            pass


def _open(path):
    """Multi-backend open -> an openslide-compatible handle exposing level_dimensions, level_downsamples,
    level_count, read_region((x,y), level, (w,h)) -> PIL image, properties.get(...), close().

    Order: openslide (PRIMARY -- fastest; the fast tiled-JPEG path is byte-identical to before) ->
    tiffslide (tifffile+zarr; opens striped / JPEG2000 / odd-compression generic TIFFs openslide refuses)
    -> cucim -> wsidicom (each only if importable; DICOM-WSI + GPU decode). A slide openslide opens with a
    <2048px level-0 (a mis-probed label/macro) is rejected so it falls through to a backend that sees the
    real series. If nothing accepts it, the native openslide error is re-raised for the caller's fallback."""
    import openslide
    try:
        s = openslide.OpenSlide(path)
        if not _suspicious(s):
            return s                                          # normal slides land here -> identical behavior
        s.close()
    except Exception:
        pass
    try:
        import tiffslide
        s = tiffslide.TiffSlide(path)
        if s.level_dimensions and s.level_dimensions[0][0] >= 1:
            return s
        s.close()
    except Exception:
        pass
    for modname, adapter in (("cucim", _CuImageSlide), ("wsidicom", _WsiDicomSlide)):
        try:
            __import__(modname)
        except Exception:
            continue
        try:
            return adapter(path)
        except Exception:
            pass
    return openslide.OpenSlide(path)                          # let the native error surface -> fallback chain


def _giant_decode_unit(path):
    """True if the level-0 atomic decode unit is huge (see _GIANT_UNIT_PX). Any probe failure -> False
    (treated as a normal slide, so we never divert a slide we cannot classify)."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            pg = tf.pages[0]
            tw = int(getattr(pg, "tilewidth", 0) or 0)
            th = int(getattr(pg, "tilelength", 0) or 0)
            if getattr(pg, "is_tiled", False) and tw and th:
                unit = tw * th
            else:
                rps = pg.tags.get("RowsPerStrip")
                rps = int(rps.value) if rps is not None else int(getattr(pg, "imagelength", 0) or 0)
                unit = int(getattr(pg, "imagewidth", 0) or 0) * rps
            return unit > _GIANT_UNIT_PX
    except Exception:
        return False


# ------------------------------------------------------------------ transcode
def resolve_tiled(path, tmpdir):
    """openslide only opens TILED tiffs efficiently. If openslide REFUSES `path` (striped deflate BigTIFFs,
    JPEG2000-compressed tiles, other odd compressions), convert it to a tiled JPEG pyramid via the system
    `vips` binary so openslide + all reads work fast. If openslide already opens it (normal slides AND
    openslide-readable giant-tile slides) return it UNCHANGED -- byte-identical for normal slides; giant
    tiles are bounded downstream in fast_thumbnail. If vips also cannot read it, return it unchanged and let
    _open's tiffslide/cucim/wsidicom fallback read it directly. Returns (readable_path, converted_temp_or_None)."""
    import openslide
    try:
        openslide.OpenSlide(path).close()
        return path, None                                     # openslide-readable -> unchanged (byte-safe)
    except Exception:
        pass
    # openslide refused: broaden past OpenSlideUnsupportedFormatError to OpenSlideError (e.g. "Unsupported
    # TIFF compression: 33004" for JPEG2000) and anything else -> attempt a bounded vips transcode.
    try:
        os.makedirs(tmpdir, exist_ok=True)
        out = os.path.join(tmpdir, os.path.basename(path) + ".tiled.tif")
        # Bound the one monolithic native step a Python-boundary check cannot reach: on expiry subprocess.run
        # SIGKILLs the vips child and raises TimeoutExpired -> propagates to predict_chain_of_thought -> fallback.
        remaining = None if DEADLINE is None else max(1.0, DEADLINE - time.monotonic())
        subprocess.run(["vips", "tiffsave", path, out, "--tile", "--tile-width", "256",
                        "--tile-height", "256", "--compression", "jpeg", "--Q", "90",
                        "--pyramid", "--bigtiff"], check=True, timeout=remaining,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out, out
    except Exception:
        # vips cannot read it either (e.g. a giant single-tile "tile size out of range", or a multi-file
        # DICOM set): leave the path so _open's richer backends handle reads, bounded by DEADLINE downstream.
        return path, None


# ------------------------------------------------------------------ per-thread handle
def _osr(path):
    """Per-thread WSI handle (openslide/tiffslide serialize reads on ONE handle -> one per thread)."""
    d = getattr(_TLS, "h", None)
    if d is None:
        d = _TLS.h = {}
    if path not in d:
        d[path] = _open(path)
    return d[path]


def dims(path):
    s = _open(path)
    wh = s.level_dimensions[0]
    s.close()
    return (int(wh[0]), int(wh[1]))


# ------------------------------------------------------------------ overview
def _pick_level(handle, ds):
    """Coarsest pyramid level whose downsample is still <= the target ds (enough resolution). For the large
    ds a tissue thumbnail uses, this is never level 0 when a coarser level exists -> cheap read."""
    try:
        lds = list(handle.level_downsamples)
    except Exception:
        return 0
    cand = [i for i, d in enumerate(lds) if d <= ds + 1e-6]
    return max(cand, key=lambda i: lds[i]) if cand else int(np.argmax(lds))


def fast_thumbnail(path, downsample, region=2048, readers=12):
    """Downsampled RGB overview. NORMAL small-tile slides: threaded block-read of level 0 (byte-identical
    to the previous implementation, so masks/tiles/bags are unchanged). GIANT-decode-unit slides (which
    would otherwise wedge on a full-res block scan): read an existing pyramid level, or a single whole-level
    decode, or a bounded strided sample -- never the whole plane block-by-block."""
    h = _open(path)
    try:
        W, H = int(h.level_dimensions[0][0]), int(h.level_dimensions[0][1])
        ds = max(1.0, float(downsample))
        ow, oh = max(1, int(W // ds)), max(1, int(H // ds))

        if _giant_decode_unit(path):
            # (A) a pyramid exists -> read a coarse existing level in one shot (no full-res scan).
            if getattr(h, "level_count", 1) > 1:
                lvl = _pick_level(h, ds)
                lw, lh = int(h.level_dimensions[lvl][0]), int(h.level_dimensions[lvl][1])
                im = h.read_region((0, 0), lvl, (lw, lh)).convert("RGB")
                return np.asarray(im.resize((ow, oh), Image.BILINEAR), np.uint8)
            # (B) no pyramid but small enough -> decode the whole plane ONCE via tifffile (each tile/strip
            # decoded exactly once) then downsample. NOT h.read_region -- openslide would re-decode the giant
            # unit per internal cell and wedge. Falls through to strided if tifffile can't read it.
            if W * H <= _WHOLE_READ_PX:
                arr = _whole_plane(path)
                if arr is not None:
                    return np.asarray(Image.fromarray(arr).resize((ow, oh), Image.BILINEAR), np.uint8)
            # (C) no pyramid and too big to whole-decode -> bounded strided sample; gaps stay 255 (=
            # background = safe mask). Best-effort under DEADLINE; the residual worst case relies on the
            # process-level hard kill in inference.py.
            return _strided_thumbnail(path, W, H, ds, ow, oh, region, readers)

        # (D) NORMAL path: threaded block-read of the whole level 0 -- unchanged from before.
        thumb = np.full((oh, ow, 3), 255, np.uint8)
        blocks = [(x, y) for y in range(0, H, region) for x in range(0, W, region)]

        def rd(xy):
            x, y = xy
            w, h_ = min(region, W - x), min(region, H - y)
            im = _osr(path).read_region((x, y), 0, (w, h_)).convert("RGB")
            tw, th = max(1, int(w // ds)), max(1, int(h_ // ds))
            return int(x // ds), int(y // ds), np.asarray(im.resize((tw, th), Image.BILINEAR), np.uint8)

        with ThreadPoolExecutor(max_workers=readers) as ex:
            for ox, oy, arr in ex.map(rd, blocks):
                if DEADLINE is not None and time.monotonic() > DEADLINE:
                    break
                thumb[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
        return thumb
    finally:
        try:
            h.close()
        except Exception:
            pass


def _whole_plane(path):
    """Decode level 0 in a single tifffile pass (each tile/strip decoded once) -> HxWx3 uint8, or None if
    tifffile cannot read it. Used only for a giant-unit slide small enough for _WHOLE_READ_PX."""
    try:
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr, np.uint8)
    except Exception:
        return None


def _strided_thumbnail(path, W, H, ds, ow, oh, region, readers):
    """Bounded strided overview for a giant slide with no pyramid that is too large to whole-decode: read at
    most _MAX_GIANT_BLOCKS aligned blocks (each re-decodes its unit, so the COUNT is what we bound), break on
    DEADLINE, leave the rest as background. Guarantees completion instead of an O(area) scan."""
    thumb = np.full((oh, ow, 3), 255, np.uint8)
    reg = max(1, min(region, W, H))
    xs = list(range(0, W, reg))
    ys = list(range(0, H, reg))
    stride = max(1, int(math.ceil(math.sqrt(max(1, len(xs) * len(ys)) / _MAX_GIANT_BLOCKS))))
    blocks = [(x, y) for iy, y in enumerate(ys) if iy % stride == 0
              for ix, x in enumerate(xs) if ix % stride == 0]

    def rd(xy):
        x, y = xy
        w, h_ = min(reg, W - x), min(reg, H - y)
        im = _osr(path).read_region((x, y), 0, (w, h_)).convert("RGB")
        tw, th = max(1, int(w // ds)), max(1, int(h_ // ds))
        return int(x // ds), int(y // ds), np.asarray(im.resize((tw, th), Image.BILINEAR), np.uint8)

    with ThreadPoolExecutor(max_workers=readers) as ex:
        for ox, oy, arr in ex.map(rd, blocks):
            if DEADLINE is not None and time.monotonic() > DEADLINE:
                break
            thumb[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return thumb


def read_tiles(path, coords, read_size, tile_px, readers=16):
    """Read tiles with per-thread handles -> list of [tile_px,tile_px,3] uint8. Unchanged for normal slides;
    a giant-tile slide re-decodes per read but the per-batch DEADLINE in extract_slide bounds the total."""
    def rd(xy):
        try:
            t = _osr(path).read_region((int(xy[0]), int(xy[1])), 0, (read_size, read_size)).convert("RGB")
            if read_size != tile_px:
                t = t.resize((tile_px, tile_px), Image.BILINEAR)
            return np.asarray(t, np.uint8)
        except Exception:
            # Unreadable/corrupt region: emit a blank (glass=255) tile instead of raising, so one bad tile can't
            # abort read_tiles -> extract_slide -> the generic fallback. Batch stays len-aligned; MIL ignores blanks.
            return np.full((tile_px, tile_px, 3), 255, np.uint8)
    with ThreadPoolExecutor(max_workers=readers) as ex:
        return list(ex.map(rd, coords))
