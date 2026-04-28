#!/usr/bin/env python3
"""
FastAPI server wrapping the Multi-Panel Figure Builder Python backend.
Launched as a Tauri sidecar — communicates via HTTP on a random port.
"""
from __future__ import annotations
import argparse
import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Fix matplotlib font cache for PyInstaller --onefile (must be before matplotlib import)
_mpl_config = os.path.join(os.path.expanduser("~"), ".multipanelfigure", "mplconfig")
os.makedirs(_mpl_config, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_config

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()

import numpy as np
from PIL import Image, ImageFile
# Accept slightly truncated images (common in scientific microscopy)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from models import (
    FigureConfig, PanelInfo, AxisLabel, HeaderLevel, HeaderGroup,
    StyledSegment, save_config, load_config, save_project, load_project,
    _to_dict, _from_dict,
)
from image_processing import process_panel
from figure_builder import assemble_figure

# ── Inline helpers (avoid importing helpers.py which requires Qt) ──
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def find_fonts(extra_dirs=None):
    fonts = {}
    import platform
    if platform.system() == "Windows":
        sys_dirs = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
    else:
        sys_dirs = [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    app_dir = str(SCRIPT_DIR)
    app_dirs = [app_dir, os.path.join(app_dir, "..")]
    persistent_dir = str(Path.home() / ".multipanelfigure" / "fonts")
    search = sys_dirs + app_dirs + [persistent_dir]
    if extra_dirs:
        search.extend(extra_dirs)
    for d in search:
        if not os.path.isdir(d):
            continue
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith((".ttf", ".otf", ".ttc")):
                    if fn not in fonts:
                        fonts[fn] = os.path.join(d, fn)
        except PermissionError:
            continue
    return fonts

# ── App State ──────────────────────────────────────────────────────────────

app = FastAPI(title="Multi-Panel Figure Builder API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Support Private Network Access (CORS-RFC1918) for Windows WebView2.
# Uses a raw ASGI middleware (avoids BaseHTTPMiddleware compatibility issues).
class PrivateNetworkMiddleware:
    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.wrapped_app(scope, receive, send)
            return

        # Handle OPTIONS preflight for private network access
        if scope.get("method") == "OPTIONS":
            headers_dict = {k: v for k, v in scope.get("headers", [])}
            if b"access-control-request-private-network" in headers_dict:
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        [b"access-control-allow-origin", b"*"],
                        [b"access-control-allow-methods", b"*"],
                        [b"access-control-allow-headers", b"*"],
                        [b"access-control-allow-private-network", b"true"],
                    ],
                })
                await send({"type": "http.response.body", "body": b""})
                return

        # Add private network header to all responses
        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append([b"access-control-allow-private-network", b"true"])
                message = {**message, "headers": headers}
            await send(message)

        await self.wrapped_app(scope, receive, send_with_header)

app.add_middleware(PrivateNetworkMiddleware)

@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Global state (single-user desktop app)
cfg = FigureConfig()
cfg.ensure_grid()
loaded_images: Dict[str, Image.Image] = {}
# Panel-internal images (e.g. 3D-volume renders assigned to a panel) that
# should NOT appear in the user-facing media timeline. Still editable through
# the panel itself.
hidden_images: set = set()
min_dims = (512, 512)
custom_fonts: Dict[str, bytes] = {}

# Video support
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts'}
loaded_videos: Dict[str, str] = {}  # name → temp file path
video_frames: Dict[str, int] = {}   # name → selected frame number

# Z-stack TIFF support
loaded_zstacks: Dict[str, str] = {}    # name → temp file path for multi-frame TIFF
zstack_frames: Dict[str, int] = {}     # name → selected frame number
zstack_counts: Dict[str, int] = {}     # name → total frame count

def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS

def _extract_video_frame(video_path: str, frame_num: int = 0) -> Image.Image:
    """Extract a single frame from a video file using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return Image.new("RGB", (100, 100), "black")
    # OpenCV uses BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


# Per-panel video frame cache. Keys are (video_name, frame_num, mtime)
# so a video file replacement invalidates entries for that name. Bounded
# LRU-ish — drop the oldest when we exceed _VIDEO_FRAME_CACHE_MAX so a
# pathological frame-scrub session can't run away with memory.
_video_frame_cache: Dict[tuple, Image.Image] = {}
_VIDEO_FRAME_CACHE_MAX = 128


def _get_panel_image(panel) -> Optional[Image.Image]:
    """Return the PIL image for *panel*, honouring its per-panel
    `frame` field for video sources so multiple panels using the
    same video name can each show a different frame. Falls back to
    loaded_images for non-video sources or unknown names.
    """
    name = getattr(panel, "image_name", None)
    if not name:
        return None
    if name in loaded_videos:
        path = loaded_videos[name]
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        frame_num = int(getattr(panel, "frame", 0) or 0)
        key = (name, frame_num, mtime)
        cached = _video_frame_cache.get(key)
        if cached is not None:
            return cached
        img = _extract_video_frame(path, frame_num)
        _video_frame_cache[key] = img
        if len(_video_frame_cache) > _VIDEO_FRAME_CACHE_MAX:
            first_key = next(iter(_video_frame_cache))
            if first_key != key:
                del _video_frame_cache[first_key]
        return img
    return loaded_images.get(name)

def _is_tiff(filename: str) -> bool:
    return Path(filename).suffix.lower() in {'.tif', '.tiff'}

def _extract_tiff_frame(tiff_path: str, frame_num: int = 0) -> Image.Image:
    """Extract a specific frame from a multi-frame TIFF (z-stack)."""
    img = Image.open(tiff_path)
    try:
        img.seek(frame_num)
    except EOFError:
        img.seek(0)
    return img.convert("RGB")

def _get_zstack_info(tiff_path: str) -> dict:
    """Get z-stack TIFF metadata."""
    img = Image.open(tiff_path)
    n_frames = getattr(img, "n_frames", 1)
    return {
        "frame_count": n_frames,
        "width": img.size[0],
        "height": img.size[1],
    }


def _get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": 0.0,
    }
    if info["fps"] > 0:
        info["duration_sec"] = info["frame_count"] / info["fps"]
    cap.release()
    return info


def _recalc_min_dims():
    global min_dims
    if loaded_images:
        ws = [img.size[0] for img in loaded_images.values()]
        hs = [img.size[1] for img in loaded_images.values()]
        min_dims = (min(ws), min(hs))


def _cfg_json():
    return _to_dict(cfg)


# ── Config Endpoints ───────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return _cfg_json()


class ConfigUpdate(BaseModel):
    config: dict


@app.put("/api/config")
def put_config(body: ConfigUpdate):
    global cfg
    cfg = _from_dict(FigureConfig, body.config)
    cfg.ensure_grid()
    return _cfg_json()


class GridUpdate(BaseModel):
    rows: int = 2
    cols: int = 2
    spacing: float = 0.02


@app.patch("/api/config/grid")
def patch_grid(body: GridUpdate):
    global cfg
    cfg.rows = body.rows
    cfg.cols = body.cols
    cfg.spacing = body.spacing
    cfg.ensure_grid()
    return _cfg_json()


class PanelUpdate(BaseModel):
    panel: dict


@app.patch("/api/config/panel/{r}/{c}")
def patch_panel(r: int, c: int, body: PanelUpdate):
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    cfg.panels[r][c] = _from_dict(PanelInfo, body.panel)
    return _to_dict(cfg.panels[r][c])


class LabelsUpdate(BaseModel):
    labels: list


@app.patch("/api/config/column-labels")
def patch_col_labels(body: LabelsUpdate):
    cfg.column_labels = [_from_dict(AxisLabel, d) for d in body.labels]
    return {"ok": True}


@app.patch("/api/config/row-labels")
def patch_row_labels(body: LabelsUpdate):
    cfg.row_labels = [_from_dict(AxisLabel, d) for d in body.labels]
    return {"ok": True}


class HeadersUpdate(BaseModel):
    headers: list


@app.patch("/api/config/column-headers")
def patch_col_headers(body: HeadersUpdate):
    cfg.column_headers = [_from_dict(HeaderLevel, d) for d in body.headers]
    return {"ok": True}


@app.patch("/api/config/row-headers")
def patch_row_headers(body: HeadersUpdate):
    cfg.row_headers = [_from_dict(HeaderLevel, d) for d in body.headers]
    return {"ok": True}


class BackgroundUpdate(BaseModel):
    background: str


@app.patch("/api/config/background")
def patch_background(body: BackgroundUpdate):
    cfg.background = body.background
    return {"ok": True}


# ── Image Endpoints ────────────────────────────────────────────────────────

@app.get("/api/images")
def list_images():
    used = set()
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            name = cfg.panels[r][c].image_name
            if name:
                used.add(name)
    # Filter out panel-internal (hidden) images from the timeline listing.
    visible_names = [n for n in loaded_images.keys() if n not in hidden_images]
    return {
        "names": visible_names,
        "used": list(used),
        "hidden": list(hidden_images),
    }


# ── Specialized image loaders ─────────────────────────────────────────────

RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.raf'}

def _load_raw(data: bytes, filename: str) -> Image.Image:
    """Load a RAW camera file using rawpy."""
    import rawpy
    tmp_path = os.path.join(tempfile.gettempdir(), f"mpf_raw_{filename}")
    try:
        with open(tmp_path, 'wb') as tmp:
            tmp.write(data)
        with rawpy.imread(tmp_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=False)
        return Image.fromarray(rgb)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _load_nd2(data: bytes, filename: str) -> Image.Image:
    """Load a Nikon .nd2 microscopy file."""
    import nd2
    tmp_path = os.path.join(tempfile.gettempdir(), f"mpf_nd2_{filename}")
    try:
        with open(tmp_path, 'wb') as tmp:
            tmp.write(data)
        with nd2.ND2File(tmp_path) as f:
            arr = f.asarray()
            # Handle multi-dimensional arrays (TZCYX)
            while arr.ndim > 3:
                arr = arr[0]  # take first of outermost dimension
            if arr.ndim == 3:
                if arr.shape[0] <= 4:  # Channels-first (CYX)
                    if arr.shape[0] == 3:
                        arr = np.moveaxis(arr, 0, -1)  # CYX -> YXC
                    else:
                        arr = arr[0]  # single channel
            # Normalize to 8-bit
            if arr.dtype != np.uint8:
                vmin, vmax = float(arr.min()), float(arr.max())
                if vmax > vmin:
                    arr = ((arr.astype(np.float32) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, 'L').convert('RGB')
            return Image.fromarray(arr).convert('RGB')
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _is_raw(filename: str) -> bool:
    return Path(filename).suffix.lower() in RAW_EXTENSIONS

def _is_nd2(filename: str) -> bool:
    return Path(filename).suffix.lower() == '.nd2'


@app.post("/api/images/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    names = []
    for f in files:
        data = await f.read()
        try:
            if _is_video(f.filename):
                # Save video to temp file and extract first frame as the image
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix)
                tmp.write(data)
                tmp.close()
                loaded_videos[f.filename] = tmp.name
                video_frames[f.filename] = 0
                img = _extract_video_frame(tmp.name, 0)
                loaded_images[f.filename] = img
                names.append(f.filename)
            elif _is_raw(f.filename):
                img = _load_raw(data, f.filename)
                loaded_images[f.filename] = img
                names.append(f.filename)
            elif _is_nd2(f.filename):
                img = _load_nd2(data, f.filename)
                loaded_images[f.filename] = img
                names.append(f.filename)
            elif _is_tiff(f.filename):
                # Check if it's a multi-frame TIFF (z-stack)
                img_obj = Image.open(io.BytesIO(data))
                n_frames = getattr(img_obj, "n_frames", 1)
                if n_frames > 1:
                    # Save to temp file for seeking (io.BytesIO doesn't persist)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix)
                    tmp.write(data)
                    tmp.close()
                    loaded_zstacks[f.filename] = tmp.name
                    zstack_frames[f.filename] = 0
                    zstack_counts[f.filename] = n_frames
                    img = _extract_tiff_frame(tmp.name, 0)
                    loaded_images[f.filename] = img
                else:
                    img = img_obj.convert("RGB")
                    loaded_images[f.filename] = img
                names.append(f.filename)
            else:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                loaded_images[f.filename] = img
                names.append(f.filename)
        except Exception as e:
            raise HTTPException(400, f"Failed to load {f.filename}: {e}")
    _recalc_min_dims()
    # Return thumbnails as base64
    thumbnails = {}
    for name in names:
        thumbnails[name] = _thumb_b64(loaded_images[name])
    return {"names": names, "thumbnails": thumbnails}


# ── Video Endpoints ────────────────────────────────────────────────────────

@app.get("/api/video/{name}/info")
def get_video_info(name: str):
    if name not in loaded_videos:
        raise HTTPException(404, f"Video '{name}' not found")
    return _get_video_info(loaded_videos[name])

@app.get("/api/video/{name}/frame/{frame_num}")
def get_video_frame(name: str, frame_num: int):
    if name not in loaded_videos:
        raise HTTPException(404, f"Video '{name}' not found")
    img = _extract_video_frame(loaded_videos[name], frame_num)
    # Update the loaded image to this frame
    loaded_images[name] = img
    video_frames[name] = frame_num
    _recalc_min_dims()
    # Return a larger preview (max 1200px) for display, not just a thumbnail
    preview = img.copy()
    max_dim = 1200
    if max(preview.size) > max_dim:
        ratio = max_dim / max(preview.size)
        preview = preview.resize((int(preview.size[0] * ratio), int(preview.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"frame": frame_num, "width": img.size[0], "height": img.size[1],
            "thumbnail": preview_b64}

@app.get("/api/video/list")
def list_videos():
    return {"videos": list(loaded_videos.keys())}


# ── Z-Stack TIFF Endpoints ────────────────────────────────────────────────

@app.get("/api/zstack/{name}/info")
def get_zstack_info(name: str):
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    return _get_zstack_info(loaded_zstacks[name])

@app.get("/api/zstack/{name}/frame/{frame_num}")
def get_zstack_frame(name: str, frame_num: int, row: Optional[int] = None, col: Optional[int] = None):
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    img = _extract_tiff_frame(loaded_zstacks[name], frame_num)

    # If row/col provided, update that specific panel's image assignment
    # so different panels can show different frames of the same z-stack
    if row is not None and col is not None:
        panel_key = f"__zstack_{name}_r{row}c{col}"
        loaded_images[panel_key] = img
        zstack_frames[panel_key] = frame_num
        # Also update the panel's image_name to use the panel-specific key
        if row < cfg.rows and col < cfg.cols:
            cfg.panels[row][col].image_name = panel_key
        # Register as a virtual z-stack so info lookups work
        loaded_zstacks[panel_key] = loaded_zstacks[name]
        zstack_counts[panel_key] = zstack_counts.get(name, 1)
    else:
        loaded_images[name] = img
        zstack_frames[name] = frame_num

    _recalc_min_dims()
    preview = img.copy()
    max_dim = 1200
    if max(preview.size) > max_dim:
        ratio = max_dim / max(preview.size)
        preview = preview.resize((int(preview.size[0] * ratio), int(preview.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"frame": frame_num, "width": img.size[0], "height": img.size[1],
            "thumbnail": preview_b64}

@app.get("/api/zstack/list")
def list_zstacks():
    return {"zstacks": list(loaded_zstacks.keys())}


class ZStackProjectRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1   # -1 = last frame
    method: str = "max"   # max | avg | min

@app.post("/api/zstack/{name}/project")
def project_zstack(name: str, body: ZStackProjectRequest):
    """Create a projection (max/avg/min) from a range of z-stack frames."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    tiff_path = loaded_zstacks[name]
    img_obj = Image.open(tiff_path)
    n_frames = getattr(img_obj, "n_frames", 1)

    start = max(0, body.start_frame)
    end = body.end_frame if body.end_frame >= 0 else n_frames - 1
    end = min(end, n_frames - 1)
    if start > end:
        start, end = end, start

    # Read frames into numpy stack
    frames = []
    for i in range(start, end + 1):
        img_obj.seek(i)
        frame = np.array(img_obj.convert("RGB"), dtype=np.float32)
        frames.append(frame)

    if not frames:
        raise HTTPException(400, "No frames in range")

    stack = np.stack(frames, axis=0)

    # Apply projection
    if body.method == "max":
        result = np.max(stack, axis=0).astype(np.uint8)
    elif body.method == "min":
        result = np.min(stack, axis=0).astype(np.uint8)
    elif body.method == "avg":
        result = np.mean(stack, axis=0).astype(np.uint8)
    else:
        raise HTTPException(400, f"Unknown projection method: {body.method}")

    projected = Image.fromarray(result)
    loaded_images[name] = projected
    _recalc_min_dims()

    # Return preview
    preview = projected.copy()
    max_dim = 1200
    if max(preview.size) > max_dim:
        ratio = max_dim / max(preview.size)
        preview = preview.resize((int(preview.size[0] * ratio), int(preview.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "method": body.method,
        "start_frame": start,
        "end_frame": end,
        "width": projected.size[0],
        "height": projected.size[1],
        "thumbnail": preview_b64,
    }


class ZStackNiftiRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1
    max_dim: int = 256  # bigger ok for NiiVue (it's optimized)
    z_spacing: float = 1.0  # z-axis voxel spacing relative to x/y (e.g. 2.0 = thicker slices)

@app.post("/api/zstack/{name}/nifti")
def get_zstack_nifti(name: str, body: ZStackNiftiRequest):
    """Return z-stack as NIfTI (.nii) bytes, base64-encoded."""
    import sys
    print(f"[nifti] Request for {name} range {body.start_frame}-{body.end_frame}", file=sys.stderr, flush=True)

    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    tiff_path = loaded_zstacks[name]

    # Try tifffile first (fast), fall back to PIL loop
    arr = None
    try:
        import tifffile
        print(f"[nifti] Using tifffile to read {tiff_path}", file=sys.stderr, flush=True)
        arr = tifffile.imread(tiff_path)
        print(f"[nifti] tifffile read {arr.shape} {arr.dtype}", file=sys.stderr, flush=True)
    except ImportError:
        print(f"[nifti] tifffile not available, falling back to PIL", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[nifti] tifffile failed: {e}, falling back to PIL", file=sys.stderr, flush=True)

    if arr is None:
        # Fallback: read with PIL seek
        print(f"[nifti] Reading with PIL seek loop", file=sys.stderr, flush=True)
        img_obj = Image.open(tiff_path)
        n = getattr(img_obj, "n_frames", 1)
        frames = []
        for i in range(n):
            img_obj.seek(i)
            frames.append(np.array(img_obj.convert("L"), dtype=np.uint8))
        arr = np.stack(frames, axis=0) if frames else np.zeros((1, 64, 64), dtype=np.uint8)

    try:
        import nibabel as nib
    except ImportError as e:
        raise HTTPException(500, f"nibabel not bundled: {e}")

    # Ensure 3D shape: (depth, height, width)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 4:
        # Multi-channel: take first channel or max across channels
        arr = arr.max(axis=-1) if arr.shape[-1] <= 4 else arr[..., 0]
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

    depth = arr.shape[0]

    # Slice to requested range
    start = max(0, body.start_frame)
    end = body.end_frame if body.end_frame >= 0 else depth - 1
    end = min(end, depth - 1)
    arr = arr[start:end + 1]

    # Normalize to uint8
    if arr.dtype != np.uint8:
        if arr.max() > 0:
            arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    # Downsample if needed
    d, h, w = arr.shape
    if max(w, h, d) > body.max_dim:
        import cv2
        scale = body.max_dim / max(w, h, d)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        new_d = max(1, int(d * scale))
        # Downsample depth by subsampling
        z_step = max(1, d // new_d)
        resized = []
        for i in range(0, d, z_step):
            if len(resized) >= new_d:
                break
            resized.append(cv2.resize(arr[i], (new_w, new_h), interpolation=cv2.INTER_AREA))
        arr = np.stack(resized, axis=0)

    # NiiVue expects (x, y, z) axis order — transpose from (z, y, x)
    arr_nii = np.transpose(arr, (2, 1, 0))  # (w, h, d) → x, y, z

    # Build affine with requested z-spacing so NiiVue renders the volume
    # with user-specified z-step thickness.
    z_spacing = max(0.05, float(body.z_spacing))
    affine = np.diag([1.0, 1.0, z_spacing, 1.0]).astype(np.float32)

    # Create NIfTI image
    nii = nib.Nifti1Image(arr_nii, affine=affine)
    nii.header.set_data_dtype(np.uint8)
    # Explicitly set pixdim for x, y, z
    nii.header["pixdim"][1] = 1.0
    nii.header["pixdim"][2] = 1.0
    nii.header["pixdim"][3] = z_spacing

    # Serialize to bytes
    buf = io.BytesIO()
    file_map = nib.Nifti1Image.make_file_map({"image": buf, "header": buf})
    nii.file_map = file_map
    nii.to_file_map()
    data_bytes = buf.getvalue()
    data_b64 = base64.b64encode(data_bytes).decode("ascii")

    return {
        "data": data_b64,
        "width": arr.shape[2],
        "height": arr.shape[1],
        "depth": arr.shape[0],
    }


class ZStackMipsRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1
    colormap: str = "gray"
    rotation_frames: int = 36       # number of rotation frames to render
    include_rotation: bool = False  # pre-render rotation? (slower)
    max_dim: int = 128


@app.post("/api/zstack/{name}/mips")
def get_zstack_mips(name: str, body: ZStackMipsRequest):
    """Fast MIP views + optional rotation animation frames."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = _load_volume(name, body.start_frame, body.end_frame, body.max_dim)
    depth, height, width = vol.shape
    norm = vol.astype(np.float32) / 255.0

    def _render_2d(data: np.ndarray, title: str) -> str:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(data, cmap=body.colormap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color="white")
        ax.axis("off")
        ax.set_facecolor("black")
        fig.patch.set_facecolor("#1c1c1e")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Fast MIPs
    mip_xy = _render_2d(np.max(norm, axis=0), "MIP XY (top)")
    mip_xz = _render_2d(np.max(norm, axis=1), "MIP XZ (front)")
    mip_yz = _render_2d(np.max(norm, axis=2), "MIP YZ (side)")

    result = {
        "mip_xy": mip_xy,
        "mip_xz": mip_xz,
        "mip_yz": mip_yz,
        "rotation_frames": [],
    }

    if body.include_rotation:
        # Pre-render rotation frames (volume voxel scatter)
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        mask = norm > 0.3
        zz, yy, xx = np.where(mask)
        max_points = 15000
        if len(zz) > max_points:
            idx = np.random.choice(len(zz), max_points, replace=False)
            zz, yy, xx = zz[idx], yy[idx], xx[idx]
        vals = norm[zz, yy, xx] if len(zz) > 0 else np.array([])
        cmap = plt.get_cmap(body.colormap)

        rotation_frames = []
        n = body.rotation_frames
        for i in range(n):
            azim = -180 + (360 * i / n)
            fig = plt.figure(figsize=(6, 6), dpi=80)
            ax = fig.add_subplot(111, projection='3d')
            if len(zz) > 0:
                colors = cmap(vals)
                colors[:, 3] = vals * 0.6
                ax.scatter(xx, yy, zz, c=colors, s=2, depthshade=True)
            ax.set_xlim(0, width); ax.set_ylim(0, height); ax.set_zlim(0, depth)
            ax.view_init(elev=25, azim=azim)
            ax.set_axis_off()
            ax.set_facecolor("black")
            fig.patch.set_facecolor("#1c1c1e")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            rotation_frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        result["rotation_frames"] = rotation_frames

    return result


class ZStackVolumeRenderRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1
    elev: float = 30.0        # camera elevation angle
    azim: float = -60.0       # camera azimuth angle
    threshold: float = 0.3    # intensity threshold (0-1)
    z_spacing: float = 1.0    # z-axis scale factor
    colormap: str = "gray"    # matplotlib colormap name
    width: int = 800          # output image width
    height: int = 600         # output image height
    method: str = "surface"   # surface | mip_xy | mip_xz | mip_yz
    show_axes: bool = True    # show 3D axes/gridlines
    zoom: float = 1.0         # zoom factor (>1 = zoom in)
    fast: bool = False        # low-res fast mode (for drag preview)

# Cache the loaded volume to avoid re-reading on every render
_volume_cache: Dict[str, np.ndarray] = {}

def _load_volume(name: str, start: int, end: int, max_dim: int = 128) -> np.ndarray:
    """Load z-stack volume with caching and downsampling."""
    cache_key = f"{name}_{start}_{end}_{max_dim}"
    if cache_key in _volume_cache:
        return _volume_cache[cache_key]

    tiff_path = loaded_zstacks[name]
    img_obj = Image.open(tiff_path)
    n_frames = getattr(img_obj, "n_frames", 1)
    end = min(end if end >= 0 else n_frames - 1, n_frames - 1)
    start = max(0, start)

    total = end - start + 1
    step = max(1, total // max_dim)

    frames = []
    for i in range(start, end + 1, step):
        img_obj.seek(i)
        frame = img_obj.convert("L")
        fw, fh = frame.size
        if max(fw, fh) > max_dim:
            s = max_dim / max(fw, fh)
            frame = frame.resize((max(1, int(fw * s)), max(1, int(fh * s))), Image.NEAREST)
        frames.append(np.array(frame, dtype=np.uint8))

    vol = np.stack(frames, axis=0) if frames else np.zeros((1, 64, 64), dtype=np.uint8)
    _volume_cache[cache_key] = vol
    # Keep cache small
    if len(_volume_cache) > 5:
        oldest = next(iter(_volume_cache))
        del _volume_cache[oldest]
    return vol


@app.post("/api/zstack/{name}/volume-render")
def render_volume(name: str, body: ZStackVolumeRenderRequest):
    """Server-side 3D volume rendering — returns a PNG image."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    vol = _load_volume(name, body.start_frame, body.end_frame)
    depth, height, width = vol.shape
    norm_vol = vol.astype(np.float32) / 255.0

    if body.method == "mip_xy":
        # Maximum intensity projection — XY view
        mip = np.max(norm_vol, axis=0)
        fig, ax = plt.subplots(figsize=(body.width / 100, body.height / 100), dpi=100)
        ax.imshow(mip, cmap=body.colormap, vmin=0, vmax=1)
        ax.set_title("MIP — XY (top-down)", fontsize=10)
        ax.axis("off")
    elif body.method == "mip_xz":
        mip = np.max(norm_vol, axis=1)
        fig, ax = plt.subplots(figsize=(body.width / 100, body.height / 100), dpi=100)
        ax.imshow(mip, cmap=body.colormap, vmin=0, vmax=1, aspect=body.z_spacing)
        ax.set_title("MIP — XZ (front)", fontsize=10)
        ax.axis("off")
    elif body.method == "mip_yz":
        mip = np.max(norm_vol, axis=2)
        fig, ax = plt.subplots(figsize=(body.width / 100, body.height / 100), dpi=100)
        ax.imshow(mip, cmap=body.colormap, vmin=0, vmax=1, aspect=body.z_spacing)
        ax.set_title("MIP — YZ (side)", fontsize=10)
        ax.axis("off")
    else:
        # 3D isosurface-like rendering using voxel scatter
        # Fast mode: use lower resolution for responsive drag
        render_scale = 0.5 if body.fast else 1.0
        w_px = int(body.width * render_scale)
        h_px = int(body.height * render_scale)
        fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Threshold and downsample for 3D scatter
        mask = norm_vol > body.threshold
        max_points = 15000 if body.fast else 50000
        zz, yy, xx = np.where(mask)
        if len(zz) > max_points:
            idx = np.random.choice(len(zz), max_points, replace=False)
            zz, yy, xx = zz[idx], yy[idx], xx[idx]

        if len(zz) > 0:
            vals = norm_vol[zz, yy, xx]
            cmap = plt.get_cmap(body.colormap)
            colors = cmap(vals)
            colors[:, 3] = vals * 0.6
            ax.scatter(xx, yy, zz * body.z_spacing, c=colors,
                       s=2 if body.fast else 1, depthshade=not body.fast)

        # Apply zoom by shrinking axis limits around center
        cx, cy, cz = width / 2, height / 2, depth * body.z_spacing / 2
        zoom = max(0.1, body.zoom)
        hx, hy, hz = width / (2 * zoom), height / (2 * zoom), depth * body.z_spacing / (2 * zoom)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_zlim(cz - hz, cz + hz)
        ax.view_init(elev=body.elev, azim=body.azim)

        if body.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.tick_params(colors='white', labelsize=6)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
        else:
            ax.set_axis_off()

        ax.set_facecolor("black")
        fig.patch.set_facecolor("#1c1c1e")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {"image": img_b64, "width": body.width, "height": body.height}


class ZStackVolumeSaveRequest(ZStackVolumeRenderRequest):
    format: str = "PNG"
    quality: int = 95
    path: str = ""


def _render_volume_to_pil(name: str, body: ZStackVolumeRenderRequest) -> Image.Image:
    """Render the volume and return a PIL Image (reused by save/as-panel endpoints)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    vol = _load_volume(name, body.start_frame, body.end_frame)
    depth, height, width = vol.shape
    norm_vol = vol.astype(np.float32) / 255.0

    if body.method in ("mip_xy", "mip_xz", "mip_yz"):
        axis = {"mip_xy": 0, "mip_xz": 1, "mip_yz": 2}[body.method]
        mip = np.max(norm_vol, axis=axis)
        fig, ax = plt.subplots(figsize=(body.width / 100, body.height / 100), dpi=100)
        aspect = body.z_spacing if body.method != "mip_xy" else "equal"
        ax.imshow(mip, cmap=body.colormap, vmin=0, vmax=1, aspect=aspect)
        if body.show_axes:
            ax.set_title(f"MIP — {body.method.upper().replace('MIP_', '')}", fontsize=10)
        else:
            ax.axis("off")
    else:
        fig = plt.figure(figsize=(body.width / 100, body.height / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        mask = norm_vol > body.threshold
        max_points = 50000
        zz, yy, xx = np.where(mask)
        if len(zz) > max_points:
            idx = np.random.choice(len(zz), max_points, replace=False)
            zz, yy, xx = zz[idx], yy[idx], xx[idx]
        if len(zz) > 0:
            vals = norm_vol[zz, yy, xx]
            cmap = plt.get_cmap(body.colormap)
            colors = cmap(vals)
            colors[:, 3] = vals * 0.6
            ax.scatter(xx, yy, zz * body.z_spacing, c=colors, s=1, depthshade=True)

        cx, cy, cz = width / 2, height / 2, depth * body.z_spacing / 2
        zoom = max(0.1, body.zoom)
        hx, hy, hz = width / (2 * zoom), height / (2 * zoom), depth * body.z_spacing / (2 * zoom)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_zlim(cz - hz, cz + hz)
        ax.view_init(elev=body.elev, azim=body.azim)

        if body.show_axes:
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.tick_params(colors='white', labelsize=6)
            ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.zaxis.label.set_color('white')
        else:
            ax.set_axis_off()

        ax.set_facecolor("black")
        fig.patch.set_facecolor("#1c1c1e" if body.show_axes else "white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


@app.post("/api/zstack/{name}/volume-save")
def save_volume_render(name: str, body: ZStackVolumeSaveRequest):
    """Render and save the volume to disk in the specified format."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    if not body.path:
        raise HTTPException(400, "Path is required")

    img = _render_volume_to_pil(name, body)
    fmt = body.format.upper()
    save_kwargs: dict = {}
    save_path = os.path.expanduser(body.path.strip() if body.path else "")
    # Pre-flight disk-space check with a conservative size estimate
    # (raw RGB is a hard upper bound for PNG/TIFF output).
    try:
        import shutil as _shutil
        target_dir = os.path.dirname(save_path) or "."
        estimated = img.size[0] * img.size[1] * 4 + 16 * 1024 * 1024
        usage = _shutil.disk_usage(target_dir)
        if usage.free < estimated:
            raise HTTPException(507, f"Not enough disk space: need ~{estimated//(1024*1024)} MB, have {usage.free//(1024*1024)} MB free.")
    except HTTPException:
        raise
    except Exception:
        pass
    if fmt == "JPEG":
        save_kwargs["quality"] = max(1, min(100, body.quality))
        img.save(save_path, format="JPEG", **save_kwargs)
    elif fmt == "TIFF":
        img.save(save_path, format="TIFF", compression="tiff_lzw")
    else:
        img.save(save_path, format="PNG")
    # Post-write verification — 0-byte means the write silently failed.
    try:
        sz = os.path.getsize(save_path)
    except OSError:
        sz = 0
    if sz <= 0:
        try: os.unlink(save_path)
        except OSError: pass
        raise HTTPException(500, f"Save failed: output file is empty ({save_path}).")
    return {"ok": True, "path": save_path, "size_bytes": sz}


class ZStackVolumeAsPanelRequest(ZStackVolumeRenderRequest):
    row: int = 0
    col: int = 0


class CanvasAsPanelRequest(BaseModel):
    image_name: str  # original z-stack name (for naming)
    row: int
    col: int
    data_b64: str    # base64 PNG from canvas
    hidden: bool = True  # keep out of the media timeline by default

@app.post("/api/save-canvas-as-panel")
def save_canvas_as_panel(body: CanvasAsPanelRequest):
    """Save a base64 PNG (from canvas) and assign as panel image.

    By default the saved image is marked 'hidden' so it does NOT show up in
    the media-timeline, but the panel still references it — subsequent edits
    (crop, adjust, etc.) applied to the panel operate on this 3D-rendered
    image rather than the original z-stack.
    """
    data = base64.b64decode(body.data_b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    panel_name = f"volume_3d_{body.image_name}_r{body.row}c{body.col}.png"

    # If this panel already had a hidden panel-internal image, clean it up so
    # we don't accumulate stale entries each time the user re-renders.
    if body.row < cfg.rows and body.col < cfg.cols:
        prev_name = cfg.panels[body.row][body.col].image_name
        if prev_name and prev_name in hidden_images and prev_name != panel_name:
            loaded_images.pop(prev_name, None)
            hidden_images.discard(prev_name)

    loaded_images[panel_name] = img
    if body.hidden:
        hidden_images.add(panel_name)
    else:
        hidden_images.discard(panel_name)
    if body.row < cfg.rows and body.col < cfg.cols:
        cfg.panels[body.row][body.col].image_name = panel_name
    _recalc_min_dims()
    return {"ok": True, "image_name": panel_name, "hidden": body.hidden}


@app.post("/api/zstack/{name}/volume-as-panel")
def use_volume_as_panel(name: str, body: ZStackVolumeAsPanelRequest):
    """Render the volume and assign it as a panel image."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    img = _render_volume_to_pil(name, body)
    # Create a unique image name for this volume render
    volume_name = f"volume_{name}_r{body.row}c{body.col}_{body.method}.png"
    loaded_images[volume_name] = img
    # Assign to panel
    if body.row < cfg.rows and body.col < cfg.cols:
        cfg.panels[body.row][body.col].image_name = volume_name
    _recalc_min_dims()
    return {"ok": True, "image_name": volume_name}


class ZStackVolumeDataRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1
    max_dim: int = 96   # small volume for client-side rendering

@app.post("/api/zstack/{name}/volume")
def get_volume_data(name: str, body: ZStackVolumeDataRequest):
    """Return z-stack as a raw 3D uint8 array for client-side rendering."""
    if name not in loaded_zstacks:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    vol = _load_volume(name, body.start_frame, body.end_frame, body.max_dim)
    depth, height, width = vol.shape
    data_b64 = base64.b64encode(vol.tobytes()).decode("ascii")
    return {
        "data": data_b64,
        "width": width,
        "height": height,
        "depth": depth,
    }


@app.delete("/api/images/{name}")
def delete_image(name: str):
    if name in loaded_images:
        del loaded_images[name]
        hidden_images.discard(name)
        # Clean up z-stack temp file if present
        if name in loaded_zstacks:
            try:
                os.unlink(loaded_zstacks[name])
            except Exception:
                pass
            del loaded_zstacks[name]
            zstack_frames.pop(name, None)
            zstack_counts.pop(name, None)
        # Clear panel assignments
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                if cfg.panels[r][c].image_name == name:
                    cfg.panels[r][c].image_name = ""
        _recalc_min_dims()
        return {"ok": True}
    raise HTTPException(404, f"Image '{name}' not found")


@app.get("/api/images/{name}/thumb")
def get_thumb(name: str):
    if name not in loaded_images:
        raise HTTPException(404, f"Image '{name}' not found")
    return {"thumbnail": _thumb_b64(loaded_images[name])}


@app.get("/api/images/{name}/info")
def get_image_info(name: str):
    """Return original full-resolution dimensions of an image."""
    if name not in loaded_images:
        raise HTTPException(404, f"Image '{name}' not found")
    img = loaded_images[name]
    return {"width": img.size[0], "height": img.size[1]}


def _thumb_b64(img: Image.Image, max_side: int = 256) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Histogram Endpoint (raw, from original image, supports 16-bit) ─────────

@app.get("/api/histogram/{r}/{c}")
def get_histogram(r: int, c: int):
    """Get RGB histogram from the ORIGINAL loaded image (pre-adjustment).
    For 16-bit images, returns the full range histogram (0-65535 binned to 256)."""
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    if not panel.image_name or panel.image_name not in loaded_images:
        return {"r": [], "g": [], "b": [], "bit_depth": 8, "max_val": 255}
    img = _get_panel_image(panel) or loaded_images[panel.image_name]
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    max_val = int(arr.max())
    bit_depth = 16 if max_val > 255 else 8
    bins = 256
    r_hist = np.histogram(arr[:, :, 0], bins=bins, range=(0, max_val + 1))[0].tolist()
    g_hist = np.histogram(arr[:, :, 1], bins=bins, range=(0, max_val + 1))[0].tolist()
    b_hist = np.histogram(arr[:, :, 2], bins=bins, range=(0, max_val + 1))[0].tolist()
    return {"r": r_hist, "g": g_hist, "b": b_hist, "bit_depth": bit_depth, "max_val": max_val}


# ── Panel Preview Endpoint ─────────────────────────────────────────────────

@app.get("/api/panel-preview/{r}/{c}")
def get_panel_preview(r: int, c: int):
    """Get a processed preview of a single panel with current settings applied."""
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    if not panel.image_name or panel.image_name not in loaded_images:
        return {"image": ""}
    # For edit dialog preview: skip ALL overlays (labels, symbols, scale bar, lines,
    # areas, zoom inset) — they are rendered as interactive UI overlays in the frontend.
    # This gives a clean base image for the user to work with.
    panel_copy = _from_dict(PanelInfo, _to_dict(panel))
    panel_copy.add_zoom_inset = False
    panel_copy.add_scale_bar = False
    panel_copy.labels = []
    panel_copy.symbols = []
    panel_copy.lines = []
    panel_copy.areas = []
    processed = process_panel(_get_panel_image(panel) or loaded_images[panel.image_name], panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
    proc_w, proc_h = processed.size  # full processed dimensions before thumbnail
    processed.thumbnail((1200, 1200), Image.LANCZOS)
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode("ascii"),
            "processed_width": proc_w, "processed_height": proc_h}


class PanelPatchAndPreview(BaseModel):
    panel: dict


@app.post("/api/panel-patch-preview/{r}/{c}")
def patch_panel_and_preview(r: int, c: int, body: PanelPatchAndPreview):
    """Atomically patch a panel AND return its processed preview in one call.
    Eliminates race conditions from separate PATCH + GET requests."""
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    # 1. Patch
    cfg.panels[r][c] = _from_dict(PanelInfo, body.panel)
    panel = cfg.panels[r][c]
    # 2. Generate preview — skip ALL overlays (rendered as interactive UI overlays)
    if not panel.image_name or panel.image_name not in loaded_images:
        return {"panel": _to_dict(panel), "image": ""}
    panel_copy = _from_dict(PanelInfo, _to_dict(panel))
    panel_copy.add_zoom_inset = False
    panel_copy.add_scale_bar = False
    panel_copy.labels = []
    panel_copy.symbols = []
    panel_copy.lines = []
    panel_copy.areas = []
    processed = process_panel(_get_panel_image(panel) or loaded_images[panel.image_name], panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
    proc_w, proc_h = processed.size
    processed.thumbnail((1200, 1200), Image.LANCZOS)
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    return {
        "panel": _to_dict(panel),
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "processed_width": proc_w, "processed_height": proc_h,
    }


# ── Single-panel rendered preview (with matplotlib overlays) ──────────────

@app.get("/api/panel-rendered-preview/{r}/{c}")
def get_panel_rendered_preview(r: int, c: int):
    """Render a single panel through matplotlib with scale bars, labels, and
    symbols — using the SAME rendering functions as the final figure output.
    Produces an image with exact pixel dimensions matching the base preview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from figure_builder import _font_props, _add_panel_scale_bars, _add_panel_labels, _add_panel_symbols

    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    if not panel.image_name or panel.image_name not in loaded_images:
        return {"image": ""}

    # Process base image (no labels/symbols/zoom — those are rendered via matplotlib/SVG)
    # Temporarily disable zoom inset to prevent double rendering with SVG overlay
    saved_zoom = panel.add_zoom_inset
    panel.add_zoom_inset = False
    processed = process_panel(
        _get_panel_image(panel) or loaded_images[panel.image_name], panel, min_dims, loaded_images,
        skip_labels=True, skip_symbols=True)
    panel.add_zoom_inset = saved_zoom

    iw, ih = processed.size
    # Match the figure builder's reference_inches (3.0) so that
    # text at N points occupies the same fraction of the panel as
    # in the final output.  DPI = pixels / inches.
    reference_inches = 3.0
    preview_dpi = max(72, iw / reference_inches)
    fig_w = iw / preview_dpi   # = reference_inches = 3.0
    fig_h = ih / preview_dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=preview_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(processed), aspect='auto', extent=[0, iw, ih, 0])
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.axis('off')

    # Use the SAME scale bar rendering as the full figure
    axes_grid = np.array([[ax]])
    _add_panel_scale_bars(fig, axes_grid, cfg, 1, 1,
                          [[processed]], panel_override=(r, c))

    # Use the SAME label rendering as the full figure
    _add_panel_labels(fig, axes_grid, cfg, 1, 1,
                      [[processed]], panel_override=(r, c))

    # Use the SAME symbol rendering as the full figure
    # Pass original (uncropped) image for consistent size scaling
    orig_img = _get_panel_image(panel) or loaded_images[panel.image_name]
    _add_panel_symbols(fig, axes_grid, cfg, 1, 1,
                       [[processed]], panel_override=(r, c),
                       original_images=[[orig_img]])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=preview_dpi, pad_inches=0)
    plt.close(fig)

    result_img = Image.open(buf)
    result_img.thumbnail((1200, 1200), Image.LANCZOS)
    out_buf = io.BytesIO()
    result_img.convert("RGB").save(out_buf, format="PNG")
    return {"image": base64.b64encode(out_buf.getvalue()).decode("ascii")}


# ── Auto-Adjust Endpoint ──────────────────────────────────────────────────

class AutoAdjustRequest(BaseModel):
    type: str   # "levels" | "contrast" | "white_balance"

@app.post("/api/auto-adjust/{r}/{c}")
def auto_adjust(r: int, c: int, body: AutoAdjustRequest):
    """Compute optimal adjustment values from the original image."""
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    if not panel.image_name or panel.image_name not in loaded_images:
        raise HTTPException(400, "No image assigned to panel")

    img = (_get_panel_image(panel) or loaded_images[panel.image_name]).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    adjustments = {}

    if body.type == "levels":
        # Per-channel percentile-based levels
        for ch_idx, suffix in enumerate(["r", "g", "b"]):
            ch = arr[:, :, ch_idx]
            p1 = float(np.percentile(ch, 1))
            p99 = float(np.percentile(ch, 99))
            ib = max(0, min(254, int(round(p1))))
            iw = max(ib + 1, min(255, int(round(p99))))
            adjustments[f"input_black_{suffix}"] = ib
            adjustments[f"input_white_{suffix}"] = iw
        adjustments["brightness"] = 1.0
        adjustments["contrast"] = 1.0

    elif body.type == "contrast":
        p5, p95 = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
        spread = max(p95 - p5, 1.0)
        adjustments["contrast"] = round(min(2.0, max(0.5, 180.0 / spread)), 3)

    elif body.type == "white_balance":
        r_mean = float(arr[:, :, 0].mean())
        g_mean = float(arr[:, :, 1].mean())
        b_mean = float(arr[:, :, 2].mean())
        overall = (r_mean + g_mean + b_mean) / 3.0
        adjustments["color_temperature"] = round(max(-100, min(100, (b_mean - r_mean) / 255.0 * 100)), 1)
        adjustments["tint"] = round(max(-100, min(100, (overall - g_mean) / 255.0 * 100)), 1)

    else:
        raise HTTPException(400, f"Unknown auto-adjust type: {body.type}")

    return {"adjustments": adjustments}


# ── Magic Wand Selection Endpoint ──────────────────────────────────────────

class MagicWandRequest(BaseModel):
    x_pct: float  # click X as percentage (0-100)
    y_pct: float  # click Y as percentage (0-100)
    tolerance: int = 30  # color tolerance 0-255
    # Optional overrides for local (unsaved) panel state
    rotation: Optional[float] = None
    crop: Optional[List[int]] = None
    crop_image: Optional[bool] = None

@app.post("/api/magic-wand/{r}/{c}")
def magic_wand_select(r: int, c: int, body: MagicWandRequest):
    """Perform flood-fill selection from a click point, return boundary polygon."""
    import numpy as np
    from collections import deque

    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    if not panel.image_name or panel.image_name not in loaded_images:
        raise HTTPException(400, "No image assigned to panel")

    import cv2

    # Use the image with rotation + crop applied (skip heavy color adjustments for speed)
    # Use local overrides if provided (EditPanelDialog sends unsaved state)
    img = (_get_panel_image(panel) or loaded_images[panel.image_name]).copy().convert("RGB")
    rotation = body.rotation if body.rotation is not None else (getattr(panel, 'rotation', 0) or 0)
    crop = body.crop if body.crop is not None else (panel.crop if panel.crop_image else None)
    crop_image = body.crop_image if body.crop_image is not None else panel.crop_image
    if rotation:
        img = img.rotate(-rotation, expand=True, resample=Image.BICUBIC)
    if crop_image and crop and len(crop) == 4:
        img = img.crop(crop)

    # Downsample for speed — max 600px on longest side
    orig_w, orig_h = img.size
    max_dim = 600
    if max(orig_w, orig_h) > max_dim:
        scale_f = max_dim / max(orig_w, orig_h)
        small = img.resize((int(orig_w * scale_f), int(orig_h * scale_f)), Image.LANCZOS)
    else:
        small = img
        scale_f = 1.0

    arr = np.array(small.convert("RGB"))
    h, w = arr.shape[:2]

    # Convert percentage to pixel coords
    cx = int(body.x_pct / 100 * w)
    cy = int(body.y_pct / 100 * h)
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    tol = body.tolerance

    # OpenCV floodFill — robust, fast C++ implementation
    # FLOODFILL_FIXED_RANGE: compare each pixel to SEED (not neighbor) — like Photoshop
    mask = np.zeros((h + 2, w + 2), np.uint8)  # mask must be 2px larger
    flood_flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(arr.copy(), mask, (cx, cy), 0,
                  loDiff=(tol, tol, tol), upDiff=(tol, tol, tol),
                  flags=flood_flags)
    # mask[1:-1, 1:-1] contains the filled region (255 = filled)
    region = mask[1:-1, 1:-1] == 255

    if not region.any():
        return {"points": [], "pixel_count": 0}

    # Use OpenCV findContours for robust contour extraction
    region_uint8 = region.astype(np.uint8) * 255
    contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"points": [], "pixel_count": 0}

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Simplify contour using Douglas-Peucker approximation
    # epsilon controls smoothness: larger = fewer points, smoother
    perimeter = cv2.arcLength(largest, True)
    epsilon = perimeter * 0.005  # 0.5% of perimeter — smooth but accurate
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Convert to percentage coordinates
    points = []
    for pt in approx:
        px, py = pt[0]
        points.append([round(float(px) / w * 100, 2), round(float(py) / h * 100, 2)])

    pixel_count = int(region.sum() / (scale_f ** 2)) if scale_f != 1.0 else int(region.sum())
    return {"points": points, "pixel_count": pixel_count, "smooth": True}


# ── Measurements Endpoint ──────────────────────────────────────────────────

@app.get("/api/measurements")
def get_measurements():
    """Compute all line/area measurements across all panels."""
    from models import compute_line_measurement, compute_area_measurement
    results = []
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            panel = cfg.panels[r][c]
            if not panel.image_name or panel.image_name not in loaded_images:
                continue
            img = _get_panel_image(panel) or loaded_images[panel.image_name]
            iw, ih = img.size
            # Apply crop dimensions if cropped
            if panel.crop_image and panel.crop:
                iw = panel.crop[2] - panel.crop[0]
                ih = panel.crop[3] - panel.crop[1]
            mpp = panel.scale_bar.micron_per_pixel if panel.scale_bar else 1.0
            label = f"R{r+1}C{c+1}"
            # Line measurements
            for line in (panel.lines or []):
                if line.show_measure and len(line.points) >= 2:
                    unit = getattr(line, 'measure_unit', 'um')
                    text = line.measure_text or compute_line_measurement(
                        line.points, iw, ih, mpp, unit)
                    results.append({"panel": label, "name": line.name, "type": "line", "value": text})
            # Area measurements
            for area in (panel.areas or []):
                if area.show_measure and len(area.points) >= 2:
                    unit = getattr(area, 'measure_unit', 'um')
                    text = area.measure_text or compute_area_measurement(
                        area.points, area.shape, iw, ih, mpp, unit)
                    results.append({"panel": label, "name": area.name, "type": "area", "value": text})
    return {"measurements": results}


# ── Preview Endpoint ───────────────────────────────────────────────────────

@app.post("/api/preview")
def generate_preview():
    rows, cols = cfg.rows, cfg.cols
    # For large grids, cap preview image resolution to keep rendering fast
    total_panels = rows * cols
    max_preview_px = 1200 if total_panels <= 9 else 800 if total_panels <= 25 else 500

    # First pass: process panels WITHOUT labels/symbols/scale bar — those are
    # rendered via matplotlib AFTER normalize for immutable font sizing
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.image_name and panel.image_name in loaded_images:
                src_img = _get_panel_image(panel) or loaded_images[panel.image_name]
                img = process_panel(
                    src_img, panel,
                    min_dims, loaded_images, skip_labels=True, skip_symbols=True)
                # Downscale for preview if image is large
                if img and max(img.size) > max_preview_px:
                    scale = max_preview_px / max(img.size)
                    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
                row_imgs.append(img)
            else:
                row_imgs.append(None)
        processed.append(row_imgs)

    # Compute per-column max width and per-row max height from actual images
    col_max_w = [max((processed[r][c].size[0] for r in range(rows)
                      if processed[r][c] is not None), default=min_dims[0]) for c in range(cols)]
    row_max_h = [max((processed[r][c].size[1] for c in range(cols)
                      if processed[r][c] is not None), default=min_dims[1]) for r in range(rows)]

    # Second pass: fill empty panels with correctly-sized placeholders
    for r in range(rows):
        for c in range(cols):
            if processed[r][c] is None:
                processed[r][c] = Image.new("RGB", (col_max_w[c], row_max_h[r]), "white")

    # Adjacent Panel zoom insets
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.add_zoom_inset and panel.zoom_inset and panel.zoom_inset.inset_type == "Adjacent Panel":
                zi = panel.zoom_inset
                ar, ac = r, c
                if zi.side == "Top": ar -= 1
                elif zi.side == "Bottom": ar += 1
                elif zi.side == "Left": ac -= 1
                elif zi.side == "Right": ac += 1
                if 0 <= ar < rows and 0 <= ac < cols:
                    ext_name = getattr(zi, 'separate_image_name', '') or ''
                    if ext_name and ext_name not in ('', 'select') and ext_name in loaded_images:
                        ext_img = loaded_images[ext_name].convert("RGB")
                        xi = getattr(zi, 'x_inset', 0) or 0
                        yi = getattr(zi, 'y_inset', 0) or 0
                        wi = getattr(zi, 'width_inset', ext_img.size[0]) or ext_img.size[0]
                        hi = getattr(zi, 'height_inset', ext_img.size[1]) or ext_img.size[1]
                        xi = max(0, min(xi, ext_img.size[0]-1))
                        yi = max(0, min(yi, ext_img.size[1]-1))
                        wi = max(1, min(wi, ext_img.size[0]-xi))
                        hi = max(1, min(hi, ext_img.size[1]-yi))
                        region = ext_img.crop((xi, yi, xi+wi, yi+hi))
                        zw = max(1, int(wi * zi.zoom_factor))
                        zh = max(1, int(hi * zi.zoom_factor))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region
                    else:
                        main_img = processed[r][c]
                        miw, mih = main_img.size
                        p_src = cfg.panels[r][c]
                        if p_src.crop_image and p_src.crop and len(p_src.crop) == 4:
                            fw = p_src.crop[2] - p_src.crop[0]
                            fh = p_src.crop[3] - p_src.crop[1]
                        elif p_src.image_name and p_src.image_name in loaded_images:
                            fw, fh = loaded_images[p_src.image_name].size
                        else:
                            fw, fh = miw, mih
                        scx = miw / max(fw, 1)
                        scy = mih / max(fh, 1)
                        # Inset by rectangle border width so we crop only
                        # pixels INSIDE the selection box, not the border
                        bw = max(1, getattr(zi, 'rectangle_width', 2) or 2)
                        bw_x = bw * scx
                        bw_y = bw * scy
                        cx1 = max(0, int(zi.x * scx + bw_x))
                        cy1 = max(0, int(zi.y * scy + bw_y))
                        cx2 = min(miw, int((zi.x + zi.width) * scx - bw_x))
                        cy2 = min(mih, int((zi.y + zi.height) * scy - bw_y))
                        cx2 = max(cx1 + 1, cx2)
                        cy2 = max(cy1 + 1, cy2)
                        region = main_img.crop((cx1, cy1, cx2, cy2))
                        zw = max(1, int(zi.width * zi.zoom_factor * scx))
                        zh = max(1, int(zi.height * zi.zoom_factor * scy))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region

    # Build full-res sizes for zoom inset line positioning
    full_res_sizes = {}
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.image_name and panel.image_name in loaded_images:
                orig_img = _get_panel_image(panel) or loaded_images[panel.image_name]
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes[(r, c)] = orig_img.size

    fig_bytes = assemble_figure(cfg, processed, full_res_sizes=full_res_sizes)
    fig_img = Image.open(io.BytesIO(fig_bytes))
    # Always convert to PNG for browser display (TIFF isn't supported by browsers)
    png_buf = io.BytesIO()
    fig_img.convert("RGBA").save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "image": b64,
        "width": fig_img.width,
        "height": fig_img.height,
        "format": "png",
    }


# ── Save/Export Endpoints ──────────────────────────────────────────────────

class SaveFigureRequest(BaseModel):
    path: str
    format: str = "TIFF"
    background: str = "White"
    dpi: int = 300


@app.post("/api/figure/save")
def save_figure(body: SaveFigureRequest):
    cfg.output_format = body.format
    cfg.background = body.background
    # Generate
    rows, cols = cfg.rows, cfg.cols
    # First pass: process all panels that have images
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.image_name and panel.image_name in loaded_images:
                row_imgs.append(process_panel(
                    _get_panel_image(panel) or loaded_images[panel.image_name], panel,
                    min_dims, loaded_images, skip_labels=True, skip_symbols=True))
            else:
                row_imgs.append(None)
        processed.append(row_imgs)

    # Compute per-column max width and per-row max height from actual images
    col_max_w = [max((processed[r][c].size[0] for r in range(rows)
                      if processed[r][c] is not None), default=min_dims[0]) for c in range(cols)]
    row_max_h = [max((processed[r][c].size[1] for c in range(cols)
                      if processed[r][c] is not None), default=min_dims[1]) for r in range(rows)]

    # Second pass: fill empty panels with correctly-sized placeholders
    for r in range(rows):
        for c in range(cols):
            if processed[r][c] is None:
                processed[r][c] = Image.new("RGB", (col_max_w[c], row_max_h[r]), "white")

    # Adjacent Panel zoom insets (must match preview pipeline)
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.add_zoom_inset and panel.zoom_inset and panel.zoom_inset.inset_type == "Adjacent Panel":
                zi = panel.zoom_inset
                ar, ac = r, c
                if zi.side == "Top": ar -= 1
                elif zi.side == "Bottom": ar += 1
                elif zi.side == "Left": ac -= 1
                elif zi.side == "Right": ac += 1
                if 0 <= ar < rows and 0 <= ac < cols:
                    ext_name = getattr(zi, 'separate_image_name', '') or ''
                    if ext_name and ext_name not in ('', 'select') and ext_name in loaded_images:
                        ext_img = loaded_images[ext_name].convert("RGB")
                        xi = getattr(zi, 'x_inset', 0) or 0
                        yi = getattr(zi, 'y_inset', 0) or 0
                        wi = getattr(zi, 'width_inset', ext_img.size[0]) or ext_img.size[0]
                        hi = getattr(zi, 'height_inset', ext_img.size[1]) or ext_img.size[1]
                        xi = max(0, min(xi, ext_img.size[0]-1))
                        yi = max(0, min(yi, ext_img.size[1]-1))
                        wi = max(1, min(wi, ext_img.size[0]-xi))
                        hi = max(1, min(hi, ext_img.size[1]-yi))
                        region = ext_img.crop((xi, yi, xi+wi, yi+hi))
                        zw = max(1, int(wi * zi.zoom_factor))
                        zh = max(1, int(hi * zi.zoom_factor))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region
                    else:
                        main_img = processed[r][c]
                        miw, mih = main_img.size
                        p_src = cfg.panels[r][c]
                        if p_src.crop_image and p_src.crop and len(p_src.crop) == 4:
                            fw = p_src.crop[2] - p_src.crop[0]
                            fh = p_src.crop[3] - p_src.crop[1]
                        elif p_src.image_name and p_src.image_name in loaded_images:
                            fw, fh = loaded_images[p_src.image_name].size
                        else:
                            fw, fh = miw, mih
                        scx = miw / max(fw, 1)
                        scy = mih / max(fh, 1)
                        bw = max(1, getattr(zi, 'rectangle_width', 2) or 2)
                        bw_x = bw * scx
                        bw_y = bw * scy
                        cx1 = max(0, int(zi.x * scx + bw_x))
                        cy1 = max(0, int(zi.y * scy + bw_y))
                        cx2 = min(miw, int((zi.x + zi.width) * scx - bw_x))
                        cy2 = min(mih, int((zi.y + zi.height) * scy - bw_y))
                        cx2 = max(cx1 + 1, cx2)
                        cy2 = max(cy1 + 1, cy2)
                        region = main_img.crop((cx1, cy1, cx2, cy2))
                        zw = max(1, int(zi.width * zi.zoom_factor * scx))
                        zh = max(1, int(zi.height * zi.zoom_factor * scy))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region

    full_res_sizes2 = {}
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.image_name and panel.image_name in loaded_images:
                orig_img = _get_panel_image(panel) or loaded_images[panel.image_name]
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes2[(r, c)] = orig_img.size
    fig_bytes = assemble_figure(cfg, processed, dpi=body.dpi, full_res_sizes=full_res_sizes2)
    save_path = os.path.expanduser(body.path.strip())
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Pre-flight disk-space check — we know how many bytes we're about to
    # write; refuse if the target volume can't hold them (plus a small
    # margin). Prevents silent 0-byte output files on low-space machines.
    try:
        import shutil as _shutil
        target_dir = os.path.dirname(save_path) or "."
        usage = _shutil.disk_usage(target_dir)
        required = len(fig_bytes) + 16 * 1024 * 1024  # +16 MB headroom
        if usage.free < required:
            raise HTTPException(
                507,
                f"Not enough disk space to save figure: need "
                f"{required // (1024*1024)} MB, have "
                f"{usage.free // (1024*1024)} MB free on the target volume.",
            )
    except HTTPException:
        raise
    except Exception as e:
        # Don't block save if disk-usage call itself fails; just log.
        import sys
        print(f"[save_figure] disk_usage check failed: {e}", file=sys.stderr, flush=True)

    with open(save_path, "wb") as f:
        f.write(fig_bytes)

    # Post-write verification — if the file somehow ended up 0 bytes or
    # much smaller than expected (partial write, quota, etc.), delete it
    # and surface an error rather than leaving a broken output behind.
    try:
        written = os.path.getsize(save_path)
    except OSError:
        written = 0
    if written < max(1, len(fig_bytes) // 2):
        try: os.unlink(save_path)
        except OSError: pass
        raise HTTPException(
            500,
            f"Save failed: wrote {written} bytes, expected {len(fig_bytes)} (possible disk-full or permission issue).",
        )

    return {"ok": True, "path": save_path, "size_bytes": written}


# ─────────────────────────────────────────────────────────────────────
# Video render — animate panels with `play_range` enabled across their
# [frame_start, frame_end] ranges and mux into an mp4/avi.
# ─────────────────────────────────────────────────────────────────────

import threading as _threading
import uuid as _uuid
import subprocess as _subprocess
import sys as _sys


def _ffmpeg_path() -> str:
    """Resolve the ffmpeg binary path.

    In the production Tauri bundle the sidecar (api-server) is invoked
    as a child process, and ffmpeg is bundled as a sibling externalBin
    via tauri.conf.json. So look next to sys.executable first. In dev
    mode (running api_server.py with the system Python) sys.executable
    points at the Python interpreter, in which case we fall back to
    `ffmpeg` from PATH.
    """
    if getattr(_sys, "frozen", False):
        exec_dir = os.path.dirname(_sys.executable)
        candidate = os.path.join(
            exec_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg",
        )
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"

# Job state — all access through _render_jobs_lock.
_render_jobs: Dict[str, Dict] = {}
_render_jobs_lock = _threading.Lock()


class RenderVideoRequest(BaseModel):
    path: str
    format: str = "mp4"           # "mp4" | "avi"
    fps: int = 30                 # 1-60
    background: str = "White"
    dpi: int = 150
    audio_panel_image_name: Optional[str] = None  # if set, must match a play_range panel's image_name


@app.get("/api/figure/render-video/ffmpeg-available")
def render_video_ffmpeg_available():
    """Probe whether ffmpeg is invocable. The 0.1.130 build bundles a
    static LGPL ffmpeg next to the sidecar via Tauri externalBin, so
    this should always be true in production. In dev mode it falls back
    to PATH ffmpeg, where availability depends on the user's machine."""
    try:
        result = _subprocess.run([_ffmpeg_path(), "-version"], capture_output=True, timeout=2)
        return {"available": result.returncode == 0}
    except (FileNotFoundError, _subprocess.TimeoutExpired, Exception):
        return {"available": False}


@app.post("/api/figure/render-video")
def render_video_start(body: RenderVideoRequest):
    if body.fps < 1 or body.fps > 60:
        raise HTTPException(400, "fps must be 1-60")
    if body.format.lower() not in ("mp4", "avi"):
        raise HTTPException(400, "format must be mp4 or avi")

    rows, cols = cfg.rows, cfg.cols
    range_panels: List[Tuple[int, int, PanelInfo]] = []
    for r in range(rows):
        for c in range(cols):
            p = cfg.panels[r][c]
            if getattr(p, "play_range", False) and p.image_name and p.image_name in loaded_videos:
                range_panels.append((r, c, p))
    if not range_panels:
        raise HTTPException(400, "No panels have 'Play range in video export' enabled.")

    total_frames = max((p.frame_end - p.frame_start + 1) for _, _, p in range_panels)
    if total_frames < 1:
        raise HTTPException(400, "Computed total frame count is 0; check the play-range bounds.")

    save_path = os.path.expanduser(body.path.strip())
    if not save_path.lower().endswith("." + body.format.lower()):
        save_path = save_path + "." + body.format.lower()
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Disk-space pre-flight: rough estimate of 800 KB per output frame
    # (ballpark for a 1080p mp4 at modest quality).
    try:
        import shutil as _shutil
        usage = _shutil.disk_usage(out_dir)
        required = total_frames * 800 * 1024 + 64 * 1024 * 1024  # +64 MB headroom
        if usage.free < required:
            raise HTTPException(
                507,
                f"Not enough disk space: need ~{required // (1024*1024)} MB, have "
                f"{usage.free // (1024*1024)} MB free on the target volume.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # disk check is best-effort

    job_id = _uuid.uuid4().hex
    with _render_jobs_lock:
        _render_jobs[job_id] = {
            "status": "running",
            "current": 0,
            "total": total_frames,
            "output_path": save_path,
            "error": None,
            "cancel_requested": False,
        }

    t = _threading.Thread(
        target=_render_video_worker,
        args=(job_id, body, total_frames, range_panels, save_path),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "total_frames": total_frames}


def _render_video_worker(job_id: str, body: RenderVideoRequest,
                         total_frames: int,
                         range_panels: List[Tuple[int, int, "PanelInfo"]],
                         save_path: str) -> None:
    import cv2 as _cv2
    import numpy as _np
    from io import BytesIO as _BytesIO
    from PIL import Image as _PIL

    rows, cols = cfg.rows, cfg.cols
    captures: Dict[str, "_cv2.VideoCapture"] = {}
    writer = None
    try:
        # Open one VideoCapture per UNIQUE video name (not per panel) so
        # multiple panels sharing the same video reuse a single capture
        # and just seek to per-panel frames per output index. Without
        # this de-dup, two panels using the same video would share their
        # range — exactly the user-reported bug.
        unique_video_names = {p.image_name for _, _, p in range_panels if p.image_name in loaded_videos}
        for name in unique_video_names:
            path = loaded_videos.get(name)
            if path:
                captures[name] = _cv2.VideoCapture(path)

        for frame_idx in range(total_frames):
            with _render_jobs_lock:
                if _render_jobs[job_id]["cancel_requested"]:
                    raise RuntimeError("Cancelled by user")

            # Per-panel image for this output frame_idx, keyed by (r, c)
            # — replaces the previous shared loaded_images[name] mutation
            # so two panels using the same video can each show their own
            # frame independently.
            panel_images: Dict[Tuple[int, int], Image.Image] = {}
            for r, c, p in range_panels:
                range_len = max(1, p.frame_end - p.frame_start + 1)
                if frame_idx < range_len:
                    src_frame = p.frame_start + frame_idx
                elif getattr(p, "return_to_selected_on_end", False):
                    # Panel's range has ended and the user opted in to
                    # snap back to the statically-selected frame.
                    src_frame = int(getattr(p, "frame", 0) or 0)
                else:
                    # Default: hold on the last frame of the range.
                    src_frame = p.frame_end
                cap = captures.get(p.image_name)
                if cap is None:
                    continue
                cap.set(_cv2.CAP_PROP_POS_FRAMES, src_frame)
                ok, bgr = cap.read()
                if not ok:
                    continue
                rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                panel_images[(r, c)] = _PIL.fromarray(rgb)

            # Mirror the save_figure pipeline for this output frame.
            processed: List[List[Optional[Image.Image]]] = []
            for r in range(rows):
                row_imgs = []
                for c in range(cols):
                    panel = cfg.panels[r][c]
                    src_img: Optional[Image.Image]
                    if (r, c) in panel_images:
                        # Range video panel — per-panel extracted frame
                        # for THIS output index.
                        src_img = panel_images[(r, c)]
                    elif panel.image_name and panel.image_name in loaded_videos:
                        # Static (non-range) video panel — show the
                        # panel's own selected frame, panel-aware so
                        # shared video sources don't alias.
                        src_img = _get_panel_image(panel)
                    elif panel.image_name and panel.image_name in loaded_images:
                        # Plain image panel.
                        src_img = loaded_images[panel.image_name]
                    else:
                        src_img = None
                    if src_img is not None:
                        row_imgs.append(process_panel(
                            src_img, panel,
                            min_dims, loaded_images,
                            skip_labels=True, skip_symbols=True,
                        ))
                    else:
                        row_imgs.append(None)
                processed.append(row_imgs)

            col_max_w = [
                max((processed[r][c].size[0] for r in range(rows) if processed[r][c] is not None),
                    default=min_dims[0]) for c in range(cols)
            ]
            row_max_h = [
                max((processed[r][c].size[1] for c in range(cols) if processed[r][c] is not None),
                    default=min_dims[1]) for r in range(rows)
            ]
            for r in range(rows):
                for c in range(cols):
                    if processed[r][c] is None:
                        processed[r][c] = _PIL.new("RGB", (col_max_w[c], row_max_h[r]), "white")

            # Adjacent Panel zoom insets — mirror the save_figure pipeline so
            # the target cell isn't left blank for each exported frame.
            for r in range(rows):
                for c in range(cols):
                    panel = cfg.panels[r][c]
                    if panel.add_zoom_inset and panel.zoom_inset and panel.zoom_inset.inset_type == "Adjacent Panel":
                        zi = panel.zoom_inset
                        ar, ac = r, c
                        if zi.side == "Top": ar -= 1
                        elif zi.side == "Bottom": ar += 1
                        elif zi.side == "Left": ac -= 1
                        elif zi.side == "Right": ac += 1
                        if 0 <= ar < rows and 0 <= ac < cols:
                            ext_name = getattr(zi, 'separate_image_name', '') or ''
                            if ext_name and ext_name not in ('', 'select') and ext_name in loaded_images:
                                ext_img = loaded_images[ext_name].convert("RGB")
                                xi = getattr(zi, 'x_inset', 0) or 0
                                yi = getattr(zi, 'y_inset', 0) or 0
                                wi = getattr(zi, 'width_inset', ext_img.size[0]) or ext_img.size[0]
                                hi = getattr(zi, 'height_inset', ext_img.size[1]) or ext_img.size[1]
                                xi = max(0, min(xi, ext_img.size[0]-1))
                                yi = max(0, min(yi, ext_img.size[1]-1))
                                wi = max(1, min(wi, ext_img.size[0]-xi))
                                hi = max(1, min(hi, ext_img.size[1]-yi))
                                region = ext_img.crop((xi, yi, xi+wi, yi+hi))
                                zw = max(1, int(wi * zi.zoom_factor))
                                zh = max(1, int(hi * zi.zoom_factor))
                                region = region.resize((zw, zh), Image.LANCZOS)
                                processed[ar][ac] = region
                            else:
                                main_img = processed[r][c]
                                miw, mih = main_img.size
                                p_src = cfg.panels[r][c]
                                if p_src.crop_image and p_src.crop and len(p_src.crop) == 4:
                                    fw = p_src.crop[2] - p_src.crop[0]
                                    fh = p_src.crop[3] - p_src.crop[1]
                                elif p_src.image_name and p_src.image_name in loaded_images:
                                    fw, fh = loaded_images[p_src.image_name].size
                                else:
                                    fw, fh = miw, mih
                                scx = miw / max(fw, 1)
                                scy = mih / max(fh, 1)
                                bw = max(1, getattr(zi, 'rectangle_width', 2) or 2)
                                bw_x = bw * scx
                                bw_y = bw * scy
                                cx1 = max(0, int(zi.x * scx + bw_x))
                                cy1 = max(0, int(zi.y * scy + bw_y))
                                cx2 = min(miw, int((zi.x + zi.width) * scx - bw_x))
                                cy2 = min(mih, int((zi.y + zi.height) * scy - bw_y))
                                cx2 = max(cx1 + 1, cx2)
                                cy2 = max(cy1 + 1, cy2)
                                region = main_img.crop((cx1, cy1, cx2, cy2))
                                zw = max(1, int(zi.width * zi.zoom_factor * scx))
                                zh = max(1, int(zi.height * zi.zoom_factor * scy))
                                region = region.resize((zw, zh), Image.LANCZOS)
                                processed[ar][ac] = region

            full_res_sizes2: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for r in range(rows):
                for c in range(cols):
                    panel = cfg.panels[r][c]
                    if panel.image_name and panel.image_name in loaded_images:
                        orig_img = loaded_images[panel.image_name]
                        if panel.crop_image and panel.crop and len(panel.crop) == 4:
                            full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                        else:
                            full_res_sizes2[(r, c)] = orig_img.size

            orig_format = cfg.output_format
            orig_bg = cfg.background
            cfg.output_format = "PNG"
            cfg.background = body.background
            try:
                fig_bytes = assemble_figure(cfg, processed, dpi=body.dpi, full_res_sizes=full_res_sizes2)
            finally:
                cfg.output_format = orig_format
                cfg.background = orig_bg

            pil_out = _PIL.open(_BytesIO(fig_bytes)).convert("RGB")
            np_rgb = _np.asarray(pil_out)
            np_bgr = _cv2.cvtColor(np_rgb, _cv2.COLOR_RGB2BGR)

            if writer is None:
                h, w = np_bgr.shape[:2]
                fourcc = _cv2.VideoWriter_fourcc(*"mp4v") if body.format.lower() == "mp4" \
                    else _cv2.VideoWriter_fourcc(*"MJPG")
                writer = _cv2.VideoWriter(save_path, fourcc, float(body.fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {save_path}")
            writer.write(np_bgr)

            with _render_jobs_lock:
                _render_jobs[job_id]["current"] = frame_idx + 1

        if writer is not None:
            writer.release()
            writer = None

        # Optional audio mux — uses ffmpeg if installed. If the user
        # selected an audio panel, take the audio from its source video
        # over the panel's [start..end] frame range, padding with silence
        # to match the output duration when the audio range is shorter.
        if body.audio_panel_image_name:
            audio_panel = next(
                (p for _, _, p in range_panels if p.image_name == body.audio_panel_image_name),
                None,
            )
            audio_path = loaded_videos.get(body.audio_panel_image_name) if audio_panel else None
            if audio_panel and audio_path:
                try:
                    src_info = _get_video_info(audio_path)
                    panel_fps = src_info.get("fps", float(body.fps)) or float(body.fps)
                    start_t = audio_panel.frame_start / max(panel_fps, 1.0)
                    audio_dur = (audio_panel.frame_end - audio_panel.frame_start + 1) / max(panel_fps, 1.0)
                    output_dur = total_frames / max(float(body.fps), 1.0)
                    use_dur = min(audio_dur, output_dur)
                    tmp_out = save_path + ".silent" + os.path.splitext(save_path)[1]
                    os.replace(save_path, tmp_out)
                    cmd = [
                        _ffmpeg_path(), "-y",
                        "-i", tmp_out,
                        "-ss", f"{start_t:.4f}",
                        "-t", f"{use_dur:.4f}",
                        "-i", audio_path,
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                        "-shortest",
                        save_path,
                    ]
                    result = _subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode != 0 or not os.path.exists(save_path) or os.path.getsize(save_path) < 1024:
                        # Restore silent video if the mux failed
                        try:
                            if os.path.exists(save_path):
                                os.unlink(save_path)
                        except OSError:
                            pass
                        os.replace(tmp_out, save_path)
                    else:
                        try:
                            os.unlink(tmp_out)
                        except OSError:
                            pass
                except FileNotFoundError:
                    # ffmpeg not available — silently keep the silent video
                    pass
                except Exception as e:
                    print(f"[render_video] audio mux failed: {e}", flush=True)

        with _render_jobs_lock:
            _render_jobs[job_id]["status"] = "done"
            _render_jobs[job_id]["output_path"] = save_path

    except Exception as e:
        with _render_jobs_lock:
            _render_jobs[job_id]["status"] = "error"
            _render_jobs[job_id]["error"] = str(e)
        # Cleanup partial output on error
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if os.path.exists(save_path):
                os.unlink(save_path)
        except OSError:
            pass
    finally:
        # Always release captures. No loaded_images restore is needed
        # any more — the worker uses a per-panel image dict instead of
        # mutating the global cache, so there's nothing to put back.
        for cap in captures.values():
            try:
                cap.release()
            except Exception:
                pass


@app.get("/api/figure/render-video/{job_id}/progress")
def render_video_progress(job_id: str):
    with _render_jobs_lock:
        job = _render_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {
            "status": job["status"],
            "current": job["current"],
            "total": job["total"],
            "output_path": job.get("output_path"),
            "error": job.get("error"),
        }


@app.post("/api/figure/render-video/{job_id}/cancel")
def render_video_cancel(job_id: str):
    with _render_jobs_lock:
        job = _render_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        job["cancel_requested"] = True
    return {"ok": True}


class ProjectSaveRequest(BaseModel):
    path: str


@app.post("/api/project/save")
def save_proj(body: ProjectSaveRequest):
    path = os.path.expanduser(body.path.strip())
    # Ensure .mpf extension (not doubled)
    if not path.lower().endswith('.mpf'):
        path = path.rsplit('.', 1)[0] + '.mpf' if '.' in os.path.basename(path) else path + '.mpf'
    # If no directory specified, save to home directory
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bytes = {}
    for name, img in loaded_images.items():
        # For video files, save the video data instead of the frame image
        if name in loaded_videos and os.path.isfile(loaded_videos[name]):
            with open(loaded_videos[name], "rb") as vf:
                img_bytes[name] = vf.read()
        else:
            img_bytes[name] = pil_to_bytes(img)

    # Pre-flight disk-space check. Project file size ≈ sum of all image
    # bytes plus ~10% overhead for config JSON + ZIP metadata.
    try:
        import shutil as _shutil
        target_dir = os.path.dirname(path) or "."
        payload_size = sum(len(b) for b in img_bytes.values())
        required = int(payload_size * 1.1) + 16 * 1024 * 1024  # +16 MB headroom
        usage = _shutil.disk_usage(target_dir)
        if usage.free < required:
            raise HTTPException(
                507,
                f"Not enough disk space to save project: need ~"
                f"{required // (1024*1024)} MB, have "
                f"{usage.free // (1024*1024)} MB free on the target volume.",
            )
    except HTTPException:
        raise
    except Exception as e:
        import sys
        print(f"[save_proj] disk_usage check failed: {e}", file=sys.stderr, flush=True)

    save_project(cfg, img_bytes, path, custom_fonts or None)
    # Verify file was written
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        try: os.unlink(path)
        except OSError: pass
        raise HTTPException(500, f"Save failed: file at {path} is empty or missing (possible disk-full or permission issue).")
    return {"ok": True, "path": path, "size_bytes": os.path.getsize(path)}


class ProjectLoadRequest(BaseModel):
    path: str


@app.post("/api/project/load")
def load_proj(body: ProjectLoadRequest):
    global cfg, loaded_images, custom_fonts, loaded_videos, video_frames
    path = os.path.expanduser(body.path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, img_bytes_dict, font_bytes_dict = load_project(path)
    cfg = loaded_cfg
    cfg.ensure_grid()
    loaded_images.clear()
    loaded_videos.clear()
    video_frames.clear()
    for name, data in img_bytes_dict.items():
        if _is_video(name):
            # Restore video: save to temp file and extract first frame
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
            tmp.write(data)
            tmp.close()
            loaded_videos[name] = tmp.name
            video_frames[name] = 0
            loaded_images[name] = _extract_video_frame(tmp.name, 0)
        else:
            loaded_images[name] = Image.open(io.BytesIO(data)).convert("RGB")
    custom_fonts = font_bytes_dict or {}
    _recalc_min_dims()
    thumbnails = {n: _thumb_b64(img) for n, img in loaded_images.items()}
    return {
        "config": _cfg_json(),
        "image_names": list(loaded_images.keys()),
        "thumbnails": thumbnails,
    }


# ── Font Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/fonts")
def list_fonts():
    fonts = find_fonts()
    # Also pick up every font matplotlib's font_manager has indexed (it walks
    # the registry on Windows and GTK/fontconfig on Linux, which catches
    # a lot that bare os.listdir misses, especially on Windows where
    # C:\Windows\Fonts is a protected shell namespace).
    try:
        from matplotlib import font_manager as _fm
        for ttf in _fm.findSystemFonts(fontext="ttf") + _fm.findSystemFonts(fontext="otf"):
            fn = os.path.basename(ttf)
            if fn.lower().endswith((".ttf", ".otf", ".ttc")) and fn not in fonts:
                fonts[fn] = ttf
    except Exception as _e:
        import sys
        print(f"[fonts] matplotlib font_manager lookup failed: {_e}", file=sys.stderr, flush=True)
    return {"fonts": fonts}


_CUSTOM_FONTS_DIR = Path.home() / ".multipanelfigure" / "fonts"

def _load_persistent_fonts():
    """Load custom fonts from persistent storage on startup."""
    if _CUSTOM_FONTS_DIR.is_dir():
        for fp in _CUSTOM_FONTS_DIR.iterdir():
            if fp.suffix.lower() in (".ttf", ".otf"):
                try:
                    custom_fonts[fp.name] = fp.read_bytes()
                except Exception:
                    pass

# Load persistent fonts on startup
_load_persistent_fonts()


@app.post("/api/fonts/upload")
async def upload_fonts(files: List[UploadFile] = File(...)):
    names = []
    _CUSTOM_FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        data = await f.read()
        custom_fonts[f.filename] = data
        # Save to persistent storage
        try:
            (_CUSTOM_FONTS_DIR / f.filename).write_bytes(data)
        except Exception as e:
            print(f"Warning: Could not save font {f.filename}: {e}")
        names.append(f.filename)
    # Invalidate the cached font-path lookups so newly uploaded fonts
    # are resolvable on the very next render.
    try:
        from figure_builder import _font_path_cache
        _font_path_cache.clear()
    except Exception:
        pass
    return {"names": names, "total": len(find_fonts())}


# ── Resolution Endpoints ──────────────────────────────────────────────────

# Persistent scale bar storage
_SCALE_BAR_FILE = Path.home() / ".multipanelfigure" / "scale_bars.json"

# Default microscope presets (μm/pixel). Values converted from the
# user-supplied calibration list; entries originally given in pixels/μm
# are stored here as 1/(px per μm).
DEFAULT_RESOLUTION_ENTRIES: Dict[str, float] = {
    # NIKON TC1 (640x480)
    "NIKON TC1 4x":  1 / 0.2925,
    "NIKON TC1 10x": 1 / 0.7412,
    "NIKON TC1 20x": 1 / 1.465,
    "NIKON TC1 40x": 1 / 2.925,
    # NIKON Ti — Microscope Room (772x618)
    "NIKON Ti 4x":   1 / 0.38,
    "NIKON Ti 10x":  1 / 0.9313,
    "NIKON Ti 20x":  1 / 1.8833,
    "NIKON Ti 40x":  1 / 3.8,
    # INCUCYTE (1408x1040)
    "INCUCYTE 4x":   2.82,
    "INCUCYTE 10x":  1.24,
    "INCUCYTE 20x":  0.62,
    "INCUCYTE Whole Well": 1000.0 / 141.5462,
    # ZEISS L12 OLD (1388x1040)
    "ZEISS L12 OLD 5x":  1.3,
    "ZEISS L12 OLD 10x": 0.6425,
    "ZEISS L12 OLD 20x": 0.3222,
    "ZEISS L12 OLD 40x": 0.1603,
    "ZEISS L12 OLD 63x": 0.1018,
    # ZEISS L12 NEW (2752x2208)
    "ZEISS L12 NEW 5x":  0.9101,
    "ZEISS L12 NEW 10x": 0.4551,
    "ZEISS L12 NEW 20x": 0.2278,
    "ZEISS L12 NEW 40x": 0.1132,
    # ZOE TC1 TIFF (2592x1944)
    "ZOE TC1 20x": 1 / 2.6233,
    # NIKON TC1 Dissecting Microscope (0.63x @ 1x → 106 px/mm)
    "NIKON TC1 Dissecting 0.63x@1x": 1000.0 / 106.0,
    # HEIDELBERG IVCM (400μm field)
    "HEIDELBERG IVCM": 1 / 0.96,
}

def _load_persistent_scale_bars() -> Dict[str, float]:
    """Load scale bars from persistent storage."""
    try:
        if _SCALE_BAR_FILE.exists():
            import json
            return json.loads(_SCALE_BAR_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_persistent_scale_bars(entries: Dict[str, float]):
    """Save scale bars to persistent storage."""
    try:
        _SCALE_BAR_FILE.parent.mkdir(parents=True, exist_ok=True)
        import json
        _SCALE_BAR_FILE.write_text(json.dumps(entries))
    except Exception as e:
        print(f"Warning: Could not save scale bars: {e}")

# Seed defaults on startup. If the user has a persistent scale_bars.json
# (custom list from prior usage), respect it as-is; otherwise populate
# with the bundled microscope presets.
_persistent_scales = _load_persistent_scale_bars()
if _persistent_scales:
    cfg.resolution_entries.update(_persistent_scales)
else:
    cfg.resolution_entries.update(DEFAULT_RESOLUTION_ENTRIES)


@app.get("/api/resolutions")
def get_resolutions():
    return cfg.resolution_entries


class ResolutionUpdate(BaseModel):
    entries: Dict[str, float]


@app.put("/api/resolutions")
def put_resolutions(body: ResolutionUpdate):
    cfg.resolution_entries = body.entries
    _save_persistent_scale_bars(body.entries)
    return {"ok": True}


# ── Collage stash management ───────────────────────────────────────────────
# When a user clicks "Add to Collage" we save a snapshot of the current
# project to a stash so the collage's "Multi-Panel Builder" button can
# round-trip back to that exact figure's editor state. The frontend
# generates the stash id; we only need a safe deletion endpoint so the
# stash isn't orphaned when the collage item is removed.

import re as _re_stash

_COLLAGE_STASH_DIR = Path.home() / ".multipanelfigure" / "collage_stash"


class CollageFigureRenderRequest(BaseModel):
    project_path: str
    """Target visual point size for headers / labels in the collage
    (pre-export-DPI). Pass null to render without any override
    (i.e., as-saved)."""
    header_pt: Optional[int] = None
    """Either: collage-side display scale (item.w / item.naturalW) for
    the legacy single-pass path, OR — preferred — the literal collage
    pixel width of the item (item.w in canvas pixels). When item_w is
    set, the backend ignores `scale` and runs an iterative two-pass
    render: first pass measures the figure's naturalW with all
    headers/labels at the target pt; second pass uses the measured
    naturalW to compute the actual override pt so the rendered
    headers compensate for the post-override naturalW change.
    Solves the matching problem for figures with row headers/labels
    (whose width grows with the override pt)."""
    scale: float = 1.0
    item_w: Optional[float] = None


# Per-process cache of decoded .mpf data, keyed by (absolute path, mtime).
# Re-renders for the global-header-pt feature otherwise re-decompress the
# zip + decode every image on every call, which dominates the wall-time
# for a typical 4-panel project (~2-3 sec per render).
_collage_mpf_cache: Dict[str, dict] = {}
_COLLAGE_CACHE_MAX = 12  # rough LRU cap; collage size is usually small


def _load_collage_mpf(path: str):
    """Return (FigureConfig, dict[name → PIL.Image]) for the .mpf at
    `path`. Caches the parsed result per-process keyed by (path, mtime)
    so subsequent collage renders skip the zip decompression entirely.
    Mutating images is unsafe — process_panel already img.copy()s, so
    callers can share the cached PIL objects safely."""
    mtime = os.path.getmtime(path)
    hit = _collage_mpf_cache.get(path)
    if hit and hit.get("mtime") == mtime:
        return hit["cfg"], hit["images"]
    loaded_cfg, img_bytes_dict, _font_bytes = load_project(path)
    loaded_cfg.ensure_grid()
    local_images: Dict[str, Image.Image] = {}
    for name, data in img_bytes_dict.items():
        if _is_video(name):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
            tmp.write(data)
            tmp.close()
            try:
                local_images[name] = _extract_video_frame(tmp.name, 0)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        else:
            local_images[name] = Image.open(io.BytesIO(data)).convert("RGB")
    _collage_mpf_cache[path] = {"mtime": mtime, "cfg": loaded_cfg, "images": local_images}
    # Naive size cap — drop the oldest entry when we exceed the limit.
    if len(_collage_mpf_cache) > _COLLAGE_CACHE_MAX:
        oldest = next(iter(_collage_mpf_cache))
        if oldest != path:
            del _collage_mpf_cache[oldest]
    return loaded_cfg, local_images


@app.post("/api/collage/render-figure")
def render_collage_figure(body: CollageFigureRenderRequest):
    """Stateless render of a saved .mpf into a PNG, optionally
    re-targeting all header/label point sizes so multiple figures
    in the same collage end up with visually-uniform header sizes
    once the collage scales them. Does NOT touch the live cfg /
    loaded_images globals — the user's current builder state is
    preserved.

    Performance notes:
    - The .mpf zip is parsed once per file via _load_collage_mpf and
      cached for the session, so subsequent calls (especially the
      auto-rerender on resize) skip the expensive decompression.
    - DPI defaults to 150 here instead of 300; the collage display
      reduces every figure significantly anyway, and a 4× pixel
      reduction roughly halves matplotlib's render time without
      visible loss in the collage view. The Save Collage path can
      always re-up to 300 DPI later if needed for export."""
    import copy as _copy
    path = os.path.expanduser(body.project_path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, local_images = _load_collage_mpf(path)

    def _apply_header_pt(cfg, new_pt):
        for level in cfg.column_headers:
            for hdr in level.headers:
                hdr.font_size = new_pt
                for seg in (hdr.styled_segments or []):
                    seg.font_size = None
        for level in cfg.row_headers:
            for hdr in level.headers:
                hdr.font_size = new_pt
                for seg in (hdr.styled_segments or []):
                    seg.font_size = None
        for lbl in cfg.column_labels:
            lbl.font_size = new_pt
            for seg in (lbl.styled_segments or []):
                seg.font_size = None
        for lbl in cfg.row_labels:
            lbl.font_size = new_pt
            for seg in (lbl.styled_segments or []):
                seg.font_size = None

    def _render_cfg(cfg_to_render):
        rows, cols = cfg_to_render.rows, cfg_to_render.cols
        if local_images:
            ws = [im.size[0] for im in local_images.values()]
            hs = [im.size[1] for im in local_images.values()]
            local_min_dims = (min(ws), min(hs))
        else:
            local_min_dims = (100, 100)
        proc: List[List[Optional[Image.Image]]] = []
        for r in range(rows):
            row_imgs: List[Optional[Image.Image]] = []
            for c in range(cols):
                panel = cfg_to_render.panels[r][c]
                if panel.image_name and panel.image_name in local_images:
                    row_imgs.append(process_panel(
                        local_images[panel.image_name], panel,
                        local_min_dims, local_images,
                        skip_labels=True, skip_symbols=True,
                    ))
                else:
                    row_imgs.append(None)
            proc.append(row_imgs)
        col_max_w = [
            max((proc[r][c].size[0] for r in range(rows) if proc[r][c] is not None),
                default=local_min_dims[0]) for c in range(cols)
        ]
        row_max_h = [
            max((proc[r][c].size[1] for c in range(cols) if proc[r][c] is not None),
                default=local_min_dims[1]) for r in range(rows)
        ]
        for r in range(rows):
            for c in range(cols):
                if proc[r][c] is None:
                    proc[r][c] = Image.new("RGB", (col_max_w[c], row_max_h[r]), "white")
        full_res_sizes_local: Dict = {}
        for r in range(rows):
            for c in range(cols):
                panel = cfg_to_render.panels[r][c]
                if panel.image_name and panel.image_name in local_images:
                    orig_img = local_images[panel.image_name]
                    if panel.crop_image and panel.crop and len(panel.crop) == 4:
                        full_res_sizes_local[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                    else:
                        full_res_sizes_local[(r, c)] = orig_img.size
        fig_bytes = assemble_figure(cfg_to_render, proc, dpi=150, full_res_sizes=full_res_sizes_local)
        img_obj = Image.open(io.BytesIO(fig_bytes)).convert("RGBA")
        return img_obj

    # Two-pass iterative path (preferred). Measures the figure's
    # post-override naturalW with a quick first pass, then uses that
    # measured width to compute the actually-correct compensation pt.
    # Without this, figures with row headers / row labels end up with
    # ~10-25% mismatched header sizes because matplotlib grows fig_w
    # to fit the larger labels and the pre-render scale calculation
    # was using the OLD naturalW.
    if body.header_pt and body.header_pt > 0 and body.item_w and body.item_w > 0:
        # Pass 1: render with all headers at target_pt — gives a
        # baseline that already INCLUDES the layout shifts caused by
        # changing header pt. The second pass then trues up.
        cfg_p1 = _copy.deepcopy(loaded_cfg)
        _apply_header_pt(cfg_p1, body.header_pt)
        img_p1 = _render_cfg(cfg_p1)
        scale_p1 = body.item_w / max(1, img_p1.width)
        new_pt = max(1, int(round(body.header_pt / max(0.001, scale_p1))))
        # Pass 2: render at the corrected pt. The corrected pt's
        # render width will be very close to but not always
        # identical to img_p1.width — typically within a few percent
        # for figures with row labels, exactly the same for figures
        # without. Returning pass 2's bytes gives the user the
        # rendered output that matches their target pt visually.
        cfg_p2 = _copy.deepcopy(loaded_cfg)
        _apply_header_pt(cfg_p2, new_pt)
        img_p2 = _render_cfg(cfg_p2)
        out_buf = io.BytesIO()
        img_p2.save(out_buf, format="PNG")
        return {
            "image": base64.b64encode(out_buf.getvalue()).decode("ascii"),
            "width": img_p2.width,
            "height": img_p2.height,
        }

    # Apply header-size override if requested. Compensates for the
    # collage's downscale: target_pt × (1 / scale) so the visible
    # size after the collage scales the figure is exactly target_pt.
    cfg2 = _copy.deepcopy(loaded_cfg)
    if body.header_pt and body.header_pt > 0:
        new_pt = max(1, int(round(body.header_pt / max(0.001, body.scale))))
        for level in cfg2.column_headers:
            for hdr in level.headers:
                hdr.font_size = new_pt
                for seg in (hdr.styled_segments or []):
                    seg.font_size = None
        for level in cfg2.row_headers:
            for hdr in level.headers:
                hdr.font_size = new_pt
                for seg in (hdr.styled_segments or []):
                    seg.font_size = None
        for lbl in cfg2.column_labels:
            lbl.font_size = new_pt
            for seg in (lbl.styled_segments or []):
                seg.font_size = None
        for lbl in cfg2.row_labels:
            lbl.font_size = new_pt
            for seg in (lbl.styled_segments or []):
                seg.font_size = None

    # Mirror the /api/preview pipeline but against local_images and
    # cfg2. We deliberately don't downscale below max_preview_px so
    # the collage receives the figure at full quality — the collage
    # itself manages display sizing.
    rows, cols = cfg2.rows, cfg2.cols
    if loaded_cfg.panels:
        # Compute min_dims from local images for per-panel processing.
        if local_images:
            ws = [im.size[0] for im in local_images.values()]
            hs = [im.size[1] for im in local_images.values()]
            local_min_dims = (min(ws), min(hs))
        else:
            local_min_dims = (100, 100)
    else:
        local_min_dims = (100, 100)

    processed: List[List[Optional[Image.Image]]] = []
    for r in range(rows):
        row_imgs: List[Optional[Image.Image]] = []
        for c in range(cols):
            panel = cfg2.panels[r][c]
            if panel.image_name and panel.image_name in local_images:
                row_imgs.append(process_panel(
                    local_images[panel.image_name], panel,
                    local_min_dims, local_images,
                    skip_labels=True, skip_symbols=True,
                ))
            else:
                row_imgs.append(None)
        processed.append(row_imgs)

    col_max_w = [
        max((processed[r][c].size[0] for r in range(rows) if processed[r][c] is not None),
            default=local_min_dims[0]) for c in range(cols)
    ]
    row_max_h = [
        max((processed[r][c].size[1] for c in range(cols) if processed[r][c] is not None),
            default=local_min_dims[1]) for r in range(rows)
    ]
    for r in range(rows):
        for c in range(cols):
            if processed[r][c] is None:
                processed[r][c] = Image.new("RGB", (col_max_w[c], row_max_h[r]), "white")

    full_res_sizes2: Dict = {}
    for r in range(rows):
        for c in range(cols):
            panel = cfg2.panels[r][c]
            if panel.image_name and panel.image_name in local_images:
                orig_img = local_images[panel.image_name]
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes2[(r, c)] = orig_img.size

    # Reduced DPI (vs the default 300) — the collage display-scales
    # every figure aggressively, so 150 keeps headers/text crisp while
    # cutting matplotlib's render time roughly in half.
    fig_bytes = assemble_figure(cfg2, processed, dpi=150, full_res_sizes=full_res_sizes2)
    img_out = Image.open(io.BytesIO(fig_bytes)).convert("RGBA")
    out_buf = io.BytesIO()
    img_out.save(out_buf, format="PNG")
    return {
        "image": base64.b64encode(out_buf.getvalue()).decode("ascii"),
        "width": img_out.width,
        "height": img_out.height,
    }


@app.delete("/api/collage/stash/{item_id}")
def delete_collage_stash(item_id: str):
    """Delete a stashed .mpf for the given collage item id. The id
    must be alphanumeric/underscore only — anything else is rejected
    so the endpoint can never be tricked into traversing outside the
    stash directory."""
    if not _re_stash.match(r"^[A-Za-z0-9_]+$", item_id):
        raise HTTPException(400, "Invalid stash id")
    path = _COLLAGE_STASH_DIR / f"{item_id}.mpf"
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
    return {"ok": True}


@app.post("/api/resolutions/restore-defaults")
def restore_default_resolutions():
    """Reset the resolution presets to the bundled microscope defaults
    and persist them to ~/.multipanelfigure/scale_bars.json."""
    cfg.resolution_entries = dict(DEFAULT_RESOLUTION_ENTRIES)
    _save_persistent_scale_bars(cfg.resolution_entries)
    return cfg.resolution_entries


# ── R Analysis Endpoints ───────────────────────────────────────────────────

import subprocess
import shutil
import glob as glob_mod

class RAnalysisRequest(BaseModel):
    code: str
    data_csv: str  # CSV string
    rscript_path: Optional[str] = None  # custom Rscript path


def _find_rscript(custom_path: Optional[str] = None) -> Optional[str]:
    """Find Rscript binary — checks custom path, PATH, and common install locations."""
    # 1. User-specified custom path
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    # 2. Standard PATH lookup
    found = shutil.which("Rscript")
    if found:
        return found

    # 3. Common installation locations by platform
    import platform
    candidates = []
    if platform.system() == "Darwin":  # macOS
        candidates = [
            "/usr/local/bin/Rscript",
            "/opt/homebrew/bin/Rscript",
            "/Library/Frameworks/R.framework/Versions/Current/Resources/bin/Rscript",
            "/Library/Frameworks/R.framework/Resources/bin/Rscript",
            os.path.expanduser("~/Library/Frameworks/R.framework/Resources/bin/Rscript"),
        ]
    elif platform.system() == "Windows":
        # Check common R install paths on Windows
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for pf in [prog_files, prog_files_x86]:
            r_dir = os.path.join(pf, "R")
            if os.path.isdir(r_dir):
                for ver_dir in sorted(os.listdir(r_dir), reverse=True):
                    cand = os.path.join(r_dir, ver_dir, "bin", "Rscript.exe")
                    candidates.append(cand)
        candidates.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "R", "bin", "Rscript.exe"))
    else:  # Linux
        candidates = [
            "/usr/bin/Rscript",
            "/usr/local/bin/Rscript",
            "/snap/bin/Rscript",
        ]

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand

    return None


@app.get("/api/analysis/check-r")
def check_r_installed(rscript_path: Optional[str] = None):
    """Check if Rscript is available."""
    rscript = _find_rscript(rscript_path)
    if not rscript:
        return {"installed": False, "version": "", "path": ""}
    try:
        result = subprocess.run([rscript, "--version"], capture_output=True, text=True, timeout=5)
        version = (result.stderr + result.stdout).strip().split("\n")[0]
        return {"installed": True, "version": version, "path": rscript}
    except Exception as e:
        return {"installed": False, "version": str(e), "path": rscript}

@app.post("/api/analysis/run-r")
def run_r_code(body: RAnalysisRequest):
    """Run R code with provided data, return stdout/stderr and any generated plots."""
    rscript = _find_rscript(body.rscript_path)
    if not rscript:
        return {"success": False, "stdout": "", "stderr": "R is not installed. Please install R from https://cran.r-project.org/ or specify the Rscript path.", "plots": []}

    with tempfile.TemporaryDirectory(prefix="mpfig_r_") as tmpdir:
        data_path = os.path.join(tmpdir, "data.csv")
        with open(data_path, "w") as f:
            csv_text = body.data_csv.rstrip() + "\n"  # ensure trailing newline
            f.write(csv_text)

        plot_dir = os.path.join(tmpdir, "plots")
        os.makedirs(plot_dir)
        script = '# Auto-install missing packages\n'
        script += 'if (!requireNamespace("ggplot2", quietly=TRUE)) install.packages("ggplot2", repos="https://cloud.r-project.org", quiet=TRUE)\n\n'
        script += f'# Auto-generated data loading\ndata <- read.csv("{data_path.replace(chr(92), "/")}")\n\n'
        script += f'# Set plot output directory\n.plot_dir <- "{plot_dir.replace(chr(92), "/")}"\n'
        script += '.plot_count <- 0\n'
        script += 'mpfig_plot <- function(filename=NULL, width=800, height=600, res=150) {\n'
        script += '  .plot_count <<- .plot_count + 1\n'
        script += '  if (is.null(filename)) filename <- paste0("plot_", .plot_count, ".png")\n'
        script += '  png(file.path(.plot_dir, filename), width=width, height=height, res=res)\n'
        script += '}\n\n'
        script += '# User code\n'
        script += body.code
        script += '\n\n# Close any open graphics devices\nwhile (dev.cur() > 1) dev.off()\n'

        script_path = os.path.join(tmpdir, "analysis.R")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            result = subprocess.run(
                [rscript, script_path],
                capture_output=True, text=True, timeout=120,
                cwd=tmpdir
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "R script timed out after 120 seconds (may be installing packages — try again).", "plots": []}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "plots": []}

        plots_b64 = []
        for png_path in sorted(glob_mod.glob(os.path.join(plot_dir, "*.png"))):
            with open(png_path, "rb") as pf:
                plots_b64.append(base64.b64encode(pf.read()).decode())

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "plots": plots_b64,
        }


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    import sys
    try:
        parser = argparse.ArgumentParser(description="Multi-Panel Figure Builder API Server")
        parser.add_argument("--port", type=int, default=0, help="Port (0 = auto)")
        parser.add_argument("--host", default="127.0.0.1")
        args = parser.parse_args()

        port = args.port
        if port == 0:
            import socket
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        print(f"READY:{port}", flush=True)
        uvicorn.run(app, host=args.host, port=port, log_level="warning")
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
