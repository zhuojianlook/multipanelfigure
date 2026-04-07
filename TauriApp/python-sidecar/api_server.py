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
min_dims = (512, 512)
custom_fonts: Dict[str, bytes] = {}

# Video support
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts'}
loaded_videos: Dict[str, str] = {}  # name → temp file path
video_frames: Dict[str, int] = {}   # name → selected frame number

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
    return {
        "names": list(loaded_images.keys()),
        "used": list(used),
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


@app.delete("/api/images/{name}")
def delete_image(name: str):
    if name in loaded_images:
        del loaded_images[name]
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
    img = loaded_images[panel.image_name]
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
    processed = process_panel(loaded_images[panel.image_name], panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
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
    processed = process_panel(loaded_images[panel.image_name], panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
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
        loaded_images[panel.image_name], panel, min_dims, loaded_images,
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
    orig_img = loaded_images[panel.image_name]
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

    img = loaded_images[panel.image_name].convert("RGB")
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
    img = loaded_images[panel.image_name].copy().convert("RGB")
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
            img = loaded_images[panel.image_name]
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
    # First pass: process panels WITHOUT labels/symbols/scale bar — those are
    # rendered via matplotlib AFTER normalize for immutable font sizing
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            panel = cfg.panels[r][c]
            if panel.image_name and panel.image_name in loaded_images:
                row_imgs.append(process_panel(
                    loaded_images[panel.image_name], panel,
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
                orig_img = loaded_images[panel.image_name]
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
                    loaded_images[panel.image_name], panel,
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
                orig_img = loaded_images[panel.image_name]
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes2[(r, c)] = orig_img.size
    fig_bytes = assemble_figure(cfg, processed, dpi=body.dpi, full_res_sizes=full_res_sizes2)
    save_path = os.path.expanduser(body.path.strip())
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(fig_bytes)
    return {"ok": True, "path": save_path}


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
    save_project(cfg, img_bytes, path, custom_fonts or None)
    # Verify file was written
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise HTTPException(500, f"Save failed: file at {path} is empty or missing")
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
    return {"names": names, "total": len(find_fonts())}


# ── Resolution Endpoints ──────────────────────────────────────────────────

# Persistent scale bar storage
_SCALE_BAR_FILE = Path.home() / ".multipanelfigure" / "scale_bars.json"

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

# Load persistent scale bars on startup
_persistent_scales = _load_persistent_scale_bars()
if _persistent_scales:
    cfg.resolution_entries.update(_persistent_scales)


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
