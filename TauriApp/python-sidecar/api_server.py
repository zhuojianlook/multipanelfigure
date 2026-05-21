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

# ── Multichannel TIFF / channel groups ────────────────────────────────────
#
# When a TIFF carries more than one channel (axes that include "C" with
# size > 1, e.g. "CYX", "ZCYX", "TZCYX"), we treat it as a *channel
# group*: each channel becomes its own greyscale plane that the user
# tints with an arbitrary colour. The composite (sum of tinted channels,
# clipped to [0,1] per RGB component) is what panels actually display
# via `loaded_images[name]` — so the rest of the render pipeline doesn't
# need to know that the source was multichannel.
#
# Per-channel state lives in `channel_groups[name]`. Mutating any field
# via `/api/zstack/{name}/channels` recomputes the composite and writes
# it back into `loaded_images[name]`, so existing panel-render code
# "just works" with per-channel tint changes.
#
# Defaults are picked to be biology-friendly: red, green, blue, magenta,
# cyan, yellow, then white for any extras. Each channel also gets its
# own black/white levels (0–255) so the user can window each channel
# independently — important for fluorescence where each fluorophore has
# a different dynamic range.
class _ChannelGroup:
    __slots__ = ("axes", "arr", "num_channels", "num_z",
                 "current_z", "tints", "enabled",
                 "black_levels", "white_levels", "max_vals", "names")

    def __init__(self, axes: str, arr: "np.ndarray"):
        self.axes = axes
        self.arr = arr            # full N-D array, dtype as-loaded
        self.num_channels = 0
        self.num_z = 1
        self.current_z = 0
        self.tints: List[str] = []
        self.enabled: List[bool] = []
        self.black_levels: List[int] = []   # 0-255
        self.white_levels: List[int] = []   # 0-255
        self.max_vals: List[float] = []     # native max per channel (for level mapping)
        # Human-readable channel names so the user can label "DAPI", "GFP",
        # etc. Surfaced in tooltips on the swatches and used in the 3D
        # viewer's channel toggles. Defaults to "Ch 1", "Ch 2", …
        self.names: List[str] = []

channel_groups: Dict[str, _ChannelGroup] = {}


def _resolve_channel_group_key(name: str) -> Optional[str]:
    """Return the canonical channel_groups key for `name`. Handles the
    per-panel snapshot keys (e.g. `__zstack_Composite.tif_r0c0`) by
    falling back to the embedded source name so the Channels block
    keeps working after a frame seek (which renames the panel's
    image_name to the per-panel snapshot key)."""
    if name in channel_groups:
        return name
    if name.startswith("__zstack_"):
        # Strip the "__zstack_" prefix and the trailing "_r{R}c{C}"
        rest = name[len("__zstack_"):]
        # Last "_rNcM" suffix
        import re as _re
        m = _re.match(r"(.+?)_r\d+c\d+$", rest)
        if m and m.group(1) in channel_groups:
            return m.group(1)
        # Or for aligned outputs: "{name}::aligned" stored under panel_key
        if rest in channel_groups:
            return rest
    return None

# Default per-channel tints — cycled for groups with > 7 channels.
_DEFAULT_CHANNEL_TINTS = [
    "#ff0000",  # red
    "#00ff00",  # green
    "#0066ff",  # blue
    "#ff00ff",  # magenta
    "#00ffff",  # cyan
    "#ffff00",  # yellow
    "#ffffff",  # white
]


def _canonicalize_tiff_axes(arr: "np.ndarray", axes: str):
    """Normalize a tifffile axes string + array to a canonical TZCYX
    (or subset) layout. Single helper used by every TIFF code path so
    they all interpret the same letter the same way.

    Behaviour by axis label:
      - T, Z, C, Y, X: kept, reordered to canonical TZCYX
      - S (sample-per-pixel, e.g. RGB-as-samples): max-merged into a
        single luminance plane — we don't treat sample-axis RGB as a
        user-tintable channel group; PIL renders those natively.
      - I (ImageJ "image" / sequential frame index): promoted to Z when
        no Z is present (the common multi-page-as-z pattern); otherwise
        max-collapsed defensively.
      - Q (tifffile's "unknown axis"): same as I — Z-promotion first,
        max-collapse if Z already exists.
      - Any other letter (rare microscope-specific dims like A=angle,
        M=mosaic, R=rotation): treated identically to I/Q — promote to
        Z if absent, else max-collapse. Better to merge an unknown dim
        than to crash with "axes don't match array".

    Returns (arr, canonical_axes_substring). The caller is responsible
    for further reductions like collapsing T to a single frame.
    """
    if not axes:
        return arr, ""
    axes = axes.upper()

    # 1) Merge S (sample-per-pixel) axis early.
    while "S" in axes:
        i = axes.index("S")
        arr = arr.max(axis=i)
        axes = axes[:i] + axes[i+1:]

    # 2) Re-label any non-canonical letter (I, Q, A, M, R, …) as either
    #    a brand-new Z (when no Z exists) or max-collapse it. We loop so
    #    multiple unknown axes all get handled deterministically.
    CANONICAL = set("TZCYX")
    while True:
        unknown = [c for c in axes if c not in CANONICAL]
        if not unknown:
            break
        letter = unknown[0]
        i = axes.index(letter)
        if "Z" not in axes:
            axes = axes[:i] + "Z" + axes[i+1:]
        else:
            arr = arr.max(axis=i)
            axes = axes[:i] + axes[i+1:]

    # 3) Reorder to canonical TZCYX (just the subset that's present).
    canonical = "".join(a for a in "TZCYX" if a in axes)
    if canonical and axes != canonical:
        arr = np.transpose(arr, [axes.index(a) for a in canonical])
        axes = canonical
    return arr, axes


def _detect_multichannel_tiff(tiff_path: str):
    """Read a TIFF with tifffile and return (canonical_axes, arr) if
    it has > 1 channel, else None. Robust to single-page, pyramidal,
    ImageJ-hyperstack, and unknown-axis TIFFs."""
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(tiff_path) as tf:
            if not tf.series:
                return None
            # Pick the largest series (typically series[0], but pyramidal
            # TIFFs put downsampled levels in later series — guard against
            # accidentally picking a 256×256 preview).
            s = max(tf.series, key=lambda ser: int(np.prod(ser.shape)))
            axes = (s.axes or "").upper()
            shape = s.shape
            if "C" not in axes:
                return None
            ci = axes.index("C")
            if ci >= len(shape) or shape[ci] <= 1:
                return None
            arr = s.asarray()
        # Canonicalize so downstream code always sees ZCYX (or a substring).
        arr, axes = _canonicalize_tiff_axes(arr, axes)
        return axes, arr
    except Exception as e:
        import sys
        print(f"[multichannel-detect] {tiff_path}: {e}", file=sys.stderr, flush=True)
        return None


def _init_channel_group(name: str, axes: str, arr: "np.ndarray") -> _ChannelGroup:
    """Build initial channel group state with sane defaults. `axes` is
    expected to be already canonicalised by `_canonicalize_tiff_axes` —
    a substring of TZCYX in that order."""
    g = _ChannelGroup(axes, arr)

    # Collapse any T axis to first (we don't expose time-as-channels yet).
    a = arr
    ax = axes
    while "T" in ax:
        i = ax.index("T")
        a = np.take(a, 0, axis=i)
        ax = ax[:i] + ax[i+1:]

    # Locate Z and C axes in the (possibly T-collapsed) array.
    g.num_z = a.shape[ax.index("Z")] if "Z" in ax else 1
    g.num_channels = a.shape[ax.index("C")] if "C" in ax else 1

    # Cache the T-collapsed view to avoid re-doing the squeeze on every
    # composite call.
    g.arr = a
    g.axes = ax

    g.tints    = [_DEFAULT_CHANNEL_TINTS[c % len(_DEFAULT_CHANNEL_TINTS)] for c in range(g.num_channels)]
    g.enabled  = [True] * g.num_channels
    g.black_levels = [0]   * g.num_channels
    g.white_levels = [255] * g.num_channels
    g.names    = [f"Ch {c + 1}" for c in range(g.num_channels)]

    # Compute per-channel max for normalization (important for 16-bit /
    # float data — a 16-bit channel with peak 4000 would otherwise
    # render near-black if we naively assumed uint8).
    g.max_vals = []
    for c in range(g.num_channels):
        ch = _extract_channel_plane(g, c, 0)  # take z=0 sample for max scan
        try:
            mx = float(ch.max()) if ch.size else 1.0
        except Exception:
            mx = 1.0
        # If the array has multiple Z slices, also sample a few more so the
        # max reflects the brightest slice (cheap heuristic — full scan
        # of every Z would be slow for big stacks).
        if g.num_z > 1:
            sample_zs = [g.num_z // 2, g.num_z - 1]
            for z in sample_zs:
                try:
                    mx = max(mx, float(_extract_channel_plane(g, c, z).max()))
                except Exception:
                    pass
        g.max_vals.append(mx if mx > 0 else 1.0)

    channel_groups[name] = g
    return g


def _extract_channel_plane(g: _ChannelGroup, c: int, z: int) -> "np.ndarray":
    """Return the (H, W) plane for channel c at z-slice z."""
    a = g.arr
    ax = g.axes
    # Collapse Z
    if "Z" in ax:
        zi = ax.index("Z")
        z = max(0, min(z, a.shape[zi] - 1))
        a = np.take(a, z, axis=zi)
        ax = ax[:zi] + ax[zi+1:]
    # Collapse C
    if "C" in ax:
        ci = ax.index("C")
        c = max(0, min(c, a.shape[ci] - 1))
        a = np.take(a, c, axis=ci)
        ax = ax[:ci] + ax[ci+1:]
    # Should be (Y, X) now
    if a.ndim == 3:
        # Trailing sample axis (e.g. RGBA) — collapse to luminance-ish.
        a = a.mean(axis=-1)
    return a


def _composite_channel_group(name: str, z: Optional[int] = None) -> Image.Image:
    """Re-render the composite RGB image for the channel group at the
    given z-slice (or the group's `current_z` when z is None). The
    z-override lets the panel render its OWN frame without mutating
    the group's shared state — important so two panels showing the
    same source can hold different frames simultaneously."""
    g = channel_groups[name]
    if z is None:
        z = g.current_z
    z = max(0, min(int(z), g.num_z - 1))
    H = W = None
    out = None

    for c in range(g.num_channels):
        if c < len(g.enabled) and not g.enabled[c]:
            continue
        ch = _extract_channel_plane(g, c, z).astype(np.float32)
        if H is None:
            H, W = ch.shape
            out = np.zeros((H, W, 3), dtype=np.float32)
        # Normalize to 0..1 against this channel's native max so 16-bit
        # data spans the full dynamic range.
        denom = max(g.max_vals[c], 1.0)
        norm = ch / denom
        # Apply black/white window: bl→0, wl→1 (both as 0..255 fractions
        # of the normalized range, matching the UI's levels semantics).
        bl = g.black_levels[c] / 255.0 if c < len(g.black_levels) else 0.0
        wl = g.white_levels[c] / 255.0 if c < len(g.white_levels) else 1.0
        if wl > bl:
            norm = np.clip((norm - bl) / (wl - bl), 0.0, 1.0)
        else:
            norm = np.zeros_like(norm)
        # Tint
        hex_ = g.tints[c] if c < len(g.tints) else "#ffffff"
        try:
            r = int(hex_[1:3], 16) / 255.0
            gn = int(hex_[3:5], 16) / 255.0
            b = int(hex_[5:7], 16) / 255.0
        except (ValueError, IndexError):
            r = gn = b = 1.0
        out[..., 0] += norm * r
        out[..., 1] += norm * gn
        out[..., 2] += norm * b

    if out is None:
        # All channels disabled — return black at first channel's plane size.
        ch0 = _extract_channel_plane(g, 0, z)
        out = np.zeros((*ch0.shape, 3), dtype=np.float32)

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")

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
    `frame` field for video / z-stack / channel-group sources so
    multiple panels using the same source can each show a different
    frame. Falls back to loaded_images for non-frame-bearing names.
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
    # Multichannel z-stack: composite at the panel's frame using the
    # group's tints / levels. This is the resolution-of-truth so the
    # panel keeps its OWN view of the source — the slider doesn't
    # mutate the group's shared `current_z` (which would bleed across
    # panels showing the same TIFF).
    if name in channel_groups:
        g = channel_groups[name]
        frame_num = int(getattr(panel, "frame", 0) or 0)
        return _composite_channel_group(name, z=frame_num)
    # Single-channel z-stack: extract the slice at panel.frame on demand.
    if name in loaded_zstacks:
        frame_num = int(getattr(panel, "frame", 0) or 0)
        return _extract_tiff_frame(loaded_zstacks[name], frame_num)
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


def _panel_zoom_insets(panel):
    """Return the zoom-insets list for a panel, falling back to the
    legacy singular `panel.zoom_inset` when the array is empty.
    Returns [] if `add_zoom_inset` is False or no inset is set."""
    if not getattr(panel, "add_zoom_inset", False):
        return []
    arr = list(getattr(panel, "zoom_insets", None) or [])
    if arr:
        return arr
    legacy = getattr(panel, "zoom_inset", None)
    if legacy is not None:
        return [legacy]
    return []


def _apply_zoom_target_self_overlays(cfg, processed, rows, cols):
    """Run the target panel's OWN overlay operations on the
    synthesized image written by _apply_adjacent_zoom_insets.

    Adjacent-zoom target panels have no `image_name`, so `process_panel`
    is never called on them — which previously meant any nested zoom
    inset (Standard / Separate), scale bar, line, or area annotation
    configured on the target was silently dropped during rendering.
    This pass replays the overlay portion of `process_panel` on each
    synthesized target image so 'parent → adjacent → standard zoom'
    chains actually display the nested zoom (and any other overlays
    the user placed on the target via Edit Panel).

    Adjacent-typed insets on the target itself are still handled by
    `_apply_adjacent_zoom_insets`'s main pass — that's how
    'parent → adjacent → adjacent' chains work — so we don't try to
    draw them here. `_draw_one_zoom_inset` already no-ops on
    Adjacent-typed entries.

    Labels and symbols continue to be added later via matplotlib in
    figure_builder._add_panel_labels / _add_panel_symbols, which
    already work for zoom targets because they iterate cfg.panels
    directly.
    """
    from image_processing import draw_zoom_inset, draw_scale_bar, draw_lines, draw_areas
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            # Only act on cells that are SYNTHESISED zoom targets:
            # no own image_name (or not loaded), but processed[r][c]
            # has been populated by _apply_adjacent_zoom_insets.
            if panel.image_name and panel.image_name in loaded_images:
                continue
            synth = processed[r][c]
            if synth is None:
                continue
            # 1. Zoom inset overlays — runs FIRST so the nested inset's
            #    internal crop reads a clean copy of the synthesised
            #    image (parallels the new ordering in process_panel).
            if getattr(panel, "add_zoom_inset", False):
                try:
                    synth = draw_zoom_inset(synth, panel, loaded_images)
                except Exception as e:
                    import sys; print(f"[zoom-target-overlay] draw_zoom_inset failed at ({r},{c}): {e}", file=sys.stderr)
            # 2. Scale bar — SKIPPED here. The matplotlib pass
            #    `_add_panel_scale_bars` runs over EVERY cell in the
            #    grid (zoom targets included) and would draw the same
            #    bar on top of our PIL one, producing the "ghosted"
            #    double-render the user reported. Drawing only via
            #    matplotlib keeps the bar crisp and matches how
            #    image-bearing panels' bars are rendered.
            mpp = panel.scale_bar.micron_per_pixel if getattr(panel, "scale_bar", None) else 1.0
            # 3. Lines
            if getattr(panel, "lines", None):
                try:
                    synth = draw_lines(synth, panel.lines, mpp)
                except Exception as e:
                    import sys; print(f"[zoom-target-overlay] draw_lines failed at ({r},{c}): {e}", file=sys.stderr)
            # 4. Areas
            if getattr(panel, "areas", None):
                try:
                    synth = draw_areas(synth, panel.areas, mpp)
                except Exception as e:
                    import sys; print(f"[zoom-target-overlay] draw_areas failed at ({r},{c}): {e}", file=sys.stderr)
            processed[r][c] = synth


def _apply_adjacent_zoom_insets(cfg, processed, rows, cols, image_override=None):
    """Replace target panels in `processed[][]` with zoomed crops for
    every Adjacent Panel zoom inset configured on any source panel.

    Iterates panel.zoom_insets (with legacy fallback to [zoom_inset]).
    Each Adjacent Panel inset's `side` field points the result at one
    of (r-1, c) / (r+1, c) / (r, c-1) / (r, c+1). Out-of-bounds
    targets are silently skipped. When two insets target the same
    cell, the LATER one wins (last write).

    Multi-pass cascade: chains like A → B → C where A is at (0,1),
    B at (0,0), C at (1,0) would have failed in single-pass row-major
    order because B's adjacent inset reads `processed[B]` which is
    still empty when we visit B before A. So we keep re-running the
    loop until no new cells get written, with a safety cap at
    rows*cols passes (any real chain is much shorter).

    Mutates processed[][] in place. Used by /api/preview,
    /api/figure/save, and the video render worker so multi-figure
    arrays land in their adjacent slots regardless of which path the
    user invokes.

    image_override: optional Dict[(r, c), PIL.Image] that overrides
    `_get_panel_image()` when reading the source panel's frame. Used
    by the video render worker to feed the CURRENT frame_idx's frame
    into the cascade — without this, video-source insets always
    sampled the panel's static `frame` field and zoom targets stayed
    frozen while the source animated.
    """
    # A cell is "ready" to act as a SOURCE when either it has an
    # image_name in loaded_images / loaded_videos, OR an earlier
    # pass already wrote a synthesised image into it. Without
    # this gate, a chained inset whose parent hasn't run yet would
    # crop stale / blank pixels from `processed[r][c]`.
    ready = [[False] * cols for _ in range(rows)]
    for rr in range(rows):
        for cc in range(cols):
            p = cfg.panels[rr][cc]
            nm = getattr(p, "image_name", "") or ""
            if nm and (nm in loaded_images or nm in loaded_videos):
                ready[rr][cc] = True

    # Insets we've already executed (by id-of-zi object) so we
    # don't re-fire the same one across passes — that'd produce
    # zoom-of-zoom-of-zoom drift on the same target.
    fired: set = set()
    max_passes = max(2, rows * cols)
    for _pass in range(max_passes):
        wrote_something = False
        for r in range(rows):
            for c in range(cols):
                panel = cfg.panels[r][c]
                # The source cell must be ready before we can read
                # pixels from it. Chained children wait their turn.
                if not ready[r][c]:
                    continue
                for zi in _panel_zoom_insets(panel):
                    if zi is None or zi.inset_type != "Adjacent Panel":
                        continue
                    if id(zi) in fired:
                        continue
                    ar, ac = r, c
                    side = zi.side or "Right"
                    if side == "Top": ar -= 1
                    elif side == "Bottom": ar += 1
                    elif side == "Left": ac -= 1
                    elif side == "Right": ac += 1
                    if not (0 <= ar < rows and 0 <= ac < cols):
                        fired.add(id(zi))
                        continue
                    ext_name = getattr(zi, "separate_image_name", "") or ""
                    if ext_name and ext_name not in ("", "select") and ext_name in loaded_images:
                        ext_img = loaded_images[ext_name].convert("RGB")
                        xi = getattr(zi, "x_inset", 0) or 0
                        yi = getattr(zi, "y_inset", 0) or 0
                        wi = getattr(zi, "width_inset", ext_img.size[0]) or ext_img.size[0]
                        hi = getattr(zi, "height_inset", ext_img.size[1]) or ext_img.size[1]
                        xi = max(0, min(xi, ext_img.size[0]-1))
                        yi = max(0, min(yi, ext_img.size[1]-1))
                        wi = max(1, min(wi, ext_img.size[0]-xi))
                        hi = max(1, min(hi, ext_img.size[1]-yi))
                        region = ext_img.crop((xi, yi, xi+wi, yi+hi))
                        zw = max(1, int(wi * zi.zoom_factor))
                        zh = max(1, int(hi * zi.zoom_factor))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region
                        ready[ar][ac] = True
                        wrote_something = True
                        fired.add(id(zi))
                    else:
                        p_src = cfg.panels[r][c]
                        # Re-process the source panel WITHOUT the zoom
                        # inset overlay so the cropped content is clean —
                        # i.e. no rectangle-border artifacts smearing into
                        # the adjacent-panel result, especially under
                        # rotation where the rect border would otherwise
                        # rotate into diagonal lines inside the crop.
                        #
                        # Video sources have their image_name in
                        # `loaded_videos`, not `loaded_images`; use
                        # `_get_panel_image` (which handles both) as the
                        # gate so video panels can still feed the
                        # adjacent zoom.
                        #
                        # When `image_override` carries an entry for
                        # this source cell (video render worker), use
                        # THAT frame so insets animate frame-by-frame
                        # instead of sampling the panel's static
                        # `frame` field repeatedly.
                        if image_override and (r, c) in image_override:
                            p_src_img = image_override[(r, c)]
                        else:
                            p_src_img = _get_panel_image(p_src) if p_src.image_name else None
                        if p_src_img is not None:
                            clean_panel = _from_dict(PanelInfo, _to_dict(p_src))
                            clean_panel.add_zoom_inset = False
                            try:
                                # skip_annotations=True keeps the source's
                                # scale bar / lines / areas out of the
                                # cropped region that becomes the adjacent
                                # zoom — annotations are overlays, not
                                # part of the image content we're zooming
                                # into. The crop here corresponds to the
                                # zoom rectangle on the source panel.
                                main_img = process_panel(
                                    p_src_img,
                                    clean_panel, min_dims, loaded_images,
                                    skip_labels=True, skip_symbols=True,
                                    skip_annotations=True,
                                )
                            except Exception:
                                main_img = processed[r][c]
                        else:
                            # For chained adjacent zooms (source itself is
                            # an adjacent-zoom TARGET with no image_name),
                            # fall back to whatever the previous iteration
                            # already wrote into processed[r][c] — that's
                            # the synthesised "primary" zoom image whose
                            # content the secondary zoom is supposed to
                            # crop from.
                            main_img = processed[r][c]
                        if main_img is None:
                            continue
                        miw, mih = main_img.size
                        if p_src.crop_image and p_src.crop and len(p_src.crop) == 4:
                            fw = p_src.crop[2] - p_src.crop[0]
                            fh = p_src.crop[3] - p_src.crop[1]
                        elif p_src.image_name and p_src.image_name in loaded_images:
                            fw, fh = loaded_images[p_src.image_name].size
                        elif p_src.image_name and p_src.image_name in loaded_videos:
                            # Video source — its image_name lives in
                            # loaded_videos, NOT loaded_images. Use the
                            # video frame's natural size as the source
                            # coord system (matches the dialog's
                            # origFullW/H from getImageInfo, which returns
                            # the video's frame size).
                            vid_img = _get_panel_image(p_src)
                            fw, fh = vid_img.size if vid_img is not None else (miw, mih)
                        else:
                            # Source is a synthesised zoom TARGET (no
                            # image_name of its own — chained adjacent
                            # zoom). The Edit Panel dialog has no
                            # origFullW for such panels, so its
                            # `ziActualW` falls back to the hardcoded
                            # 1000-px default canvas (see EditPanelDialog
                            # `const ziActualW = ... origFullW > 0 ? origFullW : 1000`).
                            # Inset coords are therefore in 1000-px
                            # space, not in main_img's natural pixels.
                            # Using main_img.size as fw, fh would treat
                            # 294-in-1000-space as 294 pixels of a 200-
                            # px image, sending the crop off the edge
                            # and producing a black secondary zoom.
                            fw, fh = 1000, 1000
                        scx = miw / max(fw, 1)
                        scy = mih / max(fh, 1)
                        # Source rectangle in main_img coords. We keep the
                        # full extent (no border-width inset needed because
                        # we cropped from a clean source above).
                        cx1 = max(0, int(zi.x * scx))
                        cy1 = max(0, int(zi.y * scy))
                        cx2 = min(miw, int((zi.x + zi.width) * scx))
                        cy2 = min(mih, int((zi.y + zi.height) * scy))
                        cx2 = max(cx1 + 1, cx2)
                        cy2 = max(cy1 + 1, cy2)
                        rot = float(getattr(zi, "rotation", 0) or 0)
                        if abs(rot) > 0.01:
                            # CSS rotate(+R) = CW; PIL Image.rotate(+R) =
                            # CCW. Use +rot (= CCW in PIL) to un-rotate
                            # the CW-tilted CSS rect so a plain crop at
                            # the original (cx1, cy1, cx2, cy2) yields
                            # axis-aligned content of a tilted source.
                            ccx = (cx1 + cx2) / 2.0
                            ccy = (cy1 + cy2) / 2.0
                            rot_img = main_img.rotate(rot, center=(ccx, ccy), resample=Image.BICUBIC)
                            region = rot_img.crop((cx1, cy1, cx2, cy2))
                        else:
                            region = main_img.crop((cx1, cy1, cx2, cy2))
                        zw = max(1, int(zi.width * zi.zoom_factor * scx))
                        zh = max(1, int(zi.height * zi.zoom_factor * scy))
                        region = region.resize((zw, zh), Image.LANCZOS)
                        processed[ar][ac] = region
                        ready[ar][ac] = True
                        wrote_something = True
                        fired.add(id(zi))
        if not wrote_something:
            break


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
                # Save to temp file first so tifffile / PIL can both
                # read it (and so multi-frame seeks work for z-stacks).
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix)
                tmp.write(data)
                tmp.close()

                # First check: is this a multichannel TIFF? If yes,
                # build a channel group and use the composite as the
                # initial image. Multichannel z-stacks (ZCYX, etc.)
                # also register a zstack mapping so the existing
                # frame-seek UI can drive `current_z`.
                multi = _detect_multichannel_tiff(tmp.name)
                if multi is not None:
                    axes, arr = multi
                    g = _init_channel_group(f.filename, axes, arr)
                    img = _composite_channel_group(f.filename)
                    loaded_images[f.filename] = img
                    if g.num_z > 1:
                        loaded_zstacks[f.filename] = tmp.name
                        zstack_frames[f.filename] = 0
                        zstack_counts[f.filename] = g.num_z
                    names.append(f.filename)
                else:
                    # Single-channel: fall back to the legacy PIL path.
                    img_obj = Image.open(tmp.name)
                    n_frames = getattr(img_obj, "n_frames", 1)
                    if n_frames > 1:
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
    # If this is a channel group, the frame_count is num_z (the user
    # advances Z, not the flat TIFF page index).
    if name in channel_groups:
        g = channel_groups[name]
        sample = _extract_channel_plane(g, 0, g.current_z)
        return {
            "frame_count": g.num_z,
            "width": int(sample.shape[1]),
            "height": int(sample.shape[0]),
            "multichannel": True,
            "num_channels": g.num_channels,
        }
    return _get_zstack_info(loaded_zstacks[name])

@app.get("/api/zstack/{name}/frame/{frame_num}")
def get_zstack_frame(name: str, frame_num: int, row: Optional[int] = None, col: Optional[int] = None):
    if name not in loaded_zstacks and name not in channel_groups:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    if name in channel_groups:
        # Multichannel: advance the group's current_z and re-composite.
        g = channel_groups[name]
        g.current_z = max(0, min(int(frame_num), g.num_z - 1))
        img = _composite_channel_group(name)
    else:
        img = _extract_tiff_frame(loaded_zstacks[name], frame_num)

    # If row/col provided, store the seeked frame ON THE PANEL ITSELF
    # via `panel.frame`. The rendering pipeline reads this via
    # `_get_panel_image` so each panel keeps its OWN view of the source
    # — no snapshot keys, no panel.image_name rewrites. This means:
    #   • Two panels can show the same multichannel TIFF at different Z
    #   • Channel toggles (tints / names) still resolve via the original
    #     image_name → channel_groups[name]
    #   • Browsing slices doesn't "freeze" the panel to a static image
    if row is not None and col is not None:
        if row < cfg.rows and col < cfg.cols:
            cfg.panels[row][col].frame = frame_num
        zstack_frames[f"__zstack_{name}_r{row}c{col}"] = frame_num
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
    # No image_name rewrite: panels keep their original multichannel
    # source and the renderer reads panel.frame to pick the slice.
    return {"frame": frame_num, "width": img.size[0], "height": img.size[1],
            "thumbnail": preview_b64}

@app.get("/api/zstack/list")
def list_zstacks():
    return {"zstacks": list(loaded_zstacks.keys())}


# ── Z-stack alignment ──────────────────────────────────────────────────────
#
# Two implementations:
#   1. ImageJ "Linear Stack Alignment with SIFT" — full SIFT feature
#      matching. Best quality but requires Fiji installed on the host.
#   2. OpenCV phase correlation — translation-only alignment using
#      `cv2.phaseCorrelate`. Always available (cv2 is already bundled).
#
# Both produce a new aligned z-stack stored under a derived name so the
# original isn't destroyed. The frontend can then re-load the aligned
# stack by switching the panel's image_name.

# Job-tracking dict for /align — keeps progress + result so the
# frontend can poll instead of blocking on a multi-minute fetch (which
# browsers abort with "Failed to fetch" after a few minutes of silence).
_align_jobs: Dict[str, Dict] = {}
_ALIGN_JOBS_MAX = 32


def _new_align_job() -> str:
    """Allocate a job id, LRU-evicting old completed entries."""
    import uuid
    job_id = uuid.uuid4().hex[:12]
    _align_jobs[job_id] = {
        "status": "starting",      # starting | running | done | error
        "progress": 0.0,           # 0..1
        "stage": "queued",          # human-readable
        "result": None,
        "error": None,
    }
    if len(_align_jobs) > _ALIGN_JOBS_MAX:
        first = next(iter(_align_jobs))
        if first != job_id:
            del _align_jobs[first]
    return job_id


@app.get("/api/zstack/align/availability")
def zstack_align_availability():
    """Tell the frontend which alignment algorithms are usable here.
    `sift` requires ImageJ/Fiji on the host; `phase_correlation` is
    always available via OpenCV."""
    path, kind = _find_imagej_executable()
    return {
        "sift": {"available": bool(path), "kind": kind, "path": path or ""},
        "phase_correlation": {"available": True, "kind": "opencv"},
    }


class ZStackAlignRequest(BaseModel):
    method: str = "sift"  # "sift" | "phase_correlation"
    start_frame: int = 0
    end_frame: int = -1
    # Channel index to use for alignment when source is multichannel
    # (default = first enabled channel). Translations computed on this
    # channel are applied to all channels so the registration stays in
    # sync across colours.
    align_channel: int = -1
    # SIFT fine controls — defaults match Fiji's "Linear Stack Alignment
    # with SIFT" defaults. See https://imagej.net/Linear_Stack_Alignment_with_SIFT
    sift_initial_gaussian_blur: float = 1.6
    sift_steps_per_scale_octave: int = 3
    sift_minimum_image_size: int = 64
    sift_maximum_image_size: int = 1024
    sift_feature_descriptor_size: int = 4
    sift_feature_descriptor_orientation_bins: int = 8
    sift_closest_next_closest_ratio: float = 0.92
    sift_maximal_alignment_error: float = 25.0
    sift_inlier_ratio: float = 0.05
    sift_expected_transformation: str = "Rigid"   # Translation | Rigid | Similarity | Affine
    sift_interpolate: bool = True
    # Phase correlation fine controls
    pc_window: str = "hann"       # hann | rect — windowing for cv2.phaseCorrelate
    pc_max_shift_frac: float = 0.25  # discard suggested shifts beyond this fraction of width
    # ── Performance knobs (SIFT) ──
    # Cap the input frame size for SIFT before invoking Fiji. SIFT runtime
    # scales superlinearly with pixel count, and the algorithm itself
    # downsamples internally to `sift_maximum_image_size` anyway — so
    # giving it 2048² when SIFT max is 1024 is just wasted I/O. Default
    # 1024 keeps a 114-frame stack alignable in a couple of minutes
    # instead of >10. Set to 0 to disable.
    align_max_dim: int = 1024
    # subprocess timeout for Fiji; users with huge stacks may want more.
    timeout_sec: int = 1800
    # ── Alignment reference source ──
    #   "max"      — element-wise max-projection across enabled channels
    #                (default; pools features from every channel, robust
    #                when no single channel has good signal everywhere)
    #   "mean"     — element-wise mean across enabled channels
    #   "sum"      — element-wise sum (clipped to uint8 max)
    #   "channel"  — use only `align_channel` (legacy single-channel)
    alignment_source: str = "max"
    # Apply CLAHE (contrast-limited adaptive histogram equalization)
    # before alignment. Dramatically improves SIFT/phase-correlation
    # on biological volumes where intensity drifts in Z.
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8


def _recover_affines_from_alignment(orig: "np.ndarray", aligned: "np.ndarray",
                                     transformation: str = "Rigid",
                                     progress_cb=None) -> List["np.ndarray"]:
    """For each frame, compute the 2×3 affine that maps `orig[i]` to
    `aligned[i]`. Used after Fiji's SIFT plugin (which doesn't expose
    its transforms directly) so we can apply the same per-frame
    registration to channels other than the alignment reference.

    Uses cv2.findTransformECC with the motion model chosen by the user
    (Translation/Euclidean/Affine/Homography). Returns identity for any
    frame where ECC fails to converge.
    """
    import cv2
    n = orig.shape[0]
    H, W = orig.shape[-2:]
    # Map our names → cv2 motion-model constants
    motion = {
        "Translation": cv2.MOTION_TRANSLATION,
        "Rigid": cv2.MOTION_EUCLIDEAN,        # rotation + translation
        "Similarity": cv2.MOTION_EUCLIDEAN,   # cv2 has no similarity, fall back to euclidean
        "Affine": cv2.MOTION_AFFINE,
    }.get(transformation, cv2.MOTION_EUCLIDEAN)
    transforms: List[np.ndarray] = []
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    for i in range(n):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            o = orig[i].astype(np.float32) / 255.0
            a = aligned[i].astype(np.float32) / 255.0
            cc, warp = cv2.findTransformECC(a, o, warp, motion, crit, None, 1)
        except Exception as e:
            print(f"[align-recover] frame {i} ECC failed: {e}", file=sys.stderr, flush=True)
        transforms.append(warp.astype(np.float32))
        if progress_cb:
            progress_cb(i + 1, n)
    return transforms


def _apply_affines_to_channel_stack(stack: "np.ndarray", transforms: List["np.ndarray"]) -> "np.ndarray":
    """Apply per-frame 2×3 affines to `stack` (Z, Y, X), returning a
    same-shape aligned stack. Frames without a transform pass through."""
    import cv2
    n = stack.shape[0]
    H, W = stack.shape[-2:]
    out = np.zeros_like(stack)
    for i in range(n):
        if i < len(transforms):
            out[i] = cv2.warpAffine(stack[i], transforms[i], (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        else:
            out[i] = stack[i]
    return out


def _align_phase_correlation(stack: "np.ndarray", body: ZStackAlignRequest):
    """OpenCV phaseCorrelate-based translation alignment.

    `stack` is (Z, Y, X) uint8 or float. Returns (aligned_stack, shifts)
    where shifts is [(dx0, dy0), ...] of length Z. Frame 0 is anchored;
    every subsequent frame is aligned to its predecessor's *aligned*
    position (cumulative shift) so we don't drift far from frame 0.
    """
    import cv2
    n = stack.shape[0]
    H, W = stack.shape[-2:]
    max_shift_px = max(2.0, float(body.pc_max_shift_frac) * min(H, W))
    # Hann window damps edge artefacts which is the standard prerequisite.
    if body.pc_window == "hann":
        win = (np.hanning(H)[:, None] * np.hanning(W)[None, :]).astype(np.float32)
    else:
        win = np.ones((H, W), dtype=np.float32)
    out = np.zeros_like(stack)
    out[0] = stack[0]
    shifts = [(0.0, 0.0)]
    cum_dx, cum_dy = 0.0, 0.0
    prev = stack[0].astype(np.float32)
    for i in range(1, n):
        cur = stack[i].astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev * win, cur * win)
        # Clamp to max_shift_px
        if abs(dx) > max_shift_px or abs(dy) > max_shift_px:
            dx, dy = 0.0, 0.0
        cum_dx += dx
        cum_dy += dy
        M = np.float32([[1, 0, -cum_dx], [0, 1, -cum_dy]])
        warped = cv2.warpAffine(stack[i], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        out[i] = warped
        shifts.append((float(cum_dx), float(cum_dy)))
        prev = cur
    return out, shifts


def _align_sift_imagej(stack: "np.ndarray", body: ZStackAlignRequest, ij_path: str):
    """Run Fiji's "Linear Stack Alignment with SIFT" headless against
    `stack` (Z, Y, X). Returns (aligned_stack, log) — log includes the
    raw stdout/stderr from Fiji which we surface to the UI on failure.

    Times out cleanly via HTTPException 504 instead of bubbling
    `TimeoutExpired` (which produces an opaque 500 / connection drop
    that the browser reports as "Failed to fetch").
    """
    import subprocess as _sp, tempfile as _tf, os as _os, time as _time
    import tifffile as _tf2
    import sys as _sys
    workdir = _tf.mkdtemp(prefix="mpfig_align_")
    try:
        in_tif = _os.path.join(workdir, "in.tif")
        out_tif = _os.path.join(workdir, "out.tif")
        _tf2.imwrite(in_tif, stack)
        # Build the SIFT command-string. We replicate Fiji's exact param
        # names so users running the same algorithm in Fiji's GUI can
        # reproduce these results.
        opts = (
            f"initial_gaussian_blur={body.sift_initial_gaussian_blur} "
            f"steps_per_scale_octave={body.sift_steps_per_scale_octave} "
            f"minimum_image_size={body.sift_minimum_image_size} "
            f"maximum_image_size={body.sift_maximum_image_size} "
            f"feature_descriptor_size={body.sift_feature_descriptor_size} "
            f"feature_descriptor_orientation_bins={body.sift_feature_descriptor_orientation_bins} "
            f"closest/next_closest_ratio={body.sift_closest_next_closest_ratio} "
            f"maximal_alignment_error={body.sift_maximal_alignment_error} "
            f"inlier_ratio={body.sift_inlier_ratio} "
            f"expected_transformation={body.sift_expected_transformation}"
            + (" interpolate" if body.sift_interpolate else "")
        )
        in_macro  = in_tif.replace("\\", "/")
        out_macro = out_tif.replace("\\", "/")
        macro = (
            f'open("{in_macro}");\n'
            f'run("Linear Stack Alignment with SIFT", "{opts}");\n'
            f'saveAs("Tiff", "{out_macro}");\n'
            f'run("Quit");\n'
        )
        macro_path = _os.path.join(workdir, "align.ijm")
        with open(macro_path, "w") as fh:
            fh.write(macro)
        cmd = [ij_path, "--headless", "--console", "-macro", macro_path]
        t0 = _time.monotonic()
        print(f"[align-sift] running {' '.join(cmd)} (timeout={body.timeout_sec}s)",
              file=_sys.stderr, flush=True)
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=max(60, int(body.timeout_sec)))
        except _sp.TimeoutExpired as te:
            # Clean 504 with a hint, instead of letting the unhandled
            # TimeoutExpired drop the HTTP connection (which the browser
            # surfaces as "Failed to fetch").
            elapsed = _time.monotonic() - t0
            raise HTTPException(
                504,
                (f"SIFT alignment timed out after {elapsed:.0f}s (Fiji subprocess limit). "
                 f"Tips: reduce frame range, lower `align_max_dim` (current={body.align_max_dim}px), "
                 f"or pick 'Translation' transformation. Partial stdout: {(te.stdout or '')[-400:]}")
            ) from None
        if not _os.path.isfile(out_tif):
            raise HTTPException(500, f"SIFT alignment produced no output. ImageJ stdout: {proc.stdout[-400:]} stderr: {proc.stderr[-400:]}")
        out_arr = _tf2.imread(out_tif)
        elapsed = _time.monotonic() - t0
        print(f"[align-sift] done in {elapsed:.1f}s, out shape {out_arr.shape}", file=_sys.stderr, flush=True)
        return out_arr, {"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "elapsed_sec": f"{elapsed:.1f}"}
    finally:
        try:
            import shutil as _shutil
            _shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def _run_alignment_job(job_id: str, name: str, body: ZStackAlignRequest):
    """Background-thread implementation of alignment. Updates the job
    record so the polling endpoint can report progress + outcome.

    Multichannel-preserving behaviour:
      1. Compute per-frame transforms on the reference (alignment) channel
         (phase_correlation gives shifts directly; SIFT runs through Fiji,
         then we recover transforms via cv2.findTransformECC by comparing
         input vs Fiji-aligned frames).
      2. Apply the same transforms to EVERY channel of the source stack.
      3. Write a ZCYX multichannel TIFF and register it as a new channel
         group inheriting the parent's tints / names / levels.
    """
    job = _align_jobs.get(job_id)
    if job is None:
        return  # job got evicted before it even started
    job["status"] = "running"
    try:
        job["stage"] = "loading source"
        job["progress"] = 0.02
        grp = channel_groups.get(name)
        method = body.method.lower()
        sift_path, _ = _find_imagej_executable()
        if method == "sift" and not sift_path:
            raise HTTPException(400, "SIFT alignment requested but ImageJ/Fiji is not installed on this host.")

        # Build full (Z, C?, Y, X) array for the source.
        if grp is not None:
            a = grp.arr
            ax = grp.axes
            while "T" in ax:
                i = ax.index("T")
                a = np.take(a, 0, axis=i)
                ax = ax[:i] + ax[i+1:]
            if "Z" not in ax:
                raise HTTPException(400, "Source has no Z axis to align over")
            order = [ax.index(c) for c in "ZCYX" if c in ax]
            sub_axes = "".join(c for c in "ZCYX" if c in ax)
            a = np.transpose(a, order)
            has_c = "C" in sub_axes
            nz = a.shape[0]
            start = max(0, body.start_frame)
            end = body.end_frame if body.end_frame >= 0 else nz - 1
            end = min(end, nz - 1)
            if has_c:
                full = a[start:end+1]   # (Z, C, Y, X)
                n_ch = full.shape[1]
                align_channel = body.align_channel
                if align_channel < 0:
                    align_channel = next((i for i, en in enumerate(grp.enabled) if en), 0)
                align_channel = max(0, min(align_channel, n_ch - 1))
            else:
                full = a[start:end+1][:, np.newaxis, :, :]  # (Z, 1, Y, X)
                n_ch = 1
                align_channel = 0
        else:
            tiff_path = loaded_zstacks[name]
            img_obj = Image.open(tiff_path)
            n_frames = getattr(img_obj, "n_frames", 1)
            start = max(0, body.start_frame)
            end = body.end_frame if body.end_frame >= 0 else n_frames - 1
            end = min(end, n_frames - 1)
            frames = []
            for i in range(start, end + 1):
                img_obj.seek(i)
                frames.append(np.array(img_obj.convert("L"), dtype=np.uint8))
            full = (np.stack(frames, axis=0)
                    if frames else np.zeros((1, 64, 64), dtype=np.uint8))
            full = full[:, np.newaxis, :, :]  # (Z, 1, Y, X)
            n_ch = 1
            align_channel = 0

        Z, C, H, W = full.shape

        # ── Build the alignment reference stack ─────────────────────
        # The reference is what SIFT/phase-correlation actually sees.
        # Pooling features across channels (max/mean/sum) is more robust
        # than picking a single channel — especially when the chosen
        # channel has weak signal in parts of the volume.
        job["stage"] = "preparing reference"
        job["progress"] = 0.04
        src_mode = (body.alignment_source or "max").lower()
        if grp is not None and C > 1:
            enabled = [bool(e) for e in (grp.enabled or [True] * C)]
            # Index of channels to combine (or just the one when mode="channel")
            keep_idx = [i for i, e in enumerate(enabled) if e] or list(range(C))
        else:
            keep_idx = [0]
            src_mode = "channel"

        if src_mode == "channel":
            ref_full_f = full[:, align_channel].astype(np.float32)
            ref_max = max(float(ref_full_f.max()), 1.0)
            stage_label = f"channel {align_channel + 1}"
        else:
            sub_f = full[:, keep_idx].astype(np.float32)  # (Z, k, Y, X)
            if src_mode == "max":
                ref_full_f = sub_f.max(axis=1)
                stage_label = f"max-projection of {len(keep_idx)} channel(s)"
            elif src_mode == "mean":
                ref_full_f = sub_f.mean(axis=1)
                stage_label = f"mean of {len(keep_idx)} channel(s)"
            elif src_mode == "sum":
                ref_full_f = sub_f.sum(axis=1)
                stage_label = f"sum of {len(keep_idx)} channel(s)"
            else:
                # Unknown mode falls back to max-projection (the safest default).
                ref_full_f = sub_f.max(axis=1)
                stage_label = f"max-projection (fallback) of {len(keep_idx)} channel(s)"
            ref_max = max(float(ref_full_f.max()), 1.0)

        # Optional CLAHE pre-processing — applied per-slice. Helps when
        # intensity drops off in Z (common in light-sheet) by locally
        # equalising contrast so SIFT finds features in dim regions.
        if body.use_clahe:
            import cv2 as _cv2
            tile = max(2, int(body.clahe_tile_grid))
            clahe = _cv2.createCLAHE(
                clipLimit=max(0.1, float(body.clahe_clip_limit)),
                tileGridSize=(tile, tile),
            )
            tmp_u8 = np.clip(ref_full_f / ref_max * 255.0, 0, 255).astype(np.uint8)
            for i in range(Z):
                tmp_u8[i] = clahe.apply(tmp_u8[i])
            ref_full_f = tmp_u8.astype(np.float32)
            ref_max = max(float(ref_full_f.max()), 1.0)
            stage_label += " + CLAHE"
        job["stage"] = f"reference = {stage_label}"

        # Optional downsample for the *transform computation only*. The
        # final aligned output stays at full resolution because we apply
        # the (rescaled) transforms back to the original-size channels.
        align_max = max(0, int(body.align_max_dim or 0))
        if align_max > 0 and max(H, W) > align_max:
            import cv2 as _cv2
            scale = align_max / max(H, W)
            new_h = max(8, int(H * scale))
            new_w = max(8, int(W * scale))
            small_ref = np.zeros((Z, new_h, new_w), dtype=np.uint8)
            for i in range(Z):
                norm = np.clip(ref_full_f[i] / ref_max * 255.0, 0, 255)
                small_ref[i] = _cv2.resize(norm.astype(np.uint8), (new_w, new_h), interpolation=_cv2.INTER_AREA)
            ref_u8 = small_ref
            ds_scale = scale
        else:
            ref_u8 = np.clip(ref_full_f / ref_max * 255.0, 0, 255).astype(np.uint8)
            ds_scale = 1.0

        job["stage"] = f"running {method} on {stage_label}"
        job["progress"] = 0.10

        if method == "phase_correlation":
            # Single-pass: phase correlation gives us (dx, dy) per frame
            # directly, so we skip ECC recovery and apply translation
            # transforms to all channels.
            _, shifts_tuples = _align_phase_correlation(ref_u8, body)
            # Rescale shifts back to original resolution
            transforms: List[np.ndarray] = []
            for (dx, dy) in shifts_tuples:
                T = np.float32([[1, 0, -dx / max(ds_scale, 1e-6)],
                                [0, 1, -dy / max(ds_scale, 1e-6)]])
                transforms.append(T)
            shifts_out = [[float(s[0]), float(s[1])] for s in shifts_tuples]
            log: Dict[str, str] = {}
        elif method == "sift":
            # Run Fiji SIFT on the reference channel
            aligned_ref_u8, log = _align_sift_imagej(ref_u8, body, sift_path)
            job["stage"] = "recovering per-frame transforms from SIFT output"
            job["progress"] = 0.55
            def _recover_cb(done, total):
                job["progress"] = 0.55 + 0.30 * (done / max(total, 1))
                job["stage"] = f"recovering transforms ({done}/{total})"
            small_transforms = _recover_affines_from_alignment(
                ref_u8, aligned_ref_u8, body.sift_expected_transformation, progress_cb=_recover_cb
            )
            # Upscale the affine translation components to full-res
            transforms = []
            for T in small_transforms:
                T_full = T.copy()
                if ds_scale != 1.0:
                    # Translation scales linearly; rotation matrix part
                    # stays identical (it's resolution-independent).
                    T_full[0, 2] = T[0, 2] / ds_scale
                    T_full[1, 2] = T[1, 2] / ds_scale
                transforms.append(T_full)
            shifts_out = [[float(T[0, 2]), float(T[1, 2])] for T in transforms]
        else:
            raise HTTPException(400, f"Unknown alignment method: {method!r}")

        # Apply transforms to every channel at original resolution
        job["stage"] = f"applying transforms to {C} channel(s)"
        job["progress"] = 0.85
        aligned_4d = np.zeros_like(full)  # (Z, C, Y, X)
        for c in range(C):
            aligned_4d[:, c] = _apply_affines_to_channel_stack(full[:, c], transforms)
            job["progress"] = 0.85 + 0.10 * ((c + 1) / max(C, 1))

        # Save: multichannel ZCYX TIFF if more than one channel,
        # otherwise the single-plane TIFF.
        job["stage"] = "writing aligned TIFF"
        job["progress"] = 0.96
        import tifffile as _tf2
        import tempfile as _tf
        aligned_path = _tf.NamedTemporaryFile(delete=False, suffix=".tif")
        aligned_path.close()
        if C > 1:
            _tf2.imwrite(aligned_path.name, aligned_4d, photometric='minisblack',
                         metadata={'axes': 'ZCYX'})
        else:
            _tf2.imwrite(aligned_path.name, aligned_4d[:, 0])

        aligned_name = f"{name}::aligned"
        loaded_zstacks[aligned_name] = aligned_path.name
        zstack_frames[aligned_name] = 0
        zstack_counts[aligned_name] = int(Z)
        if C > 1:
            # Re-register the aligned stack as a channel group so the UI
            # still shows tints / names. Inherit parent's metadata.
            new_grp = _init_channel_group(aligned_name, "ZCYX", aligned_4d)
            if grp is not None:
                # Inherit user-set tints / names / enabled / levels for
                # channels that overlap between source and aligned stack.
                for c in range(min(grp.num_channels, new_grp.num_channels)):
                    new_grp.tints[c]        = grp.tints[c]
                    new_grp.names[c]        = grp.names[c]
                    new_grp.enabled[c]      = grp.enabled[c]
                    new_grp.black_levels[c] = grp.black_levels[c]
                    new_grp.white_levels[c] = grp.white_levels[c]
            # Initial composite for the panel thumbnail
            loaded_images[aligned_name] = _composite_channel_group(aligned_name)
        else:
            loaded_images[aligned_name] = _extract_tiff_frame(aligned_path.name, 0)
        _recalc_min_dims()

        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = 1.0
        job["result"] = {
            "aligned_name": aligned_name,
            "method": method,
            "n_frames": int(Z),
            "n_channels": int(C),
            "shifts": shifts_out,
            "log": log,
        }
    except HTTPException as he:
        job["status"] = "error"
        job["error"] = he.detail if isinstance(he.detail, str) else str(he.detail)
        job["progress"] = 1.0
    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        job["progress"] = 1.0


@app.post("/api/zstack/{name}/align")
def align_zstack(name: str, body: ZStackAlignRequest):
    """Start an alignment job in the background. Returns a job_id; the
    client polls /api/zstack/align/status/{job_id} for progress and the
    final result. We chose async-with-polling over a blocking request
    because Fiji SIFT can take minutes on large stacks and the browser
    aborts long pending fetches as "Failed to fetch"."""
    if name not in loaded_zstacks and name not in channel_groups:
        raise HTTPException(404, f"Z-stack '{name}' not found")
    job_id = _new_align_job()
    import threading
    t = threading.Thread(target=_run_alignment_job, args=(job_id, name, body), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "started"}


@app.get("/api/zstack/align/status/{job_id}")
def align_status(job_id: str):
    job = _align_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Alignment job not found (may have been evicted)")
    return {
        "status": job["status"],   # starting | running | done | error
        "progress": float(job["progress"]),
        "stage": job["stage"],
        "result": job["result"],
        "error": job["error"],
    }


# ── Multichannel TIFF Endpoints ───────────────────────────────────────────

@app.get("/api/zstack/{name}/channels/info")
def get_channel_info(name: str):
    """Return per-channel state for a multichannel TIFF/z-stack.
    Returns 404 with `is_multichannel: false` for single-channel images,
    so the frontend can probe any image to discover whether the channel
    UI should be shown."""
    key = _resolve_channel_group_key(name)
    if key is None:
        return {"is_multichannel": False}
    g = channel_groups[key]
    return {
        "is_multichannel": True,
        "axes": g.axes,
        "num_channels": g.num_channels,
        "num_z": g.num_z,
        "current_z": g.current_z,
        "tints": list(g.tints),
        "enabled": list(g.enabled),
        "black_levels": list(g.black_levels),
        "white_levels": list(g.white_levels),
        "names": list(g.names),
    }


class ChannelUpdateRequest(BaseModel):
    # All optional — only provided fields are applied.
    tints: Optional[List[str]] = None
    enabled: Optional[List[bool]] = None
    black_levels: Optional[List[int]] = None
    white_levels: Optional[List[int]] = None
    current_z: Optional[int] = None
    names: Optional[List[str]] = None
    # When provided, also reassign panel.image_name = panel_key (mirrors
    # the per-panel pattern used by the frame endpoint, so two panels can
    # show the same multichannel TIFF with different tints).
    row: Optional[int] = None
    col: Optional[int] = None


@app.patch("/api/zstack/{name}/channels")
def update_channels(name: str, body: ChannelUpdateRequest):
    """Update channel-group state (tints, enabled, levels, z-slice) and
    recompose. Returns the new composite as a base64 PNG preview."""
    key = _resolve_channel_group_key(name)
    if key is None:
        raise HTTPException(404, f"'{name}' is not a multichannel image")
    # `name` may be a panel-specific snapshot key (e.g. __zstack_X_r0c0);
    # `key` is the canonical channel-group key.
    g = channel_groups[key]

    if body.tints is not None:
        # Length-guard: pad/truncate to num_channels.
        ts = list(body.tints)[:g.num_channels]
        while len(ts) < g.num_channels:
            ts.append("#ffffff")
        g.tints = ts
    if body.enabled is not None:
        es = list(body.enabled)[:g.num_channels]
        while len(es) < g.num_channels:
            es.append(True)
        g.enabled = es
    if body.black_levels is not None:
        bs = [max(0, min(255, int(v))) for v in body.black_levels][:g.num_channels]
        while len(bs) < g.num_channels:
            bs.append(0)
        g.black_levels = bs
    if body.white_levels is not None:
        ws = [max(0, min(255, int(v))) for v in body.white_levels][:g.num_channels]
        while len(ws) < g.num_channels:
            ws.append(255)
        g.white_levels = ws
    if body.names is not None:
        ns = [str(v)[:64] for v in body.names][:g.num_channels]
        while len(ns) < g.num_channels:
            ns.append(f"Ch {len(ns) + 1}")
        g.names = ns
    if body.current_z is not None:
        g.current_z = max(0, min(int(body.current_z), g.num_z - 1))

    # Tints / names / levels are SHARED state across all panels showing
    # this multichannel TIFF (analogous to a global LUT). Each panel
    # still gets its own Z slice via `panel.frame` (see _get_panel_image),
    # but the channel display settings apply uniformly.
    img = _composite_channel_group(key)
    loaded_images[key] = img

    _recalc_min_dims()
    preview = img.copy()
    max_dim = 1200
    if max(preview.size) > max_dim:
        ratio = max_dim / max(preview.size)
        preview = preview.resize((int(preview.size[0] * ratio), int(preview.size[1] * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    return {
        "thumbnail": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": img.size[0],
        "height": img.size[1],
        "current_z": g.current_z,
        "tints": list(g.tints),
        "enabled": list(g.enabled),
        "black_levels": list(g.black_levels),
        "white_levels": list(g.white_levels),
        "names": list(g.names),
    }


class ZStackProjectRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = -1   # -1 = last frame
    method: str = "max"   # max | avg | min

@app.post("/api/zstack/{name}/project")
def project_zstack(name: str, body: ZStackProjectRequest):
    """Create a projection (max/avg/min) from a range of z-stack frames.

    For multichannel z-stacks (sources registered in `channel_groups`),
    the projection is computed PER-CHANNEL and then composited using
    the user's tint colours + per-channel levels. This is what the user
    expects: a max-intensity projection that looks like the 2D view at
    each Z slice, just summarised across Z.

    For single-channel sources the legacy path applies — projection
    runs on the RGB-converted PIL frames.
    """
    if name not in loaded_zstacks and name not in channel_groups:
        raise HTTPException(404, f"Z-stack '{name}' not found")

    grp = channel_groups.get(name)
    if grp is not None:
        # ── Channel-aware projection ─────────────────────────────────
        # grp.arr is canonicalised (T already collapsed by _init_channel_group's
        # caller chain, but we re-check). Layout is "ZCYX" / "CYX" / "ZYX".
        a = grp.arr
        ax = grp.axes
        # Collapse any remaining T (defensive).
        while "T" in ax:
            i = ax.index("T")
            a = np.take(a, 0, axis=i)
            ax = ax[:i] + ax[i+1:]
        if "Z" not in ax:
            raise HTTPException(400, "Source has no Z axis to project over")
        zi = ax.index("Z")
        nz = a.shape[zi]
        start = max(0, body.start_frame)
        end = body.end_frame if body.end_frame >= 0 else nz - 1
        end = min(end, nz - 1)
        if start > end:
            start, end = end, start

        # Slice z range
        idx = [slice(None)] * a.ndim
        idx[zi] = slice(start, end + 1)
        sub = a[tuple(idx)]
        # Move Z to axis 0 if not already; move C to axis 1 if present.
        order = []
        for letter in ("Z", "C", "Y", "X"):
            if letter in ax:
                order.append(ax.index(letter))
        sub = np.transpose(sub, order)
        sub_axes = "".join(letter for letter in ("Z", "C", "Y", "X") if letter in ax)

        # Compute projection per channel
        method = body.method
        def _project(stack):
            if method == "max":  return np.max(stack, axis=0)
            if method == "min":  return np.min(stack, axis=0)
            if method == "avg":  return np.mean(stack, axis=0)
            raise HTTPException(400, f"Unknown projection method: {method}")

        if "C" in sub_axes:
            # sub now (Z, C, Y, X)
            projected_per_ch = _project(sub.astype(np.float32))  # (C, Y, X)
        else:
            # sub is (Z, Y, X); wrap as single channel for uniform code
            projected_per_ch = _project(sub.astype(np.float32))[np.newaxis, ...]  # (1, Y, X)

        # Composite using channel tints + per-channel windows.
        C = projected_per_ch.shape[0]
        H, W = projected_per_ch.shape[-2], projected_per_ch.shape[-1]
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(C):
            if c < len(grp.enabled) and not grp.enabled[c]:
                continue
            plane = projected_per_ch[c]
            denom = max(grp.max_vals[c] if c < len(grp.max_vals) else 1.0, 1.0)
            norm = plane / denom
            bl = grp.black_levels[c] / 255.0 if c < len(grp.black_levels) else 0.0
            wl = grp.white_levels[c] / 255.0 if c < len(grp.white_levels) else 1.0
            if wl > bl:
                norm = np.clip((norm - bl) / (wl - bl), 0.0, 1.0)
            else:
                norm = np.zeros_like(norm)
            hex_ = grp.tints[c] if c < len(grp.tints) else "#ffffff"
            try:
                tr = int(hex_[1:3], 16) / 255.0
                tg = int(hex_[3:5], 16) / 255.0
                tb = int(hex_[5:7], 16) / 255.0
            except (ValueError, IndexError):
                tr = tg = tb = 1.0
            rgb[..., 0] += norm * tr
            rgb[..., 1] += norm * tg
            rgb[..., 2] += norm * tb
        result = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        projected = Image.fromarray(result, "RGB")
    else:
        # ── Legacy single-channel / PIL projection path ──────────────
        tiff_path = loaded_zstacks[name]
        img_obj = Image.open(tiff_path)
        n_frames = getattr(img_obj, "n_frames", 1)

        start = max(0, body.start_frame)
        end = body.end_frame if body.end_frame >= 0 else n_frames - 1
        end = min(end, n_frames - 1)
        if start > end:
            start, end = end, start

        frames = []
        for i in range(start, end + 1):
            img_obj.seek(i)
            frame = np.array(img_obj.convert("RGB"), dtype=np.float32)
            frames.append(frame)
        if not frames:
            raise HTTPException(400, "No frames in range")
        stack = np.stack(frames, axis=0)

        if body.method == "max":   result = np.max(stack, axis=0).astype(np.uint8)
        elif body.method == "min": result = np.min(stack, axis=0).astype(np.uint8)
        elif body.method == "avg": result = np.mean(stack, axis=0).astype(np.uint8)
        else: raise HTTPException(400, f"Unknown projection method: {body.method}")
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
        "is_multichannel": grp is not None,
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

    # Try tifffile first (fast), fall back to PIL loop. We also capture
    # the axes string so we can locate channel / Z dimensions explicitly
    # — assuming "channels are last" is wrong for ZCYX / TZCYX files,
    # which would otherwise get sliced along X and produce a 1-pixel-
    # wide garbage volume.
    arr = None
    axes = ""
    try:
        import tifffile
        print(f"[nifti] Using tifffile to read {tiff_path}", file=sys.stderr, flush=True)
        with tifffile.TiffFile(tiff_path) as tf:
            if tf.series:
                axes = (tf.series[0].axes or "").upper()
            arr = tifffile.imread(tiff_path)
        print(f"[nifti] tifffile read {arr.shape} {arr.dtype} axes={axes!r}", file=sys.stderr, flush=True)
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
        axes = "ZYX"

    try:
        import nibabel as nib
    except ImportError as e:
        raise HTTPException(500, f"nibabel not bundled: {e}")

    # ── Reduce to a canonical layout, channel-aware ───────────────────
    # We try to preserve a 4D (Z, C, Y, X) shape when the source is
    # multichannel AND we have a channel_group for it, so the RGB
    # compositing branch below can apply the user's per-channel tints
    # + levels to produce a coloured 3D volume.  Otherwise we collapse
    # to (Z, Y, X) and emit a grayscale volume.
    if axes and len(axes) == arr.ndim:
        # Use the same canonicaliser the upload + channel-group paths
        # use, so I/Q/A/M/R/etc. all get interpreted identically.
        arr, axes = _canonicalize_tiff_axes(arr, axes)
        # Collapse T (we render a single time-point for 3D view).
        while "T" in axes:
            i = axes.index("T")
            arr = np.take(arr, 0, axis=i)
            axes = axes[:i] + axes[i+1:]
        # Promote pure-2D output to (1, Y, X) so we always emit at
        # least a 1-z volume.
        if "Z" not in axes:
            arr = arr[np.newaxis, ...]
            axes = "Z" + axes
    else:
        # No reliable axes string — fall back to shape-based reduction.
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
            axes = "ZYX"
        elif arr.ndim == 3:
            axes = "ZYX"
        elif arr.ndim == 4:
            # Pick the smallest-sized axis (other than 0 and the last
            # two, which are conventionally Z, Y, X) as the channel axis.
            shape = arr.shape
            candidate_axes = [i for i in range(arr.ndim) if i not in (0, arr.ndim - 1, arr.ndim - 2)]
            ch_axis = min(candidate_axes, key=lambda i: shape[i]) if candidate_axes else 1
            if ch_axis != 1:
                arr = np.moveaxis(arr, ch_axis, 1)
            axes = "ZCYX"

    # Does this volume have a channel axis AND a registered channel
    # group? If yes, take the RGB-composite path further below.
    grp = channel_groups.get(name) if "C" in axes else None
    use_rgb_composite = grp is not None and grp.num_channels == arr.shape[axes.index("C")]

    if not use_rgb_composite and "C" in axes:
        # No tint info available — collapse channels by max-projection
        # (preserves brightest signal per voxel; sensible grayscale
        # default for fluorescence and structural volumes alike).
        i = axes.index("C")
        arr = arr.max(axis=i)
        axes = axes[:i] + axes[i+1:]

    depth = arr.shape[0]

    # Slice to requested Z range
    start = max(0, body.start_frame)
    end = body.end_frame if body.end_frame >= 0 else depth - 1
    end = min(end, depth - 1)
    arr = arr[start:end + 1]

    z_spacing = max(0.05, float(body.z_spacing))

    if use_rgb_composite:
        # ─── RGB composite path ──────────────────────────────────────
        # arr is now (Z, C, Y, X). We normalize each channel against
        # its native max (so 16-bit channels span 0..1), apply the
        # user's black/white window, multiply by the channel's tint
        # colour, and sum into an RGB buffer. The same math as the
        # 2D `_composite_channel_group` so the 3D view exactly mirrors
        # what the user sees on the panel.
        z, c, h, w = arr.shape
        arr_f = arr.astype(np.float32)

        # Downsample BEFORE compositing — cheaper to interp grayscale
        # planes than per-RGB.
        max_dim = body.max_dim
        if max(w, h, z) > max_dim:
            import cv2
            scale = max_dim / max(w, h, z)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            new_z = max(1, int(z * scale))
            z_step = max(1, z // new_z)
            keep_zs = list(range(0, z, z_step))[:new_z]
            resized = np.zeros((len(keep_zs), c, new_h, new_w), dtype=np.float32)
            for out_i, in_i in enumerate(keep_zs):
                for ch in range(c):
                    resized[out_i, ch] = cv2.resize(arr_f[in_i, ch], (new_w, new_h), interpolation=cv2.INTER_AREA)
            arr_f = resized
            z, _, h, w = arr_f.shape

        # Compose RGB per voxel
        rgb = np.zeros((z, h, w, 3), dtype=np.float32)
        for ch in range(c):
            if ch < len(grp.enabled) and not grp.enabled[ch]:
                continue
            plane = arr_f[:, ch, :, :]
            denom = max(grp.max_vals[ch] if ch < len(grp.max_vals) else 1.0, 1.0)
            norm = plane / denom
            bl = grp.black_levels[ch] / 255.0 if ch < len(grp.black_levels) else 0.0
            wl = grp.white_levels[ch] / 255.0 if ch < len(grp.white_levels) else 1.0
            if wl > bl:
                norm = np.clip((norm - bl) / (wl - bl), 0.0, 1.0)
            else:
                norm = np.zeros_like(norm)
            hex_ = grp.tints[ch] if ch < len(grp.tints) else "#ffffff"
            try:
                tr = int(hex_[1:3], 16) / 255.0
                tg = int(hex_[3:5], 16) / 255.0
                tb = int(hex_[5:7], 16) / 255.0
            except (ValueError, IndexError):
                tr = tg = tb = 1.0
            rgb[..., 0] += norm * tr
            rgb[..., 1] += norm * tg
            rgb[..., 2] += norm * tb

        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255).astype(np.uint8)  # (Z, Y, X, 3)

        # NIfTI expects (X, Y, Z) spatial layout. Reorder + repack into
        # a structured-dtype array so nibabel emits NIFTI_TYPE_RGB24 (128).
        rgb_xyz = np.transpose(rgb_uint8, (2, 1, 0, 3))  # (X, Y, Z, 3)
        rgb_struct = np.zeros(rgb_xyz.shape[:3], dtype=[('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        rgb_struct['R'] = rgb_xyz[..., 0]
        rgb_struct['G'] = rgb_xyz[..., 1]
        rgb_struct['B'] = rgb_xyz[..., 2]

        affine = np.diag([1.0, 1.0, z_spacing, 1.0]).astype(np.float32)
        nii = nib.Nifti1Image(rgb_struct, affine=affine)
        nii.header["pixdim"][1] = 1.0
        nii.header["pixdim"][2] = 1.0
        nii.header["pixdim"][3] = z_spacing
        # Datatype 128 = NIFTI_TYPE_RGB24 — NiiVue renders these natively
        # without applying a colormap.

        out_d, out_h, out_w = rgb_uint8.shape[:3]
    else:
        # ─── Grayscale path (single channel or no tint info) ─────────
        # arr is (Z, Y, X)
        if arr.dtype != np.uint8:
            if arr.max() > 0:
                arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        d, h, w = arr.shape
        if max(w, h, d) > body.max_dim:
            import cv2
            scale = body.max_dim / max(w, h, d)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            new_d = max(1, int(d * scale))
            z_step = max(1, d // new_d)
            resized = []
            for i in range(0, d, z_step):
                if len(resized) >= new_d:
                    break
                resized.append(cv2.resize(arr[i], (new_w, new_h), interpolation=cv2.INTER_AREA))
            arr = np.stack(resized, axis=0)

        arr_nii = np.transpose(arr, (2, 1, 0))  # (Z, Y, X) → (X, Y, Z)
        affine = np.diag([1.0, 1.0, z_spacing, 1.0]).astype(np.float32)
        nii = nib.Nifti1Image(arr_nii, affine=affine)
        nii.header.set_data_dtype(np.uint8)
        nii.header["pixdim"][1] = 1.0
        nii.header["pixdim"][2] = 1.0
        nii.header["pixdim"][3] = z_spacing
        out_d, out_h, out_w = arr.shape

    # Serialize to bytes
    buf = io.BytesIO()
    file_map = nib.Nifti1Image.make_file_map({"image": buf, "header": buf})
    nii.file_map = file_map
    nii.to_file_map()
    data_bytes = buf.getvalue()
    data_b64 = base64.b64encode(data_bytes).decode("ascii")

    return {
        "data": data_b64,
        "width": out_w,
        "height": out_h,
        "depth": out_d,
        "rgb": bool(use_rgb_composite),
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
    """Get a processed preview of a single panel with current settings applied.

    Handles three cases:
      • Image-bearing panel (static image in loaded_images, or video in
        loaded_videos): process_panel on the source frame.
      • Adjacent-zoom TARGET panel (no image_name): synthesise from the
        source panel via the same _apply_adjacent_zoom_insets path the
        full preview uses, so the Edit dialog has a base image to show
        on its first open.
      • Empty cell: return empty image (frontend shows placeholder).
    """
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    panel = cfg.panels[r][c]
    # Try the regular image / video path first.
    src_img = _get_panel_image(panel) if panel.image_name else None
    if src_img is not None:
        # For edit dialog preview: skip ALL overlays (labels, symbols,
        # scale bar, lines, areas, zoom inset) — they're rendered as
        # interactive UI overlays in the frontend.
        panel_copy = _from_dict(PanelInfo, _to_dict(panel))
        panel_copy.add_zoom_inset = False
        panel_copy.add_scale_bar = False
        panel_copy.labels = []
        panel_copy.symbols = []
        panel_copy.lines = []
        panel_copy.areas = []
        processed = process_panel(src_img, panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
    elif not panel.image_name:
        # Adjacent-zoom target: synthesise via the cascade path so the
        # dialog can show what's actually in this cell. Without this,
        # the dialog opens with a blank base preview and the user
        # can't see what they're editing until they switch to an
        # overlay tab that triggers the rendered-preview fetch.
        rows, cols = cfg.rows, cfg.cols
        processed_grid = [[None for _ in range(cols)] for _ in range(rows)]
        # Seed every image-bearing cell.
        for sr in range(rows):
            for sc in range(cols):
                sp = cfg.panels[sr][sc]
                sp_img = _get_panel_image(sp) if sp.image_name else None
                if sp_img is None:
                    continue
                saved_z = sp.add_zoom_inset
                sp.add_zoom_inset = False
                try:
                    processed_grid[sr][sc] = process_panel(sp_img, sp, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
                except Exception:
                    pass
                finally:
                    sp.add_zoom_inset = saved_z
        if processed_grid[r][c] is None:
            processed_grid[r][c] = Image.new("RGB", (1, 1), "white")
        _apply_adjacent_zoom_insets(cfg, processed_grid, rows, cols)
        synth = processed_grid[r][c]
        if synth is None or synth.size == (1, 1):
            return {"image": ""}
        processed = synth
    else:
        return {"image": ""}
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
    Eliminates race conditions from separate PATCH + GET requests.

    Handles image, video, and adjacent-zoom-target panels (latter via
    cascade synthesis so the dialog has a base image to draw on).
    """
    if r >= cfg.rows or c >= cfg.cols:
        raise HTTPException(400, f"Panel [{r}][{c}] out of range")
    # 1. Patch
    cfg.panels[r][c] = _from_dict(PanelInfo, body.panel)
    panel = cfg.panels[r][c]
    # 2. Generate preview — skip ALL overlays (rendered as interactive UI overlays)
    src_img = _get_panel_image(panel) if panel.image_name else None
    if src_img is not None:
        panel_copy = _from_dict(PanelInfo, _to_dict(panel))
        panel_copy.add_zoom_inset = False
        panel_copy.add_scale_bar = False
        panel_copy.labels = []
        panel_copy.symbols = []
        panel_copy.lines = []
        panel_copy.areas = []
        processed = process_panel(src_img, panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
    elif not panel.image_name:
        rows, cols = cfg.rows, cfg.cols
        processed_grid = [[None for _ in range(cols)] for _ in range(rows)]
        for sr in range(rows):
            for sc in range(cols):
                sp = cfg.panels[sr][sc]
                sp_img = _get_panel_image(sp) if sp.image_name else None
                if sp_img is None:
                    continue
                saved_z = sp.add_zoom_inset
                sp.add_zoom_inset = False
                try:
                    processed_grid[sr][sc] = process_panel(sp_img, sp, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
                except Exception:
                    pass
                finally:
                    sp.add_zoom_inset = saved_z
        if processed_grid[r][c] is None:
            processed_grid[r][c] = Image.new("RGB", (1, 1), "white")
        _apply_adjacent_zoom_insets(cfg, processed_grid, rows, cols)
        synth = processed_grid[r][c]
        if synth is None or synth.size == (1, 1):
            return {"panel": _to_dict(panel), "image": ""}
        processed = synth
    else:
        return {"panel": _to_dict(panel), "image": ""}
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

    # Adjacent-zoom target panels have no image_name of their own —
    # their image content is synthesised from a SOURCE panel's zoom
    # rectangle. The Edit Panel dialog needs to show that synthesised
    # image so the user can place nested labels / scale bars /
    # annotations / further zoom insets on it. Reuse the same
    # _apply_adjacent_zoom_insets path that the full preview uses and
    # pick out the cell we want.
    if not panel.image_name or panel.image_name not in loaded_images:
        # Build a processed[][] grid by seeding EVERY image-bearing
        # cell (including videos). Without this, multi-hop chains
        # like A → B → C would fail when the dialog opens for C —
        # the previous "only-direct-sources" seeding would skip A
        # because A doesn't directly target C. The cascade pass in
        # `_apply_adjacent_zoom_insets` now propagates through chains
        # of any length, so seeding all roots is sufficient.
        rows, cols = cfg.rows, cfg.cols
        processed = [[None for _ in range(cols)] for _ in range(rows)]
        for sr in range(rows):
            for sc in range(cols):
                src = cfg.panels[sr][sc]
                # Accept both static images (loaded_images) AND videos
                # (loaded_videos). `_get_panel_image` handles both.
                if not (src.image_name and (src.image_name in loaded_images or src.image_name in loaded_videos)):
                    continue
                src_img = _get_panel_image(src)
                if src_img is None:
                    continue
                # Suppress own zoom-inset overlay; it would land
                # *on* the source panel's image, not on the
                # adjacent target, and is also wasted work for the
                # dialog preview of a different cell.
                saved_z = src.add_zoom_inset
                src.add_zoom_inset = False
                try:
                    processed[sr][sc] = process_panel(
                        src_img,
                        src, min_dims, loaded_images,
                        skip_labels=True, skip_symbols=True,
                    )
                except Exception:
                    pass
                finally:
                    src.add_zoom_inset = saved_z
        # Fill the target slot with a placeholder so the helper's
        # `processed[ar][ac] = region` assignment doesn't IndexError.
        if processed[r][c] is None:
            # Size doesn't matter — it gets replaced. Use a 1×1.
            processed[r][c] = Image.new("RGB", (1, 1), "white")
        _apply_adjacent_zoom_insets(cfg, processed, rows, cols)
        _apply_zoom_target_self_overlays(cfg, processed, rows, cols)
        synth = processed[r][c]
        if synth is None or synth.size == (1, 1):
            return {"image": ""}
        # Fall through to the shared matplotlib rendering pass below.
        # `processed` is now the synthesised PIL image; `panel` carries
        # the target cell's stamped scale_bar / labels. Routing zoom
        # targets through the same matplotlib pass as image-bearing
        # cells means scalebars / labels / symbols render on them too,
        # fixing the dialog-preview asymmetry where zoom targets
        # previously showed the raw zoomed image without a scalebar.
        processed = synth
    else:
        # Image-bearing panel: process the base image (no labels /
        # symbols / zoom — those are rendered via matplotlib / SVG).
        # Temporarily disable zoom inset to prevent double rendering
        # with the SVG overlay.
        saved_zoom = panel.add_zoom_inset
        panel.add_zoom_inset = False
        processed = process_panel(
            _get_panel_image(panel) or loaded_images[panel.image_name], panel, min_dims, loaded_images,
            skip_labels=True, skip_symbols=True)
        panel.add_zoom_inset = saved_zoom

    iw, ih = processed.size
    # Match the figure builder's reference_inches (3.0) so text /
    # bar sizes scale proportionally.
    reference_inches = 3.0
    # Match the 1×1 axes grid that we pass to matplotlib below (paired
    # with panel_override). _add_panel_scale_bars iterates r=0..rows,
    # c=0..cols and reads pre_norm_sizes[r][c] — using the override
    # `(r, c)` here would mis-index a 1×1 grid. So we store the
    # pre-resize size at [0][0].
    pre_upscale_sizes = [[None]]
    pre_upscale_sizes[0][0] = processed.size
    # Mirror the FIGURE's effective render size: when normalize_widths
    # is on, the figure stretches each panel to the grid-wide max
    # width / height. We do the same here so the matplotlib bar
    # position, length and thickness ALL match between dialog and
    # figure. Crispness is then added by PIL-upscaling the FINAL
    # OUTPUT bytes (below) — not by matplotlib upscaling, which
    # would change `iw` and shift bar_height's visual fraction.
    fig_target_w, fig_target_h = iw, ih
    if getattr(cfg, "normalize_widths", False):
        norm_mode = getattr(cfg, "normalize_mode", "width") or "width"
        max_w, max_h = 1, 1
        for sr in range(cfg.rows):
            for sc in range(cfg.cols):
                sp = cfg.panels[sr][sc]
                if not (sp.image_name and (sp.image_name in loaded_images or sp.image_name in loaded_videos)):
                    continue
                src_img = _get_panel_image(sp)
                if src_img is None:
                    continue
                try:
                    pp = process_panel(src_img, sp, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
                    if pp.size[0] > max_w: max_w = pp.size[0]
                    if pp.size[1] > max_h: max_h = pp.size[1]
                except Exception:
                    pass
        if norm_mode == "width" and max_w > iw:
            fig_target_w = max_w
            fig_target_h = int(ih * max_w / max(iw, 1))
        elif norm_mode == "height" and max_h > ih:
            fig_target_h = max_h
            fig_target_w = int(iw * max_h / max(ih, 1))
    if fig_target_w != iw or fig_target_h != ih:
        processed = processed.resize((fig_target_w, fig_target_h), Image.LANCZOS)
        iw, ih = fig_target_w, fig_target_h
    # Fix fig_w to the reference inches; preview_dpi controls output
    # crispness independently from the data range. matplotlib stretches
    # the imshow data to fill fig_w × fig_h, so the bar's position /
    # length / thickness — which live in DATA coords — stay tied to
    # `iw / ih` (the figure-equivalent size) regardless of dpi. We
    # bump dpi up to MIN_PREVIEW_DPI so text and bar rasterise crisply
    # in the dialog even when the underlying synth was small.
    fig_w = reference_inches
    fig_h = reference_inches * ih / max(iw, 1)
    MIN_PREVIEW_DPI = 400  # ~1200 px output at 3-inch fig — crisp text/bar
    preview_dpi = max(MIN_PREVIEW_DPI, iw / reference_inches)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=preview_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.array(processed), aspect='auto', extent=[0, iw, ih, 0])
    ax.set_xlim(0, iw)
    ax.set_ylim(ih, 0)
    ax.axis('off')

    # Use the SAME scale bar rendering as the full figure. Pass
    # pre_upscale_sizes so the bar/text scale with the upsample
    # we may have applied above (otherwise a 200-px bar in a 600
    # synth would become 200 px in a 1000-px upscale = visibly
    # too short).
    axes_grid = np.array([[ax]])
    _add_panel_scale_bars(fig, axes_grid, cfg, 1, 1,
                          [[processed]], panel_override=(r, c),
                          pre_norm_sizes=pre_upscale_sizes)

    # Use the SAME label rendering as the full figure
    _add_panel_labels(fig, axes_grid, cfg, 1, 1,
                      [[processed]], panel_override=(r, c))

    # Use the SAME symbol rendering as the full figure.
    # Adjacent-zoom target panels have no image_name (their content is
    # synthesised), so fall back to the synthesised image for symbol
    # size scaling — otherwise loaded_images[panel.image_name] would
    # KeyError on the empty string and the dialog preview 500's.
    if panel.image_name and panel.image_name in loaded_images:
        orig_img = _get_panel_image(panel) or loaded_images[panel.image_name]
    else:
        orig_img = processed
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
    """Compute all line/area measurements across all panels.

    Each result carries the measurement BOTH as a display string (`value`,
    which honours a user `measure_text` override) AND as separate
    structured fields — `numeric` (the computed number) and `unit` (its
    label, e.g. 'µm' / 'µm²'). The structured fields ensure the unit is
    explicitly recorded for the Analysis section rather than only being
    embedded inside the display string."""
    from models import (compute_line_measurement, compute_area_measurement,
                         compute_line_measurement_value,
                         compute_area_measurement_value, unit_label)
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
                    numeric = compute_line_measurement_value(line.points, iw, ih, mpp, unit)
                    text = line.measure_text or compute_line_measurement(
                        line.points, iw, ih, mpp, unit)
                    results.append({"panel": label, "name": line.name, "type": "line",
                                    "value": text, "numeric": round(numeric, 4),
                                    "unit": unit_label(unit, squared=False)})
            # Area measurements
            for area in (panel.areas or []):
                if area.show_measure and len(area.points) >= 2:
                    unit = getattr(area, 'measure_unit', 'um')
                    numeric = compute_area_measurement_value(
                        area.points, area.shape, iw, ih, mpp, unit)
                    text = area.measure_text or compute_area_measurement(
                        area.points, area.shape, iw, ih, mpp, unit)
                    results.append({"panel": label, "name": area.name, "type": "area",
                                    "value": text, "numeric": round(numeric, 4),
                                    "unit": unit_label(unit, squared=True)})
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
    #
    # Accept both static images (loaded_images) AND videos (loaded_videos);
    # `_get_panel_image` handles both. The previous gate of
    # `panel.image_name in loaded_images` silently dropped video panels and
    # they showed as blank placeholders in the preview.
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            panel = cfg.panels[r][c]
            src_img = _get_panel_image(panel) if panel.image_name else None
            if src_img is not None:
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

    # Adjacent Panel zoom insets — array-aware via _apply_adjacent_zoom_insets
    _apply_adjacent_zoom_insets(cfg, processed, rows, cols)
    _apply_zoom_target_self_overlays(cfg, processed, rows, cols)

    # Build full-res sizes for zoom inset line positioning.
    # For adjacent-zoom TARGET panels (no image_name), use the
    # dialog's 1000-px canvas convention so the source-rect overlay
    # the user drew on them lands in the same place as the rect
    # shown in the Edit Panel dialog (which falls back to ziActualW
    # = 1000 when origFullW is 0). Without this, figure_builder
    # would use the synth's actual size (e.g. 200×200) and draw the
    # rect off-image whenever zi.x exceeded the synth width.
    full_res_sizes = {}
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            orig_img = _get_panel_image(panel) if panel.image_name else None
            if orig_img is not None:
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes[(r, c)] = orig_img.size
            elif not panel.image_name and getattr(panel, "add_zoom_inset", False):
                # Zoom-target panel that itself has an outgoing inset
                # — record the 1000-px canvas size so figure_builder
                # scales zi.x/y correctly when drawing the rect.
                full_res_sizes[(r, c)] = (1000, 1000)

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
    # First pass: process all panels that have images (or videos —
    # `_get_panel_image` handles both, the previous gate of
    # `image_name in loaded_images` silently dropped video panels).
    processed = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            panel = cfg.panels[r][c]
            src_img = _get_panel_image(panel) if panel.image_name else None
            if src_img is not None:
                row_imgs.append(process_panel(
                    src_img, panel,
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
    _apply_adjacent_zoom_insets(cfg, processed, rows, cols)
    _apply_zoom_target_self_overlays(cfg, processed, rows, cols)

    full_res_sizes2 = {}
    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            orig_img = _get_panel_image(panel) if panel.image_name else None
            if orig_img is not None:
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes2[(r, c)] = orig_img.size
            elif not panel.image_name and getattr(panel, "add_zoom_inset", False):
                # See /api/preview: zoom-target source with an
                # outgoing inset → 1000-px canvas convention.
                full_res_sizes2[(r, c)] = (1000, 1000)
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

            # Adjacent Panel zoom insets — share the helper used by the
            # /api/preview and /api/figure/save paths so video export
            # honours every adjacent inset configured on every panel.
            # Pass `panel_images` (the per-output-frame frames we
            # extracted for range-video panels) as the cascade source
            # override, so insets animate frame-by-frame instead of
            # repeatedly sampling the static `frame` field of each
            # source panel.
            _apply_adjacent_zoom_insets(cfg, processed, rows, cols,
                                        image_override=panel_images)
            _apply_zoom_target_self_overlays(cfg, processed, rows, cols)

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
    # Optional analysis payload. Shape:
    #   {
    #     "manifest": <dict — arbitrary JSON describing tabs, etc.>,
    #     "plots":  { "<plot_id>": "<base64-png>", ... },
    #     "tables": { "<table_id>": "<csv text>", ... },
    #   }
    # The frontend ships PNG plots as base64 — we decode them here before
    # handing off to save_project().
    analysis: Optional[dict] = None


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

    # Convert the optional analysis payload from the wire format
    # ({plot_id: base64_png}) into the bytes the model layer expects.
    analysis_payload = None
    if body.analysis:
        plots_in = (body.analysis or {}).get("plots") or {}
        plots_bytes: Dict[str, bytes] = {}
        for pid, b64 in plots_in.items():
            if not b64:
                continue
            try:
                plots_bytes[pid] = base64.b64decode(b64)
            except Exception as _e:
                import sys
                print(f"[save_proj] failed to decode plot {pid}: {_e}", file=sys.stderr, flush=True)
        analysis_payload = {
            "manifest": (body.analysis or {}).get("manifest"),
            "plots": plots_bytes,
            "tables": (body.analysis or {}).get("tables") or {},
        }

    save_project(cfg, img_bytes, path, custom_fonts or None, analysis_payload)
    # Verify file was written
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        try: os.unlink(path)
        except OSError: pass
        raise HTTPException(500, f"Save failed: file at {path} is empty or missing (possible disk-full or permission issue).")
    return {"ok": True, "path": path, "size_bytes": os.path.getsize(path)}


class ProjectLoadRequest(BaseModel):
    path: str


def _hydrate_loaded_project(loaded_cfg, img_bytes_dict, font_bytes_dict):
    """Replace the global builder state (config + images + fonts) with a
    freshly-deserialized project. Returns the per-image thumbnails dict.
    Shared by the on-disk project loader (/api/project/load) and the
    in-session snapshot restore (/api/project/restore) so both paths hydrate
    identically."""
    global cfg, loaded_images, custom_fonts, loaded_videos, video_frames
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
    return {n: _thumb_b64(img) for n, img in loaded_images.items()}


@app.post("/api/project/load")
def load_proj(body: ProjectLoadRequest):
    path = os.path.expanduser(body.path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, img_bytes_dict, font_bytes_dict, analysis = load_project(path)
    thumbnails = _hydrate_loaded_project(loaded_cfg, img_bytes_dict, font_bytes_dict)

    # Re-encode any analysis plot bytes as base64 strings for transport to the
    # frontend; tables travel as plain CSV text.
    analysis_out = None
    if analysis is not None:
        plots_b64: Dict[str, str] = {}
        for pid, raw in (analysis.get("plots") or {}).items():
            if not raw:
                continue
            try:
                plots_b64[pid] = base64.b64encode(raw).decode()
            except Exception as _e:
                import sys
                print(f"[load_proj] failed to encode plot {pid}: {_e}", file=sys.stderr, flush=True)
        analysis_out = {
            "manifest": analysis.get("manifest"),
            "plots": plots_b64,
            "tables": analysis.get("tables") or {},
        }

    return {
        "config": _cfg_json(),
        "image_names": list(loaded_images.keys()),
        "thumbnails": thumbnails,
        "analysis": analysis_out,
    }


@app.get("/api/project/snapshot")
def snapshot_proj():
    """Serialize the CURRENT builder state to an in-memory .mpf blob
    (base64). Used for seamless document-tab switching: the single global
    backend state can hold only one document, so before swapping to another
    tab the frontend snapshots the outgoing tab here and restores it later
    (via /api/project/restore) without ever touching disk — so a tab's
    unsaved edits survive being switched away from."""
    img_bytes = {}
    for name, img in loaded_images.items():
        # Videos: persist the original file bytes (not the extracted frame).
        if name in loaded_videos and os.path.isfile(loaded_videos[name]):
            with open(loaded_videos[name], "rb") as vf:
                img_bytes[name] = vf.read()
        else:
            img_bytes[name] = pil_to_bytes(img)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mpf")
    tmp.close()
    try:
        save_project(cfg, img_bytes, tmp.name, custom_fonts or None, None)
        with open(tmp.name, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return {"blob": base64.b64encode(data).decode()}


class ProjectRestoreRequest(BaseModel):
    blob: str


@app.post("/api/project/restore")
def restore_proj(body: ProjectRestoreRequest):
    """Restore a builder state previously captured by /api/project/snapshot,
    replacing the global state with the blob's config + images + fonts.
    Mirrors /api/project/load but reads from an in-memory blob rather than a
    file on disk (analysis state is managed separately by the frontend)."""
    try:
        data = base64.b64decode(body.blob)
    except Exception:
        raise HTTPException(400, "Invalid snapshot blob.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mpf")
    tmp.write(data)
    tmp.close()
    try:
        loaded_cfg, img_bytes_dict, font_bytes_dict, _analysis = load_project(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    thumbnails = _hydrate_loaded_project(loaded_cfg, img_bytes_dict, font_bytes_dict)
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


@app.get("/api/fonts/file-b64/{name}")
def get_font_file_b64(name: str):
    """Same as /api/fonts/file/{name} but returns a JSON payload with
    a base64 string instead of raw bytes.  Used by the Tauri WebView
    where direct binary URLs (for @font-face src=url(...)) are blocked
    by the CSP — the frontend fetches this JSON via the api-proxy and
    constructs an in-memory Blob URL to register the font.
    """
    import sys
    safe = os.path.basename(name)
    if not safe or safe.startswith("."):
        raise HTTPException(404, "font not found")
    blob: bytes | None = custom_fonts.get(safe)
    if blob is None:
        cand = _CUSTOM_FONTS_DIR / safe
        if cand.is_file():
            try:
                blob = cand.read_bytes()
            except Exception:
                pass
    if blob is None:
        try:
            registry = find_fonts()
            sys_path = registry.get(safe)
            if not sys_path:
                try:
                    from matplotlib import font_manager as _fm
                    for ttf in (_fm.findSystemFonts(fontext="ttf")
                                + _fm.findSystemFonts(fontext="otf")):
                        if os.path.basename(ttf) == safe:
                            sys_path = ttf
                            break
                except Exception:
                    pass
            if sys_path and os.path.isfile(sys_path):
                with open(sys_path, "rb") as _fh:
                    blob = _fh.read()
        except Exception as _e:
            print(f"[fonts] system lookup for {safe} failed: {_e}", file=sys.stderr, flush=True)
    if blob is None:
        raise HTTPException(404, f"font '{safe}' not found on this host")
    return {
        "name": safe,
        "b64": base64.b64encode(blob).decode("ascii"),
        "mime": "font/ttf" if safe.lower().endswith(".ttf") else (
                "font/otf" if safe.lower().endswith(".otf") else "application/octet-stream"),
    }


@app.get("/api/fonts/file/{name}")
def get_font_file(name: str):
    """Serve raw font bytes by filename — used by the frontend's
    @font-face loader so per-character styling in the CSS overlay
    can use the same font the backend matplotlib / PIL renderer
    uses.  Without this the CSS would fall back to system fonts
    by name string (e.g. "ArialNarrowItalic.ttf" → no match → the
    overlay used sans-serif instead of the actual font).

    Looks up the font in three places, in order:
      1. The in-memory `custom_fonts` dict (uploaded fonts +
         project-bundled fonts loaded by load_project).
      2. The persistent custom fonts dir (~/.multipanelfigure/fonts).
      3. The system font index from find_fonts() / matplotlib's
         font_manager — both return absolute paths.
    """
    from fastapi import Response
    import sys
    # Sanitise — strip any path components so callers can't request
    # arbitrary files.  Only the basename is honoured.
    safe = os.path.basename(name)
    if not safe or safe.startswith("."):
        raise HTTPException(404, "font not found")
    # 1) In-memory cache.
    blob = custom_fonts.get(safe)
    if blob is None:
        # 2) Persistent custom fonts dir.
        cand = _CUSTOM_FONTS_DIR / safe
        if cand.is_file():
            try:
                blob = cand.read_bytes()
            except Exception as _e:
                print(f"[fonts] read {cand} failed: {_e}", file=sys.stderr, flush=True)
    if blob is None:
        # 3) System font registry — find_fonts + matplotlib's font_manager.
        try:
            registry = find_fonts()
            sys_path = registry.get(safe)
            if not sys_path:
                try:
                    from matplotlib import font_manager as _fm
                    for ttf in (_fm.findSystemFonts(fontext="ttf")
                                + _fm.findSystemFonts(fontext="otf")):
                        if os.path.basename(ttf) == safe:
                            sys_path = ttf
                            break
                except Exception:
                    pass
            if sys_path and os.path.isfile(sys_path):
                with open(sys_path, "rb") as _fh:
                    blob = _fh.read()
        except Exception as _e:
            print(f"[fonts] system lookup for {safe} failed: {_e}", file=sys.stderr, flush=True)
    if blob is None:
        raise HTTPException(404, f"font '{safe}' not found on this host")
    # Browsers don't strictly require a precise MIME for @font-face but
    # font/ttf is the canonical hint.
    media = "font/ttf" if safe.lower().endswith(".ttf") else (
            "font/otf" if safe.lower().endswith(".otf") else "application/octet-stream")
    return Response(content=blob, media_type=media, headers={
        "Cache-Control": "public, max-age=86400",
    })


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
    """When set, the font size override is applied ONLY to these element
    ids (from /api/collage/figure-elements). When null/empty, the legacy
    default set (column/row headers + axis labels) is targeted."""
    element_ids: Optional[List[str]] = None
    """Per-element STYLE overrides (font_name / font_style / color /
    styled_segments / font_size) keyed by element id, applied before the
    size sync. Used by the collage's double-click text customization."""
    element_overrides: Optional[Dict[str, dict]] = None


class CollageFigureElementsRequest(BaseModel):
    project_path: str


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
    loaded_cfg, img_bytes_dict, _font_bytes, _analysis = load_project(path)
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


def _iter_text_elements(cfg):
    """Return editable text-element descriptors for `cfg`. Each is a dict
    { id, type, text, font_size, _obj, _size_attr, _segs_attr }. The ids
    are stable and shared by the figure-elements list endpoint and the
    per-element font override path in render_collage_figure, so the
    frontend can let the user pick exactly which text elements get the
    synchronized font size."""
    out = []

    def add(eid, etype, obj, text_attr, size_attr, segs_attr, color_attr, style_attr):
        out.append({
            "id": eid,
            "type": etype,
            "text": (getattr(obj, text_attr, "") or ""),
            "font_size": getattr(obj, size_attr, None),
            "font_name": getattr(obj, "font_name", None),
            "color": getattr(obj, color_attr, None),
            "font_style": list(getattr(obj, style_attr, []) or []),
            "styled_segments": _segments_to_dicts(getattr(obj, segs_attr, None)),
            "_obj": obj,
            "_size_attr": size_attr,
            "_segs_attr": segs_attr,
            "_color_attr": color_attr,
            "_style_attr": style_attr,
        })

    for li, level in enumerate(getattr(cfg, "column_headers", []) or []):
        for gi, hdr in enumerate(level.headers):
            add(f"colhdr:{li}:{gi}", "Column header", hdr, "text", "font_size", "styled_segments", "default_color", "font_style")
    for li, level in enumerate(getattr(cfg, "row_headers", []) or []):
        for gi, hdr in enumerate(level.headers):
            add(f"rowhdr:{li}:{gi}", "Row header", hdr, "text", "font_size", "styled_segments", "default_color", "font_style")
    for i, lbl in enumerate(getattr(cfg, "column_labels", []) or []):
        add(f"collbl:{i}", "Column label", lbl, "text", "font_size", "styled_segments", "default_color", "font_style")
    for i, lbl in enumerate(getattr(cfg, "row_labels", []) or []):
        add(f"rowlbl:{i}", "Row label", lbl, "text", "font_size", "styled_segments", "default_color", "font_style")
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            panel = cfg.panels[r][c]
            for i, lab in enumerate(getattr(panel, "labels", []) or []):
                add(f"panellbl:{r}:{c}:{i}", f"Panel R{r+1}C{c+1} label", lab, "text", "font_size", "styled_segments", "color", "font_style")
            sb = getattr(panel, "scale_bar", None)
            if sb is not None:
                add(f"scalebar:{r}:{c}", f"Scale bar R{r+1}C{c+1}", sb, "label", "font_size", "styled_segments", "label_color", "label_font_style")
    return out


def _segments_to_dicts(segs):
    """Serialize a list of StyledSegment (or dict) objects to plain dicts
    for the figure-elements response."""
    if not segs:
        return []
    out = []
    for s in segs:
        if isinstance(s, dict):
            out.append({
                "text": s.get("text", ""), "color": s.get("color"),
                "font_name": s.get("font_name"), "font_size": s.get("font_size"),
                "font_style": list(s.get("font_style") or []) if s.get("font_style") else None,
            })
        else:
            out.append({
                "text": getattr(s, "text", ""), "color": getattr(s, "color", None),
                "font_name": getattr(s, "font_name", None), "font_size": getattr(s, "font_size", None),
                "font_style": list(getattr(s, "font_style", None) or []) if getattr(s, "font_style", None) else None,
            })
    return out


def _apply_element_overrides(cfg, element_overrides):
    """Apply per-element STYLE overrides (font_name / font_style / color /
    styled_segments / font_size) keyed by element id. Used by the collage's
    double-click text customization. Applied BEFORE the size sync so a
    later font_size sync still wins on size while keeping these styles."""
    if not element_overrides:
        return cfg
    by_id = {el["id"]: el for el in _iter_text_elements(cfg)}
    for eid, ov in element_overrides.items():
        el = by_id.get(eid)
        if not el or not ov:
            continue
        obj = el["_obj"]
        if ov.get("font_name"):
            if hasattr(obj, "font_name"):
                obj.font_name = ov["font_name"]
                if hasattr(obj, "font_path"):
                    obj.font_path = None  # let the renderer resolve by name
        if ov.get("font_style") is not None and hasattr(obj, el["_style_attr"]):
            setattr(obj, el["_style_attr"], list(ov["font_style"]))
        if ov.get("color") and hasattr(obj, el["_color_attr"]):
            setattr(obj, el["_color_attr"], ov["color"])
        if ov.get("font_size") and hasattr(obj, el["_size_attr"]):
            setattr(obj, el["_size_attr"], ov["font_size"])
        if "styled_segments" in ov:
            segs = ov.get("styled_segments") or []
            setattr(obj, el["_segs_attr"], [
                StyledSegment(
                    text=s.get("text", ""),
                    color=s.get("color", "#000000") or "#000000",
                    font_name=s.get("font_name"),
                    font_size=s.get("font_size"),
                    font_style=s.get("font_style"),
                ) for s in segs
            ])
    return cfg


# Element-id prefixes that the legacy "synchronize headers" default targets
# when no explicit element selection is supplied (column/row headers + the
# simple column/row axis labels). Panel labels + scale bars are opt-in via
# explicit element selection so default behaviour is unchanged.
_DEFAULT_FONT_SYNC_PREFIXES = ("colhdr", "rowhdr", "collbl", "rowlbl")


def _apply_font_pt(cfg, new_pt, element_ids=None):
    """Set the font size to new_pt on the figure's text elements, clearing
    any per-segment size overrides so the element-level size wins. When
    element_ids is given, only those elements are changed; otherwise the
    default header/axis-label set is changed (legacy behaviour)."""
    ids = set(element_ids) if element_ids else None
    for el in _iter_text_elements(cfg):
        if ids is not None:
            if el["id"] not in ids:
                continue
        else:
            if el["id"].split(":")[0] not in _DEFAULT_FONT_SYNC_PREFIXES:
                continue
        obj = el["_obj"]
        setattr(obj, el["_size_attr"], new_pt)
        segs = getattr(obj, el["_segs_attr"], None)
        if segs:
            for seg in segs:
                if hasattr(seg, "font_size"):
                    seg.font_size = None
    return cfg


@app.post("/api/collage/figure-elements")
def collage_figure_elements(body: "CollageFigureElementsRequest"):
    """List the editable text elements of a saved .mpf so the collage UI
    can offer per-element font synchronization. Returns id / type / text /
    current font_size for each element."""
    path = os.path.expanduser(body.project_path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, local_images = _load_collage_mpf(path)
    elements = [
        {"id": e["id"], "type": e["type"], "text": e["text"], "font_size": e["font_size"],
         "font_name": e["font_name"], "color": e["color"],
         "font_style": e["font_style"], "styled_segments": e["styled_segments"]}
        for e in _iter_text_elements(loaded_cfg)
    ]

    # Render once to collect on-figure geometry (figure fractions, y from
    # bottom) for column/row headers, so the collage can overlay clickable
    # hotspots on the figure itself. Best-effort — failures just omit geom.
    try:
        rows, cols = loaded_cfg.rows, loaded_cfg.cols
        if local_images:
            ws = [im.size[0] for im in local_images.values()]
            hs = [im.size[1] for im in local_images.values()]
            local_min_dims = (min(ws), min(hs))
        else:
            local_min_dims = (100, 100)
        proc = []
        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                panel = loaded_cfg.panels[r][c]
                if panel.image_name and panel.image_name in local_images:
                    row_imgs.append(process_panel(local_images[panel.image_name], panel,
                                                  local_min_dims, local_images,
                                                  skip_labels=True, skip_symbols=True))
                else:
                    row_imgs.append(None)
            proc.append(row_imgs)
        col_max_w = [max((proc[r][c].size[0] for r in range(rows) if proc[r][c] is not None),
                         default=local_min_dims[0]) for c in range(cols)]
        row_max_h = [max((proc[r][c].size[1] for c in range(cols) if proc[r][c] is not None),
                         default=local_min_dims[1]) for r in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if proc[r][c] is None:
                    proc[r][c] = Image.new("RGB", (col_max_w[c], row_max_h[r]), "white")
        geom_list: list = []
        assemble_figure(loaded_cfg, proc, dpi=72, header_collect=geom_list)
        geom_by_id = {}
        for g in geom_list:
            gid = g.get("id")
            if not gid:
                continue
            if g.get("orientation") == "column":
                geom_by_id[gid] = {
                    "orientation": "column",
                    "cx": g.get("cx_frac"), "cy": g.get("cy_frac"),
                    "s0": g.get("span_x0_frac"), "s1": g.get("span_x1_frac"),
                }
            else:
                geom_by_id[gid] = {
                    "orientation": "row",
                    "cx": g.get("cx_frac"), "cy": g.get("cy_frac"),
                    "s0": g.get("span_y0_frac"), "s1": g.get("span_y1_frac"),
                }
        for e in elements:
            if e["id"] in geom_by_id:
                e["geom"] = geom_by_id[e["id"]]
    except Exception as _e:
        import sys
        print(f"[figure-elements] geometry collect failed: {_e}", file=sys.stderr, flush=True)

    return {"elements": elements}


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
        _apply_element_overrides(cfg, body.element_overrides)
        _apply_font_pt(cfg, new_pt, body.element_ids)

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
    _apply_element_overrides(cfg2, body.element_overrides)
    if body.header_pt and body.header_pt > 0:
        new_pt = max(1, int(round(body.header_pt / max(0.001, body.scale))))
        _apply_font_pt(cfg2, new_pt, body.element_ids)

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


class CollageDecomposeRequest(BaseModel):
    project_path: str


@app.post("/api/collage/decompose")
def decompose_collage_figure(body: CollageDecomposeRequest):
    """Render a saved .mpf into a header-LESS body PNG plus the geometry
    of every column/row header, so the collage can lay headers out as
    live overlays. This is what makes the 'unify headers' button instant:
    the body is rendered once (and cached implicitly via _load_collage_mpf),
    and changing the unified font only re-typesets the HTML overlays —
    no matplotlib round-trip, no per-resize re-render.

    Returns:
      image      base64 PNG of the figure WITHOUT headers (header
                 margin space is still reserved, so the overlays have
                 room to sit)
      width/height  natural pixel size of the body
      headers[]  geometry in figure FRACTIONS (0..1, y from bottom):
                 each entry has orientation/position, anchor (cx_frac,
                 cy_frac), span, rotation, text + styled segments, font,
                 size_pt, colour, and bracket-line props. The frontend
                 multiplies fractions by the displayed body size and
                 renders the headers itself at the unified pt.
    """
    path = os.path.expanduser(body.project_path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, local_images = _load_collage_mpf(path)

    rows, cols = loaded_cfg.rows, loaded_cfg.cols
    if local_images:
        ws = [im.size[0] for im in local_images.values()]
        hs = [im.size[1] for im in local_images.values()]
        local_min_dims = (min(ws), min(hs))
    else:
        local_min_dims = (100, 100)

    processed: List[List[Optional[Image.Image]]] = []
    for r in range(rows):
        row_imgs: List[Optional[Image.Image]] = []
        for c in range(cols):
            panel = loaded_cfg.panels[r][c]
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
            panel = loaded_cfg.panels[r][c]
            if panel.image_name and panel.image_name in local_images:
                orig_img = local_images[panel.image_name]
                if panel.crop_image and panel.crop and len(panel.crop) == 4:
                    full_res_sizes2[(r, c)] = (panel.crop[2] - panel.crop[0], panel.crop[3] - panel.crop[1])
                else:
                    full_res_sizes2[(r, c)] = orig_img.size

    header_geom: list = []
    fig_bytes = assemble_figure(
        loaded_cfg, processed, dpi=150, full_res_sizes=full_res_sizes2,
        draw_headers=False, header_collect=header_geom,
    )
    img_out = Image.open(io.BytesIO(fig_bytes)).convert("RGBA")
    out_buf = io.BytesIO()
    img_out.save(out_buf, format="PNG")
    return {
        "image": base64.b64encode(out_buf.getvalue()).decode("ascii"),
        "width": img_out.width,
        "height": img_out.height,
        "headers": header_geom,
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
    data_csv: str = ""  # CSV string (optional — node graph sends an empty header row)
    rscript_path: Optional[str] = None        # legacy field name
    interpreter_path: Optional[str] = None    # unified field across engines
    # Tolerate the node graph's `measurements_csv` alias so older
    # frontends and the new analysis canvas can share this endpoint.
    measurements_csv: Optional[str] = None
    # When set, force a base font size (pt) on generated plots. Used by the
    # collage's "Synchronize headers" to re-render an R plot at a target
    # size. Best-effort: ggplot output is wrapped so the size is applied
    # AFTER the user's own theming (so it wins even over a full theme_*),
    # and base-graphics get par(cex) inside mpfig_plot.
    base_font_size: Optional[int] = None


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
    # `interpreter_path` is the unified name across engines; fall
    # back to the legacy `rscript_path` field if only the older name
    # is sent (keeps the analysis canvas + saved projects in sync).
    if body.interpreter_path and not body.rscript_path:
        body.rscript_path = body.interpreter_path
    # New analysis canvas sends `measurements_csv`; legacy field is
    # `data_csv`.  Coalesce.
    if (not body.data_csv) and body.measurements_csv:
        body.data_csv = body.measurements_csv
    rscript = _find_rscript(body.rscript_path)
    if not rscript:
        return {"success": False, "stdout": "", "stderr": "R is not installed. Please install R from https://cran.r-project.org/ or specify the Rscript path.", "plots": [], "tables": []}

    with tempfile.TemporaryDirectory(prefix="mpfig_r_") as tmpdir:
        data_path = os.path.join(tmpdir, "data.csv")
        with open(data_path, "w") as f:
            csv_text = body.data_csv.rstrip() + "\n"  # ensure trailing newline
            f.write(csv_text)

        plot_dir = os.path.join(tmpdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        table_dir = os.path.join(tmpdir, "tables")
        os.makedirs(table_dir, exist_ok=True)
        script = '# Auto-install missing packages used by the generated analysis\n'
        script += '#   ggplot2  — plotting\n'
        script += '#   ggprism  — GraphPad Prism-style theme + palettes, add_pvalue()\n'
        script += '#   rstatix  — t-test / Wilcoxon / ANOVA / Kruskal-Wallis stats\n'
        script += '#   dplyr    — the %>% pipe used by the stats block\n'
        script += 'for (.pkg in c("ggplot2", "ggprism", "rstatix", "dplyr")) {\n'
        script += '  if (!requireNamespace(.pkg, quietly=TRUE)) install.packages(.pkg, repos="https://cloud.r-project.org", quiet=TRUE)\n'
        script += '}\n\n'
        script += f'# Auto-generated data loading\ndata <- read.csv("{data_path.replace(chr(92), "/")}")\n\n'
        script += f'# Set plot output directory\n.plot_dir <- "{plot_dir.replace(chr(92), "/")}"\n'
        script += '.plot_count <- 0\n'
        _base_fs = body.base_font_size if (body.base_font_size and body.base_font_size > 0) else None
        script += 'mpfig_plot <- function(filename=NULL, width=800, height=600, res=150) {\n'
        script += '  .plot_count <<- .plot_count + 1\n'
        script += '  if (is.null(filename)) filename <- paste0("plot_", .plot_count, ".png")\n'
        script += '  png(file.path(.plot_dir, filename), width=width, height=height, res=res)\n'
        if _base_fs:
            # Base-graphics best-effort: scale character expansion relative to
            # R's default 12pt so base plots roughly track the target size.
            script += f'  try(par(cex={_base_fs}/12, cex.axis={_base_fs}/12, cex.lab={_base_fs}/12, cex.main={_base_fs}/12, cex.sub={_base_fs}/12), silent=TRUE)\n'
        script += '}\n\n'
        if _base_fs:
            # ggplot2: force a base font size by (1) setting the active theme's
            # base_size and (2) shadowing the common theme_*() constructors in
            # the global env so unqualified calls in user code (e.g.
            # `+ theme_classic()`, `+ theme_prism()`) inherit the size. This
            # works at the function level, so it is robust across ggplot2
            # versions (incl. the S7-based 4.x, where overriding print.ggplot
            # no longer intercepts rendering).
            script += '# ── Synchronized font size (collage) ──────────────────────\n'
            script += f'.mpfig_base_size <- {_base_fs}\n'
            script += 'options(mpfig_base_size = .mpfig_base_size)\n'
            script += 'if (requireNamespace("ggplot2", quietly=TRUE)) {\n'
            script += '  try({\n'
            script += '    ggplot2::theme_set(ggplot2::theme_grey(base_size = .mpfig_base_size))\n'
            script += '    .mpfig_wrap_theme <- function(fn) { force(fn); function(...) { a <- list(...); a$base_size <- .mpfig_base_size; do.call(fn, a) } }\n'
            script += '    for (.nm in c("theme_grey","theme_gray","theme_bw","theme_classic","theme_minimal","theme_light","theme_dark","theme_void","theme_linedraw")) {\n'
            script += '      if (exists(.nm, envir=asNamespace("ggplot2"))) assign(.nm, .mpfig_wrap_theme(get(.nm, envir=asNamespace("ggplot2"))), envir=globalenv())\n'
            script += '    }\n'
            script += '    if (requireNamespace("ggprism", quietly=TRUE) && exists("theme_prism", envir=asNamespace("ggprism")))\n'
            script += '      assign("theme_prism", .mpfig_wrap_theme(get("theme_prism", envir=asNamespace("ggprism"))), envir=globalenv())\n'
            script += '  }, silent = TRUE)\n'
            script += '}\n\n'
        script += f'# Set table output directory\n.table_dir <- "{table_dir.replace(chr(92), "/")}"\n'
        script += 'mpfig_data <- function(df, name = "table") {\n'
        script += '  write.csv(df, file = file.path(.table_dir, paste0(name, ".csv")), row.names = FALSE)\n'
        script += '  invisible(df)\n'
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
            return {"success": False, "stdout": "", "stderr": "R script timed out after 120 seconds (may be installing packages — try again).", "plots": [], "tables": []}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "plots": [], "tables": []}

        plots_b64 = []
        for png_path in sorted(glob_mod.glob(os.path.join(plot_dir, "*.png"))):
            with open(png_path, "rb") as pf:
                plots_b64.append(base64.b64encode(pf.read()).decode())

        tables_out = []
        for csv_path in sorted(glob_mod.glob(os.path.join(table_dir, "*.csv"))):
            try:
                with open(csv_path, "r", encoding="utf-8") as cf:
                    tables_out.append({
                        "name": os.path.splitext(os.path.basename(csv_path))[0],
                        "csv": cf.read(),
                    })
            except Exception as _e:
                import sys
                print(f"[run-r] failed to read table {csv_path}: {_e}", file=sys.stderr, flush=True)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "plots": plots_b64,
            "tables": tables_out,
        }


class RConsoleRequest(BaseModel):
    command: str
    rscript_path: Optional[str] = None


@app.post("/api/analysis/run-console")
def run_r_console(body: RConsoleRequest):
    """Run a RAW R command — no data-loading / plot boilerplate.

    Powers the Analysis dialog's small R console, whose main use is
    ad-hoc package management, e.g.  install.packages("ggsignif").
    Uses a long timeout because a CRAN install can be slow."""
    rscript = _find_rscript(body.rscript_path)
    if not rscript:
        return {"success": False, "stdout": "",
                "stderr": "R is not installed. Install R from https://cran.r-project.org/ "
                          "or set the Rscript path."}
    cmd = (body.command or "").strip()
    if not cmd:
        return {"success": False, "stdout": "", "stderr": "Empty command."}
    with tempfile.TemporaryDirectory(prefix="mpfig_rc_") as tmpdir:
        # Default the CRAN mirror so bare install.packages("x") works
        # without an interactive mirror prompt.
        script = ('options(repos = c(CRAN = "https://cloud.r-project.org"))\n'
                  + cmd + "\n")
        script_path = os.path.join(tmpdir, "console.R")
        with open(script_path, "w") as f:
            f.write(script)
        try:
            result = subprocess.run(
                [rscript, script_path],
                capture_output=True, text=True, timeout=600, cwd=tmpdir,
            )
            return {"success": result.returncode == 0,
                    "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": "Command timed out after 600 seconds."}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}


# ── Python pipelines on zoom-inset pixels ──────────────────────────────────
# These let the user run custom Python against the cropped pixels of
# any inset they marked "Include in Analysis" on a panel. Two endpoints:
#   GET  /api/analysis/inset-sources       → list available regions
#   POST /api/analysis/run-python          → execute user code, return
#                                            stdout / stderr / plots /
#                                            tables / modified images

def _collect_analysis_insets(include_thumbnails: bool = True):
    """Walk the current config and return one entry per zoom inset
    where `include_in_analysis` is true. The entry carries the inset's
    label (panel coords + inset index), location, dimensions, AND
    (when `include_thumbnails` is True) a base64 PNG thumbnail so the
    Analysis dialog can show the user exactly which pixels its
    pipelines will operate on.  Also ships a SECOND thumbnail of the
    PARENT panel (post-crop, post-rotation) plus the inset's bbox in
    that parent's coordinate space — the analysis library uses these
    to render a hover preview showing the inset's location on the
    host panel.

    Thumbnails go through `_extract_inset_image()` / `process_panel()`
    so they reflect the same cascade / crop math the runner uses; the
    result is PIL-thumbnailed to ≤256 px on the long edge to keep
    payload size sane.  Parent thumbnails are memoised per (row, col)
    so a panel with multiple insets doesn't re-render N times.
    """
    out: List[Dict[str, object]] = []
    if cfg is None:
        return out
    parent_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
    # Per-panel POST-CROP pixel size — keyed by (r, c).  Used to
    # build the figure-context thumbnail below; doubles as a way to
    # avoid re-running process_panel just to read the dimensions.
    panel_pix_size: Dict[Tuple[int, int], Tuple[int, int]] = {}
    # Memoise raw processed PIL images per panel for the figure-grid
    # composite (uniform-cell layout, see below).
    panel_processed: Dict[Tuple[int, int], Image.Image] = {}

    def _render_panel_thumb(panel_obj, r: int, c: int) -> Tuple[str, int, int]:
        """Render the parent panel post-crop/rotate, sans overlays.
        Returns (b64_thumb, full_w, full_h).  Memoised per (r, c).
        """
        cached = parent_cache.get((r, c))
        if cached is not None:
            return cached["thumb"], cached["w"], cached["h"]  # type: ignore[return-value]
        b64 = ""
        pw = ph = 0
        try:
            src_img = _get_panel_image(panel_obj) if panel_obj.image_name else None
            if src_img is not None:
                pc = _from_dict(PanelInfo, _to_dict(panel_obj))
                pc.add_zoom_inset = False
                pc.add_scale_bar = False
                pc.labels = []
                pc.symbols = []
                pc.lines = []
                pc.areas = []
                processed = process_panel(src_img, pc, min_dims, loaded_images,
                                          skip_labels=True, skip_symbols=True)
                pw, ph = int(processed.size[0]), int(processed.size[1])
                # Cache the FULL processed panel for the figure-grid
                # composite built after the per-source loop completes.
                panel_processed[(r, c)] = processed
                panel_pix_size[(r, c)] = (pw, ph)
                pthumb = processed.copy()
                pthumb.thumbnail((320, 320), Image.LANCZOS)
                buf = io.BytesIO()
                pthumb.convert("RGB").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as _e:
            import sys as _s
            print(f"[inset-sources] parent thumb failed for ({r},{c}): {_e}", file=_s.stderr, flush=True)
        parent_cache[(r, c)] = {"thumb": b64, "w": pw, "h": ph}
        return b64, pw, ph

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            panel = cfg.panels[r][c]

            # 1) PANEL-LEVEL flag — the cropped panel itself as a
            #    source.  Lets users analyse the whole panel image
            #    (or its crop region) without needing a zoom inset.
            if getattr(panel, "include_in_analysis", False) and panel.image_name:
                try:
                    pthumb_b64, pw, ph = _render_panel_thumb(panel, r, c)
                    if pw > 0 and ph > 0:
                        # Inset thumbnail for the library card = a
                        # smaller copy of the parent thumb (same
                        # pixels), so the library row looks right.
                        out.append({
                            "key": f"r{r}c{c}_panel",
                            "row": r, "col": c,
                            "inset_index": -1,
                            "inset_type": "Panel",
                            "x": 0, "y": 0, "width": pw, "height": ph,
                            "zoom_factor": 1.0,
                            "label": f"R{r+1}C{c+1} (panel)",
                            "natural_width": pw,
                            "natural_height": ph,
                            "thumbnail": pthumb_b64,
                            "parent_thumbnail": pthumb_b64,
                            "parent_natural_width": pw,
                            "parent_natural_height": ph,
                            "parent_bbox": [0, 0, pw, ph],
                        })
                except Exception as _e:
                    import sys as _s
                    print(f"[inset-sources] panel-as-source failed for ({r},{c}): {_e}", file=_s.stderr, flush=True)

            # 2) AREA-LEVEL flag — masked polygon regions exposed
            #    as image sources.  Each area's pixels are extracted
            #    via _extract_area_image (defined below).
            for ai, area in enumerate(getattr(panel, "areas", []) or []):
                if not getattr(area, "include_in_analysis", False):
                    continue
                try:
                    masked, bbox_px = _extract_area_image(r, c, ai)
                    if masked is None or bbox_px is None:
                        continue
                    pthumb_b64, pw, ph = _render_panel_thumb(panel, r, c)
                    mw, mh = int(masked.size[0]), int(masked.size[1])
                    thumb = masked.copy()
                    thumb.thumbnail((256, 256), Image.LANCZOS)
                    tb = io.BytesIO(); thumb.convert("RGB").save(tb, format="PNG")
                    out.append({
                        "key": f"r{r}c{c}_area{ai}",
                        "row": r, "col": c,
                        "inset_index": -1,
                        "area_index": ai,
                        "inset_type": "Area",
                        "x": int(bbox_px[0]), "y": int(bbox_px[1]),
                        "width": int(bbox_px[2]), "height": int(bbox_px[3]),
                        "zoom_factor": 1.0,
                        "label": f"R{r+1}C{c+1} area {ai+1}" + (f" ({area.name})" if area.name else ""),
                        "natural_width": mw,
                        "natural_height": mh,
                        "thumbnail": base64.b64encode(tb.getvalue()).decode("ascii"),
                        "parent_thumbnail": pthumb_b64,
                        "parent_natural_width": pw,
                        "parent_natural_height": ph,
                        "parent_bbox": [int(bbox_px[0]), int(bbox_px[1]), int(bbox_px[2]), int(bbox_px[3])],
                    })
                except Exception as _e:
                    import sys as _s
                    print(f"[inset-sources] area-as-source failed for ({r},{c}) area {ai}: {_e}", file=_s.stderr, flush=True)

            insets = _panel_zoom_insets(panel)
            for idx, zi in enumerate(insets):
                if zi is None:
                    continue
                if not getattr(zi, "include_in_analysis", False):
                    continue
                entry: Dict[str, object] = {
                    "key": f"r{r}c{c}_i{idx}",
                    "row": r,
                    "col": c,
                    "inset_index": idx,
                    "inset_type": zi.inset_type,
                    "x": zi.x, "y": zi.y, "width": zi.width, "height": zi.height,
                    "zoom_factor": zi.zoom_factor,
                    "label": f"R{r+1}C{c+1} · inset {idx+1} ({zi.inset_type})",
                    "natural_width": 0,
                    "natural_height": 0,
                    "thumbnail": "",
                    "parent_thumbnail": "",
                    "parent_natural_width": 0,
                    "parent_natural_height": 0,
                    "parent_bbox": [int(zi.x), int(zi.y), int(zi.width), int(zi.height)],
                }
                if include_thumbnails:
                    try:
                        img = _extract_inset_image(r, c, idx)
                        if img is not None:
                            entry["natural_width"] = int(img.size[0])
                            entry["natural_height"] = int(img.size[1])
                            thumb = img.copy()
                            thumb.thumbnail((256, 256), Image.LANCZOS)
                            buf = io.BytesIO()
                            thumb.convert("RGB").save(buf, format="PNG")
                            entry["thumbnail"] = base64.b64encode(buf.getvalue()).decode("ascii")
                    except Exception as _e:
                        import sys as _s
                        print(f"[inset-sources] thumb extract failed for ({r},{c}) inset {idx}: {_e}", file=_s.stderr, flush=True)
                    # Parent panel preview — memoised per (r, c).
                    try:
                        cached = parent_cache.get((r, c))
                        if cached is None:
                            pthumb_b64 = ""
                            pw = ph = 0
                            src_img = _get_panel_image(panel) if panel.image_name else None
                            if src_img is not None:
                                # Render the parent with no zoom-inset
                                # overlay so the bbox we ship lines up
                                # with what the user sees in the popover.
                                panel_copy = _from_dict(PanelInfo, _to_dict(panel))
                                panel_copy.add_zoom_inset = False
                                panel_copy.add_scale_bar = False
                                panel_copy.labels = []
                                panel_copy.symbols = []
                                panel_copy.lines = []
                                panel_copy.areas = []
                                processed = process_panel(src_img, panel_copy, min_dims, loaded_images,
                                                           skip_labels=True, skip_symbols=True)
                                pw, ph = int(processed.size[0]), int(processed.size[1])
                                pthumb = processed.copy()
                                pthumb.thumbnail((320, 320), Image.LANCZOS)
                                buf = io.BytesIO()
                                pthumb.convert("RGB").save(buf, format="PNG")
                                pthumb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                            cached = {"thumb": pthumb_b64, "w": pw, "h": ph}
                            parent_cache[(r, c)] = cached
                        entry["parent_thumbnail"] = cached["thumb"]
                        entry["parent_natural_width"] = cached["w"]
                        entry["parent_natural_height"] = cached["h"]
                    except Exception as _e:
                        import sys as _s
                        print(f"[inset-sources] parent thumb failed for ({r},{c}): {_e}", file=_s.stderr, flush=True)
                out.append(entry)

    # ── Figure-context thumbnail ─────────────────────────────────
    # Build a single low-res tiled thumbnail of the WHOLE figure
    # (every image-bearing panel, not just those flagged for
    # analysis) so the analysis library's hover preview shows where
    # the source sits relative to the entire figure.  Each cell is
    # `CELL` pixels; panels are letterboxed to a uniform cell rect
    # so the grid is coherent regardless of varying panel aspect
    # ratios.  We pre-render every panel that wasn't already in
    # `panel_processed` so even purely "context" panels appear.
    try:
        rows_n, cols_n = cfg.rows, cfg.cols
        CELL = 110
        # First pass — fill `panel_processed` for every image-bearing
        # panel so the composite shows the full figure, not just the
        # analysis-flagged subset.
        for rr in range(rows_n):
            for cc in range(cols_n):
                if (rr, cc) in panel_processed:
                    continue
                p = cfg.panels[rr][cc]
                if not p or not getattr(p, "image_name", ""):
                    continue
                try:
                    src_img = _get_panel_image(p)
                    if src_img is None:
                        continue
                    pc = _from_dict(PanelInfo, _to_dict(p))
                    pc.add_zoom_inset = False
                    pc.add_scale_bar = False
                    pc.labels = []
                    pc.symbols = []
                    pc.lines = []
                    pc.areas = []
                    proc = process_panel(src_img, pc, min_dims, loaded_images,
                                         skip_labels=True, skip_symbols=True)
                    panel_processed[(rr, cc)] = proc
                    panel_pix_size[(rr, cc)] = (int(proc.size[0]), int(proc.size[1]))
                except Exception as _e:
                    import sys as _s
                    print(f"[inset-sources] panel render for figure-context failed at ({rr},{cc}): {_e}",
                          file=_s.stderr, flush=True)
        figW = max(1, cols_n * CELL)
        figH = max(1, rows_n * CELL)
        composite = Image.new("RGB", (figW, figH), (245, 245, 245))
        cell_geom: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}  # (r,c) -> (cx, cy, cw, ch)
        for (r, c), pim in panel_processed.items():
            pw, ph = pim.size
            if pw <= 0 or ph <= 0:
                continue
            scale = min(CELL / pw, CELL / ph)
            cw, ch = max(1, int(pw * scale)), max(1, int(ph * scale))
            cell = pim.copy()
            cell.thumbnail((cw, ch), Image.LANCZOS)
            cx = c * CELL + (CELL - cw) // 2
            cy = r * CELL + (CELL - ch) // 2
            composite.paste(cell, (cx, cy))
            cell_geom[(r, c)] = (cx, cy, cw, ch)
        # Light cell borders + RxCy labels in the top-left of each
        # cell so users can identify which panel a region belongs to.
        try:
            from PIL import ImageDraw as _ImageDraw, ImageFont as _ImageFont
            d = _ImageDraw.Draw(composite)
            try:
                font = _ImageFont.load_default()
            except Exception:
                font = None
            for r in range(rows_n):
                for c in range(cols_n):
                    d.rectangle([c * CELL, r * CELL, (c + 1) * CELL - 1, (r + 1) * CELL - 1],
                                outline=(190, 190, 190), width=1)
                    if font is not None:
                        d.text((c * CELL + 3, r * CELL + 2), f"R{r+1}C{c+1}",
                               fill=(110, 110, 110), font=font)
        except Exception:
            pass
        buf = io.BytesIO()
        composite.save(buf, format="PNG", optimize=True)
        figure_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        for entry in out:
            try:
                rr = int(entry.get("row", 0)); cc = int(entry.get("col", 0))
                if (rr, cc) not in cell_geom or (rr, cc) not in panel_pix_size:
                    continue
                cx, cy, cw, ch = cell_geom[(rr, cc)]
                pw, ph = panel_pix_size[(rr, cc)]
                bbox = entry.get("parent_bbox") or [0, 0, pw, ph]
                bx, by, bw, bh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                sx = cw / max(1, pw); sy = ch / max(1, ph)
                fx = int(cx + bx * sx)
                fy = int(cy + by * sy)
                fw = max(2, int(bw * sx))
                fh = max(2, int(bh * sy))
                entry["figure_thumbnail"] = figure_b64
                entry["figure_natural_width"] = figW
                entry["figure_natural_height"] = figH
                entry["figure_bbox"] = [fx, fy, fw, fh]
                entry["figure_cell_bbox"] = [cx, cy, cw, ch]
            except Exception:
                continue
    except Exception as _e:
        import sys as _s
        print(f"[inset-sources] figure-context build failed: {_e}", file=_s.stderr, flush=True)

    return out


def _extract_area_image(row: int, col: int, area_index: int) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    """Return (masked_image, bbox_in_processed_panel_coords) for the
    area annotation at (row, col)[area_index].  The masked image has
    pixels OUTSIDE the polygon set to transparent / black so the
    analysis pipeline operates only on the area's interior.  Bbox is
    in the processed (post-crop / rotate) panel's pixel space.

    Returns (None, None) when the area / panel doesn't exist or
    can't be extracted.
    """
    if cfg is None or row < 0 or col < 0 or row >= cfg.rows or col >= cfg.cols:
        return None, None
    panel = cfg.panels[row][col]
    areas = list(getattr(panel, "areas", []) or [])
    if area_index < 0 or area_index >= len(areas):
        return None, None
    area = areas[area_index]
    src_img = _get_panel_image(panel) if panel.image_name else None
    if src_img is None:
        return None, None

    # Process the panel the same way the figure renderer does so
    # the area's % coords land on the right pixels.
    pc = _from_dict(PanelInfo, _to_dict(panel))
    pc.add_zoom_inset = False
    pc.add_scale_bar = False
    pc.labels = []
    pc.symbols = []
    pc.lines = []
    pc.areas = []
    processed = process_panel(src_img, pc, min_dims, loaded_images,
                              skip_labels=True, skip_symbols=True)
    W, H = processed.size

    # Resolve polygon points (in % of the processed panel) into pixels.
    pts: List[Tuple[float, float]] = []
    shape = (area.shape or "Rectangle")
    if shape in ("Rectangle", "Ellipse") and len(area.points) >= 2:
        # First point = centre (x%, y%); second point = (w%, h%).
        cx, cy = area.points[0]
        ww, hh = area.points[1]
        x0 = (cx - ww / 2) / 100.0 * W
        y0 = (cy - hh / 2) / 100.0 * H
        x1 = (cx + ww / 2) / 100.0 * W
        y1 = (cy + hh / 2) / 100.0 * H
        if shape == "Rectangle":
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            import math as _math
            steps = 64
            cxp = (x0 + x1) / 2; cyp = (y0 + y1) / 2
            rx = (x1 - x0) / 2; ry = (y1 - y0) / 2
            pts = [(cxp + rx * _math.cos(2 * _math.pi * i / steps),
                    cyp + ry * _math.sin(2 * _math.pi * i / steps)) for i in range(steps)]
    elif shape in ("Custom", "Magic") and len(area.points) >= 3:
        pts = [(p[0] / 100.0 * W, p[1] / 100.0 * H) for p in area.points]
    else:
        return None, None

    # Bbox-crop the masked region for a tight output image.
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bx = max(0, int(min(xs))); by = max(0, int(min(ys)))
    bw = min(W - bx, int(max(xs)) - bx + 1)
    bh = min(H - by, int(max(ys)) - by + 1)
    if bw <= 0 or bh <= 0:
        return None, None

    # Render a mask the same size as `processed`, then crop both to
    # the area's bbox so the output is just the masked ROI.
    from PIL import ImageDraw as _ImageDraw
    mask = Image.new("L", (W, H), 0)
    _ImageDraw.Draw(mask).polygon([(p[0], p[1]) for p in pts], fill=255)
    masked_full = Image.composite(processed, Image.new("RGB", (W, H), (0, 0, 0)), mask)
    masked = masked_full.crop((bx, by, bx + bw, by + bh))
    return masked, (bx, by, bw, bh)


def _extract_source_image(s: Dict[str, object]) -> Optional[Image.Image]:
    """Unified source-image dispatcher.  The analysis library now
    exposes three flavours of source (zoom insets, whole panels,
    area annotations); each materialises through a different path
    but the runners just want a PIL image.  We route by the key
    suffix ("_panel" / "_area<N>") with `inset_index` as the legacy
    fallback for normal zoom-inset sources."""
    key = str(s.get("key") or "")
    try:
        row = int(s.get("row") or 0)
        col = int(s.get("col") or 0)
    except Exception:
        return None
    if cfg is None or row < 0 or col < 0 or row >= cfg.rows or col >= cfg.cols:
        return None
    if key.endswith("_panel") or s.get("inset_type") == "Panel":
        panel = cfg.panels[row][col]
        src_img = _get_panel_image(panel) if panel.image_name else None
        if src_img is None:
            return None
        pc = _from_dict(PanelInfo, _to_dict(panel))
        pc.add_zoom_inset = False
        pc.add_scale_bar = False
        pc.labels = []; pc.symbols = []; pc.lines = []; pc.areas = []
        return process_panel(src_img, pc, min_dims, loaded_images,
                             skip_labels=True, skip_symbols=True)
    if "_area" in key or s.get("inset_type") == "Area":
        import re as _re
        m = _re.search(r"_area(\d+)$", key)
        ai = int(m.group(1)) if m else int(s.get("area_index") or 0)
        masked, _bbox = _extract_area_image(row, col, ai)
        return masked
    # Standard zoom inset.
    try:
        idx = int(s.get("inset_index") or 0)
    except Exception:
        idx = 0
    return _extract_inset_image(row, col, idx)


def _extract_inset_image(row: int, col: int, inset_index: int) -> Optional[Image.Image]:
    """Return the cropped + zoomed PIL image for the inset at
    (row, col)[inset_index]. Mirrors the cascade / draw_standard_zoom
    logic so the analysis pipeline operates on EXACTLY the same pixels
    that appear in the inset on the final figure.

    Returns None if the inset doesn't exist or can't be extracted
    (panel has no source image, indices out of range, etc.).
    """
    if cfg is None or row < 0 or col < 0 or row >= cfg.rows or col >= cfg.cols:
        return None
    panel = cfg.panels[row][col]
    insets = _panel_zoom_insets(panel)
    if inset_index < 0 or inset_index >= len(insets):
        return None
    zi = insets[inset_index]
    if zi is None:
        return None
    # Standard / Separate insets read from the panel's own image
    # (or an external image for Separate). Adjacent insets render
    # into another cell — we still want THIS panel's selection
    # crop for the pixels (zoom_factor × source rect), not the
    # final assembled cell, so the analysis is repeatable across
    # render-context changes.
    src_img = _get_panel_image(panel) if panel.image_name else None
    if src_img is None:
        # Try the synthesised path for image-less zoom-target
        # panels — extract by running the cascade and reading
        # the cell whose source is `panel`.
        rows, cols = cfg.rows, cfg.cols
        processed_grid: List[List[Optional[Image.Image]]] = [[None] * cols for _ in range(rows)]
        for sr in range(rows):
            for sc in range(cols):
                sp = cfg.panels[sr][sc]
                spi = _get_panel_image(sp) if sp.image_name else None
                if spi is None:
                    continue
                saved_z = sp.add_zoom_inset
                sp.add_zoom_inset = False
                try:
                    processed_grid[sr][sc] = process_panel(
                        spi, sp, min_dims, loaded_images,
                        skip_labels=True, skip_symbols=True,
                    )
                finally:
                    sp.add_zoom_inset = saved_z
        _apply_adjacent_zoom_insets(cfg, processed_grid, rows, cols)
        synth = processed_grid[row][col]
        if synth is None or synth.size == (1, 1):
            return None
        # Crop the inset region from the synth.
        sw, sh = synth.size
        x = max(0, min(int(zi.x), sw - 1))
        y = max(0, min(int(zi.y), sh - 1))
        w = max(1, min(int(zi.width), sw - x))
        h = max(1, min(int(zi.height), sh - y))
        region = synth.crop((x, y, x + w, y + h))
        zw = max(1, int(zi.width * zi.zoom_factor))
        zh = max(1, int(zi.height * zi.zoom_factor))
        return region.resize((zw, zh), Image.LANCZOS)
    # Image-bearing panel — process through process_panel first so
    # crop / rotation / levels are applied, then take the inset crop.
    panel_copy = _from_dict(PanelInfo, _to_dict(panel))
    panel_copy.add_zoom_inset = False
    panel_copy.add_scale_bar = False
    panel_copy.labels = []
    panel_copy.symbols = []
    panel_copy.lines = []
    panel_copy.areas = []
    img = process_panel(src_img, panel_copy, min_dims, loaded_images, skip_labels=True, skip_symbols=True)
    iw, ih = img.size
    x = max(0, min(int(zi.x), iw - 1))
    y = max(0, min(int(zi.y), ih - 1))
    w = max(1, min(int(zi.width), iw - x))
    h = max(1, min(int(zi.height), ih - y))
    rot = float(getattr(zi, "rotation", 0) or 0)
    if abs(rot) > 0.01:
        cx_r = x + w / 2.0
        cy_r = y + h / 2.0
        rotated = img.rotate(rot, center=(cx_r, cy_r), resample=Image.BICUBIC)
        region = rotated.crop((x, y, x + w, y + h))
    else:
        region = img.crop((x, y, x + w, y + h))
    zw = max(1, int(w * zi.zoom_factor))
    zh = max(1, int(h * zi.zoom_factor))
    return region.resize((zw, zh), Image.LANCZOS)


@app.get("/api/analysis/inset-sources")
def list_inset_analysis_sources():
    """List every zoom inset across the grid whose `include_in_analysis`
    flag is set. The Analysis tab uses this to populate its source
    selector — running a Python pipeline lets the user pick any of
    these regions as the input dataset."""
    return {"sources": _collect_analysis_insets()}


# Serialises the temporary global-state swap used to collect sources from a
# non-active .mpf (see below).
_source_swap_lock = _threading.Lock()


class InsetSourcesForRequest(BaseModel):
    project_path: str


@app.post("/api/analysis/inset-sources-for")
def list_inset_sources_for(body: InsetSourcesForRequest):
    """Collect analysis sources for a SPECIFIC .mpf (not the live figure), so
    the Analysis tab can show one sources drawer per loaded MPF — including
    figures that aren't the active builder doc. Reuses the active-figure
    collector by briefly swapping the global figure state to the target
    .mpf's (serialised by a lock, restored in `finally`). The Analysis tab
    is a separate workspace, so concurrent builder ops during the short swap
    window are not expected."""
    global cfg, loaded_images, loaded_videos, video_frames, loaded_zstacks, channel_groups, min_dims
    path = os.path.expanduser(body.project_path.strip())
    if not os.path.dirname(path):
        path = os.path.join(os.path.expanduser("~"), "Documents", path)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Project file not found: {path}")
    loaded_cfg, local_images = _load_collage_mpf(path)
    with _source_swap_lock:
        saved = (cfg, loaded_images, loaded_videos, video_frames, loaded_zstacks, channel_groups, min_dims)
        try:
            cfg = loaded_cfg
            loaded_images = dict(local_images)
            loaded_videos = {}
            video_frames = {}
            loaded_zstacks = {}
            channel_groups = {}
            _recalc_min_dims()
            sources = _collect_analysis_insets(include_thumbnails=True)
        finally:
            (cfg, loaded_images, loaded_videos, video_frames,
             loaded_zstacks, channel_groups, min_dims) = saved
    return {"sources": sources}


class PyAnalysisRequest(BaseModel):
    code: str
    # Sources the pipeline can read. Each entry: { key, row, col,
    # inset_index }. The runner injects them into the script as
    # `inputs[key]` → a dict { image: np.ndarray (HxWx3 uint8),
    # width: int, height: int, label: str }.
    sources: List[Dict[str, object]] = []
    # Extra inputs piped from upstream nodes in the analysis graph.
    # Each entry: { key, kind, label?, image_b64?, csv? }.
    #   kind="image" → decoded into the same inputs[key] shape as a
    #                  source (image / width / height / label).
    #   kind="table" → exposed as inputs[key]["table"] (list of dicts
    #                  parsed from the CSV via csv.DictReader).
    # Lets a downstream Python node operate on a prior node's
    # outputs without going through the inset registry.
    extra_inputs: List[Dict[str, object]] = []
    timeout_sec: int = 30
    # Optional custom Python interpreter path. When set, run the
    # script via this binary instead of `sys.executable`; lets users
    # pin a particular venv / conda environment / system Python.
    interpreter_path: Optional[str] = None


@app.post("/api/analysis/run-python")
def run_python_pipeline(body: PyAnalysisRequest):
    """Execute the user's Python code with the requested inset images
    available as `inputs`. Return stdout, stderr, any plots saved via
    `mpfig_plot(...)`, any CSV tables saved via `mpfig_data(...)`, and
    any modified images saved via `mpfig_image(...)`.

    The code runs in a subprocess with a wall-clock timeout (default
    30 s, capped at 600 s) so a runaway loop or accidental
    `while True` can't hang the sidecar. Inputs are passed via a
    pickled file in a temp dir — not as command-line args — so
    multi-MB images don't blow the argv length limit. Standard
    libraries available: PIL/Pillow, numpy, OpenCV (cv2), scipy,
    scikit-image, matplotlib — whatever the sidecar's own venv has
    (since we use the same `sys.executable`).
    """
    import sys as _sys
    import pickle as _pickle
    import textwrap as _textwrap
    import numpy as _np
    code = (body.code or "").strip()
    if not code:
        return {"success": False, "stdout": "", "stderr": "Empty code.",
                "plots": [], "tables": [], "images": []}
    sources_in = body.sources or []
    timeout_sec = max(1, min(int(body.timeout_sec or 30), 600))

    with tempfile.TemporaryDirectory(prefix="mpfig_py_") as tmpdir:
        # Materialise each requested inset to numpy bytes ahead of
        # time and stash in a pickle the worker reads.
        inputs_pickle = os.path.join(tmpdir, "inputs.pkl")
        inputs_dict: Dict[str, Dict[str, object]] = {}
        for s in sources_in:
            try:
                key = str(s.get("key", ""))
                r = int(s.get("row", -1)); c = int(s.get("col", -1))
                if not key or r < 0 or c < 0:
                    continue
                img = _extract_source_image(s)  # dispatches: inset / panel / area
                if img is None:
                    continue
                arr = _np.asarray(img.convert("RGB"))
                inputs_dict[key] = {
                    "image": arr,
                    "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]),
                    "label": str(s.get("label") or key),
                    "row": r, "col": c, "inset_index": int(s.get("inset_index", -1)),
                }
            except Exception as _e:
                import sys as __s
                print(f"[run-python] extract failed for {s}: {_e}", file=__s.stderr, flush=True)
        # Pipe upstream-node outputs in as additional inputs[key]
        # entries — same shape as inset-sourced inputs so the user
        # code doesn't need to special-case them.
        for x in (body.extra_inputs or []):
            try:
                key = str(x.get("key", ""))
                kind = str(x.get("kind", ""))
                label = str(x.get("label") or key)
                if not key:
                    continue
                if kind == "image" and x.get("image_b64"):
                    raw = base64.b64decode(str(x["image_b64"]))
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    arr = _np.asarray(img)
                    inputs_dict[key] = {
                        "image": arr,
                        "width": int(arr.shape[1]),
                        "height": int(arr.shape[0]),
                        "label": label,
                    }
                elif kind == "table" and x.get("csv"):
                    import csv as _csv
                    reader = _csv.DictReader(io.StringIO(str(x["csv"])))
                    rows = list(reader)
                    inputs_dict[key] = {"table": rows, "label": label}
            except Exception as _e:
                import sys as __s
                print(f"[run-python] extra_input {x.get('key')!r} decode failed: {_e}", file=__s.stderr, flush=True)
        with open(inputs_pickle, "wb") as f:
            _pickle.dump(inputs_dict, f)

        plot_dir = os.path.join(tmpdir, "plots")
        table_dir = os.path.join(tmpdir, "tables")
        image_dir = os.path.join(tmpdir, "images")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Harness script wraps the user's code with:
        #   • `inputs` dict pre-loaded from the pickle.
        #   • `mpfig_plot()` — save current matplotlib figure into
        #     the analysis-tab plot timeline.
        #   • `mpfig_data()` — save CSV from DataFrame / dict / list.
        #   • `mpfig_image()` — save a numpy/PIL image to the image
        #     timeline (user can drag it back into the main figure).
        #
        # NOTE: this is a plain template (NOT an f-string) so we can
        # contain literal `{` / `}` in the body without escaping
        # every brace. We substitute the four path placeholders and
        # the user code via simple string `.replace()` calls.
        harness_template = '''import sys, os, pickle, traceback
import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_INPUTS_PATH = r"__INPUTS_PATH__"
_PLOT_DIR = r"__PLOT_DIR__"
_TABLE_DIR = r"__TABLE_DIR__"
_IMAGE_DIR = r"__IMAGE_DIR__"

with open(_INPUTS_PATH, "rb") as _f:
    inputs = pickle.load(_f)

_plot_count = [0]
_table_count = [0]
_image_count = [0]

def mpfig_plot(filename=None, **savefig_kwargs):
    """Save the current matplotlib figure into the analysis-tab
    plot timeline. Call after building a figure with plt."""
    if plt is None:
        return
    _plot_count[0] += 1
    if not filename:
        filename = "plot_" + str(_plot_count[0]) + ".png"
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        filename += ".png"
    path = os.path.join(_PLOT_DIR, filename)
    fig = plt.gcf()
    fig.savefig(path, dpi=savefig_kwargs.pop("dpi", 150), bbox_inches="tight", **savefig_kwargs)
    plt.close(fig)

def mpfig_data(rows, name="table"):
    """Save a CSV table into the analysis tab's data section.
    `rows` may be a pandas DataFrame, a dict of column→list, a
    list of dicts, or a list of lists."""
    import csv as _csv
    _table_count[0] += 1
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:64] or "table"
    path = os.path.join(_TABLE_DIR, safe + ".csv")
    try:
        import pandas as _pd
        if isinstance(rows, _pd.DataFrame):
            rows.to_csv(path, index=False)
            return
    except Exception:
        pass
    if isinstance(rows, dict):
        keys = list(rows.keys())
        n = max((len(v) for v in rows.values()), default=0)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(keys)
            for i in range(n):
                w.writerow([rows[k][i] if i < len(rows[k]) else "" for k in keys])
    elif isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], dict):
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r.get(k, "") for k in keys])
    elif isinstance(rows, (list, tuple)):
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            for r in rows:
                w.writerow(r if isinstance(r, (list, tuple)) else [r])
    else:
        with open(path, "w") as f:
            f.write(str(rows))

def mpfig_image(arr, name="image"):
    """Save a NumPy array or PIL image into the analysis tab's
    image timeline."""
    _image_count[0] += 1
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:64] or "image"
    path = os.path.join(_IMAGE_DIR, safe + ".png")
    if isinstance(arr, Image.Image):
        arr.save(path)
    else:
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            mn, mx = float(np.min(a)), float(np.max(a))
            if mx > mn:
                a = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                a = np.zeros_like(a, dtype=np.uint8)
        Image.fromarray(a).save(path)

try:
__USER_CODE__
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''
        harness = (
            harness_template
            .replace("__INPUTS_PATH__", inputs_pickle.replace("\\", "/"))
            .replace("__PLOT_DIR__", plot_dir.replace("\\", "/"))
            .replace("__TABLE_DIR__", table_dir.replace("\\", "/"))
            .replace("__IMAGE_DIR__", image_dir.replace("\\", "/"))
            .replace("__USER_CODE__", _textwrap.indent(code, "    "))
        )
        script_path = os.path.join(tmpdir, "pipeline.py")
        with open(script_path, "w") as f:
            f.write(harness)

        # Resolve which Python binary to use.  User override wins
        # (Settings dialog stores `interpreter_path` per-engine);
        # otherwise we fall back to the sidecar's own sys.executable
        # so the pipeline shares the bundled deps.
        py_bin = _sys.executable
        if body.interpreter_path and os.path.isfile(body.interpreter_path):
            py_bin = body.interpreter_path
        elif body.interpreter_path:
            return {"success": False, "stdout": "",
                    "stderr": f"Configured Python interpreter not found at: {body.interpreter_path}",
                    "plots": [], "tables": [], "images": []}
        try:
            result = subprocess.run(
                [py_bin, script_path],
                capture_output=True, text=True, timeout=timeout_sec, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"Python pipeline timed out after {timeout_sec} seconds.",
                    "plots": [], "tables": [], "images": []}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "plots": [], "tables": [], "images": []}

        # Collect outputs
        plots_b64: List[str] = []
        for png_path in sorted(glob_mod.glob(os.path.join(plot_dir, "*.*"))):
            try:
                with open(png_path, "rb") as pf:
                    plots_b64.append(base64.b64encode(pf.read()).decode())
            except Exception:
                pass
        tables_out: List[Dict[str, str]] = []
        for csv_path in sorted(glob_mod.glob(os.path.join(table_dir, "*.csv"))):
            try:
                with open(csv_path, "r", encoding="utf-8") as cf:
                    tables_out.append({
                        "name": os.path.splitext(os.path.basename(csv_path))[0],
                        "csv": cf.read(),
                    })
            except Exception:
                pass
        images_out: List[Dict[str, str]] = []
        for img_path in sorted(glob_mod.glob(os.path.join(image_dir, "*.*"))):
            try:
                with open(img_path, "rb") as pf:
                    images_out.append({
                        "name": os.path.splitext(os.path.basename(img_path))[0],
                        "image": base64.b64encode(pf.read()).decode(),
                    })
            except Exception:
                pass

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "plots": plots_b64,
            "tables": tables_out,
            "images": images_out,
        }


def _find_matlab_executable(custom_path: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Locate a MATLAB-or-compatible interpreter on the host. Returns
    (path, kind) where kind ∈ {"matlab", "octave", ""}.

    Octave is preferred when both are installed because it starts in
    ~1 s while MATLAB takes 15-30 s and consumes a license.

    Detection order: custom_path > PATH > common install dirs across
    macOS / Linux / Windows.
    """
    import shutil as _shutil
    import glob as _g
    # 0. User override wins outright.
    if custom_path and os.path.isfile(custom_path):
        name = os.path.basename(custom_path).lower()
        kind = "octave" if "octave" in name else "matlab"
        return custom_path, kind
    # 1. Octave (free, fast startup)
    for c in ("octave-cli", "octave"):
        p = _shutil.which(c)
        if p:
            return p, "octave"
    for candidate in [
        "/opt/homebrew/bin/octave", "/usr/local/bin/octave", "/usr/bin/octave",
        "/Applications/Octave.app/Contents/Resources/usr/bin/octave",
    ]:
        if os.path.isfile(candidate):
            return candidate, "octave"
    # 1b. Windows Octave (Octave-Forge MinGW build).
    prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for pf in [prog_files, prog_files_x86, r"C:\Octave"]:
        for hit in _g.glob(os.path.join(pf, "Octave*", "*", "bin", "octave-cli.exe")):
            if os.path.isfile(hit):
                return hit, "octave"
        for hit in _g.glob(os.path.join(pf, "GNU Octave", "Octave-*", "mingw64", "bin", "octave-cli.exe")):
            if os.path.isfile(hit):
                return hit, "octave"
    # 2. MATLAB (commercial, slow startup)
    p = _shutil.which("matlab")
    if p:
        return p, "matlab"
    for candidate in [
        "/Applications/MATLAB_R2024b.app/bin/matlab",
        "/Applications/MATLAB_R2024a.app/bin/matlab",
        "/Applications/MATLAB_R2023b.app/bin/matlab",
        "/Applications/MATLAB_R2023a.app/bin/matlab",
    ]:
        if os.path.isfile(candidate):
            return candidate, "matlab"
    # 2b. Windows MATLAB.  C:\Program Files\MATLAB\R20*\bin\matlab.exe.
    for pf in [prog_files, prog_files_x86]:
        for hit in _g.glob(os.path.join(pf, "MATLAB", "R*", "bin", "matlab.exe")):
            if os.path.isfile(hit):
                return hit, "matlab"
    return None, ""


@app.get("/api/analysis/check-matlab")
def check_matlab_installed(path: Optional[str] = None):
    """Tell the frontend whether the Run MATLAB button should be
    enabled, and which interpreter it'd use.  Accepts an optional
    ?path= override so the Settings dialog can probe a user-pinned
    binary before the user commits to it."""
    found, kind = _find_matlab_executable(path)
    return {"installed": bool(found), "kind": kind, "path": found or ""}


def _find_imagej_executable(custom_path: Optional[str] = None):
    """Locate an ImageJ / Fiji executable on PATH or in common
    install locations. Returns (path, kind) where kind is one of
    {'fiji', 'imagej'} or '' when nothing is installed. ImageJ-*
    names (e.g. ImageJ-linux64, ImageJ-macosx, ImageJ-win64.exe)
    are version-suffixed so we glob a couple of well-known parents.

    Detection order: custom_path > PATH > common install dirs on
    macOS / Linux / Windows."""
    import shutil as _sh
    import glob as _g
    import os as _os
    # 0. User override.
    if custom_path and _os.path.isfile(custom_path):
        kind = "fiji" if "Fiji" in custom_path or "fiji" in custom_path.lower() else "imagej"
        return custom_path, kind
    # 1. shutil.which preferring Fiji.
    for cand in ("ImageJ-linux64", "ImageJ-macosx", "ImageJ-win64.exe",
                 "fiji", "fiji-macosx", "Fiji.app/ImageJ-macosx", "ImageJ"):
        path = _sh.which(cand)
        if path:
            kind = "fiji" if "Fiji" in path or "fiji" in cand.lower() else "imagej"
            return path, kind
    # 2. Common install locations on macOS / Linux.
    candidates = [
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
        "/Applications/Fiji.app/ImageJ-macosx",
        "/opt/Fiji.app/ImageJ-linux64",
        _os.path.expanduser("~/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
        _os.path.expanduser("~/Fiji.app/ImageJ-linux64"),
    ]
    # 2b. Windows install locations — Fiji ships as a portable folder
    #     so users tend to drop it in a few standard places.
    prog_files = _os.environ.get("ProgramFiles", r"C:\Program Files")
    prog_files_x86 = _os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for parent in [prog_files, prog_files_x86, r"C:\\", _os.path.expanduser("~"), _os.path.expanduser("~/Downloads")]:
        for sub in ("Fiji.app", "fiji-win64", "ImageJ"):
            candidates.append(_os.path.join(parent, sub, "ImageJ-win64.exe"))
            candidates.append(_os.path.join(parent, sub, "ImageJ.exe"))
    for c in candidates:
        if _os.path.isfile(c) and (_os.access(c, _os.X_OK) or c.lower().endswith(".exe")):
            kind = "fiji" if "Fiji" in c or "fiji" in c.lower() else "imagej"
            return c, kind
    # 3. Glob for ImageJ-* binaries on PATH so users with a versioned
    #    install (ImageJ-2.14.0) still get detected.
    for d in _os.environ.get("PATH", "").split(_os.pathsep):
        for hit in _g.glob(_os.path.join(d, "ImageJ-*")):
            if _os.access(hit, _os.X_OK) or hit.lower().endswith(".exe"):
                return hit, "fiji" if "Fiji" in hit else "imagej"
    return None, ""


@app.get("/api/analysis/check-imagej")
def check_imagej_installed(path: Optional[str] = None):
    """Tell the frontend whether the ImageJ button should be enabled.
    Accepts an optional ?path= override so Settings can probe a
    user-pinned binary before committing it."""
    found, kind = _find_imagej_executable(path)
    return {"installed": bool(found), "kind": kind, "path": found or ""}


# ── Cellpose 3 module ─────────────────────────────────────────
@app.get("/api/analysis/check-cellpose")
def check_cellpose_installed():
    """Tell the frontend whether Cellpose is importable in the sidecar's
    Python environment.  Returns the package version when available so
    the UI tooltip can show e.g. "cellpose 3.0.10"."""
    try:
        import importlib
        m = importlib.import_module("cellpose")
        ver = getattr(m, "version", None) or getattr(m, "__version__", None) or ""
        return {"installed": True, "kind": f"cellpose {ver}".strip(), "path": getattr(m, "__file__", "")}
    except Exception as e:
        return {"installed": False, "kind": "", "path": "", "error": str(e)[:200]}


@app.post("/api/analysis/install-cellpose")
def install_cellpose():
    """Run `pip install --upgrade cellpose` against the sidecar's own
    Python interpreter.  Returns a structured result so the frontend
    can surface stdout/stderr inline.  The download is large (~500 MB
    with torch); the timeout is generous.  Idempotent — re-running
    upgrades to the latest version.

    For LIVE progress, the streaming endpoint
    `/api/analysis/install-cellpose-stream` returns text/event-stream
    chunks as pip writes them.  This non-streaming version is kept
    for callers that just want a final pass/fail summary."""
    import sys as _sys
    import subprocess as _sp
    try:
        proc = _sp.run(
            [_sys.executable, "-m", "pip", "install", "--upgrade", "cellpose"],
            capture_output=True, text=True, timeout=900,
        )
        ok = proc.returncode == 0
        return {
            "success": ok,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "returncode": proc.returncode,
        }
    except _sp.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "pip install cellpose timed out (>15 min). "
                "Try running it manually in a terminal: "
                f"`{_sys.executable} -m pip install --upgrade cellpose`."}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": f"pip launch failed: {e}"}


from fastapi.responses import StreamingResponse  # noqa: E402

@app.post("/api/analysis/install-cellpose-stream")
def install_cellpose_stream():
    """Streaming variant: pipes pip's stdout/stderr line-by-line via
    Server-Sent Events so the frontend's Console panel can show
    real-time progress (download bars, "Collecting torch…", etc.).
    Ends with a final event containing the exit code.
    """
    import sys as _sys
    import subprocess as _sp

    def gen():
        # Use unbuffered output so pip's progress lines flush as they
        # arrive (PYTHONUNBUFFERED + --no-input + --progress-bar).
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        # `--break-system-packages` is a no-op inside a venv (the
        # case for the bundled Tauri sidecar) but lets us bypass
        # PEP-668's externally-managed-environment guard when the
        # sidecar happens to be the system Python (dev installs).
        # Without it pip 23+ aborts immediately with a confusing
        # error and the button looked like it "did nothing".
        cmd = [_sys.executable, "-u", "-m", "pip", "install",
               "--upgrade", "--no-input",
               "--progress-bar", "on",
               "--break-system-packages",
               "cellpose"]
        try:
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                             text=True, bufsize=1, env=env)
        except Exception as e:
            yield f"data: {{\"line\":\"pip launch failed: {e}\"}}\n\n"
            yield "data: {\"done\":true,\"returncode\":-1}\n\n"
            return
        yield f'data: {{"line":"using interpreter: {_sys.executable}"}}\n\n'
        yield 'data: {"line":"pip install --upgrade cellpose (may take 5-15 min, ~500 MB download)"}\n\n'
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip("\n")
            # Escape for JSON-ish event payload.
            esc = line.replace("\\", "\\\\").replace('"', '\\"')
            yield f'data: {{"line":"{esc}"}}\n\n'
        proc.wait()
        yield f'data: {{"done":true,"returncode":{proc.returncode}}}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream")


class CellposeAnalysisRequest(BaseModel):
    # JSON config + optional `//`-prefixed comment lines (we strip
    # them server-side).  See CELLPOSE_DEFAULT on the frontend for
    # the recognised keys.
    config: str
    sources: List[Dict[str, object]] = []
    extra_inputs: List[Dict[str, object]] = []
    timeout_sec: int = 300


@app.post("/api/analysis/run-cellpose")
def run_cellpose_pipeline(body: CellposeAnalysisRequest):
    """Run the configured Cellpose model against every upstream image.

    Emits, per input image:
      • <label>_mask.png — labelled mask (uint16 → palette-mapped PNG)
      • <label>_outlines.png — original image with outlines overlaid
    Plus a single per-pipeline CSV:
      • cellpose_counts.csv — source, n_cells, mean_area_px, median_area_px

    The cellpose import is heavy (loads torch).  We import lazily so
    the sidecar doesn't pay the cost when no Cellpose node is used.
    """
    import sys as _sys
    import os as _os
    import io as _io
    import re as _re
    import json as _json
    import base64 as _b64
    import numpy as _np

    # 1) Parse config — strip `// ...` and `# ...` comment lines so
    #    json.loads is happy.  We keep the value-token strings raw.
    raw = body.config or "{}"
    raw = "\n".join(line for line in raw.splitlines() if not _re.match(r"\s*(//|#)", line))
    try:
        cfg_dict = _json.loads(raw or "{}")
        if not isinstance(cfg_dict, dict):
            raise ValueError("config must be a JSON object")
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": f"Cellpose config is not valid JSON: {e}",
                "plots": [], "tables": [], "images": []}

    model_name        = str(cfg_dict.get("model", "cyto3"))
    diameter          = cfg_dict.get("diameter")  # None or number
    channels          = cfg_dict.get("channels", [0, 0])
    flow_threshold    = float(cfg_dict.get("flow_threshold", 0.4))
    cellprob_threshold = float(cfg_dict.get("cellprob_threshold", 0.0))
    min_size          = int(cfg_dict.get("min_size", 15))
    use_gpu           = bool(cfg_dict.get("use_gpu", False))

    # 2) Try to import cellpose.  Give a clear "install hint" on miss.
    try:
        from cellpose import models as cp_models, utils as cp_utils  # type: ignore
    except Exception as e:
        return {"success": False, "stdout": "",
                "stderr": ("Cellpose isn't installed in the sidecar's Python environment. "
                           f"Run `pip install cellpose` and restart the app.  Import error: {e}"),
                "plots": [], "tables": [], "images": []}

    # 3) Materialise every image input (sources + extras) into a
    #    common [(label, np.ndarray)] list.  Source insets are
    #    re-extracted at full resolution; extras arrive base64 PNG.
    images_in: List[Tuple[str, _np.ndarray]] = []
    for s in (body.sources or []):
        try:
            arr = _extract_source_image(s)  # dispatches: inset / panel / area
            if arr is None: continue
            images_in.append((str(s.get("label") or s.get("key") or f"src_{len(images_in)}"), _np.array(arr)))
        except Exception as _e:
            print(f"[run-cellpose] source extract failed: {_e}", file=_sys.stderr, flush=True)
    for x in (body.extra_inputs or []):
        if x.get("kind") != "image": continue
        b64 = x.get("image_b64") or ""
        if not b64: continue
        try:
            from PIL import Image as _Im
            arr = _np.array(_Im.open(_io.BytesIO(_b64.b64decode(b64))).convert("RGB"))
            images_in.append((str(x.get("label") or x.get("key") or f"in_{len(images_in)}"), arr))
        except Exception as _e:
            print(f"[run-cellpose] extra decode failed: {_e}", file=_sys.stderr, flush=True)
    if not images_in:
        return {"success": False, "stdout": "",
                "stderr": "No image inputs — wire source insets or upstream image outputs into this node.",
                "plots": [], "tables": [], "images": []}

    stdout_lines: List[str] = []
    images_out: List[Dict[str, str]] = []
    rows: List[Dict[str, object]] = []
    try:
        # 4) Detect Cellpose API surface and instantiate the model.
        # Behaviour confirmed against an installed cellpose 4.1.1
        # (the latest as of writing):
        #   v3:  cellpose.models.Cellpose(gpu, model_type=NAME)
        #        .eval(img, channels=, diameter=, …) → 4-tuple
        #   v4:  cellpose.models.CellposeModel(gpu, pretrained_model=NAME)
        #        .eval(img, channels=, diameter=, …) → 3-tuple
        #        Only 'cpsam' is a built-in name; anything else falls
        #        back to cpsam after a noisy download.  channels= is
        #        deprecated but still accepted (just emits a warning).
        import importlib as _il
        _cp_mod = _il.import_module("cellpose")
        # Version string for diagnostics.  v3 used ``__version__``;
        # v4 dropped that in favour of a plain ``version`` attribute
        # plus a multi-line ``version_str`` banner.  Take whichever
        # exists, and trim to the first line so we don't bury the log.
        cp_ver = (getattr(_cp_mod, "version", None)
                  or getattr(_cp_mod, "__version__", None)
                  or "").strip().splitlines()[0] if (
                    getattr(_cp_mod, "version", None)
                    or getattr(_cp_mod, "__version__", None)
                  ) else ""
        # v4 reliably signals itself by removing the ``Cellpose`` (size
        # model + seg model) wrapper class.  Anything that still has
        # cp_models.Cellpose is treated as the v3 API.
        is_v4 = not hasattr(cp_models, "Cellpose")

        # In v4 only "cpsam" is a built-in pretrained model.  If the
        # user's config still names a v3 cyto/nuclei model, accept it
        # but log a clear warning — cellpose will silently download &
        # use cpsam instead, which produces confusingly different
        # results from what the preset name implies.
        V4_BUILTIN = {"cpsam"}
        if is_v4 and model_name not in V4_BUILTIN:
            stdout_lines.append(
                f"WARNING: cellpose 4.x only ships the 'cpsam' model — "
                f"your requested {model_name!r} will fall back to cpsam. "
                f"Update your config's \"model\" field to \"cpsam\" to silence."
            )
            model_name = "cpsam"

        if is_v4:
            # v4 path — CellposeModel only.
            model = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
        else:
            # v3 path — combined wrapper (does size estimation + seg).
            model = cp_models.Cellpose(gpu=use_gpu, model_type=model_name)
        load_line = (
            f"loaded cellpose {cp_ver or '(version unknown)'} model={model_name!r} "
            f"gpu={use_gpu} api={'v4' if is_v4 else 'v3'}"
        )
        stdout_lines.append(load_line)
        # Real-time flush to the sidecar terminal so the user can
        # watch progress while the run is happening.  (The frontend
        # Console pane only sees stdout after the response returns
        # — a full SSE refactor is a separate change.)
        print(f"[run-cellpose] {load_line}", flush=True)

        import time as _time
        t_total_start = _time.monotonic()
        for img_idx, (label, img) in enumerate(images_in, start=1):
            seg_line = (
                f"[{img_idx}/{len(images_in)}] segmenting {label!r} "
                f"shape={img.shape} dtype={img.dtype}"
            )
            stdout_lines.append(seg_line)
            print(f"[run-cellpose] {seg_line}", flush=True)
            t_img_start = _time.monotonic()
            eval_kwargs: Dict[str, object] = dict(
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
            )
            # channels=: required in v3; deprecated-but-accepted in v4.
            # Pass it always for compatibility; both paths handle it.
            try:
                result = model.eval(img, channels=channels, **eval_kwargs)
            except TypeError:
                # Future v5 may drop channels entirely — fall back.
                result = model.eval(img, **eval_kwargs)
            # v3: (masks, flows, styles, diams).  v4: (masks, flows, styles).
            if isinstance(result, tuple) and len(result) >= 4:
                masks, flows, styles, diams = result[0], result[1], result[2], result[3]
            else:
                masks, flows, styles = result[0], result[1], result[2]
                diams = None
            # Mask → coloured palette PNG for output.
            from PIL import Image as _Im
            n_cells = int(masks.max())
            mask_rgb = _np.zeros((*masks.shape, 3), dtype=_np.uint8)
            if n_cells > 0:
                rng = _np.random.default_rng(42)
                palette = rng.integers(64, 255, size=(n_cells + 1, 3), dtype=_np.uint8)
                palette[0] = (0, 0, 0)  # background
                mask_rgb = palette[masks]
            buf = _io.BytesIO(); _Im.fromarray(mask_rgb).save(buf, format="PNG")
            images_out.append({"name": f"{label}_mask", "image": _b64.b64encode(buf.getvalue()).decode()})

            # Labels image — 8-bit grayscale where pixel value = cell
            # label ID.  This is the format ImageJ macros can iterate
            # over directly (threshold to label==N → measure).  If
            # n_cells > 255 we re-pack labels modulo 255 + 1 so the
            # image still fits in 8-bit (rare edge case — typical
            # microscopy fields have <200 cells).
            labels8 = masks.astype(_np.int64)
            if n_cells > 255:
                # Map to 1..255 cyclically; cells with the same packed
                # value won't be separable but the user gets a warning.
                labels8 = ((labels8 - 1) % 255 + 1) * (labels8 > 0)
                stdout_lines.append(
                    f"WARNING: {label!r} has {n_cells} cells (>255) — "
                    "packed label image cyclically; downstream ImageJ "
                    "analysis may merge cells with the same packed ID."
                )
            labels8 = labels8.astype(_np.uint8)
            buf_lbl = _io.BytesIO()
            _Im.fromarray(labels8, mode="L").save(buf_lbl, format="PNG")
            images_out.append({
                "name": f"{label}_labels",
                "image": _b64.b64encode(buf_lbl.getvalue()).decode(),
            })

            # Outlines overlay on the original image.
            outline_img = img.copy()
            if outline_img.ndim == 2:
                outline_img = _np.stack([outline_img]*3, axis=-1)
            try:
                outlines = cp_utils.outlines_list(masks)
                for o in outlines:
                    for (xx, yy) in o.astype(int):
                        if 0 <= yy < outline_img.shape[0] and 0 <= xx < outline_img.shape[1]:
                            outline_img[yy, xx] = (255, 80, 80)
            except Exception:
                pass
            buf2 = _io.BytesIO(); _Im.fromarray(outline_img).save(buf2, format="PNG")
            images_out.append({"name": f"{label}_outlines", "image": _b64.b64encode(buf2.getvalue()).decode()})

            # Per-cell stats for the counts table.
            sizes = _np.bincount(masks.ravel())[1:] if n_cells > 0 else _np.array([0])
            rows.append({
                "source": label,
                "n_cells": n_cells,
                "mean_area_px": float(sizes.mean()) if n_cells > 0 else 0.0,
                "median_area_px": float(_np.median(sizes)) if n_cells > 0 else 0.0,
                "model": model_name,
            })
            # Per-image timing so the post-run log makes the cost
            # visible (cellpose 4 / cpsam is heavy on first inference).
            elapsed = _time.monotonic() - t_img_start
            done_line = (
                f"[{img_idx}/{len(images_in)}] {label!r}: {n_cells} cell(s) "
                f"in {elapsed:.1f}s"
            )
            stdout_lines.append(done_line)
            print(f"[run-cellpose] {done_line}", flush=True)
        total_elapsed = _time.monotonic() - t_total_start
        total_line = f"total: {len(images_in)} image(s) in {total_elapsed:.1f}s"
        stdout_lines.append(total_line)
        print(f"[run-cellpose] {total_line}", flush=True)
    except Exception as e:
        return {"success": False, "stdout": "\n".join(stdout_lines),
                "stderr": f"Cellpose run failed: {e}",
                "plots": [], "tables": [], "images": images_out}

    # 5) Pack the per-image counts table.
    csv_lines = ["source,n_cells,mean_area_px,median_area_px,model"]
    for r in rows:
        csv_lines.append(",".join(str(r[k]) for k in ("source", "n_cells", "mean_area_px", "median_area_px", "model")))
    tables = [{"name": "cellpose_counts", "csv": "\n".join(csv_lines) + "\n"}]
    return {"success": True, "kind": "cellpose",
            "stdout": "\n".join(stdout_lines), "stderr": "",
            "plots": [], "tables": tables, "images": images_out}


class ImageJAnalysisRequest(BaseModel):
    code: str
    sources: List[Dict[str, object]] = []
    extra_inputs: List[Dict[str, object]] = []
    timeout_sec: int = 120
    # Optional path to the Fiji / ImageJ launcher binary.  When set,
    # `_find_imagej_executable` returns it directly; otherwise the
    # sidecar falls back to its auto-detection across the standard
    # macOS / Linux / Windows install locations.
    interpreter_path: Optional[str] = None


@app.post("/api/analysis/run-imagej")
def run_imagej_pipeline(body: ImageJAnalysisRequest):
    """Execute a Fiji / ImageJ macro headless against the requested
    inset images. Each input is materialised to a tempfile and the
    macro receives an `inputs` array with `path` + `label` per entry,
    plus helper functions `mpfig_data(name, ...)` and `mpfig_image(name)`.

    This is a scaffold: it actually shells out to Fiji when it's
    installed, and returns a clear "not installed" error otherwise
    so the frontend can surface an install hint inline rather than
    failing opaquely.
    """
    import sys as _sys
    import os as _os
    import io as _io
    import json as _json
    import base64 as _b64
    import subprocess as _sp
    import tempfile as _tf
    path, kind = _find_imagej_executable(body.interpreter_path)
    if not path:
        return {
            "success": False,
            "kind": "",
            "stdout": "",
            "stderr": (
                "ImageJ / Fiji not detected on this host. "
                + (f"Configured path: {body.interpreter_path!r} not found. " if body.interpreter_path else "")
                + "Install Fiji from https://imagej.net/software/fiji/ and ensure the `ImageJ-*` "
                "binary is on PATH (or place Fiji.app in /Applications, ~/Applications, "
                "or under %ProgramFiles% on Windows)."
            ),
            "plots": [], "tables": [], "images": [],
        }

    out_dir = _tf.mkdtemp(prefix="mpfig_imagej_")
    try:
        # 1. Materialise every image input (sources + upstream extras)
        #    to PNG on disk so the macro can `open(path)` them.
        materialised: List[Dict[str, str]] = []
        # 1a. Source insets — flagged regions from the host figure.
        for s in (body.sources or []):
            key = str(s.get("key") or "")
            label = str(s.get("label") or key)
            try:
                arr = _extract_source_image(s)  # dispatches: inset / panel / area
                if arr is None:
                    continue
                fpath = _os.path.join(out_dir, f"{key}.png")
                arr.save(fpath)
                materialised.append({"path": fpath, "label": label, "key": key})
            except Exception as _e:  # pragma: no cover
                print(f"[run-imagej] extract failed for {s}: {_e}", file=_sys.stderr, flush=True)
        # 1b. extra_inputs — upstream node outputs piped in by the
        #     graph runner (e.g. Cellpose's *_mask / *_labels images).
        #     Each entry is {kind, key, label, image_b64} or
        #     {kind="table", csv}.  We only materialise images here;
        #     table inputs would need a different path.
        for x in (body.extra_inputs or []):
            if x.get("kind") != "image":
                continue
            b64 = x.get("image_b64") or ""
            if not b64:
                continue
            key = str(x.get("key") or f"extra_{len(materialised)}")
            label = str(x.get("label") or key)
            try:
                from PIL import Image as _Im
                img = _Im.open(_io.BytesIO(_b64.b64decode(b64)))
                # Sanitise key for use as a filename (no slashes / colons).
                safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_")
                fpath = _os.path.join(out_dir, f"{safe_key}.png")
                img.save(fpath)
                materialised.append({"path": fpath, "label": label, "key": key})
            except Exception as _e:  # pragma: no cover
                print(f"[run-imagej] extra decode failed for {key}: {_e}", file=_sys.stderr, flush=True)

        # 2. Build the macro prelude.  ImageJ macro language has no
        #    objects, no JSON parser, and only flat arrays — so we
        #    expose the input list as THREE PARALLEL ARRAYS that the
        #    user iterates by index:
        #        input_paths[]   absolute filesystem paths
        #        input_labels[]  human labels (source name etc.)
        #        input_keys[]    stable keys (e.g. "out_image_0")
        #    The `inputs[i].path` / `inputs[i].label` aliases used by
        #    older presets are emulated via 2-D nameless arrays — no
        #    longer recommended (no member access in IJ macro).  New
        #    code should use input_paths / input_labels / input_keys.
        #
        #    mpfig_data(name, headers, ...arrays) writes a CSV with
        #    arbitrary column count.  Macro signature is limited (no
        #    real varargs), so we shim with fixed-arity variants:
        #        mpfig_data1(name, headers, a)
        #        mpfig_data2(name, headers, a, b)
        #        ... up to mpfig_data7.
        #    `headers` is a comma-joined string (e.g. "x,y,z").

        def _macro_str_array(name: str, values: List[str]) -> str:
            """Render a newArray(...) initializer for IJ macro."""
            if not values:
                return f'{name} = newArray(0);\n'
            escaped = ['"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"' for v in values]
            return f'{name} = newArray({", ".join(escaped)});\n'

        macro_path = _os.path.join(out_dir, "_main.ijm")
        results_csv = _os.path.join(out_dir, "_results.csv")

        # mpfig_dataN helpers — write CSV with N data columns.  Macro
        # functions can't loop over a variadic arg list, so we generate
        # variants for column counts 1..12 (12 covers the full cell-
        # shape metrics emit: source/group/cell_id + 9 metrics).
        # Headers are split by comma in the prelude.
        _data_fns: List[str] = []
        for ncols in range(1, 13):
            args = ", ".join(["c%d" % i for i in range(ncols)])
            row_concat = ' + "," + '.join(["c%d[k]" % i for i in range(ncols)])
            _data_fns.append(
                f'function mpfig_data{ncols}(name, headers, {args}) {{ '
                f'  out = OUT_DIR + "/" + name + ".csv"; '
                f'  f = File.open(out); '
                f'  print(f, headers); '
                f'  for (k = 0; k < lengthOf(c0); k++) print(f, {row_concat}); '
                f'  File.close(f); }}\n'
            )

        prelude_parts: List[str] = [
            f'OUT_DIR = "{out_dir}";\n',
            _macro_str_array("input_paths",  [m["path"]  for m in materialised]),
            _macro_str_array("input_labels", [m["label"] for m in materialised]),
            _macro_str_array("input_keys",   [m["key"]   for m in materialised]),
            # Back-compat: many existing presets reference inputs[i].path
            # and inputs[i].label.  IJ macro can't do member access so
            # we provide a sentinel that warns when used.
            '// Use input_paths[i] / input_labels[i] / input_keys[i].\n',
            '// (Legacy `inputs[i].path` syntax is no longer supported.)\n',
        ]
        prelude_parts.extend(_data_fns)
        prelude_parts.append(
            # 3-arg backward-compat shim (was: mpfig_data(name, labels, a, b)).
            'function mpfig_data(name, labels, a, b) { '
            '  mpfig_data3(name, "label,a,b", labels, a, b); }\n'
        )
        prelude_parts.append(
            'function mpfig_image(name) { '
            '  saveAs("PNG", OUT_DIR + "/" + name + ".png"); }\n'
        )
        prelude = "".join(prelude_parts)

        with open(macro_path, "w") as _fh:
            _fh.write(prelude)
            _fh.write(body.code or "")

        # 3. Run Fiji headless.
        cmd = [path, "--headless", "--console", "-macro", macro_path]
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=body.timeout_sec)
            success = proc.returncode == 0
            stdout = proc.stdout
            stderr = proc.stderr
        except _sp.TimeoutExpired as _te:
            return {"success": False, "kind": kind, "stdout": _te.stdout or "",
                    "stderr": f"ImageJ macro timed out after {body.timeout_sec}s",
                    "plots": [], "tables": [], "images": []}

        # 4. Collect outputs.
        plots: List[str] = []
        tables: List[Dict[str, str]] = []
        images: List[Dict[str, str]] = []
        try:
            for fn in _os.listdir(out_dir):
                fp = _os.path.join(out_dir, fn)
                if fn.startswith("_"):
                    continue
                if fn.endswith(".csv"):
                    try:
                        with open(fp) as _r: csv = _r.read()
                        tables.append({"name": fn[:-4], "csv": csv})
                    except Exception:
                        pass
                elif fn.endswith(".png"):
                    try:
                        with open(fp, "rb") as _r: b = _b64.b64encode(_r.read()).decode("ascii")
                        # ImageJ macros produce processed images, not
                        # publication plots, so we surface them under
                        # the `images` bucket; plots stay R-only.
                        images.append({"name": fn[:-4], "image": b})
                    except Exception:
                        pass
        except Exception:
            pass
        _ = results_csv  # reserved for future Results-table import
        return {"success": success, "kind": kind, "stdout": stdout, "stderr": stderr,
                "plots": plots, "tables": tables, "images": images}
    finally:
        try:
            import shutil as _shr
            _shr.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass


class MatlabAnalysisRequest(BaseModel):
    code: str
    sources: List[Dict[str, object]] = []
    extra_inputs: List[Dict[str, object]] = []
    timeout_sec: int = 60
    # Optional path to a MATLAB / Octave binary; honoured before the
    # sidecar's auto-detection across PATH and common install dirs.
    interpreter_path: Optional[str] = None


@app.post("/api/analysis/run-matlab")
def run_matlab_pipeline(body: MatlabAnalysisRequest):
    """Execute the user's MATLAB / Octave code with the requested
    inset images available as `inputs.<key>.image` (uint8 H×W×3
    matrix). Returns stdout, stderr, plots / tables / images saved
    via the matching helper functions:
        mpfig_plot(figHandle, name)
        mpfig_data(table, name)
        mpfig_image(arr, name)

    The harness writes `inputs.mat` (via scipy.io.savemat from
    Python) so the script can `load("inputs.mat")` and immediately
    have `inputs.<key>.image` in scope. Output paths go through
    fixed temp directories so we can collect after the run, the
    same way the Python pipeline works.
    """
    import sys as _sys
    import numpy as _np
    try:
        from scipy.io import savemat as _savemat
    except ImportError:
        return {"success": False, "stdout": "",
                "stderr": "scipy is required for the MATLAB pipeline (savemat). "
                          "Install scipy in the sidecar's Python environment.",
                "plots": [], "tables": [], "images": []}

    matlab_path, kind = _find_matlab_executable(body.interpreter_path)
    if not matlab_path:
        return {"success": False, "stdout": "",
                "stderr": (
                    "Neither Octave nor MATLAB found. "
                    + (f"Configured path: {body.interpreter_path!r} not found. " if body.interpreter_path else "")
                    + "Install Octave from https://octave.org/ (free) or MATLAB from MathWorks, "
                    + "or set a custom path in Analysis → ⚙ Engines."
                ),
                "plots": [], "tables": [], "images": []}

    code = (body.code or "").strip()
    if not code:
        return {"success": False, "stdout": "", "stderr": "Empty code.",
                "plots": [], "tables": [], "images": []}
    sources_in = body.sources or []
    timeout_sec = max(1, min(int(body.timeout_sec or 60), 600))

    with tempfile.TemporaryDirectory(prefix="mpfig_matlab_") as tmpdir:
        # Extract inset images and bundle into a .mat the script
        # can `load`. MATLAB struct field names can't start with a
        # digit, so we sanitise keys (the original key is kept as
        # the field `label`).
        inputs_mat = os.path.join(tmpdir, "inputs.mat")
        inputs_struct: Dict[str, Dict[str, object]] = {}
        key_map: Dict[str, str] = {}  # safe → original
        for s in sources_in:
            try:
                orig = str(s.get("key", ""))
                r = int(s.get("row", -1)); c = int(s.get("col", -1))
                if not orig or r < 0 or c < 0:
                    continue
                img = _extract_source_image(s)  # dispatches: inset / panel / area
                if img is None:
                    continue
                arr = _np.asarray(img.convert("RGB"))
                safe = "k_" + orig.replace("-", "_").replace(".", "_")
                key_map[safe] = orig
                inputs_struct[safe] = {
                    "image": arr,
                    "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]),
                    "label": str(s.get("label") or orig),
                    "row": r, "col": c, "inset_index": int(s.get("inset_index", -1)),
                }
            except Exception as _e:
                import sys as __s
                print(f"[run-matlab] extract failed for {s}: {_e}", file=__s.stderr, flush=True)
        # Pipe upstream-node outputs in as additional inputs.<safe>
        # entries. Field names must start with a letter for MATLAB,
        # so we reuse the matlabSafeKey scheme.
        for x in (body.extra_inputs or []):
            try:
                key = str(x.get("key", ""))
                kind = str(x.get("kind", ""))
                label = str(x.get("label") or key)
                if not key:
                    continue
                safe = "k_" + key.replace("-", "_").replace(".", "_").replace("/", "_")
                if kind == "image" and x.get("image_b64"):
                    raw = base64.b64decode(str(x["image_b64"]))
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    arr = _np.asarray(img)
                    inputs_struct[safe] = {
                        "image": arr,
                        "width": int(arr.shape[1]),
                        "height": int(arr.shape[0]),
                        "label": label,
                    }
                elif kind == "table" and x.get("csv"):
                    # Tables get exposed as inputs.<safe>.table — a
                    # struct of column vectors. MATLAB consumer:
                    # `inputs.<safe>.table.<column>`
                    import csv as _csv
                    reader = _csv.DictReader(io.StringIO(str(x["csv"])))
                    rows = list(reader)
                    if rows:
                        cols = {col: [r.get(col, "") for r in rows] for col in rows[0].keys()}
                        # Try to coerce numeric columns to float arrays
                        for col, vals in list(cols.items()):
                            try:
                                cols[col] = _np.asarray([float(v) for v in vals])
                            except Exception:
                                cols[col] = vals  # keep as list of strings
                        inputs_struct[safe] = {"table": cols, "label": label}
                    else:
                        inputs_struct[safe] = {"table": {}, "label": label}
            except Exception as _e:
                import sys as __s
                print(f"[run-matlab] extra_input {x.get('key')!r} decode failed: {_e}", file=__s.stderr, flush=True)
        try:
            _savemat(inputs_mat, {"inputs": inputs_struct}, do_compression=True)
        except Exception as _e:
            return {"success": False, "stdout": "", "stderr": f"savemat failed: {_e}",
                    "plots": [], "tables": [], "images": []}

        plot_dir = os.path.join(tmpdir, "plots")
        table_dir = os.path.join(tmpdir, "tables")
        image_dir = os.path.join(tmpdir, "images")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # MATLAB / Octave harness. Both share enough syntax for this
        # to work in either interpreter.
        harness_template = """% Auto-generated harness — do not edit by hand
_MPFIG_INPUTS_PATH = '__INPUTS_PATH__';
_MPFIG_PLOT_DIR    = '__PLOT_DIR__';
_MPFIG_TABLE_DIR   = '__TABLE_DIR__';
_MPFIG_IMAGE_DIR   = '__IMAGE_DIR__';

load(_MPFIG_INPUTS_PATH);  % defines `inputs`

_mpfig_plot_count = 0;
_mpfig_table_count = 0;
_mpfig_image_count = 0;

function mpfig_plot(name)
  global _MPFIG_PLOT_DIR _mpfig_plot_count
  _mpfig_plot_count = _mpfig_plot_count + 1;
  if nargin < 1 || isempty(name)
    name = sprintf('plot_%d.png', _mpfig_plot_count);
  end
  if ~ischar(name)
    name = char(name);
  end
  fp = fullfile(_MPFIG_PLOT_DIR, name);
  print(gcf(), fp, '-dpng', '-r150');
end

function mpfig_data(tbl, name)
  global _MPFIG_TABLE_DIR _mpfig_table_count
  _mpfig_table_count = _mpfig_table_count + 1;
  if nargin < 2 || isempty(name)
    name = sprintf('table_%d', _mpfig_table_count);
  end
  fp = fullfile(_MPFIG_TABLE_DIR, [char(name) '.csv']);
  if isstruct(tbl)
    fns = fieldnames(tbl);
    n = 0;
    for k = 1:numel(fns)
      n = max(n, numel(tbl.(fns{k})));
    end
    fid = fopen(fp, 'w');
    fprintf(fid, '%s', fns{1});
    for k = 2:numel(fns); fprintf(fid, ',%s', fns{k}); end
    fprintf(fid, '\\n');
    for i = 1:n
      for k = 1:numel(fns)
        v = tbl.(fns{k});
        if iscell(v); cell_v = v{min(i, numel(v))}; else; cell_v = v(min(i, numel(v))); end
        if k > 1; fprintf(fid, ','); end
        if ischar(cell_v); fprintf(fid, '%s', cell_v); else; fprintf(fid, '%g', cell_v); end
      end
      fprintf(fid, '\\n');
    end
    fclose(fid);
  elseif isnumeric(tbl)
    csvwrite(fp, tbl);
  else
    fid = fopen(fp, 'w'); fprintf(fid, '%s\\n', char(tbl)); fclose(fid);
  end
end

function mpfig_image(arr, name)
  global _MPFIG_IMAGE_DIR _mpfig_image_count
  _mpfig_image_count = _mpfig_image_count + 1;
  if nargin < 2 || isempty(name)
    name = sprintf('image_%d', _mpfig_image_count);
  end
  fp = fullfile(_MPFIG_IMAGE_DIR, [char(name) '.png']);
  if isa(arr, 'uint8')
    imwrite(arr, fp);
  else
    mn = min(arr(:)); mx = max(arr(:));
    if mx > mn
      arr = uint8(255 * (arr - mn) / (mx - mn));
    else
      arr = uint8(zeros(size(arr)));
    end
    imwrite(arr, fp);
  end
end

global _MPFIG_PLOT_DIR _MPFIG_TABLE_DIR _MPFIG_IMAGE_DIR
global _mpfig_plot_count _mpfig_table_count _mpfig_image_count

try
__USER_CODE__
catch _err
  fprintf(2, '%s\\n', _err.message);
  if isfield(_err, 'stack')
    for _ii = 1:numel(_err.stack)
      fprintf(2, '  at %s line %d\\n', _err.stack(_ii).name, _err.stack(_ii).line);
    end
  end
  exit(1);
end
"""
        harness = (
            harness_template
            .replace("__INPUTS_PATH__", inputs_mat.replace("\\", "/"))
            .replace("__PLOT_DIR__", plot_dir.replace("\\", "/"))
            .replace("__TABLE_DIR__", table_dir.replace("\\", "/"))
            .replace("__IMAGE_DIR__", image_dir.replace("\\", "/"))
            .replace("__USER_CODE__", code)
        )

        script_path = os.path.join(tmpdir, "pipeline.m")
        with open(script_path, "w") as f:
            f.write(harness)

        try:
            if kind == "octave":
                # `--no-gui --no-window-system` keeps Octave headless;
                # `-q` suppresses the welcome banner.
                cmd = [matlab_path, "--no-gui", "--no-window-system", "-q", script_path]
            else:
                # MATLAB: `-batch` evaluates and exits, suppresses
                # the splash. Pass the script content via -batch
                # since MATLAB doesn't accept a .m file as the
                # entry directly without a function block.
                cmd = [matlab_path, "-batch",
                       "run('" + script_path.replace("'", "''") + "')",
                       "-nosplash", "-nodesktop"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": f"{kind.upper()} pipeline timed out after {timeout_sec} seconds.",
                    "plots": [], "tables": [], "images": []}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e),
                    "plots": [], "tables": [], "images": []}

        plots_b64: List[str] = []
        for png_path in sorted(glob_mod.glob(os.path.join(plot_dir, "*.*"))):
            try:
                with open(png_path, "rb") as pf:
                    plots_b64.append(base64.b64encode(pf.read()).decode())
            except Exception:
                pass
        tables_out: List[Dict[str, str]] = []
        for csv_path in sorted(glob_mod.glob(os.path.join(table_dir, "*.csv"))):
            try:
                with open(csv_path, "r", encoding="utf-8") as cf:
                    tables_out.append({
                        "name": os.path.splitext(os.path.basename(csv_path))[0],
                        "csv": cf.read(),
                    })
            except Exception:
                pass
        images_out: List[Dict[str, str]] = []
        for img_path in sorted(glob_mod.glob(os.path.join(image_dir, "*.*"))):
            try:
                with open(img_path, "rb") as pf:
                    images_out.append({
                        "name": os.path.splitext(os.path.basename(img_path))[0],
                        "image": base64.b64encode(pf.read()).decode(),
                    })
            except Exception:
                pass

        return {
            "success": result.returncode == 0,
            "kind": kind,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "plots": plots_b64,
            "tables": tables_out,
            "images": images_out,
        }


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    import sys
    try:
        parser = argparse.ArgumentParser(description="Multi-Panel Figure Builder API Server")
        parser.add_argument("--port", type=int, default=0, help="Port (0 = auto)")
        parser.add_argument("--host", default="127.0.0.1")
        # Dev-only: watch this directory and auto-restart uvicorn workers
        # whenever a .py file changes.  Mirrors Vite's HMR feel for the
        # Python side, so editing api_server.py / models.py / figure_builder.py
        # no longer requires a manual kill + restart.
        # Implemented via uvicorn's `reload=True`, which needs the app
        # passed as an import string ("api_server:app") instead of the
        # already-imported `app` object.
        # NOTE: doesn't work under PyInstaller --onefile (no source files
        # on disk for the watcher).  We detect that via sys.frozen and
        # silently ignore --reload there.
        parser.add_argument(
            "--reload", action="store_true",
            help="Dev mode: auto-restart when .py files in the sidecar "
                 "directory change.  Ignored when running as a frozen binary.",
        )
        args = parser.parse_args()

        port = args.port
        if port == 0:
            import socket
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        print(f"READY:{port}", flush=True)

        # Detect PyInstaller / frozen-binary context — reload watchers
        # need source files on disk that match what's imported, which
        # isn't the case after PyInstaller bundles everything into a
        # single executable.  Fall back to the non-reload path.
        is_frozen = getattr(sys, "frozen", False)
        if args.reload and not is_frozen:
            # Watching just SCRIPT_DIR avoids the parent venv firehose
            # of irrelevant .py changes (numpy, fastapi, etc.) that
            # would otherwise restart on every git checkout.
            #
            # `reload_delay=3` collapses bursty file events (e.g. Dropbox
            # cloud-sync touching every .py mtime in one second) into a
            # single reload — otherwise every Dropbox round-trip restarts
            # the sidecar mid-request and the user loses any uploaded
            # images / channel-group state.  Three seconds is short
            # enough that a real code-save still gives quick HMR, and
            # long enough that Dropbox bursts don't thrash.
            print(f"DEV-RELOAD: watching {SCRIPT_DIR} for *.py changes (3s debounce)", flush=True)
            uvicorn.run(
                "api_server:app",
                host=args.host,
                port=port,
                log_level="warning",
                reload=True,
                reload_dirs=[str(SCRIPT_DIR)],
                reload_includes=["*.py"],
                reload_delay=3.0,
            )
        else:
            uvicorn.run(app, host=args.host, port=port, log_level="warning")
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
