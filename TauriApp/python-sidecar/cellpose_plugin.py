"""Isolated-venv Cellpose plugin host.

The shipped sidecar is a frozen PyInstaller binary, so it cannot `pip install`
into itself (and even if it could, compiled deps like torch are ABI-locked to
its exact CPython). Instead we host Cellpose in a SEPARATE virtualenv built
from a real system Python, kept in a persistent user-data dir that survives app
updates, and run it as a SUBPROCESS. The sidecar never imports cellpose — it
extracts the input images, hands them to the venv's Python via a small runner
script, and reads the results back. This is version/ABI-independent and works
identically in the packaged app and in dev.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


# ── Locations ──────────────────────────────────────────────────────────────
def plugin_base_dir() -> str:
    """Persistent, update-surviving base dir for plugin venvs + runners."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support", "MultiPanelFigure")
    elif os.name == "nt":
        base = os.path.join(os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming"), "MultiPanelFigure")
    else:
        base = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share"), "MultiPanelFigure")
    os.makedirs(base, exist_ok=True)
    return base


def cellpose_venv_dir(version: str = "v4") -> str:
    """Per-major-version venv path.  v3 and v4 coexist on disk so the
    user can pick which API to run per node — v3 has the real
    `cyto3` / `nuclei` model zoo, v4 ships only `cpsam` (a SAM-based
    generalist).  Each lives in its own venv to avoid pip's "one
    cellpose package per env" constraint.
    """
    norm = (version or "v4").lower()
    if norm in ("v3", "3", "cellpose3", "cellpose-3"):
        name = "cellpose-v3"
    else:
        name = "cellpose-v4"
    return os.path.join(plugin_base_dir(), "venvs", name)


def legacy_cellpose_venv_dir() -> str:
    """Pre-multi-version venv (single "cellpose" dir).  Users who
    installed before 0.1.324 land here.  We treat it as whichever
    major it actually has installed so existing workflows don't
    break — install_stream(version) builds the new versioned dirs."""
    return os.path.join(plugin_base_dir(), "venvs", "cellpose")


def venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def runner_path() -> str:
    return os.path.join(plugin_base_dir(), "cellpose_runner.py")


def find_bootstrap_python() -> Optional[str]:
    """Locate a REAL Python 3 (not the frozen sidecar) that can build a venv.
    Prefers an interpreter in the torch-supported range (3.9–3.13) so the
    Cellpose/torch wheels actually exist — a bleeding-edge Python (e.g. 3.14)
    or an EOL one would make `pip install` fail. Returns the best path, or None
    if nothing usable is found."""
    candidates: List[str] = []
    # Explicit, well-supported versions first.
    for name in ("python3.12", "python3.11", "python3.10", "python3.13", "python3.9"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    # The sidecar's own interpreter, but only if it's a real Python (in dev the
    # sidecar runs as system/Xcode Python; when frozen, sys.frozen is set).
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    candidates += [
        "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3",
    ]

    seen = set()
    usable: List[Tuple[Tuple[int, int], str]] = []  # (version, path), in discovery order
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            r = subprocess.run(
                [c, "-c", "import venv,ensurepip,sys;print('%d.%d' % sys.version_info[:2])"],
                capture_output=True, text=True, timeout=25,
            )
            if r.returncode != 0:
                continue
            ver = tuple(int(x) for x in (r.stdout or "").strip().split("."))[:2]
            if len(ver) == 2:
                usable.append((ver, c))  # type: ignore[arg-type]
        except Exception:
            continue
    if not usable:
        return None
    # Prefer a torch-supported version (3.9–3.13); otherwise fall back to the
    # first usable interpreter (install may fail, but with a clear pip error).
    for ver, path in usable:
        if (3, 9) <= ver <= (3, 13):
            return path
    return usable[0][1]


# ── Runner script (executed INSIDE the venv) ───────────────────────────────
# Self-contained: reads a job.json {config, inputs:[{label,path}], out_dir},
# runs cellpose on each input, writes <label>_mask/_labels/_outlines PNGs and a
# result.json. Uses only cellpose + numpy + pillow (all in the venv). The eval
# logic mirrors the sidecar's previous in-process pipeline (v3 + v4 APIs).
RUNNER_SRC = r'''
import sys, os, json, io, time

def main():
    job_path = sys.argv[1]
    with open(job_path) as f:
        job = json.load(f)
    cfg = job.get("config") or {}
    inputs = job.get("inputs") or []
    out_dir = job.get("out_dir") or os.path.dirname(job_path)
    result = {"success": False, "stdout": "", "error": "", "rows": [], "images": []}
    log = []
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        result["error"] = "Plugin venv missing numpy/pillow: %s" % e
        _write(out_dir, result); return
    try:
        from cellpose import models as cp_models, utils as cp_utils
        import importlib
        _cp = importlib.import_module("cellpose")
        cp_ver = (getattr(_cp, "version", None) or getattr(_cp, "__version__", None) or "")
        cp_ver = (cp_ver.strip().splitlines()[0] if cp_ver else "")
    except Exception as e:
        result["error"] = ("Cellpose isn't installed in the plugin environment: %s\n"
                           "Use 'Install Cellpose 3' to (re)install it." % e)
        _write(out_dir, result); return

    model_name = str(cfg.get("model", "cyto3"))
    diameter = cfg.get("diameter")
    channels = cfg.get("channels", [0, 0])
    flow_threshold = float(cfg.get("flow_threshold", 0.4))
    cellprob_threshold = float(cfg.get("cellprob_threshold", 0.0))
    min_size = int(cfg.get("min_size", 15))
    use_gpu = bool(cfg.get("use_gpu", False))

    is_v4 = not hasattr(cp_models, "Cellpose")
    if is_v4 and model_name not in {"cpsam"}:
        log.append("WARNING: cellpose 4.x only ships 'cpsam' — %r falls back to cpsam." % model_name)
        model_name = "cpsam"

    # Lazy model loader.  The model (~100 MB for cpsam) takes ~30 s
    # cold-start, so we defer the load until the FIRST image that
    # actually needs cellpose.eval().  When every image carries a
    # user-supplied mask (the common case after a dialog preview +
    # Save), the model is never loaded — turning what used to be a
    # 30 s+ run into a fraction of a second.
    _model = {"m": None, "err": None}
    def _ensure_model():
        if _model["m"] is not None: return _model["m"]
        if _model["err"] is not None: return None
        _t = time.monotonic()
        try:
            if is_v4:
                _model["m"] = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
            else:
                _model["m"] = cp_models.Cellpose(gpu=use_gpu, model_type=model_name)
            log.append("loaded cellpose %s model=%r gpu=%s api=%s in %.1fs (lazy)"
                       % (cp_ver or "(unknown)", model_name, use_gpu,
                          "v4" if is_v4 else "v3", time.monotonic() - _t))
        except Exception as _le:
            _model["err"] = "Failed to load cellpose model: %s" % _le
            log.append("cellpose model load FAILED: %s" % _le)
        return _model["m"]

    # Optional pre-supplied label rasters keyed by image label.  When
    # the user has interactively painted/deleted cells in the dialog,
    # we round-trip the corrected mask back here as a 16-bit-packed
    # RGBA PNG (R = low byte, G = high byte) on disk; the runner reads
    # it INSTEAD of running model.eval on that image.  Lets the final
    # quantification reflect the user's edits exactly.
    edited_label_paths = cfg.get("edited_label_paths") or {}

    def _decode_labels_rgba(path, target_shape=None):
        # PIL keeps the four channels intact.  Returns int32 HxW.
        # Optionally NN-resizes to target_shape (h, w) — needed when
        # the dialog painted at PREVIEW resolution (~1024 px) but the
        # cellpose input arrived at a different size.
        try:
            im = Image.open(path).convert("RGBA")
            if target_shape is not None:
                th, tw = int(target_shape[0]), int(target_shape[1])
                if (im.size[1], im.size[0]) != (th, tw):
                    im = im.resize((tw, th), resample=Image.NEAREST)
            arr = np.asarray(im)
            return arr[..., 0].astype(np.int32) | (arr[..., 1].astype(np.int32) << 8)
        except Exception:
            return None

    t0 = time.monotonic()
    for i, item in enumerate(inputs, start=1):
        label = str(item.get("label") or ("src_%d" % i))
        path = item.get("path")
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            log.append("[%d/%d] %r: failed to read input (%s)" % (i, len(inputs), label, e))
            continue
        masks = None
        flows = None
        used_edits = False
        edit_path = edited_label_paths.get(label) if isinstance(edited_label_paths, dict) else None
        if edit_path and os.path.isfile(edit_path):
            # Pass the image dims so the runner NN-upscales preview-res
            # edits (the dialog paints at ~1024 px) to the cellpose
            # input size (downsized to ~1500 px max edge by the
            # generator).  Result: an edited mask painted in the dialog
            # is faithfully applied to whatever size cellpose sees.
            ed = _decode_labels_rgba(edit_path, target_shape=img.shape[:2])
            if ed is not None and ed.shape[:2] == img.shape[:2]:
                masks = ed
                used_edits = True
                log.append("[%d/%d] %r: USING user-edited mask (%d cell(s))"
                           % (i, len(inputs), label, int(ed.max())))
            else:
                log.append("[%d/%d] %r: edited mask shape mismatch / decode failed — re-running cellpose"
                           % (i, len(inputs), label))

        if masks is None:
            m = _ensure_model()
            if m is None:
                log.append("[%d/%d] %r: skipped (model unavailable: %s)"
                           % (i, len(inputs), label, _model.get("err") or "?"))
                continue
            log.append("[%d/%d] segmenting %r shape=%s" % (i, len(inputs), label, img.shape))
            ek = dict(diameter=diameter, flow_threshold=flow_threshold,
                      cellprob_threshold=cellprob_threshold, min_size=min_size)
            try:
                res = m.eval(img, channels=channels, **ek)
            except TypeError:
                res = m.eval(img, **ek)
            masks = res[0]
            # flows is a list — flows[0] = HxWx3 RGB direction visualisation,
            # flows[2] = HxW cellprob (float).  Shape varies slightly across
            # cellpose versions; guard each access.
            try:
                flows = res[1]
            except Exception:
                flows = None
        n_cells = int(masks.max())

        # coloured mask png
        mask_rgb = np.zeros((*masks.shape, 3), dtype=np.uint8)
        if n_cells > 0:
            rng = np.random.default_rng(42)
            palette = rng.integers(64, 255, size=(n_cells + 1, 3), dtype=np.uint8)
            palette[0] = (0, 0, 0)
            mask_rgb = palette[masks]
        _save(out_dir, "%s_mask.png" % label, mask_rgb)
        result["images"].append({"name": "%s_mask" % label, "file": "%s_mask.png" % label})

        # 8-bit label image (legacy — kept for backwards compat with
        # downstream code paths that read <label>_labels by name).
        labels8 = masks.astype(np.int64)
        if n_cells > 255:
            labels8 = ((labels8 - 1) % 255 + 1) * (labels8 > 0)
            log.append("WARNING: %r has %d cells (>255) — packed cyclically." % (label, n_cells))
        _save(out_dir, "%s_labels.png" % label, labels8.astype(np.uint8), mode="L")
        result["images"].append({"name": "%s_labels" % label, "file": "%s_labels.png" % label})

        # 16-bit label image as RGBA-packed PNG so the dialog can round-
        # trip masks through the browser canvas without losing IDs >255.
        # R = low byte, G = high byte, B = 0, A = 255.
        m_lo = (masks & 0xFF).astype(np.uint8)
        m_hi = ((masks >> 8) & 0xFF).astype(np.uint8)
        rgba = np.zeros((masks.shape[0], masks.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = m_lo
        rgba[..., 1] = m_hi
        rgba[..., 3] = 255
        _save(out_dir, "%s_labels16.png" % label, rgba)
        result["images"].append({"name": "%s_labels16" % label, "file": "%s_labels16.png" % label})

        # outlines overlay
        outline_img = img.copy()
        if outline_img.ndim == 2:
            outline_img = np.stack([outline_img] * 3, axis=-1)
        try:
            for o in cp_utils.outlines_list(masks):
                for (xx, yy) in o.astype(int):
                    if 0 <= yy < outline_img.shape[0] and 0 <= xx < outline_img.shape[1]:
                        outline_img[yy, xx] = (255, 80, 80)
        except Exception:
            pass
        _save(out_dir, "%s_outlines.png" % label, outline_img)
        result["images"].append({"name": "%s_outlines" % label, "file": "%s_outlines.png" % label})

        # Flows (only available when model.eval just ran — skipped when
        # the user supplied an edited mask).  flows[0] is HxWx3 uint8
        # already; flows[2] is HxW float cellprob.
        if flows is not None:
            try:
                flow_rgb = np.asarray(flows[0])
                if flow_rgb.ndim == 3 and flow_rgb.shape[2] >= 3:
                    flow_rgb_u8 = np.ascontiguousarray(flow_rgb[..., :3].astype(np.uint8))
                    _save(out_dir, "%s_flows_rgb.png" % label, flow_rgb_u8)
                    result["images"].append({"name": "%s_flows_rgb" % label, "file": "%s_flows_rgb.png" % label})
            except Exception as _fe:
                log.append("[%d/%d] %r: flows RGB unavailable (%s)" % (i, len(inputs), label, _fe))
            try:
                cprob = np.asarray(flows[2]).astype(np.float32)
                cp_lo = float(cprob.min()); cp_hi = float(cprob.max())
                if cp_hi - cp_lo > 1e-6:
                    cp_u8 = ((cprob - cp_lo) / (cp_hi - cp_lo) * 255.0).astype(np.uint8)
                else:
                    cp_u8 = np.zeros_like(cprob, dtype=np.uint8)
                _save(out_dir, "%s_cellprob.png" % label, cp_u8, mode="L")
                result["images"].append({"name": "%s_cellprob" % label, "file": "%s_cellprob.png" % label})
            except Exception as _ce:
                log.append("[%d/%d] %r: cellprob unavailable (%s)" % (i, len(inputs), label, _ce))

        # show_segmentation-style overview: a single PNG with four
        # PIL panels side-by-side — original | flows RGB | coloured
        # masks | outlines.  Mirrors cellpose.plot.show_segmentation
        # so users get the same at-a-glance QC view they'd get
        # running cellpose from the CLI.  PIL-only (no matplotlib
        # dep) so it works on every venv we install into.
        try:
            from PIL import ImageDraw as _ID, ImageFont as _IF
            panels = [("original", img if img.ndim == 3 else np.stack([img]*3, axis=-1))]
            # Flows panel only when flows actually arrived (skipped on
            # edited masks; we show a uniform grey filler so the four
            # panels stay aligned).
            if flows is not None:
                try:
                    fr = np.asarray(flows[0])
                    if fr.ndim == 3 and fr.shape[2] >= 3:
                        panels.append(("flows", fr[..., :3].astype(np.uint8)))
                    else:
                        raise ValueError("flows[0] wrong shape")
                except Exception:
                    panels.append(("flows (n/a)", np.full((img.shape[0], img.shape[1], 3), 60, dtype=np.uint8)))
            else:
                panels.append(("flows (skipped on edit)",
                               np.full((img.shape[0], img.shape[1], 3), 60, dtype=np.uint8)))
            panels.append(("masks (N=%d)" % n_cells, mask_rgb))
            panels.append(("outlines", outline_img))
            # Resize each panel to a common height so they line up
            # cleanly; max 380 px tall keeps the overview file modest
            # while staying readable at typical preview zoom.
            target_h = min(380, img.shape[0])
            scale = target_h / float(img.shape[0])
            target_w = max(1, int(round(img.shape[1] * scale)))
            from PIL import Image as _PI
            tiles = []
            for name, arr_p in panels:
                tile = _PI.fromarray(arr_p).convert("RGB").resize((target_w, target_h), _PI.LANCZOS)
                # Stamp the panel name in the top-left corner.
                d = _ID.Draw(tile)
                try: font = _IF.load_default()
                except Exception: font = None
                d.rectangle([(0, 0), (max(110, len(name) * 7), 16)], fill=(0, 0, 0))
                d.text((4, 1), name, fill=(255, 255, 255), font=font)
                tiles.append(tile)
            # Glue them with a 2-px white separator.
            gap = 2
            stitched_w = target_w * len(tiles) + gap * (len(tiles) - 1)
            stitched = _PI.new("RGB", (stitched_w, target_h), color=(255, 255, 255))
            for k, t in enumerate(tiles):
                stitched.paste(t, (k * (target_w + gap), 0))
            # Title bar across the top with the source label.
            title_h = 22
            full = _PI.new("RGB", (stitched_w, target_h + title_h), color=(28, 28, 28))
            d2 = _ID.Draw(full)
            try: tf = _IF.load_default()
            except Exception: tf = None
            d2.text((6, 4), "%s — show_segmentation%s" % (label, " (edited)" if used_edits else ""),
                    fill=(255, 255, 255), font=tf)
            full.paste(stitched, (0, title_h))
            _save(out_dir, "%s_show_segmentation.png" % label, np.array(full))
            result["images"].append({"name": "%s_show_segmentation" % label,
                                     "file": "%s_show_segmentation.png" % label})
        except Exception as _se:
            log.append("[%d/%d] %r: show_segmentation render failed: %s"
                       % (i, len(inputs), label, _se))

        sizes = np.bincount(masks.ravel())[1:] if n_cells > 0 else np.array([0])
        result["rows"].append({
            "source": label, "n_cells": n_cells,
            "mean_area_px": float(sizes.mean()) if n_cells > 0 else 0.0,
            "median_area_px": float(np.median(sizes)) if n_cells > 0 else 0.0,
            "model": model_name,
            "edited": bool(used_edits),
        })
        log.append("[%d/%d] %r: %d cell(s)%s" % (i, len(inputs), label, n_cells,
                                                  " (edited)" if used_edits else ""))

    log.append("total: %d image(s) in %.1fs" % (len(inputs), time.monotonic() - t0))
    result["success"] = True
    result["stdout"] = "\n".join(log)
    _write(out_dir, result)


def _save(out_dir, name, arr, mode=None):
    from PIL import Image
    Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)
    (Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)).save(os.path.join(out_dir, name), format="PNG")


def _write(out_dir, result):
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
'''


def write_runner() -> str:
    p = runner_path()
    with open(p, "w") as f:
        f.write(RUNNER_SRC)
    return p


# ── Public API used by api_server endpoints ────────────────────────────────
def _probe_venv(venv_path: str) -> dict:
    """Returns {installed, kind, path, major} for a single venv path."""
    vpy = venv_python(venv_path)
    if not os.path.isfile(vpy):
        return {"installed": False, "path": vpy}
    try:
        r = subprocess.run(
            [vpy, "-c",
             "import cellpose;print(getattr(cellpose,'version',None) or getattr(cellpose,'__version__','') or 'installed')"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            ver_str = ((r.stdout or "").strip().splitlines()[0] if r.stdout else "")
            # Detect major from the version string (e.g. "3.1.2" → 3, "4.0.0" → 4).
            major = 0
            try:
                first = ver_str.split(".")[0]
                major = int("".join(ch for ch in first if ch.isdigit()))
            except Exception:
                major = 0
            return {"installed": True, "kind": ("cellpose " + ver_str).strip(),
                    "path": vpy, "major": major}
    except Exception:
        pass
    return {"installed": False, "path": vpy}


def check_all() -> dict:
    """Probe BOTH versioned venvs + the legacy single-venv directory.
    Returns a dict { v3: {...}, v4: {...}, legacy: {...} } that the
    frontend uses to drive the dual install cards."""
    v3 = _probe_venv(cellpose_venv_dir("v3"))
    v4 = _probe_venv(cellpose_venv_dir("v4"))
    legacy = _probe_venv(legacy_cellpose_venv_dir())
    # When a versioned venv isn't installed but the legacy single-venv
    # IS, surface the legacy install as the matching slot so existing
    # users keep working without re-installing.
    if not v4.get("installed") and legacy.get("installed") and int(legacy.get("major", 0)) >= 4:
        v4 = {**legacy, "legacy": True}
    if not v3.get("installed") and legacy.get("installed") and int(legacy.get("major", 0)) == 3:
        v3 = {**legacy, "legacy": True}
    return {"v3": v3, "v4": v4, "legacy": legacy}


def check() -> dict:
    """Legacy single-slot check.  Returns v4 status first (the modern
    default); falls back to v3 or legacy.  Older callers (pre-0.1.324)
    expect {installed, kind, path} so this keeps them working."""
    all_ = check_all()
    if all_["v4"].get("installed"):
        return {k: v for k, v in all_["v4"].items() if k != "legacy"}
    if all_["v3"].get("installed"):
        return {k: v for k, v in all_["v3"].items() if k != "legacy"}
    return {"installed": False, "path": all_["v4"].get("path", "")}


def install_stream(version: str = "v4"):
    """Generator yielding SSE 'data:' lines while building the venv +
    installing the requested Cellpose major version into it.  Final
    event carries the return code.

    version — "v3" pins `pip install "cellpose<4"` so the v3 model zoo
              (cyto, cyto2, cyto3, nuclei, …) is available.
              "v4" pins `pip install "cellpose>=4"` (cpsam-only).
    Each version lives in its OWN venv so they coexist on disk and the
    user can switch per-node without reinstalling.
    """
    def emit(line: str) -> str:
        esc = str(line).replace("\\", "\\\\").replace('"', '\\"')
        return 'data: {"line":"%s"}\n\n' % esc

    norm_version = (version or "v4").lower()
    if norm_version not in ("v3", "v4"):
        norm_version = "v4"

    py = find_bootstrap_python()
    if not py:
        yield emit("No system Python 3 found to host plugins.")
        yield emit("Install Python 3 (python.org, or `brew install python`) and retry.")
        yield 'data: {"done":true,"returncode":-1}\n\n'
        return

    venv = cellpose_venv_dir(norm_version)
    vpy = venv_python(venv)
    yield emit("bootstrap python: %s" % py)
    yield emit("plugin environment (%s): %s" % (norm_version, venv))

    # 1) Create the venv (idempotent).
    if not os.path.isfile(vpy):
        yield emit("creating plugin virtualenv…")
        os.makedirs(os.path.dirname(venv), exist_ok=True)
        try:
            r = subprocess.run([py, "-m", "venv", venv], capture_output=True, text=True, timeout=180)
        except Exception as e:
            yield emit("venv creation failed to launch: %s" % e)
            yield 'data: {"done":true,"returncode":-1}\n\n'
            return
        if r.returncode != 0 or not os.path.isfile(vpy):
            yield emit("venv creation failed: %s" % (r.stderr or r.stdout or "")[-800:])
            yield 'data: {"done":true,"returncode":%d}\n\n' % (r.returncode or -1)
            return
        # Ensure pip exists (CLT pythons sometimes ship without it in the venv).
        subprocess.run([vpy, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True, timeout=180)

    # 2) Bootstrap the venv's packaging toolchain. Some user-supplied
    #    Pythons (and some venv creations) end up without the `packaging`
    #    module wired in, which makes `from cellpose import models` fail
    #    with "No module named 'packaging'" even after cellpose itself
    #    installs cleanly. Upgrading pip + (re)installing setuptools /
    #    wheel / packaging FIRST fixes that — and an existing user can
    #    re-run "Install Cellpose 3" to repair an env that hit this trap.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    yield emit("bootstrap: upgrading pip + setuptools + wheel + packaging")
    boot = subprocess.run(
        [vpy, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
         "pip", "setuptools", "wheel", "packaging"],
        capture_output=True, text=True, timeout=600, env=env,
    )
    for line in (boot.stdout or "").splitlines() + (boot.stderr or "").splitlines():
        if line.strip():
            yield emit(line.rstrip())
    if boot.returncode != 0:
        yield emit("warning: bootstrap returned %d (continuing — pip may still work)" % boot.returncode)

    # 3) Install/upgrade cellpose + its I/O deps, streaming pip output.
    #    Pin the cellpose major to the requested version so v3 (the
    #    real cyto3 + nuclei model zoo) and v4 (cpsam generalist) can
    #    coexist on disk in separate venvs.
    cellpose_spec = "cellpose<4" if norm_version == "v3" else "cellpose>=4"
    cmd = [vpy, "-u", "-m", "pip", "install", "--upgrade", "--no-input",
           "--progress-bar", "on", cellpose_spec, "numpy", "pillow"]
    yield emit("pip install --upgrade %r numpy pillow (may take 5–15 min, ~500 MB)" % cellpose_spec)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
    except Exception as e:
        yield emit("pip launch failed: %s" % e)
        yield 'data: {"done":true,"returncode":-1}\n\n'
        return
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, ""):
        yield emit(raw.rstrip("\n"))
    proc.wait()
    if proc.returncode == 0:
        try:
            write_runner()
            yield emit("plugin runner installed.")
        except Exception as e:
            yield emit("warning: could not write runner: %s" % e)
    yield 'data: {"done":true,"returncode":%d}\n\n' % proc.returncode


def run(images_in: List[Tuple[str, "object"]], cfg_dict: dict, timeout_sec: int = 600) -> dict:
    """Run Cellpose on already-extracted (label, ndarray) images via the venv
    subprocess. Returns the same envelope shape the endpoint returned before:
    { success, stdout, stderr, plots, tables, images }.

    cfg_dict["cellpose_version"] (optional, "v3" / "v4", default "v4")
    selects which venv to dispatch into.  Falls back to the legacy
    single-venv install when the requested version isn't installed
    yet — keeps existing setups working without a forced re-install.
    """
    from PIL import Image  # sidecar has pillow

    # Pull the version preference OUT of cfg_dict so it doesn't end up
    # in the runner job (the runner doesn't care which venv it lives in;
    # it just runs cellpose).
    version = str(cfg_dict.pop("cellpose_version", "v4")).lower()
    if version not in ("v3", "v4"):
        version = "v4"
    vpy = venv_python(cellpose_venv_dir(version))
    chosen_venv_kind = version
    if not os.path.isfile(vpy):
        # Try the OTHER versioned venv.
        alt = "v3" if version == "v4" else "v4"
        vpy = venv_python(cellpose_venv_dir(alt))
        if os.path.isfile(vpy):
            chosen_venv_kind = alt + " (fallback)"
        else:
            # Fall back to the legacy single venv (pre-0.1.324 installs).
            vpy = venv_python(legacy_cellpose_venv_dir())
            if os.path.isfile(vpy):
                chosen_venv_kind = "legacy"
    if not os.path.isfile(vpy):
        return {"success": False, "stdout": "",
                "stderr": ("Cellpose isn't installed yet. Open Plugins → "
                           "Install Cellpose %s first." % version),
                "plots": [], "tables": [], "images": []}
    try:
        write_runner()
    except Exception as e:
        return {"success": False, "stdout": "",
                "stderr": "Couldn't write the cellpose runner: %s" % e,
                "plots": [], "tables": [], "images": []}

    workdir = tempfile.mkdtemp(prefix="mpfig_cellpose_")
    try:
        inputs = []
        for i, (label, arr) in enumerate(images_in):
            p = os.path.join(workdir, "in_%d.png" % i)
            Image.fromarray(arr).convert("RGB").save(p, format="PNG")
            inputs.append({"label": str(label), "path": p})

        # User-edited masks (from the interactive Intensity dialog):
        # base64-encoded 16-bit-packed RGBA PNGs keyed by image label.
        # Dump each to disk and pass the paths through cfg_dict so the
        # runner sub-process can read them WITHOUT re-encoding (and so
        # the JSON job stays small).
        edited_masks = cfg_dict.pop("edited_masks", None) if isinstance(cfg_dict, dict) else None
        edited_paths: Dict[str, str] = {}
        if isinstance(edited_masks, dict):
            for j, (lbl, b64s) in enumerate(edited_masks.items()):
                if not (isinstance(lbl, str) and isinstance(b64s, str) and b64s):
                    continue
                try:
                    raw = base64.b64decode(b64s.split(",")[-1])
                    ep = os.path.join(workdir, "edit_%d.png" % j)
                    with open(ep, "wb") as fh:
                        fh.write(raw)
                    edited_paths[lbl] = ep
                except Exception:
                    continue
        if edited_paths:
            cfg_dict = {**cfg_dict, "edited_label_paths": edited_paths}

        job = {"config": cfg_dict, "inputs": inputs, "out_dir": workdir}
        job_path = os.path.join(workdir, "job.json")
        with open(job_path, "w") as f:
            json.dump(job, f)

        try:
            proc = subprocess.run([vpy, runner_path(), job_path],
                                  capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "",
                    "stderr": "Cellpose timed out after %ds." % timeout_sec,
                    "plots": [], "tables": [], "images": []}

        res_path = os.path.join(workdir, "result.json")
        if not os.path.isfile(res_path):
            return {"success": False, "stdout": proc.stdout[-2000:] if proc.stdout else "",
                    "stderr": "Cellpose runner produced no result.\n" + (proc.stderr or "")[-2000:],
                    "plots": [], "tables": [], "images": []}
        with open(res_path) as f:
            res = json.load(f)

        images_out: List[Dict[str, str]] = []
        for im in res.get("images", []):
            fp = os.path.join(workdir, im.get("file", ""))
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    images_out.append({"name": im.get("name", "out"),
                                       "image": base64.b64encode(fh.read()).decode()})

        if not res.get("success"):
            return {"success": False, "stdout": res.get("stdout", ""),
                    "stderr": res.get("error", "") or (proc.stderr or "")[-2000:],
                    "plots": [], "tables": [], "images": images_out}

        rows = res.get("rows", [])
        csv_lines = ["source,n_cells,mean_area_px,median_area_px,model"]
        for r in rows:
            csv_lines.append(",".join(str(r.get(k, "")) for k in
                             ("source", "n_cells", "mean_area_px", "median_area_px", "model")))
        tables = [{"name": "cellpose_counts", "csv": "\n".join(csv_lines) + "\n"}]
        return {"success": True, "kind": "cellpose",
                "stdout": res.get("stdout", ""), "stderr": "",
                "plots": [], "tables": tables, "images": images_out}
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
