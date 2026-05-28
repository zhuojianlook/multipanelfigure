// ─────────────────────────────────────────────────────────────
//  IntensityPickerDialog — fluorescence segmentation picker
// ─────────────────────────────────────────────────────────────
//  Like the Western-blot Band Picker, but for fluorescence channel
//  intensity quantification. The user:
//    • Cycles through the upstream images with prev/next arrows.
//    • Picks a STRATEGY:
//        Simple   — per-channel rolling-ball BG + threshold. Each
//                   enabled channel becomes a binary mask; the
//                   intensity sample = mean within that channel's
//                   mask. n = images per group.
//        Cellpose — per-cell segmentation (Cellpose 3+ via the
//                   plugin venv). One row per cell per channel;
//                   n = cells per group.
//    • Tweaks the strategy's params (per-channel threshold + min
//      area for simple; model / diameter / segment-on for
//      Cellpose) with a LIVE PREVIEW overlay updating after a
//      short debounce.
//    • Renames each channel (Channel R → DAPI, Channel G →
//      Anti-VegF, …) and assigns each image to an experimental
//      GROUP (Control / Treatment / …).
//    • Picks one channel as the CONTROL — the downstream R plot
//      adds a sanity-check panel ("DAPI should be similar across
//      groups") and flags it red if the control is significantly
//      different.
//
//  The picker emits the Python code body the analysis node runs
//  (and saves the config back onto the node) so a normal "Run"
//  reproduces the configured run on the full-resolution images.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box, Typography, Button, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, Tooltip, ToggleButton, ToggleButtonGroup, MenuItem,
  CircularProgress,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import RefreshIcon from "@mui/icons-material/Refresh";

// ── Types (exported so the host can persist the config on the node) ──
export interface FluorChannels {
  r: string;
  g: string;
  b: string;
}
export interface FluorGroup {
  id: string;
  name: string;
  /** Image labels in this group (matched against each upstream input's `label`). */
  images: string[];
}
export interface FluorCellpose {
  /** Cellpose 4 model name. "cpsam" = the SAM-based default. */
  model: string;
  /** Object diameter prior in px. 0 = auto-estimate. */
  diameter: number;
  /** Which channel to segment ON (cells / nuclei). */
  segChannel: "r" | "g" | "b";
  /** Min object size in px² — filters out spurious tiny masks. */
  minSize: number;
}

/** Per-channel threshold knobs for the SIMPLE strategy. Each channel
 *  gets its own rule because fluorescent stains have very different
 *  intensity distributions — DAPI baseline ≠ a sparse marker like
 *  Anti-VegF, so one global percentile rarely works for all three. */
export interface FluorChannelThreshold {
  /** When false, this channel is skipped in the simple-mode quantification
   *  (no mask, no intensity row emitted). */
  enabled: boolean;
  thresholdMethod: "percentile" | "otsu";
  thresholdPercentile: number;
  /** Drop connected components smaller than this many px². */
  minArea: number;
  /** When true, apply a Difference-of-Gaussians band-pass to this channel
   *  BEFORE thresholding. Suppresses diffuse haze and emphasises cell-
   *  sized structures, so it works well for densely-packed cells or
   *  cells with soft edges. Stacks on top of the global rolling-ball
   *  background subtraction when both are on. */
  enhanceEdges?: boolean;
}

/** Per-image upstream input — what the dialog needs to cycle / preview. */
export interface FluorImageSource {
  /** Unique upstream key (matches detection-time descriptor for source-node
   *  inputs). Used to re-extract full-res in-sidecar. */
  key: string;
  /** Source-node inset coords — same triple the wb-* endpoints use. */
  row?: number;
  col?: number;
  inset_index?: number;
  /** Optional human label. */
  name?: string;
}

export interface FluorPickerImage {
  /** Display label (matches an entry in `groups[].images`). */
  label: string;
  /** Base64 PNG of the thumbnail — what the dialog shows as the preview
   *  fallback when the backend overlay hasn't returned yet. */
  image_b64: string;
  /** Source descriptor for full-res re-extract on the live preview call.
   *  Absent for upstream-node-derived inputs (preview falls back to b64). */
  source?: FluorImageSource;
}

export interface FluorIntensityConfig {
  version: 1;
  channels: FluorChannels;
  /** "simple"   — per-channel rolling-ball BG + threshold; intensity =
   *               mean inside each channel's mask, one row per (image,
   *               channel).
   *  "cellpose" — Cellpose 3+ segments cells in the chosen channel; one
   *               row per (cell, channel).
   *  Legacy modes ("advanced") are migrated on load. */
  mode: "simple" | "cellpose";
  /** Per-channel threshold knobs for simple mode. */
  thresholds: { r: FluorChannelThreshold; g: FluorChannelThreshold; b: FluorChannelThreshold };
  /** Rolling-ball BG subtraction radius (px). 0 = disabled. Shared across
   *  channels (per-channel BG would be expensive and rarely useful). */
  rollingRadius: number;
  /** Cellpose params (used only when mode = "cellpose"). */
  cellpose: FluorCellpose;
  /** Optional "control channel" — the downstream R plot adds a sanity-check
   *  panel comparing this channel across groups (the assumption is the
   *  control stain like DAPI should be statistically similar; if it isn't,
   *  the panel is flagged so the user knows their groups may not be
   *  biologically comparable). null = no control panel. */
  controlChannel: "r" | "g" | "b" | null;
  groups: FluorGroup[];
}

export function emptyFluorConfig(): FluorIntensityConfig {
  return {
    version: 1,
    channels: { r: "Channel R", g: "Channel G", b: "Channel B" },
    mode: "simple",
    thresholds: {
      r: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30, enhanceEdges: false },
      g: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30, enhanceEdges: false },
      b: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30, enhanceEdges: false },
    },
    rollingRadius: 35,
    cellpose: { model: "cpsam", diameter: 0, segChannel: "b", minSize: 80 },
    controlChannel: null,
    groups: [],
  };
}

/** Migrate legacy schemas (pre-spread, advanced/threshold/cellpose) onto
 *  the new 2-mode shape so older saved nodes keep loading. */
function migrateConfig(cfg: FluorIntensityConfig): FluorIntensityConfig {
  const fresh = emptyFluorConfig();
  // Old shape: { mode: "simple"|"advanced"|"cellpose", quantify: {...}, cellpose: {...} }
  const legacy = cfg as unknown as {
    mode?: string;
    quantify?: {
      maskSource?: string; maskChannels?: string[]; thresholdMethod?: string;
      thresholdPercentile?: number; rollingRadius?: number; minObjectArea?: number;
      cellpose?: FluorCellpose;
    };
    cellpose?: FluorCellpose;
    thresholds?: FluorIntensityConfig["thresholds"];
    rollingRadius?: number;
    controlChannel?: "r" | "g" | "b" | null;
  };
  const out: FluorIntensityConfig = {
    ...fresh,
    ...cfg,
  };
  // Already migrated → leave alone.
  if (cfg.thresholds && (cfg.mode === "simple" || cfg.mode === "cellpose")) {
    return {
      ...out,
      thresholds: cfg.thresholds || fresh.thresholds,
      rollingRadius: typeof cfg.rollingRadius === "number" ? cfg.rollingRadius : fresh.rollingRadius,
      controlChannel: cfg.controlChannel ?? null,
      cellpose: cfg.cellpose || fresh.cellpose,
    };
  }
  // Choose new mode based on legacy combination.
  let mode: FluorIntensityConfig["mode"] = "simple";
  if (legacy.mode === "cellpose") mode = "cellpose";
  else if (legacy.mode === "advanced" && legacy.quantify?.maskSource === "cellpose") mode = "cellpose";
  // Seed per-channel thresholds from the legacy single rule (maskChannels
  // controls which channels were enabled; threshold params apply uniformly).
  const enabledSet = new Set(legacy.quantify?.maskChannels || ["r", "g", "b"]);
  const oneThresh = (k: "r" | "g" | "b"): FluorChannelThreshold => ({
    enabled: enabledSet.has(k),
    thresholdMethod: (legacy.quantify?.thresholdMethod === "otsu" ? "otsu" : "percentile"),
    thresholdPercentile: typeof legacy.quantify?.thresholdPercentile === "number"
      ? legacy.quantify!.thresholdPercentile : 95,
    minArea: typeof legacy.quantify?.minObjectArea === "number" ? legacy.quantify!.minObjectArea : 30,
  });
  return {
    ...out,
    mode,
    thresholds: legacy.thresholds || { r: oneThresh("r"), g: oneThresh("g"), b: oneThresh("b") },
    rollingRadius: typeof legacy.rollingRadius === "number"
      ? legacy.rollingRadius
      : (typeof legacy.quantify?.rollingRadius === "number" ? legacy.quantify.rollingRadius : 35),
    cellpose: legacy.quantify?.cellpose || legacy.cellpose || fresh.cellpose,
    controlChannel: legacy.controlChannel ?? null,
  };
}

let _uid = 0;
const uid = (p = "g") => `${p}_${Date.now().toString(36)}_${(_uid++).toString(36)}_${Math.random().toString(36).slice(2, 5)}`;

// ── Python code generator ────────────────────────────────────
// Self-contained — runs inside the analysis sidecar's Python
// engine. Reads `inputs` (the upstream Source's images), follows
// the config's channel renames + group mapping + mode flag, and
// emits a `fluor_intensities` table plus a mask PNG per source.
export function generateFluorCode(cfg: FluorIntensityConfig): string {
  const json = JSON.stringify(cfg);
  return `# @name: Channel intensities (renameable channels)
# Auto-generated from the interactive Intensity picker. Edit channel
# names / groups / mode / thresholds via the "Configure intensity…"
# button on the node — manual edits here are overwritten the next
# time the picker saves.
import numpy as np, json, io as _io, base64 as _b64, sys as _sys
import urllib.request as _ur
from PIL import Image as _Im, ImageDraw as _ImD

try:
    import cv2 as _cv2
    from scipy import ndimage as _ndi
    _have_scipy = True
except Exception as _e:
    print(f"[intensity] scipy/cv2 unavailable ({_e}); simple-mode quantification will still run, "
          f"per-channel masks will fall back to a raw threshold")
    _have_scipy = False

CFG = json.loads(r'''${json}''')
mode = CFG.get("mode", "simple")
ch_name = {
    "R": (CFG.get("channels", {}).get("r") or "Channel R"),
    "G": (CFG.get("channels", {}).get("g") or "Channel G"),
    "B": (CFG.get("channels", {}).get("b") or "Channel B"),
}
control_key = CFG.get("controlChannel")  # "r" | "g" | "b" | null
control_name = ch_name.get(str(control_key).upper(), None) if control_key else None
img2group = {}
for g in CFG.get("groups", []) or []:
    nm = (g.get("name") or "").strip()
    if not nm: continue
    for im in g.get("images", []) or []:
        img2group[str(im)] = nm
thresholds = CFG.get("thresholds", {}) or {}
rolling_radius = int(CFG.get("rollingRadius", 35) or 0)
cp_cfg = CFG.get("cellpose", {}) or {}

# Gather the upstream images. Prefer FULL-BIT-DEPTH pixels (image_raw)
# when present — they preserve the native scientific dynamic range —
# but fall back to the 8-bit display image otherwise.
imgs = []
for k, v in inputs.items():
    if not isinstance(v, dict): continue
    if "image_raw" in v or "image" in v:
        imgs.append((k, v))
if not imgs:
    raise SystemExit("No image inputs — wire your fluorescence panels into this node.")

def _label_of(src, key):
    return str(src.get("label") or key).rsplit("/", 1)[-1]

def _pixels(src):
    arr = np.asarray(src.get("image_raw") if "image_raw" in src else src["image"]).astype(np.float32)
    if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] < 3:
        pad = np.repeat(arr[..., :1], 3 - arr.shape[2], axis=2)
        arr = np.concatenate([arr, pad], axis=2)
    return arr[..., :3]

def _rolling_bg(img, radius):
    if radius <= 0 or not _have_scipy:
        return np.zeros_like(img, dtype=np.float64)
    k = int(radius) * 2 + 1
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
    return _cv2.morphologyEx(img.astype(np.float32), _cv2.MORPH_OPEN, kernel).astype(np.float64)

def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r

def _threshold_mask(corr, method, pct):
    if method == "otsu" and _have_scipy:
        lo = np.percentile(corr, 0.2); hi = np.percentile(corr, 99.8)
        if hi > lo:
            scaled = np.clip((corr - lo) / (hi - lo), 0, 1)
            u8 = (scaled * 255).astype(np.uint8)
            thr_u8, _ = _cv2.threshold(u8, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
            thr = float(lo + (thr_u8 / 255.0) * (hi - lo))
        else:
            thr = float(np.percentile(corr, 99))
    else:
        thr = float(np.percentile(corr, float(pct)))
    m = corr > thr
    if _have_scipy:
        m = _ndi.binary_opening(m, structure=_disk(1))
        m = _ndi.binary_closing(m, structure=_disk(2))
        m = _ndi.binary_fill_holes(m)
    return m.astype(bool)

def _composite_u8(arr):
    """Contrast-stretched RGB composite for overlay backgrounds."""
    comp = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    for ci in range(min(3, arr.shape[2])):
        ch = arr[..., ci].astype(np.float32)
        lo, hi = np.percentile(ch, [1, 99.5])
        if hi > lo: comp[..., ci] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return (comp * 255).astype(np.uint8)

# Cellpose inference resolution cap — full-res images (4000+ px) on
# CPU can take 5+ minutes per image. Cap to this max-edge for
# segmentation, then upscale labels back to native size with nearest-
# neighbour. Intensity measurements still use the native-res image,
# so the only loss is sub-pixel cell boundary precision. Cellpose
# itself recommends this for "very large images".
CELLPOSE_MAX_EDGE = 1500

def _downsize_for_cellpose(a8_rgb):
    """Return (downsized_u8, scale_back_to_native) or (a8_rgb, 1.0)
    when no downsizing was needed."""
    h, w = a8_rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= CELLPOSE_MAX_EDGE:
        return a8_rgb, 1.0
    if not _have_scipy:
        return a8_rgb, 1.0
    scale = CELLPOSE_MAX_EDGE / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return _cv2.resize(a8_rgb, (new_w, new_h), interpolation=_cv2.INTER_AREA), scale

def _upsize_labels(lbl_arr, target_h, target_w):
    """Nearest-neighbour upscale of a label array (integer-valued)."""
    if lbl_arr.shape[0] == target_h and lbl_arr.shape[1] == target_w:
        return lbl_arr
    if not _have_scipy:
        return lbl_arr
    return _cv2.resize(lbl_arr.astype(np.int32), (target_w, target_h),
                       interpolation=_cv2.INTER_NEAREST).astype(np.int32)

def _cellpose_labels_batch(items):
    """BATCH cellpose over multiple images in ONE subprocess call so the
    ~100-MB cpsam model loads only ONCE (vs once per image when calling
    image-by-image — that was the dominant cost). "items" = list of
    (label, a8_rgb_native) tuples. Returns {label: labels_array_native_res}
    + per-image error map for the ones that failed.

    Each image is downsized to CELLPOSE_MAX_EDGE for inference, then
    its labels are NN-upscaled back to native resolution so stats are
    measured on the original pixels."""
    sent = []
    scales = {}    # label → scale factor applied (1.0 = no resize)
    sizes_native = {}  # label → (h, w) at native res
    for label, a8 in items:
        sizes_native[label] = a8.shape[:2]
        a8_small, sc = _downsize_for_cellpose(a8)
        scales[label] = sc
        sent.append({
            "kind": "image", "key": label, "label": label,
            "image_b64": _b64.b64encode(_png_bytes(a8_small)).decode(),
        })
    payload = json.dumps({
        "config": json.dumps({
            "model": cp_cfg.get("model") or "cpsam",
            "diameter": float(cp_cfg.get("diameter") or 0) or None,
            "min_size": int(cp_cfg.get("minSize") or 30),
            "channels": [{"r": 1, "g": 2, "b": 3}.get(cp_cfg.get("segChannel") or "b", 3), 0],
        }),
        "extra_inputs": sent,
        "sources": [], "timeout_sec": 600,
    })
    print(f"[intensity] cellpose: batching {len(sent)} image(s) into one model load "
          f"(cap {CELLPOSE_MAX_EDGE}px per edge, model={cp_cfg.get('model') or 'cpsam'})")
    try:
        req = _ur.Request("http://127.0.0.1:8765/api/analysis/run-cellpose",
                          data=payload.encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=600) as resp:
            cp_out = json.loads(resp.read().decode("utf-8"))
    except Exception as _e:
        # If the WHOLE batch call failed, every image errors with the
        # same message. The caller logs them per-image.
        return {}, {label: f"cellpose call failed: {_e}" for label, _ in items}
    if not cp_out.get("success"):
        msg = (cp_out.get("stderr") or "(no detail)").strip()
        return {}, {label: msg for label, _ in items}
    # Index returned images by name (each image emits <label>_labels).
    by_name = {im.get("name"): im.get("image") for im in (cp_out.get("images") or [])}
    labels_by_label = {}
    errors = {}
    for label, _ in items:
        lbl_b64 = by_name.get(f"{label}_labels")
        if not lbl_b64:
            errors[label] = "no labels image returned"
            continue
        try:
            arr = np.asarray(_Im.open(_io.BytesIO(_b64.b64decode(lbl_b64))).convert("L")).astype(np.int32)
            # Restore to native resolution for stat measurement.
            h_n, w_n = sizes_native[label]
            arr = _upsize_labels(arr, h_n, w_n)
            labels_by_label[label] = arr
        except Exception as _e:
            errors[label] = f"labels decode failed: {_e}"
    return labels_by_label, errors

def _png_bytes(arr_u8):
    buf = _io.BytesIO()
    _Im.fromarray(arr_u8).save(buf, format="PNG")
    return buf.getvalue()

rows = []

if mode == "cellpose":
    # ── Per-cell pipeline. ──
    # 1) Rolling-ball BG subtract every channel.
    # 2) BATCH cellpose call for ALL grouped images at once — the model
    #    only loads ONCE per run (vs once per image when we called
    #    image-by-image, which was the dominant runtime cost when N>1).
    # 3) Per cell: per-channel mean (raw + bg-corrected) → one row per
    #    (cell, channel).
    # 4) Save labeled-mask + outline-overlay PNG per source.
    n_images_with_cells = 0
    # First pass: prepare every grouped image's corrected channels +
    # the uint8 RGB Cellpose will see. Skipped (no group) images don't
    # go to Cellpose.
    prepared = []     # list of (key, src, label, grp, raw, corrected, a8)
    for key, src in imgs:
        label = _label_of(src, key)
        grp = img2group.get(label)
        if not grp:
            print(f"[intensity] {label}: not assigned to any group — skipping")
            continue
        raw = _pixels(src)
        corrected = np.zeros_like(raw, dtype=np.float64)
        for ci in range(3):
            bg = _rolling_bg(raw[..., ci], rolling_radius)
            c = raw[..., ci].astype(np.float64) - bg
            c[c < 0] = 0
            corrected[..., ci] = c
        a8 = np.clip(raw, 0, 255).astype(np.uint8)
        prepared.append((key, src, label, grp, raw, corrected, a8))
    if not prepared:
        raise SystemExit("[intensity] no images are in any group — assign images to groups in the picker.")
    # Second pass: ONE cellpose call for the whole batch.
    print(f"[intensity] cellpose: starting batch run on {len(prepared)} image(s) — "
          f"the model loads once (first run on a machine downloads ~100 MB)")
    _cp_t0 = __import__("time").monotonic()
    labels_by_label, cp_errors = _cellpose_labels_batch([(p[2], p[6]) for p in prepared])
    print(f"[intensity] cellpose: batch finished in {__import__('time').monotonic() - _cp_t0:.1f}s "
          f"({len(labels_by_label)} ok, {len(cp_errors)} failed)")
    # Third pass: per-image stats from the batched labels.
    for (_key, _src, label, grp, raw, corrected, _a8) in prepared:
        lbl_arr = labels_by_label.get(label)
        if lbl_arr is None or lbl_arr.shape[:2] != raw.shape[:2]:
            err = cp_errors.get(label) or "shape mismatch"
            print(f"[intensity] {label}: cellpose unusable ({err}) — skipping")
            continue
        # Drop tiny components.
        min_area = int(cp_cfg.get("minSize") or 30)
        labels = lbl_arr.astype(np.int32)
        if _have_scipy:
            sizes = np.bincount(labels.ravel())
            if sizes.size > 1:
                sizes[0] = 0
                keep = np.zeros_like(labels)
                next_id = 1
                for oid in range(1, sizes.size):
                    if sizes[oid] < min_area: continue
                    keep[labels == oid] = next_id
                    next_id += 1
                labels = keep
        cell_ids = [int(x) for x in np.unique(labels) if x != 0]
        if not cell_ids:
            print(f"[intensity] {label}: 0 cells above area {min_area} — skipped")
            continue
        n_images_with_cells += 1
        for cid in cell_ids:
            m = labels == cid
            area = int(m.sum())
            ys, xs = np.where(m)
            for ci, ck in enumerate(("R", "G", "B")):
                rv = raw[..., ci][m].astype(np.float64)
                cv = corrected[..., ci][m]
                rows.append({
                    "source": label,
                    "group": grp,
                    "channel": ch_name[ck],
                    "is_control": (control_name is not None and ch_name[ck] == control_name),
                    "object_id": int(cid),
                    "area_px": area,
                    "centroid_x": float(np.mean(xs)) if xs.size else 0.0,
                    "centroid_y": float(np.mean(ys)) if ys.size else 0.0,
                    "raw_mean": float(np.mean(rv)),
                    "raw_integrated_density": float(np.sum(rv)),
                    "background_corrected_mean": float(np.mean(cv)),
                    "background_corrected_integrated_density": float(np.sum(cv)),
                    "mean_intensity": float(np.mean(cv)),
                    "max_intensity": float(np.max(rv)),
                })
        # ── Per-image outputs: labeled-mask PNG + outline-overlay PNG. ──
        try:
            # 1) Labeled mask (raw labels as 16-bit greyscale → palette-mapped
            #    so each cell is a unique colour, easy to inspect).
            n = max(1, int(labels.max()))
            palette = np.zeros((n + 1, 3), dtype=np.uint8)
            rng = np.random.default_rng(42)
            palette[1:] = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
            paletted = palette[labels]
            mpfig_image(_Im.fromarray(paletted), name=f"{label}_mask")
            # 2) Outline overlay (yellow cell edges on the composite + ids).
            #    Symmetric 4-neighbour transitions so the outline sits on
            #    the cell edge, not 1 px above it.
            comp = _composite_u8(raw)
            b_up    = np.zeros(labels.shape, dtype=bool)
            b_down  = np.zeros(labels.shape, dtype=bool)
            b_left  = np.zeros(labels.shape, dtype=bool)
            b_right = np.zeros(labels.shape, dtype=bool)
            b_up[1:, :]     = labels[1:, :]    != labels[:-1, :]
            b_down[:-1, :]  = labels[:-1, :]   != labels[1:, :]
            b_left[:, 1:]   = labels[:, 1:]    != labels[:, :-1]
            b_right[:, :-1] = labels[:, :-1]   != labels[:, 1:]
            boundaries = (b_up | b_down | b_left | b_right) & (labels > 0)
            comp[boundaries] = (255, 255, 0)
            overlay = _Im.fromarray(comp)
            d = _ImD.Draw(overlay)
            for cid in cell_ids[:300]:
                ys, xs = np.where(labels == cid)
                if xs.size == 0: continue
                d.text((float(np.mean(xs)), float(np.mean(ys))), str(cid), fill=(255, 255, 255))
            mpfig_image(overlay, name=f"{label}_overlay")
        except Exception as _e:
            print(f"[intensity] {label}: overlay/mask render failed: {_e}", file=_sys.stderr)
        print(f"[intensity] {label}: {len(cell_ids)} cell(s) measured (cellpose)")
    if not rows:
        raise SystemExit("[intensity] cellpose produced no measurable cells across the inputs — "
                         "check the picker preview, then re-run.")
    print(f"computed per-cell intensities across {n_images_with_cells} image(s)")
    mpfig_data(rows, name="fluor_intensities")
    raise SystemExit(0)

# ── Simple strategy: per-channel rolling-ball + threshold. ──
# One row per (image, channel): mean intensity inside the channel's mask.
# Disabled channels emit no row (they don't contribute to the plot).
# Per-image PNG output: a composite where each enabled channel's mask
# boundary is drawn in that channel's colour (matches the dialog preview).
ch_colors = {"r": (255, 64, 64), "g": (96, 220, 96), "b": (96, 160, 255)}
for key, src in imgs:
    label = _label_of(src, key)
    grp = img2group.get(label, label)  # unassigned → label as own group (one bar)
    raw = _pixels(src)
    corrected = np.zeros_like(raw, dtype=np.float64)
    masks = {}
    for ci, ck in enumerate(("R", "G", "B")):
        key_lc = ck.lower()
        spec = thresholds.get(key_lc) or {}
        enabled = bool(spec.get("enabled", True))
        if not enabled:
            corrected[..., ci] = raw[..., ci].astype(np.float64)
            continue
        bg = _rolling_bg(raw[..., ci], rolling_radius)
        corr = raw[..., ci].astype(np.float64) - bg
        corr[corr < 0] = 0
        # IMPORTANT: corrected[..., ci] stores the BG-corrected intensity
        # used for the per-mask MEAN measurement — NOT the DoG response.
        # We want intensity stats to reflect actual fluorescence, not a
        # band-pass artefact. So we keep "corrected" here and build a
        # separate "thresh_in" for the threshold step below.
        corrected[..., ci] = corr
        thresh_in = corr
        # Optional Difference-of-Gaussians band-pass: better separation
        # for densely-packed cells / soft-edged signal. The threshold is
        # applied to the DoG response; intensities still come from "corr".
        if bool(spec.get("enhanceEdges", False)) and _have_scipy:
            g_small = _cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 1.0)
            g_large = _cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 20.0)
            thresh_in = (g_small - g_large).astype(np.float64)
            thresh_in[thresh_in < 0] = 0
        try:
            mask = _threshold_mask(
                thresh_in,
                str(spec.get("thresholdMethod", "percentile")),
                float(spec.get("thresholdPercentile", 95)),
            )
        except Exception as _e:
            print(f"[intensity] {label} ch {ck}: threshold failed: {_e}", file=_sys.stderr)
            continue
        min_area = int(spec.get("minArea", 30) or 0)
        if min_area > 0 and _have_scipy:
            labels_c, _n = _ndi.label(mask)
            sizes = np.bincount(labels_c.ravel())
            if sizes.size > 1:
                sizes[0] = 0
                keep_ids = np.where(sizes >= min_area)[0]
                mask = np.isin(labels_c, keep_ids)
        masks[key_lc] = mask
        n_pixels = int(mask.sum())
        if n_pixels == 0:
            continue
        rv = raw[..., ci][mask].astype(np.float64)
        cv = corrected[..., ci][mask]
        rows.append({
            "source": label,
            "group": grp,
            "channel": ch_name[ck],
            "is_control": (control_name is not None and ch_name[ck] == control_name),
            "n_pixels": n_pixels,
            "raw_mean": float(np.mean(rv)),
            "raw_integrated_density": float(np.sum(rv)),
            "background_corrected_mean": float(np.mean(cv)),
            "background_corrected_integrated_density": float(np.sum(cv)),
            "mean_intensity": float(np.mean(cv)),
            "max_intensity": float(np.max(rv)),
        })
    # ── Per-image PNG outputs: mask composite + outline overlay. ──
    try:
        comp = _composite_u8(raw)
        out = comp.copy()
        mask_rgb = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.uint8)
        for k_lc, color in ch_colors.items():
            m = masks.get(k_lc)
            if m is None: continue
            # Fill the mask (semi-transparent feel via 50% blend with composite).
            for ci, cv in enumerate(color):
                mask_rgb[..., ci] = np.maximum(mask_rgb[..., ci], (m * cv).astype(np.uint8))
            # SYMMETRIC 2-px boundary: dilation(mask) AND NOT erosion(mask).
            # The earlier slice-diff form biased the outline 1 px upward
            # (and 1 px leftward), so the saved overlay didn't quite sit
            # on the actual signal. This matches the picker preview now.
            if _have_scipy:
                boundary = _ndi.binary_dilation(m, iterations=1) & ~_ndi.binary_erosion(m, iterations=1)
            else:
                boundary = np.zeros(m.shape, dtype=bool)
                boundary[:-1, :] |= m[:-1, :] != m[1:, :]
                boundary[:, :-1] |= m[:, :-1] != m[:, 1:]
                boundary[1:, :]  |= m[1:, :]  != m[:-1, :]
                boundary[:, 1:]  |= m[:, 1:]  != m[:, :-1]
            out[boundary] = color
        mpfig_image(_Im.fromarray(mask_rgb), name=f"{label}_mask")
        mpfig_image(_Im.fromarray(out), name=f"{label}_overlay")
    except Exception as _e:
        print(f"[intensity] {label}: mask render failed: {_e}", file=_sys.stderr)
    print(f"[intensity] {label}: thresholded {sum(1 for v in masks.values() if v.any())} channel(s)")

if not rows:
    raise SystemExit("[intensity] no channels above threshold across the inputs — "
                     "tweak the picker thresholds and re-run.")
print(f"computed channel intensities for {len(imgs)} image(s); "
      f"groups = {sorted({r['group'] for r in rows})}; "
      f"control = {control_name or '<none>'}")
mpfig_data(rows, name="fluor_intensities")
`;
}

// ── Dialog ────────────────────────────────────────────────────

interface IntensityPickerDialogProps {
  open: boolean;
  /** Resolved upstream image inputs — label, thumbnail base64, optional
   *  source descriptor for full-res re-extract. */
  images: FluorPickerImage[];
  initial: FluorIntensityConfig | null;
  onClose: () => void;
  onSave: (cfg: FluorIntensityConfig) => void;
}

const CHANNEL_SUGGESTIONS = [
  "DAPI", "Hoechst", "GFP", "RFP", "YFP", "CFP",
  "mCherry", "FITC", "TRITC", "AF488", "AF555", "AF594", "AF647", "AF680",
  "Anti-VegF", "Anti-CD31", "Anti-Ki67", "α-SMA",
];

/** Compact MenuProps for all the Select dropdowns in this dialog.
 *  MUI's default popup uses the theme's normal body font (~14-16 px),
 *  which towers over the dialog's 0.78 rem fields and feels off-key
 *  ("cpsam (default)" looked huge). This shrinks the popup paper +
 *  every MenuItem to match the surrounding fields. */
const SELECT_MENU_PROPS = {
  MenuProps: {
    PaperProps: {
      sx: {
        "& .MuiMenuItem-root": { fontSize: "0.78rem", py: 0.4, minHeight: 28 },
      },
    },
  },
} as const;

/** A computed preview for one image: the base composite + a per-channel
 *  boundary overlay (each on transparent background, so the dialog can
 *  layer + toggle visibility live without re-running). For Cellpose
 *  there's a single fused `overlaySrc` and no per-channel maps. */
type PreviewLayers = {
  /** Combined / fused overlay (legacy fallback; cellpose result). */
  overlaySrc?: string;
  /** Base layer — contrast-stretched composite (simple strategy only). */
  compositeSrc?: string;
  /** Per-channel boundary on transparent bg (simple strategy only). */
  channelOverlays?: Partial<Record<"r" | "g" | "b", string>>;
  n_cells?: number;
  per_channel?: Record<string, number>;
};

export default function IntensityPickerDialog(props: IntensityPickerDialogProps) {
  const { open, images, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<FluorIntensityConfig>(initial ? migrateConfig(initial) : emptyFluorConfig());
  const [activeIdx, setActiveIdx] = useState(0);
  // Per-image preview cache: maps imageLabel → the rendered layers from
  // the most recent successful Run. Switching back to a previously-
  // previewed image restores its overlay INSTANTLY (no re-fetch); the
  // cache is invalidated whenever a preview-relevant param changes.
  const [previewByImage, setPreviewByImage] = useState<Record<string, PreviewLayers>>({});
  const [previewLoading, setPreviewLoading] = useState(false);
  // Elapsed seconds counter ticking while a preview is running. Drives
  // the Run-button label and an HUD chip so the user gets continuous
  // feedback (cellpose first-run can take 30-60 s on CPU before any
  // visible signal of progress; without this it feels stuck).
  const [elapsedSec, setElapsedSec] = useState(0);
  useEffect(() => {
    if (!previewLoading) { setElapsedSec(0); return; }
    const t0 = Date.now();
    const id = setInterval(() => setElapsedSec(Math.round((Date.now() - t0) / 1000)), 250);
    return () => clearInterval(id);
  }, [previewLoading]);
  const [previewError, setPreviewError] = useState<string | null>(null);
  // Per-channel mask visibility in the preview overlay. Independent of
  // the per-channel "enabled" flag (which controls inclusion in
  // quantification) — this is purely a display toggle and never gates
  // the actual analysis. Defaults: all visible.
  const [maskVisible, setMaskVisible] = useState<Record<"r" | "g" | "b", boolean>>({ r: true, g: true, b: true });
  // Sidecar build identifier — fetched once on dialog open so we can
  // surface it next to the title. Lets the user (and bug reports) tell
  // at a glance which python-sidecar binary is actually running.
  const [sidecarBuild, setSidecarBuild] = useState<string | null>(null);
  // Warmup PROMISE so the first Run preview can await it. The race
  // between fire-and-forget warmup + first Run is what was producing
  // the "load failed" error: fetch returning before cv2/scipy finished
  // loading caused the socket to drop or the response to be malformed.
  const warmupRef = useRef<Promise<void> | null>(null);
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    fetch("http://127.0.0.1:8765/api/version")
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (!cancelled && d?.build) setSidecarBuild(String(d.build)); })
      .catch(() => { /* not critical — leave null */ });
    // Warmup — triggers cv2 / scipy / PIL imports in the frozen sidecar
    // so the user's first Run preview click doesn't pay the cold-start
    // cost (which would otherwise blow past the simple-mode timeout
    // AND can manifest as "Failed to fetch" if the socket drops mid-
    // import on macOS). We keep the promise so fetchPreview can await.
    warmupRef.current = fetch("http://127.0.0.1:8765/api/analysis/warmup", { method: "POST" })
      .then(() => undefined)
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [open]);
  // Repair flow for a broken Cellpose plugin venv (missing packaging,
  // missing setuptools, partial install, etc.). Streams the install log
  // into a textarea below the preview pane so the user can see progress.
  const [repairing, setRepairing] = useState(false);
  const [repairLog, setRepairLog] = useState<string>("");
  // Show the repair affordance when the error looks like a plugin-env
  // problem (the 'No module named …' / "Cellpose isn't installed" pattern).
  const showRepair = !!previewError && (
    /No module named/.test(previewError)
    || /not installed/i.test(previewError)
    || /Install Cellpose/i.test(previewError)
    || /plugin (?:venv|environment)/i.test(previewError)
  );
  const runCellposeRepair = useCallback(async () => {
    if (repairing) return;
    setRepairing(true);
    setRepairLog("");
    try {
      const resp = await fetch("http://127.0.0.1:8765/api/analysis/install-cellpose-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!resp.ok || !resp.body) {
        setRepairLog(`HTTP ${resp.status}: failed to start install`);
        return;
      }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames: split on double newline, parse `data:` lines.
        let nl;
        while ((nl = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, nl);
          buf = buf.slice(nl + 2);
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            try {
              const evt = JSON.parse(payload) as { message?: string; done?: boolean; returncode?: number };
              if (evt.message) setRepairLog((p) => p + evt.message + "\n");
              if (evt.done) {
                setRepairLog((p) => p + `\n${evt.returncode === 0 ? "✔ install finished" : `✘ install exited with ${evt.returncode}`}\n`);
              }
            } catch {
              setRepairLog((p) => p + payload + "\n");
            }
          }
        }
      }
    } catch (e: unknown) {
      setRepairLog((p) => p + `\nerror: ${String((e as { message?: string })?.message ?? e)}\n`);
    } finally {
      setRepairing(false);
    }
  }, [repairing]);

  // Reset on (re-)open. Clear active idx + preview cache so we don't
  // show stale results from a previous open.
  useEffect(() => {
    if (open) {
      setCfg(initial ? migrateConfig(structuredClone(initial)) : emptyFluorConfig());
      setActiveIdx(0);
      setPreviewByImage({});
      setPreviewError(null);
      setMaskVisible({ r: true, g: true, b: true });
    }
  }, [open, initial]);

  const activeImage = images[activeIdx] || null;

  // Distinct images currently upstream — chips for group assignment.
  // Pull from props.images (the full source set) so the group panel
  // matches what the cycler is showing.
  const imageLabels = useMemo(
    () => Array.from(new Set(images.map((i) => i.label.trim()))).filter(Boolean),
    [images],
  );
  // Per-image group lookup (last group wins if duplicated; the UI prevents that).
  const imgToGroup = useMemo(() => {
    const m = new Map<string, string>();
    for (const g of cfg.groups) for (const im of g.images) m.set(im, g.name);
    return m;
  }, [cfg.groups]);

  // ── Preview fetcher ──
  // The preview ONLY runs when the user explicitly clicks "Run preview"
  // (or when they cycle to a new image — a deliberate user action that
  // does want a fresh overlay). Auto-refresh on every slider drag is
  // disabled: Cellpose is heavy enough that an accidental flurry of
  // requests stalls the sidecar, and even threshold tweaks deserve a
  // chance to settle before the user commits to a re-fetch.
  //
  // `paramsDirty` tracks whether any preview-relevant param has changed
  // since the last successful preview, so the Run button can pulse to
  // tell the user "your tweaks aren't reflected in the overlay yet".
  const [paramsDirty, setParamsDirty] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fetchPreview = useCallback(async () => {
    if (!open || !activeImage) return;
    if (abortRef.current) abortRef.current.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setPreviewLoading(true);
    setPreviewError(null);
    // Wait for the warmup ping to complete (or fail) before issuing
    // the preview call. Fixes the "load failed" error on the FIRST
    // Run when cv2/scipy were still loading in the sidecar.
    if (warmupRef.current) {
      try { await warmupRef.current; } catch { /* swallow */ }
      if (ac.signal.aborted) return;
    }
    try {
      const body: Record<string, unknown> = {
        image_b64: activeImage.image_b64 || "",
        source: activeImage.source
          ? {
              key: activeImage.source.key,
              row: activeImage.source.row,
              col: activeImage.source.col,
              inset_index: activeImage.source.inset_index,
              name: activeImage.source.name,
            }
          : undefined,
        strategy: cfg.mode,
        rolling_radius: cfg.rollingRadius,
        preview_max_w: 1024,
      };
      if (cfg.mode === "simple") {
        const chBody = (k: "r" | "g" | "b") => ({
          enabled: cfg.thresholds[k].enabled,
          threshold_method: cfg.thresholds[k].thresholdMethod,
          threshold_percentile: cfg.thresholds[k].thresholdPercentile,
          min_area: cfg.thresholds[k].minArea,
          enhance_edges: !!cfg.thresholds[k].enhanceEdges,
        });
        body.channels = { r: chBody("r"), g: chBody("g"), b: chBody("b") };
      } else {
        body.cellpose = {
          model: cfg.cellpose.model,
          diameter: cfg.cellpose.diameter,
          seg_channel: cfg.cellpose.segChannel,
          min_size: cfg.cellpose.minSize,
        };
      }
      // Hard client-side timeout so the spinner can't get stuck forever
      // when the sidecar is offline or the strategy is mis-configured.
      // Cellpose: 10 min (matches the backend's plugin call), so the
      // first run can finish downloading the ~100 MB model. Threshold:
      // 60 s — the FIRST simple-strategy call lazy-loads cv2 + scipy +
      // PIL in the frozen sidecar (can take 10-30 s on a cold start),
      // and the dialog's warmup ping at mount time helps but isn't
      // guaranteed to finish before the user clicks Run.
      const timeoutMs = cfg.mode === "cellpose" ? 600000 : 60000;
      const timeoutId = setTimeout(() => {
        // Tag the controller so the catch block can distinguish a
        // timeout-driven abort from a user-driven one (e.g. re-clicking
        // Run cancels the in-flight request via abortRef.current.abort()).
        (ac as unknown as { __timedOut?: boolean }).__timedOut = true;
        ac.abort();
      }, timeoutMs);
      // Single retry on TypeError: Failed to fetch — that's macOS Safari /
      // Chromium's generic name for a socket-level error, which can hit if
      // the sidecar is doing a heavy import (cv2/scipy) right as we POST.
      // The retry waits 600 ms and tries once more; if it still fails we
      // surface the real error.
      const doFetch = () => fetch("http://127.0.0.1:8765/api/analysis/fluor-preview-segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      let resp: Response;
      try {
        try {
          resp = await doFetch();
        } catch (firstErr: unknown) {
          const msg = String((firstErr as { message?: string })?.message ?? firstErr);
          if (/Failed to fetch|Load failed|NetworkError/i.test(msg) && !ac.signal.aborted) {
            await new Promise((r) => setTimeout(r, 600));
            resp = await doFetch();
          } else {
            throw firstErr;
          }
        }
      } finally {
        clearTimeout(timeoutId);
      }
      const data = await resp.json();
      if (ac.signal.aborted) return;
      if (data.success && (data.overlay_b64 || data.composite_b64)) {
        const layers: PreviewLayers = {
          overlaySrc: data.overlay_b64 ? `data:image/png;base64,${data.overlay_b64}` : undefined,
          compositeSrc: data.composite_b64 ? `data:image/png;base64,${data.composite_b64}` : undefined,
          channelOverlays: (() => {
            const co = (data.channel_overlays || {}) as Record<string, string>;
            const out: Partial<Record<"r" | "g" | "b", string>> = {};
            for (const k of ["r", "g", "b"] as const) {
              if (co[k]) out[k] = `data:image/png;base64,${co[k]}`;
            }
            return out;
          })(),
          n_cells: typeof data.n_cells === "number" ? data.n_cells : undefined,
          per_channel: data.per_channel,
        };
        // Cache by the ACTIVE image's label so cycling through images
        // restores their overlays instantly.
        const key = activeImage.label;
        setPreviewByImage((cur) => ({ ...cur, [key]: layers }));
        setParamsDirty(false);
      } else {
        setPreviewError(data.error || "preview failed");
      }
    } catch (e: unknown) {
      const isAbort = (e as { name?: string })?.name === "AbortError";
      if (isAbort) {
        // Distinguish a timeout (we tagged the controller) from a
        // user-driven cancel (e.g. they clicked Run again to retry).
        if ((ac as unknown as { __timedOut?: boolean }).__timedOut) {
          setPreviewError(cfg.mode === "cellpose"
            ? "Cellpose preview timed out (>10 min). The first run downloads the model (~100 MB) — check your internet, then try again, or verify the plugin venv with 'Repair plugin venv'."
            : "Preview timed out (>60 s). The sidecar may be busy or still loading scientific libraries on a cold start — try again.");
        }
        return;
      }
      setPreviewError(String((e as { message?: string })?.message ?? e));
    } finally {
      setPreviewLoading(false);
    }
  }, [open, activeImage, cfg.mode, cfg.rollingRadius, cfg.thresholds, cfg.cellpose]);

  // Cycling images: clear the inline error and dirty-state, but keep
  // the cache so we can restore THIS image's previous overlay below.
  useEffect(() => {
    if (!open) return;
    setPreviewError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIdx]);

  // Param changes invalidate the whole cache — every previously-rendered
  // image's overlay is now stale relative to the current settings.
  useEffect(() => {
    if (!open) return;
    setPreviewByImage({});
    setParamsDirty(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg.mode, cfg.rollingRadius, cfg.thresholds, cfg.cellpose]);

  // The active image's cached preview (or undefined if it hasn't been
  // Run yet, or the cache was invalidated by a param change).
  const activePreview = activeImage ? previewByImage[activeImage.label] : undefined;

  // ── Canvas-composited preview ─────────────────────────────
  // We composite the base composite + each visible channel boundary at
  // the PNG's NATIVE resolution onto a single <canvas>. CSS only scales
  // the final canvas (one element, one scale), so the per-channel
  // contours can't drift relative to the underlying composite — they
  // share the same scaling kernel because they're already burned into
  // the same pixel grid before the browser sees them. Earlier two-
  // <img> layout had subpixel drift between RGB (composite) and RGBA
  // (overlay) imgs even with identical CSS — RGB and RGBA can use
  // different interpolation kernels at scale.
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewBoxRef = useRef<HTMLDivElement>(null);
  const [compImgEl, setCompImgEl] = useState<HTMLImageElement | null>(null);
  const [chImgEls, setChImgEls] = useState<Partial<Record<"r" | "g" | "b", HTMLImageElement>>>({});
  // View state for the preview canvas: zoom + pan. zoom = 1 → fit to
  // the preview pane. Wheel zooms around the cursor (Photoshop-style);
  // drag pans; double-click resets.
  const [view, setView] = useState<{ zoom: number; panX: number; panY: number }>({ zoom: 1, panX: 0, panY: 0 });
  const resetView = useCallback(() => setView({ zoom: 1, panX: 0, panY: 0 }), []);
  // Reset on image cycle so each new image opens at fit.
  useEffect(() => { resetView(); }, [activeIdx, resetView]);
  // Wheel = zoom around cursor; preventDefault so the page doesn't scroll.
  const onPreviewWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setView((v) => {
      const newZoom = Math.max(0.2, Math.min(20, v.zoom * factor));
      // Zoom around the cursor: keep cursor's image-space point pinned.
      const box = previewBoxRef.current?.getBoundingClientRect();
      if (!box) return { ...v, zoom: newZoom };
      const cx = e.clientX - box.left - box.width / 2;
      const cy = e.clientY - box.top - box.height / 2;
      const realFactor = newZoom / v.zoom;
      return {
        zoom: newZoom,
        panX: cx - realFactor * (cx - v.panX),
        panY: cy - realFactor * (cy - v.panY),
      };
    });
  }, []);
  // Left-drag pans. Track via window listeners so dragging off-canvas works.
  const onPreviewMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX, startY = e.clientY;
    const startPan = view;
    const onMove = (ev: MouseEvent) => {
      setView((v) => ({ ...v, panX: startPan.panX + (ev.clientX - startX), panY: startPan.panY + (ev.clientY - startY) }));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [view]);

  // Load composite + channel overlays as HTMLImageElement instances
  // whenever the active preview changes. We hold them in state so
  // visibility toggles can re-composite without re-loading.
  useEffect(() => {
    if (!activePreview || cfg.mode !== "simple") {
      setCompImgEl(null);
      setChImgEls({});
      return;
    }
    let cancelled = false;
    const compSrc = activePreview.compositeSrc || activePreview.overlaySrc;
    if (compSrc) {
      const im = new window.Image();
      im.onload = () => { if (!cancelled) setCompImgEl(im); };
      im.src = compSrc;
    } else {
      setCompImgEl(null);
    }
    setChImgEls({});
    for (const k of ["r", "g", "b"] as const) {
      const src = activePreview.channelOverlays?.[k];
      if (!src) continue;
      const im = new window.Image();
      im.onload = () => {
        if (cancelled) return;
        setChImgEls((cur) => ({ ...cur, [k]: im }));
      };
      im.src = src;
    }
    return () => { cancelled = true; };
  }, [activePreview, cfg.mode]);

  // Redraw the canvas whenever the composite, any overlay, or the
  // visibility map changes. The redraw runs at the PNG's native
  // resolution — CSS scales the result for display, but the layers
  // themselves are pixel-perfect aligned.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!compImgEl) {
      // Reset to a tiny transparent canvas so the previous frame
      // doesn't linger after switching images.
      canvas.width = 1;
      canvas.height = 1;
      return;
    }
    const w = compImgEl.naturalWidth;
    const h = compImgEl.naturalHeight;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(compImgEl, 0, 0);
    for (const k of ["r", "g", "b"] as const) {
      if (!maskVisible[k]) continue;
      const im = chImgEls[k];
      if (im) ctx.drawImage(im, 0, 0);
    }
  }, [compImgEl, chImgEls, maskVisible]);

  const setChannel = useCallback((k: keyof FluorChannels, v: string) => {
    setCfg((c) => ({ ...c, channels: { ...c.channels, [k]: v } }));
  }, []);

  const setThreshold = useCallback((k: "r" | "g" | "b", patch: Partial<FluorChannelThreshold>) => {
    setCfg((c) => ({
      ...c,
      thresholds: { ...c.thresholds, [k]: { ...c.thresholds[k], ...patch } },
    }));
  }, []);

  const setCellpose = useCallback((patch: Partial<FluorCellpose>) => {
    setCfg((c) => ({ ...c, cellpose: { ...c.cellpose, ...patch } }));
  }, []);

  const addGroup = useCallback(() => {
    const used = new Set(cfg.groups.map((g) => g.name));
    let n = 1; while (used.has(`Group ${n}`)) n++;
    setCfg((c) => ({ ...c, groups: [...c.groups, { id: uid("grp"), name: `Group ${n}`, images: [] }] }));
  }, [cfg.groups]);
  const renameGroup = useCallback((id: string, name: string) => {
    setCfg((c) => ({ ...c, groups: c.groups.map((g) => g.id === id ? { ...g, name } : g) }));
  }, []);
  const deleteGroup = useCallback((id: string) => {
    setCfg((c) => ({ ...c, groups: c.groups.filter((g) => g.id !== id) }));
  }, []);
  const toggleImageInGroup = useCallback((gid: string, label: string) => {
    setCfg((c) => ({
      ...c,
      groups: c.groups.map((g) => {
        if (g.id === gid) {
          const has = g.images.includes(label);
          return { ...g, images: has ? g.images.filter((x) => x !== label) : [...g.images, label] };
        }
        // Enforce single membership — remove from other groups when adding here.
        return { ...g, images: g.images.filter((x) => x !== label) };
      }),
    }));
  }, []);

  const prevImage = () => setActiveIdx((i) => (images.length ? (i - 1 + images.length) % images.length : 0));
  const nextImage = () => setActiveIdx((i) => (images.length ? (i + 1) % images.length : 0));

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem", py: 1.25 }}>
        🌈 Fluorescence segmentation picker
        <Typography component="span" variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
          Cycle inputs · choose strategy · tweak with live preview · then quantify
        </Typography>
        {sidecarBuild && (
          <Typography component="span" variant="caption"
            sx={{
              ml: 1, px: 0.6, py: 0.1, borderRadius: 0.5,
              bgcolor: "action.hover", color: "text.disabled", fontSize: "0.6rem",
            }}>
            sidecar {sidecarBuild}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers sx={{ display: "flex", flexDirection: "row", gap: 1.5, py: 1.5, minHeight: 720 }}>
        {/* ── Left: image cycler + preview pane ─────────────────── */}
        <Box sx={{ flex: 1.4, minWidth: 360, display: "flex", flexDirection: "column", gap: 1 }}>
          {/* Cycler header */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Tooltip title="Previous image">
              <span>
                <IconButton size="small" onClick={prevImage} disabled={images.length <= 1}>
                  <ChevronLeftIcon />
                </IconButton>
              </span>
            </Tooltip>
            <Box sx={{ flex: 1, textAlign: "center" }}>
              {images.length === 0 ? (
                <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic" }}>
                  No upstream images wired
                </Typography>
              ) : (
                <Typography variant="caption" sx={{ fontWeight: 700 }}>
                  {activeImage?.label || "—"}{" "}
                  <Typography component="span" variant="caption" sx={{ color: "text.disabled" }}>
                    ({activeIdx + 1}/{images.length})
                  </Typography>
                </Typography>
              )}
            </Box>
            <Tooltip title="Next image">
              <span>
                <IconButton size="small" onClick={nextImage} disabled={images.length <= 1}>
                  <ChevronRightIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
          {/* The Run strategy button is the ONLY trigger for the
              segmentation preview — by design (see the useEffects above).
              It pulses (contained colour) whenever the current overlay
              doesn't reflect the latest settings, so the user is never
              left looking at a stale preview without knowing. */}
          <Button
            variant={paramsDirty ? "contained" : "outlined"}
            color={paramsDirty ? "primary" : "inherit"}
            startIcon={previewLoading
              ? <CircularProgress size={14} color="inherit" />
              : <RefreshIcon sx={{ fontSize: 18 }} />}
            onClick={() => void fetchPreview()}
            disabled={!activeImage || previewLoading}
            sx={{ textTransform: "none", fontWeight: 700, py: 0.5 }}
          >
            {previewLoading
              ? (cfg.mode === "cellpose"
                  ? `Running Cellpose… ${elapsedSec}s${elapsedSec < 10 ? " (loading model)" : elapsedSec < 30 ? " (inference)" : ""}`
                  : `Running… ${elapsedSec}s`)
              : (paramsDirty
                  ? `Run ${cfg.mode === "cellpose" ? "Cellpose" : "threshold"} preview`
                  : "Re-run preview")}
          </Button>
          {/* Preview canvas. For simple strategy we LAYER the response:
              the base composite goes at the bottom and each enabled
              channel's transparent boundary PNG stacks on top with a
              CSS visibility toggle — so the user can show/hide masks
              instantly without re-running. For Cellpose we just show
              the fused overlay PNG. */}
          <Box sx={{
            position: "relative", flex: 1,
            border: "1px solid", borderColor: "divider", borderRadius: 1,
            bgcolor: "#0a0a0a",
            overflow: "hidden",
            display: "flex", alignItems: "center", justifyContent: "center",
            minHeight: 560,
          }}>
            {/* Pan/zoom container — wraps the canvas (or fused-overlay
                <img> for cellpose) and a fallback thumbnail. Mouse-wheel
                zooms around the cursor, click-drag pans, double-click
                resets. Children center in the box and are transformed
                via CSS — the canvas itself keeps its native dimensions. */}
            <Box
              ref={previewBoxRef}
              onWheel={onPreviewWheel}
              onMouseDown={onPreviewMouseDown}
              onDoubleClick={resetView}
              sx={{
                position: "absolute", inset: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: previewLoading ? "wait" : "grab",
                "&:active": { cursor: previewLoading ? "wait" : "grabbing" },
                overflow: "hidden",
              }}
            >
              <Box sx={{
                transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`,
                transformOrigin: "center center",
                transition: "transform 50ms linear",
                display: "flex", alignItems: "center", justifyContent: "center",
                maxWidth: "100%", maxHeight: "100%",
              }}>
                {(() => {
                  const ap = activePreview;
                  const isSimple = cfg.mode === "simple";
                  if (ap && isSimple && (ap.compositeSrc || ap.overlaySrc)) {
                    return (
                      <canvas
                        ref={canvasRef}
                        style={{
                          display: "block",
                          maxWidth: "100%",
                          maxHeight: "calc(100vh - 280px)",
                          verticalAlign: "top",
                          imageRendering: "auto",
                        }}
                      />
                    );
                  }
                  if (ap?.overlaySrc) {
                    return (
                      <img
                        src={ap.overlaySrc}
                        alt="Segmentation preview"
                        style={{ display: "block", maxWidth: "100%", maxHeight: "calc(100vh - 280px)", objectFit: "contain" }}
                      />
                    );
                  }
                  if (activeImage?.image_b64) {
                    return (
                      <img
                        src={`data:image/png;base64,${activeImage.image_b64}`}
                        alt={activeImage.label}
                        style={{ display: "block", maxWidth: "100%", maxHeight: "calc(100vh - 280px)", objectFit: "contain", opacity: 0.65 }}
                      />
                    );
                  }
                  return (
                    <Typography variant="caption" sx={{ color: "text.disabled" }}>
                      {images.length === 0 ? "Wire upstream image sources first." : "(no preview — click Run)"}
                    </Typography>
                  );
                })()}
              </Box>
            </Box>
            {/* Pan/zoom HUD: reset button + zoom level. Floated to the
                top-right so it doesn't fight the error banner at bottom. */}
            <Box sx={{
              position: "absolute", top: 4, right: 4,
              display: "flex", alignItems: "center", gap: 0.5,
              bgcolor: "rgba(0,0,0,0.45)", color: "common.white",
              px: 0.6, py: 0.2, borderRadius: 0.5, fontSize: "0.66rem",
              pointerEvents: "auto",
            }}>
              <Typography variant="caption" sx={{ fontSize: "0.65rem", opacity: 0.85 }}>
                {Math.round(view.zoom * 100)}%
              </Typography>
              <Button size="small" onClick={resetView}
                sx={{ textTransform: "none", fontSize: "0.62rem", color: "common.white", py: 0.05, px: 0.5, minWidth: 0 }}>
                Reset
              </Button>
            </Box>
            {previewLoading && (
              <Box sx={{
                position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                bgcolor: "rgba(0,0,0,0.35)", pointerEvents: "none",
              }}>
                <CircularProgress size={28} />
              </Box>
            )}
            {previewError && (
              <Box sx={{
                position: "absolute", bottom: 4, left: 4, right: 4,
                px: 0.8, py: 0.5, borderRadius: 0.5,
                bgcolor: "rgba(180,40,40,0.92)", color: "common.white",
                fontSize: "0.7rem", display: "flex", flexDirection: "column", gap: 0.5,
              }}>
                <Box>⚠ {previewError}</Box>
                {showRepair && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                    <Typography variant="caption" sx={{ color: "common.white", opacity: 0.85, fontSize: "0.66rem" }}>
                      The plugin venv looks broken. Click to repair (pip install — 5-15 min, ~500 MB).
                    </Typography>
                    <Button size="small" variant="contained" color="warning"
                      onClick={() => void runCellposeRepair()}
                      disabled={repairing}
                      sx={{ textTransform: "none", fontWeight: 700, fontSize: "0.7rem", py: 0.2, px: 1, ml: "auto" }}>
                      {repairing ? "Repairing…" : "Repair plugin venv"}
                    </Button>
                  </Box>
                )}
              </Box>
            )}
            {/* Live install log when repair is running. Anchored at the
                top so it doesn't fight the error banner at the bottom. */}
            {(repairing || repairLog) && (
              <Box sx={{
                position: "absolute", top: 4, left: 4, right: 4, maxHeight: "55%",
                px: 0.8, py: 0.4, borderRadius: 0.5,
                bgcolor: "rgba(0,0,0,0.85)", color: "#cfe", fontFamily: "monospace",
                fontSize: "0.62rem", overflow: "auto", whiteSpace: "pre-wrap",
              }}>
                {repairLog || "starting…"}
              </Box>
            )}
            {/* Stats footer (cell count or per-channel pixel tallies). */}
            {!previewError && activePreview && (
              <Box sx={{
                position: "absolute", bottom: 4, left: 4, right: 4,
                px: 0.8, py: 0.3, borderRadius: 0.5,
                bgcolor: "rgba(0,0,0,0.55)", color: "common.white",
                fontSize: "0.66rem", display: "flex", gap: 1, justifyContent: "center",
              }}>
                {typeof activePreview.n_cells === "number" && (
                  <span>cells: <b>{activePreview.n_cells}</b></span>
                )}
                {activePreview.per_channel && (["r", "g", "b"] as const).map((k) => {
                  const sw = k === "r" ? "#ff8080" : k === "g" ? "#88e088" : "#88a8ff";
                  const n = activePreview.per_channel?.[k] ?? 0;
                  if (!cfg.thresholds[k].enabled) return null;
                  return (
                    <span key={k} style={{ color: sw, opacity: maskVisible[k] ? 1 : 0.4 }}>
                      {cfg.channels[k]}: <b>{n.toLocaleString()}px</b>
                    </span>
                  );
                })}
              </Box>
            )}
          </Box>
          {/* Per-channel mask visibility — eye toggles for the live overlay.
              Cellpose has no per-channel mask, so we only show this for
              simple strategy AND when a preview has been computed. */}
          {cfg.mode === "simple" && activePreview && (
            <Box sx={{
              display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap",
              px: 0.5, py: 0.4, borderRadius: 0.5,
              bgcolor: "rgba(255,255,255,0.04)",
            }}>
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem", fontWeight: 600 }}>
                Show masks:
              </Typography>
              {(["r", "g", "b"] as const).map((k) => {
                const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
                const hasOverlay = !!activePreview.channelOverlays?.[k];
                const enabled = cfg.thresholds[k].enabled;
                const on = maskVisible[k];
                return (
                  <Box key={k}
                    onClick={() => setMaskVisible((p) => ({ ...p, [k]: !p[k] }))}
                    sx={{
                      cursor: hasOverlay ? "pointer" : "not-allowed",
                      fontSize: "0.66rem", px: 0.6, py: 0.15, borderRadius: 0.6,
                      display: "inline-flex", alignItems: "center", gap: 0.3,
                      userSelect: "none",
                      opacity: !enabled ? 0.4 : (hasOverlay ? 1 : 0.55),
                      bgcolor: on && hasOverlay ? sw : "transparent",
                      color: on && hasOverlay ? "common.white" : "text.secondary",
                      border: "1px solid", borderColor: sw,
                      fontWeight: on ? 700 : 500,
                    }}
                    title={!enabled
                      ? `${cfg.channels[k]} is not included in quantification (unchecked below)`
                      : hasOverlay
                        ? (on ? "Hide this channel's mask in the preview" : "Show this channel's mask in the preview")
                        : "No mask for this channel — Run preview first"}>
                    <span style={{ fontSize: "0.85rem", lineHeight: 1 }}>{on && hasOverlay ? "👁" : "·"}</span>
                    <span>{cfg.channels[k]}</span>
                  </Box>
                );
              })}
            </Box>
          )}
          {/* Image-list strip (quick jump) */}
          {images.length > 1 && (
            <Box sx={{ display: "flex", gap: 0.4, flexWrap: "wrap", maxHeight: 60, overflowY: "auto" }}>
              {images.map((im, i) => (
                <Box key={`${im.label}-${i}`}
                  onClick={() => setActiveIdx(i)}
                  sx={{
                    fontSize: "0.62rem", px: 0.6, py: 0.15, borderRadius: 0.5,
                    cursor: "pointer", userSelect: "none",
                    bgcolor: i === activeIdx ? "primary.main" : "transparent",
                    color: i === activeIdx ? "primary.contrastText" : "text.secondary",
                    border: "1px solid", borderColor: i === activeIdx ? "primary.main" : "divider",
                    fontWeight: i === activeIdx ? 700 : 500,
                    maxWidth: 170, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                  {im.label}
                </Box>
              ))}
            </Box>
          )}
        </Box>

        {/* ── Right: config panel ─────────────────────────────── */}
        <Box sx={{ flex: 1.0, minWidth: 360, display: "flex", flexDirection: "column", gap: 1.25 }}>
          {/* Channel rename ──────────────────────────────────── */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.1 }}>
            <Tooltip title="Rename each colour channel to the stain it shows (DAPI, Anti-VegF, …). Names flow into the plot legend AND the control-channel dropdown below.">
              <Typography variant="caption" sx={{ fontWeight: 700, display: "block", mb: 0.6 }}>
                Channel names
              </Typography>
            </Tooltip>
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 1 }}>
              {(["r", "g", "b"] as const).map((k) => {
                const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
                return (
                  <Box key={k} sx={{ display: "flex", flexDirection: "column", gap: 0.3 }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: sw, border: "1px solid", borderColor: "divider" }} />
                      <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", fontSize: "0.6rem" }}>
                        {k === "r" ? "Red" : k === "g" ? "Green" : "Blue"}
                      </Typography>
                    </Box>
                    <TextField size="small" value={cfg.channels[k]}
                      onChange={(e) => setChannel(k, e.target.value)}
                      placeholder={`Channel ${k.toUpperCase()}`}
                      inputProps={{ list: `chsugg-${k}`, style: { fontSize: "0.78rem", padding: "5px 8px" } }} />
                    <datalist id={`chsugg-${k}`}>
                      {CHANNEL_SUGGESTIONS.map((s) => <option key={s} value={s} />)}
                    </datalist>
                  </Box>
                );
              })}
            </Box>
            {/* Control-channel selector */}
            <Box sx={{ mt: 0.8, display: "flex", alignItems: "center", gap: 0.75 }}>
              <Tooltip title="Pick the 'control' stain (typically DAPI / Hoechst). The downstream R plot adds a sanity-check panel comparing this channel across groups — if it's significantly different, the panel is flagged so you know the groups may not be biologically comparable.">
                <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 100 }}>Control channel</Typography>
              </Tooltip>
              <TextField select size="small"
                value={cfg.controlChannel ?? ""}
                onChange={(e) => setCfg((c) => ({ ...c, controlChannel: (e.target.value || null) as ("r" | "g" | "b" | null) }))}
                inputProps={{ style: { fontSize: "0.75rem", padding: "4px 6px" } }}
                sx={{ minWidth: 140 }}
                SelectProps={SELECT_MENU_PROPS}>
                <MenuItem value="" sx={{ fontSize: "0.78rem" }}>— none —</MenuItem>
                <MenuItem value="r" sx={{ fontSize: "0.78rem" }}>{cfg.channels.r}</MenuItem>
                <MenuItem value="g" sx={{ fontSize: "0.78rem" }}>{cfg.channels.g}</MenuItem>
                <MenuItem value="b" sx={{ fontSize: "0.78rem" }}>{cfg.channels.b}</MenuItem>
              </TextField>
            </Box>
          </Box>

          {/* Strategy ─────────────────────────────────────────── */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.1 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.6, flexWrap: "wrap" }}>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>Strategy</Typography>
              <ToggleButtonGroup size="small" exclusive value={cfg.mode}
                onChange={(_, v) => { if (v) setCfg((c) => ({ ...c, mode: v })); }}
                sx={{ ml: 1 }}>
                <ToggleButton value="simple" sx={{ textTransform: "none", fontSize: "0.68rem", py: 0.15 }}>
                  Simple (brightest pixels)
                </ToggleButton>
                <ToggleButton value="cellpose" sx={{ textTransform: "none", fontSize: "0.68rem", py: 0.15 }}>
                  Cellpose (per-cell)
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <Typography variant="caption" sx={{ color: "text.secondary", display: "block", mb: 0.75 }}>
              {cfg.mode === "simple"
                ? "Detects the brightest pixels in each channel. Intensity sample = mean within that channel's detected pixels. n = images per group."
                : "Cellpose 3+ segments individual cells in the chosen channel. Per-cell mean intensity per channel. n = cells per group."}
            </Typography>

            {cfg.mode === "simple" ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.6 }}>
                {/* Friendlier "Subtract background" — replaces the
                    morphological-open-radius spinner. Default radius
                    when ON is 35 px (the script's CLI default); OFF
                    sends 0 to the backend. Power users who want a
                    different radius can still edit the value via the
                    saved JSON, but the picker UI is intentionally
                    binary here. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  <Box
                    onClick={() => setCfg((c) => ({
                      ...c,
                      rollingRadius: c.rollingRadius > 0 ? 0 : 35,
                    }))}
                    sx={{
                      width: 14, height: 14, borderRadius: 0.4, cursor: "pointer",
                      bgcolor: cfg.rollingRadius > 0 ? "primary.main" : "transparent",
                      border: "1px solid", borderColor: "primary.main",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: "common.white", fontSize: "0.7rem", fontWeight: 700,
                    }}>
                    {cfg.rollingRadius > 0 ? "✓" : ""}
                  </Box>
                  <Tooltip title="Removes uneven illumination + autofluorescence by subtracting a smoothed-out version of each channel before thresholding. Recommended ON for most fluorescence images.">
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      Subtract background
                    </Typography>
                  </Tooltip>
                  <Typography variant="caption" sx={{ color: "text.disabled", ml: 0.5 }}>
                    (recommended for uneven illumination)
                  </Typography>
                </Box>
                {/* Per-channel STRICTNESS slider — replaces the method/
                    percentile/min-area triple. Strictness 0 → percentile
                    80 (loose, catches dim signal); 50 → 95 (default);
                    100 → 99.7 (very strict, only the brightest pixels).
                    The threshold method is always percentile here. */}
                <Typography variant="caption" sx={{ color: "text.disabled", display: "block", mt: 0.4 }}>
                  Drag each channel's slider to control how strict the detection is. Higher = fewer pixels (only the brightest); lower = more pixels (catches dimmer signal).
                </Typography>
                {(["r", "g", "b"] as const).map((k) => {
                  const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
                  const t = cfg.thresholds[k];
                  // Map stored percentile back to a 0..100 strictness for the
                  // slider. percentile range we expose: 80..99.7 (~19.7 pts).
                  const PCT_LO = 80, PCT_HI = 99.7;
                  const pct = Math.max(PCT_LO, Math.min(PCT_HI, t.thresholdPercentile));
                  const strictness = Math.round(((pct - PCT_LO) / (PCT_HI - PCT_LO)) * 100);
                  const edgeOn = !!t.enhanceEdges;
                  return (
                    <Box key={k} sx={{
                      display: "grid",
                      gridTemplateColumns: "115px 1fr 56px 90px",
                      gap: 0.6, alignItems: "center",
                      px: 0.5, py: 0.35, borderRadius: 0.5,
                      bgcolor: t.enabled ? "transparent" : "action.hover",
                      opacity: t.enabled ? 1 : 0.5,
                    }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Tooltip title={t.enabled
                          ? "Click to skip this channel (not included in the analysis)"
                          : "Click to include this channel"}>
                          <Box
                            onClick={() => setThreshold(k, { enabled: !t.enabled })}
                            sx={{
                              width: 14, height: 14, borderRadius: 0.4, cursor: "pointer",
                              bgcolor: t.enabled ? sw : "transparent",
                              border: "1px solid", borderColor: sw,
                              display: "flex", alignItems: "center", justifyContent: "center",
                              color: "common.white", fontSize: "0.7rem", fontWeight: 700,
                            }}>
                            {t.enabled ? "✓" : ""}
                          </Box>
                        </Tooltip>
                        <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.7rem",
                                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {cfg.channels[k]}
                        </Typography>
                      </Box>
                      {/* Native range input — minimal MUI fuss, snappy. */}
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled" }}>loose</Typography>
                        <input
                          type="range" min={0} max={100} step={1}
                          value={strictness}
                          disabled={!t.enabled}
                          onChange={(e) => {
                            const s = Math.max(0, Math.min(100, Number(e.target.value)));
                            const newPct = PCT_LO + (s / 100) * (PCT_HI - PCT_LO);
                            setThreshold(k, {
                              thresholdMethod: "percentile",
                              thresholdPercentile: Math.round(newPct * 10) / 10,
                            });
                          }}
                          style={{ flex: 1, accentColor: sw }}
                        />
                        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled" }}>strict</Typography>
                      </Box>
                      {/* Per-channel edge-enhance chip (Difference of
                          Gaussians band-pass). Helps with cells that have
                          soft edges or are densely packed — recommend the
                          user try it when the slider can't separate cell
                          signal from haze. Stacks with "Subtract bg". */}
                      <Tooltip title={edgeOn
                        ? "Edge enhancement ON — Difference-of-Gaussians band-pass before threshold. Toggle off to disable."
                        : "Toggle ON for densely-packed cells / soft edges. Applies a Difference-of-Gaussians band-pass to this channel before thresholding."}>
                        <Box
                          onClick={() => t.enabled && setThreshold(k, { enhanceEdges: !edgeOn })}
                          sx={{
                            cursor: t.enabled ? "pointer" : "not-allowed",
                            fontSize: "0.62rem", px: 0.45, py: 0.1, borderRadius: 0.5,
                            display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 0.25,
                            bgcolor: edgeOn ? sw : "transparent",
                            color: edgeOn ? "common.white" : "text.secondary",
                            border: "1px solid", borderColor: sw,
                            fontWeight: edgeOn ? 700 : 500,
                            opacity: t.enabled ? 1 : 0.4,
                            userSelect: "none",
                          }}>
                          <span style={{ fontSize: "0.7rem", lineHeight: 1 }}>✦</span>
                          <span>Edges</span>
                        </Box>
                      </Tooltip>
                      <Tooltip title="Approximate fraction of the image's brightest pixels that survive the threshold.">
                        <Typography variant="caption" sx={{ fontSize: "0.7rem", color: "text.secondary", textAlign: "right" }}>
                          {t.enabled ? `top ${Math.max(0.3, Math.round((100 - pct) * 10) / 10)}%` : "skipped"}
                        </Typography>
                      </Tooltip>
                    </Box>
                  );
                })}
              </Box>
            ) : (
              // Cellpose params, compact. Diameter + min-size default
                // to AUTO and are buried behind an Advanced disclosure —
                // for typical fluorescence cell sizes the auto values are
                // fine, and exposing them at top level just confuses
                // users. Model + Segment-on stay top-level since they
                // change frequently per experiment.
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
                  <TextField select size="small" label="Model" value={cfg.cellpose.model}
                    onChange={(e) => setCellpose({ model: e.target.value })}
                    inputProps={{ style: { fontSize: "0.78rem" } }}
                    InputLabelProps={{ style: { fontSize: "0.78rem" } }}
                    SelectProps={SELECT_MENU_PROPS}>
                    <MenuItem value="cpsam" sx={{ fontSize: "0.78rem" }}>cpsam</MenuItem>
                    <MenuItem value="cyto3" sx={{ fontSize: "0.78rem" }}>cyto3</MenuItem>
                    <MenuItem value="cyto2" sx={{ fontSize: "0.78rem" }}>cyto2</MenuItem>
                    <MenuItem value="nuclei" sx={{ fontSize: "0.78rem" }}>nuclei</MenuItem>
                  </TextField>
                  <TextField select size="small" label="Segment on" value={cfg.cellpose.segChannel}
                    onChange={(e) => setCellpose({ segChannel: e.target.value as "r" | "g" | "b" })}
                    inputProps={{ style: { fontSize: "0.78rem" } }}
                    InputLabelProps={{ style: { fontSize: "0.78rem" } }}
                    SelectProps={SELECT_MENU_PROPS}>
                    <MenuItem value="r" sx={{ fontSize: "0.78rem" }}>{cfg.channels.r}</MenuItem>
                    <MenuItem value="g" sx={{ fontSize: "0.78rem" }}>{cfg.channels.g}</MenuItem>
                    <MenuItem value="b" sx={{ fontSize: "0.78rem" }}>{cfg.channels.b}</MenuItem>
                  </TextField>
                </Box>
                {/* Auto-by-default + advanced disclosure. Setting
                    diameter=0 in our config maps to None server-side,
                    which makes cellpose auto-estimate the diameter from
                    image content; min-size has a sane default (30 px²)
                    that the user only needs to touch for unusually
                    large/small cells. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
                  <Typography variant="caption" sx={{ color: "text.disabled", fontSize: "0.65rem" }}>
                    Diameter + min-size are AUTO (recommended). Override:
                  </Typography>
                  <TextField size="small" label="Diameter (px)" type="number" value={cfg.cellpose.diameter || ""}
                    onChange={(e) => setCellpose({ diameter: Math.max(0, Number(e.target.value) || 0) })}
                    placeholder="auto"
                    inputProps={{ min: 0, max: 400, step: 5, style: { fontSize: "0.7rem", padding: "3px 5px" } }}
                    InputLabelProps={{ style: { fontSize: "0.7rem" } }}
                    sx={{ width: 110 }} />
                  <TextField size="small" label="Min size (px²)" type="number" value={cfg.cellpose.minSize}
                    onChange={(e) => setCellpose({ minSize: Math.max(0, Number(e.target.value) || 0) })}
                    inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.7rem", padding: "3px 5px" } }}
                    InputLabelProps={{ style: { fontSize: "0.7rem" } }}
                    sx={{ width: 110 }} />
                </Box>
              </Box>
            )}
          </Box>

          {/* Groups (per-image assignment) ─────────────────────── */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.1, flex: 1, overflowY: "auto" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.4 }}>
              <Tooltip title="Group images into experimental conditions (Control, Treatment, …). The R plot draws one bar per (group, channel) with mean ± SD and pairwise stats.">
                <Typography variant="caption" sx={{ fontWeight: 700 }}>Groups</Typography>
              </Tooltip>
              <Button size="small" variant="outlined" onClick={addGroup} disabled={imageLabels.length === 0}
                sx={{ textTransform: "none", fontSize: "0.62rem", py: 0.05, px: 0.6, ml: "auto" }}>
                + Group
              </Button>
            </Box>
            {imageLabels.length === 0 ? (
              <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
                Wire image sources upstream first — then come back here to assign them to groups.
              </Typography>
            ) : cfg.groups.length === 0 ? (
              <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
                No groups yet — click <b>+ Group</b>, then click each image chip you want in that group.
                An image can only be in one group at a time.
              </Typography>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.4 }}>
                {cfg.groups.map((g) => (
                  <Box key={g.id} sx={{ display: "flex", alignItems: "flex-start", gap: 0.6, py: 0.3, borderTop: "1px dashed", borderColor: "divider" }}>
                    <TextField variant="standard" value={g.name}
                      onChange={(e) => renameGroup(g.id, e.target.value)}
                      inputProps={{ style: { fontSize: "0.76rem", fontWeight: 700, width: 110 } }} />
                    <Box sx={{ flex: 1, display: "flex", flexWrap: "wrap", gap: 0.35 }}>
                      {imageLabels.map((im) => {
                        const on = g.images.includes(im);
                        const inOther = imgToGroup.has(im) && imgToGroup.get(im) !== g.name;
                        return (
                          <Tooltip key={im} title={inOther ? `Already in "${imgToGroup.get(im)}" — clicking will move it here.` : im}>
                            <Box onClick={() => toggleImageInGroup(g.id, im)}
                              sx={{
                                fontSize: "0.62rem", px: 0.5, py: 0.08, borderRadius: 0.6,
                                cursor: "pointer", userSelect: "none",
                                bgcolor: on ? "primary.main" : "transparent",
                                color: on ? "primary.contrastText" : (inOther ? "text.disabled" : "text.secondary"),
                                border: "1px solid", borderColor: on ? "primary.main" : "divider",
                                fontWeight: on ? 700 : 500,
                                opacity: inOther && !on ? 0.65 : 1,
                                maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                              }}>
                              {im}
                            </Box>
                          </Tooltip>
                        );
                      })}
                    </Box>
                    <IconButton size="small" onClick={() => deleteGroup(g.id)}>
                      <DeleteOutlineIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Box>
                ))}
              </Box>
            )}
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ flexDirection: "column", alignItems: "stretch", gap: 0.75, px: 2, py: 1.25 }}>
        {/* Save-blocker banner. The run-graph gate aborts the workflow
            with "no groups assigned" when the saved config has no group
            with any images. Mirror the band-picker UX: disable Save and
            explain exactly what's missing so the user doesn't sit there
            wondering why Run graph keeps reopening the picker. */}
        {(() => {
          const hasImages = images.length > 0;
          const groupCount = (cfg.groups ?? []).length;
          const groupsWithImages = (cfg.groups ?? []).filter((g) => (g.images?.length ?? 0) > 0).length;
          const canSave = !hasImages || (groupCount > 0 && groupsWithImages > 0);
          let reason = "";
          if (hasImages && groupCount === 0) {
            reason = "⚠ No groups defined — the analysis runner halts on save. "
              + "Click \"+ Group\" above and assign at least one image before saving.";
          } else if (hasImages && groupsWithImages === 0) {
            reason = "⚠ You have groups but no images assigned — click each image chip "
              + "(in the row next to its group name) to assign it, then save.";
          }
          return (
            <>
              {reason && (
                <Box sx={{
                  px: 1, py: 0.5, borderRadius: 0.5,
                  bgcolor: "rgba(180,120,40,0.12)",
                  border: "1px solid rgba(220,150,60,0.4)",
                  color: "#e0a060",
                  fontSize: "0.72rem",
                  textAlign: "left",
                }}>
                  {reason}
                </Box>
              )}
              <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 1 }}>
                <Button onClick={onClose} sx={{ textTransform: "none" }}>Cancel</Button>
                <Tooltip title={canSave ? "" : (reason || "Save disabled")}
                         disableHoverListener={canSave}
                         disableFocusListener={canSave}>
                  <span>
                    <Button variant="contained" disabled={!canSave}
                            onClick={() => onSave(cfg)}
                            sx={{ textTransform: "none" }}>
                      Save configuration
                    </Button>
                  </span>
                </Tooltip>
              </Box>
            </>
          );
        })()}
      </DialogActions>
    </Dialog>
  );
}
