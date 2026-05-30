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
import PanToolAltIcon from "@mui/icons-material/PanToolAlt";
import BrushIcon from "@mui/icons-material/Brush";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import CallMergeIcon from "@mui/icons-material/CallMerge";
import AutoFixOffIcon from "@mui/icons-material/AutoFixOff";
import UndoIcon from "@mui/icons-material/Undo";
import RedoIcon from "@mui/icons-material/Redo";
import LayersClearIcon from "@mui/icons-material/LayersClear";

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
  /** Which cellpose major to dispatch into.  "v3" → real cyto3 /
   *  cyto2 / nuclei model zoo (smaller, faster, more predictable
   *  semantics).  "v4" → cpsam (generalist SAM-based, ~100 MB, the
   *  only model v4 ships).  The plugin host keeps both venvs side-
   *  by-side; the user picks per-node. */
  cellposeVersion?: "v3" | "v4";
  /** Model name.  Valid choices depend on cellposeVersion:
   *  v3 → "cyto3" | "cyto2" | "cyto" | "nuclei"
   *  v4 → "cpsam" */
  model: string;
  /** Object diameter prior in px. 0 = auto-estimate. */
  diameter: number;
  /** Which channel carries the cell-body / cytoplasm signal for the
   *  cyto3/cpsam segmentation. Cellpose's "channels[0]". */
  segChannel: "r" | "g" | "b";
  /** Optional NUCLEI channel — DAPI / Hoechst. When set, cyto3 uses it
   *  as its nuclei input (channels[1]), which the model was trained on
   *  and produces noticeably better whole-cell boundaries than the
   *  single-channel form. ALSO required for "measureCompartments". */
  nucleiChannel?: "r" | "g" | "b" | null;
  /** Min object size in px² — filters out spurious tiny masks. */
  minSize: number;
  /** When true AND nucleiChannel is set, run a SECOND cellpose pass
   *  with the `nuclei` model on the nuclei channel, then emit per-cell
   *  measurements separately for WHOLE-CELL / NUCLEUS / CYTOPLASM
   *  compartments. Doubles cellpose cost but lets you compare e.g.
   *  nuclear-vs-cytoplasmic localisation of a target. */
  measureCompartments?: boolean;
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
  /** Stable input key that survives label-collision suffixing and
   *  sanitisation.  cfg.groups[].images stores THIS, not the display
   *  label — so two images with the same display name (e.g. "image1"
   *  from two upstream nodes) can still be grouped distinctly.
   *  Legacy configs stored labels here; the generator accepts both. */
  id: string;
  /** Display label (what the user sees in the cycler chips).  Suffixed
   *  with " #N" when multiple inputs share a name, but the GROUPING
   *  identity is `id`, not `label`. */
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
  /** Optional user-painted cellpose masks keyed by image label.  Each
   *  value is the RGBA-packed PNG data URL the dialog produced from
   *  its interactive editor (R = label low byte, G = label high
   *  byte).  When non-empty, the generated Python forwards these to
   *  /api/analysis/run-cellpose so the FINAL quantification uses the
   *  user's edits instead of a fresh cellpose run for those images. */
  editedMasks?: Record<string, string>;
  /** Optional user-painted per-channel SIMPLE-mode binary masks,
   *  keyed by image label → channel → "L" PNG data URL (255 in mask,
   *  0 out).  When present for an image's channel, the generated
   *  Python skips the threshold step for that (image, channel) and
   *  uses the user's mask instead. */
  editedChannelMasks?: Record<string, Partial<Record<"r" | "g" | "b", string>>>;
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
    // Default to cyto3 — much smaller (~25 MB vs ~100 MB for cpsam),
    // ~3x faster inference, works great on most fluorescence cell
    // types. Users can switch to cpsam in the dropdown when they need
    // its higher-accuracy segmentation.
    cellpose: {
      cellposeVersion: "v4",
      model: "cpsam", diameter: 0, segChannel: "b",
      nucleiChannel: null, measureCompartments: false, minSize: 80,
    },
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
    // Backfill cellposeVersion for configs saved before 0.1.324.
    // Heuristic: if the user picked a v3-only model (cyto3 / cyto2 /
    // nuclei) treat it as v3; otherwise default to v4.
    const cpIn = cfg.cellpose || fresh.cellpose;
    const cp = {
      ...cpIn,
      cellposeVersion: cpIn.cellposeVersion
        ?? (["cyto3", "cyto2", "cyto", "nuclei"].includes(String(cpIn.model || "")) ? "v3" : "v4"),
    };
    return {
      ...out,
      thresholds: cfg.thresholds || fresh.thresholds,
      rollingRadius: typeof cfg.rollingRadius === "number" ? cfg.rollingRadius : fresh.rollingRadius,
      controlChannel: cfg.controlChannel ?? null,
      cellpose: cp,
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
import re as _re_grp
for g in CFG.get("groups", []) or []:
    nm = (g.get("name") or "").strip()
    if not nm: continue
    for im in g.get("images", []) or []:
        s = str(im)
        img2group[s] = nm
        # Back-compat: 0.1.330's first cut stored the collectInputs-
        # prefixed id (inset_N_<key>) instead of the underlying
        # <key> the runtime sees, so any config saved during that
        # window misses every lookup.  Also map the stripped form
        # so re-running an existing project doesn't require the user
        # to re-assign every image.
        _m = _re_grp.match(r"^inset_\d+_(.+)$", s)
        if _m:
            img2group[_m.group(1)] = nm
        # Also map the "up_image_N_<rest>" form for upstream-node
        # outputs and the trailing-filename form so a saved label
        # with embedded slashes still matches a bare-filename
        # runtime label.
        _m2 = _re_grp.match(r"^up_(?:image|table)_\d+_(.+)$", s)
        if _m2:
            img2group[_m2.group(1)] = nm
        if "/" in s:
            img2group[s.rsplit("/", 1)[-1]] = nm

def _resolve_group(key, label, mapping):
    """Multi-level group lookup with explicit fallbacks.
    1) exact key match (the post-0.1.331 stable id)
    2) exact label match (legacy configs)
    3) substring match on either side (catches edge cases where the
       saved entry has the label as a substring, e.g. a path)"""
    if not mapping: return None
    if key and mapping.get(str(key)): return mapping[str(key)]
    if label and mapping.get(label): return mapping[label]
    # Last-resort substring match — only when nothing exact hit.
    if label:
        lbl_lc = label.lower()
        for k, v in mapping.items():
            kk = k.lower()
            if lbl_lc in kk or kk in lbl_lc:
                return v
    return None

# Diagnostic — print what img2group ended up with so the user can
# verify their picker assignments actually reached the runtime.
# (When a group is "silently missing", this surfaces whether the
# bug is in saving vs in resolution.)
print(f"[intensity] img2group has {len(img2group)} entries (showing keys): "
      f"{sorted(set(img2group.keys()))[:8]}")
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

def _cellpose_labels_batch(items, model, channels, isolate_channel=0, diameter_override=None):
    """BATCH cellpose over multiple images in ONE subprocess call so the
    ~100-MB cpsam model loads only ONCE (vs once per image when calling
    image-by-image — that was the dominant cost). "items" = list of
    (label, a8_rgb_native) tuples. Returns {label: labels_array_native_res}
    + per-image error map for the ones that failed.

    "model" and "channels" are passed through so the SAME function can
    run the cell-body pass (cyto3 with [cyto, nuclei]) AND the nucleus
    pass (nuclei with [nuc, 0]) for compartment measurement.

    isolate_channel — when > 0 (1=R, 2=G, 3=B), strips the image to a
    single channel before encoding.  Required for the NUCLEI pass on
    cellpose 4: cpsam (v4's only model) has no separate nuclei head, so
    given the full RGB it returns cell-shaped masks even when the user
    asked for nuclei.  Isolating the DAPI channel makes the nuclei the
    only thing visible, so cpsam (and v3's nuclei model) segment them
    as the requested nuclei.

    diameter_override — when set, overrides cp_cfg["diameter"] for this
    pass.  Used by the nuclei pass to default to ~60% of the cell
    diameter (nuclei are systematically smaller than whole cells)."""
    sent = []
    scales = {}    # label → scale factor applied (1.0 = no resize)
    sizes_native = {}  # label → (h, w) at native res
    for label, a8 in items:
        sizes_native[label] = a8.shape[:2]
        # Channel isolation (nuclei-pass on cellpose 4).  Done BEFORE
        # downsize so the resize works on the isolated 2D plane.
        if isolate_channel and a8.ndim == 3 and a8.shape[2] >= isolate_channel:
            ch_zero = isolate_channel - 1   # 1..3 → 0..2 RGB index
            iso = a8[..., ch_zero]
            a8_iso = np.stack([iso, iso, iso], axis=-1)
        else:
            a8_iso = a8
        a8_small, sc = _downsize_for_cellpose(a8_iso)
        scales[label] = sc
        sent.append({
            "kind": "image", "key": label, "label": label,
            "image_b64": _b64.b64encode(_png_bytes(a8_small)).decode(),
        })
    # Forward user-edited masks (Intensity dialog Tier-1 editing).
    # Only relevant for the WHOLE-CELL pass; the nuclei pass uses a
    # different model and edits aren't tracked separately for nuclei.
    _edited = {}
    if model != "nuclei":
        _edited_cfg = CFG.get("editedMasks") or {}
        if isinstance(_edited_cfg, dict):
            for _lbl in [label for label, _ in items]:
                v = _edited_cfg.get(_lbl)
                if isinstance(v, str) and v:
                    # Strip the "data:image/png;base64," prefix if present.
                    _edited[_lbl] = v.split(",", 1)[-1] if v.startswith("data:") else v

    # ── FAST PATH: every requested image already has an edited mask ──
    # Decode the supplied PNGs inline and skip the cellpose subprocess
    # ENTIRELY.  The dialog's Run preview just did the segmentation; the
    # user's saved labels ARE the result we'd otherwise re-derive.
    # Previously the 0.1.327 fix made the backend RUNNER skip model.eval,
    # but the JS still POSTed /api/analysis/run-cellpose every time,
    # which paid ~5-15 s for the subprocess spawn + cellpose Python
    # import.  This short-circuit eliminates that.
    if _edited and len(_edited) == len(items):
        print(f"[intensity] cellpose ({model}): all {len(items)} image(s) carry "
              f"user-supplied mask(s) — skipping cellpose subprocess entirely")
        labels_by_label = {}
        errors = {}
        for label, _ in items:
            try:
                # Same decode the backend runner would do: RGBA-packed
                # PNG, R = label low byte, G = label high byte.
                rgba = np.asarray(
                    _Im.open(_io.BytesIO(_b64.b64decode(_edited[label]))).convert("RGBA")
                )
                arr = (rgba[..., 0].astype(np.int32)
                       | (rgba[..., 1].astype(np.int32) << 8))
                h_n, w_n = sizes_native[label]
                arr = _upsize_labels(arr, h_n, w_n)
                labels_by_label[label] = arr
            except Exception as _e:
                errors[label] = f"edited-mask decode failed: {_e}"
        return labels_by_label, errors

    _diam = float(diameter_override) if diameter_override else float(cp_cfg.get("diameter") or 0)
    # Cellpose major: v3 (real cyto3 + nuclei model zoo) or v4 (cpsam).
    # Routes to the matching plugin venv on the backend.  Forwarded as
    # a top-level field on the request so the v3 path doesn't get its
    # model name silently remapped to cpsam.
    _cp_ver = str(cp_cfg.get("cellposeVersion") or "v4")
    payload_obj = {
        "config": json.dumps({
            "model": model,
            "diameter": _diam or None,
            "min_size": int(cp_cfg.get("minSize") or 30),
            "channels": channels,
            # Use GPU when available (CUDA / Apple Silicon MPS). Falls
            # back to CPU automatically if neither is compiled in.
            "use_gpu": True,
        }),
        "extra_inputs": sent,
        "sources": [], "timeout_sec": 600,
        "cellpose_version": _cp_ver,
    }
    if _edited:
        payload_obj["edited_masks"] = _edited
        print(f"[intensity] cellpose: applying {len(_edited)} user-edited mask(s) "
              f"({', '.join(sorted(_edited.keys()))})")
    payload = json.dumps(payload_obj)
    print(f"[intensity] cellpose ({model}, channels={channels}): batching {len(sent)} image(s) into one model load "
          f"(cap {CELLPOSE_MAX_EDGE}px per edge)")
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
    # Index returned images by name.  Prefer <label>_labels16 (RGBA-
    # packed, no >255 truncation) and fall back to the legacy 8-bit
    # <label>_labels for older sidecar builds.
    by_name = {im.get("name"): im.get("image") for im in (cp_out.get("images") or [])}
    labels_by_label = {}
    errors = {}
    for label, _ in items:
        lbl16_b64 = by_name.get(f"{label}_labels16")
        lbl_b64 = by_name.get(f"{label}_labels")
        if not (lbl16_b64 or lbl_b64):
            errors[label] = "no labels image returned"
            continue
        try:
            if lbl16_b64:
                rgba = np.asarray(_Im.open(_io.BytesIO(_b64.b64decode(lbl16_b64))).convert("RGBA"))
                arr = (rgba[..., 0].astype(np.int32) | (rgba[..., 1].astype(np.int32) << 8))
            else:
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
    _unassigned = []
    for key, src in imgs:
        label = _label_of(src, key)
        grp = _resolve_group(key, label, img2group)
        if not grp:
            print(f"[intensity] {label} (key={key}): not assigned to any group — skipping")
            _unassigned.append(label)
            continue
        print(f"[intensity] {label} (key={key}) → group {grp!r}")
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
        _hint = (f" Dropped: {_unassigned}." if _unassigned else "")
        raise SystemExit(f"[intensity] no images are in any group — "
                         f"assign images to groups in the picker.{_hint}")
    if _unassigned:
        # Cellpose mode silently drops un-grouped images by skipping
        # them above; surface that explicitly so the user notices
        # which inputs didn't reach the plot.
        print(f"[intensity] WARNING: dropped {len(_unassigned)} un-grouped image(s) "
              f"in cellpose mode: {_unassigned}")
    # Second pass: ONE cellpose call for the whole batch — for WHOLE-CELL
    # segmentation. cyto3 (the default) uses both the cytoplasm channel
    # AND the nuclei channel (when set) for noticeably better cell
    # boundaries; that's "channels: [seg_ch, nuc_ch]".
    seg_idx = {"r": 1, "g": 2, "b": 3}.get(cp_cfg.get("segChannel") or "b", 3)
    nuc_key = (cp_cfg.get("nucleiChannel") or "").lower()
    nuc_idx = {"r": 1, "g": 2, "b": 3}.get(nuc_key, 0)
    measure_compartments = bool(cp_cfg.get("measureCompartments")) and nuc_idx > 0
    print(f"[intensity] cellpose: starting batch run on {len(prepared)} image(s) — "
          f"the model loads once (first run on a machine downloads weights)")
    _cp_t0 = __import__("time").monotonic()
    labels_by_label, cp_errors = _cellpose_labels_batch(
        [(p[2], p[6]) for p in prepared],
        cp_cfg.get("model") or "cyto3",
        [seg_idx, nuc_idx],
    )
    print(f"[intensity] cellpose cell-pass: {__import__('time').monotonic() - _cp_t0:.1f}s "
          f"({len(labels_by_label)} ok, {len(cp_errors)} failed)")
    # Optional second cellpose pass: nuclei model on the nuclei channel.
    # Only when the user enabled "measure compartments" AND set a nuclei
    # channel. Returns per-image nucleus_labels at native resolution.
    nuclei_by_label = {}
    if measure_compartments:
        # Nuclei-pass: cellpose 4 only ships cpsam (no separate nuclei
        # head), so giving it the full RGB makes it return CELL-shaped
        # masks even when the user asked for nuclei.  Isolating the DAPI
        # channel makes the nuclei the only thing visible, so cpsam (and
        # v3's nuclei model) segment them as the requested nuclei.
        # Diameter defaults to ~60% of the cell diameter (nuclei are
        # systematically smaller — reusing the cell diameter merges
        # adjacent nuclei).  When the user picked auto-diameter (0) we
        # let cellpose estimate for the nuclei pass too.
        _cell_diam = float(cp_cfg.get("diameter") or 0)
        _nuc_diam = _cell_diam * 0.6 if _cell_diam > 0 else 0
        print(f"[intensity] cellpose: starting nuclei-pass on {len(prepared)} image(s) "
              f"(model=nuclei, isolated channel={nuc_idx}, diameter={_nuc_diam or 'auto'})")
        _np_t0 = __import__("time").monotonic()
        nuclei_by_label, _ne = _cellpose_labels_batch(
            [(p[2], p[6]) for p in prepared],
            "nuclei",
            [0, 0],                       # grayscale segmentation
            isolate_channel=nuc_idx,      # strip to nuclei-only signal
            diameter_override=_nuc_diam,  # shrink for nuclei-sized blobs
        )
        print(f"[intensity] cellpose nuclei-pass: {__import__('time').monotonic() - _np_t0:.1f}s "
              f"({len(nuclei_by_label)} ok, {len(_ne)} failed)")
    # Per-image control mean (for Gap 3 normalization). Computed after
    # the per-cell loop — initialize empty here.
    image_ctrl_mean = {}
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
        # Optional nucleus labels for this image — used to derive the
        # per-cell nuclear ROI when measure_compartments is on. We use
        # cell ∩ nuclei to keep each cell's nucleus paired to ITS cell.
        nuc_lbl = nuclei_by_label.get(label) if measure_compartments else None
        if nuc_lbl is not None and nuc_lbl.shape[:2] != raw.shape[:2]:
            print(f"[intensity] {label}: nucleus labels shape mismatch — ignoring nuclei", file=_sys.stderr)
            nuc_lbl = None
        # Collect control-channel BG-corrected means PER CELL so we can
        # compute the per-image mean control signal (used for Gap 3
        # normalization: mean_intensity_norm = mean_intensity / img_ctrl).
        ctrl_per_cell = []
        ctrl_ci = {"R": 0, "G": 1, "B": 2}.get(str(control_key).upper(), -1) if control_key else -1
        for cid in cell_ids:
            cell_mask = labels == cid
            cell_area = int(cell_mask.sum())
            ys, xs = np.where(cell_mask)
            # Compartments:
            #  - "whole" — full cellpose cyto3 mask (cell body)
            #  - "nucleus" — cell ∩ any nucleus label
            #  - "cytoplasm" — cell − nucleus
            compartments = [("whole_cell", cell_mask)]
            if nuc_lbl is not None:
                nuc_in_cell = cell_mask & (nuc_lbl > 0)
                cyto_in_cell = cell_mask & ~nuc_in_cell
                if int(nuc_in_cell.sum()) > 0:
                    compartments.append(("nucleus", nuc_in_cell))
                if int(cyto_in_cell.sum()) > 0:
                    compartments.append(("cytoplasm", cyto_in_cell))
            # Track control-channel mean over the WHOLE cell for the
            # per-image normalization step (one per cell, averaged across
            # cells per image later).
            if ctrl_ci >= 0:
                ctrl_vals = corrected[..., ctrl_ci][cell_mask]
                if ctrl_vals.size > 0:
                    ctrl_per_cell.append(float(np.mean(ctrl_vals)))
            for compartment, m in compartments:
                if int(m.sum()) == 0: continue
                m_area = int(m.sum())
                for ci, ck in enumerate(("R", "G", "B")):
                    rv = raw[..., ci][m].astype(np.float64)
                    cv = corrected[..., ci][m]
                    rows.append({
                        "source": label,
                        "group": grp,
                        "channel": ch_name[ck],
                        "compartment": compartment,
                        "is_control": (control_name is not None and ch_name[ck] == control_name),
                        "object_id": int(cid),
                        "area_px": m_area,
                        "cell_area_px": cell_area,
                        "centroid_x": float(np.mean(xs)) if xs.size else 0.0,
                        "centroid_y": float(np.mean(ys)) if ys.size else 0.0,
                        "raw_mean": float(np.mean(rv)),
                        "raw_integrated_density": float(np.sum(rv)),
                        "background_corrected_mean": float(np.mean(cv)),
                        "background_corrected_integrated_density": float(np.sum(cv)),
                        "mean_intensity": float(np.mean(cv)),
                        "max_intensity": float(np.max(rv)),
                    })
        # Per-image control mean (for normalization step below). Skipped
        # when there's no control channel or no cells with control signal.
        if ctrl_per_cell:
            image_ctrl_mean[label] = float(np.mean(ctrl_per_cell))
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
    # Gap 3: per-image control normalization. For each row, divide its
    # mean_intensity by the image's mean control-channel intensity (over
    # the WHOLE-cell compartment). Cancels exposure differences between
    # images. Only computed when control_name is set AND we measured
    # control intensity on enough cells.
    if control_name and image_ctrl_mean:
        for r in rows:
            ctrl = image_ctrl_mean.get(r["source"])
            if ctrl and ctrl > 0:
                r["per_image_control_mean"] = ctrl
                r["mean_intensity_norm"] = float(r["mean_intensity"] / ctrl)
            else:
                r["per_image_control_mean"] = None
                r["mean_intensity_norm"] = None
    n_compart = len({r.get("compartment", "whole_cell") for r in rows})
    print(f"computed per-cell intensities across {n_images_with_cells} image(s); "
          f"compartments={n_compart}; "
          f"normalization={'on' if (control_name and image_ctrl_mean) else 'off'}")
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
    # Multi-level lookup: exact key → exact label → substring fallback.
    # When nothing matches we surface "(unassigned)" so the image
    # shows on the plot instead of vanishing — silently collapsing
    # into the image's own label was masking missing-group bugs.
    grp = _resolve_group(key, label, img2group) or "(unassigned)"
    if grp == "(unassigned)":
        print(f"[intensity] {label} (key={key}): not assigned to any group — "
              f"surfacing as '(unassigned)' so the plot shows it dropped")
    else:
        print(f"[intensity] {label} (key={key}) → group {grp!r}")
    raw = _pixels(src)
    corrected = np.zeros_like(raw, dtype=np.float64)
    masks = {}
    # Edited channel masks (simple-mode paint/erase, 0.1.333+).  When
    # the dialog supplied a user-painted mask for an (image, channel),
    # we skip the threshold step and use the supplied mask directly —
    # NN-resized to native if needed.  Lookup tries the same multi-id
    # forms as img2group (key / label / etc).
    _edited_ch_cfg = CFG.get("editedChannelMasks") or {}
    _edited_ch_for_image = None
    if isinstance(_edited_ch_cfg, dict):
        for _candidate in (str(key), label, _re_grp.sub(r"^inset_\d+_", "", str(key))):
            v = _edited_ch_cfg.get(_candidate)
            if isinstance(v, dict):
                _edited_ch_for_image = v; break
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
        corrected[..., ci] = corr
        # ── FAST PATH: user-supplied edited mask for this channel. ──
        _user_mask_b64 = (_edited_ch_for_image or {}).get(key_lc) if _edited_ch_for_image else None
        if _user_mask_b64:
            try:
                _raw_bytes = _b64.b64decode(_user_mask_b64.split(",", 1)[-1] if _user_mask_b64.startswith("data:") else _user_mask_b64)
                _mask_img = _Im.open(_io.BytesIO(_raw_bytes)).convert("L")
                if (_mask_img.size[1], _mask_img.size[0]) != raw.shape[:2] and _have_scipy:
                    _mask_img = _mask_img.resize((raw.shape[1], raw.shape[0]), resample=_Im.NEAREST)
                mask = (np.asarray(_mask_img) > 127)
                masks[key_lc] = mask
                print(f"[intensity] {label} ch {ck}: using user-edited mask ({int(mask.sum())} px)")
                n_pixels = int(mask.sum())
                if n_pixels == 0:
                    continue
                rv = raw[..., ci][mask].astype(np.float64)
                cv = corrected[..., ci][mask]
                rows.append({
                    "source": label, "group": grp, "channel": ch_name[ck],
                    "is_control": (control_name is not None and ch_name[ck] == control_name),
                    "n_pixels": n_pixels,
                    "raw_mean": float(rv.mean()), "raw_integrated_density": float(rv.sum()),
                    "background_corrected_mean": float(cv.mean()),
                    "background_corrected_integrated_density": float(cv.sum()),
                    "mean_intensity": float(cv.mean()),
                    "max_intensity": float(rv.max()),
                    "compartment": "whole_image",
                    "edited": True,
                })
                continue
            except Exception as _ee:
                print(f"[intensity] {label} ch {ck}: edited mask decode failed: {_ee}", file=_sys.stderr)
                # fall through to threshold path
        thresh_in = corr
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
# Gap 3 (simple mode): per-image control normalization. Group rows by
# source, find the control-channel row, divide every channel's mean
# intensity by it. Cancels exposure differences between images.
if control_name:
    from collections import defaultdict as _dd
    ctrl_per_source = {}
    for r in rows:
        if r.get("is_control") and r.get("source") and r.get("mean_intensity") is not None:
            ctrl_per_source[r["source"]] = float(r["mean_intensity"])
    for r in rows:
        ctrl = ctrl_per_source.get(r.get("source"))
        if ctrl and ctrl > 0:
            r["per_image_control_mean"] = ctrl
            r["mean_intensity_norm"] = float(r["mean_intensity"] / ctrl)
        else:
            r["per_image_control_mean"] = None
            r["mean_intensity_norm"] = None
# Simple mode rows are NOT per-cell — there's no segmentation in this
# mode. They're a single mean per (image, channel) inside the channel's
# threshold mask, which covers wherever the channel happens to be
# bright in the whole image. Tag them as "whole_image" so the R plot
# can show that label honestly and the cellpose compartments
# (whole_cell / nucleus / cytoplasm) stay distinct.
for r in rows:
    r.setdefault("compartment", "whole_image")
print(f"computed channel intensities for {len(imgs)} image(s); "
      f"groups = {sorted({r['group'] for r in rows})}; "
      f"control = {control_name or '<none>'}; "
      f"normalization={'on' if control_name else 'off'}")
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

/** A computed preview for one image: a base composite + transparent
 *  boundary overlays (per-channel for simple strategy, or cells +
 *  nuclei for cellpose). The dialog layers them onto a canvas so
 *  visibility toggles are instant and pixel-perfect. */
type PreviewLayers = {
  /** Combined / fused overlay (legacy fallback; cellpose result). */
  overlaySrc?: string;
  /** Base layer — contrast-stretched composite (no outlines). */
  compositeSrc?: string;
  /** Per-channel boundary on transparent bg (simple strategy). */
  channelOverlays?: Partial<Record<"r" | "g" | "b", string>>;
  /** Cellpose cell-mask outline (yellow, transparent bg). */
  cellOverlaySrc?: string;
  /** Cellpose nucleus-mask outline (cyan, transparent bg) — only
   *  present when "Measure compartments" was on. */
  nucleusOverlaySrc?: string;
  /** Editable cell-label raster at PREVIEW resolution.  Decoded from
   *  the backend's RGBA-packed PNG (R = lo, G = hi).  Each pixel's
   *  Int32Array value is the cell ID (0 = background). */
  cellLabels?: Int32Array;
  nucleusLabels?: Int32Array;
  labelW?: number;
  labelH?: number;
  /** Editable per-channel binary masks (simple-strategy paint/erase).
   *  Each Uint8Array is `labelW * labelH` long; 1 = in mask, 0 = out.
   *  Decoded from the backend's `channel_masks_b64` PNGs. */
  channelMasks?: Partial<Record<"r" | "g" | "b", Uint8Array>>;
  /** Cellpose intermediate visualisations.  Tier 2 view modes. */
  flowsRgbSrc?: string;
  cellprobSrc?: string;
  /** True once the user has interactively painted/deleted/merged. */
  edited?: boolean;
  n_cells?: number;
  n_nuclei?: number;
  per_channel?: Record<string, number>;
};

/** Decode an 8-bit grayscale PNG into a Uint8Array (1 where pixel >
 *  127, 0 elsewhere).  Used for simple-mode per-channel binary masks
 *  the user can paint or erase. */
async function decodeBinaryMaskPng(b64: string): Promise<{ mask: Uint8Array; w: number; h: number } | null> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.onload = () => {
      const cnv = document.createElement("canvas");
      cnv.width = img.naturalWidth;
      cnv.height = img.naturalHeight;
      const ctx = cnv.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, cnv.width, cnv.height).data;
      const n = cnv.width * cnv.height;
      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) mask[i] = data[i * 4] > 127 ? 1 : 0;
      resolve({ mask, w: cnv.width, h: cnv.height });
    };
    img.onerror = () => resolve(null);
    img.src = `data:image/png;base64,${b64}`;
  });
}

/** Encode a Uint8Array binary mask back to a single-channel "L" PNG
 *  data URL for round-tripping to the backend. */
function encodeBinaryMaskPng(mask: Uint8Array, w: number, h: number): string {
  const cnv = document.createElement("canvas");
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(w, h);
  const d = imageData.data;
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 255 : 0;
    d[i * 4] = v; d[i * 4 + 1] = v; d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return cnv.toDataURL("image/png");
}

/** Derive a 1-pixel boundary from a binary mask (Uint8Array).  Same
 *  4-neighbour transition logic as deriveBoundary() but for binary
 *  data — used by the simple-mode compositor to draw per-channel
 *  outlines after the user paints / erases. */
function deriveBinaryBoundary(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const up = y > 0 ? mask[i - w] : 0;
      const dn = y < h - 1 ? mask[i + w] : 0;
      const lf = x > 0 ? mask[i - 1] : 0;
      const rt = x < w - 1 ? mask[i + 1] : 0;
      if (!up || !dn || !lf || !rt) out[i] = 1;
    }
  }
  return out;
}

/** Decode the backend's RGBA-packed labels PNG into an Int32Array.
 *  Uses a hidden canvas (browser-native PNG decode), then reads
 *  R = low byte + G = high byte for every pixel.  Returns null on
 *  any decode failure. */
async function decodeRgbaLabels(b64: string): Promise<{ labels: Int32Array; w: number; h: number } | null> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.onload = () => {
      const cnv = document.createElement("canvas");
      cnv.width = img.naturalWidth;
      cnv.height = img.naturalHeight;
      const ctx = cnv.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, cnv.width, cnv.height).data;
      const n = cnv.width * cnv.height;
      const labels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        labels[i] = data[i * 4] | (data[i * 4 + 1] << 8);
      }
      resolve({ labels, w: cnv.width, h: cnv.height });
    };
    img.onerror = () => resolve(null);
    img.src = `data:image/png;base64,${b64}`;
  });
}

/** Encode an Int32Array label image back to an RGBA-packed PNG data
 *  URL for round-tripping to the backend (R = lo, G = hi). */
function encodeRgbaLabels(labels: Int32Array, w: number, h: number): string {
  const cnv = document.createElement("canvas");
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(w, h);
  const d = imageData.data;
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i] | 0;
    d[i * 4] = v & 0xFF;
    d[i * 4 + 1] = (v >>> 8) & 0xFF;
    d[i * 4 + 2] = 0;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return cnv.toDataURL("image/png");
}

/** Derive a 4-neighbour boundary mask from an Int32Array label image. */
function deriveBoundary(labels: Int32Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = labels[i];
      if (v === 0) continue;
      // Compare against 4-neighbours.  At the edge, "no neighbour"
      // is treated as a different label so the cell's silhouette is
      // closed against the image border.
      const up = y > 0 ? labels[i - w] : -1;
      const dn = y < h - 1 ? labels[i + w] : -1;
      const lf = x > 0 ? labels[i - 1] : -1;
      const rt = x < w - 1 ? labels[i + 1] : -1;
      if (v !== up || v !== dn || v !== lf || v !== rt) out[i] = 1;
    }
  }
  return out;
}

/** Count the number of distinct non-zero IDs in a label image.
 *  Used for the live cell-count display next to the eye chip. */
function countNonZeroIds(labels: Int32Array): number {
  const seen = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (v > 0) seen.add(v);
  }
  return seen.size;
}

/** Render a boundary mask onto a transparent canvas in the given
 *  colour.  Returns the canvas so the compositor can drawImage it. */
function renderBoundaryCanvas(
  boundary: Uint8Array, w: number, h: number, rgb: [number, number, number],
): HTMLCanvasElement {
  const cnv = document.createElement("canvas");
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext("2d");
  if (!ctx) return cnv;
  const imageData = ctx.createImageData(w, h);
  const d = imageData.data;
  for (let i = 0; i < boundary.length; i++) {
    if (!boundary[i]) continue;
    d[i * 4] = rgb[0];
    d[i * 4 + 1] = rgb[1];
    d[i * 4 + 2] = rgb[2];
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return cnv;
}

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
  // Cellpose mask visibility (separate from simple-mode channel
  // visibility above): cells = yellow outline, nuclei = cyan. Both
  // default ON so the user immediately sees what cellpose found.
  const [cpMaskVisible, setCpMaskVisible] = useState<{ cells: boolean; nuclei: boolean }>({ cells: true, nuclei: true });
  // ── Tier 2: view mode (base layer) ────────────────────────────
  // "composite" = full contrast-stretched RGB; "r"/"g"/"b" isolate
  // one channel as a grayscale view; "flows" / "cellprob" swap the
  // base for cellpose's intermediate visualisations (Tier 2).  All
  // tools and toggles still work on top of any view.
  type ViewMode = "composite" | "r" | "g" | "b" | "flows" | "cellprob";
  const [viewMode, setViewMode] = useState<ViewMode>("composite");
  // Show a translucent reference circle at the cellpose `diameter` so
  // the user can sanity-check whether their cells look about that big.
  const [scaleDiskOn, setScaleDiskOn] = useState(false);
  // ── Interactive mask editor (cellpose mode only) ───────────────
  // The active tool: "pan" is the default and matches the existing
  // click-drag-to-pan behaviour; "paint" draws a new cell with a
  // round brush stroke; "delete" zeroes whichever cell the user
  // clicks; "merge" reassigns one cell's pixels to the ID of another.
  type EditTool = "pan" | "paint" | "erase" | "delete" | "merge";
  const [editTool, setEditTool] = useState<EditTool>("pan");
  // Simple-mode edit target — which channel's binary mask the paint /
  // erase tools operate on.  Hidden when in cellpose mode (where the
  // tools operate on cellLabels instead).  Disabled-checkbox channels
  // are skipped at run time so editing them is a no-op; the picker
  // still shows them so the user can paint into a previously-disabled
  // channel and then enable it.
  const [simpleEditChannel, setSimpleEditChannel] = useState<"r" | "g" | "b">("r");
  // Brush radius (preview-pixel space).  Cellpose's GUI uses 3 / 5 / 7;
  // 12 is a forgiving default for human pointing.  `[` / `]` adjusts.
  const [brushPx, setBrushPx] = useState(12);
  // Merge-tool state: the ID of the first cell clicked, awaiting the
  // second click that completes the merge.  null = no pending merge.
  const [mergeFirstId, setMergeFirstId] = useState<number | null>(null);
  // Hover cursor position + the cell ID under it (for the tooltip).
  const [hoverInfo, setHoverInfo] = useState<{
    x: number; y: number; clientX: number; clientY: number; cellId: number;
  } | null>(null);
  // Undo / redo: per-image stacks of {cells, nuclei} label snapshots.
  // We keep the active image's stack only; switching images flushes
  // the other image's history to keep peak memory bounded.
  type EditSnapshot = { cellLabels: Int32Array; nucleusLabels?: Int32Array };
  const [editHistory, setEditHistory] = useState<Record<string, { past: EditSnapshot[]; future: EditSnapshot[] }>>({});
  // Edited counts per image — surfaced in the stats footer so the
  // user can see at a glance which images have been touched.
  const editsMeta = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(editHistory)) out[k] = v.past.length;
    return out;
  }, [editHistory]);
  // Run-all progress: { current, total } while iterating images.
  // null when idle. cancelRef lets the user abort mid-batch (close
  // the dialog or click again).
  const [runAllProgress, setRunAllProgress] = useState<{ current: number; total: number } | null>(null);
  const runAllCancelRef = useRef(false);
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
  // problem (the 'No module named …' / "Cellpose isn't installed"
  // pattern). Exclude timeouts — a slow inference doesn't mean the venv
  // is broken; offering a 5-15 min reinstall on a timeout was bad UX.
  const showRepair = !!previewError
    && !/timed out|timeout/i.test(previewError)
    && (
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
      // Reinstall the version this dialog is currently configured to
      // use — so a v3 user doesn't accidentally get v4 reinstalled
      // and lose their cyto3 + nuclei model zoo.
      const ver = cfg.cellpose.cellposeVersion || "v4";
      const resp = await fetch(`http://127.0.0.1:8765/api/analysis/install-cellpose-stream?version=${ver}`, {
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
      setEditTool("pan");
      setEditHistory({});
      setMergeFirstId(null);
      setHoverInfo(null);
      setViewMode("composite");
      setScaleDiskOn(false);
    }
  }, [open, initial]);

  const activeImage = images[activeIdx] || null;

  // Distinct images currently upstream — chips for group assignment.
  // Pulled from props.images so the group panel matches what the
  // cycler is showing.  Keyed by stable `id` (input key) rather than
  // display label, so two images that share a name (e.g. "image1"
  // from two upstream nodes) stay distinct for group-assignment.
  const imageEntries = useMemo(
    () => images.map((i) => ({ id: i.id, label: i.label.trim() || i.id })),
    [images],
  );
  // Per-image group lookup keyed by id (last group wins if duplicated;
  // the UI prevents that).  Accepts legacy label entries too — saved
  // configs from before 0.1.330 stored labels in cfg.groups[].images.
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
  const fetchPreview = useCallback(async (forImage?: FluorPickerImage) => {
    const img = forImage || activeImage;
    if (!open || !img) return;
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
        image_b64: img.image_b64 || "",
        source: img.source
          ? {
              key: img.source.key,
              row: img.source.row,
              col: img.source.col,
              inset_index: img.source.inset_index,
              name: img.source.name,
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
          cellpose_version: cfg.cellpose.cellposeVersion || "v4",
          model: cfg.cellpose.model,
          diameter: cfg.cellpose.diameter,
          seg_channel: cfg.cellpose.segChannel,
          nuclei_channel: cfg.cellpose.nucleiChannel || null,
          min_size: cfg.cellpose.minSize,
          measure_compartments: !!cfg.cellpose.measureCompartments,
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
          cellOverlaySrc: data.cell_overlay_b64
            ? `data:image/png;base64,${data.cell_overlay_b64}` : undefined,
          nucleusOverlaySrc: data.nucleus_overlay_b64
            ? `data:image/png;base64,${data.nucleus_overlay_b64}` : undefined,
          flowsRgbSrc: data.flows_rgb_b64
            ? `data:image/png;base64,${data.flows_rgb_b64}` : undefined,
          cellprobSrc: data.cellprob_b64
            ? `data:image/png;base64,${data.cellprob_b64}` : undefined,
          n_cells: typeof data.n_cells === "number" ? data.n_cells : undefined,
          n_nuclei: typeof data.n_nuclei === "number" ? data.n_nuclei : undefined,
          per_channel: data.per_channel,
        };
        // Decode editable label rasters (cellpose mode only).  These
        // power the brush / delete / merge tools — the dialog stores
        // them as Int32Arrays and re-derives the boundary canvas on
        // each edit.  Async (PNG decode goes through a hidden canvas).
        if (data.cell_labels_b64) {
          const dec = await decodeRgbaLabels(data.cell_labels_b64);
          if (dec) { layers.cellLabels = dec.labels; layers.labelW = dec.w; layers.labelH = dec.h; }
        }
        if (data.nucleus_labels_b64) {
          const dec = await decodeRgbaLabels(data.nucleus_labels_b64);
          if (dec) { layers.nucleusLabels = dec.labels; }
        }
        // Simple-mode per-channel binary masks — Uint8Array per
        // channel for the paint/erase tools.
        if (data.channel_masks_b64 && typeof data.channel_masks_b64 === "object") {
          const cm = data.channel_masks_b64 as Record<string, string>;
          const out: Partial<Record<"r" | "g" | "b", Uint8Array>> = {};
          for (const k of ["r", "g", "b"] as const) {
            if (!cm[k]) continue;
            const dec = await decodeBinaryMaskPng(cm[k]);
            if (dec) {
              out[k] = dec.mask;
              // Use the channel mask dims as labelW/H for the editor
              // when no cellpose labels are present (simple mode).
              if (!layers.labelW) { layers.labelW = dec.w; layers.labelH = dec.h; }
            }
          }
          if (Object.keys(out).length > 0) layers.channelMasks = out;
        }
        // Cache by THIS image's label so Run-all stores results
        // against each image's own label, not the (stale) activeIdx.
        const key = img.label;
        setPreviewByImage((cur) => ({ ...cur, [key]: layers }));
        // Fresh preview → clear the edit history for this image.
        setEditHistory((cur) => { const c = { ...cur }; delete c[key]; return c; });
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
            ? `Cellpose timed out (>10 min). ${
                cfg.cellpose.model === "cpsam"
                  ? "cpsam is the slowest model — try 'cyto3' in the Model dropdown (~3x faster, ~25 MB vs ~100 MB) or"
                  : "Check your internet (first run downloads the model), or"
              } close the dialog and try again — the model is cached after a successful run.`
            : "Preview timed out (>60 s). The sidecar may be busy or still loading scientific libraries on a cold start — try again.");
        }
        return;
      }
      setPreviewError(String((e as { message?: string })?.message ?? e));
    } finally {
      setPreviewLoading(false);
    }
  }, [open, activeImage, cfg.mode, cfg.rollingRadius, cfg.thresholds, cfg.cellpose]);

  // Run preview SEQUENTIALLY across every image. The cellpose model
  // loads only once for the batch (Python is single-threaded server-
  // side; we just queue requests here from JS) so this is ~as cheap
  // as running each manually but without the user having to cycle +
  // click N times. cancelRef + the run state let the user abort
  // mid-batch by clicking Stop (or closing the dialog).
  const runAll = useCallback(async () => {
    if (!open || images.length === 0) return;
    if (runAllProgress) {
      // Already running — second click = cancel.
      runAllCancelRef.current = true;
      return;
    }
    runAllCancelRef.current = false;
    setRunAllProgress({ current: 0, total: images.length });
    for (let i = 0; i < images.length; i++) {
      if (runAllCancelRef.current) break;
      setRunAllProgress({ current: i + 1, total: images.length });
      // Surface the in-flight image in the cycler so the user can
      // watch progress (also primes activePreview for that image).
      setActiveIdx(i);
      await fetchPreview(images[i]);
      // Brief breather to keep the UI responsive between calls.
      if (i < images.length - 1) await new Promise((r) => setTimeout(r, 60));
    }
    setRunAllProgress(null);
    runAllCancelRef.current = false;
  }, [open, images, runAllProgress, fetchPreview]);

  // Cycling images: clear the inline error and dirty-state, but keep
  // the cache so we can restore THIS image's previous overlay below.
  useEffect(() => {
    if (!open) return;
    setPreviewError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIdx]);

  // Param changes invalidate the whole cache — every previously-rendered
  // image's overlay is now stale relative to the current settings.
  // Also resets editor history; re-running cellpose with new params
  // would produce a fresh label image that wouldn't share IDs with
  // the user's prior edits.
  useEffect(() => {
    if (!open) return;
    setPreviewByImage({});
    setEditHistory({});
    setMergeFirstId(null);
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
  // Cellpose mode mask layers — separate <Image> handles for the cell
  // outline (yellow) and nucleus outline (cyan, when measure
  // compartments was on). Loaded into HTMLImageElements so the canvas
  // composite is a synchronous drawImage on every visibility toggle.
  const [cellImgEl, setCellImgEl] = useState<HTMLImageElement | null>(null);
  const [nucImgEl, setNucImgEl] = useState<HTMLImageElement | null>(null);
  // Tier-2 view layers — cellpose's flow direction RGB + cell-probability
  // heatmap.  Loaded only on demand (when the corresponding view mode is
  // chosen) but cached so toggling between views is instant.
  const [flowsImgEl, setFlowsImgEl] = useState<HTMLImageElement | null>(null);
  const [cellprobImgEl, setCellprobImgEl] = useState<HTMLImageElement | null>(null);
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
  // Convert a DOM mouse event into IMAGE-space pixel coordinates
  // (the canvas's intrinsic resolution).  Returns null when the
  // pointer is outside the canvas bounds — which means the user is
  // hovering the dark padding area around the image.
  const clientToImage = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    const cnv = canvasRef.current;
    if (!cnv) return null;
    const rect = cnv.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right) return null;
    if (clientY < rect.top || clientY > rect.bottom) return null;
    const x = Math.round((clientX - rect.left) * (cnv.width / rect.width));
    const y = Math.round((clientY - rect.top) * (cnv.height / rect.height));
    return { x, y };
  }, []);

  // ── Editor: commits ─────────────────────────────────────────────
  // Each commit pushes a snapshot of the current cellLabels (and
  // nucleusLabels) onto the per-image undo stack, then mutates a
  // CLONE of the labels and writes it back into previewByImage.
  const pushHistorySnapshot = useCallback((label: string) => {
    setEditHistory((cur) => {
      const ap = previewByImage[label];
      if (!ap || !ap.cellLabels) return cur;
      const prev = cur[label] || { past: [], future: [] };
      const snap: EditSnapshot = {
        cellLabels: new Int32Array(ap.cellLabels),
        nucleusLabels: ap.nucleusLabels ? new Int32Array(ap.nucleusLabels) : undefined,
      };
      // Cap the per-image history depth at 20 — each Int32 snapshot
      // of a 1024×1024 mask is ~4 MB so unbounded growth would chew
      // through memory fast.
      const past = [...prev.past, snap].slice(-20);
      return { ...cur, [label]: { past, future: [] } };
    });
  }, [previewByImage]);

  // Simple-mode writeback: update ONE channel's binary mask in
  // previewByImage so the boundary canvas re-derives and the
  // compositor redraws.  Other channels keep their existing masks.
  const writeBackChannelMask = useCallback((label: string, ch: "r" | "g" | "b", next: Uint8Array) => {
    setPreviewByImage((cur) => {
      const ap = cur[label];
      if (!ap) return cur;
      const cm = { ...(ap.channelMasks || {}), [ch]: next };
      return { ...cur, [label]: { ...ap, channelMasks: cm, edited: true } };
    });
  }, []);

  const writeBackLabels = useCallback((label: string, next: Int32Array, nextNuc?: Int32Array | null) => {
    setPreviewByImage((cur) => {
      const ap = cur[label];
      if (!ap) return cur;
      return {
        ...cur,
        [label]: {
          ...ap,
          cellLabels: next,
          nucleusLabels: nextNuc === null ? undefined : (nextNuc || ap.nucleusLabels),
          edited: true,
          n_cells: countNonZeroIds(next),
        },
      };
    });
  }, []);

  // Paint a filled circle into the stroke mask at image-space (x, y).
  const stampCircle = useCallback((cx: number, cy: number, r: number, w: number, h: number) => {
    const mask = strokeMaskRef.current;
    if (!mask) return;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(h - 1, Math.ceil(cy + r));
    const r2 = r * r;
    for (let y = y0; y <= y1; y++) {
      const dy = y - cy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        if (dx * dx + dy * dy <= r2) mask[y * w + x] = 1;
      }
    }
  }, []);

  // Stamp circles along the segment between the previous and current
  // pointer positions so a fast drag still produces a continuous
  // stroke (Bresenham would do; circles every pixel is plenty).
  const paintSegment = useCallback((x: number, y: number, w: number, h: number) => {
    const r = brushPx;
    const last = lastPaintXYRef.current;
    if (!last) {
      stampCircle(x, y, r, w, h);
    } else {
      const dx = x - last.x, dy = y - last.y;
      const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        stampCircle(last.x + dx * t, last.y + dy * t, r, w, h);
      }
    }
    lastPaintXYRef.current = { x, y };
    setStrokeTick((t) => t + 1);
  }, [brushPx, stampCircle]);

  const commitPaintStroke = useCallback((label: string) => {
    const mask = strokeMaskRef.current;
    const ap = previewByImage[label];
    if (!mask || !ap?.labelW || !ap.labelH) {
      strokeMaskRef.current = null;
      paintingRef.current = false;
      lastPaintXYRef.current = null;
      setStrokeTick((t) => t + 1);
      return;
    }
    // ── Simple-mode branch: edit ONE channel's binary mask. ──
    // Cellpose's labeled-blob semantics don't apply; paint just
    // sets, erase just clears.  No new ID allocation, no merge
    // behaviour.  Undo history is per-channel-mask too — pushed
    // BEFORE the mutation so the user can revert exactly one step.
    if (cfg.mode === "simple") {
      const ch = simpleEditChannel;
      const cur = ap.channelMasks?.[ch];
      if (!cur) {
        strokeMaskRef.current = null;
        paintingRef.current = false;
        lastPaintXYRef.current = null;
        setStrokeTick((t) => t + 1);
        return;
      }
      const next = new Uint8Array(cur);
      let touched = 0;
      if (paintingToolRef.current === "erase") {
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i]) continue;
          if (next[i]) { next[i] = 0; touched++; }
        }
      } else {
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i]) continue;
          if (!next[i]) { next[i] = 1; touched++; }
        }
      }
      if (touched > 0) writeBackChannelMask(label, ch, next);
      strokeMaskRef.current = null;
      paintingRef.current = false;
      lastPaintXYRef.current = null;
      setStrokeTick((t) => t + 1);
      return;
    }
    if (!ap.cellLabels) {
      strokeMaskRef.current = null;
      paintingRef.current = false;
      lastPaintXYRef.current = null;
      setStrokeTick((t) => t + 1);
      return;
    }
    pushHistorySnapshot(label);
    const next = new Int32Array(ap.cellLabels);
    let touched = 0;
    if (paintingToolRef.current === "erase") {
      // Erase tool: set every pixel under the stroke to background.
      // Useful for pulling a cell's boundary BACK (the inverse of
      // paint-extend).  Doesn't delete an entire cell — it just
      // shaves whichever pixels the user dragged over.
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        if (next[i] !== 0) { next[i] = 0; touched++; }
      }
    } else {
      // Paint tool: if the stroke STARTED on an existing cell, every
      // stroke pixel is assigned that cell's ID — extending /
      // pushing the cell's boundary along the brush path.  Useful for
      // dendritic cells where cellpose under-segments processes.
      // Stroke starting in background → allocate a fresh cell ID.
      let targetId = paintStartIdRef.current | 0;
      if (targetId === 0) {
        let maxId = 0;
        for (let i = 0; i < next.length; i++) if (next[i] > maxId) maxId = next[i];
        targetId = maxId + 1;
      }
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        if (next[i] !== targetId) { next[i] = targetId; touched++; }
      }
    }
    if (touched > 0) writeBackLabels(label, next);
    strokeMaskRef.current = null;
    paintingRef.current = false;
    lastPaintXYRef.current = null;
    paintStartIdRef.current = 0;
    setStrokeTick((t) => t + 1);
  }, [previewByImage, pushHistorySnapshot, writeBackLabels]);

  const deleteCellAt = useCallback((label: string, x: number, y: number) => {
    const ap = previewByImage[label];
    if (!ap?.cellLabels || !ap.labelW || !ap.labelH) return;
    const idx = y * ap.labelW + x;
    const id = ap.cellLabels[idx];
    if (!id) return;
    pushHistorySnapshot(label);
    const next = new Int32Array(ap.cellLabels);
    for (let i = 0; i < next.length; i++) if (next[i] === id) next[i] = 0;
    writeBackLabels(label, next);
  }, [previewByImage, pushHistorySnapshot, writeBackLabels]);

  const mergeCellInto = useCallback((label: string, srcId: number, dstId: number) => {
    if (srcId === dstId || !srcId || !dstId) return;
    const ap = previewByImage[label];
    if (!ap?.cellLabels) return;
    pushHistorySnapshot(label);
    const next = new Int32Array(ap.cellLabels);
    for (let i = 0; i < next.length; i++) if (next[i] === srcId) next[i] = dstId;
    writeBackLabels(label, next);
  }, [previewByImage, pushHistorySnapshot, writeBackLabels]);

  const clearAllMasks = useCallback((label: string) => {
    const ap = previewByImage[label];
    if (!ap?.cellLabels) return;
    pushHistorySnapshot(label);
    writeBackLabels(label, new Int32Array(ap.cellLabels.length));
  }, [previewByImage, pushHistorySnapshot, writeBackLabels]);

  const doUndo = useCallback((label: string) => {
    setEditHistory((cur) => {
      const h = cur[label];
      if (!h || h.past.length === 0) return cur;
      const ap = previewByImage[label];
      if (!ap?.cellLabels) return cur;
      const snap = h.past[h.past.length - 1];
      const present: EditSnapshot = {
        cellLabels: new Int32Array(ap.cellLabels),
        nucleusLabels: ap.nucleusLabels ? new Int32Array(ap.nucleusLabels) : undefined,
      };
      writeBackLabels(label, snap.cellLabels, snap.nucleusLabels ?? null);
      return {
        ...cur,
        [label]: { past: h.past.slice(0, -1), future: [...h.future, present] },
      };
    });
  }, [previewByImage, writeBackLabels]);

  const doRedo = useCallback((label: string) => {
    setEditHistory((cur) => {
      const h = cur[label];
      if (!h || h.future.length === 0) return cur;
      const ap = previewByImage[label];
      if (!ap?.cellLabels) return cur;
      const snap = h.future[h.future.length - 1];
      const present: EditSnapshot = {
        cellLabels: new Int32Array(ap.cellLabels),
        nucleusLabels: ap.nucleusLabels ? new Int32Array(ap.nucleusLabels) : undefined,
      };
      writeBackLabels(label, snap.cellLabels, snap.nucleusLabels ?? null);
      return {
        ...cur,
        [label]: { past: [...h.past, present], future: h.future.slice(0, -1) },
      };
    });
  }, [previewByImage, writeBackLabels]);

  // ── Mouse dispatcher ─────────────────────────────────────────
  // Drives pan in the default tool, paint / delete / merge otherwise.
  // We don't disable wheel zoom or double-click reset — those stay
  // available regardless of tool.
  const onPreviewMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const label = activeImage?.label;
    const ap = label ? previewByImage[label] : undefined;
    // Editable in cellpose mode when cellLabels are decoded; editable
    // in simple mode when the CURRENTLY-SELECTED channel has a binary
    // mask.  Delete + merge only make sense for cellpose (labeled
    // blobs) — paint + erase are valid in both modes.
    const cellposeEditable = cfg.mode === "cellpose" && !!ap?.cellLabels;
    const simpleEditable = cfg.mode === "simple" && !!ap?.channelMasks?.[simpleEditChannel];
    const editable = (cellposeEditable || simpleEditable) && !!ap?.labelW && !!ap?.labelH;
    if (!editable || editTool === "pan") {
      // Existing pan behaviour.
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
      return;
    }
    e.preventDefault();
    const pt = clientToImage(e.clientX, e.clientY);
    if (!pt) return;
    const w = ap!.labelW!, h = ap!.labelH!;
    if (editTool === "paint" || editTool === "erase") {
      // Capture the under-cursor cell ID + the active tool at the
      // start of the stroke so changing tools mid-drag (or moving
      // off an existing cell during the drag) doesn't change the
      // commit semantics of THIS stroke.
      paintingToolRef.current = editTool === "erase" ? "erase" : "paint";
      // Cellpose: start id determines whether paint extends an
      // existing cell (>0) or makes a new one (0).  Simple mode:
      // no labeled blobs, so start id is always 0.
      paintStartIdRef.current =
        cfg.mode === "cellpose" && editTool === "paint"
          ? (ap!.cellLabels![pt.y * w + pt.x] | 0)
          : 0;
      strokeMaskRef.current = new Uint8Array(w * h);
      paintingRef.current = true;
      lastPaintXYRef.current = null;
      paintSegment(pt.x, pt.y, w, h);
      const onMove = (ev: MouseEvent) => {
        if (!paintingRef.current) return;
        const p = clientToImage(ev.clientX, ev.clientY);
        if (!p) return;
        paintSegment(p.x, p.y, w, h);
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (label) commitPaintStroke(label);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      return;
    }
    if (editTool === "delete" && label) {
      deleteCellAt(label, pt.x, pt.y);
      return;
    }
    if (editTool === "merge" && label) {
      const idx = pt.y * w + pt.x;
      const id = ap!.cellLabels![idx];
      if (!id) return;
      if (mergeFirstId == null) {
        setMergeFirstId(id);
      } else {
        mergeCellInto(label, id, mergeFirstId);
        setMergeFirstId(null);
      }
      return;
    }
  }, [view, activeImage, previewByImage, cfg.mode, editTool, clientToImage,
      paintSegment, commitPaintStroke, deleteCellAt, mergeCellInto, mergeFirstId,
      simpleEditChannel]);

  // Hover tracking — updates the brush cursor + cell-ID tooltip.
  const onPreviewMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (cfg.mode !== "cellpose") return;
    const label = activeImage?.label;
    const ap = label ? previewByImage[label] : undefined;
    if (!ap?.cellLabels || !ap.labelW || !ap.labelH) {
      setHoverInfo(null);
      return;
    }
    const pt = clientToImage(e.clientX, e.clientY);
    if (!pt) { setHoverInfo(null); return; }
    const cellId = ap.cellLabels[pt.y * ap.labelW + pt.x] | 0;
    setHoverInfo({ x: pt.x, y: pt.y, clientX: e.clientX, clientY: e.clientY, cellId });
  }, [cfg.mode, activeImage, previewByImage, clientToImage]);

  const onPreviewMouseLeave = useCallback(() => {
    setHoverInfo(null);
  }, []);

  // ── Hotkeys ──────────────────────────────────────────────────
  // Roughly mirror cellpose's GUI shortcuts: V=pan, B=paint, D=delete,
  // M=merge; [ / ] resize brush; Cmd+Z / Cmd+Shift+Z undo/redo;
  // Cmd+0 clears every mask on the active image.  We deliberately
  // SKIP hotkeys when focus is in a text input so typing channel
  // names doesn't toggle tools.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable)) return;
      const cmd = e.metaKey || e.ctrlKey;
      const label = activeImage?.label;
      if (cfg.mode !== "cellpose") return;
      if (cmd && (e.key === "z" || e.key === "Z")) {
        e.preventDefault();
        if (label) { if (e.shiftKey) doRedo(label); else doUndo(label); }
        return;
      }
      if (cmd && e.key === "0") {
        e.preventDefault();
        if (label) clearAllMasks(label);
        return;
      }
      if (e.key === "[") { e.preventDefault(); setBrushPx((p) => Math.max(2, p - 2)); return; }
      if (e.key === "]") { e.preventDefault(); setBrushPx((p) => Math.min(80, p + 2)); return; }
      const k = e.key.toLowerCase();
      if (k === "v") { setEditTool("pan"); setMergeFirstId(null); }
      else if (k === "b") { setEditTool("paint"); setMergeFirstId(null); }
      else if (k === "e") { setEditTool("erase"); setMergeFirstId(null); }
      else if (k === "d") { setEditTool("delete"); setMergeFirstId(null); }
      else if (k === "m") { setEditTool("merge"); setMergeFirstId(null); }
      else if (k === "escape") { setMergeFirstId(null); }
      // Tier-2 view-mode hotkeys (cellpose only).  We deliberately
      // SKIP R/G/B isolate keys — "B" is paint and "G" is too easy
      // to fat-finger; users pick those channels via the chip row
      // below the toolbar.  Available: W=composite, F=flows,
      // P=cell probability, S=scale disk.
      else if (k === "w") { setViewMode("composite"); }
      else if (k === "f") { setViewMode((v) => v === "flows" ? "composite" : "flows"); }
      else if (k === "p") { setViewMode((v) => v === "cellprob" ? "composite" : "cellprob"); }
      else if (k === "s") { setScaleDiskOn((p) => !p); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, cfg.mode, activeImage, doUndo, doRedo, clearAllMasks]);

  // ── Save: pack EVERY image's preview labels into cfg.editedMasks ──
  // Previously the dialog only saved masks the user had actively
  // painted on, which meant the workflow's Run step re-ran cellpose
  // from scratch on every "unedited" image — even ones the user had
  // already segmented in the preview.  That wasted ~30s of model
  // cold-start + ~1s/image of inference.
  //
  // Now we pack every image that has a cached preview, edited or
  // not.  The backend already supports per-image mask overrides
  // (cellpose_plugin's edited_label_paths); supplied masks skip
  // model.eval.  When ALL images have masks, the runner also skips
  // loading cellpose entirely (lazy-load — see runner code).
  //
  // Staleness is already prevented: every cellpose-relevant param
  // change clears previewByImage, so a non-empty preview entry is
  // by-definition current with the saved config.  Users who want
  // to FORCE a fresh full-res cellpose pass just re-open the
  // dialog, click Run preview (regenerates the labels), and save.
  const handleSave = useCallback(() => {
    // Cellpose masks (full per-image label rasters).
    const masks: Record<string, string> = {};
    // Simple-mode per-channel binary masks — only emitted for
    // (image, channel) pairs where the user actually edited (we
    // can't tell EDITED vs unmodified from the binary content
    // alone, so we conservatively emit every image's current
    // channel masks; the runtime fast-path then uses them
    // instead of re-running the threshold).
    const chMasks: Record<string, Partial<Record<"r" | "g" | "b", string>>> = {};
    for (const img of images) {
      const ap = previewByImage[img.label];
      if (!ap?.labelW || !ap.labelH) continue;
      if (ap.cellLabels) {
        const url = encodeRgbaLabels(ap.cellLabels, ap.labelW, ap.labelH);
        if (url) masks[img.label] = url;
      }
      if (ap.channelMasks) {
        const out: Partial<Record<"r" | "g" | "b", string>> = {};
        for (const k of ["r", "g", "b"] as const) {
          const m = ap.channelMasks[k];
          if (!m) continue;
          const url = encodeBinaryMaskPng(m, ap.labelW, ap.labelH);
          if (url) out[k] = url;
        }
        if (Object.keys(out).length > 0) chMasks[img.label] = out;
      }
    }
    onSave({
      ...cfg,
      editedMasks: Object.keys(masks).length ? masks : undefined,
      editedChannelMasks: Object.keys(chMasks).length ? chMasks : undefined,
    });
  }, [cfg, images, previewByImage, onSave]);

  // Load composite + overlay layers (per-channel for simple, cells +
  // nuclei for cellpose) as HTMLImageElement instances whenever the
  // active preview changes. We hold them in state so visibility
  // toggles can re-composite without re-loading.
  useEffect(() => {
    if (!activePreview) {
      setCompImgEl(null); setChImgEls({});
      setCellImgEl(null); setNucImgEl(null);
      setFlowsImgEl(null); setCellprobImgEl(null);
      return;
    }
    let cancelled = false;
    // Base composite (no outlines) when available; fall back to the
    // legacy fused overlay for very old responses.
    const compSrc = activePreview.compositeSrc || activePreview.overlaySrc;
    if (compSrc) {
      const im = new window.Image();
      im.onload = () => { if (!cancelled) setCompImgEl(im); };
      im.src = compSrc;
    } else {
      setCompImgEl(null);
    }
    setChImgEls({});
    setCellImgEl(null);
    setNucImgEl(null);
    setFlowsImgEl(null);
    setCellprobImgEl(null);
    if (cfg.mode === "simple") {
      for (const k of ["r", "g", "b"] as const) {
        const src = activePreview.channelOverlays?.[k];
        if (!src) continue;
        const im = new window.Image();
        im.onload = () => { if (!cancelled) setChImgEls((cur) => ({ ...cur, [k]: im })); };
        im.src = src;
      }
    } else {
      // cellpose: cell outline + (optional) nucleus outline
      if (activePreview.cellOverlaySrc) {
        const im = new window.Image();
        im.onload = () => { if (!cancelled) setCellImgEl(im); };
        im.src = activePreview.cellOverlaySrc;
      }
      if (activePreview.nucleusOverlaySrc) {
        const im = new window.Image();
        im.onload = () => { if (!cancelled) setNucImgEl(im); };
        im.src = activePreview.nucleusOverlaySrc;
      }
      if (activePreview.flowsRgbSrc) {
        const im = new window.Image();
        im.onload = () => { if (!cancelled) setFlowsImgEl(im); };
        im.src = activePreview.flowsRgbSrc;
      }
      if (activePreview.cellprobSrc) {
        const im = new window.Image();
        im.onload = () => { if (!cancelled) setCellprobImgEl(im); };
        im.src = activePreview.cellprobSrc;
      }
    }
    return () => { cancelled = true; };
  }, [activePreview, cfg.mode]);

  // Channel-isolation cache: when viewMode is "r" / "g" / "b" we
  // render compImgEl as a single-channel grayscale canvas.  Built
  // lazily and only when the user actually picks the isolated view.
  const channelGrayCache = useRef<Partial<Record<"r" | "g" | "b", HTMLCanvasElement>>>({});
  useEffect(() => { channelGrayCache.current = {}; }, [compImgEl]);
  const getChannelGray = useCallback((k: "r" | "g" | "b"): HTMLCanvasElement | null => {
    if (!compImgEl) return null;
    const hit = channelGrayCache.current[k];
    if (hit) return hit;
    const w = compImgEl.naturalWidth, h = compImgEl.naturalHeight;
    const cnv = document.createElement("canvas");
    cnv.width = w; cnv.height = h;
    const ctx = cnv.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(compImgEl, 0, 0);
    const id = ctx.getImageData(0, 0, w, h);
    const d = id.data;
    const ofs = k === "r" ? 0 : k === "g" ? 1 : 2;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i + ofs];
      d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    channelGrayCache.current[k] = cnv;
    return cnv;
  }, [compImgEl]);

  // ── Editor: derived boundary canvases (cellpose mode) ─────────
  // When the active preview carries an Int32Array of cell labels (i.e.
  // the new "editable" path), we re-derive the boundary client-side
  // on every change.  This is what makes paint / delete / merge appear
  // instantly — the server's overlay PNG is only used as the initial
  // bootstrap and falls back when labels weren't returned.
  const [cellBoundaryCnv, setCellBoundaryCnv] = useState<HTMLCanvasElement | null>(null);
  const [nucleusBoundaryCnv, setNucleusBoundaryCnv] = useState<HTMLCanvasElement | null>(null);
  // Simple-mode per-channel boundary canvases derived from the
  // editable binary masks.  When present, override chImgEls — the
  // server's pre-rendered overlay PNG is only used until the user
  // touches the mask.
  const [simpleBoundaryCnvs, setSimpleBoundaryCnvs] = useState<Partial<Record<"r" | "g" | "b", HTMLCanvasElement>>>({});
  useEffect(() => {
    if (!activePreview || !activePreview.cellLabels || !activePreview.labelW || !activePreview.labelH) {
      setCellBoundaryCnv(null);
      setNucleusBoundaryCnv(null);
      return;
    }
    const w = activePreview.labelW, h = activePreview.labelH;
    const b = deriveBoundary(activePreview.cellLabels, w, h);
    setCellBoundaryCnv(renderBoundaryCanvas(b, w, h, [255, 255, 0]));
    if (activePreview.nucleusLabels) {
      const nb = deriveBoundary(activePreview.nucleusLabels, w, h);
      setNucleusBoundaryCnv(renderBoundaryCanvas(nb, w, h, [96, 220, 255]));
    } else {
      setNucleusBoundaryCnv(null);
    }
  }, [activePreview]);

  // Simple-mode per-channel boundary canvases — derived from the
  // editable Uint8Array masks.  Each channel rendered in its own
  // colour matching the existing per-channel scheme.  When the user
  // paints / erases, the binary mask updates and this effect re-runs.
  useEffect(() => {
    const ap = activePreview;
    if (!ap || cfg.mode !== "simple" || !ap.channelMasks || !ap.labelW || !ap.labelH) {
      setSimpleBoundaryCnvs({});
      return;
    }
    const out: Partial<Record<"r" | "g" | "b", HTMLCanvasElement>> = {};
    const colors: Record<"r" | "g" | "b", [number, number, number]> = {
      r: [255, 80, 80], g: [120, 220, 120], b: [120, 170, 255],
    };
    for (const k of ["r", "g", "b"] as const) {
      const m = ap.channelMasks[k];
      if (!m) continue;
      const bnd = deriveBinaryBoundary(m, ap.labelW, ap.labelH);
      out[k] = renderBoundaryCanvas(bnd, ap.labelW, ap.labelH, colors[k]);
    }
    setSimpleBoundaryCnvs(out);
  }, [activePreview, cfg.mode]);

  // ── Editor: transient paint stroke + brush-cursor overlays ─────
  // While the user holds left-mouse with the paint tool, every
  // sampled pointer position is stamped into strokeMaskRef as a
  // filled circle.  The compositor reads it as a semi-transparent
  // top layer so the user sees what they're drawing IMMEDIATELY.
  // On mouseup it's committed into cellLabels with a fresh ID.
  const strokeMaskRef = useRef<Uint8Array | null>(null);
  const paintingRef = useRef(false);
  // Which tool the IN-PROGRESS stroke belongs to ("paint" vs "erase").
  // Captured at mousedown so changing the tool mid-drag doesn't alter
  // the commit semantics for an active stroke.
  const paintingToolRef = useRef<"paint" | "erase">("paint");
  // The cell ID under the cursor at mousedown.  Drives "extend an
  // existing cell vs make a new one" — when the user starts the
  // paint stroke ON a cell (non-zero label), the whole stroke is
  // assigned that cell's ID, pushing its boundary outwards along
  // the brush path.  Set to 0 when the stroke started in background;
  // commit then allocates a fresh ID.
  const paintStartIdRef = useRef<number>(0);
  const lastPaintXYRef = useRef<{ x: number; y: number } | null>(null);
  // Bumped on every paint stamp so the compositor re-runs.  Cheaper
  // than copying the Int32Array on every move.
  const [strokeTick, setStrokeTick] = useState(0);

  // Redraw the canvas whenever the composite, any overlay, the
  // visibility map, or any editor state changes.  Pixel-perfect:
  // CSS scales the final canvas, but all layers share its grid.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!compImgEl) {
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
    // Base layer — Tier 2 view modes swap what gets drawn.  Channel
    // isolation re-uses the composite via getChannelGray; flows /
    // cellprob use their dedicated HTMLImageElement.  Cellpose-only
    // modes fall back to the composite when the matching layer
    // hasn't arrived (e.g. flows is skipped on user-edited masks).
    if (cfg.mode === "cellpose" && (viewMode === "flows" || viewMode === "cellprob")) {
      const im = viewMode === "flows" ? flowsImgEl : cellprobImgEl;
      if (im) {
        // The flows / cellprob layers can be at a different intrinsic
        // resolution than the composite (cellpose caps its inference
        // res).  Stretch to fit the canvas so they overlay 1:1.
        ctx.drawImage(im, 0, 0, w, h);
      } else {
        ctx.drawImage(compImgEl, 0, 0);
      }
    } else if (viewMode === "r" || viewMode === "g" || viewMode === "b") {
      const cnv = getChannelGray(viewMode);
      if (cnv) ctx.drawImage(cnv, 0, 0);
      else ctx.drawImage(compImgEl, 0, 0);
    } else {
      ctx.drawImage(compImgEl, 0, 0);
    }
    if (cfg.mode === "simple") {
      // Prefer the LOCALLY-derived boundary canvas (always current
      // with the user's paint / erase edits); fall back to the
      // server's per-channel overlay PNG when no edits have happened
      // yet (or when channelMasks weren't supplied).
      for (const k of ["r", "g", "b"] as const) {
        if (!maskVisible[k]) continue;
        const local = simpleBoundaryCnvs[k];
        if (local) { ctx.drawImage(local, 0, 0); continue; }
        const im = chImgEls[k];
        if (im) ctx.drawImage(im, 0, 0);
      }
    } else {
      // Prefer the locally-derived boundary (always current with
      // edits); fall back to the server's overlay PNG when we
      // haven't decoded labels for this image yet.
      if (cpMaskVisible.cells) {
        if (cellBoundaryCnv) ctx.drawImage(cellBoundaryCnv, 0, 0);
        else if (cellImgEl) ctx.drawImage(cellImgEl, 0, 0);
      }
      if (cpMaskVisible.nuclei) {
        if (nucleusBoundaryCnv) ctx.drawImage(nucleusBoundaryCnv, 0, 0);
        else if (nucImgEl) ctx.drawImage(nucImgEl, 0, 0);
      }
      // Transient paint / erase stroke (semi-transparent fill).
      // Critical: putImageData on the main ctx REPLACES every pixel
      // (alpha is honoured but pixel-for-pixel — the (0,0,0,0) gaps
      // wipe the base layer to transparent → the user saw a black
      // canvas while painting).  Build a temp canvas, put the stroke
      // image there, then drawImage it so the gaps are honest no-ops
      // and the underlying composite shows through.
      const stroke = strokeMaskRef.current;
      if (stroke && stroke.length === w * h) {
        const tmp = document.createElement("canvas");
        tmp.width = w; tmp.height = h;
        const tctx = tmp.getContext("2d");
        if (tctx) {
          const id = tctx.createImageData(w, h);
          const d = id.data;
          // Yellow tint for "paint", red tint for "erase".  The
          // ref is set at mousedown; defaults to paint when missing.
          const isErase = paintingToolRef.current === "erase";
          const r = isErase ? 255 : 255;
          const g = isErase ? 60 : 255;
          const b = isErase ? 60 : 80;
          for (let i = 0; i < stroke.length; i++) {
            if (!stroke[i]) continue;
            d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 130;
          }
          tctx.putImageData(id, 0, 0);
          ctx.drawImage(tmp, 0, 0);
        }
      }
      // Highlight the cell that's mid-merge (waiting for second click).
      // Same drawImage-via-temp pattern so the highlight blends
      // instead of erasing everything around it.
      if (mergeFirstId != null && activePreview?.cellLabels) {
        const lbl = activePreview.cellLabels;
        const tmp = document.createElement("canvas");
        tmp.width = w; tmp.height = h;
        const tctx = tmp.getContext("2d");
        if (tctx) {
          const id = tctx.createImageData(w, h);
          const d = id.data;
          for (let i = 0; i < lbl.length; i++) {
            if (lbl[i] !== mergeFirstId) continue;
            d[i * 4] = 255; d[i * 4 + 1] = 80; d[i * 4 + 2] = 255; d[i * 4 + 3] = 100;
          }
          tctx.putImageData(id, 0, 0);
          ctx.drawImage(tmp, 0, 0);
        }
      }
      // Scale disk — translucent circle of `diameter` px in the
      // bottom-right corner, so the user can sanity-check whether
      // their cells look about the diameter they've entered.
      if (scaleDiskOn && cfg.cellpose.diameter > 0) {
        const r = Math.max(2, cfg.cellpose.diameter / 2);
        const cx = w - r - 24, cy = h - r - 24;
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,180,80,0.85)";
        ctx.lineWidth = 2;
        ctx.fillStyle = "rgba(255,180,80,0.18)";
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "rgba(255,180,80,0.95)";
        ctx.font = "bold 12px system-ui, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(`Ø ${Math.round(cfg.cellpose.diameter)} px`, cx + r, cy - r - 4);
        ctx.restore();
      }
      // Brush cursor — only when in an editing tool with hover info.
      if (hoverInfo && editTool !== "pan") {
        const cx = hoverInfo.x, cy = hoverInfo.y;
        ctx.save();
        ctx.lineWidth = 1.5;
        const col = editTool === "paint" ? "#fff58a"
                  : editTool === "erase" ? "#ff5050"
                  : editTool === "delete" ? "#ff7a7a"
                  : editTool === "merge" ? "#ff8aff" : "#ffffff";
        ctx.strokeStyle = col;
        if (editTool === "paint" || editTool === "erase") {
          ctx.beginPath();
          ctx.arc(cx, cy, brushPx, 0, Math.PI * 2);
          ctx.stroke();
        } else {
          // Crosshair for click-tools so the user knows there's no
          // brush radius to think about.
          ctx.beginPath();
          ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
          ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
          ctx.stroke();
        }
        ctx.restore();
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compImgEl, chImgEls, maskVisible, cellImgEl, nucImgEl, cpMaskVisible, cfg.mode,
      cellBoundaryCnv, nucleusBoundaryCnv, hoverInfo, editTool, brushPx, mergeFirstId,
      strokeTick, viewMode, flowsImgEl, cellprobImgEl, scaleDiskOn, cfg.cellpose.diameter,
      simpleBoundaryCnvs]);

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
  const toggleImageInGroup = useCallback((gid: string, imageId: string) => {
    setCfg((c) => ({
      ...c,
      groups: c.groups.map((g) => {
        if (g.id === gid) {
          const has = g.images.includes(imageId);
          return { ...g, images: has ? g.images.filter((x) => x !== imageId) : [...g.images, imageId] };
        }
        // Enforce single membership — remove from other groups when adding here.
        return { ...g, images: g.images.filter((x) => x !== imageId) };
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
          <Box sx={{ display: "flex", gap: 0.75 }}>
            <Button
              variant={paramsDirty ? "contained" : "outlined"}
              color={paramsDirty ? "primary" : "inherit"}
              startIcon={previewLoading && !runAllProgress
                ? <CircularProgress size={14} color="inherit" />
                : <RefreshIcon sx={{ fontSize: 18 }} />}
              onClick={() => void fetchPreview()}
              disabled={!activeImage || previewLoading || !!runAllProgress}
              sx={{ textTransform: "none", fontWeight: 700, py: 0.5, flex: 1 }}
            >
              {previewLoading && !runAllProgress
                ? (cfg.mode === "cellpose"
                    ? `Running Cellpose… ${elapsedSec}s${elapsedSec < 10 ? " (loading model)" : elapsedSec < 30 ? " (inference)" : ""}`
                    : `Running… ${elapsedSec}s`)
                : (paramsDirty
                    ? `Run on this image`
                    : "Re-run on this image")}
            </Button>
            {/* Run-all: iterates every wired image through fetchPreview.
                For cellpose this is dramatically faster than the user
                clicking through each image one at a time — the model
                doesn't reload between images (Python subprocess in the
                sidecar caches it). Second click while running cancels. */}
            {images.length > 1 && (
              <Button
                variant={runAllProgress ? "contained" : "outlined"}
                color={runAllProgress ? "warning" : "inherit"}
                onClick={() => void runAll()}
                disabled={!activeImage}
                sx={{ textTransform: "none", fontWeight: 700, py: 0.5, minWidth: 140 }}
              >
                {runAllProgress
                  ? `Stop (${runAllProgress.current}/${runAllProgress.total})`
                  : `Run on all (${images.length})`}
              </Button>
            )}
          </Box>
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
            {/* ── Editor toolbar (cellpose mode, after a Run) ──
                Mirrors cellpose-GUI: pan / paint / delete / merge +
                undo / redo / clear, with a brush-size slider.  Sits
                top-left so it doesn't clash with the zoom HUD on the
                right or the stats footer at the bottom.
                Hotkeys: V (pan), B (paint), D (delete), M (merge),
                [/] (brush ±2 px), Cmd+Z / Cmd+Shift+Z, Cmd+0. */}
            {((cfg.mode === "cellpose" && activePreview?.cellLabels)
              || (cfg.mode === "simple" && activePreview?.channelMasks)) && (
              <Box sx={{
                position: "absolute", top: 4, left: 4, zIndex: 5,
                display: "flex", alignItems: "center", gap: 0.4, flexWrap: "wrap",
                bgcolor: "rgba(0,0,0,0.55)", color: "common.white",
                px: 0.6, py: 0.35, borderRadius: 0.6,
              }}>
                {/* Simple-mode channel selector — picks which of R/G/B
                    the paint / erase tools modify.  Hidden in cellpose
                    mode (no per-channel concept there).  Disabled
                    channels still appear so the user can paint into
                    one even before re-enabling it in the threshold UI. */}
                {cfg.mode === "simple" && (
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.3, mr: 0.4,
                             borderRight: "1px solid rgba(255,255,255,0.25)", pr: 0.4 }}>
                    <Typography variant="caption" sx={{ fontSize: "0.6rem", opacity: 0.85, mr: 0.2 }}>
                      Edit ch:
                    </Typography>
                    {(["r", "g", "b"] as const).map((k) => {
                      const sw = k === "r" ? "#ff6868" : k === "g" ? "#6fdc6f" : "#7fa3ff";
                      const active = simpleEditChannel === k;
                      const hasMask = !!activePreview?.channelMasks?.[k];
                      return (
                        <Tooltip key={k} title={hasMask
                          ? `Edit ${cfg.channels[k]} mask (${k.toUpperCase()})`
                          : `${cfg.channels[k]} has no mask yet — Run preview first`}>
                          <Box
                            onClick={() => { if (hasMask) setSimpleEditChannel(k); }}
                            sx={{
                              cursor: hasMask ? "pointer" : "not-allowed",
                              opacity: hasMask ? 1 : 0.35, userSelect: "none",
                              px: 0.5, py: 0.1, borderRadius: 0.4,
                              border: "1px solid",
                              borderColor: active ? sw : "rgba(255,255,255,0.25)",
                              bgcolor: active ? sw : "transparent",
                              color: active ? "common.white" : "rgba(255,255,255,0.85)",
                              fontSize: "0.65rem", fontWeight: active ? 700 : 500,
                              minWidth: 18, textAlign: "center",
                            }}>
                            {k.toUpperCase()}
                          </Box>
                        </Tooltip>
                      );
                    })}
                  </Box>
                )}
                {([
                  { tool: "pan" as const,    icon: <PanToolAltIcon sx={{ fontSize: 16 }} />,    title: "Pan (V)", cellposeOnly: false },
                  { tool: "paint" as const,  icon: <BrushIcon sx={{ fontSize: 16 }} />,
                    title: cfg.mode === "simple"
                      ? "Paint (B) — adds pixels to the SELECTED channel's mask along the brush path"
                      : "Paint (B) — start on an EXISTING cell to extend / push its boundary along the brush path (great for dendrites); start in empty area to make a NEW cell",
                    cellposeOnly: false },
                  { tool: "erase" as const,  icon: <AutoFixOffIcon sx={{ fontSize: 16 }} />,
                    title: cfg.mode === "simple"
                      ? "Erase (E) — removes pixels from the SELECTED channel's mask along the brush path"
                      : "Erase pixels (E) — pulls a cell's boundary BACK by setting brushed pixels to background.  Doesn't remove the whole cell — for that, use Delete.",
                    cellposeOnly: false },
                  { tool: "delete" as const, icon: <HighlightOffIcon sx={{ fontSize: 16 }} />,  title: "Delete cell under cursor (D)", cellposeOnly: true },
                  { tool: "merge" as const,  icon: <CallMergeIcon sx={{ fontSize: 16 }} />,     title: "Merge two cells (M) — click first cell, then the cell to merge it INTO", cellposeOnly: true },
                ] as const).filter(({ cellposeOnly }) =>
                    !cellposeOnly || cfg.mode === "cellpose"
                ).map(({ tool, icon, title }) => (
                  <Tooltip key={tool} title={title}>
                    <Box
                      onClick={() => { setEditTool(tool); setMergeFirstId(null); }}
                      sx={{
                        cursor: "pointer", userSelect: "none",
                        px: 0.6, py: 0.2, borderRadius: 0.4,
                        bgcolor: editTool === tool ? "primary.main" : "transparent",
                        color: editTool === tool ? "common.white" : "rgba(255,255,255,0.85)",
                        border: "1px solid",
                        borderColor: editTool === tool ? "primary.main" : "rgba(255,255,255,0.25)",
                        display: "inline-flex", alignItems: "center",
                      }}>
                      {icon}
                    </Box>
                  </Tooltip>
                ))}
                {/* Brush radius (paint / erase tools).  Shows the
                    actual px radius so the user understands the
                    units; same control governs both tools. */}
                {(editTool === "paint" || editTool === "erase") && (
                  <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.3, ml: 0.4 }}>
                    <Typography variant="caption" sx={{ fontSize: "0.62rem", opacity: 0.85 }}>
                      brush {brushPx}px
                    </Typography>
                    <Box onClick={() => setBrushPx((p) => Math.max(2, p - 2))}
                      sx={{ cursor: "pointer", px: 0.5, py: 0.05, borderRadius: 0.3,
                            border: "1px solid rgba(255,255,255,0.25)", fontSize: "0.7rem", lineHeight: 1 }}>−</Box>
                    <Box onClick={() => setBrushPx((p) => Math.min(80, p + 2))}
                      sx={{ cursor: "pointer", px: 0.5, py: 0.05, borderRadius: 0.3,
                            border: "1px solid rgba(255,255,255,0.25)", fontSize: "0.7rem", lineHeight: 1 }}>+</Box>
                  </Box>
                )}
                {/* Spacer */}
                <Box sx={{ width: 1, alignSelf: "stretch", bgcolor: "rgba(255,255,255,0.2)", mx: 0.4 }} />
                {/* Undo / Redo / Clear.  Disabled when there's nothing
                    to act on — the chip styling reflects state. */}
                {(() => {
                  const lbl = activeImage?.label || "";
                  const h = editHistory[lbl];
                  const canUndo = !!h && h.past.length > 0;
                  const canRedo = !!h && h.future.length > 0;
                  return (
                    <>
                      <Tooltip title="Undo (Cmd/Ctrl+Z)">
                        <Box onClick={() => { if (canUndo) doUndo(lbl); }}
                          sx={{
                            cursor: canUndo ? "pointer" : "not-allowed",
                            opacity: canUndo ? 1 : 0.35,
                            px: 0.6, py: 0.2, borderRadius: 0.4,
                            border: "1px solid rgba(255,255,255,0.25)",
                            display: "inline-flex", alignItems: "center",
                          }}>
                          <UndoIcon sx={{ fontSize: 16 }} />
                        </Box>
                      </Tooltip>
                      <Tooltip title="Redo (Cmd/Ctrl+Shift+Z)">
                        <Box onClick={() => { if (canRedo) doRedo(lbl); }}
                          sx={{
                            cursor: canRedo ? "pointer" : "not-allowed",
                            opacity: canRedo ? 1 : 0.35,
                            px: 0.6, py: 0.2, borderRadius: 0.4,
                            border: "1px solid rgba(255,255,255,0.25)",
                            display: "inline-flex", alignItems: "center",
                          }}>
                          <RedoIcon sx={{ fontSize: 16 }} />
                        </Box>
                      </Tooltip>
                      <Tooltip title="Clear ALL masks on this image (Cmd/Ctrl+0)">
                        <Box onClick={() => { if (lbl) clearAllMasks(lbl); }}
                          sx={{
                            cursor: lbl ? "pointer" : "not-allowed",
                            px: 0.6, py: 0.2, borderRadius: 0.4,
                            border: "1px solid rgba(255,255,255,0.25)",
                            display: "inline-flex", alignItems: "center",
                          }}>
                          <LayersClearIcon sx={{ fontSize: 16 }} />
                        </Box>
                      </Tooltip>
                    </>
                  );
                })()}
                {/* "Edited" badge — surfaces the per-image edit count
                    so the user knows their work is captured. */}
                {(() => {
                  const lbl = activeImage?.label || "";
                  const n = editsMeta[lbl] || 0;
                  if (!n) return null;
                  return (
                    <Typography variant="caption" sx={{
                      ml: 0.6, fontSize: "0.62rem", color: "#ffd56a",
                      fontWeight: 700,
                    }}>
                      edited ✱ ({n})
                    </Typography>
                  );
                })()}
                {/* Merge-tool hint */}
                {editTool === "merge" && (
                  <Typography variant="caption" sx={{
                    ml: 0.6, fontSize: "0.62rem", color: "#ffb1ff", fontWeight: 600,
                  }}>
                    {mergeFirstId == null ? "click first cell…" : `merge → click target cell (Esc to cancel)`}
                  </Typography>
                )}
                {/* Hover cell ID */}
                {hoverInfo && hoverInfo.cellId > 0 && editTool !== "pan" && (
                  <Typography variant="caption" sx={{
                    ml: 0.6, fontSize: "0.62rem", opacity: 0.85,
                  }}>
                    cell #{hoverInfo.cellId}
                  </Typography>
                )}
              </Box>
            )}
            {/* Pan/zoom container — wraps the canvas (or fused-overlay
                <img> for cellpose) and a fallback thumbnail. Mouse-wheel
                zooms around the cursor, click-drag pans, double-click
                resets. Children center in the box and are transformed
                via CSS — the canvas itself keeps its native dimensions. */}
            <Box
              ref={previewBoxRef}
              onWheel={onPreviewWheel}
              onMouseDown={onPreviewMouseDown}
              onMouseMove={onPreviewMouseMove}
              onMouseLeave={onPreviewMouseLeave}
              onDoubleClick={resetView}
              sx={{
                position: "absolute", inset: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: previewLoading
                  ? "wait"
                  : (cfg.mode === "cellpose" && editTool !== "pan" && activePreview?.cellLabels
                      ? "crosshair"
                      : "grab"),
                "&:active": {
                  cursor: previewLoading
                    ? "wait"
                    : (cfg.mode === "cellpose" && editTool !== "pan" && activePreview?.cellLabels
                        ? "crosshair"
                        : "grabbing"),
                },
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
                  // Canvas-composited preview for BOTH modes when we
                  // have a composite layer + at least one overlay
                  // type. The compositor effect routes channelOverlays
                  // OR cell/nucleus overlays based on cfg.mode.
                  if (ap && (ap.compositeSrc || ap.overlaySrc)) {
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
            {/* ── Tier-2 view-mode chips (cellpose only) ──
                Sits at the bottom-left, above the stats footer, so it
                doesn't fight the edit toolbar or zoom HUD.  Lets the
                user swap between composite / R / G / B isolation,
                cellpose's flow direction RGB, the cell-probability
                heatmap, and a scale-disk overlay calibrated to the
                configured diameter.  Hotkeys: W (composite), F
                (flows), P (probability), S (scale disk). */}
            {cfg.mode === "cellpose" && activePreview && (
              <Box sx={{
                position: "absolute", bottom: 32, left: 4, zIndex: 5,
                display: "flex", alignItems: "center", gap: 0.3, flexWrap: "wrap",
                bgcolor: "rgba(0,0,0,0.5)", color: "common.white",
                px: 0.5, py: 0.25, borderRadius: 0.5,
                fontSize: "0.62rem",
              }}>
                {([
                  { mode: "composite" as const, label: "Composite",  hk: "W", color: "rgba(255,255,255,0.7)" },
                  { mode: "r" as const,        label: "R",           hk: "",  color: "#d35454" },
                  { mode: "g" as const,        label: "G",           hk: "",  color: "#5fa566" },
                  { mode: "b" as const,        label: "B",           hk: "",  color: "#5d80c0" },
                  { mode: "flows" as const,    label: "Flows",       hk: "F", color: "#c489ff" },
                  { mode: "cellprob" as const, label: "Cell prob.",  hk: "P", color: "#7fdc89" },
                ] as const).map(({ mode, label, hk, color }) => {
                  const active = viewMode === mode;
                  const disabled =
                    (mode === "flows" && !flowsImgEl)
                    || (mode === "cellprob" && !cellprobImgEl);
                  return (
                    <Tooltip key={mode} title={
                      disabled
                        ? `${label} not available (run cellpose first; edited masks skip flows/probability)`
                        : `Show ${label}${hk ? ` (${hk})` : ""}`}>
                      <Box
                        onClick={() => { if (!disabled) setViewMode(mode); }}
                        sx={{
                          cursor: disabled ? "not-allowed" : "pointer",
                          opacity: disabled ? 0.35 : 1,
                          userSelect: "none",
                          px: 0.55, py: 0.1, borderRadius: 0.3,
                          bgcolor: active ? color : "transparent",
                          color: active ? "common.white" : "rgba(255,255,255,0.85)",
                          border: "1px solid",
                          borderColor: active ? color : "rgba(255,255,255,0.25)",
                          fontWeight: active ? 700 : 500,
                          display: "inline-flex", alignItems: "center", gap: 0.2,
                        }}>
                        <span>{label}</span>
                      </Box>
                    </Tooltip>
                  );
                })}
                {/* Scale disk + fit-to-view + reset cluster */}
                <Box sx={{ width: 1, alignSelf: "stretch", bgcolor: "rgba(255,255,255,0.2)", mx: 0.3 }} />
                <Tooltip title="Show a calibration disk at the configured cellpose diameter (S)">
                  <Box
                    onClick={() => setScaleDiskOn((p) => !p)}
                    sx={{
                      cursor: "pointer", userSelect: "none",
                      px: 0.55, py: 0.1, borderRadius: 0.3,
                      bgcolor: scaleDiskOn ? "rgba(255,180,80,0.85)" : "transparent",
                      color: scaleDiskOn ? "common.white" : "rgba(255,255,255,0.85)",
                      border: "1px solid",
                      borderColor: scaleDiskOn ? "rgba(255,180,80,0.85)" : "rgba(255,255,255,0.25)",
                      fontWeight: scaleDiskOn ? 700 : 500,
                    }}>
                    ⊙ disk
                  </Box>
                </Tooltip>
                <Tooltip title="Fit image to view (also: double-click the image)">
                  <Box
                    onClick={resetView}
                    sx={{
                      cursor: "pointer", userSelect: "none",
                      px: 0.55, py: 0.1, borderRadius: 0.3,
                      border: "1px solid rgba(255,255,255,0.25)",
                    }}>
                    ⤢ fit
                  </Box>
                </Tooltip>
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
          {/* Cellpose-mode mask visibility row — analogue of the simple-
              mode toggles above. Cells (yellow) is always togglable when
              a preview exists; Nuclei (cyan) only appears when the user
              ran with "Measure compartments" on and the nuclei-pass
              actually produced labels. */}
          {cfg.mode === "cellpose" && activePreview && (activePreview.cellOverlaySrc || activePreview.nucleusOverlaySrc) && (
            <Box sx={{
              display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap",
              px: 0.5, py: 0.4, borderRadius: 0.5,
              bgcolor: "rgba(255,255,255,0.04)",
            }}>
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem", fontWeight: 600 }}>
                Show masks:
              </Typography>
              {activePreview.cellOverlaySrc && (
                <Box
                  onClick={() => setCpMaskVisible((p) => ({ ...p, cells: !p.cells }))}
                  sx={{
                    cursor: "pointer", fontSize: "0.66rem", px: 0.6, py: 0.15, borderRadius: 0.6,
                    display: "inline-flex", alignItems: "center", gap: 0.3, userSelect: "none",
                    bgcolor: cpMaskVisible.cells ? "#cdb336" : "transparent",
                    color: cpMaskVisible.cells ? "common.white" : "text.secondary",
                    border: "1px solid", borderColor: "#cdb336",
                    fontWeight: cpMaskVisible.cells ? 700 : 500,
                  }}>
                  <span style={{ fontSize: "0.85rem", lineHeight: 1 }}>{cpMaskVisible.cells ? "👁" : "·"}</span>
                  <span>Cells{typeof activePreview.n_cells === "number" ? ` (${activePreview.n_cells})` : ""}</span>
                </Box>
              )}
              {activePreview.nucleusOverlaySrc && (
                <Box
                  onClick={() => setCpMaskVisible((p) => ({ ...p, nuclei: !p.nuclei }))}
                  sx={{
                    cursor: "pointer", fontSize: "0.66rem", px: 0.6, py: 0.15, borderRadius: 0.6,
                    display: "inline-flex", alignItems: "center", gap: 0.3, userSelect: "none",
                    bgcolor: cpMaskVisible.nuclei ? "#5fbcdc" : "transparent",
                    color: cpMaskVisible.nuclei ? "common.white" : "text.secondary",
                    border: "1px solid", borderColor: "#5fbcdc",
                    fontWeight: cpMaskVisible.nuclei ? 700 : 500,
                  }}>
                  <span style={{ fontSize: "0.85rem", lineHeight: 1 }}>{cpMaskVisible.nuclei ? "👁" : "·"}</span>
                  <span>Nuclei{typeof activePreview.n_nuclei === "number" ? ` (${activePreview.n_nuclei})` : ""}</span>
                </Box>
              )}
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
              // Cellpose params. Each dropdown's explanation lives in a
              // small "ⓘ" icon-tooltip beside the label — wrapping the
              // whole TextField in a Tooltip used to capture hover and
              // block the click on the dropdown trigger itself.
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                {/* Version chip row — picks v3 (real cyto3 + nuclei
                    model zoo) or v4 (cpsam-only).  Routes to a
                    separate plugin venv per version; install each
                    independently from Plugins → Install Cellpose 3/4.
                    Switching version auto-snaps the model to the
                    default for that major. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.6, flexWrap: "wrap" }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.7rem" }}>
                    Cellpose
                  </Typography>
                  {([
                    { v: "v3" as const, label: "v3 (cyto3 / nuclei)", hint: "Real model zoo — smaller, faster, separate nuclei head" },
                    { v: "v4" as const, label: "v4 (cpsam)",            hint: "SAM-based generalist — one model, ~100 MB" },
                  ] as const).map(({ v, label, hint }) => {
                    const active = (cfg.cellpose.cellposeVersion || "v4") === v;
                    return (
                      <Tooltip key={v} title={hint}>
                        <Box
                          onClick={() => setCellpose({
                            cellposeVersion: v,
                            model: v === "v3" ? "cyto3" : "cpsam",
                          })}
                          sx={{
                            cursor: "pointer", userSelect: "none",
                            fontSize: "0.62rem", px: 0.6, py: 0.15, borderRadius: 0.5,
                            border: "1px solid",
                            borderColor: active ? "primary.main" : "divider",
                            bgcolor: active ? "primary.main" : "transparent",
                            color: active ? "primary.contrastText" : "text.secondary",
                            fontWeight: active ? 700 : 500,
                          }}>
                          {label}
                        </Box>
                      </Tooltip>
                    );
                  })}
                  <Typography variant="caption" sx={{ color: "text.disabled", ml: "auto", fontSize: "0.6rem" }}>
                    install via Plugins menu
                  </Typography>
                </Box>
                {/* Row 1 — model + cell-body channel.  Model list is
                    filtered by the version above; v3 surfaces the
                    full cyto + nuclei zoo, v4 only cpsam. */}
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
                  <TextField select size="small" label="Model" value={cfg.cellpose.model}
                    onChange={(e) => setCellpose({ model: e.target.value })}
                    inputProps={{ style: { fontSize: "0.78rem" } }}
                    InputLabelProps={{ style: { fontSize: "0.78rem" } }}
                    SelectProps={SELECT_MENU_PROPS}>
                    {(cfg.cellpose.cellposeVersion || "v4") === "v3" ? [
                      <MenuItem key="cyto3"  value="cyto3"  sx={{ fontSize: "0.78rem" }}>cyto3 (fast, default)</MenuItem>,
                      <MenuItem key="cyto2"  value="cyto2"  sx={{ fontSize: "0.78rem" }}>cyto2</MenuItem>,
                      <MenuItem key="cyto"   value="cyto"   sx={{ fontSize: "0.78rem" }}>cyto</MenuItem>,
                      <MenuItem key="nuclei" value="nuclei" sx={{ fontSize: "0.78rem" }}>nuclei (DAPI-only)</MenuItem>,
                    ] : [
                      <MenuItem key="cpsam" value="cpsam" sx={{ fontSize: "0.78rem" }}>cpsam (only v4 model)</MenuItem>,
                    ]}
                  </TextField>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
                    <TextField select size="small" label="Cell channel" value={cfg.cellpose.segChannel}
                      onChange={(e) => setCellpose({ segChannel: e.target.value as "r" | "g" | "b" })}
                      inputProps={{ style: { fontSize: "0.78rem" } }}
                      InputLabelProps={{ style: { fontSize: "0.78rem" } }}
                      SelectProps={SELECT_MENU_PROPS}
                      sx={{ flex: 1 }}>
                      <MenuItem value="r" sx={{ fontSize: "0.78rem" }}>{cfg.channels.r}</MenuItem>
                      <MenuItem value="g" sx={{ fontSize: "0.78rem" }}>{cfg.channels.g}</MenuItem>
                      <MenuItem value="b" sx={{ fontSize: "0.78rem" }}>{cfg.channels.b}</MenuItem>
                    </TextField>
                    <Tooltip title={(cfg.cellpose.cellposeVersion || "v4") === "v4"
                      ? "cpsam (cellpose 4) is a SAM-based generalist — it segments whichever objects are visible.  This dropdown still picks the channel cpsam isolates as its single grayscale input, but the v3-style dual-channel cyto+nuclei pairing doesn't apply (only the nuclei pass uses the Nuclei channel below)."
                      : "Channel cellpose uses as the cell-body / cytoplasm signal. cyto3 uses this PLUS the nuclei channel to find whole-cell boundaries. Intensity is measured in EVERY channel inside each detected cell — you don't need to pick this as the channel of interest, only the one whose signal best outlines cell bodies."}
                      placement="top" enterDelay={200}>
                      <Box sx={{
                        width: 16, height: 16, borderRadius: "50%",
                        bgcolor: "action.hover", color: "text.secondary",
                        fontSize: "0.7rem", display: "flex", alignItems: "center",
                        justifyContent: "center", cursor: "help", flexShrink: 0,
                      }}>ⓘ</Box>
                    </Tooltip>
                  </Box>
                </Box>
                {/* Row 2 — nuclei channel + compartment toggle. */}
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, alignItems: "center" }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
                    <TextField select size="small" label="Nuclei channel" value={cfg.cellpose.nucleiChannel ?? ""}
                      onChange={(e) => setCellpose({ nucleiChannel: (e.target.value || null) as "r" | "g" | "b" | null })}
                      inputProps={{ style: { fontSize: "0.78rem" } }}
                      InputLabelProps={{ style: { fontSize: "0.78rem" } }}
                      SelectProps={SELECT_MENU_PROPS}
                      sx={{ flex: 1 }}>
                      <MenuItem value="" sx={{ fontSize: "0.78rem" }}>— none —</MenuItem>
                      <MenuItem value="r" sx={{ fontSize: "0.78rem" }}>{cfg.channels.r}</MenuItem>
                      <MenuItem value="g" sx={{ fontSize: "0.78rem" }}>{cfg.channels.g}</MenuItem>
                      <MenuItem value="b" sx={{ fontSize: "0.78rem" }}>{cfg.channels.b}</MenuItem>
                    </TextField>
                    <Tooltip title={(cfg.cellpose.cellposeVersion || "v4") === "v4"
                      ? "DAPI / Hoechst channel.  cpsam (cellpose 4) does NOT use it for the cell-body pass (that's a cyto3-only feature).  Required ONLY for 'Measure compartments' — the second pass isolates this channel and segments the nuclei from it."
                      : "DAPI / Hoechst channel. Feeding cyto3 a nuclei channel as well gives noticeably better whole-cell boundaries (the model was trained on both inputs). Also REQUIRED for 'Measure compartments'."}
                      placement="top" enterDelay={200}>
                      <Box sx={{
                        width: 16, height: 16, borderRadius: "50%",
                        bgcolor: "action.hover", color: "text.secondary",
                        fontSize: "0.7rem", display: "flex", alignItems: "center",
                        justifyContent: "center", cursor: "help", flexShrink: 0,
                      }}>ⓘ</Box>
                    </Tooltip>
                  </Box>
                  <Box
                    onClick={() => {
                      if (!cfg.cellpose.nucleiChannel) return;
                      setCellpose({ measureCompartments: !cfg.cellpose.measureCompartments });
                    }}
                    title={cfg.cellpose.nucleiChannel
                      ? "When ON: also run cellpose's nuclei model on the nuclei channel, then for each cell emit THREE rows (whole_cell / nucleus / cytoplasm). Lets you compare nuclear vs cytoplasmic localisation. Roughly doubles cellpose time."
                      : "Pick a Nuclei channel first."}
                    sx={{
                      cursor: cfg.cellpose.nucleiChannel ? "pointer" : "not-allowed",
                      display: "flex", alignItems: "center", gap: 0.6,
                      px: 0.75, py: 0.55, borderRadius: 0.5,
                      border: "1px solid", borderColor: "divider",
                      opacity: cfg.cellpose.nucleiChannel ? 1 : 0.5,
                      userSelect: "none",
                      bgcolor: cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel
                        ? "primary.main" : "transparent",
                      color: cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel
                        ? "primary.contrastText" : "text.primary",
                    }}>
                    <Box sx={{
                      width: 14, height: 14, borderRadius: 0.4,
                      bgcolor: cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel
                        ? "common.white" : "transparent",
                      border: "1px solid",
                      borderColor: cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel
                        ? "common.white" : "text.secondary",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel
                        ? "primary.main" : "transparent",
                      fontSize: "0.78rem", fontWeight: 700,
                    }}>
                      {cfg.cellpose.measureCompartments && cfg.cellpose.nucleiChannel ? "✓" : ""}
                    </Box>
                    <Typography variant="caption" sx={{ fontSize: "0.72rem", fontWeight: 600 }}>
                      Measure compartments
                    </Typography>
                  </Box>
                </Box>
                {/* Auto-by-default override row. alignItems: "center" so
                    the "Diameter + min-size are AUTO..." caption and
                    the two number fields share a baseline. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
                  <Typography variant="caption" sx={{ color: "text.disabled", fontSize: "0.65rem", alignSelf: "center" }}>
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
                {/* Quick clarifier line: cellpose segments ONCE using the
                    chosen channels — the OTHER channels are not
                    re-segmented but ARE measured inside each detected
                    cell. This is the standard IF workflow and prevents
                    the common "why isn't my third channel being
                    segmented" confusion. */}
                <Typography variant="caption" sx={{ color: "text.disabled", fontSize: "0.65rem", display: "block", mt: 0.2 }}>
                  Cellpose produces ONE cell mask per image (using the channels above). Intensities are then measured inside that mask in EVERY channel — not just the one used for segmentation.
                </Typography>
              </Box>
            )}
          </Box>

          {/* Groups (per-image assignment) ─────────────────────── */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.1, flex: 1, overflowY: "auto" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.4 }}>
              <Tooltip title="Group images into experimental conditions (Control, Treatment, …). The R plot draws one bar per (group, channel) with mean ± SD and pairwise stats.">
                <Typography variant="caption" sx={{ fontWeight: 700 }}>Groups</Typography>
              </Tooltip>
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.62rem" }}>
                · {imageEntries.length} image{imageEntries.length === 1 ? "" : "s"} wired
              </Typography>
              <Button size="small" variant="outlined" onClick={addGroup} disabled={imageEntries.length === 0}
                sx={{ textTransform: "none", fontSize: "0.62rem", py: 0.05, px: 0.6, ml: "auto" }}>
                + Group
              </Button>
            </Box>
            {/* Stale-assignment banner: the saved config references
                images that aren't wired to this node right now.  This
                hits when the user uploaded N images to the source node
                but only drew a connection from SOME of those outputs
                to this picker — the unwired ones can't be analysed
                but their group assignments still take up cfg.groups
                slots, hiding the wiring gap.  Telling them up-front
                avoids the "Group 2 has no data" mystery. */}
            {(() => {
              const wiredKeys = new Set<string>(imageEntries.flatMap((e) => [e.id, e.label]));
              const orphans = cfg.groups.flatMap((g) =>
                g.images.filter((im) => !wiredKeys.has(im))
              );
              if (orphans.length === 0) return null;
              return (
                <Box sx={{
                  mb: 0.5, px: 0.7, py: 0.4, borderRadius: 0.4,
                  bgcolor: "rgba(220,150,60,0.12)",
                  border: "1px solid rgba(220,150,60,0.4)",
                  fontSize: "0.62rem", color: "#c0a060",
                }}>
                  ⚠ {orphans.length} group assignment{orphans.length === 1 ? "" : "s"} reference image{orphans.length === 1 ? "" : "s"} not wired to this node — those won't be analysed.
                  Draw a connection from each upstream source-node output you want included.
                </Box>
              );
            })()}
            {imageEntries.length === 0 ? (
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
                      {imageEntries.map((entry) => {
                        // Toggle membership by stable id; chip text is the
                        // display label.  Backward-compat: accept legacy
                        // configs that stored labels in g.images by
                        // matching either id or label.
                        const on = g.images.includes(entry.id) || g.images.includes(entry.label);
                        const inOtherGroup =
                          (imgToGroup.has(entry.id) && imgToGroup.get(entry.id) !== g.name)
                          || (imgToGroup.has(entry.label) && imgToGroup.get(entry.label) !== g.name);
                        const otherGroupName =
                          imgToGroup.get(entry.id) || imgToGroup.get(entry.label) || "";
                        return (
                          <Tooltip key={entry.id} title={inOtherGroup ? `Already in "${otherGroupName}" — clicking will move it here.` : entry.label}>
                            <Box onClick={() => toggleImageInGroup(g.id, entry.id)}
                              sx={{
                                fontSize: "0.62rem", px: 0.5, py: 0.08, borderRadius: 0.6,
                                cursor: "pointer", userSelect: "none",
                                bgcolor: on ? "primary.main" : "transparent",
                                color: on ? "primary.contrastText" : (inOtherGroup ? "text.disabled" : "text.secondary"),
                                border: "1px solid", borderColor: on ? "primary.main" : "divider",
                                fontWeight: on ? 700 : 500,
                                opacity: inOtherGroup && !on ? 0.65 : 1,
                                maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                              }}>
                              {entry.label}
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
                            onClick={handleSave}
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
