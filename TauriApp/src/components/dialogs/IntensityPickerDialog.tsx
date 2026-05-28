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
      r: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30 },
      g: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30 },
      b: { enabled: true, thresholdMethod: "percentile", thresholdPercentile: 95, minArea: 30 },
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

def _cellpose_labels_for_image(label, a8_rgb):
    """Single-image cellpose call via the loopback API. Returns the
    integer labels array (or None on failure)."""
    payload = json.dumps({
        "config": json.dumps({
            "model": cp_cfg.get("model") or "cpsam",
            "diameter": float(cp_cfg.get("diameter") or 0) or None,
            "min_size": int(cp_cfg.get("minSize") or 30),
            "channels": [{"r": 1, "g": 2, "b": 3}.get(cp_cfg.get("segChannel") or "b", 3), 0],
        }),
        "extra_inputs": [{
            "kind": "image", "key": label, "label": label,
            "image_b64": _b64.b64encode(_png_bytes(a8_rgb)).decode(),
        }],
        "sources": [], "timeout_sec": 600,
    })
    try:
        req = _ur.Request("http://127.0.0.1:8765/api/analysis/run-cellpose",
                          data=payload.encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=600) as resp:
            cp_out = json.loads(resp.read().decode("utf-8"))
    except Exception as _e:
        return None, f"cellpose call failed: {_e}"
    if not cp_out.get("success"):
        return None, (cp_out.get("stderr") or "(no detail)").strip()
    lbl_b64 = next((im["image"] for im in (cp_out.get("images") or [])
                    if im.get("name") == f"{label}_labels"), None)
    if not lbl_b64:
        return None, "no labels image returned"
    try:
        return np.asarray(_Im.open(_io.BytesIO(_b64.b64decode(lbl_b64))).convert("L")).astype(np.int32), None
    except Exception as _e:
        return None, f"labels decode failed: {_e}"

def _png_bytes(arr_u8):
    buf = _io.BytesIO()
    _Im.fromarray(arr_u8).save(buf, format="PNG")
    return buf.getvalue()

rows = []

if mode == "cellpose":
    # ── Per-cell pipeline. ──
    # 1) Rolling-ball BG subtract every channel.
    # 2) Get cell labels from Cellpose.
    # 3) Per cell: per-channel mean (raw + bg-corrected) → one row per
    #    (cell, channel). n = cells per group across all images in that
    #    group.
    # 4) Save labeled-mask + outline-overlay PNG per source.
    n_images_with_cells = 0
    for key, src in imgs:
        label = _label_of(src, key)
        grp = img2group.get(label)
        if not grp:
            print(f"[intensity] {label}: not assigned to any group — skipping")
            continue
        raw = _pixels(src)                                  # HxWx3 float32
        corrected = np.zeros_like(raw, dtype=np.float64)
        for ci in range(3):
            bg = _rolling_bg(raw[..., ci], rolling_radius)
            c = raw[..., ci].astype(np.float64) - bg
            c[c < 0] = 0
            corrected[..., ci] = c
        a8 = np.clip(raw, 0, 255).astype(np.uint8)
        lbl_arr, err = _cellpose_labels_for_image(label, a8)
        if err or lbl_arr is None or lbl_arr.shape[:2] != raw.shape[:2]:
            print(f"[intensity] {label}: cellpose unusable ({err or 'shape mismatch'}) — skipping")
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
        corrected[..., ci] = corr
        try:
            mask = _threshold_mask(
                corr,
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
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    fetch("http://127.0.0.1:8765/api/version")
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (!cancelled && d?.build) setSidecarBuild(String(d.build)); })
      .catch(() => { /* not critical — leave null */ });
    // Fire-and-forget warmup — triggers cv2 / scipy / PIL imports in the
    // frozen sidecar so the user's first Run preview click doesn't pay
    // the cold-start cost (which would otherwise blow past the 30 s
    // simple-mode timeout). Idempotent; subsequent dialog opens are
    // ~instant because the modules stay cached for the process lifetime.
    fetch("http://127.0.0.1:8765/api/analysis/warmup", { method: "POST" })
      .catch(() => { /* warm-up is best-effort; the picker still works without it */ });
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
        body.channels = {
          r: {
            enabled: cfg.thresholds.r.enabled,
            threshold_method: cfg.thresholds.r.thresholdMethod,
            threshold_percentile: cfg.thresholds.r.thresholdPercentile,
            min_area: cfg.thresholds.r.minArea,
          },
          g: {
            enabled: cfg.thresholds.g.enabled,
            threshold_method: cfg.thresholds.g.thresholdMethod,
            threshold_percentile: cfg.thresholds.g.thresholdPercentile,
            min_area: cfg.thresholds.g.minArea,
          },
          b: {
            enabled: cfg.thresholds.b.enabled,
            threshold_method: cfg.thresholds.b.thresholdMethod,
            threshold_percentile: cfg.thresholds.b.thresholdPercentile,
            min_area: cfg.thresholds.b.minArea,
          },
        };
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
      let resp: Response;
      try {
        resp = await fetch("http://127.0.0.1:8765/api/analysis/fluor-preview-segment", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: ac.signal,
        });
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
      <DialogContent dividers sx={{ display: "flex", flexDirection: "row", gap: 1.5, py: 1.5, minHeight: 600 }}>
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
              ? (cfg.mode === "cellpose" ? "Running Cellpose…" : "Running…")
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
            minHeight: 400,
          }}>
            {(() => {
              // 1) Best-available preview source for this image.
              const ap = activePreview;
              const isSimple = cfg.mode === "simple";
              // Simple strategy: prefer composite + per-channel overlays.
              // Fallback to fused overlaySrc if composite missing.
              if (ap && isSimple && (ap.compositeSrc || ap.overlaySrc)) {
                // Layered preview: the wrapper sizes to the composite img
                // (inline-block + lineHeight:0 to suppress baseline-descender
                // space). Overlay imgs are absolutely positioned to the
                // wrapper's full bounds with NO objectFit — the overlays
                // are guaranteed by the backend to be the EXACT same pixel
                // dimensions as the composite, so stretching to 100%/100%
                // is a pixel-perfect match. Earlier objectFit:contain on
                // each overlay introduced independent fit-rounding that
                // could shift overlays 1-2 px off the underlying signal.
                return (
                  <Box sx={{
                    position: "relative", maxWidth: "100%", maxHeight: "100%",
                    display: "inline-block", lineHeight: 0, fontSize: 0,
                  }}>
                    <img
                      src={ap.compositeSrc || ap.overlaySrc}
                      alt="Composite"
                      style={{
                        display: "block",
                        maxWidth: "100%",
                        maxHeight: "calc(100vh - 320px)",
                        verticalAlign: "top",
                      }}
                    />
                    {(["r", "g", "b"] as const).map((k) => {
                      const src = ap.channelOverlays?.[k];
                      if (!src) return null;
                      return (
                        <img
                          key={k}
                          src={src}
                          alt={`${cfg.channels[k]} mask`}
                          style={{
                            position: "absolute",
                            top: 0, left: 0,
                            width: "100%", height: "100%",
                            display: maskVisible[k] ? "block" : "none",
                            pointerEvents: "none",
                          }}
                        />
                      );
                    })}
                  </Box>
                );
              }
              // Cellpose: single fused overlay.
              if (ap?.overlaySrc) {
                return (
                  <img
                    src={ap.overlaySrc}
                    alt="Segmentation preview"
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                  />
                );
              }
              // No preview yet — fall back to the raw thumbnail (dimmed).
              if (activeImage?.image_b64) {
                return (
                  <img
                    src={`data:image/png;base64,${activeImage.image_b64}`}
                    alt={activeImage.label}
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", opacity: 0.65 }}
                  />
                );
              }
              return (
                <Typography variant="caption" sx={{ color: "text.disabled" }}>
                  {images.length === 0 ? "Wire upstream image sources first." : "(no preview — click Run)"}
                </Typography>
              );
            })()}
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
                  return (
                    <Box key={k} sx={{
                      display: "grid",
                      gridTemplateColumns: "115px 1fr 100px",
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
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1 }}>
                <TextField select size="small" label="Model" value={cfg.cellpose.model}
                  onChange={(e) => setCellpose({ model: e.target.value })}
                  inputProps={{ style: { fontSize: "0.78rem" } }}
                  SelectProps={SELECT_MENU_PROPS}>
                  <MenuItem value="cpsam" sx={{ fontSize: "0.78rem" }}>cpsam (default)</MenuItem>
                  <MenuItem value="cyto3" sx={{ fontSize: "0.78rem" }}>cyto3</MenuItem>
                  <MenuItem value="cyto2" sx={{ fontSize: "0.78rem" }}>cyto2</MenuItem>
                  <MenuItem value="nuclei" sx={{ fontSize: "0.78rem" }}>nuclei</MenuItem>
                </TextField>
                <TextField select size="small" label="Segment on" value={cfg.cellpose.segChannel}
                  onChange={(e) => setCellpose({ segChannel: e.target.value as "r" | "g" | "b" })}
                  inputProps={{ style: { fontSize: "0.78rem" } }}
                  SelectProps={SELECT_MENU_PROPS}>
                  <MenuItem value="r" sx={{ fontSize: "0.78rem" }}>{cfg.channels.r}</MenuItem>
                  <MenuItem value="g" sx={{ fontSize: "0.78rem" }}>{cfg.channels.g}</MenuItem>
                  <MenuItem value="b" sx={{ fontSize: "0.78rem" }}>{cfg.channels.b}</MenuItem>
                </TextField>
                <TextField size="small" label="Diameter (px, 0=auto)" type="number" value={cfg.cellpose.diameter}
                  onChange={(e) => setCellpose({ diameter: Math.max(0, Number(e.target.value) || 0) })}
                  inputProps={{ min: 0, max: 400, step: 5, style: { fontSize: "0.78rem" } }}
                  InputLabelProps={{ style: { fontSize: "0.78rem" } }} />
                <TextField size="small" label="Min size (px²)" type="number" value={cfg.cellpose.minSize}
                  onChange={(e) => setCellpose({ minSize: Math.max(0, Number(e.target.value) || 0) })}
                  inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.78rem" } }}
                  InputLabelProps={{ style: { fontSize: "0.78rem" } }} />
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
