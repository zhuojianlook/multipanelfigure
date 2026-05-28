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
            comp = _composite_u8(raw)
            boundaries = np.zeros(labels.shape, dtype=bool)
            boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
            boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
            boundaries &= labels > 0
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
            # Outline boundary on the overlay so the user can spot small dots.
            boundary = np.zeros(m.shape, dtype=bool)
            boundary[:-1, :] |= m[:-1, :] != m[1:, :]
            boundary[:, :-1] |= m[:, :-1] != m[:, 1:]
            if _have_scipy:
                boundary = _ndi.binary_dilation(boundary, iterations=1)
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

export default function IntensityPickerDialog(props: IntensityPickerDialogProps) {
  const { open, images, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<FluorIntensityConfig>(initial ? migrateConfig(initial) : emptyFluorConfig());
  const [activeIdx, setActiveIdx] = useState(0);
  // Live preview state — overlay returned by /api/analysis/fluor-preview-segment.
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewStats, setPreviewStats] = useState<{ n_cells?: number; per_channel?: Record<string, number> } | null>(null);

  // Reset on (re-)open. Clear active idx + preview so we don't show stale.
  useEffect(() => {
    if (open) {
      setCfg(initial ? migrateConfig(structuredClone(initial)) : emptyFluorConfig());
      setActiveIdx(0);
      setPreviewSrc(null);
      setPreviewError(null);
      setPreviewStats(null);
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
      const resp = await fetch("http://127.0.0.1:8765/api/analysis/fluor-preview-segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ac.signal,
      });
      const data = await resp.json();
      if (ac.signal.aborted) return;
      if (data.success && data.overlay_b64) {
        setPreviewSrc(`data:image/png;base64,${data.overlay_b64}`);
        setPreviewStats({
          n_cells: typeof data.n_cells === "number" ? data.n_cells : undefined,
          per_channel: data.per_channel,
        });
        setParamsDirty(false);    // current overlay reflects current params
      } else {
        setPreviewError(data.error || "preview failed");
        setPreviewSrc(null);
      }
    } catch (e: unknown) {
      if ((e as { name?: string })?.name === "AbortError") return;
      setPreviewError(String((e as { message?: string })?.message ?? e));
      setPreviewSrc(null);
    } finally {
      setPreviewLoading(false);
    }
  }, [open, activeImage, cfg.mode, cfg.rollingRadius, cfg.thresholds, cfg.cellpose]);

  // Auto-fetch ONLY on image cycle (a deliberate user action that wants
  // a fresh overlay for the newly-shown image). Param changes leave the
  // previous overlay in place and just mark the params dirty — the user
  // clicks Run to commit.
  useEffect(() => {
    if (!open || !activeImage) return;
    // Clear the previous image's stats so the footer doesn't lie about
    // the new image while the next preview is being fetched.
    setPreviewStats(null);
    setParamsDirty(false);
    void fetchPreview();
    // We intentionally depend ONLY on activeIdx + open, so changing
    // strategy/thresholds doesn't trigger an auto-fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, activeIdx]);

  // Mark the preview as out-of-date whenever a preview-relevant param
  // changes. Doesn't fire any fetch — just lights up the Run button so
  // the user knows the current overlay isn't the latest.
  useEffect(() => {
    if (!open) return;
    setParamsDirty(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg.mode, cfg.rollingRadius, cfg.thresholds, cfg.cellpose]);

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
            <Tooltip title={paramsDirty
              ? "Preview is out of date — click to re-run the segmentation with the current settings"
              : "Re-run the segmentation preview"}>
              <span>
                <Button size="small" variant={paramsDirty ? "contained" : "outlined"}
                  color={paramsDirty ? "primary" : "inherit"}
                  startIcon={<RefreshIcon sx={{ fontSize: 16 }} />}
                  onClick={() => void fetchPreview()}
                  disabled={!activeImage || previewLoading}
                  sx={{ textTransform: "none", fontSize: "0.72rem", py: 0.25, px: 1, minWidth: 0 }}>
                  Run preview
                </Button>
              </span>
            </Tooltip>
          </Box>
          {/* Preview canvas */}
          <Box sx={{
            position: "relative", flex: 1,
            border: "1px solid", borderColor: "divider", borderRadius: 1,
            bgcolor: "#0a0a0a",
            overflow: "hidden",
            display: "flex", alignItems: "center", justifyContent: "center",
            minHeight: 400,
          }}>
            {previewSrc ? (
              <img
                src={previewSrc}
                alt="Segmentation preview"
                style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
              />
            ) : activeImage?.image_b64 ? (
              <img
                src={`data:image/png;base64,${activeImage.image_b64}`}
                alt={activeImage.label}
                style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", opacity: 0.65 }}
              />
            ) : (
              <Typography variant="caption" sx={{ color: "text.disabled" }}>
                {images.length === 0 ? "Wire upstream image sources first." : "(no preview)"}
              </Typography>
            )}
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
                px: 0.8, py: 0.3, borderRadius: 0.5,
                bgcolor: "rgba(180,40,40,0.85)", color: "common.white",
                fontSize: "0.66rem",
              }}>
                ⚠ {previewError}
              </Box>
            )}
            {/* Stats footer (cell count or per-channel pixel tallies) */}
            {!previewError && previewStats && (
              <Box sx={{
                position: "absolute", bottom: 4, left: 4, right: 4,
                px: 0.8, py: 0.3, borderRadius: 0.5,
                bgcolor: "rgba(0,0,0,0.55)", color: "common.white",
                fontSize: "0.66rem", display: "flex", gap: 1, justifyContent: "center",
              }}>
                {typeof previewStats.n_cells === "number" && (
                  <span>cells: <b>{previewStats.n_cells}</b></span>
                )}
                {previewStats.per_channel && (["r", "g", "b"] as const).map((k) => {
                  const sw = k === "r" ? "#ff8080" : k === "g" ? "#88e088" : "#88a8ff";
                  const n = previewStats.per_channel?.[k] ?? 0;
                  if (!cfg.thresholds[k].enabled) return null;
                  return (
                    <span key={k} style={{ color: sw }}>
                      {cfg.channels[k]}: <b>{n.toLocaleString()}px</b>
                    </span>
                  );
                })}
              </Box>
            )}
          </Box>
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
                sx={{ minWidth: 140 }}>
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
                  Simple (per-channel threshold)
                </ToggleButton>
                <ToggleButton value="cellpose" sx={{ textTransform: "none", fontSize: "0.68rem", py: 0.15 }}>
                  Cellpose (per-cell)
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <Typography variant="caption" sx={{ color: "text.secondary", display: "block", mb: 0.75 }}>
              {cfg.mode === "simple"
                ? "Each enabled channel gets a rolling-ball BG + threshold mask. Intensity sample = mean within that channel's mask. n = images per group."
                : "Cellpose 3+ segments cells in the chosen channel. Per-cell mean intensity per channel. n = cells per group."}
            </Typography>

            {cfg.mode === "simple" ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.6 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 90 }}>Rolling BG (px)</Typography>
                  <TextField size="small" type="number" value={cfg.rollingRadius}
                    onChange={(e) => setCfg((c) => ({ ...c, rollingRadius: Math.max(0, Number(e.target.value) || 0) }))}
                    inputProps={{ min: 0, max: 200, step: 1, style: { fontSize: "0.78rem", padding: "4px 6px" } }}
                    sx={{ width: 110 }} />
                  <Typography variant="caption" sx={{ color: "text.disabled", ml: 1 }}>
                    Subtract a morphological-open background per channel (0 disables).
                  </Typography>
                </Box>
                {/* Per-channel threshold rows */}
                {(["r", "g", "b"] as const).map((k) => {
                  const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
                  const t = cfg.thresholds[k];
                  return (
                    <Box key={k} sx={{
                      display: "grid",
                      gridTemplateColumns: "100px 90px 90px 90px 90px",
                      gap: 0.6, alignItems: "center",
                      px: 0.5, py: 0.4, borderRadius: 0.5,
                      bgcolor: t.enabled ? "transparent" : "action.hover",
                      opacity: t.enabled ? 1 : 0.7,
                    }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Box
                          onClick={() => setThreshold(k, { enabled: !t.enabled })}
                          sx={{
                            width: 12, height: 12, borderRadius: 0.4,
                            bgcolor: t.enabled ? sw : "transparent",
                            border: "1px solid", borderColor: sw, cursor: "pointer",
                          }} />
                        <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.66rem",
                                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {cfg.channels[k]}
                        </Typography>
                      </Box>
                      <TextField select size="small" value={t.thresholdMethod}
                        onChange={(e) => setThreshold(k, { thresholdMethod: e.target.value as "percentile" | "otsu" })}
                        disabled={!t.enabled}
                        inputProps={{ style: { fontSize: "0.72rem", padding: "3px 5px" } }}>
                        <MenuItem value="percentile" sx={{ fontSize: "0.78rem" }}>Percentile</MenuItem>
                        <MenuItem value="otsu" sx={{ fontSize: "0.78rem" }}>Otsu</MenuItem>
                      </TextField>
                      <TextField size="small" type="number" label="%" value={t.thresholdPercentile}
                        onChange={(e) => setThreshold(k, { thresholdPercentile: Math.max(0, Math.min(100, Number(e.target.value) || 0)) })}
                        disabled={!t.enabled || t.thresholdMethod !== "percentile"}
                        inputProps={{ min: 0, max: 100, step: 0.5, style: { fontSize: "0.72rem", padding: "3px 5px" } }} />
                      <TextField size="small" type="number" label="Min area" value={t.minArea}
                        onChange={(e) => setThreshold(k, { minArea: Math.max(0, Number(e.target.value) || 0) })}
                        disabled={!t.enabled}
                        inputProps={{ min: 0, max: 100000, step: 5, style: { fontSize: "0.72rem", padding: "3px 5px" } }} />
                      <Typography variant="caption" sx={{ color: "text.disabled", fontSize: "0.6rem", textAlign: "right" }}>
                        {!t.enabled ? "skipped" : ""}
                      </Typography>
                    </Box>
                  );
                })}
              </Box>
            ) : (
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1 }}>
                <TextField select size="small" label="Model" value={cfg.cellpose.model}
                  onChange={(e) => setCellpose({ model: e.target.value })}
                  inputProps={{ style: { fontSize: "0.78rem" } }}>
                  <MenuItem value="cpsam">cpsam (default)</MenuItem>
                  <MenuItem value="cyto3">cyto3</MenuItem>
                  <MenuItem value="cyto2">cyto2</MenuItem>
                  <MenuItem value="nuclei">nuclei</MenuItem>
                </TextField>
                <TextField select size="small" label="Segment on" value={cfg.cellpose.segChannel}
                  onChange={(e) => setCellpose({ segChannel: e.target.value as "r" | "g" | "b" })}
                  inputProps={{ style: { fontSize: "0.78rem" } }}>
                  <MenuItem value="r">{cfg.channels.r}</MenuItem>
                  <MenuItem value="g">{cfg.channels.g}</MenuItem>
                  <MenuItem value="b">{cfg.channels.b}</MenuItem>
                </TextField>
                <TextField size="small" label="Diameter (px, 0=auto)" type="number" value={cfg.cellpose.diameter}
                  onChange={(e) => setCellpose({ diameter: Math.max(0, Number(e.target.value) || 0) })}
                  inputProps={{ min: 0, max: 400, step: 5, style: { fontSize: "0.78rem" } }} />
                <TextField size="small" label="Min size (px²)" type="number" value={cfg.cellpose.minSize}
                  onChange={(e) => setCellpose({ minSize: Math.max(0, Number(e.target.value) || 0) })}
                  inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.78rem" } }} />
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
