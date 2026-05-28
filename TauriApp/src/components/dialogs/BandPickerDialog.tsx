// ─────────────────────────────────────────────────────────────
//  BandPickerDialog — interactive western-blot band picker
// ─────────────────────────────────────────────────────────────
//  The user drops the WHOLE blot membrane onto a Source node and
//  wires it into a "Band picker" node.  This dialog shows that
//  membrane and lets them:
//    • auto-detect bands (backend lane/level detector),
//    • draw / move / resize a rectangular ROI per band,
//    • label each band's LANE (column / sample) and LEVEL (target /
//      MW row) — "lanes of the same level" are compared to each other,
//    • pick which LEVEL is the loading control (per-lane normaliser),
//    • optionally draw a background ROI (toggle) for subtraction.
//
//  The ROIs are stored as NORMALISED 0..1 coords so they survive
//  image resizing / DPI changes, and are baked into auto-generated
//  Python at run time (see generateBandPickerCode) that maps them
//  onto the full-res membrane, crops each band, inverts, subtracts
//  background and sums → integrated optical density (IOD) per band.
//  The emitted table schema (lane / level / is_loading_control / iod)
//  feeds the Normalize node (per-lane loading-control division) and
//  the R plot (grouped bars, mean±SD, significance, faceted by level).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box, Typography, Button, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, Tooltip, MenuItem,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

// ── Types (shared with AnalysisNodeGraph via import) ──────────
export interface BandRoi {
  id: string;
  /** Normalised 0..1 rect over the membrane image (top-left origin). */
  x: number; y: number; w: number; h: number;
  /** UNIQUE display name for this band, e.g. "B1". Auto-generated so every
   *  row in the table is distinguishable even when multiple bands share a
   *  lane (column). User-editable. Falls back to `label` for old configs. */
  name?: string;
  /** Lane / sample name (the COLUMN), e.g. "S1". Shared across bands in the
   *  same column — this is the GROUPING key for loading-control normalisation
   *  and the user-defined group→condition mapping. */
  label: string;
  /** Target / protein row (the LEVEL): "lanes of the same level" are bands
   *  at the same apparent molecular weight across columns. e.g. "G1" / "GAPDH". */
  level: string;
  /** @deprecated — loading control is now chosen per LEVEL
   *  (BandPickerConfig.loadingControlLevel). Kept so older saved nodes load. */
  isReference?: boolean;
}

/** A user-defined experimental group — maps a condition NAME to a set of
 *  individual BANDS (by `name`, e.g. "B5", "B12") that get compared
 *  against other groups in the downstream R plot. `lanes` is kept for
 *  back-compat with older saved configs (whole-lane membership); new UI
 *  populates `bands` exclusively. */
export interface BandGroup {
  id: string;
  name: string;          // condition name shown on the x-axis
  bands?: string[];      // band NAMES (B1, B5, …) — preferred
  lanes?: string[];      // legacy: lane LABELs (S1, …) — fallback for old configs
}

/** Detected lane geometry (vertical strip on the membrane) returned by
 *  the backend. Used for (a) on-preview guide lines and (b) snapping
 *  freshly-drawn boxes to the nearest lane so every box has a lane
 *  identity — required for per-lane loading-control normalisation. */
export interface LaneBound {
  name: string;
  x1: number; y1: number; x2: number; y2: number;   // all 0..1 normalised
}

export interface BandPickerConfig {
  version: 1;
  lanes: BandRoi[];
  /** Master switch for loading-control normalisation. When true, downstream
   *  Normalize divides each lane's bands by that lane's chosen LC band's
   *  IOD → "relative density". When false, IOD passes through unchanged
   *  and the R plot reports raw IOD values directly. */
  loadingControlEnabled?: boolean;
  /** When `loadingControlEnabled`, maps each LANE NAME → the BAND NAME
   *  (B1, B5, …) chosen as that lane's loading control. Defaults to the
   *  LOWEST-row (highest y) band in each lane (the canonical bottom-of-
   *  membrane housekeeping band). */
  lcBandByLane?: Record<string, string>;
  /** Legacy: which `level` (target row) is the loading control. Kept so
   *  older saved nodes still load; new picker UI uses lcBandByLane. */
  loadingControlLevel?: string | null;
  /** "percentile" → per-lane Nth-percentile floor (default);
   *  "roi" → subtract the mean of a user-drawn background rect. */
  bgMode: "percentile" | "roi";
  bgRoi?: { x: number; y: number; w: number; h: number } | null;
  /** Percentile (0..100) used when bgMode === "percentile". */
  bgPercentile: number;
  /** User-defined experimental groups for the downstream R plot. The picker
   *  emits a per-row `group` column on the band_iod table; the plot uses it
   *  to draw grouped bars + significance. Empty → every lane is its own
   *  group (the old behaviour). */
  groups?: BandGroup[];
  /** Default box dimensions (NORMALISED 0..1) used for the "+ Band" button
   *  and snap-on-release after drag — so every box matches the auto-detect
   *  74×33 px ROI regardless of the source's resolution. */
  defaultBoxW?: number;
  defaultBoxH?: number;
  /** Lane geometry returned by the detection endpoint — drives the lane
   *  guide lines + snap-to-lane on new boxes. */
  laneBounds?: LaneBound[];
  /** Plot layout: "combined" (default) faceted side-by-side in one figure,
   *  "separate" emits one PNG per group so each group can be moved into
   *  the figure timeline independently. */
  plotPerGroup?: boolean;
  /** Image orientation for IOD math.  Drives whether the measurement
   *  plane is inverted (vmax − pixel) before summing.
   *    "dark_on_light" — chemiluminescent / film: dark bands on light
   *      background.  Stronger band = darker pixels = LOWER raw sum.
   *      We invert so signal = integrated density (higher = more).
   *    "light_on_dark" — fluorescent (LI-COR / Odyssey, etc.): bright
   *      bands on dark background.  Raw sum already IS the density,
   *      no inversion needed.
   *    "auto" (default) — pick based on the median of the measurement
   *      plane.  Wrong only if your image was contrast-adjusted in a
   *      way that fooled the heuristic (e.g. histogram-stretched
   *      chemilum with a black background applied in software).
   *  Saved per-blot; the generated Python honours whatever you pick. */
  bandOrientation?: "auto" | "dark_on_light" | "light_on_dark";
}

export function emptyBandConfig(): BandPickerConfig {
  return {
    version: 1, lanes: [],
    loadingControlEnabled: false, lcBandByLane: {},
    loadingControlLevel: null,
    bgMode: "percentile", bgRoi: null, bgPercentile: 5,
    groups: [], laneBounds: [],
    plotPerGroup: false,
    bandOrientation: "auto",
  };
}

/** Pick a band's lane label from the laneBounds by x-center. Falls back to
 *  the existing `desiredLabel` (or "Lane") when no bounds intersect. */
function snapLaneFromBounds(
  cx: number,
  bounds: LaneBound[] | undefined,
  desiredLabel: string,
): string {
  if (!bounds || bounds.length === 0) return desiredLabel || "Lane";
  let best: LaneBound | null = null; let bestDist = Infinity;
  for (const b of bounds) {
    const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2);
    if (cx >= x1 && cx <= x2) return b.name;                 // inside → exact hit
    const d = cx < x1 ? x1 - cx : cx - x2;
    if (d < bestDist) { bestDist = d; best = b; }
  }
  return best ? best.name : (desiredLabel || "Lane");
}

/** Lowest-row (highest y-center) band per lane = the default LC pick when
 *  the user first enables LC normalisation. Returns lane → band name. */
function defaultLcByLane(lanes: BandRoi[]): Record<string, string> {
  const byLane: Record<string, BandRoi[]> = {};
  for (const l of lanes) {
    const ln = (l.label || "Lane").trim() || "Lane";
    (byLane[ln] = byLane[ln] || []).push(l);
  }
  const out: Record<string, string> = {};
  for (const [ln, bands] of Object.entries(byLane)) {
    const lowest = bands.reduce((a, b) =>
      (a.y + a.h / 2) > (b.y + b.h / 2) ? a : b);
    out[ln] = (lowest.name || lowest.label || "");
  }
  return out;
}

// Unique-id generator with a process-wide counter so same-millisecond
// auto-detect / import-CSV loops can't collide. (The previous 4-char
// random suffix produced ~1.7M combinations → birthday-collisions
// became plausible for >100 IDs/tick, which manifested as missing-IOD
// rows in the table because multiple bands shared one map key.)
let _uidCounter = 0;
const uid = (prefix = "lane") =>
  `${prefix}_${Date.now().toString(36)}_${(_uidCounter++).toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** Read a ROI's displayed name with back-compat for old saved configs that
 *  only have `label`. New rows always have an explicit `name`. */
const roiName = (l: BandRoi): string => (l.name || l.label || "Band");

// Distinct, color-blind-friendly palette to tint ROIs by their level (row), so
// "lanes of the same level" share a color in the editor.
const LEVEL_PALETTE = [
  "#1976d2", "#43a047", "#e65100", "#8e24aa", "#00838f",
  "#c62828", "#5d4037", "#558b2f", "#6a1b9a", "#00695c",
];
/** Stable color for a level, by its index in the (sorted) distinct-level list. */
function levelColor(level: string, order: string[]): string {
  const i = order.indexOf(level);
  return LEVEL_PALETTE[(i < 0 ? 0 : i) % LEVEL_PALETTE.length];
}
/** Read a ROI's level with back-compat default for older saved nodes. */
const roiLevel = (l: BandRoi): string => (l.level || "Target");

// ── Python code generator ────────────────────────────────────
// Emits a self-contained script.  The config is embedded as a JSON
// literal so the run is fully reproducible from the saved node.
export function generateBandPickerCode(cfg: BandPickerConfig): string {
  const json = JSON.stringify(cfg);
  return `# @name: Band picker (IOD per lane)
# Auto-generated from the interactive band picker.  Edit lanes via
# the "Pick bands…" button on the node rather than by hand — manual
# edits here are overwritten the next time you open the picker.
#
# Quantification matches detect_bands_equal_boxes.py's quantify():
#   * raw_integrated_density = sum of measurement-plane pixels in each ROI box,
#   * background = robust mean of the same-width slabs immediately ABOVE and
#     BELOW the band (with a 6 px gap), computed with median + MAD outlier reject
#     so neighbouring bands don't bias it,
#   * background_corrected_integrated_density = raw - bg_mean * area.
# This is the same number the CLI's annotated CSV emits — so you can sanity
# check by running the script and comparing the values.
import numpy as np, json

CFG = json.loads(r'''${json}''')

imgs = [v for v in inputs.values() if isinstance(v, dict) and ("image_raw" in v or "image" in v)]
if not imgs:
    raise SystemExit("No membrane image — wire the blot (Source) into this node.")
src = imgs[0]
# FULL-BIT-DEPTH pixels (e.g. 16-bit) for quantitative IOD; fall back to the
# 8-bit display image only when raw isn't available.
use_raw = "image_raw" in src
arr = np.asarray(src["image_raw"] if use_raw else src["image"]).astype(np.float32)
# Measurement plane — max over RGB to match detect_bands_equal_boxes.py
# (--measurement-channel max default).
if arr.ndim == 3 and arr.shape[2] >= 3:
    plane = np.max(arr[..., :3], axis=2)
elif arr.ndim == 3:
    plane = arr[..., 0]
else:
    plane = arr
H, W = plane.shape[:2]

# Orientation handling — DARK-on-LIGHT blots (chemiluminescent / film:
# dark bands on a light background) need the plane inverted before
# summing, so that "stronger band = LARGER IOD" matches the conventional
# Western-blot semantics.  LIGHT-on-DARK blots (fluorescent, LI-COR /
# Odyssey: bright bands on dark background) already satisfy that and
# need no inversion.
#
# CFG["bandOrientation"]:
#   "dark_on_light" — force inversion
#   "light_on_dark" — force no inversion (raw sum = density)
#   "auto" / missing — pick by the median of the plane (median > vmax/2
#     → mostly-light background → dark-on-light → invert).  Wrong only
#     when the image was contrast-stretched in a way that confuses the
#     heuristic; flip the picker's "Image orientation" radio to fix.
vmax = 65535.0 if float(plane.max()) > 255.0 else 255.0
median_v = float(np.median(plane))
_orient = str(CFG.get("bandOrientation") or "auto").lower()
if _orient == "dark_on_light":
    dark_on_light = True
elif _orient == "light_on_dark":
    dark_on_light = False
else:
    dark_on_light = median_v > vmax * 0.5
if dark_on_light:
    plane = vmax - plane
    print(f"IOD orientation: DARK-on-LIGHT (median={median_v:.1f}/{vmax:.0f}, source={_orient}) "
          f"— inverted plane so signal = integrated density (higher = more protein)")
else:
    print(f"IOD orientation: LIGHT-on-DARK (median={median_v:.1f}/{vmax:.0f}, source={_orient}) "
          f"— summing raw pixels directly (bright bands → larger IOD)")

def _px(rect):
    x0 = int(round(rect["x"] * W)); x1 = int(round((rect["x"] + rect["w"]) * W))
    y0 = int(round(rect["y"] * H)); y1 = int(round((rect["y"] + rect["h"]) * H))
    x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
    y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
    return x0, y0, x1, y1

def _robust_background(values):
    flat = np.asarray(values).reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    if mad > 0:
        sigma = 1.4826 * mad
        keep = np.abs(flat - median) <= 3.0 * sigma
        if int(np.sum(keep)) >= max(20, flat.size // 5):
            flat = flat[keep]
    return float(np.mean(flat))

# Loading-control configuration:
#   - loadingControlEnabled == True → use the per-lane band picks (lcBandByLane:
#       { laneName → bandName }) to flag is_loading_control=True on those rows.
#   - loadingControlEnabled == False → no LC flagging; downstream Normalize
#       will pass IOD through unchanged.
#   - Legacy fallback: loadingControlLevel string flags any band whose level
#       matches (kept so older saved nodes still normalise correctly).
lc_enabled = bool(CFG.get("loadingControlEnabled"))
plot_per_group = bool(CFG.get("plotPerGroup"))
# Picker UI saves { lane_being_normalized: band_name_to_use_as_LC }, which
# CAN be cross-lane (user-drawn box in lane S2 chosen as the LC for lane S1).
lc_pick_for_lane = {str(k): str(v) for k, v in (CFG.get("lcBandByLane") or {}).items()}
# Reverse map: bandName → set of lanes it serves as the LC for. A single
# band MAY serve multiple lanes (e.g. when the user has only one good
# housekeeping band on the membrane and reuses it).
lc_for_by_band: dict = {}
for _ln, _bn in lc_pick_for_lane.items():
    if _bn:
        lc_for_by_band.setdefault(_bn, []).append(_ln)
lc_level   = CFG.get("loadingControlLevel")          # legacy

# Band → group(s) and lane → group fallback from the picker's Groups UI.
# Groups can now contain band NAMES (B1, B5, ...) for arbitrary band-set
# comparisons, OR lane labels (S1, S2, ...) for the older lane-based form.
band2group = {}; lane2group = {}
for g in CFG.get("groups", []) or []:
    nm = (g.get("name") or "").strip()
    if not nm: continue
    for bn in g.get("bands", []) or []:
        band2group[str(bn)] = nm
    for ln in g.get("lanes", []) or []:
        lane2group[str(ln)] = nm

# Pre-compute ALL ROI rects in pixel space so the background sampling can
# exclude pixels that fall inside ANOTHER band's box. Without this, densely
# packed bands (3-row blots etc.) get IOD = 0 because the above/below
# slabs land on neighbouring bands and inflate the background.
def _band_px(b):
    bx0, by0, bx1, by1 = _px(b)
    return (bx0, by0, bx1, by1)
all_rects = [(_band_px(b), id(b)) for b in (CFG.get("lanes", []) or [])]

rows = []
for i, band in enumerate(CFG.get("lanes", [])):
    x0, y0, x1, y1 = _px(band)
    if x1 <= x0 or y1 <= y0:
        continue
    roi = plane[y0:y1, x0:x1].astype(np.float64)
    area = int(roi.size)
    # Other-band exclusion mask for the bg slabs.
    others = [r for (r, ref) in all_rects if ref != id(band)]
    def _excluded_mask(yslab1, yslab2):
        if yslab2 <= yslab1: return np.zeros((0, x1 - x0), dtype=bool)
        m = np.zeros((yslab2 - yslab1, x1 - x0), dtype=bool)
        for (ox0, oy0, ox1, oy1) in others:
            ix0 = max(x0, ox0); ix1 = min(x1, ox1)
            iy0 = max(yslab1, oy0); iy1 = min(yslab2, oy1)
            if ix1 > ix0 and iy1 > iy0:
                m[iy0 - yslab1:iy1 - yslab1, ix0 - x0:ix1 - x0] = True
        return m
    # Local background: slabs above and below the ROI, same width, gap=6.
    gap, hh = 6, (y1 - y0)
    above_y2 = max(0, y0 - gap); above_y1 = max(0, above_y2 - hh)
    below_y1 = min(H, y1 + gap); below_y2 = min(H, below_y1 + hh)
    parts = []
    if above_y2 > above_y1:
        slab = plane[above_y1:above_y2, x0:x1]
        msk = _excluded_mask(above_y1, above_y2)
        parts.append(slab[~msk].reshape(-1))
    if below_y2 > below_y1:
        slab = plane[below_y1:below_y2, x0:x1]
        msk = _excluded_mask(below_y1, below_y2)
        parts.append(slab[~msk].reshape(-1))
    bg_pixels = np.concatenate(parts) if parts else np.array([])
    # If neighbouring boxes ate too much of the slab, fall back to a per-ROI
    # 5th-percentile floor — guarantees a meaningful (non-zero) IOD even
    # when bands are packed shoulder-to-shoulder.
    if bg_pixels.size >= max(20, area // 4):
        bg_mean = _robust_background(bg_pixels)
    else:
        bg_mean = float(np.percentile(roi, 5)) if area else 0.0
    raw_sum = float(np.sum(roi))
    corrected = raw_sum - bg_mean * area
    lane = (band.get("label") or "").strip() or f"Lane {i + 1}"
    level = (band.get("level") or "").strip() or "Target"
    band_name = (band.get("name") or "").strip() or f"B{i + 1}"
    # Group resolution: band-name match wins over lane match. Empty string
    # means "ungrouped" — the downstream R plot leaves these out of the
    # grouped bars (or shows them as their own bar, depending on plot mode).
    grp = band2group.get(band_name) or lane2group.get(lane) or ""
    # is_loading_control: new path uses the picker's per-lane band pick,
    # which can be CROSS-LANE (user picks B5 in lane S2 as the LC for
    # lane S1). A band is "an LC" if it serves as the chosen LC for any
    # lane. lc_for_lane carries the comma-joined list of lanes this band
    # serves, so the downstream Normalize node can divide the correct
    # rows by THIS band's IOD — without it, cross-lane picks silently
    # fell through and the user's custom LC pixel went unused.
    target_lanes_for_this_band = lc_for_by_band.get(band_name, []) if lc_enabled else []
    if lc_enabled:
        is_lc = len(target_lanes_for_this_band) > 0
    elif lc_level:
        is_lc = (level == lc_level)
    else:
        is_lc = False
    rows.append({
        "band": band_name,                         # UNIQUE per-row identifier
        "lane": lane,                              # column / sample
        "level": level,                            # target / MW row
        "group": grp,                              # condition / experimental group
        "lc_enabled": bool(lc_enabled),            # plot picks normalised vs raw
        "plot_per_group": bool(plot_per_group),    # split groups across figures
        "is_loading_control": bool(is_lc),
        "lc_for_lane": ",".join(target_lanes_for_this_band),  # lanes this band normalises
        "iod": float(max(corrected, 0.0)),
        "raw_integrated_density": raw_sum,
        "background_corrected_integrated_density": corrected,
        "background_mean": bg_mean,
        "mean_signal": float(roi.mean()) if area else 0.0,
        "area_px": area,
    })

if not rows:
    raise SystemExit("No bands defined — open 'Pick bands…' and mark each band.")

n_lanes = len({r["lane"] for r in rows})
n_levels = len({r["level"] for r in rows})
print(f"quantified {len(rows)} band(s) across {n_lanes} lane(s) x {n_levels} level(s)")
if lc_level:
    print(f"loading control level = {lc_level!r}")
else:
    print("no loading-control level set — Normalize will pass IOD through unchanged")
mpfig_data(rows, name="band_iod")

# ── Annotated preview output ──────────────────────────────────
# Always emit an annotated_bands image so the downstream collage /
# timeline has something to show. We build it in stages, each wrapped
# defensively: as long as the basic contrast-stretch succeeds we emit
# SOMETHING (raw membrane). Failing to draw the band-name labels (e.g.,
# missing DejaVuSans font in the frozen sidecar) no longer wipes out
# the entire image.
import sys as _sys
annot = None
try:
    from PIL import Image as _PIL_I, ImageDraw as _PIL_D, ImageFont as _PIL_F
    # Stage 1: contrast-stretched display plane. Always runs.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb_disp = arr[..., :3].astype(np.float32)
    elif arr.ndim == 3:
        g = arr[..., 0]
        rgb_disp = np.stack([g, g, g], axis=-1).astype(np.float32)
    else:
        rgb_disp = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
    _lo, _hi = np.percentile(rgb_disp, [0.5, 99.95])
    _scaled = np.clip((rgb_disp - _lo) / max(_hi - _lo, 1), 0, 1)
    _disp_u8 = np.clip(_scaled ** 0.5 * 255.0, 0, 255).astype(np.uint8)
    annot = _PIL_I.fromarray(_disp_u8).convert("RGB")
except Exception as _e:
    print(f"[band-picker] contrast-stretch failed: {_e}", file=_sys.stderr)
    # Final fallback: emit the raw bytes as a uint8 PNG so the user still
    # sees the membrane (no detection overlay, but no missing-output).
    try:
        from PIL import Image as _PIL_I2
        a8 = np.clip(np.asarray(arr).astype(np.float32), 0, 255).astype(np.uint8)
        if a8.ndim == 2:
            annot = _PIL_I2.fromarray(a8).convert("RGB")
        else:
            annot = _PIL_I2.fromarray(a8[..., :3]).convert("RGB")
    except Exception as _e2:
        print(f"[band-picker] raw fallback also failed: {_e2}", file=_sys.stderr)

# Stage 2: draw the band rectangles. Done in its own try block so a
# font / draw failure doesn't drop the image we already built.
if annot is not None:
    try:
        draw = _PIL_D.Draw(annot)
        PALETTE = ["#1976d2", "#43a047", "#e65100", "#8e24aa", "#00838f",
                   "#c62828", "#5d4037", "#558b2f", "#6a1b9a", "#00695c"]
        level_to_color = {}
        for r in rows:
            lv = str(r.get("level") or "Target")
            if lv not in level_to_color:
                level_to_color[lv] = PALETTE[len(level_to_color) % len(PALETTE)]
        # Try to load DejaVuSans (bundled in the sidecar); fall back to the
        # PIL default font. font.size is unreliable across Pillow versions
        # for the default font, so we cache a fallback size separately.
        try:
            target_font_size = max(10, int(min(H, W) * 0.012))
            font = _PIL_F.truetype("DejaVuSans.ttf", target_font_size)
        except Exception:
            font = _PIL_F.load_default()
            target_font_size = 14
        # Use getattr — Pillow's bitmap default doesn't have .size on
        # every version, but FreeTypeFont does. Either way we fall back.
        font_size = int(getattr(font, "size", None) or target_font_size)
        for band in (CFG.get("lanes", []) or []):
            try:
                x0, y0, x1, y1 = _px(band)
                if x1 <= x0 or y1 <= y0: continue
                lv = (band.get("level") or "Target").strip() or "Target"
                col = level_to_color.get(lv, PALETTE[0])
                nm = (band.get("name") or "").strip()
                sw = max(1, int(round(min(H, W) / 600)))
                is_lc_band = lc_enabled and lc_by_lane.get(str(band.get("label") or ""), "") == nm
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=sw)
                if is_lc_band:
                    draw.rectangle([x0 + sw, y0 + sw, x1 - sw - 1, y1 - sw - 1], outline=col, width=max(1, sw // 2))
                label_y = max(0, y0 - int(font_size * 1.1) - 2)
                draw.text((x0 + 2, label_y), nm or "?", fill=col, font=font)
            except Exception as _be:
                print(f"[band-picker] band draw skipped ({band.get('name', '?')}): {_be}", file=_sys.stderr)
        if lc_enabled:
            for lb in (CFG.get("laneBounds", []) or []):
                try:
                    lx0 = int(round(float(lb.get("x1", 0)) * W))
                    lx1 = int(round(float(lb.get("x2", 0)) * W))
                    ly0 = int(round(float(lb.get("y1", 0)) * H))
                    ly1 = int(round(float(lb.get("y2", 0)) * H))
                    if lx1 > lx0 and ly1 > ly0:
                        draw.rectangle([lx0, ly0, lx1 - 1, ly1 - 1], outline="#2196f3", width=1)
                except Exception: pass
    except Exception as _de:
        print(f"[band-picker] band-overlay draw failed: {_de}", file=_sys.stderr)

# Stage 3: cap + emit. ALWAYS runs if we have any annot image at all.
if annot is not None:
    try:
        SAVE_MAX_W = 3000
        if annot.width > SAVE_MAX_W:
            ratio = SAVE_MAX_W / annot.width
            annot = annot.resize((SAVE_MAX_W, max(1, int(round(annot.height * ratio)))), _PIL_I.LANCZOS)
        mpfig_image(annot, name="annotated_bands")
        print(f"emitted annotated_bands ({annot.width}x{annot.height})")
    except Exception as _ee:
        print(f"[band-picker] mpfig_image emit failed: {_ee}", file=_sys.stderr)
else:
    print("[band-picker] no annotated image could be built — see errors above", file=_sys.stderr)
`;
}

// ── Auto-detect lanes from a grayscale intensity profile ──────
// Western-blot lanes run vertically (one sample per column). We:
//   1. Decide polarity — dark bands on a light membrane (colorimetric)
//      vs light bands on a dark background (chemiluminescence/fluor) —
//      from the image median, so "signal" is always bright.
//   2. Build a column profile of mean signal, smooth it, and pick the
//      contiguous runs above a baseline-relative threshold as lanes.
//   3. For EACH lane, build a row profile within that lane's columns
//      and tighten the ROI to the band's actual vertical extent
//      (largest contiguous run), instead of a generic full-height box.
function smooth1d(arr: Float32Array, k: number): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, n = 0;
    for (let j = -k; j <= k; j++) { const ii = i + j; if (ii >= 0 && ii < arr.length) { s += arr[ii]; n++; } }
    out[i] = s / n;
  }
  return out;
}
/** Contiguous runs where val > thr, each at least minLen long. */
function runsAbove(arr: Float32Array, thr: number, minLen: number): Array<[number, number]> {
  const runs: Array<[number, number]> = [];
  let i = 0;
  while (i < arr.length) {
    if (arr[i] > thr) { const s = i; while (i < arr.length && arr[i] > thr) i++; if (i - s >= minLen) runs.push([s, i]); }
    else i++;
  }
  return runs;
}
function autoDetectLanes(gray: Float32Array, w: number, h: number): BandRoi[] {
  if (w < 4 || h < 4) return [];
  // 1) Polarity from the median (sampled for speed).
  const step = Math.max(1, Math.floor((w * h) / 20000));
  const samp: number[] = [];
  for (let i = 0; i < gray.length; i += step) samp.push(gray[i]);
  samp.sort((a, b) => a - b);
  const median = samp[Math.floor(samp.length / 2)] ?? 128;
  const darkOnLight = median >= 128;
  const sig = (v: number) => (darkOnLight ? 255 - v : v); // bands → bright

  // 2) Column profile → lane x-segments.
  const col = new Float32Array(w);
  for (let x = 0; x < w; x++) { let s = 0; for (let y = 0; y < h; y++) s += sig(gray[y * w + x]); col[x] = s / h; }
  const colS = smooth1d(col, Math.max(2, Math.round(w * 0.01)));
  let cmin = Infinity, cmax = -Infinity;
  for (const v of colS) { if (v < cmin) cmin = v; if (v > cmax) cmax = v; }
  if (cmax - cmin < 1e-3) return [];
  const cThr = cmin + 0.22 * (cmax - cmin);
  const xRuns = runsAbove(colS, cThr, Math.max(3, Math.round(w * 0.012)));

  // 3) Per-lane band y-extent.
  const lanes: BandRoi[] = [];
  for (const [x0, x1] of xRuns) {
    const row = new Float32Array(h);
    for (let y = 0; y < h; y++) { let s = 0; for (let x = x0; x < x1; x++) s += sig(gray[y * w + x]); row[y] = s / (x1 - x0); }
    const rowS = smooth1d(row, Math.max(2, Math.round(h * 0.01)));
    let rmin = Infinity, rmax = -Infinity;
    for (const v of rowS) { if (v < rmin) rmin = v; if (v > rmax) rmax = v; }
    const rThr = rmin + 0.4 * (rmax - rmin);
    const yRuns = runsAbove(rowS, rThr, Math.max(2, Math.round(h * 0.02)));
    // Pick the strongest band (largest integrated signal over the run).
    let best: [number, number] | null = null, bestScore = -1;
    for (const [s, e] of yRuns) { let sum = 0; for (let y = s; y < e; y++) sum += rowS[y] - rThr; if (sum > bestScore) { bestScore = sum; best = [s, e]; } }
    let y0: number, y1: number;
    if (best) { const padY = (best[1] - best[0]) * 0.3; y0 = Math.max(0, best[0] - padY); y1 = Math.min(h, best[1] + padY); }
    else { y0 = h * 0.1; y1 = h * 0.9; }
    const padX = (x1 - x0) * 0.06;
    const nx0 = clamp01((x0 - padX) / w), nx1 = clamp01((x1 + padX) / w);
    lanes.push({
      id: uid(), x: nx0, y: clamp01(y0 / h), w: nx1 - nx0, h: clamp01((y1 - y0) / h),
      label: `S${lanes.length + 1}`, level: "Target",
    });
  }
  return lanes;
}

// ── Approximate IOD for the live readout (display-res, guidance) ─
// Mirrors the emitted Python's quantify(): measurement-plane sum minus the
// mean of the slabs immediately above + below the ROI, but with neighbour
// exclusion — when bands are packed close together (3-row WB blots etc.)
// the above/below slabs sit on TOP of other bands, so a naive mean is too
// high and the corrected IOD clamps to 0. We skip pixels that fall inside
// any other box, and if too few clean pixels remain, fall back to a 5th-
// percentile-of-ROI floor (a per-band background that doesn't depend on
// neighbouring regions at all).
function approxIod(
  gray: Float32Array, w: number, h: number,
  lane: { x: number; y: number; w: number; h: number },
  others: Array<{ x: number; y: number; w: number; h: number }>,
): number {
  const x0 = Math.max(0, Math.round(lane.x * w));
  const x1 = Math.min(w, Math.round((lane.x + lane.w) * w));
  const y0 = Math.max(0, Math.round(lane.y * h));
  const y1 = Math.min(h, Math.round((lane.y + lane.h) * h));
  if (x1 <= x0 || y1 <= y0) return 0;
  const hh = y1 - y0;
  // ROI: raw sum + the values themselves (for the percentile fallback).
  let roiSum = 0; const roiVals: number[] = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const v = 255 - gray[y * w + x];
    roiSum += v; roiVals.push(v);
  }
  const n = roiVals.length;
  // Other-box rects in pixel coords for fast point-in-box checks.
  const others_px = others.map((o) => ({
    x0: Math.round(o.x * w), x1: Math.round((o.x + o.w) * w),
    y0: Math.round(o.y * h), y1: Math.round((o.y + o.h) * h),
  }));
  const inOther = (x: number, y: number) => {
    for (const o of others_px) if (x >= o.x0 && x < o.x1 && y >= o.y0 && y < o.y1) return true;
    return false;
  };
  // Local background slabs above + below, same width, gap=6, neighbour-excluded.
  const gap = 6;
  const ay2 = Math.max(0, y0 - gap), ay1 = Math.max(0, ay2 - hh);
  const by1Lo = Math.min(h, y1 + gap), by2 = Math.min(h, by1Lo + hh);
  let bgSum = 0, bgN = 0;
  for (let y = ay1; y < ay2; y++) for (let x = x0; x < x1; x++) {
    if (!inOther(x, y)) { bgSum += 255 - gray[y * w + x]; bgN++; }
  }
  for (let y = by1Lo; y < by2; y++) for (let x = x0; x < x1; x++) {
    if (!inOther(x, y)) { bgSum += 255 - gray[y * w + x]; bgN++; }
  }
  let bgMean: number;
  if (bgN >= Math.max(20, Math.floor(n / 4))) {
    bgMean = bgSum / bgN;
  } else {
    // Densely-packed bands → no clean slab pixels. Fall back to the 5th
    // percentile of the ROI itself (per-band floor — what the picker used
    // before the slab method was introduced).
    const sorted = roiVals.slice().sort((a, b) => a - b);
    bgMean = sorted[Math.floor(sorted.length * 0.05)] || 0;
  }
  return Math.max(0, roiSum - bgMean * n);
}

interface DragState {
  mode: "move" | "resize" | "draw";
  id: string | null;
  handle?: string;
  startNx: number; startNy: number;
  orig: { x: number; y: number; w: number; h: number };
}

interface BandPickerDialogProps {
  open: boolean;
  /** Membrane image as a base64 PNG (no data: prefix) or a data URL — used for
   *  the editor DISPLAY and as the detection fallback. */
  imageSrc: string | null;
  /** Source descriptor so the backend can re-extract the FULL-RES (and
   *  full-bit-depth) image for detection. Either a builder inset
   *  (row/col/inset_index) or a standalone analysis upload ({name}). */
  source?: { key: string; name?: string; row?: number; col?: number; inset_index?: number } | null;
  initial: BandPickerConfig | null;
  onClose: () => void;
  onSave: (cfg: BandPickerConfig) => void;
}

const MAX_DISP_W = 900;
const MAX_DISP_H = 760;

export default function BandPickerDialog(props: BandPickerDialogProps) {
  const { open, imageSrc, source, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<BandPickerConfig>(initial ?? emptyBandConfig());
  const [selId, setSelId] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectInfo, setDetectInfo] = useState<string | null>(null);
  const [natW, setNatW] = useState(0);
  const [natH, setNatH] = useState(0);
  const [gray, setGray] = useState<Float32Array | null>(null);
  // Contrast-stretched preview from the backend (bright, clearly-visible bands).
  // Replaces the dark raw thumbnail in the display once auto-detect has run.
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  // Measured size of the image area so the image fills it (no empty space).
  const [box, setBox] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  // Zoom + pan over the membrane preview. zoom = 1 fits the image to the
  // available area (the previous behaviour); >1 magnifies, centred via panX/Y
  // (translation in CSS-pixel space — applied as a CSS transform on the
  // image+overlay wrapper, so the SVG geometry stays in disp-pixel coords
  // and pointer→normalised math doesn't need to know about zoom).
  const [view, setView] = useState<{ zoom: number; panX: number; panY: number }>({ zoom: 1, panX: 0, panY: 0 });
  // Optional level filter for the Groups chip rows (so picking from many
  // bands at the same MW row stays manageable).
  // Per-group "Show: <level>" filter. Each group has its own chip row, so
  // narrowing group A's band candidates to level G1 doesn't change what
  // group B can see. Map: groupId → level (or "" / undefined = show all).
  const [groupLevelFilter, setGroupLevelFilter] = useState<Record<string, string>>({});
  const svgRef = useRef<SVGSVGElement | null>(null);
  const imgAreaRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);

  // Reset state whenever the dialog (re)opens with fresh inputs.
  useEffect(() => {
    if (open) {
      setCfg(initial ? structuredClone(initial) : emptyBandConfig());
      setSelId(null);
      setDetectInfo(null);
      setPreviewSrc(null);
      setView({ zoom: 1, panX: 0, panY: 0 });
    }
  }, [open, initial]);

  // High-resolution preview on open. The upstream Source node's image_b64 is
  // a sub-512 px thumbnail (that's all collectInputs ships through the graph),
  // so without this the band picker would open showing a soft, low-resolution
  // membrane until the user runs Auto-detect. We hit the lightweight
  // /api/analysis/wb-preview endpoint (no detection — just contrast-stretch
  // + 3k px PNG) so the display upgrades to native-ish detail immediately.
  useEffect(() => {
    if (!open) return;
    if (!source && !imageSrc) return;
    let cancelled = false;
    (async () => {
      try {
        const b64 = imageSrc ? (imageSrc.startsWith("data:") ? imageSrc.split(",")[1] : imageSrc) : "";
        const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
        const resp = await fetch(`${apiBase}/api/analysis/wb-preview`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64, source: source ?? undefined }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        if (cancelled) return;
        if (data?.preview_b64) {
          setPreviewSrc(`data:image/png;base64,${data.preview_b64}`);
        }
      } catch { /* backend unreachable → stay on the thumbnail, no harm */ }
    })();
    return () => { cancelled = true; };
  }, [open, source, imageSrc]);

  const srcUrl = useMemo(() => {
    if (!imageSrc) return null;
    return imageSrc.startsWith("data:") ? imageSrc : `data:image/png;base64,${imageSrc}`;
  }, [imageSrc]);

  // The image actually shown: the bright contrast-stretched preview once
  // auto-detect has run, otherwise the raw thumbnail.
  const dispUrl = previewSrc || srcUrl;

  // Decode the displayed image to a grayscale buffer for auto-detect + IOD,
  // and to drive the display geometry (so the box sizes to the shown image).
  useEffect(() => {
    if (!dispUrl) { setGray(null); setNatW(0); setNatH(0); return; }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const w = img.naturalWidth, h = img.naturalHeight;
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, w, h).data;
      const g = new Float32Array(w * h);
      for (let i = 0, p = 0; i < g.length; i++, p += 4) {
        g[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      }
      setNatW(w); setNatH(h); setGray(g);
    };
    img.src = dispUrl;
    return () => { cancelled = true; };
  }, [dispUrl]);

  // Measure the image area (ResizeObserver) so the image fills the available
  // space instead of a fixed cap — no empty gap, image as large as it fits.
  useEffect(() => {
    if (!open) return;
    const el = imgAreaRef.current;
    if (!el) return;
    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open, srcUrl, previewSrc]);

  // Display geometry — fill the measured image area, preserving aspect.
  // Allows upscaling so a small thumbnail still expands; the post-detect
  // preview (~1100px) downscales to fit crisply.
  const disp = useMemo(() => {
    const aw = box.w || MAX_DISP_W;
    const ah = box.h || MAX_DISP_H;
    if (!natW || !natH) return { w: Math.max(1, aw), h: Math.min(ah, 360) };
    const scale = Math.min(aw / natW, ah / natH);
    return { w: Math.max(1, Math.round(natW * scale)), h: Math.max(1, Math.round(natH * scale)) };
  }, [natW, natH, box]);

  // Live IOD per lane (approximate, from display-res grayscale).
  const iodById = useMemo(() => {
    const out: Record<string, number> = {};
    if (gray && natW && natH) {
      for (const lane of cfg.lanes) {
        const others = cfg.lanes.filter((o) => o.id !== lane.id);
        out[lane.id] = approxIod(gray, natW, natH, lane, others);
      }
    }
    return out;
  }, [gray, natW, natH, cfg]);

  const clientToNorm = useCallback((clientX: number, clientY: number) => {
    const el = svgRef.current;
    if (!el) return { nx: 0, ny: 0 };
    const r = el.getBoundingClientRect();
    return { nx: clamp01((clientX - r.left) / r.width), ny: clamp01((clientY - r.top) / r.height) };
  }, []);

  /** Add a new band of the same size as the rest at a convenient spot
   *  (offset from the selected box, or centred on the image). Triggered by
   *  the "+ Band" toolbar button — keeps every box uniform with auto-detect
   *  so quantification stays apples-to-apples across the membrane. */
  const addBand = useCallback(() => {
    // Same box dimensions as auto-detect (or median of existing boxes,
    // falling back to a reasonable 5%×3% default).
    const stored = cfg.defaultBoxW && cfg.defaultBoxH;
    let bw = stored ? cfg.defaultBoxW! : 0;
    let bh = stored ? cfg.defaultBoxH! : 0;
    if (!stored && cfg.lanes.length > 0) {
      const ws = cfg.lanes.map((l) => l.w).sort((a, b) => a - b);
      const hs = cfg.lanes.map((l) => l.h).sort((a, b) => a - b);
      bw = ws[Math.floor(ws.length / 2)] || 0.05;
      bh = hs[Math.floor(hs.length / 2)] || 0.03;
    }
    if (!bw || !bh) { bw = 0.05; bh = 0.03; }

    // Anchor near the selected box (offset by one box-height below) so
    // the user can build a column quickly; fall back to centre.
    const sel = cfg.lanes.find((l) => l.id === selId);
    let x: number, y: number;
    if (sel) {
      x = clamp01(sel.x);
      y = clamp01(sel.y + sel.h + bh * 0.4);
      if (y + bh > 1) y = clamp01(sel.y - bh * 1.4);
    } else {
      x = 0.5 - bw / 2; y = 0.5 - bh / 2;
    }
    const lastLevel = cfg.lanes.length ? cfg.lanes[cfg.lanes.length - 1].level || "Target" : "Target";
    const newId = uid("band");
    // Snap to lane: prefer the detected lane whose horizontal extent contains
    // the new box's x-center. Falls back to an aligned existing box's lane,
    // then to a fresh "S{n}" if there's nothing nearby.
    const cx = x + bw / 2;
    const aligned = cfg.lanes.find((l) => Math.abs((l.x + l.w / 2) - cx) < bw * 0.6);
    let lane = aligned ? aligned.label : `S${cfg.lanes.filter((l) => !/^L\d*$|^Ladder|^Marker/i.test(l.label)).length + 1}`;
    if (cfg.laneBounds && cfg.laneBounds.length > 0) lane = snapLaneFromBounds(cx, cfg.laneBounds, lane);
    const newBand: BandRoi = {
      id: newId, x, y, w: bw, h: bh,
      name: `B${cfg.lanes.length + 1}`,
      label: lane, level: lastLevel,
    };
    setCfg((c) => ({ ...c, lanes: [...c.lanes, newBand] }));
    setSelId(newId);
  }, [cfg.lanes, cfg.defaultBoxW, cfg.defaultBoxH, cfg.laneBounds, selId]);

  /** Wheel → zoom around the cursor (cursor stays anchored to the same
   *  point in image space). Trackpad pinches arrive as ctrl+wheel; both
   *  paths route through here. */
  const onWheel = useCallback((e: React.WheelEvent) => {
    if (!disp.w || !disp.h) return;
    e.preventDefault();
    const ar = imgAreaRef.current?.getBoundingClientRect();
    if (!ar) return;
    const cx = e.clientX - ar.left;   // cursor in imgArea coords
    const cy = e.clientY - ar.top;
    // Center offset — the wrapper is positioned at (cxOff, cyOff) within
    // imgArea when pan=0, so it appears flex-centred at zoom=1.
    const cxOff = Math.max(0, (ar.width - disp.w) / 2);
    const cyOff = Math.max(0, (ar.height - disp.h) / 2);
    setView((v) => {
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = Math.max(1, Math.min(8, v.zoom * factor));
      // Anchor the cursor: cursor in pre-transform wrapper coords is
      //   (cx - cxOff - panX) / zoom
      // After zoom change we want that same wrapper-coord to land at (cx).
      const wx = (cx - cxOff - v.panX) / v.zoom;
      const wy = (cy - cyOff - v.panY) / v.zoom;
      let panX = cx - cxOff - wx * next;
      let panY = cy - cyOff - wy * next;
      // Clamp pan so the wrapper edge doesn't leave the viewport.
      const minX = ar.width - cxOff - disp.w * next;
      const minY = ar.height - cyOff - disp.h * next;
      panX = Math.min(-cxOff > 0 ? 0 : -cxOff, Math.max(minX < 0 ? minX : 0, panX));
      panY = Math.min(-cyOff > 0 ? 0 : -cyOff, Math.max(minY < 0 ? minY : 0, panY));
      // Reset pan to 0 when we're back at zoom=1 so the image re-centres.
      if (next <= 1.001) return { zoom: 1, panX: 0, panY: 0 };
      return { zoom: next, panX, panY };
    });
  }, [disp.w, disp.h]);

  /** Keyboard-driven nudging + delete for the currently selected box. */
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack arrow keys when the user is typing into a TextField.
      const t = e.target as HTMLElement | null;
      const tag = (t?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || (t as HTMLElement)?.isContentEditable) return;
      if (!selId) return;
      const step = e.shiftKey ? 0.01 : 0.002;  // shift = coarser step
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === selId ? { ...l, x: clamp01(l.x - step) } : l) }));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === selId ? { ...l, x: clamp01(Math.min(1 - l.w, l.x + step)) } : l) }));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === selId ? { ...l, y: clamp01(l.y - step) } : l) }));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === selId ? { ...l, y: clamp01(Math.min(1 - l.h, l.y + step)) } : l) }));
      } else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        setCfg((c) => ({ ...c, lanes: c.lanes.filter((l) => l.id !== selId) }));
        setSelId(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, selId]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const t = e.target as Element;
    const role = t.getAttribute("data-role");
    const id = t.getAttribute("data-id");
    const handle = t.getAttribute("data-handle") || undefined;
    const { nx, ny } = clientToNorm(e.clientX, e.clientY);
    (e.target as Element).setPointerCapture?.(e.pointerId);

    if (role === "handle" && id) {
      const lane = cfg.lanes.find((l) => l.id === id);
      if (lane) { setSelId(id); dragRef.current = { mode: "resize", id, handle, startNx: nx, startNy: ny, orig: { ...lane } }; }
      return;
    }
    if (role === "lane" && id) {
      const lane = cfg.lanes.find((l) => l.id === id);
      if (lane) { setSelId(id); dragRef.current = { mode: "move", id, startNx: nx, startNy: ny, orig: { ...lane } }; }
      return;
    }
    // Empty area → draw a new band rectangle.
    const newId = uid();
    dragRef.current = { mode: "draw", id: newId, startNx: nx, startNy: ny, orig: { x: nx, y: ny, w: 0, h: 0 } };
    setCfg((c) => {
      // Inherit the level (row) from the selected band so you can add more
      // bands to the same level by drawing; fall back to the last band's level.
      const sel = c.lanes.find((l) => l.id === selId);
      const defLevel = sel ? roiLevel(sel) : (c.lanes.length ? roiLevel(c.lanes[c.lanes.length - 1]) : "Target");
      // Snap the new box's lane label to the lane bounds at the click position
      // (or default to "S{n}" when no detection has run yet).
      const defLane = snapLaneFromBounds(nx, c.laneBounds, `S${c.lanes.length + 1}`);
      const nextBandIdx = c.lanes.length + 1;
      return {
        ...c,
        lanes: [...c.lanes, {
          id: newId, x: nx, y: ny, w: 0, h: 0,
          name: `B${nextBandIdx}`, label: defLane, level: defLevel,
        }],
      };
    });
    setSelId(newId);
  }, [cfg.lanes, clientToNorm, selId]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const { nx, ny } = clientToNorm(e.clientX, e.clientY);
    const dx = nx - d.startNx, dy = ny - d.startNy;

    const resizeRect = (o: { x: number; y: number; w: number; h: number }, handle: string | undefined) => {
      let { x, y, w, h } = o;
      const right = o.x + o.w, bottom = o.y + o.h;
      if (handle?.includes("w")) { x = clamp01(o.x + dx); w = right - x; }
      if (handle?.includes("e")) { w = clamp01(right + dx) - x; }
      if (handle?.includes("n")) { y = clamp01(o.y + dy); h = bottom - y; }
      if (handle?.includes("s")) { h = clamp01(bottom + dy) - y; }
      return { x, y, w: Math.max(0, w), h: Math.max(0, h) };
    };

    if (d.mode === "draw") {
      const x = Math.min(d.startNx, nx), y = Math.min(d.startNy, ny);
      const w = Math.abs(nx - d.startNx), h = Math.abs(ny - d.startNy);
      if (d.id) {
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, x, y, w, h } : l) }));
      }
    } else if (d.mode === "move" && d.id) {
      const x = clamp01(d.orig.x + dx), y = clamp01(d.orig.y + dy);
      setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, x: Math.min(x, 1 - l.w), y: Math.min(y, 1 - l.h) } : l) }));
    } else if (d.mode === "resize" && d.id) {
      const r = resizeRect(d.orig, d.handle);
      setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, ...r } : l) }));
    }
  }, [clientToNorm]);

  const onPointerUp = useCallback(() => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;
    // Discard zero-area scribbles.
    if (d.mode === "draw" && d.id) {
      setCfg((c) => {
        const lane = c.lanes.find((l) => l.id === d.id);
        if (lane && (lane.w < 0.01 || lane.h < 0.01)) return { ...c, lanes: c.lanes.filter((l) => l.id !== d.id) };
        return c;
      });
    }
  }, []);

  const setName = (id: string, name: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === id ? { ...l, name } : l) }));
  const setLabel = (id: string, label: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === id ? { ...l, label } : l) }));
  const setLevel = (id: string, level: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === id ? { ...l, level } : l) }));
  const setLoadingControlLevel = (level: string | null) =>
    setCfg((c) => ({ ...c, loadingControlLevel: level }));
  const deleteLane = (id: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.filter((l) => l.id !== id) }));

  // Groups (condition mappings used by the downstream R plot to build
  // `conditions <- list(...)` automatically). Each group is a free-form
  // set of individual BAND names — so the user can compare any subset of
  // bands against any other subset, regardless of lane or level.
  const groups: BandGroup[] = cfg.groups ?? [];
  // Helper: membership in a group is "bands" first, "lanes" fallback (so
  // old saved configs still light up correctly until the user re-assigns).
  const groupBands = (g: BandGroup): string[] => g.bands ?? [];
  const isBandInGroup = (g: BandGroup, bandName: string, laneLabel: string) =>
    (g.bands ?? []).includes(bandName) || (g.lanes ?? []).includes(laneLabel);

  const addGroup = () => {
    const used = new Set(groups.map((g) => g.name));
    let n = 1; while (used.has(`G${n}`)) n++;
    const g: BandGroup = { id: uid("group"), name: `G${n}`, bands: [] };
    setCfg((c) => ({ ...c, groups: [...(c.groups ?? []), g] }));
  };
  const renameGroup = (id: string, name: string) =>
    setCfg((c) => ({ ...c, groups: (c.groups ?? []).map((g) => g.id === id ? { ...g, name } : g) }));
  const deleteGroup = (id: string) =>
    setCfg((c) => ({ ...c, groups: (c.groups ?? []).filter((g) => g.id !== id) }));
  const toggleBandInGroup = (gid: string, bandName: string) =>
    setCfg((c) => ({
      ...c,
      groups: (c.groups ?? []).map((g) => {
        if (g.id !== gid) return g;
        const current = g.bands ?? [];
        const has = current.includes(bandName);
        return { ...g, bands: has ? current.filter((b) => b !== bandName) : [...current, bandName] };
      }),
    }));

  // Distinct levels (rows) present, in first-seen order, for coloring + the
  // loading-control selector.
  const levelOrder = useMemo(() => {
    const seen: string[] = [];
    for (const l of cfg.lanes) { const lv = roiLevel(l); if (!seen.includes(lv)) seen.push(lv); }
    return seen;
  }, [cfg.lanes]);

  const applyDetectedLanes = (
    rects: Array<{ x: number; y: number; w: number; h: number; lane?: string; level?: string }>,
    bounds?: LaneBound[],
  ): boolean => {
    if (!rects.length) return false;
    // Each band gets a UNIQUE display name (B1..BN) AND a lane label (the
    // column identity, shared across same-column bands so the loading-control
    // normalisation can group them). Keep the ladder (lane "L", level "Ladder").
    const lanes: BandRoi[] = rects.map((r, i) => ({
      id: uid("band"), x: clamp01(r.x), y: clamp01(r.y), w: clamp01(r.w), h: clamp01(r.h),
      name: `B${i + 1}`,
      label: (r.lane || `S${i + 1}`), level: (r.level || "Target"),
    }));
    // Median ROI size from the detector = the auto-detect 74×33 box at this
    // image's resolution. Stored so the "+ Band" button and snap-to-uniform
    // on drag-release keep every box exactly the same size as the rest.
    const ws = rects.map((r) => r.w).sort((a, b) => a - b);
    const hs = rects.map((r) => r.h).sort((a, b) => a - b);
    const defaultBoxW = ws[Math.floor(ws.length / 2)] || 0.05;
    const defaultBoxH = hs[Math.floor(hs.length / 2)] || 0.03;
    // Seed sensible LC defaults: the LOWEST-row band in each lane (the
    // canonical bottom-of-membrane housekeeping band). Doesn't enable LC
    // — the user still has to toggle "Normalize to loading control" — but
    // when they do, every lane already has a pre-picked LC band.
    const seededLc = defaultLcByLane(lanes);
    setCfg((c) => ({
      ...c, lanes, defaultBoxW, defaultBoxH,
      laneBounds: bounds && bounds.length ? bounds : (c.laneBounds ?? []),
      lcBandByLane: seededLc,
    }));
    setSelId(lanes[0]?.id ?? null);
    return true;
  };

  const runAutoDetect = async () => {
    if (detecting) return;
    setDetecting(true);
    setDetectInfo(null);
    try {
      // Backend detector — vendored detect_bands_equal_boxes.py. Defaults
      // mirror the CLI exactly (expected_lanes=8, threshold_percentile=98,
      // min_gap=45, min_group_lanes=3, roi 74×33, channel=max), so the
      // result is byte-for-byte what the reference script's annotated PNG /
      // CSV would emit on the same input.
      if (imageSrc || source) {
        try {
          const b64 = imageSrc ? (imageSrc.startsWith("data:") ? imageSrc.split(",")[1] : imageSrc) : "";
          const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
          const resp = await fetch(`${apiBase}/api/analysis/wb-detect-bands`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_b64: b64, source: source ?? undefined }),
          });
          if (resp.ok) {
            const data = await resp.json();
            if (Array.isArray(data.lanes) && applyDetectedLanes(data.lanes, data.lane_bounds)) {
              if (data.preview_b64) setPreviewSrc(`data:image/png;base64,${data.preview_b64}`);
              const sw = Number(data.src_w) || 0;
              const sh = Number(data.src_h) || 0;
              const raw = !!data.raw_depth;
              const used = !!data.used_source;
              if (!used || (sw > 0 && sw < 700)) {
                setDetectInfo(
                  `⚠ Detected on a ${sw || "low"}px preview — the full-resolution membrane wasn't available, ` +
                  `so results are coarse. Re-drop the original blot onto the Source node and auto-detect again.`,
                );
              } else {
                setDetectInfo(
                  `Detected ${data.band_count ?? data.lanes.length} band(s) on ${sw}×${sh} px ` +
                  `(${raw ? "16-bit raw" : "8-bit display"} pixels). ` +
                  `Same pipeline + defaults as detect_bands_equal_boxes.py — drag any box to fine-tune.`,
                );
              }
              return;
            }
          }
        } catch { /* backend unavailable → local fallback */ }
      }
      if (gray && natW && natH) {
        applyDetectedLanes(autoDetectLanes(gray, natW, natH));
        setDetectInfo("Detected with the in-app fallback (backend unavailable) — clean up as needed.");
      }
    } finally {
      setDetecting(false);
    }
  };

  // Render the 4 CORNER handles only — small dots placed slightly OUTSIDE
  // the box so they don't obscure the band underneath. A large transparent
  // hit-area square (~14 px) sits over each dot to keep them easy to grab.
  // Edge midpoint handles are gone (they were rarely used and the corners
  // already let you change both axes; cleaner band view, less clutter).
  const handlesFor = (
    rect: { x: number; y: number; w: number; h: number },
    id: string | null,
    roleHandle: string,
    _selected: boolean = false,
  ) => {
    const corners = [
      { h: "nw", cx: rect.x,            cy: rect.y,            ox: -1, oy: -1 },
      { h: "ne", cx: rect.x + rect.w,   cy: rect.y,            ox:  1, oy: -1 },
      { h: "sw", cx: rect.x,            cy: rect.y + rect.h,   ox: -1, oy:  1 },
      { h: "se", cx: rect.x + rect.w,   cy: rect.y + rect.h,   ox:  1, oy:  1 },
    ];
    const cur: Record<string, string> = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };
    // Visible dot stays ~5 px on-screen at any zoom. Hit-target is ~14 px so
    // grabbing stays easy. Both are positioned slightly OUTSIDE the corner.
    const z = Math.max(1, view.zoom);
    const dotSz = 5 / z;
    const hitSz = 14 / z;
    const dotOff = 3 / z;   // how far the dot sits outside the corner
    return corners.flatMap((c) => {
      const cxPx = c.cx * disp.w + c.ox * dotOff;
      const cyPx = c.cy * disp.h + c.oy * dotOff;
      return [
        // Generous hit area (invisible).
        <rect key={`${c.h}-hit`} data-role={roleHandle} data-id={id ?? undefined} data-handle={c.h}
          x={cxPx - hitSz / 2} y={cyPx - hitSz / 2} width={hitSz} height={hitSz}
          fill="transparent" style={{ cursor: cur[c.h] }} />,
        // Tiny visible dot — pointer-events none so the hit rect catches clicks.
        <rect key={`${c.h}-dot`} pointerEvents="none"
          x={cxPx - dotSz / 2} y={cyPx - dotSz / 2} width={dotSz} height={dotSz}
          fill="#fff" stroke="#1976d2" strokeWidth={1.2 / z} rx={1 / z} />,
      ];
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xl" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem", py: 1.25 }}>
        🩻 Pick bands
        <Typography component="span" variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
          Auto-detect, then refine — scroll to zoom · drag empty area to add · arrow keys nudge selected box · Delete to remove
        </Typography>
      </DialogTitle>
      {/* Image on the RIGHT and FLEX-FILLING, so the user gets a clear look at
          the membrane; controls are a fixed column on the left. */}
      <DialogContent dividers sx={{ display: "flex", flexDirection: "row-reverse", gap: 2, alignItems: "stretch", height: "80vh" }}>
        {/* Image + ROI overlay — flex-fills the remaining width/height, image
            centred and scaled to fill it (see `disp`, measured from this box). */}
        <Box ref={imgAreaRef} onWheel={onWheel}
          sx={{ flex: 1, minWidth: 0, minHeight: 0, overflow: "hidden", position: "relative" }}>
          {/* Zoom-out control — visible only when zoomed in; resets the view. */}
          {dispUrl && view.zoom > 1.0001 && (
            <Box sx={{ position: "absolute", top: 6, left: 6, zIndex: 5, display: "flex", gap: 0.5 }}>
              <Button size="small" variant="contained" color="inherit" sx={{ fontSize: "0.65rem", py: 0.2, px: 1, minWidth: 0, textTransform: "none" }}
                onClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}>
                ⤢ Reset zoom ({view.zoom.toFixed(1)}×)
              </Button>
            </Box>
          )}
          {dispUrl ? (
            <Box sx={{
              position: "absolute",
              // Center the wrapper at zoom=1 (cxOff/cyOff baked in here);
              // panX/Y from the wheel handler adds on top. Box dims fall back
              // to 0 when the measurement isn't ready (initial mount).
              left: Math.max(0, ((box.w || 0) - disp.w) / 2),
              top: Math.max(0, ((box.h || 0) - disp.h) / 2),
              width: disp.w, height: disp.h, userSelect: "none",
              border: "1px solid", borderColor: "divider",
              transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`,
              transformOrigin: "0 0",
            }}>
              <Box component="img" src={dispUrl} alt="blot" draggable={false}
                sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "fill", pointerEvents: "none" }} />
              <svg ref={svgRef} width={disp.w} height={disp.h}
                style={{ position: "absolute", inset: 0, cursor: "crosshair", touchAction: "none" }}
                onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}>
                {/* Lane guide lines — visible when LC normalisation is on, so
                    the user sees which vertical strip each box belongs to.
                    Drawn UNDER the bands so they don't fight for attention. */}
                {cfg.loadingControlEnabled && (cfg.laneBounds || []).map((lb) => {
                  const x = lb.x1 * disp.w;
                  const w = (lb.x2 - lb.x1) * disp.w;
                  const y = lb.y1 * disp.h;
                  const h = (lb.y2 - lb.y1) * disp.h;
                  const sw = 1 / Math.max(1, view.zoom);
                  const fs = 10 / Math.max(1, view.zoom);
                  return (
                    <g key={`lane-${lb.name}`} pointerEvents="none">
                      <rect x={x} y={y} width={w} height={h}
                        fill="rgba(33,150,243,0.04)"
                        stroke="rgba(33,150,243,0.55)"
                        strokeWidth={sw} strokeDasharray={`${4 / view.zoom} ${3 / view.zoom}`} />
                      <text x={x + 2 / view.zoom} y={y - 2 / view.zoom}
                        fontSize={fs} fill="rgba(33,150,243,0.85)"
                        style={{ fontWeight: 700 }}>
                        {lb.name}
                      </text>
                    </g>
                  );
                })}
                {/* Bands — colored by level (row); same level = same color.
                    Stroke widths are scaled by 1/zoom so the visible thickness
                    stays constant when the user zooms in. */}
                {cfg.lanes.map((lane) => {
                  const sel = lane.id === selId;
                  const lv = roiLevel(lane);
                  const col = levelColor(lv, levelOrder);
                  const isLC = cfg.loadingControlEnabled
                    ? ((cfg.lcBandByLane || {})[lane.label || ""] === roiName(lane))
                    : (cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel);
                  const sw = (sel ? 2.5 : 1.5) / Math.max(1, view.zoom);
                  const fs = 11 / Math.max(1, view.zoom);
                  return (
                    <g key={lane.id}>
                      <rect data-role="lane" data-id={lane.id}
                        x={lane.x * disp.w} y={lane.y * disp.h} width={lane.w * disp.w} height={lane.h * disp.h}
                        fill={col} fillOpacity={sel ? 0.22 : 0.1}
                        stroke={col} strokeWidth={sw}
                        strokeDasharray={isLC ? `${4 / view.zoom} ${2 / view.zoom}` : undefined}
                        style={{ cursor: "move" }} />
                      <text x={lane.x * disp.w + 3 / view.zoom} y={lane.y * disp.h + 12 / view.zoom}
                        fontSize={fs} fill={col} style={{ pointerEvents: "none", fontWeight: 700 }}>
                        {roiName(lane)}{isLC ? " (LC)" : ""}
                      </text>
                      {sel && handlesFor(lane, lane.id, "handle", true)}
                    </g>
                  );
                })}
              </svg>
            </Box>
          ) : (
            <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", border: "1px dashed", borderColor: "divider", color: "text.secondary", textAlign: "center", p: 3 }}>
              <Typography variant="body2">
                No blot image reached this node.<br />Wire a Source node (with the whole membrane dropped on it) into this Band picker, then reopen.
              </Typography>
            </Box>
          )}
        </Box>

        {/* Controls — fixed-width column on the left, scrolls if tall. */}
        <Box sx={{ width: 380, flexShrink: 0, display: "flex", flexDirection: "column", gap: 1.25, overflowY: "auto", overflowX: "hidden" }}>
          <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
            <Button size="small" variant="contained" startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
              onClick={runAutoDetect} disabled={!srcUrl || detecting}>
              {detecting ? "Detecting…" : "Auto-detect"}
            </Button>
            <Tooltip title="Add a new band of the same size as the rest. The new box is offset below the selected one (or centred when nothing is selected); same column/lane as the box above it.">
              <span>
                <Button size="small" variant="outlined" onClick={addBand} disabled={!dispUrl}
                  sx={{ textTransform: "none" }}>
                  + Band
                </Button>
              </span>
            </Tooltip>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto" }}>
              {cfg.lanes.length} band(s)
            </Typography>
          </Box>
          {detectInfo && (
            <Typography variant="caption" sx={{
              color: detectInfo.startsWith("⚠") ? "warning.main" : "text.secondary",
              lineHeight: 1.4, mt: -0.5,
            }}>
              {detectInfo}
            </Typography>
          )}

          {/* Loading-control normalisation — opt-in toggle. OFF (default) →
              raw IOD passes through, no Normalize step in the plot. ON →
              user picks one BAND per lane as that lane's loading control;
              every band in that lane is divided by its LC band's IOD. */}
          {(() => {
            const lcEnabled = !!cfg.loadingControlEnabled;
            const lanesPresent = Array.from(new Set(cfg.lanes.map((l) => (l.label || "Lane").trim() || "Lane")));
            const lcMap = cfg.lcBandByLane || {};
            const lcCount = lanesPresent.filter((ln) => lcMap[ln]).length;
            return (
              <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Tooltip title="When ON: each lane's bands are divided by that lane's chosen loading-control band's IOD → relative density (vs LC). Pick a different LC band per lane below. When OFF: raw IOD is plotted directly (no Normalize step).">
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      Normalize to loading control
                    </Typography>
                  </Tooltip>
                  <Box
                    role="switch"
                    aria-checked={lcEnabled}
                    onClick={() => {
                      setCfg((c) => {
                        const next = !c.loadingControlEnabled;
                        // First time enabling → seed defaults (lowest-row band per lane).
                        const seed = next && Object.keys(c.lcBandByLane || {}).length === 0
                          ? defaultLcByLane(c.lanes) : (c.lcBandByLane || {});
                        return { ...c, loadingControlEnabled: next, lcBandByLane: seed };
                      });
                    }}
                    sx={{
                      cursor: "pointer", width: 32, height: 18, borderRadius: 9,
                      bgcolor: lcEnabled ? "primary.main" : "divider",
                      position: "relative", transition: "background 0.15s",
                      "&:before": {
                        content: '""', position: "absolute", top: 2,
                        left: lcEnabled ? 16 : 2, width: 14, height: 14,
                        borderRadius: "50%", bgcolor: "background.paper",
                        transition: "left 0.15s",
                      },
                    }}
                  />
                  <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto" }}>
                    {lcEnabled
                      ? `${lcCount}/${lanesPresent.length} lane(s) configured`
                      : "raw IOD will be plotted (Normalize step skipped)"}
                  </Typography>
                </Box>
                {/* Plot layout: combined facets vs one figure per group. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5, pt: 0.5, borderTop: "1px dashed", borderColor: "divider" }}>
                  <Tooltip title="Combined: all groups share one figure (faceted side-by-side). Separate: emit one PNG per group, each can be moved into the figure timeline independently.">
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      Plot layout
                    </Typography>
                  </Tooltip>
                  <Box
                    role="switch"
                    aria-checked={!!cfg.plotPerGroup}
                    onClick={() => setCfg((c) => ({ ...c, plotPerGroup: !c.plotPerGroup }))}
                    sx={{
                      cursor: "pointer", width: 32, height: 18, borderRadius: 9,
                      bgcolor: cfg.plotPerGroup ? "primary.main" : "divider",
                      position: "relative", transition: "background 0.15s",
                      "&:before": {
                        content: '""', position: "absolute", top: 2,
                        left: cfg.plotPerGroup ? 16 : 2, width: 14, height: 14,
                        borderRadius: "50%", bgcolor: "background.paper",
                        transition: "left 0.15s",
                      },
                    }}
                  />
                  <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto", fontSize: "0.65rem" }}>
                    {cfg.plotPerGroup ? "one figure per group" : "combined (faceted)"}
                  </Typography>
                </Box>
                {/* Image orientation — drives whether IOD is computed as
                    sum(pixel) (light-on-dark; fluorescent) or
                    sum(vmax - pixel) (dark-on-light; chemilum / film).
                    Default "auto" picks by the median of the measurement
                    plane.  Wrong only when the image was contrast-
                    adjusted in a way that fooled the heuristic — in
                    that case the user picks explicitly here. */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5, pt: 0.5, borderTop: "1px dashed", borderColor: "divider", flexWrap: "wrap" }}>
                  <Tooltip title="How are the bands rendered in this image? Bright (light-on-dark, fluorescent / LI-COR) → sum raw pixels = signal. Dark (dark-on-light, chemilum / film) → invert plane (vmax − pixel) so darker bands give larger IOD. Auto = pick by the median of the measurement plane (works for most blots).">
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      Image orientation
                    </Typography>
                  </Tooltip>
                  {([
                    { v: "auto",          label: "Auto",                hint: "Pick by median" },
                    { v: "light_on_dark", label: "Bright bands",        hint: "Fluorescent / LI-COR — sum raw" },
                    { v: "dark_on_light", label: "Dark bands",          hint: "Chemilum / film — invert" },
                  ] as const).map(({ v, label, hint }) => {
                    const active = (cfg.bandOrientation ?? "auto") === v;
                    return (
                      <Tooltip key={v} title={hint}>
                        <Box
                          onClick={() => setCfg((c) => ({ ...c, bandOrientation: v }))}
                          sx={{
                            cursor: "pointer", userSelect: "none",
                            fontSize: "0.65rem", px: 0.7, py: 0.2, borderRadius: 0.5,
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
                  <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto", fontSize: "0.62rem" }}>
                    {(cfg.bandOrientation ?? "auto") === "auto"
                      ? "auto-detect from image"
                      : ((cfg.bandOrientation === "light_on_dark")
                          ? "sum raw pixels = density"
                          : "invert plane (vmax − pixel)")}
                  </Typography>
                </Box>
                {lcEnabled && lanesPresent.length > 0 && (
                  <Box sx={{ mt: 0.75, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 0.75 }}>
                    {lanesPresent.map((ln) => {
                      const chosen = lcMap[ln] || "";
                      // Show ALL bands as candidates (not just same-lane) so the
                      // user can pick a custom-drawn box even if its label
                      // doesn't match a detected lane. Bands in the SAME lane
                      // sort first; everything else falls below in a secondary
                      // group so the natural pick is at the top.
                      const sameLane = cfg.lanes.filter((l) => (l.label || "Lane") === ln);
                      const otherBands = cfg.lanes.filter((l) => (l.label || "Lane") !== ln);
                      return (
                        <Box key={ln} sx={{ display: "flex", flexDirection: "column", gap: 0.2 }}>
                          <Typography variant="caption" sx={{ fontSize: "0.62rem", fontWeight: 700, color: "text.secondary" }}>
                            Lane {ln}
                          </Typography>
                          <TextField select size="small" value={chosen}
                            onChange={(e) => setCfg((c) => ({
                              ...c, lcBandByLane: { ...(c.lcBandByLane || {}), [ln]: e.target.value },
                            }))}
                            inputProps={{ style: { fontSize: "0.74rem", padding: "4px 8px" } }}
                            SelectProps={{ displayEmpty: true }}>
                            <MenuItem value=""><em>—</em></MenuItem>
                            {sameLane.map((b) => (
                              <MenuItem key={b.id} value={roiName(b)} sx={{ fontSize: "0.78rem" }}>
                                {roiName(b)} ({roiLevel(b)})
                              </MenuItem>
                            ))}
                            {otherBands.length > 0 && (
                              <MenuItem disabled value="__sep__" sx={{ fontSize: "0.62rem", opacity: 0.6, py: 0.2 }}>
                                — other lanes —
                              </MenuItem>
                            )}
                            {otherBands.map((b) => (
                              <MenuItem key={b.id} value={roiName(b)} sx={{ fontSize: "0.78rem", color: "text.secondary" }}>
                                {roiName(b)} · lane {b.label || "Lane"} ({roiLevel(b)})
                              </MenuItem>
                            ))}
                          </TextField>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            );
          })()}

          {/* Groups (experimental conditions) — FREE assignment of any band
              (by its unique Name) to any group, across levels and lanes.
              The downstream R plot makes one bar per group from the bands
              in it, with mean ± SD + pairwise significance — no R editing. */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
              <Tooltip title="Click bands by NAME (B1, B2, …) into groups — any subset across any lane / level. The downstream R plot draws one bar per group with mean ± SD and pairwise significance.">
                <Typography variant="caption" sx={{ fontWeight: 700 }}>Groups (free band selection)</Typography>
              </Tooltip>
              <Button size="small" variant="outlined" onClick={addGroup} disabled={cfg.lanes.length === 0}
                sx={{ textTransform: "none", fontSize: "0.65rem", py: 0.1, px: 0.75, ml: "auto" }}>
                + Group
              </Button>
            </Box>
            {groups.length === 0 ? (
              <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
                {cfg.lanes.length === 0
                  ? "Add bands first, then group them into conditions."
                  : "No groups yet — click + Group, then click each band (B1, B2, …) you want in that group."}
              </Typography>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.6 }}>
                {groups.map((g) => {
                  const groupFilter = groupLevelFilter[g.id] || "";   // "" = show all
                  const filteredBands = cfg.lanes.filter((l) => !groupFilter || roiLevel(l) === groupFilter);
                  return (
                    <Box key={g.id} sx={{ display: "flex", flexDirection: "column", gap: 0.3, py: 0.5, borderTop: "1px dashed", borderColor: "divider" }}>
                      <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.75 }}>
                        <TextField variant="standard" value={g.name}
                          onChange={(e) => renameGroup(g.id, e.target.value)}
                          inputProps={{ style: { fontSize: "0.78rem", fontWeight: 700, width: 80 } }} />
                        {/* Per-GROUP "Show: <level>" pill row — independent
                            per group so narrowing one doesn't affect siblings. */}
                        {levelOrder.length > 1 && (
                          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.3, alignItems: "center", flex: 1 }}>
                            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.disabled", mr: 0.3 }}>
                              Show:
                            </Typography>
                            {(["__all__", ...levelOrder] as const).map((lv) => {
                              const on = (groupFilter || "__all__") === lv;
                              return (
                                <Box key={lv}
                                  onClick={() => setGroupLevelFilter((prev) => {
                                    const next = { ...prev };
                                    if (lv === "__all__") delete next[g.id];
                                    else next[g.id] = lv;
                                    return next;
                                  })}
                                  sx={{
                                    fontSize: "0.55rem", px: 0.5, py: 0.05, borderRadius: 0.75,
                                    cursor: "pointer", userSelect: "none",
                                    bgcolor: on ? "action.selected" : "transparent",
                                    border: "1px solid", borderColor: on ? "primary.main" : "divider",
                                    color: on ? "primary.main" : "text.secondary",
                                    fontWeight: on ? 700 : 500,
                                  }}>
                                  {lv === "__all__" ? "all" : lv}
                                </Box>
                              );
                            })}
                          </Box>
                        )}
                        <IconButton size="small" onClick={() => deleteGroup(g.id)}>
                          <DeleteOutlineIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Box>
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.4, pl: 0.5 }}>
                        {filteredBands.length === 0 ? (
                          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled", fontStyle: "italic" }}>
                            No bands at this level.
                          </Typography>
                        ) : filteredBands.map((b) => {
                          const bn = roiName(b);
                          const on = isBandInGroup(g, bn, b.label);
                          const lv = roiLevel(b);
                          const col = levelColor(lv, levelOrder);
                          return (
                            <Tooltip key={b.id} title={`${bn}  ·  Lane ${b.label}  ·  Level ${lv}`}>
                              <Box onClick={() => toggleBandInGroup(g.id, bn)}
                                sx={{
                                  fontSize: "0.66rem", px: 0.55, py: 0.1, borderRadius: 0.75,
                                  cursor: "pointer", userSelect: "none",
                                  bgcolor: on ? col : "transparent",
                                  color: on ? "primary.contrastText" : "text.secondary",
                                  border: "1px solid", borderColor: on ? col : "divider",
                                  fontWeight: on ? 700 : 500,
                                  display: "inline-flex", alignItems: "center", gap: 0.3,
                                }}>
                                <Box component="span" sx={{ width: 6, height: 6, borderRadius: 0.3, bgcolor: col, opacity: on ? 0 : 1 }} />
                                {bn}
                              </Box>
                            </Tooltip>
                          );
                        })}
                      </Box>
                    </Box>
                  );
                })}
                {/* "Bands in this group" summary count */}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.3 }}>
                  {groups.map((g) => (
                    <Typography key={g.id} variant="caption" sx={{ fontSize: "0.58rem", color: "text.disabled" }}>
                      {g.name}: {groupBands(g).length} band(s)
                    </Typography>
                  ))}
                </Box>
              </Box>
            )}
          </Box>

          {/* Band list — fixed scrolling viewport so it doesn't get squashed
              when the Groups panel above grows tall. The whole controls
              column is already overflow-auto, so anything beyond fits via
              the column's own scrollbar. */}
          <Box sx={{ minHeight: 280, maxHeight: 520, overflow: "auto", border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
            <Box sx={{ display: "grid", gridTemplateColumns: "55px 70px 1fr 60px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.5, position: "sticky", top: 0, bgcolor: "background.paper", borderBottom: "1px solid", borderColor: "divider" }}>
              <Tooltip title="Unique band name — each row is distinct, even when multiple bands share a lane"><Typography variant="caption" sx={{ fontWeight: 700 }}>Name</Typography></Tooltip>
              <Tooltip title="Lane / sample (the column) — shared across bands in the same column for loading-control normalisation"><Typography variant="caption" sx={{ fontWeight: 700 }}>Lane</Typography></Tooltip>
              <Tooltip title="Target / protein row — 'lanes of the same level' are compared to each other"><Typography variant="caption" sx={{ fontWeight: 700 }}>Level</Typography></Tooltip>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>IOD</Typography>
              <span />
            </Box>
            {cfg.lanes.length === 0 ? (
              <Typography variant="caption" sx={{ display: "block", p: 1.5, color: "text.secondary", fontStyle: "italic" }}>
                No bands yet. Click <b>Auto-detect</b>, drag on the blot, or <b>+ Band</b> for a same-size box.
              </Typography>
            ) : cfg.lanes.map((lane, i) => {
              const lv = roiLevel(lane);
              const col = levelColor(lv, levelOrder);
              const isLC = cfg.loadingControlEnabled
                ? ((cfg.lcBandByLane || {})[lane.label || ""] === roiName(lane))
                : (cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel);
              return (
                <Box key={lane.id}
                  onMouseEnter={() => setSelId(lane.id)}
                  sx={{ display: "grid", gridTemplateColumns: "55px 70px 1fr 60px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.4,
                    bgcolor: lane.id === selId ? "action.hover" : undefined, borderBottom: "1px solid", borderColor: "divider" }}>
                  <TextField variant="standard" value={roiName(lane)} placeholder={`B${i + 1}`}
                    onChange={(e) => setName(lane.id, e.target.value)}
                    inputProps={{ style: { fontSize: "0.74rem", fontWeight: 700, color: col } }} />
                  <TextField variant="standard" value={lane.label} placeholder={`S${i + 1}`}
                    onChange={(e) => setLabel(lane.id, e.target.value)}
                    inputProps={{ style: { fontSize: "0.74rem" } }} />
                  <TextField variant="standard" value={lv} placeholder="Target"
                    onChange={(e) => setLevel(lane.id, e.target.value)}
                    inputProps={{ style: { fontSize: "0.74rem", color: col, fontWeight: 600 } }} />
                  <Typography variant="caption" sx={{ fontVariantNumeric: "tabular-nums", color: "text.secondary" }}>
                    {Number.isFinite(iodById[lane.id])
                      ? Math.round(iodById[lane.id]).toLocaleString()
                      : (lane.w < 0.001 || lane.h < 0.001 ? "0" : "—")}
                    {isLC && <span title="loading control" style={{ marginLeft: 4 }}>★</span>}
                  </Typography>
                  <IconButton size="small" onClick={() => deleteLane(lane.id)}><DeleteOutlineIcon sx={{ fontSize: 16 }} /></IconButton>
                </Box>
              );
            })}
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.4 }}>
            <b>Name</b> uniquely identifies a band · <b>Lane</b> is the column (shared with same-column bands; used for loading-control normalisation) · <b>Level</b> is the protein row (compared across lanes). IOD shown here is a display-image preview; the run recomputes on the full-resolution membrane.
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions sx={{ flexDirection: "column", alignItems: "stretch", gap: 0.75, px: 2, py: 1.25 }}>
        {/* Save-blocker banner. The analysis runner aborts the workflow
            with "no groups defined" when the saved config has bands but
            zero groups — explain that here and disable Save until at
            least one group has at least one band assigned, so the user
            doesn't keep cycling Run graph → picker → save → halt. */}
        {(() => {
          const hasBands = cfg.lanes.length > 0;
          const groupCount = (cfg.groups ?? []).length;
          const groupsWithBands = (cfg.groups ?? []).filter((g) => (g.bands?.length ?? 0) > 0).length;
          const canSave = !hasBands || (groupCount > 0 && groupsWithBands > 0);
          let reason = "";
          if (hasBands && groupCount === 0) {
            reason = "⚠ No groups defined yet — the analysis runner halts on save. "
              + "Click \"+ Group\" above and assign at least one band before saving.";
          } else if (hasBands && groupsWithBands === 0) {
            reason = "⚠ You have groups but no bands assigned to any of them — "
              + "click the band-name chips (B1, B2, …) inside a group row to assign them, then save.";
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
                      Save bands
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
