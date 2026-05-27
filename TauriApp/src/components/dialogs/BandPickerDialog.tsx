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

/** A user-defined experimental group — maps a condition NAME to the set of
 *  lane labels that are its biological replicates. Used by the R plot node
 *  to build the `conditions <- list(...)` mapping automatically. */
export interface BandGroup {
  id: string;
  name: string;          // condition name shown on the x-axis
  lanes: string[];       // lane LABELs (the "column" identity) in this group
}

export interface BandPickerConfig {
  version: 1;
  lanes: BandRoi[];
  /** Which `level` (target row) is the loading control. Per-lane normalisation
   *  divides every other level's IOD by this level's IOD in the same lane.
   *  null → no loading-control normalisation (raw IOD passes through). */
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
}

export function emptyBandConfig(): BandPickerConfig {
  return {
    version: 1, lanes: [], loadingControlLevel: null,
    bgMode: "percentile", bgRoi: null, bgPercentile: 5,
    groups: [],
  };
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

lc_level = CFG.get("loadingControlLevel")          # which level is the loading control

# Lane → experimental group (condition name) from the picker's Groups UI.
lane2group = {}
for g in CFG.get("groups", []) or []:
    name = (g.get("name") or "").strip()
    if not name: continue
    for ln in g.get("lanes", []) or []:
        lane2group[str(ln)] = name

rows = []
for i, band in enumerate(CFG.get("lanes", [])):
    x0, y0, x1, y1 = _px(band)
    if x1 <= x0 or y1 <= y0:
        continue
    roi = plane[y0:y1, x0:x1].astype(np.float64)
    area = int(roi.size)
    # Local background: slabs above and below the ROI, same width, gap=6
    # — matches detect_bands_equal_boxes.py quantify().
    gap, hh = 6, (y1 - y0)
    above_y2 = max(0, y0 - gap); above_y1 = max(0, above_y2 - hh)
    below_y1 = min(H, y1 + gap); below_y2 = min(H, below_y1 + hh)
    parts = []
    if above_y2 > above_y1: parts.append(plane[above_y1:above_y2, x0:x1].reshape(-1))
    if below_y2 > below_y1: parts.append(plane[below_y1:below_y2, x0:x1].reshape(-1))
    bg_pixels = np.concatenate(parts) if parts else np.array([])
    bg_mean = _robust_background(bg_pixels)
    raw_sum = float(np.sum(roi))
    corrected = raw_sum - bg_mean * area
    lane = (band.get("label") or "").strip() or f"Lane {i + 1}"
    level = (band.get("level") or "").strip() or "Target"
    band_name = (band.get("name") or "").strip() or f"B{i + 1}"
    rows.append({
        "band": band_name,                         # UNIQUE per-row identifier
        "lane": lane,                              # column / sample (group key)
        "level": level,                            # target / MW row
        "group": lane2group.get(lane, ""),         # condition / experimental group
        "is_loading_control": bool(lc_level is not None and level == lc_level),
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
// mean of the slabs immediately above + below the ROI. The display-image
// readout is APPROXIMATE — the run-time recomputes at full bit depth.
function approxIod(
  gray: Float32Array, w: number, h: number,
  lane: { x: number; y: number; w: number; h: number },
): number {
  const x0 = Math.max(0, Math.round(lane.x * w));
  const x1 = Math.min(w, Math.round((lane.x + lane.w) * w));
  const y0 = Math.max(0, Math.round(lane.y * h));
  const y1 = Math.min(h, Math.round((lane.y + lane.h) * h));
  if (x1 <= x0 || y1 <= y0) return 0;
  const hh = y1 - y0;
  // ROI raw sum on the inverted (dark→bright) display plane.
  let roiSum = 0; let n = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { roiSum += 255 - gray[y * w + x]; n++; }
  // Local background: slabs above + below, same width, gap=6 px.
  const gap = 6;
  const ay2 = Math.max(0, y0 - gap), ay1 = Math.max(0, ay2 - hh);
  const by1Lo = Math.min(h, y1 + gap), by2 = Math.min(h, by1Lo + hh);
  let bgSum = 0, bgN = 0;
  for (let y = ay1; y < ay2; y++) for (let x = x0; x < x1; x++) { bgSum += 255 - gray[y * w + x]; bgN++; }
  for (let y = by1Lo; y < by2; y++) for (let x = x0; x < x1; x++) { bgSum += 255 - gray[y * w + x]; bgN++; }
  const bgMean = bgN > 0 ? bgSum / bgN : 0;
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
      for (const lane of cfg.lanes) out[lane.id] = approxIod(gray, natW, natH, lane);
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
    // Inherit the LANE label from the band directly above (same column)
    // when there's an aligned candidate, otherwise pick "S{n}".
    let lane = `S${cfg.lanes.filter((l) => !/^L\d*$|^Ladder|^Marker/i.test(l.label)).length + 1}`;
    const aligned = cfg.lanes.find((l) => Math.abs((l.x + l.w / 2) - (x + bw / 2)) < bw * 0.5);
    if (aligned) lane = aligned.label;
    const newBand: BandRoi = {
      id: newId, x, y, w: bw, h: bh,
      name: `B${cfg.lanes.length + 1}`,
      label: lane, level: lastLevel,
    };
    setCfg((c) => ({ ...c, lanes: [...c.lanes, newBand] }));
    setSelId(newId);
  }, [cfg.lanes, cfg.defaultBoxW, cfg.defaultBoxH, selId]);

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
      return {
        ...c,
        lanes: [...c.lanes, { id: newId, x: nx, y: ny, w: 0, h: 0, label: `S${c.lanes.length + 1}`, level: defLevel }],
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
  // `conditions <- list(...)` automatically).
  const groups = cfg.groups ?? [];
  const addGroup = () => {
    const used = new Set(groups.map((g) => g.name));
    let n = 1; while (used.has(`G${n}`)) n++;
    const g: BandGroup = { id: uid("group"), name: `G${n}`, lanes: [] };
    setCfg((c) => ({ ...c, groups: [...(c.groups ?? []), g] }));
  };
  const renameGroup = (id: string, name: string) =>
    setCfg((c) => ({ ...c, groups: (c.groups ?? []).map((g) => g.id === id ? { ...g, name } : g) }));
  const deleteGroup = (id: string) =>
    setCfg((c) => ({ ...c, groups: (c.groups ?? []).filter((g) => g.id !== id) }));
  const toggleLaneInGroup = (gid: string, laneLabel: string) =>
    setCfg((c) => ({
      ...c,
      groups: (c.groups ?? []).map((g) => {
        if (g.id !== gid) return g;
        const has = g.lanes.includes(laneLabel);
        return { ...g, lanes: has ? g.lanes.filter((l) => l !== laneLabel) : [...g.lanes, laneLabel] };
      }),
    }));
  // Distinct lane labels available to assign — built from the bands.
  const laneLabels = useMemo(() => {
    const seen: string[] = [];
    for (const l of cfg.lanes) if (l.label && !seen.includes(l.label)) seen.push(l.label);
    return seen;
  }, [cfg.lanes]);

  // Distinct levels (rows) present, in first-seen order, for coloring + the
  // loading-control selector.
  const levelOrder = useMemo(() => {
    const seen: string[] = [];
    for (const l of cfg.lanes) { const lv = roiLevel(l); if (!seen.includes(lv)) seen.push(lv); }
    return seen;
  }, [cfg.lanes]);

  const applyDetectedLanes = (
    rects: Array<{ x: number; y: number; w: number; h: number; lane?: string; level?: string }>,
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
    setCfg((c) => ({ ...c, lanes, defaultBoxW, defaultBoxH }));
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
            if (Array.isArray(data.lanes) && applyDetectedLanes(data.lanes)) {
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

  // Render an 8-handle set for a normalised rect. When a band is selected
  // (`selected`) the handles are notably larger (14×14 vs 8×8) and use a
  // wider hit area so they're easy to grab — small handles on a zoomable
  // canvas were unforgiving (the previous 8 px square at 1× already fell
  // below the macOS pointer's effective grab radius).
  const handlesFor = (
    rect: { x: number; y: number; w: number; h: number },
    id: string | null,
    roleHandle: string,
    selected: boolean = false,
  ) => {
    const hs = [
      { h: "nw", cx: rect.x, cy: rect.y }, { h: "n", cx: rect.x + rect.w / 2, cy: rect.y }, { h: "ne", cx: rect.x + rect.w, cy: rect.y },
      { h: "w", cx: rect.x, cy: rect.y + rect.h / 2 }, { h: "e", cx: rect.x + rect.w, cy: rect.y + rect.h / 2 },
      { h: "sw", cx: rect.x, cy: rect.y + rect.h }, { h: "s", cx: rect.x + rect.w / 2, cy: rect.y + rect.h }, { h: "se", cx: rect.x + rect.w, cy: rect.y + rect.h },
    ];
    const cur: Record<string, string> = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };
    // Zoom-aware handle size — keep the on-screen size constant so handles
    // don't visually shrink/disappear when the user zooms in/out.
    const sz = (selected ? 14 : 8) / Math.max(1, view.zoom);
    const half = sz / 2;
    return hs.map((hh) => (
      <rect key={hh.h} data-role={roleHandle} data-id={id ?? undefined} data-handle={hh.h}
        x={hh.cx * disp.w - half} y={hh.cy * disp.h - half} width={sz} height={sz}
        fill="#fff" stroke="#1976d2" strokeWidth={selected ? 2 : 1.5} rx={2}
        style={{ cursor: cur[hh.h] }} />
    ));
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
                {/* Bands — colored by level (row); same level = same color.
                    Stroke widths are scaled by 1/zoom so the visible thickness
                    stays constant when the user zooms in. */}
                {cfg.lanes.map((lane) => {
                  const sel = lane.id === selId;
                  const lv = roiLevel(lane);
                  const col = levelColor(lv, levelOrder);
                  const isLC = cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel;
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

          {/* Levels & loading control */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1, display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
            <Tooltip title="The target row whose band each lane is divided by, for loading-control normalisation. Bands on the same row share one level.">
              <Typography variant="caption" sx={{ fontWeight: 700 }}>Loading control level</Typography>
            </Tooltip>
            <TextField select size="small" value={cfg.loadingControlLevel ?? ""}
              onChange={(e) => setLoadingControlLevel(e.target.value === "" ? null : e.target.value)}
              disabled={levelOrder.length === 0}
              sx={{ minWidth: 130 }}
              SelectProps={{ displayEmpty: true }}
              inputProps={{ style: { fontSize: "0.78rem", padding: "4px 8px" } }}>
              <MenuItem value=""><em>None (raw IOD)</em></MenuItem>
              {levelOrder.map((lv) => (
                <MenuItem key={lv} value={lv} sx={{ fontSize: "0.8rem" }}>
                  <Box component="span" sx={{ display: "inline-block", width: 10, height: 10, mr: 0.75, borderRadius: 0.3, bgcolor: levelColor(lv, levelOrder) }} />
                  {lv}
                </MenuItem>
              ))}
            </TextField>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto" }}>
              {levelOrder.length} level(s)
            </Typography>
          </Box>

          {/* Groups (experimental conditions) — defines which LANES are
              replicates of which condition. The downstream R plot turns
              this into grouped bars (mean ± SD + significance) with no
              R-code editing required. */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
              <Tooltip title="Group replicate lanes into experimental conditions (Control, Treatment, …). The downstream R plot draws one bar per group per level with mean ± SD and pairwise significance. Lanes left out of every group plot individually.">
                <Typography variant="caption" sx={{ fontWeight: 700 }}>Groups (conditions)</Typography>
              </Tooltip>
              <Button size="small" variant="outlined" onClick={addGroup} disabled={laneLabels.length === 0}
                sx={{ textTransform: "none", fontSize: "0.65rem", py: 0.1, px: 0.75, ml: "auto" }}>
                + Group
              </Button>
            </Box>
            {groups.length === 0 ? (
              <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
                {laneLabels.length === 0
                  ? "Add bands first, then group lanes into conditions."
                  : "No groups yet — click + Group, then tick the lanes (e.g. S1 S2 S3) that are replicates of one condition."}
              </Typography>
            ) : groups.map((g) => (
              <Box key={g.id} sx={{ display: "flex", alignItems: "flex-start", gap: 0.75, py: 0.4, borderTop: "1px dashed", borderColor: "divider" }}>
                <TextField variant="standard" value={g.name}
                  onChange={(e) => renameGroup(g.id, e.target.value)}
                  inputProps={{ style: { fontSize: "0.78rem", fontWeight: 700, width: 80 } }} />
                <Box sx={{ flex: 1, display: "flex", flexWrap: "wrap", gap: 0.4 }}>
                  {laneLabels.map((ln) => {
                    const on = g.lanes.includes(ln);
                    return (
                      <Box key={ln} onClick={() => toggleLaneInGroup(g.id, ln)}
                        sx={{
                          fontSize: "0.66rem", px: 0.6, py: 0.1, borderRadius: 0.75,
                          cursor: "pointer", userSelect: "none",
                          bgcolor: on ? "primary.main" : "transparent",
                          color: on ? "primary.contrastText" : "text.secondary",
                          border: "1px solid", borderColor: on ? "primary.main" : "divider",
                          fontWeight: on ? 700 : 500,
                        }}>
                        {ln}
                      </Box>
                    );
                  })}
                </Box>
                <IconButton size="small" onClick={() => deleteGroup(g.id)}><DeleteOutlineIcon sx={{ fontSize: 16 }} /></IconButton>
              </Box>
            ))}
          </Box>

          {/* Band list */}
          <Box sx={{ flex: 1, overflow: "auto", maxHeight: 520, border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
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
              const isLC = cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel;
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
      <DialogActions>
        <Button onClick={onClose} sx={{ textTransform: "none" }}>Cancel</Button>
        <Button variant="contained" onClick={() => onSave(cfg)} sx={{ textTransform: "none" }}>
          Save bands
        </Button>
      </DialogActions>
    </Dialog>
  );
}
