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
  DialogActions, TextField, Tooltip, ToggleButton, ToggleButtonGroup, MenuItem,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

// ── Types (shared with AnalysisNodeGraph via import) ──────────
export interface BandRoi {
  id: string;
  /** Normalised 0..1 rect over the membrane image (top-left origin). */
  x: number; y: number; w: number; h: number;
  /** Lane / sample name (the COLUMN), e.g. "S1". */
  label: string;
  /** Target / protein row (the LEVEL): "lanes of the same level" are bands
   *  at the same apparent molecular weight across columns. e.g. "G1" / "GAPDH". */
  level: string;
  /** @deprecated — loading control is now chosen per LEVEL
   *  (BandPickerConfig.loadingControlLevel). Kept so older saved nodes load. */
  isReference?: boolean;
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
}

export function emptyBandConfig(): BandPickerConfig {
  return { version: 1, lanes: [], loadingControlLevel: null, bgMode: "percentile", bgRoi: null, bgPercentile: 5 };
}

const uid = () => `lane_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

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
import numpy as np, json

CFG = json.loads(r'''${json}''')

imgs = [v for v in inputs.values() if isinstance(v, dict) and ("image_raw" in v or "image" in v)]
if not imgs:
    raise SystemExit("No membrane image — wire the blot (Source) into this node.")
src = imgs[0]
# Prefer FULL-BIT-DEPTH pixels (e.g. 16-bit) for quantitative IOD; fall back to
# the 8-bit display image when raw isn't available.
use_raw = "image_raw" in src
arr = np.asarray(src["image_raw"] if use_raw else src["image"]).astype(np.float32)
if arr.ndim == 3:
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2] if arr.shape[2] >= 3 else arr[..., 0]
else:
    gray = arr
H, W = gray.shape[:2]
# Invert relative to the white level so dark protein bands -> bright signal,
# preserving the native dynamic range (16-bit) when available.
white = float(src.get("raw_max") or 0) if use_raw else 255.0
if white <= 0:
    white = float(gray.max()) or 1.0
inv_full = white - gray

def _px(rect):
    x0 = int(round(rect["x"] * W)); x1 = int(round((rect["x"] + rect["w"]) * W))
    y0 = int(round(rect["y"] * H)); y1 = int(round((rect["y"] + rect["h"]) * H))
    x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
    y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
    return x0, y0, x1, y1

# Optional global background from a user-drawn ROI.
bg_global = None
if CFG.get("bgMode") == "roi" and CFG.get("bgRoi"):
    bx0, by0, bx1, by1 = _px(CFG["bgRoi"])
    if bx1 > bx0 and by1 > by0:
        bg_global = float(inv_full[by0:by1, bx0:bx1].mean())

pct = float(CFG.get("bgPercentile", 5))
lc_level = CFG.get("loadingControlLevel")          # which level is the loading control
rows = []
for i, band in enumerate(CFG.get("lanes", [])):
    x0, y0, x1, y1 = _px(band)
    if x1 <= x0 or y1 <= y0:
        continue
    inv = inv_full[y0:y1, x0:x1]
    bg = bg_global if bg_global is not None else float(np.percentile(inv, pct))
    signal = np.clip(inv - bg, 0, None)
    lane = (band.get("label") or "").strip() or f"Lane {i + 1}"
    level = (band.get("level") or "").strip() or "Target"
    rows.append({
        "lane": lane,                              # column / sample
        "level": level,                            # target / MW row
        "is_loading_control": bool(lc_level is not None and level == lc_level),
        "iod": float(signal.sum()),
        "mean_signal": float(signal.mean()),
        "area_px": int(inv.size),
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
function approxIod(
  gray: Float32Array, w: number, h: number,
  lane: { x: number; y: number; w: number; h: number },
  cfg: BandPickerConfig,
): number {
  const x0 = Math.max(0, Math.round(lane.x * w));
  const x1 = Math.min(w, Math.round((lane.x + lane.w) * w));
  const y0 = Math.max(0, Math.round(lane.y * h));
  const y1 = Math.min(h, Math.round((lane.y + lane.h) * h));
  if (x1 <= x0 || y1 <= y0) return 0;
  const vals: number[] = [];
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) vals.push(255 - gray[y * w + x]);
  let bg: number;
  if (cfg.bgMode === "roi" && cfg.bgRoi) {
    const bx0 = Math.max(0, Math.round(cfg.bgRoi.x * w));
    const bx1 = Math.min(w, Math.round((cfg.bgRoi.x + cfg.bgRoi.w) * w));
    const by0 = Math.max(0, Math.round(cfg.bgRoi.y * h));
    const by1 = Math.min(h, Math.round((cfg.bgRoi.y + cfg.bgRoi.h) * h));
    let s = 0, n = 0;
    for (let y = by0; y < by1; y++) for (let x = bx0; x < bx1; x++) { s += 255 - gray[y * w + x]; n++; }
    bg = n > 0 ? s / n : 0;
  } else {
    const sorted = [...vals].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.floor((cfg.bgPercentile / 100) * sorted.length));
    bg = sorted[idx] || 0;
  }
  let sum = 0;
  for (const v of vals) { const s = v - bg; if (s > 0) sum += s; }
  return sum;
}

interface DragState {
  mode: "move" | "resize" | "draw" | "drawbg" | "movebg" | "resizebg";
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

const MAX_DISP_W = 620;
const MAX_DISP_H = 520;

export default function BandPickerDialog(props: BandPickerDialogProps) {
  const { open, imageSrc, source, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<BandPickerConfig>(initial ?? emptyBandConfig());
  const [selId, setSelId] = useState<string | null>(null);
  const [drawingBg, setDrawingBg] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [detectInfo, setDetectInfo] = useState<string | null>(null);
  const [natW, setNatW] = useState(0);
  const [natH, setNatH] = useState(0);
  const [gray, setGray] = useState<Float32Array | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<DragState | null>(null);

  // Reset state whenever the dialog (re)opens with fresh inputs.
  useEffect(() => {
    if (open) {
      setCfg(initial ? structuredClone(initial) : emptyBandConfig());
      setSelId(null);
      setDrawingBg(false);
      setDetectInfo(null);
    }
  }, [open, initial]);

  const srcUrl = useMemo(() => {
    if (!imageSrc) return null;
    return imageSrc.startsWith("data:") ? imageSrc : `data:image/png;base64,${imageSrc}`;
  }, [imageSrc]);

  // Decode the image once to a grayscale buffer for auto-detect + IOD.
  useEffect(() => {
    if (!srcUrl) { setGray(null); setNatW(0); setNatH(0); return; }
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
    img.src = srcUrl;
    return () => { cancelled = true; };
  }, [srcUrl]);

  // Display geometry — fit within the max box, preserve aspect.
  const disp = useMemo(() => {
    if (!natW || !natH) return { w: MAX_DISP_W, h: 360 };
    const scale = Math.min(MAX_DISP_W / natW, MAX_DISP_H / natH, 1.5);
    return { w: Math.round(natW * scale), h: Math.round(natH * scale) };
  }, [natW, natH]);

  // Live IOD per lane (approximate, from display-res grayscale).
  const iodById = useMemo(() => {
    const out: Record<string, number> = {};
    if (gray && natW && natH) {
      for (const lane of cfg.lanes) out[lane.id] = approxIod(gray, natW, natH, lane, cfg);
    }
    return out;
  }, [gray, natW, natH, cfg]);

  const clientToNorm = useCallback((clientX: number, clientY: number) => {
    const el = svgRef.current;
    if (!el) return { nx: 0, ny: 0 };
    const r = el.getBoundingClientRect();
    return { nx: clamp01((clientX - r.left) / r.width), ny: clamp01((clientY - r.top) / r.height) };
  }, []);

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
    if (role === "bg-handle" && cfg.bgRoi) {
      dragRef.current = { mode: "resizebg", id: null, handle, startNx: nx, startNy: ny, orig: { ...cfg.bgRoi } };
      return;
    }
    if (role === "lane" && id) {
      const lane = cfg.lanes.find((l) => l.id === id);
      if (lane) { setSelId(id); dragRef.current = { mode: "move", id, startNx: nx, startNy: ny, orig: { ...lane } }; }
      return;
    }
    if (role === "bg" && cfg.bgRoi) {
      dragRef.current = { mode: "movebg", id: null, startNx: nx, startNy: ny, orig: { ...cfg.bgRoi } };
      return;
    }
    // Empty area → draw a new lane (or background rect if in bg-draw mode).
    if (drawingBg) {
      dragRef.current = { mode: "drawbg", id: null, startNx: nx, startNy: ny, orig: { x: nx, y: ny, w: 0, h: 0 } };
    } else {
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
    }
  }, [cfg.lanes, cfg.bgRoi, clientToNorm, drawingBg, selId]);

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

    if (d.mode === "draw" || d.mode === "drawbg") {
      const x = Math.min(d.startNx, nx), y = Math.min(d.startNy, ny);
      const w = Math.abs(nx - d.startNx), h = Math.abs(ny - d.startNy);
      if (d.mode === "draw" && d.id) {
        setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, x, y, w, h } : l) }));
      } else {
        setCfg((c) => ({ ...c, bgRoi: { x, y, w, h } }));
      }
    } else if (d.mode === "move" && d.id) {
      const x = clamp01(d.orig.x + dx), y = clamp01(d.orig.y + dy);
      setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, x: Math.min(x, 1 - l.w), y: Math.min(y, 1 - l.h) } : l) }));
    } else if (d.mode === "resize" && d.id) {
      const r = resizeRect(d.orig, d.handle);
      setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === d.id ? { ...l, ...r } : l) }));
    } else if (d.mode === "movebg") {
      const x = clamp01(d.orig.x + dx), y = clamp01(d.orig.y + dy);
      setCfg((c) => c.bgRoi ? ({ ...c, bgRoi: { ...c.bgRoi, x: Math.min(x, 1 - c.bgRoi.w), y: Math.min(y, 1 - c.bgRoi.h) } }) : c);
    } else if (d.mode === "resizebg") {
      const r = resizeRect(d.orig, d.handle);
      setCfg((c) => ({ ...c, bgRoi: r }));
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
    if (d.mode === "drawbg") {
      setCfg((c) => (c.bgRoi && (c.bgRoi.w < 0.01 || c.bgRoi.h < 0.01)) ? { ...c, bgRoi: null } : c);
      setDrawingBg(false);
    }
  }, []);

  const setLabel = (id: string, label: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === id ? { ...l, label } : l) }));
  const setLevel = (id: string, level: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => l.id === id ? { ...l, level } : l) }));
  const setLoadingControlLevel = (level: string | null) =>
    setCfg((c) => ({ ...c, loadingControlLevel: level }));
  const deleteLane = (id: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.filter((l) => l.id !== id) }));

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
    // Drop ladder/marker rows — they're a visual reference, not a quantified
    // target. The user can still add them by hand if needed.
    const usable = rects.filter((r) => (r.level || "") !== "Ladder" && (r.lane || "") !== "L");
    const src = usable.length ? usable : rects;
    const lanes: BandRoi[] = src.map((r, i) => ({
      id: uid(), x: clamp01(r.x), y: clamp01(r.y), w: clamp01(r.w), h: clamp01(r.h),
      label: (r.lane || `S${i + 1}`), level: (r.level || "Target"),
    }));
    setCfg((c) => ({ ...c, lanes }));
    setSelId(lanes[0]?.id ?? null);
    return true;
  };

  const runAutoDetect = async () => {
    if (detecting) return;
    setDetecting(true);
    setDetectInfo(null);
    try {
      // Prefer the backend detector (robust band-row-constrained algorithm,
      // pure-numpy so it ships in the frozen sidecar). Fall back to the local
      // heuristic if the endpoint isn't reachable (e.g. older sidecar / no backend).
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
              // Warn if detection ran on a low-res preview rather than the
              // full-res membrane — the single biggest cause of poor results.
              const sw = Number(data.src_w) || 0;
              if (!data.used_source || (sw > 0 && sw < 700)) {
                setDetectInfo(
                  `⚠ Detected on a ${sw || "low"}px preview — the full-resolution membrane wasn't available, so results are coarse. ` +
                  `Re-drop the original blot onto the Source node (and make sure the backend is connected), then auto-detect again.`,
                );
              } else {
                setDetectInfo(`Detected on the full-resolution membrane (${sw}px).`);
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

  // Snap each existing lane's vertical position to its actual band, keeping the
  // user's x-grid. Sends the current lanes to the backend's per-lane detector
  // Render an 8-handle set for a normalised rect.
  const handlesFor = (rect: { x: number; y: number; w: number; h: number }, id: string | null, roleHandle: string) => {
    const hs = [
      { h: "nw", cx: rect.x, cy: rect.y }, { h: "n", cx: rect.x + rect.w / 2, cy: rect.y }, { h: "ne", cx: rect.x + rect.w, cy: rect.y },
      { h: "w", cx: rect.x, cy: rect.y + rect.h / 2 }, { h: "e", cx: rect.x + rect.w, cy: rect.y + rect.h / 2 },
      { h: "sw", cx: rect.x, cy: rect.y + rect.h }, { h: "s", cx: rect.x + rect.w / 2, cy: rect.y + rect.h }, { h: "se", cx: rect.x + rect.w, cy: rect.y + rect.h },
    ];
    const cur: Record<string, string> = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };
    return hs.map((hh) => (
      <rect key={hh.h} data-role={roleHandle} data-id={id ?? undefined} data-handle={hh.h}
        x={hh.cx * disp.w - 4} y={hh.cy * disp.h - 4} width={8} height={8}
        fill="#fff" stroke="#1976d2" strokeWidth={1.5} style={{ cursor: cur[hh.h] }} />
    ));
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem", py: 1.25 }}>
        🩻 Pick bands
        <Typography component="span" variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
          Auto-detect bands, then clean up: drag to move · handles to resize · select + Delete to remove · drag on the blot to add
        </Typography>
      </DialogTitle>
      <DialogContent dividers sx={{ display: "flex", gap: 2, alignItems: "flex-start" }}>
        {/* Image + ROI overlay */}
        <Box sx={{ flexShrink: 0 }}>
          {srcUrl ? (
            <Box sx={{ position: "relative", width: disp.w, height: disp.h, userSelect: "none", border: "1px solid", borderColor: "divider" }}>
              <Box component="img" src={srcUrl} alt="blot" draggable={false}
                sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "fill", pointerEvents: "none" }} />
              <svg ref={svgRef} width={disp.w} height={disp.h}
                style={{ position: "absolute", inset: 0, cursor: drawingBg ? "crosshair" : "crosshair", touchAction: "none" }}
                onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}>
                {/* Background ROI */}
                {cfg.bgRoi && (
                  <>
                    <rect data-role="bg" x={cfg.bgRoi.x * disp.w} y={cfg.bgRoi.y * disp.h}
                      width={cfg.bgRoi.w * disp.w} height={cfg.bgRoi.h * disp.h}
                      fill="rgba(255,193,7,0.12)" stroke="#ffc107" strokeDasharray="4 3" strokeWidth={1.5} style={{ cursor: "move" }} />
                    {handlesFor(cfg.bgRoi, null, "bg-handle")}
                  </>
                )}
                {/* Bands — colored by level (row); same level = same color */}
                {cfg.lanes.map((lane) => {
                  const sel = lane.id === selId;
                  const lv = roiLevel(lane);
                  const col = levelColor(lv, levelOrder);
                  const isLC = cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel;
                  return (
                    <g key={lane.id}>
                      <rect data-role="lane" data-id={lane.id}
                        x={lane.x * disp.w} y={lane.y * disp.h} width={lane.w * disp.w} height={lane.h * disp.h}
                        fill={col} fillOpacity={sel ? 0.22 : 0.1}
                        stroke={col} strokeWidth={sel ? 2.5 : 1.5}
                        strokeDasharray={isLC ? "4 2" : undefined}
                        style={{ cursor: "move" }} />
                      <text x={lane.x * disp.w + 3} y={lane.y * disp.h + 12} fontSize={11}
                        fill={col} style={{ pointerEvents: "none", fontWeight: 700 }}>
                        {lane.label || "?"}{isLC ? " (LC)" : ""}
                      </text>
                      {sel && handlesFor(lane, lane.id, "handle")}
                    </g>
                  );
                })}
              </svg>
            </Box>
          ) : (
            <Box sx={{ width: MAX_DISP_W, height: 320, display: "flex", alignItems: "center", justifyContent: "center", border: "1px dashed", borderColor: "divider", color: "text.secondary", textAlign: "center", p: 3 }}>
              <Typography variant="body2">
                No blot image reached this node.<br />Wire a Source node (with the whole membrane dropped on it) into this Band picker, then reopen.
              </Typography>
            </Box>
          )}
        </Box>

        {/* Controls */}
        <Box sx={{ flex: 1, minWidth: 280, display: "flex", flexDirection: "column", gap: 1.25 }}>
          <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
            <Button size="small" variant="contained" startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
              onClick={runAutoDetect} disabled={!srcUrl || detecting}>
              {detecting ? "Detecting…" : "Auto-detect bands"}
            </Button>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: "auto" }}>
              {cfg.lanes.length} band(s) — delete / drag / resize to clean up
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

          {/* Background subtraction */}
          <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 700, display: "block", mb: 0.5 }}>Background subtraction</Typography>
            <ToggleButtonGroup size="small" exclusive value={cfg.bgMode}
              onChange={(_, v) => { if (v) setCfg((c) => ({ ...c, bgMode: v })); }}>
              <ToggleButton value="percentile" sx={{ textTransform: "none", fontSize: "0.7rem" }}>Per-lane percentile</ToggleButton>
              <ToggleButton value="roi" sx={{ textTransform: "none", fontSize: "0.7rem" }}>Background ROI</ToggleButton>
            </ToggleButtonGroup>
            {cfg.bgMode === "percentile" ? (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.75 }}>
                <Typography variant="caption">Floor percentile</Typography>
                <TextField type="number" size="small" value={cfg.bgPercentile}
                  onChange={(e) => setCfg((c) => ({ ...c, bgPercentile: Math.max(0, Math.min(100, Number(e.target.value) || 0)) }))}
                  inputProps={{ min: 0, max: 100, step: 1, style: { width: 56, padding: "4px 6px" } }} />
              </Box>
            ) : (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.75 }}>
                <Button size="small" variant={drawingBg ? "contained" : "outlined"} color="warning"
                  onClick={() => setDrawingBg((v) => !v)} sx={{ textTransform: "none", fontSize: "0.7rem" }}>
                  {drawingBg ? "Click-drag on blot…" : cfg.bgRoi ? "Redraw background" : "Draw background"}
                </Button>
                {cfg.bgRoi && (
                  <IconButton size="small" onClick={() => setCfg((c) => ({ ...c, bgRoi: null }))}><DeleteOutlineIcon sx={{ fontSize: 16 }} /></IconButton>
                )}
              </Box>
            )}
          </Box>

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

          {/* Band list */}
          <Box sx={{ flex: 1, overflow: "auto", maxHeight: 360, border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
            <Box sx={{ display: "grid", gridTemplateColumns: "20px 1fr 1fr 60px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.5, position: "sticky", top: 0, bgcolor: "background.paper", borderBottom: "1px solid", borderColor: "divider" }}>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>#</Typography>
              <Tooltip title="Lane / sample (the column)"><Typography variant="caption" sx={{ fontWeight: 700 }}>Lane</Typography></Tooltip>
              <Tooltip title="Target / protein row — 'lanes of the same level' are compared to each other"><Typography variant="caption" sx={{ fontWeight: 700 }}>Level</Typography></Tooltip>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>IOD</Typography>
              <span />
            </Box>
            {cfg.lanes.length === 0 ? (
              <Typography variant="caption" sx={{ display: "block", p: 1.5, color: "text.secondary", fontStyle: "italic" }}>
                No bands yet. Click “Auto-detect bands”, or drag rectangles on the blot.
              </Typography>
            ) : cfg.lanes.map((lane, i) => {
              const lv = roiLevel(lane);
              const col = levelColor(lv, levelOrder);
              const isLC = cfg.loadingControlLevel != null && lv === cfg.loadingControlLevel;
              return (
                <Box key={lane.id}
                  onMouseEnter={() => setSelId(lane.id)}
                  sx={{ display: "grid", gridTemplateColumns: "20px 1fr 1fr 60px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.4,
                    bgcolor: lane.id === selId ? "action.hover" : undefined, borderBottom: "1px solid", borderColor: "divider" }}>
                  <Box sx={{ width: 9, height: 9, borderRadius: 0.3, bgcolor: col, border: isLC ? "1.5px dashed" : "none", borderColor: "text.primary" }} />
                  <TextField variant="standard" value={lane.label} placeholder={`S${i + 1}`}
                    onChange={(e) => setLabel(lane.id, e.target.value)} inputProps={{ style: { fontSize: "0.78rem" } }} />
                  <TextField variant="standard" value={lv} placeholder="Target"
                    onChange={(e) => setLevel(lane.id, e.target.value)}
                    inputProps={{ style: { fontSize: "0.78rem", color: col, fontWeight: 600 } }} />
                  <Typography variant="caption" sx={{ fontVariantNumeric: "tabular-nums", color: "text.secondary" }}>
                    {iodById[lane.id] != null ? Math.round(iodById[lane.id]).toLocaleString() : "—"}
                  </Typography>
                  <IconButton size="small" onClick={() => deleteLane(lane.id)}><DeleteOutlineIcon sx={{ fontSize: 16 }} /></IconButton>
                </Box>
              );
            })}
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.4 }}>
            Set a <b>Level</b> per band (same level = same protein row, compared across lanes). Pick the
            <b> loading control level</b> for per-lane normalisation. IOD here is a display-image preview; the
            run recomputes on the full-resolution membrane.
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
