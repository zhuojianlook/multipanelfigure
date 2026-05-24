// ─────────────────────────────────────────────────────────────
//  BandPickerDialog — interactive western-blot band picker
// ─────────────────────────────────────────────────────────────
//  The user drops the WHOLE blot membrane onto a Source node and
//  wires it into a "Band picker" node.  This dialog shows that
//  membrane and lets them:
//    • auto-detect lanes (column intensity-profile peak finder),
//    • draw / move / resize a rectangular ROI per lane,
//    • label each lane and tick exactly one as the loading control,
//    • optionally draw a background ROI (toggle) for subtraction.
//
//  The ROIs are stored as NORMALISED 0..1 coords so they survive
//  image resizing / DPI changes, and are baked into auto-generated
//  Python at run time (see generateBandPickerCode) that maps them
//  onto the full-res membrane, crops each lane, inverts, subtracts
//  background and sums → integrated optical density (IOD) per lane.
//  The emitted table schema (source / is_reference / iod) matches
//  the existing Normalize node, so Source → Band picker → Normalize
//  → R plot works unchanged.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box, Typography, Button, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, Tooltip, ToggleButton, ToggleButtonGroup, Radio,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";

// ── Types (shared with AnalysisNodeGraph via import) ──────────
export interface BandRoi {
  id: string;
  /** Normalised 0..1 rect over the membrane image (top-left origin). */
  x: number; y: number; w: number; h: number;
  label: string;
  isReference: boolean;
}
export interface BandPickerConfig {
  version: 1;
  lanes: BandRoi[];
  /** "percentile" → per-lane Nth-percentile floor (default);
   *  "roi" → subtract the mean of a user-drawn background rect. */
  bgMode: "percentile" | "roi";
  bgRoi?: { x: number; y: number; w: number; h: number } | null;
  /** Percentile (0..100) used when bgMode === "percentile". */
  bgPercentile: number;
}

export function emptyBandConfig(): BandPickerConfig {
  return { version: 1, lanes: [], bgMode: "percentile", bgRoi: null, bgPercentile: 5 };
}

const uid = () => `lane_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

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

imgs = [v for v in inputs.values() if isinstance(v, dict) and "image" in v]
if not imgs:
    raise SystemExit("No membrane image — wire the blot (Source) into this node.")
arr = np.asarray(imgs[0]["image"]).astype(np.float32)
if arr.ndim == 3:
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
else:
    gray = arr
H, W = gray.shape[:2]
inv_full = 255.0 - gray            # dark protein bands -> bright signal

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
rows = []
for i, lane in enumerate(CFG.get("lanes", [])):
    x0, y0, x1, y1 = _px(lane)
    if x1 <= x0 or y1 <= y0:
        continue
    inv = inv_full[y0:y1, x0:x1]
    bg = bg_global if bg_global is not None else float(np.percentile(inv, pct))
    signal = np.clip(inv - bg, 0, None)
    label = (lane.get("label") or "").strip() or f"Lane {i + 1}"
    rows.append({
        "source": label,
        "is_reference": bool(lane.get("isReference", False)),
        "iod": float(signal.sum()),
        "mean_signal": float(signal.mean()),
        "area_px": int(inv.size),
    })

if not rows:
    raise SystemExit("No lanes defined — open 'Pick bands…' and mark each lane.")

# Guarantee exactly one reference (loading control).  If the user
# didn't tick one, assume the first lane; if they ticked several,
# keep the first ticked.
ref_seen = False
for r in rows:
    if r["is_reference"] and not ref_seen:
        ref_seen = True
    else:
        r["is_reference"] = False
if not ref_seen:
    rows[0]["is_reference"] = True

ref = next(r for r in rows if r["is_reference"])
print(f"quantified {len(rows)} lane(s); reference = {ref['source']!r}")
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
      label: `Lane ${lanes.length + 1}`, isReference: lanes.length === 0,
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
  /** Membrane image as a base64 PNG (no data: prefix) or a data URL. */
  imageSrc: string | null;
  initial: BandPickerConfig | null;
  onClose: () => void;
  onSave: (cfg: BandPickerConfig) => void;
}

const MAX_DISP_W = 620;
const MAX_DISP_H = 520;

export default function BandPickerDialog(props: BandPickerDialogProps) {
  const { open, imageSrc, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<BandPickerConfig>(initial ?? emptyBandConfig());
  const [selId, setSelId] = useState<string | null>(null);
  const [drawingBg, setDrawingBg] = useState(false);
  const [detecting, setDetecting] = useState(false);
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
      setCfg((c) => ({
        ...c,
        lanes: [...c.lanes, { id: newId, x: nx, y: ny, w: 0, h: 0, label: `Lane ${c.lanes.length + 1}`, isReference: c.lanes.length === 0 }],
      }));
      setSelId(newId);
    }
  }, [cfg.lanes, cfg.bgRoi, clientToNorm, drawingBg]);

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
  const setReference = (id: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.map((l) => ({ ...l, isReference: l.id === id })) }));
  const deleteLane = (id: string) =>
    setCfg((c) => ({ ...c, lanes: c.lanes.filter((l) => l.id !== id) }));

  const applyDetectedLanes = (rects: Array<{ x: number; y: number; w: number; h: number }>): boolean => {
    if (!rects.length) return false;
    const lanes: BandRoi[] = rects.map((r, i) => ({
      id: uid(), x: clamp01(r.x), y: clamp01(r.y), w: clamp01(r.w), h: clamp01(r.h),
      label: `Lane ${i + 1}`, isReference: i === 0,
    }));
    setCfg((c) => ({ ...c, lanes }));
    setSelId(lanes[0]?.id ?? null);
    return true;
  };

  const runAutoDetect = async () => {
    if (detecting) return;
    setDetecting(true);
    try {
      // Prefer the backend detector (robust band-row-constrained algorithm,
      // pure-numpy so it ships in the frozen sidecar). Fall back to the local
      // heuristic if the endpoint isn't reachable (e.g. older sidecar / no backend).
      if (imageSrc) {
        try {
          const b64 = imageSrc.startsWith("data:") ? imageSrc.split(",")[1] : imageSrc;
          const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
          const resp = await fetch(`${apiBase}/api/analysis/wb-detect-bands`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_b64: b64 }),
          });
          if (resp.ok) {
            const data = await resp.json();
            if (Array.isArray(data.lanes) && applyDetectedLanes(data.lanes)) return;
          }
        } catch { /* backend unavailable → local fallback */ }
      }
      if (gray && natW && natH) applyDetectedLanes(autoDetectLanes(gray, natW, natH));
    } finally {
      setDetecting(false);
    }
  };

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
          Drag on the blot to draw a lane · drag a lane to move · handles to resize
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
                {/* Lanes */}
                {cfg.lanes.map((lane, i) => {
                  const sel = lane.id === selId;
                  return (
                    <g key={lane.id}>
                      <rect data-role="lane" data-id={lane.id}
                        x={lane.x * disp.w} y={lane.y * disp.h} width={lane.w * disp.w} height={lane.h * disp.h}
                        fill={sel ? "rgba(25,118,210,0.16)" : "rgba(25,118,210,0.08)"}
                        stroke={lane.isReference ? "#e53935" : "#1976d2"} strokeWidth={sel ? 2 : 1.5}
                        style={{ cursor: "move" }} />
                      <text x={lane.x * disp.w + 3} y={lane.y * disp.h + 12} fontSize={11}
                        fill={lane.isReference ? "#e53935" : "#1976d2"} style={{ pointerEvents: "none", fontWeight: 600 }}>
                        {i + 1}{lane.isReference ? " ★" : ""}
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
            <Button size="small" variant="outlined" startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
              onClick={runAutoDetect} disabled={!srcUrl || detecting}>
              {detecting ? "Detecting…" : "Auto-detect lanes"}
            </Button>
            <Typography variant="caption" sx={{ color: "text.secondary" }}>{cfg.lanes.length} lane(s)</Typography>
          </Box>

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

          {/* Lane list */}
          <Box sx={{ flex: 1, overflow: "auto", maxHeight: 360, border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
            <Box sx={{ display: "grid", gridTemplateColumns: "24px 1fr 70px 28px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.5, position: "sticky", top: 0, bgcolor: "background.paper", borderBottom: "1px solid", borderColor: "divider" }}>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>#</Typography>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>Label</Typography>
              <Typography variant="caption" sx={{ fontWeight: 700 }}>IOD</Typography>
              <Tooltip title="Loading control (reference)"><Typography variant="caption" sx={{ fontWeight: 700 }}>Ref</Typography></Tooltip>
              <span />
            </Box>
            {cfg.lanes.length === 0 ? (
              <Typography variant="caption" sx={{ display: "block", p: 1.5, color: "text.secondary", fontStyle: "italic" }}>
                No lanes yet. Click “Auto-detect lanes” or drag rectangles on the blot.
              </Typography>
            ) : cfg.lanes.map((lane, i) => (
              <Box key={lane.id}
                onMouseEnter={() => setSelId(lane.id)}
                sx={{ display: "grid", gridTemplateColumns: "24px 1fr 70px 28px 28px", alignItems: "center", gap: 0.5, px: 1, py: 0.4,
                  bgcolor: lane.id === selId ? "action.hover" : undefined, borderBottom: "1px solid", borderColor: "divider" }}>
                <Typography variant="caption">{i + 1}</Typography>
                <TextField variant="standard" value={lane.label} placeholder={`Lane ${i + 1}`}
                  onChange={(e) => setLabel(lane.id, e.target.value)} inputProps={{ style: { fontSize: "0.78rem" } }} />
                <Typography variant="caption" sx={{ fontVariantNumeric: "tabular-nums", color: "text.secondary" }}>
                  {iodById[lane.id] != null ? Math.round(iodById[lane.id]).toLocaleString() : "—"}
                </Typography>
                <Radio size="small" checked={lane.isReference} onChange={() => setReference(lane.id)} sx={{ p: 0.25 }} />
                <IconButton size="small" onClick={() => deleteLane(lane.id)}><DeleteOutlineIcon sx={{ fontSize: 16 }} /></IconButton>
              </Box>
            ))}
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.4 }}>
            IOD shown is a quick preview from the display image; the run recomputes it on the full-resolution membrane.
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
