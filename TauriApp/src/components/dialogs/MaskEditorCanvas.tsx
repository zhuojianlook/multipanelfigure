/**
 * MaskEditorCanvas — interactive per-cell mask editor for the Cellpose modal.
 *
 * Cells-only distillation of the fluorescence Intensity picker's editor: one
 * <canvas> at the label raster's native resolution shows the input image with
 * derived cell outlines on top; the user paints new cells, extends/erases
 * existing ones, deletes, or merges. Labels are an Int32Array (0 = background,
 * N = the Nth cell), controlled by the parent; every commit calls onChange with
 * a fresh array. Undo/redo/pan/zoom are internal.
 *
 * Remount with a new `key` when the edited image changes — that resets the
 * tool/brush/history cleanly.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Box, Stack, IconButton, Typography, Tooltip, Slider } from "@mui/material";
import PanToolAltIcon from "@mui/icons-material/PanToolAlt";
import BrushIcon from "@mui/icons-material/Brush";
import AutoFixOffIcon from "@mui/icons-material/AutoFixOff";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import CallMergeIcon from "@mui/icons-material/CallMerge";
import UndoIcon from "@mui/icons-material/Undo";
import RedoIcon from "@mui/icons-material/Redo";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import { deriveBoundary, renderBoundaryCanvas, countNonZeroIds } from "../../utils/maskEdit";

type Tool = "pan" | "paint" | "erase" | "delete" | "merge";

type Props = {
  /** Input image (any resolution) drawn scaled behind the outlines. */
  baseImageB64: string;
  /** Current label raster (0 = background, N = cell N). */
  labels: Int32Array;
  width: number;
  height: number;
  /** Called with a fresh Int32Array after every edit. */
  onChange: (next: Int32Array) => void;
  /** Display height of the editor viewport in CSS px. */
  viewportHeight?: number;
};

const TOOLS: { id: Tool; icon: ReactNode; label: string }[] = [
  { id: "pan", icon: <PanToolAltIcon fontSize="small" />, label: "Pan / zoom (or drag)" },
  { id: "paint", icon: <BrushIcon fontSize="small" />, label: "Paint — drag on a cell to grow it, on empty space to add one" },
  { id: "erase", icon: <AutoFixOffIcon fontSize="small" />, label: "Erase — remove painted pixels" },
  { id: "delete", icon: <HighlightOffIcon fontSize="small" />, label: "Delete — click a cell to remove it whole" },
  { id: "merge", icon: <CallMergeIcon fontSize="small" />, label: "Merge — click two cells to join them" },
];

export default function MaskEditorCanvas(props: Props) {
  const { baseImageB64, labels, width, height, onChange, viewportHeight = 380 } = props;

  const [tool, setTool] = useState<Tool>("paint");
  const [brushPx, setBrushPx] = useState(10);
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  const [mergeFirstId, setMergeFirstId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [strokeTick, setStrokeTick] = useState(0);

  // Undo/redo of the FULL label raster (cap depth — each snapshot is w*h*4 B).
  const pastRef = useRef<Int32Array[]>([]);
  const futureRef = useRef<Int32Array[]>([]);
  const [histTick, setHistTick] = useState(0);

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const baseImgRef = useRef<HTMLImageElement | null>(null);
  const boundaryRef = useRef<HTMLCanvasElement | null>(null);
  const strokeMaskRef = useRef<Uint8Array | null>(null);
  const paintingRef = useRef(false);
  const paintToolRef = useRef<Tool>("paint");
  const paintStartIdRef = useRef(0);
  const lastXYRef = useRef<{ x: number; y: number } | null>(null);
  const panRef = useRef<{ cx: number; cy: number; tx: number; ty: number } | null>(null);
  const hoverXYRef = useRef<{ x: number; y: number } | null>(null);
  // Keep the latest labels in a ref so the (window-level) pointer handlers
  // never read a stale array mid-stroke.
  const labelsRef = useRef(labels);
  labelsRef.current = labels;

  const nCells = useMemo(() => countNonZeroIds(labels), [labels]);

  // ── Load the base image ────────────────────────────────────────────────
  useEffect(() => {
    if (!baseImageB64) { baseImgRef.current = null; setStrokeTick((t) => t + 1); return; }
    const img = new window.Image();
    img.onload = () => { baseImgRef.current = img; setStrokeTick((t) => t + 1); };
    img.src = baseImageB64.startsWith("data:") ? baseImageB64 : `data:image/png;base64,${baseImageB64.split(",").pop()}`;
  }, [baseImageB64]);

  // ── Re-derive cell outlines whenever labels change ─────────────────────
  useEffect(() => {
    boundaryRef.current = renderBoundaryCanvas(deriveBoundary(labels, width, height), width, height, [255, 230, 0]);
    setStrokeTick((t) => t + 1);
  }, [labels, width, height]);

  // ── Fit-to-viewport on mount / size change ─────────────────────────────
  const fitView = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap || !width || !height) return;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    const fit = Math.min(cw / width, ch / height) || 1;
    setView({ scale: fit, tx: (cw - width * fit) / 2, ty: (ch - height * fit) / 2 });
  }, [width, height]);
  useLayoutEffect(() => { fitView(); }, [fitView]);

  // ── Map a client point to integer image (label) coordinates ────────────
  const clientToImage = useCallback((clientX: number, clientY: number): { x: number; y: number } | null => {
    const cnv = canvasRef.current;
    if (!cnv) return null;
    const rect = cnv.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return null;
    const x = Math.round((clientX - rect.left) * (cnv.width / rect.width));
    const y = Math.round((clientY - rect.top) * (cnv.height / rect.height));
    if (x < 0 || y < 0 || x >= width || y >= height) return null;
    return { x, y };
  }, [width, height]);

  // ── Compositor ─────────────────────────────────────────────────────────
  useEffect(() => {
    const cnv = canvasRef.current; if (!cnv) return;
    const ctx = cnv.getContext("2d"); if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (baseImgRef.current) ctx.drawImage(baseImgRef.current, 0, 0, width, height);
    else { ctx.fillStyle = "#101014"; ctx.fillRect(0, 0, width, height); }
    if (boundaryRef.current) ctx.drawImage(boundaryRef.current, 0, 0);
    // Live stroke overlay (green = paint, red = erase).
    const sm = strokeMaskRef.current;
    if (sm && paintingRef.current) {
      const erase = paintToolRef.current === "erase";
      const img = ctx.getImageData(0, 0, width, height);
      const d = img.data;
      for (let i = 0; i < sm.length; i++) {
        if (!sm[i]) continue;
        const j = i * 4;
        if (erase) { d[j] = 255; d[j + 1] = 60; d[j + 2] = 60; }
        else { d[j] = 60; d[j + 1] = 230; d[j + 2] = 90; }
      }
      ctx.putImageData(img, 0, 0);
    }
    // Brush cursor ring.
    const hv = hoverXYRef.current;
    if (hv && (tool === "paint" || tool === "erase")) {
      ctx.beginPath();
      ctx.arc(hv.x, hv.y, brushPx, 0, Math.PI * 2);
      ctx.strokeStyle = tool === "erase" ? "rgba(255,90,90,0.9)" : "rgba(120,230,140,0.9)";
      ctx.lineWidth = Math.max(1, 2 / view.scale);
      ctx.stroke();
    }
  }, [strokeTick, tool, brushPx, width, height, view.scale]);

  // ── History ────────────────────────────────────────────────────────────
  const snapshot = useCallback(() => {
    pastRef.current = [...pastRef.current, new Int32Array(labelsRef.current)].slice(-20);
    futureRef.current = [];
    setHistTick((t) => t + 1);
  }, []);
  const undo = useCallback(() => {
    if (pastRef.current.length === 0) return;
    const prev = pastRef.current[pastRef.current.length - 1];
    pastRef.current = pastRef.current.slice(0, -1);
    futureRef.current = [...futureRef.current, new Int32Array(labelsRef.current)];
    setHistTick((t) => t + 1);
    onChange(prev);
  }, [onChange]);
  const redo = useCallback(() => {
    if (futureRef.current.length === 0) return;
    const nxt = futureRef.current[futureRef.current.length - 1];
    futureRef.current = futureRef.current.slice(0, -1);
    pastRef.current = [...pastRef.current, new Int32Array(labelsRef.current)];
    setHistTick((t) => t + 1);
    onChange(nxt);
  }, [onChange]);
  const clearAll = useCallback(() => {
    snapshot();
    onChange(new Int32Array(width * height));
  }, [snapshot, onChange, width, height]);

  // ── Stroke helpers ─────────────────────────────────────────────────────
  const stamp = useCallback((cx: number, cy: number, r: number) => {
    const m = strokeMaskRef.current; if (!m) return;
    const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(width - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(height - 1, Math.ceil(cy + r));
    const r2 = r * r;
    for (let y = y0; y <= y1; y++) {
      const dy = y - cy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx;
        if (dx * dx + dy * dy <= r2) m[y * width + x] = 1;
      }
    }
  }, [width, height]);

  const paintSegment = useCallback((x: number, y: number) => {
    const last = lastXYRef.current;
    if (!last) stamp(x, y, brushPx);
    else {
      const dx = x - last.x, dy = y - last.y;
      const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
      for (let i = 0; i <= steps; i++) stamp(last.x + dx * (i / steps), last.y + dy * (i / steps), brushPx);
    }
    lastXYRef.current = { x, y };
    setStrokeTick((t) => t + 1);
  }, [brushPx, stamp]);

  const commitStroke = useCallback(() => {
    const m = strokeMaskRef.current;
    if (m) {
      const cur = labelsRef.current;
      const next = new Int32Array(cur);
      let touched = 0;
      if (paintToolRef.current === "erase") {
        for (let i = 0; i < m.length; i++) { if (m[i] && next[i] !== 0) { next[i] = 0; touched++; } }
      } else {
        let id = paintStartIdRef.current | 0;
        if (id === 0) { let mx = 0; for (let i = 0; i < next.length; i++) if (next[i] > mx) mx = next[i]; id = mx + 1; }
        for (let i = 0; i < m.length; i++) { if (m[i] && next[i] !== id) { next[i] = id; touched++; } }
      }
      if (touched > 0) { snapshot(); onChange(next); }
    }
    strokeMaskRef.current = null;
    paintingRef.current = false;
    lastXYRef.current = null;
    paintStartIdRef.current = 0;
    setStrokeTick((t) => t + 1);
  }, [snapshot, onChange]);

  const deleteAt = useCallback((x: number, y: number) => {
    const cur = labelsRef.current;
    const id = cur[y * width + x];
    if (!id) return;
    snapshot();
    const next = new Int32Array(cur);
    for (let i = 0; i < next.length; i++) if (next[i] === id) next[i] = 0;
    onChange(next);
  }, [width, snapshot, onChange]);

  const mergeAt = useCallback((x: number, y: number) => {
    const cur = labelsRef.current;
    const id = cur[y * width + x];
    if (!id) return;
    if (mergeFirstId == null) { setMergeFirstId(id); return; }
    if (id === mergeFirstId) { setMergeFirstId(null); return; }
    snapshot();
    const next = new Int32Array(cur);
    for (let i = 0; i < next.length; i++) if (next[i] === id) next[i] = mergeFirstId;
    onChange(next);
    setMergeFirstId(null);
  }, [width, mergeFirstId, snapshot, onChange]);

  // ── Pointer handling ───────────────────────────────────────────────────
  const onDown = useCallback((e: React.MouseEvent) => {
    const pt = clientToImage(e.clientX, e.clientY);
    // Pan: pan tool, OR middle/right button, OR outside the image.
    if (tool === "pan" || e.button === 1 || e.button === 2 || !pt) {
      panRef.current = { cx: e.clientX, cy: e.clientY, tx: view.tx, ty: view.ty };
      return;
    }
    if (tool === "delete") { deleteAt(pt.x, pt.y); return; }
    if (tool === "merge") { mergeAt(pt.x, pt.y); return; }
    // paint / erase
    paintingRef.current = true;
    paintToolRef.current = tool;
    strokeMaskRef.current = new Uint8Array(width * height);
    lastXYRef.current = null;
    paintStartIdRef.current = tool === "paint" ? (labelsRef.current[pt.y * width + pt.x] | 0) : 0;
    paintSegment(pt.x, pt.y);
  }, [clientToImage, tool, view.tx, view.ty, width, height, deleteAt, mergeAt, paintSegment]);

  const onMove = useCallback((e: React.MouseEvent) => {
    if (panRef.current) {
      const p = panRef.current;
      setView((v) => ({ ...v, tx: p.tx + (e.clientX - p.cx), ty: p.ty + (e.clientY - p.cy) }));
      return;
    }
    const pt = clientToImage(e.clientX, e.clientY);
    hoverXYRef.current = pt;
    if (pt) setHoverId(labelsRef.current[pt.y * width + pt.x] || null);
    if (paintingRef.current && pt) paintSegment(pt.x, pt.y);
    else setStrokeTick((t) => t + 1); // refresh brush cursor
  }, [clientToImage, width, paintSegment]);

  const endStroke = useCallback(() => {
    if (panRef.current) { panRef.current = null; return; }
    if (paintingRef.current) commitStroke();
  }, [commitStroke]);

  // Window-level up so a stroke that ends off-canvas still commits.
  useEffect(() => {
    const up = () => endStroke();
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [endStroke]);

  const onWheel = useCallback((e: React.WheelEvent) => {
    const wrap = wrapRef.current; if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const lx = e.clientX - rect.left, ly = e.clientY - rect.top;
    setView((v) => {
      const s2 = Math.min(12, Math.max(0.2, v.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      // Keep the point under the cursor fixed (origin 0 0).
      const k = s2 / v.scale;
      return { scale: s2, tx: lx - (lx - v.tx) * k, ty: ly - (ly - v.ty) * k };
    });
  }, []);

  // Keyboard shortcuts while the editor is focused/hovered.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!wrapRef.current?.matches(":hover")) return;
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
      if (e.key === "v") setTool("pan");
      else if (e.key === "b") setTool("paint");
      else if (e.key === "e") setTool("erase");
      else if (e.key === "d") setTool("delete");
      else if (e.key === "m") setTool("merge");
      else if (e.key === "[") setBrushPx((b) => Math.max(2, b - 2));
      else if (e.key === "]") setBrushPx((b) => Math.min(80, b + 2));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

  const cursor = tool === "pan" ? (panRef.current ? "grabbing" : "grab")
    : tool === "delete" || tool === "merge" ? "pointer" : "crosshair";

  return (
    <Stack spacing={0.5}>
      {/* Toolbar */}
      <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap">
        {TOOLS.map((t) => (
          <Tooltip key={t.id} title={t.label} placement="top">
            <IconButton size="small" onClick={() => { setTool(t.id); setMergeFirstId(null); }}
              sx={{ bgcolor: tool === t.id ? "primary.main" : "action.hover", color: tool === t.id ? "primary.contrastText" : "text.primary", "&:hover": { bgcolor: tool === t.id ? "primary.dark" : "action.selected" }, borderRadius: 1, p: 0.5 }}>
              {t.icon}
            </IconButton>
          </Tooltip>
        ))}
        <Box sx={{ width: 8 }} />
        <Tooltip title="Undo (⌘Z)"><span><IconButton size="small" onClick={undo} disabled={pastRef.current.length === 0}><UndoIcon fontSize="small" /></IconButton></span></Tooltip>
        <Tooltip title="Redo (⌘⇧Z)"><span><IconButton size="small" onClick={redo} disabled={futureRef.current.length === 0}><RedoIcon fontSize="small" /></IconButton></span></Tooltip>
        <Tooltip title="Clear all masks"><span><IconButton size="small" onClick={clearAll}><DeleteSweepIcon fontSize="small" /></IconButton></span></Tooltip>
        <Tooltip title="Reset view"><span><IconButton size="small" onClick={fitView}><CenterFocusStrongIcon fontSize="small" /></IconButton></span></Tooltip>
        <Box sx={{ flex: 1 }} />
        <Typography variant="caption" color="text.secondary">
          {nCells} cell{nCells === 1 ? "" : "s"}{hoverId ? ` · #${hoverId}` : ""}{mergeFirstId ? ` · merge #${mergeFirstId} → …` : ""}
        </Typography>
      </Stack>

      {/* Brush size (paint / erase only) */}
      {(tool === "paint" || tool === "erase") && (
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="caption" sx={{ width: 70 }}>Brush {brushPx}px</Typography>
          <Slider size="small" min={2} max={60} step={1} value={brushPx} onChange={(_, v) => setBrushPx(v as number)} sx={{ maxWidth: 200 }} />
        </Stack>
      )}

      {/* Canvas viewport */}
      <Box ref={wrapRef}
        onMouseDown={onDown} onMouseMove={onMove}
        onMouseLeave={() => { hoverXYRef.current = null; setStrokeTick((t) => t + 1); }}
        onWheel={onWheel}
        onContextMenu={(e) => e.preventDefault()}
        sx={{ position: "relative", width: "100%", height: viewportHeight, overflow: "hidden",
          border: "1px solid", borderColor: "divider", borderRadius: 1, background: "#0c0c10", cursor }}>
        <canvas ref={canvasRef} width={width} height={height}
          style={{ position: "absolute", left: 0, top: 0, width, height,
            transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`, transformOrigin: "0 0",
            imageRendering: "pixelated" }} />
      </Box>
      <Typography variant="caption" color="text.secondary">
        Paint on a cell to grow it, on empty space to add one · Delete/Merge click cells · scroll to zoom · V/B/E/D/M keys
      </Typography>
    </Stack>
  );
}
