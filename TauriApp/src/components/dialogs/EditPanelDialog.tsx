/* ------------------------------------------------------------------
   EditPanelDialog -- MUI Dialog with tabs for editing a
   single panel's properties.
   Tabs: Crop/Resize, Image Adjustments, Labels,
         Scale Bar, Annotations, Zoom Inset.
   Includes a live image preview at the top of the dialog
   that updates via debounced fetch to /api/panel-preview.
   ------------------------------------------------------------------ */

import { useState, useEffect, useRef, useCallback, useLayoutEffect } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  TextField,
  Slider,
  Typography,
  Switch,
  FormControlLabel,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  ToggleButtonGroup,
  ToggleButton,
  ButtonGroup,
  Checkbox,
  Divider,
  Tooltip,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DeleteIcon from "@mui/icons-material/Delete";
import AddIcon from "@mui/icons-material/Add";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import FlipIcon from "@mui/icons-material/Flip";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import StarIcon from "@mui/icons-material/Star";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import { symbolToSvgPoints } from "../../utils/symbolDefs";
import CircleIcon from "@mui/icons-material/Circle";
import CloseIcon from "@mui/icons-material/Close";
import UndoIcon from "@mui/icons-material/Undo";
import RedoIcon from "@mui/icons-material/Redo";
import { VolumeViewerDialog } from "./VolumeViewer";
import { useFigureStore } from "../../store/figureStore";
import { api } from "../../api/client";
import type {
  PanelInfo,
  LabelSettings,
  ScaleBarSettings,
  SymbolSettings,
  ZoomInsetSettings,
} from "../../api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  row: number;
  col: number;
}

function TabPanel({ children, value, index }: { children: React.ReactNode; value: number; index: number }) {
  // Keep every tab mounted but hide inactive ones. Unmounting on tab switch
  // threw away CropCanvas state (image ref, `loaded` flag, canvas bitmap),
  // leaving the crop overlay invisible on the first render after a switch.
  // With CSS hiding, the canvas keeps its pixels and its drawn overlay is
  // preserved across tab changes.
  const active = value === index;
  return (
    <Box
      role="tabpanel"
      hidden={!active}
      aria-hidden={!active}
      sx={{ py: active ? 2 : 0, display: active ? "block" : "none" }}
    >
      {children}
    </Box>
  );
}

// Position presets (percentage 0-100)
type PositionPreset = "Custom" | "Top-Left" | "Top-Right" | "Bottom-Left" | "Bottom-Right" | "Center";
const POSITION_PRESETS: { label: PositionPreset; x: (d: number) => number; y: (d: number) => number }[] = [
  { label: "Custom", x: () => 5, y: () => 5 },
  { label: "Top-Left", x: (d) => d, y: (d) => d },
  { label: "Top-Right", x: (d) => 100 - d, y: (d) => d },
  { label: "Bottom-Left", x: (d) => d, y: (d) => 100 - d },
  { label: "Bottom-Right", x: (d) => 100 - d, y: (d) => 100 - d },
  { label: "Center", x: () => 50, y: () => 50 },
];

function defaultLabel(): LabelSettings {
  return {
    text: "Label",
    font_path: null,
    font_name: "arial.ttf",
    font_size: 20,
    font_style: [],
    color: "#FFFFFF",
    position_x: 5,     // percentage 0-100 (matching backend convention)
    position_y: 5,
    rotation: 0,
    default_color: "#FFFFFF",
    position_preset: "Custom",
    edge_distance: 3,
    linked_to_header: false,
    styled_segments: [],
  };
}

function panelLetterLabel(row: number, col: number, cols: number): LabelSettings {
  const idx = row * cols + col;
  const letter = idx < 26 ? String.fromCharCode(97 + idx) : `${String.fromCharCode(97 + Math.floor(idx / 26) - 1)}${String.fromCharCode(97 + (idx % 26))}`;
  return {
    text: letter,
    font_path: null,
    font_name: "arial.ttf",
    font_size: 20,
    font_style: [],
    color: "#FFFFFF",
    position_x: 3,
    position_y: 3,
    rotation: 0,
    default_color: "#FFFFFF",
    position_preset: "Top-Left",
    edge_distance: 3,
    linked_to_header: true,
    styled_segments: [],
  };
}

function defaultScaleBar(): ScaleBarSettings {
  return {
    micron_per_pixel: 1.0,
    bar_length_microns: 100,
    bar_position: [0.05, 0.9],
    bar_height: 5,
    bar_color: "#FFFFFF",
    label: "",
    font_size: 10,
    font_name: "arial.ttf",
    font_path: null,
    label_x_offset: 0,
    label_font_style: [],
    label_color: "#FFFFFF",
    position_preset: "Bottom-Right",
    position_x: 90,
    position_y: 90,
    edge_distance: 5,
    unit: "um",
    scale_name: "",
    styled_segments: [],
    draggable: false,
  };
}

function defaultSymbol(fontName?: string): SymbolSettings {
  return {
    name: "",
    shape: "Arrow",
    x: 50,          // percentage 0-100
    y: 50,          // percentage 0-100
    color: "#FF0000",
    size: 30,
    rotation: -45,
    label_text: "",
    label_color: "#FFFFFF",
    label_offset_x: 0,
    label_offset_y: 0,
    label_position_x: -1,
    label_position_y: -1,
    label_font_name: fontName || "arial.ttf",
    label_font_size: 12,
    label_font_path: null,
    label_font_style: [],
    label_styled_segments: [],
  };
}

function defaultZoomInset(): ZoomInsetSettings {
  return {
    inset_type: "Standard Zoom",
    zoom_factor: 2.0,
    rectangle_color: "#FFFFFF",
    rectangle_width: 2,
    line_color: "#FFFFFF",
    line_width: 1,
    x: 10,
    y: 10,
    width: 100,
    height: 100,
    target_x: 200,
    target_y: 200,
    separate_image_name: "",
    x_main: 0.0,
    y_main: 0.0,
    width_main: 1.0,
    height_main: 1.0,
    x_inset: 0.0,
    y_inset: 0.0,
    width_inset: 1.0,
    height_inset: 1.0,
    side: "right",
    margin_offset: 5,
    scale_bar: null,
    zoom_label: null,
  };
}

/* ================================================================
   Aspect-ratio presets
   ================================================================ */
type AspectPreset = "Free" | "Custom" | "1:1" | "4:3" | "16:9";
const ASPECT_PRESETS: AspectPreset[] = ["Free", "Custom", "1:1", "4:3", "16:9"];

function ratioStrToPreset(s: string): AspectPreset {
  if (!s || s.trim() === "") return "Free";
  if (s === "1:1") return "1:1";
  if (s === "4:3") return "4:3";
  if (s === "16:9") return "16:9";
  if (s.includes(":")) return "Custom"; // custom ratio like "3:2"
  return "Free";
}

function parseRatioStr(s: string): number | null {
  const m = s.match(/^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/);
  if (!m) return null;
  const w = parseFloat(m[1]), h = parseFloat(m[2]);
  return h > 0 ? w / h : null;
}

/* ================================================================
   Compute rotated bounding box dimensions
   ================================================================ */
function computeRotatedDims(w: number, h: number, angleDeg: number): { rotW: number; rotH: number } {
  const rad = (angleDeg || 0) * Math.PI / 180;
  const cosA = Math.abs(Math.cos(rad));
  const sinA = Math.abs(Math.sin(rad));
  return {
    rotW: Math.ceil(w * cosA + h * sinA),
    rotH: Math.ceil(w * sinA + h * cosA),
  };
}

/* ================================================================
   CropCanvas -- renders the source image with a draggable /
   resizable crop rectangle overlay.
   Supports rotation: image is drawn rotated, crop rect stays
   axis-aligned in the rotated image's coordinate space.
   ================================================================ */
interface CropCanvasProps {
  imageSrc: string;           // base64 data-url of the ORIGINAL (unrotated) thumbnail
  aspectPreset: AspectPreset;
  customRatio?: number | null; // for Custom preset
  cropRect: { x: number; y: number; w: number; h: number };  // in rotated-image pixel coords
  imgNatW: number;            // original (unrotated) thumbnail width
  imgNatH: number;            // original (unrotated) thumbnail height
  rotation: number;           // degrees (0-360)
  flipH?: boolean;            // horizontal flip
  flipV?: boolean;            // vertical flip
  active?: boolean;           // true when the hosting tab is currently visible
  onChange: (rect: { x: number; y: number; w: number; h: number }) => void;
  onCommit?: (rect: { x: number; y: number; w: number; h: number }) => void;
}

const HANDLE_SIZE = 10;

type DragMode = "none" | "move" | "nw" | "ne" | "sw" | "se" | "n" | "s" | "e" | "w";

function CropCanvas({ imageSrc, aspectPreset, customRatio, cropRect, imgNatW, imgNatH, rotation, flipH, flipV, active = true, onChange, onCommit }: CropCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [canvasW, setCanvasW] = useState(256);
  const dragRef = useRef<{ mode: DragMode; startMx: number; startMy: number; startRect: typeof cropRect }>({
    mode: "none", startMx: 0, startMy: 0, startRect: cropRect,
  });
  const latestRectRef = useRef(cropRect);
  latestRectRef.current = cropRect;

  // When the tab becomes active, re-measure container width (it was 0
  // while display:none'd). The rAF-based redraw below handles repainting.
  useEffect(() => {
    if (!active) return;
    const el = containerRef.current;
    if (el) {
      const w = Math.floor(el.clientWidth);
      if (w > 0) {
        setCanvasW((cur) => (Math.abs(w - cur) >= 1 ? w : cur));
      }
    }
  }, [active]);

  // Measure container width so canvas always fits.
  // Debounce via rAF and ignore sub-pixel jitter to avoid a ResizeObserver →
  // state → re-render → ResizeObserver feedback loop that shows up on
  // Windows/WebView2 as a "ghost" rapid resize when the user switches
  // aspect-ratio presets.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let raf = 0;
    let pending: number | null = null;
    const applyPending = () => {
      raf = 0;
      if (pending != null) {
        setCanvasW((cur) => (Math.abs(pending! - cur) >= 1 ? pending! : cur));
        pending = null;
      }
    };
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w > 0) {
          pending = w;
          if (!raf) raf = requestAnimationFrame(applyPending);
        }
      }
    });
    ro.observe(el);
    // Set initial width
    setCanvasW(Math.floor(el.clientWidth) || 256);
    return () => {
      ro.disconnect();
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  // Compute rotated bounding box
  const { rotW, rotH } = computeRotatedDims(imgNatW, imgNatH, rotation);
  const rad = (rotation || 0) * Math.PI / 180;

  // Computed scale (fit rotated image into measured canvas width)
  const scale = rotW > 0 ? canvasW / rotW : 1;
  const canvasH = Math.max(1, Math.round(rotH * scale));

  // Keep refs for values needed in window event listeners (avoid stale closures)
  const scaleRef = useRef(scale);
  scaleRef.current = scale;
  const rotWRef = useRef(rotW);
  rotWRef.current = rotW;
  const rotHRef = useRef(rotH);
  rotHRef.current = rotH;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onCommitRef = useRef(onCommit);
  onCommitRef.current = onCommit;
  const aspectPresetRef = useRef(aspectPreset);
  aspectPresetRef.current = aspectPreset;

  // Load image element once. Cover every known edge case:
  // - async onload for non-cached URLs
  // - already-complete (cached / data URL) images where onload may not fire
  //   before useLayoutEffect runs on the next render
  // - decode errors → log but don't hang in a loading state
  useEffect(() => {
    if (!imageSrc) return;
    const img = new window.Image();
    const markLoaded = () => {
      imgRef.current = img;
      setLoaded(true);
    };
    img.onload = markLoaded;
    img.onerror = () => {
      console.warn("[CropCanvas] failed to load image", imageSrc.slice(0, 64));
    };
    img.src = imageSrc;
    // Synchronous path for images already decoded (browsers differ on
    // whether onload fires for data URLs that are already ready).
    if (img.complete && img.naturalWidth > 0) {
      markLoaded();
    }
  }, [imageSrc]);

  // Every time the image becomes loaded, schedule an extra draw on the
  // next animation frame. useLayoutEffect (no deps, below) will usually
  // pick this up, but this is a belt-and-suspenders guarantee.
  useEffect(() => {
    if (!loaded) return;
    const id = requestAnimationFrame(() => drawRef.current());
    return () => cancelAnimationFrame(id);
  }, [loaded]);

  // Absolute nuclear fallback: while the tab is active, poll the draw
  // function every 250 ms. If the canvas bitmap has been cleared for
  // ANY reason (browser optimisation, layout change, whatever), this
  // guarantees it gets repainted within a quarter-second. Cheap — a
  // fully-populated draw is microseconds.
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => drawRef.current(), 250);
    return () => window.clearInterval(id);
  }, [active]);

  // Final safety net — on initial mount and whenever props change such
  // that the canvas would need repainting, schedule a cascade of draws
  // over the first few seconds. Covers any remaining race where the
  // browser hadn't laid out / decoded the image when earlier draws fired.
  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (const ms of [16, 50, 150, 400, 800, 1500, 3000]) {
      timers.push(setTimeout(() => drawRef.current(), ms));
    }
    return () => timers.forEach(clearTimeout);
  }, [imageSrc]);

  // Draw — extracted into a ref'd function so it can be called from multiple
  // places (layout effect, rAF tick, active-change effect). Canvas bitmaps
  // can be discarded by the browser in a display:none'd tab, so we rely on
  // multiple triggers to guarantee a repaint whenever the tab is shown.
  const drawRef = useRef<() => void>(() => {});
  drawRef.current = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !loaded) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasW;
    canvas.height = canvasH;

    // Draw rotated + flipped image: translate to center, rotate, flip, draw centered
    const displayW = imgNatW * scale;
    const displayH = imgNatH * scale;
    ctx.save();
    ctx.translate(canvasW / 2, canvasH / 2);
    ctx.rotate(rad);
    ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    ctx.drawImage(img, -displayW / 2, -displayH / 2, displayW, displayH);
    ctx.restore();

    // Darken outside crop rect (in rotated space = canvas space)
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    const rx = cropRect.x * scale;
    const ry = cropRect.y * scale;
    const rw = cropRect.w * scale;
    const rh = cropRect.h * scale;

    // Top
    ctx.fillRect(0, 0, canvasW, ry);
    // Bottom
    ctx.fillRect(0, ry + rh, canvasW, canvasH - ry - rh);
    // Left
    ctx.fillRect(0, ry, rx, rh);
    // Right
    ctx.fillRect(rx + rw, ry, canvasW - rx - rw, rh);

    // Crop rectangle border
    ctx.strokeStyle = "#2196f3";
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);

    // Corner handles (4 corners)
    ctx.fillStyle = "#2196f3";
    const hs = HANDLE_SIZE;
    const corners: [number, number][] = [
      [rx - hs / 2, ry - hs / 2],
      [rx + rw - hs / 2, ry - hs / 2],
      [rx - hs / 2, ry + rh - hs / 2],
      [rx + rw - hs / 2, ry + rh - hs / 2],
    ];
    for (const [hx, hy] of corners) {
      ctx.fillRect(hx, hy, hs, hs);
    }
    // Edge midpoint handles (4 edges) — smaller than corners
    const ehs = 8;
    const edges: [number, number][] = [
      [rx + rw / 2 - ehs / 2, ry - ehs / 2],           // N
      [rx + rw / 2 - ehs / 2, ry + rh - ehs / 2],      // S
      [rx - ehs / 2, ry + rh / 2 - ehs / 2],            // W
      [rx + rw - ehs / 2, ry + rh / 2 - ehs / 2],       // E
    ];
    ctx.fillStyle = "rgba(33,150,243,0.7)";
    for (const [ex, ey] of edges) {
      ctx.fillRect(ex, ey, ehs, ehs);
    }
  };

  // Trigger 1 — synchronous after every render (catches the common case
  // where cropRect / canvas size / rotation etc. change).
  useLayoutEffect(() => {
    drawRef.current();
  }); // no deps: every render

  // Trigger 2 — when the hosting tab becomes active, run several rAF
  // passes. Some browsers (notably WKWebView) throw away the canvas
  // bitmap while the container is display:none'd; rAF-ing after the
  // DOM re-shows lets us repaint AFTER the browser has re-allocated
  // the canvas surface.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const frames: number[] = [];
    const schedule = (n: number) => {
      if (n <= 0 || cancelled) return;
      frames.push(requestAnimationFrame(() => {
        drawRef.current();
        schedule(n - 1);
      }));
    };
    schedule(3); // 3 consecutive frames
    return () => {
      cancelled = true;
      for (const f of frames) cancelAnimationFrame(f);
    };
  }, [active, loaded]);

  // Hit-test for drag mode — prioritises corners > edges > move
  // Grab zones scale with crop size (min 14px, max 20px) for reliable interaction
  const hitTest = (mx: number, my: number): DragMode => {
    const rx = cropRect.x * scale;
    const ry = cropRect.y * scale;
    const rw = cropRect.w * scale;
    const rh = cropRect.h * scale;

    // Dynamic grab size: larger when crop is large, but always at least 14px
    const minDim = Math.min(rw, rh);
    const cornerR = Math.max(14, Math.min(20, minDim * 0.3));
    const edgeZone = Math.max(10, Math.min(16, minDim * 0.2));

    // Distance from each corner
    const dNW = Math.hypot(mx - rx, my - ry);
    const dNE = Math.hypot(mx - (rx + rw), my - ry);
    const dSW = Math.hypot(mx - rx, my - (ry + rh));
    const dSE = Math.hypot(mx - (rx + rw), my - (ry + rh));

    // Corners first (always take priority)
    if (dNW < cornerR) return "nw";
    if (dNE < cornerR) return "ne";
    if (dSW < cornerR) return "sw";
    if (dSE < cornerR) return "se";

    // Edge midpoint hit-test — check proximity to edge midpoint handles first
    const midN = { x: rx + rw / 2, y: ry };
    const midS = { x: rx + rw / 2, y: ry + rh };
    const midW = { x: rx, y: ry + rh / 2 };
    const midE = { x: rx + rw, y: ry + rh / 2 };
    const edgeMidR = cornerR;  // Same generous grab radius as corners
    if (Math.hypot(mx - midN.x, my - midN.y) < edgeMidR) return "n";
    if (Math.hypot(mx - midS.x, my - midS.y) < edgeMidR) return "s";
    if (Math.hypot(mx - midW.x, my - midW.y) < edgeMidR) return "w";
    if (Math.hypot(mx - midE.x, my - midE.y) < edgeMidR) return "e";

    // Edge line proximity — checked BEFORE move so edges are always grabbable
    const inRectX = mx >= rx - edgeZone && mx <= rx + rw + edgeZone;
    const inRectY = my >= ry - edgeZone && my <= ry + rh + edgeZone;
    if (inRectY && Math.abs(my - ry) <= edgeZone && inRectX) return "n";
    if (inRectY && Math.abs(my - (ry + rh)) <= edgeZone && inRectX) return "s";
    if (inRectX && Math.abs(mx - rx) <= edgeZone && inRectY) return "w";
    if (inRectX && Math.abs(mx - (rx + rw)) <= edgeZone && inRectY) return "e";

    // Interior = move
    if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) return "move";
    return "none";
  };

  const getCanvasPos = (e: React.MouseEvent | MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { mx: 0, my: 0 };
    const rect = canvas.getBoundingClientRect();
    // Canvas logical size matches CSS size (no maxWidth scaling needed)
    return {
      mx: (e.clientX - rect.left) * (canvas.width / rect.width),
      my: (e.clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const clampRect = (r: typeof cropRect) => {
    const minSize = 10;
    const maxW = rotWRef.current;
    const maxH = rotHRef.current;
    let { x, y, w, h } = r;
    w = Math.max(minSize, Math.min(w, maxW));
    h = Math.max(minSize, Math.min(h, maxH));
    x = Math.max(0, Math.min(x, maxW - w));
    y = Math.max(0, Math.min(y, maxH - h));
    return { x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) };
  };

  const enforceAspect = (r: typeof cropRect, mode: DragMode): typeof cropRect => {
    const ap = aspectPresetRef.current;
    if (ap === "Free") return r;
    let ratio = 1;
    if (ap === "1:1") ratio = 1;
    else if (ap === "4:3") ratio = 4 / 3;
    else if (ap === "16:9") ratio = 16 / 9;
    else if (ap === "Custom" && customRatio) ratio = customRatio;
    else return r;

    if (mode === "move") return r;
    const newH = r.w / ratio;
    return { ...r, h: newH };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const { mx, my } = getCanvasPos(e);
    const mode = hitTest(mx, my);
    if (mode === "none") return;
    e.preventDefault();
    dragRef.current = { mode, startMx: mx, startMy: my, startRect: { ...cropRect } };

    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      const { mx: cmx, my: cmy } = getCanvasPos(ev);
      const curScale = scaleRef.current;
      const dx = (cmx - dragRef.current.startMx) / curScale;
      const dy = (cmy - dragRef.current.startMy) / curScale;
      const sr = dragRef.current.startRect;
      let newR = { ...sr };

      const m = dragRef.current.mode;
      if (m === "move") {
        newR = { x: sr.x + dx, y: sr.y + dy, w: sr.w, h: sr.h };
      } else if (m === "se") {
        newR = { x: sr.x, y: sr.y, w: sr.w + dx, h: sr.h + dy };
      } else if (m === "nw") {
        newR = { x: sr.x + dx, y: sr.y + dy, w: sr.w - dx, h: sr.h - dy };
      } else if (m === "ne") {
        newR = { x: sr.x, y: sr.y + dy, w: sr.w + dx, h: sr.h - dy };
      } else if (m === "sw") {
        newR = { x: sr.x + dx, y: sr.y, w: sr.w - dx, h: sr.h + dy };
      } else if (m === "n") {
        newR = { x: sr.x, y: sr.y + dy, w: sr.w, h: sr.h - dy };
      } else if (m === "s") {
        newR = { x: sr.x, y: sr.y, w: sr.w, h: sr.h + dy };
      } else if (m === "w") {
        newR = { x: sr.x + dx, y: sr.y, w: sr.w - dx, h: sr.h };
      } else if (m === "e") {
        newR = { x: sr.x, y: sr.y, w: sr.w + dx, h: sr.h };
      }

      newR = enforceAspect(newR, m);
      const clamped = clampRect(newR);
      // Update ref IMMEDIATELY so onUp always sees the latest rect
      // (don't wait for React render which may not have flushed yet)
      latestRectRef.current = clamped;
      onChangeRef.current(clamped);
    };

    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (onCommitRef.current) onCommitRef.current(latestRectRef.current);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Cursor style
  const onMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { mx, my } = getCanvasPos(e);
    const mode = hitTest(mx, my);
    const cursors: Record<DragMode, string> = {
      none: "default", move: "grab",
      nw: "nw-resize", ne: "ne-resize", sw: "sw-resize", se: "se-resize",
      n: "n-resize", s: "s-resize", w: "w-resize", e: "e-resize",
    };
    canvas.style.cursor = cursors[mode] || "default";
  };

  // Compute crop overlay pixel positions (container-space). The dark
  // strips sit outside the crop rect; the blue border wraps the rect.
  const ox = cropRect.x * scale;
  const oy = cropRect.y * scale;
  const ow = cropRect.w * scale;
  const oh = cropRect.h * scale;
  // Image display size (pre-rotation) in container space.
  const imgDispW = imgNatW * scale;
  const imgDispH = imgNatH * scale;
  const rotDeg = rotation || 0;

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: (canvasH || 200) + "px",
        position: "relative",
        borderRadius: 4,
        border: "1px solid #555",
        overflow: "hidden",
        backgroundColor: "#222",
        userSelect: "none",
      }}
    >
      {/* Always-visible HTML image, rotated/flipped via CSS. This is the
          definitive source of pixels the user sees — never subject to any
          canvas-bitmap timing issue. */}
      {imageSrc && imgDispW > 0 && imgDispH > 0 && (
        <img
          src={imageSrc}
          alt=""
          draggable={false}
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            width: imgDispW + "px",
            height: imgDispH + "px",
            transform: `translate(-50%, -50%) rotate(${rotDeg}deg) scaleX(${flipH ? -1 : 1}) scaleY(${flipV ? -1 : 1})`,
            transformOrigin: "center",
            pointerEvents: "none",
            display: "block",
          }}
        />
      )}

      {/* Dark overlay strips outside the crop rect. */}
      {oh > 0 && ow > 0 && (
        <>
          {/* top */}
          <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: Math.max(0, oy) + "px", backgroundColor: "rgba(0,0,0,0.5)", pointerEvents: "none" }} />
          {/* bottom */}
          <div style={{ position: "absolute", top: (oy + oh) + "px", left: 0, width: "100%", height: Math.max(0, canvasH - oy - oh) + "px", backgroundColor: "rgba(0,0,0,0.5)", pointerEvents: "none" }} />
          {/* left */}
          <div style={{ position: "absolute", top: oy + "px", left: 0, width: Math.max(0, ox) + "px", height: oh + "px", backgroundColor: "rgba(0,0,0,0.5)", pointerEvents: "none" }} />
          {/* right */}
          <div style={{ position: "absolute", top: oy + "px", left: (ox + ow) + "px", width: Math.max(0, canvasW - ox - ow) + "px", height: oh + "px", backgroundColor: "rgba(0,0,0,0.5)", pointerEvents: "none" }} />
          {/* Crop border */}
          <div style={{ position: "absolute", top: oy + "px", left: ox + "px", width: ow + "px", height: oh + "px", border: "2px solid #2196f3", boxSizing: "border-box", pointerEvents: "none" }} />
          {/* 4 corner handles */}
          {[
            [ox - HANDLE_SIZE / 2, oy - HANDLE_SIZE / 2],
            [ox + ow - HANDLE_SIZE / 2, oy - HANDLE_SIZE / 2],
            [ox - HANDLE_SIZE / 2, oy + oh - HANDLE_SIZE / 2],
            [ox + ow - HANDLE_SIZE / 2, oy + oh - HANDLE_SIZE / 2],
          ].map(([hx, hy], i) => (
            <div key={`c-${i}`} style={{ position: "absolute", top: hy + "px", left: hx + "px", width: HANDLE_SIZE + "px", height: HANDLE_SIZE + "px", backgroundColor: "#2196f3", pointerEvents: "none" }} />
          ))}
          {/* 4 edge midpoint handles */}
          {[
            [ox + ow / 2 - 4, oy - 4],
            [ox + ow / 2 - 4, oy + oh - 4],
            [ox - 4, oy + oh / 2 - 4],
            [ox + ow - 4, oy + oh / 2 - 4],
          ].map(([hx, hy], i) => (
            <div key={`e-${i}`} style={{ position: "absolute", top: hy + "px", left: hx + "px", width: "8px", height: "8px", backgroundColor: "rgba(33,150,243,0.7)", pointerEvents: "none" }} />
          ))}
        </>
      )}

      {/* The canvas is retained as an invisible event capture layer; its
          existing drag/resize hit-testing reads mouse positions relative
          to this element so rewiring mouse handlers to work on the
          container div alone would have been an unrelated refactor. */}
      <canvas
        ref={canvasRef}
        width={canvasW}
        height={canvasH || 200}
        style={{
          position: "absolute",
          top: 0, left: 0,
          display: "block",
          width: "100%",
          height: "100%",
          opacity: 0, // invisible — HTML elements above render the overlay
        }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
      />
    </div>
  );
}

/* ================================================================
   InteractiveHistogram — Per-channel RGB histogram with draggable
   black/white point markers (like ImageJ/Fiji LUT).
   Supports Composite / R / G / B channel modes.
   ================================================================ */
const HIST_H = 80;
const SLIDER_H = 18;
const TOTAL_H = HIST_H + SLIDER_H;

type HistChannel = "composite" | "r" | "g" | "b";

interface PerChannelLevels {
  input_black_r: number; input_white_r: number;
  input_black_g: number; input_white_g: number;
  input_black_b: number; input_white_b: number;
}

interface InteractiveHistogramProps {
  imageSrc: string;
  levels: PerChannelLevels;
  onChange: (levels: PerChannelLevels) => void;
  onCommit: (levels: PerChannelLevels) => void;
}

const CHANNEL_COLORS: Record<HistChannel, { fill: string; line: string; markerB: string; markerW: string; grad0: string; grad1: string }> = {
  composite: { fill: "", line: "#aaa", markerB: "#222", markerW: "#ddd", grad0: "#000", grad1: "#fff" },
  r: { fill: "rgba(255,60,60,0.5)", line: "#f44", markerB: "#600", markerW: "#faa", grad0: "#000", grad1: "#f00" },
  g: { fill: "rgba(60,220,60,0.5)", line: "#4c4", markerB: "#060", markerW: "#afa", grad0: "#000", grad1: "#0f0" },
  b: { fill: "rgba(60,120,255,0.5)", line: "#48f", markerB: "#006", markerW: "#aaf", grad0: "#000", grad1: "#00f" },
};

function InteractiveHistogram({ imageSrc, levels, onChange, onCommit }: InteractiveHistogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasW, setCanvasW] = useState(400);
  const [channel, setChannel] = useState<HistChannel>("composite");
  const [histData, setHistData] = useState<{ r: Uint32Array; g: Uint32Array; b: Uint32Array; maxR: number; maxG: number; maxB: number } | null>(null);
  const dragRef = useRef<"none" | "black" | "white">("none");
  const latestLevelsRef = useRef(levels);
  latestLevelsRef.current = levels;

  const getBlack = (ch: HistChannel, lv: PerChannelLevels) => {
    if (ch === "r") return lv.input_black_r;
    if (ch === "g") return lv.input_black_g;
    if (ch === "b") return lv.input_black_b;
    return Math.min(lv.input_black_r, lv.input_black_g, lv.input_black_b);
  };
  const getWhite = (ch: HistChannel, lv: PerChannelLevels) => {
    if (ch === "r") return lv.input_white_r;
    if (ch === "g") return lv.input_white_g;
    if (ch === "b") return lv.input_white_b;
    return Math.max(lv.input_white_r, lv.input_white_g, lv.input_white_b);
  };

  const setBlackWhite = (ch: HistChannel, black: number, white: number, base: PerChannelLevels): PerChannelLevels => {
    const next = { ...base };
    if (ch === "r" || ch === "composite") { next.input_black_r = black; next.input_white_r = white; }
    if (ch === "g" || ch === "composite") { next.input_black_g = black; next.input_white_g = white; }
    if (ch === "b" || ch === "composite") { next.input_black_b = black; next.input_white_b = white; }
    return next;
  };

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) { const w = Math.floor(entry.contentRect.width); if (w > 0) setCanvasW(w); }
    });
    ro.observe(el);
    setCanvasW(Math.floor(el.clientWidth) || 400);
    return () => ro.disconnect();
  }, []);

  // Compute histogram data
  useEffect(() => {
    if (!imageSrc) return;
    const img = new window.Image();
    img.onload = () => {
      const off = document.createElement("canvas");
      off.width = img.naturalWidth; off.height = img.naturalHeight;
      const octx = off.getContext("2d");
      if (!octx) return;
      octx.drawImage(img, 0, 0);
      const data = octx.getImageData(0, 0, off.width, off.height).data;
      const r = new Uint32Array(256), g = new Uint32Array(256), b = new Uint32Array(256);
      for (let i = 0; i < data.length; i += 4) { r[data[i]]++; g[data[i+1]]++; b[data[i+2]]++; }
      let maxR = 1, maxG = 1, maxB = 1;
      for (let i = 1; i < 255; i++) { maxR = Math.max(maxR, r[i]); maxG = Math.max(maxG, g[i]); maxB = Math.max(maxB, b[i]); }
      setHistData({ r, g, b, maxR, maxG, maxB });
    };
    img.src = imageSrc.startsWith("data:") ? imageSrc : `data:image/png;base64,${imageSrc}`;
  }, [imageSrc]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvasW;
    canvas.width = W; canvas.height = TOTAL_H;
    const hd = histData;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, W, HIST_H);

    if (hd) {
      const drawCh = (hist: Uint32Array, maxV: number, color: string, alpha: number) => {
        ctx.fillStyle = color.replace(/[\d.]+\)$/, `${alpha})`);
        ctx.beginPath(); ctx.moveTo(0, HIST_H);
        for (let i = 0; i < 256; i++) { ctx.lineTo((i / 255) * W, HIST_H - (hist[i] / maxV) * HIST_H); }
        ctx.lineTo(W, HIST_H); ctx.closePath(); ctx.fill();
      };
      if (channel === "composite") {
        const maxAll = Math.max(hd.maxR, hd.maxG, hd.maxB);
        drawCh(hd.r, maxAll, "rgba(255,60,60,0.35)", 0.35);
        drawCh(hd.g, maxAll, "rgba(60,220,60,0.35)", 0.35);
        drawCh(hd.b, maxAll, "rgba(60,120,255,0.35)", 0.35);
      } else {
        // Dim other channels, highlight selected
        const maxAll = Math.max(hd.maxR, hd.maxG, hd.maxB);
        if (channel !== "r") drawCh(hd.r, maxAll, "rgba(255,60,60,0.12)", 0.12);
        if (channel !== "g") drawCh(hd.g, maxAll, "rgba(60,220,60,0.12)", 0.12);
        if (channel !== "b") drawCh(hd.b, maxAll, "rgba(60,120,255,0.12)", 0.12);
        const chData = channel === "r" ? hd.r : channel === "g" ? hd.g : hd.b;
        const chMax = channel === "r" ? hd.maxR : channel === "g" ? hd.maxG : hd.maxB;
        drawCh(chData, chMax, CHANNEL_COLORS[channel].fill, 0.5);
      }
    }

    // Active channel markers
    const curBlack = getBlack(channel, levels);
    const curWhite = getWhite(channel, levels);
    const bx = (curBlack / 255) * W;
    const wx = (curWhite / 255) * W;
    const cc = CHANNEL_COLORS[channel];

    // Shading
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.fillRect(0, 0, bx, HIST_H);
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillRect(wx, 0, W - wx, HIST_H);

    // Vertical lines
    ctx.strokeStyle = cc.line; ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(bx, 0); ctx.lineTo(bx, HIST_H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(wx, 0); ctx.lineTo(wx, HIST_H); ctx.stroke();
    ctx.setLineDash([]);

    // Gradient bar
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, cc.grad0); grad.addColorStop(1, cc.grad1);
    ctx.fillStyle = grad;
    ctx.fillRect(0, HIST_H, W, SLIDER_H);

    // Triangle markers
    const drawTri = (x: number, fill: string, stroke: string) => {
      const ty = HIST_H + 2, by2 = HIST_H + SLIDER_H - 1, hw = 6;
      ctx.beginPath(); ctx.moveTo(x, ty); ctx.lineTo(x - hw, by2); ctx.lineTo(x + hw, by2); ctx.closePath();
      ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke();
    };
    drawTri(bx, cc.markerB, "#888");
    drawTri(wx, cc.markerW, "#888");
  }, [canvasW, levels, imageSrc, channel, histData]);

  // Mouse interaction
  const valFromMouse = (e: MouseEvent | React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    return Math.round(Math.max(0, Math.min(255, ((e.clientX - rect.left) / rect.width) * 255)));
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const my = e.clientY - rect.top;
    if (my < HIST_H - 8) return;
    const mx = e.clientX - rect.left;
    const curB = getBlack(channel, levels);
    const curW = getWhite(channel, levels);
    const bx = (curB / 255) * rect.width;
    const wx = (curW / 255) * rect.width;
    const distB = Math.abs(mx - bx), distW = Math.abs(mx - wx);
    dragRef.current = (distB < distW && distB < 20) ? "black" : (distW < 20) ? "white" : (mx < (bx + wx) / 2) ? "black" : "white";
    e.preventDefault();

    const onMove = (ev: MouseEvent) => {
      ev.preventDefault();
      const v = valFromMouse(ev);
      const cur = latestLevelsRef.current;
      const curB2 = getBlack(channel, cur);
      const curW2 = getWhite(channel, cur);
      if (dragRef.current === "black") {
        onChange(setBlackWhite(channel, Math.min(v, curW2 - 1), curW2, cur));
      } else {
        onChange(setBlackWhite(channel, curB2, Math.max(v, curB2 + 1), cur));
      }
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      onCommit(latestLevelsRef.current);
      dragRef.current = "none";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const onMouseMove2 = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.style.cursor = (e.clientY - rect.top) >= HIST_H - 8 ? "ew-resize" : "default";
  };

  const curBlack = getBlack(channel, levels);
  const curWhite = getWhite(channel, levels);

  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
        <ToggleButtonGroup size="small" exclusive value={channel} onChange={(_, v) => { if (v) setChannel(v); }}>
          <ToggleButton value="composite" sx={{ py: 0, px: 1, fontSize: "0.65rem" }}>RGB</ToggleButton>
          <ToggleButton value="r" sx={{ py: 0, px: 1, fontSize: "0.65rem", color: "#f44" }}>R</ToggleButton>
          <ToggleButton value="g" sx={{ py: 0, px: 1, fontSize: "0.65rem", color: "#4c4" }}>G</ToggleButton>
          <ToggleButton value="b" sx={{ py: 0, px: 1, fontSize: "0.65rem", color: "#48f" }}>B</ToggleButton>
        </ToggleButtonGroup>
        <Typography variant="caption" color="text.secondary">{curBlack} – {curWhite}</Typography>
      </Box>
      <div ref={containerRef} style={{ width: "100%", height: TOTAL_H, flexShrink: 0 }}>
        <canvas
          ref={canvasRef}
          width={canvasW}
          height={TOTAL_H}
          style={{ width: "100%", height: TOTAL_H, borderRadius: 4, border: "1px solid #333", display: "block" }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove2}
        />
      </div>
    </Box>
  );
}

/* ================================================================
   AdjustSlider — reusable slider row with editable numeric input
   ================================================================ */
interface AdjustSliderProps {
  label: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  marks?: { value: number }[];
  onChange: (v: number) => void;
  suffix?: string;
  format?: (v: number) => string;
}

function AdjustSlider({ label, value, defaultValue, min, max, step, marks, onChange, suffix, format }: AdjustSliderProps) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
      <Typography
        variant="caption"
        sx={{ width: 80, minWidth: 80, flexShrink: 0, cursor: "pointer", fontSize: "0.7rem" }}
        onDoubleClick={() => onChange(defaultValue)}
      >
        {label}
      </Typography>
      <Slider
        size="small"
        sx={{ mx: 0.5 }}
        value={value}
        min={min}
        max={max}
        step={step}
        marks={marks}
        onChange={(_, v) => onChange(v as number)}
      />
      <TextField
        type="number"
        size="small"
        value={format ? format(value) : value}
        onChange={(e) => {
          let v = Number(e.target.value);
          v = Math.max(min, Math.min(max, v));
          onChange(v);
        }}
        inputProps={{ min, max, step }}
        sx={{ width: 68, flexShrink: 0, "& input": { textAlign: "center", py: 0.5, px: 1, fontSize: "0.7rem" } }}
      />
      <Typography variant="caption" sx={{ flexShrink: 0, width: 16, fontSize: "0.65rem" }}>{suffix || ""}</Typography>
    </Box>
  );
}

/* ================================================================
   Crop Pixel Fields — controlled component with Enter-to-commit
   ================================================================ */

function CropPixelFields({ cropRect, imgNatW, imgNatH, origFullW, origFullH, displayRotation, aspectPreset, customRatioStr, onCommit }: {
  cropRect: { x: number; y: number; w: number; h: number };
  imgNatW: number; imgNatH: number;
  origFullW: number; origFullH: number;
  displayRotation: number;
  aspectPreset: string;
  customRatioStr: string;
  onCommit: (rect: { x: number; y: number; w: number; h: number }) => void;
}) {
  const scaleW = origFullW > 0 && imgNatW > 0 ? origFullW / imgNatW : 1;
  const scaleH = origFullH > 0 && imgNatH > 0 ? origFullH / imgNatH : 1;
  // Defensively coerce any non-finite values (NaN / Infinity / undefined)
  // back to 0 so the pixel fields never render as blank boxes with a
  // "Position: NaN, NaN" footer underneath.
  const safeCropW = Number.isFinite(cropRect.w) ? cropRect.w : 0;
  const safeCropH = Number.isFinite(cropRect.h) ? cropRect.h : 0;
  const fullW = Math.max(0, Math.round(safeCropW * scaleW));
  const fullH = Math.max(0, Math.round(safeCropH * scaleH));

  const [editW, setEditW] = useState(String(fullW));
  const [editH, setEditH] = useState(String(fullH));
  const justCommittedRef = useRef(false);

  // Sync local state when cropRect changes externally (e.g. canvas drag)
  // but NOT right after a manual commit (which already set the correct values)
  useEffect(() => {
    if (justCommittedRef.current) {
      justCommittedRef.current = false;
      return;
    }
    setEditW(String(Math.round(cropRect.w * scaleW)));
    setEditH(String(Math.round(cropRect.h * scaleH)));
  }, [cropRect.w, cropRect.h, scaleW, scaleH]);

  const commit = (changedAxis: "w" | "h") => {
    const { rotW, rotH } = computeRotatedDims(imgNatW, imgNatH, displayRotation);
    const typedW = Math.max(10, Number(editW) || fullW);
    const typedH = Math.max(10, Number(editH) || fullH);

    // Work in full-res space for aspect ratio to avoid rounding issues
    let finalFullW = typedW;
    let finalFullH = typedH;

    // Enforce aspect ratio in full-res space
    const ratio = aspectPreset === "Custom" ? parseRatioStr(customRatioStr)
      : aspectPreset === "1:1" ? 1 : aspectPreset === "4:3" ? 4 / 3 : aspectPreset === "16:9" ? 16 / 9 : null;
    if (ratio && aspectPreset !== "Free") {
      if (changedAxis === "w") {
        finalFullH = Math.round(finalFullW / ratio);
      } else {
        finalFullW = Math.round(finalFullH * ratio);
      }
    }

    // Clamp to rotated image bounds (in full-res)
    const maxFullW = Math.round(rotW * scaleW);
    const maxFullH = Math.round(rotH * scaleH);
    finalFullW = Math.min(finalFullW, maxFullW);
    finalFullH = Math.min(finalFullH, maxFullH);

    // Convert to thumbnail coordinates for the crop rect
    const thumbW = Math.max(1, Math.round(finalFullW / scaleW));
    const thumbH = Math.max(1, Math.round(finalFullH / scaleH));

    const newRect = {
      x: Math.max(0, Math.min(cropRect.x, rotW - thumbW)),
      y: Math.max(0, Math.min(cropRect.y, rotH - thumbH)),
      w: thumbW, h: thumbH,
    };

    // Update display with the full-res values (not round-tripped)
    justCommittedRef.current = true;
    setEditW(String(finalFullW));
    setEditH(String(finalFullH));
    onCommit(newRect);
  };

  return (
    <>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Typography variant="caption" sx={{ flexShrink: 0 }}>Crop</Typography>
        <TextField
          type="number"
          size="small"
          label="W"
          value={editW}
          onChange={(e) => setEditW(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") commit("w"); }}
          inputProps={{ min: 10 }}
          sx={{ flex: 1, "& input": { py: 0.5, fontSize: "0.75rem" } }}
        />
        <Typography variant="caption">&times;</Typography>
        <TextField
          type="number"
          size="small"
          label="H"
          value={editH}
          onChange={(e) => setEditH(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") commit("h"); }}
          inputProps={{ min: 10 }}
          sx={{ flex: 1, "& input": { py: 0.5, fontSize: "0.75rem" } }}
        />
        <Typography variant="caption" sx={{ flexShrink: 0 }}>px</Typography>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ mt: -0.5, fontStyle: "italic" }}>
        Press Enter to apply
      </Typography>
    </>
  );
}

/* ================================================================
   Main dialog
   ================================================================ */
// ── Video Frame Selector Component ──────────────────────────────────────
function VideoFrameSelector({
  imageName, onFrameChange, onFrameImage, frame, setFrame,
  frameStart, setFrameStart, frameEnd, setFrameEnd,
  playRange, setPlayRange,
  returnToSelectedOnEnd, setReturnToSelectedOnEnd,
}: {
  imageName: string;
  onFrameChange: () => void;
  onFrameImage?: (b64: string) => void;
  frame: number;
  setFrame: (f: number) => void;
  frameStart: number;
  setFrameStart: (f: number) => void;
  frameEnd: number;
  setFrameEnd: (f: number) => void;
  playRange: boolean;
  setPlayRange: (b: boolean) => void;
  returnToSelectedOnEnd: boolean;
  setReturnToSelectedOnEnd: (b: boolean) => void;
}) {
  const [videoInfo, setVideoInfo] = useState<{ frame_count: number; fps: number; duration_sec: number } | null>(null);
  const currentFrame = frame;
  const setCurrentFrame = setFrame;
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(false);
  const frameRef = useRef(0);
  // Local thumb values for the dual-range slider (separate from
  // committed frameStart/frameEnd so dragging doesn't seek per-pixel).
  const [rangeDraft, setRangeDraft] = useState<[number, number]>([frameStart, frameEnd]);

  useEffect(() => {
    api.getVideoInfo(imageName).then(info => {
      setVideoInfo(info);
      // Initialise sensible range defaults on first load if model still
      // shows the legacy 0/0 (= "not yet configured"): use full clip.
      if (info.frame_count > 1 && frameStart === 0 && frameEnd === 0) {
        setFrameStart(0);
        setFrameEnd(info.frame_count - 1);
        setRangeDraft([0, info.frame_count - 1]);
      } else {
        setRangeDraft([frameStart, frameEnd]);
      }
    }).catch(() => {});
  }, [imageName]); // eslint-disable-line

  const seekToFrame = async (frame: number) => {
    setLoading(true);
    try {
      await api.getVideoFrame(imageName, frame);
      setCurrentFrame(frame);
      frameRef.current = frame;
      onFrameChange();
    } catch { /* ignore */ }
    setLoading(false);
  };

  // Play loop — waits for each frame to render before advancing
  useEffect(() => {
    if (!playing || !videoInfo) return;
    playRef.current = true;
    let cancelled = false;
    const advance = async () => {
      while (playRef.current && !cancelled) {
        const next = frameRef.current + 1;
        if (next >= videoInfo.frame_count) {
          setPlaying(false); playRef.current = false; return;
        }
        try {
          const resp = await api.getVideoFrame(imageName, next);
          if (cancelled) return;
          frameRef.current = next;
          setCurrentFrame(next);
          if (onFrameImage && resp.thumbnail) onFrameImage(resp.thumbnail);
          await new Promise(r => setTimeout(r, 30));
        } catch { break; }
      }
    };
    advance();
    return () => { cancelled = true; playRef.current = false; };
  }, [playing, videoInfo, imageName]); // eslint-disable-line

  if (!videoInfo) return <Typography variant="caption" sx={{ color: "text.secondary" }}>Loading video info...</Typography>;

  const timeStr = videoInfo.fps > 0 ? `${(currentFrame / videoInfo.fps).toFixed(2)}s` : "";
  const maxFrame = Math.max(0, videoInfo.frame_count - 1);
  const rangeLen = Math.max(1, rangeDraft[1] - rangeDraft[0] + 1);
  const rangeDur = videoInfo.fps > 0 ? `${(rangeLen / videoInfo.fps).toFixed(2)}s` : "";

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mt: 1 }}>
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Typography variant="caption" sx={{ flexShrink: 0, fontSize: "0.65rem" }}>
          Frame {currentFrame} / {maxFrame} {timeStr && `(${timeStr})`}
        </Typography>
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>
          {videoInfo.fps.toFixed(1)} fps, {videoInfo.duration_sec.toFixed(1)}s
        </Typography>
      </Box>
      <Slider
        value={currentFrame} min={0} max={maxFrame} step={1}
        onChange={(_, v) => { setCurrentFrame(v as number); setPlaying(false); }}
        onChangeCommitted={(_, v) => seekToFrame(v as number)}
        sx={{ mt: 0, mx: 1 }}
      />
      <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
        <Button size="small" variant="outlined" onClick={() => seekToFrame(Math.max(0, currentFrame - 1))}
          disabled={loading || playing || currentFrame <= 0} sx={{ minWidth: 28, px: 0.5, fontSize: "0.75rem" }}>⏮</Button>
        <Button size="small" variant={playing ? "contained" : "outlined"} color={playing ? "error" : "primary"}
          onClick={() => setPlaying(!playing)}
          sx={{ minWidth: 36, px: 0.5, fontSize: "0.85rem" }}>
          {playing ? "⏸" : "▶"}
        </Button>
        <Button size="small" variant="outlined" onClick={() => seekToFrame(Math.min(maxFrame, currentFrame + 1))}
          disabled={loading || playing || currentFrame >= maxFrame} sx={{ minWidth: 28, px: 0.5, fontSize: "0.75rem" }}>⏭</Button>
        <TextField type="number" value={currentFrame}
          onChange={(e) => setCurrentFrame(Number(e.target.value))}
          onKeyDown={(e) => { if (e.key === "Enter") seekToFrame(currentFrame); }}
          size="small" inputProps={{ min: 0, max: maxFrame }}
          sx={{ width: 70, "& input": { fontSize: "0.7rem", px: 0.75, py: 0.4, textAlign: "center" } }}
        />
      </Box>

      {/* ── Video export range ─────────────────────────── */}
      <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: "divider" }}>
        <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 0.5 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={playRange}
                onChange={(e) => setPlayRange(e.target.checked)}
                size="small"
                sx={{ p: 0.5 }}
              />
            }
            label={
              <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>
                Play range in video export
              </Typography>
            }
            sx={{ ml: 0, mr: 1 }}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={returnToSelectedOnEnd}
                disabled={!playRange}
                onChange={(e) => setReturnToSelectedOnEnd(e.target.checked)}
                size="small"
                sx={{ p: 0.5 }}
              />
            }
            label={
              <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>
                Return to selected frame on end
              </Typography>
            }
            title="When this panel's range ends before others finish, snap back to the static-selected frame instead of holding on the last range frame."
            sx={{ ml: 0, mr: 1 }}
          />
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>
            {`${rangeDraft[0]}–${rangeDraft[1]} (${rangeLen} frames${rangeDur ? `, ${rangeDur}` : ""})`}
          </Typography>
        </Box>
        <Slider
          value={rangeDraft}
          min={0}
          max={maxFrame}
          step={1}
          onChange={(_, v) => {
            const arr = v as number[];
            setRangeDraft([Math.min(arr[0], arr[1]), Math.max(arr[0], arr[1])]);
          }}
          onChangeCommitted={(_, v) => {
            const arr = v as number[];
            const lo = Math.min(arr[0], arr[1]);
            const hi = Math.max(arr[0], arr[1]);
            setFrameStart(lo);
            setFrameEnd(hi);
          }}
          disableSwap
          sx={{ mt: 0, mx: 1 }}
        />
      </Box>
    </Box>
  );
}


// ── Z-Stack TIFF Frame Selector Component ─────────────────────────────
function ZStackFrameSelector({ imageName, onFrameChange, onFrameImage, frame, setFrame, panelRow, panelCol, onAppliedToPanel }: { imageName: string; onFrameChange: () => void; onFrameImage?: (b64: string) => void; frame: number; setFrame: (f: number) => void; panelRow?: number; panelCol?: number; onAppliedToPanel?: () => void }) {
  const [info, setInfo] = useState<{ frame_count: number; width: number; height: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [projRange, setProjRange] = useState<[number, number]>([0, 0]);
  const [projMethod, setProjMethod] = useState<"max" | "avg" | "min">("max");
  const [projecting, setProjecting] = useState(false);
  const [volumeOpen, setVolumeOpen] = useState(false);

  useEffect(() => {
    api.getZStackInfo(imageName).then(i => {
      setInfo(i);
      setProjRange([0, i.frame_count - 1]);
    }).catch(() => setInfo(null));
  }, [imageName]);

  const seekToFrame = async (f: number) => {
    setLoading(true);
    try {
      const resp = await api.getZStackFrame(imageName, f, panelRow, panelCol);
      setFrame(f);
      if (onFrameImage && resp.thumbnail) onFrameImage(resp.thumbnail);
      onFrameChange();
    } catch { /* ignore */ }
    setLoading(false);
  };

  const applyProjection = async () => {
    setProjecting(true);
    try {
      const resp = await api.projectZStack(imageName, projRange[0], projRange[1], projMethod);
      if (onFrameImage && resp.thumbnail) onFrameImage(resp.thumbnail);
      onFrameChange();
    } catch (e) {
      console.error("Projection failed:", e);
    }
    setProjecting(false);
  };

  if (!info) return <Typography variant="caption" sx={{ color: "text.secondary" }}>Loading z-stack info...</Typography>;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, mt: 1 }}>
      {/* Single slice selection */}
      <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.7rem" }}>Single Slice</Typography>
      <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>
        Slice {frame} / {info.frame_count - 1}
      </Typography>
      <Slider
        value={frame} min={0} max={Math.max(0, info.frame_count - 1)} step={1}
        onChange={(_, v) => setFrame(v as number)}
        onChangeCommitted={(_, v) => seekToFrame(v as number)}
        sx={{ mt: 0, mx: 1 }}
      />
      <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
        <Button size="small" variant="outlined" onClick={() => seekToFrame(Math.max(0, frame - 1))}
          disabled={loading || frame <= 0} sx={{ minWidth: 28, px: 0.5, fontSize: "0.75rem" }}>◀</Button>
        <Button size="small" variant="outlined" onClick={() => seekToFrame(Math.min(info.frame_count - 1, frame + 1))}
          disabled={loading || frame >= info.frame_count - 1} sx={{ minWidth: 28, px: 0.5, fontSize: "0.75rem" }}>▶</Button>
        <TextField type="number" value={frame}
          onChange={(e) => setFrame(Number(e.target.value))}
          onKeyDown={(e) => { if (e.key === "Enter") seekToFrame(frame); }}
          size="small" inputProps={{ min: 0, max: info.frame_count - 1 }}
          sx={{ width: 70, "& input": { fontSize: "0.7rem", px: 0.75, py: 0.4, textAlign: "center" } }}
        />
      </Box>

      <Divider sx={{ my: 0.5 }} />

      {/* Projection */}
      <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.7rem" }}>Projection</Typography>
      <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>
        Range: slice {projRange[0]} – {projRange[1]} ({projRange[1] - projRange[0] + 1} slices)
      </Typography>
      <Slider
        value={projRange}
        min={0}
        max={Math.max(0, info.frame_count - 1)}
        step={1}
        onChange={(_, v) => setProjRange(v as [number, number])}
        valueLabelDisplay="auto"
        sx={{ mx: 1 }}
      />
      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Select
          size="small"
          value={projMethod}
          onChange={(e) => setProjMethod(e.target.value as "max" | "avg" | "min")}
          sx={{ fontSize: "0.65rem", minWidth: 120, "& .MuiSelect-select": { py: 0.3, px: 1 } }}
        >
          <MenuItem value="max" sx={{ fontSize: "0.65rem" }}>Maximum Intensity</MenuItem>
          <MenuItem value="avg" sx={{ fontSize: "0.65rem" }}>Average Intensity</MenuItem>
          <MenuItem value="min" sx={{ fontSize: "0.65rem" }}>Minimum Intensity</MenuItem>
        </Select>
        <Button
          size="small"
          variant="contained"
          disabled={projecting}
          onClick={applyProjection}
          sx={{ fontSize: "0.6rem", textTransform: "none" }}
        >
          {projecting ? "Projecting..." : "Apply"}
        </Button>
      </Box>

      <Divider sx={{ my: 0.5 }} />

      {/* 3D View */}
      <Button
        size="small"
        variant="outlined"
        onClick={() => setVolumeOpen(true)}
        sx={{ fontSize: "0.6rem", textTransform: "none" }}
      >
        🔬 3D Volume View
      </Button>

      {volumeOpen && (
        <VolumeViewerDialog
          open={volumeOpen}
          onClose={() => setVolumeOpen(false)}
          imageName={imageName}
          startFrame={projRange[0]}
          endFrame={projRange[1]}
          panelRow={panelRow}
          panelCol={panelCol}
          onAppliedToPanel={onAppliedToPanel}
        />
      )}
    </Box>
  );
}


// Persist last-used tab across dialog opens
let _lastTabIdx = 0;

// Cache of which image names are z-stacks (populated lazily)
const _zstackCache = new Map<string, number>(); // name → frame_count (1 = not z-stack)

export function EditPanelDialog({ open, onClose, row, col }: Props) {
  const config = useFigureStore((s) => s.config);
  const updatePanel = useFigureStore((s) => s.updatePanel);
  const fonts = useFigureStore((s) => s.fonts);
  const loadedImages = useFigureStore((s) => s.loadedImages);

  const [tabIdx, _setTabIdx] = useState(_lastTabIdx);
  const setTabIdx = (v: number) => { _lastTabIdx = v; _setTabIdx(v); };
  const [local, setLocal] = useState<PanelInfo | null>(null);

  // Tab-ownership types (used by the per-tab undo/redo block further below,
  // which needs `tabIdx + tOff` from after the z-stack / video tab has been
  // resolved, so the actual history setup is deferred until after TAB_*
  // constants are in scope).
  type TabKey = "crop" | "adj" | "labels" | "scale" | "annot" | "zoom";
  const TAB_FIELDS: Record<TabKey, (keyof PanelInfo)[]> = {
    crop: [
      "rotation", "flip_horizontal", "flip_vertical",
      "crop", "crop_image", "aspect_ratio_str",
      "final_resize", "final_width", "final_height",
    ] as unknown as (keyof PanelInfo)[],
    adj: [
      "brightness", "contrast", "hue", "saturation", "gamma",
      "exposure", "vibrance", "color_temperature", "tint",
      "sharpen", "blur", "denoise",
      "highlights", "shadows", "midtones",
      "input_black_r", "input_white_r",
      "input_black_g", "input_white_g",
      "input_black_b", "input_white_b",
      "invert", "grayscale", "pseudocolor",
    ] as unknown as (keyof PanelInfo)[],
    labels: ["labels"] as unknown as (keyof PanelInfo)[],
    scale: ["scale_bar", "add_scale_bar"] as unknown as (keyof PanelInfo)[],
    annot: ["symbols", "lines", "areas"] as unknown as (keyof PanelInfo)[],
    zoom: ["zoom_inset", "add_zoom_inset"] as unknown as (keyof PanelInfo)[],
  };
  const historyRef = useRef<Record<TabKey, { stack: Partial<PanelInfo>[]; idx: number }>>({
    crop: { stack: [], idx: -1 },
    adj: { stack: [], idx: -1 },
    labels: { stack: [], idx: -1 },
    scale: { stack: [], idx: -1 },
    annot: { stack: [], idx: -1 },
    zoom: { stack: [], idx: -1 },
  });
  const historyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);
  const isRestoringRef = useRef(false);
  const extractTabFields = (panelState: PanelInfo, key: TabKey): Partial<PanelInfo> => {
    const fields = TAB_FIELDS[key];
    const out: Record<string, unknown> = {};
    for (const f of fields) out[f as string] = (panelState as unknown as Record<string, unknown>)[f as string];
    return out as Partial<PanelInfo>;
  };

  // Detect if current panel has a video file
  const VIDEO_EXTS = ['.mp4','.avi','.mov','.mkv','.webm','.wmv','.flv','.m4v','.mpg','.mpeg','.3gp','.ts','.mts'];
  const isVideoPanel = VIDEO_EXTS.some(ext => (local?.image_name || "").toLowerCase().endsWith(ext));

  // Detect if current panel has a z-stack TIFF (multi-frame)
  const [isZStackPanel, setIsZStackPanel] = useState(false);
  useEffect(() => {
    const name = local?.image_name || "";
    if (!name || !(/\.(tif|tiff)$/i.test(name))) {
      setIsZStackPanel(false);
      return;
    }
    // Check cache first
    if (_zstackCache.has(name)) {
      setIsZStackPanel((_zstackCache.get(name) ?? 1) > 1);
      return;
    }
    api.listZStacks().then(resp => {
      const isZ = resp.zstacks.includes(name);
      _zstackCache.set(name, isZ ? 2 : 1); // 2 = is zstack, 1 = not
      setIsZStackPanel(isZ);
    }).catch(() => setIsZStackPanel(false));
  }, [local?.image_name]);

  // Tab offset: video OR z-stack panels have an extra tab at index 0
  const hasFrameTab = isVideoPanel || isZStackPanel;
  const tOff = hasFrameTab ? 1 : 0;
  // Logical tab indices (adjusted for video offset)
  const TAB_CROP = 0 + tOff;
  const TAB_ADJ = 1 + tOff;
  const TAB_LABELS = 2 + tOff;
  const TAB_SCALE = 3 + tOff;
  const TAB_ANNOT = 4 + tOff;
  const TAB_ZOOM = 5 + tOff;
  const OVERLAY_TABS = [TAB_ADJ, TAB_LABELS, TAB_SCALE, TAB_ANNOT, TAB_ZOOM];

  // ── Per-tab undo / redo ──────────────────────────────────────
  // Each tab has its own history stack; undo/redo only affects fields
  // owned by the currently-active tab, leaving other tabs untouched.
  const currentTabKey = (): TabKey => {
    if (tabIdx === TAB_CROP) return "crop";
    if (tabIdx === TAB_ADJ) return "adj";
    if (tabIdx === TAB_LABELS) return "labels";
    if (tabIdx === TAB_SCALE) return "scale";
    if (tabIdx === TAB_ANNOT) return "annot";
    if (tabIdx === TAB_ZOOM) return "zoom";
    return "crop";
  };
  const pushHistory = useCallback((snapshot: PanelInfo) => {
    if (isRestoringRef.current) return;
    const key = currentTabKey();
    if (historyTimerRef.current) clearTimeout(historyTimerRef.current);
    historyTimerRef.current = setTimeout(() => {
      const h = historyRef.current[key];
      const fields = extractTabFields(snapshot, key);
      const serialized = JSON.stringify(fields);
      const last = h.stack[h.idx];
      if (last && JSON.stringify(last) === serialized) return;
      h.stack = h.stack.slice(0, h.idx + 1);
      h.stack.push(JSON.parse(serialized));
      const MAX = 60;
      if (h.stack.length > MAX) h.stack.splice(0, h.stack.length - MAX);
      h.idx = h.stack.length - 1;
      setHistoryVersion((v) => v + 1);
    }, 350);
  }, [tabIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyPartialSnapshot = useCallback((partial: Partial<PanelInfo>) => {
    isRestoringRef.current = true;
    setLocal((prev) => {
      if (!prev) return prev;
      const next: PanelInfo = { ...prev, ...partial } as PanelInfo;
      localRef.current = next;
      if ("rotation" in partial) {
        const rot = (partial.rotation as number | undefined) ?? 0;
        setDisplayRotation(rot > 180 ? rot - 360 : rot);
      }
      if ("aspect_ratio_str" in partial) {
        const preset = ratioStrToPreset((partial.aspect_ratio_str as string) ?? "");
        setAspectPreset(preset);
        if (preset === "Custom" && partial.aspect_ratio_str) setCustomRatioStr(partial.aspect_ratio_str as string);
      }
      return next;
    });
    setTimeout(() => { isRestoringRef.current = false; }, 50);
    setTimeout(() => { refreshPreview(); }, 0);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const doUndo = useCallback(() => {
    const key = currentTabKey();
    const h = historyRef.current[key];
    if (h.idx <= 0) return;
    h.idx -= 1;
    setHistoryVersion((v) => v + 1);
    applyPartialSnapshot(h.stack[h.idx]);
  }, [applyPartialSnapshot, tabIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  const doRedo = useCallback(() => {
    const key = currentTabKey();
    const h = historyRef.current[key];
    if (h.idx >= h.stack.length - 1) return;
    h.idx += 1;
    setHistoryVersion((v) => v + 1);
    applyPartialSnapshot(h.stack[h.idx]);
  }, [applyPartialSnapshot, tabIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeHistory = (() => {
    const key: TabKey =
      tabIdx === TAB_CROP ? "crop" :
      tabIdx === TAB_ADJ ? "adj" :
      tabIdx === TAB_LABELS ? "labels" :
      tabIdx === TAB_SCALE ? "scale" :
      tabIdx === TAB_ANNOT ? "annot" :
      tabIdx === TAB_ZOOM ? "zoom" : "crop";
    return historyRef.current[key];
  })();
  const canUndo = activeHistory.idx > 0;
  const canRedo = activeHistory.idx < activeHistory.stack.length - 1;
  void historyVersion;

  const [previewB64, setPreviewB64] = useState<string>("");
  // Natural dimensions of the rendered preview (post-crop / -process).
  // Drives the preview wrapper's aspect-ratio so the image consistently
  // fills the available pane regardless of how small the cropped output
  // is. Without this, a 200×150 cropped result rendered as a 200px-wide
  // image inside a 1000px pane — looking unhelpfully tiny.
  const [previewNatW, setPreviewNatW] = useState(0);
  const [previewNatH, setPreviewNatH] = useState(0);
  const [processedW, setProcessedW] = useState(0);
  const [processedH, setProcessedH] = useState(0);
  const [renderedPreviewB64, setRenderedPreviewB64] = useState<string>("");
  const renderedPreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [selectedLabelIdx, setSelectedLabelIdx] = useState<number>(-1);
  const [extImageThumb, setExtImageThumb] = useState<string>("");
  const extImageNameRef = useRef<string>("");
  const [extImageDims, setExtImageDims] = useState<{ w: number; h: number }>({ w: 1000, h: 1000 });
  const [videoFrame, setVideoFrame] = useState(0);
  const [videoFrameStart, setVideoFrameStart] = useState(0);
  const [videoFrameEnd, setVideoFrameEnd] = useState(0);
  const [videoPlayRange, setVideoPlayRange] = useState(false);
  const [videoReturnToSelectedOnEnd, setVideoReturnToSelectedOnEnd] = useState(false);
  // Result dialog for the "Sync seek to row/column" actions. Replaces a
  // raw window.alert() with the same MUI Dialog style used by the
  // toolbar's "New" confirmation, so the popup matches the app theme.
  const [syncSeekResult, setSyncSeekResult] = useState<{ title: string; message: string } | null>(null);
  const [selectedAnnotIdx, setSelectedAnnotIdx] = useState<{ type: "symbol" | "line" | "area"; idx: number } | null>(null);
  const [magicWandLoading, setMagicWandLoading] = useState(false);
  const previewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Crop canvas state
  const [origImgSrc, setOrigImgSrc] = useState<string>("");
  const [imgNatW, setImgNatW] = useState(0);       // thumbnail (unrotated) width
  const [imgNatH, setImgNatH] = useState(0);       // thumbnail (unrotated) height
  const [origFullW, setOrigFullW] = useState(0);    // original full-res width
  const [origFullH, setOrigFullH] = useState(0);    // original full-res height
  const [cropRect, setCropRect] = useState({ x: 0, y: 0, w: 100, h: 100 });
  const [aspectPreset, setAspectPreset] = useState<AspectPreset>("Free");
  const [customRatioStr, setCustomRatioStr] = useState("3:2");
  const [customW, setCustomW] = useState(0);
  const [customH, setCustomH] = useState(0);
  const [displayRotation, setDisplayRotation] = useState(0);  // fast visual rotation (no preview refresh)
  const rotationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previewImgRef = useRef<HTMLImageElement | null>(null);  // loaded thumbnail for client-side preview

  const panel = config?.panels[row]?.[col];

  useEffect(() => {
    if (panel && open) {
      const p = JSON.parse(JSON.stringify(panel)) as PanelInfo;
      // Ensure new fields exist (for configs saved before these were added)
      if (p.rotation === undefined) p.rotation = 0;
      if (p.flip_horizontal === undefined) p.flip_horizontal = false;
      if (p.flip_vertical === undefined) p.flip_vertical = false;
      if (p.saturation === undefined) p.saturation = 1.0;
      if (p.gamma === undefined) p.gamma = 1.0;
      if (p.color_temperature === undefined) p.color_temperature = 0;
      if (p.tint === undefined) p.tint = 0;
      if (p.sharpen === undefined) p.sharpen = 0;
      if (p.blur === undefined) p.blur = 0;
      if (p.denoise === undefined) p.denoise = 0;
      if (p.exposure === undefined) p.exposure = 0;
      if (p.vibrance === undefined) p.vibrance = 0;
      if (p.highlights === undefined) p.highlights = 0;
      if (p.shadows === undefined) p.shadows = 0;
      if (p.invert === undefined) p.invert = false;
      if (p.grayscale === undefined) p.grayscale = false;
      if ((p as unknown as Record<string, unknown>).midtones === undefined) (p as unknown as Record<string, unknown>).midtones = 0;
      const pu = p as unknown as Record<string, unknown>;
      if (pu.input_black_r === undefined) pu.input_black_r = 0;
      if (pu.input_white_r === undefined) pu.input_white_r = 255;
      if (pu.input_black_g === undefined) pu.input_black_g = 0;
      if (pu.input_white_g === undefined) pu.input_white_g = 255;
      if (pu.input_black_b === undefined) pu.input_black_b = 0;
      if (pu.input_white_b === undefined) pu.input_white_b = 255;
      // Migrate old single input_black/input_white to per-channel
      if (pu.input_black !== undefined) { pu.input_black_r = pu.input_black; pu.input_black_g = pu.input_black; pu.input_black_b = pu.input_black; delete pu.input_black; }
      if (pu.input_white !== undefined) { pu.input_white_r = pu.input_white; pu.input_white_g = pu.input_white; pu.input_white_b = pu.input_white; delete pu.input_white; }
      setLocal(p);
      // Hydrate the video range / play-range fields from the model.
      // Defaults preserve old behaviour for legacy projects.
      setVideoFrame(p.frame ?? 0);
      setVideoFrameStart(p.frame_start ?? 0);
      setVideoFrameEnd(p.frame_end ?? 0);
      setVideoPlayRange(p.play_range ?? false);
      setVideoReturnToSelectedOnEnd(p.return_to_selected_on_end ?? false);
      // Reset per-tab undo/redo history to just the initial snapshot of
      // each tab's fields so each panel edit session starts clean.
      const fresh: Record<TabKey, { stack: Partial<PanelInfo>[]; idx: number }> = {
        crop: { stack: [extractTabFields(p, "crop")], idx: 0 },
        adj: { stack: [extractTabFields(p, "adj")], idx: 0 },
        labels: { stack: [extractTabFields(p, "labels")], idx: 0 },
        scale: { stack: [extractTabFields(p, "scale")], idx: 0 },
        annot: { stack: [extractTabFields(p, "annot")], idx: 0 },
        zoom: { stack: [extractTabFields(p, "zoom")], idx: 0 },
      };
      historyRef.current = fresh;
      setHistoryVersion((v) => v + 1);
      // Restore last-used tab (don't reset to 0)
      _setTabIdx(_lastTabIdx);
      setPreviewB64("");
      const preset = ratioStrToPreset(p.aspect_ratio_str);
      setAspectPreset(preset);
      if (preset === "Custom" && p.aspect_ratio_str) setCustomRatioStr(p.aspect_ratio_str);
      // Convert stored 0-360 to -180..180 for display
      const rot = p.rotation ?? 0;
      setDisplayRotation(rot > 180 ? rot - 360 : rot);
    }
  }, [panel, open]);

  // Hook undo/redo into every local-state mutation. We watch `local` itself
  // (not the updateLocal fn) so direct setLocal calls and slider updates
  // are all captured via the same debounced push.
  useEffect(() => {
    if (local && open) pushHistory(local);
  }, [local, open, pushHistory]);

  // Keyboard shortcuts: Cmd/Ctrl+Z = undo, Cmd/Ctrl+Shift+Z or Cmd/Ctrl+Y = redo.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (!meta) return;
      const k = e.key.toLowerCase();
      const target = e.target as HTMLElement | null;
      // Don't hijack while the user is typing in a text input/textarea.
      const isTyping = !!target && ["INPUT", "TEXTAREA"].includes(target.tagName) && !target.getAttribute("readonly");
      if (isTyping) return;
      if (k === "z" && !e.shiftKey) { e.preventDefault(); doUndo(); }
      else if ((k === "z" && e.shiftKey) || k === "y") { e.preventDefault(); doRedo(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, doUndo, doRedo]);

  // Fetch preview image on open — use panel from store (not local which may not be set yet)
  const panelImageName = panel?.image_name ?? "";
  useEffect(() => {
    if (open && panelImageName) {
      // Small delay to ensure the dialog has rendered
      const t = setTimeout(() => {
        api.getPanelPreview(row, col).then((resp) => {
          if (resp.image) { setPreviewB64(resp.image); if ((resp as any).processed_width) { setProcessedW((resp as any).processed_width); setProcessedH((resp as any).processed_height || 0); } }
        }).catch(() => {});
      }, 100);
      return () => clearTimeout(t);
    }
  }, [open, row, col, panelImageName]);

  // Load original image for the crop canvas (thumbnail) + fetch original dimensions.
  // Runs on every dialog-open transition. When the dialog is closed we
  // also reset the crop-related state back to defaults so reopening goes
  // through a clean init path (otherwise stale imgNatW / cropRect values
  // from a previous session could linger and cause "overlay disappears
  // after Apply then reopen" symptoms).
  useEffect(() => {
    if (!open) {
      // Reset on close so the next open is fresh.
      setImgNatW(0);
      setImgNatH(0);
      setOrigFullW(0);
      setOrigFullH(0);
      setOrigImgSrc("");
      setCropRect({ x: 0, y: 0, w: 100, h: 100 });
      return;
    }
    const panelImageName = panel?.image_name;
    if (!panelImageName) return;

    const thumbObj = loadedImages[panelImageName];
    if (!thumbObj?.thumbnailB64) return;

    const thumb = thumbObj.thumbnailB64;
    const src = thumb.startsWith("data:") ? thumb : `data:image/png;base64,${thumb}`;

    // Read saved crop and rotation from the STORE panel
    const savedCrop = panel?.crop ?? null;
    const savedRotation = panel?.rotation ?? 0;

    // Fetch original dimensions, then load thumbnail to restore saved crop
    const infoPromise = api.getImageInfo(panelImageName).catch(() => null);

    const img = new window.Image();
    img.onload = async () => {
      const natW = img.naturalWidth;
      const natH = img.naturalHeight;
      setImgNatW(natW);
      setImgNatH(natH);

      const rot = savedRotation;
      const { rotW, rotH } = computeRotatedDims(natW, natH, rot);

      // Sanitizer — guard every coordinate so a malformed saved crop (null,
      // undefined or NaN values from an older save) can never propagate
      // into cropRect as NaN (which then shows up as "Position: NaN, NaN"
      // and blank W/H boxes in the pixel fields).
      const finite = (n: unknown): number => {
        const v = typeof n === "number" ? n : Number(n);
        return Number.isFinite(v) ? v : 0;
      };

      const info = await infoPromise;
      const fullW = info ? finite(info.width) : 0;
      const fullH = info ? finite(info.height) : 0;
      if (info) {
        setOrigFullW(fullW);
        setOrigFullH(fullH);
      }

      // Helper that derives the on-canvas crop rectangle from whichever
      // saved state we have: explicit crop coords → aspect-ratio preset →
      // whole (rotated) image. Centralized so we never end up with a stale
      // full-image rect when the user actually had an aspect crop applied.
      const computeCropRectFromSaved = (): { x: number; y: number; w: number; h: number } => {
        if (savedCrop && fullW > 0 && natW > 0) {
          const scaleToThumb = natW / fullW;
          const left = finite(savedCrop[0]);
          const top = finite(savedCrop[1]);
          const right = finite(savedCrop[2]);
          const bottom = finite(savedCrop[3]);
          const x = Math.round(left * scaleToThumb);
          const y = Math.round(top * scaleToThumb);
          const w = Math.max(1, Math.round((right - left) * scaleToThumb));
          const h = Math.max(1, Math.round((bottom - top) * scaleToThumb));
          // Reject a degenerate / zero crop (shouldn't happen but defensive).
          if (w > 0 && h > 0) return { x, y, w, h };
        }

        // No explicit crop but an aspect-ratio preset is saved → materialize
        // it. This is the fix for "crop area disappears on reopen": the
        // aspect-preset path in applyCropRatio may have skipped saving the
        // crop array when dimensions weren't ready, so on reopen we
        // recompute from the ratio now that they are.
        const aspectStr = panel?.aspect_ratio_str || "";
        const ratio = aspectStr === "1:1" ? 1
          : aspectStr === "4:3" ? 4 / 3
          : aspectStr === "16:9" ? 16 / 9
          : (() => {
              const m = aspectStr.match(/^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/);
              if (!m) return null;
              const rw = parseFloat(m[1]); const rh = parseFloat(m[2]);
              return rh > 0 ? rw / rh : null;
            })();
        if (ratio && rotW > 0 && rotH > 0) {
          let rw = rotW;
          let rh = Math.round(rw / ratio);
          if (rh > rotH) { rh = rotH; rw = Math.round(rh * ratio); }
          const rx = Math.round((rotW - rw) / 2);
          const ry = Math.round((rotH - rh) / 2);
          return { x: rx, y: ry, w: rw, h: rh };
        }

        // Fallback: whole rotated image.
        return { x: 0, y: 0, w: Math.max(1, rotW), h: Math.max(1, rotH) };
      };

      setCropRect(computeCropRectFromSaved());

      setOrigImgSrc(src);
    };
    img.onerror = () => {
      console.warn("[EditPanelDialog] failed to load thumbnail for", panelImageName);
    };
    img.src = src;
    // Fire onload synchronously for already-decoded (cached / data URL)
    // images — matches the behaviour in CropCanvas.
    if (img.complete && img.naturalWidth > 0) {
      img.onload && (img.onload as () => void)(/* synthetic */);
    }
  }, [open, panel?.image_name]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load thumbnail image for client-side preview generation
  useEffect(() => {
    if (!origImgSrc) { previewImgRef.current = null; return; }
    const img = new window.Image();
    img.onload = () => { previewImgRef.current = img; };
    img.src = origImgSrc;
  }, [origImgSrc]);

  // Client-side crop preview — renders instantly using Canvas (no backend round-trip).
  // Called during drag for real-time feedback.
  const generateClientPreview = useCallback((rect: { x: number; y: number; w: number; h: number }) => {
    const img = previewImgRef.current;
    if (!img || imgNatW <= 0 || rect.w <= 0 || rect.h <= 0) return;

    const rot = displayRotation;
    const { rotW, rotH } = computeRotatedDims(imgNatW, imgNatH, rot);
    const rad = (rot || 0) * Math.PI / 180;

    // Draw full rotated image on offscreen canvas
    const off = document.createElement("canvas");
    off.width = rotW;
    off.height = rotH;
    const octx = off.getContext("2d");
    if (!octx) return;
    octx.translate(rotW / 2, rotH / 2);
    octx.rotate(rad);
    octx.drawImage(img, -imgNatW / 2, -imgNatH / 2, imgNatW, imgNatH);

    // Extract just the crop region
    const cropW = Math.min(rect.w, rotW - rect.x);
    const cropH = Math.min(rect.h, rotH - rect.y);
    if (cropW <= 0 || cropH <= 0) return;
    const crop = document.createElement("canvas");
    crop.width = cropW;
    crop.height = cropH;
    const cctx = crop.getContext("2d");
    if (!cctx) return;
    cctx.drawImage(off, rect.x, rect.y, cropW, cropH, 0, 0, cropW, cropH);

    // Convert to base64 and set as preview
    const dataUrl = crop.toDataURL("image/png");
    const b64 = dataUrl.split(",")[1];
    if (b64) setPreviewB64(b64);
  }, [imgNatW, imgNatH, displayRotation]);

  // Keep a ref to local so refreshPreview always reads the latest value
  // (avoids stale-closure bugs where setLocal hasn't flushed yet)
  const localRef = useRef(local);
  localRef.current = local;

  // Request sequencing — only apply the response from the LATEST request.
  // Prevents stale previews from overwriting newer ones when requests overlap.
  const previewSeqRef = useRef(0);

  // Debounced preview refresh — uses atomic patch+preview endpoint
  // to guarantee the preview always matches the panel state sent.
  const refreshPreview = useCallback(() => {
    if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
    previewTimerRef.current = setTimeout(async () => {
      const cur = localRef.current;
      if (!cur?.image_name) return;
      const seq = ++previewSeqRef.current;
      try {
        const resp = await api.patchPanelAndPreview(row, col, cur as unknown as Record<string, unknown>);
        // Only apply if this is still the latest request (no newer request was sent)
        if (seq === previewSeqRef.current && resp.image) setPreviewB64(resp.image);
      } catch {
        // ignore preview errors
      }
    }, 300);
  }, [row, col]); // eslint-disable-line react-hooks/exhaustive-deps

  // Rendered preview: matplotlib-rendered single panel with labels + scale bar
  const refreshRenderedPreview = useCallback(() => {
    if (renderedPreviewTimerRef.current) clearTimeout(renderedPreviewTimerRef.current);
    renderedPreviewTimerRef.current = setTimeout(async () => {
      try {
        const cur = localRef.current;
        if (!cur?.image_name) return;
        // Sync panel state first
        await api.patchPanelAndPreview(row, col, cur as unknown as Record<string, unknown>);
        const resp = await api.getPanelRenderedPreview(row, col);
        if (resp.image) setRenderedPreviewB64(resp.image);
      } catch { /* ignore */ }
    }, 200);
  }, [row, col]);

  // Refresh rendered preview when overlays change or tab switches to overlay tabs
  const overlayHash = `${local?.labels?.length || 0}-${local?.add_scale_bar || false}-${local?.scale_bar?.bar_length_microns || 0}-${local?.scale_bar?.font_size || 0}-${local?.symbols?.length || 0}`;
  useEffect(() => {
    if (OVERLAY_TABS.includes(tabIdx) && local?.image_name) {
      refreshRenderedPreview();
    }
  }, [tabIdx, refreshRenderedPreview, overlayHash]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch external image thumbnail and dimensions when it changes
  useEffect(() => {
    const extName = local?.zoom_inset?.separate_image_name;
    if (extName && extName !== "select" && extName !== extImageNameRef.current) {
      extImageNameRef.current = extName;
      api.getImageThumb(extName).then(r => {
        if (r?.thumbnail) setExtImageThumb(r.thumbnail);
      }).catch(() => setExtImageThumb(""));
      // Also fetch dimensions and set default crop to full image
      api.getImageInfo(extName)
        .then(info => {
          if (info.width && info.height) {
            setExtImageDims({ w: info.width, h: info.height });
            if (local?.zoom_inset) {
              const zi = local.zoom_inset;
              if ((zi.width_inset ?? 1) <= 1 && (zi.height_inset ?? 1) <= 1) {
                updateLocal({
                  zoom_inset: { ...zi, x_inset: 0, y_inset: 0, width_inset: info.width, height_inset: info.height },
                });
              }
            }
          }
        }).catch(() => {});
    } else if (!extName || extName === "select") {
      setExtImageThumb("");
      extImageNameRef.current = "";
    }
  }, [local?.zoom_inset?.separate_image_name]); // eslint-disable-line

  if (!local) return null;

  const handleSave = () => {
    updatePanel(row, col, local);
    onClose();
  };

  const updateLocal = (patch: Partial<PanelInfo>) => {
    setLocal((prev) => {
      if (!prev) return prev;
      let next = { ...prev, ...patch };

      // When crop changes, transform line/area/symbol coordinates to maintain
      // physical position on the original image (critical for measurement accuracy)
      if (patch.crop !== undefined || patch.crop_image !== undefined) {
        const oW = origFullW > 0 ? origFullW : 1000;
        const oH = origFullH > 0 ? origFullH : 1000;
        const oldCrop = (prev.crop_image && prev.crop?.length === 4) ? prev.crop : [0, 0, oW, oH];
        const newCrop = (next.crop_image && next.crop?.length === 4) ? next.crop : [0, 0, oW, oH];
        const oldCx = oldCrop[0], oldCy = oldCrop[1], oldCw = oldCrop[2] - oldCrop[0], oldCh = oldCrop[3] - oldCrop[1];
        const newCx = newCrop[0], newCy = newCrop[1], newCw = newCrop[2] - newCrop[0], newCh = newCrop[3] - newCrop[1];

        if (oldCw > 0 && oldCh > 0 && newCw > 0 && newCh > 0) {
          // Annotations are stored in absolute pixel coords relative to the
          // ORIGINAL image. They do NOT move when crop changes — they stay
          // fixed to the pixels they were drawn on.
          // Only zoom inset coordinates need transformation (they're in pixel coords).

          const transformPt = (pctX: number, pctY: number): [number, number] => {
            const origPx = pctX / 100 * oldCw + oldCx;
            const origPy = pctY / 100 * oldCh + oldCy;
            return [(origPx - newCx) / newCw * 100, (origPy - newCy) / newCh * 100];
          };

          // Transform zoom inset source region (pixel coords → % → transform → % → pixel coords)
          if (next.zoom_inset && next.add_zoom_inset) {
            const zi = { ...next.zoom_inset };
            // Source region: convert pixel coords to old-crop-% then transform to new-crop-%
            const srcXPct = zi.x / oldCw * 100;
            const srcYPct = zi.y / oldCh * 100;
            const srcWPct = zi.width / oldCw * 100;
            const srcHPct = zi.height / oldCh * 100;
            const [newSrcXPct, newSrcYPct] = transformPt(srcXPct, srcYPct);
            const [newSrcEndXPct, newSrcEndYPct] = transformPt(srcXPct + srcWPct, srcYPct + srcHPct);
            zi.x = Math.round(newSrcXPct / 100 * newCw);
            zi.y = Math.round(newSrcYPct / 100 * newCh);
            zi.width = Math.round((newSrcEndXPct - newSrcXPct) / 100 * newCw);
            zi.height = Math.round((newSrcEndYPct - newSrcYPct) / 100 * newCh);
            // Target position
            const tgtXPct = zi.target_x / oldCw * 100;
            const tgtYPct = zi.target_y / oldCh * 100;
            const [newTgtXPct, newTgtYPct] = transformPt(tgtXPct, tgtYPct);
            zi.target_x = Math.round(newTgtXPct / 100 * newCw);
            zi.target_y = Math.round(newTgtYPct / 100 * newCh);
            next = { ...next, zoom_inset: zi };
          }
        }
      }

      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
    if (OVERLAY_TABS.includes(tabIdx)) setTimeout(() => refreshRenderedPreview(), 0);
  };

  // Label helpers
  const updateLabel = (idx: number, patch: Partial<LabelSettings>) => {
    setLocal((prev) => {
      if (!prev) return prev;
      const labels = [...prev.labels];
      labels[idx] = { ...labels[idx], ...patch };
      const next = { ...prev, labels };
      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
  };

  const addLabel = () => {
    // Use global font from config (first column label's font) instead of hardcoded arial
    const globalFont = config?.column_labels?.[0]?.font_name || "arial.ttf";
    const globalFontSize = config?.column_labels?.[0]?.font_size || 12;
    let addedIdx = -1;
    setLocal((prev) => {
      if (!prev) return prev;
      const lbl = defaultLabel();
      lbl.font_name = globalFont;
      lbl.font_size = globalFontSize;
      addedIdx = prev.labels.length;
      const next = { ...prev, labels: [...prev.labels, lbl] };
      localRef.current = next;
      return next;
    });
    // Select the newly-added label and refresh the server-rendered preview
    // so the label is baked in (otherwise the CSS overlay stays invisible
    // because it keys off whether a rendered preview exists).
    if (addedIdx >= 0) setSelectedLabelIdx(addedIdx);
    setTimeout(() => refreshPreview(), 0);
  };

  const removeLabel = (idx: number) => {
    setLocal((prev) => {
      if (!prev) return prev;
      const labels = prev.labels.filter((_, i) => i !== idx);
      const next = { ...prev, labels };
      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
  };

  // Symbol helpers
  const updateSymbol = (idx: number, patch: Partial<SymbolSettings>) => {
    setLocal((prev) => {
      if (!prev) return prev;
      const symbols = [...prev.symbols];
      symbols[idx] = { ...symbols[idx], ...patch };
      const next = { ...prev, symbols };
      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
  };

  const addSymbol = () => {
    const globalFont = config?.column_labels?.[0]?.font_name || "arial.ttf";
    setLocal((prev) => {
      if (!prev) return prev;
      const next = { ...prev, symbols: [...prev.symbols, defaultSymbol(globalFont)] };
      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
    setTimeout(() => refreshRenderedPreview(), 0);
  };

  const removeSymbol = (idx: number) => {
    setLocal((prev) => {
      if (!prev) return prev;
      const symbols = prev.symbols.filter((_, i) => i !== idx);
      const next = { ...prev, symbols };
      localRef.current = next;
      return next;
    });
    setTimeout(() => refreshPreview(), 0);
    setTimeout(() => refreshRenderedPreview(), 0);
  };

  // Crop rect -> update visual + client-side preview (no backend call during drag)
  const handleCropRectChange = (rect: { x: number; y: number; w: number; h: number }) => {
    setCropRect(rect);
    // Real-time client-side preview (instant, no network)
    generateClientPreview(rect);
  };

  // Commit crop to backend when user finishes dragging.
  // Shows instant client-side preview, then fetches full-quality backend preview.
  const handleCropCommit = async (rect: { x: number; y: number; w: number; h: number }) => {
    // Instant client-side preview (no network delay)
    generateClientPreview(rect);

    // Compute scale factor: thumbnail → original (must be > 1 for real images)
    if (origFullW <= 0 || imgNatW <= 0) return; // safety: don't commit if dimensions unknown
    const scaleToOrig = origFullW / imgNatW;

    const left = Math.round(rect.x * scaleToOrig);
    const top = Math.round(rect.y * scaleToOrig);
    const right = Math.round((rect.x + rect.w) * scaleToOrig);
    const bottom = Math.round((rect.y + rect.h) * scaleToOrig);

    // Check if this is effectively the full rotated image (no crop needed)
    const { rotW: rotOrigW, rotH: rotOrigH } = computeRotatedDims(origFullW, origFullH, displayRotation);
    const isFull = left <= 1 && top <= 1 && Math.abs(right - rotOrigW) <= 2 && Math.abs(bottom - rotOrigH) <= 2;

    const cropPatch = {
      crop: isFull ? null : [left, top, right, bottom] as [number, number, number, number] | null,
      crop_image: !isFull,
    };

    // Build the full panel to send BEFORE calling setLocal (avoid React batching issues).
    // localRef.current always has the latest panel state from previous updates.
    const panelToSend = { ...localRef.current, ...cropPatch };

    // Update local state + ref immediately
    localRef.current = panelToSend as PanelInfo;
    setLocal(panelToSend as PanelInfo);

    // Cancel any pending debounced preview (we're doing it now)
    if (previewTimerRef.current) clearTimeout(previewTimerRef.current);

    // Atomically patch + preview in a single request (no race condition)
    const seq = ++previewSeqRef.current;
    try {
      const resp = await api.patchPanelAndPreview(row, col, panelToSend as unknown as Record<string, unknown>);
      // Only apply if still the latest request
      if (seq === previewSeqRef.current && resp.image) setPreviewB64(resp.image);
    } catch {
      // ignore
    }
  };

  // Aspect preset change -- uses rotated dimensions
  // Apply a given ratio to the crop rect and save
  const applyCropRatio = (ratio: number, ratioLabel: string) => {
    const { rotW, rotH } = computeRotatedDims(imgNatW, imgNatH, displayRotation);
    let rw = rotW;
    let rh = Math.round(rw / ratio);
    if (rh > rotH) { rh = rotH; rw = Math.round(rh * ratio); }
    const rx = Math.round((rotW - rw) / 2);
    const ry = Math.round((rotH - rh) / 2);
    setCropRect({ x: rx, y: ry, w: rw, h: rh });
    const rect = { x: rx, y: ry, w: rw, h: rh };
    if (origFullW > 0 && imgNatW > 0 && rw > 0 && rh > 0) {
      const s = origFullW / imgNatW;
      updateLocal({ aspect_ratio_str: ratioLabel, crop_image: true, crop: [Math.round(rect.x * s), Math.round(rect.y * s), Math.round((rect.x + rect.w) * s), Math.round((rect.y + rect.h) * s)] });
    } else {
      // Can't compute an accurate full-res crop yet. Still record the
      // aspect-ratio preset so a future reopen can reconstruct the rect
      // from it (see computeCropRectFromSaved), and leave crop null.
      updateLocal({ aspect_ratio_str: ratioLabel, crop_image: true, crop: null });
    }
  };

  const handleAspectPresetChange = (preset: AspectPreset) => {
    setAspectPreset(preset);
    const { rotW, rotH } = computeRotatedDims(imgNatW, imgNatH, displayRotation);
    if (preset === "Free") {
      updateLocal({ aspect_ratio_str: "", crop: null, crop_image: false });
      setCropRect({ x: 0, y: 0, w: rotW, h: rotH });
    } else if (preset === "Custom") {
      const r = parseRatioStr(customRatioStr);
      if (r) applyCropRatio(r, customRatioStr);
    } else {
      const ratio = preset === "1:1" ? 1 : preset === "4:3" ? 4/3 : 16/9;
      applyCropRatio(ratio, preset);
    }
  };

  const fontList = fonts.length > 0 ? fonts : ["arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf"];
  const imageNames = Object.keys(loadedImages);

  // Compute stable SVG overlay dimensions from crop settings (not stale DOM reads)
  const svgDims = (() => {
    if (local?.crop_image && local?.crop && local.crop.length === 4) {
      const cw = local.crop[2] - local.crop[0];
      const ch = local.crop[3] - local.crop[1];
      if (cw > 0 && ch > 0) return { w: cw, h: ch };
    }
    if (origFullW > 0 && origFullH > 0) return { w: origFullW, h: origFullH };
    if (processedW > 0 && processedH > 0) return { w: processedW, h: processedH };
    return { w: 1000, h: 1000 };
  })();

  // Convert percentage coords (0-100) from CROPPED image space to ORIGINAL image space and back.
  // Line/area points are stored as % of the cropped image. When crop changes,
  // we need to re-map so points stay at the same physical location.
  // origPct = point as % of original image
  // cropPct = point as % of cropped image
  const origW = origFullW > 0 ? origFullW : 1000;
  const origH = origFullH > 0 ? origFullH : 1000;
  const cropRect2 = (local?.crop_image && local?.crop?.length === 4) ? local.crop : [0, 0, origW, origH];
  const cropX = cropRect2[0], cropY = cropRect2[1];
  const cropW = cropRect2[2] - cropRect2[0];
  const cropH = cropRect2[3] - cropRect2[1];

  // Convert a point from crop-% to pixel in crop space
  const cropPctToPixel = (pctX: number, pctY: number): [number, number] => [
    pctX / 100 * cropW,
    pctY / 100 * cropH,
  ];
  // Convert a pixel in display (image element) to crop-%
  const displayToCropPct = (dispX: number, dispY: number, dispW: number, dispH: number): [number, number] => [
    Math.max(0, Math.min(100, (dispX / dispW) * 100)),
    Math.max(0, Math.min(100, (dispY / dispH) * 100)),
  ];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1 }}>
        <Box component="span">Edit Panel R{row + 1} C{col + 1}</Box>
        {/* Undo / Redo — applies to every tab (crop, resize, adjustments,
            labels, scale bar, …). Hotkeys: Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z. */}
        <Box sx={{ display: "flex", gap: 0.5 }}>
          <Tooltip title="Undo (⌘/Ctrl+Z)">
            <span>
              <IconButton size="small" disabled={!canUndo} onClick={doUndo}>
                <UndoIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Redo (⌘/Ctrl+⇧+Z)">
            <span>
              <IconButton size="small" disabled={!canRedo} onClick={doRedo}>
                <RedoIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: "flex", gap: 2, minHeight: 400 }}>
          {/* -- Left: Controls -------------------------------- */}
          <Box sx={{ width: 400, minWidth: 400, flexShrink: 0, overflowY: "auto", overflowX: "hidden", maxHeight: "70vh", pr: 2 }}>
            {(() => {
              const VIDEO_EXTS = ['.mp4','.avi','.mov','.mkv','.webm','.wmv','.flv','.m4v','.mpg','.mpeg','.3gp','.ts','.mts'];
              const imgName = local?.image_name || "";
              const isVid = VIDEO_EXTS.some(ext => imgName.toLowerCase().endsWith(ext));
              return (
                <Tabs value={tabIdx} onChange={(_, v) => setTabIdx(v)} variant="scrollable" scrollButtons="auto">
                  {isVid && <Tab label="Play & Seek" />}
                  {!isVid && isZStackPanel && <Tab label="Z-Stack Slice" />}
                  <Tab label="Crop/Resize" />
                  <Tab label="Adjustments" />
                  <Tab label="Labels" />
                  <Tab label="Scale Bar" />
                  <Tab label="Annotations" />
                  <Tab label="Zoom Inset" />
                </Tabs>
              );
            })()}

        {/* -- Tab 0: Crop / Resize ----------------------------- */}
        {/* Video Play & Seek tab (only for video files) */}
        {isVideoPanel && (
          <TabPanel value={tabIdx} index={0}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.85rem" }}>
                🎬 Video Frame Selector
              </Typography>
              <VideoFrameSelector
                imageName={local?.image_name || ""}
                frame={videoFrame}
                setFrame={(f) => {
                  setVideoFrame(f);
                  setLocal((prev) => prev ? { ...prev, frame: f } : prev);
                }}
                frameStart={videoFrameStart}
                setFrameStart={(f) => {
                  setVideoFrameStart(f);
                  setLocal((prev) => prev ? { ...prev, frame_start: f } : prev);
                }}
                frameEnd={videoFrameEnd}
                setFrameEnd={(f) => {
                  setVideoFrameEnd(f);
                  setLocal((prev) => prev ? { ...prev, frame_end: f } : prev);
                }}
                playRange={videoPlayRange}
                setPlayRange={(b) => {
                  setVideoPlayRange(b);
                  setLocal((prev) => prev ? { ...prev, play_range: b } : prev);
                }}
                returnToSelectedOnEnd={videoReturnToSelectedOnEnd}
                setReturnToSelectedOnEnd={(b) => {
                  setVideoReturnToSelectedOnEnd(b);
                  setLocal((prev) => prev ? { ...prev, return_to_selected_on_end: b } : prev);
                }}
                onFrameChange={() => { refreshPreview(); }}
                onFrameImage={(b64) => { setPreviewB64(b64); }}
              />
              {/* Sync seek to other panels using the SAME video in this
                  row / column. Anchors on local.frame so the user gets
                  what they currently see, not what's on disk. */}
              {config && local?.image_name && (
                <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
                  <Button
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: "0.6rem", textTransform: "none" }}
                    onClick={() => {
                      if (!config || !local) return;
                      const tgt = local.frame ?? 0;
                      const name = local.image_name;
                      const updatePanel = useFigureStore.getState().updatePanel;
                      let applied = 0;
                      for (let c2 = 0; c2 < config.cols; c2++) {
                        if (c2 === col) continue;
                        const p2 = config.panels[row]?.[c2];
                        if (p2?.image_name && p2.image_name === name) {
                          updatePanel(row, c2, { frame: tgt });
                          applied++;
                        }
                      }
                      setSyncSeekResult({
                        title: "Sync seek to row",
                        message: applied === 0
                          ? "No other panels in this row use the same video."
                          : `Synced ${applied} panel${applied === 1 ? "" : "s"} in row ${row + 1} to frame ${tgt}.`,
                      });
                    }}
                  >
                    Sync seek to row
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: "0.6rem", textTransform: "none" }}
                    onClick={() => {
                      if (!config || !local) return;
                      const tgt = local.frame ?? 0;
                      const name = local.image_name;
                      const updatePanel = useFigureStore.getState().updatePanel;
                      let applied = 0;
                      for (let r2 = 0; r2 < config.rows; r2++) {
                        if (r2 === row) continue;
                        const p2 = config.panels[r2]?.[col];
                        if (p2?.image_name && p2.image_name === name) {
                          updatePanel(r2, col, { frame: tgt });
                          applied++;
                        }
                      }
                      setSyncSeekResult({
                        title: "Sync seek to column",
                        message: applied === 0
                          ? "No other panels in this column use the same video."
                          : `Synced ${applied} panel${applied === 1 ? "" : "s"} in column ${col + 1} to frame ${tgt}.`,
                      });
                    }}
                  >
                    Sync seek to column
                  </Button>
                </Box>
              )}
            </Box>
          </TabPanel>
        )}
        {/* Z-Stack Frame tab (only for multi-frame TIFF files) */}
        {isZStackPanel && !isVideoPanel && (
          <TabPanel value={tabIdx} index={0}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.85rem" }}>
                📚 Z-Stack Slice Selector
              </Typography>
              <ZStackFrameSelector imageName={local?.image_name || ""} frame={videoFrame} setFrame={setVideoFrame} panelRow={row} panelCol={col} onFrameChange={() => {
                refreshPreview();
              }} onFrameImage={(b64) => {
                setPreviewB64(b64);
              }} onAppliedToPanel={() => {
                // 3D volume was applied as the panel image — close the edit
                // dialog so the user re-opens it on the updated image (which
                // loads the correct frame/adjustment state for the new PNG).
                onClose();
              }} />
            </Box>
          </TabPanel>
        )}
        <TabPanel value={tabIdx} index={0 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>

            {/* Rotation control — -180° to 180°, 0.1° fine step */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 1 }}>
              <Typography variant="caption" sx={{ width: 56, flexShrink: 0 }}>Rotation</Typography>
              <Slider
                size="small"
                value={displayRotation}
                min={-180}
                max={180}
                step={0.1}
                marks={[{ value: -180 }, { value: -90 }, { value: 0 }, { value: 90 }, { value: 180 }]}
                sx={{ mx: 1 }}
                onChange={(_, v) => {
                  const deg = v as number;
                  setDisplayRotation(deg);
                  // Internally store as 0-360 for backend compatibility
                  const absDeg = ((deg % 360) + 360) % 360;
                  const { rotW: newRotW, rotH: newRotH } = computeRotatedDims(imgNatW, imgNatH, absDeg);
                  setCropRect((prev) => {
                    const w = Math.min(prev.w, newRotW);
                    const h = Math.min(prev.h, newRotH);
                    return {
                      x: Math.max(0, Math.min(prev.x, newRotW - w)),
                      y: Math.max(0, Math.min(prev.y, newRotH - h)),
                      w, h,
                    };
                  });
                  setLocal((prev) => prev ? { ...prev, rotation: absDeg } : prev);
                }}
                onChangeCommitted={(_, v) => {
                  const absDeg = (((v as number) % 360) + 360) % 360;
                  updateLocal({ rotation: absDeg });
                }}
              />
              {/* Fine adjustment: -0.1° */}
              <IconButton size="small" title="-0.1°" sx={{ p: 0.25 }} onClick={() => {
                const deg = Math.max(-180, Math.round((displayRotation - 0.1) * 10) / 10);
                setDisplayRotation(deg);
                const absDeg = ((deg % 360) + 360) % 360;
                const { rotW: newRotW, rotH: newRotH } = computeRotatedDims(imgNatW, imgNatH, absDeg);
                setCropRect((prev) => {
                  const w = Math.min(prev.w, newRotW);
                  const h = Math.min(prev.h, newRotH);
                  return { x: Math.max(0, Math.min(prev.x, newRotW - w)), y: Math.max(0, Math.min(prev.y, newRotH - h)), w, h };
                });
                updateLocal({ rotation: absDeg });
              }}>
                <Typography sx={{ fontSize: 10, lineHeight: 1 }}>&#9660;</Typography>
              </IconButton>
              <TextField
                type="number"
                size="small"
                value={displayRotation}
                onChange={(e) => {
                  let v = Number(e.target.value);
                  v = Math.max(-180, Math.min(180, v));
                  setDisplayRotation(v);
                  const absDeg = ((v % 360) + 360) % 360;
                  const { rotW: newRotW, rotH: newRotH } = computeRotatedDims(imgNatW, imgNatH, absDeg);
                  setCropRect((prev) => {
                    const w = Math.min(prev.w, newRotW);
                    const h = Math.min(prev.h, newRotH);
                    return { x: Math.max(0, Math.min(prev.x, newRotW - w)), y: Math.max(0, Math.min(prev.y, newRotH - h)), w, h };
                  });
                  updateLocal({ rotation: absDeg });
                }}
                inputProps={{ min: -180, max: 180, step: 0.1 }}
                sx={{ width: 72, flexShrink: 0, "& input": { textAlign: "center", py: 0.5, px: 1, fontSize: "0.75rem" } }}
              />
              {/* Fine adjustment: +0.1° */}
              <IconButton size="small" title="+0.1°" sx={{ p: 0.25 }} onClick={() => {
                const deg = Math.min(180, Math.round((displayRotation + 0.1) * 10) / 10);
                setDisplayRotation(deg);
                const absDeg = ((deg % 360) + 360) % 360;
                const { rotW: newRotW, rotH: newRotH } = computeRotatedDims(imgNatW, imgNatH, absDeg);
                setCropRect((prev) => {
                  const w = Math.min(prev.w, newRotW);
                  const h = Math.min(prev.h, newRotH);
                  return { x: Math.max(0, Math.min(prev.x, newRotW - w)), y: Math.max(0, Math.min(prev.y, newRotH - h)), w, h };
                });
                updateLocal({ rotation: absDeg });
              }}>
                <Typography sx={{ fontSize: 10, lineHeight: 1 }}>&#9650;</Typography>
              </IconButton>
              <Typography variant="caption" sx={{ flexShrink: 0 }}>&deg;</Typography>
              <IconButton size="small" title="Reset rotation" onClick={() => {
                setDisplayRotation(0);
                setCropRect({ x: 0, y: 0, w: imgNatW, h: imgNatH });
                updateLocal({ rotation: 0 });
              }}>
                <RestartAltIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </Box>

            {/* Copy rotation TO row / column — parallel to copy-crop below.
                Only offered when other panels with images exist in the row/col. */}
            {config && local && (
              <Box sx={{ display: "flex", gap: 0.5 }}>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config || !local) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    const rot = local.rotation ?? 0;
                    let applied = 0;
                    for (let c2 = 0; c2 < config.cols; c2++) {
                      if (c2 === col) continue;
                      const p2 = config.panels[row]?.[c2];
                      if (!p2?.image_name) continue;
                      updatePanel(row, c2, { rotation: rot });
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this row have images assigned.");
                  }}
                >Copy rotation to row</Button>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config || !local) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    const rot = local.rotation ?? 0;
                    let applied = 0;
                    for (let r2 = 0; r2 < config.rows; r2++) {
                      if (r2 === row) continue;
                      const p2 = config.panels[r2]?.[col];
                      if (!p2?.image_name) continue;
                      updatePanel(r2, col, { rotation: rot });
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this column have images assigned.");
                  }}
                >Copy rotation to column</Button>
              </Box>
            )}

            {/* Flip buttons */}
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Flip</Typography>
              <Box sx={{ display: "flex", gap: 1 }}>
                <ToggleButton
                  value="flip_h"
                  selected={local.flip_horizontal ?? false}
                  onChange={() => updateLocal({ flip_horizontal: !local.flip_horizontal })}
                  size="small"
                >
                  <FlipIcon sx={{ mr: 0.5 }} />
                  Horizontal
                </ToggleButton>
                <ToggleButton
                  value="flip_v"
                  selected={local.flip_vertical ?? false}
                  onChange={() => updateLocal({ flip_vertical: !local.flip_vertical })}
                  size="small"
                >
                  <FlipIcon sx={{ transform: "rotate(90deg)", mr: 0.5 }} />
                  Vertical
                </ToggleButton>
              </Box>
            </Box>

            {/* Aspect ratio preset */}
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Aspect Ratio</Typography>
              <ToggleButtonGroup
                value={aspectPreset}
                exclusive
                onChange={(_, v) => { if (v !== null) handleAspectPresetChange(v as AspectPreset); }}
                size="small"
                sx={{ flexWrap: "wrap" }}
              >
                {ASPECT_PRESETS.map((p) => (
                  <ToggleButton key={p} value={p} sx={{ px: 1.5, fontSize: "0.7rem" }}>
                    {p}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
              {aspectPreset === "Custom" && (
                <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mt: 0.5 }}>
                  <TextField
                    size="small"
                    value={customRatioStr}
                    onChange={(e) => setCustomRatioStr(e.target.value)}
                    onBlur={() => { const r = parseRatioStr(customRatioStr); if (r) applyCropRatio(r, customRatioStr); }}
                    onKeyDown={(e) => { if (e.key === "Enter") { const r = parseRatioStr(customRatioStr); if (r) applyCropRatio(r, customRatioStr); } }}
                    placeholder="W:H"
                    sx={{ width: 80, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }}
                  />
                  <Typography variant="caption" color="text.secondary">e.g. 3:2, 5:4</Typography>
                </Box>
              )}
            </Box>

            {/* Copy crop from adjacent panel */}
            {(() => {
              if (!config) return null;
              const adjPanels: { r: number; c: number; label: string; crop: [number, number, number, number] }[] = [];
              for (let r2 = 0; r2 < config.rows; r2++) {
                for (let c2 = 0; c2 < config.cols; c2++) {
                  if (r2 === row && c2 === col) continue;
                  const p2 = config.panels[r2]?.[c2];
                  if (p2?.crop && p2.image_name) {
                    adjPanels.push({ r: r2, c: c2, label: `R${r2+1}C${c2+1}`, crop: p2.crop as [number,number,number,number] });
                  }
                }
              }
              if (adjPanels.length === 0) return null;
              return (
                <FormControl size="small" fullWidth>
                  <InputLabel sx={{ fontSize: "0.75rem" }}>Copy crop from…</InputLabel>
                  <Select
                    value=""
                    label="Copy crop from…"
                    onChange={(e) => {
                      const key = e.target.value as string;
                      const src = adjPanels.find(p => `${p.r}-${p.c}` === key);
                      if (!src) return;
                      const [sl, st, sr, sb] = src.crop;
                      const cw = sr - sl, ch = sb - st;
                      // Check if crop fits in current image
                      const { rotW: rotOrigW, rotH: rotOrigH } = computeRotatedDims(origFullW, origFullH, displayRotation);
                      if (cw > rotOrigW || ch > rotOrigH) {
                        alert(`Crop from ${src.label} (${cw}×${ch}) exceeds this image (${rotOrigW}×${rotOrigH}). Cannot apply.`);
                        return;
                      }
                      // Apply: center the crop if it would go out of bounds
                      const left = Math.min(sl, Math.max(0, rotOrigW - cw));
                      const top2 = Math.min(st, Math.max(0, rotOrigH - ch));
                      updateLocal({ crop: [left, top2, left + cw, top2 + ch], crop_image: true });
                      // Update visual crop rect (scaled to thumbnail)
                      if (origFullW > 0 && imgNatW > 0) {
                        const s = imgNatW / origFullW;
                        setCropRect({ x: Math.round(left * s), y: Math.round(top2 * s), w: Math.round(cw * s), h: Math.round(ch * s) });
                      }
                    }}
                    sx={{ fontSize: "0.75rem" }}
                  >
                    {adjPanels.map(p => (
                      <MenuItem key={`${p.r}-${p.c}`} value={`${p.r}-${p.c}`} sx={{ fontSize: "0.75rem" }}>
                        {p.label}: {p.crop[2]-p.crop[0]}×{p.crop[3]-p.crop[1]}px
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              );
            })()}

            {/* Copy crop TO row/column */}
            {local.crop && config && (
              <Box sx={{ display: "flex", gap: 0.5 }}>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config || !local.crop) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    let applied = 0;
                    for (let c2 = 0; c2 < config.cols; c2++) {
                      if (c2 === col) continue;
                      const p2 = config.panels[row]?.[c2];
                      if (!p2?.image_name) continue;
                      updatePanel(row, c2, { crop: local.crop, crop_image: true, aspect_ratio_str: local.aspect_ratio_str });
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this row have images assigned.");
                  }}
                >Copy crop to row</Button>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config || !local.crop) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    let applied = 0;
                    for (let r2 = 0; r2 < config.rows; r2++) {
                      if (r2 === row) continue;
                      const p2 = config.panels[r2]?.[col];
                      if (!p2?.image_name) continue;
                      updatePanel(r2, col, { crop: local.crop, crop_image: true, aspect_ratio_str: local.aspect_ratio_str });
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this column have images assigned.");
                  }}
                >Copy crop to column</Button>
              </Box>
            )}

            {/* Crop pixel dimensions — shows full-res pixels, commits on Enter */}
            <CropPixelFields
              cropRect={cropRect}
              imgNatW={imgNatW} imgNatH={imgNatH}
              origFullW={origFullW} origFullH={origFullH}
              displayRotation={displayRotation}
              aspectPreset={aspectPreset}
              customRatioStr={customRatioStr}
              onCommit={(newRect) => {
                setCropRect(newRect);
                generateClientPreview(newRect);
                handleCropCommit(newRect);
              }}
            />

            {/* Visual crop canvas */}
            {origImgSrc && imgNatW > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
                  Drag to move crop area. Drag corners/edges to resize.
                </Typography>
                <CropCanvas
                  imageSrc={origImgSrc}
                  aspectPreset={aspectPreset}
                  customRatio={aspectPreset === "Custom" ? parseRatioStr(customRatioStr) : null}
                  cropRect={cropRect}
                  imgNatW={imgNatW}
                  imgNatH={imgNatH}
                  rotation={displayRotation}
                  flipH={local.flip_horizontal}
                  flipV={local.flip_vertical}
                  active={tabIdx === TAB_CROP}
                  onChange={handleCropRectChange}
                  onCommit={handleCropCommit}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                  Position: {Number.isFinite(cropRect.x) ? cropRect.x : 0}, {Number.isFinite(cropRect.y) ? cropRect.y : 0}
                </Typography>
              </Box>
            )}

          </Box>
        </TabPanel>

        {/* -- Tab 1: Image Adjustments ----------------------------- */}
        <TabPanel value={tabIdx} index={1 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>

            {/* Interactive per-channel histogram — uses ORIGINAL image for stable display */}
            {origImgSrc && (
              <InteractiveHistogram
                imageSrc={origImgSrc}
                levels={{
                  input_black_r: local.input_black_r ?? 0, input_white_r: local.input_white_r ?? 255,
                  input_black_g: local.input_black_g ?? 0, input_white_g: local.input_white_g ?? 255,
                  input_black_b: local.input_black_b ?? 0, input_white_b: local.input_white_b ?? 255,
                }}
                onChange={(lv) => updateLocal(lv as unknown as Partial<PanelInfo>)}
                onCommit={(lv) => updateLocal(lv as unknown as Partial<PanelInfo>)}
              />
            )}

            {/* Copy adjustments from another panel */}
            {(() => {
              if (!config) return null;
              const adjPanels: { r: number; c: number; label: string }[] = [];
              for (let r2 = 0; r2 < config.rows; r2++) {
                for (let c2 = 0; c2 < config.cols; c2++) {
                  if (r2 === row && c2 === col) continue;
                  const p2 = config.panels[r2]?.[c2];
                  if (p2?.image_name) {
                    adjPanels.push({ r: r2, c: c2, label: `R${r2+1}C${c2+1}` });
                  }
                }
              }
              if (adjPanels.length === 0) return null;
              return (
                <FormControl size="small" fullWidth sx={{ mb: 1 }}>
                  <InputLabel sx={{ fontSize: "0.75rem" }}>Copy adjustments from...</InputLabel>
                  <Select
                    value=""
                    label="Copy adjustments from..."
                    onChange={(e) => {
                      const key = e.target.value as string;
                      const src = adjPanels.find(p => `${p.r}-${p.c}` === key);
                      if (!src || !config) return;
                      const srcP = config.panels[src.r][src.c];
                      updateLocal({
                        brightness: srcP.brightness, contrast: srcP.contrast, exposure: srcP.exposure,
                        gamma: srcP.gamma, hue: srcP.hue, saturation: srcP.saturation, vibrance: srcP.vibrance,
                        color_temperature: srcP.color_temperature, tint: srcP.tint,
                        sharpen: srcP.sharpen, blur: srcP.blur, denoise: srcP.denoise,
                        highlights: srcP.highlights, shadows: srcP.shadows,
                        midtones: (srcP as unknown as Record<string, number>).midtones ?? 0,
                        input_black_r: srcP.input_black_r, input_white_r: srcP.input_white_r,
                        input_black_g: srcP.input_black_g, input_white_g: srcP.input_white_g,
                        input_black_b: srcP.input_black_b, input_white_b: srcP.input_white_b,
                        invert: srcP.invert, grayscale: srcP.grayscale,
                      } as unknown as Partial<PanelInfo>);
                    }}
                    sx={{ fontSize: "0.75rem" }}
                  >
                    {adjPanels.map(p => (
                      <MenuItem key={`${p.r}-${p.c}`} value={`${p.r}-${p.c}`} sx={{ fontSize: "0.75rem" }}>
                        {p.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              );
            })()}

            {/* Copy adjustments TO row/column */}
            {config && (
              <Box sx={{ display: "flex", gap: 0.5, mb: 1 }}>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    const adjPatch = {
                      brightness: local.brightness, contrast: local.contrast, exposure: local.exposure,
                      gamma: local.gamma, hue: local.hue, saturation: local.saturation, vibrance: local.vibrance,
                      color_temperature: local.color_temperature, tint: local.tint,
                      sharpen: local.sharpen, blur: local.blur, denoise: local.denoise,
                      highlights: local.highlights, shadows: local.shadows, midtones: local.midtones,
                      input_black_r: local.input_black_r, input_white_r: local.input_white_r,
                      input_black_g: local.input_black_g, input_white_g: local.input_white_g,
                      input_black_b: local.input_black_b, input_white_b: local.input_white_b,
                      invert: local.invert, grayscale: local.grayscale,
                    };
                    let applied = 0;
                    for (let c2 = 0; c2 < config.cols; c2++) {
                      if (c2 === col) continue;
                      if (!config.panels[row]?.[c2]?.image_name) continue;
                      updatePanel(row, c2, adjPatch as unknown as Partial<PanelInfo>);
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this row have images assigned.");
                  }}
                >Copy adjustments to row</Button>
                <Button size="small" variant="outlined" sx={{ fontSize: "0.6rem", textTransform: "none" }}
                  onClick={() => {
                    if (!config) return;
                    const updatePanel = useFigureStore.getState().updatePanel;
                    const adjPatch = {
                      brightness: local.brightness, contrast: local.contrast, exposure: local.exposure,
                      gamma: local.gamma, hue: local.hue, saturation: local.saturation, vibrance: local.vibrance,
                      color_temperature: local.color_temperature, tint: local.tint,
                      sharpen: local.sharpen, blur: local.blur, denoise: local.denoise,
                      highlights: local.highlights, shadows: local.shadows, midtones: local.midtones,
                      input_black_r: local.input_black_r, input_white_r: local.input_white_r,
                      input_black_g: local.input_black_g, input_white_g: local.input_white_g,
                      input_black_b: local.input_black_b, input_white_b: local.input_white_b,
                      invert: local.invert, grayscale: local.grayscale,
                    };
                    let applied = 0;
                    for (let r2 = 0; r2 < config.rows; r2++) {
                      if (r2 === row) continue;
                      if (!config.panels[r2]?.[col]?.image_name) continue;
                      updatePanel(r2, col, adjPatch as unknown as Partial<PanelInfo>);
                      applied++;
                    }
                    if (applied === 0) alert("No other panels in this column have images assigned.");
                  }}
                >Copy adjustments to column</Button>
              </Box>
            )}

            {/* Action buttons row */}
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <ButtonGroup size="small" variant="outlined">
                <Button startIcon={<AutoFixHighIcon />} onClick={async () => {
                  try {
                    const resp = await api.autoAdjust(row, col, "levels");
                    updateLocal(resp.adjustments as Partial<PanelInfo>);
                  } catch {}
                }}>Levels</Button>
                <Button onClick={async () => {
                  try {
                    const resp = await api.autoAdjust(row, col, "contrast");
                    updateLocal(resp.adjustments as Partial<PanelInfo>);
                  } catch {}
                }}>Contrast</Button>
                <Button onClick={async () => {
                  try {
                    const resp = await api.autoAdjust(row, col, "white_balance");
                    updateLocal(resp.adjustments as Partial<PanelInfo>);
                  } catch {}
                }}>WB</Button>
              </ButtonGroup>
              <Button
                size="small"
                startIcon={<RestartAltIcon />}
                onClick={() => updateLocal({
                  brightness: 1.0, contrast: 1.0, exposure: 0, gamma: 1.0,
                  hue: 0, saturation: 1.0, vibrance: 0, color_temperature: 0, tint: 0,
                  sharpen: 0, blur: 0, denoise: 0,
                  highlights: 0, shadows: 0, midtones: 0,
                  input_black_r: 0, input_white_r: 255,
                  input_black_g: 0, input_white_g: 255,
                  input_black_b: 0, input_white_b: 255,
                  invert: false, grayscale: false,
                } as unknown as Partial<PanelInfo>)}
              >
                Reset All
              </Button>
            </Box>

            {/* ---- Tone section ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600 }}>Tone</Typography>
            <Divider />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 2 }}>
              <AdjustSlider label="Brightness" value={local.brightness} defaultValue={1.0} min={0} max={2} step={0.01} marks={[{value:0},{value:1},{value:2}]} onChange={(v) => updateLocal({ brightness: v })} />
              <AdjustSlider label="Contrast" value={local.contrast} defaultValue={1.0} min={0} max={2} step={0.01} marks={[{value:0},{value:1},{value:2}]} onChange={(v) => updateLocal({ contrast: v })} />
              <AdjustSlider label="Exposure" value={local.exposure ?? 0} defaultValue={0} min={-3} max={3} step={0.1} marks={[{value:-3},{value:0},{value:3}]} onChange={(v) => updateLocal({ exposure: v })} suffix="EV" />
              <AdjustSlider label="Gamma" value={local.gamma ?? 1.0} defaultValue={1.0} min={0.1} max={3.0} step={0.05} marks={[{value:0.1},{value:1.0},{value:3.0}]} onChange={(v) => updateLocal({ gamma: v })} />
            </Box>

            {/* ---- Color section ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Color</Typography>
            <Divider />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 2 }}>
              <AdjustSlider label="Hue Shift" value={local.hue ?? 0} defaultValue={0} min={0} max={360} step={1} marks={[{value:0},{value:180},{value:360}]} onChange={(v) => updateLocal({ hue: v })} suffix="°" />
              <AdjustSlider label="Saturation" value={local.saturation ?? 1.0} defaultValue={1.0} min={0} max={2} step={0.01} marks={[{value:0},{value:1},{value:2}]} onChange={(v) => updateLocal({ saturation: v })} />
              <AdjustSlider label="Vibrance" value={local.vibrance ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ vibrance: v })} />
              <AdjustSlider label="Temperature" value={local.color_temperature ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ color_temperature: v })} />
              <AdjustSlider label="Tint" value={local.tint ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ tint: v })} />
            </Box>

            {/* ---- Detail section ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Detail</Typography>
            <Divider />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 2 }}>
              <AdjustSlider label="Sharpen" value={local.sharpen ?? 0} defaultValue={0} min={0} max={3} step={0.01} marks={[{value:0},{value:1},{value:2},{value:3}]} onChange={(v) => updateLocal({ sharpen: v })} />
              <AdjustSlider label="Blur" value={local.blur ?? 0} defaultValue={0} min={0} max={20} step={0.5} marks={[{value:0},{value:10},{value:20}]} onChange={(v) => updateLocal({ blur: v })} />
              <AdjustSlider label="Denoise" value={local.denoise ?? 0} defaultValue={0} min={0} max={1} step={0.01} marks={[{value:0},{value:0.5},{value:1}]} onChange={(v) => updateLocal({ denoise: v })} />
            </Box>

            {/* ---- Tone Curve section ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Tone Curve</Typography>
            <Divider />
            <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 2 }}>
              <AdjustSlider label="Highlights" value={local.highlights ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ highlights: v })} />
              <AdjustSlider label="Midtones" value={(local as unknown as Record<string, number>).midtones ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ midtones: v } as unknown as Partial<PanelInfo>)} />
              <AdjustSlider label="Shadows" value={local.shadows ?? 0} defaultValue={0} min={-100} max={100} step={1} marks={[{value:-100},{value:0},{value:100}]} onChange={(v) => updateLocal({ shadows: v })} />
            </Box>

            {/* ---- Effects section ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Effects</Typography>
            <Divider />
            <Box sx={{ display: "flex", gap: 2 }}>
              <FormControlLabel
                control={<Checkbox size="small" checked={local.invert ?? false} onChange={(e) => updateLocal({ invert: e.target.checked })} />}
                label={<Typography variant="caption">Invert</Typography>}
              />
              <FormControlLabel
                control={<Checkbox size="small" checked={local.grayscale ?? false} onChange={(e) => updateLocal({ grayscale: e.target.checked })} />}
                label={<Typography variant="caption">Grayscale</Typography>}
              />
            </Box>

            {/* Pseudocolor */}
            <Box>
              <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 600, mb: 0.5, display: "block" }}>Pseudocolor (LUT)</Typography>
              <Select
                size="small"
                value={local.pseudocolor || ""}
                onChange={(e) => updateLocal({ pseudocolor: e.target.value })}
                displayEmpty
                sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.4, px: 1 } }}
              >
                <MenuItem value="" sx={{ fontSize: "0.65rem" }}>None (original colors)</MenuItem>
                <MenuItem value="green" sx={{ fontSize: "0.65rem" }}>🟢 Green</MenuItem>
                <MenuItem value="red" sx={{ fontSize: "0.65rem" }}>🔴 Red</MenuItem>
                <MenuItem value="blue" sx={{ fontSize: "0.65rem" }}>🔵 Blue</MenuItem>
                <MenuItem value="cyan" sx={{ fontSize: "0.65rem" }}>🔵 Cyan</MenuItem>
                <MenuItem value="magenta" sx={{ fontSize: "0.65rem" }}>🟣 Magenta</MenuItem>
                <MenuItem value="yellow" sx={{ fontSize: "0.65rem" }}>🟡 Yellow</MenuItem>
                <MenuItem value="hot" sx={{ fontSize: "0.65rem" }}>🔥 Hot</MenuItem>
                <MenuItem value="cool" sx={{ fontSize: "0.65rem" }}>❄️ Cool</MenuItem>
                <MenuItem value="viridis" sx={{ fontSize: "0.65rem" }}>🌿 Viridis</MenuItem>
                <MenuItem value="magma" sx={{ fontSize: "0.65rem" }}>🌋 Magma</MenuItem>
                <MenuItem value="inferno" sx={{ fontSize: "0.65rem" }}>🔥 Inferno</MenuItem>
                <MenuItem value="plasma" sx={{ fontSize: "0.65rem" }}>⚡ Plasma</MenuItem>
              </Select>
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", mt: 0.5, display: "block" }}>
                Applies a false-color mapping to grayscale images
              </Typography>
            </Box>
          </Box>
        </TabPanel>

        {/* -- Tab 2: Labels ------------------------------------ */}
        <TabPanel value={tabIdx} index={2 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
            {local.labels.map((lbl, i) => (
              <Box key={i} sx={{
                border: "1px solid",
                borderColor: selectedLabelIdx === i ? "primary.main" : "divider",
                borderRadius: 1,
                p: 1.5,
                cursor: "pointer",
                "&:hover": { borderColor: "primary.light" },
              }}
                onClick={() => setSelectedLabelIdx(i)}
              >
                {/* Header row: label name + delete */}
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600, fontSize: "0.8rem" }}>{lbl.text || `Label ${i + 1}`}</Typography>
                  <IconButton onClick={(e) => { e.stopPropagation(); removeLabel(i); }} size="small"><DeleteIcon sx={{ fontSize: 16 }} /></IconButton>
                </Box>

                {/* Text + Color */}
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1.5 }}>
                  <TextField label="Text" value={lbl.text} onChange={(e) => updateLabel(i, { text: e.target.value })} fullWidth size="small" sx={{ "& input": { fontSize: "0.8rem" } }} />
                  <Tooltip title="Label color">
                    <input type="color" value={lbl.color} onChange={(e) => updateLabel(i, { color: e.target.value, default_color: e.target.value })} style={{ width: 36, height: 36, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                  </Tooltip>
                </Box>

                {/* Font row */}
                <Typography variant="caption" sx={{ fontWeight: 600, mb: 0.5, display: "block", color: "text.secondary" }}>Font</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1 }}>
                  <FormControl size="small" sx={{ flex: 1 }}>
                    <Select value={lbl.font_name} onChange={(e) => updateLabel(i, { font_name: e.target.value })} sx={{ fontSize: "0.75rem" }}>
                      {fontList.slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map((f) => <MenuItem key={f} value={f} sx={{ fontSize: "0.75rem" }}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <TextField label="Size" type="number" value={lbl.font_size} onChange={(e) => updateLabel(i, { font_size: Number(e.target.value) })}
                    size="small" inputProps={{ min: 4, max: 200 }}
                    sx={{ width: 72, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }}
                  />
                  <Tooltip title={lbl.linked_to_header ? "Linked to header size (click to unlink)" : "Link to header font size"}>
                    <IconButton size="small" onClick={(e) => { e.stopPropagation(); updateLabel(i, { linked_to_header: !lbl.linked_to_header }); }}
                      sx={{ color: lbl.linked_to_header ? "primary.main" : "text.disabled" }}>
                      {lbl.linked_to_header ? <span style={{ fontSize: "0.85rem" }}>{"\uD83D\uDD17"}</span> : <span style={{ fontSize: "0.85rem" }}>{"\u26D3"}</span>}
                    </IconButton>
                  </Tooltip>
                </Box>

                {/* Style buttons */}
                <Box sx={{ mb: 1.5 }}>
                  <ToggleButtonGroup size="small" value={lbl.font_style} onChange={(_, v) => updateLabel(i, { font_style: v })}>
                    <ToggleButton value="Bold" sx={{ px: 1, py: 0.25, fontWeight: 700, fontSize: "0.8rem" }}>B</ToggleButton>
                    <ToggleButton value="Italic" sx={{ px: 1, py: 0.25, fontStyle: "italic", fontSize: "0.8rem" }}>I</ToggleButton>
                    <ToggleButton value="Strikethrough" sx={{ px: 1, py: 0.25, textDecoration: "line-through", fontSize: "0.8rem" }}>S</ToggleButton>
                    <ToggleButton value="Superscript" sx={{ px: 1, py: 0.25, fontSize: "0.7rem" }}>X<sup style={{ fontSize: "0.55rem" }}>2</sup></ToggleButton>
                    <ToggleButton value="Subscript" sx={{ px: 1, py: 0.25, fontSize: "0.7rem" }}>X<sub style={{ fontSize: "0.55rem" }}>2</sub></ToggleButton>
                  </ToggleButtonGroup>
                </Box>

                {/* Position */}
                <Typography variant="caption" sx={{ fontWeight: 600, mb: 0.5, display: "block", color: "text.secondary" }}>Position</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1 }}>
                  <FormControl size="small" sx={{ flex: 1 }}>
                    <InputLabel sx={{ fontSize: "0.75rem" }}>Preset</InputLabel>
                    <Select
                      label="Preset"
                      value={lbl.position_preset || (() => {
                        for (const p of POSITION_PRESETS) {
                          if (p.label === "Custom") continue;
                          if (Math.abs(lbl.position_x - p.x(3)) < 1 && Math.abs(lbl.position_y - p.y(3)) < 1) return p.label;
                          if (Math.abs(lbl.position_x - p.x(5)) < 1 && Math.abs(lbl.position_y - p.y(5)) < 1) return p.label;
                        }
                        return "Custom";
                      })()}
                      onChange={(e) => {
                        const preset = POSITION_PRESETS.find(p => p.label === e.target.value);
                        if (preset) {
                          if (preset.label === "Custom") {
                            // Switch to custom — keep current X/Y, just make it draggable
                            updateLabel(i, { position_preset: "Custom" });
                          } else {
                            const d = 3;
                            updateLabel(i, { position_preset: preset.label, position_x: preset.x(d), position_y: preset.y(d) });
                          }
                        }
                      }}
                      sx={{ fontSize: "0.75rem" }}
                    >
                      {POSITION_PRESETS.map(p => <MenuItem key={p.label} value={p.label} sx={{ fontSize: "0.75rem" }}>{p.label}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Box>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1 }}>
                  <TextField label="X %" type="number" value={Math.round(lbl.position_x)} onChange={(e) => updateLabel(i, { position_x: Number(e.target.value), position_preset: "Custom" })}
                    size="small" inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ flex: 1, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }}
                  />
                  <TextField label="Y %" type="number" value={Math.round(lbl.position_y)} onChange={(e) => updateLabel(i, { position_y: Number(e.target.value), position_preset: "Custom" })}
                    size="small" inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ flex: 1, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }}
                  />
                </Box>

                {/* Rotation */}
                <Typography variant="caption" sx={{ fontWeight: 600, mb: 0.5, display: "block", color: "text.secondary" }}>Rotation</Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Slider size="small" value={lbl.rotation} min={-180} max={180} step={1}
                    marks={[{ value: -180 }, { value: -90 }, { value: 0 }, { value: 90 }, { value: 180 }]}
                    onChange={(_, v) => updateLabel(i, { rotation: v as number })}
                    sx={{ mx: 0.5 }}
                  />
                  <TextField type="number" size="small" value={lbl.rotation}
                    onChange={(e) => updateLabel(i, { rotation: Number(e.target.value) })}
                    inputProps={{ min: -360, max: 360 }}
                    sx={{ width: 68, flexShrink: 0, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }}
                  />
                  <Typography variant="caption" sx={{ flexShrink: 0 }}>&deg;</Typography>
                  <IconButton size="small" title="Reset rotation" onClick={() => updateLabel(i, { rotation: 0 })}>
                    <RestartAltIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Box>
              </Box>
            ))}
            <Button startIcon={<AddIcon />} onClick={addLabel} size="small" variant="outlined">Add Label</Button>
          </Box>
        </TabPanel>

        {/* -- Tab 3: Scale Bar --------------------------------- */}
        <TabPanel value={tabIdx} index={3 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
            <FormControlLabel
              sx={{ ml: 0, pl: 0.5 }}
              control={
                <Switch
                  checked={local.add_scale_bar}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    updateLocal({
                      add_scale_bar: enabled,
                      scale_bar: enabled && !local.scale_bar ? defaultScaleBar() : local.scale_bar,
                    });
                  }}
                />
              }
              label="Enable Scale Bar"
            />
            {local.add_scale_bar && local.scale_bar && (() => {
              const sb = local.scale_bar!;
              const unitLabels: Record<string, string> = { km: "km", m: "m", cm: "cm", mm: "mm", um: "\u00B5m", nm: "nm", pm: "pm" };
              const unitToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
              const unitLabel = unitLabels[sb.unit] || "\u00B5m";
              const umPerUnit = unitToUm[sb.unit] || 1;
              // Display bar length in current unit (internally stored in µm)
              const barLengthInUnit = sb.bar_length_microns / umPerUnit;
              const barPx = sb.bar_length_microns / Math.max(sb.micron_per_pixel, 1e-9);
              const barTooWide = origFullW > 0 && barPx > origFullW;
              const autoLabel = `${Number(barLengthInUnit.toPrecision(6))} ${unitLabel}`;
              // Predefined scale: extract unit from name (format: "name|unit")
              const isPredefined = !!sb.scale_name;
              const predefinedUnit = sb.scale_name ? (sb.scale_name.split("|")[1] || "um") : null;
              return (
              <>
                {barTooWide && (
                  <Alert severity="warning" sx={{ py: 0, fontSize: "0.75rem" }}>
                    Scale bar ({Math.round(barPx)}px) exceeds image width ({origFullW}px).
                  </Alert>
                )}

                {/* ── Scale Source ── */}
                <Typography variant="caption" sx={{ fontWeight: 600, color: "text.secondary" }}>Scale Source</Typography>
                <FormControl size="small" fullWidth>
                  <InputLabel sx={{ fontSize: "0.75rem" }}>Source</InputLabel>
                  <Select
                    value={sb.scale_name ?? ""}
                    label="Source"
                    onChange={(e) => {
                      const name = e.target.value;
                      if (name && config && config.resolution_entries[name]) {
                        // Extract unit from predefined name (format: "name|unit")
                        const predUnit = name.split("|")[1] || "um";
                        const predUmPerUnit = unitToUm[predUnit] || 1;
                        // Convert existing bar length to new unit
                        const barInNewUnit = sb.bar_length_microns / predUmPerUnit;
                        updateLocal({ scale_bar: { ...sb, scale_name: name, micron_per_pixel: config.resolution_entries[name], unit: predUnit } });
                      } else {
                        updateLocal({ scale_bar: { ...sb, scale_name: "" } });
                      }
                    }}
                    sx={{ fontSize: "0.75rem" }}
                  >
                    <MenuItem value="" sx={{ fontSize: "0.75rem" }}>Custom</MenuItem>
                    {config && Object.entries(config.resolution_entries).map(([name, val]) => {
                      const parts = name.split("|");
                      const displayName = parts[0];
                      const u = parts[1] || "um";
                      const uDisp = unitLabels[u] || "\u00B5m";
                      return <MenuItem key={name} value={name} sx={{ fontSize: "0.75rem" }}>{displayName} ({val} {uDisp}/px)</MenuItem>;
                    })}
                  </Select>
                </FormControl>

                {/* ── Measurement ── */}
                <Divider />
                <Typography variant="caption" sx={{ fontWeight: 600, color: "text.secondary" }}>Measurement</Typography>
                <Box sx={{ display: "flex", gap: 1 }}>
                  <TextField
                    label={`${unitLabel}/pixel`}
                    type="number"
                    value={Number((sb.micron_per_pixel / umPerUnit).toPrecision(6))}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, micron_per_pixel: Number(e.target.value) * umPerUnit, scale_name: "" } })}
                    inputProps={{ step: 0.01 }}
                    size="small"
                    fullWidth
                    disabled={!!sb.scale_name}
                    sx={{ "& input": { fontSize: "0.75rem" } }}
                  />
                  <TextField
                    label={`Bar length (${unitLabel})`}
                    type="number"
                    value={Number(barLengthInUnit.toPrecision(6))}
                    onChange={(e) => {
                      const valInUnit = Number(e.target.value);
                      updateLocal({ scale_bar: { ...sb, bar_length_microns: valInUnit * umPerUnit } });
                    }}
                    size="small"
                    fullWidth
                    sx={{ "& input": { fontSize: "0.75rem" } }}
                  />
                </Box>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <Typography variant="caption" sx={{ flexShrink: 0, width: 32, color: isPredefined ? "text.disabled" : "text.primary" }}>Unit</Typography>
                  <ToggleButtonGroup value={sb.unit ?? "um"} exclusive size="small"
                    disabled={isPredefined}
                    onChange={(_, v) => {
                      if (!v) return;
                      const oldUnit = sb.unit || "um";
                      const newUnit = v as string;
                      // bar_length_microns stays the same (it's in µm internally)
                      // Only update the unit — the display will auto-convert
                      const oldAutoLabel = `${Number((sb.bar_length_microns / umPerUnit).toPrecision(6))} ${unitLabel}`;
                      const shouldClearLabel = !sb.label || sb.label === oldAutoLabel;
                      updateLocal({ scale_bar: { ...sb, unit: newUnit, label: shouldClearLabel ? "" : sb.label } });
                    }}>
                    <ToggleButton value="km" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>km</ToggleButton>
                    <ToggleButton value="m" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>m</ToggleButton>
                    <ToggleButton value="cm" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>cm</ToggleButton>
                    <ToggleButton value="mm" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>mm</ToggleButton>
                    <ToggleButton value="um" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>{"\u00B5m"}</ToggleButton>
                    <ToggleButton value="nm" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>nm</ToggleButton>
                    <ToggleButton value="pm" sx={{ px: 0.75, py: 0.25, fontSize: "0.65rem" }}>pm</ToggleButton>
                  </ToggleButtonGroup>
                </Box>

                {/* ── Appearance ── */}
                <Divider />
                <Typography variant="caption" sx={{ fontWeight: 600, color: "text.secondary" }}>Appearance</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <TextField label="Bar height (px)" type="number" value={sb.bar_height}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, bar_height: Number(e.target.value) } })}
                    size="small" sx={{ flex: 1, "& input": { fontSize: "0.75rem" } }} />
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Typography variant="caption" sx={{ fontSize: "0.7rem" }}>Bar color</Typography>
                    <input type="color" value={sb.bar_color}
                      onChange={(e) => updateLocal({ scale_bar: { ...sb, bar_color: e.target.value } })}
                      style={{ width: 32, height: 32, border: "none", padding: 0, cursor: "pointer", borderRadius: 4 }} />
                  </Box>
                </Box>

                {/* ── Label ── */}
                <Divider />
                <Typography variant="caption" sx={{ fontWeight: 600, color: "text.secondary" }}>Label</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <TextField
                    label="Label text"
                    value={sb.label || autoLabel}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, label: e.target.value } })}
                    size="small" fullWidth
                    sx={{ "& input": { fontSize: "0.8rem" } }}
                  />
                  <Tooltip title="Label color">
                    <input type="color" value={sb.label_color || sb.bar_color}
                      onChange={(e) => updateLocal({ scale_bar: { ...sb, label_color: e.target.value } })}
                      style={{ width: 32, height: 32, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                  </Tooltip>
                </Box>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <FormControl size="small" sx={{ flex: 1 }}>
                    <Select value={sb.font_name} onChange={(e) => updateLocal({ scale_bar: { ...sb, font_name: e.target.value } })} sx={{ fontSize: "0.75rem" }}>
                      {fontList.slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map((f) => <MenuItem key={f} value={f} sx={{ fontSize: "0.75rem" }}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <TextField label="Size" type="number" value={sb.font_size}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, font_size: Number(e.target.value) } })}
                    size="small" inputProps={{ min: 4, max: 100 }}
                    sx={{ width: 72, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }} />
                </Box>
                <ToggleButtonGroup size="small" value={sb.label_font_style || []}
                  onChange={(_, v) => updateLocal({ scale_bar: { ...sb, label_font_style: v } })}>
                  <ToggleButton value="Bold" sx={{ px: 1, py: 0.25, fontWeight: 700, fontSize: "0.8rem" }}>B</ToggleButton>
                  <ToggleButton value="Italic" sx={{ px: 1, py: 0.25, fontStyle: "italic", fontSize: "0.8rem" }}>I</ToggleButton>
                  <ToggleButton value="Strikethrough" sx={{ px: 1, py: 0.25, textDecoration: "line-through", fontSize: "0.8rem" }}>S</ToggleButton>
                </ToggleButtonGroup>

                {/* ── Position ── */}
                <Divider />
                <Typography variant="caption" sx={{ fontWeight: 600, color: "text.secondary" }}>Position</Typography>
                <FormControl size="small" fullWidth>
                  <InputLabel sx={{ fontSize: "0.75rem" }}>Preset</InputLabel>
                  <Select value={sb.position_preset ?? "Bottom-Right"} label="Preset"
                    onChange={(e) => {
                      const preset = POSITION_PRESETS.find(p => p.label === e.target.value);
                      if (preset && preset.label !== "Custom") {
                        const d = sb.edge_distance ?? 5;
                        updateLocal({ scale_bar: { ...sb, position_preset: preset.label, position_x: preset.x(d), position_y: preset.y(d), bar_position: [preset.x(d) / 100, preset.y(d) / 100] } });
                      } else {
                        updateLocal({ scale_bar: { ...sb, position_preset: "Custom" } });
                      }
                    }}
                    sx={{ fontSize: "0.75rem" }}>
                    {POSITION_PRESETS.map(p => <MenuItem key={p.label} value={p.label} sx={{ fontSize: "0.75rem" }}>{p.label}</MenuItem>)}
                  </Select>
                </FormControl>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <TextField label="X %" type="number" value={Math.round(sb.position_x ?? 90)}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, position_x: Number(e.target.value), position_preset: "Custom", bar_position: [Number(e.target.value) / 100, (sb.position_y ?? 90) / 100] } })}
                    size="small" inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ flex: 1, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }} />
                  <TextField label="Y %" type="number" value={Math.round(sb.position_y ?? 90)}
                    onChange={(e) => updateLocal({ scale_bar: { ...sb, position_y: Number(e.target.value), position_preset: "Custom", bar_position: [(sb.position_x ?? 90) / 100, Number(e.target.value) / 100] } })}
                    size="small" inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ flex: 1, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center" } }} />
                </Box>
              </>
              );
            })()}
          </Box>
        </TabPanel>

        {/* -- Tab 4: Annotations ------------------------------- */}
        <TabPanel value={tabIdx} index={4 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            {/* Active editing mode indicator */}
            {selectedAnnotIdx && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1, py: 0.5, bgcolor: "primary.dark", borderRadius: 1, mb: 0.5 }}>
                <Typography variant="caption" sx={{ color: "#fff", fontWeight: 600, fontSize: "0.7rem" }}>
                  Editing: {selectedAnnotIdx.type === "symbol" ? `Annotation ${selectedAnnotIdx.idx + 1}` : selectedAnnotIdx.type === "line" ? `Line ${selectedAnnotIdx.idx + 1}` : `Area ${selectedAnnotIdx.idx + 1}`}
                </Typography>
                <Box sx={{ flex: 1 }} />
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.6)", fontSize: "0.6rem" }}>
                  {selectedAnnotIdx.type === "line" ? "Click preview to add points" : selectedAnnotIdx.type === "area" ? "Click preview to place area" : "Drag on preview to reposition"}
                </Typography>
                <Button size="small" sx={{ color: "#fff", fontSize: "0.6rem", minWidth: 0, p: 0.25 }} onClick={() => setSelectedAnnotIdx(null)}>Clear</Button>
              </Box>
            )}
            {local.symbols.map((sym, i) => {
              const isActive = selectedAnnotIdx?.type === "symbol" && selectedAnnotIdx.idx === i;
              return (
              <Box key={i} onClick={() => setSelectedAnnotIdx({ type: "symbol", idx: i })} sx={{ border: "1px solid", borderColor: isActive ? "primary.main" : "divider", borderRadius: 1, p: 1.5, cursor: "pointer", "&:hover": { borderColor: "primary.light" }, opacity: selectedAnnotIdx && !isActive ? 0.6 : 1, transition: "opacity 0.15s" }}>
                {/* Header row */}
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.75rem", color: isActive ? "primary.main" : "text.primary" }}>
                    {sym.name || `${sym.shape} Annotation ${i + 1}`} {isActive ? "(editing)" : "- click to edit"}
                  </Typography>
                  <IconButton onClick={(e) => { e.stopPropagation(); removeSymbol(i); }} size="small" sx={{ p: 0.25 }}>
                    <DeleteIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Box>
                {/* Controls only shown when active */}
                {isActive && <>
                {/* Name */}
                <TextField
                  label="Name"
                  value={sym.name || ""}
                  onChange={(e) => updateSymbol(i, { name: e.target.value })}
                  size="small"
                  fullWidth
                  placeholder={`Annotation ${i + 1}`}
                  sx={{ mt: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }}
                />
                {/* Shape selector */}
                <Box sx={{ mt: 1.5 }}>
                  <ToggleButtonGroup
                    value={sym.shape}
                    exclusive
                    onChange={(_, v) => { if (v !== null) updateSymbol(i, { shape: v }); }}
                    size="small"
                    sx={{ flexWrap: "wrap", "& .MuiToggleButton-root": { px: 1, py: 0.5 } }}
                  >
                    <ToggleButton value="Arrow" title="Arrow"><ArrowUpwardIcon sx={{ fontSize: 16 }} /></ToggleButton>
                    <ToggleButton value="NarrowTriangle" title="Narrow Triangle">{"\u25B4"}</ToggleButton>
                    <ToggleButton value="Star" title="Star"><StarIcon sx={{ fontSize: 16 }} /></ToggleButton>
                    <ToggleButton value="Rectangle" title="Rectangle"><CropSquareIcon sx={{ fontSize: 16 }} /></ToggleButton>
                    <ToggleButton value="Ellipse" title="Circle"><CircleIcon sx={{ fontSize: 16 }} /></ToggleButton>
                    <ToggleButton value="Cross" title="Cross"><CloseIcon sx={{ fontSize: 16 }} /></ToggleButton>
                    <ToggleButton value="Triangle" title="Triangle">{"\u25B2"}</ToggleButton>
                  </ToggleButtonGroup>
                  <input
                    type="color"
                    value={sym.color}
                    onChange={(e) => updateSymbol(i, { color: e.target.value })}
                    title="Symbol color"
                    style={{ width: 28, height: 28, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, marginLeft: 8, verticalAlign: "middle" }}
                  />
                </Box>

                {/* Position */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Position</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <TextField label="X %" type="number" value={Math.round(sym.x)} onChange={(e) => updateSymbol(i, { x: Number(e.target.value) })} size="small" inputProps={{ min: 0, max: 100, step: 1 }} sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                  <TextField label="Y %" type="number" value={Math.round(sym.y)} onChange={(e) => updateSymbol(i, { y: Number(e.target.value) })} size="small" inputProps={{ min: 0, max: 100, step: 1 }} sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                </Box>

                {/* Size & Rotation */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Size & Rotation</Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                  <Typography variant="caption" sx={{ width: 32, flexShrink: 0 }}>Size</Typography>
                  <Slider size="small" value={sym.size || 25} min={5} max={200} step={1} onChange={(_, v) => updateSymbol(i, { size: v as number })} sx={{ flex: 1, mx: 1 }} />
                  <TextField type="number" value={sym.size || 25} onChange={(e) => updateSymbol(i, { size: Number(e.target.value) || 25 })} size="small" inputProps={{ min: 1, max: 200 }} sx={{ width: 60, "& input": { fontSize: "0.8rem", px: 1, py: 0.75, textAlign: "center" } }} />
                  <IconButton size="small" title="Reset size" onClick={() => updateSymbol(i, { size: 25 })} sx={{ p: 0.25, ml: 0.5 }}><RestartAltIcon sx={{ fontSize: 14 }} /></IconButton>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="caption" sx={{ width: 32, flexShrink: 0 }}>Rot</Typography>
                  <Slider size="small" value={sym.rotation || 0} min={-180} max={180} step={1} onChange={(_, v) => updateSymbol(i, { rotation: v as number })} sx={{ flex: 1, mx: 1 }} />
                  <TextField type="number" value={sym.rotation || 0} onChange={(e) => updateSymbol(i, { rotation: Number(e.target.value) || 0 })} size="small" inputProps={{ min: -360, max: 360 }} sx={{ width: 60, "& input": { fontSize: "0.8rem", px: 1, py: 0.75, textAlign: "center" } }} />
                  <IconButton size="small" title="Reset rotation" onClick={() => updateSymbol(i, { rotation: 0 })} sx={{ p: 0.25, ml: 0.5 }}><RestartAltIcon sx={{ fontSize: 14 }} /></IconButton>
                </Box>

                {/* Label */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Label</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1 }}>
                  <TextField label="Text" value={sym.label_text} onChange={(e) => updateSymbol(i, { label_text: e.target.value })} size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                  <input type="color" value={sym.label_color} onChange={(e) => updateSymbol(i, { label_color: e.target.value })} title="Label color" style={{ width: 32, height: 32, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                </Box>
                {/* Label font */}
                <Typography variant="caption" sx={{ fontWeight: 600, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Font</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 1 }}>
                  <select
                    value={sym.label_font_name || "arial.ttf"}
                    onChange={(e) => updateSymbol(i, { label_font_name: e.target.value })}
                    style={{ flex: 1, fontSize: "0.8rem", padding: "8px 10px", background: "transparent", color: "inherit", border: "1px solid rgba(255,255,255,0.23)", borderRadius: 4, minWidth: 0 }}
                  >
                    {fontList.slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map(f => (
                      <option key={f} value={f}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</option>
                    ))}
                  </select>
                  <TextField
                    label="Size"
                    type="number"
                    value={sym.label_font_size}
                    onChange={(e) => updateSymbol(i, { label_font_size: Number(e.target.value) || 10 })}
                    size="small"
                    inputProps={{ min: 4, max: 100 }}
                    sx={{ width: 68, "& input": { fontSize: "0.8rem", px: 1, py: 0.75, textAlign: "center" } }}
                  />
                </Box>
                <Box sx={{ display: "flex", gap: 0.5 }}>
                  {["Bold", "Italic", "Strikethrough"].map(style => {
                    const active = (sym.label_font_style || []).includes(style);
                    return (
                      <Button
                        key={style}
                        size="small"
                        variant={active ? "contained" : "outlined"}
                        onClick={() => {
                          const styles = [...(sym.label_font_style || [])];
                          if (active) styles.splice(styles.indexOf(style), 1);
                          else styles.push(style);
                          updateSymbol(i, { label_font_style: styles });
                        }}
                        sx={{ minWidth: 32, px: 0.5, py: 0.25, fontSize: "0.7rem",
                              fontWeight: style === "Bold" ? 700 : 400,
                              fontStyle: style === "Italic" ? "italic" : "normal",
                              textDecoration: style === "Strikethrough" ? "line-through" : "none" }}
                      >
                        {style === "Bold" ? "B" : style === "Italic" ? "I" : "S"}
                      </Button>
                    );
                  })}
                </Box>
                {/* Label position hint */}
                {sym.label_text && (
                  <Typography variant="caption" sx={{ mt: 0.5, color: "text.secondary", fontSize: "0.65rem", fontStyle: "italic" }}>
                    Drag the label text on the preview to reposition
                  </Typography>
                )}
                </>}
              </Box>
              );
            })}
            <Button startIcon={<AddIcon />} onClick={addSymbol} size="small" variant="outlined">Add Annotation</Button>

            {/* ---- Line Annotations ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Lines</Typography>
            <Divider />
            {(local.lines ?? []).map((line, i) => {
              const isLineActive = selectedAnnotIdx?.type === "line" && selectedAnnotIdx.idx === i;
              return (
              <Box key={`line-${i}`} onClick={() => setSelectedAnnotIdx({ type: "line", idx: i })} sx={{ border: "2px solid", borderColor: isLineActive ? "primary.main" : "divider", borderRadius: 1, p: 1, mb: 0.5, cursor: "pointer", "&:hover": { borderColor: "primary.light" }, opacity: selectedAnnotIdx && !isLineActive ? 0.6 : 1, transition: "opacity 0.15s" }}>
                {!isLineActive ? (
                  <Typography variant="caption" sx={{ fontWeight: 600, color: "text.primary" }}>
                    {line.name || `Line ${i + 1}`} ({line.points?.length ?? 0} pts) - click to edit
                  </Typography>
                ) : <>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
                  <TextField label="Name" value={line.name} onClick={(e) => e.stopPropagation()} onChange={(e) => {
                    const lines = [...(local.lines ?? [])];
                    lines[i] = { ...lines[i], name: e.target.value };
                    updateLocal({ lines } as unknown as Partial<PanelInfo>);
                  }} size="small" sx={{ flex: 1, mr: 1, "& input": { fontSize: "0.7rem", py: 0.5 } }} />
                  <input type="color" value={line.color} onChange={(e) => {
                    const lines = [...(local.lines ?? [])];
                    lines[i] = { ...lines[i], color: e.target.value };
                    updateLocal({ lines } as unknown as Partial<PanelInfo>);
                  }} style={{ width: 24, height: 24, border: "none", padding: 0, cursor: "pointer", borderRadius: 4 }} />
                  <IconButton size="small" onClick={() => {
                    const lines = (local.lines ?? []).filter((_, idx) => idx !== i);
                    updateLocal({ lines } as unknown as Partial<PanelInfo>);
                  }}><DeleteIcon fontSize="small" /></IconButton>
                </Box>
                <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mb: 0.5 }}>
                  <AdjustSlider label="Width" value={line.width} defaultValue={2} min={0.5} max={10} step={0.5} onChange={(v) => {
                    const lines = [...(local.lines ?? [])];
                    lines[i] = { ...lines[i], width: v };
                    updateLocal({ lines } as unknown as Partial<PanelInfo>);
                  }} />
                </Box>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center", mb: 0.5 }}>
                  <FormControl size="small" sx={{ minWidth: 80 }}>
                    <Select value={line.dash_style} onChange={(e) => {
                      const lines = [...(local.lines ?? [])];
                      lines[i] = { ...lines[i], dash_style: e.target.value };
                      updateLocal({ lines } as unknown as Partial<PanelInfo>);
                    }} sx={{ fontSize: "0.7rem" }}>
                      <MenuItem value="solid" sx={{ fontSize: "0.7rem" }}>Solid</MenuItem>
                      <MenuItem value="dashed" sx={{ fontSize: "0.7rem" }}>Dashed</MenuItem>
                      <MenuItem value="dotted" sx={{ fontSize: "0.7rem" }}>Dotted</MenuItem>
                      <MenuItem value="dash-dot" sx={{ fontSize: "0.7rem" }}>Dash-Dot</MenuItem>
                    </Select>
                  </FormControl>
                  <ToggleButtonGroup size="small" value={line.line_type || (line.is_curved ? "curved" : "straight")} exclusive
                    onChange={(_, v) => {
                      if (!v) return;
                      const lines = [...(local.lines ?? [])];
                      let pts = lines[i].points ?? [];
                      // Straight: trim to 2 points max
                      if (v === "straight" && pts.length > 2) {
                        pts = [pts[0], pts[pts.length - 1]];
                      }
                      lines[i] = { ...lines[i], line_type: v, is_curved: v === "curved", points: pts };
                      updateLocal({ lines } as unknown as Partial<PanelInfo>);
                    }}>
                    <ToggleButton value="straight" sx={{ px: 0.5, py: 0, fontSize: "0.6rem" }}>
                      Straight
                    </ToggleButton>
                    <ToggleButton value="multijointed" sx={{ px: 0.5, py: 0, fontSize: "0.6rem" }}>
                      Multi-point
                    </ToggleButton>
                    <ToggleButton value="curved" sx={{ px: 0.5, py: 0, fontSize: "0.6rem" }}>
                      Smoothed
                    </ToggleButton>
                  </ToggleButtonGroup>
                  <FormControlLabel
                    control={<Checkbox size="small" checked={line.show_measure} onChange={(e) => {
                      const lines = [...(local.lines ?? [])];
                      lines[i] = { ...lines[i], show_measure: e.target.checked };
                      updateLocal({ lines } as unknown as Partial<PanelInfo>);
                    }} />}
                    label={<Typography variant="caption">Measure</Typography>}
                  />
                </Box>
                {/* Scale selection for measurement */}
                {line.show_measure && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mb: 0.5 }}>
                    <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                      <Typography variant="caption" sx={{ flexShrink: 0, fontSize: "0.6rem" }}>Scale:</Typography>
                      <FormControl size="small" sx={{ flex: 1 }}>
                        <Select
                          value={local.scale_bar?.scale_name ?? ""}
                          onChange={(e) => {
                            const name = e.target.value as string;
                            if (name && config && config.resolution_entries[name]) {
                              updateLocal({
                                scale_bar: {
                                  ...(local.scale_bar || { micron_per_pixel: 1, bar_length_microns: 100, bar_height: 5, bar_color: "#FFFFFF", bar_position: [0.9, 0.9] as [number, number], label: "", font_size: 10, font_name: "arial.ttf", font_path: "", label_x_offset: 0, label_font_style: [] as string[], position_preset: "Bottom-Right", position_x: 90, position_y: 90, edge_distance: 5, unit: "um", scale_name: "", styled_segments: [] as any[], draggable: false, label_color: "#FFFFFF" }),
                                  scale_name: name,
                                  micron_per_pixel: config.resolution_entries[name],
                                },
                                add_scale_bar: true,
                              });
                            } else {
                              // Custom selected
                              updateLocal({
                                scale_bar: {
                                  ...(local.scale_bar || { micron_per_pixel: 1, bar_length_microns: 100, bar_height: 5, bar_color: "#FFFFFF", bar_position: [0.9, 0.9] as [number, number], label: "", font_size: 10, font_name: "arial.ttf", font_path: "", label_x_offset: 0, label_font_style: [] as string[], position_preset: "Bottom-Right", position_x: 90, position_y: 90, edge_distance: 5, unit: "um", scale_name: "", styled_segments: [] as any[], draggable: false, label_color: "#FFFFFF" }),
                                  scale_name: "",
                                },
                              });
                            }
                          }}
                          sx={{ fontSize: "0.6rem" }}
                          displayEmpty
                        >
                          <MenuItem value="" sx={{ fontSize: "0.6rem" }}>Custom</MenuItem>
                          {config && Object.entries(config.resolution_entries).map(([name, val]) => (
                            <MenuItem key={name} value={name} sx={{ fontSize: "0.6rem" }}>{name} ({val} {"\u00B5m/px"})</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>
                    {/* Custom scale value input when no predefined scale selected */}
                    {!(local.scale_bar?.scale_name) && (
                      <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                        <TextField
                          label={`${local.scale_bar?.unit === "nm" ? "nm" : local.scale_bar?.unit === "mm" ? "mm" : "\u00B5m"}/px`}
                          type="number"
                          value={local.scale_bar?.micron_per_pixel ?? 1}
                          onChange={(e) => {
                            const val = Number(e.target.value);
                            if (val > 0) {
                              updateLocal({
                                scale_bar: {
                                  ...(local.scale_bar || { micron_per_pixel: 1, bar_length_microns: 100, bar_height: 5, bar_color: "#FFFFFF", bar_position: [0.9, 0.9] as [number, number], label: "", font_size: 10, font_name: "arial.ttf", font_path: "", label_x_offset: 0, label_font_style: [] as string[], position_preset: "Bottom-Right", position_x: 90, position_y: 90, edge_distance: 5, unit: "um", scale_name: "", styled_segments: [] as any[], draggable: false, label_color: "#FFFFFF" }),
                                  micron_per_pixel: val,
                                  scale_name: "",
                                },
                              });
                            }
                          }}
                          size="small"
                          inputProps={{ step: 0.001, min: 0.0001 }}
                          sx={{ flex: 1, "& input": { fontSize: "0.75rem", px: 1, py: 0.75 } }}
                        />
                      </Box>
                    )}
                  </Box>
                )}
                {/* Measurement text customization */}
                {line.show_measure && (
                  <Box sx={{ mt: 1, display: "flex", flexDirection: "column", gap: 1 }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.7rem" }}>Measurement Label</Typography>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      <TextField label="Text (auto if empty)" value={line.measure_text || ""} onChange={(e) => {
                        const lines2 = [...(local.lines ?? [])];
                        lines2[i] = { ...lines2[i], measure_text: e.target.value };
                        updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                      }} size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                      <input type="color" value={line.measure_color || "#FFFF00"} onChange={(e) => {
                        const lines2 = [...(local.lines ?? [])];
                        lines2[i] = { ...lines2[i], measure_color: e.target.value };
                        updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                      }} style={{ width: 32, height: 32, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      <select value={line.measure_font_name || "arial.ttf"} onChange={(e) => {
                        const lines2 = [...(local.lines ?? [])];
                        lines2[i] = { ...lines2[i], measure_font_name: e.target.value };
                        updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                      }} style={{ flex: 1, fontSize: "0.8rem", padding: "8px 10px", background: "transparent", color: "inherit", border: "1px solid rgba(255,255,255,0.23)", borderRadius: 4, minWidth: 0 }}>
                        {fontList.slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map(f => (
                          <option key={f} value={f}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</option>
                        ))}
                      </select>
                      <TextField label="Size" type="number" value={line.measure_font_size || 12} onChange={(e) => {
                        const lines2 = [...(local.lines ?? [])];
                        lines2[i] = { ...lines2[i], measure_font_size: Number(e.target.value) || 12 };
                        updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                      }} size="small" inputProps={{ min: 4, max: 100 }} sx={{ width: 68, "& input": { fontSize: "0.8rem", px: 1, py: 0.75, textAlign: "center" } }} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      {["Bold", "Italic", "Strikethrough"].map(style => {
                        const mfs = (line as any).measure_font_style || [];
                        const active = mfs.includes(style);
                        return (
                          <Button key={style} size="small" variant={active ? "contained" : "outlined"}
                            onClick={() => {
                              const styles = [...mfs];
                              if (active) styles.splice(styles.indexOf(style), 1); else styles.push(style);
                              const lines2 = [...(local.lines ?? [])];
                              lines2[i] = { ...lines2[i], measure_font_style: styles } as any;
                              updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                            }}
                            sx={{ minWidth: 32, px: 0.5, py: 0.25, fontSize: "0.7rem",
                              fontWeight: style === "Bold" ? 700 : 400,
                              fontStyle: style === "Italic" ? "italic" : "normal",
                              textDecoration: style === "Strikethrough" ? "line-through" : "none" }}
                          >{style === "Bold" ? "B" : style === "Italic" ? "I" : "S"}</Button>
                        );
                      })}
                    </Box>
                    <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem", fontStyle: "italic" }}>
                      Drag measurement text on preview to reposition
                    </Typography>
                  </Box>
                )}
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                  Points: {line.points?.length ?? 0} — Click on preview to add points
                  {line.points && line.points.length >= 2 && (line.line_type || "straight") === "straight" && " (2-point straight)"}
                  {line.points && line.points.length >= 2 && (line.line_type || "straight") === "multijointed" && " (multi-point discrete)"}
                  {line.points && line.points.length >= 2 && (line.line_type || "straight") === "curved" && " (multi-point smoothed)"}
                </Typography>
                </>}
              </Box>
              );
            })}
            <Button size="small" variant="outlined" onClick={() => {
              const lines = [...(local.lines ?? []), {
                name: `Line ${(local.lines ?? []).length + 1}`,
                points: [], color: "#FFFF00", width: 2, dash_style: "solid",
                line_type: "straight", is_curved: false, show_measure: false, measure_text: "",
                measure_unit: "um", measure_font_size: 12, measure_color: "#FFFF00",
                measure_font_name: config?.column_labels?.[0]?.font_name || "arial.ttf", measure_font_style: [],
                measure_styled_segments: [], measure_position_x: -1, measure_position_y: -1,
              }];
              updateLocal({ lines } as unknown as Partial<PanelInfo>);
            }}>Add Line</Button>

            {/* ---- Area Annotations ---- */}
            <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Areas</Typography>
            <Divider />
            {(local.areas ?? []).map((area, i) => {
              const isAreaActive = selectedAnnotIdx?.type === "area" && selectedAnnotIdx.idx === i;
              return (
              <Box key={`area-${i}`} onClick={() => setSelectedAnnotIdx({ type: "area", idx: i })} sx={{ border: "2px solid", borderColor: isAreaActive ? "primary.main" : "divider", borderRadius: 1, p: 1, mb: 0.5, cursor: "pointer", "&:hover": { borderColor: "primary.light" }, opacity: selectedAnnotIdx && !isAreaActive ? 0.6 : 1, transition: "opacity 0.15s" }}>
                {!isAreaActive ? (
                  <Typography variant="caption" sx={{ fontWeight: 600, color: "text.primary" }}>
                    {area.name || `Area ${i + 1}`} ({area.shape}) - click to edit
                  </Typography>
                ) : <>
                {/* Name + delete */}
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <TextField label="Name" value={area.name} onClick={(e) => e.stopPropagation()} onChange={(e) => {
                    const areas = [...(local.areas ?? [])];
                    areas[i] = { ...areas[i], name: e.target.value };
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }} size="small" sx={{ flex: 1, mr: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                  <IconButton size="small" onClick={() => {
                    const areas = (local.areas ?? []).filter((_, idx) => idx !== i);
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }}><DeleteIcon sx={{ fontSize: 16 }} /></IconButton>
                </Box>

                {/* Shape selector */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Shape</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <ToggleButtonGroup value={area.shape} exclusive size="small" onChange={(_, v) => {
                    if (!v) return;
                    const areas = [...(local.areas ?? [])];
                    // Clear points when switching shape type
                    areas[i] = { ...areas[i], shape: v, points: [] };
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }} sx={{ "& .MuiToggleButton-root": { px: 1, py: 0.5, fontSize: "0.7rem" } }}>
                    <ToggleButton value="Rectangle">Rect</ToggleButton>
                    <ToggleButton value="Ellipse">Ellipse</ToggleButton>
                    <ToggleButton value="Custom">Custom</ToggleButton>
                    <ToggleButton value="Magic">{"\u2728"} Magic</ToggleButton>
                  </ToggleButtonGroup>
                  <input type="color" value={area.color?.slice(0, 7) ?? "#FF0000"} onChange={(e) => {
                    const areas = [...(local.areas ?? [])];
                    areas[i] = { ...areas[i], color: e.target.value + "40", border_color: e.target.value };
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }} title="Fill color" style={{ width: 28, height: 28, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                </Box>

                {/* Magic wand: tolerance slider with live re-selection */}
                {area.shape === "Magic" && (
                  <Box sx={{ mt: 1 }}>
                    <AdjustSlider label="Tolerance" value={(area as any).magic_tolerance ?? 30} defaultValue={30} min={1} max={128} step={1} onChange={(v) => {
                      const areas = [...(local.areas ?? [])];
                      areas[i] = { ...areas[i], magic_tolerance: v } as any;
                      updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      // Re-run magic wand if we have a stored click point
                      const clickX = (area as any).magic_click_x;
                      const clickY = (area as any).magic_click_y;
                      if (clickX != null && clickY != null) {
                        setMagicWandLoading(true);
                        (async () => {
                          try {
                            const resp = await api.magicWandSelect(row, col, clickX, clickY, v, { rotation: local.rotation, crop: local.crop as number[] | undefined, crop_image: local.crop_image });
                            if (resp.points && resp.points.length >= 3) {
                              const areas2 = [...(local.areas ?? [])];
                              areas2[i] = { ...areas2[i], points: resp.points as [number, number][], smooth: true, magic_tolerance: v } as any;
                              updateLocal({ areas: areas2 } as unknown as Partial<PanelInfo>);
                            }
                          } catch (err) { console.error("Magic wand re-select failed:", err); }
                          finally { setMagicWandLoading(false); }
                        })();
                      }
                    }} />
                    <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", mt: 0.5, display: "block" }}>
                      {area.points.length > 0 ? `${area.points.length} boundary points — Adjust tolerance to refine` : "Click on preview to select region"}
                    </Typography>
                  </Box>
                )}

                {/* Custom polygon: smoothed vs multi-point toggle */}
                {area.shape === "Custom" && (
                  <Box sx={{ mt: 1 }}>
                    <ToggleButtonGroup value={(area as any).smooth ? "smooth" : "discrete"} exclusive size="small"
                      onChange={(_, v) => {
                        if (!v) return;
                        const areas = [...(local.areas ?? [])];
                        areas[i] = { ...areas[i], smooth: v === "smooth" } as any;
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      }}
                      sx={{ "& .MuiToggleButton-root": { px: 1, py: 0.25, fontSize: "0.65rem" } }}>
                      <ToggleButton value="discrete">Multi-point</ToggleButton>
                      <ToggleButton value="smooth">Smoothed</ToggleButton>
                    </ToggleButtonGroup>
                  </Box>
                )}

                {/* Border width */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Appearance</Typography>
                <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                  <AdjustSlider label="Border" value={area.border_width ?? 1} defaultValue={1} min={1} max={10} step={1} onChange={(v) => {
                    const areas = [...(local.areas ?? [])];
                    areas[i] = { ...areas[i], border_width: v };
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }} />
                </Box>
                <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mt: 0.5 }}>
                  <Typography variant="caption" sx={{ fontSize: "0.65rem", flexShrink: 0 }}>Style:</Typography>
                  <ToggleButtonGroup value={(area as any).dash_style || "solid"} exclusive size="small"
                    onChange={(_, v) => {
                      if (!v) return;
                      const areas = [...(local.areas ?? [])];
                      areas[i] = { ...areas[i], dash_style: v } as any;
                      updateLocal({ areas } as unknown as Partial<PanelInfo>);
                    }}
                    sx={{ "& .MuiToggleButton-root": { px: 0.75, py: 0.15, fontSize: "0.6rem", textTransform: "none" } }}>
                    <ToggleButton value="solid">Solid</ToggleButton>
                    <ToggleButton value="dashed">Dashed</ToggleButton>
                    <ToggleButton value="dotted">Dotted</ToggleButton>
                  </ToggleButtonGroup>
                </Box>

                {/* Measurement */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Measurement</Typography>
                <FormControlLabel sx={{ ml: 0 }}
                  control={<Checkbox size="small" checked={area.show_measure} onChange={(e) => {
                    const areas = [...(local.areas ?? [])];
                    areas[i] = { ...areas[i], show_measure: e.target.checked };
                    updateLocal({ areas } as unknown as Partial<PanelInfo>);
                  }} />}
                  label={<Typography variant="caption" sx={{ fontSize: "0.75rem" }}>Show measurement</Typography>}
                />
                {area.show_measure && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mt: 0.5 }}>
                    {/* Scale source */}
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                      <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                        <Typography variant="caption" sx={{ flexShrink: 0, fontSize: "0.7rem" }}>Scale:</Typography>
                        <FormControl size="small" sx={{ flex: 1 }}>
                          <Select
                            value={local.scale_bar?.scale_name ?? ""}
                            onChange={(e) => {
                              const name = e.target.value as string;
                              const defaultSb = { micron_per_pixel: 1, bar_length_microns: 100, bar_height: 5, bar_color: "#FFFFFF", bar_position: [0.9, 0.9] as [number, number], label: "", font_size: 10, font_name: "arial.ttf", font_path: "", label_x_offset: 0, label_font_style: [] as string[], position_preset: "Bottom-Right", position_x: 90, position_y: 90, edge_distance: 5, unit: "um", scale_name: "", styled_segments: [] as any[], draggable: false, label_color: "#FFFFFF" };
                              if (name && config && config.resolution_entries[name]) {
                                updateLocal({
                                  scale_bar: { ...(local.scale_bar || defaultSb), scale_name: name, micron_per_pixel: config.resolution_entries[name] },
                                  add_scale_bar: true,
                                });
                              } else {
                                updateLocal({ scale_bar: { ...(local.scale_bar || defaultSb), scale_name: "" } });
                              }
                            }}
                            sx={{ fontSize: "0.7rem" }}
                            displayEmpty
                          >
                            <MenuItem value="" sx={{ fontSize: "0.7rem" }}>Custom</MenuItem>
                            {config && Object.entries(config.resolution_entries).map(([name, val]) => (
                              <MenuItem key={name} value={name} sx={{ fontSize: "0.7rem" }}>{name} ({val} µm/px)</MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Box>
                      {!(local.scale_bar?.scale_name) && (
                        <TextField
                          label={`${local.scale_bar?.unit === "nm" ? "nm" : local.scale_bar?.unit === "mm" ? "mm" : "\u00B5m"}/px`}
                          type="number"
                          value={local.scale_bar?.micron_per_pixel ?? 1}
                          onChange={(e) => {
                            const val = Number(e.target.value);
                            if (val > 0) {
                              const defaultSb = { micron_per_pixel: 1, bar_length_microns: 100, bar_height: 5, bar_color: "#FFFFFF", bar_position: [0.9, 0.9] as [number, number], label: "", font_size: 10, font_name: "arial.ttf", font_path: "", label_x_offset: 0, label_font_style: [] as string[], position_preset: "Bottom-Right", position_x: 90, position_y: 90, edge_distance: 5, unit: "um", scale_name: "", styled_segments: [] as any[], draggable: false, label_color: "#FFFFFF" };
                              updateLocal({ scale_bar: { ...(local.scale_bar || defaultSb), micron_per_pixel: val, scale_name: "" } });
                            }
                          }}
                          size="small"
                          inputProps={{ step: 0.001, min: 0.0001 }}
                          sx={{ "& input": { fontSize: "0.75rem", px: 1, py: 0.75 } }}
                        />
                      )}
                    </Box>
                    {/* Unit selection */}
                    <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                      <Typography variant="caption" sx={{ flexShrink: 0, fontSize: "0.7rem" }}>Unit:</Typography>
                      <ToggleButtonGroup value={(area as any).measure_unit || local.scale_bar?.unit || "um"} exclusive size="small"
                        onChange={(_, v) => {
                          if (!v) return;
                          const areas = [...(local.areas ?? [])];
                          areas[i] = { ...areas[i], measure_unit: v } as any;
                          updateLocal({ areas } as unknown as Partial<PanelInfo>);
                        }}
                        sx={{ flexWrap: "wrap", "& .MuiToggleButton-root": { px: 0.75, py: 0.15, fontSize: "0.6rem", textTransform: "none" } }}>
                        <ToggleButton value="km">km{"\u00B2"}</ToggleButton>
                        <ToggleButton value="m">m{"\u00B2"}</ToggleButton>
                        <ToggleButton value="cm">cm{"\u00B2"}</ToggleButton>
                        <ToggleButton value="mm">mm{"\u00B2"}</ToggleButton>
                        <ToggleButton value="um">{"\u00B5"}m{"\u00B2"}</ToggleButton>
                        <ToggleButton value="nm">nm{"\u00B2"}</ToggleButton>
                      </ToggleButtonGroup>
                    </Box>

                    {/* Measurement label customization */}
                    <Typography variant="caption" sx={{ fontWeight: 600, mt: 1, mb: 0.5, display: "block", fontSize: "0.7rem" }}>Measurement Label</Typography>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      <TextField label="Text (auto if empty)" value={area.measure_text || ""} onChange={(e) => {
                        const areas = [...(local.areas ?? [])];
                        areas[i] = { ...areas[i], measure_text: e.target.value };
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      }} size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1.5, py: 1 } }} />
                      <input type="color" value={area.measure_color || "#FFFF00"} onChange={(e) => {
                        const areas = [...(local.areas ?? [])];
                        areas[i] = { ...areas[i], measure_color: e.target.value };
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      }} title="Text color" style={{ width: 28, height: 28, border: "none", padding: 0, cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      <select value={area.measure_font_name || "arial.ttf"} onChange={(e) => {
                        const areas = [...(local.areas ?? [])];
                        areas[i] = { ...areas[i], measure_font_name: e.target.value };
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      }} style={{ flex: 1, fontSize: "0.8rem", padding: "8px 10px", background: "transparent", color: "inherit", border: "1px solid rgba(255,255,255,0.23)", borderRadius: 4, minWidth: 0 }}>
                        {fontList.slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map(f => (
                          <option key={f} value={f}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</option>
                        ))}
                      </select>
                      <TextField label="Size" type="number" value={area.measure_font_size || 12} onChange={(e) => {
                        const areas = [...(local.areas ?? [])];
                        areas[i] = { ...areas[i], measure_font_size: Number(e.target.value) || 12 };
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                      }} size="small" inputProps={{ min: 4, max: 100 }} sx={{ width: 68, "& input": { fontSize: "0.8rem", px: 1, py: 0.75, textAlign: "center" } }} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      {["Bold", "Italic", "Strikethrough"].map(style => {
                        const mfs = (area as any).measure_font_style || [];
                        const active = mfs.includes(style);
                        return (
                          <Button key={style} size="small" variant={active ? "contained" : "outlined"}
                            onClick={() => {
                              const styles = [...mfs];
                              if (active) styles.splice(styles.indexOf(style), 1); else styles.push(style);
                              const areas = [...(local.areas ?? [])];
                              areas[i] = { ...areas[i], measure_font_style: styles } as any;
                              updateLocal({ areas } as unknown as Partial<PanelInfo>);
                            }}
                            sx={{ minWidth: 32, px: 0.5, py: 0.25, fontSize: "0.7rem",
                              fontWeight: style === "Bold" ? 700 : 400,
                              fontStyle: style === "Italic" ? "italic" : "normal",
                              textDecoration: style === "Strikethrough" ? "line-through" : "none" }}
                          >{style === "Bold" ? "B" : style === "Italic" ? "I" : "S"}</Button>
                        );
                      })}
                    </Box>
                    <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem", fontStyle: "italic" }}>
                      Drag measurement text on preview to reposition
                    </Typography>
                  </Box>
                )}

                {/* Instructions */}
                <Typography variant="caption" sx={{ mt: 1, color: "text.secondary", fontSize: "0.65rem", fontStyle: "italic" }}>
                  {area.shape === "Custom"
                    ? `Points: ${area.points?.length ?? 0} — Click preview to add vertices. Drag points to adjust.`
                    : "Click preview to place center, then click again to set size"}
                </Typography>
                </>}
              </Box>
              );
            })}
            <Button size="small" variant="outlined" onClick={() => {
              const areas = [...(local.areas ?? []), {
                name: `Area ${(local.areas ?? []).length + 1}`,
                shape: "Custom", points: [], color: "#FF000040", border_color: "#FF0000",
                border_width: 1, show_measure: true, measure_text: "",
                measure_unit: local.scale_bar?.unit || "um",
                measure_font_size: 12, measure_color: "#FF0000",
                measure_font_name: config?.column_labels?.[0]?.font_name || "arial.ttf",
                measure_styled_segments: [],
              }];
              updateLocal({ areas } as unknown as Partial<PanelInfo>);
            }}>Add Area</Button>
          </Box>
        </TabPanel>

        {/* -- Tab 5: Zoom Inset -------------------------------- */}
        <TabPanel value={tabIdx} index={5 + tOff}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <FormControlLabel
              sx={{ ml: 0, pl: 0.5 }}
              control={
                <Switch
                  size="small"
                  checked={local.add_zoom_inset}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    updateLocal({
                      add_zoom_inset: enabled,
                      zoom_inset: enabled && !local.zoom_inset ? defaultZoomInset() : local.zoom_inset,
                    });
                  }}
                />
              }
              label={<Typography variant="caption" sx={{ fontSize: "0.8rem" }}>Enable Zoom Inset</Typography>}
            />
            {local.add_zoom_inset && local.zoom_inset && (
              <Accordion
                defaultExpanded
                disableGutters
                sx={{
                  bgcolor: "transparent",
                  boxShadow: "none",
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: 1,
                  "&:before": { display: "none" },
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  sx={{
                    minHeight: 36,
                    "& .MuiAccordionSummary-content": { my: 0.5 },
                  }}
                >
                  {/* Title summarises the current inset so a collapsed
                      group still tells the user what it is. */}
                  <Typography variant="caption" sx={{ fontSize: "0.75rem", fontWeight: 600 }}>
                    Zoom Inset 1
                  </Typography>
                  <Typography variant="caption" sx={{ ml: 1, fontSize: "0.65rem", color: "text.secondary" }}>
                    {local.zoom_inset.inset_type}
                    {local.zoom_inset.inset_type === "Standard Zoom" && local.zoom_inset.zoom_factor
                      ? ` · ${local.zoom_inset.zoom_factor.toFixed(1)}×`
                      : ""}
                    {local.zoom_inset.inset_type === "Adjacent Panel" && local.zoom_inset.side
                      ? ` · ${local.zoom_inset.side}`
                      : ""}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ pt: 1, pb: 1.5, px: 1.5, display: "flex", flexDirection: "column", gap: 1.5 }}>
                <FormControl size="small" fullWidth>
                  <InputLabel>Inset Type</InputLabel>
                  <Select
                    value={local.zoom_inset.inset_type}
                    label="Inset Type"
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, inset_type: e.target.value },
                      })
                    }
                  >
                    <MenuItem value="Standard Zoom">Standard Zoom</MenuItem>
                    <MenuItem value="Adjacent Panel">Adjacent Panel</MenuItem>
                  </Select>
                </FormControl>

                {/* Zoom factor — hidden when external image is used in Standard Zoom */}
                {!(local.zoom_inset.inset_type === "Standard Zoom" && local.zoom_inset.separate_image_name && local.zoom_inset.separate_image_name !== "select") && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Zoom Factor: {local.zoom_inset.zoom_factor.toFixed(1)}x
                  </Typography>
                  <Slider
                    value={local.zoom_inset.zoom_factor}
                    min={1}
                    max={10}
                    step={0.5}
                    marks={[{ value: 1, label: "1x" }, { value: 5, label: "5x" }, { value: 10, label: "10x" }]}
                    onChange={(_, v) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, zoom_factor: v as number },
                      })
                    }
                  />
                </Box>
                )}

                {/* External image option */}
                <FormControlLabel
                  sx={{ ml: 0, mt: 1 }}
                  control={
                    <Switch
                      size="small"
                      checked={!!local.zoom_inset.separate_image_name}
                      onChange={(e) => {
                        updateLocal({
                          zoom_inset: {
                            ...local.zoom_inset!,
                            separate_image_name: e.target.checked ? "select" : "",
                          },
                        });
                      }}
                    />
                  }
                  label={<Typography variant="caption" sx={{ fontSize: "0.75rem" }}>Use external image</Typography>}
                />
                {!!local.zoom_inset.separate_image_name ? (
                  <Box sx={{ mb: 1 }}>
                    <FormControl size="small" fullWidth>
                      <InputLabel sx={{ fontSize: "0.75rem" }}>External Image</InputLabel>
                      <Select
                        value={local.zoom_inset.separate_image_name || ""}
                        label="External Image"
                        onChange={(e) =>
                          updateLocal({
                            zoom_inset: { ...local.zoom_inset!, separate_image_name: e.target.value },
                          })
                        }
                        sx={{ fontSize: "0.75rem" }}
                      >
                        <MenuItem value="" sx={{ fontSize: "0.75rem" }}>Select image...</MenuItem>
                        {imageNames.map((name) => (
                          <MenuItem key={name} value={name} sx={{ fontSize: "0.75rem" }}>{name}</MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem", mt: 0.5, display: "block" }}>
                      The external image replaces the zoomed content (e.g., higher magnification view)
                    </Typography>
                    {/* External image preview and crop area */}
                    {local.zoom_inset.separate_image_name && local.zoom_inset.separate_image_name !== "select" && (() => {
                      const extName = local.zoom_inset.separate_image_name;
                      if (!extImageThumb) return <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem" }}>Loading preview...</Typography>;
                      return (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.7rem" }}>External Image Preview</Typography>
                          <Box sx={{ position: "relative", mt: 0.5, border: "1px solid", borderColor: "divider", borderRadius: 1, overflow: "hidden" }}>
                            <img src={`data:image/png;base64,${extImageThumb}`} alt="External" style={{ width: "100%", display: "block" }}
                              onLoad={(e) => {
                                // Store natural dimensions for coordinate mapping
                                const img = e.currentTarget;
                                img.dataset.natW = String(img.naturalWidth);
                                img.dataset.natH = String(img.naturalHeight);
                              }}
                            />
                            {/* Crop rectangle overlay */}
                            {(() => {
                              const ezi = local.zoom_inset;
                              const xi = ezi.x_inset ?? 0;
                              const yi = ezi.y_inset ?? 0;
                              const wi = ezi.width_inset ?? 100;
                              const hi = ezi.height_inset ?? 100;
                              const fullW = extImageDims.w;
                              const fullH = extImageDims.h;
                              const xPct = (xi / fullW) * 100;
                              const yPct = (yi / fullH) * 100;
                              const wPct = (wi / fullW) * 100;
                              const hPct = (hi / fullH) * 100;
                              return (
                                <Box
                                  sx={{
                                    position: "absolute",
                                    left: `${xPct}%`, top: `${yPct}%`,
                                    width: `${wPct}%`, height: `${hPct}%`,
                                    border: "2px solid #4FC3F7",
                                    cursor: "move",
                                    boxSizing: "border-box",
                                    "&::after": {
                                      content: '""', position: "absolute", inset: 0,
                                      bgcolor: "rgba(0,0,0,0.15)",
                                    },
                                  }}
                                  onMouseDown={(e) => {
                                    e.preventDefault();
                                    const imgEl = e.currentTarget.parentElement?.querySelector("img") as HTMLImageElement;
                                    if (!imgEl) return;
                                    const dispW = imgEl.clientWidth;
                                    const dispH = imgEl.clientHeight;
                                    const fW = fullW;
                                    const fH = fullH;
                                    const startX = e.clientX, startY = e.clientY;
                                    const startXi = xi, startYi = yi;
                                    const onMove = (ev: MouseEvent) => {
                                      const dx = ((ev.clientX - startX) / dispW) * fW;
                                      const dy = ((ev.clientY - startY) / dispH) * fH;
                                      const newX = Math.max(0, Math.min(fW - wi, Math.round(startXi + dx)));
                                      const newY = Math.max(0, Math.min(fH - hi, Math.round(startYi + dy)));
                                      updateLocal({ zoom_inset: { ...local.zoom_inset!, x_inset: newX, y_inset: newY } });
                                    };
                                    const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                    window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                                  }}
                                >
                                  {/* Corner resize handles */}
                                  {["nw","ne","sw","se"].map(corner => (
                                    <Box key={corner} sx={{
                                      position: "absolute",
                                      width: 10, height: 10, bgcolor: "#4FC3F7", border: "1px solid #fff",
                                      ...(corner.includes("n") ? { top: -5 } : { bottom: -5 }),
                                      ...(corner.includes("w") ? { left: -5 } : { right: -5 }),
                                      cursor: `${corner}-resize`, zIndex: 2,
                                    }}
                                    onMouseDown={(ev) => {
                                      ev.preventDefault(); ev.stopPropagation();
                                      const container = ev.currentTarget.parentElement?.parentElement;
                                      const imgEl2 = container?.querySelector("img") as HTMLImageElement;
                                      if (!imgEl2) return;
                                      const rect2 = imgEl2.getBoundingClientRect();
                                      const startMX = ev.clientX, startMY = ev.clientY;
                                      const sXi = xi, sYi = yi, sWi = wi, sHi = hi;
                                      const onMove2 = (em: MouseEvent) => {
                                        const dx = ((em.clientX - startMX) / rect2.width) * fullW;
                                        const dy = ((em.clientY - startMY) / rect2.height) * fullH;
                                        let nX = sXi, nY = sYi, nW = sWi, nH = sHi;
                                        if (corner.includes("e")) nW = Math.max(20, Math.round(sWi + dx));
                                        if (corner.includes("w")) { nX = Math.max(0, Math.round(sXi + dx)); nW = Math.max(20, Math.round(sWi - dx)); }
                                        if (corner.includes("s")) nH = Math.max(20, Math.round(sHi + dy));
                                        if (corner.includes("n")) { nY = Math.max(0, Math.round(sYi + dy)); nH = Math.max(20, Math.round(sHi - dy)); }
                                        updateLocal({ zoom_inset: { ...local.zoom_inset!, x_inset: nX, y_inset: nY, width_inset: nW, height_inset: nH } });
                                      };
                                      const onUp2 = () => { window.removeEventListener("mousemove", onMove2); window.removeEventListener("mouseup", onUp2); };
                                      window.addEventListener("mousemove", onMove2); window.addEventListener("mouseup", onUp2);
                                    }}
                                    />
                                  ))}
                                </Box>
                              );
                            })()}
                          </Box>
                          <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.7rem", mt: 1, display: "block" }}>External Crop Area</Typography>
                          <Box sx={{ display: "flex", gap: 1 }}>
                            <TextField label="X" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.x_inset ?? 0}
                              onChange={(e) => updateLocal({ zoom_inset: { ...local.zoom_inset!, x_inset: Number(e.target.value) } })}
                              size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                            <TextField label="Y" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.y_inset ?? 0}
                              onChange={(e) => updateLocal({ zoom_inset: { ...local.zoom_inset!, y_inset: Number(e.target.value) } })}
                              size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                            <TextField label="W" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.width_inset ?? 100}
                              onChange={(e) => updateLocal({ zoom_inset: { ...local.zoom_inset!, width_inset: Number(e.target.value) } })}
                              size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                            <TextField label="H" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.height_inset ?? 100}
                              onChange={(e) => updateLocal({ zoom_inset: { ...local.zoom_inset!, height_inset: Number(e.target.value) } })}
                              size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                          </Box>
                          <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.55rem", fontStyle: "italic" }}>
                            Crop area in pixels on the external image. Leave default (0,0,full) for entire image.
                          </Typography>
                        </Box>
                      );
                    })()}
                  </Box>
                ) : null}

                {/* Both Standard Zoom and Adjacent Panel use the SAME
                    rectangle on the source image; the historical name
                    "Zoom Area" was confusingly different from the
                    "Source Area" label used in the Adjacent Panel
                    branch. Unified to "Source Area" so the two
                    branches don't look like duplicate but unrelated
                    sets of fields. */}
                <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Source Area</Typography>
                <Divider />
                <Box sx={{ display: "flex", gap: 1 }}>
                  <TextField
                    label="X"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.x}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, x: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ step: 1, min: 0, max: 9999 }}
                    fullWidth
                  />
                  <TextField
                    label="Y"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.y}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, y: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ step: 1, min: 0, max: 9999 }}
                    fullWidth
                  />
                  <TextField
                    label="W"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.width}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, width: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ step: 1, min: 0, max: 9999 }}
                    fullWidth
                  />
                  <TextField
                    label="H"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.height}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, height: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ step: 1, min: 0, max: 9999 }}
                    fullWidth
                  />
                </Box>

                <Typography variant="caption" sx={{ fontWeight: 600, mt: 2 }}>Styling</Typography>
                <Divider />
                <Box sx={{ display: "flex", gap: 1 }}>
                  <TextField
                    label="Rect color"
                    type="color"
                    value={local.zoom_inset.rectangle_color}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, rectangle_color: e.target.value },
                      })
                    }
                    size="small"
                    sx={{ width: 80, "& input": { cursor: "pointer", p: 0.5 } }}
                  />
                  <TextField
                    label="Line color"
                    type="color"
                    value={local.zoom_inset.line_color}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, line_color: e.target.value },
                      })
                    }
                    size="small"
                    sx={{ width: 80, "& input": { cursor: "pointer", p: 0.5 } }}
                  />
                </Box>

                <Box sx={{ display: "flex", gap: 1 }}>
                  <TextField
                    label="Rect width"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.rectangle_width}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, rectangle_width: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ min: 0, max: 10 }}
                    fullWidth
                  />
                  <TextField
                    label="Line width"
                    type="number"
                    onFocus={(e) => (e.target as HTMLInputElement).select()}
                    value={local.zoom_inset.line_width}
                    onChange={(e) =>
                      updateLocal({
                        zoom_inset: { ...local.zoom_inset!, line_width: Number(e.target.value) },
                      })
                    }
                    size="small"
                    inputProps={{ min: 0, max: 10 }}
                    fullWidth
                  />
                </Box>

                {/* Adjacent Panel options */}
                {local.zoom_inset.inset_type === "Adjacent Panel" && (
                  <>
                    <FormControl size="small" fullWidth>
                      <InputLabel>Target Side</InputLabel>
                      <Select
                        value={local.zoom_inset.side}
                        label="Target Side"
                        onChange={(e) =>
                          updateLocal({
                            zoom_inset: { ...local.zoom_inset!, side: e.target.value },
                          })
                        }
                      >
                        {(() => {
                          const sides = [
                            { value: "Top", dr: -1, dc: 0 },
                            { value: "Bottom", dr: 1, dc: 0 },
                            { value: "Left", dr: 0, dc: -1 },
                            { value: "Right", dr: 0, dc: 1 },
                          ];
                          return sides.map(s => {
                            const tr = row + s.dr;
                            const tc = col + s.dc;
                            const inBounds = tr >= 0 && tr < (config?.rows ?? 0) && tc >= 0 && tc < (config?.cols ?? 0);
                            const hasImage = inBounds && config?.panels?.[tr]?.[tc]?.image_name;
                            const disabled = !inBounds || !!hasImage;
                            return (
                              <MenuItem key={s.value} value={s.value} disabled={disabled}>
                                {s.value}{!inBounds ? " (out of grid)" : hasImage ? " (occupied)" : ""}
                              </MenuItem>
                            );
                          });
                        })()}
                      </Select>
                    </FormControl>
                    {(() => {
                      const s = local.zoom_inset.side || "Right";
                      const dr = s === "Top" ? -1 : s === "Bottom" ? 1 : 0;
                      const dc = s === "Left" ? -1 : s === "Right" ? 1 : 0;
                      const tr = row + dr, tc = col + dc;
                      const inBounds = tr >= 0 && tr < (config?.rows ?? 0) && tc >= 0 && tc < (config?.cols ?? 0);
                      const hasImage = inBounds && config?.panels?.[tr]?.[tc]?.image_name;
                      if (!inBounds) return <Typography variant="caption" color="error" sx={{ fontSize: "0.65rem" }}>Target panel is outside the grid</Typography>;
                      if (hasImage) return <Typography variant="caption" color="error" sx={{ fontSize: "0.65rem" }}>Target panel is occupied — clear it first</Typography>;
                      return null;
                    })()}

                    {/* Zoom factor */}
                    <Typography variant="caption" sx={{ mt: 1 }}>
                      Zoom Factor: {local.zoom_inset.zoom_factor.toFixed(1)}x
                    </Typography>
                    <Slider
                      value={local.zoom_inset.zoom_factor}
                      min={1} max={10} step={0.5}
                      marks={[{ value: 1, label: "1x" }, { value: 5, label: "5x" }, { value: 10, label: "10x" }]}
                      onChange={(_, val) =>
                        updateLocal({ zoom_inset: { ...local.zoom_inset!, zoom_factor: val as number } })
                      }
                      sx={{ mt: 0 }}
                    />

                    {/* Source area */}
                    <Typography variant="caption" sx={{ fontWeight: 600, mt: 1 }}>Source Area</Typography>
                    <Box sx={{ display: "flex", gap: 1 }}>
                      <TextField label="X" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.x} onChange={(e) =>
                        updateLocal({ zoom_inset: { ...local.zoom_inset!, x: Number(e.target.value) } })
                      } size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                      <TextField label="Y" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.y} onChange={(e) =>
                        updateLocal({ zoom_inset: { ...local.zoom_inset!, y: Number(e.target.value) } })
                      } size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                      <TextField label="W" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.width} onChange={(e) =>
                        updateLocal({ zoom_inset: { ...local.zoom_inset!, width: Number(e.target.value) } })
                      } size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                      <TextField label="H" type="number" onFocus={(e) => (e.target as HTMLInputElement).select()} value={local.zoom_inset.height} onChange={(e) =>
                        updateLocal({ zoom_inset: { ...local.zoom_inset!, height: Number(e.target.value) } })
                      } size="small" sx={{ flex: 1, "& input": { fontSize: "0.8rem", px: 1, py: 0.75 } }} />
                    </Box>
                    <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem", fontStyle: "italic" }}>
                      Drag the source rectangle on the preview to reposition
                    </Typography>
                  </>
                )}

                {/* Separate Image options */}
                </AccordionDetails>
              </Accordion>
            )}
          </Box>
        </TabPanel>
          </Box>{/* end left controls panel */}

          {/* -- Right: Live Preview with draggable labels ----------- */}
          <Box sx={{ flex: 1, display: "flex", alignItems: "flex-start", justifyContent: "center", p: 1, bgcolor: "action.hover", borderRadius: 2, border: "1px solid", borderColor: "divider", maxHeight: "70vh", overflow: "hidden" }}>
            {previewB64 ? (
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", width: "100%", height: "100%" }}>
                {/* Wrapper sized via CSS aspect-ratio to the rendered
                    preview's natural dimensions. Combined with width:100%
                    and max-height:68vh, the browser fills the available
                    area (height- or width-bound, whichever bites first)
                    while preserving aspect — so a small crop result is
                    blown up to fill the pane instead of rendering at its
                    tiny natural size. Overlays inside continue to use
                    percentage-based positioning, which now maps to the
                    enlarged image footprint correctly. */}
                <Box sx={{
                  position: "relative",
                  width: "100%",
                  maxWidth: "100%",
                  maxHeight: "68vh",
                  aspectRatio: previewNatW > 0 && previewNatH > 0
                    ? `${previewNatW} / ${previewNatH}`
                    : undefined,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}>
                {(() => {
                  // Use PIL-rendered preview on Adjustments, Labels, Scale Bar tabs.
                  // Annotations tab (4) uses base preview + SVG overlays for responsive drag.
                  // Crop tab uses base preview for crop handles.
                  // Zoom Inset tab (5) uses rendered preview to show all overlays.
                  const useRendered = [TAB_ADJ, TAB_LABELS, TAB_SCALE, TAB_ZOOM].includes(tabIdx) && renderedPreviewB64;
                  return (
                    <Box
                      component="img"
                      src={`data:image/png;base64,${useRendered ? renderedPreviewB64 : previewB64}`}
                      alt="Panel preview"
                      onLoad={(e) => {
                        const img = e.currentTarget as HTMLImageElement;
                        if (img.naturalWidth && img.naturalHeight) {
                          setPreviewNatW(img.naturalWidth);
                          setPreviewNatH(img.naturalHeight);
                        }
                      }}
                      sx={{
                        // Fill the aspect-ratio'd wrapper. objectFit:contain
                        // is a safety net — since the wrapper aspect matches
                        // the image, no letterboxing actually occurs.
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                        borderRadius: 1,
                      }}
                      draggable={false}
                    />
                  );
                })()}
                {/* Draggable label indicators overlay — matches actual rendering */}
                {/* Label overlays — interactive on Labels tab, visible but non-interactive on Scale Bar tab */}
                {[TAB_LABELS, TAB_SCALE].includes(tabIdx) && local.labels.map((lbl, i) => {
                  const isLabelsTab = tabIdx === TAB_LABELS;
                  const isPreset = (lbl.position_preset ?? "Custom") !== "Custom";
                  const isDraggable = isLabelsTab && !isPreset;
                  const isBold = lbl.font_style?.includes("Bold");
                  const isItalic = lbl.font_style?.includes("Italic");
                  const hasStrike = lbl.font_style?.includes("Strikethrough");
                  const hasSub = lbl.font_style?.includes("Subscript");
                  const hasSup = lbl.font_style?.includes("Superscript");
                  // Scale font size for preview display.
                  // In the final output at 300 DPI, font_size pts = font_size/72*300 pixels
                  // on the full processed image. The preview displays at clientWidth pixels.
                  // Match matplotlib: font_size pts on a 3-inch (216pt) panel.
                  // CSS pixels = font_size / 216 * displayed_width
                  const previewEl = document.querySelector('[alt="Panel preview"]') as HTMLImageElement | null;
                  const previewW = previewEl?.clientWidth || 400;
                  const scaledSize = Math.max(6, lbl.font_size * previewW / 216);
                  // Map font_name to web-safe font family
                  const fontFamilyMap: Record<string, string> = {
                    "arial.ttf": "Arial, sans-serif",
                    "times.ttf": "Times New Roman, serif",
                    "cour.ttf": "Courier New, monospace",
                    "verdana.ttf": "Verdana, sans-serif",
                  };
                  const fontFamily = fontFamilyMap[lbl.font_name] || lbl.font_name.replace(/\.ttf$/i, "") + ", sans-serif";
                  // When rendered preview is active, labels are baked in.
                  // CSS overlays serve as invisible drag hitboxes only —
                  // EXCEPT the currently-selected label, which we keep
                  // visible so the user sees what they are dragging (fixes
                  // the "new label invisible while dragging" bug).
                  const hasRendered = !!renderedPreviewB64;
                  const isSelected = selectedLabelIdx === i;
                  const hideText = hasRendered && !isSelected;
                  return (
                    <Box
                      key={`label-${i}`}
                      sx={{
                        position: "absolute",
                        left: `${lbl.position_x}%`,
                        top: `${lbl.position_y}%`,
                        transform: `rotate(${lbl.rotation || 0}deg)`,
                        cursor: isDraggable ? "grab" : "default",
                        userSelect: "none",
                        px: 0.5, py: 0.25,
                        bgcolor: isDraggable && isSelected ? "rgba(33,150,243,0.3)" : "transparent",
                        color: hideText ? "transparent" : (lbl.color || "#fff"),
                        borderRadius: 0.5,
                        fontSize: `${scaledSize}px`,
                        fontFamily: fontFamily,
                        fontWeight: isBold ? 700 : 400,
                        fontStyle: isItalic ? "italic" : "normal",
                        textDecoration: hasStrike ? "line-through" : "none",
                        verticalAlign: hasSup ? "super" : hasSub ? "sub" : "baseline",
                        border: isLabelsTab && isSelected ? "1px solid rgba(255,255,255,0.5)" : "1px solid transparent",
                        whiteSpace: "nowrap",
                        zIndex: isLabelsTab && isSelected ? 10 : 1,
                        textShadow: hideText ? "none" : "0 1px 3px rgba(0,0,0,0.8)",
                        pointerEvents: isDraggable ? "auto" : "none",
                        "&:hover": isDraggable ? { border: "1px solid rgba(255,255,255,0.5)" } : {},
                      }}
                      onMouseDown={isDraggable ? (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setSelectedLabelIdx(i);
                        const imgEl = (e.currentTarget.parentElement?.querySelector("img")) as HTMLImageElement | null;
                        if (!imgEl) return;
                        const onMove = (ev: MouseEvent) => {
                          ev.preventDefault();
                          const rect = imgEl.getBoundingClientRect();
                          const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                          const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                          updateLabel(i, { position_x: Math.round(px), position_y: Math.round(py), position_preset: "Custom" });
                        };
                        const onUp = () => {
                          window.removeEventListener("mousemove", onMove);
                          window.removeEventListener("mouseup", onUp);
                          setTimeout(() => refreshPreview(), 0);
                        };
                        window.addEventListener("mousemove", onMove);
                        window.addEventListener("mouseup", onUp);
                      } : undefined}
                      onClick={() => setSelectedLabelIdx(i)}
                    >
                      {lbl.text || `L${i + 1}`}
                    </Box>
                  );
                })}
                {/* Scale bar CSS overlay: only on Crop tab when no rendered preview, hidden elsewhere */}
                {tabIdx !== TAB_CROP && !([TAB_ADJ, TAB_LABELS, TAB_SCALE, TAB_ANNOT].includes(tabIdx) && renderedPreviewB64) && local.add_scale_bar && local.scale_bar && (() => {
                  const isScaleBarTab = tabIdx === TAB_SCALE;
                  const sb = local.scale_bar!;
                  const edgePct = sb.edge_distance ?? 5;
                  let posXPct = sb.position_x ?? 90;
                  let posYPct = sb.position_y ?? 90;
                  // Pre-compute bar dimensions for position calculations
                  const barPx = sb.bar_length_microns / Math.max(sb.micron_per_pixel || 1, 1e-9);
                  const sbPrevEl = document.querySelector('[alt="Panel preview"]') as HTMLImageElement | null;
                  const sbDispW = sbPrevEl?.clientWidth || 400;
                  const sbDispH = sbPrevEl?.clientHeight || 400;
                  const sbActW = origFullW > 0 ? origFullW : 1000;
                  const sbScale = sbDispW / sbActW;
                  // For presets, override with calculated positions that keep bar inside
                  if (sb.position_preset === "Bottom-Right") {
                    posXPct = 100 - edgePct;
                    posYPct = 100 - edgePct;
                  } else if (sb.position_preset === "Bottom-Left") {
                    posXPct = edgePct;
                    posYPct = 100 - edgePct;
                  } else if (sb.position_preset === "Top-Right") {
                    posXPct = 100 - edgePct;
                    posYPct = edgePct;
                  } else if (sb.position_preset === "Top-Left") {
                    posXPct = edgePct;
                    posYPct = edgePct;
                  }
                  const isBottom = posYPct > 50;
                  const isRight = posXPct > 50;
                  // Position matching backend: bar right edge at (100-edge)%,
                  // bar bottom at (100-edge)% - 5px offset
                  const barPxDisp = barPx * sbScale;
                  const barHDisp = sb.bar_height * sbScale;
                  const barPct = (barPxDisp / sbDispW) * 100;

                  // Use right/bottom anchoring for bottom-right presets (matches backend clamping)
                  // Position matching backend pixel formula exactly:
                  // Backend: bx = iw*(1-edge) - bar_length (Right), iw*edge (Left)
                  //          by = ih*(1-edge) - bar_height - 5 (Bottom), ih*edge + 5 (Top)
                  // CSS uses left% for bar's left edge position
                  // Position using bar CENTER point, then center the box on it.
                  // Backend: bx = iw*(1-edge)-bar_length (Right) → bar center at (1-edge) - bar_length/2
                  //          by = ih*(1-edge)-bar_height-5 (Bottom) → bar center at that + bar_height/2
                  const posStyle: Record<string, string> = {};
                  let centerXPct: number, centerYPct: number;
                  if (!sb.position_preset || sb.position_preset === "Custom") {
                    centerXPct = posXPct;
                    centerYPct = posYPct;
                  } else {
                    if (isRight) {
                      // bar center X = (100-edge)% - barWidth/2%
                      centerXPct = 100 - edgePct - barPct / 2;
                    } else {
                      centerXPct = edgePct + barPct / 2;
                    }
                    if (isBottom) {
                      const barHPct = barHDisp / sbDispH * 100;
                      const offsetPct = 5 * sbScale / sbDispH * 100;
                      centerYPct = 100 - edgePct - barHPct / 2 - offsetPct;
                    } else {
                      const offsetPct = 5 * sbScale / sbDispH * 100;
                      centerYPct = edgePct + offsetPct + (barHDisp / sbDispH * 100) / 2;
                    }
                  }
                  posStyle.left = `${centerXPct}%`;
                  posStyle.top = `${centerYPct}%`;
                  const transformStr = `translate(-50%, ${isBottom ? "0%" : "-50%"})`;
                  return (
                  <Box
                    sx={{
                      position: "absolute",
                      ...posStyle,
                      transform: transformStr,
                      cursor: isScaleBarTab && (sb.position_preset ?? "Bottom-Right") === "Custom" ? "grab" : "default",
                      pointerEvents: isScaleBarTab ? "auto" : "none",
                      userSelect: "none",
                      display: "flex",
                      flexDirection: isBottom ? "column-reverse" : "column",
                      alignItems: "center",  // center text over bar, matching PIL rendering
                      gap: 0.25,
                      zIndex: 5,
                    }}
                    onMouseDown={(local.scale_bar.position_preset ?? "Bottom-Right") === "Custom" ? (e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      const imgEl = (e.currentTarget.parentElement?.querySelector("img")) as HTMLImageElement | null;
                      if (!imgEl) return;
                      const onMove = (ev: MouseEvent) => {
                        ev.preventDefault();
                        const rect = imgEl.getBoundingClientRect();
                        const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                        const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                        updateLocal({ scale_bar: { ...local.scale_bar!, position_x: Math.round(px), position_y: Math.round(py), position_preset: "Custom", bar_position: [px / 100, py / 100] } });
                      };
                      const onUp = () => {
                        window.removeEventListener("mousemove", onMove);
                        window.removeEventListener("mouseup", onUp);
                        setTimeout(() => refreshPreview(), 0);
                      };
                      window.addEventListener("mousemove", onMove);
                      window.addEventListener("mouseup", onUp);
                    } : undefined}
                  >
                    {(() => {
                      // Use pre-computed outer scope variables for consistency
                      const barW = Math.max(8, barPx * sbScale);
                      const barH = Math.max(2, sb.bar_height * sbScale);
                      // Match matplotlib: font_size pts on 3-inch (216pt) panel
                      const fontSize = Math.max(6, (sb.font_size || 10) * sbDispW / 216);
                      return (
                        <>
                          <Box sx={{ width: barW, height: barH, bgcolor: local.scale_bar!.bar_color, borderRadius: 0.25 }} />
                          <Typography sx={{ fontSize: `${fontSize}px`, color: local.scale_bar!.label_color || local.scale_bar!.bar_color, textShadow: "0 1px 2px rgba(0,0,0,0.8)", whiteSpace: "nowrap" }}>
                            {local.scale_bar!.label || (() => {
                              const sbU = local.scale_bar!.unit || "um";
                              const uToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
                              const uLabels: Record<string, string> = { km: "km", m: "m", cm: "cm", mm: "mm", um: "\u00B5m", nm: "nm", pm: "pm" };
                              const val = local.scale_bar!.bar_length_microns / (uToUm[sbU] || 1);
                              return `${Number(val.toPrecision(6))} ${uLabels[sbU] || sbU}`;
                            })()}
                          </Typography>
                        </>
                      );
                    })()}
                  </Box>
                  ); })()}
                {/* Scale bar drag hitbox — invisible, for dragging on Scale Bar tab with rendered preview */}
                {tabIdx === TAB_SCALE && renderedPreviewB64 && local.add_scale_bar && local.scale_bar && (local.scale_bar.position_preset ?? "Bottom-Right") === "Custom" && (() => {
                  const sb = local.scale_bar!;
                  const posX = sb.position_x ?? 50;
                  const posY = sb.position_y ?? 90;
                  return (
                    <Box
                      sx={{
                        position: "absolute",
                        left: `${posX}%`,
                        top: `${posY}%`,
                        transform: "translate(-50%, -50%)",
                        width: "15%",
                        height: "10%",
                        cursor: "grab",
                        zIndex: 10,
                      }}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const imgEl = (e.currentTarget.parentElement?.querySelector("img")) as HTMLImageElement | null;
                        if (!imgEl) return;
                        const onMove = (ev: MouseEvent) => {
                          ev.preventDefault();
                          const rect = imgEl.getBoundingClientRect();
                          const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                          const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                          updateLocal({ scale_bar: { ...local.scale_bar!, position_x: Math.round(px), position_y: Math.round(py), position_preset: "Custom", bar_position: [px / 100, py / 100] } });
                        };
                        const onUp = () => {
                          window.removeEventListener("mousemove", onMove);
                          window.removeEventListener("mouseup", onUp);
                        };
                        window.addEventListener("mousemove", onMove);
                        window.addEventListener("mouseup", onUp);
                      }}
                    />
                  );
                })()}
                {/* Line annotation overlays */}
                {tabIdx === TAB_ANNOT && (local.lines ?? []).length > 0 && (() => {
                  const lnVbW = svgDims.w || 1000;
                  const lnVbH = svgDims.h || 1000;
                  return (
                  <svg
                    viewBox={`0 0 ${lnVbW} ${lnVbH}`}
                    preserveAspectRatio="none"
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 3 }}
                  >
                    {(local.lines ?? []).map((line, li) => {
                      if (!line.points || line.points.length === 0) return null;
                      // Convert percentage coords to pixel coords for the viewBox
                      const pxPoints = line.points.map(([px, py]: [number, number]) => [px / 100 * lnVbW, py / 100 * lnVbH] as [number, number]);
                      const lineType = line.line_type || (line.is_curved ? "curved" : "straight");
                      const ptsStr = pxPoints.length >= 2
                        ? pxPoints.map(([px, py]) => `${px},${py}`).join(" ")
                        : "";
                      // Generate smooth SVG path for "curved" mode
                      let smoothPath = "";
                      if (lineType === "curved" && pxPoints.length >= 3) {
                        // Catmull-Rom to cubic bezier approximation
                        const pts = pxPoints;
                        smoothPath = `M ${pts[0][0]},${pts[0][1]}`;
                        for (let pi = 0; pi < pts.length - 1; pi++) {
                          const p0 = pts[Math.max(0, pi - 1)];
                          const p1 = pts[pi];
                          const p2 = pts[pi + 1];
                          const p3 = pts[Math.min(pts.length - 1, pi + 2)];
                          const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
                          const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
                          const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
                          const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
                          smoothPath += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
                        }
                      }
                      const dashArr = line.dash_style === "dashed" ? "2,1" : line.dash_style === "dotted" ? "0.5,1" : line.dash_style === "dash-dot" ? "2,1,0.5,1" : "";
                      // Compute length for measure using real units from scale bar
                      const scaleBar = local.scale_bar;
                      const mpp = scaleBar?.micron_per_pixel || 1;
                      const scaleUnit = scaleBar?.unit || "um";
                      // Use ACTUAL image pixel dimensions (svgDims = crop dimensions)
                      // NOT thumbnail dimensions — must match PIL's computation exactly
                      const imgW = lnVbW;  // = svgDims.w = actual crop pixel width
                      const imgH = lnVbH;  // = svgDims.h = actual crop pixel height
                      let totalLen = 0;
                      if (lineType === "curved" && line.points.length >= 3) {
                        // Approximate spline arc length by sampling the Catmull-Rom curve
                        const pts = line.points;
                        const SAMPLES_PER_SEG = 20;
                        for (let si = 0; si < pts.length - 1; si++) {
                          const p0 = pts[Math.max(0, si - 1)];
                          const p1 = pts[si];
                          const p2 = pts[si + 1];
                          const p3 = pts[Math.min(pts.length - 1, si + 2)];
                          let prevX = p1[0] / 100 * imgW, prevY = p1[1] / 100 * imgH;
                          for (let t = 1; t <= SAMPLES_PER_SEG; t++) {
                            const tt = t / SAMPLES_PER_SEG;
                            // Catmull-Rom interpolation
                            const t2 = tt * tt, t3 = t2 * tt;
                            const cx = 0.5 * (2*p1[0] + (-p0[0]+p2[0])*tt + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3);
                            const cy = 0.5 * (2*p1[1] + (-p0[1]+p2[1])*tt + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3);
                            const curX = cx / 100 * imgW, curY = cy / 100 * imgH;
                            totalLen += Math.sqrt((curX - prevX) ** 2 + (curY - prevY) ** 2);
                            prevX = curX; prevY = curY;
                          }
                        }
                      } else {
                        // Straight segments — sum of chord lengths
                        for (let pi = 1; pi < line.points.length; pi++) {
                          const dxPx = (line.points[pi][0] - line.points[pi - 1][0]) / 100 * imgW;
                          const dyPx = (line.points[pi][1] - line.points[pi - 1][1]) / 100 * imgH;
                          totalLen += Math.sqrt(dxPx * dxPx + dyPx * dyPx);
                        }
                      }
                      // Convert pixel distance to real units
                      const uToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
                      const realDistUm = totalLen * mpp;
                      const realDist = realDistUm / (uToUm[scaleUnit] || 1);
                      const unitLabels: Record<string, string> = { km: "km", m: "m", cm: "cm", mm: "mm", um: "\u00B5m", nm: "nm", pm: "pm" };
                      const unitLabel = unitLabels[scaleUnit] || scaleUnit;
                      const midPt = pxPoints.length >= 2
                        ? [(pxPoints[0][0] + pxPoints[pxPoints.length - 1][0]) / 2,
                           (pxPoints[0][1] + pxPoints[pxPoints.length - 1][1]) / 2]
                        : [lnVbW / 2, lnVbH / 2];
                      return (
                        <g key={`line-${li}`}>
                          {ptsStr && !smoothPath && (
                            <polyline
                              points={ptsStr}
                              fill="none"
                              stroke={line.color}
                              strokeWidth={Math.max(1, line.width * Math.max(1, lnVbW / 1000))}
                              strokeDasharray={dashArr ? dashArr.split(",").map(v => String(Number(v) * lnVbW / 100)).join(",") : ""}
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          )}
                          {smoothPath && (
                            <path
                              d={smoothPath}
                              fill="none"
                              stroke={line.color}
                              strokeWidth={Math.max(1, line.width * Math.max(1, lnVbW / 1000))}
                              strokeDasharray={dashArr ? dashArr.split(",").map(v => String(Number(v) * lnVbW / 100)).join(",") : ""}
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          )}
                          {/* Draggable point handles */}
                          {pxPoints.map(([px, py], pi: number) => (
                            <circle key={pi} cx={px} cy={py} r={Math.max(lnVbW * 0.005, line.width * Math.max(1, lnVbW / 1000) * 2)} fill={line.color} stroke="#fff" strokeWidth={Math.max(1, lnVbW * 0.001)}
                              style={{ pointerEvents: "auto", cursor: "grab" }}
                              onMouseDown={(e) => {
                                e.preventDefault(); e.stopPropagation();
                                const svgEl = e.currentTarget.closest("svg");
                                const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                if (!imgEl) return;
                                const onMove = (ev: MouseEvent) => {
                                  ev.preventDefault();
                                  const rect = imgEl.getBoundingClientRect();
                                  const nx = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                  const ny = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                  const lines2 = [...(local.lines ?? [])];
                                  const pts2 = [...(lines2[li].points ?? [])];
                                  pts2[pi] = [Math.round(nx * 10) / 10, Math.round(ny * 10) / 10];
                                  lines2[li] = { ...lines2[li], points: pts2 };
                                  updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                                };
                                const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                              }}
                            />
                          ))}
                          {/* Measurement label */}
                          {line.show_measure && pxPoints.length >= 2 && (() => {
                            const isLineActive = selectedAnnotIdx?.type === "line" && selectedAnnotIdx.idx === li;
                            // Use absolute position if set, otherwise auto (midpoint)
                            const mPosX = (line as any).measure_position_x;
                            const mPosY = (line as any).measure_position_y;
                            const tx = (mPosX >= 0) ? mPosX / 100 * lnVbW : midPt[0];
                            const ty = (mPosY >= 0) ? mPosY / 100 * lnVbH : midPt[1] - lnVbW * 0.02;
                            const mFontSize = Math.max(lnVbW * 0.015, (line.measure_font_size || 12) * lnVbW / 216);
                            return (
                              <text x={tx} y={ty} fill={line.measure_color || line.color}
                                fontSize={mFontSize} textAnchor="middle"
                                fontWeight={(line.measure_font_style || []).includes("Bold") ? "bold" : "normal"}
                                fontStyle={(line.measure_font_style || []).includes("Italic") ? "italic" : "normal"}
                                textDecoration={(line.measure_font_style || []).includes("Strikethrough") ? "line-through" : "none"}
                                fontFamily={(line.measure_font_name || "arial.ttf").replace(/\.(ttf|otf|ttc)$/i, "")}
                                style={{ pointerEvents: isLineActive ? "auto" : "none", cursor: isLineActive ? "grab" : "default" }}
                                onMouseDown={isLineActive ? (e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  const svgEl = e.currentTarget.closest("svg");
                                  const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                  if (!imgEl) return;
                                  const onMove = (ev: MouseEvent) => {
                                    const rect = imgEl.getBoundingClientRect();
                                    const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                    const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                    const lines2 = [...(local.lines ?? [])];
                                    lines2[li] = { ...lines2[li], measure_position_x: Math.round(px * 10) / 10, measure_position_y: Math.round(py * 10) / 10 } as any;
                                    updateLocal({ lines: lines2 } as unknown as Partial<PanelInfo>);
                                  };
                                  const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                  window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                                } : undefined}
                              >
                                {line.measure_text || `${realDist.toFixed(1)} ${unitLabel}`}
                              </text>
                            );
                          })()}
                        </g>
                      );
                    })}
                  </svg>
                  );
                })()}
                {/* Area annotation overlays */}
                {tabIdx === TAB_ANNOT && (local.areas ?? []).some(a => a.points && a.points.length >= 1) && (() => {
                  const arVbW = svgDims.w || 1000;
                  const arVbH = svgDims.h || 1000;
                  return (
                  <svg
                    viewBox={`0 0 ${arVbW} ${arVbH}`}
                    preserveAspectRatio="none"
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 5 }}
                  >
                    {(local.areas ?? []).map((area, ai) => {
                      if (!area.points || area.points.length === 0) return null;
                      const isAreaActive = selectedAnnotIdx?.type === "area" && selectedAnnotIdx.idx === ai;
                      // Convert hex+alpha to rgba for SVG compatibility
                      const rawColor = area.color || "#FF000040";
                      let fillColor: string;
                      if (rawColor.length === 9 && rawColor.startsWith("#")) {
                        const r = parseInt(rawColor.slice(1, 3), 16);
                        const g = parseInt(rawColor.slice(3, 5), 16);
                        const b = parseInt(rawColor.slice(5, 7), 16);
                        const a = parseInt(rawColor.slice(7, 9), 16) / 255;
                        fillColor = `rgba(${r},${g},${b},${a.toFixed(2)})`;
                      } else {
                        fillColor = rawColor;
                      }
                      const strokeColor = area.border_color || "#FF0000";
                      const sw = Math.max(arVbW * 0.001, (area.border_width || 1) * Math.max(1, arVbW / 1000));
                      // Dash pattern for area border
                      const areaDash = (area as any).dash_style === "dashed" ? `${sw * 3},${sw * 2}` : (area as any).dash_style === "dotted" ? `${sw},${sw * 2}` : (area as any).dash_style === "dash-dot" ? `${sw * 3},${sw},${sw},${sw}` : "";

                      // Custom/Magic polygon: render points and connecting lines
                      if (area.shape === "Custom" || area.shape === "Magic") {
                        const pxPts = area.points.map(([px, py]: [number, number]) => [px / 100 * arVbW, py / 100 * arVbH] as [number, number]);
                        const polyStr = pxPts.map(([px, py]) => `${px},${py}`).join(" ");
                        // Compute area using Shoelace formula
                        // For smoothed polygons, sample the Bezier curve to get accurate area
                        let areaVal = 0;
                        if (pxPts.length >= 3) {
                          const isSmoothed = (area as any).smooth;
                          if (isSmoothed) {
                            // Sample Catmull-Rom spline at many points, then Shoelace
                            const sampledPts: [number, number][] = [];
                            const n = pxPts.length;
                            const samplesPerSeg = 20;
                            for (let si = 0; si < n; si++) {
                              const p0 = pxPts[(si - 1 + n) % n];
                              const p1 = pxPts[si];
                              const p2 = pxPts[(si + 1) % n];
                              const p3 = pxPts[(si + 2) % n];
                              const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
                              const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
                              const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
                              const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
                              for (let ti = 0; ti < samplesPerSeg; ti++) {
                                const t = ti / samplesPerSeg;
                                const mt = 1 - t;
                                const x = mt*mt*mt*p1[0] + 3*mt*mt*t*cp1x + 3*mt*t*t*cp2x + t*t*t*p2[0];
                                const y = mt*mt*mt*p1[1] + 3*mt*mt*t*cp1y + 3*mt*t*t*cp2y + t*t*t*p2[1];
                                sampledPts.push([x, y]);
                              }
                            }
                            for (let pi = 0; pi < sampledPts.length; pi++) {
                              const [x1, y1] = sampledPts[pi];
                              const [x2, y2] = sampledPts[(pi + 1) % sampledPts.length];
                              areaVal += x1 * y2 - x2 * y1;
                            }
                            areaVal = Math.abs(areaVal) / 2;
                          } else {
                            for (let pi = 0; pi < pxPts.length; pi++) {
                              const [x1, y1] = pxPts[pi];
                              const [x2, y2] = pxPts[(pi + 1) % pxPts.length];
                              areaVal += x1 * y2 - x2 * y1;
                            }
                            areaVal = Math.abs(areaVal) / 2;
                          }
                        }
                        const mpp = local.scale_bar?.micron_per_pixel || 1;
                        const areaUnit = (area as any).measure_unit || local.scale_bar?.unit || "um";
                        const uToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
                        const unitLabels: Record<string, string> = { km: "km\u00B2", m: "m\u00B2", cm: "cm\u00B2", mm: "mm\u00B2", um: "\u00B5m\u00B2", nm: "nm\u00B2", pm: "pm\u00B2" };
                        const areaUm2 = areaVal * (mpp ** 2);
                        const areaInUnit = areaUm2 / ((uToUm[areaUnit] || 1) ** 2);
                        // Centroid for text placement
                        const centX = pxPts.reduce((s, p) => s + p[0], 0) / pxPts.length;
                        const centY = pxPts.reduce((s, p) => s + p[1], 0) / pxPts.length;

                        return (
                          <g key={`area-${ai}`}>
                            {pxPts.length >= 3 && (() => {
                              const isSmooth = (area as any).smooth;
                              if (isSmooth) {
                                // Generate closed smooth cubic bezier path through all points
                                const pts = pxPts;
                                const n = pts.length;
                                let d = `M ${pts[0][0]},${pts[0][1]}`;
                                for (let si = 0; si < n; si++) {
                                  const p0 = pts[(si - 1 + n) % n];
                                  const p1 = pts[si];
                                  const p2 = pts[(si + 1) % n];
                                  const p3 = pts[(si + 2) % n];
                                  // Catmull-Rom to cubic bezier control points (alpha=0.5 for centripetal)
                                  const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
                                  const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
                                  const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
                                  const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
                                  d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
                                }
                                d += " Z";
                                return <path d={d} fill={fillColor} stroke={strokeColor}
                                  strokeWidth={sw} strokeLinejoin="round" strokeDasharray={areaDash || undefined}
                                  style={{ pointerEvents: "auto", cursor: "grab" }}
                                  onClick={() => setSelectedAnnotIdx({ type: "area", idx: ai })} />;
                              }
                              return <polygon points={polyStr} fill={fillColor} stroke={strokeColor}
                                strokeWidth={sw} strokeLinejoin="round" strokeDasharray={areaDash || undefined}
                                style={{ pointerEvents: "auto", cursor: "grab" }}
                                onClick={() => setSelectedAnnotIdx({ type: "area", idx: ai })} />;
                            })()}
                            {pxPts.length < 3 && pxPts.length >= 2 && (
                              <polyline points={polyStr} fill="none" stroke={strokeColor}
                                strokeWidth={sw} strokeLinecap="round" />
                            )}
                            {/* Draggable point handles */}
                            {pxPts.map(([px, py], pi) => (
                              <circle key={pi} cx={px} cy={py}
                                r={Math.max(arVbW * 0.005, sw * 2)}
                                fill={strokeColor} stroke="#fff" strokeWidth={Math.max(1, arVbW * 0.001)}
                                style={{ pointerEvents: "auto", cursor: "grab" }}
                                onMouseDown={(e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  const svgEl = e.currentTarget.closest("svg");
                                  const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                  if (!imgEl) return;
                                  const onMove = (ev: MouseEvent) => {
                                    const rect = imgEl.getBoundingClientRect();
                                    const nx = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                    const ny = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                    const areas2 = [...(local.areas ?? [])];
                                    const pts2 = [...(areas2[ai].points ?? [])];
                                    pts2[pi] = [nx, ny];
                                    areas2[ai] = { ...areas2[ai], points: pts2 };
                                    updateLocal({ areas: areas2 } as unknown as Partial<PanelInfo>);
                                  };
                                  const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                  window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                                }}
                              />
                            ))}
                            {area.show_measure && pxPts.length >= 3 && (() => {
                              const mPosX = (area as any).measure_position_x ?? -1;
                              const mPosY = (area as any).measure_position_y ?? -1;
                              const textX = mPosX >= 0 ? mPosX / 100 * arVbW : centX;
                              const textY = mPosY >= 0 ? mPosY / 100 * arVbH : centY;
                              const measFontSz = Math.max(arVbW * 0.012, (area.measure_font_size || 12) * arVbW / 216);
                              const amfs = (area as any).measure_font_style || [];
                              const amfn = (area.measure_font_name || "arial.ttf").replace(/\.(ttf|otf|ttc)$/i, "");
                              // Use toPrecision for better precision across unit switches
                              const measDisplayVal = areaInUnit < 0.01 ? areaInUnit.toExponential(3) : areaInUnit < 1 ? areaInUnit.toPrecision(4) : areaInUnit < 1000 ? areaInUnit.toFixed(2) : areaInUnit.toPrecision(6);
                              return (
                                <text x={textX} y={textY} fill={area.measure_color || "#FF0"}
                                  fontSize={measFontSz} textAnchor="middle" dominantBaseline="middle"
                                  fontWeight={amfs.includes("Bold") ? "bold" : "normal"}
                                  fontStyle={amfs.includes("Italic") ? "italic" : "normal"}
                                  textDecoration={amfs.includes("Strikethrough") ? "line-through" : "none"}
                                  fontFamily={amfn}
                                  style={{ cursor: "grab", pointerEvents: "auto" }}
                                  onMouseDown={(e) => {
                                    e.preventDefault(); e.stopPropagation();
                                    const svg = (e.target as SVGElement).closest("svg");
                                    if (!svg) return;
                                    const onMove = (ev: MouseEvent) => {
                                      ev.preventDefault();
                                      const rect = svg.getBoundingClientRect();
                                      const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                      const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                      const areas = [...(local.areas || [])];
                                      areas[ai] = { ...areas[ai], measure_position_x: Math.round(px * 10) / 10, measure_position_y: Math.round(py * 10) / 10 } as any;
                                      updateLocal({ areas });
                                    };
                                    const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                    window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                                  }}>
                                  {area.measure_text || `${measDisplayVal} ${unitLabels[areaUnit] || areaUnit}`}
                                </text>
                              );
                            })()}
                          </g>
                        );
                      }

                      // Standard shapes need at least 2 points (center + size)
                      if (area.points.length < 2) return null;
                      const [cx, cy] = [area.points[0][0] / 100 * arVbW, area.points[0][1] / 100 * arVbH];
                      const [w, h] = [area.points[1][0] / 100 * arVbW, area.points[1][1] / 100 * arVbH];
                      if (area.shape === "Ellipse") {
                        return (
                          <g key={`area-${ai}`}>
                            <ellipse cx={cx} cy={cy} rx={w / 2} ry={h / 2}
                              fill={fillColor} stroke={strokeColor} strokeWidth={Math.max(arVbW * 0.001, area.border_width * arVbW / 500)}
                              strokeDasharray={areaDash || undefined}
                              style={{ pointerEvents: "auto", cursor: "grab" }}
                              onMouseDown={(e) => {
                                e.preventDefault(); e.stopPropagation();
                                const svgEl = e.currentTarget.closest("svg");
                                const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                if (!imgEl) return;
                                const startX = e.clientX, startY = e.clientY;
                                // Use percentage coordinates (0-100), not SVG viewBox units
                                const startPctX = area.points[0][0];
                                const startPctY = area.points[0][1];
                                const onMove = (ev: MouseEvent) => {
                                  const rect = imgEl.getBoundingClientRect();
                                  const dx = ((ev.clientX - startX) / rect.width) * 100;
                                  const dy = ((ev.clientY - startY) / rect.height) * 100;
                                  const areas2 = [...(local.areas ?? [])];
                                  const pts2 = [...(areas2[ai].points ?? [])];
                                  pts2[0] = [Math.round((startPctX + dx) * 10) / 10, Math.round((startPctY + dy) * 10) / 10];
                                  areas2[ai] = { ...areas2[ai], points: pts2 };
                                  updateLocal({ areas: areas2 } as unknown as Partial<PanelInfo>);
                                };
                                const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                                window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                              }}
                            />
                            {/* Resize handles for standard shapes */}
                            {isAreaActive && [
                              { hx: cx + w/2, hy: cy, cursor: "e-resize", dim: "w", sign: 1 },
                              { hx: cx - w/2, hy: cy, cursor: "w-resize", dim: "w", sign: -1 },
                              { hx: cx, hy: cy + h/2, cursor: "s-resize", dim: "h", sign: 1 },
                              { hx: cx, hy: cy - h/2, cursor: "n-resize", dim: "h", sign: -1 },
                            ].map((handle, hi) => (
                              <circle key={`resize-${hi}`} cx={handle.hx} cy={handle.hy}
                                r={arVbW * 0.008} fill="#4FC3F7" stroke="#fff" strokeWidth={arVbW * 0.002}
                                style={{ cursor: handle.cursor, pointerEvents: "auto" }}
                                onMouseDown={(e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  const svgEl2 = e.currentTarget.closest("svg");
                                  if (!svgEl2) return;
                                  const startX2 = e.clientX, startY2 = e.clientY;
                                  const startW = w, startH = h;
                                  const startCx2 = area.points[0][0], startCy2 = area.points[0][1];
                                  const onMove2 = (ev: MouseEvent) => {
                                    const rect2 = svgEl2.getBoundingClientRect();
                                    const dxPct = ((ev.clientX - startX2) / rect2.width) * 100;
                                    const dyPct = ((ev.clientY - startY2) / rect2.height) * 100;
                                    const areas3 = [...(local.areas ?? [])];
                                    const pts3 = [...(areas3[ai].points ?? [])];
                                    // Asymmetric resize: move center by half delta, change size by full delta
                                    const dW = handle.dim === "w" ? dxPct * handle.sign : 0;
                                    const dH = handle.dim === "h" ? dyPct * handle.sign : 0;
                                    const newWPct = Math.max(2, startW / arVbW * 100 + dW);
                                    const newHPct = Math.max(2, startH / arVbH * 100 + dH);
                                    const newCx = startCx2 + (handle.dim === "w" ? dxPct / 2 : 0);
                                    const newCy = startCy2 + (handle.dim === "h" ? dyPct / 2 : 0);
                                    pts3[0] = [Math.round(newCx * 10) / 10, Math.round(newCy * 10) / 10];
                                    pts3[1] = [Math.round(newWPct * 10) / 10, Math.round(newHPct * 10) / 10];
                                    areas3[ai] = { ...areas3[ai], points: pts3 };
                                    updateLocal({ areas: areas3 } as unknown as Partial<PanelInfo>);
                                  };
                                  const onUp2 = () => { window.removeEventListener("mousemove", onMove2); window.removeEventListener("mouseup", onUp2); };
                                  window.addEventListener("mousemove", onMove2); window.addEventListener("mouseup", onUp2);
                                }}
                              />
                            ))}
                            {area.show_measure && (() => {
                              const mPosX2 = (area as any).measure_position_x ?? -1;
                              const mPosY2 = (area as any).measure_position_y ?? -1;
                              const tX2 = mPosX2 >= 0 ? mPosX2 / 100 * arVbW : cx;
                              const tY2 = mPosY2 >= 0 ? mPosY2 / 100 * arVbH : cy;
                              const mfs2 = Math.max(arVbW * 0.012, (area.measure_font_size || 12) * arVbW / 216);
                              const amfs2 = (area as any).measure_font_style || [];
                              const amfn2 = (area.measure_font_name || "arial.ttf").replace(/\.(ttf|otf|ttc)$/i, "");
                              // Compute area measurement
                              const aUnit2 = (area as any).measure_unit || local.scale_bar?.unit || "um";
                              const uToUm2: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
                              const uLabels2: Record<string, string> = { km: "km\u00B2", m: "m\u00B2", cm: "cm\u00B2", mm: "mm\u00B2", um: "\u00B5m\u00B2", nm: "nm\u00B2", pm: "pm\u00B2" };
                              const mpp2 = local.scale_bar?.micron_per_pixel || 1;
                              const hw2 = w / 2, hh2 = h / 2;
                              let areaPx2 = 0;
                              if (area.shape === "Ellipse") areaPx2 = Math.PI * hw2 * hh2;
                              else if (area.shape === "Triangle" && area.points.length >= 3) {
                                const tp = area.points.slice(0, 3).map((p: number[]) => [p[0] / 100 * arVbW, p[1] / 100 * arVbH]);
                                areaPx2 = Math.abs((tp[1][0] - tp[0][0]) * (tp[2][1] - tp[0][1]) - (tp[2][0] - tp[0][0]) * (tp[1][1] - tp[0][1])) / 2;
                              } else areaPx2 = w * h;
                              const areaUm2 = areaPx2 * (mpp2 ** 2);
                              const areaInU2 = areaUm2 / ((uToUm2[aUnit2] || 1) ** 2);
                              const measDisp2 = areaInU2 < 0.01 ? areaInU2.toExponential(3) : areaInU2 < 1 ? areaInU2.toPrecision(4) : areaInU2 < 1000 ? areaInU2.toFixed(2) : areaInU2.toPrecision(6);
                              const measText2 = area.measure_text || `${measDisp2} ${uLabels2[aUnit2] || aUnit2}`;
                              return (
                                <text x={tX2} y={tY2} fill={area.measure_color || "#FF0"}
                                  fontSize={mfs2} textAnchor="middle" dominantBaseline="middle"
                                  fontWeight={amfs2.includes("Bold") ? "bold" : "normal"}
                                  fontStyle={amfs2.includes("Italic") ? "italic" : "normal"}
                                  textDecoration={amfs2.includes("Strikethrough") ? "line-through" : "none"}
                                  fontFamily={amfn2}
                                  style={{ cursor: "grab", pointerEvents: "auto" }}
                                  onMouseDown={(e) => {
                                    e.preventDefault(); e.stopPropagation();
                                    const svg2 = (e.target as SVGElement).closest("svg");
                                    if (!svg2) return;
                                    const onMove2 = (ev: MouseEvent) => {
                                      ev.preventDefault();
                                      const r2 = svg2.getBoundingClientRect();
                                      const px2 = Math.max(0, Math.min(100, ((ev.clientX - r2.left) / r2.width) * 100));
                                      const py2 = Math.max(0, Math.min(100, ((ev.clientY - r2.top) / r2.height) * 100));
                                      const ar2 = [...(local.areas || [])];
                                      ar2[ai] = { ...ar2[ai], measure_position_x: Math.round(px2 * 10) / 10, measure_position_y: Math.round(py2 * 10) / 10 } as any;
                                      updateLocal({ areas: ar2 });
                                    };
                                    const onUp2 = () => { window.removeEventListener("mousemove", onMove2); window.removeEventListener("mouseup", onUp2); };
                                    window.addEventListener("mousemove", onMove2); window.addEventListener("mouseup", onUp2);
                                  }}>
                                  {measText2}
                                </text>
                              );
                            })()}
                          </g>
                        );
                      }
                      return (
                        <g key={`area-${ai}`}>
                          <rect x={cx - w / 2} y={cy - h / 2} width={w} height={h}
                            fill={fillColor} stroke={strokeColor} strokeWidth={Math.max(arVbW * 0.001, area.border_width * arVbW / 500)}
                            strokeDasharray={areaDash || undefined}
                            style={{ pointerEvents: "auto", cursor: "grab" }}
                            onMouseDown={(e) => {
                              e.preventDefault(); e.stopPropagation();
                              const svgEl = e.currentTarget.closest("svg");
                              const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                              if (!imgEl) return;
                              const startX = e.clientX, startY = e.clientY;
                              // Use percentage coordinates (0-100), not SVG viewBox units
                              const startPctX = area.points[0][0];
                              const startPctY = area.points[0][1];
                              const onMove = (ev: MouseEvent) => {
                                const rect = imgEl.getBoundingClientRect();
                                const dx = ((ev.clientX - startX) / rect.width) * 100;
                                const dy = ((ev.clientY - startY) / rect.height) * 100;
                                const areas2 = [...(local.areas ?? [])];
                                const pts2 = [...(areas2[ai].points ?? [])];
                                pts2[0] = [Math.round((startPctX + dx) * 10) / 10, Math.round((startPctY + dy) * 10) / 10];
                                areas2[ai] = { ...areas2[ai], points: pts2 };
                                updateLocal({ areas: areas2 } as unknown as Partial<PanelInfo>);
                              };
                              const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
                              window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                            }}
                          />
                          {/* Resize handles for rectangle */}
                          {isAreaActive && [
                            { hx: cx + w/2, hy: cy, cursor: "e-resize", dim: "w" as const, sign: 1 },
                            { hx: cx - w/2, hy: cy, cursor: "w-resize", dim: "w" as const, sign: -1 },
                            { hx: cx, hy: cy + h/2, cursor: "s-resize", dim: "h" as const, sign: 1 },
                            { hx: cx, hy: cy - h/2, cursor: "n-resize", dim: "h" as const, sign: -1 },
                          ].map((handle, hi) => (
                            <circle key={`rresize-${hi}`} cx={handle.hx} cy={handle.hy}
                              r={arVbW * 0.008} fill="#4FC3F7" stroke="#fff" strokeWidth={arVbW * 0.002}
                              style={{ cursor: handle.cursor, pointerEvents: "auto" }}
                              onMouseDown={(e) => {
                                e.preventDefault(); e.stopPropagation();
                                const svgEl3 = e.currentTarget.closest("svg");
                                if (!svgEl3) return;
                                const startX3 = e.clientX, startY3 = e.clientY;
                                const startW3 = w, startH3 = h;
                                const startCx3 = area.points[0][0], startCy3 = area.points[0][1];
                                const onMove3 = (ev: MouseEvent) => {
                                  const rect3 = svgEl3.getBoundingClientRect();
                                  const dxR = ((ev.clientX - startX3) / rect3.width) * 100;
                                  const dyR = ((ev.clientY - startY3) / rect3.height) * 100;
                                  const areas4 = [...(local.areas ?? [])];
                                  const pts4 = [...(areas4[ai].points ?? [])];
                                  const dW3 = handle.dim === "w" ? dxR * handle.sign : 0;
                                  const dH3 = handle.dim === "h" ? dyR * handle.sign : 0;
                                  const nWPct = Math.max(2, startW3 / arVbW * 100 + dW3);
                                  const nHPct = Math.max(2, startH3 / arVbH * 100 + dH3);
                                  const nCx3 = startCx3 + (handle.dim === "w" ? dxR / 2 : 0);
                                  const nCy3 = startCy3 + (handle.dim === "h" ? dyR / 2 : 0);
                                  pts4[0] = [Math.round(nCx3 * 10) / 10, Math.round(nCy3 * 10) / 10];
                                  pts4[1] = [Math.round(nWPct * 10) / 10, Math.round(nHPct * 10) / 10];
                                  areas4[ai] = { ...areas4[ai], points: pts4 };
                                  updateLocal({ areas: areas4 } as unknown as Partial<PanelInfo>);
                                };
                                const onUp3 = () => { window.removeEventListener("mousemove", onMove3); window.removeEventListener("mouseup", onUp3); };
                                window.addEventListener("mousemove", onMove3); window.addEventListener("mouseup", onUp3);
                              }}
                            />
                          ))}
                          {area.show_measure && (() => {
                              const mPosX2 = (area as any).measure_position_x ?? -1;
                              const mPosY2 = (area as any).measure_position_y ?? -1;
                              const tX2 = mPosX2 >= 0 ? mPosX2 / 100 * arVbW : cx;
                              const tY2 = mPosY2 >= 0 ? mPosY2 / 100 * arVbH : cy;
                              const mfs2 = Math.max(arVbW * 0.012, (area.measure_font_size || 12) * arVbW / 216);
                              const amfs2 = (area as any).measure_font_style || [];
                              const amfn2 = (area.measure_font_name || "arial.ttf").replace(/\.(ttf|otf|ttc)$/i, "");
                              // Compute area measurement
                              const aUnit2 = (area as any).measure_unit || local.scale_bar?.unit || "um";
                              const uToUm2: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
                              const uLabels2: Record<string, string> = { km: "km\u00B2", m: "m\u00B2", cm: "cm\u00B2", mm: "mm\u00B2", um: "\u00B5m\u00B2", nm: "nm\u00B2", pm: "pm\u00B2" };
                              const mpp2 = local.scale_bar?.micron_per_pixel || 1;
                              const hw2 = w / 2, hh2 = h / 2;
                              let areaPx2 = 0;
                              if (area.shape === "Ellipse") areaPx2 = Math.PI * hw2 * hh2;
                              else if (area.shape === "Triangle" && area.points.length >= 3) {
                                const tp = area.points.slice(0, 3).map((p: number[]) => [p[0] / 100 * arVbW, p[1] / 100 * arVbH]);
                                areaPx2 = Math.abs((tp[1][0] - tp[0][0]) * (tp[2][1] - tp[0][1]) - (tp[2][0] - tp[0][0]) * (tp[1][1] - tp[0][1])) / 2;
                              } else areaPx2 = w * h;
                              const areaUm2 = areaPx2 * (mpp2 ** 2);
                              const areaInU2 = areaUm2 / ((uToUm2[aUnit2] || 1) ** 2);
                              const measDisp2 = areaInU2 < 0.01 ? areaInU2.toExponential(3) : areaInU2 < 1 ? areaInU2.toPrecision(4) : areaInU2 < 1000 ? areaInU2.toFixed(2) : areaInU2.toPrecision(6);
                              const measText2 = area.measure_text || `${measDisp2} ${uLabels2[aUnit2] || aUnit2}`;
                              return (
                                <text x={tX2} y={tY2} fill={area.measure_color || "#FF0"}
                                  fontSize={mfs2} textAnchor="middle" dominantBaseline="middle"
                                  fontWeight={amfs2.includes("Bold") ? "bold" : "normal"}
                                  fontStyle={amfs2.includes("Italic") ? "italic" : "normal"}
                                  textDecoration={amfs2.includes("Strikethrough") ? "line-through" : "none"}
                                  fontFamily={amfn2}
                                  style={{ cursor: "grab", pointerEvents: "auto" }}
                                  onMouseDown={(e) => {
                                    e.preventDefault(); e.stopPropagation();
                                    const svg2 = (e.target as SVGElement).closest("svg");
                                    if (!svg2) return;
                                    const onMove2 = (ev: MouseEvent) => {
                                      ev.preventDefault();
                                      const r2 = svg2.getBoundingClientRect();
                                      const px2 = Math.max(0, Math.min(100, ((ev.clientX - r2.left) / r2.width) * 100));
                                      const py2 = Math.max(0, Math.min(100, ((ev.clientY - r2.top) / r2.height) * 100));
                                      const ar2 = [...(local.areas || [])];
                                      ar2[ai] = { ...ar2[ai], measure_position_x: Math.round(px2 * 10) / 10, measure_position_y: Math.round(py2 * 10) / 10 } as any;
                                      updateLocal({ areas: ar2 });
                                    };
                                    const onUp2 = () => { window.removeEventListener("mousemove", onMove2); window.removeEventListener("mouseup", onUp2); };
                                    window.addEventListener("mousemove", onMove2); window.addEventListener("mouseup", onUp2);
                                  }}>
                                  {measText2}
                                </text>
                              );
                            })()}
                        </g>
                      );
                    })}
                  </svg>
                  );
                })()}
                {/* Click-to-add-point for lines / Click-to-place areas */}
                {tabIdx === TAB_ANNOT && ((local.lines ?? []).length > 0 || (local.areas ?? []).some(a => !a.points || a.points.length < 2) || (selectedAnnotIdx?.type === "area" && ["Custom", "Magic"].includes((local.areas ?? [])[selectedAnnotIdx.idx]?.shape))) && (
                  <Box
                    sx={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", zIndex: 1, cursor: "crosshair" }}
                    onClick={(e) => {
                      const imgEl = e.currentTarget.parentElement?.querySelector("img") as HTMLImageElement | null;
                      if (!imgEl) return;
                      const rect = imgEl.getBoundingClientRect();
                      const px = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
                      const py = Math.max(0, Math.min(100, ((e.clientY - rect.top) / rect.height) * 100));
                      const rpx = Math.round(px * 10) / 10;
                      const rpy = Math.round(py * 10) / 10;

                      // If there's an active area being edited, add points to it
                      const activeAreaIdx = selectedAnnotIdx?.type === "area" ? selectedAnnotIdx.idx : -1;
                      // Or find an area without enough points
                      const areaIdx = activeAreaIdx >= 0 ? activeAreaIdx : (local.areas ?? []).findIndex(a => !a.points || a.points.length < 2);
                      if (areaIdx >= 0) {
                        const areas = [...(local.areas ?? [])];
                        const area = areas[areaIdx];
                        const pts = [...(area.points ?? [])];

                        if (area.shape === "Magic") {
                          // Magic wand: call backend to flood-fill select
                          const tolerance = (area as any).magic_tolerance ?? 30;
                          setMagicWandLoading(true);
                          (async () => {
                            try {
                              const resp = await api.magicWandSelect(row, col, rpx, rpy, tolerance, { rotation: local.rotation, crop: local.crop as number[] | undefined, crop_image: local.crop_image });
                              if (resp.points && resp.points.length >= 3) {
                                const areas2 = [...(local.areas ?? [])];
                                areas2[areaIdx] = {
                                  ...areas2[areaIdx],
                                  points: resp.points as [number, number][],
                                  shape: "Magic",
                                  smooth: true,
                                  magic_click_x: rpx,
                                  magic_click_y: rpy,
                                } as any;
                                setSelectedAnnotIdx({ type: "area", idx: areaIdx });
                                updateLocal({ areas: areas2 } as unknown as Partial<PanelInfo>);
                              }
                            } catch (err) {
                              console.error("Magic wand failed:", err);
                            } finally {
                              setMagicWandLoading(false);
                            }
                          })();
                          return;
                        }

                        if (area.shape === "Custom") {
                          // Custom polygon: keep adding points on each click
                          pts.push([rpx, rpy]);
                          areas[areaIdx] = { ...areas[areaIdx], points: pts };
                          setSelectedAnnotIdx({ type: "area", idx: areaIdx });
                          updateLocal({ areas } as unknown as Partial<PanelInfo>);
                          return;
                        }

                        // Standard shapes (Rect/Ellipse/Tri): center + size
                        if (pts.length === 0) {
                          pts.push([rpx, rpy]); // center
                          pts.push([15, 15]);    // default size
                        } else if (pts.length === 1) {
                          const dx = Math.abs(rpx - pts[0][0]) * 2;
                          const dy = Math.abs(rpy - pts[0][1]) * 2;
                          pts.push([Math.max(5, dx), Math.max(5, dy)]);
                        }
                        areas[areaIdx] = { ...areas[areaIdx], points: pts };
                        setSelectedAnnotIdx({ type: "area", idx: areaIdx });
                        updateLocal({ areas } as unknown as Partial<PanelInfo>);
                        return;
                      }

                      // Otherwise add point to last line
                      const lines = [...(local.lines ?? [])];
                      const lastIdx = lines.length - 1;
                      if (lastIdx >= 0) {
                        const lastLine = lines[lastIdx];
                        const pts = [...(lastLine.points ?? [])];
                        // Enforce 2-point max for straight lines only
                        const lt = lastLine.line_type || (lastLine.is_curved ? "curved" : "straight");
                        if (lt === "straight" && pts.length >= 2) return;
                        pts.push([rpx, rpy]);
                        lines[lastIdx] = { ...lines[lastIdx], points: pts };
                        updateLocal({ lines } as unknown as Partial<PanelInfo>);
                      }
                    }}
                  />
                )}
                {/* Draggable symbol indicators overlay */}
                {tabIdx === TAB_ANNOT && local.symbols.length > 0 && (() => {
                  const vbW = svgDims.w || 1000;
                  const vbH = svgDims.h || 1000;
                  return (
                  <svg
                    viewBox={`0 0 ${vbW} ${vbH}`}
                    preserveAspectRatio="none"
                    style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 4 }}
                  >
                    {local.symbols.map((sym, i) => {
                      const isActive = selectedAnnotIdx?.type === "symbol" && selectedAnnotIdx.idx === i;
                      // Convert percentage coords to pixel coords in viewBox space
                      const cx = sym.x / 100 * vbW;
                      const cy = sym.y / 100 * vbH;
                      // Size in pixels: sym.size * iw / 216 (matches backend)
                      const sz = Math.max(5, sym.size * vbW / 216);
                      const svgColor = sym.color;
                      const sw = Math.max(0.2, sz / 12);

                      // Use shared symbol definitions (matches backend exactly)
                      const { fillPolys, strokePolys, filled } = symbolToSvgPoints(
                        sym.shape, cx, cy, sz, sym.rotation || 0
                      );
                      // Stroke scales with symbol size (pixel units)
                      const thinSw = Math.max(vbW * 0.001, sz / 20);

                      // Always show SVG shapes on Annotations tab (base preview is used, not rendered)
                      const shape = (
                        <g>
                          {fillPolys.map((pts, pi) => (
                            <polygon key={`f${pi}`} points={pts}
                              fill={filled ? svgColor : "none"}
                              stroke={svgColor}
                              strokeWidth={filled ? 0.15 : thinSw}
                              strokeLinejoin="round"
                            />
                          ))}
                          {strokePolys.map((pts, pi) => (
                            <polyline key={`s${pi}`} points={pts}
                              fill="none" stroke={svgColor} strokeWidth={thinSw}
                              strokeLinecap="round"
                            />
                          ))}
                        </g>
                      );

                      return (
                        <g key={`sym-svg-${i}`}>
                          {shape}
                          {/* Invisible larger hit area for dragging */}
                          <circle cx={cx} cy={cy}
                            r={Math.max(sz * 0.8, vbW * 0.015)}
                            fill="transparent"
                            stroke={isActive ? "#2196f3" : "transparent"}
                            strokeWidth={isActive ? vbW * 0.002 : 0}
                            strokeDasharray={isActive ? `${vbW*0.005},${vbW*0.003}` : ""}
                            style={{ pointerEvents: "auto", cursor: "grab" }}
                            onClick={() => setSelectedAnnotIdx({ type: "symbol", idx: i })}
                            onMouseDown={(e) => {
                              e.preventDefault(); e.stopPropagation();
                              setSelectedAnnotIdx({ type: "symbol", idx: i });
                              const svgEl = e.currentTarget.closest("svg");
                              const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                              if (!imgEl) return;
                              const onMove = (ev: MouseEvent) => {
                                const rect = imgEl.getBoundingClientRect();
                                const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                updateSymbol(i, { x: Math.round(px), y: Math.round(py) });
                              };
                              const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); setTimeout(() => refreshPreview(), 0); };
                              window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                            }}
                          />
                          {sym.label_text && (() => {
                            // Label position: absolute (pixel coords) if set, otherwise auto
                            const autoX = cx + sz * 0.5;
                            const autoY = cy - sz * 0.25;
                            const lx = (sym.label_position_x >= 0) ? sym.label_position_x / 100 * vbW : autoX;
                            const ly = (sym.label_position_y >= 0) ? sym.label_position_y / 100 * vbH : autoY;
                            const labelFontSize = Math.max(vbW * 0.015, sym.label_font_size * vbW / 216);
                            return (
                              <text x={lx} y={ly}
                                fill={sym.label_color || "#fff"}
                                fontSize={labelFontSize}
                                fontWeight={(sym.label_font_style || []).includes("Bold") ? 700 : 400}
                                fontStyle={(sym.label_font_style || []).includes("Italic") ? "italic" : "normal"}
                                textDecoration={(sym.label_font_style || []).includes("Strikethrough") ? "line-through" : "none"}
                                style={{ pointerEvents: isActive ? "auto" : "none", cursor: isActive ? "grab" : "default" }}
                                onMouseDown={isActive ? (e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  const svgEl = e.currentTarget.closest("svg");
                                  const imgEl = svgEl?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                  if (!imgEl) return;
                                  const onMove = (ev: MouseEvent) => {
                                    const rect = imgEl.getBoundingClientRect();
                                    const px = Math.max(0, Math.min(100, ((ev.clientX - rect.left) / rect.width) * 100));
                                    const py = Math.max(0, Math.min(100, ((ev.clientY - rect.top) / rect.height) * 100));
                                    updateSymbol(i, { label_position_x: Math.round(px * 10) / 10, label_position_y: Math.round(py * 10) / 10 });
                                  };
                                  const onUp = () => {
                                    window.removeEventListener("mousemove", onMove);
                                    window.removeEventListener("mouseup", onUp);
                                  };
                                  window.addEventListener("mousemove", onMove);
                                  window.addEventListener("mouseup", onUp);
                                } : undefined}
                              >
                                {sym.label_text}
                              </text>
                            );
                          })()}
                        </g>
                      );
                    })}
                  </svg>
                  );
                })()}
                {/* Legacy symbol overlay removed — now using SVG shapes above */}
                {/* Magic wand loading indicator */}
                {magicWandLoading && (
                  <Box sx={{
                    position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    bgcolor: "rgba(0,0,0,0.4)", zIndex: 20, borderRadius: 1,
                  }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, bgcolor: "background.paper", px: 2, py: 1, borderRadius: 1, boxShadow: 3 }}>
                      <CircularProgress size={16} />
                      <Typography variant="caption">Processing magic wand selection...</Typography>
                    </Box>
                  </Box>
                )}
                {/* Zoom inset overlay — draggable rectangle showing zoom area */}
                {tabIdx === TAB_ZOOM && local.add_zoom_inset && local.zoom_inset && ["Standard Zoom", "Adjacent Panel"].includes(local.zoom_inset.inset_type) && (() => {
                  const zi = local.zoom_inset!;
                  const ziPreviewEl = document.querySelector('[alt="Panel preview"]') as HTMLImageElement | null;
                  const ziImgW = ziPreviewEl?.clientWidth || 400;
                  const ziImgH = ziPreviewEl?.clientHeight || 400;
                  // Use cropped dimensions (not original) since zoom coords are in crop space
                  const ziActualW = (local.crop_image && local.crop?.length === 4)
                    ? (local.crop[2] - local.crop[0]) : (origFullW > 0 ? origFullW : 1000);
                  const ziActualH = (local.crop_image && local.crop?.length === 4)
                    ? (local.crop[3] - local.crop[1]) : (origFullH > 0 ? origFullH : 1000);
                  // Convert pixel coords to percentage
                  const xPct = (zi.x / ziActualW) * 100;
                  const yPct = (zi.y / ziActualH) * 100;
                  const wPct = (zi.width / ziActualW) * 100;
                  const hPct = (zi.height / ziActualH) * 100;
                  const tgtXPct = ((zi.target_x ?? 200) / ziActualW) * 100;
                  const tgtYPct = ((zi.target_y ?? 200) / ziActualH) * 100;
                  return (
                    <>
                      {/* Source area rectangle — draggable + resizable like crop */}
                      <Box
                        sx={{
                          position: "absolute",
                          left: `${xPct}%`, top: `${yPct}%`,
                          width: `${wPct}%`, height: `${hPct}%`,
                          border: `2px solid ${zi.rectangle_color}`,
                          cursor: "grab", zIndex: 8,
                          "&:hover": { boxShadow: `0 0 8px ${zi.rectangle_color}` },
                        }}
                        onMouseDown={(e) => {
                          e.preventDefault(); e.stopPropagation();
                          const imgEl = (e.currentTarget.parentElement?.querySelector("img")) as HTMLImageElement | null;
                          if (!imgEl) return;
                          const startX = e.clientX, startY = e.clientY;
                          const startZx = zi.x, startZy = zi.y;
                          const onMove = (ev: MouseEvent) => {
                            const rect = imgEl.getBoundingClientRect();
                            const dx = ((ev.clientX - startX) / rect.width) * ziActualW;
                            const dy = ((ev.clientY - startY) / rect.height) * ziActualH;
                            updateLocal({ zoom_inset: { ...zi, x: Math.max(0, Math.round(startZx + dx)), y: Math.max(0, Math.round(startZy + dy)) } });
                          };
                          const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); setTimeout(() => refreshPreview(), 0); };
                          window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                        }}
                      >
                        {/* Corner resize handles */}
                        {(["nw", "ne", "sw", "se"] as const).map((corner) => {
                          const isRight = corner.includes("e");
                          const isBottom = corner.includes("s");
                          return (
                            <Box
                              key={corner}
                              sx={{
                                position: "absolute",
                                [isRight ? "right" : "left"]: -4,
                                [isBottom ? "bottom" : "top"]: -4,
                                width: 8, height: 8,
                                bgcolor: zi.rectangle_color,
                                border: "1px solid rgba(0,0,0,0.5)",
                                cursor: `${corner}-resize`,
                                zIndex: 10,
                              }}
                              onMouseDown={(e) => {
                                e.preventDefault(); e.stopPropagation();
                                // DOM walk: corner handle → source-rect → wrapper.
                                // The previous `closest("[style]")` matched the
                                // first ancestor with an inline style attribute,
                                // which is unpredictable since MUI's sx compiles
                                // to emotion classes (not inline style). It often
                                // landed on the dialog backdrop, whose
                                // getBoundingClientRect bears no relation to the
                                // panel image — so dx/dy scaled by the wrong rect
                                // and the source rectangle "jumped" on tiny drags.
                                const imgEl = e.currentTarget.parentElement?.parentElement?.querySelector("img") as HTMLImageElement | null;
                                if (!imgEl) return;
                                const startX = e.clientX, startY = e.clientY;
                                const startZi = { ...zi };
                                const onMove = (ev: MouseEvent) => {
                                  const rect = imgEl.getBoundingClientRect();
                                  const dx = ((ev.clientX - startX) / rect.width) * ziActualW;
                                  const dy = ((ev.clientY - startY) / rect.height) * ziActualH;
                                  let newX = startZi.x, newY = startZi.y, newW = startZi.width, newH = startZi.height;
                                  if (corner.includes("e")) { newW = Math.max(20, Math.round(startZi.width + dx)); }
                                  if (corner.includes("w")) { newX = Math.max(0, Math.round(startZi.x + dx)); newW = Math.max(20, Math.round(startZi.width - dx)); }
                                  if (corner.includes("s")) { newH = Math.max(20, Math.round(startZi.height + dy)); }
                                  if (corner.includes("n")) { newY = Math.max(0, Math.round(startZi.y + dy)); newH = Math.max(20, Math.round(startZi.height - dy)); }
                                  updateLocal({ zoom_inset: { ...zi, x: newX, y: newY, width: newW, height: newH } });
                                };
                                const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); setTimeout(() => refreshPreview(), 0); };
                                window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                              }}
                            />
                          );
                        })}
                      </Box>
                      {/* Target: shows zoomed content from source area (Standard Zoom only) */}
                      {zi.inset_type !== "Standard Zoom" ? null : (() => {
                        const hasExt = zi.separate_image_name && zi.separate_image_name !== "select";
                        // When external image: inset size follows external crop aspect ratio
                        const insetWPct = hasExt
                          ? wPct * zi.zoom_factor  // width stays as zoom * source width
                          : wPct * zi.zoom_factor;
                        const insetHPct = hasExt && extImageDims.w > 0 && extImageDims.h > 0
                          ? insetWPct * ((zi.height_inset || extImageDims.h) / (zi.width_inset || extImageDims.w)) * (ziActualW / ziActualH)
                          : hPct * zi.zoom_factor;
                        return <Box
                        sx={{
                          position: "absolute",
                          left: `${tgtXPct}%`, top: `${tgtYPct}%`,
                          width: `${insetWPct}%`, height: `${insetHPct}%`,
                          border: `2px solid ${zi.rectangle_color}`,
                          overflow: "hidden", cursor: "grab", zIndex: 7,
                          bgcolor: "rgba(0,0,0,0.3)",
                        }}
                        ref={(el: HTMLDivElement | null) => {
                          if (!el) return;
                          const existingCanvas = el.querySelector("canvas");
                          if (existingCanvas) existingCanvas.remove();
                          const canvas = document.createElement("canvas");
                          const tgtW = el.clientWidth;
                          const tgtH = el.clientHeight;
                          if (tgtW < 2 || tgtH < 2) return;
                          canvas.width = tgtW;
                          canvas.height = tgtH;
                          canvas.style.width = "100%";
                          canvas.style.height = "100%";
                          canvas.style.pointerEvents = "none";
                          const ctx = canvas.getContext("2d");
                          if (!ctx) return;
                          ctx.imageSmoothingEnabled = true;
                          ctx.imageSmoothingQuality = "high";

                          const extName = zi.separate_image_name || "";
                          if (extName && extName !== "select" && extImageThumb) {
                            // Draw external image crop
                            const extImg = new window.Image();
                            extImg.onload = () => {
                              const exi = zi.x_inset ?? 0;
                              const eyi = zi.y_inset ?? 0;
                              const ewi = zi.width_inset ?? extImg.naturalWidth;
                              const ehi = zi.height_inset ?? extImg.naturalHeight;
                              const sx = (exi / extImageDims.w) * extImg.naturalWidth;
                              const sy = (eyi / extImageDims.h) * extImg.naturalHeight;
                              const sw = (ewi / extImageDims.w) * extImg.naturalWidth;
                              const sh = (ehi / extImageDims.h) * extImg.naturalHeight;
                              try { ctx.drawImage(extImg, sx, sy, sw, sh, 0, 0, tgtW, tgtH); } catch {}
                            };
                            extImg.src = `data:image/png;base64,${extImageThumb}`;
                          } else {
                            // Default: crop source region from preview image
                            const imgEl = el.parentElement?.querySelector("img") as HTMLImageElement | null;
                            if (!imgEl || !imgEl.naturalWidth) return;
                            const srcX = (zi.x / ziActualW) * imgEl.naturalWidth;
                            const srcY = (zi.y / ziActualH) * imgEl.naturalHeight;
                            const srcW = (zi.width / ziActualW) * imgEl.naturalWidth;
                            const srcH = (zi.height / ziActualH) * imgEl.naturalHeight;
                            try { ctx.drawImage(imgEl, srcX, srcY, srcW, srcH, 0, 0, tgtW, tgtH); } catch {}
                          }
                          el.appendChild(canvas);
                        }}
                        onMouseDown={(e) => {
                          e.preventDefault(); e.stopPropagation();
                          const imgEl = (e.currentTarget.parentElement?.querySelector("img")) as HTMLImageElement | null;
                          if (!imgEl) return;
                          const startX = e.clientX, startY = e.clientY;
                          const startTx = zi.target_x ?? 200, startTy = zi.target_y ?? 200;
                          const onMove = (ev: MouseEvent) => {
                            const rect = imgEl.getBoundingClientRect();
                            const dx = ((ev.clientX - startX) / rect.width) * ziActualW;
                            const dy = ((ev.clientY - startY) / rect.height) * ziActualH;
                            updateLocal({ zoom_inset: { ...zi, target_x: Math.round(startTx + dx), target_y: Math.round(startTy + dy) } });
                          };
                          const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); setTimeout(() => refreshPreview(), 0); };
                          window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
                        }}
                      >
                        {/* Bottom-right resize handle (proportional) */}
                        <Box sx={{
                          position: "absolute", bottom: -4, right: -4, width: 10, height: 10,
                          bgcolor: "#4FC3F7", border: "1px solid #fff", cursor: "se-resize", zIndex: 2,
                        }}
                        onMouseDown={(ev) => {
                          ev.preventDefault(); ev.stopPropagation();
                          // Same fix as the source-rect corner handles: walk
                          // the DOM explicitly (handle → target-rect →
                          // wrapper) instead of using closest("[style]"),
                          // which lands on whichever ancestor first has an
                          // inline style attribute and is therefore
                          // unpredictable under MUI's emotion-based sx.
                          const imgEl2 = ev.currentTarget.parentElement?.parentElement?.querySelector("img") as HTMLImageElement | null;
                          if (!imgEl2) return;
                          const rect2 = imgEl2.getBoundingClientRect();
                          const startMX = ev.clientX;
                          const startZoom = zi.zoom_factor;
                          const onMove2 = (em: MouseEvent) => {
                            const dx = ((em.clientX - startMX) / rect2.width) * 100;
                            // Scale zoom factor proportionally to drag distance
                            const newZoom = Math.max(0.5, startZoom + dx * startZoom / 50);
                            updateLocal({ zoom_inset: { ...zi, zoom_factor: Math.round(newZoom * 10) / 10 } });
                          };
                          const onUp2 = () => { window.removeEventListener("mousemove", onMove2); window.removeEventListener("mouseup", onUp2); setTimeout(() => refreshPreview(), 0); };
                          window.addEventListener("mousemove", onMove2); window.addEventListener("mouseup", onUp2);
                        }}
                        />
                      </Box>
                      })()}
                      {/* Connecting lines — funnel from source corners to inset corners (Standard Zoom only) */}
                      {zi.inset_type === "Standard Zoom" && (() => {
                        // Source corners (TL, TR, BR, BL)
                        const sc = [
                          [xPct, yPct], [xPct + wPct, yPct],
                          [xPct + wPct, yPct + hPct], [xPct, yPct + hPct]
                        ];
                        // Inset corners — use same sizing as the inset box
                        const hasExtLine = zi.separate_image_name && zi.separate_image_name !== "select";
                        const tgtW = wPct * zi.zoom_factor;
                        const tgtH = hasExtLine && extImageDims.w > 0 && extImageDims.h > 0
                          ? tgtW * ((zi.height_inset || extImageDims.h) / (zi.width_inset || extImageDims.w)) * (ziActualW / ziActualH)
                          : hPct * zi.zoom_factor;
                        const dc = [
                          [tgtXPct, tgtYPct], [tgtXPct + tgtW, tgtYPct],
                          [tgtXPct + tgtW, tgtYPct + tgtH], [tgtXPct, tgtYPct + tgtH]
                        ];
                        // Centers
                        const srcCx = xPct + wPct / 2, srcCy = yPct + hPct / 2;
                        const dstCx = tgtXPct + tgtW / 2, dstCy = tgtYPct + tgtH / 2;
                        // Sort source corners by distance to inset center, take 2 closest
                        const srcSorted = sc.map((c, i) => ({ c, i, d: Math.hypot(c[0] - dstCx, c[1] - dstCy) }))
                          .sort((a, b) => a.d - b.d);
                        const s1 = srcSorted[0].c, s2 = srcSorted[1].c;
                        // Sort inset corners by distance to source center, take 2 closest
                        const dstSorted = dc.map((c, i) => ({ c, i, d: Math.hypot(c[0] - srcCx, c[1] - srcCy) }))
                          .sort((a, b) => a.d - b.d);
                        const d1 = dstSorted[0].c, d2 = dstSorted[1].c;
                        // Connect: try both pairings, pick the non-crossing one
                        const ccw = (a: number[], b: number[], c: number[]) =>
                          (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0]);
                        const cross = (p1: number[], p2: number[], q1: number[], q2: number[]) =>
                          ccw(p1,q1,q2) !== ccw(p2,q1,q2) && ccw(p1,p2,q1) !== ccw(p1,p2,q2);
                        let l1s = s1, l1e = d1, l2s = s2, l2e = d2;
                        if (cross(l1s, l1e, l2s, l2e)) {
                          // Swap destination pairing
                          l1e = d2; l2e = d1;
                        }
                        return (
                          <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 6 }}>
                            <line x1={`${l1s[0]}%`} y1={`${l1s[1]}%`} x2={`${l1e[0]}%`} y2={`${l1e[1]}%`} stroke={zi.line_color} strokeWidth={zi.line_width} opacity={0.6} />
                            <line x1={`${l2s[0]}%`} y1={`${l2s[1]}%`} x2={`${l2e[0]}%`} y2={`${l2e[1]}%`} stroke={zi.line_color} strokeWidth={zi.line_width} opacity={0.6} />
                          </svg>
                        );
                      })()}
                    </>
                  );
                })()}
                {/* Adjacent Panel: connecting lines from source to panel edge + label */}
                {tabIdx === TAB_ZOOM && local.add_zoom_inset && local.zoom_inset?.inset_type === "Adjacent Panel" && (() => {
                  const zi = local.zoom_inset!;
                  const adjActW = (local.crop_image && local.crop?.length === 4) ? (local.crop[2] - local.crop[0]) : (origFullW > 0 ? origFullW : 1000);
                  const adjActH = (local.crop_image && local.crop?.length === 4) ? (local.crop[3] - local.crop[1]) : (origFullH > 0 ? origFullH : 1000);
                  const sxP = (zi.x / adjActW) * 100;
                  const syP = (zi.y / adjActH) * 100;
                  const swP = (zi.width / adjActW) * 100;
                  const shP = (zi.height / adjActH) * 100;
                  // Explicit corner matching — source edge corners to panel edge
                  const side = zi.side || "Right";
                  let s1: [number, number], s2: [number, number];
                  let e1: [number, number], e2: [number, number];
                  if (side === "Right") {
                    s1 = [sxP + swP, syP]; s2 = [sxP + swP, syP + shP];
                    e1 = [100, syP]; e2 = [100, syP + shP];
                  } else if (side === "Left") {
                    s1 = [sxP, syP]; s2 = [sxP, syP + shP];
                    e1 = [0, syP]; e2 = [0, syP + shP];
                  } else if (side === "Top") {
                    s1 = [sxP, syP]; s2 = [sxP + swP, syP];
                    e1 = [sxP, 0]; e2 = [sxP + swP, 0];
                  } else {
                    s1 = [sxP, syP + shP]; s2 = [sxP + swP, syP + shP];
                    e1 = [sxP, 100]; e2 = [sxP + swP, 100];
                  }
                  return (
                    <>
                      <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 6 }}>
                        <line x1={`${s1[0]}%`} y1={`${s1[1]}%`} x2={`${e1[0]}%`} y2={`${e1[1]}%`} stroke={zi.line_color} strokeWidth={zi.line_width} opacity={0.6} />
                        <line x1={`${s2[0]}%`} y1={`${s2[1]}%`} x2={`${e2[0]}%`} y2={`${e2[1]}%`} stroke={zi.line_color} strokeWidth={zi.line_width} opacity={0.6} />
                      </svg>
                      <Typography variant="caption" sx={{ position: "absolute", bottom: 8, right: 8, bgcolor: "rgba(0,0,0,0.7)", color: "#fff", px: 1, py: 0.25, borderRadius: 1, fontSize: "0.6rem", zIndex: 8 }}>
                        Zoomed region → {side} panel
                      </Typography>
                    </>
                  );
                })()}
              </Box>{/* end position wrapper */}
            </Box>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 4 }}>
                {local.image_name ? "Loading preview..." : "No image assigned"}
              </Typography>
            )}
          </Box>
        </Box>{/* end horizontal split */}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleSave}
          disabled={(() => {
            if (!local?.add_zoom_inset || !local?.zoom_inset || local.zoom_inset.inset_type !== "Adjacent Panel") return false;
            const s = local.zoom_inset.side || "Right";
            const dr = s === "Top" ? -1 : s === "Bottom" ? 1 : 0;
            const dc = s === "Left" ? -1 : s === "Right" ? 1 : 0;
            const tr = row + dr, tc = col + dc;
            const inBounds = tr >= 0 && tr < (config?.rows ?? 0) && tc >= 0 && tc < (config?.cols ?? 0);
            if (!inBounds) return true;
            const hasImage = !!config?.panels?.[tr]?.[tc]?.image_name;
            return hasImage;
          })()}
        >
          Apply
        </Button>
      </DialogActions>
      {/* Sync seek to row/column result. Sits inside the EditPanelDialog
          render so it's visually anchored to the Edit Panel context.
          maxWidth=xs keeps it compact for a short status message. */}
      <Dialog
        open={!!syncSeekResult}
        onClose={() => setSyncSeekResult(null)}
        maxWidth="xs"
      >
        <DialogTitle>{syncSeekResult?.title ?? ""}</DialogTitle>
        <DialogContent>
          <Typography>{syncSeekResult?.message ?? ""}</Typography>
        </DialogContent>
        <DialogActions>
          <Button variant="contained" onClick={() => setSyncSeekResult(null)}>
            OK
          </Button>
        </DialogActions>
      </Dialog>
    </Dialog>
  );
}
