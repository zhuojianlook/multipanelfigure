/* ──────────────────────────────────────────────────────────
   CollageView — Collage Assembly canvas.

   Pan + zoom matches the multi-panel preview pane: scroll
   wheel to zoom, drag empty canvas to pan, "Reset view"
   recentres at fit-to-window. Items are click-drag positioned
   with snap-to-grid (default 50 px) and resized via
   aspect-locked handles on all four corners — no stretching.
   ────────────────────────────────────────────────────────── */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Button,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
  Divider,
  ToggleButton,
  Menu,
  MenuItem,
  ListSubheader,
} from "@mui/material";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import GridOnIcon from "@mui/icons-material/GridOn";
import GridOffIcon from "@mui/icons-material/GridOff";
import StraightenIcon from "@mui/icons-material/Straighten";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import AspectRatioIcon from "@mui/icons-material/AspectRatio";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import TextFieldsIcon from "@mui/icons-material/TextFields";
import CropIcon from "@mui/icons-material/Crop";
import HorizontalRuleIcon from "@mui/icons-material/HorizontalRule";
import {
  useCollageStore,
  DEFAULT_CANVAS_W,
  DEFAULT_CANVAS_H,
} from "../../store/collageStore";
import type { CollageItem } from "../../store/collageStore";
import { useFigureStore } from "../../store/figureStore";
import { CollageStrip } from "./CollageStrip";
import { RichTextEditor } from "../dialogs/RichTextEditor";
import type { StyledSegment } from "../../api/types";
import { api } from "../../api/client";
import { confirm as confirmDialog, alert as alertDialog } from "../shared/ConfirmDialog";

type Corner = "nw" | "ne" | "sw" | "se";

/** Map a stored font value to a CSS font-family. Custom fonts are
 *  registered (loadCustomFonts.ts) under a family equal to the file name
 *  without its extension, so `arial.ttf` → `"arial", Arial, sans-serif`.
 *  Legacy values that are already plain family names (e.g. "Arial") pass
 *  through unchanged. */
function cssFamily(name?: string): string {
  if (!name) return "Arial, sans-serif";
  const stripped = name.replace(/\.(ttf|otf|ttc|woff2?|TTF|OTF|TTC)$/i, "");
  return `"${stripped}", Arial, sans-serif`;
}

/** Build the React inline style for one rich-text segment (per-character
 *  styling), inheriting the text box's base size/colour where the segment
 *  doesn't override them. Used for live preview AND export parity. */
function segmentStyle(
  seg: StyledSegment,
  baseSize: number,
  baseColor: string,
): React.CSSProperties {
  const styles = seg.font_style ?? [];
  const isSuper = styles.includes("Superscript");
  const isSub = styles.includes("Subscript");
  const baseSeg = seg.font_size ?? baseSize;
  return {
    fontFamily: cssFamily(seg.font_name),
    fontSize: `${isSuper || isSub ? baseSeg * 0.7 : baseSeg}px`,
    color: seg.color || baseColor,
    fontWeight: styles.includes("Bold") ? 700 : 400,
    fontStyle: styles.includes("Italic") ? "italic" : "normal",
    textDecoration:
      [
        styles.includes("Underline") ? "underline" : "",
        styles.includes("Strikethrough") ? "line-through" : "",
      ]
        .filter(Boolean)
        .join(" ") || "none",
    verticalAlign: isSuper ? "super" : isSub ? "sub" : "baseline",
    whiteSpace: "pre-wrap",
  };
}

/* Canvas size presets for common journals + paper sizes. Dimensions are
   in pixels at 300 DPI (px = mm / 25.4 × 300). Widths follow each
   journal's column/figure-width guidelines; heights use the journal's
   typical maximum figure height (or full page for paper sizes) so a
   plate composed to a preset will fit the page. */
const MM = (mm: number) => Math.round((mm / 25.4) * 300);
/** Standard inter-column gutter for column guide overlays (px @ 300 DPI). */
const GUIDE_GUTTER_PX = MM(4);
// `cols` = number of journal columns this preset implies; drives the column
// guide-line overlay (0 / 1 → no internal dividers).
interface CanvasPreset { label: string; w: number; h: number; cols: number; }
interface PresetGroup { group: string; presets: CanvasPreset[]; }
const CANVAS_PRESET_GROUPS: PresetGroup[] = [
  {
    group: "Nature",
    presets: [
      { label: "Single column (89 mm)", w: MM(89), h: MM(247), cols: 1 },
      { label: "Double column (183 mm)", w: MM(183), h: MM(247), cols: 2 },
    ],
  },
  {
    group: "Science",
    presets: [
      { label: "1 column (55 mm)", w: MM(55), h: MM(240), cols: 1 },
      { label: "2 columns (120 mm)", w: MM(120), h: MM(240), cols: 2 },
      { label: "3 columns (183 mm)", w: MM(183), h: MM(240), cols: 3 },
    ],
  },
  {
    group: "Cell",
    presets: [
      { label: "1 column (85 mm)", w: MM(85), h: MM(240), cols: 1 },
      { label: "1.5 column (114 mm)", w: MM(114), h: MM(240), cols: 0 },
      { label: "2 columns (174 mm)", w: MM(174), h: MM(240), cols: 2 },
    ],
  },
  {
    group: "PNAS",
    presets: [
      { label: "1 column (87 mm)", w: MM(87), h: MM(240), cols: 1 },
      { label: "1.5 column (114 mm)", w: MM(114), h: MM(240), cols: 0 },
      { label: "2 columns (178 mm)", w: MM(178), h: MM(240), cols: 2 },
    ],
  },
  {
    group: "eLife / general",
    presets: [
      { label: "eLife full width (175 mm)", w: MM(175), h: MM(240), cols: 2 },
      { label: "Square (180 mm)", w: MM(180), h: MM(180), cols: 0 },
    ],
  },
  {
    group: "Paper size",
    presets: [
      { label: "A4 portrait", w: MM(210), h: MM(297), cols: 0 },
      { label: "A4 landscape", w: MM(297), h: MM(210), cols: 0 },
      { label: "US Letter portrait", w: MM(216), h: MM(279), cols: 0 },
      { label: "US Letter landscape", w: MM(279), h: MM(216), cols: 0 },
    ],
  },
];

export function CollageView() {
  const items = useCollageStore((s) => s.items);
  const canvasW = useCollageStore((s) => s.canvasW);
  const canvasH = useCollageStore((s) => s.canvasH);
  const background = useCollageStore((s) => s.background);
  const selectedId = useCollageStore((s) => s.selectedId);
  const selectedIds = useCollageStore((s) => s.selectedIds);
  const elemSyncItemId = useCollageStore((s) => s.elemSyncItemId);
  const elemListByItem = useCollageStore((s) => s.elemListByItem);
  const elemSelByItem = useCollageStore((s) => s.elemSelByItem);
  const elemOverridesByItem = useCollageStore((s) => s.elemOverridesByItem);
  const hoveredElem = useCollageStore((s) => s.hoveredElem);
  const toggleElemSel = useCollageStore((s) => s.toggleElemSel);
  const setElemOverride = useCollageStore((s) => s.setElemOverride);
  const setHoveredElem = useCollageStore((s) => s.setHoveredElem);
  const setElemSyncItem = useCollageStore((s) => s.setElemSyncItem);
  const setElemList = useCollageStore((s) => s.setElemList);
  const fonts = useFigureStore((s) => s.fonts);
  const gridVisible = useCollageStore((s) => s.gridVisible);
  const snapEnabled = useCollageStore((s) => s.snapEnabled);
  const gridStep = useCollageStore((s) => s.gridStep);
  const guideColumns = useCollageStore((s) => s.guideColumns);
  const guideGutter = useCollageStore((s) => s.guideGutter);
  const guidesVisible = useCollageStore((s) => s.guidesVisible);
  const setSelectedId = useCollageStore((s) => s.setSelectedId);
  const setSelectedIds = useCollageStore((s) => s.setSelectedIds);
  const toggleSelected = useCollageStore((s) => s.toggleSelected);
  const updateItem = useCollageStore((s) => s.updateItem);
  const moveItem = useCollageStore((s) => s.moveItem);
  const bringToFront = useCollageStore((s) => s.bringToFront);
  const setCanvasSize = useCollageStore((s) => s.setCanvasSize);
  const setBackground = useCollageStore((s) => s.setBackground);
  const setColumnGuides = useCollageStore((s) => s.setColumnGuides);
  const setGuidesVisible = useCollageStore((s) => s.setGuidesVisible);
  const setGridVisible = useCollageStore((s) => s.setGridVisible);
  const setSnapEnabled = useCollageStore((s) => s.setSnapEnabled);
  const setGridStep = useCollageStore((s) => s.setGridStep);
  const addItem = useCollageStore((s) => s.addItem);
  const loadProject = useFigureStore((s) => s.loadProject);
  const setMode = useCollageStore((s) => s.setMode);

  const containerRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const projectInputRef = useRef<HTMLInputElement>(null);
  // Pan + user-zoom state. The display scale that gets applied to the
  // canvas wrapper is fitScale × userZoom, where fitScale is the
  // automatically-computed scale that makes the page fit the viewport
  // (≤1, recomputed on container/canvas resize). userZoom and pan let
  // the user override that — wheel to zoom, drag to pan.
  const [fitScale, setFitScale] = useState(1);
  const [userZoom, setUserZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Free-text drafts for the canvas W/H fields so the user can type
  // intermediate values (e.g. clearing to type a new number) without the
  // controlled clamp coercing every keystroke. Committed (clamped) on
  // blur / Enter. Re-seeded whenever the store value changes (e.g. preset).
  const [canvasWText, setCanvasWText] = useState(String(canvasW));
  const [canvasHText, setCanvasHText] = useState(String(canvasH));
  useEffect(() => { setCanvasWText(String(canvasW)); }, [canvasW]);
  useEffect(() => { setCanvasHText(String(canvasH)); }, [canvasH]);
  const commitCanvasW = () => {
    const w = Math.max(100, Math.min(8000, Math.round(Number(canvasWText) || canvasW)));
    setCanvasSize(w, canvasH);
    setCanvasWText(String(w));
  };
  const commitCanvasH = () => {
    const h = Math.max(100, Math.min(8000, Math.round(Number(canvasHText) || canvasH)));
    setCanvasSize(canvasW, h);
    setCanvasHText(String(h));
  };

  // Canvas-size presets menu anchor.
  const [presetAnchor, setPresetAnchor] = useState<null | HTMLElement>(null);

  // Marquee (rubber-band) selection rect, in canvas coordinates. Null when
  // not dragging a selection box.
  const [marquee, setMarquee] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  // Inline text editing: id of the text item currently being edited.
  const [editingTextId, setEditingTextId] = useState<string | null>(null);
  // Rich-text editor for a whole text box (double-click). Gives all fonts,
  // bold/italic/underline/strikethrough/super/subscript and per-character
  // styling. Anchored to the text box's DOM node.
  const [textEditor, setTextEditor] = useState<
    { itemId: string; anchorEl: HTMLElement; segments: StyledSegment[]; plainText: string } | null
  >(null);

  /** Open the rich-text editor for a text item, seeding it with the item's
   *  existing styled segments (or a single segment built from its whole-box
   *  font props so nothing is lost on first open). */
  const openTextEditor = (it: CollageItem, anchorEl: HTMLElement) => {
    const seed: StyledSegment[] = it.styledSegments?.length
      ? it.styledSegments
      : [
          {
            text: it.text || "",
            color: it.fontColor || "#000000",
            font_name: it.fontFamily || "arial.ttf",
            font_size: it.fontSize || 28,
            font_style: [
              ...(it.fontBold ? ["Bold"] : []),
              ...(it.fontItalic ? ["Italic"] : []),
              ...(it.fontUnderline ? ["Underline"] : []),
            ],
          },
        ];
    setTextEditor({ itemId: it.id, anchorEl, segments: seed, plainText: it.text || "" });
  };
  // Per-element rich-text customization editor (opened by double-clicking a
  // hotspot). Anchored to the clicked hotspot.
  const [elemEditor, setElemEditor] = useState<
    { itemId: string; elemId: string; anchorEl: HTMLElement; segments: StyledSegment[]; plainText: string } | null
  >(null);

  // Re-render one figure item with the current sync pt + element selection +
  // per-element style overrides (used after a customization edit).
  const rerenderFigure = async (itemId: string) => {
    const it = useCollageStore.getState().items.find((i) => i.id === itemId);
    if (!it || it.kind !== "figure" || !it.projectPath) return;
    const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
    const pt = useCollageStore.getState().globalHeaderPt;
    const sel = useCollageStore.getState().elemSelByItem[itemId];
    const elementIds = sel ? Object.keys(sel).filter((k) => sel[k]) : null;
    const overrides = useCollageStore.getState().elemOverridesByItem[itemId] || null;
    try {
      const resp = await api.renderCollageFigure(
        it.projectPath, pt ?? null, Math.max(0.001, scale), it.w, elementIds,
        overrides as Record<string, unknown> | null,
      );
      if (resp?.image && resp.width && resp.height) {
        updateItem(itemId, {
          src: `data:image/png;base64,${resp.image}`,
          naturalW: resp.width, naturalH: resp.height,
          h: it.w / (resp.width / resp.height),
        });
      }
    } catch (e) {
      console.error("[collage] re-render figure failed", e);
    }
  };
  // Crop mode: the image item being cropped + the crop rect in the item's
  // displayed pixel coordinates (0..it.w, 0..it.h).
  const [cropItemId, setCropItemId] = useState<string | null>(null);
  const [cropRect, setCropRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  // Selecting a figure on the canvas reveals its text-element hotspots (so
  // they're discoverable without expanding the sidebar tree). Loads the
  // element list + geometry on first select. Clicking empty canvas hides them.
  useEffect(() => {
    const it = items.find((i) => i.id === selectedId);
    if (it && it.kind === "figure" && it.projectPath) {
      setElemSyncItem(it.id);
      if (!elemListByItem[it.id]) {
        api.getFigureElements(it.projectPath)
          .then(({ elements }) => setElemList(it.id, elements))
          .catch((e) => console.error("[collage] load elements (select) failed", e));
      }
    } else if (!it) {
      setElemSyncItem(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  // Add a new text item at the canvas center and start editing it.
  const insertText = () => {
    const id = addItem({
      kind: "text",
      src: "",
      name: "Text",
      text: "Double-click to edit",
      x: Math.round(canvasW / 2 - 160),
      y: Math.round(canvasH / 2 - 30),
      w: 320, h: 60, naturalW: 320, naturalH: 60,
      fontSize: 28, fontColor: "#000000", fontFamily: "Arial",
      fontBold: false, fontItalic: false, align: "left",
    });
    setSelectedId(id);
    setEditingTextId(id);
  };

  // Add a horizontal divider line at the canvas center.
  const insertLine = () => {
    const id = addItem({
      kind: "line",
      src: "",
      name: "Line",
      x: Math.round(canvasW / 2 - 200),
      y: Math.round(canvasH / 2 - 12),
      w: 400, h: 24, naturalW: 400, naturalH: 24,
      lineColor: "#000000", lineThickness: 3, lineStyle: "solid",
    });
    setSelectedId(id);
  };

  const selectedItem = items.find((it) => it.id === selectedId) || null;
  const cropEligible = selectedIds.length === 1 && selectedItem?.kind === "image";

  const startCrop = () => {
    if (!cropEligible || !selectedItem) return;
    setCropItemId(selectedItem.id);
    // Initialise the crop rect to a 10% inset of the displayed item.
    setCropRect({
      x: selectedItem.w * 0.1, y: selectedItem.h * 0.1,
      w: selectedItem.w * 0.8, h: selectedItem.h * 0.8,
    });
  };
  const cancelCrop = () => { setCropItemId(null); setCropRect(null); };
  const applyCrop = async () => {
    const it = items.find((i) => i.id === cropItemId);
    if (!it || !cropRect) { cancelCrop(); return; }
    const scaleX = it.naturalW / Math.max(1, it.w);
    const scaleY = it.naturalH / Math.max(1, it.h);
    const sx = Math.max(0, cropRect.x * scaleX);
    const sy = Math.max(0, cropRect.y * scaleY);
    const sw = Math.max(1, cropRect.w * scaleX);
    const sh = Math.max(1, cropRect.h * scaleY);
    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const img = new window.Image();
        img.onload = () => {
          const cv = document.createElement("canvas");
          cv.width = Math.round(sw); cv.height = Math.round(sh);
          const ctx = cv.getContext("2d");
          if (!ctx) { reject(new Error("no ctx")); return; }
          ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cv.width, cv.height);
          resolve(cv.toDataURL("image/png"));
        };
        img.onerror = () => reject(new Error("img load failed"));
        img.src = it.src;
      });
      updateItem(it.id, {
        src: dataUrl,
        cropOrigSrc: it.cropOrigSrc ?? it.src,
        naturalW: Math.round(sw),
        naturalH: Math.round(sh),
        w: Math.round(cropRect.w),
        h: Math.round(cropRect.h),
      });
    } catch (e) {
      console.error("[collage] crop failed", e);
    }
    cancelCrop();
  };

  /** Re-fit the page to the viewport — runs on mount + resize. */
  useEffect(() => {
    const compute = () => {
      const el = containerRef.current;
      if (!el) return;
      const pad = 32;
      const availW = el.clientWidth - pad;
      const availH = el.clientHeight - pad;
      if (availW <= 0 || availH <= 0) return;
      const sx = availW / canvasW;
      const sy = availH / canvasH;
      setFitScale(Math.min(1, Math.min(sx, sy)));
    };
    compute();
    const ro = new ResizeObserver(compute);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [canvasW, canvasH]);

  const displayScale = fitScale * userZoom;

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.z - b.z),
    [items],
  );

  /** Snap value to grid step when snap is enabled. */
  const snap = (v: number) => (snapEnabled ? Math.round(v / gridStep) * gridStep : v);

  const resetView = () => {
    setUserZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const applyCanvasPreset = (w: number, h: number, cols: number) => {
    setCanvasSize(w, h);
    // Set column guides for multi-column presets; clear them otherwise.
    setColumnGuides(cols >= 2 ? cols : 0, cols >= 2 ? GUIDE_GUTTER_PX : 0);
    if (cols >= 2) setGuidesVisible(true);
    resetView();
    setPresetAnchor(null);
  };

  /** Wheel: zoom around the cursor for an intuitive Photoshop-style
   *  feel — the point under the mouse stays put while everything else
   *  scales toward or away from it. */
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newUserZoom = Math.max(0.1, Math.min(20, userZoom * factor));
    const realFactor = newUserZoom / userZoom;
    if (realFactor === 1) return;
    // Adjust pan so the cursor's canvas-coordinate stays fixed under the
    // pointer through the zoom. Standard zoom-around-point math.
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setPan((p) => ({
        x: mx - realFactor * (mx - p.x),
        y: my - realFactor * (my - p.y),
      }));
    }
    setUserZoom(newUserZoom);
  };

  /** Pan with left-drag on empty canvas (or middle-drag anywhere). */
  const handleViewportMouseDown = (e: React.MouseEvent) => {
    if (e.target !== e.currentTarget) return;
    setSelectedId(null);
    const startX = e.clientX;
    const startY = e.clientY;
    const startPan = { ...pan };
    const onMove = (ev: MouseEvent) => {
      setPan({ x: startPan.x + (ev.clientX - startX), y: startPan.y + (ev.clientY - startY) });
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const onPickImageFile = async (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const src = String(reader.result);
      const img = new window.Image();
      img.onload = () => {
        const targetMax = Math.min(canvasW, canvasH) * 0.4;
        const aspect = img.naturalWidth / img.naturalHeight;
        const w = aspect >= 1 ? targetMax : targetMax * aspect;
        const h = aspect >= 1 ? targetMax / aspect : targetMax;
        const offset = items.length * 24;
        addItem({
          kind: "image",
          src,
          name: file.name,
          x: snap(40 + offset),
          y: snap(40 + offset),
          w,
          h,
          naturalW: img.naturalWidth,
          naturalH: img.naturalHeight,
        });
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  const handleImportClick = () => fileInputRef.current?.click();
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    files.forEach(onPickImageFile);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleImportProjectPick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (projectInputRef.current) projectInputRef.current.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".mpf")) {
      await alertDialog({
        title: "Wrong file type",
        body: "Please choose a .mpf project file.",
      });
      return;
    }
    const ok = await confirmDialog({
      title: "Load project into builder",
      body: `Load "${file.name}" into the Multi-Panel Builder?\n\n`
        + "Your current builder state will be replaced. Then you can review "
        + 'the figure and click "Add to Collage" to insert it.',
      confirmLabel: "Load",
      destructive: true,
    });
    if (!ok) return;
    try {
      try {
        const { open } = await import("@tauri-apps/plugin-dialog");
        const picked = await open({
          multiple: false,
          filters: [{ name: "Project", extensions: ["mpf"] }],
        });
        if (picked && typeof picked === "string") {
          await loadProject(picked);
          setMode("builder");
          return;
        }
      } catch {
        /* not in Tauri — fall through */
      }
      await alertDialog({
        title: "Use desktop file picker",
        body: "Project import currently requires the desktop app's native file "
          + "dialog. Use Sidebar → Load Project from the Multi-Panel Builder, "
          + "then return here and click Add to Collage.",
      });
    } catch (err) {
      console.error(err);
      await alertDialog({
        title: "Load failed",
        body: "Could not load project. Check the console for details.",
      });
    }
  };

  /** Build a corner resize-handle. Aspect-locked: dragging changes the
   *  scale factor relative to the natural aspect ratio of the source
   *  image, never the aspect itself. The opposite corner stays
   *  anchored. The "drive" axis is whichever drag delta is larger in
   *  absolute terms — if the user drags 100px right and 5px down,
   *  width drives; if they drag 5px right and 100px down, height
   *  drives. Either way both dimensions move together. */
  const renderResizeHandle = (
    it: { id: string; x: number; y: number; w: number; h: number; naturalW: number; naturalH: number },
    corner: Corner,
  ) => {
    const isWest = corner.endsWith("w");
    const isNorth = corner.startsWith("n");
    return (
      <Box
        key={corner}
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
          const startX = e.clientX;
          const startY = e.clientY;
          const startW = it.w;
          const startH = it.h;
          const startItemX = it.x;
          const startItemY = it.y;
          // Anchor = the corner opposite the one we're dragging.
          const anchorX = isWest ? startItemX + startW : startItemX;
          const anchorY = isNorth ? startItemY + startH : startItemY;
          const aspect = it.naturalW / Math.max(1, it.naturalH);
          const onMove = (ev: MouseEvent) => {
            // Convert mouse movement in screen-px back to canvas-px.
            const dxScreen = ev.clientX - startX;
            const dyScreen = ev.clientY - startY;
            const dx = dxScreen / displayScale;
            const dy = dyScreen / displayScale;
            // Sign per corner: east-side handles grow on +dx, west on -dx;
            // south on +dy, north on -dy.
            const sx = isWest ? -1 : 1;
            const sy = isNorth ? -1 : 1;
            const tentativeW = startW + sx * dx;
            const tentativeH = startH + sy * dy;
            // Whichever absolute screen-delta is larger drives the
            // size. Aspect lock: derive the other dim from it.
            const widthDriven = Math.abs(dxScreen) >= Math.abs(dyScreen);
            let newW = widthDriven ? tentativeW : tentativeH * aspect;
            let newH = widthDriven ? tentativeW / aspect : tentativeH;
            // Snap dimensions so resized items keep gridline alignment.
            newW = Math.max(20, snap(newW));
            newH = Math.max(20, snap(newH));
            // Aspect-lock: re-derive the smaller-driven dim from the
            // snapped one so the snap-rounding doesn't break aspect.
            if (widthDriven) newH = Math.max(20, newW / aspect);
            else newW = Math.max(20, newH * aspect);
            // Re-anchor: keep the opposite corner pinned. New top-left
            // depends on which corner is being dragged.
            const newX = isWest ? anchorX - newW : anchorX;
            const newY = isNorth ? anchorY - newH : anchorY;
            updateItem(it.id, {
              x: snap(newX),
              y: snap(newY),
              w: newW,
              h: newH,
            });
          };
          const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
            // Decomposed figures need NO re-render on resize — the body
            // raster just scales and the header overlays re-typeset from
            // it.w/it.h automatically. (Legacy non-decomposed figure
            // items, which bake headers into the raster, are left as-is;
            // they simply don't auto-unify until re-added.)
          };
          window.addEventListener("mousemove", onMove);
          window.addEventListener("mouseup", onUp);
        }}
        sx={{
          position: "absolute",
          width: 12,
          height: 12,
          backgroundColor: "#4FC3F7",
          border: "1px solid #fff",
          borderRadius: 0.5,
          cursor: `${corner}-resize`,
          ...(isWest ? { left: -6 } : { right: -6 }),
          ...(isNorth ? { top: -6 } : { bottom: -6 }),
        }}
      />
    );
  };

  /** On-canvas rotation handle: a small circle on a stalk above the item's
   *  top-centre. Dragging it rotates the item about its centre — the angle
   *  is computed from the item-centre→cursor vector in screen space, so it's
   *  immune to zoom/pan. Hold Shift to snap to 15° steps. The handle lives
   *  inside the (CSS-rotated) item box, so it visually tracks the rotation. */
  const renderRotationHandle = (
    it: { id: string; x: number; y: number; w: number; h: number },
  ) => {
    const STALK = 26; // px above the box top edge (item-local)
    return (
      <Box
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
          const pageRect = pageRef.current?.getBoundingClientRect();
          if (!pageRect) return;
          const cx = pageRect.left + (it.x + it.w / 2) * displayScale;
          const cy = pageRect.top + (it.y + it.h / 2) * displayScale;
          const onMove = (ev: MouseEvent) => {
            const ang = (Math.atan2(ev.clientY - cy, ev.clientX - cx) * 180) / Math.PI + 90;
            let deg = ((ang % 360) + 360) % 360;
            if (ev.shiftKey) deg = (Math.round(deg / 15) * 15) % 360;
            updateItem(it.id, { rotation: deg });
          };
          const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
          };
          window.addEventListener("mousemove", onMove);
          window.addEventListener("mouseup", onUp);
        }}
        title="Drag to rotate · hold Shift for 15° steps"
        sx={{
          position: "absolute",
          left: it.w / 2 - 7,
          top: -(STALK + 14),
          width: 14,
          height: 14,
          borderRadius: "50%",
          backgroundColor: "#4FC3F7",
          border: "2px solid #fff",
          cursor: "grab",
          zIndex: 4,
          boxShadow: "0 1px 4px rgba(0,0,0,0.45)",
        }}
      >
        {/* Stalk connecting the handle down to the box top edge. */}
        <Box
          sx={{
            position: "absolute",
            left: "50%",
            top: 14,
            width: 0,
            height: STALK,
            transform: "translateX(-50%)",
            borderLeft: "1.5px solid #4FC3F7",
            pointerEvents: "none",
          }}
        />
      </Box>
    );
  };

  // Crop overlay (item-local coords): dark mask + draggable/resizable crop
  // rect + Apply/Cancel. Mouse deltas are divided by displayScale because
  // the whole page is scaled by the parent transform.
  const renderCropOverlay = (it: { w: number; h: number }) => {
    if (!cropRect) return null;
    const cr = cropRect;
    const mask = "rgba(0,0,0,0.45)";
    const resizeHandle = (corner: Corner) => (e: React.MouseEvent) => {
      e.preventDefault(); e.stopPropagation();
      const sx = e.clientX, sy = e.clientY; const start = { ...cr };
      const isW = corner.endsWith("w"); const isN = corner.startsWith("n");
      const onMove = (ev: MouseEvent) => {
        const dx = (ev.clientX - sx) / displayScale;
        const dy = (ev.clientY - sy) / displayScale;
        let { x, y, w, h } = start; const MIN = 12;
        if (isW) { x = start.x + dx; w = start.w - dx; } else { w = start.w + dx; }
        if (isN) { y = start.y + dy; h = start.h - dy; } else { h = start.h + dy; }
        if (w < MIN) { if (isW) x = start.x + start.w - MIN; w = MIN; }
        if (h < MIN) { if (isN) y = start.y + start.h - MIN; h = MIN; }
        x = Math.max(0, x); y = Math.max(0, y);
        w = Math.min(w, it.w - x); h = Math.min(h, it.h - y);
        setCropRect({ x, y, w, h });
      };
      const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
      window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
    };
    const moveDrag = (e: React.MouseEvent) => {
      e.preventDefault(); e.stopPropagation();
      const sx = e.clientX, sy = e.clientY; const start = { ...cr };
      const onMove = (ev: MouseEvent) => {
        const dx = (ev.clientX - sx) / displayScale;
        const dy = (ev.clientY - sy) / displayScale;
        const x = Math.max(0, Math.min(start.x + dx, it.w - start.w));
        const y = Math.max(0, Math.min(start.y + dy, it.h - start.h));
        setCropRect({ x, y, w: start.w, h: start.h });
      };
      const onUp = () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
      window.addEventListener("mousemove", onMove); window.addEventListener("mouseup", onUp);
    };
    return (
      <>
        <Box sx={{ position: "absolute", left: 0, top: 0, width: "100%", height: cr.y, backgroundColor: mask, pointerEvents: "none" }} />
        <Box sx={{ position: "absolute", left: 0, top: cr.y + cr.h, width: "100%", bottom: 0, backgroundColor: mask, pointerEvents: "none" }} />
        <Box sx={{ position: "absolute", left: 0, top: cr.y, width: cr.x, height: cr.h, backgroundColor: mask, pointerEvents: "none" }} />
        <Box sx={{ position: "absolute", left: cr.x + cr.w, top: cr.y, right: 0, height: cr.h, backgroundColor: mask, pointerEvents: "none" }} />
        <Box onMouseDown={moveDrag} sx={{ position: "absolute", left: cr.x, top: cr.y, width: cr.w, height: cr.h, border: "1px solid #4FC3F7", cursor: "move", boxSizing: "border-box" }} />
        {(["nw", "ne", "sw", "se"] as const).map((c) => {
          const isW = c.endsWith("w"); const isN = c.startsWith("n");
          return (
            <Box key={c} onMouseDown={resizeHandle(c)} sx={{
              position: "absolute",
              left: isW ? cr.x - 5 : cr.x + cr.w - 5,
              top: isN ? cr.y - 5 : cr.y + cr.h - 5,
              width: 10, height: 10, backgroundColor: "#4FC3F7", border: "1px solid #fff",
              borderRadius: 0.5, cursor: `${c}-resize`, zIndex: 2,
            }} />
          );
        })}
        <Box sx={{ position: "absolute", left: cr.x, top: cr.y + cr.h + 6, display: "flex", gap: 0.5, zIndex: 3 }} onMouseDown={(e) => e.stopPropagation()}>
          <Button size="small" variant="contained" onClick={applyCrop} sx={{ fontSize: "0.6rem", py: 0, minWidth: 0, px: 0.75 }}>Apply</Button>
          <Button size="small" variant="outlined" onClick={cancelCrop} sx={{ fontSize: "0.6rem", py: 0, minWidth: 0, px: 0.75 }}>Cancel</Button>
        </Box>
      </>
    );
  };

  // Clickable element hotspots overlaid on a figure item while it's
  // expanded for per-element font sync (item-local coords). Positions come
  // from element geometry in figure fractions (y from bottom). Clicking
  // toggles the element's membership in the figure's sync selection.
  const renderElementHotspots = (it: { id: string; w: number; h: number }) => {
    const els = elemListByItem[it.id];
    if (!els) return null;
    const sel = elemSelByItem[it.id] || {};
    const bandH = Math.max(10, it.h * 0.05);
    const bandW = Math.max(10, it.w * 0.05);
    return (
      <>
        {els.filter((e) => e.geom).map((e) => {
          const g = e.geom!;
          let left: number, top: number, w: number, h: number;
          if (g.orientation === "column") {
            left = g.s0 * it.w;
            w = Math.max(8, (g.s1 - g.s0) * it.w);
            top = (1 - g.cy) * it.h - bandH / 2;
            h = bandH;
          } else {
            // Row header: vertical band; s0/s1 are y-fractions from bottom.
            left = Math.max(0, g.cx * it.w - bandW / 2);
            w = bandW;
            top = (1 - g.s1) * it.h;
            h = Math.max(8, (g.s1 - g.s0) * it.h);
          }
          const on = !!sel[e.id];
          const hov = hoveredElem?.itemId === it.id && hoveredElem?.elemId === e.id;
          return (
            <Box
              key={e.id}
              onMouseDown={(ev) => { ev.preventDefault(); ev.stopPropagation(); }}
              onMouseEnter={() => setHoveredElem({ itemId: it.id, elemId: e.id })}
              onMouseLeave={() => setHoveredElem(null)}
              onClick={(ev) => { ev.stopPropagation(); toggleElemSel(it.id, e.id); }}
              onDoubleClick={(ev) => {
                ev.preventDefault(); ev.stopPropagation();
                const ov = (elemOverridesByItem[it.id] || {})[e.id];
                const segs = (ov?.styled_segments as StyledSegment[] | undefined) ?? e.styled_segments ?? [];
                setElemEditor({ itemId: it.id, elemId: e.id, anchorEl: ev.currentTarget as HTMLElement, segments: segs, plainText: e.text || "" });
              }}
              title={`${e.type}: ${e.text}\nClick to ${on ? "deselect" : "select"} · double-click to style`}
              sx={{
                position: "absolute", left, top, width: w, height: h,
                cursor: "pointer", borderRadius: "4px", boxSizing: "border-box",
                border: hov ? "2px solid #FFD54F" : on ? "1.5px solid #4FC3F7" : "1px dashed rgba(79,195,247,0.55)",
                backgroundColor: hov ? "rgba(255,213,79,0.30)" : on ? "rgba(79,195,247,0.18)" : "rgba(79,195,247,0.05)",
                boxShadow: hov ? "0 0 0 2px rgba(255,213,79,0.5), 0 1px 6px rgba(0,0,0,0.3)" : on ? "0 0 0 1px rgba(79,195,247,0.3), 0 1px 4px rgba(0,0,0,0.25)" : "none",
                transition: "background-color 120ms, border-color 120ms, box-shadow 120ms",
                "&:hover": { backgroundColor: "rgba(255,213,79,0.30)", borderColor: "#FFD54F", borderStyle: "solid" },
              }}
            >
              {on && (
                <Box sx={{
                  position: "absolute", top: -7, right: -7,
                  width: 14, height: 14, borderRadius: "50%",
                  backgroundColor: "#4FC3F7", color: "#fff",
                  fontSize: 9, lineHeight: "14px", textAlign: "center", fontWeight: 700,
                  boxShadow: "0 1px 2px rgba(0,0,0,0.4)",
                }}>✓</Box>
              )}
            </Box>
          );
        })}
      </>
    );
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* ── Toolbar row ─────────────────────────────────────── */}
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        sx={{ px: 1.5, py: 1, borderBottom: "1px solid var(--c-border)", flexShrink: 0, flexWrap: "wrap" }}
      >
        <Tooltip title="Import an arbitrary image into the collage">
          <Button
            size="small"
            variant="outlined"
            startIcon={<AddPhotoAlternateIcon />}
            onClick={handleImportClick}
          >
            Import image
          </Button>
        </Tooltip>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        <Tooltip title="Open a saved .mpf project in the Multi-Panel Builder so you can render and add it to the collage">
          <Button
            size="small"
            variant="outlined"
            startIcon={<FolderOpenIcon />}
            onClick={() => projectInputRef.current?.click()}
          >
            Import project
          </Button>
        </Tooltip>
        <input
          ref={projectInputRef}
          type="file"
          accept=".mpf"
          style={{ display: "none" }}
          onChange={handleImportProjectPick}
        />

        <Tooltip title="Insert a text box">
          <Button size="small" variant="outlined" startIcon={<TextFieldsIcon />} onClick={insertText}>
            Text
          </Button>
        </Tooltip>
        <Tooltip title="Insert a divider line">
          <Button size="small" variant="outlined" startIcon={<HorizontalRuleIcon />} onClick={insertLine}>
            Line
          </Button>
        </Tooltip>
        <Tooltip title={cropEligible ? "Crop the selected image" : "Select a single imported image to crop"}>
          <span>
            <Button size="small" variant="outlined" startIcon={<CropIcon />} disabled={!cropEligible} onClick={startCrop}>
              Crop
            </Button>
          </span>
        </Tooltip>

        {/* Text styling controls — shown when a single text item is selected.
            Plain text boxes get quick whole-box controls; once a box carries
            per-character rich styling we show a compact summary + the editor
            button instead (so the simple controls can't silently flatten it). */}
        {selectedIds.length === 1 && selectedItem?.kind === "text" && (() => {
          const t = selectedItem;
          const isRich = !!t.styledSegments?.length;
          return (
            <>
              <Divider orientation="vertical" flexItem />
              {isRich ? (
                <>
                  <Typography variant="caption" sx={{ color: "text.secondary", fontStyle: "italic" }}>
                    Rich text
                  </Typography>
                  <Button size="small" variant="outlined"
                    onClick={() => updateItem(t.id, { styledSegments: undefined })}
                    sx={{ fontSize: "0.65rem", textTransform: "none", py: 0, minWidth: 0, px: 0.75 }}>
                    Reset style
                  </Button>
                </>
              ) : (
                <>
                  <TextField
                    type="number" size="small" title="Font size (px)"
                    value={t.fontSize ?? 28}
                    onChange={(e) => updateItem(t.id, { fontSize: Math.max(4, Math.min(400, Number(e.target.value) || 28)) })}
                    inputProps={{ min: 4, max: 400, step: 1 }}
                    sx={{ width: 64, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
                      "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": { filter: "invert(1)", opacity: 1 } }}
                  />
                  <Tooltip title="Text color">
                    <Box component="input" type="color" value={t.fontColor ?? "#000000"}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateItem(t.id, { fontColor: e.target.value })}
                      sx={{ width: 28, height: 28, p: 0, border: "none", bgcolor: "transparent", cursor: "pointer" }} />
                  </Tooltip>
                  <ToggleButton value="bold" selected={!!t.fontBold} size="small" onChange={() => updateItem(t.id, { fontBold: !t.fontBold })}
                    sx={{ p: 0.5, border: "1px solid var(--c-border)", fontWeight: 700, fontSize: "0.75rem", lineHeight: 1, minWidth: 26 }}>B</ToggleButton>
                  <ToggleButton value="italic" selected={!!t.fontItalic} size="small" onChange={() => updateItem(t.id, { fontItalic: !t.fontItalic })}
                    sx={{ p: 0.5, border: "1px solid var(--c-border)", fontStyle: "italic", fontSize: "0.75rem", lineHeight: 1, minWidth: 26 }}>i</ToggleButton>
                  <ToggleButton value="underline" selected={!!t.fontUnderline} size="small" onChange={() => updateItem(t.id, { fontUnderline: !t.fontUnderline })}
                    sx={{ p: 0.5, border: "1px solid var(--c-border)", textDecoration: "underline", fontSize: "0.75rem", lineHeight: 1, minWidth: 26 }}>U</ToggleButton>
                  <Box component="select" value={t.fontFamily ?? "arial.ttf"}
                    onChange={(e: React.ChangeEvent<HTMLSelectElement>) => updateItem(t.id, { fontFamily: e.target.value })}
                    title="Font family"
                    sx={{ fontSize: "0.7rem", height: 26, maxWidth: 150, bgcolor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 1, px: 0.5 }}>
                    {(fonts.length > 0 ? fonts : ["arial.ttf"]).map((f) => (
                      <option key={f} value={f}>{f.replace(/\.(ttf|otf|ttc|woff2?)$/i, "")}</option>
                    ))}
                  </Box>
                </>
              )}
              <Tooltip title="Bold/italic/underline, super/subscript, per-character fonts & colours">
                <Button size="small" variant="outlined"
                  onClick={(e) => openTextEditor(t, e.currentTarget)}
                  sx={{ fontSize: "0.65rem", textTransform: "none", py: 0, minWidth: 0, px: 0.75 }}>
                  Style…
                </Button>
              </Tooltip>
              <Tooltip title="Rotation (°) — or drag the round handle above the box">
                <TextField
                  type="number" size="small"
                  value={Math.round(t.rotation ?? 0)}
                  onChange={(e) => updateItem(t.id, { rotation: ((Number(e.target.value) || 0) % 360 + 360) % 360 })}
                  inputProps={{ min: 0, max: 359, step: 5 }}
                  sx={{ width: 60, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
                    "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": { filter: "invert(1)", opacity: 1 } }}
                />
              </Tooltip>
            </>
          );
        })()}

        {/* Line styling controls — shown when a single line item is selected. */}
        {selectedIds.length === 1 && selectedItem?.kind === "line" && (() => {
          const t = selectedItem;
          return (
            <>
              <Divider orientation="vertical" flexItem />
              <Tooltip title="Line color">
                <Box component="input" type="color" value={t.lineColor ?? "#000000"}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateItem(t.id, { lineColor: e.target.value })}
                  sx={{ width: 28, height: 28, p: 0, border: "none", bgcolor: "transparent", cursor: "pointer" }} />
              </Tooltip>
              <Tooltip title="Thickness (px)">
                <TextField type="number" size="small" value={t.lineThickness ?? 3}
                  onChange={(e) => updateItem(t.id, { lineThickness: Math.max(1, Math.min(100, Number(e.target.value) || 3)) })}
                  inputProps={{ min: 1, max: 100, step: 1 }}
                  sx={{ width: 60, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
                    "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": { filter: "invert(1)", opacity: 1 } }}
                />
              </Tooltip>
              <Box component="select" value={t.lineStyle ?? "solid"}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => updateItem(t.id, { lineStyle: e.target.value as "solid" | "dashed" | "dotted" })}
                title="Line style"
                sx={{ fontSize: "0.7rem", height: 26, bgcolor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 1, px: 0.5 }}>
                {["solid", "dashed", "dotted"].map((s) => (<option key={s} value={s}>{s}</option>))}
              </Box>
              <Tooltip title="Rotation (°) — or drag the round handle above the line">
                <TextField type="number" size="small" value={Math.round(t.rotation ?? 0)}
                  onChange={(e) => updateItem(t.id, { rotation: ((Number(e.target.value) || 0) % 360 + 360) % 360 })}
                  inputProps={{ min: 0, max: 359, step: 5 }}
                  sx={{ width: 60, "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
                    "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": { filter: "invert(1)", opacity: 1 } }}
                />
              </Tooltip>
            </>
          );
        })()}

        <Divider orientation="vertical" flexItem />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>Canvas</Typography>
        <TextField
          type="number"
          size="small"
          value={canvasWText}
          onChange={(e) => setCanvasWText(e.target.value)}
          onBlur={commitCanvasW}
          onKeyDown={(e) => { if (e.key === "Enter") { commitCanvasW(); (e.target as HTMLInputElement).blur(); } }}
          inputProps={{ min: 100, max: 8000, step: 50 }}
          sx={{
            width: 84,
            "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
            "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
              filter: "invert(1)", opacity: 1,
            },
          }}
        />
        <Typography variant="caption" sx={{ color: "text.secondary" }}>×</Typography>
        <TextField
          type="number"
          size="small"
          value={canvasHText}
          onChange={(e) => setCanvasHText(e.target.value)}
          onBlur={commitCanvasH}
          onKeyDown={(e) => { if (e.key === "Enter") { commitCanvasH(); (e.target as HTMLInputElement).blur(); } }}
          inputProps={{ min: 100, max: 8000, step: 50 }}
          sx={{
            width: 84,
            "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
            "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
              filter: "invert(1)", opacity: 1,
            },
          }}
        />
        <Tooltip title="Canvas size presets (journals + paper sizes, 300 DPI)">
          <Button
            size="small"
            variant="outlined"
            startIcon={<AspectRatioIcon fontSize="small" />}
            endIcon={<ArrowDropDownIcon fontSize="small" />}
            onClick={(e) => setPresetAnchor(e.currentTarget)}
            sx={{ fontSize: "0.7rem", textTransform: "none", whiteSpace: "nowrap" }}
          >
            Presets
          </Button>
        </Tooltip>
        <Menu
          anchorEl={presetAnchor}
          open={Boolean(presetAnchor)}
          onClose={() => setPresetAnchor(null)}
          MenuListProps={{ dense: true }}
          slotProps={{ paper: { sx: { maxHeight: 460 } } }}
        >
          <MenuItem onClick={() => applyCanvasPreset(DEFAULT_CANVAS_W, DEFAULT_CANVAS_H, 2)}>
            Default (Nature double column)
          </MenuItem>
          {CANVAS_PRESET_GROUPS.flatMap((g) => [
            <ListSubheader key={`h-${g.group}`} sx={{ bgcolor: "background.paper", fontSize: "0.62rem", lineHeight: 2, color: "text.secondary" }}>
              {g.group}
            </ListSubheader>,
            ...g.presets.map((p) => (
              <MenuItem key={`${g.group}-${p.label}`} onClick={() => applyCanvasPreset(p.w, p.h, p.cols)} sx={{ fontSize: "0.72rem" }}>
                {p.label}
                <Typography component="span" variant="caption" sx={{ ml: "auto", pl: 2, color: "text.secondary" }}>
                  {p.cols >= 2 ? `${p.cols}-col · ` : ""}{p.w}×{p.h}
                </Typography>
              </MenuItem>
            )),
          ])}
        </Menu>
        <Tooltip title={background === "transparent" ? "Background: transparent (click swatch to pick a color)" : "Canvas background color"}>
          <Box
            component="input"
            type="color"
            value={background === "transparent" ? "#ffffff" : background}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBackground(e.target.value)}
            sx={{
              width: 28, height: 28, p: 0, border: "none", cursor: "pointer",
              bgcolor: "transparent",
              opacity: background === "transparent" ? 0.4 : 1,
            }}
          />
        </Tooltip>
        <Tooltip title="Transparent background (exports PNG with alpha)">
          <ToggleButton
            value="transparent"
            selected={background === "transparent"}
            size="small"
            onChange={() => setBackground(background === "transparent" ? "#FFFFFF" : "transparent")}
            sx={{ p: 0.5, border: "1px solid var(--c-border)", fontSize: "0.6rem", textTransform: "none", lineHeight: 1 }}
          >
            Transp.
          </ToggleButton>
        </Tooltip>
        {guideColumns >= 2 && (
          <Tooltip title={guidesVisible ? "Hide column guides" : "Show column guides"}>
            <ToggleButton
              value="guides"
              selected={guidesVisible}
              size="small"
              onChange={() => setGuidesVisible(!guidesVisible)}
              sx={{ p: 0.5, border: "1px solid var(--c-border)", fontSize: "0.6rem", textTransform: "none", lineHeight: 1 }}
            >
              Guides
            </ToggleButton>
          </Tooltip>
        )}

        <Divider orientation="vertical" flexItem />

        <Tooltip title={gridVisible ? "Hide gridlines" : "Show gridlines"}>
          <ToggleButton
            value="grid"
            selected={gridVisible}
            size="small"
            onChange={() => setGridVisible(!gridVisible)}
            sx={{ p: 0.5, border: "1px solid var(--c-border)" }}
          >
            {gridVisible ? <GridOnIcon fontSize="small" /> : <GridOffIcon fontSize="small" />}
          </ToggleButton>
        </Tooltip>
        <Tooltip title={snapEnabled ? "Snap-to-grid is on" : "Snap-to-grid is off"}>
          <ToggleButton
            value="snap"
            selected={snapEnabled}
            size="small"
            onChange={() => setSnapEnabled(!snapEnabled)}
            sx={{ p: 0.5, border: "1px solid var(--c-border)" }}
          >
            <StraightenIcon fontSize="small" />
          </ToggleButton>
        </Tooltip>
        <TextField
          type="number"
          size="small"
          value={gridStep}
          onChange={(e) => setGridStep(Number(e.target.value) || 50)}
          inputProps={{ min: 2, max: 500, step: 5 }}
          title="Grid step (px)"
          sx={{
            width: 64,
            "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
            "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
              filter: "invert(1)", opacity: 1,
            },
          }}
        />

        <Divider orientation="vertical" flexItem />

        {/* Zoom controls — same UX as the multi-panel preview pane. */}
        <Tooltip title="Zoom out">
          <IconButton size="small" onClick={() => setUserZoom((z) => Math.max(0.1, z * 0.8))}>
            <ZoomOutIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Typography variant="caption" sx={{ minWidth: 48, textAlign: "center", color: "text.secondary" }}>
          {Math.round(displayScale * 100)}%
        </Typography>
        <Tooltip title="Zoom in">
          <IconButton size="small" onClick={() => setUserZoom((z) => Math.min(20, z * 1.25))}>
            <ZoomInIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Reset view (fit canvas to window)">
          <IconButton size="small" onClick={resetView}>
            <CenterFocusStrongIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Box sx={{ flex: 1 }} />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>
          {items.length} item{items.length === 1 ? "" : "s"}
        </Typography>
      </Stack>

      {/* ── Canvas viewport ─────────────────────────────────── */}
      <Box
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleViewportMouseDown}
        sx={{
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
          position: "relative",
          backgroundColor: "var(--c-bg)",
          backgroundImage:
            "linear-gradient(45deg, rgba(255,255,255,0.04) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.04) 75%)," +
            "linear-gradient(45deg, rgba(255,255,255,0.04) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.04) 75%)",
          backgroundSize: "20px 20px",
          backgroundPosition: "0 0, 10px 10px",
        }}
      >
        {/* Pan/zoom wrapper. Positioned absolute at (pan.x, pan.y) and
            scaled to displayScale. Items live in canvas-pixel space
            inside, so all drag/snap math stays simple. */}
        <Box
          sx={{
            position: "absolute",
            left: pan.x,
            top: pan.y,
            transform: `scale(${displayScale})`,
            transformOrigin: "top left",
          }}
        >
          {/* The "page" — solid color or a checkerboard for transparent. */}
          <Box
            ref={pageRef}
            sx={{
              position: "relative",
              width: canvasW,
              height: canvasH,
              boxShadow: "0 0 0 1px rgba(255,255,255,0.15), 0 8px 24px rgba(0,0,0,0.4)",
              ...(background === "transparent"
                ? {
                    backgroundColor: "#ffffff",
                    backgroundImage:
                      `linear-gradient(45deg, #cfcfcf 25%, transparent 25%),` +
                      `linear-gradient(-45deg, #cfcfcf 25%, transparent 25%),` +
                      `linear-gradient(45deg, transparent 75%, #cfcfcf 75%),` +
                      `linear-gradient(-45deg, transparent 75%, #cfcfcf 75%)`,
                    backgroundSize: "24px 24px",
                    backgroundPosition: "0 0, 0 12px, 12px -12px, -12px 0",
                  }
                : { backgroundColor: background }),
            }}
            onMouseDown={(e) => {
              // Only start a marquee when the click lands on the empty page
              // (not on an item, which stops propagation).
              if (e.target !== e.currentTarget) return;
              const additive = e.shiftKey || e.metaKey || e.ctrlKey;
              if (!additive) setSelectedId(null);
              const pageRect = e.currentTarget.getBoundingClientRect();
              const toCanvas = (cx: number, cy: number) => ({
                x: (cx - pageRect.left) / displayScale,
                y: (cy - pageRect.top) / displayScale,
              });
              const start = toCanvas(e.clientX, e.clientY);
              const baseSel = additive ? [...useCollageStore.getState().selectedIds] : [];
              let moved = false;
              const onMove = (ev: MouseEvent) => {
                const p = toCanvas(ev.clientX, ev.clientY);
                moved = true;
                const rect = {
                  x0: Math.min(start.x, p.x), y0: Math.min(start.y, p.y),
                  x1: Math.max(start.x, p.x), y1: Math.max(start.y, p.y),
                };
                setMarquee(rect);
                const hits = useCollageStore.getState().items
                  .filter((it) => it.x < rect.x1 && it.x + it.w > rect.x0 && it.y < rect.y1 && it.y + it.h > rect.y0)
                  .map((it) => it.id);
                setSelectedIds(Array.from(new Set([...baseSel, ...hits])));
              };
              const onUp = () => {
                window.removeEventListener("mousemove", onMove);
                window.removeEventListener("mouseup", onUp);
                setMarquee(null);
                if (!moved && !additive) setSelectedId(null);
              };
              window.addEventListener("mousemove", onMove);
              window.addEventListener("mouseup", onUp);
            }}
          >
            {/* Gridline overlay (kept separate from the page background so it
                composes cleanly over a transparent checkerboard). */}
            {gridVisible && (
              <Box sx={{
                position: "absolute", inset: 0, pointerEvents: "none",
                backgroundImage:
                  `linear-gradient(to right, rgba(0,0,0,0.10) 1px, transparent 1px),` +
                  `linear-gradient(to bottom, rgba(0,0,0,0.10) 1px, transparent 1px)`,
                backgroundSize: `${gridStep}px ${gridStep}px, ${gridStep}px ${gridStep}px`,
              }} />
            )}
            {/* Journal column guides — vertical lines at each column edge with
                shaded gutters, so figures can be aligned to journal columns. */}
            {guidesVisible && guideColumns >= 2 && (() => {
              const colW = (canvasW - guideGutter * (guideColumns - 1)) / guideColumns;
              const nodes: React.ReactNode[] = [];
              for (let i = 0; i < guideColumns; i++) {
                const left = i * (colW + guideGutter);
                // Shaded gutter to the right of every column except the last.
                if (i < guideColumns - 1 && guideGutter > 0) {
                  nodes.push(
                    <Box key={`gut-${i}`} sx={{
                      position: "absolute", top: 0, bottom: 0,
                      left: left + colW, width: guideGutter,
                      backgroundColor: "rgba(79,195,247,0.10)", pointerEvents: "none",
                    }} />,
                  );
                }
                // Column edges (left + right).
                for (const x of [left, left + colW]) {
                  nodes.push(
                    <Box key={`g-${i}-${x}`} sx={{
                      position: "absolute", top: 0, bottom: 0, left: x, width: 0,
                      borderLeft: "1px dashed rgba(79,195,247,0.7)", pointerEvents: "none",
                    }} />,
                  );
                }
              }
              return <>{nodes}</>;
            })()}
            {sortedItems.map((it) => {
              const isSelected = selectedIds.includes(it.id);
              return (
                <Box
                  key={it.id}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    // Shift/Cmd-click toggles membership without dragging.
                    if (e.shiftKey || e.metaKey || e.ctrlKey) {
                      toggleSelected(it.id);
                      return;
                    }
                    // Clicking an item outside the current selection selects
                    // just it; clicking one already selected keeps the group.
                    const curSel = useCollageStore.getState().selectedIds;
                    const dragIds = curSel.includes(it.id) ? curSel : [it.id];
                    if (!curSel.includes(it.id)) setSelectedId(it.id);
                    bringToFront(it.id);
                    const startMouseX = e.clientX;
                    const startMouseY = e.clientY;
                    // Snapshot each dragged item's start position so the whole
                    // group moves by the same (snapped) delta.
                    const starts = new Map<string, { x: number; y: number }>();
                    for (const id of dragIds) {
                      const cur = useCollageStore.getState().items.find((i) => i.id === id);
                      if (cur) starts.set(id, { x: cur.x, y: cur.y });
                    }
                    const primaryStart = starts.get(it.id) ?? { x: it.x, y: it.y };
                    const onMove = (ev: MouseEvent) => {
                      const dx = (ev.clientX - startMouseX) / displayScale;
                      const dy = (ev.clientY - startMouseY) / displayScale;
                      const targetX = snap(primaryStart.x + dx);
                      const targetY = snap(primaryStart.y + dy);
                      const appliedDx = targetX - primaryStart.x;
                      const appliedDy = targetY - primaryStart.y;
                      for (const [id, st] of starts) {
                        const cur = useCollageStore.getState().items.find((i) => i.id === id);
                        if (!cur) continue;
                        moveItem(id, (st.x + appliedDx) - cur.x, (st.y + appliedDy) - cur.y);
                      }
                    };
                    const onUp = () => {
                      window.removeEventListener("mousemove", onMove);
                      window.removeEventListener("mouseup", onUp);
                    };
                    window.addEventListener("mousemove", onMove);
                    window.addEventListener("mouseup", onUp);
                  }}
                  onDoubleClick={(e) => {
                    if (it.kind !== "text") return;
                    // Rich-styled boxes open the full editor (so inline editing
                    // can't flatten their per-character styling); plain boxes
                    // get the quick inline textarea for fast content edits.
                    if (it.styledSegments?.length) openTextEditor(it, e.currentTarget as HTMLElement);
                    else setEditingTextId(it.id);
                  }}
                  sx={{
                    position: "absolute",
                    left: it.x,
                    top: it.y,
                    width: it.w,
                    height: it.h,
                    cursor: "grab",
                    ...(it.rotation ? { transform: `rotate(${it.rotation}deg)`, transformOrigin: "center center" } : {}),
                    outline: isSelected ? "2px solid #4FC3F7" : "1px solid rgba(255,255,255,0.0)",
                    outlineOffset: isSelected ? 2 : 0,
                    "&:hover": { outline: "2px solid rgba(79,195,247,0.6)" },
                  }}
                  title={it.name}
                >
                  {it.kind === "text" ? (
                    editingTextId === it.id ? (
                      <textarea
                        autoFocus
                        defaultValue={it.text ?? ""}
                        onMouseDown={(e) => e.stopPropagation()}
                        onBlur={(e) => { updateItem(it.id, { text: e.target.value }); setEditingTextId(null); }}
                        style={{
                          width: "100%", height: "100%", boxSizing: "border-box",
                          border: "none", outline: "none", resize: "none", background: "transparent",
                          padding: 0, lineHeight: 1.2, overflow: "hidden",
                          fontSize: it.fontSize ?? 28, color: it.fontColor ?? "#000000",
                          fontFamily: cssFamily(it.fontFamily),
                          fontWeight: it.fontBold ? "bold" : "normal",
                          fontStyle: it.fontItalic ? "italic" : "normal",
                          textDecoration: it.fontUnderline ? "underline" : "none",
                          textAlign: it.align ?? "left",
                        }}
                      />
                    ) : it.styledSegments?.length ? (
                      <div style={{
                        width: "100%", height: "100%", overflow: "hidden",
                        whiteSpace: "pre-wrap", wordBreak: "break-word", lineHeight: 1.2,
                        pointerEvents: "none", userSelect: "none",
                        textAlign: it.align ?? "left",
                      }}>
                        {it.styledSegments.map((seg, i) => (
                          <span key={`${i}-${seg.text.slice(0, 6)}`} style={segmentStyle(seg, it.fontSize ?? 28, it.fontColor ?? "#000000")}>
                            {seg.text}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <div style={{
                        width: "100%", height: "100%", overflow: "hidden",
                        whiteSpace: "pre-wrap", wordBreak: "break-word", lineHeight: 1.2,
                        pointerEvents: "none", userSelect: "none",
                        fontSize: it.fontSize ?? 28, color: it.fontColor ?? "#000000",
                        fontFamily: cssFamily(it.fontFamily),
                        fontWeight: it.fontBold ? "bold" : "normal",
                        fontStyle: it.fontItalic ? "italic" : "normal",
                        textDecoration: it.fontUnderline ? "underline" : "none",
                        textAlign: it.align ?? "left",
                      }}>
                        {it.text || "Text"}
                      </div>
                    )
                  ) : it.kind === "line" ? (
                    <Box sx={{
                      position: "absolute", left: 0, right: 0, top: "50%",
                      transform: "translateY(-50%)", pointerEvents: "none",
                      borderTop: `${it.lineThickness ?? 3}px ${it.lineStyle ?? "solid"} ${it.lineColor ?? "#000000"}`,
                    }} />
                  ) : (
                    <img
                      src={it.src}
                      alt={it.name}
                      draggable={false}
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "fill",
                        pointerEvents: "none",
                        userSelect: "none",
                      }}
                    />
                  )}
                  {/* Resize handles only when exactly one item is selected and
                      not mid-crop (group resize isn't supported — only move). */}
                  {isSelected && selectedIds.length === 1 && cropItemId !== it.id && (
                    <>
                      {(["nw", "ne", "sw", "se"] as const).map((c) => renderResizeHandle(it, c))}
                    </>
                  )}
                  {/* On-canvas rotation handle for text + line items. */}
                  {isSelected && selectedIds.length === 1 && cropItemId !== it.id
                    && (it.kind === "text" || it.kind === "line")
                    && renderRotationHandle(it)}
                  {/* Crop overlay for image items in crop mode. */}
                  {cropItemId === it.id && cropRect && renderCropOverlay(it)}
                  {/* Per-element font-sync hotspots while this figure is
                      expanded in the sidebar. */}
                  {elemSyncItemId === it.id && it.kind === "figure" && renderElementHotspots(it)}
                </Box>
              );
            })}
            {/* Marquee (rubber-band) selection rectangle. */}
            {marquee && (
              <Box sx={{
                position: "absolute",
                left: marquee.x0, top: marquee.y0,
                width: Math.max(0, marquee.x1 - marquee.x0),
                height: Math.max(0, marquee.y1 - marquee.y0),
                border: "1px dashed #4FC3F7",
                backgroundColor: "rgba(79,195,247,0.12)",
                pointerEvents: "none",
              }} />
            )}
          </Box>
        </Box>
      </Box>

      <CollageStrip />

      {/* Whole text-box rich editor (double-click a styled box, or the
          "Style…" toolbar button). Saves per-character segments + plain text. */}
      {textEditor && (
        <RichTextEditor
          open
          anchorEl={textEditor.anchorEl}
          segments={textEditor.segments}
          plainText={textEditor.plainText}
          fonts={fonts}
          onClose={() => setTextEditor(null)}
          onSave={(segments, plainText) => {
            const ed = textEditor;
            updateItem(ed.itemId, { styledSegments: segments, text: plainText });
            setTextEditor(null);
          }}
        />
      )}

      {/* Per-element rich-text customization (double-click a hotspot). On
          save we store the style override and re-render that figure. */}
      {elemEditor && (
        <RichTextEditor
          open
          anchorEl={elemEditor.anchorEl}
          segments={elemEditor.segments}
          plainText={elemEditor.plainText}
          fonts={fonts}
          onClose={() => setElemEditor(null)}
          onSave={(segments) => {
            const ed = elemEditor;
            setElemOverride(ed.itemId, ed.elemId, { styled_segments: segments });
            setElemEditor(null);
            void rerenderFigure(ed.itemId);
          }}
        />
      )}
    </Box>
  );
}
