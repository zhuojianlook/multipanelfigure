/* ──────────────────────────────────────────────────────────
   PanelGrid — CSS Grid layout for the panel builder.
   Redesigned header system:
   - Empty header lanes (click to add headers)
   - Click-to-create individual headers
   - Drag-to-resize header span
   - X buttons at end of lanes (right for col, bottom for row)
   - Floating formatting toolbar on click
   - Right-click context menus for header manipulation
   ────────────────────────────────────────────────────────── */

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Checkbox,
  FormControlLabel,
  Slider,
  TextField,
  Typography,
  Box,
  Select,
  FormControl,
  InputLabel,
  Tooltip,
  Switch,
  IconButton,
} from "@mui/material";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import DeleteIcon from "@mui/icons-material/Delete";
import CallSplitIcon from "@mui/icons-material/CallSplit";
import AddIcon from "@mui/icons-material/Add";
import TuneIcon from "@mui/icons-material/Tune";
import ClearIcon from "@mui/icons-material/Clear";
import ViewColumnIcon from "@mui/icons-material/ViewColumn";
import CloseIcon from "@mui/icons-material/Close";
import { useFigureStore, type LoadedImage } from "../../store/figureStore";
import { PanelCell } from "./PanelCell";
import { FloatingToolbar } from "./FloatingToolbar";
import { HeaderEditor, type HeaderEditorTarget } from "./HeaderEditor";
import { type StyledTextEditorHandle } from "./StyledTextEditor";
import type { HeaderGroup, PanelInfo } from "../../api/types";
import { api } from "../../api/client";

/* ── Types ──────────────────────────────────────────────── */

interface ContextMenuState {
  mouseX: number;
  mouseY: number;
  axis: "col" | "row";
  level: number;
  groupIdx: number;
  type: "header" | "empty";
  cellIndex?: number;
}

interface SpanDialogState {
  axis: "col" | "row";
  level: number;
  groupIdx: number;
  selectedIndices: number[];
}

interface HeaderStyledSegment {
  text: string;
  color: string;
  font_name?: string;
  font_size?: number;
  font_style?: string[];
}

interface HeaderPropsDialogState {
  axis: "col" | "row";
  level: number;
  groupIdx: number;
  distance: number;
  lineColor: string;
  lineWidth: number;
  lineStyle: string;
  lineLength: number;
  position: string;
  showLine: boolean;
  endCaps: boolean;
  defaultColor: string;
  styledSegments: HeaderStyledSegment[];
  headerText: string;  // full header text for reference when building segments
}

interface LabelPropsDialogState {
  axis: "col" | "row";
  index: number;
  distance: number;      // displayed as percentage (0–10) of figure
  position: string;
  rotation: number;
  applyToAll: boolean;
}

interface DragState {
  axis: "col" | "row";
  level: number;
  groupIdx: number;
  edge: "start" | "end";
  initialIndices: number[];
  startClientPos: number; // clientX or clientY at drag start
}

interface ToolbarTarget {
  type: "header" | "colLabel" | "rowLabel";
  axis: "col" | "row";
  level?: number;
  groupIdx?: number;
  index?: number;
}

/* ── Drawer Strip Component ────────────────────────────── */

interface DrawerStripProps {
  cols: number;
  drawerPanels: PanelInfo[];
  loadedImages: Record<string, LoadedImage>;
  movePanelToDrawer: (r: number, c: number, drawerIdx: number) => void;
  movePanelFromDrawer: (drawerIdx: number, r: number, c: number) => void;
}

function DrawerStrip({ cols, drawerPanels, loadedImages }: DrawerStripProps) {
  const [dragOverIdx, setDragOverIdx] = useState<number | null>(null);
  const config = useFigureStore((s) => s.config);
  const drawerThumbnails = useFigureStore((s) => s.drawerThumbnails);
  const totalPanels = config ? config.rows * config.cols : cols;

  // Show at least totalPanels slots, plus any extra drawer entries, plus one empty for new drops
  const filledCount = drawerPanels.filter((p) => p && p.image_name).length;
  const slots = Math.max(totalPanels, filledCount + 1);

  return (
    <Box sx={{ mt: 2, p: 1, border: "1px dashed", borderColor: "divider", borderRadius: 1, backgroundColor: "var(--c-surface)" }}>
      <Typography variant="caption" sx={{ color: "var(--c-text-dim)", display: "block", mb: 0.5, fontSize: "0.6rem", textTransform: "uppercase", letterSpacing: 1 }}>
        Parking Drawer
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
        {Array.from({ length: slots }, (_, i) => {
          const panel = drawerPanels[i];
          const hasImage = panel && panel.image_name;
          // Prefer the processed drawer thumbnail; fall back to the raw upload thumbnail
          const drawerThumb = drawerThumbnails[i];
          const rawThumb = hasImage ? loadedImages[panel.image_name]?.thumbnailB64 : undefined;
          const thumbSrc = drawerThumb || rawThumb;
          return (
            <Box
              key={i}
              sx={{
                width: 80,
                height: 80,
                minWidth: 80,
                border: dragOverIdx === i ? "2px solid #2196f3" : "1px dashed",
                borderColor: dragOverIdx === i ? "#2196f3" : "divider",
                borderRadius: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: hasImage ? "grab" : "default",
                backgroundColor: hasImage ? "var(--c-surface2)" : "transparent",
                transition: "border-color 0.15s",
                position: "relative",
              }}
              draggable={!!hasImage}
              onDragStart={(e: React.DragEvent) => {
                if (!hasImage) { e.preventDefault(); return; }
                e.dataTransfer.setData("application/x-drawer-index", String(i));
                e.dataTransfer.effectAllowed = "move";
              }}
              onDragOver={(e: React.DragEvent) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                setDragOverIdx(i);
              }}
              onDragLeave={() => setDragOverIdx(null)}
              onDrop={(e: React.DragEvent) => {
                e.preventDefault();
                setDragOverIdx(null);
                const panelSrc = e.dataTransfer.getData("application/x-panel-source");
                if (panelSrc) {
                  const src = JSON.parse(panelSrc) as { row: number; col: number };
                  const store = useFigureStore.getState();
                  store.movePanelToDrawer(src.row, src.col, i);
                }
              }}
            >
              <Tooltip title={hasImage ? panel.image_name : ""} placement="top" arrow enterDelay={300}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", width: "100%", height: "100%", pointerEvents: "auto" }}>
                  {thumbSrc ? (
                    <Box
                      component="img"
                      src={`data:image/png;base64,${thumbSrc}`}
                      alt={hasImage ? panel.image_name : ""}
                      sx={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: 0.5 }}
                      draggable={false}
                    />
                  ) : (
                    <Typography sx={{ fontSize: "0.55rem", color: "var(--c-text-dim)" }}>
                      Drop here
                    </Typography>
                  )}
                </Box>
              </Tooltip>
              {hasImage && (
                <Typography sx={{
                  position: "absolute", bottom: 1, left: 2, right: 2,
                  fontSize: "0.45rem", color: "var(--c-text-dim)",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  textAlign: "center",
                }}>
                  {panel.image_name.split("/").pop()?.substring(0, 15)}
                </Typography>
              )}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

/* ── Component ──────────────────────────────────────────── */

export function PanelGrid() {
  const config = useFigureStore((s) => s.config);
  const drawerPanels = useFigureStore((s) => s.drawerPanels);
  const movePanelToDrawer = useFigureStore((s) => s.movePanelToDrawer);
  const movePanelFromDrawer = useFigureStore((s) => s.movePanelFromDrawer);
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const updateColumnLabel = useFigureStore((s) => s.updateColumnLabel);
  const updateRowLabel = useFigureStore((s) => s.updateRowLabel);
  const updateHeaderGroupText = useFigureStore((s) => s.updateHeaderGroupText);
  const addColumnHeaderLevel = useFigureStore((s) => s.addColumnHeaderLevel);
  const addRowHeaderLevel = useFigureStore((s) => s.addRowHeaderLevel);
  const removeColumnHeaderLevel = useFigureStore((s) => s.removeColumnHeaderLevel);
  const removeRowHeaderLevel = useFigureStore((s) => s.removeRowHeaderLevel);
  const extendHeaderGroup = useFigureStore((s) => s.extendHeaderGroup);
  const removeHeaderGroup = useFigureStore((s) => s.removeHeaderGroup);
  const splitHeaderGroup = useFigureStore((s) => s.splitHeaderGroup);
  const createHeaderGroupAt = useFigureStore((s) => s.createHeaderGroupAt);
  const resizeHeaderGroup = useFigureStore((s) => s.resizeHeaderGroup);
  const updateHeaderGroupFormatting = useFigureStore((s) => s.updateHeaderGroupFormatting);
  const updateLabelFormatting = useFigureStore((s) => s.updateLabelFormatting);
  const swapColumnHeaderLevels = useFigureStore((s) => s.swapColumnHeaderLevels);
  const swapRowHeaderLevels = useFigureStore((s) => s.swapRowHeaderLevels);
  const setConfig = useFigureStore((s) => s.setConfig);
  const fonts = useFigureStore((s) => s.fonts);
  const requestPreview = useFigureStore((s) => s.requestPreview);

  // Context menus
  const [ctxMenu, setCtxMenu] = useState<ContextMenuState | null>(null);

  // Dialogs
  const [spanDialog, setSpanDialog] = useState<SpanDialogState | null>(null);
  const [headerPropsDialog, setHeaderPropsDialog] = useState<HeaderPropsDialogState | null>(null);
  const [labelPropsDialog, setLabelPropsDialog] = useState<LabelPropsDialogState | null>(null);

  // Floating toolbar
  const [toolbarAnchor, setToolbarAnchor] = useState<HTMLElement | null>(null);
  const [toolbarTarget, setToolbarTarget] = useState<ToolbarTarget | null>(null);

  // Drag-to-resize
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [dragPreviewIndices, setDragPreviewIndices] = useState<number[] | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  /* ── Context menu handlers ──────────────────────────── */

  const handleHeaderContextMenu = useCallback(
    (e: React.MouseEvent, axis: "col" | "row", level: number, groupIdx: number) => {
      e.preventDefault();
      setCtxMenu({ mouseX: e.clientX, mouseY: e.clientY, axis, level, groupIdx, type: "header" });
    },
    [],
  );

  const handleEmptyCellContextMenu = useCallback(
    (e: React.MouseEvent, axis: "col" | "row", level: number, cellIndex: number) => {
      e.preventDefault();
      setCtxMenu({ mouseX: e.clientX, mouseY: e.clientY, axis, level, groupIdx: -1, type: "empty", cellIndex });
    },
    [],
  );

  const closeCtxMenu = () => setCtxMenu(null);

  /* ── Click-to-create on empty lane cells ────────────── */

  const handleEmptyCellClick = useCallback(
    (axis: "col" | "row", level: number, cellIndex: number) => {
      createHeaderGroupAt(axis, level, cellIndex);
    },
    [createHeaderGroupAt],
  );

  /* ── Floating toolbar handlers ──────────────────────── */

  const handleHeaderClick = useCallback(
    (e: React.MouseEvent, axis: "col" | "row", level: number, groupIdx: number) => {
      // Only open toolbar on single left click (not right-click)
      if (e.button !== 0) return;
      setToolbarAnchor(e.currentTarget as HTMLElement);
      setToolbarTarget({ type: "header", axis, level, groupIdx });
    },
    [],
  );

  const handleLabelClick = useCallback(
    (e: React.MouseEvent, axis: "col" | "row", index: number) => {
      if (e.button !== 0) return;
      setToolbarAnchor(e.currentTarget as HTMLElement);
      setToolbarTarget({
        type: axis === "col" ? "colLabel" : "rowLabel",
        axis,
        index,
      });
    },
    [],
  );

  const closeToolbar = () => {
    setToolbarAnchor(null);
    setToolbarTarget(null);
    setSelectionPreview("");
  };

  // Close the floating toolbar when the user clicks anywhere that isn't
  // the toolbar itself or the textarea/input it's anchored to. Required
  // because hideBackdrop+pointerEvents:none on the Popover (which was
  // necessary to keep textarea drag-selection working) also disabled
  // MUI's native click-outside-to-close behaviour.
  useEffect(() => {
    if (!toolbarAnchor) return;
    const onDocMouseDown = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (!t) return;
      // Inside the toolbar popover (paper or any descendant) — don't close.
      if (t.closest('.MuiPopover-paper')) return;
      // Inside the anchor element (the textarea / input's wrapper) — don't close.
      if (toolbarAnchor.contains(t)) return;
      // Inside ANY header / label textarea / input (user is switching
      // between adjacent editable fields — new onFocus will re-anchor).
      if (t.tagName === "TEXTAREA" || t.tagName === "INPUT") {
        const a = t.getAttribute("aria-label") || "";
        if (/\b(header|label)\b/i.test(a)) return;
      }
      closeToolbar();
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [toolbarAnchor]); // eslint-disable-line react-hooks/exhaustive-deps

  /* ── Drag-to-resize handlers ────────────────────────── */

  const handleDragStart = useCallback(
    (
      e: React.MouseEvent,
      axis: "col" | "row",
      level: number,
      groupIdx: number,
      edge: "start" | "end",
      currentIndices: number[],
    ) => {
      e.preventDefault();
      e.stopPropagation();
      const clientPos = axis === "col" ? e.clientX : e.clientY;
      setDragState({
        axis,
        level,
        groupIdx,
        edge,
        initialIndices: [...currentIndices].sort((a, b) => a - b),
        startClientPos: clientPos,
      });
      setDragPreviewIndices([...currentIndices].sort((a, b) => a - b));
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragState || !config || !gridRef.current) return;

      const { axis, level, groupIdx, edge, initialIndices } = dragState;
      const cellSize = 160;
      const gap = 3;
      const cellStep = cellSize + gap;

      const clientPos = axis === "col" ? e.clientX : e.clientY;
      const delta = clientPos - dragState.startClientPos;
      const cellDelta = Math.round(delta / cellStep);

      const minIdx = initialIndices[0];
      const maxIdx = initialIndices[initialIndices.length - 1];
      const maxBound = axis === "col" ? config.cols - 1 : config.rows - 1;

      // Compute new indices based on which edge is being dragged
      let newMin = minIdx;
      let newMax = maxIdx;
      if (edge === "start") {
        newMin = Math.max(0, Math.min(maxIdx, minIdx + cellDelta));
      } else {
        newMax = Math.min(maxBound, Math.max(minIdx, maxIdx + cellDelta));
      }

      // Build new indices array
      const newIndices: number[] = [];
      for (let i = newMin; i <= newMax; i++) {
        newIndices.push(i);
      }

      // Check for overlaps with other groups
      const headers = axis === "col" ? config.column_headers : config.row_headers;
      const usedByOthers = new Set<number>();
      headers[level].headers.forEach((g, gi) => {
        if (gi !== groupIdx) g.columns_or_rows.forEach((idx) => usedByOthers.add(idx));
      });

      // Trim to avoid overlap
      const validIndices = newIndices.filter((i) => !usedByOthers.has(i));
      // Must stay contiguous - find the contiguous range containing the original group
      if (validIndices.length > 0) {
        const sortedValid = validIndices.sort((a, b) => a - b);
        // Find the contiguous block that overlaps with original indices
        let bestStart = 0;
        let bestEnd = 0;
        let currentStart = 0;
        for (let i = 1; i <= sortedValid.length; i++) {
          if (i === sortedValid.length || sortedValid[i] !== sortedValid[i - 1] + 1) {
            // End of contiguous block
            const blockIndices = sortedValid.slice(currentStart, i);
            // Does this block overlap with any original index?
            if (blockIndices.some((idx) => initialIndices.includes(idx))) {
              if (i - currentStart > bestEnd - bestStart) {
                bestStart = currentStart;
                bestEnd = i;
              }
            }
            currentStart = i;
          }
        }
        setDragPreviewIndices(sortedValid.slice(bestStart, bestEnd));
      }
    },
    [dragState, config],
  );

  const handleMouseUp = useCallback(() => {
    if (dragState && dragPreviewIndices && dragPreviewIndices.length > 0) {
      resizeHeaderGroup(dragState.axis, dragState.level, dragState.groupIdx, dragPreviewIndices);
    }
    setDragState(null);
    setDragPreviewIndices(null);
  }, [dragState, dragPreviewIndices, resizeHeaderGroup]);

  /* ── Early return ──────────────────────────────────── */

  if (!config) return null;

  const {
    rows,
    cols,
    column_headers,
    column_labels,
    row_labels,
    row_headers,
  } = config;

  // Insert a "\n" into a header's text AND its styled_segments, preserving
  // per-character styling across the break. Used for both the toolbar's
  // Insert-Line-Break button AND intercepted Shift+Enter keypresses — going
  // through updateHeaderGroupText alone would clear styled_segments because
  // the segment concat no longer equals the new text.
  const insertLineBreakInHeader = (
    axis: "col" | "row",
    level: number,
    groupIdx: number,
    caretStart: number,
    caretEnd?: number,
  ) => {
    const headers = axis === "col" ? column_headers : row_headers;
    const grp = headers[level]?.headers[groupIdx];
    if (!grp) return;
    const currentText = grp.text || "";
    const currentSegs = (grp.styled_segments as HeaderStyledSegment[]) || [];
    const defaultColor = grp.default_color || "#000000";

    const rawStart = caretStart;
    const rawEnd = caretEnd ?? caretStart;
    const start = Math.max(0, Math.min(rawStart, currentText.length));
    const end = Math.max(start, Math.min(rawEnd, currentText.length));
    const newText = currentText.slice(0, start) + "\n" + currentText.slice(end);

    let newSegs: HeaderStyledSegment[] = [];
    if (currentSegs.length) {
      const out: HeaderStyledSegment[] = [];
      let charCount = 0;
      let inserted = false;
      for (const seg of currentSegs) {
        const segEnd = charCount + seg.text.length;
        if (!inserted && start >= charCount && start <= segEnd) {
          const beforeLocal = seg.text.slice(0, start - charCount);
          const afterLocal = seg.text.slice(end - charCount);
          if (beforeLocal.length > 0) out.push({ ...seg, text: beforeLocal });
          out.push({ ...seg, text: "\n" });
          if (afterLocal.length > 0) out.push({ ...seg, text: afterLocal });
          inserted = true;
        } else {
          out.push(seg);
        }
        charCount = segEnd;
      }
      if (!inserted) out.push({ text: "\n", color: defaultColor });
      newSegs = out;
    }

    updateHeaderGroupText(axis, level, groupIdx, newText);
    if (newSegs.length > 0) {
      updateHeaderGroupFormatting(axis, level, groupIdx, { styled_segments: newSegs });
    }
    return start + 1;
  };

  // Insert "\n" at the cursor of a primary col/row LABEL textarea, preserving
  // per-char styled_segments — same pattern as insertLineBreakInHeader, but
  // for the primary (next-to-panel) labels.
  const insertLineBreakInLabel = (
    axis: "col" | "row",
    index: number,
    caretStart: number,
    caretEnd?: number,
  ) => {
    const labels = axis === "col" ? column_labels : row_labels;
    const lbl = labels[index];
    if (!lbl) return;
    const currentText = lbl.text || "";
    const currentSegs = (lbl.styled_segments as HeaderStyledSegment[]) || [];
    const defaultColor = lbl.default_color || "#000000";

    const rawStart = caretStart;
    const rawEnd = caretEnd ?? caretStart;
    const start = Math.max(0, Math.min(rawStart, currentText.length));
    const end = Math.max(start, Math.min(rawEnd, currentText.length));
    const newText = currentText.slice(0, start) + "\n" + currentText.slice(end);

    let newSegs: HeaderStyledSegment[] = [];
    if (currentSegs.length) {
      const out: HeaderStyledSegment[] = [];
      let charCount = 0;
      let inserted = false;
      for (const seg of currentSegs) {
        const segEnd = charCount + seg.text.length;
        if (!inserted && start >= charCount && start <= segEnd) {
          const beforeLocal = seg.text.slice(0, start - charCount);
          const afterLocal = seg.text.slice(end - charCount);
          if (beforeLocal.length > 0) out.push({ ...seg, text: beforeLocal });
          out.push({ ...seg, text: "\n" });
          if (afterLocal.length > 0) out.push({ ...seg, text: afterLocal });
          inserted = true;
        } else {
          out.push(seg);
        }
        charCount = segEnd;
      }
      if (!inserted) out.push({ text: "\n", color: defaultColor });
      newSegs = out;
    }

    if (axis === "col") {
      updateColumnLabel(index, newText);
    } else {
      updateRowLabel(index, newText);
    }
    if (newSegs.length > 0) {
      updateLabelFormatting(axis, index, { styled_segments: newSegs });
    }
    return start + 1;
  };

  /* ── Span dialog helpers ───────────────────────────── */

  const openSpanDialog = (axis: "col" | "row", level: number, groupIdx: number) => {
    const headers = axis === "col" ? column_headers : row_headers;
    const group = headers[level]?.headers[groupIdx];
    if (!group) return;
    setSpanDialog({
      axis,
      level,
      groupIdx,
      selectedIndices: [...group.columns_or_rows],
    });
  };

  const getAvailableIndicesForSpan = (): number[] => {
    if (!spanDialog) return [];
    const { axis, level, groupIdx } = spanDialog;
    const headers = axis === "col" ? column_headers : row_headers;
    const maxCount = axis === "col" ? cols : rows;
    const usedByOthers = new Set<number>();
    headers[level].headers.forEach((g, gi) => {
      if (gi !== groupIdx) g.columns_or_rows.forEach((i) => usedByOthers.add(i));
    });
    return Array.from({ length: maxCount }, (_, i) => i).filter((i) => !usedByOthers.has(i));
  };

  const applySpanDialog = () => {
    if (!spanDialog || !config) return;
    const { axis, level, groupIdx, selectedIndices } = spanDialog;
    if (selectedIndices.length === 0) {
      setSpanDialog(null);
      return;
    }
    // Deep clone to avoid mutating Zustand state directly
    const headers = JSON.parse(JSON.stringify(axis === "col" ? config.column_headers : config.row_headers));
    headers[level].headers[groupIdx].columns_or_rows = selectedIndices.sort((a, b) => a - b);
    const setConfig = useFigureStore.getState().setConfig;
    if (axis === "col") {
      setConfig({ ...config, column_headers: headers });
    } else {
      setConfig({ ...config, row_headers: headers });
    }
    const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
    patchFn.call(api, headers).catch(console.error);
    requestPreview();
    setSpanDialog(null);
  };

  /* ── Header properties dialog helpers ──────────────── */

  const openHeaderPropsDialog = (axis: "col" | "row", level: number, groupIdx: number) => {
    const headers = axis === "col" ? column_headers : row_headers;
    const group = headers[level]?.headers[groupIdx];
    if (!group) return;
    setHeaderPropsDialog({
      axis,
      level,
      groupIdx,
      distance: group.distance * 1000,
      lineColor: group.line_color,
      lineWidth: group.line_width,
      lineStyle: group.line_style ?? "solid",
      lineLength: group.line_length ?? 1.0,
      position: group.position,
      showLine: (group.line_width ?? 1) > 0,
      endCaps: (group as any).end_caps ?? false,
      defaultColor: group.default_color || "#000000",
      styledSegments: (group.styled_segments || []).map((s: any) => ({
        text: s.text || "",
        color: s.color || group.default_color || "#000000",
        font_name: s.font_name,
        font_size: s.font_size,
        font_style: s.font_style,
      })),
      headerText: group.text || "",
    });
  };

  const applyHeaderPropsDialog = () => {
    if (!headerPropsDialog || !config) return;
    const { axis, level, groupIdx, distance, lineColor, lineWidth, lineStyle, lineLength, position, showLine, endCaps, defaultColor, styledSegments } = headerPropsDialog;
    // Deep clone to avoid mutating Zustand state directly
    const headers = JSON.parse(JSON.stringify(axis === "col" ? config.column_headers : config.row_headers));
    const group = headers[level].headers[groupIdx];
    group.distance = distance / 1000;
    group.line_color = lineColor;
    group.line_width = showLine ? lineWidth : 0;
    group.line_style = lineStyle;
    group.line_length = lineLength;
    group.position = position;
    group.end_caps = endCaps;
    group.default_color = defaultColor;
    // Filter out empty segments; when no segments are defined, the full
    // header text renders in the default colour (as before).
    const cleanSegs = (styledSegments || []).filter((s) => s.text && s.text.length > 0);
    group.styled_segments = cleanSegs;
    // Keep header.text in sync with the segment concatenation so downstream
    // logic (width measurement, plain-text fallback) stays coherent.
    if (cleanSegs.length > 0) {
      group.text = cleanSegs.map((s) => s.text).join("");
    }
    // Update store and backend
    const setConfig = useFigureStore.getState().setConfig;
    if (axis === "col") {
      setConfig({ ...config, column_headers: headers });
    } else {
      setConfig({ ...config, row_headers: headers });
    }
    const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
    patchFn.call(api, headers).catch(console.error);
    requestPreview();
    setHeaderPropsDialog(null);
  };

  /* ── Label Properties Dialog (for primary column/row labels) ── */

  const openLabelPropsDialog = (axis: "col" | "row", index: number) => {
    const labels = axis === "col" ? column_labels : row_labels;
    const lbl = labels[index];
    if (!lbl) return;
    setLabelPropsDialog({
      axis,
      index,
      distance: lbl.distance * 100,   // convert figure-fraction → percentage
      position: lbl.position,
      rotation: lbl.rotation ?? (axis === "row" ? 90 : 0),
      applyToAll: true,
    });
  };

  const applyLabelPropsDialog = () => {
    if (!labelPropsDialog || !config) return;
    const { axis, index, distance, position, rotation, applyToAll } = labelPropsDialog;
    const colLabels = JSON.parse(JSON.stringify(config.column_labels));
    const rowLabels = JSON.parse(JSON.stringify(config.row_labels));
    const fracDist = distance / 100;    // percentage → figure-fraction

    if (applyToAll) {
      // Sync distance across BOTH column and row labels for consistent spacing
      for (const lbl of colLabels) {
        lbl.distance = fracDist;
      }
      for (const lbl of rowLabels) {
        lbl.distance = fracDist;
      }
      // Apply position and rotation to same-axis labels only
      const sameAxisLabels = axis === "col" ? colLabels : rowLabels;
      for (const lbl of sameAxisLabels) {
        lbl.position = position;
        lbl.rotation = rotation;
      }
    } else {
      const labels = axis === "col" ? colLabels : rowLabels;
      labels[index].distance = fracDist;
      labels[index].position = position;
      labels[index].rotation = rotation;
    }

    const setConfig = useFigureStore.getState().setConfig;
    setConfig({ ...config, column_labels: colLabels, row_labels: rowLabels });
    // Patch both axes to backend
    api.patchColumnLabels(colLabels).catch(console.error);
    api.patchRowLabels(rowLabels).catch(console.error);
    requestPreview();
    setLabelPropsDialog(null);
  };

  /* ── Floating toolbar data ─────────────────────────── */

  const getToolbarData = () => {
    if (!toolbarTarget || !config) {
      return { text: "", fontSize: 10, fontName: "arial.ttf", fontStyle: [] as string[], color: "#000000" };
    }
    if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
      const headers = toolbarTarget.axis === "col" ? config.column_headers : config.row_headers;
      const group = headers[toolbarTarget.level]?.headers[toolbarTarget.groupIdx];
      if (group) {
        return {
          text: group.text,
          fontSize: group.font_size,
          fontName: group.font_name,
          fontStyle: group.font_style,
          color: group.default_color,
        };
      }
    }
    if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
      const labels = toolbarTarget.axis === "col" ? config.column_labels : config.row_labels;
      const lbl = labels[toolbarTarget.index];
      if (lbl) {
        return {
          text: lbl.text,
          fontSize: lbl.font_size,
          fontName: lbl.font_name,
          fontStyle: lbl.font_style,
          color: lbl.default_color,
        };
      }
    }
    return { text: "", fontSize: 10, fontName: "arial.ttf", fontStyle: [] as string[], color: "#000000" };
  };

  const toolbarData = getToolbarData();

  const handleToolbarTextChange = (text: string) => {
    if (!toolbarTarget) return;
    if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
      updateHeaderGroupText(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, text);
    } else if (toolbarTarget.type === "colLabel" && toolbarTarget.index !== undefined) {
      updateColumnLabel(toolbarTarget.index, text);
    } else if (toolbarTarget.type === "rowLabel" && toolbarTarget.index !== undefined) {
      updateRowLabel(toolbarTarget.index, text);
    }
  };

  // These three handlers now share the applyStylingPatch dispatcher
  // defined below — keeping the per-selection scoping logic in one place
  // and ensuring headers AND labels both get per-character styling.
  // (applyStylingPatch is defined below this; it calls these handlers'
  // fall-through via the `fullPatchFn` argument. TypeScript hoisting
  // means forward refs are fine at runtime.)
  const handleToolbarFontSizeChange = (size: number) => {
    applyStylingPatch({ font_size: size }, () => {
      if (!toolbarTarget) return;
      if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
        updateHeaderGroupFormatting(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, { font_size: size });
      } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
        updateLabelFormatting(toolbarTarget.axis, toolbarTarget.index, { font_size: size });
      }
    });
  };

  const handleToolbarFontNameChange = (name: string) => {
    applyStylingPatch({ font_name: name }, () => {
      if (!toolbarTarget) return;
      if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
        updateHeaderGroupFormatting(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, { font_name: name });
      } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
        updateLabelFormatting(toolbarTarget.axis, toolbarTarget.index, { font_name: name });
      }
    });
  };

  // Resolve the effective font_style for the currently-selected characters
  // (intersection of per-char styles). If there's no selection, falls back
  // to the element's default font_style. This lets the toggle handler ADD
  // a style to chars that don't have it and REMOVE it from chars that all
  // already do — i.e. proper stacking (bold+italic+underline+...).
  const effectiveFontStylesForSelection = (): string[] => {
    const sel = resolveSelection();
    if (!sel || !toolbarTarget) return [...toolbarData.fontStyle];
    let fullText = "";
    let segs: HeaderStyledSegment[] | undefined;
    let defaultStyles: string[] = [];
    if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
      const headers = toolbarTarget.axis === "col" ? column_headers : row_headers;
      const g = headers[toolbarTarget.level]?.headers[toolbarTarget.groupIdx];
      if (!g) return [...toolbarData.fontStyle];
      fullText = g.text || "";
      segs = (g.styled_segments as HeaderStyledSegment[]) || undefined;
      defaultStyles = g.font_style || [];
    } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
      const labels = toolbarTarget.axis === "col" ? (config?.column_labels ?? []) : (config?.row_labels ?? []);
      const lbl = labels[toolbarTarget.index];
      if (!lbl) return [...toolbarData.fontStyle];
      fullText = lbl.text || "";
      segs = (lbl.styled_segments as HeaderStyledSegment[]) || undefined;
      defaultStyles = lbl.font_style || [];
    }
    // Build per-char style list (mirrors applySegmentPatch's flatten step).
    type CharStyle = string[];
    const flat: CharStyle[] = [];
    if (!segs || segs.length === 0) {
      for (let i = 0; i < fullText.length; i++) flat.push([...defaultStyles]);
    } else {
      for (const seg of segs) {
        const sty = seg.font_style && seg.font_style.length > 0 ? seg.font_style : defaultStyles;
        for (let i = 0; i < seg.text.length; i++) flat.push([...sty]);
      }
      if (flat.length !== fullText.length) {
        // Desync — fall back to defaults.
        flat.length = 0;
        for (let i = 0; i < fullText.length; i++) flat.push([...defaultStyles]);
      }
    }
    const lo = Math.max(0, Math.min(sel.start, sel.end));
    const hi = Math.min(flat.length, Math.max(sel.start, sel.end));
    if (hi <= lo) return [...toolbarData.fontStyle];
    // Intersection: a style is "present in selection" iff EVERY char in
    // the selection has it. That matches most word-processor toggle
    // semantics — partial presence means the next toggle ADDs.
    const first = flat[lo];
    const inter = first.filter((s) =>
      flat.slice(lo, hi).every((cs) => cs.includes(s)),
    );
    return inter;
  };

  const handleToolbarFontStyleToggle = (style: string) => {
    // Start from the SELECTION's effective styles (intersection) so
    // stacking works — previously we pulled from toolbarData.fontStyle
    // which is the HEADER-LEVEL default, so toggling Italic on an
    // already-Bold char would blow away the Bold.
    const current = effectiveFontStylesForSelection();
    const idx = current.indexOf(style);
    if (idx >= 0) current.splice(idx, 1);
    else current.push(style);
    applyStylingPatch({ font_style: current }, () => {
      if (!toolbarTarget) return;
      // "No selection" path — toggle against the element's full current
      // default font_style list instead of the bare selection-effective one.
      const elementCurrent = [...toolbarData.fontStyle];
      const ei = elementCurrent.indexOf(style);
      if (ei >= 0) elementCurrent.splice(ei, 1);
      else elementCurrent.push(style);
      if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
        updateHeaderGroupFormatting(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, { font_style: elementCurrent });
      } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
        updateLabelFormatting(toolbarTarget.axis, toolbarTarget.index, { font_style: elementCurrent });
      }
    });
  };

  // Tracks the text selection inside the currently-focused header / label
  // textarea so toolbar actions (colour, font, size, style) can be scoped
  // to just the selected range instead of overwriting the whole string.
  // { start, end } are character offsets into the full text; if start === end
  // there is no selection and the action applies to the whole element.
  // Tracks (a) the textarea the user is currently editing, and (b) the last
  // *non-empty* selection range the user made inside it.
  //
  // Relying on React synthetic `onSelect` and on live DOM reads at click
  // time was fragile — in several browsers, the moment the user clicks a
  // toolbar control the textarea's selection collapses before any handler
  // runs. The reliable source of truth is the document-level
  // `selectionchange` event, which fires every time the textarea's
  // selection changes (including caret moves). We listen globally and
  // cache only *non-empty* ranges, so the ref still holds the user's last
  // real selection when the toolbar handler finally fires.
  const toolbarTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Parallel channel for the new StyledTextEditor-based surfaces. When
  // a HeaderEditor activates, it sets activeEditorRef; captureSelection-
  // FromActive and the toolbar's Shift+Enter handler prefer it over the
  // textarea ref. Textarea-based surfaces still populate
  // toolbarTextareaRef via their existing onFocus/onSelect handlers.
  const activeEditorRef = useRef<StyledTextEditorHandle | null>(null);
  const activeEditorTargetRef = useRef<HeaderEditorTarget | null>(null);
  const toolbarSelectionRef = useRef<{ start: number; end: number } | null>(null);
  // Mirror the cached selection into reactive state so the toolbar can
  // display a preview of what substring will be styled next.
  const [selectionPreview, setSelectionPreview] = useState<string>("");
  const updateSelectionPreview = (ta: HTMLTextAreaElement, start: number, end: number) => {
    if (start === end) return;
    const text = (ta.value || "").slice(Math.min(start, end), Math.max(start, end));
    setSelectionPreview(text);
  };
  const updateSelectionPreviewFromText = (text: string, start: number, end: number) => {
    if (start === end) return;
    setSelectionPreview(text.slice(Math.min(start, end), Math.max(start, end)));
  };
  // Read the LIVE text from the store rather than from closure-captured
  // const refs. The selectionchange listener can fire before React's
  // re-render commits the latest store update — when the user deletes
  // and retypes text, the handler's closure still saw "Column 1" while
  // the store already had "Test", which made the chip slice the wrong
  // string and show old characters.
  const getActiveEditorText = (): string => {
    const t = activeEditorTargetRef.current;
    if (!t) return "";
    const cfg = useFigureStore.getState().config;
    if (!cfg) return "";
    if (t.type === "header") {
      const headers = t.axis === "col" ? cfg.column_headers : cfg.row_headers;
      return headers[t.level]?.headers[t.groupIdx]?.text || "";
    }
    const labels = t.axis === "col" ? cfg.column_labels : cfg.row_labels;
    return labels[t.index]?.text || "";
  };
  // Capture a non-empty selection from whichever surface is active
  // (HeaderEditor OR the legacy textarea).
  const captureSelectionFromActive = () => {
    const ed = activeEditorRef.current;
    if (ed) {
      const sel = ed.getSelection();
      if (sel && sel.start !== sel.end) {
        toolbarSelectionRef.current = sel;
        updateSelectionPreviewFromText(getActiveEditorText(), sel.start, sel.end);
        return;
      }
    }
    const ae = document.activeElement as HTMLElement | null;
    if (!ae || !(ae.tagName === "TEXTAREA" || ae.tagName === "INPUT")) return;
    const ta = ae as HTMLTextAreaElement | HTMLInputElement;
    const a = ta.getAttribute("aria-label") || "";
    if (!/\b(header|label)\b/i.test(a)) return;
    const s = (ta as HTMLTextAreaElement).selectionStart ?? 0;
    const en = (ta as HTMLTextAreaElement).selectionEnd ?? 0;
    toolbarTextareaRef.current = ta as HTMLTextAreaElement;
    if (s !== en) {
      toolbarSelectionRef.current = { start: s, end: en };
      updateSelectionPreview(ta as HTMLTextAreaElement, s, en);
    }
  };
  // Called by HeaderEditor on every non-empty selection change.
  const handleEditorSelectionChange = (sel: { start: number; end: number }) => {
    toolbarSelectionRef.current = sel;
    updateSelectionPreviewFromText(getActiveEditorText(), sel.start, sel.end);
  };
  // Called by HeaderEditor when the selection collapses or the user
  // edits the text. Drops the cached range so a subsequent toolbar
  // action doesn't apply to stale offsets, and clears the preview chip
  // so the user isn't looking at e.g. "Column" after deleting that text.
  const handleEditorSelectionCleared = () => {
    toolbarSelectionRef.current = null;
    setSelectionPreview("");
  };
  // Activated on editor focus — records it as "the current one" and
  // positions the floating toolbar to match.
  const activateEditor = (target: HeaderEditorTarget, handle: StyledTextEditorHandle) => {
    activeEditorRef.current = handle;
    activeEditorTargetRef.current = target;
    toolbarTextareaRef.current = null;
    toolbarSelectionRef.current = null;
    setSelectionPreview("");
    setToolbarTarget(target as ToolbarTarget);
  };
  const handleToolbarSelectionChange = (start: number, end: number) => {
    if (start !== end) {
      toolbarSelectionRef.current = { start, end };
      const ta = toolbarTextareaRef.current;
      if (ta) updateSelectionPreview(ta, start, end);
    }
  };
  useEffect(() => {
    // Document-level keydown / input logger — diagnostic. Logs EVERY key
    // event + target so we can see when a key does / doesn't reach a
    // textarea, and which element actually has focus.
    const onAnyKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      console.log("[mpf-global] keydown", {
        key: e.key,
        code: e.code,
        shift: e.shiftKey,
        ctrl: e.ctrlKey,
        meta: e.metaKey,
        alt: e.altKey,
        targetTag: t?.tagName,
        targetAria: t?.getAttribute?.("aria-label") || "",
        activeTag: (document.activeElement as HTMLElement | null)?.tagName,
        activeAria: (document.activeElement as HTMLElement | null)?.getAttribute?.("aria-label") || "",
      });
    };
    const onAnyKeyWindow = (e: KeyboardEvent) => {
      console.log("[mpf-window] keydown", { key: e.key, code: e.code, shift: e.shiftKey });
    };
    const onAnyKeyUp = (e: KeyboardEvent) => {
      console.log("[mpf-global] keyup", { key: e.key, code: e.code, shift: e.shiftKey });
    };
    const onBeforeInput = (e: Event) => {
      const t = e.target as HTMLElement | null;
      const a = t?.getAttribute?.("aria-label") || "";
      if (/\b(header|label)\b/i.test(a)) {
        console.log("[mpf-global] beforeinput", {
          tag: t?.tagName,
          aria: a,
          inputType: (e as unknown as { inputType?: string }).inputType,
          data: (e as unknown as { data?: string }).data,
        });
      }
    };
    const onFocusChange = () => {
      const ae = document.activeElement as HTMLElement | null;
      console.log("[mpf-global] focuschange", {
        tag: ae?.tagName,
        aria: ae?.getAttribute?.("aria-label") || "",
      });
    };
    document.addEventListener("focusin", onFocusChange);
    document.addEventListener("focusout", onFocusChange);
    document.addEventListener("keyup", onAnyKeyUp, true);
    document.addEventListener("beforeinput", onBeforeInput, true);
    window.addEventListener("keydown", onAnyKeyWindow, true);
    const onAnyInput = (e: Event) => {
      const t = e.target as HTMLTextAreaElement | HTMLInputElement | null;
      const a = t?.getAttribute?.("aria-label") || "";
      if (/\b(header|label)\b/i.test(a)) {
        console.log("[mpf-global] input", { targetTag: t?.tagName, aria: a, value: JSON.stringify(t?.value || "") });
      }
    };
    document.addEventListener("keydown", onAnyKey, true); // capture phase
    document.addEventListener("input", onAnyInput, true);
    // Document-level selectionchange (works in Chrome/Firefox for textarea)
    document.addEventListener("selectionchange", captureSelectionFromActive);
    // Global fallbacks — mouseup and keyup anywhere. Essential for WebKit
    // where selectionchange on document doesn't catch textarea selections,
    // and for the common case where drag-selecting ends with the cursor
    // OUTSIDE the textarea (React's onMouseUp on the element never fires).
    document.addEventListener("mouseup", captureSelectionFromActive);
    document.addEventListener("keyup", captureSelectionFromActive);
    // Final brute-force safety net: poll the active textarea's selection
    // 10× per second. If ALL the event-based capture paths above somehow
    // miss the user's selection (selectionchange not firing for textareas
    // in WebKit, React synthetic events being swallowed, etc.), this
    // guarantees the preview eventually reflects the current selection
    // while the user is still holding it.
    const pollId = window.setInterval(captureSelectionFromActive, 100);
    return () => {
      document.removeEventListener("focusin", onFocusChange);
      document.removeEventListener("focusout", onFocusChange);
      document.removeEventListener("keyup", onAnyKeyUp, true);
      document.removeEventListener("beforeinput", onBeforeInput, true);
      window.removeEventListener("keydown", onAnyKeyWindow, true);
      document.removeEventListener("keydown", onAnyKey, true);
      document.removeEventListener("input", onAnyInput, true);
      document.removeEventListener("selectionchange", captureSelectionFromActive);
      document.removeEventListener("mouseup", captureSelectionFromActive);
      document.removeEventListener("keyup", captureSelectionFromActive);
      window.clearInterval(pollId);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Resolve the selection at toolbar-action time: prefer the cached range
  // (captured by the selectionchange listener above right when the user
  // made the selection), falling back to a live DOM read.
  const resolveSelection = (): { start: number; end: number } | null => {
    const cached = toolbarSelectionRef.current;
    if (cached && cached.start !== cached.end) return cached;
    const ta = toolbarTextareaRef.current;
    if (ta) {
      const s = ta.selectionStart ?? 0;
      const e = ta.selectionEnd ?? 0;
      if (s !== e) return { start: s, end: e };
    }
    return null;
  };

  // Build a styled_segments array given the current full text, an existing
  // segments array (may be empty), a [start,end) selection and a patch to
  // apply to the selected span.
  const applySegmentPatch = (
    fullText: string,
    existing: HeaderStyledSegment[] | undefined,
    selection: { start: number; end: number },
    defaultColor: string,
    patch: Partial<Pick<HeaderStyledSegment, "color" | "font_name" | "font_size" | "font_style">>,
  ): HeaderStyledSegment[] => {
    // Flatten existing segments into per-character style array, falling back
    // to default_color when no segments are defined.
    const flat: Array<{ char: string; color: string; font_name?: string; font_size?: number; font_style?: string[] }> = [];
    if (!existing || existing.length === 0) {
      for (const ch of fullText) flat.push({ char: ch, color: defaultColor });
    } else {
      for (const seg of existing) {
        for (const ch of seg.text) {
          flat.push({
            char: ch,
            color: seg.color || defaultColor,
            font_name: seg.font_name,
            font_size: seg.font_size,
            font_style: seg.font_style,
          });
        }
      }
      // If the textarea has diverged from the concatenated segments (e.g.
      // user edited the plain text afterwards), re-sync to plain text.
      const concat = flat.map((f) => f.char).join("");
      if (concat !== fullText) {
        flat.length = 0;
        for (const ch of fullText) flat.push({ char: ch, color: defaultColor });
      }
    }

    // Apply the patch to characters in the selection range.
    const lo = Math.max(0, Math.min(selection.start, selection.end));
    const hi = Math.min(flat.length, Math.max(selection.start, selection.end));
    for (let i = lo; i < hi; i++) {
      if (patch.color !== undefined) flat[i].color = patch.color;
      if (patch.font_name !== undefined) flat[i].font_name = patch.font_name;
      if (patch.font_size !== undefined) flat[i].font_size = patch.font_size;
      if (patch.font_style !== undefined) flat[i].font_style = patch.font_style;
    }

    // Re-group adjacent chars with identical styling back into segments.
    const out: HeaderStyledSegment[] = [];
    for (const f of flat) {
      const last = out[out.length - 1];
      const sameStyle =
        last &&
        last.color === f.color &&
        last.font_name === f.font_name &&
        last.font_size === f.font_size &&
        JSON.stringify(last.font_style) === JSON.stringify(f.font_style);
      if (sameStyle) {
        last!.text += f.char;
      } else {
        out.push({
          text: f.char,
          color: f.color,
          font_name: f.font_name,
          font_size: f.font_size,
          font_style: f.font_style,
        });
      }
    }
    return out;
  };

  // Unified dispatcher that applies a selection-aware styling patch to
  // headers OR labels. If the user has a non-empty text selection in the
  // source control, the patch is scoped to that range via styled_segments;
  // otherwise it falls back to the element's default formatting field
  // (and clears styled_segments so the new default wins).
  const applyStylingPatch = (
    patch: Partial<Pick<HeaderStyledSegment, "color" | "font_name" | "font_size" | "font_style">>,
    fullPatchFn: () => void,   // called in the "no selection" case
  ) => {
    if (!toolbarTarget) return;
    const sel = resolveSelection();

    if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
      if (sel) {
        const headers = toolbarTarget.axis === "col" ? column_headers : row_headers;
        const group = headers[toolbarTarget.level]?.headers[toolbarTarget.groupIdx];
        if (!group) return;
        const nextSegs = applySegmentPatch(
          group.text || "",
          (group.styled_segments as HeaderStyledSegment[]) || [],
          sel,
          group.default_color || "#000000",
          patch,
        );
        updateHeaderGroupFormatting(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, { styled_segments: nextSegs });
      } else {
        fullPatchFn();
      }
    } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
      if (sel) {
        const labels = toolbarTarget.axis === "col" ? (config?.column_labels ?? []) : (config?.row_labels ?? []);
        const lbl = labels[toolbarTarget.index];
        if (!lbl) return;
        const nextSegs = applySegmentPatch(
          lbl.text || "",
          (lbl.styled_segments as HeaderStyledSegment[]) || [],
          sel,
          lbl.default_color || "#000000",
          patch,
        );
        updateLabelFormatting(toolbarTarget.axis, toolbarTarget.index, { styled_segments: nextSegs });
      } else {
        fullPatchFn();
      }
    }

    // Restore the editor's visual selection so it doesn't fade after a
    // toolbar action — but skip the re-focus if the user is currently
    // typing into a form input (e.g. the font-size field). Re-focusing
    // would steal focus from the input mid-type. If activeElement is
    // a focusable form control, leave focus alone and only re-apply
    // the selection range so the highlight stays visible in the editor.
    if (sel) {
      const ed = activeEditorRef.current;
      if (ed) {
        const ae = document.activeElement as HTMLElement | null;
        const focusElsewhere =
          ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" || ae.tagName === "SELECT");
        requestAnimationFrame(() => {
          if (focusElsewhere) {
            ed.setSelection(sel.start, sel.end);
          } else {
            ed.focus();
            ed.setSelection(sel.start, sel.end);
          }
        });
      }
    }
  };

  const handleToolbarColorChange = (color: string) => {
    applyStylingPatch({ color }, () => {
      if (!toolbarTarget) return;
      if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
        updateHeaderGroupFormatting(toolbarTarget.axis, toolbarTarget.level, toolbarTarget.groupIdx, { default_color: color, styled_segments: [] });
      } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
        updateLabelFormatting(toolbarTarget.axis, toolbarTarget.index, { default_color: color, styled_segments: [] });
      }
    });
  };

  /* ── Layout math ───────────────────────────────────── */

  const colHdrTiers = column_headers.length;
  const rowHdrTiers = row_headers.length;
  const addColBtnRow = 1;
  const hdrRowStart = 2;
  const labelRow = hdrRowStart + colHdrTiers;
  const panelRowStart = labelRow + 1;
  const addRowBtnCol = 1;
  const hdrColStart = 2;
  const labelCol = hdrColStart + rowHdrTiers;
  const panelColStart = labelCol + 1;
  // X buttons for col headers go in the column AFTER the last panel column
  const colHdrXBtnCol = panelColStart + cols;
  // X buttons for row headers go in the row AFTER the last panel row
  const rowHdrXBtnRow = panelRowStart + rows;

  const cellSize = 160;

  // Precompute the widest line-count for each row-header tier and the
  // primary row label. The grid track for those columns is sized to fit
  // the unwrapped content because CSS grid's `max-content` track sizing
  // doesn't propagate cleanly through a vertical-rl + transform-rotated
  // flex container.
  //
  // Width formula matches the col-header row template's growth pattern:
  // 28px minimum, plus 12px per line beyond the first.  For a 10px
  // font with line-height 1.2, that's exactly the line-stride. So row
  // header thickness ≡ col header height for the same line count.
  const rowHdrLineCounts = row_headers.map((level) =>
    level.headers.reduce(
      (max, h) => Math.max(max, (h.text || "").split("\n").length || 1),
      1,
    ),
  );
  const rowLabelMaxLines = (config.row_labels || []).reduce(
    (max, l) => Math.max(max, (l.text || "").split("\n").length || 1),
    1,
  );
  const trackWidthForLines = (n: number) => `${Math.max(28, 4 + n * 12)}px`;
  const rowHdrTrackWidths = rowHdrLineCounts.map(trackWidthForLines);
  const rowLabelTrackWidth = trackWidthForLines(rowLabelMaxLines);

  const needsColXBtnCol = colHdrTiers > 0 || config.show_column_labels !== false;
  const colTemplate = [
    "28px",                                     // +Row button column
    ...rowHdrTrackWidths,                       // secondary row header tiers — grow with line breaks
    rowLabelTrackWidth,                         // primary row label column — grows with line breaks
    ...Array(cols).fill(`${cellSize}px`),       // panel columns
    ...(needsColXBtnCol ? ["28px"] : []),       // X button column for col headers + primary col labels
  ].join(" ");

  const needsRowXBtnRow = rowHdrTiers > 0 || config.show_row_labels !== false;
  const rowTemplate = [
    "24px",                                     // +Col button row
    // Col header tier rows: auto-grow to fit multi-line headers so 5+
    // line breaks aren't clipped. 28px minimum (single line).
    ...Array(colHdrTiers).fill("minmax(28px, max-content)"),
    // Col label row: same auto-grow behavior.
    "minmax(28px, max-content)",                // col label row
    ...Array(rows).fill(`${cellSize}px`),       // panel rows
    ...(needsRowXBtnRow ? ["28px"] : []),       // X button row for row headers + primary row labels
  ].join(" ");

  /* ── Render helpers ────────────────────────────────── */

  /** Render a drag handle for a header group edge */
  const renderDragHandle = (
    axis: "col" | "row",
    level: number,
    groupIdx: number,
    edge: "start" | "end",
    currentIndices: number[],
  ) => {
    const isCol = axis === "col";
    const isDragging = dragState?.axis === axis && dragState?.level === level && dragState?.groupIdx === groupIdx && dragState?.edge === edge;
    return (
      <div
        style={{
          position: "absolute",
          ...(isCol
            ? {
                [edge === "start" ? "left" : "right"]: -4,
                top: "50%",
                transform: "translateY(-50%)",
                height: 20,
                width: 6,
                cursor: "col-resize",
              }
            : {
                [edge === "start" ? "top" : "bottom"]: -4,
                left: "50%",
                transform: "translateX(-50%)",
                width: 20,
                height: 6,
                cursor: "row-resize",
              }),
          backgroundColor: isDragging ? "var(--c-accent)" : "rgba(255,255,255,0.15)",
          zIndex: 10,
          transition: "background-color 0.15s",
          borderRadius: "3px",
        }}
        className="hover:!bg-[rgba(59,130,246,0.5)]"
        onMouseDown={(e) => {
          e.stopPropagation();
          e.preventDefault();
          handleDragStart(e, axis, level, groupIdx, edge, currentIndices);
        }}
      />
    );
  };

  return (
    <div
      className="p-3 overflow-auto"
      onMouseMove={dragState ? handleMouseMove : undefined}
      onMouseUp={dragState ? handleMouseUp : undefined}
      onMouseLeave={dragState ? handleMouseUp : undefined}
      style={dragState ? { cursor: dragState.axis === "col" ? "col-resize" : "row-resize" } : undefined}
    >
      <div
        ref={gridRef}
        className="inline-grid gap-[3px]"
        style={{
          gridTemplateColumns: colTemplate,
          gridTemplateRows: rowTemplate,
          minWidth: cols > 5 ? `${cols * 120}px` : undefined,
        }}
      >
        {/* ── Corner: + Col Header button ──────────────── */}
        <div
          style={{
            gridRow: addColBtnRow,
            gridColumn: `${panelColStart} / span ${cols}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
          }}
        >
          <button
            className="flex items-center justify-center rounded text-[11px]
                       hover:opacity-80 transition-opacity"
            style={{
              backgroundColor: "var(--c-surface2)",
              color: "var(--c-text-dim)",
              border: "1px dashed var(--c-border)",
              width: 22,
              height: 22,
              lineHeight: 1,
            }}
            onClick={addColumnHeaderLevel}
            title="Add column header level"
            aria-label="Add column header level"
          >
            +
          </button>
        </div>

        {/* ── Corner: + Row Header button ──────────────── */}
        <div
          style={{
            gridRow: `${panelRowStart} / span ${rows}`,
            gridColumn: addRowBtnCol,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <button
            className="flex items-center justify-center rounded text-[11px]
                       hover:opacity-80 transition-opacity"
            style={{
              backgroundColor: "var(--c-surface2)",
              color: "var(--c-text-dim)",
              border: "1px dashed var(--c-border)",
              width: 22,
              height: 22,
              lineHeight: 1,
            }}
            onClick={addRowHeaderLevel}
            title="Add row header level"
            aria-label="Add row header level"
          >
            +
          </button>
        </div>

        {/* ── Column header tiers ──────────────────────── */}
        {column_headers.map((level, li) => {
          const gridRow = hdrRowStart + li;

          // Swap UP/DOWN buttons — at the LEFT of the lane (only when 2+ levels)
          const swapBtns = colHdrTiers >= 2 ? (
            <div
              key={`ch-swap-${li}`}
              style={{
                gridRow,
                gridColumn: addRowBtnCol,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 0,
              }}
            >
              {li > 0 && (
                <Tooltip title="Move this header level up" placement="left" arrow>
                  <button
                    className="flex items-center justify-center rounded-full
                               hover:bg-blue-500/30 transition-colors"
                    style={{
                      color: "var(--c-text-dim)",
                      width: 18,
                      height: 14,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      padding: 0,
                    }}
                    onClick={() => swapColumnHeaderLevels(li, li - 1)}
                    aria-label={`Move column header level ${li + 1} up`}
                  >
                    <ArrowUpwardIcon sx={{ fontSize: 12 }} />
                  </button>
                </Tooltip>
              )}
              {li < colHdrTiers - 1 && (
                <Tooltip title="Move this header level down" placement="left" arrow>
                  <button
                    className="flex items-center justify-center rounded-full
                               hover:bg-blue-500/30 transition-colors"
                    style={{
                      color: "var(--c-text-dim)",
                      width: 18,
                      height: 14,
                      border: "none",
                      background: "transparent",
                      cursor: "pointer",
                      padding: 0,
                    }}
                    onClick={() => swapColumnHeaderLevels(li, li + 1)}
                    aria-label={`Move column header level ${li + 1} down`}
                  >
                    <ArrowDownwardIcon sx={{ fontSize: 12 }} />
                  </button>
                </Tooltip>
              )}
            </div>
          ) : null;

          // X button at the RIGHT end of the lane
          const removeBtn = (
            <Tooltip key={`ch-rm-${li}`} title="Remove this header level" placement="right" arrow>
              <button
                className="flex items-center justify-center rounded-full
                           hover:bg-red-500/30 transition-colors"
                style={{
                  gridRow,
                  gridColumn: colHdrXBtnCol,
                  color: "var(--c-text-dim)",
                  width: 20,
                  height: 20,
                  margin: "auto",
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                }}
                onClick={() => removeColumnHeaderLevel(li)}
                aria-label={`Remove column header level ${li + 1}`}
              >
                <CloseIcon sx={{ fontSize: 14 }} />
              </button>
            </Tooltip>
          );

          // Render existing header groups
          const groups = level.headers.map((group, gi) => {
            const groupCols = [...group.columns_or_rows].sort((a, b) => a - b);
            const startCol = groupCols.length > 0 ? groupCols[0] : 0;
            const span = groupCols.length > 0 ? groupCols.length : 1;

            // Use drag preview indices if this group is being dragged
            const isBeingDragged =
              dragState?.axis === "col" && dragState?.level === li && dragState?.groupIdx === gi;
            const displayIndices = isBeingDragged && dragPreviewIndices ? dragPreviewIndices : groupCols;
            const displayStart = displayIndices.length > 0 ? displayIndices[0] : startCol;
            const displaySpan = displayIndices.length > 0 ? displayIndices.length : span;

            return (
              <div
                key={`ch-${li}-${gi}`}
                className="flex items-center justify-center gap-0.5 text-[10px] group/hdr"
                style={{
                  gridRow,
                  gridColumn: `${panelColStart + displayStart} / span ${displaySpan}`,
                  position: "relative",
                  opacity: isBeingDragged ? 0.8 : 1,
                  cursor: "pointer",
                }}
                onClick={(e) => handleHeaderClick(e, "col", li, gi)}
                onContextMenu={(e) => handleHeaderContextMenu(e, "col", li, gi)}
              >
                {/* Left drag handle */}
                {renderDragHandle("col", li, gi, "start", groupCols)}

                <div
                  style={{ position: "relative", flex: 1, minWidth: 0 }}
                  onFocus={(e) => { setToolbarAnchor(e.currentTarget); }}
                >
                  <HeaderEditor
                    target={{ type: "header", axis: "col", level: li, groupIdx: gi }}
                    text={group.text}
                    styledSegments={group.styled_segments as HeaderStyledSegment[] | undefined}
                    defaultColor={group.default_color}
                    fontStyle={group.font_style}
                    className="text-center text-[10px] rounded px-1 py-0.5 no-overlay-scrollbar hover:ring-1 hover:ring-blue-400/40 transition-shadow"
                    style={{
                      minHeight: "28px",
                      // Flex-center so the text sits visually centered
                      // in the box whether it's single- or multi-line.
                      // text-align:center (via the class) handles the
                      // inline (horizontal) axis; align-items:center
                      // handles the block (vertical) axis.
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      backgroundColor: isBeingDragged ? "var(--c-accent)" : "rgba(255,255,255,0.15)",
                      borderBottom: `${group.line_width}px ${group.line_style === "dashed" ? "dashed" : group.line_style === "dotted" ? "dotted" : "solid"} ${group.line_color}`,
                      textDecoration: group.font_style?.includes("Strikethrough") ? "line-through" : group.font_style?.includes("Underline") ? "underline" : undefined,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      lineHeight: 1.2,
                    }}
                    onTextChange={(newText) => {
                      updateHeaderGroupText("col", li, gi, newText);
                      if (newText === "" && (group.styled_segments?.length ?? 0) > 0) {
                        updateHeaderGroupFormatting("col", li, gi, { styled_segments: [] });
                      }
                    }}
                    onActivate={activateEditor}
                    onSelectionNonEmpty={handleEditorSelectionChange}
                    onSelectionCleared={handleEditorSelectionCleared}
                    onShiftEnter={(sel) => {
                      const newCaret = insertLineBreakInHeader("col", li, gi, sel.start, sel.end);
                      if (newCaret !== undefined) {
                        const h = activeEditorRef.current;
                        requestAnimationFrame(() => { h?.focus(); h?.setSelection(newCaret, newCaret); });
                      }
                    }}
                    enterBlurs
                    onClick={(e) => e.stopPropagation()}
                    onBeforeAction={captureSelectionFromActive}
                  />
                </div>

                {/* Always-visible edit button */}
                <Tooltip title="Edit header properties" placement="right" arrow>
                  <button
                    style={{
                      background: "transparent",
                      border: "none",
                      cursor: "pointer",
                      color: "var(--c-text-dim)",
                      padding: 0,
                      lineHeight: 1,
                      flexShrink: 0,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      openHeaderPropsDialog("col", li, gi);
                    }}
                    aria-label={`Edit column header ${li + 1} group ${gi + 1} properties`}
                  >
                    <TuneIcon sx={{ fontSize: 12 }} />
                  </button>
                </Tooltip>
                {/* Position swap button */}
                <Tooltip title={`Move to ${group.position === "Top" ? "Bottom" : "Top"}`} placement="right" arrow>
                  <button
                    style={{
                      background: "transparent",
                      border: "none",
                      cursor: "pointer",
                      color: "var(--c-text-dim)",
                      fontSize: "10px",
                      padding: "0 2px",
                      lineHeight: 1,
                      flexShrink: 0,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      const newPos = group.position === "Top" ? "Bottom" : "Top";
                      updateHeaderGroupFormatting("col", li, gi, { position: newPos });
                    }}
                    aria-label={`Move column header to ${group.position === "Top" ? "Bottom" : "Top"}`}
                  >
                    ↕
                  </button>
                </Tooltip>

                {/* Right drag handle */}
                {renderDragHandle("col", li, gi, "end", groupCols)}
              </div>
            );
          });

          // Fill empty columns not covered by any group with click-to-create cells
          const coveredCols = new Set(
            level.headers.flatMap((g) => g.columns_or_rows),
          );
          const emptyCols = Array.from({ length: cols }, (_, i) => i).filter(
            (c) => !coveredCols.has(c),
          );
          const emptySlots = emptyCols.map((c) => (
            <Tooltip key={`ch-empty-${li}-${c}`} title="Click to add header" placement="top" arrow>
              <div
                className="flex items-center justify-center rounded cursor-pointer
                           hover:bg-[var(--c-surface2)] transition-colors"
                style={{
                  gridRow,
                  gridColumn: panelColStart + c,
                  border: "1px dashed var(--c-border)",
                  opacity: 0.5,
                }}
                onClick={() => handleEmptyCellClick("col", li, c)}
                onContextMenu={(e) => handleEmptyCellContextMenu(e, "col", li, c)}
              />
            </Tooltip>
          ));

          return [swapBtns, removeBtn, ...groups, ...emptySlots].filter(Boolean);
        })}

        {/* ── Column labels row ────────────────────────── */}
        {config.show_column_labels !== false ? (
          <>
          {column_labels.map((lbl, ci) => {
            return (
            <div
              key={`cl-${ci}`}
              className="flex items-center justify-center gap-0.5 group/cl"
              style={{
                gridRow: labelRow,
                gridColumn: panelColStart + ci,
                position: "relative",
              }}
              onClick={(e) => handleLabelClick(e, "col", ci)}
              onFocus={(e) => { setToolbarAnchor(e.currentTarget); }}
            >
              <HeaderEditor
                target={{ type: "colLabel", axis: "col", index: ci }}
                text={lbl.text}
                styledSegments={lbl.styled_segments as HeaderStyledSegment[] | undefined}
                defaultColor={lbl.default_color}
                fontStyle={lbl.font_style}
                className="text-center text-[10px] rounded px-1 py-0.5 flex-1 min-w-0 no-overlay-scrollbar"
                style={{
                  minHeight: "28px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "rgba(255,255,255,0.15)",
                  textDecoration: lbl.font_style?.includes("Strikethrough") ? "line-through" : lbl.font_style?.includes("Underline") ? "underline" : undefined,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  lineHeight: 1.2,
                }}
                onTextChange={(newText) => {
                  updateColumnLabel(ci, newText);
                  if (newText === "" && (lbl.styled_segments?.length ?? 0) > 0) {
                    updateLabelFormatting("col", ci, { styled_segments: [] });
                  }
                }}
                onActivate={activateEditor}
                onSelectionNonEmpty={handleEditorSelectionChange}
                onSelectionCleared={handleEditorSelectionCleared}
                onShiftEnter={(sel) => {
                  const newCaret = insertLineBreakInLabel("col", ci, sel.start, sel.end);
                  if (newCaret !== undefined) {
                    const h = activeEditorRef.current;
                    requestAnimationFrame(() => { h?.focus(); h?.setSelection(newCaret, newCaret); });
                  }
                }}
                enterBlurs
                onClick={(e) => e.stopPropagation()}
                onBeforeAction={captureSelectionFromActive}
              />
              <Tooltip title="Edit header properties" placement="right" arrow>
                <button
                  className=""
                  style={{
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "var(--c-text-dim)",
                    padding: 0,
                    lineHeight: 1,
                    flexShrink: 0,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    openLabelPropsDialog("col", ci);
                  }}
                  aria-label={`Edit column ${ci + 1} label properties`}
                >
                  <TuneIcon sx={{ fontSize: 12 }} />
                </button>
              </Tooltip>
              <Tooltip title={`Move to ${lbl.position === "Top" ? "Bottom" : "Top"}`} placement="right" arrow>
                <button
                  className=""
                  style={{
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "var(--c-text-dim)",
                    fontSize: "10px",
                    padding: "0 2px",
                    lineHeight: 1,
                    flexShrink: 0,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    const newPos = lbl.position === "Top" ? "Bottom" : "Top";
                    updateLabelFormatting("col", ci, { position: newPos });
                  }}
                  aria-label={`Move column ${ci + 1} label to ${lbl.position === "Top" ? "Bottom" : "Top"}`}
                >
                  ↕
                </button>
              </Tooltip>
            </div>
            );
          })}
          {/* X button to remove column labels — same style as secondary header X */}
          <Tooltip title="Remove column labels" placement="right" arrow>
            <button
              className="flex items-center justify-center rounded-full hover:bg-red-500/30 transition-colors"
              style={{
                gridRow: labelRow,
                gridColumn: panelColStart + cols,
                color: "var(--c-text-dim)",
                width: 20,
                height: 20,
                margin: "auto",
                border: "none",
                background: "transparent",
                cursor: "pointer",
              }}
              onClick={() => setConfig({ ...config, show_column_labels: false })}
              aria-label="Remove column labels"
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </button>
          </Tooltip>
          </>
        ) : (
          /* Per-column + buttons to restore column labels */
          <>
          {Array.from({ length: cols }, (_, ci) => (
            <Tooltip key={`cl-add-${ci}`} title="Add column labels" placement="bottom" arrow>
              <button
                style={{
                  gridRow: labelRow,
                  gridColumn: panelColStart + ci,
                  background: "transparent",
                  border: "1px dashed rgba(255,255,255,0.15)",
                  borderRadius: "4px",
                  cursor: "pointer",
                  color: "var(--c-text-dim)",
                  fontSize: "12px",
                  padding: "2px 4px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "28px",
                }}
                onClick={() => setConfig({ ...config, show_column_labels: true })}
                aria-label={`Add column ${ci + 1} label`}
              >
                +
              </button>
            </Tooltip>
          ))}
          </>
        )}

        {/* ── Row header tiers + labels + panel cells ──── */}
        {Array.from({ length: rows }, (_, ri) => {
          const panelRow = panelRowStart + ri;

          // Row header tier cells
          const hdrCells = row_headers.map((level, li) => {
            const group = level.headers.find((g) =>
              g.columns_or_rows.includes(ri),
            );
            const gridCol = hdrColStart + li;

            if (!group) {
              // Empty cell - click to create
              return (
                <Tooltip key={`rh-${li}-${ri}`} title="Click to add header" placement="left" arrow>
                  <div
                    className="flex items-center justify-center rounded cursor-pointer
                               hover:bg-[var(--c-surface2)] transition-colors"
                    style={{
                      gridRow: panelRow,
                      gridColumn: gridCol,
                      border: "1px dashed var(--c-border)",
                      opacity: 0.5,
                    }}
                    onClick={() => handleEmptyCellClick("row", li, ri)}
                    onContextMenu={(e) => handleEmptyCellContextMenu(e, "row", li, ri)}
                  />
                </Tooltip>
              );
            }

            // Only render for the first row in the span
            const firstRow = Math.min(...group.columns_or_rows);
            if (ri !== firstRow) return null;

            const span = group.columns_or_rows.length;
            const gi = level.headers.indexOf(group);
            const sortedIndices = [...group.columns_or_rows].sort((a, b) => a - b);

            // Check if being dragged
            const isBeingDragged =
              dragState?.axis === "row" && dragState?.level === li && dragState?.groupIdx === gi;
            const displayIndices = isBeingDragged && dragPreviewIndices ? dragPreviewIndices : sortedIndices;
            const displayStart = displayIndices[0];
            const displaySpan = displayIndices.length;

            return (
              <div
                key={`rh-${li}-${ri}`}
                className="flex flex-col items-center justify-center text-[10px] group/rhdr"
                style={{
                  gridRow: `${panelRowStart + displayStart} / span ${displaySpan}`,
                  gridColumn: gridCol,
                  position: "relative",
                  opacity: isBeingDragged ? 0.8 : 1,
                  cursor: "pointer",
                  // Separator LINE lives on the outer grid-cell so the
                  // inner wrapper can have symmetric padding around the
                  // text (centered both axes). Previously the wrapper
                  // carried borderRight + paddingRight which pushed the
                  // text visually toward the line side.
                  borderRight: `${group.line_width}px ${group.line_style === "dashed" ? "dashed" : group.line_style === "dotted" ? "dotted" : "solid"} ${group.line_color}`,
                }}
                onClick={(e) => handleHeaderClick(e, "row", li, gi)}
                onContextMenu={(e) => handleHeaderContextMenu(e, "row", li, gi)}
              >
                {/* Top drag handle */}
                {renderDragHandle("row", li, gi, "start", sortedIndices)}

                {(() => {
                  const isRotated = (group.rotation ?? 90) !== 0;
                  return (
                  <div
                    style={{
                      ...(isRotated
                        ? { writingMode: "vertical-rl" as const, transform: "rotate(180deg)", textOrientation: "mixed" as const }
                        : {}),
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      // Stretch to fill the cell so the editor inside
                      // centers within the full cell width (= same width
                      // for every row in the column). With shrink-to-
                      // content the wrapper varied per row and the text
                      // looked off-centre between rows.
                      width: "100%",
                      height: "100%",
                      minHeight: 0,
                      backgroundColor: isBeingDragged
                        ? "var(--c-accent)"
                        : "rgba(255,255,255,0.15)",
                      borderRadius: 4,
                    }}
                    onFocus={(e) => { setToolbarAnchor(e.currentTarget.parentElement || e.currentTarget); }}
                  >
                    <HeaderEditor
                      target={{ type: "header", axis: "row", level: li, groupIdx: gi }}
                      text={group.text}
                      styledSegments={group.styled_segments as HeaderStyledSegment[] | undefined}
                      defaultColor={group.default_color}
                      fontStyle={group.font_style}
                      className="text-center text-[10px] rounded px-0.5 py-1 no-overlay-scrollbar hover:ring-1 hover:ring-blue-400/40 transition-shadow"
                      style={{
                        // Flex-center the text along both axes, regardless
                        // of writing mode. In vertical-rl, align-items
                        // centers along the BLOCK axis (= horizontal
                        // physical) so text doesn't pile against the
                        // block-start (right) edge. justify-content centers
                        // along the inline axis (= vertical), redundant
                        // with text-align: center but harmless.
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        textDecoration: group.font_style?.includes("Strikethrough") ? "line-through" : group.font_style?.includes("Underline") ? "underline" : undefined,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        lineHeight: 1.2,
                        ...(isRotated
                          ? { width: "100%", height: "100%" }
                          : { minHeight: "28px" }),
                      }}
                      onTextChange={(newText) => {
                        updateHeaderGroupText("row", li, gi, newText);
                        if (newText === "" && (group.styled_segments?.length ?? 0) > 0) {
                          updateHeaderGroupFormatting("row", li, gi, { styled_segments: [] });
                        }
                      }}
                      onActivate={activateEditor}
                      onSelectionNonEmpty={handleEditorSelectionChange}
                      onSelectionCleared={handleEditorSelectionCleared}
                      onShiftEnter={(sel) => {
                        const newCaret = insertLineBreakInHeader("row", li, gi, sel.start, sel.end);
                        if (newCaret !== undefined) {
                          const h = activeEditorRef.current;
                          requestAnimationFrame(() => { h?.focus(); h?.setSelection(newCaret, newCaret); });
                        }
                      }}
                      enterBlurs
                      onClick={(e) => e.stopPropagation()}
                      onBeforeAction={captureSelectionFromActive}
                    />
                  </div>
                  );
                })()}

                {/* Buttons below the bounding box — absolute-positioned so
                    they don't consume cell vertical space, which would push
                    the rotated wrapper up and make text appear off-centre. */}
                <div style={{ position: "absolute", bottom: 2, left: 0, right: 0, display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 2 }}>
                  <Tooltip title="Edit header properties" placement="bottom" arrow>
                    <button
                      style={{
                        background: "transparent",
                        border: "none",
                        cursor: "pointer",
                        color: "var(--c-text-dim)",
                        padding: 0,
                        lineHeight: 1,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        openHeaderPropsDialog("row", li, gi);
                      }}
                      aria-label={`Edit row header ${li + 1} group ${gi + 1} properties`}
                    >
                      <TuneIcon sx={{ fontSize: 10 }} />
                    </button>
                  </Tooltip>
                  <Tooltip title={`Move to ${group.position === "Left" ? "Right" : "Left"}`} placement="bottom" arrow>
                    <button
                      style={{
                        background: "transparent",
                        border: "none",
                        cursor: "pointer",
                        color: "var(--c-text-dim)",
                        fontSize: "9px",
                        padding: 0,
                        lineHeight: 1,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        const newPos = group.position === "Left" ? "Right" : "Left";
                        updateHeaderGroupFormatting("row", li, gi, { position: newPos });
                      }}
                      aria-label={`Move row header to ${group.position === "Left" ? "Right" : "Left"}`}
                    >
                      ↔
                    </button>
                  </Tooltip>
                </div>

                {/* Bottom drag handle */}
                {renderDragHandle("row", li, gi, "end", sortedIndices)}
              </div>
            );
          });

          // X buttons + swap buttons for row header levels (render at BOTTOM after last row)
          const removeRowHdrBtns =
            ri === rows - 1
              ? row_headers.map((_, li) => (
                  <div
                    key={`rh-ctrl-${li}`}
                    style={{
                      gridRow: rowHdrXBtnRow,
                      gridColumn: hdrColStart + li,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 0,
                    }}
                  >
                    {/* Swap LEFT/RIGHT buttons (only when 2+ levels) */}
                    {rowHdrTiers >= 2 && (
                      <div style={{ display: "flex", gap: 0 }}>
                        {li > 0 && (
                          <Tooltip title="Move this header level left" placement="bottom" arrow>
                            <button
                              className="flex items-center justify-center rounded-full
                                         hover:bg-blue-500/30 transition-colors"
                              style={{
                                color: "var(--c-text-dim)",
                                width: 14,
                                height: 14,
                                border: "none",
                                background: "transparent",
                                cursor: "pointer",
                                padding: 0,
                              }}
                              onClick={() => swapRowHeaderLevels(li, li - 1)}
                              aria-label={`Move row header level ${li + 1} left`}
                            >
                              <ArrowBackIcon sx={{ fontSize: 10 }} />
                            </button>
                          </Tooltip>
                        )}
                        {li < rowHdrTiers - 1 && (
                          <Tooltip title="Move this header level right" placement="bottom" arrow>
                            <button
                              className="flex items-center justify-center rounded-full
                                         hover:bg-blue-500/30 transition-colors"
                              style={{
                                color: "var(--c-text-dim)",
                                width: 14,
                                height: 14,
                                border: "none",
                                background: "transparent",
                                cursor: "pointer",
                                padding: 0,
                              }}
                              onClick={() => swapRowHeaderLevels(li, li + 1)}
                              aria-label={`Move row header level ${li + 1} right`}
                            >
                              <ArrowForwardIcon sx={{ fontSize: 10 }} />
                            </button>
                          </Tooltip>
                        )}
                      </div>
                    )}
                    {/* Remove button */}
                    <Tooltip title="Remove this header level" placement="bottom" arrow>
                      <button
                        className="flex items-center justify-center rounded-full
                                   hover:bg-red-500/30 transition-colors"
                        style={{
                          color: "var(--c-text-dim)",
                          width: 20,
                          height: 20,
                          border: "none",
                          background: "transparent",
                          cursor: "pointer",
                        }}
                        onClick={() => removeRowHeaderLevel(li)}
                        aria-label={`Remove row header level ${li + 1}`}
                      >
                        <CloseIcon sx={{ fontSize: 14 }} />
                      </button>
                    </Tooltip>
                  </div>
                ))
              : [];

          // X button for primary row labels (in the same X-button row as secondary headers)
          const removeRowLabelBtn = ri === rows - 1 && config.show_row_labels !== false ? (
            <Tooltip key="rl-rm" title="Remove row labels" placement="bottom" arrow>
              <button
                className="flex items-center justify-center rounded-full hover:bg-red-500/30 transition-colors"
                style={{
                  gridRow: rowHdrXBtnRow,
                  gridColumn: labelCol,
                  color: "var(--c-text-dim)",
                  width: 20,
                  height: 20,
                  margin: "auto",
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                }}
                onClick={() => setConfig({ ...config, show_row_labels: false })}
                aria-label="Remove row labels"
              >
                <CloseIcon sx={{ fontSize: 14 }} />
              </button>
            </Tooltip>
          ) : null;

          // Row label
          const rowLabel = config.show_row_labels !== false ? (
            <div
              key={`rl-${ri}`}
              className="group/rl"
              style={{
                gridRow: panelRow,
                gridColumn: labelCol,
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
              onClick={(e) => handleLabelClick(e, "row", ri)}
            >
              {(() => {
                const isRotated = (row_labels[ri]?.rotation ?? 90) !== 0;
                const rlbl = row_labels[ri];
                return (
                <div
                  style={{
                    ...(isRotated
                      ? { writingMode: "vertical-rl" as const, transform: "rotate(180deg)", textOrientation: "mixed" as const }
                      : {}),
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    // Stretch to fill the cell so the editor inside is
                    // centered within the full cell width — same wrapper
                    // width for every row in the column means the text
                    // looks consistently centred between rows.
                    width: "100%",
                    height: "100%",
                    minHeight: 0,
                    position: "relative",
                    backgroundColor: "rgba(255,255,255,0.15)",
                    borderRadius: 4,
                  }}
                  onFocus={(e) => { setToolbarAnchor(e.currentTarget.parentElement || e.currentTarget); }}
                >
                  <HeaderEditor
                    target={{ type: "rowLabel", axis: "row", index: ri }}
                    text={rlbl?.text ?? ""}
                    styledSegments={rlbl?.styled_segments as HeaderStyledSegment[] | undefined}
                    defaultColor={rlbl?.default_color}
                    fontStyle={rlbl?.font_style}
                    className="text-center text-[10px] rounded px-0.5 py-1 no-overlay-scrollbar"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      textDecoration: rlbl?.font_style?.includes("Strikethrough") ? "line-through" : rlbl?.font_style?.includes("Underline") ? "underline" : undefined,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      lineHeight: 1.2,
                      ...(isRotated
                        ? { width: "100%", height: "100%" }
                        : { minHeight: "28px", width: "100%" }),
                    }}
                    onTextChange={(newText) => {
                      updateRowLabel(ri, newText);
                      if (newText === "" && (rlbl?.styled_segments?.length ?? 0) > 0) {
                        updateLabelFormatting("row", ri, { styled_segments: [] });
                      }
                    }}
                    onActivate={activateEditor}
                    onSelectionNonEmpty={handleEditorSelectionChange}
                    onSelectionCleared={handleEditorSelectionCleared}
                    onShiftEnter={(sel) => {
                      const newCaret = insertLineBreakInLabel("row", ri, sel.start, sel.end);
                      if (newCaret !== undefined) {
                        const h = activeEditorRef.current;
                        requestAnimationFrame(() => { h?.focus(); h?.setSelection(newCaret, newCaret); });
                      }
                    }}
                    enterBlurs
                    onClick={(e) => e.stopPropagation()}
                    onBeforeAction={captureSelectionFromActive}
                  />
                </div>
                );
              })()}
              {/* Buttons below the bounding box — absolute-positioned so
                  they don't consume cell vertical space, which would push
                  the rotated wrapper up and make text appear off-centre. */}
              <div style={{ position: "absolute", bottom: 2, left: 0, right: 0, display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 2 }}>
                <Tooltip title="Edit header properties" placement="bottom" arrow>
                  <button
                    style={{
                      background: "transparent",
                      border: "none",
                      cursor: "pointer",
                      color: "var(--c-text-dim)",
                      padding: 0,
                      lineHeight: 1,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      openLabelPropsDialog("row", ri);
                    }}
                    aria-label={`Edit row ${ri + 1} label properties`}
                  >
                    <TuneIcon sx={{ fontSize: 10 }} />
                  </button>
                </Tooltip>
                <Tooltip title={`Move to ${row_labels[ri]?.position === "Left" ? "Right" : "Left"}`} placement="bottom" arrow>
                  <button
                    style={{
                      background: "transparent",
                      border: "none",
                      cursor: "pointer",
                      color: "var(--c-text-dim)",
                      fontSize: "9px",
                      padding: 0,
                      lineHeight: 1,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      const newPos = row_labels[ri]?.position === "Left" ? "Right" : "Left";
                      updateLabelFormatting("row", ri, { position: newPos });
                    }}
                    aria-label={`Move row ${ri + 1} label to ${row_labels[ri]?.position === "Left" ? "Right" : "Left"}`}
                  >
                    ↔
                  </button>
                </Tooltip>
              </div>
            </div>
          ) : (
            /* Per-row + button to restore row labels */
            <Tooltip key={`rl-add-${ri}`} title="Add row labels" placement="right" arrow>
              <button
                style={{
                  gridRow: panelRow,
                  gridColumn: labelCol,
                  background: "transparent",
                  border: "1px dashed rgba(255,255,255,0.15)",
                  borderRadius: "4px",
                  cursor: "pointer",
                  color: "var(--c-text-dim)",
                  fontSize: "12px",
                  padding: "4px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "28px",
                }}
                onClick={() => setConfig({ ...config, show_row_labels: true })}
                aria-label={`Add row ${ri + 1} label`}
              >
                +
              </button>
            </Tooltip>
          );

          // Panel cells
          const cells = Array.from({ length: cols }, (_, ci) => (
            <div
              key={`cell-${ri}-${ci}`}
              style={{
                gridRow: panelRow,
                gridColumn: panelColStart + ci,
                aspectRatio: "1 / 1",
              }}
            >
              <PanelCell
                row={ri}
                col={ci}
                imageName={config.panels[ri]?.[ci]?.image_name ?? ""}
              />
            </div>
          ));

          return [
            ...removeRowHdrBtns,
            ...(removeRowLabelBtn ? [removeRowLabelBtn] : []),
            ...hdrCells,
            rowLabel,
            ...cells,
          ];
        })}
      </div>

      {/* ── Panel Parking Drawer ────────────────────── */}
      <DrawerStrip
        cols={cols}
        drawerPanels={drawerPanels}
        loadedImages={loadedImages}
        movePanelToDrawer={movePanelToDrawer}
        movePanelFromDrawer={movePanelFromDrawer}
      />

      {/* ── Context Menu for Header Groups ──────────── */}
      <Menu
        open={ctxMenu !== null && ctxMenu.type === "header"}
        onClose={closeCtxMenu}
        anchorReference="anchorPosition"
        anchorPosition={ctxMenu ? { top: ctxMenu.mouseY, left: ctxMenu.mouseX } : undefined}
      >
        {ctxMenu?.axis === "col" ? (
          [
            <MenuItem key="ext-right" onClick={() => { extendHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx, "right"); closeCtxMenu(); }}>
              <ListItemIcon><ArrowForwardIcon fontSize="small" /></ListItemIcon>
              <ListItemText>Extend right</ListItemText>
            </MenuItem>,
            <MenuItem key="ext-left" onClick={() => { extendHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx, "left"); closeCtxMenu(); }}>
              <ListItemIcon><ArrowBackIcon fontSize="small" /></ListItemIcon>
              <ListItemText>Extend left</ListItemText>
            </MenuItem>,
          ]
        ) : ctxMenu?.axis === "row" ? (
          [
            <MenuItem key="ext-down" onClick={() => { extendHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx, "down"); closeCtxMenu(); }}>
              <ListItemIcon><ArrowDownwardIcon fontSize="small" /></ListItemIcon>
              <ListItemText>Extend down</ListItemText>
            </MenuItem>,
            <MenuItem key="ext-up" onClick={() => { extendHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx, "up"); closeCtxMenu(); }}>
              <ListItemIcon><ArrowUpwardIcon fontSize="small" /></ListItemIcon>
              <ListItemText>Extend up</ListItemText>
            </MenuItem>,
          ]
        ) : null}
        <Divider />
        <MenuItem onClick={() => { if (ctxMenu) openSpanDialog(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx); closeCtxMenu(); }}>
          <ListItemIcon><ViewColumnIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Select columns to span...</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { if (ctxMenu) openHeaderPropsDialog(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx); closeCtxMenu(); }}>
          <ListItemIcon><TuneIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Edit properties...</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => { if (ctxMenu) removeHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx); closeCtxMenu(); }}>
          <ListItemIcon><DeleteIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Remove header</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { if (ctxMenu) splitHeaderGroup(ctxMenu.axis, ctxMenu.level, ctxMenu.groupIdx); closeCtxMenu(); }}>
          <ListItemIcon><CallSplitIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Split into individual headers</ListItemText>
        </MenuItem>
      </Menu>

      {/* ── Context Menu for Empty Header Cells ─────── */}
      <Menu
        open={ctxMenu !== null && ctxMenu.type === "empty"}
        onClose={closeCtxMenu}
        anchorReference="anchorPosition"
        anchorPosition={ctxMenu ? { top: ctxMenu.mouseY, left: ctxMenu.mouseX } : undefined}
      >
        <MenuItem onClick={() => {
          if (ctxMenu && ctxMenu.cellIndex !== undefined) {
            createHeaderGroupAt(ctxMenu.axis, ctxMenu.level, ctxMenu.cellIndex);
          }
          closeCtxMenu();
        }}>
          <ListItemIcon><AddIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Add header here</ListItemText>
        </MenuItem>
      </Menu>

      {/* ── Span Selection Dialog ────────────────────── */}
      <Dialog open={spanDialog !== null} onClose={() => setSpanDialog(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Select {spanDialog?.axis === "col" ? "columns" : "rows"} to span</DialogTitle>
        <DialogContent>
          {spanDialog && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
              {getAvailableIndicesForSpan().map((idx) => (
                <FormControlLabel
                  key={idx}
                  control={
                    <Checkbox
                      checked={spanDialog.selectedIndices.includes(idx)}
                      onChange={(e) => {
                        setSpanDialog((prev) => {
                          if (!prev) return prev;
                          const sel = e.target.checked
                            ? [...prev.selectedIndices, idx]
                            : prev.selectedIndices.filter((i) => i !== idx);
                          return { ...prev, selectedIndices: sel };
                        });
                      }}
                    />
                  }
                  label={`${spanDialog.axis === "col" ? "Column" : "Row"} ${idx + 1}`}
                />
              ))}
              {getAvailableIndicesForSpan().length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No available {spanDialog.axis === "col" ? "columns" : "rows"} (all occupied by other groups)
                </Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSpanDialog(null)}>Cancel</Button>
          <Button variant="contained" onClick={applySpanDialog}>Apply</Button>
        </DialogActions>
      </Dialog>

      {/* ── Header Properties Dialog ─────────────────── */}
      <Dialog open={headerPropsDialog !== null} onClose={() => setHeaderPropsDialog(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Header Properties</DialogTitle>
        <DialogContent>
          {headerPropsDialog && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 1 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Offset from previous header: {(headerPropsDialog.distance / 10).toFixed(1)}%
                </Typography>
                <Slider
                  value={headerPropsDialog.distance}
                  min={0}
                  max={100}
                  onChange={(_, v) => setHeaderPropsDialog((prev) => prev ? { ...prev, distance: v as number } : prev)}
                />
              </Box>

              <FormControl fullWidth size="small">
                <InputLabel>Position</InputLabel>
                <Select
                  value={headerPropsDialog.position}
                  label="Position"
                  onChange={(e) => setHeaderPropsDialog((prev) => prev ? { ...prev, position: e.target.value } : prev)}
                >
                  {headerPropsDialog.axis === "col" ? (
                    [
                      <MenuItem key="Top" value="Top">Top</MenuItem>,
                      <MenuItem key="Bottom" value="Bottom">Bottom</MenuItem>,
                    ]
                  ) : (
                    [
                      <MenuItem key="Left" value="Left">Left</MenuItem>,
                      <MenuItem key="Right" value="Right">Right</MenuItem>,
                    ]
                  )}
                </Select>
              </FormControl>

              {/* Line controls */}
              <Typography variant="caption" sx={{ fontWeight: 600 }}>Line</Typography>

              <FormControlLabel
                sx={{ ml: 0 }}
                control={
                  <Switch
                    size="small"
                    checked={headerPropsDialog.showLine}
                    onChange={(_e, checked) => setHeaderPropsDialog((prev) => prev ? { ...prev, showLine: checked } : prev)}
                  />
                }
                label={<Typography variant="caption">Show line</Typography>}
              />

              {headerPropsDialog.showLine && (
                <>
                  <TextField
                    type="color"
                    label="Line color"
                    value={headerPropsDialog.lineColor}
                    onChange={(e) => setHeaderPropsDialog((prev) => prev ? { ...prev, lineColor: e.target.value } : prev)}
                    fullWidth
                    size="small"
                    sx={{ "& input": { cursor: "pointer" } }}
                  />

                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Line width: {headerPropsDialog.lineWidth.toFixed(1)}
                    </Typography>
                    <Slider
                      value={headerPropsDialog.lineWidth}
                      min={0.5}
                      max={5}
                      step={0.5}
                      onChange={(_, v) => setHeaderPropsDialog((prev) => prev ? { ...prev, lineWidth: v as number } : prev)}
                    />
                  </Box>

                  <FormControl fullWidth size="small">
                    <InputLabel>Line style</InputLabel>
                    <Select
                      value={headerPropsDialog.lineStyle}
                      label="Line style"
                      onChange={(e) => setHeaderPropsDialog((prev) => prev ? { ...prev, lineStyle: e.target.value } : prev)}
                    >
                      <MenuItem value="solid">Solid</MenuItem>
                      <MenuItem value="dashed">Dashed</MenuItem>
                      <MenuItem value="dotted">Dotted</MenuItem>
                      <MenuItem value="dash-dot">Dash-dot</MenuItem>
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Line length: {Math.round(headerPropsDialog.lineLength * 100)}%
                    </Typography>
                    <Slider
                      value={headerPropsDialog.lineLength}
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      onChange={(_, v) => setHeaderPropsDialog((prev) => prev ? { ...prev, lineLength: v as number } : prev)}
                    />
                  </Box>

                  <FormControlLabel
                    sx={{ ml: 0 }}
                    control={
                      <Switch
                        size="small"
                        checked={headerPropsDialog.endCaps}
                        onChange={(_e, checked) => setHeaderPropsDialog((prev) => prev ? { ...prev, endCaps: checked } : prev)}
                      />
                    }
                    label={<Typography variant="caption">End caps (toward previous header)</Typography>}
                  />
                </>
              )}

              {/* ── Per-character styling (styled segments) ───────── */}
              <Typography variant="caption" sx={{ fontWeight: 600, mt: 1 }}>Per-character styling</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.65rem" }}>
                Split the header into segments so parts like "DAPI" and "GFP" in "DAPI/GFP" can each take their own colour / font / size.
                Leave empty to render the whole header in the default colour.
              </Typography>

              <TextField
                label="Default colour"
                type="color"
                value={headerPropsDialog.defaultColor}
                onChange={(e) => setHeaderPropsDialog((prev) => prev ? { ...prev, defaultColor: e.target.value } : prev)}
                size="small"
                sx={{ "& input": { cursor: "pointer" } }}
              />

              {headerPropsDialog.styledSegments.map((seg, i) => (
                <Box key={i} sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                  <TextField
                    label={`#${i + 1}`}
                    value={seg.text}
                    size="small"
                    onChange={(e) => setHeaderPropsDialog((prev) => {
                      if (!prev) return prev;
                      const segs = [...prev.styledSegments];
                      segs[i] = { ...segs[i], text: e.target.value };
                      return { ...prev, styledSegments: segs };
                    })}
                    sx={{ flex: 1, "& input": { fontSize: "0.7rem" } }}
                  />
                  <TextField
                    type="color"
                    value={seg.color}
                    size="small"
                    onChange={(e) => setHeaderPropsDialog((prev) => {
                      if (!prev) return prev;
                      const segs = [...prev.styledSegments];
                      segs[i] = { ...segs[i], color: e.target.value };
                      return { ...prev, styledSegments: segs };
                    })}
                    sx={{ width: 48, "& input": { cursor: "pointer", p: 0.25 } }}
                  />
                  <TextField
                    type="number"
                    value={seg.font_size ?? ""}
                    placeholder="sz"
                    size="small"
                    onChange={(e) => setHeaderPropsDialog((prev) => {
                      if (!prev) return prev;
                      const segs = [...prev.styledSegments];
                      const v = e.target.value === "" ? undefined : Number(e.target.value);
                      segs[i] = { ...segs[i], font_size: v };
                      return { ...prev, styledSegments: segs };
                    })}
                    inputProps={{ min: 4, max: 200, step: 1 }}
                    sx={{ width: 52, "& input": { fontSize: "0.65rem", textAlign: "center" } }}
                  />
                  <IconButton size="small" title="Remove segment" onClick={() => setHeaderPropsDialog((prev) => {
                    if (!prev) return prev;
                    const segs = prev.styledSegments.filter((_, j) => j !== i);
                    return { ...prev, styledSegments: segs };
                  })}>
                    <DeleteIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Box>
              ))}

              <Box sx={{ display: "flex", gap: 0.5 }}>
                <Button size="small" variant="outlined" startIcon={<AddIcon />} onClick={() => setHeaderPropsDialog((prev) => {
                  if (!prev) return prev;
                  const defColor = prev.defaultColor || "#000000";
                  // First segment prepopulates with the full current header text
                  // so the user can split it immediately.
                  const initial: HeaderStyledSegment = prev.styledSegments.length === 0
                    ? { text: prev.headerText || "", color: defColor }
                    : { text: "", color: defColor };
                  return { ...prev, styledSegments: [...prev.styledSegments, initial] };
                })}>
                  Add segment
                </Button>
                {headerPropsDialog.styledSegments.length > 0 && (
                  <Button size="small" onClick={() => setHeaderPropsDialog((prev) => prev ? { ...prev, styledSegments: [] } : prev)}>
                    Clear
                  </Button>
                )}
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHeaderPropsDialog(null)}>Cancel</Button>
          <Button variant="contained" onClick={applyHeaderPropsDialog}>Apply</Button>
        </DialogActions>
      </Dialog>

      {/* ── Label Properties Dialog ────────────────────── */}
      <Dialog open={labelPropsDialog !== null} onClose={() => setLabelPropsDialog(null)} maxWidth="xs" fullWidth>
        <DialogTitle>
          {labelPropsDialog?.axis === "col" ? "Column" : "Row"} Label Properties
        </DialogTitle>
        <DialogContent>
          {labelPropsDialog && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 1 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Offset from panel edge: {labelPropsDialog.distance.toFixed(1)}%
                </Typography>
                <Slider
                  value={labelPropsDialog.distance}
                  min={0}
                  max={10}
                  step={0.1}
                  onChange={(_, v) => setLabelPropsDialog((prev) => prev ? { ...prev, distance: v as number } : prev)}
                />
              </Box>

              <FormControl fullWidth size="small">
                <InputLabel>Position</InputLabel>
                <Select
                  value={labelPropsDialog.position}
                  label="Position"
                  onChange={(e) => setLabelPropsDialog((prev) => prev ? { ...prev, position: e.target.value } : prev)}
                >
                  {labelPropsDialog.axis === "col" ? (
                    [
                      <MenuItem key="Top" value="Top">Top</MenuItem>,
                      <MenuItem key="Bottom" value="Bottom">Bottom</MenuItem>,
                    ]
                  ) : (
                    [
                      <MenuItem key="Left" value="Left">Left</MenuItem>,
                      <MenuItem key="Right" value="Right">Right</MenuItem>,
                    ]
                  )}
                </Select>
              </FormControl>

              <Box>
                <Typography variant="caption" color="text.secondary">
                  Rotation: {labelPropsDialog.rotation}°
                </Typography>
                <Slider
                  value={labelPropsDialog.rotation}
                  min={0}
                  max={360}
                  step={5}
                  marks={[
                    { value: 0, label: "0°" },
                    { value: 90, label: "90°" },
                    { value: 180, label: "180°" },
                    { value: 270, label: "270°" },
                  ]}
                  onChange={(_, v) => setLabelPropsDialog((prev) => prev ? { ...prev, rotation: v as number } : prev)}
                />
              </Box>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={labelPropsDialog.applyToAll}
                    onChange={(e) => setLabelPropsDialog((prev) => prev ? { ...prev, applyToAll: e.target.checked } : prev)}
                    size="small"
                  />
                }
                label={
                  <Typography variant="caption">
                    Sync offset across all row &amp; column labels
                  </Typography>
                }
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLabelPropsDialog(null)}>Cancel</Button>
          <Button variant="contained" onClick={applyLabelPropsDialog}>Apply</Button>
        </DialogActions>
      </Dialog>

      {/* ── Floating Formatting Toolbar ──────────────── */}
      <FloatingToolbar
        anchorEl={toolbarAnchor}
        open={toolbarAnchor !== null && toolbarTarget !== null}
        onClose={closeToolbar}
        text={toolbarData.text}
        fontSize={toolbarData.fontSize}
        fontName={toolbarData.fontName}
        fontStyle={toolbarData.fontStyle}
        color={toolbarData.color}
        fonts={fonts}
        onTextChange={handleToolbarTextChange}
        onFontSizeChange={handleToolbarFontSizeChange}
        onFontNameChange={handleToolbarFontNameChange}
        onFontStyleToggle={handleToolbarFontStyleToggle}
        onColorChange={handleToolbarColorChange}
        onBeforeAction={() => {
          // Snapshot the currently-focused textarea's selection BEFORE
          // focus moves to a toolbar button, so resolveSelection() can
          // still see it when the handler fires. We read from
          // document.activeElement to cover cases where toolbarTextareaRef
          // wasn't set (e.g. user opened the toolbar via a previous focus
          // and focused another element before clicking the button).
          const ae = document.activeElement as HTMLTextAreaElement | null;
          const target = ae && ae.tagName === "TEXTAREA" ? ae : toolbarTextareaRef.current;
          if (target) {
            toolbarTextareaRef.current = target;
            const s = target.selectionStart ?? 0;
            const en = target.selectionEnd ?? 0;
            if (s !== en) {
              toolbarSelectionRef.current = { start: s, end: en };
              updateSelectionPreview(target, s, en);
            }
          }
        }}
        selectionPreview={selectionPreview}
        onInsertLineBreak={() => {
          // Insert a newline at the current cursor of the field the
          // toolbar is anchored to. Works as a keyboard-independent
          // alternative to Shift+Enter, which doesn't reach the
          // editor in WKWebView for some users.
          if (!toolbarTarget) return;
          // Prefer the active contentEditable editor's selection. Fall
          // back to a textarea's selectionStart/End for any legacy
          // surface that hasn't been migrated yet.
          let start: number | null = null;
          let end: number | null = null;
          const ed = activeEditorRef.current;
          if (ed) {
            const sel = ed.getSelection();
            if (sel) { start = sel.start; end = sel.end; }
          }
          if (start === null) {
            const ta = toolbarTextareaRef.current as HTMLTextAreaElement | HTMLInputElement | null;
            if (ta) {
              start = ta.selectionStart ?? 0;
              end = ta.selectionEnd ?? start;
            }
          }
          if (start === null) return;
          let newCaret: number | undefined;
          if (toolbarTarget.type === "header" && toolbarTarget.level !== undefined && toolbarTarget.groupIdx !== undefined) {
            newCaret = insertLineBreakInHeader(
              toolbarTarget.axis,
              toolbarTarget.level,
              toolbarTarget.groupIdx,
              start,
              end ?? start,
            );
          } else if ((toolbarTarget.type === "colLabel" || toolbarTarget.type === "rowLabel") && toolbarTarget.index !== undefined) {
            newCaret = insertLineBreakInLabel(
              toolbarTarget.axis,
              toolbarTarget.index,
              start,
              end ?? start,
            );
          }
          if (newCaret === undefined) return;
          // Restore caret after the inserted newline.
          requestAnimationFrame(() => {
            if (ed) {
              ed.focus();
              ed.setSelection(newCaret!, newCaret!);
              return;
            }
            const ta = toolbarTextareaRef.current as HTMLTextAreaElement | HTMLInputElement | null;
            if (ta) {
              try {
                (ta as HTMLTextAreaElement).focus();
                (ta as HTMLTextAreaElement).setSelectionRange(newCaret!, newCaret!);
              } catch { /* ignore */ }
            }
          });
        }}
      />
    </div>
  );
}
