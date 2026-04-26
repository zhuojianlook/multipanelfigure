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
} from "@mui/material";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import GridOnIcon from "@mui/icons-material/GridOn";
import GridOffIcon from "@mui/icons-material/GridOff";
import StraightenIcon from "@mui/icons-material/Straighten";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import {
  useCollageStore,
  DEFAULT_CANVAS_W,
  DEFAULT_CANVAS_H,
} from "../../store/collageStore";
import { useFigureStore } from "../../store/figureStore";
import { CollageStrip } from "./CollageStrip";

type Corner = "nw" | "ne" | "sw" | "se";

export function CollageView() {
  const items = useCollageStore((s) => s.items);
  const canvasW = useCollageStore((s) => s.canvasW);
  const canvasH = useCollageStore((s) => s.canvasH);
  const background = useCollageStore((s) => s.background);
  const selectedId = useCollageStore((s) => s.selectedId);
  const gridVisible = useCollageStore((s) => s.gridVisible);
  const snapEnabled = useCollageStore((s) => s.snapEnabled);
  const gridStep = useCollageStore((s) => s.gridStep);
  const setSelectedId = useCollageStore((s) => s.setSelectedId);
  const updateItem = useCollageStore((s) => s.updateItem);
  const moveItem = useCollageStore((s) => s.moveItem);
  const bringToFront = useCollageStore((s) => s.bringToFront);
  const setCanvasSize = useCollageStore((s) => s.setCanvasSize);
  const setBackground = useCollageStore((s) => s.setBackground);
  const setGridVisible = useCollageStore((s) => s.setGridVisible);
  const setSnapEnabled = useCollageStore((s) => s.setSnapEnabled);
  const setGridStep = useCollageStore((s) => s.setGridStep);
  const addItem = useCollageStore((s) => s.addItem);
  const loadProject = useFigureStore((s) => s.loadProject);
  const setMode = useCollageStore((s) => s.setMode);

  const containerRef = useRef<HTMLDivElement>(null);
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
  const resetCanvasSize = () => {
    setCanvasSize(DEFAULT_CANVAS_W, DEFAULT_CANVAS_H);
    resetView();
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
      window.alert("Please choose a .mpf project file.");
      return;
    }
    const ok = window.confirm(
      `Load "${file.name}" into the Multi-Panel Builder?\n\n` +
      "Your current builder state will be replaced. Then you can review " +
      "the figure and click \"Add to Collage\" to insert it.",
    );
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
      window.alert(
        "Project import currently requires the desktop app's native file " +
        "dialog. Use Sidebar → Load Project from the Multi-Panel Builder, " +
        "then return here and click Add to Collage.",
      );
    } catch (err) {
      console.error(err);
      window.alert("Could not load project. Check the console for details.");
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

        <Divider orientation="vertical" flexItem />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>Canvas</Typography>
        <TextField
          type="number"
          size="small"
          value={canvasW}
          onChange={(e) => setCanvasSize(Math.max(100, Number(e.target.value) || 100), canvasH)}
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
          value={canvasH}
          onChange={(e) => setCanvasSize(canvasW, Math.max(100, Number(e.target.value) || 100))}
          inputProps={{ min: 100, max: 8000, step: 50 }}
          sx={{
            width: 84,
            "& input": { fontSize: "0.75rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
            "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
              filter: "invert(1)", opacity: 1,
            },
          }}
        />
        <Tooltip title={`Reset canvas to Nature page (${DEFAULT_CANVAS_W} × ${DEFAULT_CANVAS_H})`}>
          <IconButton size="small" onClick={resetCanvasSize}>
            <RestartAltIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Canvas background color">
          <Box
            component="input"
            type="color"
            value={background}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBackground(e.target.value)}
            sx={{ width: 28, height: 28, p: 0, border: "none", bgcolor: "transparent", cursor: "pointer" }}
          />
        </Tooltip>

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
          {/* The "page" — backgroundColor + (optional) gridlines. */}
          <Box
            sx={{
              position: "relative",
              width: canvasW,
              height: canvasH,
              backgroundColor: background,
              boxShadow: "0 0 0 1px rgba(255,255,255,0.15), 0 8px 24px rgba(0,0,0,0.4)",
              ...(gridVisible
                ? {
                    backgroundImage:
                      `linear-gradient(to right, rgba(0,0,0,0.10) 1px, transparent 1px),` +
                      `linear-gradient(to bottom, rgba(0,0,0,0.10) 1px, transparent 1px)`,
                    backgroundSize: `${gridStep}px ${gridStep}px, ${gridStep}px ${gridStep}px`,
                  }
                : {}),
            }}
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) setSelectedId(null);
            }}
          >
            {sortedItems.map((it) => {
              const isSelected = it.id === selectedId;
              return (
                <Box
                  key={it.id}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setSelectedId(it.id);
                    bringToFront(it.id);
                    const startMouseX = e.clientX;
                    const startMouseY = e.clientY;
                    const startX = it.x;
                    const startY = it.y;
                    const onMove = (ev: MouseEvent) => {
                      const dx = (ev.clientX - startMouseX) / displayScale;
                      const dy = (ev.clientY - startMouseY) / displayScale;
                      const targetX = snap(startX + dx);
                      const targetY = snap(startY + dy);
                      const cur = useCollageStore.getState().items.find((i) => i.id === it.id);
                      if (!cur) return;
                      moveItem(it.id, targetX - cur.x, targetY - cur.y);
                    };
                    const onUp = () => {
                      window.removeEventListener("mousemove", onMove);
                      window.removeEventListener("mouseup", onUp);
                    };
                    window.addEventListener("mousemove", onMove);
                    window.addEventListener("mouseup", onUp);
                  }}
                  sx={{
                    position: "absolute",
                    left: it.x,
                    top: it.y,
                    width: it.w,
                    height: it.h,
                    cursor: "grab",
                    outline: isSelected ? "2px solid #4FC3F7" : "1px solid rgba(255,255,255,0.0)",
                    outlineOffset: isSelected ? 2 : 0,
                    "&:hover": { outline: "2px solid rgba(79,195,247,0.6)" },
                  }}
                  title={it.name}
                >
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
                  {isSelected && (
                    <>
                      {(["nw", "ne", "sw", "se"] as const).map((c) => renderResizeHandle(it, c))}
                    </>
                  )}
                </Box>
              );
            })}
          </Box>
        </Box>
      </Box>

      <CollageStrip />
    </Box>
  );
}
