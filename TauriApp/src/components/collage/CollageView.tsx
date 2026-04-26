/* ──────────────────────────────────────────────────────────
   CollageView — main canvas for the Collage Assembly
   workspace. Free-form positioning of items (rendered figures
   or imported images) via click-and-drag, with optional
   snap-to-grid alignment.
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
import { useCollageStore } from "../../store/collageStore";
import { useFigureStore } from "../../store/figureStore";
import { CollageStrip } from "./CollageStrip";

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
  const [fitScale, setFitScale] = useState(1);

  // Recompute fit-scale on container/canvas resize so the page always fits.
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

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.z - b.z),
    [items],
  );

  /** Round v to the nearest grid step when snap is enabled. */
  const snap = (v: number) => (snapEnabled ? Math.round(v / gridStep) * gridStep : v);

  const onPickImageFile = async (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const src = String(reader.result);
      const img = new window.Image();
      img.onload = () => {
        // Default render size: fit within ~40% of the canvas while
        // preserving aspect ratio.
        const targetMax = Math.min(canvasW, canvasH) * 0.4;
        const aspect = img.naturalWidth / img.naturalHeight;
        const w = aspect >= 1 ? targetMax : targetMax * aspect;
        const h = aspect >= 1 ? targetMax / aspect : targetMax;
        // Stagger newly-added items so they don't all land on top.
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

  /** Import .mpf project: load it into the multi-panel builder, then
   *  switch to builder mode so the user can review and click "Add to
   *  Collage". This route reuses the existing project loader rather
   *  than introducing a stateless render endpoint. */
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
      // The backend's loadProject takes a server-side path, but in dev/
      // browser we have a File object. Write it through the file-upload
      // shim if available; otherwise fall back to picking via the OS
      // dialog inside Tauri.
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
        /* not in Tauri — fall through to alert */
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

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* ── Toolbar row ─────────────────────────────────────── */}
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        sx={{ px: 1.5, py: 1, borderBottom: "1px solid var(--c-border)", flexShrink: 0 }}
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

        <Box sx={{ flex: 1 }} />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>
          {items.length} item{items.length === 1 ? "" : "s"} · zoom {(fitScale * 100).toFixed(0)}%
        </Typography>
      </Stack>

      {/* ── Canvas viewport ─────────────────────────────────── */}
      <Box
        ref={containerRef}
        onMouseDown={(e) => {
          if (e.target === e.currentTarget) setSelectedId(null);
        }}
        sx={{
          flex: 1,
          minHeight: 0,
          overflow: "auto",
          position: "relative",
          backgroundColor: "var(--c-bg)",
          backgroundImage:
            "linear-gradient(45deg, rgba(255,255,255,0.04) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.04) 75%)," +
            "linear-gradient(45deg, rgba(255,255,255,0.04) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.04) 75%)",
          backgroundSize: "20px 20px",
          backgroundPosition: "0 0, 10px 10px",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "flex-start",
          p: 2,
        }}
      >
        {/* The "page" — virtual logical-pixel canvas, transform-scaled to
            fit the available area. */}
        <Box
          sx={{
            position: "relative",
            width: canvasW,
            height: canvasH,
            backgroundColor: background,
            transform: `scale(${fitScale})`,
            transformOrigin: "top left",
            boxShadow: "0 0 0 1px rgba(255,255,255,0.15), 0 8px 24px rgba(0,0,0,0.4)",
            flexShrink: 0,
            // Gridline overlay rendered as repeating linear-gradients.
            // Two overlays — vertical lines + horizontal lines — at gridStep
            // pixels with subtle opacity so they don't fight with the items.
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
                  // Track absolute x/y instead of incremental deltas so the
                  // snap-to-grid math stays correct (snapping deltas would
                  // accumulate jitter as the cursor moves).
                  const startMouseX = e.clientX;
                  const startMouseY = e.clientY;
                  const startX = it.x;
                  const startY = it.y;
                  const onMove = (ev: MouseEvent) => {
                    const dx = (ev.clientX - startMouseX) / fitScale;
                    const dy = (ev.clientY - startMouseY) / fitScale;
                    const targetX = snap(startX + dx);
                    const targetY = snap(startY + dy);
                    // moveItem expects deltas — compute the delta from the
                    // current position to the new snapped target.
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
                  <Box
                    onMouseDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      const startW = it.w;
                      const startH = it.h;
                      const startX = e.clientX;
                      const startY = e.clientY;
                      const aspect = it.naturalW / Math.max(1, it.naturalH);
                      const onMove = (ev: MouseEvent) => {
                        const dx = (ev.clientX - startX) / fitScale;
                        const dy = (ev.clientY - startY) / fitScale;
                        const tentativeW = startW + dx;
                        const tentativeH = startH + dy;
                        const widthDriven = Math.abs(dx) > Math.abs(dy);
                        let newW = widthDriven ? tentativeW : tentativeH * aspect;
                        let newH = widthDriven ? tentativeW / aspect : tentativeH;
                        // Snap the dimensions too so resized items keep
                        // gridline alignment.
                        newW = Math.max(20, snap(newW));
                        newH = Math.max(20, snap(newH));
                        if (newW > 20 && newH > 20) {
                          updateItem(it.id, { w: newW, h: newH });
                        }
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
                      right: -6,
                      bottom: -6,
                      width: 12,
                      height: 12,
                      backgroundColor: "#4FC3F7",
                      border: "1px solid #fff",
                      cursor: "se-resize",
                      borderRadius: 0.5,
                    }}
                  />
                )}
              </Box>
            );
          })}
        </Box>
      </Box>

      {/* ── Item strip (timeline-style) at the bottom ────────── */}
      <CollageStrip />
    </Box>
  );
}
