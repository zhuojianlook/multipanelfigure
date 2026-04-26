/* ──────────────────────────────────────────────────────────
   CollageView — main canvas for the Collage Assembly
   workspace. Free-form positioning of items (rendered figures
   or imported images) via click-and-drag.
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
} from "@mui/material";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import DeleteForeverIcon from "@mui/icons-material/DeleteForever";
import VerticalAlignTopIcon from "@mui/icons-material/VerticalAlignTop";
import DeleteIcon from "@mui/icons-material/Delete";
import { useCollageStore } from "../../store/collageStore";

export function CollageView() {
  const items = useCollageStore((s) => s.items);
  const canvasW = useCollageStore((s) => s.canvasW);
  const canvasH = useCollageStore((s) => s.canvasH);
  const background = useCollageStore((s) => s.background);
  const selectedId = useCollageStore((s) => s.selectedId);
  const setSelectedId = useCollageStore((s) => s.setSelectedId);
  const updateItem = useCollageStore((s) => s.updateItem);
  const moveItem = useCollageStore((s) => s.moveItem);
  const removeItem = useCollageStore((s) => s.removeItem);
  const bringToFront = useCollageStore((s) => s.bringToFront);
  const setCanvasSize = useCollageStore((s) => s.setCanvasSize);
  const setBackground = useCollageStore((s) => s.setBackground);
  const clear = useCollageStore((s) => s.clear);
  const addItem = useCollageStore((s) => s.addItem);

  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fitScale, setFitScale] = useState(1);

  // Recompute fit-scale whenever the container or canvas dimensions change.
  // The canvas is rendered at its logical size, then transform-scaled so it
  // fits inside the available pane with a small margin.
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

  const onPickImageFile = async (file: File) => {
    // Read file → data URL → measure natural dims via Image() → push to store.
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
        // Stagger newly-added items so they don't all land on top of each
        // other — count of existing items × 24 px offset.
        const offset = items.length * 24;
        addItem({
          src,
          name: file.name,
          x: 40 + offset,
          y: 40 + offset,
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
    // Reset so the same file can be re-selected after removal.
    if (fileInputRef.current) fileInputRef.current.value = "";
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
        <Tooltip title="Import image (PNG, JPEG, TIFF, etc.) into the collage">
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

        <Divider orientation="vertical" flexItem />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>Canvas</Typography>
        <TextField
          type="number"
          size="small"
          value={canvasW}
          onChange={(e) => setCanvasSize(Math.max(100, Number(e.target.value) || 100), canvasH)}
          inputProps={{ min: 100, max: 8000, step: 100 }}
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
          inputProps={{ min: 100, max: 8000, step: 100 }}
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

        <Box sx={{ flex: 1 }} />

        <Typography variant="caption" sx={{ color: "text.secondary" }}>
          {items.length} item{items.length === 1 ? "" : "s"} · zoom {(fitScale * 100).toFixed(0)}%
        </Typography>

        <Tooltip title="Remove all items">
          <span>
            <IconButton
              size="small"
              disabled={items.length === 0}
              onClick={() => {
                if (window.confirm(`Remove all ${items.length} item(s) from the collage?`)) clear();
              }}
            >
              <DeleteForeverIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>

      {/* ── Canvas viewport ─────────────────────────────────── */}
      <Box
        ref={containerRef}
        onMouseDown={(e) => {
          // Clicking the empty viewport deselects.
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
        {/* The "page" (canvas). Rendered at logical pixel dimensions but
            transform-scaled to fit. Using transform keeps item child math
            simple — drag deltas are computed against the unscaled size and
            then divided by fitScale at the source. */}
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
                  // Use a running last-x/y so each mousemove only contributes
                  // its incremental delta (cumulative deltas would double-apply
                  // because moveItem mutates the item's x/y on every tick).
                  let lastX = e.clientX;
                  let lastY = e.clientY;
                  const onMove = (ev: MouseEvent) => {
                    const dx = (ev.clientX - lastX) / fitScale;
                    const dy = (ev.clientY - lastY) / fitScale;
                    lastX = ev.clientX;
                    lastY = ev.clientY;
                    moveItem(it.id, dx, dy);
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
                  /* Bottom-right resize handle — preserves aspect ratio while
                     dragging (user can hold Shift to ignore aspect — phase 2). */
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
                        // Pick the larger of the two deltas to keep aspect.
                        const tentativeW = startW + dx;
                        const tentativeH = startH + dy;
                        // Drive width by whichever delta is larger in proportion.
                        const widthDriven = Math.abs(dx) > Math.abs(dy);
                        const newW = widthDriven ? tentativeW : tentativeH * aspect;
                        const newH = widthDriven ? tentativeW / aspect : tentativeH;
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

      {/* ── Selected-item action bar ────────────────────────── */}
      {selectedId && (() => {
        const it = items.find((i) => i.id === selectedId);
        if (!it) return null;
        return (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1.5, py: 0.75, borderTop: "1px solid var(--c-border)", flexShrink: 0 }}>
            <Typography variant="caption" sx={{ color: "text.secondary", maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {it.name}
            </Typography>
            <Box sx={{ flex: 1 }} />
            <Typography variant="caption" sx={{ color: "text.secondary" }}>
              {Math.round(it.w)} × {Math.round(it.h)} px @ ({Math.round(it.x)}, {Math.round(it.y)})
            </Typography>
            <Tooltip title="Bring to front">
              <IconButton size="small" onClick={() => bringToFront(it.id)}>
                <VerticalAlignTopIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Remove from collage">
              <IconButton size="small" color="error" onClick={() => removeItem(it.id)}>
                <DeleteIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        );
      })()}
    </Box>
  );
}
