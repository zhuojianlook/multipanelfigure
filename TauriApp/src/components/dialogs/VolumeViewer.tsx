/* ──────────────────────────────────────────────────────────
   VolumeViewer — Server-side 3D volume rendering.
   Python renders the view, frontend displays and controls.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Dialog, DialogTitle, DialogContent, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress, ToggleButtonGroup, ToggleButton,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import { api } from "../../api/client";

interface Props {
  open: boolean;
  onClose: () => void;
  imageName: string;
  startFrame: number;
  endFrame: number;
}

export function VolumeViewerDialog({ open, onClose, imageName, startFrame, endFrame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageSrc, setImageSrc] = useState("");

  // View controls
  const [elev, setElev] = useState(30);
  const [azim, setAzim] = useState(-60);
  const [threshold, setThreshold] = useState(0.3);
  const [zSpacing, setZSpacing] = useState(1.0);
  const [colormap, setColormap] = useState("gray");
  const [method, setMethod] = useState("surface");

  // Drag state for rotation
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, elev: 0, azim: 0 });

  const renderView = useCallback(async (opts?: { e?: number; a?: number }) => {
    setLoading(true);
    setError("");
    try {
      const result = await api.renderVolume(imageName, {
        startFrame,
        endFrame,
        elev: opts?.e ?? elev,
        azim: opts?.a ?? azim,
        threshold,
        zSpacing,
        colormap,
        width: 900,
        height: 700,
        method,
      });
      setImageSrc(`data:image/png;base64,${result.image}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setLoading(false);
  }, [imageName, startFrame, endFrame, elev, azim, threshold, zSpacing, colormap, method]);

  // Initial render
  useEffect(() => {
    if (open) renderView();
  }, [open]); // eslint-disable-line

  // Re-render when controls change (debounced)
  const renderTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!open || !imageSrc) return; // skip initial
    if (renderTimer.current) clearTimeout(renderTimer.current);
    renderTimer.current = setTimeout(() => renderView(), 300);
    return () => { if (renderTimer.current) clearTimeout(renderTimer.current); };
  }, [threshold, zSpacing, colormap, method]); // eslint-disable-line

  // Mouse drag for rotation
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY, elev, azim };
  }, [elev, azim]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    setAzim(dragStart.current.azim + dx * 0.5);
    setElev(Math.max(-90, Math.min(90, dragStart.current.elev - dy * 0.5)));
  }, []);

  const handleMouseUp = useCallback(() => {
    if (isDragging.current) {
      isDragging.current = false;
      renderView();
    }
  }, [renderView]);

  const saveView = () => {
    if (!imageSrc) return;
    const link = document.createElement("a");
    link.href = imageSrc;
    link.download = `volume_${imageName.replace(/\.\w+$/, "")}_${method}.png`;
    link.click();
  };

  return (
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>3D Volume View — {imageName}</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", height: "100%", overflow: "hidden" }}>
        {/* Rendered image */}
        <Box
          sx={{ flex: 1, position: "relative", cursor: isDragging.current ? "grabbing" : "grab", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "#1c1c1e" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {imageSrc && (
            <img src={imageSrc} alt="Volume render" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", userSelect: "none" }} draggable={false} />
          )}
          {loading && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.4)" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, bgcolor: "background.paper", px: 2, py: 1, borderRadius: 1 }}>
                <CircularProgress size={16} />
                <Typography variant="caption">Rendering...</Typography>
              </Box>
            </Box>
          )}
          {error && !loading && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Typography color="error" variant="caption">{error}</Typography>
            </Box>
          )}
        </Box>

        {/* Controls */}
        <Box sx={{ width: 240, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 2, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Controls</Typography>

          {/* View mode */}
          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem", mb: 0.5, display: "block" }}>View Mode</Typography>
            <ToggleButtonGroup
              value={method}
              exclusive
              onChange={(_, v) => { if (v) setMethod(v); }}
              size="small"
              sx={{ flexWrap: "wrap" }}
            >
              <ToggleButton value="surface" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>3D</ToggleButton>
              <ToggleButton value="mip_xy" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP XY</ToggleButton>
              <ToggleButton value="mip_xz" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP XZ</ToggleButton>
              <ToggleButton value="mip_yz" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP YZ</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* Threshold (3D only) */}
          {method === "surface" && (
            <Box>
              <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Threshold: {threshold.toFixed(2)}</Typography>
              <Slider size="small" value={threshold} min={0} max={1} step={0.01}
                onChange={(_, v) => setThreshold(v as number)} />
            </Box>
          )}

          {/* Z Spacing */}
          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Z Spacing: {zSpacing.toFixed(1)}</Typography>
            <Slider size="small" value={zSpacing} min={0.1} max={5} step={0.1}
              onChange={(_, v) => setZSpacing(v as number)} />
          </Box>

          {/* Colormap */}
          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Colormap</Typography>
            <Select size="small" value={colormap} onChange={(e) => setColormap(e.target.value)}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}>
              <MenuItem value="gray" sx={{ fontSize: "0.65rem" }}>Grayscale</MenuItem>
              <MenuItem value="hot" sx={{ fontSize: "0.65rem" }}>Hot</MenuItem>
              <MenuItem value="cool" sx={{ fontSize: "0.65rem" }}>Cool</MenuItem>
              <MenuItem value="viridis" sx={{ fontSize: "0.65rem" }}>Viridis</MenuItem>
              <MenuItem value="magma" sx={{ fontSize: "0.65rem" }}>Magma</MenuItem>
              <MenuItem value="inferno" sx={{ fontSize: "0.65rem" }}>Inferno</MenuItem>
              <MenuItem value="plasma" sx={{ fontSize: "0.65rem" }}>Plasma</MenuItem>
            </Select>
          </Box>

          {/* Camera (3D only) */}
          {method === "surface" && (
            <>
              <Box>
                <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Elevation: {elev.toFixed(0)}°</Typography>
                <Slider size="small" value={elev} min={-90} max={90} step={1}
                  onChange={(_, v) => setElev(v as number)}
                  onChangeCommitted={() => renderView()} />
              </Box>
              <Box>
                <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Azimuth: {azim.toFixed(0)}°</Typography>
                <Slider size="small" value={azim} min={-180} max={180} step={1}
                  onChange={(_, v) => setAzim(v as number)}
                  onChangeCommitted={() => renderView()} />
              </Box>
            </>
          )}

          <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary" }}>
            {method === "surface" ? "Drag image to rotate. Adjust controls and rendering updates automatically." : "MIP views update when controls change."}
          </Typography>

          <Button size="small" variant="outlined" onClick={saveView} startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
            sx={{ fontSize: "0.6rem", textTransform: "none" }}>
            Save View as PNG
          </Button>

          <Button size="small" variant="contained" onClick={() => renderView()} disabled={loading}
            sx={{ fontSize: "0.6rem", textTransform: "none" }}>
            {loading ? "Rendering..." : "Re-render"}
          </Button>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
