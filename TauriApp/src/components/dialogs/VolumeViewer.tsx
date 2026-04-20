/* ──────────────────────────────────────────────────────────
   VolumeViewer — Server-side 3D volume rendering.
   Fast drag preview + high-res on release.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress, ToggleButtonGroup, ToggleButton,
  FormControlLabel, Checkbox, TextField,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import ImageIcon from "@mui/icons-material/Image";
import { api } from "../../api/client";
import { useFigureStore } from "../../store/figureStore";

interface Props {
  open: boolean;
  onClose: () => void;
  imageName: string;
  startFrame: number;
  endFrame: number;
  panelRow?: number;
  panelCol?: number;
}

export function VolumeViewerDialog({ open, onClose, imageName, startFrame, endFrame, panelRow, panelCol }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageSrc, setImageSrc] = useState("");
  const fetchImages = useFigureStore((s) => s.fetchImages);

  // View controls
  const [elev, setElev] = useState(30);
  const [azim, setAzim] = useState(-60);
  const [threshold, setThreshold] = useState(0.3);
  const [zSpacing, setZSpacing] = useState(1.0);
  const [colormap, setColormap] = useState("gray");
  const [method, setMethod] = useState("surface");
  const [showAxes, setShowAxes] = useState(true);
  const [zoom, setZoom] = useState(1.0);

  // Save dialog state
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveFormat, setSaveFormat] = useState("PNG");
  const [saveQuality, setSaveQuality] = useState(95);
  const [savePath, setSavePath] = useState("");
  const [saving, setSaving] = useState(false);

  // Drag state
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0, elev: 0, azim: 0 });
  const pendingHiRes = useRef<ReturnType<typeof setTimeout> | null>(null);

  const renderView = useCallback(async (opts?: { e?: number; a?: number; z?: number; fast?: boolean }) => {
    if (!opts?.fast) setLoading(true);
    setError("");
    try {
      const result = await api.renderVolume(imageName, {
        startFrame, endFrame,
        elev: opts?.e ?? elev,
        azim: opts?.a ?? azim,
        threshold, zSpacing, colormap,
        width: 900, height: 700, method,
        showAxes, zoom: opts?.z ?? zoom,
        fast: opts?.fast ?? false,
      });
      setImageSrc(`data:image/png;base64,${result.image}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setLoading(false);
  }, [imageName, startFrame, endFrame, elev, azim, threshold, zSpacing, colormap, method, showAxes, zoom]);

  // Initial render
  useEffect(() => {
    if (open) renderView();
  }, [open]); // eslint-disable-line

  // Re-render when settings change (debounced)
  const renderTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!open || !imageSrc) return;
    if (renderTimer.current) clearTimeout(renderTimer.current);
    renderTimer.current = setTimeout(() => renderView(), 200);
    return () => { if (renderTimer.current) clearTimeout(renderTimer.current); };
  }, [threshold, zSpacing, colormap, method, showAxes]); // eslint-disable-line

  // Mouse drag — fast preview during drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY, elev, azim };
  }, [elev, azim]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    const newAzim = dragStart.current.azim + dx * 0.5;
    const newElev = Math.max(-90, Math.min(90, dragStart.current.elev - dy * 0.5));
    setAzim(newAzim);
    setElev(newElev);
    // Throttle fast preview rendering
    if (pendingHiRes.current) clearTimeout(pendingHiRes.current);
    pendingHiRes.current = setTimeout(() => {
      renderView({ e: newElev, a: newAzim, fast: true });
    }, 30);
  }, [renderView]);

  const handleMouseUp = useCallback(() => {
    if (isDragging.current) {
      isDragging.current = false;
      if (pendingHiRes.current) clearTimeout(pendingHiRes.current);
      renderView(); // high-res final render
    }
  }, [renderView]);

  // Scroll wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.2, Math.min(10, zoom * delta));
    setZoom(newZoom);
    if (pendingHiRes.current) clearTimeout(pendingHiRes.current);
    // Fast render immediately, high-res after pause
    renderView({ z: newZoom, fast: true });
    pendingHiRes.current = setTimeout(() => {
      renderView({ z: newZoom });
    }, 200);
  }, [zoom, renderView]);

  const useAsPanel = async () => {
    if (panelRow == null || panelCol == null) return;
    try {
      await api.useVolumeAsPanel(imageName, panelRow, panelCol, {
        startFrame, endFrame, elev, azim, threshold, zSpacing, colormap,
        method, showAxes, zoom,
        width: 1600, height: 1200,
      });
      await fetchImages();
      onClose();
    } catch (e) {
      setError("Failed to set as panel image: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const openSaveDialog = async () => {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const ext = saveFormat === "TIFF" ? "tiff" : saveFormat.toLowerCase();
      const path = await save({
        defaultPath: `volume_${imageName.replace(/\.\w+$/, "")}.${ext}`,
        filters: [{ name: saveFormat, extensions: [ext] }],
      });
      if (path) {
        setSavePath(path as string);
        setSaveOpen(true);
      }
    } catch {
      // Fallback: just open the format dialog
      setSaveOpen(true);
    }
  };

  const performSave = async () => {
    if (!savePath) return;
    setSaving(true);
    try {
      await api.saveVolumeRenderAsImage(imageName, {
        startFrame, endFrame, elev, azim, threshold, zSpacing, colormap,
        method, showAxes, zoom,
        width: 1600, height: 1200,
        format: saveFormat, quality: saveQuality, filePath: savePath,
      });
      setSaveOpen(false);
    } catch (e) {
      setError("Save failed: " + (e instanceof Error ? e.message : String(e)));
    }
    setSaving(false);
  };

  return (
    <>
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
          onWheel={handleWheel}
        >
          {imageSrc && (
            <img src={imageSrc} alt="Volume render" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", userSelect: "none" }} draggable={false} />
          )}
          {loading && (
            <Box sx={{ position: "absolute", top: 16, right: 16, bgcolor: "background.paper", px: 1.5, py: 0.5, borderRadius: 1, display: "flex", alignItems: "center", gap: 1 }}>
              <CircularProgress size={14} />
              <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>Rendering...</Typography>
            </Box>
          )}
          {error && !loading && (
            <Box sx={{ position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)", bgcolor: "error.dark", color: "white", px: 2, py: 1, borderRadius: 1 }}>
              <Typography variant="caption">{error}</Typography>
            </Box>
          )}
        </Box>

        {/* Controls */}
        <Box sx={{ width: 240, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 1.5, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Controls</Typography>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem", mb: 0.5, display: "block" }}>View Mode</Typography>
            <ToggleButtonGroup value={method} exclusive onChange={(_, v) => { if (v) setMethod(v); }} size="small" sx={{ flexWrap: "wrap" }}>
              <ToggleButton value="surface" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>3D</ToggleButton>
              <ToggleButton value="mip_xy" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP XY</ToggleButton>
              <ToggleButton value="mip_xz" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP XZ</ToggleButton>
              <ToggleButton value="mip_yz" sx={{ fontSize: "0.55rem", px: 1, py: 0.25 }}>MIP YZ</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          <FormControlLabel
            sx={{ ml: 0 }}
            control={<Checkbox size="small" checked={showAxes} onChange={(e) => setShowAxes(e.target.checked)} sx={{ p: 0.25 }} />}
            label={<Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Show axes & grid</Typography>}
          />

          {method === "surface" && (
            <Box>
              <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Threshold: {threshold.toFixed(2)}</Typography>
              <Slider size="small" value={threshold} min={0} max={1} step={0.01}
                onChange={(_, v) => setThreshold(v as number)} />
            </Box>
          )}

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Z Spacing: {zSpacing.toFixed(1)}</Typography>
            <Slider size="small" value={zSpacing} min={0.1} max={5} step={0.1}
              onChange={(_, v) => setZSpacing(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Zoom: {zoom.toFixed(1)}×</Typography>
            <Slider size="small" value={zoom} min={0.2} max={10} step={0.1}
              onChange={(_, v) => setZoom(v as number)}
              onChangeCommitted={() => renderView()} />
          </Box>

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

          <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary" }}>
            Drag to rotate. Scroll to zoom.
          </Typography>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
            <Button size="small" variant="outlined" onClick={openSaveDialog} startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
              sx={{ fontSize: "0.6rem", textTransform: "none" }}>
              Save View...
            </Button>
            {panelRow != null && panelCol != null && (
              <Button size="small" variant="contained" color="primary" onClick={useAsPanel} startIcon={<ImageIcon sx={{ fontSize: 12 }} />}
                sx={{ fontSize: "0.6rem", textTransform: "none" }}>
                Use as Panel Image
              </Button>
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>

    {/* Save dialog */}
    <Dialog open={saveOpen} onClose={() => setSaveOpen(false)} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem" }}>Save Volume Render</DialogTitle>
      <DialogContent>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
          <TextField
            label="File path" size="small" fullWidth value={savePath}
            onChange={(e) => setSavePath(e.target.value)}
          />
          <Box>
            <Typography variant="caption">Format</Typography>
            <ToggleButtonGroup value={saveFormat} exclusive onChange={(_, v) => { if (v) setSaveFormat(v); }} size="small" sx={{ mt: 0.5 }}>
              <ToggleButton value="PNG">PNG</ToggleButton>
              <ToggleButton value="TIFF">TIFF</ToggleButton>
              <ToggleButton value="JPEG">JPEG</ToggleButton>
            </ToggleButtonGroup>
          </Box>
          {saveFormat === "JPEG" && (
            <Box>
              <Typography variant="caption">Quality: {saveQuality}</Typography>
              <Slider size="small" value={saveQuality} min={1} max={100} step={1}
                onChange={(_, v) => setSaveQuality(v as number)} />
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setSaveOpen(false)}>Cancel</Button>
        <Button variant="contained" onClick={performSave} disabled={saving || !savePath}>
          {saving ? "Saving..." : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
    </>
  );
}
