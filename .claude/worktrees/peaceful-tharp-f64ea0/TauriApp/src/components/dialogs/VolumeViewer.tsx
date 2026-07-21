/* ──────────────────────────────────────────────────────────
   VolumeViewer — NiiVue WebGL2 volume rendering.
   Fast, interactive, supports real-time rotation/zoom.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress, ToggleButtonGroup, ToggleButton,
  TextField, FormControlLabel, Checkbox,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import ImageIcon from "@mui/icons-material/Image";
import { Niivue, SLICE_TYPE } from "@niivue/niivue";
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
  /**
   * Fired after the 3D render is successfully applied to the panel.
   * Parent dialogs (e.g. EditPanelDialog) can listen to this to close
   * themselves so the user re-opens the editor on the updated image.
   */
  onAppliedToPanel?: () => void;
}

const COLORMAPS = ["gray", "hot", "cool", "viridis", "magma", "inferno", "plasma", "bone", "blues", "greens"];

export function VolumeViewerDialog({ open, onClose, imageName, startFrame, endFrame, panelRow, panelCol, onAppliedToPanel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Initializing...");
  const [error, setError] = useState("");
  const [sliceType, setSliceType] = useState<number>(SLICE_TYPE.RENDER);
  const [colormap, setColormap] = useState("gray");
  const [opacity, setOpacity] = useState(1.0);
  const [showCrosshairs, setShowCrosshairs] = useState(false);
  const [showOrientationCube, setShowOrientationCube] = useState(true);
  // New: z-step thickness (relative voxel spacing) and interpolation mode
  const [zThickness, setZThickness] = useState(1.0);
  const [interpolation, setInterpolation] = useState<"linear" | "nearest">("linear");
  // Debounced z-spacing that actually triggers a backend refetch
  const [appliedZThickness, setAppliedZThickness] = useState(1.0);

  // Save dialog
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveFormat, setSaveFormat] = useState("PNG");
  const [saveQuality, setSaveQuality] = useState(95);
  const [savePath, setSavePath] = useState("");
  const [saving, setSaving] = useState(false);
  const fetchImages = useFigureStore((s) => s.fetchImages);
  const refreshPanelThumbnail = useFigureStore((s) => s.refreshPanelThumbnail);
  const requestPreview = useFigureStore((s) => s.requestPreview);

  // Initialize NiiVue and load volume
  useEffect(() => {
    if (!open) return;
    let disposed = false;

    const init = async () => {
      // Wait for canvas to be mounted (Dialog may not mount children immediately)
      setLoadingStage("Waiting for canvas...");
      const waitForCanvas = async (maxWait = 3000) => {
        const start = Date.now();
        while (!canvasRef.current && Date.now() - start < maxWait) {
          await new Promise(r => setTimeout(r, 50));
        }
        return canvasRef.current != null;
      };
      const hasCanvas = await waitForCanvas();
      if (disposed) return;
      if (!hasCanvas) {
        setError("Canvas element did not mount");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");
      try {
        setLoadingStage("Reading TIFF (fast)...");
        const t0 = performance.now();
        const resp = await api.getZStackNifti(imageName, { startFrame, endFrame, maxDim: 256, zSpacing: appliedZThickness });
        console.log(`[VolumeViewer] NIfTI fetched (${resp.width}×${resp.height}×${resp.depth}, z=${appliedZThickness}) in ${(performance.now() - t0).toFixed(0)}ms`);
        if (disposed) return;

        setLoadingStage("Decoding volume...");
        const raw = atob(resp.data);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

        setLoadingStage("Initializing 3D renderer...");
        const nv = new Niivue({
          backColor: [0.11, 0.11, 0.12, 1],
          show3Dcrosshair: false,
          isOrientCube: showOrientationCube,
          crosshairWidth: 0,
          colorbarHeight: 0,
          sagittalNoseLeft: true,
          isColorbar: false,
        });
        await nv.attachToCanvas(canvasRef.current!);
        nv.setSliceType(sliceType);

        setLoadingStage("Loading volume...");
        await nv.loadFromArrayBuffer(bytes.buffer, `volume.nii`);

        if (nv.volumes.length > 0) {
          nv.volumes[0].colormap = colormap;
          nv.volumes[0].opacity = opacity;
          nv.updateGLVolume();
        }

        // Apply current interpolation (nearest vs linear)
        try {
          nv.setInterpolation(interpolation === "nearest");
        } catch { /* ignore */ }

        nvRef.current = nv;
        setLoading(false);
      } catch (e) {
        console.error("[VolumeViewer] Error:", e);
        if (!disposed) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    };

    init();

    return () => {
      disposed = true;
      if (nvRef.current) {
        try {
          (nvRef.current as unknown as { closeDrawing?: () => void }).closeDrawing?.();
        } catch { /* ignore */ }
      }
    };
  }, [open, imageName, startFrame, endFrame, appliedZThickness]); // eslint-disable-line

  // Update slice type when changed
  useEffect(() => {
    if (nvRef.current) {
      nvRef.current.setSliceType(sliceType);
    }
  }, [sliceType]);

  // Update colormap
  useEffect(() => {
    if (nvRef.current && nvRef.current.volumes.length > 0) {
      nvRef.current.volumes[0].colormap = colormap;
      nvRef.current.updateGLVolume();
    }
  }, [colormap]);

  // Update opacity
  useEffect(() => {
    if (nvRef.current && nvRef.current.volumes.length > 0) {
      nvRef.current.volumes[0].opacity = opacity;
      nvRef.current.updateGLVolume();
    }
  }, [opacity]);

  // Update orientation cube
  useEffect(() => {
    if (nvRef.current) {
      (nvRef.current.opts as { isOrientCube: boolean }).isOrientCube = showOrientationCube;
      nvRef.current.drawScene();
    }
  }, [showOrientationCube]);

  // Update crosshairs
  useEffect(() => {
    if (nvRef.current) {
      (nvRef.current.opts as { crosshairWidth: number }).crosshairWidth = showCrosshairs ? 1 : 0;
      nvRef.current.drawScene();
    }
  }, [showCrosshairs]);

  // Update interpolation (nearest vs linear) without reloading the volume.
  useEffect(() => {
    if (nvRef.current) {
      try {
        nvRef.current.setInterpolation(interpolation === "nearest");
        nvRef.current.drawScene();
      } catch { /* ignore */ }
    }
  }, [interpolation]);

  // Debounce z-thickness slider: only re-fetch once the user stops sliding.
  useEffect(() => {
    if (!open) return;
    if (zThickness === appliedZThickness) return;
    const h = setTimeout(() => setAppliedZThickness(zThickness), 350);
    return () => clearTimeout(h);
  }, [zThickness, appliedZThickness, open]);

  const canvasToPngDataUrl = (): string | null => {
    if (!canvasRef.current || !nvRef.current) return null;
    nvRef.current.drawScene();
    return canvasRef.current.toDataURL("image/png");
  };

  const saveViewAsPng = () => {
    const dataUrl = canvasToPngDataUrl();
    if (!dataUrl) return;
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = `volume_${imageName.replace(/\.\w+$/, "")}.png`;
    link.click();
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
      } else {
        saveViewAsPng();
      }
    } catch {
      saveViewAsPng();
    }
  };

  const performSave = async () => {
    if (!savePath) return;
    setSaving(true);
    try {
      const dataUrl = canvasToPngDataUrl();
      if (!dataUrl) throw new Error("Canvas not ready");
      const b64 = dataUrl.split(",")[1];

      if (saveFormat === "PNG") {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("save_base64_to_path", { path: savePath, dataB64: b64 });
        } catch {
          const link = document.createElement("a");
          link.href = dataUrl;
          link.download = savePath.split("/").pop() || "volume.png";
          link.click();
        }
      } else {
        // For TIFF/JPEG: send canvas PNG to backend and let it convert
        const { invoke } = await import("@tauri-apps/api/core");
        // Write PNG first, then tell backend to convert
        await invoke("save_base64_to_path", { path: savePath + ".tmp.png", dataB64: b64 });
        // Note: for a proper TIFF/JPEG conversion, we'd call a backend endpoint.
        // For now, save as PNG with requested extension (PIL can be used later).
        await invoke("save_base64_to_path", { path: savePath, dataB64: b64 });
      }
      setSaveOpen(false);
    } catch (e) {
      setError("Save failed: " + (e instanceof Error ? e.message : String(e)));
    }
    setSaving(false);
  };

  const useAsPanel = async () => {
    if (panelRow == null || panelCol == null) return;
    try {
      const dataUrl = canvasToPngDataUrl();
      if (!dataUrl) throw new Error("Canvas not ready");
      const b64 = dataUrl.split(",")[1];
      // Backend will store the render as a hidden, panel-internal image
      // (not visible in the media timeline) and assign it to this panel.
      await api.saveCanvasAsPanel(imageName, panelRow, panelCol, b64);
      // fetchImages() will refresh the timeline — because the new image is
      // hidden on the backend, the timeline stays clean.
      await fetchImages();
      // Refresh the panel thumbnail so the editor sees the new 3D image,
      // and request a preview so the main canvas updates.
      try { await refreshPanelThumbnail(panelRow, panelCol); } catch { /* ignore */ }
      requestPreview();
      // Close the 3D dialog AND notify the parent (e.g. EditPanelDialog) that
      // the panel's underlying image has been replaced. The user will re-open
      // the edit panel on the updated image to continue cropping/adjusting.
      onClose();
      if (onAppliedToPanel) onAppliedToPanel();
    } catch (e) {
      setError("Failed to set as panel: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  return (
    <>
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>3D Volume View — {imageName}</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", height: "100%", overflow: "hidden" }}>
        <Box sx={{ flex: 1, position: "relative", bgcolor: "#1c1c1e" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
          {loading && (
            <Box sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "rgba(0,0,0,0.6)" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, bgcolor: "background.paper", px: 2, py: 1, borderRadius: 1 }}>
                <CircularProgress size={16} />
                <Typography variant="caption">{loadingStage}</Typography>
              </Box>
            </Box>
          )}
          {error && (
            <Box sx={{ position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)", bgcolor: "error.dark", color: "white", px: 2, py: 1, borderRadius: 1 }}>
              <Typography variant="caption">{error}</Typography>
            </Box>
          )}
        </Box>

        <Box sx={{ width: 240, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 1.5, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>View</Typography>

          <ToggleButtonGroup value={sliceType} exclusive onChange={(_, v) => { if (v != null) setSliceType(v); }} size="small" orientation="vertical">
            <ToggleButton value={SLICE_TYPE.RENDER} sx={{ fontSize: "0.6rem" }}>3D Volume</ToggleButton>
            <ToggleButton value={SLICE_TYPE.AXIAL} sx={{ fontSize: "0.6rem" }}>Axial (top)</ToggleButton>
            <ToggleButton value={SLICE_TYPE.CORONAL} sx={{ fontSize: "0.6rem" }}>Coronal (front)</ToggleButton>
            <ToggleButton value={SLICE_TYPE.SAGITTAL} sx={{ fontSize: "0.6rem" }}>Sagittal (side)</ToggleButton>
            <ToggleButton value={SLICE_TYPE.MULTIPLANAR} sx={{ fontSize: "0.6rem" }}>Multi-planar</ToggleButton>
          </ToggleButtonGroup>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Colormap</Typography>
            <Select size="small" value={colormap} onChange={(e) => setColormap(e.target.value)}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}>
              {COLORMAPS.map(c => (
                <MenuItem key={c} value={c} sx={{ fontSize: "0.65rem" }}>{c.charAt(0).toUpperCase() + c.slice(1)}</MenuItem>
              ))}
            </Select>
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Opacity: {opacity.toFixed(2)}</Typography>
            <Slider size="small" value={opacity} min={0.05} max={1.0} step={0.01} onChange={(_, v) => setOpacity(v as number)} />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>
              Z-step thickness: {zThickness.toFixed(2)}×
              {zThickness !== appliedZThickness && " (applying…)"}
            </Typography>
            <Slider
              size="small"
              value={zThickness}
              min={0.25}
              max={5.0}
              step={0.05}
              marks={[{ value: 1, label: "" }]}
              onChange={(_, v) => setZThickness(v as number)}
            />
          </Box>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Interpolation</Typography>
            <ToggleButtonGroup
              value={interpolation}
              exclusive
              size="small"
              onChange={(_, v) => { if (v) setInterpolation(v); }}
              sx={{ mt: 0.25 }}
              fullWidth
            >
              <ToggleButton value="linear" sx={{ fontSize: "0.6rem", py: 0.25 }}>Linear</ToggleButton>
              <ToggleButton value="nearest" sx={{ fontSize: "0.6rem", py: 0.25 }}>Nearest</ToggleButton>
            </ToggleButtonGroup>
            <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", display: "block", mt: 0.25 }}>
              Nearest = sharp voxels · Linear = smooth
            </Typography>
          </Box>

          <FormControlLabel sx={{ ml: 0 }}
            control={<Checkbox size="small" checked={showOrientationCube} onChange={(e) => setShowOrientationCube(e.target.checked)} sx={{ p: 0.25 }} />}
            label={<Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Orientation cube</Typography>}
          />
          <FormControlLabel sx={{ ml: 0 }}
            control={<Checkbox size="small" checked={showCrosshairs} onChange={(e) => setShowCrosshairs(e.target.checked)} sx={{ p: 0.25 }} />}
            label={<Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Crosshairs (2D views)</Typography>}
          />

          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>
            Drag to rotate. Scroll to zoom. Powered by NiiVue (WebGL2).
          </Typography>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
            <Button size="small" variant="outlined" onClick={openSaveDialog} startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
              disabled={loading} sx={{ fontSize: "0.6rem", textTransform: "none" }}>
              Save View...
            </Button>
            {panelRow != null && panelCol != null && (
              <Button size="small" variant="contained" color="primary" onClick={useAsPanel} startIcon={<ImageIcon sx={{ fontSize: 12 }} />}
                disabled={loading} sx={{ fontSize: "0.6rem", textTransform: "none" }}
                title="Replace this panel's image with the current 3D view. You can continue cropping/adjusting it afterwards.">
                Apply to Panel
              </Button>
            )}
          </Box>
        </Box>
      </DialogContent>
    </Dialog>

    <Dialog open={saveOpen} onClose={() => setSaveOpen(false)} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem" }}>Save Volume View</DialogTitle>
      <DialogContent>
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
          <TextField label="File path" size="small" fullWidth value={savePath} onChange={(e) => setSavePath(e.target.value)} />
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
              <Slider size="small" value={saveQuality} min={1} max={100} step={1} onChange={(_, v) => setSaveQuality(v as number)} />
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
