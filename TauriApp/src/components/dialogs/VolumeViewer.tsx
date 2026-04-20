/* ──────────────────────────────────────────────────────────
   VolumeViewer — Fast MIP views + optional rotation animation.
   Scientific-standard approach (like Fiji/ImageJ).
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions, IconButton, Box, Typography,
  Slider, Select, MenuItem, Button, CircularProgress, ToggleButtonGroup, ToggleButton,
  TextField,
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
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Loading...");
  const [error, setError] = useState("");
  const [colormap, setColormap] = useState("gray");
  const [mode, setMode] = useState<"mip_xy" | "mip_xz" | "mip_yz" | "rotate">("mip_xy");
  const [rotationIndex, setRotationIndex] = useState(18); // middle
  const [playing, setPlaying] = useState(false);
  const [hasRotation, setHasRotation] = useState(false);
  const [loadingRotation, setLoadingRotation] = useState(false);

  const mipsRef = useRef<{ mip_xy: string; mip_xz: string; mip_yz: string; rotation_frames: string[] }>({
    mip_xy: "", mip_xz: "", mip_yz: "", rotation_frames: [],
  });

  // Force re-render by using state that updates alongside ref
  const [, setTick] = useState(0);

  // Save dialog
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveFormat, setSaveFormat] = useState("PNG");
  const [saveQuality, setSaveQuality] = useState(95);
  const [savePath, setSavePath] = useState("");
  const [saving, setSaving] = useState(false);
  const fetchImages = useFigureStore((s) => s.fetchImages);

  // Load MIPs (fast)
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      setLoadingStage("Computing projections...");
      try {
        const result = await api.getZStackMips(imageName, {
          startFrame, endFrame, colormap, includeRotation: false,
        });
        if (cancelled) return;
        mipsRef.current = result;
        setTick(t => t + 1);
        setHasRotation(false);
        setLoading(false);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [open, imageName, startFrame, endFrame, colormap]);

  // Load rotation frames on-demand
  const loadRotation = async () => {
    setLoadingRotation(true);
    try {
      const result = await api.getZStackMips(imageName, {
        startFrame, endFrame, colormap, includeRotation: true, rotationFrames: 36,
      });
      mipsRef.current = result;
      setHasRotation(result.rotation_frames.length > 0);
      setTick(t => t + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setLoadingRotation(false);
  };

  // Auto-play rotation
  useEffect(() => {
    if (!playing || mode !== "rotate" || !hasRotation) return;
    const interval = setInterval(() => {
      setRotationIndex(i => (i + 1) % mipsRef.current.rotation_frames.length);
    }, 100);
    return () => clearInterval(interval);
  }, [playing, mode, hasRotation]);

  const currentImage = (() => {
    const m = mipsRef.current;
    if (mode === "mip_xy") return m.mip_xy;
    if (mode === "mip_xz") return m.mip_xz;
    if (mode === "mip_yz") return m.mip_yz;
    if (mode === "rotate" && m.rotation_frames[rotationIndex]) return m.rotation_frames[rotationIndex];
    return "";
  })();

  const saveViewAsPng = () => {
    if (!currentImage) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${currentImage}`;
    link.download = `volume_${imageName.replace(/\.\w+$/, "")}_${mode}.png`;
    link.click();
  };

  const openSaveDialog = async () => {
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const ext = saveFormat === "TIFF" ? "tiff" : saveFormat.toLowerCase();
      const path = await save({
        defaultPath: `volume_${imageName.replace(/\.\w+$/, "")}_${mode}.${ext}`,
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
      if (saveFormat === "PNG" && currentImage) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("save_base64_to_path", { path: savePath, dataB64: currentImage });
        } catch {
          const link = document.createElement("a");
          link.href = `data:image/png;base64,${currentImage}`;
          link.download = savePath.split("/").pop() || "volume.png";
          link.click();
        }
      } else {
        // TIFF/JPEG: render via backend
        const methodMap: Record<string, string> = { mip_xy: "mip_xy", mip_xz: "mip_xz", mip_yz: "mip_yz", rotate: "surface" };
        await api.saveVolumeRenderAsImage(imageName, {
          startFrame, endFrame,
          colormap, method: methodMap[mode] ?? "mip_xy",
          width: 1600, height: 1200,
          format: saveFormat, quality: saveQuality, filePath: savePath,
        });
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
      const methodMap: Record<string, string> = { mip_xy: "mip_xy", mip_xz: "mip_xz", mip_yz: "mip_yz", rotate: "surface" };
      await api.useVolumeAsPanel(imageName, panelRow, panelCol, {
        startFrame, endFrame, colormap,
        method: methodMap[mode] ?? "mip_xy",
        width: 1600, height: 1200,
      });
      await fetchImages();
      onClose();
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
        <Box sx={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "#1c1c1e" }}>
          {currentImage && (
            <img src={`data:image/png;base64,${currentImage}`} alt="Volume"
              style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", userSelect: "none" }} draggable={false} />
          )}
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

        <Box sx={{ width: 240, flexShrink: 0, borderLeft: 1, borderColor: "divider", p: 2, display: "flex", flexDirection: "column", gap: 2, overflow: "auto" }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>View Mode</Typography>

          <ToggleButtonGroup value={mode} exclusive onChange={(_, v) => { if (v) setMode(v); }} size="small" orientation="vertical">
            <ToggleButton value="mip_xy" sx={{ fontSize: "0.65rem" }}>MIP XY (top)</ToggleButton>
            <ToggleButton value="mip_xz" sx={{ fontSize: "0.65rem" }}>MIP XZ (front)</ToggleButton>
            <ToggleButton value="mip_yz" sx={{ fontSize: "0.65rem" }}>MIP YZ (side)</ToggleButton>
            <ToggleButton value="rotate" sx={{ fontSize: "0.65rem" }}>3D Rotation</ToggleButton>
          </ToggleButtonGroup>

          <Box>
            <Typography variant="caption" sx={{ fontSize: "0.6rem", mb: 0.5, display: "block" }}>Colormap</Typography>
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

          {mode === "rotate" && (
            <Box>
              {!hasRotation ? (
                <Button size="small" variant="contained" fullWidth onClick={loadRotation} disabled={loadingRotation}
                  sx={{ fontSize: "0.65rem", textTransform: "none" }}>
                  {loadingRotation ? "Rendering 36 frames..." : "Generate 3D rotation"}
                </Button>
              ) : (
                <>
                  <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>
                    Angle: {Math.round(-180 + (360 * rotationIndex / 36))}°
                  </Typography>
                  <Slider size="small" value={rotationIndex} min={0} max={35} step={1}
                    onChange={(_, v) => setRotationIndex(v as number)} />
                  <Box sx={{ display: "flex", gap: 0.5, mt: 1 }}>
                    <Button size="small" variant={playing ? "contained" : "outlined"} onClick={() => setPlaying(!playing)}
                      sx={{ fontSize: "0.6rem", flex: 1 }}>
                      {playing ? "⏸ Pause" : "▶ Play"}
                    </Button>
                    <Button size="small" variant="outlined" onClick={loadRotation}
                      sx={{ fontSize: "0.6rem" }}>Re-render</Button>
                  </Box>
                </>
              )}
            </Box>
          )}

          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", mt: 1 }}>
            MIP = Maximum Intensity Projection. Shows the brightest voxels along each axis.
          </Typography>

          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, mt: 1 }}>
            <Button size="small" variant="outlined" onClick={openSaveDialog} startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
              disabled={!currentImage} sx={{ fontSize: "0.6rem", textTransform: "none" }}>
              Save View...
            </Button>
            {panelRow != null && panelCol != null && (
              <Button size="small" variant="contained" color="primary" onClick={useAsPanel} startIcon={<ImageIcon sx={{ fontSize: 12 }} />}
                disabled={!currentImage} sx={{ fontSize: "0.6rem", textTransform: "none" }}>
                Use as Panel Image
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
