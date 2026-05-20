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
  // True when the backend returned an RGB24 NIfTI (multichannel TIFF
  // composited with the per-channel tints from the Channels block in the
  // Edit Panel). In that case NiiVue ignores the colormap dropdown, so
  // we surface a hint instead of leaving the control there confusingly
  // disabled.
  const [isRgbVolume, setIsRgbVolume] = useState(false);
  const [opacity, setOpacity] = useState(1.0);
  const [showCrosshairs, setShowCrosshairs] = useState(false);
  const [showOrientationCube, setShowOrientationCube] = useState(true);
  // New: z-step thickness (relative voxel spacing) and interpolation mode
  const [zThickness, setZThickness] = useState(1.0);
  const [interpolation, setInterpolation] = useState<"linear" | "nearest">("linear");
  // Debounced z-spacing that actually triggers a backend refetch
  const [appliedZThickness, setAppliedZThickness] = useState(1.0);

  // Per-channel state for the in-viewer Channels block. Mirrors the
  // EditPanelDialog ChannelsBlock state shape so the UI feels identical;
  // toggling a channel here PATCH'es the same /channels endpoint and we
  // re-fetch the NIfTI so the 3D render picks up the new composite.
  type ChInfo = {
    is_multichannel: boolean;
    num_channels?: number; num_z?: number; current_z?: number;
    tints?: string[]; enabled?: boolean[];
    black_levels?: number[]; white_levels?: number[];
    names?: string[];
  };
  const [chanInfo, setChanInfo] = useState<ChInfo | null>(null);
  // Used by both Channels + Align sections — bumps to force a NIfTI
  // re-fetch when something mutates the underlying data.
  const [reloadTick, setReloadTick] = useState(0);
  // What's actually being rendered (after Align replaces the source).
  const [activeImageName, setActiveImageName] = useState(imageName);
  useEffect(() => { setActiveImageName(imageName); }, [imageName]);

  // Alignment state.  Availability is fetched once per open and tells us
  // which methods can run on this host (SIFT requires ImageJ/Fiji).
  type AlignAvail = {
    sift: { available: boolean; kind: string; path: string };
    phase_correlation: { available: boolean; kind: string };
  };
  const [alignAvail, setAlignAvail] = useState<AlignAvail | null>(null);
  const [alignMethod, setAlignMethod] = useState<"sift" | "phase_correlation">("phase_correlation");
  const [alignRunning, setAlignRunning] = useState(false);
  const [alignError, setAlignError] = useState("");
  const [alignAdvancedOpen, setAlignAdvancedOpen] = useState(false);
  // Progress reporting from the polling job.
  const [alignProgress, setAlignProgress] = useState(0);
  const [alignStage, setAlignStage] = useState("");
  // SIFT controls — same defaults as Fiji's "Linear Stack Alignment with SIFT"
  const [siftBlur, setSiftBlur] = useState(1.6);
  const [siftSteps, setSiftSteps] = useState(3);
  const [siftMinSize, setSiftMinSize] = useState(64);
  const [siftMaxSize, setSiftMaxSize] = useState(1024);
  const [siftDescSize, setSiftDescSize] = useState(4);
  const [siftDescBins, setSiftDescBins] = useState(8);
  const [siftRatio, setSiftRatio] = useState(0.92);
  const [siftError, setSiftError] = useState(25);
  const [siftInlier, setSiftInlier] = useState(0.05);
  const [siftTransform, setSiftTransform] = useState<"Translation" | "Rigid" | "Similarity" | "Affine">("Rigid");
  const [siftInterpolate, setSiftInterpolate] = useState(true);
  const [pcWindow, setPcWindow] = useState<"hann" | "rect">("hann");
  const [pcMaxShift, setPcMaxShift] = useState(0.25);
  // Shared performance knobs — cap the per-frame size before alignment
  // (Fiji caps it anyway at sift_maximum_image_size) and subprocess
  // timeout. 1024px keeps 100-frame stacks alignable in ~2-3 minutes.
  const [alignMaxDim, setAlignMaxDim] = useState(1024);
  const [alignTimeout, setAlignTimeout] = useState(1800);
  // Reference source for alignment + optional CLAHE pre-processing.
  // Default 'max' pools features from all enabled channels and is more
  // robust than picking a single channel — see the backend docstring
  // for the strategy trade-offs.
  const [alignSource, setAlignSource] = useState<"max" | "mean" | "sum" | "channel">("max");
  const [alignUseClahe, setAlignUseClahe] = useState(false);

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
        const resp = await api.getZStackNifti(activeImageName, { startFrame, endFrame, maxDim: 256, zSpacing: appliedZThickness });
        console.log(`[VolumeViewer] NIfTI fetched (${resp.width}×${resp.height}×${resp.depth}, z=${appliedZThickness}, rgb=${resp.rgb ?? false}) in ${(performance.now() - t0).toFixed(0)}ms`);
        if (disposed) return;
        // RGB volumes carry baked-in channel tints; NiiVue ignores the
        // colormap dropdown for them, so surface a hint instead.
        setIsRgbVolume(Boolean(resp.rgb));
        // Fetch channel info in parallel with the rest of the init.
        api.getChannelInfo(activeImageName).then(ci => { if (!disposed) setChanInfo(ci as ChInfo); }).catch(() => { if (!disposed) setChanInfo(null); });
        // Detect ImageJ availability once per session of the dialog.
        if (!alignAvail) {
          api.getAlignAvailability().then(a => { if (!disposed) setAlignAvail(a); }).catch(() => {/*ignore*/});
        }

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
  }, [open, activeImageName, startFrame, endFrame, appliedZThickness, reloadTick]); // eslint-disable-line

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
      // NiiVue ignores `opts.crosshairWidth` mutations unless we call
      // setCrosshairWidth() AND force a GL refresh (drawScene alone
      // doesn't repaint the overlay). Setting the crosshair colour
      // explicitly too — the default is fully transparent on some
      // builds which makes "width = 1" invisible.
      const w = showCrosshairs ? 2 : 0;
      try {
        const nv = nvRef.current as unknown as {
          setCrosshairWidth?: (w: number) => void;
          setCrosshairColor?: (rgba: number[]) => void;
          opts: { crosshairWidth?: number; show3Dcrosshair?: boolean };
          updateGLVolume?: () => void;
          drawScene?: () => void;
        };
        if (nv.setCrosshairWidth) nv.setCrosshairWidth(w);
        else nv.opts.crosshairWidth = w;
        if (nv.setCrosshairColor) nv.setCrosshairColor([1, 0.5, 0.1, 1]);
        // Also flip the 3D-mode crosshair so users see something when
        // they tick the box while the viewer is on 3D Volume mode.
        nv.opts.show3Dcrosshair = showCrosshairs;
        if (nv.updateGLVolume) nv.updateGLVolume();
        if (nv.drawScene) nv.drawScene();
      } catch (e) {
        console.warn("[VolumeViewer] crosshair toggle failed:", e);
      }
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
      await api.saveCanvasAsPanel(activeImageName, panelRow, panelCol, b64);
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
            <Select
              size="small"
              value={colormap}
              onChange={(e) => setColormap(e.target.value)}
              disabled={isRgbVolume}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}
            >
              {COLORMAPS.map(c => (
                <MenuItem key={c} value={c} sx={{ fontSize: "0.65rem" }}>{c.charAt(0).toUpperCase() + c.slice(1)}</MenuItem>
              ))}
            </Select>
            {isRgbVolume && (
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "primary.light", display: "block", mt: 0.5 }}>
                Multichannel volume — colors come from per-channel tints set in the Edit Panel → Channels block. The colormap dropdown has no effect.
              </Typography>
            )}
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

          {/* ── Channels block (multichannel sources only) ─────────────
              Toggling a channel here PATCHes the same /channels endpoint
              used in EditPanelDialog and bumps reloadTick so the NIfTI
              gets refetched with the updated tints/visibility. */}
          {chanInfo?.is_multichannel && (chanInfo.num_channels ?? 0) > 1 && (
            <Box sx={{ mt: 1, p: 1, border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, display: "block", mb: 0.5 }}>
                Channels ({chanInfo.num_channels})
              </Typography>
              {Array.from({ length: chanInfo.num_channels ?? 0 }).map((_, c) => {
                const tint = chanInfo.tints?.[c] ?? "#ffffff";
                const enabled = chanInfo.enabled?.[c] ?? true;
                const name = chanInfo.names?.[c] ?? `Ch ${c + 1}`;
                return (
                  <Box key={c} sx={{ display: "flex", gap: 0.5, alignItems: "center", mb: 0.25 }}>
                    <Checkbox
                      size="small" checked={enabled} sx={{ p: 0.25 }}
                      onChange={async (e) => {
                        const next = [...(chanInfo.enabled ?? Array(chanInfo.num_channels).fill(true))];
                        next[c] = e.target.checked;
                        try {
                          const res = await api.updateChannels(activeImageName, { enabled: next });
                          setChanInfo(prev => prev ? { ...prev, ...res } : prev);
                          setReloadTick(t => t + 1);
                        } catch (err) { console.error("[channels]", err); }
                      }}
                    />
                    <Box sx={{ width: 16, height: 16, bgcolor: tint, border: "1px solid #999", borderRadius: 0.5, opacity: enabled ? 1 : 0.4 }} title={`${name} tint`} />
                    <Typography variant="caption" sx={{ fontSize: "0.6rem", flex: 1, ml: 0.5, opacity: enabled ? 1 : 0.5 }} title={name}>
                      {name}
                    </Typography>
                  </Box>
                );
              })}
              <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", display: "block", mt: 0.5 }}>
                Tints + names come from Edit Panel → Channels. Toggle here to refresh the 3D composite.
              </Typography>
            </Box>
          )}

          {/* ── Z-stack alignment ───────────────────────────────────────
              SIFT runs through ImageJ/Fiji (greyed out when Fiji isn't
              installed). Phase correlation is always available. After
              alignment runs we swap `activeImageName` to the new
              ::aligned stack so the 3D viewer shows the registered
              result; the user can also Apply-to-Panel to commit it. */}
          <Box sx={{ mt: 1, p: 1, border: "1px solid", borderColor: "divider", borderRadius: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, display: "block", mb: 0.5 }}>
              Slice alignment
            </Typography>
            <Select
              size="small" value={alignMethod} onChange={(e) => setAlignMethod(e.target.value as "sift" | "phase_correlation")}
              sx={{ fontSize: "0.65rem", width: "100%", "& .MuiSelect-select": { py: 0.3 } }}
            >
              <MenuItem value="phase_correlation" sx={{ fontSize: "0.65rem" }}>
                Phase correlation (always available)
              </MenuItem>
              <MenuItem
                value="sift"
                disabled={!alignAvail?.sift.available}
                sx={{ fontSize: "0.65rem" }}
                title={alignAvail?.sift.available
                  ? `Uses Fiji at ${alignAvail.sift.path}`
                  : "Requires ImageJ/Fiji on this host (not detected)"}
              >
                ImageJ SIFT — Linear Stack Align {alignAvail && !alignAvail.sift.available ? "(ImageJ not detected)" : ""}
              </MenuItem>
            </Select>
            {/* Reference source — what the alignment algorithm actually
                sees. Max-projection pools features across enabled
                channels, giving SIFT more to lock onto and avoiding
                misalignment when a single chosen channel has weak signal
                in some slices. Channel toggles in the Channels block
                above directly influence what's pooled. */}
            <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mt: 0.5 }}>
              <Typography variant="caption" sx={{ fontSize: "0.55rem", minWidth: 56 }}
                title="Which image SIFT/phase corr sees. 'Max' pools features across all enabled channels.">
                Reference
              </Typography>
              <Select
                size="small" value={alignSource}
                onChange={(e) => setAlignSource(e.target.value as "max" | "mean" | "sum" | "channel")}
                sx={{ flex: 1, fontSize: "0.6rem", "& .MuiSelect-select": { py: 0.2, px: 0.75 } }}
              >
                <MenuItem value="max"     sx={{ fontSize: "0.6rem" }}>Max-projection of channels</MenuItem>
                <MenuItem value="mean"    sx={{ fontSize: "0.6rem" }}>Mean of channels</MenuItem>
                <MenuItem value="sum"     sx={{ fontSize: "0.6rem" }}>Sum of channels</MenuItem>
                <MenuItem value="channel" sx={{ fontSize: "0.6rem" }}>Single channel (first enabled)</MenuItem>
              </Select>
            </Box>
            <FormControlLabel sx={{ ml: 0 }}
              control={<Checkbox size="small" checked={alignUseClahe} onChange={(e) => setAlignUseClahe(e.target.checked)} sx={{ p: 0.25 }} />}
              label={<Typography variant="caption" sx={{ fontSize: "0.6rem" }}
                title="Local contrast normalization — helps SIFT find features when Z intensity drifts.">
                CLAHE pre-process
              </Typography>}
            />
            <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mt: 0.5 }}>
              <Button
                size="small" variant="contained"
                disabled={alignRunning || loading || (alignMethod === "sift" && !alignAvail?.sift.available)}
                onClick={async () => {
                  setAlignRunning(true);
                  setAlignError("");
                  setAlignProgress(0);
                  setAlignStage("starting…");
                  try {
                    const res = await api.alignAndWait(activeImageName, {
                      method: alignMethod,
                      startFrame, endFrame,
                      siftInitialGaussianBlur: siftBlur,
                      siftStepsPerScaleOctave: siftSteps,
                      siftMinimumImageSize: siftMinSize,
                      siftMaximumImageSize: siftMaxSize,
                      siftFeatureDescriptorSize: siftDescSize,
                      siftFeatureDescriptorOrientationBins: siftDescBins,
                      siftClosestNextClosestRatio: siftRatio,
                      siftMaximalAlignmentError: siftError,
                      siftInlierRatio: siftInlier,
                      siftExpectedTransformation: siftTransform,
                      siftInterpolate,
                      pcWindow, pcMaxShiftFrac: pcMaxShift,
                      alignMaxDim, timeoutSec: alignTimeout,
                      alignmentSource: alignSource,
                      useClahe: alignUseClahe,
                    }, (p, stage) => {
                      setAlignProgress(p);
                      setAlignStage(stage);
                    });
                    // Swap the active image so the NIfTI refetch picks
                    // up the aligned stack; the original stays intact.
                    setActiveImageName(res.aligned_name);
                    setReloadTick(t => t + 1);
                    // Re-fetch channels info since the aligned stack
                    // gets its own channel-group entry inheriting
                    // tints/names from the source.
                    api.getChannelInfo(res.aligned_name).then(ci => setChanInfo(ci as ChInfo)).catch(() => {});
                  } catch (e) {
                    setAlignError(e instanceof Error ? e.message : String(e));
                  }
                  setAlignRunning(false);
                }}
                sx={{ fontSize: "0.6rem", textTransform: "none", flex: 1 }}
              >
                {alignRunning ? "Aligning…" : "Run alignment"}
              </Button>
              {activeImageName !== imageName && (
                <Button
                  size="small" variant="outlined" disabled={alignRunning}
                  onClick={() => { setActiveImageName(imageName); setReloadTick(t => t + 1); }}
                  sx={{ fontSize: "0.55rem", textTransform: "none" }}
                  title="Revert to the unaligned source stack"
                >
                  Revert
                </Button>
              )}
            </Box>
            {alignRunning && (
              <Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block" }}>
                  {alignStage} — {Math.round(alignProgress * 100)}%
                </Typography>
                <Box sx={{ height: 4, bgcolor: "action.disabledBackground", borderRadius: 2, overflow: "hidden", mt: 0.25 }}>
                  <Box sx={{
                    height: "100%",
                    width: `${Math.max(2, alignProgress * 100)}%`,
                    bgcolor: "primary.main",
                    transition: "width 0.2s ease",
                  }} />
                </Box>
              </Box>
            )}
            {alignError && (
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "error.light", display: "block", mt: 0.5 }}>
                {alignError}
              </Typography>
            )}
            {activeImageName !== imageName && !alignError && (
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "primary.light", display: "block", mt: 0.5 }}>
                Showing aligned stack. Click Apply to Panel below to commit.
              </Typography>
            )}
            <Button
              size="small" variant="text"
              onClick={() => setAlignAdvancedOpen(o => !o)}
              sx={{ fontSize: "0.55rem", textTransform: "none", mt: 0.5, p: 0 }}
            >
              {alignAdvancedOpen ? "▼" : "▶"} Advanced
            </Button>
            {alignAdvancedOpen && (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.4, mt: 0.5, pl: 0.5 }}>
                {/* Performance knobs — applies to both methods. Capping
                    frame size before SIFT keeps runtime under control
                    for big light-sheet stacks; the algorithm internally
                    downsamples to sift_maximum_image_size anyway. */}
                <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>
                  Align max-dim (px / frame, 0 = no cap): {alignMaxDim}
                </Typography>
                <Slider size="small" value={alignMaxDim} min={0} max={2048} step={64}
                  marks={[{ value: 0, label: "off" }, { value: 1024, label: "1024" }, { value: 2048, label: "2048" }]}
                  onChange={(_, v) => setAlignMaxDim(v as number)} />
                <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>
                  SIFT subprocess timeout: {alignTimeout}s
                </Typography>
                <Slider size="small" value={alignTimeout} min={120} max={3600} step={60}
                  onChange={(_, v) => setAlignTimeout(v as number)} />
                {alignMethod === "phase_correlation" ? (
                  <>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Window</Typography>
                    <Select size="small" value={pcWindow} onChange={(e) => setPcWindow(e.target.value as "hann" | "rect")}
                      sx={{ fontSize: "0.55rem", "& .MuiSelect-select": { py: 0.2 } }}>
                      <MenuItem value="hann" sx={{ fontSize: "0.55rem" }}>Hann (recommended)</MenuItem>
                      <MenuItem value="rect" sx={{ fontSize: "0.55rem" }}>Rectangular</MenuItem>
                    </Select>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Max shift (fraction of image): {pcMaxShift.toFixed(2)}</Typography>
                    <Slider size="small" value={pcMaxShift} min={0.05} max={0.5} step={0.01}
                      onChange={(_, v) => setPcMaxShift(v as number)} />
                  </>
                ) : (
                  <>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Initial Gaussian blur: {siftBlur.toFixed(2)}</Typography>
                    <Slider size="small" value={siftBlur} min={0.5} max={5} step={0.1} onChange={(_, v) => setSiftBlur(v as number)} />
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Steps per scale octave: {siftSteps}</Typography>
                    <Slider size="small" value={siftSteps} min={1} max={8} step={1} onChange={(_, v) => setSiftSteps(v as number)} />
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Min/Max image size</Typography>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      <TextField type="number" size="small" value={siftMinSize}
                        onChange={(e) => setSiftMinSize(Math.max(8, Number(e.target.value)))}
                        inputProps={{ min: 8, max: 4096 }}
                        sx={{ width: 70, "& input": { fontSize: "0.55rem", py: 0.25 } }} />
                      <TextField type="number" size="small" value={siftMaxSize}
                        onChange={(e) => setSiftMaxSize(Math.max(siftMinSize, Number(e.target.value)))}
                        inputProps={{ min: 8, max: 8192 }}
                        sx={{ width: 70, "& input": { fontSize: "0.55rem", py: 0.25 } }} />
                    </Box>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Descriptor size / orient. bins</Typography>
                    <Box sx={{ display: "flex", gap: 0.5 }}>
                      <TextField type="number" size="small" value={siftDescSize}
                        onChange={(e) => setSiftDescSize(Math.max(2, Number(e.target.value)))}
                        inputProps={{ min: 2, max: 32 }}
                        sx={{ width: 70, "& input": { fontSize: "0.55rem", py: 0.25 } }} />
                      <TextField type="number" size="small" value={siftDescBins}
                        onChange={(e) => setSiftDescBins(Math.max(2, Number(e.target.value)))}
                        inputProps={{ min: 2, max: 32 }}
                        sx={{ width: 70, "& input": { fontSize: "0.55rem", py: 0.25 } }} />
                    </Box>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Closest/next ratio: {siftRatio.toFixed(2)}</Typography>
                    <Slider size="small" value={siftRatio} min={0.5} max={0.99} step={0.01} onChange={(_, v) => setSiftRatio(v as number)} />
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Max alignment error (px): {siftError}</Typography>
                    <Slider size="small" value={siftError} min={1} max={100} step={1} onChange={(_, v) => setSiftError(v as number)} />
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Inlier ratio: {siftInlier.toFixed(2)}</Typography>
                    <Slider size="small" value={siftInlier} min={0.01} max={0.5} step={0.01} onChange={(_, v) => setSiftInlier(v as number)} />
                    <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Expected transformation</Typography>
                    <Select size="small" value={siftTransform} onChange={(e) => setSiftTransform(e.target.value as "Translation" | "Rigid" | "Similarity" | "Affine")}
                      sx={{ fontSize: "0.55rem", "& .MuiSelect-select": { py: 0.2 } }}>
                      <MenuItem value="Translation" sx={{ fontSize: "0.55rem" }}>Translation</MenuItem>
                      <MenuItem value="Rigid"       sx={{ fontSize: "0.55rem" }}>Rigid</MenuItem>
                      <MenuItem value="Similarity"  sx={{ fontSize: "0.55rem" }}>Similarity</MenuItem>
                      <MenuItem value="Affine"      sx={{ fontSize: "0.55rem" }}>Affine</MenuItem>
                    </Select>
                    <FormControlLabel sx={{ ml: 0 }}
                      control={<Checkbox size="small" checked={siftInterpolate} onChange={(e) => setSiftInterpolate(e.target.checked)} sx={{ p: 0.25 }} />}
                      label={<Typography variant="caption" sx={{ fontSize: "0.55rem" }}>Bilinear interpolation</Typography>} />
                  </>
                )}
              </Box>
            )}
          </Box>

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
