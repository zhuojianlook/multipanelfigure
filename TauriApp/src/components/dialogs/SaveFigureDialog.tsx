/* ──────────────────────────────────────────────────────────
   SaveFigureDialog — MUI Dialog for saving the final figure.
   File path input, format selector, quality slider.
   Adds a "Video" tab when any panel has play_range enabled.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect, useRef, useMemo } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Select,
  MenuItem,
  Slider,
  TextField,
  Box,
  Typography,
  FormControl,
  FormControlLabel,
  Checkbox,
  InputLabel,
  Alert,
  Tabs,
  Tab,
  LinearProgress,
} from "@mui/material";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import { useFigureStore } from "../../store/figureStore";
import { api } from "../../api/client";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function SaveFigureDialog({ open, onClose }: Props) {
  const saveFigure = useFigureStore((s) => s.saveFigure);
  const configBg = useFigureStore((s) => s.config?.background ?? "White");
  const config = useFigureStore((s) => s.config);

  // Detect any panel with play_range enabled — gates the Video tab.
  const playRangePanels = useMemo(() => {
    if (!config) return [] as Array<{ row: number; col: number; image_name: string; frame_start: number; frame_end: number }>;
    const out: Array<{ row: number; col: number; image_name: string; frame_start: number; frame_end: number }> = [];
    for (let r = 0; r < config.rows; r++) {
      for (let c = 0; c < config.cols; c++) {
        const p = config.panels[r]?.[c];
        if (p?.play_range && p.image_name) {
          out.push({
            row: r,
            col: c,
            image_name: p.image_name,
            frame_start: p.frame_start ?? 0,
            frame_end: p.frame_end ?? 0,
          });
        }
      }
    }
    return out;
  }, [config]);
  const hasVideoExport = playRangePanels.length > 0;
  const longestRangeFrames = playRangePanels.reduce(
    (m, p) => Math.max(m, p.frame_end - p.frame_start + 1), 0,
  );

  // Tab: 0 = Image, 1 = Video. Auto-pick Image when no play_range panels.
  const [tab, setTab] = useState<0 | 1>(0);
  useEffect(() => {
    if (!hasVideoExport && tab === 1) setTab(0);
  }, [hasVideoExport, tab]);

  // Image-export state
  const [format, setFormat] = useState("TIFF");
  const [quality, setQuality] = useState(95);
  const [dpi, setDpi] = useState(300);
  const today = new Date();
  const yyyymmdd = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
  const [filePath, setFilePath] = useState("");
  const [saveError, setSaveError] = useState("");
  const [saving, setSaving] = useState(false);

  // Video-export state
  const [videoFormat, setVideoFormat] = useState<"mp4" | "avi">("mp4");
  const [videoFps, setVideoFps] = useState(30);
  const [videoDpi, setVideoDpi] = useState(150);
  const [videoFilePath, setVideoFilePath] = useState("");
  const [retainAudio, setRetainAudio] = useState(false);
  const [audioPanelName, setAudioPanelName] = useState<string>("");
  const [ffmpegAvailable, setFfmpegAvailable] = useState<boolean | null>(null);
  const [renderProgress, setRenderProgress] = useState<{ current: number; total: number } | null>(null);
  const [renderJobId, setRenderJobId] = useState<string | null>(null);
  const [renderError, setRenderError] = useState("");
  const [confirmLargeJob, setConfirmLargeJob] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const ext = format.toLowerCase() === "png" ? "png" : format.toLowerCase() === "jpeg" ? "jpg" : "tiff";
  const videoExt = videoFormat;

  // Detect ffmpeg on dialog open so the audio option can be hidden if missing.
  useEffect(() => {
    if (!open) return;
    api.renderVideoFfmpegAvailable().then((r) => setFfmpegAvailable(!!r.available)).catch(() => setFfmpegAvailable(false));
  }, [open]);

  // Auto-populate filenames on open
  useEffect(() => {
    if (open) {
      setFilePath((prev) => prev?.trim() ? prev : `${yyyymmdd}_auto.${ext}`);
      setVideoFilePath((prev) => prev?.trim() ? prev : `${yyyymmdd}_auto.${videoExt}`);
      setSaveError("");
      setRenderError("");
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update extension for image
  useEffect(() => {
    setFilePath((prev) => {
      if (!prev) return `${yyyymmdd}_auto.${ext}`;
      const parts = prev.split(".");
      if (parts.length > 1) { parts[parts.length - 1] = ext; return parts.join("."); }
      return `${prev}.${ext}`;
    });
  }, [format]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update extension for video
  useEffect(() => {
    setVideoFilePath((prev) => {
      if (!prev) return `${yyyymmdd}_auto.${videoExt}`;
      const parts = prev.split(".");
      if (parts.length > 1) { parts[parts.length - 1] = videoExt; return parts.join("."); }
      return `${prev}.${videoExt}`;
    });
  }, [videoFormat]); // eslint-disable-line react-hooks/exhaustive-deps

  // Default the audio panel selector to the first play_range panel.
  useEffect(() => {
    if (playRangePanels.length > 0 && !audioPanelName) {
      setAudioPanelName(playRangePanels[0].image_name);
    }
  }, [playRangePanels, audioPanelName]);

  const handleBrowse = async (target: "image" | "video") => {
    const isVideo = target === "video";
    const currentPath = isVideo ? videoFilePath : filePath;
    const currentExt = isVideo ? videoExt : ext;
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const selected = await save({
        defaultPath: currentPath || `${yyyymmdd}_auto.${currentExt}`,
        filters: [{ name: isVideo ? "Video" : "Image", extensions: [currentExt] }],
      });
      if (selected) {
        if (isVideo) setVideoFilePath(selected); else setFilePath(selected);
        setSaveError(""); setRenderError("");
        return;
      }
    } catch { /* not in Tauri */ }
    const fname = (currentPath || "").split("/").pop() || `${yyyymmdd}_auto.${currentExt}`;
    if (isVideo) setVideoFilePath(`~/Documents/${fname}`);
    else setFilePath(`~/Documents/${fname}`);
    setSaveError(""); setRenderError("");
  };

  const handleSave = async () => {
    const path = filePath.trim();
    if (!path) { setSaveError("Please specify a file path before saving."); return; }
    if (!path.includes("/") && !path.includes("\\")) {
      setSaveError("Please specify a full file path (e.g., /Users/you/Desktop/figure.tiff)");
      return;
    }
    setSaveError("");
    setSaving(true);
    try {
      await saveFigure(path, format, configBg, dpi);
      onClose();
    } catch (err) {
      setSaveError(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => stopPolling();
  }, []);

  const handleRenderVideo = async (skipLargeWarning = false) => {
    const path = videoFilePath.trim();
    if (!path) { setRenderError("Please specify a video file path before rendering."); return; }
    if (!path.includes("/") && !path.includes("\\")) {
      setRenderError("Please specify a full file path (e.g., /Users/you/Desktop/figure.mp4)");
      return;
    }
    if (longestRangeFrames > 1800 && !skipLargeWarning) {
      setConfirmLargeJob(true);
      return;
    }
    setRenderError("");
    setRenderProgress({ current: 0, total: longestRangeFrames });
    try {
      const { job_id, total_frames } = await api.renderVideoStart(
        path, videoFormat, videoFps, configBg, videoDpi,
        retainAudio && audioPanelName ? audioPanelName : null,
      );
      setRenderJobId(job_id);
      setRenderProgress({ current: 0, total: total_frames });
      // Poll progress every 500 ms
      pollRef.current = setInterval(async () => {
        try {
          const r = await api.renderVideoProgress(job_id);
          setRenderProgress({ current: r.current, total: r.total });
          if (r.status === "done") {
            stopPolling(); setRenderJobId(null); setRenderProgress(null); onClose();
          } else if (r.status === "error") {
            stopPolling(); setRenderJobId(null);
            setRenderError(`Render failed: ${r.error || "unknown error"}`);
          }
        } catch (err) {
          stopPolling(); setRenderJobId(null);
          setRenderError(`Progress poll failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }, 500);
    } catch (err) {
      setRenderProgress(null);
      setRenderError(`Render start failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleCancelRender = async () => {
    if (!renderJobId) return;
    try { await api.renderVideoCancel(renderJobId); } catch { /* ignore */ }
    stopPolling();
    setRenderJobId(null);
    setRenderProgress(null);
  };

  const isRendering = renderProgress !== null && renderJobId !== null;

  return (
    <Dialog open={open} onClose={isRendering ? undefined : onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Save Figure</DialogTitle>
      <DialogContent>
        {hasVideoExport && (
          <Tabs value={tab} onChange={(_, v) => setTab(v as 0 | 1)} sx={{ mb: 1 }}>
            <Tab label="Image" />
            <Tab label="Video" />
          </Tabs>
        )}

        {tab === 0 && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, mt: 1 }}>
            {/* File path */}
            <Box>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                <TextField
                  fullWidth size="small"
                  label="File path"
                  placeholder={`/Users/you/Desktop/figure.${ext}`}
                  value={filePath}
                  onChange={(e) => { setFilePath(e.target.value); setSaveError(""); }}
                />
                <Button variant="outlined" size="small" onClick={() => handleBrowse("image")}
                  sx={{ minWidth: 80, flexShrink: 0 }} startIcon={<FolderOpenIcon />}>Browse</Button>
              </Box>
              <Typography variant="caption" sx={{ color: "text.secondary", ml: 1.5, mt: 0.25, display: "block", fontSize: "0.65rem" }}>
                Full path including filename and extension
              </Typography>
            </Box>

            {/* Format */}
            <FormControl fullWidth size="small">
              <InputLabel>Output Format</InputLabel>
              <Select value={format} label="Output Format" onChange={(e) => setFormat(e.target.value)}>
                <MenuItem value="TIFF">TIFF (lossless, max quality)</MenuItem>
                <MenuItem value="PNG">PNG (lossless)</MenuItem>
                <MenuItem value="JPEG">JPEG</MenuItem>
              </Select>
            </FormControl>

            {/* DPI */}
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                DPI: {dpi} — {dpi <= 150 ? "screen" : dpi <= 300 ? "publication" : "high-resolution"}
              </Typography>
              <Slider value={dpi} min={72} max={600} step={1}
                marks={[{ value: 72, label: "72" }, { value: 150, label: "150" }, { value: 300, label: "300" }, { value: 600, label: "600" }]}
                onChange={(_, val) => setDpi(val as number)} />
            </Box>

            {/* JPEG quality */}
            {format === "JPEG" && (
              <Box>
                <Typography variant="caption" color="text.secondary" gutterBottom>Quality: {quality}</Typography>
                <Slider value={quality} min={1} max={100} onChange={(_, val) => setQuality(val as number)} />
              </Box>
            )}
          </Box>
        )}

        {tab === 1 && hasVideoExport && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5, mt: 1 }}>
            <Alert severity="info" sx={{ py: 0.5 }}>
              {playRangePanels.length} panel{playRangePanels.length === 1 ? "" : "s"} will animate.
              Longest range: {longestRangeFrames} frames
              {videoFps > 0 && ` (~${(longestRangeFrames / videoFps).toFixed(1)} s at ${videoFps} fps)`}.
              Shorter panels freeze on their last frame.
            </Alert>

            {/* File path */}
            <Box>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                <TextField fullWidth size="small" label="Video file path"
                  placeholder={`/Users/you/Desktop/figure.${videoExt}`}
                  value={videoFilePath}
                  onChange={(e) => { setVideoFilePath(e.target.value); setRenderError(""); }} />
                <Button variant="outlined" size="small" onClick={() => handleBrowse("video")}
                  sx={{ minWidth: 80, flexShrink: 0 }} startIcon={<FolderOpenIcon />}>Browse</Button>
              </Box>
            </Box>

            {/* Format */}
            <FormControl fullWidth size="small">
              <InputLabel>Video Format</InputLabel>
              <Select value={videoFormat} label="Video Format" onChange={(e) => setVideoFormat(e.target.value as "mp4" | "avi")}>
                <MenuItem value="mp4">MP4 (H.264-compatible, recommended)</MenuItem>
                <MenuItem value="avi">AVI (MJPEG)</MenuItem>
              </Select>
            </FormControl>

            {/* FPS */}
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>Frame rate: {videoFps} fps</Typography>
              <Slider value={videoFps} min={1} max={60} step={1}
                marks={[{ value: 12, label: "12" }, { value: 24, label: "24" }, { value: 30, label: "30" }, { value: 60, label: "60" }]}
                onChange={(_, v) => setVideoFps(v as number)} />
            </Box>

            {/* DPI */}
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Render DPI: {videoDpi} — higher = sharper labels but larger file & slower render
              </Typography>
              <Slider value={videoDpi} min={72} max={300} step={1}
                marks={[{ value: 72, label: "72" }, { value: 150, label: "150" }, { value: 300, label: "300" }]}
                onChange={(_, v) => setVideoDpi(v as number)} />
            </Box>

            {/* Audio retention */}
            {ffmpegAvailable !== false && (
              <Box>
                <FormControlLabel
                  control={<Checkbox checked={retainAudio} onChange={(e) => setRetainAudio(e.target.checked)} size="small" />}
                  label={<Typography variant="caption">Retain audio from a panel</Typography>}
                />
                {retainAudio && (
                  <FormControl fullWidth size="small" sx={{ mt: 0.5 }}>
                    <InputLabel>Audio source panel</InputLabel>
                    <Select value={audioPanelName} label="Audio source panel"
                      onChange={(e) => setAudioPanelName(e.target.value)}>
                      {playRangePanels.map((p) => (
                        <MenuItem key={`${p.row}-${p.col}-${p.image_name}`} value={p.image_name}>
                          R{p.row + 1} C{p.col + 1} — {p.image_name} (frames {p.frame_start}–{p.frame_end})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              </Box>
            )}
            {ffmpegAvailable === false && (
              <Alert severity="info" sx={{ py: 0.5 }}>
                Install ffmpeg and add it to your PATH to enable audio retention.
              </Alert>
            )}

            {/* Render progress */}
            {isRendering && renderProgress && (
              <Box>
                <Typography variant="caption" color="text.secondary" gutterBottom>
                  Rendering frame {renderProgress.current} / {renderProgress.total}
                  {renderProgress.total > 0 && ` (${Math.round(100 * renderProgress.current / renderProgress.total)}%)`}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={renderProgress.total > 0 ? (100 * renderProgress.current / renderProgress.total) : 0}
                />
              </Box>
            )}

            {/* Confirm large job */}
            {confirmLargeJob && (
              <Alert
                severity="warning"
                action={
                  <Box sx={{ display: "flex", gap: 0.5 }}>
                    <Button size="small" onClick={() => setConfirmLargeJob(false)}>Cancel</Button>
                    <Button size="small" color="warning" variant="contained"
                      onClick={() => { setConfirmLargeJob(false); handleRenderVideo(true); }}>
                      Render anyway
                    </Button>
                  </Box>
                }
              >
                {longestRangeFrames} frames is a large render — this may take several minutes and use significant disk space. Continue?
              </Alert>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        {tab === 0 && saveError && <Alert severity="error" sx={{ mr: 2, py: 0, flex: 1 }}>{saveError}</Alert>}
        {tab === 1 && renderError && <Alert severity="error" sx={{ mr: 2, py: 0, flex: 1 }}>{renderError}</Alert>}
        <Button onClick={isRendering ? undefined : onClose} disabled={isRendering}>Cancel</Button>
        {tab === 0 && (
          <Button variant="contained" color="secondary" onClick={handleSave} disabled={saving || !filePath.trim()}>
            {saving ? "Saving..." : "Save"}
          </Button>
        )}
        {tab === 1 && hasVideoExport && (
          isRendering
            ? <Button variant="outlined" color="error" onClick={handleCancelRender}>Cancel render</Button>
            : <Button variant="contained" color="secondary" onClick={() => handleRenderVideo()} disabled={!videoFilePath.trim() || confirmLargeJob}>
                Render video
              </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
