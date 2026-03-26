/* ──────────────────────────────────────────────────────────
   Toolbar — horizontal bar above the image strip.
   Load Images button, image count badge, Save Figure button,
   Help/About button.
   ────────────────────────────────────────────────────────── */

import { useRef, useState } from "react";
import {
  Box,
  Button,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Tooltip,
  Divider,
  CircularProgress,
  Alert,
} from "@mui/material";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import SaveIcon from "@mui/icons-material/Save";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import InfoIcon from "@mui/icons-material/Info";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import SystemUpdateAltIcon from "@mui/icons-material/SystemUpdateAlt";
import DownloadIcon from "@mui/icons-material/Download";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

const APP_VERSION = "0.1.3";
const CHANGELOG = [
  { version: "0.1.3", date: "2026-03-26", changes: [
    "Updated about section description",
  ]},
  { version: "0.1.2", date: "2026-03-26", changes: [
    "Native in-app auto-updater — download and install updates without leaving the app",
    "Restart button after update install for seamless upgrade experience",
  ]},
  { version: "0.1.1", date: "2026-03-26", changes: [
    "Updated about section description",
  ]},
  { version: "0.1.0", date: "2026-03-25", changes: [
    "Initial standalone release as native desktop app",
    "Multi-panel grid layout with drag-and-drop image management",
    "Image editing: crop, rotate, flip, brightness, contrast, levels, color adjustments",
    "Primary and secondary headers with spanning, full typography, and position control",
    "Scale bars with auto unit conversion (km to pm) and predefined scale definitions",
    "Annotations: symbols, measurement lines, area tools (rect, ellipse, custom polygon, magic wand)",
    "Zoomed insets: standard overlay, adjacent panel, and external image support",
    "Video support: frame extraction with play/seek controls",
    "Save/Load projects (.mpf) with all media bundled for sharing",
    "Export to TIFF/PNG at configurable DPI (72-600)",
    "Custom font support with global font application",
    "Dark mode interface with parking drawer for panel organization",
    "Cross-platform: macOS (Apple Silicon + Intel) and Windows",
  ]},
];
import { useFigureStore } from "../../store/figureStore";
import { SaveFigureDialog } from "../dialogs/SaveFigureDialog";
import { api } from "../../api/client";

export function Toolbar() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const uploadImages = useFigureStore((s) => s.uploadImages);
  const fileRef = useRef<HTMLInputElement>(null);
  const [saveDlgOpen, setSaveDlgOpen] = useState(false);
  const [newConfirmOpen, setNewConfirmOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(true);
  const [helpMenuAnchor, setHelpMenuAnchor] = useState<null | HTMLElement>(null);
  const [updateStatus, setUpdateStatus] = useState<"idle" | "checking" | "up-to-date" | "available" | "downloading" | "ready" | "error">("idle");
  const [latestVersion, setLatestVersion] = useState<string | null>(null);
  const [releaseNotes, setReleaseNotes] = useState("");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [updateRef, setUpdateRef] = useState<Awaited<ReturnType<typeof check>> | null>(null);
  const [citationCopied, setCitationCopied] = useState(false);

  const imageCount = Object.keys(loadedImages).length;

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const fileArr = Array.from(files);
    await uploadImages(fileArr);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1.5,
        px: 1.5,
        py: 0.75,
        borderBottom: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        flexShrink: 0,
        flexWrap: "wrap",
      }}
    >
      {/* Load images */}
      <Button
        variant="contained"
        startIcon={<AddPhotoAlternateIcon />}
        onClick={() => fileRef.current?.click()}
      >
        Load Media
      </Button>

      <input
        ref={fileRef}
        type="file"
        accept=".tif,.tiff,.png,.jpg,.jpeg,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf,.nd2,.mp4,.avi,.mov,.mkv,.webm,.wmv,.flv,.m4v,.mpg,.mpeg,.3gp,.ts,.mts"
        multiple
        style={{ display: "none" }}
        aria-label="Load image files"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {/* Badge */}
      <Chip
        label={`${imageCount} file${imageCount !== 1 ? "s" : ""}`}
        size="small"
        variant="outlined"
      />

      {/* New / Clear Session */}
      <Tooltip title="New figure — clears all panels, images, and settings">
        <Button
          size="small"
          variant="outlined"
          color="error"
          startIcon={<RestartAltIcon />}
          onClick={() => {
            setNewConfirmOpen(true);
          }}
        >
          New
        </Button>
      </Tooltip>

      {/* New figure confirmation dialog */}
      <Dialog open={newConfirmOpen} onClose={() => setNewConfirmOpen(false)}>
        <DialogTitle>New Figure</DialogTitle>
        <DialogContent>
          <Typography>Start a new figure? All current images, settings, and panels will be cleared.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewConfirmOpen(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={async () => {
            setNewConfirmOpen(false);
            try {
              // Preserve user-defined scale bars
              const savedScales = await api.getResolutions().catch(() => ({}));
              // Reset backend to fresh 2x2 grid
              await api.updateConfig({
                rows: 2, cols: 2, spacing: 0.02, output_format: "TIFF", background: "White",
                panels: [[{} as never, {} as never], [{} as never, {} as never]],
                column_labels: [
                  { text: "Column 1", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Top", rotation: 0, styled_segments: [], visible: true },
                  { text: "Column 2", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Top", rotation: 0, styled_segments: [], visible: true },
                ] as never,
                row_labels: [
                  { text: "Row 1", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Left", rotation: 90, styled_segments: [], visible: true },
                  { text: "Row 2", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Left", rotation: 90, styled_segments: [], visible: true },
                ] as never,
                column_headers: [], row_headers: [],
                resolution_entries: savedScales, dpi: 300,
              });
              // Delete all loaded images
              const imgs = await api.listImages();
              for (const name of imgs.names) {
                await api.deleteImage(name).catch(() => {});
              }
            } catch (err) {
              console.error("Clear session failed", err);
            }
            // Full reload to reset frontend state
            window.location.reload();
          }}>Confirm</Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ flex: 1 }} />

      {/* Save figure */}
      <Button
        variant="contained"
        color="secondary"
        startIcon={<SaveIcon />}
        onClick={() => setSaveDlgOpen(true)}
      >
        Save Figure
      </Button>

      {/* Help menu */}
      <Tooltip title="Help">
        <IconButton size="small" onClick={(e) => setHelpMenuAnchor(e.currentTarget)}>
          <HelpOutlineIcon sx={{ fontSize: 20 }} />
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={helpMenuAnchor}
        open={Boolean(helpMenuAnchor)}
        onClose={() => setHelpMenuAnchor(null)}
      >
        <MenuItem onClick={() => { setAboutOpen(true); setHelpMenuAnchor(null); }}>
          <InfoIcon sx={{ mr: 1, fontSize: 18 }} /> About
        </MenuItem>
      </Menu>

      <SaveFigureDialog open={saveDlgOpen} onClose={() => setSaveDlgOpen(false)} />

      {/* About Dialog */}
      <Dialog open={aboutOpen} onClose={() => { setAboutOpen(false); setUpdateStatus("idle"); }} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>About</DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: "center", py: 2 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              Multi-Panel Figure Builder
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Version {APP_VERSION}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Created by <strong>Zhuojian Look</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
              A professional tool for creating multi-panel scientific figures
              with full control over layout, annotations, scale bars, and image adjustments. For the benefit of scientists and the pursuit of science.
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Check for Updates */}
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1, mb: 2 }}>
            <Button
              variant="outlined"
              size="small"
              startIcon={updateStatus === "checking" ? <CircularProgress size={14} /> : <SystemUpdateAltIcon />}
              disabled={updateStatus === "checking" || updateStatus === "downloading"}
              onClick={async () => {
                setUpdateStatus("checking");
                setLatestVersion(null);
                setUpdateRef(null);
                try {
                  const update = await check();
                  if (update) {
                    setLatestVersion(update.version);
                    setReleaseNotes(update.body || "");
                    setUpdateRef(update);
                    setUpdateStatus("available");
                  } else {
                    setUpdateStatus("up-to-date");
                  }
                } catch (e) {
                  console.error("Update check failed:", e);
                  setUpdateStatus("error");
                }
              }}
            >
              {updateStatus === "checking" ? "Checking..." : "Check for Updates"}
            </Button>

            {updateStatus === "up-to-date" && (
              <Alert severity="success" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                You are running the latest version ({APP_VERSION}).
              </Alert>
            )}
            {updateStatus === "available" && (
              <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Typography sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
                  Version {latestVersion} is available!
                </Typography>
                {releaseNotes && (
                  <Typography sx={{ fontSize: "0.65rem", mt: 0.5, maxHeight: 80, overflowY: "auto", whiteSpace: "pre-wrap", color: "text.secondary" }}>
                    {releaseNotes}
                  </Typography>
                )}
                <Button size="small" variant="contained" color="primary" sx={{ mt: 0.5, fontSize: "0.65rem", textTransform: "none" }}
                  startIcon={<DownloadIcon />}
                  onClick={async () => {
                    if (!updateRef) return;
                    try {
                      setUpdateStatus("downloading");
                      setDownloadProgress(0);
                      let downloaded = 0;
                      await updateRef.downloadAndInstall((event) => {
                        if (event.event === "Started" && event.data.contentLength) {
                          setDownloadProgress(0);
                        } else if (event.event === "Progress") {
                          downloaded += event.data.chunkLength;
                          setDownloadProgress(downloaded);
                        } else if (event.event === "Finished") {
                          setDownloadProgress(100);
                        }
                      });
                      setUpdateStatus("ready");
                    } catch (e) {
                      console.error("Update download failed:", e);
                      setUpdateStatus("error");
                    }
                  }}
                >
                  Download &amp; Install Update
                </Button>
              </Alert>
            )}
            {updateStatus === "downloading" && (
              <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <CircularProgress size={16} />
                  <Typography sx={{ fontSize: "0.75rem" }}>
                    Downloading update... {downloadProgress > 0 ? `(${(downloadProgress / 1024 / 1024).toFixed(1)} MB)` : ""}
                  </Typography>
                </Box>
              </Alert>
            )}
            {updateStatus === "ready" && (
              <Alert severity="success" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Typography sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
                  Update installed! Restart to apply.
                </Typography>
                <Button size="small" variant="contained" color="success" sx={{ mt: 0.5, fontSize: "0.65rem", textTransform: "none" }}
                  onClick={async () => { await relaunch(); }}
                >
                  Restart Now
                </Button>
              </Alert>
            )}
            {updateStatus === "error" && (
              <Alert severity="warning" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                Could not check for updates. Please check your internet connection.
              </Alert>
            )}
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Citation */}
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Citation</Typography>
          <Box sx={{
            bgcolor: "action.hover",
            borderRadius: 1,
            p: 1.5,
            mb: 2,
            position: "relative",
            fontFamily: "monospace",
            fontSize: "0.7rem",
            lineHeight: 1.5,
            color: "text.secondary",
          }}>
            <Typography sx={{ fontFamily: "inherit", fontSize: "inherit", lineHeight: "inherit", color: "inherit" }}>
              Look, Z. (2026). Multi-Panel Figure Builder (Version {APP_VERSION}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure
            </Typography>
            <Tooltip title={citationCopied ? "Copied!" : "Copy citation"}>
              <IconButton
                size="small"
                sx={{ position: "absolute", top: 4, right: 4 }}
                onClick={() => {
                  navigator.clipboard.writeText(
                    `Look, Z. (2026). Multi-Panel Figure Builder (Version ${APP_VERSION}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure`
                  );
                  setCitationCopied(true);
                  setTimeout(() => setCitationCopied(false), 2000);
                }}
              >
                <ContentCopyIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Changelog */}
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Changelog</Typography>
          {CHANGELOG.map((entry) => (
            <Box key={entry.version} sx={{ mb: 1.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                v{entry.version} — {entry.date}
              </Typography>
              <Box component="ul" sx={{ m: 0, pl: 2.5, "& li": { fontSize: "0.75rem", color: "text.secondary" } }}>
                {entry.changes.map((change, i) => (
                  <li key={i}>{change}</li>
                ))}
              </Box>
            </Box>
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setAboutOpen(false); setUpdateStatus("idle"); }}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
