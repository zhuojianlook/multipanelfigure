/* ──────────────────────────────────────────────────────────
   Toolbar — horizontal bar above the image strip.
   Load Images button, image count badge, Save Figure button,
   Help/About button.
   ────────────────────────────────────────────────────────── */

import { useRef, useState, useEffect } from "react";
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
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
import { getVersion } from "@tauri-apps/api/app";

// Dynamic changelog — fetched from GitHub releases on dialog open
interface ChangelogEntry {
  version: string;
  date: string;
  changes: string[];
}

let _changelogCache: ChangelogEntry[] | null = null;

async function fetchChangelog(): Promise<ChangelogEntry[]> {
  if (_changelogCache) return _changelogCache;
  try {
    const resp = await fetch(
      "https://api.github.com/repos/zhuojianlook/multipanelfigure/releases?per_page=30",
      { headers: { Accept: "application/vnd.github.v3+json" } }
    );
    if (!resp.ok) throw new Error(`GitHub API: ${resp.status}`);
    const releases = await resp.json();
    const entries: ChangelogEntry[] = [];
    for (const rel of releases) {
      const tag = (rel.tag_name || "").replace(/^(v|exp-)/, "");
      const date = (rel.published_at || "").slice(0, 10);
      const body = rel.body || "";
      // Parse commit messages from auto-generated release notes
      const changes: string[] = [];
      for (const line of body.split("\n")) {
        const trimmed = line.trim();
        // Match lines starting with * or - (markdown list items)
        if (/^[*\-]\s+/.test(trimmed)) {
          let text = trimmed.replace(/^[*\-]\s+/, "").trim();
          // Remove PR links and commit hashes
          text = text.replace(/\s+by\s+@\S+.*$/i, "").replace(/\s+in\s+https:\/\/\S+/g, "").trim();
          if (text && text.length > 5) changes.push(text);
        }
      }
      if (changes.length === 0 && body.trim()) {
        // Use the first line of the body as a single change entry
        const firstLine = body.trim().split("\n")[0].replace(/^#+\s*/, "").trim();
        if (firstLine) changes.push(firstLine);
      }
      if (tag) entries.push({ version: tag, date, changes: changes.length > 0 ? changes : ["Release " + tag] });
    }
    _changelogCache = entries;
    return entries;
  } catch {
    return [{ version: "?", date: "", changes: ["Could not fetch changelog. Check your internet connection."] }];
  }
}
import { useFigureStore } from "../../store/figureStore";
import { SaveFigureDialog } from "../dialogs/SaveFigureDialog";
import { api } from "../../api/client";

export function Toolbar() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const uploadImages = useFigureStore((s) => s.uploadImages);
  const uploadImagesFromPaths = useFigureStore((s) => s.uploadImagesFromPaths);
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
  const [appVersion, setAppVersion] = useState("...");
  const [updateChannel, setUpdateChannel] = useState<"stable" | "experimental">(() => {
    return (localStorage.getItem("mpfig_update_channel") as "stable" | "experimental") || "stable";
  });

  const toggleChannel = (channel: "stable" | "experimental") => {
    setUpdateChannel(channel);
    localStorage.setItem("mpfig_update_channel", channel);
    setUpdateStatus("idle");
  };

  const [changelog, setChangelog] = useState<ChangelogEntry[]>([]);

  useEffect(() => {
    getVersion().then((v) => setAppVersion(v)).catch(() => setAppVersion("unknown"));
  }, []);

  // Fetch changelog when About dialog opens
  useEffect(() => {
    if (aboutOpen) {
      fetchChangelog().then(setChangelog);
    }
  }, [aboutOpen]);

  const imageCount = Object.keys(loadedImages).length;

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const fileArr = Array.from(files);
    try {
      await uploadImages(fileArr);
    } catch (err) {
      console.error("Image upload failed:", err);
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleLoadMedia = async () => {
    try {
      // Try Tauri native file dialog — returns file paths, avoids base64/IPC limits
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: true,
        filters: [{
          name: "Images & Video",
          extensions: ["tif", "tiff", "png", "jpg", "jpeg", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "pef", "raf", "nd2", "mp4", "avi", "mov", "mkv", "webm", "wmv", "flv", "m4v", "mpg", "mpeg", "3gp", "ts", "mts"],
        }],
      });
      if (selected) {
        const items = Array.isArray(selected) ? selected : [selected];
        // open() may return strings or {path, name} objects depending on version
        const paths = items.map((item: unknown) =>
          typeof item === "string" ? item : (item as { path: string }).path
        ).filter(Boolean);
        if (paths.length > 0) {
          await uploadImagesFromPaths(paths);
        }
      }
    } catch {
      // If dialog import fails (dev mode), fall back to HTML file input
      fileRef.current?.click();
    }
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
        onClick={handleLoadMedia}
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
              Version {appVersion}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Created by <strong>Zhuojian Look</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
              A tool for creating professional multi-panel scientific figures
              with full control over layout, annotations, scale bars, and image adjustments. For the benefit of scientists.
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Update Channel Toggle */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>Update channel:</Typography>
            <Button
              size="small"
              variant={updateChannel === "stable" ? "contained" : "outlined"}
              onClick={() => toggleChannel("stable")}
              sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.1, px: 1, minWidth: 0 }}
            >Stable</Button>
            <Button
              size="small"
              variant={updateChannel === "experimental" ? "contained" : "outlined"}
              color="warning"
              onClick={() => toggleChannel("experimental")}
              sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.1, px: 1, minWidth: 0 }}
            >Experimental</Button>
          </Box>
          {updateChannel === "experimental" && (
            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "warning.main", textAlign: "center", mb: 0.5 }}>
              Experimental updates may contain unstable features
            </Typography>
          )}

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
                  // Fetch the correct manifest based on channel
                  const manifestFile = updateChannel === "experimental" ? "latest-experimental.json" : "latest.json";
                  const manifestUrl = `https://raw.githubusercontent.com/zhuojianlook/multipanelfigure/updater/${manifestFile}`;

                  // First check if there's an update by fetching the manifest directly
                  const manifestResp = await fetch(manifestUrl);
                  if (!manifestResp.ok) throw new Error(`Failed to fetch manifest: ${manifestResp.status}`);
                  const manifest = await manifestResp.json();
                  const latestVer = manifest.version || "";

                  // Compare versions
                  const current = appVersion.split(".").map(Number);
                  const latest = latestVer.split(".").map(Number);
                  const isNewer = latest[0] > current[0] ||
                    (latest[0] === current[0] && latest[1] > current[1]) ||
                    (latest[0] === current[0] && latest[1] === current[1] && (latest[2] || 0) > (current[2] || 0));

                  if (isNewer) {
                    setLatestVersion(latestVer);
                    setReleaseNotes(manifest.notes || "");
                    // Try Tauri updater for the actual download
                    try {
                      const update = await check();
                      if (update) {
                        setUpdateRef(update);
                      }
                    } catch {
                      // check() may fail for experimental channel — that's ok,
                      // we'll still show the update is available
                    }
                    setUpdateStatus("available");
                  } else {
                    setUpdateStatus("up-to-date");
                  }
                } catch (e: unknown) {
                  console.error("Update check failed:", e);
                  const msg = e instanceof Error ? e.message : String(e);
                  setReleaseNotes(msg);
                  setUpdateStatus("error");
                }
              }}
            >
              {updateStatus === "checking" ? "Checking..." : "Check for Updates"}
            </Button>

            {updateStatus === "up-to-date" && (
              <Alert severity="success" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                You are running the latest version ({appVersion}).
              </Alert>
            )}
            {updateStatus === "available" && (
              <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Typography sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
                  Version {latestVersion} is available!
                </Typography>
                {/* Show changelog of what's new since current version */}
                <Box sx={{ mt: 1, maxHeight: 160, overflowY: "auto" }}>
                  {changelog.filter((entry: ChangelogEntry) => {
                    // Show entries newer than current version
                    const current = appVersion.split(".").map(Number);
                    const entry_v = entry.version.split(".").map(Number);
                    for (let i = 0; i < 3; i++) {
                      if ((entry_v[i] || 0) > (current[i] || 0)) return true;
                      if ((entry_v[i] || 0) < (current[i] || 0)) return false;
                    }
                    return false;
                  }).map((entry: ChangelogEntry) => (
                    <Box key={entry.version} sx={{ mb: 1 }}>
                      <Typography sx={{ fontWeight: 600, fontSize: "0.7rem" }}>
                        v{entry.version} — {entry.date}
                      </Typography>
                      <Box component="ul" sx={{ m: 0, pl: 2, "& li": { fontSize: "0.65rem", color: "text.secondary", lineHeight: 1.4 } }}>
                        {entry.changes.map((change: string, i: number) => (
                          <li key={i}>{change}</li>
                        ))}
                      </Box>
                    </Box>
                  ))}
                </Box>
                <Button size="small" variant="contained" color="primary" sx={{ mt: 0.5, fontSize: "0.65rem", textTransform: "none" }}
                  startIcon={<DownloadIcon />}
                  onClick={async () => {
                    if (updateRef) {
                      try {
                        // Kill sidecar before update to avoid file locks
                        try {
                          const { invoke } = await import("@tauri-apps/api/core");
                          await invoke("kill_sidecar");
                        } catch { /* ignore if not available */ }
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
                      } catch (e: unknown) {
                        console.error("Update download failed:", e);
                        const errMsg = e instanceof Error ? e.message : String(e);
                        setReleaseNotes(errMsg);
                        setUpdateStatus("error");
                      }
                    } else {
                      // No Tauri updater ref (experimental channel) — open release page
                      const tag = updateChannel === "experimental" ? `exp-${latestVersion}` : `v${latestVersion}`;
                      window.open(`https://github.com/zhuojianlook/multipanelfigure/releases/tag/${tag}`, "_blank");
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
                  onClick={async () => {
                    try {
                      const { invoke } = await import("@tauri-apps/api/core");
                      await invoke("kill_sidecar");
                    } catch { /* ignore */ }
                    await relaunch();
                  }}
                >
                  Restart Now
                </Button>
              </Alert>
            )}
            {updateStatus === "error" && (
              <Alert severity="warning" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                Could not check for updates. {releaseNotes ? `Error: ${releaseNotes}` : "Please check your internet connection."}
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
              Look, Z. (2026). Multi-Panel Figure Builder (Version {appVersion}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure
            </Typography>
            <Tooltip title={citationCopied ? "Copied!" : "Copy citation"}>
              <IconButton
                size="small"
                sx={{ position: "absolute", top: 4, right: 4 }}
                onClick={() => {
                  navigator.clipboard.writeText(
                    `Look, Z. (2026). Multi-Panel Figure Builder (Version ${appVersion}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure`
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

          {/* Changelog — collapsible */}
          <Accordion disableGutters elevation={0} sx={{ bgcolor: "transparent", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0, minHeight: 32 }}>
              <Typography variant="subtitle2">Changelog</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0, pt: 0 }}>
              {changelog.length === 0 ? (
                <Typography variant="caption" sx={{ color: "text.disabled" }}>Loading changelog...</Typography>
              ) : changelog.map((entry) => (
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
            </AccordionDetails>
          </Accordion>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setAboutOpen(false); setUpdateStatus("idle"); }}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
