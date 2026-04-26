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
import LibraryAddIcon from "@mui/icons-material/LibraryAdd";
import DashboardIcon from "@mui/icons-material/Dashboard";
import ViewModuleIcon from "@mui/icons-material/ViewModule";
import { useCollageStore } from "../../store/collageStore";
import { api } from "../../api/client";
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

/** Fetch a URL via Rust proxy (bypasses WebView CORS), fallback to browser fetch */
async function proxyFetch(url: string): Promise<string> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke("fetch_url", { url }) as string;
  } catch {
    const resp = await fetch(url);
    return resp.text();
  }
}

async function fetchChangelog(): Promise<ChangelogEntry[]> {
  if (_changelogCache) return _changelogCache;
  try {
    const REPO = "zhuojianlook/multipanelfigure";
    const relText = await proxyFetch(`https://api.github.com/repos/${REPO}/releases?per_page=30`);
    const releases = JSON.parse(relText);
    if (!Array.isArray(releases)) throw new Error("Invalid releases response");

    // Fetch recent commits to extract commit messages (much more useful than release notes)
    let commitMessages: Record<string, string[]> = {};
    try {
      const commitsText = await proxyFetch(`https://api.github.com/repos/${REPO}/commits?per_page=100`);
      const commits = JSON.parse(commitsText);
      if (Array.isArray(commits)) {
        // Group commits by their closest tag (based on release dates)
        const tagDates = releases.map((r: { tag_name: string; published_at: string }) => ({
          tag: r.tag_name,
          date: new Date(r.published_at).getTime(),
        })).sort((a: { date: number }, b: { date: number }) => b.date - a.date);

        for (const commit of commits) {
          const msg = (commit.commit?.message || "").split("\n")[0].trim();
          if (!msg || msg.startsWith("Merge") || msg.includes("Co-Authored-By")) continue;
          // Clean up: remove conventional commit prefixes for readability
          const cleaned = msg.replace(/^(feat|fix|chore|docs|refactor|style|test|ci|build)(\(.+?\))?:\s*/i, "").trim();
          if (cleaned.length < 8) continue;
          // Find which release this commit belongs to
          const commitDate = new Date(commit.commit?.author?.date || "").getTime();
          let assignedTag = tagDates[0]?.tag || "";
          for (let i = 0; i < tagDates.length - 1; i++) {
            if (commitDate <= tagDates[i].date && commitDate > tagDates[i + 1].date) {
              assignedTag = tagDates[i].tag;
              break;
            }
          }
          if (assignedTag) {
            if (!commitMessages[assignedTag]) commitMessages[assignedTag] = [];
            if (commitMessages[assignedTag].length < 8) { // cap per release
              commitMessages[assignedTag].push(cleaned);
            }
          }
        }
      }
    } catch { /* commits fetch failed, fall back to release body */ }

    const entries: ChangelogEntry[] = [];
    for (const rel of releases) {
      const tagName = rel.tag_name || "";
      const version = tagName.replace(/^(v|exp-)/, "");
      const date = (rel.published_at || "").slice(0, 10);
      const isExp = tagName.startsWith("exp-");

      // Use commit messages if available, otherwise parse release body
      let changes: string[] = commitMessages[tagName] || [];

      if (changes.length === 0) {
        // Parse release body for meaningful lines
        const body = rel.body || "";
        for (const line of body.split("\n")) {
          const trimmed = line.trim();
          if (/^[*\-]\s+/.test(trimmed)) {
            let text = trimmed.replace(/^[*\-]\s+/, "").trim();
            text = text.replace(/\s+by\s+@\S+.*$/i, "").replace(/\s+in\s+https:\/\/\S+/g, "").trim();
            // Strip conventional-commit prefix (feat/fix/chore/...) so the
            // changelog reads cleanly. CI now writes release bodies from
            // git log including those prefixes; this is the corresponding
            // display-side cleanup.
            text = text.replace(/^(feat|fix|chore|docs|refactor|style|test|ci|build|perf)(\(.+?\))?:\s*/i, "").trim();
            // Skip "Full Changelog" links
            if (text.includes("Full Changelog") || text.includes("github.com/compare")) continue;
            if (text.length > 5) changes.push(text);
          }
        }
      }

      if (version) {
        const label = isExp ? `${version} (experimental)` : version;
        entries.push({
          version: label,
          date,
          changes: changes.length > 0 ? changes : [`Release ${tagName}`],
        });
      }
    }
    _changelogCache = entries;
    return entries;
  } catch (e) {
    console.error("Changelog fetch failed:", e);
    // Return a static fallback changelog
    return [
      { version: "0.1.70", date: "2026-04-16", changes: [
        "Stable/experimental update channels with channel switcher",
        "Dynamic changelog fetched from GitHub releases",
        "Z-stack TIFF slice selection",
        "Drag-and-drop files from OS into timeline",
        "Right-click to copy preview to clipboard",
      ]},
      { version: "0.1.57", date: "2026-04-08", changes: [
        "Preview pan & zoom with controls",
        "Header margin fixes for all positions",
        "Grid horizontal scrolling, 50 row/col limit",
        "R analysis integration with presets",
        "Media groups for organizing images",
      ]},
      { version: "0.1.0", date: "2026-03-25", changes: [
        "Initial release: multi-panel scientific figure builder",
      ]},
    ];
  }
}
import { useFigureStore } from "../../store/figureStore";
import { SaveFigureDialog } from "../dialogs/SaveFigureDialog";

/* ── SaveCollageButton ───────────────────────────────────────
   Renders the collage canvas to PNG client-side (compositing
   each item at its x/y/w/h on a single offscreen <canvas>),
   then writes the bytes to a user-chosen path via the
   existing save_base64_to_path Tauri command. In a non-Tauri
   browser preview, falls back to a download anchor. */
function SaveCollageButton() {
  const items = useCollageStore((s) => s.items);
  const canvasW = useCollageStore((s) => s.canvasW);
  const canvasH = useCollageStore((s) => s.canvasH);
  const background = useCollageStore((s) => s.background);

  const handleSave = async () => {
    if (items.length === 0) {
      window.alert("No items in the collage to save. Add a figure or image first.");
      return;
    }
    // Compose items onto an offscreen canvas at full virtual resolution.
    // We load each <img> first (they may already be cached from the data
    // URL paint, but Image() resolves the decode lifecycle cleanly) so
    // ctx.drawImage doesn't draw blank tiles for any straggler.
    const canvas = document.createElement("canvas");
    canvas.width = canvasW;
    canvas.height = canvasH;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      window.alert("Could not initialise a 2D canvas context.");
      return;
    }
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvasW, canvasH);

    const sorted = [...items].sort((a, b) => a.z - b.z);
    for (const it of sorted) {
      await new Promise<void>((resolve) => {
        const img = new window.Image();
        img.onload = () => {
          try {
            ctx.drawImage(img, it.x, it.y, it.w, it.h);
          } catch (err) {
            console.warn("[collage] drawImage failed for", it.name, err);
          }
          resolve();
        };
        img.onerror = () => resolve();
        img.src = it.src;
      });
    }

    const dataUrl = canvas.toDataURL("image/png");
    const b64 = dataUrl.split(",")[1] ?? "";

    // Try Tauri save flow first.
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const { invoke } = await import("@tauri-apps/api/core");
      const path = await save({
        defaultPath: "collage.png",
        filters: [{ name: "PNG image", extensions: ["png"] }],
      });
      if (!path) return;
      await invoke("save_base64_to_path", { path, dataB64: b64 });
      window.alert(`Collage saved to ${path}`);
      return;
    } catch {
      /* Not running inside Tauri — fall back to browser download. */
    }
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = "collage.png";
    a.click();
  };

  return (
    <Button
      variant="contained"
      color="secondary"
      startIcon={<SaveIcon />}
      onClick={handleSave}
    >
      Save Collage
    </Button>
  );
}

/* ── CollageWorkspaceControls ────────────────────────────────
   Workspace toggle (Builder ↔ Collage) and the "Add to Collage"
   action that captures the currently-rendered figure preview
   and pushes it into the collage store. */
function CollageWorkspaceControls() {
  const mode = useCollageStore((s) => s.mode);
  const setMode = useCollageStore((s) => s.setMode);
  const addItem = useCollageStore((s) => s.addItem);
  const itemCount = useCollageStore((s) => s.items.length);
  const configDirty = useFigureStore((s) => s.configDirty);

  const handleAddToCollage = async () => {
    if (configDirty) {
      const ok = window.confirm(
        "You have unsaved changes in the multi-panel builder.\n\n" +
        "Adding to the collage uses the current rendered preview, but the " +
        "underlying project file is not saved. Use Sidebar → Save Project " +
        "first if you want to keep this exact state recoverable.\n\n" +
        "Continue and add to collage?",
      );
      if (!ok) return;
    }
    try {
      // Render at full preview quality (matplotlib pipeline) so the
      // collage shows the same labels/headers/scale bars the user sees.
      const resp = await api.getPreview();
      if (!resp?.image) {
        window.alert("Preview is empty — add some images to your panels first.");
        return;
      }
      const naturalW = resp.width || 0;
      const naturalH = resp.height || 0;
      const aspect = naturalH > 0 ? naturalW / naturalH : 1;
      // Default insert size: keep a sensible visual scale on a 1600×1200
      // canvas. Use 600 px on the longer axis; subsequent items can be
      // resized via the corner handle.
      const targetMax = 600;
      const w = aspect >= 1 ? targetMax : targetMax * aspect;
      const h = aspect >= 1 ? targetMax / aspect : targetMax;
      const offset = itemCount * 24;
      addItem({
        kind: "figure",
        src: `data:image/png;base64,${resp.image}`,
        name: `Figure ${itemCount + 1}`,
        x: 40 + offset,
        y: 40 + offset,
        w,
        h,
        naturalW,
        naturalH,
      });
      setMode("collage");
    } catch (e) {
      console.error("Add to collage failed:", e);
      window.alert("Failed to capture the figure preview. Check the console.");
    }
  };

  return (
    <>
      <Tooltip title={mode === "collage" ? "Back to Multi-Panel Builder" : "Open Collage Assembly"}>
        <Button
          variant={mode === "collage" ? "contained" : "outlined"}
          color="primary"
          size="small"
          startIcon={mode === "collage" ? <ViewModuleIcon /> : <DashboardIcon />}
          onClick={() => setMode(mode === "collage" ? "builder" : "collage")}
          sx={{ textTransform: "none" }}
        >
          {mode === "collage" ? "Multi-Panel Builder" : "Collage Assembly"}
        </Button>
      </Tooltip>
      {mode === "builder" && (
        <Tooltip title="Render the current figure and add it to the Collage Assembly">
          <Button
            variant="outlined"
            color="primary"
            size="small"
            startIcon={<LibraryAddIcon />}
            onClick={handleAddToCollage}
            sx={{ textTransform: "none" }}
          >
            Add to Collage
          </Button>
        </Tooltip>
      )}
    </>
  );
}

export function Toolbar() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const uploadImages = useFigureStore((s) => s.uploadImages);
  const uploadImagesFromPaths = useFigureStore((s) => s.uploadImagesFromPaths);
  const mode = useCollageStore((s) => s.mode);
  const fileRef = useRef<HTMLInputElement>(null);
  const [saveDlgOpen, setSaveDlgOpen] = useState(false);
  const [newConfirmOpen, setNewConfirmOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(true);
  const [helpMenuAnchor, setHelpMenuAnchor] = useState<null | HTMLElement>(null);
  const [updateStatus, setUpdateStatus] = useState<"idle" | "checking" | "up-to-date" | "available" | "downloading" | "ready" | "error">("idle");
  const [latestVersion, setLatestVersion] = useState<string | null>(null);
  const [releaseNotes, setReleaseNotes] = useState("");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadTotal, setDownloadTotal] = useState<number | null>(null);
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
      {/* The Load Media / file-count chip / New trio belongs to the
          multi-panel builder workflow only. In collage mode they're
          hidden — the collage has its own Import image / Import
          project buttons inside CollageView's toolbar. */}
      {mode === "builder" && (
        <>
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

          <Chip
            label={`${imageCount} file${imageCount !== 1 ? "s" : ""}`}
            size="small"
            variant="outlined"
          />

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
        </>
      )}

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

      {/* Collage workspace toggle + Add to Collage */}
      <CollageWorkspaceControls />

      {/* Save figure / Save collage — same button, two modes. The
          collage path renders the canvas client-side via <canvas>
          and ships the bytes through the existing save_base64_to_path
          Tauri command (or download fallback in browser). */}
      {mode === "builder" ? (
        <Button
          variant="contained"
          color="secondary"
          startIcon={<SaveIcon />}
          onClick={() => setSaveDlgOpen(true)}
        >
          Save Figure
        </Button>
      ) : (
        <SaveCollageButton />
      )}

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

                  // Fetch manifest via Rust proxy (WebView blocks cross-origin)
                  const manifestText = await proxyFetch(manifestUrl);
                  const manifest = JSON.parse(manifestText) as { version: string; notes: string };
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
                You are running the latest {updateChannel} version ({appVersion}).
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
                    const { invoke } = await import("@tauri-apps/api/core");
                    const { listen } = await import("@tauri-apps/api/event");
                    try {
                      try { await invoke("kill_sidecar"); } catch { /* ignore */ }
                      setUpdateStatus("downloading");
                      setDownloadProgress(0);
                      setDownloadTotal(null);

                      if (updateChannel === "stable" && updateRef) {
                        let downloaded = 0;
                        await updateRef.downloadAndInstall((event) => {
                          if (event.event === "Started") {
                            setDownloadProgress(0);
                            setDownloadTotal(event.data.contentLength ?? null);
                          } else if (event.event === "Progress") {
                            downloaded += event.data.chunkLength;
                            setDownloadProgress(downloaded);
                          }
                        });
                      } else {
                        // Experimental: use Rust command with custom endpoint
                        // Listen for the same progress events the Rust side
                        // emits, so the UI can show "X MB / Y MB" exactly
                        // like the stable path.
                        const unlisten = await listen<{ downloaded: number; total: number | null }>(
                          "updater://progress",
                          (e) => {
                            setDownloadProgress(e.payload.downloaded);
                            if (e.payload.total) setDownloadTotal(e.payload.total);
                          }
                        );
                        const manifestFile = updateChannel === "experimental" ? "latest-experimental.json" : "latest.json";
                        const manifestUrl = `https://raw.githubusercontent.com/zhuojianlook/multipanelfigure/updater/${manifestFile}`;
                        const timeoutMs = 3 * 60 * 1000;
                        const timeoutPromise = new Promise((_, reject) =>
                          setTimeout(() => reject(new Error("Download timed out after 3 minutes")), timeoutMs)
                        );
                        try {
                          await Promise.race([
                            invoke("download_and_install_update", { manifestUrl }),
                            timeoutPromise,
                          ]);
                        } finally {
                          unlisten();
                        }
                      }
                      setUpdateStatus("ready");
                    } catch (e: unknown) {
                      console.error("Update failed:", e);
                      const errMsg = e instanceof Error ? e.message : String(e);
                      // On failure, offer browser download fallback
                      const tag = updateChannel === "experimental" ? `exp-${latestVersion}` : `v${latestVersion}`;
                      const releaseUrl = `https://github.com/zhuojianlook/multipanelfigure/releases/tag/${tag}`;
                      try {
                        const { open } = await import("@tauri-apps/plugin-shell");
                        await open(releaseUrl);
                        setReleaseNotes(`In-app update failed (${errMsg}). Opened browser for manual download.`);
                      } catch {
                        setReleaseNotes(errMsg);
                      }
                      setUpdateStatus("error");
                    }
                  }}
                >
                  Download & Install Update
                </Button>
              </Alert>
            )}
            {updateStatus === "downloading" && (() => {
              const dlMB = downloadProgress / 1024 / 1024;
              const totalMB = downloadTotal ? downloadTotal / 1024 / 1024 : null;
              const pct = downloadTotal && downloadTotal > 0
                ? Math.min(100, Math.round((downloadProgress / downloadTotal) * 100))
                : null;
              const text = downloadProgress > 0
                ? (totalMB != null
                    ? `(${dlMB.toFixed(1)} MB / ${totalMB.toFixed(1)} MB${pct != null ? ` — ${pct}%` : ""})`
                    : `(${dlMB.toFixed(1)} MB)`)
                : "";
              return (
                <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <CircularProgress size={16} />
                    <Typography sx={{ fontSize: "0.75rem" }}>
                      Downloading update... {text}
                    </Typography>
                  </Box>
                </Alert>
              );
            })()}
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
