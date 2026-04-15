/* ──────────────────────────────────────────────────────────
   SaveFigureDialog — MUI Dialog for saving the final figure.
   File path input, format selector, quality slider.
   Uses Tauri dialog plugin for native file picker.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect } from "react";
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
  InputLabel,
  Alert,
} from "@mui/material";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import { useFigureStore } from "../../store/figureStore";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function SaveFigureDialog({ open, onClose }: Props) {
  const saveFigure = useFigureStore((s) => s.saveFigure);
  const configBg = useFigureStore((s) => s.config?.background ?? "White");

  const [format, setFormat] = useState("TIFF");
  const [quality, setQuality] = useState(95);
  const [dpi, setDpi] = useState(300);
  const today = new Date();
  const yyyymmdd = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
  const [filePath, setFilePath] = useState("");
  const [saveError, setSaveError] = useState("");
  const [saving, setSaving] = useState(false);

  const ext = format.toLowerCase() === "png" ? "png" : format.toLowerCase() === "jpeg" ? "jpg" : "tiff";

  // Auto-populate filename with today's date when dialog opens
  useEffect(() => {
    if (open) {
      setFilePath((prev) => {
        if (!prev || prev.trim() === "") {
          return `${yyyymmdd}_auto.${ext}`;
        }
        return prev;
      });
      setSaveError("");
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update extension when format changes
  useEffect(() => {
    setFilePath((prev) => {
      if (!prev) return `${yyyymmdd}_auto.${ext}`;
      // Replace the extension in the current filename
      const parts = prev.split(".");
      if (parts.length > 1) {
        parts[parts.length - 1] = ext;
        return parts.join(".");
      }
      return `${prev}.${ext}`;
    });
  }, [format]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleBrowse = async () => {
    try {
      // Try Tauri dialog (works in native app)
      const { save } = await import("@tauri-apps/plugin-dialog");
      const selected = await save({
        defaultPath: filePath || `${yyyymmdd}_auto.${ext}`,
        filters: [{ name: "Image", extensions: [ext] }],
      });
      if (selected) { setFilePath(selected); setSaveError(""); return; }
    } catch { /* not in Tauri */ }
    // Web fallback: pre-fill ~/Documents/ path
    const fname = (filePath || "").split("/").pop() || `${yyyymmdd}_auto.${ext}`;
    setFilePath(`~/Documents/${fname}`);
    setSaveError("");
  };

  const handleSave = async () => {
    const path = filePath.trim();
    if (!path) {
      setSaveError("Please specify a file path before saving.");
      return;
    }
    // Validate path has a directory component (not just a filename)
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

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Save Figure</DialogTitle>
      <DialogContent>
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
              <Button
                variant="outlined" size="small"
                onClick={handleBrowse}
                sx={{ minWidth: 80, flexShrink: 0 }}
                startIcon={<FolderOpenIcon />}
              >
                Browse
              </Button>
            </Box>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: 1.5, mt: 0.25, display: "block", fontSize: "0.65rem" }}>
              Full path including filename and extension
            </Typography>
          </Box>

          {/* Format */}
          <FormControl fullWidth size="small">
            <InputLabel>Output Format</InputLabel>
            <Select
              value={format}
              label="Output Format"
              onChange={(e) => setFormat(e.target.value)}
            >
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
            <Slider
              value={dpi}
              min={72}
              max={600}
              step={1}
              marks={[
                { value: 72, label: "72" },
                { value: 150, label: "150" },
                { value: 300, label: "300" },
                { value: 600, label: "600" },
              ]}
              onChange={(_, val) => setDpi(val as number)}
            />
          </Box>

          {/* Quality (only for JPEG) */}
          {format === "JPEG" && (
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Quality: {quality}
              </Typography>
              <Slider
                value={quality}
                min={1}
                max={100}
                onChange={(_, val) => setQuality(val as number)}
              />
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        {saveError && <Alert severity="error" sx={{ mr: 2, py: 0, flex: 1 }}>{saveError}</Alert>}
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          color="secondary"
          onClick={handleSave}
          disabled={saving || !filePath.trim()}
        >
          {saving ? "Saving..." : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
