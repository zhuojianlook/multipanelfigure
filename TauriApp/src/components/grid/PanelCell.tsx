/* ──────────────────────────────────────────────────────────
   PanelCell — a single cell in the panel grid.
   Shows position label, thumbnail, image dropdown, edit button.
   Right-click context menu for copy/paste panel settings.
   ────────────────────────────────────────────────────────── */

import { useState } from "react";
import {
  Box,
  Typography,
  Select,
  MenuItem,
  IconButton,
  Menu,
  ListItemIcon,
  ListItemText,
  Snackbar,
  Alert,
  Tooltip,
} from "@mui/material";
import SettingsIcon from "@mui/icons-material/Settings";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ContentPasteIcon from "@mui/icons-material/ContentPaste";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import { useFigureStore, type LoadedImage } from "../../store/figureStore";
import { EditPanelDialog } from "../dialogs/EditPanelDialog";
import { getSelectedImageName, clearSelectedImage, useSelectedImage } from "../image-strip/ImageStrip";
import type { PanelInfo } from "../../api/types";

// Global clipboard for panel settings (excludes zoom_inset per spec 3.1.7)
let copiedPanelSettings: Partial<PanelInfo> | null = null;

interface Props {
  row: number;
  col: number;
  imageName: string;
}

export function PanelCell({ row, col, imageName }: Props) {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const panelThumbnails = useFigureStore((s) => s.panelThumbnails);
  const setPanelImage = useFigureStore((s) => s.setPanelImage);
  const updatePanel = useFigureStore((s) => s.updatePanel);
  const config = useFigureStore((s) => s.config);
  const swapPanels = useFigureStore((s) => s.swapPanels);
  const [editOpen, setEditOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [ctxMenu, setCtxMenu] = useState<{ mouseX: number; mouseY: number } | null>(null);
  const [snackMsg, setSnackMsg] = useState("");
  const selectedImage = useSelectedImage();

  const img: LoadedImage | undefined = imageName
    ? loadedImages[imageName]
    : undefined;

  const processedThumb = panelThumbnails[`${row}-${col}`];
  const imageEntries = Object.values(loadedImages);
  const panel = config?.panels[row]?.[col];

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setCtxMenu({ mouseX: e.clientX + 2, mouseY: e.clientY - 6 });
  };

  const handleCopySettings = () => {
    if (!panel) return;
    // Copy everything except image_name and zoom_inset
    const { image_name, zoom_inset, add_zoom_inset, ...settings } = panel;
    copiedPanelSettings = settings;
    setSnackMsg("Settings copied");
    setCtxMenu(null);
  };

  const handlePasteSettings = () => {
    if (!copiedPanelSettings || !panel) { setCtxMenu(null); return; }
    const statusParts: string[] = [];

    // Check crop compatibility
    if (copiedPanelSettings.crop) {
      statusParts.push("crop area...applied");
    }
    if (copiedPanelSettings.brightness !== undefined || copiedPanelSettings.contrast !== undefined) {
      statusParts.push("adjustments...ok");
    }
    if (copiedPanelSettings.scale_bar || copiedPanelSettings.add_scale_bar) {
      statusParts.push("scale bar...ok");
    }
    if (copiedPanelSettings.labels && (copiedPanelSettings.labels as unknown[]).length > 0) {
      statusParts.push("labels...ok");
    }
    if (copiedPanelSettings.symbols && (copiedPanelSettings.symbols as unknown[]).length > 0) {
      statusParts.push("symbols...ok");
    }

    // Apply settings (excluding image_name and zoom_inset)
    updatePanel(row, col, copiedPanelSettings);
    const msg = statusParts.length > 0 ? `Settings pasted: ${statusParts.join(", ")}` : "Settings pasted";
    setSnackMsg(msg);
    setCtxMenu(null);
  };

  const handleClearPanel = () => {
    if (!imageName) { setCtxMenu(null); return; }
    if (window.confirm(`Clear image from R${row + 1}C${col + 1}? All settings will be lost.`)) {
      setPanelImage(row, col, "");
    }
    setCtxMenu(null);
  };

  // Check if this panel is a zoom inset target (protected from drops)
  const isZoomTarget = (() => {
    const cfg = useFigureStore.getState().config;
    if (!cfg) return false;
    for (let r = 0; r < cfg.rows; r++) {
      for (let c = 0; c < cfg.cols; c++) {
        const p = cfg.panels[r]?.[c];
        if (!p?.add_zoom_inset || !p.zoom_inset || p.zoom_inset.inset_type !== "Adjacent Panel") continue;
        const side = p.zoom_inset.side || "Right";
        let tr = r, tc = c;
        if (side === "Top") tr--; else if (side === "Bottom") tr++; else if (side === "Left") tc--; else if (side === "Right") tc++;
        if (tr === row && tc === col) return true;
      }
    }
    return false;
  })();

  // Centralized drop handler — extracted so we can use it on a transparent overlay too
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (isZoomTarget) {
      alert("This panel is reserved for a zoom inset. Disable the zoom inset first.");
      return;
    }
    const imgName = e.dataTransfer.getData("application/x-image-name");
    if (imgName) {
      if (imageName) {
        if (!window.confirm(`Replace image in R${row + 1}C${col + 1}? Current settings will be lost.`)) return;
      }
      setPanelImage(row, col, imgName);
      return;
    }
    const panelSrc = e.dataTransfer.getData("application/x-panel-source");
    if (panelSrc) {
      const src = JSON.parse(panelSrc) as { row: number; col: number };
      swapPanels(src.row, src.col, row, col);
      return;
    }
    const drawerSrc = e.dataTransfer.getData("application/x-drawer-index");
    if (drawerSrc) {
      const drawerIdx = Number(drawerSrc);
      const movePanelFromDrawer = useFigureStore.getState().movePanelFromDrawer;
      movePanelFromDrawer(drawerIdx, row, col);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Accept both "copy" (from filmstrip) and "move" (from drawer/panel swap)
    const allowed = e.dataTransfer.effectAllowed;
    e.dataTransfer.dropEffect = (allowed === "move") ? "move" : "copy";
    setDragOver(true);
  };

  return (
    <Box
      onContextMenu={handleContextMenu}
      onDragEnter={(e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true); }}
      onDragOver={handleDragOver}
      onDragLeave={(e) => {
        const relTarget = e.relatedTarget as HTMLElement | null;
        if (relTarget && (e.currentTarget as HTMLElement).contains(relTarget)) return;
        setDragOver(false);
      }}
      onDrop={handleDrop}
      draggable={!!imageName}
      onDragStart={(e) => {
        if (!imageName) { e.preventDefault(); return; }
        e.dataTransfer.setData("application/x-panel-source", JSON.stringify({ row, col }));
        e.dataTransfer.setData("text/plain", `Panel R${row+1}C${col+1}`);
        e.dataTransfer.effectAllowed = "move";
        // Set custom drag image to show only this panel's thumbnail
        const thumbEl = e.currentTarget.querySelector("img");
        if (thumbEl) {
          e.dataTransfer.setDragImage(thumbEl, thumbEl.offsetWidth / 2, thumbEl.offsetHeight / 2);
        }
      }}
      onClick={() => {
        // Click-to-assign: if an image is selected in the filmstrip, assign it here
        const selImg = getSelectedImageName();
        if (selImg) {
          if (imageName && imageName !== selImg) {
            if (!window.confirm(`Replace image in R${row + 1}C${col + 1}? Current settings will be lost.`)) return;
          }
          setPanelImage(row, col, selImg);
          clearSelectedImage();
          return;
        }
      }}
      sx={{
        position: "relative",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "space-between",
        borderRadius: 1,
        border: dragOver || selectedImage ? 3 : imageName ? 2 : 1,
        borderColor: dragOver ? "#2196f3" : selectedImage ? "#2196f3" : imageName ? "primary.main" : "divider",
        height: "100%",
        overflow: "hidden",
        p: 0.75,
        transition: "border-color 0.15s",
        "&:hover .edit-btn, & .edit-btn:focus-visible": { opacity: 1 },
        cursor: selectedImage ? "pointer" : imageName ? "grab" : "default",
      }}
    >
      {/* Invisible drop overlay — ensures drops always work even over child elements */}
      {dragOver && (
        <Box
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onDragLeave={() => setDragOver(false)}
          sx={{
            position: "absolute",
            inset: 0,
            zIndex: 100,
            backgroundColor: "rgba(33, 150, 243, 0.1)",
          }}
        />
      )}
      {/* Position label + Edit button row */}
      <Box sx={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }}>
        <Typography
          variant="caption"
          sx={{ fontSize: "0.55rem", fontFamily: "monospace", color: "text.secondary" }}
        >
          R{row + 1} C{col + 1}
        </Typography>
        <IconButton
          className="edit-btn"
          size="small"
          onClick={() => setEditOpen(true)}
          sx={{ opacity: 0.7, transition: "opacity 0.15s", p: 0.25 }}
          aria-label="Edit panel"
          title="Edit panel"
        >
          <SettingsIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Box>

      {/* Thumbnail area — also handles click-to-assign */}
      <Box
        onClick={() => {
          if (isZoomTarget) return;
          const selImg = getSelectedImageName();
          if (selImg) {
            if (imageName && imageName !== selImg) {
              if (!window.confirm(`Replace image in R${row + 1}C${col + 1}? Current settings will be lost.`)) return;
            }
            setPanelImage(row, col, selImg);
            clearSelectedImage();
          }
        }}
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "100%",
          minHeight: 0,
          cursor: selectedImage ? "pointer" : "default",
          border: selectedImage && !imageName ? "2px dashed #2196f3" : "none",
          borderRadius: 1,
          transition: "border-color 0.15s",
        }}
      >
        {img ? (
          <Box
            component="img"
            src={`data:image/png;base64,${processedThumb || img.thumbnailB64}`}
            alt={img.name}
            sx={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: 0.5 }}
            draggable={false}
          />
        ) : (
          <Typography sx={{ fontSize: selectedImage ? "0.7rem" : "1.5rem", color: selectedImage ? "#2196f3" : "divider" }}>
            {selectedImage ? "Click to assign" : "\u25A2"}
          </Typography>
        )}
      </Box>

      {/* Zoom target indicator */}
      {isZoomTarget && (
        <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "warning.main", textAlign: "center", mt: 0.25, fontStyle: "italic" }}>
          🔒 Zoom inset target
        </Typography>
      )}
      {/* Image dropdown */}
      <Tooltip title={isZoomTarget ? "Protected: zoom inset target" : (imageName || "")} placement="bottom" arrow enterDelay={200}>
      <div>
      <Select
        value={imageName}
        onChange={(e) => {
          if (isZoomTarget) return;
          if (!e.target.value && imageName) {
            if (!window.confirm(`Remove image from R${row + 1}C${col + 1}? All settings will be lost.`)) return;
          }
          setPanelImage(row, col, e.target.value);
        }}
        disabled={isZoomTarget}
        displayEmpty
        fullWidth
        sx={{
          fontSize: "0.625rem",
          mt: 0.5,
          "& .MuiSelect-select": { py: 0.25, px: 0.5 },
        }}
      >
        <MenuItem value="" sx={{ fontSize: "0.625rem", py: 0.25, px: 0.5 }}>
          No Image
        </MenuItem>
        {imageEntries.map((im) => (
          <MenuItem key={im.name} value={im.name} title={im.name} sx={{ fontSize: "0.625rem", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
            {im.name}
          </MenuItem>
        ))}
      </Select>
      </div>
      </Tooltip>

      {/* Right-click context menu */}
      <Menu
        open={ctxMenu !== null}
        onClose={() => setCtxMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={ctxMenu ? { top: ctxMenu.mouseY, left: ctxMenu.mouseX } : undefined}
      >
        <MenuItem onClick={handleCopySettings} disabled={!imageName}>
          <ListItemIcon><ContentCopyIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Copy Settings</ListItemText>
        </MenuItem>
        <MenuItem onClick={handlePasteSettings} disabled={!copiedPanelSettings || !imageName}>
          <ListItemIcon><ContentPasteIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Paste Settings</ListItemText>
        </MenuItem>
        <MenuItem onClick={handleClearPanel} disabled={!imageName}>
          <ListItemIcon><DeleteSweepIcon fontSize="small" /></ListItemIcon>
          <ListItemText>Clear Panel</ListItemText>
        </MenuItem>
      </Menu>

      {/* Status snackbar */}
      <Snackbar
        open={!!snackMsg}
        autoHideDuration={2000}
        onClose={() => setSnackMsg("")}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity="success" onClose={() => setSnackMsg("")} sx={{ py: 0 }}>
          {snackMsg}
        </Alert>
      </Snackbar>

      <EditPanelDialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        row={row}
        col={col}
      />
    </Box>
  );
}
