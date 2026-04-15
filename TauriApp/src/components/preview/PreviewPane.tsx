/* ──────────────────────────────────────────────────────────
   PreviewPane — displays the rendered figure preview.
   Always fills available width. Right-click to copy.
   Checkerboard bg for transparency.
   Background color dropdown in header area.
   ────────────────────────────────────────────────────────── */

import { useCallback, useRef, useState } from "react";
import { Select, MenuItem, Button as MuiButton, IconButton, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions, Typography, Button } from "@mui/material";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import { useFigureStore } from "../../store/figureStore";

/** Check if a color is dark (close to black) */
function isDark(color: string): boolean {
  const c = color.toLowerCase().replace(/\s/g, "");
  if (c === "#000000" || c === "#000" || c === "black" || c === "rgb(0,0,0)") return true;
  // Check hex colors close to black
  if (c.startsWith("#")) {
    const hex = c.slice(1);
    const r = parseInt(hex.length === 3 ? hex[0] + hex[0] : hex.slice(0, 2), 16);
    const g = parseInt(hex.length === 3 ? hex[1] + hex[1] : hex.slice(2, 4), 16);
    const b = parseInt(hex.length === 3 ? hex[2] + hex[2] : hex.slice(4, 6), 16);
    return (r + g + b) / 3 < 40; // average < 40 is very dark
  }
  return false;
}

/** Check if a color is light (close to white) */
function isLight(color: string): boolean {
  const c = color.toLowerCase().replace(/\s/g, "");
  if (c === "#ffffff" || c === "#fff" || c === "white" || c === "rgb(255,255,255)") return true;
  if (c.startsWith("#")) {
    const hex = c.slice(1);
    const r = parseInt(hex.length === 3 ? hex[0] + hex[0] : hex.slice(0, 2), 16);
    const g = parseInt(hex.length === 3 ? hex[1] + hex[1] : hex.slice(2, 4), 16);
    const b = parseInt(hex.length === 3 ? hex[2] + hex[2] : hex.slice(4, 6), 16);
    return (r + g + b) / 3 > 215; // average > 215 is very light
  }
  return false;
}

export function PreviewPane() {
  const previewImageB64 = useFigureStore((s) => s.previewImageB64);
  const previewLoading = useFigureStore((s) => s.previewLoading);
  const requestPreview = useFigureStore((s) => s.requestPreview);
  const background = useFigureStore((s) => s.config?.background ?? "White");
  const setBackground = useFigureStore((s) => s.setBackground);
  const config = useFigureStore((s) => s.config);
  const containerRef = useRef<HTMLDivElement>(null);

  // Pan & Zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const resetView = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.1, Math.min(20, z * delta)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // left button only
    isPanning.current = true;
    panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    e.preventDefault();
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;
    setPan({
      x: panStart.current.panX + (e.clientX - panStart.current.x),
      y: panStart.current.panY + (e.clientY - panStart.current.y),
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
  }, []);

  const previewSrc = previewImageB64
    ? `data:image/png;base64,${previewImageB64}`
    : null;

  const [copyFeedback, setCopyFeedback] = useState("");

  const handleContextMenu = useCallback(
    async (e: React.MouseEvent) => {
      e.preventDefault();
      if (!previewImageB64) return;
      try {
        // Try Tauri clipboard write via IPC (works in WebView)
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("copy_image_to_clipboard", { imageB64: previewImageB64 });
        setCopyFeedback("Copied to clipboard!");
        setTimeout(() => setCopyFeedback(""), 2000);
      } catch {
        // Fallback: try browser clipboard API
        try {
          if (previewSrc) {
            const res = await fetch(previewSrc);
            const blob = await res.blob();
            await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
            setCopyFeedback("Copied to clipboard!");
            setTimeout(() => setCopyFeedback(""), 2000);
          }
        } catch {
          setCopyFeedback("Copy not supported in this context — use Save Figure instead");
          setTimeout(() => setCopyFeedback(""), 3000);
        }
      }
    },
    [previewSrc],
  );

  /** Switch all dark text to white, or all light text to black */
  const switchTextColors = (toColor: string, checkFn: (c: string) => boolean) => {
    if (!config) return;
    const store = useFigureStore.getState();
    // Update column labels
    config.column_labels.forEach((lbl, i) => {
      if (checkFn(lbl.default_color)) {
        store.updateLabelFormatting("col", i, { default_color: toColor });
      }
    });
    // Update row labels
    config.row_labels.forEach((lbl, i) => {
      if (checkFn(lbl.default_color)) {
        store.updateLabelFormatting("row", i, { default_color: toColor });
      }
    });
    // Update column headers
    config.column_headers.forEach((level, li) => {
      level.headers.forEach((hdr, gi) => {
        if (checkFn(hdr.default_color)) {
          store.updateHeaderGroupFormatting("col", li, gi, { default_color: toColor });
        }
      });
    });
    // Update row headers
    config.row_headers.forEach((level, li) => {
      level.headers.forEach((hdr, gi) => {
        if (checkFn(hdr.default_color)) {
          store.updateHeaderGroupFormatting("row", li, gi, { default_color: toColor });
        }
      });
    });
  };

  // Background color change dialog
  const [colorDialog, setColorDialog] = useState<{ message: string; toColor: string; checkFn: (c: string) => boolean; newBg: string } | null>(null);

  const handleBackgroundChange = (newBg: string) => {
    if (!config) {
      setBackground(newBg);
      return;
    }

    const allColors: string[] = [];
    config.column_labels.forEach((l) => allColors.push(l.default_color));
    config.row_labels.forEach((l) => allColors.push(l.default_color));
    config.column_headers.forEach((lv) => lv.headers.forEach((h) => allColors.push(h.default_color)));
    config.row_headers.forEach((lv) => lv.headers.forEach((h) => allColors.push(h.default_color)));

    if (newBg === "Black" && allColors.some(isDark)) {
      setColorDialog({ message: "Switch dark header/label text to white for visibility on black background?", toColor: "#FFFFFF", checkFn: isDark, newBg });
    } else if (newBg === "White" && allColors.some(isLight)) {
      setColorDialog({ message: "Switch light header/label text to black for visibility on white background?", toColor: "#000000", checkFn: isLight, newBg });
    } else {
      setBackground(newBg);
    }
  };

  const isTransparent = background === "Transparent";

  return (
    <div
      ref={containerRef}
      className={`h-full flex flex-col items-center p-3 relative overflow-hidden
                  ${isTransparent ? "checkerboard" : ""}`}
      style={{
        backgroundColor: isTransparent
          ? undefined
          : "var(--c-bg)",
      }}
    >
      {/* Section header */}
      <div
        className="flex items-center justify-between w-full mb-2 px-1"
        style={{ color: "var(--c-text-dim)", zIndex: 5, position: "relative" }}
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold tracking-wider uppercase">
            Preview
          </span>
          <Select
            value={background}
            onChange={(e) => handleBackgroundChange(e.target.value)}
            size="small"
            variant="standard"
            sx={{
              fontSize: "0.625rem",
              minWidth: 140,
              "& .MuiSelect-select": { py: 0, px: 0.5 },
              "& .MuiInput-underline:before": { borderBottom: "none" },
              "& .MuiInput-underline:after": { borderBottom: "1px solid var(--c-accent)" },
            }}
          >
            <MenuItem value="White" sx={{ fontSize: "0.625rem" }}>White</MenuItem>
            <MenuItem value="Black" sx={{ fontSize: "0.625rem" }}>Black</MenuItem>
            <MenuItem value="Transparent" sx={{ fontSize: "0.625rem" }}>Transparent</MenuItem>
          </Select>
        </div>
        <div className="flex items-center gap-1">
          {previewSrc && (
            <>
              <Tooltip title="Zoom out"><IconButton size="small" onClick={() => setZoom(z => Math.max(0.1, z * 0.8))} sx={{ p: 0.25 }}><ZoomOutIcon sx={{ fontSize: 14 }} /></IconButton></Tooltip>
              <span className="text-[9px]" style={{ minWidth: 32, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
              <Tooltip title="Zoom in"><IconButton size="small" onClick={() => setZoom(z => Math.min(20, z * 1.2))} sx={{ p: 0.25 }}><ZoomInIcon sx={{ fontSize: 14 }} /></IconButton></Tooltip>
              <Tooltip title="Reset view"><IconButton size="small" onClick={resetView} sx={{ p: 0.25 }}><CenterFocusStrongIcon sx={{ fontSize: 14 }} /></IconButton></Tooltip>
            </>
          )}
          <span className="text-[9px] ml-1">
            {previewLoading ? "Rendering..." : previewSrc ? "Right-click to copy" : ""}
          </span>
        </div>
      </div>

      {previewLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/30 z-10">
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            Rendering...
          </span>
        </div>
      )}

      {/* Copy feedback toast */}
      {copyFeedback && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 z-20 px-3 py-1.5 rounded text-xs font-medium"
          style={{ backgroundColor: "var(--c-accent)", color: "#fff", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }}>
          {copyFeedback}
        </div>
      )}

      {previewSrc ? (
        <div
          style={{ flex: 1, width: "100%", overflow: "hidden", cursor: isPanning.current ? "grabbing" : "grab", position: "relative" }}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={handleContextMenu}
        >
        <img
          src={previewSrc}
          alt="Figure preview"
          className="rounded"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            maxWidth: "100%",
            maxHeight: "80vh",
            display: "block",
            margin: "0 auto",
            userSelect: "none",
          }}
          onContextMenu={handleContextMenu}
          draggable={false}
        />
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center gap-2">
          <span
            className="text-sm font-medium"
            style={{ color: "var(--c-text-dim)" }}
          >
            No preview yet
          </span>
          <MuiButton
            variant="contained"
            size="small"
            onClick={requestPreview}
          >
            Generate Preview
          </MuiButton>
        </div>
      )}
      {/* Background color text switch dialog */}
      <Dialog open={!!colorDialog} onClose={() => { if (colorDialog) setBackground(colorDialog.newBg); setColorDialog(null); }}>
        <DialogTitle>Switch Text Colors</DialogTitle>
        <DialogContent>
          <Typography>{colorDialog?.message}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { if (colorDialog) setBackground(colorDialog.newBg); setColorDialog(null); }}>No</Button>
          <Button variant="contained" onClick={() => {
            if (colorDialog) {
              switchTextColors(colorDialog.toColor, colorDialog.checkFn);
              setBackground(colorDialog.newBg);
            }
            setColorDialog(null);
          }}>Yes</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
