/* ──────────────────────────────────────────────────────────
   PreviewPane — displays the rendered figure preview.
   Always fills available width. Right-click to copy.
   Checkerboard bg for transparency.
   Background color dropdown in header area.
   ────────────────────────────────────────────────────────── */

import { useCallback, useRef } from "react";
import { Select, MenuItem, Button as MuiButton } from "@mui/material";
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

  const previewSrc = previewImageB64
    ? `data:image/png;base64,${previewImageB64}`
    : null;

  const handleContextMenu = useCallback(
    async (e: React.MouseEvent) => {
      e.preventDefault();
      if (!previewSrc) return;
      try {
        const res = await fetch(previewSrc);
        const blob = await res.blob();
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]);
      } catch {
        // clipboard write may fail in non-secure contexts
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

  const handleBackgroundChange = (newBg: string) => {
    if (!config) {
      setBackground(newBg);
      return;
    }

    // Collect all text colors
    const allColors: string[] = [];
    config.column_labels.forEach((l) => allColors.push(l.default_color));
    config.row_labels.forEach((l) => allColors.push(l.default_color));
    config.column_headers.forEach((lv) => lv.headers.forEach((h) => allColors.push(h.default_color)));
    config.row_headers.forEach((lv) => lv.headers.forEach((h) => allColors.push(h.default_color)));

    if (newBg === "Black" && allColors.some(isDark)) {
      if (window.confirm("Switch dark header/label text to white for visibility on black background?")) {
        switchTextColors("#FFFFFF", isDark);
      }
    } else if (newBg === "White" && allColors.some(isLight)) {
      if (window.confirm("Switch light header/label text to black for visibility on white background?")) {
        switchTextColors("#000000", isLight);
      }
    }

    setBackground(newBg);
  };

  const isTransparent = background === "Transparent";

  return (
    <div
      ref={containerRef}
      className={`h-full flex flex-col items-center p-3 relative overflow-auto
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
        style={{ color: "var(--c-text-dim)" }}
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
        <span className="text-[9px]">
          {previewLoading
            ? "Rendering..."
            : previewSrc
              ? "Right-click to copy"
              : ""}
        </span>
      </div>

      {previewLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/30 z-10">
          <span className="text-xs" style={{ color: "var(--c-text-dim)" }}>
            Rendering...
          </span>
        </div>
      )}

      {previewSrc ? (
        <img
          src={previewSrc}
          alt="Figure preview"
          className="w-full object-contain rounded"
          style={{ maxHeight: "80vh" }}
          onContextMenu={handleContextMenu}
          draggable={false}
        />
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
    </div>
  );
}
