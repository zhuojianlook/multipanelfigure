/* ──────────────────────────────────────────────────────────
   PreviewPane — displays the rendered figure preview.
   Always fills available width. Right-click to copy.
   Checkerboard bg for transparency.
   Background color dropdown in header area.
   ────────────────────────────────────────────────────────── */

import { useCallback, useRef } from "react";
import { Select, MenuItem, Button as MuiButton } from "@mui/material";
import { useFigureStore } from "../../store/figureStore";

export function PreviewPane() {
  const previewImageB64 = useFigureStore((s) => s.previewImageB64);
  const previewLoading = useFigureStore((s) => s.previewLoading);
  const requestPreview = useFigureStore((s) => s.requestPreview);
  const background = useFigureStore((s) => s.config?.background ?? "White");
  const setBackground = useFigureStore((s) => s.setBackground);
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
            onChange={(e) => setBackground(e.target.value)}
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
