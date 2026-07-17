/* ──────────────────────────────────────────────────────────
   ColorPickerButton — the app's single colour control.

   A swatch that opens an INLINE palette. Two reasons it isn't a native
   <input type="color">:

     1. WKWebView (this app's macOS webview) hands a native colour input
        to the OS NSColorPanel — a free-floating window that opens
        wherever macOS last left it, frequently nowhere near the app.
     2. Consistency: the styled-text hover toolbar already uses this
        palette, so a native swatch elsewhere made the same concept look
        like two different controls.

   Positioned by CSS relative to the button — NOT a portal. MUI's
   portal-based widgets mis-position when nested inside a <Popper> (the
   text toolbar is one), and this control has to work in both places.

   "Custom…" still reaches the OS picker for colours outside the palette.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import { Box, Paper, Typography } from "@mui/material";

/** Figure-annotation staples: a greyscale ramp plus high-contrast
 *  primaries that read well over both bright-field and fluorescence. */
export const COLOR_SWATCHES = [
  "#FFFFFF", "#C0C0C0", "#808080", "#404040", "#000000", "#FF0000", "#FF6D00", "#FFD600",
  "#FFFF00", "#00E676", "#00C853", "#00E5FF", "#2979FF", "#0000FF", "#D500F9", "#FF00FF",
];

export interface ColorPickerButtonProps {
  value: string;
  onChange: (color: string) => void;
  /** Tooltip / accessible name. */
  title?: string;
  /** Swatch edge length in px. */
  size?: number;
  /** Called on mousedown BEFORE focus moves — lets a parent toolbar
   *  preventDefault so it doesn't lose its text selection. */
  onBeforeOpen?: (e: React.MouseEvent<HTMLElement>) => void;
  /** Where the palette opens relative to the swatch. */
  align?: "left" | "right";
  disabled?: boolean;
}

export function ColorPickerButton({
  value,
  onChange,
  title = "Colour",
  size = 22,
  onBeforeOpen,
  align = "left",
  disabled = false,
}: ColorPickerButtonProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const nativeRef = useRef<HTMLInputElement>(null);

  // Close on outside click. Guarded on `open` so we don't keep a listener
  // around for every swatch on the page.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (t && wrapRef.current?.contains(t)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <Box ref={wrapRef} sx={{ position: "relative", display: "inline-flex" }}>
      <Box
        role="button"
        aria-label={title}
        title={title}
        onMouseDown={onBeforeOpen}
        onClick={() => { if (!disabled) setOpen((v) => !v); }}
        sx={{
          width: size, height: size, borderRadius: 0.5,
          bgcolor: value || "#000",
          border: "1px solid",
          borderColor: "divider",
          cursor: disabled ? "default" : "pointer",
          opacity: disabled ? 0.5 : 1,
          flexShrink: 0,
          "&:hover": disabled ? {} : { borderColor: "primary.main" },
        }}
      />
      {open && (
        <Paper
          elevation={6}
          sx={{
            position: "absolute", top: "100%", mt: 0.25,
            ...(align === "right" ? { right: 0 } : { left: 0 }),
            p: 0.75, zIndex: 1600,
            border: "1px solid", borderColor: "divider",
          }}
          // Keep focus where it was (e.g. a text selection in the editor).
          onMouseDown={(e) => e.preventDefault()}
        >
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(8, 16px)", gap: 0.375 }}>
            {COLOR_SWATCHES.map((c) => {
              const selected = (value || "").toLowerCase() === c.toLowerCase();
              return (
                <Box
                  key={c}
                  title={c}
                  onMouseDown={onBeforeOpen}
                  onClick={() => { onChange(c); setOpen(false); }}
                  sx={{
                    width: 16, height: 16, borderRadius: 0.25,
                    bgcolor: c, cursor: "pointer",
                    border: "1px solid",
                    borderColor: selected ? "primary.main" : "divider",
                    boxShadow: selected ? 2 : 0,
                    "&:hover": { transform: "scale(1.15)" },
                    transition: "transform 80ms",
                  }}
                />
              );
            })}
          </Box>
          <Box
            onMouseDown={onBeforeOpen}
            onClick={() => {
              const el = nativeRef.current as
                (HTMLInputElement & { showPicker?: () => void }) | null;
              if (!el) return;
              setOpen(false);
              // showPicker() opens the OS picker even for an off-screen input;
              // a plain .click() is a silent no-op in WKWebView.
              try {
                if (typeof el.showPicker === "function") { el.showPicker(); return; }
              } catch { /* fall through */ }
              el.click();
            }}
            sx={{
              mt: 0.75, px: 0.5, py: 0.25, borderRadius: 0.5,
              fontSize: "0.65rem", textAlign: "center", cursor: "pointer",
              color: "text.secondary",
              border: "1px dashed", borderColor: "divider",
              "&:hover": { bgcolor: "action.hover" },
            }}
          >
            <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>Custom…</Typography>
          </Box>
        </Paper>
      )}
      <input
        ref={nativeRef}
        type="color"
        value={value || "#000000"}
        onChange={(e) => onChange(e.target.value)}
        tabIndex={-1}
        // Only reachable via "Custom…". Kept mounted + 1×1 so showPicker()
        // has a live, rendered element to open from.
        style={{ position: "absolute", left: 4, bottom: 0, width: 1, height: 1, opacity: 0, border: "none", padding: 0, pointerEvents: "none" }}
      />
    </Box>
  );
}
