/* ──────────────────────────────────────────────────────────
   FloatingToolbar — small formatting toolbar that appears
   above a selected header group or column/row label.
   Contains: B | I | U | S̶ | Font dropdown | Size input | Color picker
   Uses MUI Popover anchored to the selected element.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect } from "react";
import {
  Popover,
  Box,
  IconButton,
  TextField,
  Select,
  MenuItem,
  Tooltip,
} from "@mui/material";
import FormatBoldIcon from "@mui/icons-material/FormatBold";
import FormatItalicIcon from "@mui/icons-material/FormatItalic";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import SuperscriptIcon from "@mui/icons-material/Superscript";
import SubscriptIcon from "@mui/icons-material/Subscript";
import SubdirectoryArrowLeftIcon from "@mui/icons-material/SubdirectoryArrowLeft";

interface Props {
  anchorEl: HTMLElement | null;
  open: boolean;
  onClose: () => void;
  text: string;
  fontSize: number;
  fontName: string;
  fontStyle: string[];
  color: string;
  fonts: Record<string, string> | string[];
  onTextChange: (text: string) => void;
  onFontSizeChange: (size: number) => void;
  onFontNameChange: (name: string) => void;
  onFontStyleToggle: (style: string) => void;
  onColorChange: (color: string) => void;
  /** Fired on mousedown anywhere in the toolbar, BEFORE focus moves off
   *  the source textarea. Lets the parent snapshot the current text
   *  selection so it can be applied later when the toolbar action
   *  actually fires. */
  onBeforeAction?: () => void;
  /** Optional label rendered in the toolbar indicating the current
   *  selected substring (if any). Lets the user see what their next
   *  style change will apply to. */
  selectionPreview?: string;
  /** Click handler for the "insert line break" button — works as a
   *  keyboard-independent alternative to Shift+Enter for users on
   *  environments where that key combo doesn't reach the textarea. */
  onInsertLineBreak?: () => void;
}

export function FloatingToolbar({
  anchorEl,
  open,
  onClose,
  fontSize,
  fontName,
  fontStyle,
  color,
  fonts,
  onFontSizeChange,
  onFontNameChange,
  onFontStyleToggle,
  onColorChange,
  onBeforeAction,
  selectionPreview,
  onInsertLineBreak,
}: Props) {
  // Normalize fonts to an array of strings
  const fontList: string[] = Array.isArray(fonts)
    ? fonts
    : Object.keys(fonts);

  const fallbackFonts = ["arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf"];
  const displayFonts = (fontList.length > 0 ? fontList : fallbackFonts)
    .slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

  // Resolve the Select's value to a font that actually exists in the
  // current option list — falls back to the first case-insensitive
  // Arial match, then to the first font in the list. Prevents the
  // dropdown from rendering blank when the model's font_name is unset
  // or points at a font not installed on the current machine (common
  // when a project is opened on a different OS).
  const effectiveFontName = (() => {
    if (fontName && displayFonts.includes(fontName)) return fontName;
    if (fontName) {
      const lower = fontName.toLowerCase();
      const match = displayFonts.find((f) => f.toLowerCase() === lower);
      if (match) return match;
    }
    const arial = displayFonts.find((f) => /^arial\b/i.test(f.replace(/\.(ttf|otf|ttc)$/i, "")));
    if (arial) return arial;
    return displayFonts[0] || "";
  })();

  const hasStyle = (style: string) => fontStyle.includes(style);

  // Local buffer for the font-size input. We commit only on blur or
  // Enter — the previous per-keystroke onChange made typing "16" fire a
  // patch at "1" first (which applied size=1 to the selected chars and
  // shrank the planner text), and the value would snap around while the
  // user was still typing. Keeping a local string buffer lets the user
  // type any intermediate value without the model reacting until they
  // explicitly commit.
  const [sizeBuf, setSizeBuf] = useState<string>(String(fontSize));
  useEffect(() => {
    setSizeBuf(String(fontSize));
  }, [fontSize]);
  const commitSize = () => {
    const v = Math.max(1, Math.min(200, Number(sizeBuf) || fontSize));
    if (v !== fontSize) onFontSizeChange(v);
    setSizeBuf(String(v));
  };

  return (
    <Popover
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: "top", horizontal: "center" }}
      transformOrigin={{ vertical: "bottom", horizontal: "center" }}
      // hideBackdrop is critical — Popover wraps Modal, which by default
      // renders an (invisible) Backdrop that catches every click on the
      // page to close the popover. That backdrop was also swallowing the
      // user's mousedown-to-start-selection in the header textarea below,
      // so drag-selecting individual characters was impossible.
      hideBackdrop
      disableScrollLock
      slotProps={{
        root: {
          // Even without the backdrop, Modal's root container still spans
          // the viewport and can swallow pointer events. Make the root
          // completely click-through; only the paper (toolbar itself)
          // stays interactive.
          sx: { pointerEvents: "none" },
        },
        paper: {
          sx: {
            mt: -0.5,
            overflow: "auto",
            boxShadow: 3,
            maxWidth: "90vw",
            pointerEvents: "auto",
          },
        },
      }}
      // Don't steal focus from the text input
      disableAutoFocus
      disableEnforceFocus
      disableRestoreFocus
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.25,
          px: 0.75,
          py: 0.5,
        }}
        // Pointer-down bubbles to this wrapper from every toolbar control
        // (buttons, dropdowns, color picker). At this moment focus is still
        // on the source textarea, so onBeforeAction can snapshot the live
        // selection. We fire it on BOTH the capture and bubble phases so
        // it runs before any child handler has a chance to change focus.
        onMouseDownCapture={() => { if (onBeforeAction) onBeforeAction(); }}
        onMouseDown={(e) => {
          if (onBeforeAction) onBeforeAction();
          e.stopPropagation();
          // preventDefault is applied ONLY to targets that don't need focus
          // — buttons/dropdowns. The native color input needs focus so the
          // OS colour picker opens, so we leave those alone.
          const t = e.target as HTMLElement | null;
          if (t && t.tagName !== "INPUT") {
            // Keep text selection in the source textarea alive.
            e.preventDefault();
          }
        }}
      >
        {/* Bold */}
        <Tooltip title="Bold" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Bold")}
            sx={{
              bgcolor: hasStyle("Bold") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <FormatBoldIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Italic */}
        <Tooltip title="Italic" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Italic")}
            sx={{
              bgcolor: hasStyle("Italic") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <FormatItalicIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Underline */}
        <Tooltip title="Underline" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Underline")}
            sx={{
              bgcolor: hasStyle("Underline") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <FormatUnderlinedIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Strikethrough */}
        <Tooltip title="Strikethrough" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Strikethrough")}
            sx={{
              bgcolor: hasStyle("Strikethrough") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <StrikethroughSIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Superscript */}
        <Tooltip title="Superscript" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Superscript")}
            sx={{
              bgcolor: hasStyle("Superscript") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <SuperscriptIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Subscript */}
        <Tooltip title="Subscript" arrow>
          <IconButton
            size="small"
            onClick={() => onFontStyleToggle("Subscript")}
            sx={{
              bgcolor: hasStyle("Subscript") ? "action.selected" : "transparent",
              width: 28,
              height: 28,
            }}
          >
            <SubscriptIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>

        {/* Insert line break — keyboard-independent alternative to
            Shift+Enter, which doesn't always reach the textarea in
            WKWebView / WebView2 environments. */}
        {onInsertLineBreak && (
          <Tooltip title="Insert line break (same as Shift+Enter)" arrow>
            <IconButton
              size="small"
              onClick={onInsertLineBreak}
              sx={{ width: 28, height: 28 }}
            >
              <SubdirectoryArrowLeftIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
        )}

        {/* Separator */}
        <Box
          sx={{
            mx: 0.25,
            borderLeft: 1,
            borderColor: "divider",
            height: 20,
          }}
        />

        {/* Font dropdown */}
        <Select
          value={effectiveFontName}
          onChange={(e) => onFontNameChange(e.target.value)}
          size="small"
          variant="standard"
          sx={{
            fontSize: "0.65rem",
            minWidth: 80,
            maxWidth: 140,
            "& .MuiSelect-select": { py: 0, px: 0.5 },
            "& .MuiInput-underline:before": { borderBottom: "none" },
          }}
        >
          {displayFonts.map((f) => (
            <MenuItem key={f} value={f} sx={{ fontSize: "0.65rem" }}>
              {f.replace(/\.(ttf|otf|ttc)$/i, "")}
            </MenuItem>
          ))}
        </Select>

        {/* Size input — commits on blur/Enter, not per-keystroke */}
        <TextField
          type="number"
          value={sizeBuf}
          onChange={(e) => setSizeBuf(e.target.value)}
          onBlur={commitSize}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commitSize();
              (e.currentTarget as HTMLInputElement).blur();
            }
          }}
          size="small"
          variant="standard"
          slotProps={{
            htmlInput: { min: 1, max: 200, step: 1 },
          }}
          sx={{
            width: 40,
            "& .MuiInputBase-input": {
              fontSize: "0.65rem",
              textAlign: "center",
              py: 0,
            },
            "& .MuiInput-underline:before": { borderBottom: "none" },
          }}
        />

        {/* Color picker */}
        <Tooltip title="Text color" arrow>
          <Box
            component="input"
            type="color"
            value={color}
            aria-label="Text color"
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              onColorChange(e.target.value)
            }
            sx={{
              width: 24,
              height: 24,
              border: "none",
              borderRadius: 0.5,
              cursor: "pointer",
              p: 0,
              bgcolor: "transparent",
              "&::-webkit-color-swatch-wrapper": { p: 0 },
              "&::-webkit-color-swatch": {
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 4,
              },
            }}
          />
        </Tooltip>

        {/* Visual hint: shows what substring will be styled by the next
            action. Empty = the whole header is targeted. */}
        {selectionPreview ? (
          <Box
            sx={{
              ml: 0.5,
              px: 0.5,
              py: 0.1,
              fontSize: "0.55rem",
              bgcolor: "action.selected",
              borderRadius: 0.5,
              maxWidth: 100,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              color: "text.secondary",
            }}
            title={`Styling will apply to: "${selectionPreview}"`}
          >
            "{selectionPreview.length > 12 ? selectionPreview.slice(0, 12) + "…" : selectionPreview}"
          </Box>
        ) : (
          <Box sx={{ ml: 0.5, fontSize: "0.55rem", color: "text.disabled" }}>whole</Box>
        )}
      </Box>
    </Popover>
  );
}
