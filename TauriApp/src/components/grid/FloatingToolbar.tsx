/* ──────────────────────────────────────────────────────────
   FloatingToolbar — small formatting toolbar that appears
   above a selected header group or column/row label.
   Contains: B | I | U | S̶ | Font dropdown | Size input | Color picker
   Uses MUI Popover anchored to the selected element.
   ────────────────────────────────────────────────────────── */

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
}: Props) {
  // Normalize fonts to an array of strings
  const fontList: string[] = Array.isArray(fonts)
    ? fonts
    : Object.keys(fonts);

  const fallbackFonts = ["arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf"];
  const displayFonts = (fontList.length > 0 ? fontList : fallbackFonts)
    .slice().sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));

  const hasStyle = (style: string) => fontStyle.includes(style);

  return (
    <Popover
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: "top", horizontal: "center" }}
      transformOrigin={{ vertical: "bottom", horizontal: "center" }}
      slotProps={{
        paper: {
          sx: {
            mt: -0.5,
            overflow: "auto",
            boxShadow: 3,
            maxWidth: "90vw",
          },
        },
      }}
      // Don't steal focus from the text input
      disableAutoFocus
      disableEnforceFocus
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
        // selection. stopPropagation prevents the outer popover from
        // auto-closing on the click.
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
          value={fontName}
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

        {/* Size input */}
        <TextField
          type="number"
          value={fontSize}
          onChange={(e) => {
            const v = Math.max(1, Math.min(200, Number(e.target.value) || 1));
            onFontSizeChange(v);
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
      </Box>
    </Popover>
  );
}
