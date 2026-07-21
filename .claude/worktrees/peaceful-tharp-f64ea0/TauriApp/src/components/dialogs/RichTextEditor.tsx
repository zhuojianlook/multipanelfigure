/* ──────────────────────────────────────────────────────────
   RichTextEditor — per-character rich text formatting editor.
   Used for column/row labels and header groups.
   Stores formatted text as an array of StyledSegment objects.
   ────────────────────────────────────────────────────────── */

import { useState, useRef, useCallback, useEffect } from "react";
import {
  Popover,
  Box,
  IconButton,
  TextField,
  Select,
  MenuItem,
  Typography,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from "@mui/material";
import FormatBoldIcon from "@mui/icons-material/FormatBold";
import FormatItalicIcon from "@mui/icons-material/FormatItalic";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import SuperscriptIcon from "@mui/icons-material/Superscript";
import SubscriptIcon from "@mui/icons-material/Subscript";
import FormatTextdirectionLToRIcon from "@mui/icons-material/FormatTextdirectionLToR";
import FormatTextdirectionRToLIcon from "@mui/icons-material/FormatTextdirectionRToL";
import type { StyledSegment } from "../../api/types";

interface Props {
  open: boolean;
  anchorEl: HTMLElement | null;
  onClose: () => void;
  segments: StyledSegment[];
  plainText: string;
  onSave: (segments: StyledSegment[], plainText: string) => void;
  fonts: string[];
}

export function RichTextEditor({ open, anchorEl, onClose, segments: initialSegments, plainText: initialText, onSave, fonts }: Props) {
  const [localSegments, setLocalSegments] = useState<StyledSegment[]>([]);
  const [selStart, setSelStart] = useState(0);
  const [selEnd, setSelEnd] = useState(0);
  const [isRtl, setIsRtl] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Current formatting for toolbar display
  const [currentFont, setCurrentFont] = useState("arial.ttf");
  const [currentSize, setCurrentSize] = useState(10);
  const [currentColor, setCurrentColor] = useState("#000000");

  useEffect(() => {
    if (open) {
      if (initialSegments.length > 0) {
        setLocalSegments(JSON.parse(JSON.stringify(initialSegments)));
      } else {
        // Convert plain text to a single segment
        setLocalSegments(initialText ? [{ text: initialText, color: "#000000" }] : []);
      }
      setSelStart(0);
      setSelEnd(0);
    }
  }, [open, initialSegments, initialText]);

  // Get full text from segments
  const getFullText = useCallback(() => {
    return localSegments.map((s) => s.text).join("");
  }, [localSegments]);

  // Map a character index to its segment
  const getSegmentAtIndex = useCallback((charIdx: number): StyledSegment | null => {
    let pos = 0;
    for (const seg of localSegments) {
      if (charIdx >= pos && charIdx < pos + seg.text.length) {
        return seg;
      }
      pos += seg.text.length;
    }
    return localSegments.length > 0 ? localSegments[localSegments.length - 1] : null;
  }, [localSegments]);

  // Update toolbar display based on selection
  useEffect(() => {
    const seg = getSegmentAtIndex(selStart);
    if (seg) {
      setCurrentFont(seg.font_name ?? "arial.ttf");
      setCurrentSize(seg.font_size ?? 10);
      setCurrentColor(seg.color ?? "#000000");
    }
  }, [selStart, getSegmentAtIndex]);

  const handleSelectionChange = () => {
    const el = inputRef.current;
    if (el) {
      setSelStart(el.selectionStart ?? 0);
      setSelEnd(el.selectionEnd ?? 0);
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newText = e.target.value;
    if (localSegments.length === 0) {
      setLocalSegments([{ text: newText, color: currentColor, font_name: currentFont, font_size: currentSize }]);
    } else {
      // Simple approach: reconstruct keeping segment boundaries where possible
      const oldText = getFullText();
      if (newText.length > oldText.length) {
        // Characters were added - insert into the segment at cursor
        const insertPos = e.target.selectionStart ?? newText.length;
        const addedCount = newText.length - oldText.length;
        const addedText = newText.slice(insertPos - addedCount, insertPos);

        const newSegs: StyledSegment[] = [];
        let pos = 0;
        let inserted = false;
        for (const seg of localSegments) {
          const segEnd = pos + seg.text.length;
          if (!inserted && insertPos - addedCount >= pos && insertPos - addedCount <= segEnd) {
            const localPos = insertPos - addedCount - pos;
            const before = seg.text.slice(0, localPos);
            const after = seg.text.slice(localPos);
            if (before || addedText || after) {
              newSegs.push({ ...seg, text: before + addedText + after });
            }
            inserted = true;
          } else {
            newSegs.push({ ...seg });
          }
          pos = segEnd;
        }
        if (!inserted) {
          newSegs.push({ text: addedText, color: currentColor, font_name: currentFont, font_size: currentSize });
        }
        setLocalSegments(newSegs.filter((s) => s.text.length > 0));
      } else {
        // Characters were removed
        const deleteCount = oldText.length - newText.length;
        const deleteStart = e.target.selectionStart ?? 0;
        const deleteEnd = deleteStart + deleteCount;

        const newSegs: StyledSegment[] = [];
        let pos = 0;
        for (const seg of localSegments) {
          const segEnd = pos + seg.text.length;
          if (segEnd <= deleteStart || pos >= deleteEnd) {
            newSegs.push({ ...seg });
          } else {
            const before = seg.text.slice(0, Math.max(0, deleteStart - pos));
            const after = seg.text.slice(Math.max(0, deleteEnd - pos));
            const remaining = before + after;
            if (remaining.length > 0) {
              newSegs.push({ ...seg, text: remaining });
            }
          }
          pos = segEnd;
        }
        setLocalSegments(newSegs);
      }
    }
  };

  const applyStyleToSelection = (styleName: string) => {
    if (selStart === selEnd) return;
    const start = Math.min(selStart, selEnd);
    const end = Math.max(selStart, selEnd);

    const newSegs: StyledSegment[] = [];
    let pos = 0;

    for (const seg of localSegments) {
      const segEnd = pos + seg.text.length;

      if (segEnd <= start || pos >= end) {
        // Outside selection
        newSegs.push({ ...seg });
      } else {
        // Overlaps with selection - split
        const overlapStart = Math.max(start, pos) - pos;
        const overlapEnd = Math.min(end, segEnd) - pos;

        const before = seg.text.slice(0, overlapStart);
        const selected = seg.text.slice(overlapStart, overlapEnd);
        const after = seg.text.slice(overlapEnd);

        if (before) newSegs.push({ ...seg, text: before });
        if (selected) {
          const existingStyles = seg.font_style ? [...seg.font_style] : [];
          const styleIdx = existingStyles.indexOf(styleName);
          if (styleIdx >= 0) {
            existingStyles.splice(styleIdx, 1);
          } else {
            existingStyles.push(styleName);
          }
          newSegs.push({ ...seg, text: selected, font_style: existingStyles });
        }
        if (after) newSegs.push({ ...seg, text: after });
      }
      pos = segEnd;
    }
    setLocalSegments(newSegs);
  };

  const applyPropertyToSelection = (prop: "font_name" | "font_size" | "color", value: string | number) => {
    if (selStart === selEnd) {
      // No selection: update defaults for new text
      if (prop === "font_name") setCurrentFont(value as string);
      else if (prop === "font_size") setCurrentSize(value as number);
      else if (prop === "color") setCurrentColor(value as string);
      return;
    }

    const start = Math.min(selStart, selEnd);
    const end = Math.max(selStart, selEnd);
    const newSegs: StyledSegment[] = [];
    let pos = 0;

    for (const seg of localSegments) {
      const segEnd = pos + seg.text.length;
      if (segEnd <= start || pos >= end) {
        newSegs.push({ ...seg });
      } else {
        const overlapStart = Math.max(start, pos) - pos;
        const overlapEnd = Math.min(end, segEnd) - pos;

        const before = seg.text.slice(0, overlapStart);
        const selected = seg.text.slice(overlapStart, overlapEnd);
        const after = seg.text.slice(overlapEnd);

        if (before) newSegs.push({ ...seg, text: before });
        if (selected) newSegs.push({ ...seg, text: selected, [prop]: value });
        if (after) newSegs.push({ ...seg, text: after });
      }
      pos = segEnd;
    }
    setLocalSegments(newSegs);
  };

  const handleSave = () => {
    const text = getFullText();
    onSave(localSegments, text);
    onClose();
  };

  const fullText = getFullText();

  return (
    <Popover
      open={open}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      transformOrigin={{ vertical: "top", horizontal: "left" }}
    >
      <Box sx={{ width: 420, maxWidth: "90vw", p: 2 }}>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
          Rich Text Editor
        </Typography>

        {/* Toolbar row 1: style buttons */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 1, flexWrap: "wrap" }}>
          <Tooltip title="Bold">
            <IconButton size="small" onClick={() => applyStyleToSelection("Bold")}>
              <FormatBoldIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Italic">
            <IconButton size="small" onClick={() => applyStyleToSelection("Italic")}>
              <FormatItalicIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Underline">
            <IconButton size="small" onClick={() => applyStyleToSelection("Underline")}>
              <FormatUnderlinedIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Strikethrough">
            <IconButton size="small" onClick={() => applyStyleToSelection("Strikethrough")}>
              <StrikethroughSIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Superscript">
            <IconButton size="small" onClick={() => applyStyleToSelection("Superscript")}>
              <SuperscriptIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Subscript">
            <IconButton size="small" onClick={() => applyStyleToSelection("Subscript")}>
              <SubscriptIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Box sx={{ mx: 0.5, borderLeft: 1, borderColor: "divider", height: 24 }} />

          <Tooltip title="Text direction">
            <ToggleButtonGroup
              value={isRtl ? "rtl" : "ltr"}
              exclusive
              size="small"
              onChange={(_, val) => { if (val) setIsRtl(val === "rtl"); }}
            >
              <ToggleButton value="ltr" sx={{ p: 0.5 }}>
                <FormatTextdirectionLToRIcon sx={{ fontSize: 18 }} />
              </ToggleButton>
              <ToggleButton value="rtl" sx={{ p: 0.5 }}>
                <FormatTextdirectionRToLIcon sx={{ fontSize: 18 }} />
              </ToggleButton>
            </ToggleButtonGroup>
          </Tooltip>
        </Box>

        {/* Toolbar row 2: font, size, color */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
          <Select
            value={currentFont}
            onChange={(e) => {
              setCurrentFont(e.target.value);
              applyPropertyToSelection("font_name", e.target.value);
            }}
            size="small"
            sx={{ fontSize: "0.7rem", minWidth: 120, maxWidth: 180 }}
          >
            {(fonts.length > 0 ? fonts : ["arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf"]).map((f) => (
              <MenuItem key={f} value={f} sx={{ fontSize: "0.7rem" }} title={f}>{f.replace(/\.(ttf|otf)$/i, "")}</MenuItem>
            ))}
          </Select>

          <TextField
            type="number"
            value={currentSize}
            onChange={(e) => {
              const v = Number(e.target.value);
              setCurrentSize(v);
              applyPropertyToSelection("font_size", v);
            }}
            size="small"
            inputProps={{ min: 1, max: 200, step: 1 }}
            sx={{ width: 70 }}
            label="Size"
          />

          <TextField
            type="color"
            value={currentColor}
            onChange={(e) => {
              setCurrentColor(e.target.value);
              applyPropertyToSelection("color", e.target.value);
            }}
            size="small"
            sx={{ width: 60, "& input": { cursor: "pointer", p: 0.5 } }}
            label="Color"
          />
        </Box>

        {/* Text input */}
        <TextField
          inputRef={inputRef}
          fullWidth
          multiline
          minRows={2}
          maxRows={4}
          value={fullText}
          onChange={handleTextChange}
          onSelect={handleSelectionChange}
          onClick={handleSelectionChange}
          onKeyUp={handleSelectionChange}
          placeholder="Enter text..."
          sx={{
            mb: 1.5,
            "& .MuiInputBase-input": {
              direction: isRtl ? "rtl" : "ltr",
            },
          }}
        />

        {/* Segments preview */}
        {localSegments.length > 0 && (
          <Box sx={{ mb: 1.5, display: "flex", flexWrap: "wrap", gap: 0, p: 1, bgcolor: "background.default", borderRadius: 1, minHeight: 28, direction: isRtl ? "rtl" : "ltr" }}>
            {localSegments.map((seg, i) => (
              <span
                key={`${i}-${seg.text.slice(0, 8)}`}
                style={{
                  color: seg.color,
                  fontWeight: seg.font_style?.includes("Bold") ? 700 : 400,
                  fontStyle: seg.font_style?.includes("Italic") ? "italic" : "normal",
                  textDecoration: [
                    seg.font_style?.includes("Underline") ? "underline" : "",
                    seg.font_style?.includes("Strikethrough") ? "line-through" : "",
                  ].filter(Boolean).join(" ") || "none",
                  fontSize: seg.font_size ? `${Math.min(seg.font_size, 20)}px` : "12px",
                  verticalAlign: seg.font_style?.includes("Superscript") ? "super" : seg.font_style?.includes("Subscript") ? "sub" : "baseline",
                }}
              >
                {seg.text}
              </span>
            ))}
          </Box>
        )}

        {/* Actions */}
        <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 1 }}>
          <Button size="small" onClick={onClose}>Cancel</Button>
          <Button size="small" variant="contained" onClick={handleSave}>Apply</Button>
        </Box>
      </Box>
    </Popover>
  );
}
