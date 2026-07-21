/* ──────────────────────────────────────────────────────────
   StyledTextField — drop-in replacement for an MUI <TextField/>
   on places where the body text should support per-character
   formatting (different font, size, bold/italic/underline/
   strikethrough/sub/sup, colour).

   This wraps the existing <StyledTextEditor/> (which is otherwise
   only used by HeaderEditor inside PanelGrid) into a self-
   contained component:
     • single-line by default, multi-line opt-in
     • MUI-styled outlined container that mirrors a real TextField
       (label, border, focus ring) so it looks at home in dialogs
     • inline floating toolbar above the editor whenever a non-
       empty selection exists, exposing the same formatting
       options as the PanelGrid header toolbar (B/I/U/S, sup/sub,
       font picker, size, colour)

   Output contract:
     • text: plain text mirror of the editor content
     • segments: optional [{ text, color, font_name?, font_size?,
       font_style? }] runs whose concatenated .text MUST equal
       `text`.  When the user hasn't applied any formatting the
       parent gets an empty / single-segment array; consumers
       should handle both shapes.

   Used by EditPanelDialog for: panel.labels[i].text, scale_bar.label,
   symbols[i].label_text, lines[i].measure_text, areas[i].measure_text.
   Also reusable in Collage.
   ────────────────────────────────────────────────────────── */

import { useEffect, useId, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import {
  Box,
  IconButton,
  Popper,
  Paper,
  Typography,
  Button,
} from "@mui/material";
import FormatBoldIcon from "@mui/icons-material/FormatBold";
import FormatItalicIcon from "@mui/icons-material/FormatItalic";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import SuperscriptIcon from "@mui/icons-material/Superscript";
import SubscriptIcon from "@mui/icons-material/Subscript";
import FormatClearIcon from "@mui/icons-material/FormatClear";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import {
  StyledTextEditor,
  type StyledTextEditorHandle,
  type StyledTextSegment,
} from "../grid/StyledTextEditor";

/* ── segment helpers ──────────────────────────────────────────
   The segments array is a string of runs whose concatenated text
   MUST equal `text`.  Empty arrays mean "no formatting; use
   element defaults" — we treat those as equivalent to a single
   segment with the defaults.
   ───────────────────────────────────────────────────────────── */

/** Normalize an arbitrary segments array against `text` so that
 *  concatenated lengths match.  Used after any text edit to keep
 *  segments aligned.  If segments is empty / null / out-of-sync,
 *  rebuilds a single "default style" segment covering the whole
 *  text. */
export function reconcileSegments(
  text: string,
  segments: StyledTextSegment[] | null | undefined,
  defaultColor: string,
): StyledTextSegment[] {
  if (!segments || segments.length === 0) {
    return text.length > 0 ? [{ text, color: defaultColor }] : [];
  }
  const total = segments.reduce((acc, s) => acc + s.text.length, 0);
  if (total === text.length) return segments;
  // Rough realignment: walk segments left-to-right, take chars
  // from text up to each segment's original length, drop trailing
  // segments if text is shorter, append a fresh default segment for
  // any extra trailing text.
  const out: StyledTextSegment[] = [];
  let cursor = 0;
  for (const seg of segments) {
    if (cursor >= text.length) break;
    const take = text.slice(cursor, cursor + seg.text.length);
    if (!take) break;
    out.push({ ...seg, text: take });
    cursor += take.length;
  }
  if (cursor < text.length) {
    const tail = text.slice(cursor);
    out.push({ text: tail, color: defaultColor });
  }
  return out;
}

/** Split a segments array at a char offset.  Returns two segment
 *  arrays whose concatenated text reconstructs the original. */
function splitAt(
  segments: StyledTextSegment[],
  offset: number,
): [StyledTextSegment[], StyledTextSegment[]] {
  if (offset <= 0) return [[], segments];
  const left: StyledTextSegment[] = [];
  const right: StyledTextSegment[] = [];
  let cursor = 0;
  for (const seg of segments) {
    if (cursor >= offset) {
      right.push(seg);
      cursor += seg.text.length;
      continue;
    }
    const end = cursor + seg.text.length;
    if (end <= offset) {
      left.push(seg);
    } else {
      const cut = offset - cursor;
      left.push({ ...seg, text: seg.text.slice(0, cut) });
      right.push({ ...seg, text: seg.text.slice(cut) });
    }
    cursor = end;
  }
  return [left, right];
}

/** Merge adjacent segments that have identical formatting — keeps
 *  the array compact after a series of edits. */
function mergeAdjacent(segments: StyledTextSegment[]): StyledTextSegment[] {
  if (segments.length <= 1) return segments;
  const out: StyledTextSegment[] = [];
  for (const seg of segments) {
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.color === seg.color &&
      (prev.font_name || "") === (seg.font_name || "") &&
      (prev.font_size || 0) === (seg.font_size || 0) &&
      (prev.font_style || []).slice().sort().join("|") ===
        (seg.font_style || []).slice().sort().join("|")
    ) {
      prev.text += seg.text;
    } else {
      out.push({ ...seg });
    }
  }
  return out;
}

/** Apply a partial formatting patch to every character in the
 *  half-open range [start, end) within `segments`.  Returns a new
 *  segments array.  The patch may include any of: color, font_name,
 *  font_size, font_style (replace), or `toggleStyle` (xor a single
 *  style name into the existing list). */
export function applyStyleToRange(
  segments: StyledTextSegment[],
  start: number,
  end: number,
  patch: Partial<Omit<StyledTextSegment, "text">> & { toggleStyle?: string; clearAll?: boolean },
  defaultColor: string,
): StyledTextSegment[] {
  if (end <= start || segments.length === 0) return segments;
  const [left, rest] = splitAt(segments, start);
  const [middle, right] = splitAt(rest, end - start);
  const patched = middle.map<StyledTextSegment>((seg) => {
    if (patch.clearAll) {
      return { text: seg.text, color: defaultColor };
    }
    let nextStyle = seg.font_style ? [...seg.font_style] : [];
    if (patch.toggleStyle) {
      const t = patch.toggleStyle;
      if (nextStyle.includes(t)) nextStyle = nextStyle.filter((s) => s !== t);
      else nextStyle.push(t);
    }
    if (patch.font_style !== undefined) nextStyle = patch.font_style;
    return {
      ...seg,
      color: patch.color !== undefined ? patch.color : seg.color,
      font_name: patch.font_name !== undefined ? patch.font_name : seg.font_name,
      font_size: patch.font_size !== undefined ? patch.font_size : seg.font_size,
      font_style: nextStyle,
    };
  });
  return mergeAdjacent([...left, ...patched, ...right]);
}

/* ── component ────────────────────────────────────────────────── */

export interface StyledTextFieldProps {
  /** Plain-text mirror.  Concatenated segments must equal this. */
  text: string;
  /** Per-character formatting runs.  Pass `[]` or `undefined` for
   *  no formatting (the element defaults apply to every char). */
  segments?: StyledTextSegment[];
  /** Element-wide default colour, applied to chars with no per-seg
   *  override. */
  defaultColor?: string;
  /** Element-wide bold/italic/underline/strike — applied as a
   *  container default for chars whose seg doesn't override. */
  fontStyle?: string[];
  /** Element-wide base font size (points).  Used as the displayed
   *  value in the toolbar's size +/- widget when the current
   *  selection's segments don't carry their own font_size override.
   *  Without this the size readout shows "—" by default which makes
   *  the user think they have to set a size from scratch. */
  baseFontSize?: number;
  /** Field label — same role as MUI TextField's `label` prop. */
  label?: string;
  /** Placeholder shown when text is empty. */
  placeholder?: string;
  /** Multi-line mode (Shift+Enter inserts newlines, Enter blurs). */
  multiline?: boolean;
  /** Width: pass-through to MUI sx for the outer container. */
  fullWidth?: boolean;
  size?: "small" | "medium";
  disabled?: boolean;
  /** Available font filenames for the inline font picker.  When
   *  empty / not passed, the font dropdown is hidden. */
  fonts?: string[];
  onChange: (text: string, segments: StyledTextSegment[]) => void;
}

export function StyledTextField({
  text,
  segments,
  defaultColor = "#000000",
  fontStyle,
  baseFontSize,
  label,
  placeholder,
  multiline = false,
  fullWidth = true,
  size = "small",
  disabled = false,
  fonts = [],
  onChange,
}: StyledTextFieldProps) {
  const editorRef = useRef<StyledTextEditorHandle>(null);
  // Both a ref (for synchronous reads inside event handlers, e.g. the
  // outside-click handler) AND a state slot (so the Popper auto-reposi-
  // tions when the DOM node attaches).  The previous "ref only" setup
  // could leave the Popper rendering at (0, 0) because anchorEl was
  // null on the first render and the Popper didn't know to update
  // when the ref later attached.
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerEl, setContainerEl] = useState<HTMLDivElement | null>(null);
  const colorInputRef = useRef<HTMLInputElement | null>(null);
  const fieldId = useId();
  const [focused, setFocused] = useState(false);
  // Cached selection range — the toolbar's button mousedown steals
  // focus from the contentEditable, so we snapshot the current
  // range on toolbar mousedown and use that snapshot to apply the
  // style after the focus dance settles.
  const cachedSelRef = useRef<{ start: number; end: number } | null>(null);
  const [hasSel, setHasSel] = useState(false);
  // Open state for the inline Font dropdown.  We render a plain
  // <Paper> positioned by CSS directly below the Font button —
  // no MUI Menu / Popover, because their portal-based positioning
  // misbehaves when nested inside our parent <Popper>.
  const [fontMenuOpen, setFontMenuOpen] = useState(false);

  // Keep segments coherent with text — if the parent passes
  // inconsistent data (e.g. only updates text without segments),
  // we reconcile here so the editor doesn't render glitched runs.
  const safeSegments = reconcileSegments(text, segments, defaultColor);

  const fireChange = (newText: string, newSegs: StyledTextSegment[]) => {
    onChange(newText, mergeAdjacent(reconcileSegments(newText, newSegs, defaultColor)));
  };

  // Inline toolbar visibility: show whenever a non-empty selection
  // exists in the editor.  Hidden on blur (except when focus moves
  // INTO the toolbar — handled below via the mousedown snapshot).
  const toolbarOpen = focused && hasSel;

  const handleTextChange = (newText: string) => {
    fireChange(newText, safeSegments);
  };

  const handleSelectionChange = (sel: { start: number; end: number } | null) => {
    if (sel && sel.start !== sel.end) {
      cachedSelRef.current = sel;
      setHasSel(true);
    } else if (!sel) {
      setHasSel(false);
    }
    // Don't drop cachedSel on collapse — toolbar buttons need it.
  };

  const applyPatch = (patch: Parameters<typeof applyStyleToRange>[3]) => {
    const sel = cachedSelRef.current;
    if (!sel) return;
    const nextSegs = applyStyleToRange(safeSegments, sel.start, sel.end, patch, defaultColor);
    fireChange(text, nextSegs);
    // Any patch dismisses any open dropdown (font picker etc.).
    setFontMenuOpen(false);
    // Re-focus + restore the selection so the user can chain
    // formatting actions without re-selecting.
    requestAnimationFrame(() => {
      editorRef.current?.focus();
      editorRef.current?.setSelection(sel.start, sel.end);
    });
  };

  /** Buttons that open popups (font menu, size +/-, colour picker)
   *  need to:
   *    1. Prevent default on mousedown so focus doesn't leave the
   *       editor.  Without this, opening the menu collapses the
   *       editor's selection and `cachedSelRef` becomes irrelevant
   *       because the editor itself is the "anchor" for any
   *       subsequent focus restore.
   *    2. Snapshot the current selection right NOW (in case the
   *       user clicked the button without the selection-change
   *       handler having fired yet — possible on touch / very
   *       quick drag interactions).
   *  Returned from `swallowMouseDown` so callers can spread it
   *  on any toolbar element. */
  const swallowMouseDown = (e: ReactMouseEvent<HTMLElement>) => {
    e.preventDefault();
    // Snapshot fresh just in case.  getSelection() returns null
    // when the editor isn't focused; in that case we trust the
    // previously-cached value.
    const live = editorRef.current?.getSelection?.();
    if (live && live.start !== live.end) cachedSelRef.current = live;
  };

  // Inspect the styling at the current selection — for active-state
  // highlights on the toolbar buttons (so the user can see which
  // styles are already on).
  const selStyle = (() => {
    const sel = cachedSelRef.current;
    if (!sel || sel.start === sel.end) return null;
    // Find the segment(s) overlapping the range; take the first one
    // for "current" state.  A range that straddles different styles
    // shows the leading segment's state — minor UX caveat that
    // matches most word processors.
    let cursor = 0;
    for (const seg of safeSegments) {
      const end = cursor + seg.text.length;
      if (end > sel.start) {
        return seg;
      }
      cursor = end;
    }
    return safeSegments[safeSegments.length - 1] || null;
  })();
  const styleActive = (name: string) =>
    !!selStyle && Array.isArray(selStyle.font_style) && selStyle.font_style.includes(name);

  // Sync the toolbar's "preview" of the selected substring.
  const selPreview = (() => {
    const sel = cachedSelRef.current;
    if (!sel) return "";
    return text.slice(sel.start, sel.end);
  })();

  // Hide toolbar on outside click.  Clicks ON the toolbar itself
  // are handled by onMouseDown preventing the focus loss.
  // CRITICAL: MUI portals (the Font Menu's `<Menu>` opens its
  // surface in a portal at document.body, OUTSIDE our toolbar
  // Paper).  An "outside click" detector that only knows about
  // the toolbar's DOM subtree would treat clicks on menu items as
  // outside, drop hasSel + cachedSelRef, and tear down the toolbar
  // before applyPatch could fire.  Skip any click whose target is
  // inside an MUI Popover / Menu / Modal portal as well.
  useEffect(() => {
    if (!toolbarOpen) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Element | null;
      if (!target) return;
      if (containerRef.current && containerRef.current.contains(target)) return;
      const toolbar = document.getElementById(`stf-toolbar-${fieldId}`);
      if (toolbar && toolbar.contains(target)) return;
      // Bail when the click landed inside any MUI overlay surface —
      // these are rendered in portals so they're not descendants of
      // our toolbar DOM, but they're conceptually part of our UI
      // (font picker menu, colour picker, etc.).  closest() walks
      // up to the document root looking for the marker class.
      if (target.closest?.(".MuiPopover-root, .MuiMenu-root, .MuiModal-root, .MuiPopper-root")) return;
      // Outside click also closes our custom Font dropdown.  The
      // dropdown's own mousedown PreventDefaults so clicks INSIDE
      // it land here as "not outside" (the Paper is inside the
      // toolbar's Popper, which contains the click target).
      setFontMenuOpen(false);
      setHasSel(false);
      cachedSelRef.current = null;
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [toolbarOpen, fieldId]);

  return (
    <Box ref={(el: HTMLDivElement | null) => { containerRef.current = el; setContainerEl(el); }} sx={{ width: fullWidth ? "100%" : "auto", minWidth: 0 }}>
      {label && (
        <Typography
          variant="caption"
          sx={{
            display: "block",
            mb: 0.25,
            fontSize: size === "small" ? "0.65rem" : "0.75rem",
            color: focused ? "primary.main" : "text.secondary",
            transition: "color 120ms",
          }}
        >
          {label}
        </Typography>
      )}
      <Box
        // Mimic MUI outlined TextField chrome so this looks native
        // alongside the other dialog inputs.
        sx={{
          position: "relative",
          border: "1px solid",
          borderColor: focused ? "primary.main" : "divider",
          borderRadius: 1,
          bgcolor: disabled ? "action.disabledBackground" : "background.paper",
          transition: "border-color 120ms",
          "&:hover": { borderColor: disabled ? "divider" : (focused ? "primary.main" : "text.primary") },
          minHeight: size === "small" ? 30 : 38,
          px: 1, py: 0.5,
          opacity: disabled ? 0.6 : 1,
          cursor: disabled ? "not-allowed" : "text",
        }}
        onClick={() => editorRef.current?.focus()}
      >
        <StyledTextEditor
          ref={editorRef}
          text={text}
          styledSegments={safeSegments}
          defaultColor={defaultColor}
          fontStyle={fontStyle}
          style={{
            outline: "none",
            fontSize: size === "small" ? "0.8rem" : "0.9rem",
            // Match MUI's textbaseline so multi-line wraps stay aligned.
            lineHeight: 1.5,
            whiteSpace: multiline ? "pre-wrap" : "nowrap",
            overflow: multiline ? "auto" : "hidden",
            // Keep a single-line field actually single-line — strip newlines.
          }}
          onTextChange={(t) => handleTextChange(multiline ? t : t.replace(/\n+/g, " "))}
          onFocus={() => setFocused(true)}
          onBlur={() => {
            // Defer so a toolbar-button click can swap focus back
            // without immediately collapsing the panel.
            //
            // CRITICAL: when the Font Menu opens, MUI auto-focuses
            // its list item — that focus lands inside the Menu's
            // PORTAL (rendered at document.body), NOT inside our
            // toolbar Paper.  Without the portal check below the
            // blur handler would conclude "focus left our UI",
            // flip `focused` to false, collapse `toolbarOpen`,
            // unmount the Popper, and take the Menu down with it.
            // Symptom: clicking the font dropdown made the whole
            // toolbar disappear.  The .MuiPopover-root / .MuiMenu-
            // root / .MuiPopper-root / .MuiModal-root class check
            // tells the blur handler that these portal surfaces
            // are still "us" for the purpose of staying open.
            setTimeout(() => {
              const active = document.activeElement as Element | null;
              const toolbar = document.getElementById(`stf-toolbar-${fieldId}`);
              if (toolbar && active && toolbar.contains(active)) return;
              if (active && active.closest?.(".MuiPopover-root, .MuiMenu-root, .MuiModal-root, .MuiPopper-root")) return;
              setFocused(false);
              setHasSel(false);
            }, 0);
          }}
          onSelectionChange={handleSelectionChange}
          onKeyDown={(e) => {
            // Single-line mode: Enter blurs.  Multi-line: Enter inserts \n.
            if (!multiline && e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              editorRef.current?.blur();
            }
          }}
        />
        {!text && placeholder && (
          <Box
            sx={{
              position: "absolute",
              top: size === "small" ? 6 : 9,
              left: 9,
              fontSize: size === "small" ? "0.8rem" : "0.9rem",
              color: "text.disabled",
              pointerEvents: "none",
            }}
          >
            {placeholder}
          </Box>
        )}
      </Box>

      {/* Floating formatting toolbar.  Anchored to the field
          container, shown only when there's a real selection.
          Use the state-tracked `containerEl` (not the ref) so the
          Popper re-positions when the container DOM node mounts —
          otherwise the first render sees a null anchorEl and the
          Popper falls back to (0, 0) i.e. the top-left of the
          viewport.  `keepMounted` is intentionally false so the
          Popper unmounts cleanly when there's no selection. */}
      <Popper
        open={toolbarOpen && containerEl !== null}
        anchorEl={containerEl}
        placement="top-start"
        modifiers={[
          { name: "offset", options: { offset: [0, 6] } },
          { name: "flip", enabled: true },
          { name: "preventOverflow", enabled: true, options: { padding: 8 } },
        ]}
        sx={{ zIndex: 1500 }}
      >
        <Paper
          id={`stf-toolbar-${fieldId}`}
          elevation={6}
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.25,
            px: 0.5, py: 0.25,
            border: "1px solid",
            borderColor: "divider",
          }}
          // CRITICAL: preventDefault on mousedown so clicks inside the
          // toolbar don't move focus out of the editor.  Without this
          // the StyledTextEditor would blur, collapse its selection,
          // and the format button click would have no range to target.
          onMouseDown={(e) => e.preventDefault()}
        >
          {/* Native HTML `title` attributes instead of MUI <Tooltip>.
              MUI Tooltip uses its own Popper to position itself relative
              to the wrapped child; when that child sits inside ANOTHER
              Popper (this toolbar), the tooltip's anchor measurement
              fires before the parent Popper has finished positioning,
              and the tooltip falls back to (0, 0) — the upper-left
              corner of the viewport, outside the EditPanel modal.
              Browser-native title tooltips are positioned by the OS so
              they never have this problem; the small downside is a
              short delay before they appear, which is acceptable for
              a dense toolbar of icon buttons. */}
          <IconButton title="Bold (selection)" size="small" sx={{ p: 0.25, color: styleActive("Bold") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Bold" })}>
            <FormatBoldIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton title="Italic (selection)" size="small" sx={{ p: 0.25, color: styleActive("Italic") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Italic" })}>
            <FormatItalicIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton title="Underline (selection)" size="small" sx={{ p: 0.25, color: styleActive("Underline") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Underline" })}>
            <FormatUnderlinedIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton title="Strikethrough (selection)" size="small" sx={{ p: 0.25, color: styleActive("Strikethrough") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Strikethrough" })}>
            <StrikethroughSIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton title="Superscript (selection)" size="small" sx={{ p: 0.25, color: styleActive("Superscript") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Superscript" })}>
            <SuperscriptIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton title="Subscript (selection)" size="small" sx={{ p: 0.25, color: styleActive("Subscript") ? "primary.main" : "text.primary" }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ toggleStyle: "Subscript" })}>
            <SubscriptIcon sx={{ fontSize: 14 }} />
          </IconButton>

          {/* Font picker — fully custom inline dropdown instead of an
              MUI Menu / Select / Popover.  None of the MUI portal-
              based widgets positioned correctly when nested inside
              our parent <Popper> (the toolbar): the inner Popover's
              anchor-rect measurement fired before the outer Popper
              settled, so the menu opened at viewport (0,0).  Both
              `disablePortal` and explicit `anchorEl` attempts had
              their own quirks (transform-relative coordinates).
              A plain <Box> with `position: absolute, top: 100%, left: 0`
              relative to the Button wrapper is positioned by CSS
              directly under the button — no JS measurement, no
              portal indirection, can't be wrong. */}
          {fonts.length > 0 && (
            <Box sx={{ position: "relative", display: "inline-flex" }}>
              <Button
                title="Font (selection)"
                size="small"
                variant="text"
                onMouseDown={swallowMouseDown}
                onClick={() => setFontMenuOpen((v) => !v)}
                endIcon={<ArrowDropDownIcon sx={{ fontSize: 14 }} />}
                sx={{
                  ml: 0.5,
                  fontSize: "0.6rem",
                  textTransform: "none",
                  color: "text.primary",
                  minWidth: 0,
                  px: 0.5,
                  py: 0.25,
                  maxWidth: 110,
                  "& .MuiButton-endIcon": { ml: 0.25 },
                }}
              >
                <Box sx={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {selStyle?.font_name ? selStyle.font_name.replace(/\.(ttf|otf|ttc)$/i, "") : "Font"}
                </Box>
              </Button>
              {fontMenuOpen && (
                <Paper
                  elevation={6}
                  sx={{
                    position: "absolute",
                    top: "100%",
                    left: 0,
                    mt: 0.25,
                    minWidth: 180,
                    maxHeight: 280,
                    overflowY: "auto",
                    zIndex: 1600,
                    border: "1px solid",
                    borderColor: "divider",
                  }}
                  onMouseDown={(e) => e.preventDefault()}
                >
                  <Box
                    onMouseDown={swallowMouseDown}
                    onClick={() => { applyPatch({ font_name: undefined }); setFontMenuOpen(false); }}
                    sx={{
                      px: 1.25, py: 0.5,
                      fontSize: "0.7rem",
                      fontStyle: "italic",
                      color: "text.secondary",
                      cursor: "pointer",
                      "&:hover": { bgcolor: "action.hover" },
                    }}
                  >
                    (default)
                  </Box>
                  {fonts.slice().sort().map((f) => {
                    const selected = selStyle?.font_name === f;
                    return (
                      <Box
                        key={f}
                        onMouseDown={swallowMouseDown}
                        onClick={() => { applyPatch({ font_name: f }); setFontMenuOpen(false); }}
                        sx={{
                          px: 1.25, py: 0.5,
                          fontSize: "0.7rem",
                          cursor: "pointer",
                          bgcolor: selected ? "action.selected" : "transparent",
                          // Preview each option in its own typeface so
                          // the user can read what they're picking.
                          // Wrapped in a <span> further down to keep
                          // the actual menu label readable even when
                          // the font is a symbol font (Wingdings etc).
                          "&:hover": { bgcolor: "action.hover" },
                        }}
                      >
                        {f.replace(/\.(ttf|otf|ttc)$/i, "")}
                      </Box>
                    );
                  })}
                </Paper>
              )}
            </Box>
          )}

          {/* Size — +/- buttons instead of a typing input.  Each click
              is a synchronous applyPatch call; no focus ping-pong.
              The displayed value falls through three layers:
                1. selStyle.font_size  — explicit segment override
                2. baseFontSize        — element-level size (label.font_size)
                3. 12                  — hard fallback
              so the user sees the meaningful starting value instead
              of "—" before any explicit override. */}
          {(() => {
            const effectiveSize = selStyle?.font_size ?? baseFontSize ?? 12;
            return (
              <>
                <IconButton title="Decrease font size" size="small" sx={{ p: 0.25, ml: 0.5 }}
                  onMouseDown={swallowMouseDown}
                  onClick={() => applyPatch({ font_size: Math.max(4, effectiveSize - 1) })}>
                  <RemoveIcon sx={{ fontSize: 12 }} />
                </IconButton>
                <Box sx={{ fontSize: "0.6rem", color: "text.primary", minWidth: 18, textAlign: "center", userSelect: "none" }}>
                  {effectiveSize}
                </Box>
                <IconButton title="Increase font size" size="small" sx={{ p: 0.25 }}
                  onMouseDown={swallowMouseDown}
                  onClick={() => applyPatch({ font_size: Math.min(200, effectiveSize + 1) })}>
                  <AddIcon sx={{ fontSize: 12 }} />
                </IconButton>
              </>
            );
          })()}

          {/* Colour swatch — IconButton triggers the hidden native
              color input via .click().  The native picker opens in
              a separate OS window so it can't be embedded inline;
              we route through a ref instead of a label so the
              mousedown can preventDefault and keep our selection. */}
          <IconButton
            title="Colour (selection)"
            size="small"
            sx={{ p: 0.25, ml: 0.5 }}
            onMouseDown={swallowMouseDown}
            onClick={() => colorInputRef.current?.click()}
          >
            <Box sx={{
              width: 14, height: 14, borderRadius: 0.25,
              border: "1px solid",
              borderColor: "divider",
              bgcolor: selStyle?.color || defaultColor,
            }} />
          </IconButton>
          <input
            ref={colorInputRef}
            type="color"
            value={selStyle?.color || defaultColor}
            onChange={(e) => applyPatch({ color: e.target.value })}
            // Keep the input mounted but invisible — clicking the
            // swatch button calls .click() programmatically, which
            // opens the native picker without moving keyboard focus.
            // tabIndex=-1 keeps it out of tab order so Tab still
            // moves between meaningful UI.
            tabIndex={-1}
            style={{ position: "absolute", width: 0, height: 0, opacity: 0, pointerEvents: "none" }}
          />

          <IconButton title="Clear formatting on selection" size="small" sx={{ p: 0.25, ml: 0.5 }}
            onMouseDown={swallowMouseDown}
            onClick={() => applyPatch({ clearAll: true })}>
            <FormatClearIcon sx={{ fontSize: 14 }} />
          </IconButton>

          {selPreview && (
            <Typography variant="caption" sx={{ ml: 0.5, fontSize: "0.55rem", color: "text.secondary", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              "{selPreview}"
            </Typography>
          )}
        </Paper>
      </Popper>
    </Box>
  );
}
