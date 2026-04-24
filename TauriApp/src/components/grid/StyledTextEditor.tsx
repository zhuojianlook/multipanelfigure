/*
 * StyledTextEditor — contentEditable-based replacement for the
 * textarea+overlay pair previously used to edit per-char-styled
 * header/label text. The visible styled glyphs ARE the editable
 * element, so selection/clicks/typing land exactly on what the user
 * sees. Eliminates the alignment drift that comes from keeping a
 * transparent-text textarea aligned with a separate styled overlay.
 *
 * Text model is plain-text (`text: string`) + an optional
 * `styledSegments` array whose concatenated `.text` must equal
 * `text`. When they don't match (e.g. user just typed a new char),
 * the component falls back to rendering plain text with the default
 * colour — the parent is free to clear/resync segments on the next
 * styling patch.
 *
 * Parents get text changes via `onTextChange`, and caret-range
 * selections via `onSelectionChange` (in character offsets counted
 * against the plain-text, with <br> elements counted as one "\n").
 */

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
} from "react";
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
} from "react";

export interface StyledTextSegment {
  text: string;
  color: string;
  font_name?: string;
  font_size?: number;
  font_style?: string[];
}

export interface StyledTextEditorHandle {
  focus: () => void;
  blur: () => void;
  /** Current caret/selection range as char offsets. Returns null when
   *  the editor isn't focused. */
  getSelection: () => { start: number; end: number } | null;
  setSelection: (start: number, end: number) => void;
}

export interface StyledTextEditorProps {
  text: string;
  styledSegments?: StyledTextSegment[];
  /** Element-level default text colour — applied to any chars/segs
   *  that don't specify their own. */
  defaultColor?: string;
  /** Element-level bold/italic/underline/strike — applied as container
   *  defaults so segs that don't explicitly override still inherit. */
  fontStyle?: string[];
  className?: string;
  style?: CSSProperties;

  onTextChange: (text: string) => void;
  /** Fired on mousedown BEFORE focus moves. Lets a parent toolbar
   *  snapshot the current selection before its own click steals focus. */
  onBeforeAction?: () => void;
  onSelectionChange?: (sel: { start: number; end: number } | null) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onClick?: (e: ReactMouseEvent<HTMLDivElement>) => void;
  onKeyDown?: (e: ReactKeyboardEvent<HTMLDivElement>) => void;
  onContextMenu?: (e: ReactMouseEvent<HTMLDivElement>) => void;
}

/* ── styling helpers ──────────────────────────────────────────── */

function segSpanStyle(
  seg: StyledTextSegment,
  fallbackColor?: string,
): CSSProperties {
  const st = seg.font_style ?? [];
  const bold = st.includes("Bold");
  const italic = st.includes("Italic");
  const underline = st.includes("Underline");
  const strike = st.includes("Strikethrough");
  const sup = st.includes("Superscript");
  const sub = st.includes("Subscript");
  const deco: string[] = [];
  if (underline) deco.push("underline");
  if (strike) deco.push("line-through");
  return {
    color: seg.color || fallbackColor || "inherit",
    fontSize:
      sup || sub
        ? seg.font_size
          ? `${seg.font_size * 0.7}px`
          : "0.7em"
        : seg.font_size
        ? `${seg.font_size}px`
        : undefined,
    fontFamily: seg.font_name
      ? seg.font_name.replace(/\.(ttf|otf|ttc)$/i, "")
      : undefined,
    fontWeight: bold ? 700 : 400,
    fontStyle: italic ? "italic" : "normal",
    textDecoration: deco.length > 0 ? deco.join(" ") : undefined,
    verticalAlign: sup ? "super" : sub ? "sub" : undefined,
  };
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cssObjectToString(css: CSSProperties): string {
  const out: string[] = [];
  for (const [k, v] of Object.entries(css)) {
    if (v === undefined || v === null) continue;
    const kebab = k.replace(/([A-Z])/g, "-$1").toLowerCase();
    out.push(`${kebab}:${String(v).replace(/"/g, "'")}`);
  }
  return out.join(";");
}

function buildHtml(
  text: string,
  segs: StyledTextSegment[] | undefined,
  defaultColor?: string,
): string {
  if (text === "") return "";
  const renderPlain = (t: string): string => {
    const lines = t.split("\n").map(escapeHtml);
    return lines.join("<br>");
  };
  if (!segs || segs.length === 0) {
    return renderPlain(text);
  }
  const concat = segs.map((s) => s.text).join("");
  if (concat !== text) {
    // Segs are out-of-sync with plain text (user probably just typed).
    // Render plain to keep the DOM consistent with the text model.
    return renderPlain(text);
  }
  let html = "";
  for (const seg of segs) {
    const style = cssObjectToString(segSpanStyle(seg, defaultColor));
    const parts = seg.text.split("\n").map(escapeHtml).join("<br>");
    html += `<span style="${style}">${parts}</span>`;
  }
  return html;
}

/* ── DOM ↔ char-offset helpers ────────────────────────────────── */

/** Extract plain text from a contentEditable root. <br> → "\n".
 *  innerText would work in Chromium but isn't 100% reliable cross-WebView
 *  (and can depend on rendered layout), so we walk the DOM explicitly. */
function extractText(root: Node): string {
  let out = "";
  const walk = (n: Node) => {
    if (n.nodeType === Node.TEXT_NODE) {
      out += n.textContent || "";
      return;
    }
    if (n.nodeType === Node.ELEMENT_NODE) {
      if ((n as Element).tagName === "BR") {
        out += "\n";
        return;
      }
      for (const c of Array.from(n.childNodes)) walk(c);
    }
  };
  walk(root);
  return out;
}

/** Convert a DOM Range endpoint (node + offset) to a char index within
 *  the plain text produced by extractText(root). */
function endpointToCharIndex(root: Node, node: Node, offset: number): number {
  let count = 0;
  let found = false;

  const walk = (n: Node): void => {
    if (found) return;
    if (n === node) {
      if (n.nodeType === Node.TEXT_NODE) {
        count += offset;
      } else {
        // Element endpoint: `offset` is the index into childNodes.
        const children = Array.from(n.childNodes);
        for (let i = 0; i < offset && i < children.length; i++) {
          walk(children[i]);
          if (found) return;
        }
      }
      found = true;
      return;
    }
    if (n.nodeType === Node.TEXT_NODE) {
      count += (n.textContent || "").length;
      return;
    }
    if ((n as Element).tagName === "BR") {
      count += 1;
      return;
    }
    for (const c of Array.from(n.childNodes)) {
      walk(c);
      if (found) return;
    }
  };
  walk(root);
  return count;
}

/** Inverse: given a char offset, return a DOM position suitable for
 *  Range.setStart/setEnd. */
function charIndexToEndpoint(
  root: Node,
  target: number,
): { node: Node; offset: number } {
  let remaining = target;
  let fallback: { node: Node; offset: number } = { node: root, offset: 0 };

  const walk = (n: Node): { node: Node; offset: number } | null => {
    if (n.nodeType === Node.TEXT_NODE) {
      const len = (n.textContent || "").length;
      if (remaining <= len) return { node: n, offset: remaining };
      remaining -= len;
      fallback = { node: n, offset: len };
      return null;
    }
    if ((n as Element).tagName === "BR") {
      if (remaining === 0) {
        const parent = n.parentNode!;
        const idx = Array.from(parent.childNodes).indexOf(n as ChildNode);
        return { node: parent, offset: idx };
      }
      remaining -= 1;
      const parent = n.parentNode!;
      const idx = Array.from(parent.childNodes).indexOf(n as ChildNode);
      fallback = { node: parent, offset: idx + 1 };
      return null;
    }
    for (const c of Array.from(n.childNodes)) {
      const found = walk(c);
      if (found) return found;
    }
    return null;
  };
  const found = walk(root);
  return found ?? fallback;
}

function readSelectionOffsets(
  root: HTMLElement,
): { start: number; end: number } | null {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0);
  if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
    return null;
  }
  const start = endpointToCharIndex(root, range.startContainer, range.startOffset);
  const end = endpointToCharIndex(root, range.endContainer, range.endOffset);
  return { start, end };
}

function applySelectionOffsets(
  root: HTMLElement,
  start: number,
  end: number,
): void {
  const s = charIndexToEndpoint(root, start);
  const e = charIndexToEndpoint(root, end);
  const range = document.createRange();
  try {
    range.setStart(s.node, s.offset);
    range.setEnd(e.node, e.offset);
  } catch {
    return;
  }
  const sel = window.getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  sel.addRange(range);
}

/* ── Component ────────────────────────────────────────────────── */

export const StyledTextEditor = forwardRef<
  StyledTextEditorHandle,
  StyledTextEditorProps
>(function StyledTextEditor(props, ref) {
  const {
    text,
    styledSegments,
    defaultColor,
    fontStyle = [],
    className,
    style,
    onTextChange,
    onBeforeAction,
    onSelectionChange,
    onFocus,
    onBlur,
    onClick,
    onKeyDown,
    onContextMenu,
  } = props;

  const rootRef = useRef<HTMLDivElement>(null);
  // Flips true for one tick when the user types — lets the layout effect
  // that syncs model→DOM skip the frame so we don't clobber the caret
  // position the browser just placed.
  const skipNextSync = useRef(false);

  const html = useMemo(
    () => buildHtml(text, styledSegments, defaultColor),
    [text, styledSegments, defaultColor],
  );

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (skipNextSync.current) {
      skipNextSync.current = false;
      return;
    }
    if (root.innerHTML === html) return;
    const hadFocus = document.activeElement === root;
    const savedSel = hadFocus ? readSelectionOffsets(root) : null;
    root.innerHTML = html;
    if (hadFocus && savedSel) {
      try {
        applySelectionOffsets(root, savedSel.start, savedSel.end);
      } catch {
        /* best-effort */
      }
    }
  }, [html]);

  useImperativeHandle(
    ref,
    () => ({
      focus: () => rootRef.current?.focus(),
      blur: () => rootRef.current?.blur(),
      getSelection: () => {
        const root = rootRef.current;
        if (!root) return null;
        return readSelectionOffsets(root);
      },
      setSelection: (start, end) => {
        const root = rootRef.current;
        if (!root) return;
        root.focus();
        applySelectionOffsets(root, start, end);
      },
    }),
    [],
  );

  // Fire selection changes to the parent so the floating toolbar can
  // track caret/range updates without polling.
  useEffect(() => {
    if (!onSelectionChange) return;
    const handler = () => {
      const root = rootRef.current;
      if (!root) return;
      if (document.activeElement !== root) return;
      onSelectionChange(readSelectionOffsets(root));
    };
    document.addEventListener("selectionchange", handler);
    return () => document.removeEventListener("selectionchange", handler);
  }, [onSelectionChange]);

  const handleInput = () => {
    const root = rootRef.current;
    if (!root) return;
    skipNextSync.current = true;
    const newText = extractText(root);
    onTextChange(newText);
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLDivElement>) => {
    // Plain-text paste — strip any formatting the source supplied.
    e.preventDefault();
    const pasted = e.clipboardData.getData("text/plain");
    if (!pasted) return;
    const root = rootRef.current;
    if (!root) return;
    const sel = readSelectionOffsets(root) ?? { start: text.length, end: text.length };
    const lo = Math.min(sel.start, sel.end);
    const hi = Math.max(sel.start, sel.end);
    const newText = text.slice(0, lo) + pasted + text.slice(hi);
    skipNextSync.current = false; // we WANT the model→DOM sync to run
    onTextChange(newText);
    // After the parent re-renders with new text, place caret after
    // the pasted region.
    requestAnimationFrame(() => {
      if (rootRef.current) {
        applySelectionOffsets(
          rootRef.current,
          lo + pasted.length,
          lo + pasted.length,
        );
      }
    });
  };

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    // Suppress the built-in rich-text shortcuts that contentEditable
    // normally honours — we're a plain-text editor with external
    // styling, so Ctrl/Cmd+B/I/U mustn't wrap selections in <b>/<i>/<u>
    // tags inside our root.
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
      const k = e.key.toLowerCase();
      if (k === "b" || k === "i" || k === "u") {
        e.preventDefault();
      }
    }
    onKeyDown?.(e);
  };

  return (
    <div
      ref={rootRef}
      className={className}
      contentEditable
      suppressContentEditableWarning
      spellCheck={false}
      style={{
        // Element-level styles inherited by unstyled chars.
        fontWeight: fontStyle.includes("Bold") ? 700 : 400,
        fontStyle: fontStyle.includes("Italic") ? "italic" : "normal",
        color: defaultColor,
        outline: "none",
        cursor: "text",
        ...style,
      }}
      onInput={handleInput}
      onPaste={handlePaste}
      onMouseDown={onBeforeAction}
      onFocus={onFocus}
      onBlur={onBlur}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      onContextMenu={onContextMenu}
    />
  );
});
