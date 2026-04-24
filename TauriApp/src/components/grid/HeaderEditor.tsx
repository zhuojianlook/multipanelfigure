/*
 * HeaderEditor — thin wrapper around StyledTextEditor used by the
 * PanelGrid for all four header/label edit surfaces (secondary col
 * header, primary col label, secondary row header, primary row label).
 *
 * Owns its own StyledTextEditorHandle ref so the parent's
 * toolbar/selection machinery can call focus/setSelection/getSelection
 * without having to allocate a ref per instance in render. On focus,
 * it calls `onActivate` with (target, handle) so PanelGrid can pick
 * this instance as "the one the toolbar is anchored to".
 */

import { useRef } from "react";
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
} from "react";
import {
  StyledTextEditor,
  type StyledTextEditorHandle,
  type StyledTextSegment,
} from "./StyledTextEditor";

/** Target descriptor used by the parent to map an editor back to its
 *  model slot (header group or label). Mirrors the existing
 *  `toolbarTarget` shape in PanelGrid. */
export type HeaderEditorTarget =
  | { type: "header"; axis: "col" | "row"; level: number; groupIdx: number }
  | { type: "colLabel" | "rowLabel"; axis: "col" | "row"; index: number };

export interface HeaderEditorProps {
  target: HeaderEditorTarget;
  text: string;
  styledSegments?: StyledTextSegment[];
  defaultColor?: string;
  /** Element-level bold/italic. Underline/strikethrough from the
   *  element are best applied via `style.textDecoration` by the caller. */
  fontStyle?: string[];
  className?: string;
  style?: CSSProperties;
  onTextChange: (text: string) => void;
  onActivate: (target: HeaderEditorTarget, handle: StyledTextEditorHandle) => void;
  onSelectionNonEmpty?: (sel: { start: number; end: number }) => void;
  onShiftEnter?: (sel: { start: number; end: number }) => void;
  /** Regular Enter collapses focus (blur) to match the existing
   *  behaviour of the row-label textarea. Disabled by default because
   *  col headers previously allowed inline \n via Shift+Enter only. */
  enterBlurs?: boolean;
  onClick?: (e: ReactMouseEvent<HTMLDivElement>) => void;
  onContextMenu?: (e: ReactMouseEvent<HTMLDivElement>) => void;
  onBeforeAction?: () => void;
}

export function HeaderEditor(props: HeaderEditorProps) {
  const {
    target,
    text,
    styledSegments,
    defaultColor,
    fontStyle,
    className,
    style,
    onTextChange,
    onActivate,
    onSelectionNonEmpty,
    onShiftEnter,
    enterBlurs,
    onClick,
    onContextMenu,
    onBeforeAction,
  } = props;
  const ref = useRef<StyledTextEditorHandle>(null);

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" && e.shiftKey) {
      e.preventDefault();
      const sel = ref.current?.getSelection() ?? { start: text.length, end: text.length };
      onShiftEnter?.(sel);
      return;
    }
    if (enterBlurs && e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ref.current?.blur();
      return;
    }
  };

  return (
    <StyledTextEditor
      ref={ref}
      text={text}
      styledSegments={styledSegments}
      defaultColor={defaultColor}
      fontStyle={fontStyle}
      className={className}
      style={style}
      onTextChange={onTextChange}
      onFocus={() => {
        if (ref.current) onActivate(target, ref.current);
      }}
      onSelectionChange={(sel) => {
        if (sel && sel.start !== sel.end) onSelectionNonEmpty?.(sel);
      }}
      onKeyDown={handleKeyDown}
      onClick={onClick}
      onContextMenu={onContextMenu}
      onBeforeAction={onBeforeAction}
    />
  );
}
