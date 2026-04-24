/*
 * Dev-only visual harness for StyledTextEditor. Accessible via
 * http://localhost:1421/?editor-test — renders a few representative
 * header/label configurations side-by-side with the same visual
 * styling they use in PanelGrid, so we can verify selection lands on
 * visible glyphs, text centers correctly, and rotated rows align.
 */

import { useState } from "react";
import {
  StyledTextEditor,
  type StyledTextSegment,
} from "./StyledTextEditor";

interface CaseSpec {
  label: string;
  initialText: string;
  segs?: StyledTextSegment[];
  fontStyle?: string[];
  defaultColor?: string;
  rotated?: boolean;
}

const cases: CaseSpec[] = [
  {
    label: "Col header — plain (no styling)",
    initialText: "Column 2",
    defaultColor: "#c9a96e",
  },
  {
    label: 'Col header — "Co" pink, "lumn 2" default',
    initialText: "Column 2",
    defaultColor: "#c9a96e",
    segs: [
      { text: "Co", color: "#ec4899" },
      { text: "lumn 2", color: "#c9a96e" },
    ],
  },
  {
    label: "Col label — Bold+Italic stacking on some chars",
    initialText: "Hello World",
    defaultColor: "#c9a96e",
    segs: [
      { text: "He", color: "#ec4899", font_style: ["Bold"] },
      { text: "llo ", color: "#c9a96e" },
      { text: "Wor", color: "#3b82f6", font_style: ["Bold", "Italic"] },
      { text: "ld", color: "#c9a96e", font_style: ["Italic"] },
    ],
  },
  {
    label: "Row header — rotated, 'Ro' pink, 'w 2' default",
    initialText: "Row 2",
    defaultColor: "#c9a96e",
    rotated: true,
    segs: [
      { text: "Ro", color: "#ec4899" },
      { text: "w 2", color: "#c9a96e" },
    ],
  },
  {
    label: "Row label — rotated, underline + strikethrough + sup",
    initialText: "Row1 H2O",
    defaultColor: "#c9a96e",
    rotated: true,
    segs: [
      { text: "Row1 ", color: "#c9a96e", font_style: ["Underline", "Strikethrough"] },
      { text: "H", color: "#ec4899" },
      { text: "2", color: "#3b82f6", font_style: ["Subscript"] },
      { text: "O", color: "#ec4899" },
    ],
  },
];

function CaseRow({ spec }: { spec: CaseSpec }) {
  const [text, setText] = useState(spec.initialText);
  const [segs, setSegs] = useState<StyledTextSegment[] | undefined>(spec.segs);
  const [sel, setSel] = useState<{ start: number; end: number } | null>(null);

  // Clear segs on text divergence — mirrors the parent's behavior.
  const onTextChange = (newText: string) => {
    if (segs && segs.map((s) => s.text).join("") !== newText) {
      setSegs(undefined);
    }
    setText(newText);
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "280px 1fr 180px",
        gap: 12,
        alignItems: "center",
        padding: "8px 12px",
        borderBottom: "1px solid #333",
      }}
    >
      <div style={{ color: "#aaa", fontSize: 12 }}>{spec.label}</div>
      <div
        style={{
          // Simulate the panel-grid cell around the editor: small
          // bordered box with a border to indicate the "line".
          display: spec.rotated ? "block" : "flex",
          alignItems: "center",
          justifyContent: "center",
          width: spec.rotated ? 44 : 220,
          height: spec.rotated ? 180 : 28,
          minHeight: 28,
          backgroundColor: "rgba(255,255,255,0.10)",
          borderRadius: 4,
          borderBottom: spec.rotated ? "none" : "1px solid #666",
          borderRight: spec.rotated ? "1px solid #666" : "none",
          margin: spec.rotated ? "0 auto" : undefined,
          ...(spec.rotated
            ? {
                writingMode: "vertical-rl",
                transform: "rotate(180deg)",
                textOrientation: "mixed",
              }
            : {}),
        }}
      >
        <StyledTextEditor
          text={text}
          styledSegments={segs}
          defaultColor={spec.defaultColor}
          fontStyle={spec.fontStyle}
          className="text-center"
          style={{
            fontSize: 10,
            lineHeight: 1.2,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            minWidth: spec.rotated ? undefined : "100%",
            minHeight: spec.rotated ? "100%" : undefined,
            padding: spec.rotated ? "4px 8px" : "2px 4px",
            textAlign: "center" as const,
          }}
          onTextChange={onTextChange}
          onSelectionChange={setSel}
        />
      </div>
      <div style={{ color: "#888", fontSize: 11, fontFamily: "monospace" }}>
        sel={sel ? `${sel.start}..${sel.end}` : "—"}
        <br />
        text={JSON.stringify(text).slice(0, 30)}
      </div>
    </div>
  );
}

export function EditorTest() {
  return (
    <div
      style={{
        fontFamily: "-apple-system, sans-serif",
        color: "#ddd",
        padding: 16,
        minHeight: "100vh",
      }}
    >
      <h1 style={{ fontSize: 20, marginBottom: 16 }}>
        StyledTextEditor visual test
      </h1>
      <p style={{ color: "#888", fontSize: 12, marginBottom: 16 }}>
        Click into each editor, select a range, type to edit. The selection
        highlight should sit exactly on the visible glyphs. Row cases are
        rotated (vertical-rl + 180° transform) to match the panel grid.
      </p>
      <div style={{ border: "1px solid #333", borderRadius: 4 }}>
        {cases.map((c, i) => (
          <CaseRow key={i} spec={c} />
        ))}
      </div>
    </div>
  );
}
