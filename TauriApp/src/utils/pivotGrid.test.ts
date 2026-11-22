/* The planner's Pivot (utils/pivotGrid): an R×C figure becomes C×R, and
   pivoting twice gives back the same figure — there is no undo stack, so
   the second press IS the undo. The fixture is shared with the sidecar's
   contract test (python-sidecar/test_pivot_config.py): `after` was generated
   once from `before` with pivotFigureConfig, reviewed, and committed. */

import { describe, it, expect } from "vitest";
import type { FigureConfig, PanelInfo } from "../api/types";
import fixture from "./__fixtures__pivotGrid.json";
import {
  pivotAnalysisKey, pivotAxisRotation, pivotCellRecord, pivotFigureConfig, pivotRawKey, pivotSide,
} from "./pivotGrid";

const before = fixture.before as unknown as FigureConfig;
const after = fixture.after as unknown as FigureConfig;
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

/** A bare R×C config whose panels are named by their cell. */
function grid(rows: number, cols: number): FigureConfig {
  return {
    rows, cols, spacing: 0.02, output_format: "TIFF", background: "White",
    panels: Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) => (
      { image_name: `r${r}c${c}.png`, labels: [], zoom_insets: [], zoom_inset: null, add_zoom_inset: false } as unknown as PanelInfo
    ))),
    column_labels: Array.from({ length: cols }, (_, i) => (
      { text: `Column ${i + 1}`, position: "Top", rotation: 0 } as FigureConfig["column_labels"][number])),
    row_labels: Array.from({ length: rows }, (_, i) => (
      { text: `Row ${i + 1}`, position: "Left", rotation: 90 } as FigureConfig["row_labels"][number])),
    column_headers: [], row_headers: [], resolution_entries: {}, dpi: 300,
  };
}

describe("pivotFigureConfig — the shared fixture", () => {
  it("before → after, and after → before (pivoting twice is exact)", () => {
    expect(pivotFigureConfig(before)).toEqual(after);
    expect(pivotFigureConfig(after)).toEqual(before);
  });

  it("does not touch its input", () => {
    const copy = clone(before);
    pivotFigureConfig(before);
    expect(before).toEqual(copy);
  });

  it("swaps the grid size and moves panel (r, c) to (c, r)", () => {
    const out = pivotFigureConfig(before);
    expect([out.rows, out.cols]).toEqual([3, 2]);
    expect(out.panels).toHaveLength(3);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        expect(out.panels[c][r].image_name).toBe(`r${r}c${c}.png`);
        // Its auto letter travels with it (the store re-letters afterwards).
        expect(out.panels[c][r].labels[0].text).toBe(before.panels[r][c].labels[0].text);
      }
    }
  });

  it("column labels become row labels and the other way round, sides and rotations swapped", () => {
    const out = pivotFigureConfig(before);
    expect(out.row_labels.map((l) => [l.text, l.position, l.rotation])).toEqual([
      ["C1", "Left", 90], ["C2", "Left", 90], ["C3", "Right", 0],
    ]);
    expect(out.column_labels.map((l) => [l.text, l.position, l.rotation])).toEqual([
      ["R1", "Top", 0], ["R2", "Bottom", 90],
    ]);
    // Text, font and distance are the label's own.
    expect(out.row_labels[2].font_size).toBe(before.column_labels[2].font_size);
    expect(out.row_labels[2].distance).toBe(before.column_labels[2].distance);
  });

  it("headers swap axes keeping level order (0 = outermost) and every span", () => {
    const out = pivotFigureConfig(before);
    expect(out.row_headers.map((lv) => lv.headers.map((g) => [g.text, g.columns_or_rows, g.position, g.rotation]))).toEqual([
      [["All", [0, 1, 2], "Left", 90]],
      [["AB", [0, 1], "Left", 90], ["C", [2], "Right", 90]],
    ]);
    expect(out.column_headers.map((lv) => lv.headers.map((g) => [g.text, g.columns_or_rows, g.position, g.rotation]))).toEqual([
      [["Rows", [0, 1], "Top", 0]],
    ]);
  });

  it("the show-labels toggles swap", () => {
    const out = pivotFigureConfig(before);
    expect(out.show_column_labels).toBe(false);
    expect(out.show_row_labels).toBe(true);
    // Absent (= shown) stays absent: a figure that never set them doesn't gain them.
    const bare = pivotFigureConfig(grid(2, 2));
    expect("show_column_labels" in bare).toBe(false);
    expect("show_row_labels" in bare).toBe(false);
  });

  it("an Adjacent Panel inset still draws into the panel it drew into", () => {
    const out = pivotFigureConfig(before);
    // (0,1) drew Right into (0,2); now at (1,0) it draws Bottom into (2,0) —
    // where (0,2) went.
    const moved = out.panels[1][0];
    expect(moved.image_name).toBe("r0c1.png");
    const zi = moved.zoom_insets[0] as unknown as Record<string, unknown>;
    expect(zi.side).toBe("Bottom");
    expect([zi.adjacent_panel_row, zi.adjacent_panel_col]).toEqual([2, 0]);
    expect(out.panels[2][0].image_name).toBe("r0c2.png");
    // The legacy single inset of (1,0) drew Top into (0,0); at (0,1) it draws Left into (0,0).
    expect(out.panels[0][1].image_name).toBe("r1c0.png");
    expect(out.panels[0][1].zoom_inset?.side).toBe("Left");
    // A Standard Zoom inset draws inside its own panel: untouched.
    expect(out.panels[1][1].image_name).toBe("r1c1.png");
    expect(out.panels[2][1].zoom_insets[0].side).toBe("Left");
    // The target's (cosmetic) source coordinates point at the source's new cell.
    const tgt = out.panels[2][0] as unknown as Record<string, unknown>;
    expect([tgt.zoom_source_row, tgt.zoom_source_col]).toEqual([1, 0]);
  });

  it("an Adjacent inset with no side (= Right) comes back as an explicit Right", () => {
    const cfg = grid(1, 2);
    cfg.panels[0][0].add_zoom_inset = true;
    cfg.panels[0][0].zoom_insets = [{ inset_type: "Adjacent Panel", side: "" } as PanelInfo["zoom_insets"][number]];
    const once = pivotFigureConfig(cfg);
    expect(once.panels[0][0].zoom_insets[0].side).toBe("Bottom");
    expect(pivotFigureConfig(once).panels[0][0].zoom_insets[0].side).toBe("Right");
  });
});

describe("pivotFigureConfig — other shapes", () => {
  it("1×1: only labels and headers swap", () => {
    const cfg = grid(1, 1);
    const out = pivotFigureConfig(cfg);
    expect([out.rows, out.cols]).toEqual([1, 1]);
    expect(out.panels[0][0].image_name).toBe("r0c0.png");
    expect(out.column_labels.map((l) => [l.text, l.position, l.rotation])).toEqual([["Row 1", "Top", 0]]);
    expect(out.row_labels.map((l) => [l.text, l.position, l.rotation])).toEqual([["Column 1", "Left", 90]]);
    expect(pivotFigureConfig(out)).toEqual(cfg);
  });

  it("3×3: the diagonal stays, the rest mirror across it", () => {
    const cfg = grid(3, 3);
    const out = pivotFigureConfig(cfg);
    expect(out.panels.map((row) => row.map((p) => p.image_name))).toEqual([
      ["r0c0.png", "r1c0.png", "r2c0.png"],
      ["r0c1.png", "r1c1.png", "r2c1.png"],
      ["r0c2.png", "r1c2.png", "r2c2.png"],
    ]);
    expect(pivotFigureConfig(out)).toEqual(cfg);
  });

  it("4×3 becomes 3×4 with nothing lost", () => {
    const out = pivotFigureConfig(grid(4, 3));
    expect([out.rows, out.cols]).toEqual([3, 4]);
    expect(out.panels.map((row) => row.length)).toEqual([4, 4, 4]);
    expect(out.column_labels).toHaveLength(4);
    expect(out.row_labels).toHaveLength(3);
    expect(out.panels[2][3].image_name).toBe("r3c2.png");
  });

  it("a missing position or rotation counts as the default (Top, 0)", () => {
    const cfg = grid(1, 1);
    const lbl = cfg.column_labels[0] as unknown as Record<string, unknown>;
    delete lbl.position;
    delete lbl.rotation;
    const out = pivotFigureConfig(cfg);
    expect([out.row_labels[0].position, out.row_labels[0].rotation]).toEqual(["Left", 90]);
    expect([pivotFigureConfig(out).column_labels[0].position, pivotFigureConfig(out).column_labels[0].rotation]).toEqual(["Top", 0]);
  });
});

describe("helpers", () => {
  it("pivotSide swaps Top↔Left and Bottom↔Right, anything else as it is", () => {
    expect(["Top", "Left", "Bottom", "Right"].map(pivotSide)).toEqual(["Left", "Top", "Right", "Bottom"]);
    expect(pivotSide("Center")).toBe("Center");
    expect(pivotSide(undefined)).toBe("");
  });

  it("pivotAxisRotation swaps exactly 0↔90 and keeps other angles", () => {
    expect(pivotAxisRotation(0)).toBe(90);
    expect(pivotAxisRotation(90)).toBe(0);
    expect(pivotAxisRotation(undefined)).toBe(90);
    expect(pivotAxisRotation(45)).toBe(45);
    expect(pivotAxisRotation(270)).toBe(270);
  });

  it("pivotCellRecord moves every 'r-c' key to 'c-r'", () => {
    expect(pivotCellRecord({ "0-1": "A", "1-0": "B", "2-2": "C", other: "D" })).toEqual({
      "1-0": "A", "0-1": "B", "2-2": "C", other: "D",
    });
  });

  it("pivotRawKey moves raw source keys only", () => {
    expect(pivotRawKey("r0c1_i0")).toBe("r1c0_i0");
    expect(pivotRawKey("r12c3_panel")).toBe("r3c12_panel");
    expect(pivotRawKey("r2c0_area4")).toBe("r0c2_area4");
    expect(pivotRawKey("r1c1_i2")).toBe("r1c1_i2");
    expect(pivotRawKey("img:r0c1_i0")).toBe("img:r0c1_i0");
    expect(pivotRawKey("r0c1_i0x")).toBe("r0c1_i0x");
  });
});

describe("pivotAnalysisKey", () => {
  const fk = "p:/exp/a.mpf";

  it("source keys of the figure swap with their twin — never both onto one", () => {
    expect(pivotAnalysisKey(`${fk}::r0c1_i0`, fk)).toBe(`${fk}::r1c0_i0`);
    expect(pivotAnalysisKey(`${fk}::r1c0_i0`, fk)).toBe(`${fk}::r0c1_i0`);
    expect(pivotAnalysisKey(`${fk}::r0c2_panel`, fk)).toBe(`${fk}::r2c0_panel`);
    expect(pivotAnalysisKey(`${fk}::r1c0_area2`, fk)).toBe(`${fk}::r0c1_area2`);
  });

  it("a diagonal panel's keys stay", () => {
    expect(pivotAnalysisKey(`${fk}::r1c1_i0`, fk)).toBe(`${fk}::r1c1_i0`);
    expect(pivotAnalysisKey(`${fk}|R2C2|0|1`, fk)).toBe(`${fk}|R2C2|0|1`);
  });

  it("measurement rows move to the transposed cell", () => {
    expect(pivotAnalysisKey(`${fk}|R1C2|0|3`, fk)).toBe(`${fk}|R2C1|0|3`);
    expect(pivotAnalysisKey(`${fk}|R10C3||`, fk)).toBe(`${fk}|R3C10||`);
    // Not exactly the three fields of a measurement row: left alone.
    expect(pivotAnalysisKey(`${fk}|R1C2|0`, fk)).toBe(`${fk}|R1C2|0`);
    expect(pivotAnalysisKey(`${fk}|x|R1C2|0|3`, fk)).toBe(`${fk}|x|R1C2|0|3`);
  });

  it("another figure of the document, another document and other keys are untouched", () => {
    expect(pivotAnalysisKey(`${fk}#fig-2::r0c1_i0`, fk)).toBe(`${fk}#fig-2::r0c1_i0`);
    expect(pivotAnalysisKey(`${fk}#fig-2|R1C2|0|0`, fk)).toBe(`${fk}#fig-2|R1C2|0|0`);
    expect(pivotAnalysisKey("p:/exp/b.mpf::r0c1_i0", fk)).toBe("p:/exp/b.mpf::r0c1_i0");
    // A path that merely starts like this one's.
    expect(pivotAnalysisKey("p:/exp/a.mpf.bak::r0c1_i0", fk)).toBe("p:/exp/a.mpf.bak::r0c1_i0");
    expect(pivotAnalysisKey(`${fk}::x::r0c1_i0`, fk)).toBe(`${fk}::x::r0c1_i0`);
    expect(pivotAnalysisKey("img:r0c1.png", fk)).toBe("img:r0c1.png");
    expect(pivotAnalysisKey("r0c1_i0", fk)).toBe("r0c1_i0");
    // A figure tab's own key base.
    const fk2 = `${fk}#fig-2`;
    expect(pivotAnalysisKey(`${fk2}::r0c1_i0`, fk2)).toBe(`${fk2}::r1c0_i0`);
    expect(pivotAnalysisKey(`${fk}::r0c1_i0`, fk2)).toBe(`${fk}::r0c1_i0`);
  });
});
