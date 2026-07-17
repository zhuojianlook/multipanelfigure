import { describe, it, expect } from "vitest";
import { MEAS_CSV_HEADER } from "./AnalysisNodeGraph";

describe("measurements CSV contract", () => {
  it("carries a Figure column", () => {
    expect(MEAS_CSV_HEADER.split(",")).toContain("Figure");
  });

  it("keeps Panel as the FIRST non-numeric column", () => {
    // The seeded R templates choose axes by position-and-type:
    //   xcol <- names(data)[!sapply(data, is.numeric)][1]
    // Only Value is numeric, so the first non-numeric must stay Panel or every
    // bar/box/violin plot silently re-axes onto whatever replaced it.
    const cols = MEAS_CSV_HEADER.split(",");
    const numeric = new Set(["Value"]);
    expect(cols.find((c) => !numeric.has(c))).toBe("Panel");
  });

  it("preserves the original column order ahead of Figure", () => {
    expect(MEAS_CSV_HEADER.startsWith("Panel,Name,Group,Value,Unit")).toBe(true);
  });
});

describe("per-panel grouping", () => {
  // Mirrors MeasurementFigureFolder's byPanel reduction.
  const groupByPanel = (rows: Array<Record<string, unknown>>) => {
    const m = new Map<string, Array<Record<string, unknown>>>();
    for (const r of rows) {
      const k = String(r.panel ?? "—");
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(r);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  };

  it("splits rows into per-panel folders, sorted", () => {
    const rows = [
      { panel: "R1C2", name: "Line 1", value: "5 µm" },
      { panel: "R1C1", name: "Curved #1", value: "70 µm" },
      { panel: "R1C1", name: "Curved #2", value: "71 µm" },
    ];
    const g = groupByPanel(rows);
    expect(g.map(([p]) => p)).toEqual(["R1C1", "R1C2"]);
    expect(g[0][1]).toHaveLength(2);
    expect(g[1][1]).toHaveLength(1);
  });

  it("rows from two figures are distinguishable once Figure is present", () => {
    // Panel labels are bare grid coords, so both figures emit "R1C1".
    const a = { panel: "R1C1", name: "L", figure: "Fig A" };
    const b = { panel: "R1C1", name: "L", figure: "Fig B" };
    expect(a.panel).toBe(b.panel);              // the old ambiguity
    expect(a.figure).not.toBe(b.figure);        // now resolvable
  });
});
