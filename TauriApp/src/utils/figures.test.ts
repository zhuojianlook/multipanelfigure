import { describe, it, expect } from "vitest";
import type { FigureConfig, FigureTab } from "../api/types";
import {
  DEFAULT_FIGURE_ID,
  normalizeFigures,
  serializeFigures,
  stripProjectKeys,
  switchActiveFigure,
  addFigure,
  duplicateFigure,
  deleteFigure,
  renameFigure,
  moveFigure,
  uniqueFigureName,
  imageUsage,
  stripImageFromFigures,
  type FiguresState,
} from "./figures";

/** Minimal config with one row of panels showing `images` ("" = empty). */
function cfg(images: string[], extra: Partial<FigureConfig> = {}): FigureConfig {
  return {
    rows: 1,
    cols: images.length,
    spacing: 0.02,
    output_format: "TIFF",
    background: "White",
    panels: [images.map((image_name) => ({ image_name } as unknown as FigureConfig["panels"][0][0]))],
    column_labels: [],
    row_labels: [],
    column_headers: [],
    row_headers: [],
    resolution_entries: {},
    dpi: 300,
    ...extra,
  };
}

function twoFigures(): FiguresState {
  return {
    config: cfg(["a.png", "b.png"], { resolution_entries: { scope: 0.5 } }),
    figures: [
      { id: "fig-1", name: "Figure 1", config: null },
      { id: "fig-x", name: "WB", config: cfg(["b.png", "c.png", "b.png"]) },
    ],
    activeFigureId: "fig-1",
  };
}

describe("normalizeFigures", () => {
  it("turns an old single-figure file into one default figure", () => {
    for (const raw of [undefined, null, [], "junk", 42]) {
      expect(normalizeFigures(raw, undefined)).toEqual({
        figures: [{ id: DEFAULT_FIGURE_ID, name: "Figure 1", config: null }],
        activeFigureId: DEFAULT_FIGURE_ID,
      });
    }
  });

  it("keeps a well-formed list, its order and unknown keys", () => {
    const raw = [
      { id: "fig-1", name: "Main", future: { x: 1 }, config: cfg(["a.png"]) },
      { id: "fig-x", name: "WB" },
    ];
    const out = normalizeFigures(raw, "fig-x");
    expect(out.activeFigureId).toBe("fig-x");
    expect(out.figures.map((f) => f.id)).toEqual(["fig-1", "fig-x"]);
    expect(out.figures[0].future).toEqual({ x: 1 });
    expect(out.figures[0].config?.panels[0][0].image_name).toBe("a.png");
    expect(out.figures[1].config).toBeNull();
  });

  it("repairs duplicate / unsafe ids and missing names", () => {
    const raw = [
      { id: "dup", name: "", config: cfg(["a.png"]) },
      { id: "dup", config: cfg(["b.png"]) },
      { id: "bad id_with#chars", name: "X" },
      "not-an-object",
    ];
    const out = normalizeFigures(raw, "nope");
    const ids = out.figures.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) expect(id).toMatch(/^[A-Za-z0-9-]+$/);
    expect(out.figures[0].name).toBe("Figure 1");
    expect(out.figures[1].name).toBe("Figure 2");
    // No id matched → the config-less entry is the active (top-level) figure.
    expect(out.figures.find((f) => f.id === out.activeFigureId)?.name).toBe("X");
  });

  it("drops parked entries that have no usable config", () => {
    const raw = [
      { id: "fig-1", name: "A" },
      { id: "fig-2", name: "B", config: { rows: "2" } },
      { id: "fig-3", name: "C", config: cfg(["c.png"]) },
    ];
    const out = normalizeFigures(raw, "fig-1");
    expect(out.figures.map((f) => f.id)).toEqual(["fig-1", "fig-3"]);
  });

  it("adds an entry for the top level when every entry carries a config", () => {
    const out = normalizeFigures([{ id: "fig-1", name: "Figure 1", config: cfg(["a.png"]) }], "missing");
    expect(out.figures).toHaveLength(2);
    expect(out.figures[0].config).toBeNull();
    expect(out.figures[0].id).toBe(out.activeFigureId);
    expect(out.activeFigureId).not.toBe("fig-1");
  });

  it("forces the active entry's config to null", () => {
    const out = normalizeFigures([{ id: "fig-1", name: "A", config: cfg(["a.png"]) }], "fig-1");
    expect(out.figures[0].config).toBeNull();
  });

  it("serialize ∘ normalize is stable", () => {
    const s = twoFigures();
    const wire = serializeFigures(s.figures, s.activeFigureId);
    const back = normalizeFigures(JSON.parse(JSON.stringify(wire)), s.activeFigureId);
    expect(back.activeFigureId).toBe("fig-1");
    expect(back.figures).toEqual(JSON.parse(JSON.stringify(s.figures)));
    expect(serializeFigures(back.figures, back.activeFigureId)).toEqual(wire);
  });
});

describe("serializeFigures / stripProjectKeys", () => {
  it("omits the active config and strips project keys from parked ones", () => {
    const parked = cfg(["z.png"], { image_groups: [], figures: [1], active_figure_id: "q" });
    const wire = serializeFigures(
      [{ id: "fig-1", name: "A", config: null }, { id: "fig-2", name: "B", config: parked }],
      "fig-1",
    );
    expect(wire[0]).toEqual({ id: "fig-1", name: "A" });
    const c = wire[1].config as Record<string, unknown>;
    expect(c).not.toHaveProperty("image_groups");
    expect(c).not.toHaveProperty("figures");
    expect(c).not.toHaveProperty("active_figure_id");
    expect(stripProjectKeys(parked)).not.toBe(parked);
  });
});

describe("switchActiveFigure", () => {
  it("parks the outgoing config and activates the target", () => {
    const s = twoFigures();
    const out = switchActiveFigure(s, "fig-x");
    expect(out.activeFigureId).toBe("fig-x");
    expect(out.config.panels[0].map((p) => p.image_name)).toEqual(["b.png", "c.png", "b.png"]);
    expect(out.figures[0].config?.panels[0].map((p) => p.image_name)).toEqual(["a.png", "b.png"]);
    expect(out.figures[1].config).toBeNull();
  });

  it("carries the project-shared keys to the incoming figure", () => {
    const out = switchActiveFigure(twoFigures(), "fig-x");
    expect(out.config.resolution_entries).toEqual({ scope: 0.5 });
  });

  it("round-trips A → B → A to the original config", () => {
    const s = twoFigures();
    const back = switchActiveFigure(switchActiveFigure(s, "fig-x"), "fig-1");
    expect(back.config).toEqual(s.config);
    expect(back.activeFigureId).toBe("fig-1");
    expect(back.figures[0].config).toBeNull();
    // The parked figure keeps its own layout but now carries the project's
    // shared keys, which followed the user while it was active.
    expect(back.figures[1].config).toEqual({ ...s.figures[1].config, resolution_entries: { scope: 0.5 } });
  });

  it("returns the input for the active or an unknown id", () => {
    const s = twoFigures();
    expect(switchActiveFigure(s, "fig-1")).toBe(s);
    expect(switchActiveFigure(s, "nope")).toBe(s);
  });

  it("never writes to its (frozen) input", () => {
    const s = twoFigures();
    const frozen = JSON.parse(JSON.stringify(s));
    Object.freeze(frozen.config); Object.freeze(frozen.figures);
    expect(() => switchActiveFigure(frozen, "fig-x")).not.toThrow();
    expect(frozen).toEqual(JSON.parse(JSON.stringify(s)));
  });
});

describe("add / duplicate / delete / rename / move", () => {
  it("adds a named figure and activates it", () => {
    const out = addFigure(twoFigures(), { config: cfg(["", ""]) });
    expect(out.figures).toHaveLength(3);
    expect(out.figures[2].name).toBe("Figure 3");
    expect(out.activeFigureId).toBe(out.figures[2].id);
    expect(out.config.panels[0].map((p) => p.image_name)).toEqual(["", ""]);
    expect(out.config.resolution_entries).toEqual({ scope: 0.5 });
  });

  it("duplicates the active figure as an independent copy next to it", () => {
    const s = twoFigures();
    const out = duplicateFigure(s, "fig-1");
    expect(out.figures.map((f) => f.name)).toEqual(["Figure 1", "Figure 1 copy", "WB"]);
    expect(out.activeFigureId).toBe(out.figures[1].id);
    expect(out.config.panels).toEqual(s.config.panels);
    expect(out.config.panels).not.toBe(s.config.panels);
    expect(duplicateFigure(out, "fig-1").figures.map((f) => f.name)).toContain("Figure 1 copy 2");
  });

  it("duplicates a parked figure", () => {
    const out = duplicateFigure(twoFigures(), "fig-x");
    expect(out.figures.map((f) => f.name)).toEqual(["Figure 1", "WB", "WB copy"]);
    expect(out.config.panels[0].map((p) => p.image_name)).toEqual(["b.png", "c.png", "b.png"]);
  });

  it("refuses to delete the last figure", () => {
    const one: FiguresState = { config: cfg(["a.png"]), figures: [{ id: "fig-1", name: "F", config: null }], activeFigureId: "fig-1" };
    expect(deleteFigure(one, "fig-1")).toBe(one);
  });

  it("deleting the active figure activates its neighbour", () => {
    const out = deleteFigure(twoFigures(), "fig-1");
    expect(out.figures.map((f) => f.id)).toEqual(["fig-x"]);
    expect(out.activeFigureId).toBe("fig-x");
    expect(out.config.panels[0][1].image_name).toBe("c.png");
    expect(out.figures[0].config).toBeNull();
  });

  it("deleting a parked figure keeps the active config", () => {
    const s = twoFigures();
    const out = deleteFigure(s, "fig-x");
    expect(out.config).toBe(s.config);
    expect(out.figures.map((f) => f.id)).toEqual(["fig-1"]);
  });

  it("renames and moves, returning the same array on a no-op", () => {
    const { figures } = twoFigures();
    expect(renameFigure(figures, "fig-x", "  Blots ")[1].name).toBe("Blots");
    expect(renameFigure(figures, "fig-x", "   ")).toBe(figures);
    expect(renameFigure(figures, "fig-x", "WB")).toBe(figures);
    expect(moveFigure(figures, "fig-x", -1).map((f) => f.id)).toEqual(["fig-x", "fig-1"]);
    expect(moveFigure(figures, "fig-x", 1)).toBe(figures);
  });

  it("names new figures past the existing ones", () => {
    const figs: FigureTab[] = [
      { id: "a", name: "Figure 1", config: null },
      { id: "b", name: "Figure 3", config: null },
    ];
    expect(uniqueFigureName(figs)).toBe("Figure 4");
  });
});

describe("imageUsage / stripImageFromFigures", () => {
  it("counts the active figure and lists the other figures", () => {
    const s = twoFigures();
    const u = imageUsage(s.figures, s.activeFigureId, s.config);
    expect(u.get("a.png")).toMatchObject({ inCurrent: 1, others: [] });
    expect(u.get("b.png")).toMatchObject({ inCurrent: 1, others: [{ id: "fig-x", name: "WB", count: 2 }] });
    expect(u.get("c.png")).toMatchObject({ inCurrent: 0, others: [{ id: "fig-x", name: "WB", count: 1 }] });
    expect(u.has("")).toBe(false);
  });

  it("lists every figure using an image in tab order, numbered by tab position", () => {
    const s = twoFigures();
    const u = imageUsage(s.figures, s.activeFigureId, s.config);
    const marks = (n: string) => u.get(n)?.byFigure.map((f) => [f.index, f.current, f.count]);
    const activeIdx = s.figures.findIndex((f) => f.id === s.activeFigureId) + 1;
    const otherIdx = s.figures.findIndex((f) => f.id === "fig-x") + 1;
    expect(marks("a.png")).toEqual([[activeIdx, true, 1]]);
    expect(marks("b.png")).toEqual(
      [[activeIdx, true, 1], [otherIdx, false, 2]].sort((p, q) => (p[0] as number) - (q[0] as number)),
    );
    expect(marks("c.png")).toEqual([[otherIdx, false, 1]]);
  });

  it("blanks an image in parked figures only", () => {
    const s = twoFigures();
    const out = stripImageFromFigures(s.figures, "b.png");
    expect(out[0]).toBe(s.figures[0]);
    expect(out[1].config?.panels[0].map((p) => p.image_name)).toEqual(["", "c.png", ""]);
    expect(stripImageFromFigures(s.figures, "zzz.png")).toBe(s.figures);
  });
});
