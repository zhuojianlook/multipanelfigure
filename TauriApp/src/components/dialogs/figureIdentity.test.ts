/* Multi-MPF source identity.
 *
 * The Analysis tab can hold several figures at once, and the backend keys every
 * inset by bare grid coords ("r0c0_i0") — identical across figures. Everything
 * here exists to stop one figure's source resolving to another's. */
import { describe, it, expect } from "vitest";
import {
  figKey, rawKeyOf, nsKey, figureLabels, migrateLegacySource, measArrToCsv,
  runtimeKeyFor, sourceProvenance, measRowKey, sourceOriginLabel,
  type InsetSource,
} from "./AnalysisNodeGraph";

const doc = (id: string, name: string, path: string | null = null) => ({ id, name, path });

describe("figKey — stable figure identity", () => {
  it("identifies a saved figure by path, not by session doc id", () => {
    // openDocs is NOT persisted while workflows ARE, so doc ids are re-minted
    // every launch. Binding on the id makes saved workflows resolve to nothing
    // (and previously, to whatever figure happened to be active).
    const before = figKey(doc("doc_3", "fig1", "/exp/a/fig1.mpf"));
    const afterRestart = figKey(doc("doc_99", "fig1", "/exp/a/fig1.mpf"));
    expect(before).toBe(afterRestart);
  });

  it("separates two figures that share a basename in different folders", () => {
    // docEnsure dedupes on the exact path string, so both tabs open, both named
    // "fig1". This is the likeliest real collision.
    expect(figKey(doc("d1", "fig1", "/exp/A/fig1.mpf")))
      .not.toBe(figKey(doc("d2", "fig1", "/exp/B/fig1.mpf")));
  });

  it("falls back to the doc id for an Untitled figure", () => {
    expect(figKey(doc("doc_7", "Untitled_1", null))).toBe("d:doc_7");
  });

  it("separates two Untitled figures that share a name", () => {
    // _nextUntitledName reuses the smallest free number, so closing Untitled_1
    // and opening a new tab reproduces the name with a different id.
    expect(figKey(doc("doc_7", "Untitled_1"))).not.toBe(figKey(doc("doc_8", "Untitled_1")));
  });
});

describe("rawKeyOf / nsKey — the round trip", () => {
  it("recovers the backend key from a namespaced key", () => {
    const k = nsKey(figKey(doc("d1", "f", "/exp/a.mpf")), "r0c0_i0");
    expect(rawKeyOf({ key: k })).toBe("r0c0_i0");
  });

  it("prefers the stored rawKey over parsing", () => {
    expect(rawKeyOf({ key: "anything::at::all", rawKey: "r2c3_i1" })).toBe("r2c3_i1");
  });

  it("passes a bare legacy key through unchanged", () => {
    expect(rawKeyOf({ key: "r0c0_i0" })).toBe("r0c0_i0");
  });

  it("strips legacy doc-id namespacing, despite ids containing underscores", () => {
    expect(rawKeyOf({ key: "doc_initial__r0c0_i0" })).toBe("r0c0_i0");
    expect(rawKeyOf({ key: "doc_abc_12__r1c2_area3" })).toBe("r1c2_area3");
  });

  it("leaves an upload key alone even when the filename contains __", () => {
    // Uploads belong to no figure and are never namespaced; the legacy "__"
    // split would otherwise chop the filename down to "image.png".
    expect(rawKeyOf({ key: "img:my__image.png", name: "my__image.png" }))
      .toBe("img:my__image.png");
  });

  it("handles every backend key shape", () => {
    const fk = figKey(doc("d1", "f", "/a.mpf"));
    for (const raw of ["r0c0_i0", "r1c2_panel", "r3c4_area2"]) {
      expect(rawKeyOf({ key: nsKey(fk, raw) })).toBe(raw);
    }
  });
});

describe("cross-figure collision — the rebind regression", () => {
  it("gives two figures' same-coords insets DIFFERENT keys", () => {
    // THE BUG: the active doc kept the backend's bare key while other docs were
    // namespaced. Attach figure A's r0c0_i0 while A is active, switch to B, and
    // the refresh's `list.find(l => l.key === s.key)` matched B's r0c0_i0 — the
    // node silently rebound to B, and the run then read B's pixels.
    const a = nsKey(figKey(doc("d1", "A", "/exp/A.mpf")), "r0c0_i0");
    const b = nsKey(figKey(doc("d2", "B", "/exp/B.mpf")), "r0c0_i0");
    expect(a).not.toBe(b);

    const list = [{ key: b, rawKey: "r0c0_i0" }];
    expect(list.find((l) => l.key === a)).toBeUndefined();
  });

  it("still re-resolves a source to its OWN figure on refresh", () => {
    // The fix must not break the legitimate refresh (new thumbnail after the
    // figure changed underneath the workflow).
    const fk = figKey(doc("d1", "A", "/exp/A.mpf"));
    const attached = { key: nsKey(fk, "r0c0_i0") };
    const list = [
      { key: nsKey(figKey(doc("d2", "B", "/exp/B.mpf")), "r0c0_i0"), thumbnail: "B" },
      { key: nsKey(fk, "r0c0_i0"), thumbnail: "A-fresh" },
    ];
    expect(list.find((l) => l.key === attached.key)?.thumbnail).toBe("A-fresh");
  });
});

describe("runtimeKeyFor — the sidecar's inputs key", () => {
  // The binding key and the runtime key are DIFFERENT strings. The binding key
  // namespaces by figure and embeds a path; the runtime key must stay bare for
  // the active figure, because existing scripts say inputs["r0c0_i0"], the
  // ImageJ runner writes `<key>.png`, and _extract_source_image routes on the
  // key's suffix.
  const src = (over = {}) => ({ key: "p:/exp/A.mpf::r0c0_i0", rawKey: "r0c0_i0", ...over });

  it("keeps the bare backend key for the ACTIVE figure", () => {
    expect(runtimeKeyFor(src(), "d1", "d1")).toBe("r0c0_i0");
  });

  it("namespaces a non-active figure so two figures can't collide in inputs", () => {
    // Wire both figures' insets into one node: inputs would otherwise have a
    // single "r0c0_i0" and one figure would clobber the other.
    const active = runtimeKeyFor(src(), "d1", "d1");
    const other = runtimeKeyFor(src({ key: "p:/exp/B.mpf::r0c0_i0" }), "d2", "d1");
    expect(other).toBe("d2__r0c0_i0");
    expect(active).not.toBe(other);
  });

  it("never emits a path separator — the ImageJ runner writes <key>.png", () => {
    for (const ownerId of ["d1", "d2"]) {
      const k = runtimeKeyFor(src(), ownerId, "d1");
      expect(k).not.toContain("/");
      expect(k).not.toContain(":");
    }
  });

  it("preserves the suffix _extract_source_image routes on", () => {
    expect(runtimeKeyFor({ key: "p:/exp/A.mpf::r0c0_panel", rawKey: "r0c0_panel" }, "d1", "d1"))
      .toBe("r0c0_panel");
    expect(runtimeKeyFor({ key: "p:/exp/A.mpf::r1c2_area3", rawKey: "r1c2_area3" }, "d2", "d1"))
      .toMatch(/_area3$/);
  });

  it("passes an upload's key straight through", () => {
    expect(runtimeKeyFor({ key: "img:blot.png", name: "blot.png" }, undefined, "d1"))
      .toBe("img:blot.png");
  });
});

describe("figureLabels — disambiguating the Figure column", () => {
  it("leaves unique names untouched", () => {
    expect(figureLabels([doc("d1", "alpha"), doc("d2", "beta")]))
      .toEqual({ d1: "alpha", d2: "beta" });
  });

  it("suffixes collisions so a group-by cannot merge two figures", () => {
    const l = figureLabels([doc("d1", "fig1"), doc("d2", "fig1"), doc("d3", "fig1")]);
    expect(l).toEqual({ d1: "fig1", d2: "fig1 (2)", d3: "fig1 (3)" });
    expect(new Set(Object.values(l)).size).toBe(3);
  });

  it("keys by doc id, so identically-named tabs stay separate", () => {
    const l = figureLabels([doc("d1", "fig1"), doc("d2", "fig1")]);
    expect(l.d1).not.toBe(l.d2);
  });
});

describe("migrateLegacySource — saved workflows", () => {
  const docs = [doc("d1", "A", "/exp/A.mpf"), doc("d2", "B", "/exp/B.mpf")];
  const legacy = (over: Partial<InsetSource>) => ({
    key: "r0c0_i0", row: 0, col: 0, inset_index: 0, label: "R1C1·1",
    natural_width: 10, natural_height: 10, thumbnail: "", ...over,
  }) as InsetSource;

  it("binds by mpfId when that doc is still open", () => {
    const m = migrateLegacySource(legacy({ mpfId: "d2" }), docs, "d1");
    expect(m.figKey).toBe(figKey(docs[1]));
    expect(m.rawKey).toBe("r0c0_i0");
  });

  it("binds by name when the id is stale but the name is unambiguous", () => {
    // The id came from a previous session and no longer exists.
    const m = migrateLegacySource(legacy({ mpfId: "doc_gone", mpf: "B" }), docs, "d1");
    expect(m.figKey).toBe(figKey(docs[1]));
  });

  it("falls back to the active doc when nothing else resolves", () => {
    // What the old code did implicitly for every bare key.
    const m = migrateLegacySource(legacy({}), docs, "d2");
    expect(m.figKey).toBe(figKey(docs[1]));
  });

  it("does not guess when the name is ambiguous — uses the active doc", () => {
    const dupes = [doc("d1", "fig1", "/exp/A/fig1.mpf"), doc("d2", "fig1", "/exp/B/fig1.mpf")];
    const m = migrateLegacySource(legacy({ mpfId: "doc_gone", mpf: "fig1" }), dupes, "d2");
    expect(m.figKey).toBe(figKey(dupes[1]));
  });

  it("is idempotent — a migrated key survives a second pass", () => {
    const once = migrateLegacySource(legacy({ mpfId: "d1" }), docs, "d1");
    const twice = migrateLegacySource(once, docs, "d1");
    expect(twice.key).toBe(once.key);
    expect(twice.rawKey).toBe("r0c0_i0");
  });
});

describe("measArrToCsv — figure attribution", () => {
  const rows = [{ panel: "R1C1", name: "Membrane", numeric: 252.98, unit: "µm" }];

  it("stamps the figure label into the last column", () => {
    const line = measArrToCsv(rows, "fig1 (2)").split("\n")[1];
    expect(line.endsWith('"fig1 (2)"')).toBe(true);
  });

  it("keeps two identically-named figures apart once labelled", () => {
    const a = measArrToCsv(rows, "fig1").split("\n")[1];
    const b = measArrToCsv(rows, "fig1 (2)").split("\n")[1];
    expect(a).not.toBe(b);
  });

  it("emits a header-only table for no rows", () => {
    expect(measArrToCsv([], "fig1")).toBe(`${measArrToCsv([], "").split("\n")[0]}\n`);
  });
});

/* Figure tabs: one .mpf can hold several figures built from the same images.
 * Sources saved before figure tabs carry no figure id and must keep binding to
 * the project's first figure ("fig-1"), the only figure such a project had. */
describe("figure tabs inside one document", () => {
  const A = doc("d1", "A", "/exp/A.mpf");
  const B = doc("d2", "B", "/exp/B.mpf");

  it("figKey: the first figure keeps the legacy key; other tabs append their id", () => {
    expect(figKey(A, "fig-1")).toBe(figKey(A));
    expect(figKey(A, undefined)).toBe("p:/exp/A.mpf");
    expect(figKey(A, "fig-x")).toBe("p:/exp/A.mpf#fig-x");
    expect(figKey(A, "fig-x")).not.toBe(figKey(A, "fig-y"));
  });

  it("namespaced keys of two tabs at the same grid coords never collide", () => {
    expect(nsKey(figKey(A, "fig-1"), "r0c0_i0")).not.toBe(nsKey(figKey(A, "fig-x"), "r0c0_i0"));
    expect(rawKeyOf({ key: nsKey(figKey(A, "fig-x"), "r0c0_i0") })).toBe("r0c0_i0");
  });

  it("runtimeKeyFor: bare only for the figure on screen of the active doc", () => {
    const s = (figureId?: string) => ({ key: "k", rawKey: "r0c0_i0", figureId });
    expect(runtimeKeyFor(s(), "d1", "d1")).toBe("r0c0_i0");                     // legacy, unchanged
    expect(runtimeKeyFor(s("fig-1"), "d1", "d1", "fig-1")).toBe("r0c0_i0");
    expect(runtimeKeyFor(s("fig-x"), "d1", "d1", "fig-x")).toBe("r0c0_i0");
    expect(runtimeKeyFor(s("fig-1"), "d1", "d1", "fig-x")).toBe("fig-1__r0c0_i0");
    expect(runtimeKeyFor(s("fig-x"), "d1", "d1", "fig-1")).toBe("fig-x__r0c0_i0");
    expect(runtimeKeyFor(s(), "d2", "d1")).toBe("d2__r0c0_i0");                  // legacy, unchanged
    expect(runtimeKeyFor(s("fig-x"), "d2", "d1")).toBe("d2-fig-x__r0c0_i0");
    const all = [
      runtimeKeyFor(s("fig-1"), "d1", "d1", "fig-x"), runtimeKeyFor(s("fig-x"), "d1", "d1", "fig-x"),
      runtimeKeyFor(s("fig-1"), "d2", "d1", "fig-x"), runtimeKeyFor(s("fig-x"), "d2", "d1", "fig-x"),
    ];
    expect(new Set(all).size).toBe(all.length);
    for (const k of all) { expect(k).not.toContain("/"); expect(k).not.toContain(":"); }
    // Routing suffixes survive.
    expect(runtimeKeyFor({ key: "k", rawKey: "r0c0_panel", figureId: "fig-x" }, "d2", "d1")).toMatch(/_panel$/);
  });

  it("sourceProvenance: resolves the owner by figKey and always names the figure", () => {
    const docs = [A, B];
    const own = sourceProvenance({ figKey: figKey(A, "fig-x"), figureId: "fig-x" }, docs, "d1");
    expect(own.owner?.id).toBe("d1");
    expect(own.fields).toEqual({ figure_id: "fig-x" });
    const other = sourceProvenance({ figKey: figKey(B, "fig-x"), figureId: "fig-x" }, docs, "d1");
    expect(other.fields).toEqual({ figure_id: "fig-x", mpf_doc_id: "d2", project_path: "/exp/B.mpf" });
    // Legacy binding (no figure id) → the project's first figure.
    const legacy = sourceProvenance({ figKey: figKey(B) }, docs, "d1");
    expect(legacy.owner?.id).toBe("d2");
    expect(legacy.fields.figure_id).toBe("fig-1");
    // A figKey whose figure id doesn't match resolves to no owner (stale).
    expect(sourceProvenance({ figKey: figKey(B, "fig-x") }, docs, "d1").owner).toBeUndefined();
    // Standalone uploads carry nothing.
    expect(sourceProvenance({ name: "blot.png" }, docs, "d1").fields).toEqual({});
  });

  it("measurements: the Figure column and group keys separate tabs", () => {
    const rows = [
      { panel: "R1C1", name: "Line 1", value: "1", figure_id: "fig-1", figure_name: "Main" },
      { panel: "R1C1", name: "Line 1", value: "2", figure_id: "fig-x", figure_name: "WB" },
    ];
    const csv = measArrToCsv(rows, "A").split("\n");
    expect(csv[1]).toContain('"A › Main"');
    expect(csv[2]).toContain('"A › WB"');
    expect(measRowKey("p:/exp/A.mpf", rows[0])).toBe(measRowKey("p:/exp/A.mpf", { panel: "R1C1" }));
    expect(measRowKey("p:/exp/A.mpf", rows[0])).not.toBe(measRowKey("p:/exp/A.mpf", rows[1]));
    // Single-figure rows (no figure keys) are exactly as before.
    expect(measArrToCsv([{ panel: "R1C1", name: "L", value: "1" }], "A")).not.toContain("›");
  });
});

describe("sourceOriginLabel — which document and figure a source came from", () => {
  it("is the document name alone for a single-figure document", () => {
    expect(sourceOriginLabel("A.mpf", "Figure 1", 1)).toBe("A.mpf");
    expect(sourceOriginLabel("A.mpf", undefined, 0)).toBe("A.mpf");
  });
  it("names the figure once the document has several, even if only one has sources", () => {
    expect(sourceOriginLabel("A.mpf", "Blots", 2)).toBe("A.mpf › Blots");
  });
  it("falls back to the document name when the figure name is missing", () => {
    expect(sourceOriginLabel("A.mpf", undefined, 3)).toBe("A.mpf");
  });
});
