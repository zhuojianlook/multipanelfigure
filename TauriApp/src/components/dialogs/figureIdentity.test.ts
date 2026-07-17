/* Multi-MPF source identity.
 *
 * The Analysis tab can hold several figures at once, and the backend keys every
 * inset by bare grid coords ("r0c0_i0") — identical across figures. Everything
 * here exists to stop one figure's source resolving to another's. */
import { describe, it, expect } from "vitest";
import {
  figKey, rawKeyOf, nsKey, figureLabels, migrateLegacySource, measArrToCsv,
  runtimeKeyFor,
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
