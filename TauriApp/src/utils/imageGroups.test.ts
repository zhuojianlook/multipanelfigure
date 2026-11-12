import { describe, it, expect } from "vitest";
import type { ImageGroup } from "../api/types";
import {
  normalizeImageGroups,
  sectionsOf,
  stripImageFromGroup,
  stripImageFromGroups,
  moveImageToSection,
  addDividerToGroup,
  renameDividerInGroup,
  deleteDividerFromGroup,
  nextSubgroupName,
} from "./imageGroups";

/* The store feeds these helpers its COMMITTED state, which immer has
   deep-frozen — so every mutation test runs on frozen input: a helper that
   writes to its argument throws here instead of corrupting state in-app. */
function deepFreeze<T>(v: T): T {
  if (v && typeof v === "object") {
    Object.values(v as object).forEach(deepFreeze);
    Object.freeze(v);
  }
  return v;
}

/** The data-model invariants every helper must preserve. */
function expectInvariants(groups: ImageGroup[]) {
  for (const g of groups) {
    expect(new Set(g.imageNames).size).toBe(g.imageNames.length);
    if ("dividers" in g) expect(g.dividers!.length).toBeGreaterThan(0);
    const member = new Set(g.imageNames);
    const claimed = new Set<string>();
    for (const d of g.dividers ?? []) {
      for (const n of d.imageNames) {
        expect(member.has(n)).toBe(true);        // divider ⊆ flat membership
        expect(claimed.has(n)).toBe(false);      // at most one divider
        claimed.add(n);
      }
    }
    // The flat list is kept in display order (what an older build shows).
    expect(g.imageNames).toEqual(sectionsOf(g).flatMap((s) => s.imageNames));
  }
}

/** Shorthand: what each section of a group shows, as [name|null, images]. */
const view = (g: ImageGroup) => sectionsOf(g).map((s) => [s.name, s.imageNames]);

/* ── normalizeImageGroups ─────────────────────────────── */

describe("normalizeImageGroups — old-shape input (pre-divider .mpf)", () => {
  // The exact inline map every load path used before dividers existed.
  const oldInlineMap = (raw: ImageGroup[], valid: Set<string>) =>
    raw.map((g) => ({ ...g, imageNames: (g.imageNames ?? []).filter((n) => valid.has(n)) }));

  it("is deep-equal to the old inline map, unknown keys included", () => {
    const raw = [
      { id: "g1", name: "Controls", imageNames: ["a.png", "gone.png", "b.png"] },
      { id: "g2", name: "Treated", imageNames: [], color: "#f00" },
      { id: "g3", name: "Late", imageNames: ["c.png"] },
    ] as ImageGroup[];
    const valid = new Set(["a.png", "b.png", "c.png"]);
    const out = normalizeImageGroups(raw, valid);
    expect(out).toEqual(oldInlineMap(raw, valid));
    expect(out.every((g) => !("dividers" in g))).toBe(true);
  });

  it("never adds a dividers key to groups that have none", () => {
    const out = normalizeImageGroups([{ id: "g", name: "G", imageNames: ["x"] }]);
    expect(Object.keys(out[0]).sort()).toEqual(["id", "imageNames", "name"]);
  });

  it("without validNames keeps every (string) name", () => {
    expect(normalizeImageGroups([{ id: "g", name: "G", imageNames: ["x", "y"] }])[0].imageNames)
      .toEqual(["x", "y"]);
  });

  it("treats absent image_groups as no groups", () => {
    expect(normalizeImageGroups(undefined)).toEqual([]);
  });
});

describe("normalizeImageGroups — malformed input never throws", () => {
  it("non-array input → []", () => {
    for (const raw of [null, undefined, {}, "groups", 42, true]) {
      expect(normalizeImageGroups(raw)).toEqual([]);
    }
  });

  it("skips null / primitive / array entries", () => {
    const out = normalizeImageGroups([null, 3, "x", ["a"], { id: "g", name: "G", imageNames: [] }]);
    expect(out.map((g) => g.id)).toEqual(["g"]);
  });

  it("fills missing ids/names and makes duplicate group ids unique", () => {
    const out = normalizeImageGroups([
      { imageNames: ["a"] },
      { id: "", name: 5, imageNames: [] },
      { id: "dup", name: "One", imageNames: [] },
      { id: "dup", name: "Two", imageNames: [] },
    ]);
    expect(out.map((g) => g.id)).toEqual(["group-1", "group-2", "dup", "dup-2"]);
    expect(out[0].name).toBe("Group 1");
    expect(out[1].name).toBe("Group 2");
  });

  it("drops non-string names, de-dupes within a group, handles non-array imageNames", () => {
    const out = normalizeImageGroups([
      { id: "a", name: "A", imageNames: ["x", 7, null, "x", "y", { n: 1 }] },
      { id: "b", name: "B", imageNames: "x" },
      { id: "c", name: "C" },
    ]);
    expect(out[0].imageNames).toEqual(["x", "y"]);
    expect(out[1].imageNames).toEqual([]);
    expect(out[2].imageNames).toEqual([]);
  });

  it("dividers that aren't an array (or end up empty) are removed", () => {
    const out = normalizeImageGroups([
      { id: "a", name: "A", imageNames: ["x"], dividers: "nope" },
      { id: "b", name: "B", imageNames: ["x"], dividers: null },
      { id: "c", name: "C", imageNames: ["x"], dividers: [] },
      { id: "d", name: "D", imageNames: ["x"], dividers: [null, 4, "s"] },
    ]);
    for (const g of out) expect("dividers" in g).toBe(false);
  });

  it("divider images must be group members; first divider claiming an image wins", () => {
    const [g] = normalizeImageGroups([{
      id: "g", name: "G", imageNames: ["a", "b", "c", "d"],
      dividers: [
        { id: "d1", name: "One", imageNames: ["b", "stray", "c", "b", 9] },
        { id: "d2", name: "Two", imageNames: ["c", "d"] },
      ],
    }]);
    expect(g.dividers!.map((d) => d.imageNames)).toEqual([["b", "c"], ["d"]]);
    expect(g.imageNames).toEqual(["a", "b", "c", "d"]);   // flat membership untouched
    expectInvariants([g]);
  });

  it("drops names not in validNames from groups AND dividers", () => {
    const [g] = normalizeImageGroups([{
      id: "g", name: "G", imageNames: ["a", "gone", "b"],
      dividers: [{ id: "d1", name: "One", imageNames: ["gone", "b"] }],
    }], new Set(["a", "b"]));
    expect(g.imageNames).toEqual(["a", "b"]);
    expect(g.dividers![0].imageNames).toEqual(["b"]);
  });

  it("generates deterministic, unique ids/names for dividers missing them", () => {
    const raw = [{
      id: "g", name: "G", imageNames: [],
      dividers: [{ imageNames: [] }, { id: "x" }, { id: "x", name: "Dup" }],
    }];
    const a = normalizeImageGroups(raw);
    const b = normalizeImageGroups(raw);
    expect(a).toEqual(b);
    expect(a[0].dividers!.map((d) => d.id)).toEqual(["g-d1", "x", "x-2"]);
    expect(a[0].dividers!.map((d) => d.name)).toEqual(["Subgroup 1", "Subgroup 2", "Dup"]);
    expect(a[0].dividers!.every((d) => Array.isArray(d.imageNames))).toBe(true);
  });

  it("is idempotent", () => {
    const raw = [{ imageNames: ["a", "a", 1], dividers: [{ imageNames: ["a"] }, "junk"] }, null];
    const once = normalizeImageGroups(raw);
    expect(normalizeImageGroups(once)).toEqual(once);
  });
});

describe("normalizeImageGroups — new-shape input", () => {
  it("round-trips dividers and preserves unknown fields on groups and dividers", () => {
    const raw = [{
      id: "g", name: "G", imageNames: ["a", "b", "c"], futureGroupKey: { x: 1 },
      dividers: [{ id: "d1", name: "Day 3", imageNames: ["b", "c"], collapsed: true }],
    }];
    const out = normalizeImageGroups(raw, new Set(["a", "b", "c"]));
    expect(out).toEqual(raw);
    expect(view(out[0])).toEqual([[null, ["a"]], ["Day 3", ["b", "c"]]]);
  });
});

/* ── sectionsOf ───────────────────────────────────────── */

describe("sectionsOf", () => {
  it("old-shape group is a single unsectioned section", () => {
    expect(sectionsOf({ id: "g", name: "G", imageNames: ["a", "b"] }))
      .toEqual([{ dividerId: null, name: null, imageNames: ["a", "b"] }]);
  });

  it("unsectioned first (flat order minus divider images), then dividers in array order", () => {
    const g: ImageGroup = {
      id: "g", name: "G", imageNames: ["d", "a", "c", "b", "e"],
      dividers: [
        { id: "d2", name: "Second", imageNames: ["e", "c"] },
        { id: "d1", name: "First", imageNames: ["b"] },
      ],
    };
    expect(sectionsOf(g)).toEqual([
      { dividerId: null, name: null, imageNames: ["d", "a"] },
      { dividerId: "d2", name: "Second", imageNames: ["e", "c"] },
      { dividerId: "d1", name: "First", imageNames: ["b"] },
    ]);
  });

  it("always includes the (possibly empty) unsectioned area and empty dividers", () => {
    const g: ImageGroup = {
      id: "g", name: "G", imageNames: ["a"],
      dividers: [{ id: "d1", name: "One", imageNames: ["a"] }, { id: "d2", name: "Two", imageNames: [] }],
    };
    expect(view(g)).toEqual([[null, []], ["One", ["a"]], ["Two", []]]);
  });
});

/* ── Fixtures for mutations ───────────────────────────── */

/** g1: unsectioned [a,b] | Day 1 [c,d,e] | Day 2 [f];  g2: [x] | Late [y] */
const fixture = (): ImageGroup[] => deepFreeze([
  {
    id: "g1", name: "Group 1", imageNames: ["a", "b", "c", "d", "e", "f"],
    dividers: [
      { id: "d1", name: "Day 1", imageNames: ["c", "d", "e"] },
      { id: "d2", name: "Day 2", imageNames: ["f"] },
    ],
  },
  {
    id: "g2", name: "Group 2", imageNames: ["x", "y"],
    dividers: [{ id: "dl", name: "Late", imageNames: ["y"] }],
  },
  { id: "g3", name: "Plain", imageNames: ["p", "q"] },
]);

/* ── stripImageFromGroup(s) ───────────────────────────── */

describe("stripImageFromGroup(s)", () => {
  it("removes the image from the flat list AND its divider", () => {
    const out = stripImageFromGroups(fixture(), "d");
    expect(view(out[0])).toEqual([[null, ["a", "b"]], ["Day 1", ["c", "e"]], ["Day 2", ["f"]]]);
    expect(out[0].imageNames).not.toContain("d");
    expectInvariants(out);
  });

  it("leaves untouched groups (and the array) as the same references", () => {
    const groups = fixture();
    const out = stripImageFromGroups(groups, "d");
    expect(out[1]).toBe(groups[1]);
    expect(out[2]).toBe(groups[2]);
    expect(stripImageFromGroups(groups, "nope")).toBe(groups);
    expect(stripImageFromGroup(groups[2], "a")).toBe(groups[2]);
  });

  it("keeps an emptied divider (the user placed it) and never adds dividers to plain groups", () => {
    const out = stripImageFromGroups(fixture(), "f");
    expect(view(out[0])).toEqual([[null, ["a", "b"]], ["Day 1", ["c", "d", "e"]], ["Day 2", []]]);
    const plain = stripImageFromGroups(fixture(), "p")[2];
    expect(plain).toEqual({ id: "g3", name: "Plain", imageNames: ["q"] });
    expect("dividers" in plain).toBe(false);
  });
});

/* ── moveImageToSection ───────────────────────────────── */

describe("moveImageToSection", () => {
  it("2-arg form (old moveImageToGroup): appends to the target's unsectioned area", () => {
    const out = moveImageToSection(fixture(), "p", "g1");
    expect(view(out[0])[0]).toEqual([null, ["a", "b", "p"]]);
    expect(out[2].imageNames).toEqual(["q"]);
    expectInvariants(out);
  });

  it("old-shape groups stay old-shape across a move", () => {
    const groups = deepFreeze<ImageGroup[]>([
      { id: "A", name: "A", imageNames: ["a", "b"] },
      { id: "B", name: "B", imageNames: [] },
    ]);
    expect(moveImageToSection(groups, "a", "B")).toEqual([
      { id: "A", name: "A", imageNames: ["b"] },
      { id: "B", name: "B", imageNames: ["a"] },
    ]);
  });

  it("cross-group move strips the image from the source group's divider", () => {
    const out = moveImageToSection(fixture(), "d", "g2", { dividerId: "dl" });
    expect(view(out[0])).toEqual([[null, ["a", "b"]], ["Day 1", ["c", "e"]], ["Day 2", ["f"]]]);
    expect(out[0].imageNames).not.toContain("d");
    expect(view(out[1])).toEqual([[null, ["x"]], ["Late", ["y", "d"]]]);
    expectInvariants(out);
  });

  it("moves from the timeline (in no group) straight into a divider", () => {
    const out = moveImageToSection(fixture(), "new", "g1", { dividerId: "d2" });
    expect(view(out[0])[2]).toEqual(["Day 2", ["f", "new"]]);
    expectInvariants(out);
  });

  it("dropping into the image's own section with no beforeImage is a no-op", () => {
    const groups = fixture();
    expect(moveImageToSection(groups, "a", "g1")).toBe(groups);
    expect(moveImageToSection(groups, "c", "g1", { dividerId: "d1" })).toBe(groups);
    expect(moveImageToSection(groups, "c", "g1", { dividerId: "d1", beforeImage: null })).toBe(groups);
    expect(moveImageToSection(groups, "c", "g1", { dividerId: "d1", beforeImage: "c" })).toBe(groups);
    // Before the tile right after it = where it already is.
    expect(moveImageToSection(groups, "c", "g1", { dividerId: "d1", beforeImage: "d" })).toBe(groups);
  });

  it("beforeImage inserts before that image (cross-section and within-section reorder)", () => {
    const into = moveImageToSection(fixture(), "a", "g1", { dividerId: "d1", beforeImage: "d" });
    expect(view(into[0])).toEqual([[null, ["b"]], ["Day 1", ["c", "a", "d", "e"]], ["Day 2", ["f"]]]);
    const reorder = moveImageToSection(fixture(), "e", "g1", { dividerId: "d1", beforeImage: "c" });
    expect(view(reorder[0])[1]).toEqual(["Day 1", ["e", "c", "d"]]);
    const back = moveImageToSection(fixture(), "d", "g1", { dividerId: null, beforeImage: "a" });
    expect(view(back[0])).toEqual([[null, ["d", "a", "b"]], ["Day 1", ["c", "e"]], ["Day 2", ["f"]]]);
    expectInvariants([...into, ...reorder, ...back]);
  });

  it("a beforeImage outside the target section appends", () => {
    const out = moveImageToSection(fixture(), "p", "g1", { dividerId: "d2", beforeImage: "a" });
    expect(view(out[0])[2]).toEqual(["Day 2", ["f", "p"]]);
  });

  it("unknown group is a no-op; unknown divider falls back to unsectioned", () => {
    const groups = fixture();
    expect(moveImageToSection(groups, "a", "nope")).toBe(groups);
    const out = moveImageToSection(groups, "p", "g1", { dividerId: "ghost" });
    expect(view(out[0])[0]).toEqual([null, ["a", "b", "p"]]);
  });

  it("only touches the source and target groups", () => {
    const groups = fixture();
    const out = moveImageToSection(groups, "x", "g1");
    expect(out[2]).toBe(groups[2]);
  });
});

/* ── addDividerToGroup ────────────────────────────────── */

describe("addDividerToGroup", () => {
  it("appends an empty 'Subgroup N' divider; an old-shape group gains a dividers key", () => {
    const groups = fixture();
    const { groups: out, dividerId } = addDividerToGroup(groups, "g3", { id: "new" });
    expect(dividerId).toBe("new");
    expect(out[2]).toEqual({
      id: "g3", name: "Plain", imageNames: ["p", "q"],
      dividers: [{ id: "new", name: "Subgroup 1", imageNames: [] }],
    });
    const again = addDividerToGroup(out, "g3", { id: "new" });
    expect(again.dividerId).toBe("new-2");                 // ids stay unique
    expect(again.groups[2].dividers![1].name).toBe("Subgroup 2");
    expect(out[0]).toBe(groups[0]);
    expectInvariants(again.groups);
  });

  it("uses a given name and generates an id when none is supplied", () => {
    const { groups: out, dividerId } = addDividerToGroup(fixture(), "g1", { name: "  Day 9 " });
    expect(typeof dividerId).toBe("string");
    expect(dividerId).not.toBe("");
    expect(view(out[0])[3]).toEqual(["Day 9", []]);
  });

  it("split at the FIRST unsectioned image moves the whole area into a new first divider", () => {
    const { groups: out } = addDividerToGroup(fixture(), "g1", { id: "n", splitBeforeImage: "a" });
    expect(view(out[0])).toEqual([
      [null, []], ["Subgroup 3", ["a", "b"]], ["Day 1", ["c", "d", "e"]], ["Day 2", ["f"]],
    ]);
    expectInvariants(out);
  });

  it("split in the MIDDLE of a plain group", () => {
    const { groups: out } = addDividerToGroup(fixture(), "g3", { id: "n", splitBeforeImage: "q" });
    expect(view(out[2])).toEqual([[null, ["p"]], ["Subgroup 1", ["q"]]]);
    expectInvariants(out);
  });

  it("split in the middle of a divider inserts right after that divider", () => {
    const { groups: out } = addDividerToGroup(fixture(), "g1", { id: "n", splitBeforeImage: "d" });
    expect(view(out[0])).toEqual([
      [null, ["a", "b"]], ["Day 1", ["c"]], ["Subgroup 3", ["d", "e"]], ["Day 2", ["f"]],
    ]);
    expectInvariants(out);
  });

  it("split at the LAST image of a section moves just that image", () => {
    const { groups: out } = addDividerToGroup(fixture(), "g1", { id: "n", splitBeforeImage: "e" });
    expect(view(out[0])).toEqual([
      [null, ["a", "b"]], ["Day 1", ["c", "d"]], ["Subgroup 3", ["e"]], ["Day 2", ["f"]],
    ]);
  });

  it("split on an image not in the group appends an empty divider; unknown group is a no-op", () => {
    const { groups: out } = addDividerToGroup(fixture(), "g3", { id: "n", splitBeforeImage: "zzz" });
    expect(view(out[2])).toEqual([[null, ["p", "q"]], ["Subgroup 1", []]]);
    const groups = fixture();
    expect(addDividerToGroup(groups, "nope")).toEqual({ groups, dividerId: null });
  });

  it("nextSubgroupName skips names already taken", () => {
    expect(nextSubgroupName({ id: "g", name: "G", imageNames: [] })).toBe("Subgroup 1");
    expect(nextSubgroupName({
      id: "g", name: "G", imageNames: [],
      dividers: [{ id: "1", name: "Subgroup 2", imageNames: [] }],
    })).toBe("Subgroup 3");
  });
});

/* ── renameDividerInGroup ─────────────────────────────── */

describe("renameDividerInGroup", () => {
  it("renames only that divider, preserving its unknown fields", () => {
    const groups = deepFreeze<ImageGroup[]>([{
      id: "g", name: "G", imageNames: ["a"],
      dividers: [{ id: "d", name: "Old", imageNames: ["a"], extra: 1 } as never],
    }]);
    const out = renameDividerInGroup(groups, "g", "d", "New");
    expect(out[0].dividers![0]).toEqual({ id: "d", name: "New", imageNames: ["a"], extra: 1 });
    expect(out[0].imageNames).toBe(groups[0].imageNames);
  });

  it("no-ops (same array) for an unchanged name or a missing group/divider", () => {
    const groups = fixture();
    expect(renameDividerInGroup(groups, "g1", "d1", "Day 1")).toBe(groups);
    expect(renameDividerInGroup(groups, "g1", "ghost", "X")).toBe(groups);
    expect(renameDividerInGroup(groups, "nope", "d1", "X")).toBe(groups);
    expect(renameDividerInGroup(groups, "g3", "d1", "X")).toBe(groups);
  });
});

/* ── deleteDividerFromGroup ───────────────────────────── */

describe("deleteDividerFromGroup", () => {
  it("deleting the FIRST divider merges its images onto the end of the unsectioned area", () => {
    const out = deleteDividerFromGroup(fixture(), "g1", "d1");
    expect(view(out[0])).toEqual([[null, ["a", "b", "c", "d", "e"]], ["Day 2", ["f"]]]);
    expectInvariants(out);
  });

  it("deleting a later divider merges into the divider above it", () => {
    const out = deleteDividerFromGroup(fixture(), "g1", "d2");
    expect(view(out[0])).toEqual([[null, ["a", "b"]], ["Day 1", ["c", "d", "e", "f"]]]);
    expectInvariants(out);
  });

  it("deleting the last remaining divider removes the dividers key (back to old shape)", () => {
    const out = deleteDividerFromGroup(fixture(), "g2", "dl");
    expect(out[1]).toEqual({ id: "g2", name: "Group 2", imageNames: ["x", "y"] });
    expect("dividers" in out[1]).toBe(false);
  });

  it("no image is ever lost from the group", () => {
    const groups = fixture();
    const out = deleteDividerFromGroup(deleteDividerFromGroup(groups, "g1", "d1"), "g1", "d2");
    expect([...out[0].imageNames].sort()).toEqual([...groups[0].imageNames].sort());
  });

  it("missing group/divider is a no-op", () => {
    const groups = fixture();
    expect(deleteDividerFromGroup(groups, "g1", "ghost")).toBe(groups);
    expect(deleteDividerFromGroup(groups, "nope", "d1")).toBe(groups);
  });
});

describe("moveImageToSection — atEnd (a section's explicit end drop zone)", () => {
  const groups = () => deepFreeze<ImageGroup[]>([{
    id: "g", name: "G", imageNames: ["x", "a", "b", "c"],
    dividers: [{ id: "d", name: "D", imageNames: ["a", "b", "c"] }],
  }]);
  it("moves an image to the end of its OWN section", () => {
    const out = moveImageToSection(groups(), "a", "g", { dividerId: "d", atEnd: true });
    expect(out[0].dividers![0].imageNames).toEqual(["b", "c", "a"]);
    expect(out[0].imageNames.slice().sort()).toEqual(["a", "b", "c", "x"]);
  });
  it("is a no-op when the image is already last", () => {
    const g = groups();
    expect(moveImageToSection(g, "c", "g", { dividerId: "d", atEnd: true })).toBe(g);
  });
  it("without atEnd, a same-section drop stays a no-op", () => {
    const g = groups();
    expect(moveImageToSection(g, "a", "g", { dividerId: "d" })).toBe(g);
  });
});
