/* A document's Analysis labels in its .mpf manifest (plan step 8): source
   renames, measurement group labels and source-library folders, stored
   document-relative ("@…") so the file can move, merged back into the
   local labels on read without ever touching another document's keys. */

import { describe, it, expect } from "vitest";
import {
  absKeyOf, asLabels, docMetaOf, fileUnderFolder, mergeDocMeta, metaAsRead, metaSigForm, readDocMeta, relKeyOf,
  renameMetaUploads, uploadKey, DIRECT_UPLOAD_FOLDER, type AnalysisLabels, type DocMeta,
} from "./analysisMeta";

const A = "p:/exp/a.mpf";
const A2 = "p:/moved/a.mpf";
const B = "p:/exp/b.mpf";
const HASH = "p:/exp/a.mpf#b.mpf";            // another document whose path starts like A's
const NONE = new Set<string>();
const UP = new Set([uploadKey("stack.tif")]);

function labels(p: Partial<AnalysisLabels> = {}): AnalysisLabels {
  return { sourceNames: {}, measurementGroups: {}, sourceGroups: [], ...p };
}

describe("relative keys", () => {
  it("the document's own keys go relative — sources, figure tabs, measurement rows; uploads only when it owns them", () => {
    expect(relKeyOf(`${A}::r0c0_i0`, A, NONE)).toBe("@::r0c0_i0");
    expect(relKeyOf(`${A}#fig-2::r1c0_panel`, A, NONE)).toBe("@#fig-2::r1c0_panel");
    expect(relKeyOf(`${A}|R1C1|0|1`, A, NONE)).toBe("@|R1C1|0|1");
    expect(relKeyOf(`${A}#fig-2|R1C1|0|1`, A, NONE)).toBe("@#fig-2|R1C1|0|1");
    expect(relKeyOf(uploadKey("stack.tif"), A, UP)).toBe("img:stack.tif");
    expect(relKeyOf(uploadKey("other.tif"), A, UP)).toBeNull();
    // Other documents' keys — including one whose path merely starts with A's.
    expect(relKeyOf(`${B}::r0c0_i0`, A, NONE)).toBeNull();
    expect(relKeyOf(`${HASH}::r0c0_i0`, A, NONE)).toBeNull();
    expect(relKeyOf("r0c0_i0", A, NONE)).toBeNull();                // legacy bare key
  });

  it("back under wherever the document is now; anything that isn't one of its keys is refused", () => {
    expect(absKeyOf("@::r0c0_i0", A2, NONE)).toBe(`${A2}::r0c0_i0`);
    expect(absKeyOf("@#fig-2|R1C1|0|1", A2, NONE)).toBe(`${A2}#fig-2|R1C1|0|1`);
    expect(absKeyOf("img:stack.tif", A2, UP)).toBe("img:stack.tif");
    expect(absKeyOf("img:other.tif", A2, UP)).toBeNull();
    expect(absKeyOf(`${B}::r0c0_i0`, A2, NONE)).toBeNull();          // a file never writes B's keys
    expect(absKeyOf("@garbage", A2, NONE)).toBeNull();
  });
});

describe("docMetaOf", () => {
  it("keeps only the document's labels, relative; folders keep only its keys", () => {
    const l = labels({
      sourceNames: { [`${A}::r0c0_i0`]: "Control_1", [`${B}::r0c0_i0`]: "B one", "img:stack.tif": "Stack", "img:x.tif": "X", [`${A}::r0c1_i0`]: "  " },
      measurementGroups: { [`${A}|R1C1|0|0`]: "Control", [`${B}|R1C1|0|0`]: "Treated" },
      sourceGroups: [
        { id: "g1", name: "Mixed", sourceKeys: [`${B}::r0c0_i0`, `${A}::r0c0_i0`, "img:stack.tif"] },
        { id: "g2", name: "Only B", sourceKeys: [`${B}::r0c1_i0`] },
      ],
    });
    expect(docMetaOf(l, A, UP)).toEqual({
      sourceNames: { "@::r0c0_i0": "Control_1", "img:stack.tif": "Stack" },
      measurementGroups: { "@|R1C1|0|0": "Control" },
      sourceGroups: [{ id: "g1", name: "Mixed", sourceKeys: ["@::r0c0_i0", "img:stack.tif"] }],
    });
    expect(docMetaOf(l, "p:/exp/nothing.mpf", NONE)).toBeUndefined();
  });
});

describe("mergeDocMeta (hydrate)", () => {
  const fileMeta: DocMeta = {
    sourceNames: { "@::r0c0_i0": "Control_1", "img:stack.tif": "Stack" },
    measurementGroups: { "@|R1C1|0|0": "Control" },
    sourceGroups: [{ id: "g_file", name: "Conditions", sourceKeys: ["@::r0c0_i0", "img:stack.tif"] }],
  };

  it("restores the file's labels under the document's keys as they are now (the file moved)", () => {
    const out = mergeDocMeta(labels(), A2, fileMeta, UP);
    expect(out.sourceNames).toEqual({ [`${A2}::r0c0_i0`]: "Control_1", "img:stack.tif": "Stack" });
    expect(out.measurementGroups).toEqual({ [`${A2}|R1C1|0|0`]: "Control" });
    expect(out.sourceGroups).toEqual([{ id: "g_file", name: "Conditions", sourceKeys: [`${A2}::r0c0_i0`, "img:stack.tif"] }]);
  });

  it("the file's labels override; local-only labels and other documents' labels stay", () => {
    const local = labels({
      sourceNames: { [`${A}::r0c0_i0`]: "old name", [`${A}::r0c1_i0`]: "local only", [`${B}::r0c0_i0`]: "B one" },
      measurementGroups: { [`${A}|R1C1|0|0`]: "Treated", [`${B}|R1C1|0|0`]: "B group" },
    });
    const out = mergeDocMeta(local, A, fileMeta, UP);
    expect(out.sourceNames).toEqual({
      [`${A}::r0c0_i0`]: "Control_1", [`${A}::r0c1_i0`]: "local only", [`${B}::r0c0_i0`]: "B one", "img:stack.tif": "Stack",
    });
    expect(out.measurementGroups).toEqual({ [`${A}|R1C1|0|0`]: "Control", [`${B}|R1C1|0|0`]: "B group" });
  });

  it("folders merge by id, else by name, else are added; a key the file files moves out of any other folder", () => {
    const byId = mergeDocMeta(labels({
      sourceGroups: [{ id: "g_file", name: "Renamed here", sourceKeys: [`${B}::r0c0_i0`] }],
    }), A, fileMeta, UP);
    expect(byId.sourceGroups).toEqual([
      { id: "g_file", name: "Renamed here", sourceKeys: [`${B}::r0c0_i0`, `${A}::r0c0_i0`, "img:stack.tif"] },
    ]);
    const byName = mergeDocMeta(labels({
      sourceGroups: [
        { id: "g_other", name: "Elsewhere", sourceKeys: [`${A}::r0c0_i0`, `${B}::r0c1_i0`] },
        { id: "g_local", name: "Conditions", sourceKeys: [] },
      ],
    }), A, fileMeta, UP);
    expect(byName.sourceGroups).toEqual([
      { id: "g_other", name: "Elsewhere", sourceKeys: [`${B}::r0c1_i0`] },
      { id: "g_local", name: "Conditions", sourceKeys: [`${A}::r0c0_i0`, "img:stack.tif"] },
    ]);
    const added = mergeDocMeta(labels({ sourceGroups: [{ id: "g_x", name: "X", sourceKeys: [] }] }), A, fileMeta, UP);
    expect(added.sourceGroups.map((g) => g.id)).toEqual(["g_x", "g_file"]);
  });

  it("never writes another document's keys or uploads it doesn't own; merging again changes nothing", () => {
    const crafted = {
      sourceNames: { [`${B}::r0c0_i0`]: "hijack", "img:not-mine.tif": "no", "@::r0c0_i0": "mine" },
      sourceGroups: [{ id: "g", name: "G", sourceKeys: [`${B}::r0c0_i0`, "img:not-mine.tif"] }],
    } as DocMeta;
    const out = mergeDocMeta(labels(), A, crafted, UP);
    expect(out.sourceNames).toEqual({ [`${A}::r0c0_i0`]: "mine" });
    expect(out.sourceGroups).toEqual([]);
    const once = mergeDocMeta(labels(), A, fileMeta, UP);
    expect(mergeDocMeta(once, A, fileMeta, UP)).toBe(once);
    const l = labels();
    expect(mergeDocMeta(l, A, undefined, UP)).toBe(l);
  });

  it("round trip: what a document saves is what its file restores", () => {
    const local = labels({
      sourceNames: { [`${A}::r0c0_i0`]: "Control_1" },
      measurementGroups: { [`${A}#fig-2|R2C1|1|0`]: "Treated" },
      sourceGroups: [{ id: "g1", name: "Conditions", sourceKeys: [`${A}::r0c0_i0`] }],
    });
    const saved = docMetaOf(local, A, NONE);
    const restored = mergeDocMeta(labels(), A2, readDocMeta(JSON.parse(JSON.stringify(saved)), NONE), NONE);
    expect(docMetaOf(restored, A2, NONE)).toEqual(saved);
  });
});

describe("readDocMeta / metaSigForm / renames", () => {
  it("readDocMeta keeps well-formed relative keys and owned uploads only", () => {
    expect(readDocMeta({
      sourceNames: { "@::r0c0_i0": "ok", [`${B}::r0c0_i0`]: "no", "img:stack.tif": "up", "img:x.tif": "no", "@::r1": 3 },
      measurementGroups: "nonsense",
      sourceGroups: [{ id: "g", name: "G", sourceKeys: ["@::r0c0_i0", 7, "@::r0c0_i0"] }, { id: 1 }],
    }, UP)).toEqual({
      sourceNames: { "@::r0c0_i0": "ok", "img:stack.tif": "up" },
      sourceGroups: [{ id: "g", name: "G", sourceKeys: ["@::r0c0_i0"] }],
    });
    expect(readDocMeta(null, UP)).toBeUndefined();
    expect(readDocMeta({ sourceNames: {} }, UP)).toBeUndefined();
  });

  it("metaSigForm ignores folder ids and order (a folder merged by name is the same folder)", () => {
    const a: DocMeta = { sourceGroups: [{ id: "g1", name: "X", sourceKeys: ["@::a", "@::b"] }, { id: "g2", name: "Y", sourceKeys: ["@::c"] }] };
    const b: DocMeta = { sourceGroups: [{ id: "zz", name: "Y", sourceKeys: ["@::c"] }, { id: "g9", name: "X", sourceKeys: ["@::b", "@::a"] }] };
    expect(metaSigForm(a)).toEqual(metaSigForm(b));
    expect(metaSigForm({ sourceGroups: [{ id: "g1", name: "Z", sourceKeys: ["@::a", "@::b"] }, { id: "g2", name: "Y", sourceKeys: ["@::c"] }] }))
      .not.toEqual(metaSigForm(a));
    expect(metaSigForm(undefined)).toBeUndefined();
  });

  it("renameMetaUploads follows restored uploads' new names, in one pass", () => {
    const m = {
      sourceNames: { "img:a.tif": "A", "img:b.tif": "B", "@::r0c0_i0": "x" },
      sourceGroups: [{ id: "g", name: "G", sourceKeys: ["img:a.tif", "@::r0c0_i0"] }],
    };
    expect(renameMetaUploads(m, { "a.tif": "b.tif", "b.tif": "c.tif" })).toEqual({
      sourceNames: { "img:b.tif": "A", "img:c.tif": "B", "@::r0c0_i0": "x" },
      sourceGroups: [{ id: "g", name: "G", sourceKeys: ["img:b.tif", "@::r0c0_i0"] }],
    });
    expect(renameMetaUploads(m, {})).toBe(m);
  });
});

describe("Direct upload", () => {
  it("fileUnderFolder parks keys that are in no folder (creating the folder first when missing)", () => {
    const l = labels({ sourceGroups: [{ id: "g", name: "Mine", sourceKeys: ["img:kept.tif"] }] });
    const out = fileUnderFolder(l, ["img:kept.tif", "img:new.tif"], DIRECT_UPLOAD_FOLDER, () => "grp_direct_1");
    expect(out.sourceGroups).toEqual([
      { id: "grp_direct_1", name: DIRECT_UPLOAD_FOLDER, sourceKeys: ["img:new.tif"] },
      { id: "g", name: "Mine", sourceKeys: ["img:kept.tif"] },
    ]);
    const again = fileUnderFolder(out, ["img:another.tif"], DIRECT_UPLOAD_FOLDER, () => "never");
    expect(again.sourceGroups[0]).toEqual({ id: "grp_direct_1", name: DIRECT_UPLOAD_FOLDER, sourceKeys: ["img:new.tif", "img:another.tif"] });
    expect(fileUnderFolder(out, ["img:new.tif"], DIRECT_UPLOAD_FOLDER, () => "x")).toBe(out);
  });

  it("metaAsRead: a restored upload the file files nowhere counts as filed in Direct upload", () => {
    const m: DocMeta = { sourceNames: { "img:a.tif": "A" } };
    expect(metaSigForm(metaAsRead(m, ["img:a.tif"]))).toEqual(metaSigForm({
      sourceNames: { "img:a.tif": "A" },
      sourceGroups: [{ id: "whatever", name: DIRECT_UPLOAD_FOLDER, sourceKeys: ["img:a.tif"] }],
    }));
    const filed: DocMeta = { sourceGroups: [{ id: "g", name: "Mine", sourceKeys: ["img:a.tif"] }] };
    expect(metaAsRead(filed, ["img:a.tif"])).toBe(filed);
  });

  it("asLabels drops malformed localStorage contents", () => {
    expect(asLabels({ a: "x", b: 2 }, [], [{ id: "g", name: "G", sourceKeys: ["k", 1] }, "bad"])).toEqual({
      sourceNames: { a: "x" }, measurementGroups: {}, sourceGroups: [{ id: "g", name: "G", sourceKeys: ["k"] }],
    });
  });
});
