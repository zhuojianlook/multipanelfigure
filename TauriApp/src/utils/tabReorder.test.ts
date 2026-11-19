/* Arithmetic behind drag-to-reorder for the .mpf document tabs. */

import { describe, it, expect } from "vitest";
import { dropSlot, moveToSlot } from "./tabReorder";

describe("dropSlot", () => {
  const mids = [50, 150, 250];   // three 100px tabs starting at x=0
  it("counts the tabs whose midpoint is left of the pointer", () => {
    expect(dropSlot(mids, 10)).toBe(0);
    expect(dropSlot(mids, 120)).toBe(1);
    expect(dropSlot(mids, 199)).toBe(2);
    expect(dropSlot(mids, 400)).toBe(3);
  });
  it("handles an empty strip", () => {
    expect(dropSlot([], 100)).toBe(0);
  });
});

describe("moveToSlot", () => {
  const ids = ["a", "b", "c", "d"] as const;
  it("moves right: the slot is measured before the move", () => {
    expect(moveToSlot(ids, "a", 2)).toEqual(["b", "a", "c", "d"]);
    expect(moveToSlot(ids, "a", 4)).toEqual(["b", "c", "d", "a"]);
    expect(moveToSlot(ids, "b", 4)).toEqual(["a", "c", "d", "b"]);
  });
  it("moves left", () => {
    expect(moveToSlot(ids, "d", 0)).toEqual(["d", "a", "b", "c"]);
    expect(moveToSlot(ids, "c", 1)).toEqual(["a", "c", "b", "d"]);
  });
  it("returns the SAME array when nothing changes", () => {
    // The slots on either side of a tab leave it where it is.
    expect(moveToSlot(ids, "b", 1)).toBe(ids);
    expect(moveToSlot(ids, "b", 2)).toBe(ids);
    expect(moveToSlot(ids, "zz" as never, 0)).toBe(ids);
  });
  it("clamps an out-of-range slot", () => {
    expect(moveToSlot(ids, "b", -5)).toEqual(["b", "a", "c", "d"]);
    expect(moveToSlot(ids, "b", 99)).toEqual(["a", "c", "d", "b"]);
  });
  it("never adds or drops an id", () => {
    for (const id of ids) {
      for (let s = 0; s <= ids.length; s++) {
        expect([...moveToSlot(ids, id, s)].sort()).toEqual([...ids]);
      }
    }
  });
});
