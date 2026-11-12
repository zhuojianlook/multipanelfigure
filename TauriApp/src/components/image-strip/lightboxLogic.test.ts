import { describe, it, expect } from "vitest";
import {
  resolveTileClick,
  lightboxMaxSide,
  LIGHTBOX_DEFAULT_MAX_SIDE,
  type TileClickSnapshot,
} from "./lightboxLogic";

/** Feed a sequence of (detail, tile) clicks through resolveTileClick the way
 *  ImageTile does, starting from `initial`, and return the final selection. */
function runClicks(initial: string | null, clicks: Array<[number, string]>): string | null {
  let sel = initial;
  let snap: TileClickSnapshot | null = null;
  for (const [detail, tile] of clicks) {
    const r = resolveTileClick(detail, tile, sel, snap);
    sel = r.next;
    snap = r.snapshot;
  }
  return sel;
}

describe("resolveTileClick — single click still toggles immediately", () => {
  it("selects an unselected tile", () => {
    expect(resolveTileClick(1, "a.png", null, null).next).toBe("a.png");
  });
  it("deselects the selected tile", () => {
    expect(resolveTileClick(1, "a.png", "a.png", null).next).toBeNull();
  });
  it("switches selection from another tile", () => {
    expect(resolveTileClick(1, "a.png", "b.png", null).next).toBe("a.png");
  });
  it("treats a synthesized click (detail 0) as a single click", () => {
    expect(resolveTileClick(0, "a.png", null, null).next).toBe("a.png");
  });
  it("records what the selection was before the click", () => {
    expect(resolveTileClick(1, "a.png", "b.png", null).snapshot).toEqual({ tile: "a.png", before: "b.png" });
  });
});

describe("resolveTileClick — a double-click nets to no selection change", () => {
  // Every starting state: nothing, this tile, and a DIFFERENT tile. The last
  // one is the case a naive double toggle gets wrong (it would clear B).
  for (const initial of [null, "a.png", "b.png"]) {
    it(`from selection=${String(initial)}`, () => {
      expect(runClicks(initial, [[1, "a.png"], [2, "a.png"]])).toBe(initial);
    });
  }
  it("works for non-ASCII filenames (U+202F narrow no-break space)", () => {
    const n = "Screenshot 2026-03-01 at 9.41.12\u202FAM.png";
    expect(runClicks("b.png", [[1, n], [2, n]])).toBe("b.png");
  });
  it("two separate single clicks still toggle on then off", () => {
    expect(runClicks(null, [[1, "a.png"], [1, "a.png"]])).toBeNull();
  });
  it("a third click (triple-click) changes nothing further", () => {
    expect(runClicks("b.png", [[1, "a.png"], [2, "a.png"], [3, "a.png"]])).toBe("b.png");
  });
});

describe("resolveTileClick — never replays a snapshot from another tile", () => {
  it("detail=2 on a tile whose detail=1 landed on a different tile changes nothing", () => {
    const snap: TileClickSnapshot = { tile: "b.png", before: "x.png" };
    const r = resolveTileClick(2, "a.png", "b.png", snap);
    expect(r.next).toBe("b.png");
    expect(r.snapshot).toBeNull();
  });
  it("detail=2 with no snapshot changes nothing", () => {
    expect(resolveTileClick(2, "a.png", "c.png", null).next).toBe("c.png");
  });
  it("the snapshot is consumed so it cannot be replayed", () => {
    const r = resolveTileClick(2, "a.png", "a.png", { tile: "a.png", before: null });
    expect(r.next).toBeNull();
    expect(r.snapshot).toBeNull();
  });
});

describe("lightboxMaxSide", () => {
  it("sizes from the 92vw × 88vh image box (the longer side), not the viewport", () => {
    expect(lightboxMaxSide(1440, 900, 1)).toBe(1325);  // 0.92·1440 = 1324.8
    expect(lightboxMaxSide(900, 1440, 1)).toBe(1268);  // 0.88·1440 = 1267.2
  });
  it("scales by devicePixelRatio and rounds up", () => {
    expect(lightboxMaxSide(1000, 700, 1.25)).toBe(1150);  // 920 × 1.25
    expect(lightboxMaxSide(1001, 700, 1.5)).toBe(1382);   // 920.92 × 1.5 = 1381.38
  });
  it("caps at 2048 to keep the base64 IPC payload sane", () => {
    expect(lightboxMaxSide(1440, 900, 2)).toBe(2048);   // a 2× Retina laptop
    expect(lightboxMaxSide(3840, 2160, 2)).toBe(2048);
  });
  it("never asks for less than 512", () => {
    expect(lightboxMaxSide(300, 200, 1)).toBe(512);
  });
  it("treats a missing / bogus DPR as 1", () => {
    expect(lightboxMaxSide(1200, 800, 0)).toBe(1104);
    expect(lightboxMaxSide(1200, 800, Number.NaN)).toBe(1104);
  });
  it("falls back to the default for an unlaid-out viewport", () => {
    expect(lightboxMaxSide(0, 0, 2)).toBe(LIGHTBOX_DEFAULT_MAX_SIDE);
    expect(lightboxMaxSide(Number.NaN, 800, 2)).toBe(LIGHTBOX_DEFAULT_MAX_SIDE);
  });
});
