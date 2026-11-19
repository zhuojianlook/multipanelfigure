/* Drag-to-reorder on the .mpf document tab strip. Pointer events (not HTML5
   drag-and-drop: a native drag that ends off a target soft-locks WKWebView),
   a 5px threshold so clicks / double-clicks are untouched, Esc cancels, and
   the click that ends a drag doesn't also switch documents. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, fireEvent } from "@testing-library/react";

const nav = vi.hoisted(() => ({
  enterCollage: vi.fn(async () => {}),
  enterAnalysis: vi.fn(async () => {}),
  switchToDocument: vi.fn(async () => {}),
  newBlankDoc: vi.fn(async () => {}),
  closeDoc: vi.fn(async () => {}),
  saveDocument: vi.fn(async () => {}),
}));
vi.mock("../../utils/projectNav", () => nav);
vi.mock("./Toolbar", () => ({ RecordAppButton: () => null, DEV_OPTIONS_KEY: "mpfig-dev-options" }));
vi.mock("../../utils/sessionRestore", () => ({ getReopenPref: () => false, setReopenPref: vi.fn() }));
vi.mock("../../api/client", () => ({
  api: new Proxy({}, { get: () => vi.fn(async () => ({})) }),
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));

import { DocumentTabs } from "./DocumentTabs";
import { useCollageStore } from "../../store/collageStore";

const DOCS = [
  { id: "d1", path: "/x/one.mpf", name: "one" },
  { id: "d2", path: "/x/two.mpf", name: "two" },
  { id: "d3", path: "/x/three.mpf", name: "three" },
];
const TAB_W = 100;

// jsdom has no PointerEvent: a MouseEvent carrying a pointerId is all the
// component reads.
class TestPointerEvent extends MouseEvent {
  pointerId: number;
  constructor(type: string, init: MouseEventInit & { pointerId?: number } = {}) {
    super(type, { bubbles: true, cancelable: true, ...init });
    this.pointerId = init.pointerId ?? 1;
  }
}
const ptr = (target: EventTarget, type: string, clientX: number, buttons = 1) =>
  act(() => { target.dispatchEvent(new TestPointerEvent(type, { clientX, button: 0, buttons })); });

const order = () => useCollageStore.getState().openDocs.map((d) => d.id);
const tab = (id: string) => document.querySelector(`[data-doc-tab="${id}"]`) as HTMLElement;

let rectSpy: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  useCollageStore.setState({ openDocs: DOCS.map((d) => ({ ...d })), activeDocId: "d1", mode: "builder" });
  // Lay the doc tabs out side by side, 100px each, in their CURRENT order.
  rectSpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    const id = this.getAttribute("data-doc-tab");
    const i = id ? order().indexOf(id) : -1;
    const left = i >= 0 ? i * TAB_W : 0;
    const width = i >= 0 ? TAB_W : 0;
    return { left, right: left + width, width, top: 0, bottom: 30, height: 30, x: left, y: 0, toJSON: () => ({}) } as DOMRect;
  });
  Object.values(nav).forEach((f) => f.mockClear());
});
afterEach(() => { rectSpy.mockRestore(); cleanup(); });

describe("DocumentTabs drag to reorder", () => {
  it("dragging the first tab past the third puts it last, without switching documents", () => {
    render(<DocumentTabs />);
    ptr(tab("d1"), "pointerdown", 50);
    ptr(window, "pointermove", 120);
    ptr(window, "pointermove", 280);
    // The dragged tab is dimmed and an insertion line marks the drop slot
    // (after the last tab).
    expect(tab("d1").getAttribute("data-dragging")).toBe("true");
    expect(tab("d3").getAttribute("data-drop-line")).toBe("right");
    ptr(window, "pointerup", 280, 0);
    expect(tab("d1").getAttribute("data-dragging")).toBeNull();
    expect(document.querySelector("[data-drop-line]")).toBeNull();
    expect(order()).toEqual(["d2", "d3", "d1"]);
    // The click the release produces must not switch to the dragged tab.
    fireEvent.click(tab("d1"));
    expect(nav.switchToDocument).not.toHaveBeenCalled();
    // ...but the next real click does.
    ptr(tab("d2"), "pointerdown", 50);
    ptr(window, "pointerup", 50, 0);
    fireEvent.click(tab("d2"));
    expect(nav.switchToDocument).toHaveBeenCalledWith("d2");
  });

  it("dragging the last tab to the front", () => {
    render(<DocumentTabs />);
    ptr(tab("d3"), "pointerdown", 250);
    ptr(window, "pointermove", 10);
    ptr(window, "pointerup", 10, 0);
    expect(order()).toEqual(["d3", "d1", "d2"]);
  });

  it("a press that moves less than the threshold is a click, not a drag", () => {
    render(<DocumentTabs />);
    ptr(tab("d2"), "pointerdown", 150);
    ptr(window, "pointermove", 153);
    ptr(window, "pointerup", 153, 0);
    fireEvent.click(tab("d2"));
    expect(order()).toEqual(["d1", "d2", "d3"]);
    expect(nav.switchToDocument).toHaveBeenCalledWith("d2");
  });

  it("Esc cancels a drag in progress", () => {
    render(<DocumentTabs />);
    ptr(tab("d1"), "pointerdown", 50);
    ptr(window, "pointermove", 280);
    act(() => { window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); });
    ptr(window, "pointerup", 280, 0);
    expect(order()).toEqual(["d1", "d2", "d3"]);
    expect(document.body.style.cursor).toBe("");
  });

  it("dropping where it started changes nothing", () => {
    render(<DocumentTabs />);
    const before = useCollageStore.getState().openDocs;
    ptr(tab("d2"), "pointerdown", 150);
    ptr(window, "pointermove", 190);
    ptr(window, "pointerup", 190, 0);
    // Same array: no store write, so nothing downstream re-runs.
    expect(useCollageStore.getState().openDocs).toBe(before);
  });

  it("a press on the close button never starts a drag", () => {
    render(<DocumentTabs />);
    const close = tab("d1").querySelector('[data-tab-action="close"]') as HTMLElement;
    ptr(close, "pointerdown", 90);
    ptr(window, "pointermove", 280);
    ptr(window, "pointerup", 280, 0);
    expect(order()).toEqual(["d1", "d2", "d3"]);
  });

  it("a move with the button already released ends the press instead of dragging", () => {
    render(<DocumentTabs />);
    ptr(tab("d1"), "pointerdown", 50);
    ptr(window, "pointermove", 280, 0);   // buttons = 0: released somewhere unseen
    ptr(window, "pointermove", 280, 1);
    ptr(window, "pointerup", 280, 0);
    expect(order()).toEqual(["d1", "d2", "d3"]);
  });

  it("double-click still renames the tab", () => {
    render(<DocumentTabs />);
    fireEvent.doubleClick(screen.getByText("two"));
    expect(document.querySelector("input")).not.toBeNull();
  });
});

describe("collageStore.docReorder", () => {
  beforeEach(() => {
    useCollageStore.setState({ openDocs: DOCS.map((d) => ({ ...d })), activeDocId: "d1" });
  });
  it("applies a permutation and keeps every tab's data", () => {
    useCollageStore.getState().docReorder(["d3", "d1", "d2"]);
    expect(useCollageStore.getState().openDocs).toEqual([DOCS[2], DOCS[0], DOCS[1]]);
    expect(useCollageStore.getState().activeDocId).toBe("d1");
  });
  it("ignores lists that are not a permutation of the current tabs", () => {
    const before = useCollageStore.getState().openDocs;
    const s = useCollageStore.getState();
    s.docReorder(["d1", "d2"]);                 // a tab missing
    s.docReorder(["d1", "d2", "d3", "d4"]);     // an extra one
    s.docReorder(["d1", "d1", "d2"]);           // a duplicate
    s.docReorder(["d1", "d2", "zz"]);           // an unknown id
    s.docReorder(["d1", "d2", "d3"]);           // unchanged
    expect(useCollageStore.getState().openDocs).toBe(before);
  });
});
