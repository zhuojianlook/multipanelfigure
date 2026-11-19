/* The workflow tab's menu for where its uploaded images are saved:
   "Store uploaded images in ▸" (shared) / "Save this analysis in…" (uploads only). */

import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { UploadHomeMenu } from "./UploadHomeMenu";

const AT = { top: 10, left: 10 };
const CHOICES = [{ uid: "du_a", label: "a" }, { uid: "du_b", label: "b" }];

afterEach(() => cleanup());

describe("UploadHomeMenu", () => {
  it("shared: lists the workflow's documents, the current one checked; picking another moves the uploads", () => {
    const onPick = vi.fn();
    const onClose = vi.fn();
    render(<UploadHomeMenu anchor={AT} onClose={onClose} choices={CHOICES} home="du_a" onPick={onPick} />);
    expect(screen.getByText("Store uploaded images in ▸")).toBeTruthy();
    const items = screen.getAllByRole("menuitemradio");
    expect(items.map((i) => i.getAttribute("aria-checked"))).toEqual(["true", "false"]);
    fireEvent.click(items[0]);                                 // already there
    expect(onPick).not.toHaveBeenCalled();
    fireEvent.click(items[1]);
    expect(onPick).toHaveBeenCalledWith("du_b");
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("read-only while the workflow is locked", () => {
    const onPick = vi.fn();
    render(<UploadHomeMenu anchor={AT} onClose={vi.fn()} choices={CHOICES} home="du_a" onPick={onPick} disabled />);
    for (const item of screen.getAllByRole("menuitemradio")) expect(item.getAttribute("aria-disabled")).toBe("true");
  });

  it("uploads only: offers the choose-document dialog; nothing shown when closed", () => {
    const onChoose = vi.fn();
    const { rerender } = render(<UploadHomeMenu anchor={AT} onClose={vi.fn()} onChooseDocument={onChoose} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "Save this analysis in…" }));
    expect(onChoose).toHaveBeenCalledTimes(1);
    rerender(<UploadHomeMenu anchor={null} onClose={vi.fn()} onChooseDocument={onChoose} />);
    expect(screen.queryByRole("menuitem", { name: "Save this analysis in…" })).toBeNull();
  });
});
