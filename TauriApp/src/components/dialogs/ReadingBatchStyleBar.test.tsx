/* Batch label size / colour for a Curved Surface measurement's readings. */

import { describe, it, expect, vi, afterEach } from "vitest";
import { useState } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ReadingBatchStyleBar, type ReadingStylePatch } from "./ReadingBatchStyleBar";

afterEach(() => cleanup());

type Rd = { font_size?: number | null; color?: string | null };

/** Holds the selection like the Edit Panel does; applies patches in place. */
function Harness({ initial, onApply }: { initial: Rd[]; onApply: (idx: number[], p: ReadingStylePatch) => void }) {
  const [rds, setRds] = useState<Rd[]>(initial);
  const [sel, setSel] = useState<number[]>([]);
  return (
    <ReadingBatchStyleBar
      readings={rds} selected={sel} defaultFontSize={5} defaultColor="#00E5FF"
      onSelect={setSel}
      onApply={(idx, p) => {
        onApply(idx, p);
        setRds((prev) => prev.map((r, i) => (idx.includes(i) ? { ...r, ...p } : r)));
      }}
    />
  );
}

describe("ReadingBatchStyleBar", () => {
  it("All → one size change writes every reading in a single call", () => {
    const onApply = vi.fn();
    render(<Harness initial={[{}, {}, {}]} onApply={onApply} />);
    expect(screen.getByText("none selected")).toBeTruthy();
    expect(screen.queryByLabelText("Label size in points for the selected readings")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Select all readings" }));
    expect(screen.getByText("3 selected")).toBeTruthy();
    const size = screen.getByLabelText("Label size in points for the selected readings") as HTMLInputElement;
    expect(size.value).toBe("5");                       // common value = measurement default
    fireEvent.change(size, { target: { value: "9" } });
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply).toHaveBeenCalledWith([0, 1, 2], { font_size: 9 });
  });

  it("shows blank when sizes differ and ignores out-of-range input", () => {
    const onApply = vi.fn();
    render(<Harness initial={[{ font_size: 9 }, {}]} onApply={onApply} />);
    fireEvent.click(screen.getByRole("button", { name: "Select all readings" }));
    const size = screen.getByLabelText("Label size in points for the selected readings") as HTMLInputElement;
    expect(size.value).toBe("");                        // mixed
    fireEvent.change(size, { target: { value: "0" } });
    fireEvent.change(size, { target: { value: "" } });
    expect(onApply).not.toHaveBeenCalled();
  });

  it("colour applies to the selection; 'Use measurement default' sends nulls", () => {
    const onApply = vi.fn();
    render(<Harness initial={[{}, {}, {}]} onApply={onApply} />);
    fireEvent.click(screen.getByRole("button", { name: "Select all readings" }));
    const reset = screen.getByRole("button", { name: "Use measurement default" }) as HTMLButtonElement;
    expect(reset.disabled).toBe(true);                  // nothing overridden yet
    fireEvent.click(screen.getByRole("button", { name: "Label colour for the 3 selected readings" }));
    fireEvent.click(screen.getByTitle("#FF0000"));
    expect(onApply).toHaveBeenLastCalledWith([0, 1, 2], { color: "#FF0000" });
    expect(reset.disabled).toBe(false);
    fireEvent.click(reset);
    expect(onApply).toHaveBeenLastCalledWith([0, 1, 2], { font_size: null, color: null });
    expect(onApply).toHaveBeenCalledTimes(2);
  });

  it("a typed size is dropped when the selection changes without a blur", () => {
    // Shift-clicking a reading on the preview adds it to the selection
    // while the Size field keeps focus (the click preventDefaults), so
    // onBlur never runs. The field must not keep showing the typed value
    // for a selection that doesn't share it.
    const onApply = vi.fn();
    const props = { defaultFontSize: 5, defaultColor: "#00E5FF", onSelect: vi.fn(), onApply };
    const { rerender } = render(<ReadingBatchStyleBar readings={[{}, {}, {}]} selected={[0, 1]} {...props} />);
    const size = screen.getByLabelText("Label size in points for the selected readings") as HTMLInputElement;
    size.focus();
    fireEvent.change(size, { target: { value: "14" } });
    expect(onApply).toHaveBeenCalledWith([0, 1], { font_size: 14 });
    const applied = [{ font_size: 14 }, { font_size: 14 }, {}];
    rerender(<ReadingBatchStyleBar readings={applied} selected={[0, 1]} {...props} />);
    expect(size.value).toBe("14");
    // Reading 3 (still at the 5 pt default) joins: sizes now differ.
    rerender(<ReadingBatchStyleBar readings={applied} selected={[0, 1, 2]} {...props} />);
    const size2 = screen.getByLabelText("Label size in points for the selected readings") as HTMLInputElement;
    expect(size2.value).toBe("");                      // "mixed", not the stale 14
    // Same set in another order is the same selection: a fresh draft stays.
    fireEvent.change(size2, { target: { value: "8" } });
    rerender(<ReadingBatchStyleBar readings={applied} selected={[2, 0, 1]} {...props} />);
    expect((screen.getByLabelText("Label size in points for the selected readings") as HTMLInputElement).value).toBe("8");
  });

  it("None clears the selection and hides the bar", () => {
    render(<Harness initial={[{}, {}]} onApply={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Select all readings" }));
    expect(screen.getByText("2 selected")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Select no readings" }));
    expect(screen.getByText("none selected")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Use measurement default" })).toBeNull();
  });
});
