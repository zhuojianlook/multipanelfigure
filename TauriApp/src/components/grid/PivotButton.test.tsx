/* The planner's Pivot button, where the row and column headers meet. */

import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { PivotButton } from "./PivotButton";

afterEach(() => cleanup());

describe("PivotButton", () => {
  it("reads 'Pivot' and names what it does", () => {
    render(<PivotButton rows={4} cols={3} onPivot={vi.fn()} />);
    const btn = screen.getByRole("button", { name: "Pivot rows and columns (4×3 → 3×4)" });
    expect(btn.textContent).toBe("Pivot");
  });

  it("a click pivots", () => {
    const onPivot = vi.fn();
    render(<PivotButton rows={2} cols={3} onPivot={onPivot} />);
    fireEvent.click(screen.getByRole("button", { name: /^Pivot rows and columns/ }));
    expect(onPivot).toHaveBeenCalledTimes(1);
  });

  it("disabled (busy, or a header being resized): a click does nothing", () => {
    const onPivot = vi.fn();
    render(<PivotButton rows={2} cols={3} disabled onPivot={onPivot} />);
    const btn = screen.getByRole("button", { name: /^Pivot rows and columns/ });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(btn);
    expect(onPivot).not.toHaveBeenCalled();
  });
});
