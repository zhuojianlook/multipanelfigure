/* The Curved Surface status chip floats over the bottom of the preview. */

import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ThicknessSurfaceChip } from "./ThicknessSurfaceChip";

afterEach(() => cleanup());

describe("ThicknessSurfaceChip", () => {
  it("lets clicks through to the image; only Save takes the mouse", () => {
    render(<ThicknessSurfaceChip which="bottom" kind="trace" nPts={1} onDone={vi.fn()} />);
    const chip = screen.getByTestId("thickness-surface-chip");
    // Regression: the chip used to take (and stop) every click and
    // mousedown over the bottom-centre of the image, where a bottom
    // surface often runs, so points there could not be added or grabbed.
    expect(chip.style.pointerEvents).toBe("none");
    const done = screen.getByRole("button", { name: "Save the bottom surface" });
    expect(done.textContent).toBe("Save");
    expect(done.style.pointerEvents).toBe("auto");
    expect(chip.textContent).toContain("Tracing bottom surface · 1 point —");
    expect(chip.textContent).toContain("double-click to finish");
  });

  it("Save finishes once and keeps its click / press off the preview", () => {
    const onDone = vi.fn();
    const outerClick = vi.fn();
    const outerDown = vi.fn();
    render(
      <div onClick={outerClick} onMouseDown={outerDown}>
        <ThicknessSurfaceChip which="top" kind="edit" nPts={4} message="" onDone={onDone} />
      </div>,
    );
    const done = screen.getByRole("button", { name: "Save the top surface" });
    fireEvent.mouseDown(done);
    fireEvent.click(done, { detail: 1 });
    expect(onDone).toHaveBeenCalledTimes(1);
    // The second click of a double-click is not a second Save.
    fireEvent.click(done, { detail: 2 });
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(outerClick).not.toHaveBeenCalled();
    expect(outerDown).not.toHaveBeenCalled();
    expect(screen.getByTestId("thickness-surface-chip").textContent).toContain("Editing top surface · 4 points");
  });

  it("a transient message replaces the hint", () => {
    render(<ThicknessSurfaceChip which="top" kind="trace" nPts={1} message="A surface needs at least 2 points (Esc cancels)" onDone={vi.fn()} />);
    const chip = screen.getByTestId("thickness-surface-chip");
    expect(chip.textContent).toContain("A surface needs at least 2 points");
    expect(chip.textContent).not.toContain("double-click to finish");
  });
});
