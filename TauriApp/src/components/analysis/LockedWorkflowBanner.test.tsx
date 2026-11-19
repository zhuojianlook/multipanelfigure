/* The banner over a locked (greyed-out) Analysis workflow: it names the
   closed documents and offers Open missing / Locate… / Remove their sources. */

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { LockedWorkflowBanner } from "./LockedWorkflowBanner";
import type { MissingDoc } from "../../store/analysisWorkspaceStore";

const B: MissingDoc = { uid: "du_b", label: "b.mpf", locatable: true };
const C: MissingDoc = { uid: "du_c", label: "c.mpf", locatable: true };
const U: MissingDoc = { uid: "du_u", label: "Untitled_1 (never saved)", locatable: false };

function setup(missing: MissingDoc[], busy = false) {
  const cbs = { onOpenMissing: vi.fn(), onLocate: vi.fn(), onRemoveSources: vi.fn() };
  render(<LockedWorkflowBanner missing={missing} busy={busy} {...cbs} />);
  return cbs;
}

afterEach(() => cleanup());

describe("LockedWorkflowBanner", () => {
  it("names the closed documents and opens them", () => {
    const cbs = setup([B, C]);
    const banner = screen.getByTestId("locked-workflow-banner");
    expect(banner.textContent).toContain("This workflow also uses b.mpf, c.mpf. Open them to edit and re-run.");
    fireEvent.click(screen.getByRole("button", { name: "Open missing" }));
    expect(cbs.onOpenMissing).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Remove their sources" }));
    expect(cbs.onRemoveSources).toHaveBeenCalledTimes(1);
  });

  it("Locate… asks for the one missing file directly, or offers a choice of several", () => {
    let cbs = setup([B]);
    expect(screen.getByTestId("locked-workflow-banner").textContent).toContain("Open it to edit");
    fireEvent.click(screen.getByRole("button", { name: "Locate…" }));
    expect(cbs.onLocate).toHaveBeenCalledWith("du_b");
    cleanup();
    cbs = setup([B, C]);
    fireEvent.click(screen.getByRole("button", { name: "Locate…" }));
    expect(cbs.onLocate).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("menuitem", { name: "Locate c.mpf…" }));
    expect(cbs.onLocate).toHaveBeenCalledWith("du_c");
  });

  it("nothing to look for (never saved) or busy: Open missing is off", () => {
    setup([U]);
    const btn = (name: string) => screen.getByRole("button", { name }) as HTMLButtonElement;
    expect(btn("Open missing").disabled).toBe(true);
    expect(btn("Remove their sources").disabled).toBe(false);
    cleanup();
    setup([B], true);
    expect(btn("Open missing").disabled).toBe(true);
    expect(btn("Locate…").disabled).toBe(true);
  });
});
