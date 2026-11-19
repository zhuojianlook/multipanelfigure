/* chooseDocument(): which document an uploads-only Analysis workflow is saved
   in — the open documents (current one pre-selected) or "New document…". */

import { describe, it, expect, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ChooseDocumentHost, chooseDocument, type ChooseDocumentOpts, type ChooseDocumentResult } from "./ChooseDocumentDialog";

const OPTS = {
  title: "Save this analysis in…",
  body: "Choose a document",
  options: [
    { id: "doc_a", label: "a", detail: "a.mpf" },
    { id: "doc_b", label: "b", detail: "not saved yet" },
  ],
  newLabel: "New document…",
  confirmLabel: "Save here",
};

afterEach(() => cleanup());

function ask(extra: Partial<ChooseDocumentOpts> = {}) {
  let p!: Promise<ChooseDocumentResult>;
  act(() => { p = chooseDocument({ ...OPTS, ...extra }); });
  return p;
}
const radio = (name: string) => screen.getByRole("radio", { name: new RegExp(`^${name}`) }) as HTMLInputElement;

describe("ChooseDocumentDialog", () => {
  it("resolves null (keep what you had) when no host is mounted", async () => {
    await expect(chooseDocument(OPTS)).resolves.toBeNull();
  });

  it("opens on the current choice and resolves the picked document", async () => {
    render(<ChooseDocumentHost />);
    const p = ask({ selectedId: "doc_b" });
    expect(screen.getByText("Save this analysis in…")).toBeTruthy();
    expect(radio("b").checked).toBe(true);
    expect(radio("a").checked).toBe(false);
    fireEvent.click(radio("a"));
    fireEvent.click(screen.getByRole("button", { name: "Save here" }));
    await expect(p).resolves.toEqual({ kind: "doc", id: "doc_a" });
    expect(screen.queryByText("Save this analysis in…")).toBeNull();
  });

  it("New document… resolves 'new'; Cancel resolves null", async () => {
    render(<ChooseDocumentHost />);
    const p1 = ask();
    expect(radio("a").checked).toBe(true);                   // no current choice: the first
    fireEvent.click(radio("New document"));
    fireEvent.click(screen.getByRole("button", { name: "Save here" }));
    await expect(p1).resolves.toEqual({ kind: "new" });
    const p2 = ask({ selectedId: "doc_a" });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await expect(p2).resolves.toBeNull();
  });

  it("hides New document… when asked to; queued requests are shown one at a time", async () => {
    render(<ChooseDocumentHost />);
    const p1 = ask({ newLabel: null, selectedId: "doc_b" });
    const p2 = ask({ title: "Second", selectedId: "doc_a" });
    expect(screen.queryByRole("radio", { name: /New document/ })).toBeNull();
    expect(screen.queryByText("Second")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Save here" }));
    await expect(p1).resolves.toEqual({ kind: "doc", id: "doc_b" });
    expect(screen.getByText("Second")).toBeTruthy();
    expect(radio("a").checked).toBe(true);                   // its own pre-selection
    fireEvent.click(screen.getByRole("button", { name: "Save here" }));
    await expect(p2).resolves.toEqual({ kind: "doc", id: "doc_a" });
  });
});
