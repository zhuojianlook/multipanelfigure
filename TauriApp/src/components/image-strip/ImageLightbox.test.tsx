import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, cleanup, within } from "@testing-library/react";
import { api } from "../../api/client";
import { useFigureStore } from "../../store/figureStore";
import { ImageLightbox } from "./ImageLightbox";
import { ImageStrip, getSelectedImageName, setSelectedImageName } from "./ImageStrip";
import { tileClickMemory } from "./lightboxLogic";

/* DOM-level checks for "double-click a media tile to view it large".
   api.getImagePreview is spied on the real client singleton, so nothing
   here ever reaches a sidecar. */

const NAME = "Screenshot 2026-03-24 at 11.59.37\u202FAM.png"; // real fixture's U+202F
/** Testing Library's default text normalizer collapses every \s run (U+202F
 *  included) to a plain space, so DOM queries use this form. Data and
 *  assertions keep the real name, so the component still handles U+202F. */
const Q = NAME.replace(/\s+/g, " ");
/** Accessible-name matcher with the same whitespace folding (role queries
 *  compare names verbatim, without the normalizer). */
const named = (want: string) => (n: string) => n.replace(/\s+/g, " ") === want;
const THUMB = "VEhVTUI=";
const LARGE = "TEFSR0U=";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const imgSrc = () => within(screen.getByRole("dialog")).getByAltText(Q).getAttribute("src");

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ImageLightbox", () => {
  it("shows the thumbnail immediately, then swaps in the large preview", async () => {
    const d = deferred<{ thumbnail: string }>();
    const spy = vi.spyOn(api, "getImagePreview").mockReturnValue(d.promise);
    render(<ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);

    expect(imgSrc()).toBe(`data:image/png;base64,${THUMB}`);
    expect(screen.getByLabelText("Loading full-size preview")).toBeTruthy();
    // jsdom: 1024×768 at DPR 1 → the 92vw × 88vh image box's longer side,
    // max(942.08, 675.84) → 943 (see lightboxMaxSide).
    expect(spy).toHaveBeenCalledWith(NAME, 943);

    await act(async () => { d.resolve({ thumbnail: LARGE }); });
    expect(imgSrc()).toBe(`data:image/png;base64,${LARGE}`);
    expect(screen.queryByLabelText("Loading full-size preview")).toBeNull();
  });

  it("keeps the thumbnail and shows a note when the preview request fails", async () => {
    const d = deferred<{ thumbnail: string }>();
    vi.spyOn(api, "getImagePreview").mockReturnValue(d.promise);
    render(<ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);

    await act(async () => { d.reject(new Error("API error: not found")); });
    expect(imgSrc()).toBe(`data:image/png;base64,${THUMB}`);
    expect(screen.getByText(/full-size preview unavailable/)).toBeTruthy();
  });

  it("does not request anything while closed", () => {
    const spy = vi.spyOn(api, "getImagePreview");
    render(<ImageLightbox open={false} name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);
    expect(spy).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes on the X button, Esc and the backdrop — but not on the image or caption", () => {
    vi.spyOn(api, "getImagePreview").mockReturnValue(new Promise(() => {}));
    const onClose = vi.fn();
    render(<ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={onClose} />);
    const dialog = screen.getByRole("dialog");

    fireEvent.click(within(dialog).getByAltText(Q));
    fireEvent.click(within(dialog).getByText(Q));
    expect(onClose).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);

    fireEvent.click(dialog); // the dark surround itself
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it("ignores a response that lands after it closed", async () => {
    const d = deferred<{ thumbnail: string }>();
    vi.spyOn(api, "getImagePreview").mockReturnValue(d.promise);
    const { rerender } = render(<ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);
    rerender(<ImageLightbox open={false} name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);
    // Still mounted for the fade-out; the late payload must not be swapped in.
    await act(async () => { d.resolve({ thumbnail: LARGE }); });
    const stillThere = screen.getByRole("dialog", { hidden: true });
    expect(within(stillThere).getByAltText(Q).getAttribute("src")).toBe(`data:image/png;base64,${THUMB}`);
  });

  it("ignores a response that lands after unmount", async () => {
    const d = deferred<{ thumbnail: string }>();
    vi.spyOn(api, "getImagePreview").mockReturnValue(d.promise);
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const { unmount } = render(<ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={() => {}} />);
    unmount();
    await act(async () => { d.resolve({ thumbnail: LARGE }); });
    expect(err).not.toHaveBeenCalled();
  });

  it("does not let clicks / double-clicks inside it bubble to the React parent", () => {
    vi.spyOn(api, "getImagePreview").mockReturnValue(new Promise(() => {}));
    const parentClick = vi.fn();
    const parentDbl = vi.fn();
    const parentDown = vi.fn();
    render(
      <div onClick={parentClick} onDoubleClick={parentDbl} onMouseDown={parentDown}>
        <ImageLightbox open name={NAME} thumbnailB64={THUMB} onClose={() => {}} />
      </div>,
    );
    const dialog = screen.getByRole("dialog");
    const img = within(dialog).getByAltText(Q);
    fireEvent.mouseDown(img);
    fireEvent.click(img);
    fireEvent.doubleClick(img);
    fireEvent.click(dialog);
    fireEvent.doubleClick(dialog);
    expect(parentClick).not.toHaveBeenCalled();
    expect(parentDbl).not.toHaveBeenCalled();
    expect(parentDown).not.toHaveBeenCalled();
  });
});

describe("ImageTile — click / double-click / drag", () => {
  beforeEach(() => {
    setSelectedImageName(null);
    tileClickMemory.snapshot = null;
    vi.spyOn(api, "getImagePreview").mockResolvedValue({ thumbnail: LARGE });
    useFigureStore.setState({
      config: null,
      loadedImages: {
        [NAME]: { name: NAME, thumbnailB64: THUMB },
        "b.png": { name: "b.png", thumbnailB64: THUMB },
      },
    });
  });
  afterEach(() => {
    cleanup(); // unmount first so the resets below don't re-render anything
    setSelectedImageName(null);
    tileClickMemory.snapshot = null;
    useFigureStore.setState({ loadedImages: {} });
  });

  // The strip tile (not the lightbox's copy of the same image, once open).
  const tile = () =>
    screen.getAllByAltText(Q).map((i) => i.closest('[draggable="true"]')).find(Boolean) as HTMLElement;

  it("a single click still selects immediately (no timers)", () => {
    render(<ImageStrip />);
    fireEvent.click(tile(), { detail: 1 });
    expect(getSelectedImageName()).toBe(NAME);
    fireEvent.click(tile(), { detail: 1 });
    expect(getSelectedImageName()).toBeNull();
  });

  for (const before of [null, NAME, "b.png"]) {
    it(`a double-click opens the lightbox and leaves selection=${String(before)} unchanged`, () => {
      render(<ImageStrip />);
      act(() => setSelectedImageName(before));
      fireEvent.click(tile(), { detail: 1 });
      fireEvent.click(tile(), { detail: 2 });
      fireEvent.doubleClick(tile(), { detail: 2 });
      expect(getSelectedImageName()).toBe(before);
      expect(screen.getByRole("dialog", { name: named(`Image preview: ${Q}`) })).toBeTruthy();
    });
  }

  it("clicking inside the open lightbox never toggles the tile's selection", () => {
    render(<ImageStrip />);
    fireEvent.doubleClick(tile());
    const dialog = screen.getByRole("dialog");
    fireEvent.click(within(dialog).getByAltText(Q), { detail: 1 });
    fireEvent.doubleClick(within(dialog).getByAltText(Q));
    fireEvent.click(dialog, { detail: 1 }); // backdrop → closes
    expect(getSelectedImageName()).toBeNull();
  });

  it("double-clicking the remove (×) button does not open the lightbox", () => {
    render(<ImageStrip />);
    fireEvent.doubleClick(screen.getByRole("button", { name: named(`Remove image ${Q}`) }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("drag start still carries the image name", () => {
    render(<ImageStrip />);
    const setData = vi.fn();
    fireEvent.dragStart(tile(), { dataTransfer: { setData, effectAllowed: "" } });
    expect(setData).toHaveBeenCalledWith("application/x-image-name", NAME);
  });
});
