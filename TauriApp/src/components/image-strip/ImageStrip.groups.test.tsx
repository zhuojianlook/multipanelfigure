/* User-made groups must stay visible when no images are loaded. ImageStrip
   used to early-return a one-line "No images loaded" hint whenever the
   project had no images, which hid the whole Groups section with it — so
   emptying the image list made the groups vanish from the UI even though the
   store kept them. */

import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { useFigureStore } from "../../store/figureStore";
import { ImageStrip } from "./ImageStrip";

afterEach(() => {
  cleanup();
  useFigureStore.setState({ loadedImages: {}, imageGroups: [], pendingUploads: [] });
});

describe("ImageStrip with no images", () => {
  it("still shows the groups (and the empty-timeline hint)", () => {
    useFigureStore.setState({
      config: null,
      loadedImages: {},
      pendingUploads: [],
      imageGroups: [
        { id: "g1", name: "Control", imageNames: [] },
        { id: "g2", name: "Treated", imageNames: [] },
      ],
    });
    render(<ImageStrip />);
    expect(screen.getByText(/No images loaded/)).toBeTruthy();
    expect(screen.getByText("Groups (2)")).toBeTruthy();
    expect(screen.getByText("Control")).toBeTruthy();
    expect(screen.getByText("Treated")).toBeTruthy();
    // Each group can be reordered by its grip, even with no images.
    expect(screen.getAllByTestId("group-drag-handle")).toHaveLength(2);
  });

  it("with no images and no groups, still offers creating a group", () => {
    useFigureStore.setState({ config: null, loadedImages: {}, pendingUploads: [], imageGroups: [] });
    render(<ImageStrip />);
    expect(screen.getByText(/No images loaded/)).toBeTruthy();
    expect(screen.getByText("Groups (0)")).toBeTruthy();
    expect(screen.getByTitle("Create new group")).toBeTruthy();
  });
});
