import { describe, it, expect } from "vitest";
import { rRerunPng, rRerenderError, rPlotNotRerenderable } from "./rRerender";

describe("rPlotNotRerenderable", () => {
  it("is true only for reasons that re-running the same code cannot change", () => {
    for (const code of ["not_ggplot", "not_mpfig_plot", "no_such_plot"]) {
      expect(rPlotNotRerenderable({ success: true, plots: [], override_code: code })).toBe(true);
    }
    for (const code of ["script_error", "mismatch", "render_failed", null, undefined]) {
      expect(rPlotNotRerenderable({ success: true, plots: [], override_code: code })).toBe(false);
    }
    // A failed run (R missing, timeout) says nothing about the plot.
    expect(rPlotNotRerenderable({ success: false, override_code: "not_ggplot" })).toBe(false);
    expect(rPlotNotRerenderable(undefined)).toBe(false);
  });
});

describe("rRerunPng", () => {
  it("takes the override image when only the override was asked for", () => {
    expect(rRerunPng({ success: true, plots: ["OV"] }, { override: true, plotIndex: 3 })).toBe("OV");
  });

  it("takes the item's own plot from a full run", () => {
    expect(rRerunPng({ success: true, plots: ["p0", "p1", "p2"] }, { override: false, plotIndex: 2 })).toBe("p2");
    expect(rRerunPng({ success: true, plots: ["p0", "p1"] }, { override: false })).toBe("p0");
  });

  it("never falls back to another plot", () => {
    // The script now makes fewer plots: plot 0 is a DIFFERENT plot.
    expect(rRerunPng({ success: true, plots: ["p0", "p1"] }, { override: false, plotIndex: 2 })).toBeUndefined();
    // The sidecar could not re-render this plot (override_error set, no image).
    expect(rRerunPng({ success: true, plots: [], override_error: "not a ggplot" }, { override: true })).toBeUndefined();
    expect(rRerunPng({ success: false, plots: ["p0"] }, { override: false })).toBeUndefined();
    expect(rRerunPng(undefined, { override: true })).toBeUndefined();
  });
});

describe("rRerenderError", () => {
  it("prefers the sidecar's reason", () => {
    expect(rRerenderError({ override_error: "Plot 2 (b.png) is not a ggplot", stderr: "Loading ggplot2" })).toBe(
      "Plot 2 (b.png) is not a ggplot",
    );
  });

  it("falls back to the end of stderr, then to the default", () => {
    const stderr = "x".repeat(1000) + "Error: boom";
    expect(rRerenderError({ stderr })).toBe(stderr.slice(-400));
    expect(rRerenderError({ stderr: "  " }, "nothing")).toBe("nothing");
    expect(rRerenderError(null)).toBe("Re-render failed");
  });
});
