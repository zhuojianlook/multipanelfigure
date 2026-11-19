/* The "Plot intensities" options dialog: which graphs are drawn and how each
   one looks. Nothing reaches the node until Apply. */

import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { ThemeProvider, createTheme } from "@mui/material";
import IntensityPlotOptionsDialog from "./IntensityPlotOptionsDialog";
import type {
  IntensityPlotCodeState, IntensityPlotConfig, IntensityPlotContext,
} from "../../utils/intensityPlotOptions";

const FULL: IntensityPlotContext = {
  groups: ["Ctrl", "IL-6"], channels: ["GFP", "RFP"], controlChannels: ["DAPI"],
  compartments: ["whole_cell", "nucleus", "cytoplasm"], mode: "cellpose", hasNorm: true, source: "run",
};
/** One group, simple mode, no control channel. */
const SPARSE: IntensityPlotContext = {
  groups: ["A"], channels: ["GFP"], controlChannels: [], compartments: ["whole_image"],
  mode: "simple", hasNorm: false, source: "run",
};
const UNKNOWN: IntensityPlotContext = {
  groups: [], channels: [], controlChannels: [], compartments: [], mode: null, hasNorm: false, source: "unknown",
};

function open(opts: {
  initial?: IntensityPlotConfig | null; context?: IntensityPlotContext; codeState?: IntensityPlotCodeState;
} = {}) {
  const onSave = vi.fn<(cfg: IntensityPlotConfig) => void>();
  const onClose = vi.fn();
  render(
    <IntensityPlotOptionsDialog open initial={opts.initial ?? null} context={opts.context ?? FULL}
      codeState={opts.codeState ?? "generated"} nodeLabel="Plot intensities" onClose={onClose} onSave={onSave} />,
  );
  const apply = () => fireEvent.click(screen.getByRole("button", { name: "Apply" }));
  const saved = () => {
    expect(onSave).toHaveBeenCalledTimes(1);
    return onSave.mock.calls[0][0];
  };
  return { onSave, onClose, apply, saved };
}
const graphBox = (name: RegExp) => screen.getByRole("button", { name });
const drawBox = (label: string) => screen.getByRole("checkbox", { name: `Draw ${label}` }) as HTMLInputElement;

afterEach(() => cleanup());

describe("IntensityPlotOptionsDialog", () => {
  it("unticking a graph and applying passes it as disabled", () => {
    const d = open();
    expect(drawBox("Fold change").checked).toBe(true);
    fireEvent.click(drawBox("Fold change"));
    expect(drawBox("Fold change").checked).toBe(false);
    d.apply();
    expect(d.saved()).toEqual({ version: 1, graphs: { channel_intensity_fc: { enabled: false } } });
  });

  it("a per-graph setting changes only that graph", () => {
    const d = open();
    fireEvent.click(graphBox(/^Per-image means/));
    expect(screen.getByRole("heading", { name: "Per-image means" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    fireEvent.click(screen.getByRole("button", { name: "Move IL-6 up" }));
    d.apply();
    expect(d.saved()).toEqual({
      version: 1,
      graphs: { channel_intensity_per_image: { chart: "box", order: { groups: ["IL-6", "Ctrl"], channels: [], compartments: [] } } },
    });
  });

  it("fold-change graphs hide the settings they do not use", () => {
    open();
    expect(screen.getByRole("button", { name: "Box" })).toBeTruthy();
    fireEvent.click(graphBox(/^Fold change/));
    expect(screen.queryByRole("button", { name: "Box" })).toBeNull();
    expect(screen.queryByRole("combobox", { name: /^Test/ })).toBeNull();
    expect(screen.queryByRole("checkbox", { name: "Log scale y axis" })).toBeNull();
    expect(screen.getByRole("combobox", { name: /^Baseline group/ })).toBeTruthy();
  });

  it("disables settings that do nothing, with the reason", () => {
    open();
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(screen.getByText("Box and violin plots show their own spread")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: /^Error bars/ }).getAttribute("aria-disabled")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Bars" }));
    const log = screen.getByRole("checkbox", { name: "Log scale y axis" }) as HTMLInputElement;
    expect(log.disabled).toBe(true);
    expect(screen.getByText("A bar starts at 0, which a log axis cannot show")).toBeTruthy();
  });

  it("apply to all copies only what each graph offers, never titles", () => {
    const d = open();
    expect(screen.getByRole("heading", { name: "Channel intensities" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    fireEvent.click(screen.getByRole("button", { name: "Right" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Title" }), { target: { value: "My title " } });
    fireEvent.click(screen.getByRole("button", { name: "Apply these settings to all graphs" }));
    expect(screen.getByRole("status").textContent).toMatch(/Copied/);
    d.apply();
    const { graphs } = d.saved();
    expect(graphs.channel_intensity).toEqual({ chart: "box", legend: "right", title: "My title" });
    for (const id of ["channel_intensity_norm", "channel_intensity_per_image", "channel_intensity_norm_per_image", "control_intensity"] as const) {
      expect(graphs[id]).toEqual({ chart: "box", legend: "right" });
    }
    expect(graphs.channel_intensity_fc).toEqual({ legend: "right" });
    expect(graphs.channel_intensity_norm_fc).toEqual({ legend: "right" });
  });

  it("unavailable graphs are disabled with their reason, and keep their stored state", () => {
    const d = open({ context: SPARSE });
    for (const label of ["Normalised intensities", "Normalised per-image means", "Control channel"]) {
      expect(drawBox(label).disabled).toBe(true);
    }
    expect(screen.getAllByText("No control channel set in the Intensity picker").length).toBeGreaterThanOrEqual(3);
    expect(drawBox("Per-image means").disabled).toBe(true);
    expect(screen.getByText("Simple mode already measures one value per image")).toBeTruthy();
    expect(drawBox("Fold change").disabled).toBe(true);
    expect(screen.getAllByText("Needs at least 2 groups").length).toBeGreaterThanOrEqual(1);
    expect(drawBox("Channel intensities").disabled).toBe(false);
    d.apply();
    expect(d.saved()).toEqual({ version: 1, graphs: {} });
  });

  it("Apply needs at least one graph the data can draw", () => {
    const d = open({ context: SPARSE });
    fireEvent.click(drawBox("Channel intensities"));
    const apply = screen.getByRole("button", { name: "Apply" }) as HTMLButtonElement;
    expect(apply.disabled).toBe(true);
    expect(screen.getByText("Tick at least one graph this data can draw.")).toBeTruthy();
    expect(d.onSave).not.toHaveBeenCalled();
  });

  it("shows the hand-edit warning only for edited code", () => {
    open({ codeState: "edited" });
    expect(screen.getByText(/R code was edited by hand/)).toBeTruthy();
    cleanup();
    open({ codeState: "legacy" });
    expect(screen.queryByText(/R code was edited by hand/)).toBeNull();
  });

  it("Cancel writes nothing", () => {
    const d = open();
    fireEvent.click(drawBox("Control channel"));
    fireEvent.click(screen.getByRole("button", { name: "Violin" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(d.onClose).toHaveBeenCalledTimes(1);
    expect(d.onSave).not.toHaveBeenCalled();
  });

  it("a click on the backdrop does not close it (the second click of a double-click lands there); Escape does", () => {
    const d = open();
    const container = document.querySelector(".MuiDialog-container") as HTMLElement;
    expect(container).toBeTruthy();
    fireEvent.mouseDown(container);
    fireEvent.click(container);
    expect(d.onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(d.onClose).toHaveBeenCalledTimes(1);
    expect(d.onSave).not.toHaveBeenCalled();
  });

  it("a number keeps its decimal point while it is being edited", () => {
    const d = open();
    const width = screen.getByRole("textbox", { name: "Panel width" }) as HTMLInputElement;
    fireEvent.change(width, { target: { value: "2.5" } });
    // Deleting the last digit leaves "2." (= 2): the field must not become "2".
    fireEvent.change(width, { target: { value: "2." } });
    expect(width.value).toBe("2.");
    fireEvent.change(width, { target: { value: "2.7" } });
    fireEvent.blur(width);
    expect(width.value).toBe("2.7");
    const font = screen.getByRole("textbox", { name: "Font size" }) as HTMLInputElement;
    fireEvent.change(font, { target: { value: "10." } });
    expect(font.value).toBe("10.");
    fireEvent.change(font, { target: { value: "10.5" } });
    fireEvent.blur(font);
    d.apply();
    expect(d.saved().graphs.channel_intensity).toEqual({ panelW: 2.7, fontSize: 10.5 });
  });

  it("bars always start at 0, so that box is ticked and disabled for them", () => {
    open({ initial: { version: 1, graphs: { channel_intensity: { yZero: false } } } });
    const zero = () => screen.getByRole("checkbox", { name: "Start the y axis at 0" }) as HTMLInputElement;
    expect(zero().disabled).toBe(true);
    expect(zero().checked).toBe(true);
    expect(screen.getByText("Bars always start at 0")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Box" }));
    expect(zero().disabled).toBe(false);
    expect(zero().checked).toBe(false);
  });

  it("starts from the stored options; reset brings one graph back to the defaults", () => {
    const initial: IntensityPlotConfig = {
      version: 1,
      graphs: { channel_intensity: { chart: "violin", title: "Kept" }, channel_intensity_fc: { enabled: false } },
    };
    const d = open({ initial });
    expect(drawBox("Fold change").checked).toBe(false);
    expect((screen.getByRole("textbox", { name: "Title" }) as HTMLInputElement).value).toBe("Kept");
    expect(screen.getByRole("button", { name: "Violin" }).getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Reset this graph" }));
    expect((screen.getByRole("textbox", { name: "Title" }) as HTMLInputElement).value).toBe("");
    d.apply();
    expect(d.saved()).toEqual({ version: 1, graphs: { channel_intensity_fc: { enabled: false } } });
  });

  it("with no data yet, lists only stored names, tagged, and says how to get the rest", () => {
    open({
      context: UNKNOWN,
      initial: { version: 1, graphs: { channel_intensity: { exclude: { groups: ["Old"], channels: [], compartments: [] } } } },
    });
    expect(screen.getByText("Run the Intensity picker once to list groups and channels.")).toBeTruthy();
    const old = screen.getByRole("checkbox", { name: "Include Old" }) as HTMLInputElement;
    expect(old.checked).toBe(false);
    const row = old.closest("div") as HTMLElement;
    expect(within(row).getByText("not in current data")).toBeTruthy();
    // Nothing is ruled out when the data is unknown.
    expect(drawBox("Fold change").disabled).toBe(false);
  });

  it("renders in both light and dark themes", () => {
    for (const mode of ["light", "dark"] as const) {
      const onSave = vi.fn();
      render(
        <ThemeProvider theme={createTheme({ palette: { mode } })}>
          <IntensityPlotOptionsDialog open initial={null} context={FULL} codeState="edited"
            onClose={vi.fn()} onSave={onSave} />
        </ThemeProvider>,
      );
      fireEvent.click(screen.getByRole("button", { name: "Apply" }));
      expect(onSave).toHaveBeenCalledWith({ version: 1, graphs: {} });
      cleanup();
    }
  });
});
