import { describe, it, expect } from "vitest";
import type { FluorIntensityConfig } from "../components/dialogs/IntensityPickerDialog";
import { emptyFluorConfig } from "../components/dialogs/IntensityPickerDialog";
import {
  DEFAULT_INTENSITY_PLOT_CODE, DEFAULT_INTENSITY_PLOT_HASH, INTENSITY_GRAPH_IDS, LEGACY_HASHES,
  applyIntensityGraphToAll, applyIntensityPlotOptions, defaultIntensityPlotConfig, defaultIntensityPlotFields,
  intensityPlotEditable, restoreIntensityPlotBackup, deriveIntensityPlotContext, generateIntensityPlotR,
  graphHasSetting, hashCode, intensityGraphUnavailableReason, intensityPlotState, intensityPlotSummary,
  isFutureIntensityPlotConfig, normalizeIntensityPlotConfig, resetIntensityGraph, resolveGraph, resolveOrder,
  rStr, settingInactiveReason, updateIntensityGraph, withoutIntensityPlot, yAxisStartsAtZero,
  type IntensityGraphId, type IntensityPlotConfig,
} from "./intensityPlotOptions";
import { INTENSITY_PLOT_RUNTIME_R } from "./intensityPlotRuntime";
import { rInputsPrelude } from "./rPrelude";
import type { NodeData } from "./analysisTypes";

/** The patterns AnalysisNodeGraph's parseDeclaredOutputs reads node code with. */
function declared(code: string): { plots: string[]; other: string[] } {
  const names = (re: RegExp) => [...code.matchAll(re)].map((m) => m[1].trim().replace(/\.(png|pdf|svg|csv)$/i, ""));
  return {
    plots: names(/mpfig_plot\s*\(\s*["']([^"']+?)\.?(?:png|pdf|svg)?["']/g),
    other: [
      ...names(/mpfig_data\s*\(\s*(?:[^"']*?,\s*)*?(?:name\s*=\s*)?["']([^"']+)["']/g),
      ...names(/mpfig_image\s*\(\s*(?:[^"']*?,\s*)*?(?:name\s*=\s*)?["']([^"']+)["']/g),
    ],
  };
}
const fileOrder = (ids: readonly string[]) => [...ids].sort((a, b) => {
  const x = `${a}.png`, y = `${b}.png`;
  return x < y ? -1 : x > y ? 1 : 0;
});
const cfgWith = (graphs: IntensityPlotConfig["graphs"]): IntensityPlotConfig => ({ version: 1, graphs });
const onlyGraphs = (...ids: IntensityGraphId[]) => cfgWith(Object.fromEntries(
  INTENSITY_GRAPH_IDS.filter((id) => !ids.includes(id)).map((id) => [id, { enabled: false }])));
/** The `.ipo` entry generated for one graph. */
const optionLine = (code: string, id: string) => {
  const lines = code.split("\n");
  const block = lines.slice(lines.indexOf(".ipo <- list("), lines.indexOf(")"));
  return block.find((l) => l.startsWith(`  ${id} = list(`)) ?? "";
};

describe("rStr", () => {
  it("escapes quotes, backslashes and whitespace", () => {
    expect(rStr('a"b')).toBe('"a\\"b"');
    expect(rStr("a'b")).toBe('"a\\\'b"');
    expect(rStr("a\\b")).toBe('"a\\\\b"');
    expect(rStr("a\nb\tc")).toBe('"a\\nb\\tc"');
  });
  it("drops control characters and lone surrogates", () => {
    expect(rStr("a\u0000b\rc\u0007d\u007fe")).toBe('"abcde"');
    expect(rStr("x\ud83dy\ude00z")).toBe('"xyz"');
  });
  it("writes non-ASCII by code point", () => {
    expect(rStr("µM")).toBe('"\\U{b5}M"');
    expect(rStr("\u{1f600}")).toBe('"\\U{1f600}"');
  });
  it("is always printable ASCII", () => {
    const nasty = 'q"\'\\\n\t\u0000\rµ—\u{1f600}\ud83d end';
    expect(rStr(nasty)).toMatch(/^[\x20-\x7e]*$/);
  });
  it("breaks up the harness helper names", () => {
    expect(rStr("mpfig_plot")).toBe('"mpfig\\U{5f}plot"');
  });
});

describe("generateIntensityPlotR", () => {
  it("declares exactly the enabled graphs, in file-name order", () => {
    expect(declared(DEFAULT_INTENSITY_PLOT_CODE)).toEqual({ plots: fileOrder(INTENSITY_GRAPH_IDS), other: [] });
    const two = generateIntensityPlotR(onlyGraphs("control_intensity", "channel_intensity_norm"));
    expect(declared(two).plots).toEqual(["channel_intensity_norm", "control_intensity"]);
  });
  it("sorts by file name, where '.' comes before '_'", () => {
    expect(declared(DEFAULT_INTENSITY_PLOT_CODE).plots.slice(0, 3))
      .toEqual(["channel_intensity", "channel_intensity_fc", "channel_intensity_norm"]);
  });
  it("runs each graph's preparation right before its own plot, once", () => {
    const calls = [...DEFAULT_INTENSITY_PLOT_CODE.matchAll(/^\.g <- \.ipo_prepare\("(\w+)"\); .*mpfig_plot\("(\w+)\.png"/gm)];
    expect(calls.map((m) => m[1])).toEqual(fileOrder(INTENSITY_GRAPH_IDS));
    expect(calls.every((m) => m[1] === m[2])).toBe(true);
  });
  it("puts the warning panel at its sorted slot, only when a fold-change graph is on", () => {
    const lines = DEFAULT_INTENSITY_PLOT_CODE.split("\n");
    const at = (needle: string) => lines.findIndex((l) => l.includes(needle));
    const warn = at('"warning.png"');
    expect(warn).toBeGreaterThan(at('mpfig_plot("channel_intensity_per_image.png"'));
    expect(warn).toBeLessThan(at('mpfig_plot("control_intensity.png"'));
    expect(generateIntensityPlotR(onlyGraphs("channel_intensity"))).not.toContain("warning.png");
    expect(generateIntensityPlotR(onlyGraphs("channel_intensity_norm_fc"))).toContain('"warning.png"');
  });
  it("still produces valid code with nothing enabled", () => {
    const code = generateIntensityPlotR(onlyGraphs());
    expect(code).toContain(".ipo <- list()");
    expect(code).toContain(".ipo_empty()");
    expect(declared(code)).toEqual({ plots: [], other: [] });
  });
  it("user text cannot declare an output", () => {
    for (const title of ["mpfig_plot('x')", 'mpfig_plot("x.png")', "mpfig_plot(", "mpfig_data(x, ", "mpfig_image(a, "]) {
      const code = generateIntensityPlotR(cfgWith({
        channel_intensity: { title, yLabel: title, refGroup: title,
          order: { groups: [title], channels: [], compartments: [] } },
      }));
      expect(declared(code)).toEqual({ plots: fileOrder(INTENSITY_GRAPH_IDS), other: [] });
    }
  });
  it("is ASCII apart from the two header comment lines", () => {
    expect(INTENSITY_PLOT_RUNTIME_R).toMatch(/^[\x00-\x7f]*$/);
    const code = generateIntensityPlotR(cfgWith({ channel_intensity: { title: "µ — \u{1f600}" } }));
    const lines = code.split("\n");
    expect(lines[0]).toBe("# @name: Plot intensities (groups × channels × compartments)");
    expect(lines[1]).toMatch(/^# @mpf-intensity-plot gen=1 /);
    expect(lines.slice(2).join("\n")).toMatch(/^[\x00-\x7f]*$/);
  });
  it("leaves disabled graphs out of the options", () => {
    const code = generateIntensityPlotR(onlyGraphs("channel_intensity"));
    expect(optionLine(code, "channel_intensity")).not.toBe("");
    expect(optionLine(code, "control_intensity")).toBe("");
  });
  it("emits only the settings a fold-change graph has", () => {
    const line = optionLine(generateIntensityPlotR(cfgWith({
      channel_intensity_fc: { chart: "box", test: "welch", refGroup: "B" },
    })), "channel_intensity_fc");
    expect(line).toContain('ref_group = "B"');
    for (const hidden of ["chart =", "error =", "points =", "test =", "compare =", "sig =", "hide_ns =", "y_zero =", "y_log ="]) {
      expect(line).not.toContain(hidden);
    }
  });
  it("applies the option interactions", () => {
    const gen = (o: NonNullable<IntensityPlotConfig["graphs"]["channel_intensity"]>) =>
      optionLine(generateIntensityPlotR(cfgWith({ channel_intensity: o })), "channel_intensity");
    expect(gen({ chart: "bar", yLog: true })).toContain("y_log = FALSE");
    const logBox = gen({ chart: "box", yLog: true });
    expect(logBox).toContain("y_log = TRUE");
    expect(logBox).toContain("y_zero = FALSE");
    expect(gen({ chart: "violin", error: "sem" })).toContain('error = "none"');
    expect(gen({ chart: "mean", error: "sem" })).toContain('error = "sem"');
  });
  it("never trusts a stored enum value", () => {
    const code = generateIntensityPlotR(cfgWith({
      channel_intensity: { chart: 'bar"); system("x' as never, palette: "nope" as never },
    }));
    expect(code).not.toContain("system(");
    expect(optionLine(code, "channel_intensity")).toContain('chart = "bar"');
  });
  it("reads names from the table's text, not read.csv's converted values", () => {
    // The dialog lists names as the CSV spells them ("0.10", "01", "T");
    // the runtime re-reads the prelude's text with every column as text.
    expect(INTENSITY_PLOT_RUNTIME_R).toContain(".mpfig_csv_text[[1]]");
    expect(INTENSITY_PLOT_RUNTIME_R).toContain('read.csv(text = txt, colClasses = "character"');
    expect(rInputsPrelude([{ label: "t", csv: "group\n0.10\n" }])).toContain(".mpfig_csv_text[[length(.mpfig_csv_text) + 1L]] <<- s");
  });
  it("is stable, and the exported default matches its hash", () => {
    expect(generateIntensityPlotR(defaultIntensityPlotConfig())).toBe(DEFAULT_INTENSITY_PLOT_CODE);
    expect(hashCode(DEFAULT_INTENSITY_PLOT_CODE)).toBe(DEFAULT_INTENSITY_PLOT_HASH);
  });
});

describe("normalizeIntensityPlotConfig", () => {
  it("rejects what is not a config", () => {
    for (const junk of [null, undefined, 3, "x", []]) expect(normalizeIntensityPlotConfig(junk)).toBeNull();
    expect(normalizeIntensityPlotConfig({})).toEqual(defaultIntensityPlotConfig());
    expect(normalizeIntensityPlotConfig({ graphs: "nope" })).toEqual(defaultIntensityPlotConfig());
  });
  it("leaves a newer version alone", () => {
    const future = { version: 2, graphs: { channel_intensity: { chart: "ridgeline" } } };
    expect(isFutureIntensityPlotConfig(future)).toBe(true);
    expect(normalizeIntensityPlotConfig(future)).toBeNull();
    expect(future.graphs.channel_intensity.chart).toBe("ridgeline");
  });
  it("drops unknown graphs, unknown keys, bad values and defaults", () => {
    expect(normalizeIntensityPlotConfig({
      version: 1,
      graphs: {
        nonsense: { chart: "box" },
        channel_intensity: {
          chart: "pie", error: "sem", legend: 5, xAngle: 31, wat: 1, points: "no",
          test: "tukey", enabled: true, title: "   ", panelW: "2",
        },
      },
    })).toEqual({ version: 1, graphs: { channel_intensity: { error: "sem" } } });
  });
  it("clamps numbers and de-duplicates lists", () => {
    const n = normalizeIntensityPlotConfig({
      graphs: { channel_intensity: {
        fontSize: 99, panelW: 0.01, panelH: 500,
        exclude: { groups: ["a", "a", 3, ""], channels: "x", compartments: ["nucleus", "bogus"] },
      } },
    })!;
    expect(n.graphs.channel_intensity).toEqual({
      fontSize: 24, panelW: 0.5, panelH: 12,
      exclude: { groups: ["a"], channels: [], compartments: ["nucleus"] },
    });
  });
  it("drops yLog on a bar chart but keeps it on others", () => {
    expect(normalizeIntensityPlotConfig({ graphs: { channel_intensity: { yLog: true } } })!.graphs).toEqual({});
    expect(normalizeIntensityPlotConfig({ graphs: { channel_intensity: { yLog: true, chart: "box" } } })!
      .graphs.channel_intensity).toEqual({ yLog: true, chart: "box" });
  });
  it("is idempotent", () => {
    const once = normalizeIntensityPlotConfig({
      graphs: {
        control_intensity: { enabled: false, title: " T ", chart: "violin", refGroup: "A" },
        channel_intensity_fc: { order: { groups: ["b", "a"], channels: [], compartments: [] }, xAngle: 90 },
      },
    });
    expect(normalizeIntensityPlotConfig(JSON.parse(JSON.stringify(once)))).toEqual(once);
    expect(once!.graphs.control_intensity!.title).toBe("T");
  });
});

describe("resolveGraph and resolveOrder", () => {
  it("fills in defaults and does not share arrays between calls", () => {
    const cfg = cfgWith({ channel_intensity: { chart: "box", enabled: false } });
    const r = resolveGraph(cfg, "channel_intensity");
    expect(r).toMatchObject({ chart: "box", enabled: false, error: "sd", legend: "bottom" });
    expect(resolveGraph(cfg, "control_intensity").enabled).toBe(true);
    r.exclude.groups.push("x");
    expect(resolveGraph(cfg, "channel_intensity").exclude.groups).toEqual([]);
  });
  it("orders stored names first and appends new ones, minus exclusions", () => {
    expect(resolveOrder(["a", "b", "c"], ["c", "gone", "a"])).toEqual(["c", "a", "b"]);
    expect(resolveOrder(["a", "b", "c", "new"], ["c", "a"], ["a"])).toEqual(["c", "b", "new"]);
    expect(resolveOrder(["a", "a"], [])).toEqual(["a"]);
  });
});

describe("editing helpers", () => {
  it("apply to all copies only what each graph type offers", () => {
    const cfg = cfgWith({
      channel_intensity: { chart: "box", test: "welch", palette: "okabe", title: "Mine", yLabel: "Y", enabled: false },
      control_intensity: { enabled: false, title: "Keep" },
    });
    const out = applyIntensityGraphToAll(cfg, "channel_intensity");
    expect(out.graphs.channel_intensity_norm).toEqual({ chart: "box", test: "welch", palette: "okabe" });
    expect(out.graphs.channel_intensity_fc).toEqual({ palette: "okabe" });
    expect(out.graphs.control_intensity).toEqual({ enabled: false, title: "Keep", chart: "box", test: "welch", palette: "okabe" });
    expect(graphHasSetting("channel_intensity_fc", "chart")).toBe(false);
  });
  it("apply to all keeps channel lists across the control / analysis split", () => {
    const lists = (channels: string[], groups: string[] = []) => ({ groups, channels, compartments: [] });
    const cfg = cfgWith({
      channel_intensity: { exclude: lists(["mCherry"], ["Old"]), order: lists(["GFP", "RFP"]) },
      channel_intensity_norm: { exclude: lists(["mCherry"]) },
      control_intensity: { palette: "dark", exclude: lists([], ["Ctrl"]), order: lists(["DAPI"]) },
    });
    // From the control graph: groups and the rest copy, analysis channels stay.
    const fromControl = applyIntensityGraphToAll(cfg, "control_intensity");
    expect(fromControl.graphs.channel_intensity).toEqual({
      palette: "dark", exclude: lists(["mCherry"], ["Ctrl"]), order: lists(["GFP", "RFP"]),
    });
    expect(fromControl.graphs.channel_intensity_norm?.exclude).toEqual(lists(["mCherry"], ["Ctrl"]));
    expect(fromControl.graphs.channel_intensity_fc?.exclude).toEqual(lists([], ["Ctrl"]));
    // To the control graph: its own channel list stays.
    const toControl = applyIntensityGraphToAll(cfg, "channel_intensity");
    expect(toControl.graphs.control_intensity).toMatchObject({ exclude: lists([], ["Old"]), order: lists(["DAPI"]) });
    // Between analysis graphs, channels copy as before.
    expect(toControl.graphs.channel_intensity_norm).toEqual({ exclude: lists(["mCherry"], ["Old"]), order: lists(["GFP", "RFP"]) });
  });
  it("apply to all from a fold-change graph leaves chart settings alone", () => {
    const cfg = cfgWith({
      channel_intensity_fc: { palette: "dark" },
      channel_intensity: { chart: "violin" },
    });
    expect(applyIntensityGraphToAll(cfg, "channel_intensity_fc").graphs.channel_intensity)
      .toEqual({ chart: "violin", palette: "dark" });
  });
  it("update and reset keep the config canonical", () => {
    let c = updateIntensityGraph(defaultIntensityPlotConfig(), "channel_intensity", { chart: "box", enabled: false });
    expect(c.graphs.channel_intensity).toEqual({ chart: "box", enabled: false });
    c = updateIntensityGraph(c, "channel_intensity", { chart: "bar" });
    expect(c.graphs.channel_intensity).toEqual({ enabled: false });
    c = updateIntensityGraph(c, "channel_intensity", { title: "x" });
    expect(resetIntensityGraph(c, "channel_intensity").graphs.channel_intensity).toEqual({ enabled: false });
    expect(resetIntensityGraph(cfgWith({ control_intensity: { chart: "box" } }), "control_intensity").graphs).toEqual({});
  });
  it("says why a setting has no effect", () => {
    const o = (over: object) => ({ ...resolveGraph(defaultIntensityPlotConfig(), "channel_intensity"), ...over });
    expect(settingInactiveReason(o({ chart: "box" }), "error")).toBeTruthy();
    expect(settingInactiveReason(o({}), "error")).toBeNull();
    expect(settingInactiveReason(o({ test: "none" }), "sig")).toBeTruthy();
    expect(settingInactiveReason(o({}), "yLog")).toBeTruthy();
    expect(settingInactiveReason(o({ chart: "box", yLog: true }), "yZero")).toBeTruthy();
    expect(settingInactiveReason(o({ chart: "box" }), "yZero")).toBeNull();
  });
  it("bars always start at 0, whatever the stored setting", () => {
    const o = (over: object) => ({ ...resolveGraph(defaultIntensityPlotConfig(), "channel_intensity"), ...over });
    expect(settingInactiveReason(o({ yZero: false }), "yZero")).toBe("Bars always start at 0");
    expect(yAxisStartsAtZero(o({ yZero: false }))).toBe(true);
    expect(yAxisStartsAtZero(o({ chart: "box", yZero: false }))).toBe(false);
    expect(yAxisStartsAtZero(o({ chart: "mean", yLog: true }))).toBe(false);
    const line = (over: object) =>
      optionLine(generateIntensityPlotR(cfgWith({ channel_intensity: over })), "channel_intensity");
    expect(line({ yZero: false })).toContain("y_zero = TRUE");
    expect(line({ yZero: false, chart: "violin" })).toContain("y_zero = FALSE");
  });
  it("summarises the configuration", () => {
    expect(intensityPlotSummary(defaultIntensityPlotConfig())).toBe("7 of 7 graphs · bars + SD · Tukey");
    expect(intensityPlotSummary(onlyGraphs("channel_intensity", "channel_intensity_fc"))).toBe("2 of 7 graphs · bars + SD · Tukey");
    expect(intensityPlotSummary(onlyGraphs("channel_intensity_fc"))).toBe("1 of 7 graphs");
    expect(intensityPlotSummary(cfgWith({ channel_intensity: { chart: "box" }, channel_intensity_norm: { test: "none" } })))
      .toBe("7 of 7 graphs · mixed charts · mixed tests");
  });
});

describe("hashCode and intensityPlotState", () => {
  const r = (code: string, extra: object = {}) => ({ kind: "r" as const, code, ...extra });
  const boxCfg = cfgWith({ channel_intensity: { chart: "box" } });
  it("ignores line endings and trailing whitespace", () => {
    expect(hashCode("a\r\nb\n\n  ")).toBe(hashCode("a\nb"));
    expect(hashCode("a\nb")).not.toBe(hashCode("a\nc"));
    expect(hashCode("a\nb")).toMatch(/^3:[0-9a-f]{8}$/);
  });
  it("tells generated code from edited code", () => {
    const data = r(DEFAULT_INTENSITY_PLOT_CODE, { intensityPlotHash: DEFAULT_INTENSITY_PLOT_HASH });
    expect(intensityPlotState(data)).toBe("generated");
    expect(intensityPlotState({ ...data, code: data.code.replace(/\n/g, "\r\n") })).toBe("generated");
    expect(intensityPlotState({ ...data, code: `${data.code}\n# mine` })).toBe("edited");
  });
  it("recognises every legacy hash", () => {
    expect(LEGACY_HASHES.size).toBe(8);
    for (const h of LEGACY_HASHES) expect(h).toMatch(/^\d+:[0-9a-f]{8}$/);
    const code = "# @name: Plot intensities\n# a template this test registers";
    const set = LEGACY_HASHES as Set<string>;
    set.add(hashCode(code));
    try {
      expect(intensityPlotState(r(code))).toBe("legacy");
      expect(intensityPlotState(r(`${code}\n# edited`))).toBe("none");
    } finally {
      set.delete(hashCode(code));
    }
  });
  it("recognises an edited old template and generated code without a hash", () => {
    expect(intensityPlotState(r("# @name: Plot intensities (x)\n.render_panel <- function(d, title, flag_warning, y) 1\n"))).toBe("edited");
    expect(intensityPlotState(r(DEFAULT_INTENSITY_PLOT_CODE))).toBe("edited");
  });
  it("ignores other nodes", () => {
    expect(intensityPlotState(r("plot(1)"))).toBe("none");
    expect(intensityPlotState(r("# @name: Plot intensities\nplot(1)"))).toBe("none");
    expect(intensityPlotState({ kind: "python", code: DEFAULT_INTENSITY_PLOT_CODE })).toBe("none");
  });
  it("other code loaded onto the node makes it an ordinary R node", () => {
    const fields = { intensityPlotHash: DEFAULT_INTENSITY_PLOT_HASH, intensityPlot: boxCfg };
    const preset = "library(ggplot2)\nmpfig_plot(\"bar.png\")\nprint(ggplot(data, aes(group, value)) + geom_col())";
    expect(intensityPlotState(r(preset, fields))).toBe("none");
    expect(intensityPlotEditable(r(preset, fields))).toBe(false);
    // A hand edit of the generated code is still one, even without the header.
    const body = DEFAULT_INTENSITY_PLOT_CODE.split("\n").slice(2).join("\n");
    expect(intensityPlotState(r(body, fields))).toBe("edited");
    // Loading a preset drops the options, hash and backup together.
    const data = { kind: "r" as const, code: "x", label: "Plot", ...fields, intensityPlotBackup: { code: "y", savedAt: "t" } };
    const cleared = withoutIntensityPlot(data);
    expect(cleared).toEqual({ kind: "r", code: "x", label: "Plot" });
    expect(data.intensityPlotBackup).toBeDefined();
    const plain = { kind: "r" as const, code: "x" };
    expect(withoutIntensityPlot(plain)).toBe(plain);
  });
  it("treats a config from a newer build as hand-edited", () => {
    expect(intensityPlotState(r(DEFAULT_INTENSITY_PLOT_CODE, {
      intensityPlotHash: DEFAULT_INTENSITY_PLOT_HASH, intensityPlot: { version: 2, graphs: {} },
    }))).toBe("edited");
  });
});

describe("deriveIntensityPlotContext", () => {
  const CSV = [
    "source,group,channel,compartment,is_control,mean_intensity,mean_intensity_norm",
    'a.tif,"Drug, high",VEGF,whole_cell,False,1,0.5',
    'a.tif,"Drug, high",DAPI,nucleus,True,1,1',
    'b.tif,"say ""hi""",VEGF,cytoplasm,False,2,',
  ].join("\r\n");
  const picker = (over: Partial<FluorIntensityConfig> = {}): FluorIntensityConfig => ({
    ...emptyFluorConfig(),
    channels: { r: "VEGF", g: "Actin", b: "DAPI" },
    groups: [{ id: "1", name: " Ctrl ", images: [] }, { id: "2", name: "Drug", images: [] }, { id: "3", name: "  ", images: [] }],
    ...over,
  });

  it("reads a run's table, including quoted fields", () => {
    expect(deriveIntensityPlotContext({ tableCsv: `﻿${CSV}` })).toEqual({
      groups: ["Drug, high", 'say "hi"'], channels: ["VEGF"], controlChannels: ["DAPI"],
      compartments: ["whole_cell", "nucleus", "cytoplasm"], mode: "cellpose", hasNorm: true, source: "run",
    });
  });
  it("needs a real value for normalisation and spots simple mode", () => {
    const ctx = deriveIntensityPlotContext({
      tableCsv: "group,channel,compartment,is_control,mean_intensity_norm\nA,X,whole_image,False,\nB,X,whole_image,False,None\n",
    });
    expect(ctx).toMatchObject({ mode: "simple", hasNorm: false, compartments: ["whole_image"], groups: ["A", "B"] });
  });
  it("falls back to the picker, which follows its own naming rules", () => {
    expect(deriveIntensityPlotContext({ pickerCfg: picker() })).toEqual({
      groups: ["Ctrl", "Drug"], channels: ["VEGF", "Actin"], controlChannels: ["DAPI"],
      compartments: ["whole_cell", "nucleus", "cytoplasm"], mode: "cellpose", hasNorm: true, source: "picker",
    });
    const base = emptyFluorConfig();
    const simple = deriveIntensityPlotContext({ pickerCfg: picker({
      mode: "simple", controlChannel: null,
      thresholds: { ...base.thresholds, g: { ...base.thresholds.g, enabled: false } },
    }) });
    expect(simple).toMatchObject({ channels: ["VEGF", "DAPI"], controlChannels: [], compartments: ["whole_image"], hasNorm: false, mode: "simple" });
    const nuclei = deriveIntensityPlotContext({ pickerCfg: picker({
      purpose: "nuclei", controlChannel: null, cellpose: { ...base.cellpose, nucleiChannel: null, measureCompartments: false },
    }) });
    expect(nuclei).toMatchObject({ compartments: ["whole_cell"], controlChannels: [], hasNorm: false });
  });
  it("prefers the run's table unless the picker changed since", () => {
    expect(deriveIntensityPlotContext({ tableCsv: CSV, pickerCfg: picker() }).groups).toEqual(["Drug, high", 'say "hi"']);
    const changed = deriveIntensityPlotContext({ tableCsv: CSV, pickerCfg: picker(), pickerChanged: true });
    expect(changed.groups).toEqual(["Ctrl", "Drug", "Drug, high", 'say "hi"']);
    expect(changed.channels).toEqual(["VEGF", "Actin"]);
    expect(changed.source).toBe("picker");
  });
  it("is unknown with nothing usable, and then rules nothing out", () => {
    for (const input of [{}, { tableCsv: "" }, { tableCsv: "a,b\n1,2\n" }, { tableCsv: "group,channel\n" }]) {
      const ctx = deriveIntensityPlotContext(input);
      expect(ctx.source).toBe("unknown");
      for (const id of INTENSITY_GRAPH_IDS) expect(intensityGraphUnavailableReason(id, ctx)).toBeNull();
    }
  });
  it("explains which graphs the data cannot support", () => {
    const ctx = deriveIntensityPlotContext({
      tableCsv: "group,channel,compartment,is_control\nOnly,X,whole_image,False\n(unassigned),X,whole_image,False\n",
    });
    const why = (id: IntensityGraphId) => intensityGraphUnavailableReason(id, ctx);
    expect(why("channel_intensity")).toBeNull();
    expect(why("channel_intensity_per_image")).toMatch(/Simple mode/);
    expect(why("channel_intensity_norm")).toMatch(/control channel/);
    expect(why("control_intensity")).toMatch(/control channel/);
    expect(why("channel_intensity_fc")).toMatch(/2 groups/);
  });
});

describe("node updates", () => {
  const AT = "2026-11-19T10:00:00.000Z";
  type Fields = Pick<NodeData,
    "kind" | "code" | "intensityPlot" | "intensityPlotHash" | "intensityPlotBackup" | "status" | "currentPreset">;
  const fresh = (): Fields => ({ kind: "r", ...defaultIntensityPlotFields(), status: "ok", currentPreset: "custom" });
  const boxes = cfgWith({ channel_intensity: { chart: "box" } });

  it("a new node holds the defaults and reads as generated", () => {
    const d = fresh();
    expect(d.code).toBe(DEFAULT_INTENSITY_PLOT_CODE);
    expect(intensityPlotState(d)).toBe("generated");
    expect(intensityPlotEditable(d)).toBe(true);
    expect(intensityPlotEditable({ ...d, intensityPlot: { version: 2, graphs: {} } as unknown as IntensityPlotConfig })).toBe(false);
    expect(intensityPlotEditable({ kind: "r", code: "plot(1)" })).toBe(false);
  });

  it("apply regenerates the code and marks an ok node stale, like a code edit", () => {
    const next = applyIntensityPlotOptions(fresh(), boxes, AT);
    expect(next.code).toBe(generateIntensityPlotR(boxes));
    expect(next.intensityPlot).toEqual(boxes);
    expect(next.intensityPlotHash).toBe(hashCode(next.code ?? ""));
    expect(next.status).toBe("stale");
    expect(next.intensityPlotBackup).toBeUndefined();
    expect(intensityPlotState(next)).toBe("generated");
    expect(applyIntensityPlotOptions({ ...fresh(), status: "idle" as const }, boxes, AT).status).toBe("idle");
  });

  it("apply with nothing changed returns the same object", () => {
    const d = fresh();
    expect(applyIntensityPlotOptions(d, defaultIntensityPlotConfig(), AT)).toBe(d);
  });

  it("options of a disabled graph change the config but not the code or status", () => {
    const off = onlyGraphs("channel_intensity");
    const d = applyIntensityPlotOptions(fresh(), off, AT);
    const more = updateIntensityGraph(off, "control_intensity", { chart: "violin" });
    const next = applyIntensityPlotOptions({ ...d, status: "ok" as const }, more, AT);
    expect(next.code).toBe(d.code);
    expect(next.status).toBe("ok");
    expect(next.intensityPlot).toEqual(more);
  });

  it("apply over hand-edited code keeps it as the backup; restore puts it back", () => {
    const edited = { ...fresh(), code: `${DEFAULT_INTENSITY_PLOT_CODE}\n# mine` };
    expect(intensityPlotState(edited)).toBe("edited");
    const applied = applyIntensityPlotOptions(edited, boxes, AT);
    expect(applied.intensityPlotBackup).toEqual({ code: edited.code, savedAt: AT });
    expect(intensityPlotState(applied)).toBe("generated");
    const restored = restoreIntensityPlotBackup({ ...applied, status: "ok" as const }, AT);
    expect(restored.code).toBe(edited.code);
    expect(restored.intensityPlotBackup).toBeUndefined();
    expect(restored.status).toBe("stale");
    expect(intensityPlotState(restored)).toBe("edited");
  });

  it("restoring over a second hand edit swaps the two, so neither is lost", () => {
    const first = `${DEFAULT_INTENSITY_PLOT_CODE}\n# first`;
    const applied = applyIntensityPlotOptions({ ...fresh(), code: first }, boxes, AT);
    const second = { ...applied, code: `${applied.code}\n# second` };
    const restored = restoreIntensityPlotBackup(second, AT);
    expect(restored.code).toBe(first);
    expect(restored.intensityPlotBackup?.code).toBe(second.code);
    expect(restoreIntensityPlotBackup(fresh(), AT)).toEqual(fresh());
  });

  it("an untouched legacy template gets no backup", () => {
    const code = "# @name: Plot intensities\n# a template this test registers";
    const set = LEGACY_HASHES as Set<string>;
    set.add(hashCode(code));
    try {
      const next = applyIntensityPlotOptions<Fields>({ kind: "r", code, status: "ok" }, boxes, AT);
      expect(next.intensityPlotBackup).toBeUndefined();
      expect(intensityPlotState(next)).toBe("generated");
    } finally {
      set.delete(hashCode(code));
    }
  });
});
