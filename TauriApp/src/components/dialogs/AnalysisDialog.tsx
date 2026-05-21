/* ──────────────────────────────────────────────────────────
   AnalysisDialog — Full-screen dialog for R-based analysis.

   Top: horizontal plots timeline (aggregated across ALL code tabs).
   Bottom-left:  Data panel  (measurements & manual rows).
   Bottom-right: R-Code panel — a CodeMirror 6 editor wrapped in a
   TABS bar so users can keep multiple code snippets side-by-side
   and produce multiple plots from different scripts. Each tab
   carries its own plot type / statistical test / code text /
   last-run plots. Clicking Run executes the ACTIVE tab; that
   tab's plots replace its previous batch in the top strip.

   The R script is AUTO-GENERATED from the figure's measurements:
   real panel names ("R1C1"), annotation names ("Line 1"), values
   and units are inlined as an editable `data.frame(...)` — which
   ALSO carries a `Group` column so measurements can be split into
   comparison groups. Stats use rstatix; styling uses ggprism.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Box,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Alert,
  CircularProgress,
  Divider,
  Tooltip,
  Tabs,
  Tab,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import DownloadIcon from "@mui/icons-material/Download";
import PhotoLibraryIcon from "@mui/icons-material/PhotoLibrary";
import ViewQuiltIcon from "@mui/icons-material/ViewQuilt";
import CodeMirror from "@uiw/react-codemirror";
import { StreamLanguage } from "@codemirror/language";
import { r as rLanguageMode } from "@codemirror/legacy-modes/mode/r";
import { oneDark } from "@codemirror/theme-one-dark";
import { api } from "../../api/client";
import { AnalysisNodeGraph } from "./AnalysisNodeGraph";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import {
  useAnalysisStore,
  type AnalysisManifest,
  type AnalysisTableSnapshot,
} from "../../store/analysisStore";
import type { AnalysisPayload } from "../../api/types";

// base64 PNG → File (for adding analysis plots to the main image timeline)
function b64ToFile(b64: string, filename: string): File {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new File([arr], filename, { type: "image/png" });
}

// natural pixel dimensions of a base64 PNG (for collage item sizing)
function imageDims(b64: string): Promise<{ w: number; h: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth || 800, h: img.naturalHeight || 600 });
    img.onerror = () => resolve({ w: 800, h: 600 });
    img.src = `data:image/png;base64,${b64}`;
  });
}

// Parse a small CSV text into a 2-D string array. Good enough for
// the modest stats / data tables we emit from R (no embedded
// newlines, simple quoted-string handling).
function parseCsv(csv: string): string[][] {
  const out: string[][] = [];
  csv.split(/\r?\n/).forEach((rawLine) => {
    if (!rawLine.trim() && out.length > 0 && out[out.length - 1].length === 0) return;
    if (!rawLine && out.length === 0) return;
    const row: string[] = [];
    let cell = "";
    let inQ = false;
    for (let i = 0; i < rawLine.length; i++) {
      const ch = rawLine[i];
      if (inQ) {
        if (ch === '"' && rawLine[i + 1] === '"') { cell += '"'; i++; }
        else if (ch === '"') { inQ = false; }
        else cell += ch;
      } else {
        if (ch === '"') inQ = true;
        else if (ch === ",") { row.push(cell); cell = ""; }
        else cell += ch;
      }
    }
    row.push(cell);
    out.push(row);
  });
  return out;
}

// Download a CSV string as a file.
function downloadCsv(csv: string, basename: string) {
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${basename}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

interface Measurement {
  panel: string;
  name: string;
  type: string;
  value: string;            // display string (may be a user override)
  numeric?: number;         // computed numeric value
  unit?: string;            // unit label, e.g. "µm" / "µm²"
}

interface Props {
  open: boolean;
  onClose: () => void;
  measurements: Measurement[];
}

// Each measurement gets bucketed into one of these kinds.
//   "distance" = line annotations           (m.type === "line")
//   "area"     = area annotations           (m.type === "area")
//   "manual"   = a row added in the Data panel's manual table
// Tabs filter on this so distance and area analyses stay separate
// by default (area-vs-distance comparisons are nonsense — different
// units, different magnitudes).
export type MeasureRowType = "distance" | "area" | "manual";

// Per-tab filter on which row kinds are inlined into that tab's
// data.frame. "all" = no filter.
export type MeasureTypeFilter = "all" | "distance" | "area";

// One normalised data row used to build the inline R `data.frame`.
interface DataRow {
  panel: string;
  name: string;
  group: string;            // comparison group (defaults to the panel name)
  value: number | null;
  unit: string;
  measureType: MeasureRowType; // used by the per-tab filter; NOT emitted to R
}

// One plot produced by a tab's last Run. Stable id so removing a
// plot from the middle of the array doesn't shift everyone else's
// keys (preserves the selection / destination-tracking links).
interface AnalysisPlot {
  id: string;
  b64: string;
  /** When set: this plot has been copied into the main image
   *  timeline under this filename. Used to (a) skip duplicate adds
   *  and (b) remove it from the timeline if the user discards
   *  the plot. */
  mainName?: string;
  /** When set: this plot has been copied into the Collage Assembly
   *  as the item with this id. Same role as mainName. */
  collageId?: string;
  /** Table CSVs emitted by the R script via mpfig_data(...). Each
   *  entry is one named data.frame (the inline data + e.g. stat.test).
   *  Filled from the run-r response. */
  tables?: AnalysisTable[];
}

// One CSV-encoded data table captured from R (the inlined `data`
// and any stat.test result). The R `mpfig_data()` helper writes
// these into <tmpdir>/tables/<name>.csv and the backend returns
// them in the run-r response.
interface AnalysisTable {
  id: string;
  name: string;   // e.g. "data", "stat_test" (the file's basename)
  csv: string;    // raw CSV text
}

// One code tab — its own measurement-type filter, plot type, stat
// test, editor text, and plots produced by its last Run. The dialog
// keeps an array of these in state so the user can flip between
// multiple analyses.
interface CodeTab {
  id: string;
  name: string;             // "Distances" / "Areas" / "Code N" (display only)
  measureType: MeasureTypeFilter;
  plotType: string;
  statTest: string;
  code: string;
  plots: AnalysisPlot[];    // PNGs + dest tracking from this tab's last Run
  /** Which interpreter the primary Run button dispatches for this
   *  tab. Defaults to "r" (the historical default) so existing
   *  tabs / saved projects keep working. Pipeline-seeded tabs
   *  (Python starter, MATLAB starter) carry "python" or "matlab"
   *  so Run does the right thing without an extra selector click. */
  engine?: "r" | "python" | "matlab";
}

function makeTabId(): string {
  return `tab_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}
function makePlotId(): string {
  return `plot_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}
function makeTableId(): string {
  return `tbl_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

// ── R string / number escaping ─────────────────────────────
const rStr = (s: string) =>
  '"' + String(s ?? "").replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
const rNum = (n: number | null | undefined) =>
  n == null || Number.isNaN(n) ? "NA" : String(n);

// ── Plot types ─────────────────────────────────────────────
// Each entry supplies the ggplot geom layers for that plot
// style. ggprism theming + the optional stats overlay are added
// by buildRCode(). `usesGroupBrackets` = the plot can carry
// add_pvalue() significance brackets across Groups.
const PLOT_TYPES: Record<string, { label: string; file: string; usesGroupBrackets: boolean; geoms: string }> = {
  bar: {
    label: "Bar (mean ± SE)",
    file: "bar_chart.png",
    usesGroupBrackets: true,
    geoms: `  stat_summary(aes(fill = Group), fun = mean, geom = "col", width = 0.7) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  geom_jitter(width = 0.12, size = 1.6, alpha = 0.55) +`,
  },
  box: {
    label: "Box plot",
    file: "box_plot.png",
    usesGroupBrackets: true,
    geoms: `  geom_boxplot(aes(fill = Group), width = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.12, size = 1.6, alpha = 0.55) +`,
  },
  violin: {
    label: "Violin plot",
    file: "violin_plot.png",
    usesGroupBrackets: true,
    geoms: `  geom_violin(aes(fill = Group), trim = FALSE, alpha = 0.85) +
  geom_boxplot(width = 0.14, fill = "white", alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.1, size = 1.4, alpha = 0.5) +`,
  },
  scatter: {
    label: "Scatter (+ linear fit)",
    file: "scatter.png",
    usesGroupBrackets: false,   // a scatter-by-index can't carry comparison brackets
    geoms: `  geom_point(aes(color = Group), size = 3) +
  geom_smooth(aes(color = Group), method = "lm", se = TRUE, alpha = 0.15) +`,
  },
};

// ── Statistical tests (rstatix) ────────────────────────────
// `compute` runs inside an `if (length(unique(data$Group)) >= 2)`
// guard in buildRCode() and must assign `stat.test`.
const STAT_TESTS: Record<string, { label: string; needsRstatix: boolean; console?: string; compute?: string }> = {
  none: {
    label: "No statistics",
    needsRstatix: false,
  },
  ttest: {
    label: "t-test (parametric, pairwise)",
    needsRstatix: true,
    compute: `  stat.test <- data %>%
    t_test(Value ~ Group) %>%                # pairwise Welch t-tests
    adjust_pvalue(method = "BH") %>%          # EDIT: "bonferroni", "holm", "none"...
    add_significance() %>%
    add_xy_position(x = "Group")`,
  },
  wilcox: {
    label: "Wilcoxon (non-parametric, pairwise)",
    needsRstatix: true,
    compute: `  stat.test <- data %>%
    wilcox_test(Value ~ Group) %>%            # pairwise Wilcoxon rank-sum
    adjust_pvalue(method = "BH") %>%
    add_significance() %>%
    add_xy_position(x = "Group")`,
  },
  anova: {
    label: "ANOVA + Tukey HSD",
    needsRstatix: true,
    console: `  cat("\\n=== One-way ANOVA ===\\n")
  print(data %>% anova_test(Value ~ Group))`,
    compute: `  # Tukey HSD post-hoc — its p.adj.signif drives the brackets
  stat.test <- data %>%
    tukey_hsd(Value ~ Group) %>%
    add_xy_position(x = "Group")`,
  },
  kruskal: {
    label: "Kruskal-Wallis + Dunn",
    needsRstatix: true,
    console: `  cat("\\n=== Kruskal-Wallis test ===\\n")
  print(data %>% kruskal_test(Value ~ Group))`,
    compute: `  # Dunn post-hoc — its p.adj.signif drives the brackets
  stat.test <- data %>%
    dunn_test(Value ~ Group) %>%
    add_xy_position(x = "Group")`,
  },
};

// Example rows shown when there are no real measurements yet —
// two groups so the stats run out of the box.
const EXAMPLE_ROWS: DataRow[] = [
  { panel: "R1C1", name: "Line 1", group: "Control", value: 12.5, unit: "µm", measureType: "distance" },
  { panel: "R1C1", name: "Line 2", group: "Control", value: 13.8, unit: "µm", measureType: "distance" },
  { panel: "R1C2", name: "Line 1", group: "Treated", value: 19.2, unit: "µm", measureType: "distance" },
  { panel: "R1C2", name: "Line 2", group: "Treated", value: 21.0, unit: "µm", measureType: "distance" },
];

/** Build the verbose, auto-populated `data <- data.frame(...)` block. */
function buildDataSection(rows: DataRow[], filter: MeasureTypeFilter = "all"): string {
  const isExample = rows.length === 0;
  const r = isExample ? EXAMPLE_ROWS : rows;
  const filterNote =
    filter === "distance"
      ? `# (This tab is filtered to DISTANCE measurements — line annotations only.)\n`
      : filter === "area"
        ? `# (This tab is filtered to AREA measurements — area annotations only.)\n`
        : "";
  const emptyHint =
    filter === "distance"
      ? "no distance (line) measurements were found"
      : filter === "area"
        ? "no area measurements were found"
        : "no measurements were found";
  const note = isExample
    ? `# NOTE: ${emptyHint}, so these are EXAMPLE rows.
#   Add lines/areas with "measure" enabled in a panel's Annotations
#   tab — or just edit the numbers/groups below directly.\n`
    : "";
  return `# ── Your data  (auto-filled from the figure's measurements) ──
# Each row is one measurement. EVERYTHING here is editable —
# change a Value, rename things, add or delete rows. Whatever is
# in \`data\` when you press Run is what gets analysed.
#   Panel = which figure panel the measurement came from (e.g. "R1C1")
#   Name  = the annotation's name (e.g. "Line 1", "Area 2")
#   Group = the COMPARISON GROUP this row belongs to. <<< RENAME THIS
#           FOR PUBLICATION: it defaults to the panel coordinate
#           ("R1C1", "R1C2", ...) which is NOT publication-friendly
#           — those labels print on the x-axis and in the stats.
#           See the "Rename your groups" block below for the easy
#           commented-out patterns, or just edit the Group entries
#           in the vector directly.
#   Value = the measured number   (numeric — edit freely)
#   Unit  = the measurement unit  (e.g. "µm", "µm²")
${filterNote}${note}data <- data.frame(
  Panel = c(${r.map((x) => rStr(x.panel)).join(", ")}),
  Name  = c(${r.map((x) => rStr(x.name)).join(", ")}),
  Group = c(${r.map((x) => rStr(x.group)).join(", ")}),
  Value = c(${r.map((x) => rNum(x.value)).join(", ")}),
  Unit  = c(${r.map((x) => rStr(x.unit)).join(", ")}),
  stringsAsFactors = FALSE
)
`;
}

/** Build a commented-out "rename your groups" block keyed to the
 *  actual group labels present in `rows`. The default Group values
 *  are panel coordinates ("R1C1", ...) which are not publication-
 *  friendly; this block gives the user three drop-in patterns
 *  (in-place rename / dplyr::recode / factor levels) pre-populated
 *  with their real group names so they just have to uncomment and
 *  edit the right-hand side. */
function buildGroupRenameBlock(rows: DataRow[]): string {
  const isExample = rows.length === 0;
  const sourceRows = isExample ? EXAMPLE_ROWS : rows;

  // Unique groups in order of first appearance.
  const seen = new Set<string>();
  const groups: string[] = [];
  sourceRows.forEach((r) => {
    if (!seen.has(r.group)) {
      seen.add(r.group);
      groups.push(r.group);
    }
  });
  if (groups.length === 0) return "";

  // Suggested replacement labels. The user will edit these — we
  // just want each option to be a plausible starting point.
  const suggestions: string[] =
    groups.length === 2
      ? ["Control", "Treated"]
      : groups.length === 3
        ? ["Control", "Condition 1", "Condition 2"]
        : groups.map((_, i) => `Group ${i + 1}`);

  const inPlaceLines = groups
    .map((g, i) => `# data$Group[data$Group == ${rStr(g)}] <- ${rStr(suggestions[i])}`)
    .join("\n");

  const recodeLines = groups
    .map((g, i) => `#   ${rStr(g)} = ${rStr(suggestions[i])}${i < groups.length - 1 ? "," : ""}`)
    .join("\n");

  const factorLine = `# data$Group <- factor(data$Group, levels = c(${suggestions.map(rStr).join(", ")}))`;

  return `
# ── Rename your groups for publication ─────────────────────
# IMPORTANT: the Group column above defaults to the figure's
# PANEL COORDINATE (e.g. "R1C1", "R1C2"). Those labels are what
# the plot prints on the x-axis and what the statistics output
# uses to identify groups — they are NOT publication-friendly.
# Rename them to something meaningful ("Control", "Treated",
# "Wild-type", "KO", "Drug X", ...) before you save the plot.
# Pick ONE of the three patterns below, uncomment it, and edit
# the right-hand side. You can also collapse two panels into the
# same group (e.g. two R1C2 replicates and the R1C3 panel all
# become "Treated") by giving them the same new name.
#
# Option A — rename one panel-name at a time (UNCOMMENT + edit):
${inPlaceLines}
#
# Option B — bulk rename via dplyr::recode (UNCOMMENT + edit;
#   also uncomment library(dplyr) at the top if needed):
# data$Group <- dplyr::recode(data$Group,
${recodeLines}
# )
#
# Option C — also lock the DISPLAY ORDER on the x-axis. The plot
# draws groups left-to-right in this order, and the FIRST level
# becomes the reference group for the statistical test. Uncomment
# + reorder once your names are settled:
${factorLine}
`;
}

/** Assemble the full R script: header + data + libraries + stats + plot. */
function buildRCode(plotKey: string, statKey: string, rows: DataRow[], measureType: MeasureTypeFilter = "all"): string {
  const plot = PLOT_TYPES[plotKey] ?? PLOT_TYPES.bar;
  const stat = STAT_TESTS[statKey] ?? STAT_TESTS.none;
  const hasStats = statKey !== "none";
  const drawBrackets = hasStats && plot.usesGroupBrackets;
  const unitExpr = `if (length(unique(data$Unit)) == 1) data$Unit[1] else "value"`;

  const header = `# ============================================================
#  Multi-Panel Figure  —  Analysis script   (auto-generated)
#
#  HOW TO USE THIS EDITOR
#   • Everything below is EDITABLE. Tweak the data + Group
#     assignments, the plot type, the test, titles, colours —
#     then press  ▶ Run.
#   • The \`data\` table was filled in from your figure's
#     measurements. Its \`Group\` column controls which rows are
#     COMPARED against each other in the statistics.
#   • The plot uses ggprism (GraphPad Prism style). Swap the
#     palette / theme below to taste.
#   • Pick a plot type and a statistical test from the toolbar,
#     or press  ↻ Reset  to regenerate this script.
#   • Add more  +  tabs to keep multiple code snippets / plot
#     types side-by-side — each tab has its own Run.
#   • mpfig_plot("name.png") opens a plot device — rename the
#     file to control what the saved plot is called.
# ============================================================

`;

  // ── Libraries ──
  let libs = `# ── Libraries ───────────────────────────────────────────────
library(ggplot2)
library(ggprism)   # GraphPad Prism-style theme, palettes, add_pvalue()
`;
  if (stat.needsRstatix) {
    libs += `library(rstatix)   # statistical tests (t-test / Wilcoxon / ANOVA / ...)
library(dplyr)     # the %>% pipe used below
`;
  }

  // ── Statistics block ──
  let statsBlock = "";
  if (hasStats && stat.compute) {
    statsBlock = `
# ── Statistics: ${stat.label} ────────────────────────────────
# EDIT: pick a different test from the toolbar's "Statistics"
# menu, or tweak this block directly. Needs >= 2 Groups in \`data\`.
stat.test <- NULL
if (length(unique(data$Group)) >= 2) {
${stat.console ? stat.console + "\n" : ""}${stat.compute}
  cat("\\n=== ${stat.label} ===\\n"); print(stat.test)
  # Persist the test result so the app can show it as a table and
  # export it to CSV from the plot-preview modal.
  if (!is.null(stat.test)) mpfig_data(as.data.frame(stat.test), "stat_test")
} else {
  cat("Only one Group in \`data\` — assign rows to >= 2 Groups",
      "above to run statistics.\\n")
}
`;
  }

  // Always persist the analysed data table too — even when there's
  // no test — so the app can show it alongside the plot.
  const persistDataBlock = `
# ── Persist the data table for the app's table viewer ───────
# (mpfig_data writes <name>.csv into the run's temp directory,
#  which the app reads back and surfaces as an exportable table.)
mpfig_data(data, "data")
`;

  // ── Plot block ──
  const groupBracketLine = drawBrackets
    ? `# Significance brackets — p.adj.signif shows stars (***, **, *, ns).
# EDIT: use label = "p.adj" to print the numeric p-value instead.
if (!is.null(stat.test)) {
  p <- p + add_pvalue(stat.test, label = "p.adj.signif", tip.length = 0.01)
}
`
    : "";
  const scatterStatNote =
    hasStats && !plot.usesGroupBrackets
      ? `# (A scatter-by-index can't carry comparison brackets — the test
#  result above is printed to the console instead.)
`
      : "";

  const plotBlock = `
# ── Plot: ${plot.label} ─────────────────────────────────────
mpfig_plot("${plot.file}")            # <- saved plot file name (editable)
.unit <- ${unitExpr}
p <- ggplot(data, aes(x = ${plotKey === "scatter" ? "seq_along(Value)" : "Group"}, y = Value)) +
${plot.geoms}
  theme_prism(base_size = 14) +                  # <- ggprism theme
  scale_fill_prism(palette = "floral") +         # <- EDIT palette: "candy_bright", "office", "viridis"...
  scale_colour_prism(palette = "floral") +
  labs(
    title = "Measurements by Group",             # <- EDIT chart title for publication
    x = ${plotKey === "scatter" ? '"Measurement #"' : '"Group"'},               # <- EDIT x-axis label (e.g. "Condition")
    y = paste0("Value (", .unit, ")")             # <- EDIT y-axis label (e.g. "Spine length (µm)")
  )
${scatterStatNote}${groupBracketLine}print(p)
`;

  return (
    header +
    buildDataSection(rows, measureType) +
    buildGroupRenameBlock(rows) +
    "\n" +
    libs +
    persistDataBlock +
    statsBlock +
    plotBlock
  );
}

// Compose a stable key for one plot in the aggregated top strip.
// Plot ids are unique, so `tabId|plotId` is overdetermined — but
// keeping the tabId in the key makes selection/added-to bookkeeping
// trivial to filter when a tab is closed or re-run.
const plotKey = (tabId: string, plotId: string) => `${tabId}|${plotId}`;
const parsePlotKey = (k: string): { tabId: string; plotId: string } => {
  const sep = k.indexOf("|");
  return { tabId: k.slice(0, sep), plotId: k.slice(sep + 1) };
};

// ─────────────────────────────────────────────────────────────
// Python / MATLAB pipelines are for IMAGE DATA EXTRACTION only —
// plotting is delegated to R (better-looking figures, consistent
// styling with the rest of the analysis). Each pipeline:
//   • computes per-source numeric values from the inset pixels,
//   • writes them as CSV via mpfig_data(...) — that table can
//     then be sent into an R tab for plotting,
//   • optionally writes a derived image via mpfig_image(...) for
//     downstream use in the figure.
// ─────────────────────────────────────────────────────────────

// "Custom" starter — empty template the user fills in. Includes
// the contract docstring so beginners know what's available.
const PYTHON_STARTER_CODE = `# ============================================================
# Python pipeline — runs against zoom-inset pixels.
# Use this to EXTRACT numeric data; let R plot it afterwards.
#
# Available globals:
#   inputs[key]   → dict with:
#       image  : uint8 numpy array, shape (H, W, 3) — RGB pixels
#       width  : int
#       height : int
#       label  : str — human-readable "R1C1 · inset 1 (...)"
#       row, col, inset_index — original grid coords
#
# Output helpers (data only — no plotting):
#   mpfig_data(rows, name)→ save CSV (DataFrame / dict / list).
#   mpfig_image(arr, name)→ save numpy / PIL image to the
#                           "Python images" strip (for use as a
#                           Separate Image inset later).
# ============================================================

import numpy as np

rows = []
for key, src in inputs.items():
    img = src["image"]               # H x W x 3, uint8
    gray = img.mean(axis=2)          # luminance
    rows.append({
        "source": src["label"],
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "min_gray": float(gray.min()),
        "max_gray": float(gray.max()),
    })
mpfig_data(rows, name="pixel_summary")
`;

// "Custom" MATLAB starter — data extraction only.
const MATLAB_STARTER_CODE = `% =============================================================
% MATLAB / Octave pipeline — runs against zoom-inset pixels.
% Use this to EXTRACT numeric data; let R plot it afterwards.
%
% Available variables (after \`load('inputs.mat')\`):
%   inputs.<safe_key>.image  — uint8 H x W x 3 matrix
%   inputs.<safe_key>.label  — human-readable source name
%
% Output helpers (data only — no plotting):
%   mpfig_data(struct, name)  — save CSV (struct of column vectors)
%   mpfig_image(arr, name)    — save image to the images strip
% =============================================================

keys = fieldnames(inputs);
labels   = cell(numel(keys), 1);
mean_g   = zeros(numel(keys), 1);
std_g    = zeros(numel(keys), 1);
min_g    = zeros(numel(keys), 1);
max_g    = zeros(numel(keys), 1);
for k = 1:numel(keys)
  src = inputs.(keys{k});
  gray = mean(double(src.image), 3);
  labels{k} = src.label;
  mean_g(k) = mean(gray(:));
  std_g(k)  = std(gray(:));
  min_g(k)  = min(gray(:));
  max_g(k)  = max(gray(:));
end
mpfig_data(struct('source', {labels}, 'mean_gray', mean_g, 'std_gray', std_g, ...
                  'min_gray', min_g, 'max_gray', max_g), 'pixel_summary');
`;

// ── Preset registry ─────────────────────────────────────────
// Each preset has a label, an engine-specific code generator,
// and an optional "needs reference" flag that surfaces a
// reference-inset selector in the UI. Code generators receive
// the reference key (if any) and the safe-name mapping for
// MATLAB so the snippets address inputs correctly.

type PipelinePreset = "custom" | "haze";

interface PresetSpec {
  label: string;
  needsReference: boolean;
  description: string;
  generatePython: (referenceKey: string | null) => string;
  generateMatlab: (referenceKey: string | null) => string;
}

// MATLAB safe-key mirror of the backend's `key_map`: backend
// replaces non-alnum chars in `key` with `_` and prefixes `k_`.
// Keep this in sync with `_extract_inset_image` / run-matlab.
function matlabSafeKey(key: string): string {
  return "k_" + key.replace(/[^a-zA-Z0-9]/g, "_");
}

const PIPELINE_PRESETS: Record<PipelinePreset, PresetSpec> = {
  custom: {
    label: "Custom",
    needsReference: false,
    description: "Empty template — write your own analysis.",
    generatePython: () => PYTHON_STARTER_CODE,
    generateMatlab: () => MATLAB_STARTER_CODE,
  },
  haze: {
    label: "Haze analysis",
    needsReference: true,
    description:
      "Compares grayscale mean of each inset to a reference inset. " +
      "Output: per-source mean_gray, delta_vs_reference, and " +
      "ratio_to_reference. Useful for haze / fog / contrast " +
      "quantification against a clean control region.",
    generatePython: (refKey) => {
      const refLit = refKey ? `"${refKey}"` : "None";
      return `# ============================================================
# HAZE ANALYSIS — compares each inset's grayscale mean against
# a reference inset. Output: per-source mean_gray,
# delta_vs_reference (mean - ref_mean), ratio_to_reference
# (mean / ref_mean). Plot the resulting CSV from an R tab.
# ============================================================

import numpy as np

REFERENCE_KEY = ${refLit}  # set by the Reference selector above

if REFERENCE_KEY is None or REFERENCE_KEY not in inputs:
    raise SystemExit("Pick a reference inset above before running haze analysis.")

ref_img = inputs[REFERENCE_KEY]["image"]
ref_label = inputs[REFERENCE_KEY]["label"]
ref_mean = float(np.mean(ref_img.mean(axis=2)))
print(f"Reference: {ref_label} → mean_gray = {ref_mean:.3f}")

rows = []
for key, src in inputs.items():
    gray = src["image"].mean(axis=2)
    m = float(gray.mean())
    rows.append({
        "source": src["label"],
        "key": key,
        "is_reference": (key == REFERENCE_KEY),
        "mean_gray": m,
        "delta_vs_reference": m - ref_mean,
        "ratio_to_reference": m / ref_mean if ref_mean != 0 else float("nan"),
    })

# Sort with the reference first
rows.sort(key=lambda r: (not r["is_reference"], r["source"]))
mpfig_data(rows, name="haze_analysis")
`;
    },
    generateMatlab: (refKey) => {
      const refLit = refKey ? `'${matlabSafeKey(refKey)}'` : "''";
      return `% =============================================================
% HAZE ANALYSIS — compares each inset's grayscale mean against
% a reference inset. Output: per-source mean_gray,
% delta_vs_reference, ratio_to_reference. Plot in R afterwards.
% =============================================================

REFERENCE_KEY = ${refLit};

if isempty(REFERENCE_KEY) || ~isfield(inputs, REFERENCE_KEY)
  error('Pick a reference inset above before running haze analysis.');
end

ref = inputs.(REFERENCE_KEY);
ref_mean = mean(mean(mean(double(ref.image), 3)));
fprintf('Reference: %s → mean_gray = %.3f\\n', ref.label, ref_mean);

keys = fieldnames(inputs);
n = numel(keys);
sources   = cell(n, 1);
key_col   = cell(n, 1);
is_ref    = false(n, 1);
mean_gray = zeros(n, 1);
delta     = zeros(n, 1);
ratio     = zeros(n, 1);
for k = 1:n
  src = inputs.(keys{k});
  m = mean(mean(mean(double(src.image), 3)));
  sources{k}   = src.label;
  key_col{k}   = keys{k};
  is_ref(k)    = strcmp(keys{k}, REFERENCE_KEY);
  mean_gray(k) = m;
  delta(k)     = m - ref_mean;
  if ref_mean ~= 0
    ratio(k) = m / ref_mean;
  else
    ratio(k) = NaN;
  end
end

mpfig_data(struct( ...
  'source', {sources}, 'key', {key_col}, ...
  'is_reference', is_ref, 'mean_gray', mean_gray, ...
  'delta_vs_reference', delta, 'ratio_to_reference', ratio), ...
  'haze_analysis');
`;
    },
  },
};

export function AnalysisDialog({ open, onClose, measurements }: Props) {
  // R status
  const [rInstalled, setRInstalled] = useState<boolean | null>(null);
  const [rVersion, setRVersion] = useState("");
  const [customRPath, setCustomRPath] = useState("");
  const [rPathInput, setRPathInput] = useState("");

  // Data
  const [selectedPanels, setSelectedPanels] = useState<Set<string>>(new Set());
  const [manualRows, setManualRows] = useState<string[][]>([]);
  const manualCols = ["Panel", "Name", "Group", "Value"];

  // ── Code tabs ──
  // Each tab carries its own plot type / test / code / plot batch.
  // The very first tab is created on first open() once the data
  // rows are known.
  const [tabs, setTabs] = useState<CodeTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [runningPy, setRunningPy] = useState(false);
  const [runningMatlab, setRunningMatlab] = useState(false);
  const [matlabInfo, setMatlabInfo] = useState<{ installed: boolean; kind: string }>({ installed: false, kind: "" });
  /** Which engine the secondary run button currently dispatches.
   *  Persists across runs so the user's last choice is remembered. */
  const [pipelineEngine, setPipelineEngine] = useState<"python" | "matlab">("python");
  /** Active pipeline preset. Choosing one regenerates the editor
   *  contents (only when the current text matches the previous
   *  preset's generated code — custom edits are preserved). */
  const [pipelinePreset, setPipelinePreset] = useState<PipelinePreset>("custom");
  /** For haze analysis: which inset key is the reference. Auto-
   *  defaults to the first available source. */
  const [hazeReferenceKey, setHazeReferenceKey] = useState<string>("");

  // Zoom inset sources marked `include_in_analysis`. Refreshed on
  // dialog open so the Python / MATLAB pipeline button stays
  // current. Each entry carries a small thumbnail so the Analysis
  // dialog can show the user exactly which pixels the pipelines
  // will operate on.
  const [insetSources, setInsetSources] = useState<Array<{
    key: string; row: number; col: number; inset_index: number;
    label: string;
    natural_width: number; natural_height: number;
    thumbnail: string;
  }>>([]);
  // Modified images produced by the most recent pipeline run —
  // surfaced as a small horizontal strip below the console.
  const [pyImages, setPyImages] = useState<Array<{ name: string; image: string }>>([]);
  // CSV tables from the most recent pipeline run — Python/MATLAB
  // pipelines are data-extraction only (per spec), so the tables
  // bucket gets its own dedicated panel with a "Send to R tab"
  // affordance per table (creates a new R tab whose `data` table
  // is populated from this CSV — the user then picks a plot type
  // and runs as usual).
  const [pyTables, setPyTables] = useState<AnalysisTable[]>([]);

  // Modal preview key and selection are GLOBAL across tabs so the
  // top strip can mix plots from different tabs.
  const [selectedPlot, setSelectedPlot] = useState<string | null>(null); // plot-key for modal
  const [selectedPlotKeys, setSelectedPlotKeys] = useState<Set<string>>(new Set());

  // (Per-plot "added to main/collage" tracking now lives on each
  // AnalysisPlot directly — see plot.mainName / plot.collageId.
  // No separate Set state required.)

  // Discard-confirmation modal. Set when the user clicks the trash
  // icon on a plot that's already been pushed somewhere — we need
  // to ask before removing from those destinations.
  const [discardConfirm, setDiscardConfirm] = useState<{
    plotKey: string;
    plotName: string;
    inMain: boolean;
    inCollage: boolean;
  } | null>(null);

  // Transient status flash (auto-dismisses). Used for "already added"
  // notices and similar one-line confirmations.
  const [flash, setFlash] = useState<{ msg: string; severity: "info" | "success" | "warning" } | null>(null);
  const showFlash = useCallback((msg: string, severity: "info" | "success" | "warning" = "info") => {
    setFlash({ msg, severity });
    // Auto-clear after a few seconds — only if it's still the same msg
    // (avoid clobbering a newer flash that replaced it).
    window.setTimeout(() => {
      setFlash((cur) => (cur && cur.msg === msg ? null : cur));
    }, 3500);
  }, []);

  // Console — accumulates output from "Run" AND from the inline R
  // console below the editor (whose main use is package management,
  // e.g. install.packages("...") to add R libraries used by edited code).
  const [consoleOut, setConsoleOut] = useState("");
  const [consoleCmd, setConsoleCmd] = useState("");
  const [consoleRunning, setConsoleRunning] = useState(false);

  // ── Active tab helpers ──
  const activeTab: CodeTab | undefined = tabs.find((t) => t.id === activeTabId) ?? tabs[0];
  const updateActiveTab = useCallback(
    (patch: Partial<CodeTab>) => {
      setTabs((prev) => prev.map((t) => (t.id === activeTabId ? { ...t, ...patch } : t)));
    },
    [activeTabId],
  );

  // Aggregated plot list across ALL tabs, in tab order. idx is just
  // the display position within the tab (1-based filenames); the
  // identity of the plot is plot.id (and plotKey() in selection state).
  const allPlots: {
    key: string;
    tabId: string;
    tabName: string;
    plotId: string;
    idx: number;
    b64: string;
    plotType: string;
    statTest: string;
    code: string;
    mainName?: string;
    collageId?: string;
    tables?: AnalysisTable[];
  }[] = tabs.flatMap((t) =>
    t.plots.map((p, idx) => ({
      key: plotKey(t.id, p.id),
      tabId: t.id,
      tabName: t.name,
      plotId: p.id,
      idx,
      b64: p.b64,
      plotType: t.plotType,
      statTest: t.statTest,
      code: t.code,
      mainName: p.mainName,
      collageId: p.collageId,
      tables: p.tables,
    })),
  );

  // Normalised data rows (selected measurements + non-empty manual rows)
  // — this is what gets inlined into the generated R `data.frame`.
  // Group defaults to the panel name; the user re-assigns groups in the code.
  // The optional `filter` arg gates rows by their measureType so a
  // "Distances" tab gets line rows only and an "Areas" tab gets areas
  // only. Manual rows pass through every filter (they have no fixed type).
  const getDataRows = useCallback(
    (filter: MeasureTypeFilter = "all"): DataRow[] => {
      const rows: DataRow[] = [];
      measurements.forEach((m) => {
        if (!selectedPanels.has(m.panel)) return;
        const mt: MeasureRowType = m.type === "area" ? "area" : "distance";
        if (filter !== "all" && mt !== filter) return;
        const v = m.numeric ?? parseFloat(String(m.value).replace(/[^0-9.\-]/g, ""));
        rows.push({
          panel: m.panel,
          name: m.name,
          group: m.panel, // default: each panel is its own group
          value: Number.isNaN(v) ? null : v,
          unit: m.unit ?? "",
          measureType: mt,
        });
      });
      manualRows.forEach((row) => {
        if (row.some((c) => c.trim())) {
          const v = parseFloat(String(row[3] ?? "").replace(/[^0-9.\-]/g, ""));
          rows.push({
            panel: row[0] || "Manual",
            name: row[1] || "row",
            group: row[2] || row[0] || "Manual",
            value: Number.isNaN(v) ? null : v,
            unit: "",
            measureType: "manual",
          });
        }
      });
      return rows;
    },
    [measurements, selectedPanels, manualRows],
  );

  // Regenerate the active tab's editor contents from its current
  // measure-type filter + plot type + test + the shared Data panel.
  const regenerateCode = useCallback(
    (measureType: MeasureTypeFilter, plotType: string, statTest: string) => {
      updateActiveTab({
        code: buildRCode(plotType, statTest, getDataRows(measureType), measureType),
      });
    },
    [getDataRows, updateActiveTab],
  );

  // Check R on open
  const checkRStatus = useCallback(async (path?: string) => {
    try {
      const r = await api.checkR(path);
      setRInstalled(r.installed);
      setRVersion(r.version);
      if (r.installed && path) setCustomRPath(path);
    } catch {
      setRInstalled(false);
    }
  }, []);

  // ── Project save bridge ───────────────────────────────────
  // Publish the current tabs/plots/tables as an AnalysisPayload
  // to the analysisStore on every state change. The Sidebar's
  // save handler reads it synchronously to embed in the .mpf zip.
  useEffect(() => {
    const tableMeta: Record<string, AnalysisTableSnapshot> = {};
    const tablesMap: Record<string, string> = {};
    const plotsMap: Record<string, string> = {};
    const tabsSnap = tabs.map((t) => ({
      id: t.id,
      name: t.name,
      measureType: t.measureType,
      plotType: t.plotType,
      statTest: t.statTest,
      code: t.code,
      plots: t.plots.map((p) => {
        plotsMap[p.id] = p.b64;
        const tableIds = (p.tables ?? []).map((tb) => {
          tableMeta[tb.id] = { id: tb.id, name: tb.name };
          tablesMap[tb.id] = tb.csv;
          return tb.id;
        });
        return {
          id: p.id,
          mainName: p.mainName,
          collageId: p.collageId,
          tableIds,
        };
      }),
    }));
    const manifest: AnalysisManifest = {
      version: 1,
      tabs: tabsSnap,
      activeTabId,
      tableMeta,
    };
    // Empty-state: no tabs → publish null so the .mpf doesn't
    // carry an empty analysis/ folder.
    if (tabs.length === 0) {
      useAnalysisStore.getState().publishSnapshot(null);
      return;
    }
    const payload: AnalysisPayload = {
      manifest,
      plots: plotsMap,
      tables: tablesMap,
    };
    useAnalysisStore.getState().publishSnapshot(payload);
  }, [tabs, activeTabId]);

  // ── Project load bridge ───────────────────────────────────
  // When the Sidebar loads a project that contains an analysis
  // blob, it calls useAnalysisStore.requestHydrate(payload). We
  // pick it up here and rebuild local React state.
  const hydratePayload = useAnalysisStore((s) => s.hydrate);
  useEffect(() => {
    if (!hydratePayload) return;
    try {
      const m = hydratePayload.manifest as AnalysisManifest | null;
      if (!m || !Array.isArray(m.tabs)) return;
      const newTabs: CodeTab[] = m.tabs.map((ts) => ({
        id: ts.id,
        name: ts.name,
        measureType: (ts.measureType as MeasureTypeFilter) ?? "all",
        plotType: ts.plotType,
        statTest: ts.statTest,
        code: ts.code,
        plots: ts.plots.map((ps) => {
          const tables: AnalysisTable[] = (ps.tableIds ?? []).map((tid) => {
            const meta = m.tableMeta?.[tid];
            return {
              id: tid,
              name: meta?.name ?? tid,
              csv: hydratePayload.tables?.[tid] ?? "",
            };
          });
          return {
            id: ps.id,
            b64: hydratePayload.plots?.[ps.id] ?? "",
            mainName: ps.mainName,
            collageId: ps.collageId,
            tables: tables.length > 0 ? tables : undefined,
          };
        }),
      }));
      setTabs(newTabs);
      setActiveTabId(m.activeTabId && newTabs.some((t) => t.id === m.activeTabId)
        ? m.activeTabId
        : newTabs[0]?.id ?? "");
      setSelectedPlotKeys(new Set());
      setSelectedPlot(null);
    } catch (err) {
      console.error("[AnalysisDialog] hydrate failed", err);
    } finally {
      // Always consume — even on failure — so we don't retry on every render.
      useAnalysisStore.getState().consumeHydrate();
    }
  }, [hydratePayload]);

  // On open:
  //   • always re-check R + re-default the selected-panels set
  //     (cheap; reflects newly added panels)
  //   • only SEED tabs if none exist yet. The dialog component stays
  //     mounted by its parent, so existing tabs (and their plots,
  //     and the plot-timeline selection / added-to-X tracking)
  //     persist across closes and reopens. Subsequent opens never
  //     clobber the user's tabs.
  //
  // Default seed: split DISTANCE and AREA measurements into two
  // separate tabs — comparing distances against areas is nonsense
  // (different units / magnitudes). If only one kind is present,
  // create one tab. If neither, create one "Code 1" tab with the
  // example rows.
  useEffect(() => {
    if (!open) return;
    checkRStatus(customRPath || undefined);
    const panels = new Set(measurements.map((m) => m.panel));
    setSelectedPanels(panels);

    // Refresh the list of zoom-inset sources flagged include_in_analysis
    // so the "Run as Python" button knows what inputs to feed into the
    // pipeline. Cheap (just metadata, no pixel extraction).
    // Probe MATLAB / Octave availability so we can show / hide
    // the Run MATLAB button. Cached on state; one call per open.
    api.checkMatlab()
      .then((m) => setMatlabInfo({ installed: !!m.installed, kind: m.kind || "" }))
      .catch(() => setMatlabInfo({ installed: false, kind: "" }));

    let insetSourcesForSeed: typeof insetSources = [];
    api.listInsetAnalysisSources()
      .then((r) => {
        const list = r.sources || [];
        setInsetSources(list);
        // Default haze-analysis reference to the FIRST source so the
        // preset is immediately runnable without an extra click.
        if (list.length > 0) {
          setHazeReferenceKey((cur) => cur && list.some((s) => s.key === cur) ? cur : list[0].key);
        } else {
          setHazeReferenceKey("");
        }
        // Stash on the closure so the tab-seeding logic below can
        // pick a Python starter when there are no measurements but
        // the user has flagged insets for analysis.
        insetSourcesForSeed = list;
        // If the dialog opened with no tabs AND no measurements but
        // there ARE insets, seed a Python starter tab so the Run
        // Python button has something sensible to execute.
        if (tabs.length === 0 && measurements.length === 0 && list.length > 0) {
          const id = makeTabId();
          const starter: CodeTab = {
            id,
            name: "Python: insets",
            measureType: "all",
            plotType: "bar",
            statTest: "none",
            code: PYTHON_STARTER_CODE,
            plots: [],
            engine: "python",
          };
          setTabs([starter]);
          setActiveTabId(id);
        }
      })
      .catch(() => setInsetSources([]));
    void insetSourcesForSeed;  // not used outside the promise

    if (tabs.length > 0) return; // user's existing tabs are preserved

    // Build the rows we WOULD see with all panels selected, ignoring
    // selectedPanels state (which we just set via the async setter).
    const buildRows = (filter: MeasureTypeFilter): DataRow[] =>
      measurements
        .filter((m) => panels.has(m.panel))
        .filter((m) => {
          const mt: MeasureRowType = m.type === "area" ? "area" : "distance";
          return filter === "all" || mt === filter;
        })
        .map((m) => {
          const v = m.numeric ?? parseFloat(String(m.value).replace(/[^0-9.\-]/g, ""));
          const mt: MeasureRowType = m.type === "area" ? "area" : "distance";
          return {
            panel: m.panel,
            name: m.name,
            group: m.panel,
            value: Number.isNaN(v) ? null : v,
            unit: m.unit ?? "",
            measureType: mt,
          };
        });

    const hasDistance = measurements.some((m) => m.type !== "area");
    const hasArea = measurements.some((m) => m.type === "area");

    const newTabs: CodeTab[] = [];
    const mk = (name: string, mt: MeasureTypeFilter): CodeTab => {
      const id = makeTabId();
      return {
        id,
        name,
        measureType: mt,
        plotType: "bar",
        statTest: "ttest",
        code: buildRCode("bar", "ttest", buildRows(mt), mt),
        plots: [],
      };
    };

    if (hasDistance && hasArea) {
      newTabs.push(mk("Distances", "distance"));
      newTabs.push(mk("Areas", "area"));
    } else if (hasArea) {
      newTabs.push(mk("Areas", "area"));
    } else if (hasDistance) {
      newTabs.push(mk("Distances", "distance"));
    } else {
      newTabs.push(mk("Code 1", "all"));
    }

    setTabs(newTabs);
    setActiveTabId(newTabs[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, measurements]);

  // CSV still passed to the backend (the generated code defines `data`
  // inline and so overrides it, but the API contract expects a CSV).
  const buildCsv = useCallback(() => {
    const rows = getDataRows();
    const header = "Panel,Name,Group,Value,Unit";
    const body = rows
      .map((r) => [r.panel, r.name, r.group, r.value ?? "", r.unit].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","))
      .join("\n");
    return `${header}\n${body}`;
  }, [getDataRows]);

  // Strip every plot-key tied to a given tab from a Set state.
  // Used for selection bookkeeping when a tab is closed / re-run.
  const dropTabKeys = (id: string) => (prev: Set<string>) => {
    const next = new Set<string>();
    prev.forEach((k) => {
      if (parsePlotKey(k).tabId !== id) next.add(k);
    });
    return next;
  };
  // Strip a single plot-key from a Set state.
  const dropPlotKey = (k: string) => (prev: Set<string>) => {
    if (!prev.has(k)) return prev;
    const next = new Set(prev);
    next.delete(k);
    return next;
  };

  // ── Tab management ──
  const addTab = () => {
    const id = makeTabId();
    setTabs((prev) => {
      // Auto-number ("Code N"); skip any names already in use.
      const taken = new Set(prev.map((t) => t.name));
      let n = prev.length + 1;
      while (taken.has(`Code ${n}`)) n++;
      const tab: CodeTab = {
        id,
        name: `Code ${n}`,
        measureType: "all",
        plotType: "bar",
        statTest: "ttest",
        code: buildRCode("bar", "ttest", getDataRows("all"), "all"),
        plots: [],
      };
      return [...prev, tab];
    });
    setActiveTabId(id);
  };

  const removeTab = (id: string) => {
    setTabs((prev) => {
      if (prev.length <= 1) return prev; // keep at least one tab
      const next = prev.filter((t) => t.id !== id);
      // If we removed the active tab, focus the previous one.
      if (id === activeTabId) {
        const wasIdx = prev.findIndex((t) => t.id === id);
        const newActive = next[Math.max(0, wasIdx - 1)] ?? next[0];
        if (newActive) setActiveTabId(newActive.id);
      }
      return next;
    });
    // Drop any selected plot keys + open preview tied to this tab.
    // (Destination tracking lives on each plot, so removing the tab
    // discards those references with it.)
    setSelectedPlotKeys(dropTabKeys(id));
    setSelectedPlot((cur) => (cur && parsePlotKey(cur).tabId === id ? null : cur));
  };

  const handleRun = async () => {
    if (!activeTab) return;
    // Dispatch based on the tab's engine so the primary Run button
    // does the right thing regardless of whether the tab is R,
    // Python, or MATLAB. The secondary "Run Python/MATLAB" button
    // remains as a way to FORCE-run with a specific engine — handy
    // when the user wants to sanity-check the same code under
    // both, or override a tab's stored engine.
    const eng = activeTab.engine || "r";
    if (eng === "python") { await handleRunPython(); return; }
    if (eng === "matlab") { await handleRunMatlab(); return; }
    setRunning(true);
    // Clear the active tab's plots + any of its keys in the selection.
    // Re-running produces fresh images so we drop tracking for the
    // previous batch — the new plots get fresh ids.
    updateActiveTab({ plots: [] });
    const tabId = activeTab.id;
    setSelectedPlotKeys(dropTabKeys(tabId));
    setSelectedPlot((cur) => (cur && parsePlotKey(cur).tabId === tabId ? null : cur));
    try {
      const result = await api.runR(activeTab.code, buildCsv(), customRPath || undefined);
      // Cache tables (if the backend returned any) onto the FIRST plot
      // for now — the run is one analysis, so the tables apply to its
      // plot batch as a unit. If the backend payload is empty, this is
      // simply [].
      const tablesFromRun: AnalysisTable[] = (result.tables ?? []).map((t) => ({
        id: makeTableId(),
        name: t.name,
        csv: t.csv,
      }));
      const fresh: AnalysisPlot[] = result.plots.map((b64, i) => ({
        id: makePlotId(),
        b64,
        // Attach all tables to every plot in the batch (cheap; the
        // tables are small text). Lets the modal preview show them
        // alongside whichever plot the user clicked.
        tables: tablesFromRun.length > 0 ? tablesFromRun : undefined,
        // mainName / collageId left undefined — fresh plots haven't
        // been sent anywhere yet.
      }));
      // Use a function form so we patch the LATEST tabs state, not
      // a stale closure copy (updateActiveTab depends on activeTabId
      // which is fine, but the tabs map is captured fresh here).
      setTabs((prev) => prev.map((t) => (t.id === tabId ? { ...t, plots: fresh } : t)));
      // Default: every freshly-generated plot starts check-selected.
      setSelectedPlotKeys((prev) => {
        const next = new Set(prev);
        fresh.forEach((p) => next.add(plotKey(tabId, p.id)));
        return next;
      });
      const out = (result.stdout || "") + (result.stderr ? `\n${result.stderr}` : "");
      setConsoleOut((prev) => prev + `\n=== Run [${activeTab.name}] ===\n` + (out.trim() || "(no console output)") + "\n");
    } catch (err) {
      setConsoleOut((prev) => prev + `\n=== Run [${activeTab.name}] failed ===\n` + (err instanceof Error ? err.message : String(err)) + "\n");
    } finally {
      setRunning(false);
    }
  };

  // Run the ACTIVE tab's code as a Python pipeline against every
  // zoom inset flagged `include_in_analysis`. Outputs route into the
  // same plot timeline + table cache that R runs use; modified
  // images go into `pyImages` (rendered below the data panel).
  // Push a pipeline-output table into a new R tab. The R script
  // reads the CSV INLINE via `read.csv(text = "...")` so it
  // travels with the tab (round-trips through save/load and is
  // self-contained — no fragile filesystem path). The default
  // plot is a bar chart of the first numeric column grouped by
  // the first string column; the user can edit afterwards.
  const sendTableToRTab = useCallback((tbl: AnalysisTable) => {
    const id = makeTabId();
    const lines = (tbl.csv || "").split(/\r?\n/).filter((l) => l.length > 0);
    const header = (lines[0] || "").split(",");
    // Detect first non-source column as the default y axis.
    const stringCol = header[0] || "source";
    let yCol = "value";
    if (lines.length > 1) {
      const firstRow = lines[1].split(",");
      for (let i = 0; i < header.length; i++) {
        const v = firstRow[i];
        if (v !== undefined && v !== "" && !Number.isNaN(parseFloat(v.replace(/[^0-9.\-]/g, "")))) {
          yCol = header[i];
          break;
        }
      }
    }
    const escapedCsv = (tbl.csv || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    const code = `# ============================================================
#  R plot for pipeline output: ${tbl.name}
#
#  CSV was generated by the Python / MATLAB pipeline above and
#  pasted inline below. Edit the ggplot block to taste; the
#  default is a bar chart of the first numeric column.
# ============================================================
library(ggplot2)
library(ggprism)

data <- read.csv(text = "${escapedCsv}", stringsAsFactors = FALSE)

mpfig_plot("${tbl.name}.png")
ggplot(data, aes(x = ${stringCol}, y = ${yCol})) +
  geom_col(aes(fill = ${stringCol}), width = 0.7) +
  theme_prism() +
  labs(title = "${tbl.name}", x = NULL, y = "${yCol}") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
`;
    setTabs((prev) => {
      const taken = new Set(prev.map((t) => t.name));
      let candidate = `R: ${tbl.name}`;
      let i = 2;
      while (taken.has(candidate)) { candidate = `R: ${tbl.name} (${i++})`; }
      const tab: CodeTab = {
        id,
        name: candidate,
        measureType: "all",
        plotType: "bar",
        statTest: "none",
        code,
        plots: [],
        engine: "r",
      };
      return [...prev, tab];
    });
    setActiveTabId(id);
    setConsoleOut((prev) => prev + `\n=== Sent table '${tbl.name}' to a new R tab. Press Run to plot. ===\n`);
  }, []);

  // Generate the preset's starter code for the current engine /
  // reference choice. Single source of truth used by:
  //   • Engine swap auto-fill (Python ↔ MATLAB)
  //   • Preset change auto-fill
  //   • Reference change auto-fill (haze analysis)
  const generatePresetCode = useCallback(
    (engine: "python" | "matlab", preset: PipelinePreset, refKey: string): string => {
      const spec = PIPELINE_PRESETS[preset];
      const ref = spec.needsReference ? (refKey || null) : null;
      return engine === "python" ? spec.generatePython(ref) : spec.generateMatlab(ref);
    },
    [],
  );

  // When the user switches tabs, sync the engine selector to the
  // tab's stored engine (if it has a python / matlab one). R tabs
  // leave the selector at its last value so flipping between an R
  // and a Python tab doesn't lose the Python/MATLAB choice.
  useEffect(() => {
    if (!activeTab) return;
    const eng = activeTab.engine;
    if (eng === "python" || eng === "matlab") {
      setPipelineEngine(eng);
    }
  }, [activeTabId, activeTab?.engine]);

  // Whenever the user toggles the preset OR (for haze) picks a new
  // reference, replace the active tab's code IF it still matches
  // any of the other preset×engine outputs — keeps custom edits
  // intact while allowing rapid preset swapping.
  useEffect(() => {
    if (!activeTab || !open) return;
    const cur = activeTab.code || "";
    const matchesAnyPreset = (Object.keys(PIPELINE_PRESETS) as PipelinePreset[]).some((pk) => {
      const spec = PIPELINE_PRESETS[pk];
      const py = spec.generatePython(spec.needsReference ? hazeReferenceKey || null : null).trim();
      const ml = spec.generateMatlab(spec.needsReference ? hazeReferenceKey || null : null).trim();
      return cur.trim() === py || cur.trim() === ml;
    });
    if (!matchesAnyPreset && cur.trim().length > 0) return;  // user-edited — leave alone
    const next = generatePresetCode(pipelineEngine, pipelinePreset, hazeReferenceKey);
    if (next !== cur) updateActiveTab({ code: next });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipelinePreset, hazeReferenceKey, pipelineEngine, activeTabId]);

  const handleRunPython = async () => {
    if (!activeTab) return;
    if (insetSources.length === 0) {
      setConsoleOut((prev) => prev +
        "\n=== Run as Python ===\nNo zoom insets are marked 'Include in Analysis'. " +
        "Open an Edit Panel → Zoom Inset tab and tick the checkbox first.\n");
      return;
    }
    setRunningPy(true);
    updateActiveTab({ plots: [] });
    const tabId = activeTab.id;
    setSelectedPlotKeys(dropTabKeys(tabId));
    setSelectedPlot((cur) => (cur && parsePlotKey(cur).tabId === tabId ? null : cur));
    setPyImages([]);
    setPyTables([]);
    try {
      const result = await api.runPython(
        activeTab.code,
        insetSources.map((s) => ({
          key: s.key, row: s.row, col: s.col, inset_index: s.inset_index, label: s.label,
        })),
      );
      const tablesFromRun: AnalysisTable[] = (result.tables ?? []).map((t) => ({
        id: makeTableId(), name: t.name, csv: t.csv,
      }));
      const fresh: AnalysisPlot[] = result.plots.map((b64) => ({
        id: makePlotId(),
        b64,
        tables: tablesFromRun.length > 0 ? tablesFromRun : undefined,
      }));
      setTabs((prev) => prev.map((t) => (t.id === tabId ? { ...t, plots: fresh } : t)));
      setSelectedPlotKeys((prev) => {
        const next = new Set(prev);
        fresh.forEach((p) => next.add(plotKey(tabId, p.id)));
        return next;
      });
      setPyImages(result.images || []);
      setPyTables(tablesFromRun);
      const out = (result.stdout || "") + (result.stderr ? `\n${result.stderr}` : "");
      setConsoleOut((prev) => prev +
        `\n=== Run Python [${activeTab.name}] (${insetSources.length} source${insetSources.length === 1 ? "" : "s"}) ===\n` +
        (out.trim() || "(no console output)") + "\n");
    } catch (err) {
      setConsoleOut((prev) => prev + `\n=== Run Python [${activeTab.name}] failed ===\n` +
        (err instanceof Error ? err.message : String(err)) + "\n");
    } finally {
      setRunningPy(false);
    }
  };

  // Run the ACTIVE tab's code as a MATLAB / Octave pipeline against
  // every flagged zoom inset. Mirrors handleRunPython end-to-end —
  // same flow into the plot timeline + Python-images strip — only
  // the backend interpreter differs.
  const handleRunMatlab = async () => {
    if (!activeTab) return;
    if (insetSources.length === 0) {
      setConsoleOut((prev) => prev +
        "\n=== Run as MATLAB ===\nNo zoom insets are marked 'Include in Analysis'. " +
        "Open an Edit Panel → Zoom Inset tab and tick the checkbox first.\n");
      return;
    }
    if (!matlabInfo.installed) {
      setConsoleOut((prev) => prev +
        "\n=== Run as MATLAB ===\nNo MATLAB / Octave interpreter detected. " +
        "Install Octave (free) from https://octave.org/ or MATLAB from MathWorks.\n");
      return;
    }
    setRunningMatlab(true);
    updateActiveTab({ plots: [] });
    const tabId = activeTab.id;
    setSelectedPlotKeys(dropTabKeys(tabId));
    setSelectedPlot((cur) => (cur && parsePlotKey(cur).tabId === tabId ? null : cur));
    setPyImages([]);
    setPyTables([]);
    try {
      const result = await api.runMatlab(
        activeTab.code,
        insetSources.map((s) => ({
          key: s.key, row: s.row, col: s.col, inset_index: s.inset_index, label: s.label,
        })),
      );
      const tablesFromRun: AnalysisTable[] = (result.tables ?? []).map((t) => ({
        id: makeTableId(), name: t.name, csv: t.csv,
      }));
      const fresh: AnalysisPlot[] = result.plots.map((b64) => ({
        id: makePlotId(),
        b64,
        tables: tablesFromRun.length > 0 ? tablesFromRun : undefined,
      }));
      setTabs((prev) => prev.map((t) => (t.id === tabId ? { ...t, plots: fresh } : t)));
      setSelectedPlotKeys((prev) => {
        const next = new Set(prev);
        fresh.forEach((p) => next.add(plotKey(tabId, p.id)));
        return next;
      });
      setPyImages(result.images || []);
      setPyTables(tablesFromRun);
      const kind = result.kind ? `(${result.kind})` : "";
      const out = (result.stdout || "") + (result.stderr ? `\n${result.stderr}` : "");
      setConsoleOut((prev) => prev +
        `\n=== Run MATLAB ${kind} [${activeTab.name}] (${insetSources.length} source${insetSources.length === 1 ? "" : "s"}) ===\n` +
        (out.trim() || "(no console output)") + "\n");
    } catch (err) {
      setConsoleOut((prev) => prev + `\n=== Run MATLAB [${activeTab.name}] failed ===\n` +
        (err instanceof Error ? err.message : String(err)) + "\n");
    } finally {
      setRunningMatlab(false);
    }
  };

  // Run an ad-hoc R command from the inline console (package installs etc.)
  const handleRunConsole = async () => {
    const cmd = consoleCmd.trim();
    if (!cmd) return;
    setConsoleRunning(true);
    setConsoleOut((prev) => prev + (prev && !prev.endsWith("\n") ? "\n" : "") + "> " + cmd + "\n");
    try {
      const res = await api.runRConsole(cmd, customRPath || undefined);
      const out = (res.stdout || "") + (res.stderr ? `\n${res.stderr}` : "");
      setConsoleOut((prev) => prev + (out.trim() || "(done — no output)") + "\n");
    } catch (err) {
      setConsoleOut((prev) => prev + "Error: " + (err instanceof Error ? err.message : String(err)) + "\n");
    } finally {
      setConsoleRunning(false);
      setConsoleCmd("");
    }
  };

  const browseForR = async () => {
    try {
      const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
      const selected = await openDialog({
        multiple: false,
        filters: [{ name: "Rscript", extensions: ["*"] }],
      });
      if (selected) {
        const path = typeof selected === "string" ? selected : (selected as { path: string }).path;
        if (path) {
          setRPathInput(path);
          checkRStatus(path);
        }
      }
    } catch {
      // Not in Tauri context
    }
  };

  const handlePlotChange = (key: string) => {
    if (!activeTab) return;
    const mt = activeTab.measureType;
    updateActiveTab({ plotType: key, code: buildRCode(key, activeTab.statTest, getDataRows(mt), mt) });
  };
  const handleStatChange = (key: string) => {
    if (!activeTab) return;
    const mt = activeTab.measureType;
    updateActiveTab({ statTest: key, code: buildRCode(activeTab.plotType, key, getDataRows(mt), mt) });
  };
  const handleMeasureTypeChange = (mt: MeasureTypeFilter) => {
    if (!activeTab) return;
    updateActiveTab({
      measureType: mt,
      code: buildRCode(activeTab.plotType, activeTab.statTest, getDataRows(mt), mt),
    });
  };

  const togglePanel = (panel: string) => {
    setSelectedPanels((prev) => {
      const next = new Set(prev);
      if (next.has(panel)) next.delete(panel);
      else next.add(panel);
      return next;
    });
  };

  const addManualRow = () => setManualRows((prev) => [...prev, manualCols.map(() => "")]);

  const updateManualCell = (rowIdx: number, colIdx: number, value: string) => {
    setManualRows((prev) => {
      const next = [...prev];
      next[rowIdx] = [...next[rowIdx]];
      next[rowIdx][colIdx] = value;
      return next;
    });
  };

  const deleteManualRow = (rowIdx: number) =>
    setManualRows((prev) => prev.filter((_, i) => i !== rowIdx));

  // A stable, descriptive base name for one plot — used for filenames
  // (downloads, main-timeline uploads, collage labels).
  const plotName = (tabName: string, plotType: string, statTest: string, idx: number) =>
    `analysis_${tabName.replace(/\s+/g, "_")}_${plotType}_${statTest}_${idx + 1}`;

  const savePlot = (b64: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${b64}`;
    link.download = `${filename}.png`;
    link.click();
  };

  // ── Plot selection (checkboxes) ───────────────────────────
  const toggleSelectPlot = (k: string) => {
    setSelectedPlotKeys((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };
  const allPlotsSelected = allPlots.length > 0 && selectedPlotKeys.size === allPlots.length;
  const toggleSelectAllPlots = () => {
    setSelectedPlotKeys(allPlotsSelected ? new Set() : new Set(allPlots.map((p) => p.key)));
  };
  // The full plot objects that are currently checked, in display order.
  const selectedPlotList = () => allPlots.filter((p) => selectedPlotKeys.has(p.key));

  // Download the SELECTED plots to the user's disk.
  const downloadSelectedPlots = () => {
    selectedPlotList().forEach((p) => savePlot(p.b64, plotName(p.tabName, p.plotType, p.statTest, p.idx)));
  };

  // Mutate a single plot inside the tabs tree (id-keyed).
  const patchPlot = (tabId: string, plotId: string, patch: Partial<AnalysisPlot>) => {
    setTabs((prev) =>
      prev.map((t) =>
        t.id === tabId
          ? { ...t, plots: t.plots.map((pp) => (pp.id === plotId ? { ...pp, ...patch } : pp)) }
          : t,
      ),
    );
  };

  // Move the SELECTED plots into the MAIN app's image timeline
  // (loadedImages) via the normal upload path. Skips any plot that
  // already has a `mainName` (i.e. has been added before); a flash
  // message surfaces the dedup result. On success, the returned
  // canonical names are stored on each plot so the discard flow
  // can find them again to remove from the timeline.
  const moveSelectedToMainTimeline = async () => {
    const list = selectedPlotList();
    if (list.length === 0) return;
    const already = list.filter((p) => !!p.mainName);
    const fresh = list.filter((p) => !p.mainName);
    if (fresh.length === 0) {
      showFlash(
        already.length === 1
          ? "This plot is already in the main timeline."
          : `All ${already.length} selected plots are already in the main timeline.`,
        "warning",
      );
      return;
    }
    const stamp = Date.now();
    const files = fresh.map((p) =>
      b64ToFile(p.b64, `${plotName(p.tabName, p.plotType, p.statTest, p.idx)}_${stamp}.png`),
    );
    try {
      const names = await useFigureStore.getState().uploadImages(files);
      // Map each file → canonical returned name (best-effort by index).
      fresh.forEach((p, i) => {
        const name = names[i] ?? files[i].name;
        patchPlot(p.tabId, p.plotId, { mainName: name });
      });
      if (already.length > 0) {
        showFlash(
          `${fresh.length} added to main timeline — ${already.length} already there (skipped).`,
          "info",
        );
      } else {
        showFlash(
          `${fresh.length} plot${fresh.length === 1 ? "" : "s"} added to main timeline.`,
          "success",
        );
      }
      setConsoleOut((prev) => prev + `\n[${fresh.length} plot(s) added to the main image timeline${already.length ? `; ${already.length} skipped (already there)` : ""}]\n`);
    } catch (err) {
      setConsoleOut((prev) => prev + `\n[Failed to add plots to timeline: ${err instanceof Error ? err.message : String(err)}]\n`);
    }
  };

  // Move the SELECTED plots into the Collage Assembly canvas as image
  // items. They're flagged `fromAnalysis` so app-close can warn the user
  // — the collage builder doesn't yet persist to disk. Skips any plot
  // that already has a `collageId`. The returned item id is stored on
  // each plot so discard can find + remove it.
  const moveSelectedToCollage = async () => {
    const list = selectedPlotList();
    if (list.length === 0) return;
    const already = list.filter((p) => !!p.collageId);
    const fresh = list.filter((p) => !p.collageId);
    if (fresh.length === 0) {
      showFlash(
        already.length === 1
          ? "This plot is already in the Collage Assembly."
          : `All ${already.length} selected plots are already in the Collage Assembly.`,
        "warning",
      );
      return;
    }
    const addItem = useCollageStore.getState().addItem;
    let placed = 0;
    for (const p of fresh) {
      const { w, h } = await imageDims(p.b64);
      const scale = Math.min(1, 700 / Math.max(w, h));
      const newId = addItem({
        kind: "image",
        fromAnalysis: true,
        src: `data:image/png;base64,${p.b64}`,
        name: plotName(p.tabName, p.plotType, p.statTest, p.idx),
        x: 60 + placed * 40,
        y: 60 + placed * 40,
        w: Math.round(w * scale),
        h: Math.round(h * scale),
        naturalW: w,
        naturalH: h,
        // Provenance so the collage can re-render this R plot at a new
        // font size via "Synchronize headers". Data CSV is frozen at
        // add-time so later measurement edits don't change this plot.
        rCode: p.code,
        rDataCsv: buildCsv(),
        rInterpreter: customRPath || null,
        rPlotIndex: p.idx,
      });
      patchPlot(p.tabId, p.plotId, { collageId: newId });
      placed++;
    }
    if (already.length > 0) {
      showFlash(
        `${fresh.length} added to Collage — ${already.length} already there (skipped).`,
        "info",
      );
    } else {
      showFlash(
        `${fresh.length} plot${fresh.length === 1 ? "" : "s"} added to Collage Assembly.`,
        "success",
      );
    }
    setConsoleOut((prev) => prev +
      `\n[${fresh.length} plot(s) added to the Collage Assembly${already.length ? `; ${already.length} skipped (already there)` : ""}. ` +
      `NOTE: the collage builder doesn't save to disk yet — use "Download" to keep these plots.]\n`);
  };

  // ── Discard a plot ───────────────────────────────────────
  // Triggered by the trash icon on each thumbnail. If the plot has
  // been pushed to the main timeline or the Collage Assembly, open
  // the confirm modal first; otherwise drop it immediately.
  const requestDiscardPlot = (key: string) => {
    const p = allPlots.find((x) => x.key === key);
    if (!p) return;
    const inMain = !!p.mainName;
    const inCollage = !!p.collageId;
    if (!inMain && !inCollage) {
      doDiscardPlot(key, false, false);
      return;
    }
    setDiscardConfirm({
      plotKey: key,
      plotName: plotName(p.tabName, p.plotType, p.statTest, p.idx),
      inMain,
      inCollage,
    });
  };

  const doDiscardPlot = async (key: string, removeFromMain: boolean, removeFromCollage: boolean) => {
    const p = allPlots.find((x) => x.key === key);
    if (!p) return;
    // 1. Cascade-remove from destinations (best-effort; failures are
    //    logged to the console but don't block the local discard).
    if (removeFromMain && p.mainName) {
      try {
        await useFigureStore.getState().removeImage(p.mainName);
      } catch (err) {
        setConsoleOut((prev) => prev + `\n[Failed to remove ${p.mainName} from main timeline: ${err instanceof Error ? err.message : String(err)}]\n`);
      }
    }
    if (removeFromCollage && p.collageId) {
      try {
        useCollageStore.getState().removeItem(p.collageId);
      } catch (err) {
        setConsoleOut((prev) => prev + `\n[Failed to remove plot from Collage: ${err instanceof Error ? err.message : String(err)}]\n`);
      }
    }
    // 2. Drop the plot from its tab.
    setTabs((prev) =>
      prev.map((t) =>
        t.id === p.tabId ? { ...t, plots: t.plots.filter((pp) => pp.id !== p.plotId) } : t,
      ),
    );
    // 3. Drop the plot from the selection + close any preview / confirm tied to it.
    setSelectedPlotKeys(dropPlotKey(p.key));
    setSelectedPlot((cur) => (cur === p.key ? null : cur));
    setDiscardConfirm(null);
    const dests = [
      removeFromMain && p.mainName ? "main timeline" : null,
      removeFromCollage && p.collageId ? "Collage" : null,
    ].filter(Boolean) as string[];
    showFlash(
      dests.length > 0
        ? `Discarded plot — also removed from ${dests.join(" + ")}.`
        : "Plot discarded.",
      "info",
    );
  };

  const panels = [...new Set(measurements.map((m) => m.panel))];
  const dataRowCount = getDataRows().length;
  const previewPlot = selectedPlot ? allPlots.find((p) => p.key === selectedPlot) : undefined;

  return (
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2, bgcolor: "background.paper" }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>Analysis</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
        <AnalysisNodeGraph open={open} measurementsCsv={buildCsv()} />
      </DialogContent>

      {/* Plot preview modal — opens on thumbnail click */}
      <Dialog
        open={previewPlot !== undefined}
        onClose={() => setSelectedPlot(null)}
        maxWidth="lg"
      >
        <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>
            {previewPlot ? `${plotName(previewPlot.tabName, previewPlot.plotType, previewPlot.statTest, previewPlot.idx)}.png` : ""}
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
            <Button size="small" startIcon={<SaveAltIcon sx={{ fontSize: 14 }} />}
              sx={{ fontSize: "0.6rem", textTransform: "none" }}
              onClick={() => {
                if (previewPlot) savePlot(previewPlot.b64, plotName(previewPlot.tabName, previewPlot.plotType, previewPlot.statTest, previewPlot.idx));
              }}
            >Download</Button>
            <IconButton size="small" onClick={() => setSelectedPlot(null)}><CloseIcon fontSize="small" /></IconButton>
          </Box>
        </DialogTitle>
        <DialogContent sx={{ p: 1 }}>
          {previewPlot && (
            <>
              <img
                src={`data:image/png;base64,${previewPlot.b64}`}
                alt={`${previewPlot.tabName} Plot ${previewPlot.idx + 1}`}
                style={{ display: "block", maxWidth: "85vw", maxHeight: "60vh", height: "auto", width: "auto" }}
              />
              {/* ── Data & statistics tables, if the R run emitted any ── */}
              {previewPlot.tables && previewPlot.tables.length > 0 && (
                <Box sx={{ mt: 1, display: "flex", flexDirection: "column", gap: 1, maxHeight: "30vh", overflowY: "auto" }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, fontSize: "0.65rem" }}>
                    Data & Statistics
                  </Typography>
                  {previewPlot.tables.map((t) => {
                    const grid = parseCsv(t.csv);
                    const header = grid[0] ?? [];
                    const body = grid.slice(1).filter((r) => r.some((c) => c !== ""));
                    return (
                      <Box key={t.id} sx={{ border: 1, borderColor: "divider", borderRadius: 1, overflow: "hidden" }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1, py: 0.5, bgcolor: "background.default" }}>
                          <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.65rem" }}>{t.name}</Typography>
                          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.disabled" }}>
                            {body.length} row{body.length === 1 ? "" : "s"} · {header.length} column{header.length === 1 ? "" : "s"}
                          </Typography>
                          <Box sx={{ flex: 1 }} />
                          <Button size="small" startIcon={<DownloadIcon sx={{ fontSize: 12 }} />}
                            onClick={() => downloadCsv(t.csv, `${plotName(previewPlot.tabName, previewPlot.plotType, previewPlot.statTest, previewPlot.idx)}_${t.name}`)}
                            sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.25, minWidth: 0 }}>
                            Export CSV
                          </Button>
                        </Box>
                        <Box sx={{ maxHeight: 200, overflow: "auto" }}>
                          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.6rem", fontFamily: "monospace" }}>
                            <thead>
                              <tr>
                                {header.map((h, i) => (
                                  <th key={i} style={{ padding: "2px 6px", borderBottom: "1px solid var(--c-border)", textAlign: "left", position: "sticky", top: 0, background: "var(--c-surface)", whiteSpace: "nowrap" }}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {body.map((row, ri) => (
                                <tr key={ri}>
                                  {row.map((cell, ci) => (
                                    <td key={ci} style={{ padding: "1px 6px", borderBottom: "1px solid var(--c-border)", whiteSpace: "nowrap" }}>{cell}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </Box>
                      </Box>
                    );
                  })}
                </Box>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* ── Discard-plot confirm modal ──────────────────────────
          Opens when the user clicks the trash icon on a plot that
          has previously been pushed to the main timeline or the
          Collage. The user picks per-destination whether to also
          remove the copy there. */}
      <Dialog open={discardConfirm !== null} onClose={() => setDiscardConfirm(null)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ fontSize: "0.9rem", py: 1, px: 2 }}>Discard plot?</DialogTitle>
        <DialogContent sx={{ pt: 1, pb: 0 }}>
          {discardConfirm && (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <code style={{ fontSize: "0.75rem" }}>{discardConfirm.plotName}.png</code>
              </Typography>
              <Typography variant="caption" sx={{ color: "warning.main", display: "block", mb: 1 }}>
                This plot has been added to other places. Choose what else to remove:
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                {discardConfirm.inMain && (
                  <FormControlLabel
                    control={
                      <Checkbox
                        defaultChecked
                        size="small"
                        inputProps={{ "aria-label": "Also remove from main timeline" }}
                        id="discard-remove-main"
                      />
                    }
                    label={<span style={{ fontSize: "0.75rem" }}>Also remove from <strong>main image timeline</strong></span>}
                  />
                )}
                {discardConfirm.inCollage && (
                  <FormControlLabel
                    control={
                      <Checkbox
                        defaultChecked
                        size="small"
                        inputProps={{ "aria-label": "Also remove from Collage" }}
                        id="discard-remove-collage"
                      />
                    }
                    label={<span style={{ fontSize: "0.75rem" }}>Also remove from <strong>Collage Assembly</strong></span>}
                  />
                )}
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 2, py: 1 }}>
          <Button size="small" onClick={() => setDiscardConfirm(null)} sx={{ textTransform: "none" }}>Cancel</Button>
          <Button
            size="small"
            variant="contained"
            color="error"
            sx={{ textTransform: "none" }}
            onClick={() => {
              if (!discardConfirm) return;
              const wantMain = discardConfirm.inMain &&
                (document.getElementById("discard-remove-main") as HTMLInputElement | null)?.checked !== false;
              const wantCollage = discardConfirm.inCollage &&
                (document.getElementById("discard-remove-collage") as HTMLInputElement | null)?.checked !== false;
              doDiscardPlot(discardConfirm.plotKey, wantMain, wantCollage);
            }}
          >
            Discard
          </Button>
        </DialogActions>
      </Dialog>
    </Dialog>
  );
}
