/* ──────────────────────────────────────────────────────────
   AnalysisNodeGraph — node-graph editor for the Analysis tab.

   Replaces the old multi-tab editor. Workflow is a DAG of typed
   data streams:
     • Source node → exposes flagged zoom-inset images +
       line/area measurements as draggable output ports.
     • Python / MATLAB / R nodes → consume any mix of upstream
       image/table edges, produce CSVs (+ images for Py/Matlab,
       + plots for R).
     • Output drawer (bottom) → aggregates every node's outputs
       so the user can pin / re-route them into the main figure.

   Plots are R-only by design: the user wants pixel analysis
   on the Python/MATLAB side and publication-quality plotting
   on the R side. `mpfig_plot()` is intentionally a no-op in
   the Py / MATLAB harnesses (see /api/analysis/run-python).
   ────────────────────────────────────────────────────────── */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  type Node,
  type Edge,
  type Connection,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Box,
  Typography,
  Button,
  IconButton,
  Select,
  MenuItem,
  Tooltip,
  CircularProgress,
  Tabs,
  Tab,
  Drawer,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AddIcon from "@mui/icons-material/Add";
import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";
import DownloadIcon from "@mui/icons-material/Download";
import CodeMirror from "@uiw/react-codemirror";
import { python as cmPython } from "@codemirror/lang-python";
import { StreamLanguage } from "@codemirror/language";
import { r as cmR } from "@codemirror/legacy-modes/mode/r";
import { oneDark } from "@codemirror/theme-one-dark";
import { api } from "../../api/client";
import { useFigureStore } from "../../store/figureStore";

// ── Types ────────────────────────────────────────────────────

export type DataKind = "image" | "table" | "plot";
export type EngineKind = "python" | "matlab" | "r";

export interface InsetSource {
  key: string;
  row: number;
  col: number;
  inset_index: number;
  label: string;
  natural_width: number;
  natural_height: number;
  thumbnail: string;
}

export interface NodeOutput {
  /** Stable id within a node — `out_image_<idx>`, `out_table_<name>`,
   *  `out_plot_<idx>`. Used as the handle id. */
  id: string;
  kind: DataKind;
  name: string;
  /** PNG bytes (base64) for `image` / `plot`, CSV text for `table`. */
  payload: string;
  /** Whether the user pinned this output (sticky in the drawer's
   *  summary view; pinned items survive node re-runs). */
  pinned?: boolean;
}

export interface NodeData {
  /** Display label on the node card. */
  label: string;
  /** "source" | "python" | "matlab" | "r" */
  kind: "source" | EngineKind;
  /** Engine-specific code (only relevant for python/matlab/r). */
  code?: string;
  /** Last-run outputs, populated after a successful execution. */
  outputs?: NodeOutput[];
  /** Per-input handle: which upstream `nodeId:outputId` it's
   *  connected to. Populated from edges; cached so the runner
   *  knows what to feed in. */
  inputs?: { name: string; sourceNodeId: string; sourceOutputId: string; kind: DataKind }[];
  status?: "idle" | "running" | "ok" | "error" | "stale";
  /** Last error message (when status === "error"). */
  error?: string;
  /** For source nodes only — list of inset sources to expose as
   *  output handles. */
  sources?: InsetSource[];
  /** When true, the user has explicitly removed the node from
   *  the canvas — used to no-op stale auto-edges. */
  deleted?: boolean;
  [key: string]: unknown;  // index signature for ReactFlow's Record
}

// ── Helpers ──────────────────────────────────────────────────

const newId = (prefix: string) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;

const KIND_COLOR: Record<"source" | EngineKind, string> = {
  source: "#673ab7",
  python: "#1976d2",
  matlab: "#ef6c00",
  r: "#2e7d32",
};

const KIND_ICON: Record<"source" | EngineKind, string> = {
  source: "📥",
  python: "🐍",
  matlab: "📐",
  r: "📊",
};

const PORT_COLOR: Record<DataKind, string> = {
  image: "#26a69a",
  table: "#ab47bc",
  plot: "#ffa726",
};

// ── Node card components ─────────────────────────────────────

interface NodeCardProps {
  data: NodeData;
  id: string;
  selected?: boolean;
}

function StatusPip({ status }: { status?: NodeData["status"] }) {
  const colour = status === "running" ? "#ffb300"
    : status === "ok" ? "#43a047"
    : status === "error" ? "#e53935"
    : status === "stale" ? "#9e9e9e"
    : "#cfd8dc";
  const label = status === "running" ? "running"
    : status === "ok" ? "fresh"
    : status === "error" ? "error"
    : status === "stale" ? "stale"
    : "idle";
  return (
    <Tooltip placement="top" title={label}>
      <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: colour }} />
    </Tooltip>
  );
}

/** Source node — exposes inset thumbnails as draggable output ports. */
function SourceNode({ data }: NodeCardProps) {
  const sources = data.sources || [];
  return (
    <Box sx={{
      minWidth: 200,
      bgcolor: "background.paper",
      border: "2px solid",
      borderColor: KIND_COLOR.source,
      borderRadius: 1,
      boxShadow: 2,
    }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.5, bgcolor: KIND_COLOR.source, color: "white", borderRadius: "4px 4px 0 0" }}>
        <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 700 }}>
          {KIND_ICON.source} Source
        </Typography>
        <Box sx={{ flex: 1 }} />
        <StatusPip status={data.status} />
      </Box>
      <Box sx={{ p: 0.5 }}>
        {sources.length === 0 ? (
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", fontStyle: "italic", display: "block", px: 1, py: 0.5 }}>
            No flagged insets. Open Edit Panel → Zoom Inset tab and tick "Include in Analysis".
          </Typography>
        ) : (
          sources.map((s, idx) => (
            <Box key={s.key} sx={{ position: "relative", display: "flex", alignItems: "center", gap: 0.5, py: 0.25, pr: 1.5 }}>
              {s.thumbnail && (
                <Box component="img" src={`data:image/png;base64,${s.thumbnail}`} alt={s.label}
                  sx={{ width: 24, height: 24, objectFit: "contain", border: "1px solid", borderColor: "divider", borderRadius: 0.25 }}
                />
              )}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="caption" sx={{ fontSize: "0.62rem", fontWeight: 600, display: "block", lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  R{s.row + 1}C{s.col + 1}·{s.inset_index + 1}
                </Typography>
                <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", lineHeight: 1 }}>
                  {s.natural_width}×{s.natural_height}
                </Typography>
              </Box>
              <Handle
                type="source"
                position={Position.Right}
                id={`out_image_${idx}`}
                style={{ background: PORT_COLOR.image, width: 10, height: 10, border: "2px solid white", right: -5 }}
              />
            </Box>
          ))
        )}
        {/* Measurements port — single table output. Bound to the
            R-tab measurements CSV via the runner. */}
        <Box sx={{ position: "relative", display: "flex", alignItems: "center", gap: 0.5, py: 0.5, pr: 1.5, borderTop: "1px dashed", borderColor: "divider" }}>
          <Typography variant="caption" sx={{ fontSize: "0.62rem", fontWeight: 600, flex: 1 }}>📋 measurements</Typography>
          <Handle
            type="source"
            position={Position.Right}
            id="out_table_measurements"
            style={{ background: PORT_COLOR.table, width: 10, height: 10, border: "2px solid white", right: -5 }}
          />
        </Box>
      </Box>
    </Box>
  );
}

/** Process node — Python / MATLAB / R. */
function ProcessNode({ data }: NodeCardProps) {
  const engine = data.kind as EngineKind;
  const colour = KIND_COLOR[engine];
  const inputs = data.inputs || [];
  const outputs = data.outputs || [];
  // Allow plot outputs ONLY for R nodes (Python/MATLAB are
  // data-extraction by design).
  const allowedOutKinds: DataKind[] = engine === "r" ? ["plot", "table"] : ["image", "table"];
  void allowedOutKinds;

  return (
    <Box sx={{
      minWidth: 220,
      bgcolor: "background.paper",
      border: "2px solid",
      borderColor: colour,
      borderRadius: 1,
      boxShadow: 2,
    }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.5, bgcolor: colour, color: "white", borderRadius: "4px 4px 0 0" }}>
        <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 700 }}>
          {KIND_ICON[engine]} {data.label}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <StatusPip status={data.status} />
      </Box>
      <Box sx={{ display: "flex" }}>
        {/* Input ports (left edge) */}
        <Box sx={{ position: "relative", width: 18, py: 0.5 }}>
          {/* image input */}
          <Handle
            type="target"
            position={Position.Left}
            id="in_image"
            style={{ background: PORT_COLOR.image, top: 24, width: 10, height: 10, border: "2px solid white", left: -5 }}
          />
          {/* table input */}
          <Handle
            type="target"
            position={Position.Left}
            id="in_table"
            style={{ background: PORT_COLOR.table, top: 52, width: 10, height: 10, border: "2px solid white", left: -5 }}
          />
        </Box>
        <Box sx={{ flex: 1, p: 0.5, minWidth: 0 }}>
          <Box sx={{ fontSize: 11 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, py: 0.25 }}>
              <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: PORT_COLOR.image }} />
              <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>image in ({inputs.filter((i) => i.kind === "image").length})</Typography>
            </Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, py: 0.25 }}>
              <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: PORT_COLOR.table }} />
              <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>table in ({inputs.filter((i) => i.kind === "table").length})</Typography>
            </Box>
            {outputs.length > 0 && (
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", mt: 0.25, fontStyle: "italic" }}>
                {outputs.length} output{outputs.length === 1 ? "" : "s"}
              </Typography>
            )}
            {data.error && (
              <Tooltip placement="top" title={data.error}>
                <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "error.main", display: "block", mt: 0.25, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  ⚠ {data.error}
                </Typography>
              </Tooltip>
            )}
          </Box>
        </Box>
        {/* Output ports (right edge) — three handles per kind so
            the user can route plot/table/image separately. */}
        <Box sx={{ position: "relative", width: 18, py: 0.5 }}>
          {engine !== "r" && (
            <Handle
              type="source"
              position={Position.Right}
              id="out_image"
              style={{ background: PORT_COLOR.image, top: 24, width: 10, height: 10, border: "2px solid white", right: -5 }}
            />
          )}
          <Handle
            type="source"
            position={Position.Right}
            id="out_table"
            style={{ background: PORT_COLOR.table, top: engine === "r" ? 24 : 52, width: 10, height: 10, border: "2px solid white", right: -5 }}
          />
          {engine === "r" && (
            <Handle
              type="source"
              position={Position.Right}
              id="out_plot"
              style={{ background: PORT_COLOR.plot, top: 52, width: 10, height: 10, border: "2px solid white", right: -5 }}
            />
          )}
        </Box>
      </Box>
    </Box>
  );
}

const nodeTypes = {
  source: SourceNode,
  python: ProcessNode,
  matlab: ProcessNode,
  r: ProcessNode,
};

// ── Per-engine code preset registries ───────────────────────
// Each engine ships a small library of ready-to-run snippets the
// user can pick from a dropdown on the node card. The user can
// also save their own (stored in localStorage under a per-engine
// key) — see loadUserPresets / saveUserPresets below.

interface CodePreset { name: string; code: string; }

const PY_HAZE = `# HAZE ANALYSIS — compares grayscale mean of each image input
# against the FIRST input (the "reference"). Output:
# delta_vs_reference + ratio_to_reference per source. Wire into
# an R Plot node downstream for visualisation.

import numpy as np
imgs = {k: v for k, v in inputs.items() if isinstance(v, dict) and "image" in v}
if not imgs:
    raise SystemExit("No image inputs — connect at least one inset → reference + sample(s).")
ref_key = next(iter(imgs))
ref_mean = float(imgs[ref_key]["image"].mean(axis=2).mean())
rows = []
for key, src in imgs.items():
    g = float(src["image"].mean(axis=2).mean())
    rows.append({"source": src.get("label", key),
                 "is_reference": (key == ref_key),
                 "mean_gray": g,
                 "delta_vs_reference": g - ref_mean,
                 "ratio_to_reference": g / ref_mean if ref_mean else float("nan")})
mpfig_data(rows, name="haze_analysis")
`;

const PY_CHANNELS = `# CHANNEL STATISTICS — per-source R/G/B means and stdevs.

import numpy as np
rows = []
for key, src in inputs.items():
    if not (isinstance(src, dict) and "image" in src): continue
    img = src["image"]
    rows.append({
        "source": src.get("label", key),
        "R_mean": float(img[..., 0].mean()), "R_std": float(img[..., 0].std()),
        "G_mean": float(img[..., 1].mean()), "G_std": float(img[..., 1].std()),
        "B_mean": float(img[..., 2].mean()), "B_std": float(img[..., 2].std()),
    })
mpfig_data(rows, name="channel_stats")
`;

const PY_HISTOGRAM = `# INTENSITY HISTOGRAM — bins luminance pixel counts per source.
# Output: long-format CSV ready for ggplot facet_wrap(~ source).

import numpy as np
rows = []
N_BINS = 32
edges = np.linspace(0, 255, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
for key, src in inputs.items():
    if not (isinstance(src, dict) and "image" in src): continue
    lum = src["image"].mean(axis=2).ravel()
    h, _ = np.histogram(lum, bins=edges)
    for c, n in zip(centers, h):
        rows.append({"source": src.get("label", key),
                     "bin_center": float(c), "count": int(n)})
mpfig_data(rows, name="intensity_histogram")
`;

const PY_THRESHOLD = `# THRESHOLD MASK — saves a thresholded copy of each image to
# the images-output bucket. Drop the resulting PNG into a
# Separate-Image inset on the main figure.

import numpy as np
THRESHOLD = 128
for key, src in inputs.items():
    if not (isinstance(src, dict) and "image" in src): continue
    img = src["image"]
    lum = img.mean(axis=2)
    out = img.copy()
    out[lum <= THRESHOLD] = 0
    mpfig_image(out, name=f"{src.get('label', key)}_thresholded")
`;

const PY_DEFAULT = `# Python node — extracts data from upstream images / tables.
# Plots are R-only; mpfig_plot() is intentionally a no-op here.

import numpy as np

rows = []
for key, src in inputs.items():
    if "image" in src:
        gray = src["image"].mean(axis=2)
        rows.append({"source": src.get("label", key),
                     "mean_gray": float(gray.mean()),
                     "std_gray": float(gray.std())})

mpfig_data(rows, name="stats")
`;

const PYTHON_PRESETS: CodePreset[] = [
  { name: "Custom (starter)", code: PY_DEFAULT },
  { name: "Haze analysis", code: PY_HAZE },
  { name: "Channel statistics (R/G/B)", code: PY_CHANNELS },
  { name: "Intensity histogram", code: PY_HISTOGRAM },
  { name: "Threshold mask", code: PY_THRESHOLD },
];

const ML_DEFAULT = `% MATLAB / Octave node — extracts data from upstream inputs.
% Plots are R-only.

keys   = fieldnames(inputs);
labels = cell(numel(keys), 1);
means  = zeros(numel(keys), 1);
stds   = zeros(numel(keys), 1);
for k = 1:numel(keys)
  src = inputs.(keys{k});
  if isfield(src, 'image')
    g = mean(double(src.image), 3);
    labels{k} = src.label;
    means(k)  = mean(g(:));
    stds(k)   = std(g(:));
  end
end
mpfig_data(struct('source', {labels}, 'mean_gray', means, 'std_gray', stds), 'stats');
`;

const ML_HAZE = `% HAZE ANALYSIS — compares each image input's grayscale mean
% against the FIRST input (the reference). Output: per-source
% delta_vs_reference and ratio_to_reference; pipe to R for plot.

keys = fieldnames(inputs);
mask = false(numel(keys), 1);
for k = 1:numel(keys); mask(k) = isfield(inputs.(keys{k}), 'image'); end
keys = keys(mask);
if isempty(keys); error('No image inputs.'); end
ref_mean = mean(mean(mean(double(inputs.(keys{1}).image), 3)));
sources = cell(numel(keys), 1);
mean_g  = zeros(numel(keys), 1);
delta   = zeros(numel(keys), 1);
ratio   = zeros(numel(keys), 1);
is_ref  = false(numel(keys), 1);
for k = 1:numel(keys)
  src = inputs.(keys{k});
  m = mean(mean(mean(double(src.image), 3)));
  sources{k} = src.label;
  mean_g(k)  = m;
  delta(k)   = m - ref_mean;
  ratio(k)   = m / ref_mean;
  is_ref(k)  = (k == 1);
end
mpfig_data(struct('source', {sources}, 'is_reference', is_ref, ...
  'mean_gray', mean_g, 'delta_vs_reference', delta, ...
  'ratio_to_reference', ratio), 'haze_analysis');
`;

const ML_CHANNELS = `% CHANNEL STATISTICS — R/G/B means/stdevs per source.

keys = fieldnames(inputs);
n = numel(keys);
labels = cell(n, 1); R = zeros(n,1); G = zeros(n,1); B = zeros(n,1);
Rs = zeros(n,1); Gs = zeros(n,1); Bs = zeros(n,1);
for k = 1:n
  src = inputs.(keys{k});
  if ~isfield(src, 'image'); continue; end
  img = double(src.image);
  labels{k} = src.label;
  R(k)  = mean(mean(img(:,:,1))); Rs(k) = std(reshape(img(:,:,1),[],1));
  G(k)  = mean(mean(img(:,:,2))); Gs(k) = std(reshape(img(:,:,2),[],1));
  B(k)  = mean(mean(img(:,:,3))); Bs(k) = std(reshape(img(:,:,3),[],1));
end
mpfig_data(struct('source', {labels}, ...
  'R_mean', R, 'R_std', Rs, 'G_mean', G, 'G_std', Gs, 'B_mean', B, 'B_std', Bs), ...
  'channel_stats');
`;

const ML_DEFAULT = `% MATLAB / Octave node — extracts data from upstream inputs.
% Plots are R-only.

keys   = fieldnames(inputs);
labels = cell(numel(keys), 1);
means  = zeros(numel(keys), 1);
stds   = zeros(numel(keys), 1);
for k = 1:numel(keys)
  src = inputs.(keys{k});
  if isfield(src, 'image')
    g = mean(double(src.image), 3);
    labels{k} = src.label;
    means(k)  = mean(g(:));
    stds(k)   = std(g(:));
  end
end
mpfig_data(struct('source', {labels}, 'mean_gray', means, 'std_gray', stds), 'stats');
`;

const MATLAB_PRESETS: CodePreset[] = [
  { name: "Custom (starter)", code: ML_DEFAULT },
  { name: "Haze analysis", code: ML_HAZE },
  { name: "Channel statistics (R/G/B)", code: ML_CHANNELS },
];

const R_BAR = `# BAR CHART (mean + SE).  `+`Plots the first numeric column of
# the first upstream table grouped by its first string column.

library(ggplot2); library(ggprism)
data <- inputs[[1]]
ycol <- names(data)[sapply(data, is.numeric)][1]
xcol <- names(data)[!sapply(data, is.numeric)][1]
mpfig_plot("bar.png")
ggplot(data, aes(x = .data[[xcol]], y = .data[[ycol]], fill = .data[[xcol]])) +
  stat_summary(fun = mean, geom = "col", width = 0.7) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  theme_prism() + theme(legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL)
`;

const R_BOX = `# BOX PLOT.
library(ggplot2); library(ggprism)
data <- inputs[[1]]
ycol <- names(data)[sapply(data, is.numeric)][1]
xcol <- names(data)[!sapply(data, is.numeric)][1]
mpfig_plot("box.png")
ggplot(data, aes(x = .data[[xcol]], y = .data[[ycol]], fill = .data[[xcol]])) +
  geom_boxplot(width = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.12, alpha = 0.55) +
  theme_prism() + theme(legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL)
`;

const R_VIOLIN = `# VIOLIN PLOT.
library(ggplot2); library(ggprism)
data <- inputs[[1]]
ycol <- names(data)[sapply(data, is.numeric)][1]
xcol <- names(data)[!sapply(data, is.numeric)][1]
mpfig_plot("violin.png")
ggplot(data, aes(x = .data[[xcol]], y = .data[[ycol]], fill = .data[[xcol]])) +
  geom_violin(trim = FALSE, alpha = 0.85) +
  geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA, alpha = 0.7) +
  theme_prism() + theme(legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL)
`;

const R_SCATTER = `# SCATTER (+ linear fit) — needs two numeric columns.
library(ggplot2); library(ggprism)
data <- inputs[[1]]
ncols <- names(data)[sapply(data, is.numeric)]
if (length(ncols) < 2) stop("Need at least two numeric columns")
mpfig_plot("scatter.png")
ggplot(data, aes(x = .data[[ncols[1]]], y = .data[[ncols[2]]])) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  theme_prism()
`;

const R_HEATMAP = `# HEATMAP — pivot the first table into a matrix, plot tile fills.
library(ggplot2); library(ggprism)
data <- inputs[[1]]
ycol <- names(data)[sapply(data, is.numeric)][1]
xcol <- names(data)[!sapply(data, is.numeric)][1]
mpfig_plot("heatmap.png")
ggplot(data, aes(x = .data[[xcol]], y = .data[[xcol]], fill = .data[[ycol]])) +
  geom_tile() +
  scale_fill_viridis_c() +
  theme_prism() + theme(axis.text.x = element_text(angle = 30, hjust = 1))
`;

const R_DEFAULT = `# R node — receives upstream tables as data frames in \`inputs\`.
# Plot with ggplot/ggprism, then save via mpfig_plot().

library(ggplot2)
library(ggprism)

# If there's exactly one upstream table, use it directly.
data <- inputs[[1]]

mpfig_plot("plot.png")
ggplot(data, aes(x = source, y = mean_gray)) +
  geom_col(aes(fill = source), width = 0.7) +
  theme_prism() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL, y = "mean_gray")
`;

const R_PRESETS: CodePreset[] = [
  { name: "Custom (starter)", code: R_DEFAULT },
  { name: "Bar chart (mean + SE)", code: R_BAR },
  { name: "Box plot", code: R_BOX },
  { name: "Violin plot", code: R_VIOLIN },
  { name: "Scatter (+ linear fit)", code: R_SCATTER },
  { name: "Heatmap", code: R_HEATMAP },
];

const BUILTIN_PRESETS: Record<EngineKind, CodePreset[]> = {
  python: PYTHON_PRESETS,
  matlab: MATLAB_PRESETS,
  r: R_PRESETS,
};

// ── User-saved code presets (localStorage) ──────────────────
const USER_PRESET_KEY: Record<EngineKind, string> = {
  python: "mpfig.user_presets.python",
  matlab: "mpfig.user_presets.matlab",
  r: "mpfig.user_presets.r",
};

function loadUserPresets(engine: EngineKind): CodePreset[] {
  try {
    const raw = localStorage.getItem(USER_PRESET_KEY[engine]);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.filter((x) => x?.name && x?.code) : [];
  } catch { return []; }
}
function saveUserPresets(engine: EngineKind, presets: CodePreset[]) {
  try { localStorage.setItem(USER_PRESET_KEY[engine], JSON.stringify(presets)); } catch { /* ignore */ }
}

// ── Saved workflows (localStorage) ───────────────────────────
const WORKFLOW_KEY = "mpfig.saved_workflows";

interface SavedWorkflow {
  id: string;
  name: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
  createdAt: number;
}

function loadSavedWorkflows(): SavedWorkflow[] {
  try {
    const raw = localStorage.getItem(WORKFLOW_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr.filter((x) => x?.id && x?.name && Array.isArray(x.nodes) && Array.isArray(x.edges)) : [];
  } catch { return []; }
}
function saveSavedWorkflows(workflows: SavedWorkflow[]) {
  try { localStorage.setItem(WORKFLOW_KEY, JSON.stringify(workflows)); } catch { /* ignore */ }
}

// ── Workflow tabs (in-session) ──────────────────────────────
// Each tab is an independent graph (nodes + edges). The user can
// add multiple to run the same analysis on different image sets.
interface WorkflowTab {
  id: string;
  name: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
}

function newSourceNode(sources: InsetSource[]): Node<NodeData> {
  return {
    id: "source",
    type: "source",
    position: { x: 30, y: 30 },
    data: { label: "Source", kind: "source", sources, status: "ok" } as NodeData,
    draggable: true,
    deletable: false,
  };
}

// ── Main component ───────────────────────────────────────────

interface Props {
  /** Provided by the parent dialog. */
  open: boolean;
  /** Built-in measurement CSV (same content the R tab used). */
  measurementsCsv: string;
  /** Pinned-only filter for the drawer (toggled by the user). */
  pinnedFilter?: boolean;
  /** Pushed by the runner whenever a node finishes — the parent
   *  uses this to mirror outputs into the AnalysisStore for
   *  save/load. Optional; falls back to internal state otherwise. */
  onOutputsChanged?: (outputs: AggregatedOutput[]) => void;
}

export interface AggregatedOutput {
  nodeId: string;
  nodeLabel: string;
  outputId: string;
  kind: DataKind;
  name: string;
  payload: string;
  pinned?: boolean;
}

export function AnalysisNodeGraph({ open, measurementsCsv, onOutputsChanged }: Props) {
  // Inset sources from the backend (image ports for the source node).
  const [insetSources, setInsetSources] = useState<InsetSource[]>([]);
  const [matlabKind, setMatlabKind] = useState<string>("");

  // ── Layer-2 canvases: multiple workflow tabs ────────────────
  // Each tab is an independent graph. The user creates new tabs
  // to run the same analysis topology on a different image set,
  // or loads a previously-saved workflow from localStorage.
  const [workflowTabs, setWorkflowTabs] = useState<WorkflowTab[]>(() => [{
    id: newId("wf"),
    name: "Workflow 1",
    nodes: [newSourceNode([])],
    edges: [],
  }]);
  const [activeWfId, setActiveWfId] = useState<string>(() => "");
  const activeWf = useMemo(() => workflowTabs.find((w) => w.id === activeWfId) || workflowTabs[0], [workflowTabs, activeWfId]);
  // Reactive references to the active tab's nodes / edges so the
  // existing handlers below (which were written against a single
  // {nodes, edges} pair) stay correct.
  const nodes = activeWf.nodes;
  const edges = activeWf.edges;
  const setNodes = useCallback((updater: (n: Node<NodeData>[]) => Node<NodeData>[]) => {
    setWorkflowTabs((tabs) => tabs.map((t) => t.id === activeWf.id ? { ...t, nodes: updater(t.nodes) } : t));
  }, [activeWf.id]);
  const setEdges = useCallback((updater: (e: Edge[]) => Edge[]) => {
    setWorkflowTabs((tabs) => tabs.map((t) => t.id === activeWf.id ? { ...t, edges: updater(t.edges) } : t));
  }, [activeWf.id]);

  useEffect(() => {
    // Initial active id wire-up — first render sets it to the
    // first tab so memoisation lands cleanly.
    if (!activeWfId && workflowTabs.length > 0) setActiveWfId(workflowTabs[0].id);
  }, [activeWfId, workflowTabs]);

  // Saved workflow library (localStorage).
  const [savedWorkflows, setSavedWorkflows] = useState<SavedWorkflow[]>(() => loadSavedWorkflows());

  // React Flow instance captured via onInit — lets us project
  // viewport coordinates so new nodes land in the visible area
  // rather than at hard-coded x=350 (off-screen when the user
  // has panned far away).
  type RFInstance = {
    getViewport: () => { x: number; y: number; zoom: number };
    screenToFlowPosition: (p: { x: number; y: number }) => { x: number; y: number };
  };
  const rfRef = useRef<RFInstance | null>(null);

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [runningGraph, setRunningGraph] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [drawerTab, setDrawerTab] = useState<DataKind>("plot");
  const consoleRef = useRef<string>("");
  const [consoleOut, setConsoleOut] = useState("");

  const addPanelToFigure = useFigureStore((s) => s.setPanelImage);
  void addPanelToFigure;  // wired in a later commit — see Output drawer

  // Load inset sources + MATLAB availability on open.
  useEffect(() => {
    if (!open) return;
    api.listInsetAnalysisSources()
      .then((r) => {
        const list = r.sources || [];
        setInsetSources(list);
        // Push the latest source list into the Source node of EVERY
        // workflow tab so the user's saved tabs stay in sync with
        // the figure's current inset-flag selections.
        setWorkflowTabs((tabs) => tabs.map((t) => ({
          ...t,
          nodes: t.nodes.map((n) => n.id === "source" ? { ...n, data: { ...n.data, sources: list } } : n),
        })));
      })
      .catch(() => setInsetSources([]));
    api.checkMatlab().then((m) => setMatlabKind(m.kind || "")).catch(() => setMatlabKind(""));
  }, [open]);

  // ── Graph handlers ─────────────────────────────────────────

  const onNodesChange = useCallback((changes: NodeChange<Node<NodeData>>[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  }, [setNodes]);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  }, [setEdges]);

  /** Validate an edge before adding: input and output kinds must
   *  match. Source-image → image-in; source-table → table-in. */
  const isValidConnection = useCallback((c: Connection | Edge): boolean => {
    const srcHandle = (c as Connection).sourceHandle || "";
    const tgtHandle = (c as Connection).targetHandle || "";
    const srcKind = srcHandle.startsWith("out_image") ? "image"
      : srcHandle.startsWith("out_table") ? "table"
      : srcHandle.startsWith("out_plot") ? "plot"
      : "";
    const tgtKind = tgtHandle === "in_image" ? "image"
      : tgtHandle === "in_table" ? "table"
      : "";
    if (!srcKind || !tgtKind) return false;
    // `plot` outputs are terminal — they can't feed another node's
    // input.
    if (srcKind === "plot") return false;
    return srcKind === tgtKind;
  }, []);

  const onConnect = useCallback((connection: Connection) => {
    if (!isValidConnection(connection)) return;
    const srcHandle = connection.sourceHandle || "";
    const kind: DataKind = srcHandle.includes("image") ? "image" : "table";
    setEdges((eds) => addEdge({
      ...connection,
      type: "default",
      animated: false,
      style: { stroke: PORT_COLOR[kind], strokeWidth: 2 },
    }, eds));
  }, [isValidConnection]);

  // ── Add a new node ─────────────────────────────────────────

  const addProcessNode = useCallback((engine: EngineKind, presetIdx: number = 0) => {
    const id = newId(engine);
    const defaultCode = BUILTIN_PRESETS[engine][presetIdx]?.code
      ?? (engine === "python" ? PY_DEFAULT : engine === "matlab" ? ML_DEFAULT : R_DEFAULT);
    // Place the new node inside the CURRENTLY VISIBLE viewport so
    // the user always sees it, regardless of pan/zoom. Without
    // this fallback to hardcoded screen coords the node was
    // routinely off-screen on a panned canvas.
    let pos = { x: 350, y: 120 };
    const inst = rfRef.current;
    if (inst) {
      try {
        // Pick a point roughly 1/3 across the visible viewport so
        // the node has room for outgoing edges. The container's
        // bounding rect is the viewport's screen extent.
        const container = document.querySelector(".react-flow") as HTMLElement | null;
        if (container) {
          const rect = container.getBoundingClientRect();
          const screen = {
            x: rect.left + rect.width * 0.35 + Math.random() * 60,
            y: rect.top + rect.height * 0.30 + Math.random() * 60,
          };
          pos = inst.screenToFlowPosition(screen);
        }
      } catch { /* fall back to default pos */ }
    }
    setNodes((cur) => [
      ...cur,
      {
        id,
        type: engine,
        position: pos,
        data: {
          label: engine === "python" ? "Python" : engine === "matlab" ? "MATLAB" : "R Plot",
          kind: engine,
          code: defaultCode,
          outputs: [],
          inputs: [],
          status: "idle",
        } as NodeData,
      },
    ]);
    setSelectedNodeId(id);
  }, [setNodes]);

  const removeNode = useCallback((nodeId: string) => {
    if (nodeId === "source") return;
    setNodes((cur) => cur.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  }, [selectedNodeId, setNodes, setEdges]);

  // ── Layer-2 workflow tab management ─────────────────────────

  const addWorkflowTab = useCallback(() => {
    const id = newId("wf");
    const idx = workflowTabs.length + 1;
    setWorkflowTabs((tabs) => [...tabs, {
      id, name: `Workflow ${idx}`,
      nodes: [newSourceNode(insetSources)], edges: [],
    }]);
    setActiveWfId(id);
    setSelectedNodeId(null);
  }, [workflowTabs.length, insetSources]);

  const removeWorkflowTab = useCallback((id: string) => {
    setWorkflowTabs((tabs) => {
      if (tabs.length <= 1) return tabs;  // keep at least one
      const idx = tabs.findIndex((t) => t.id === id);
      const next = tabs.filter((t) => t.id !== id);
      if (id === activeWfId) {
        setActiveWfId(next[Math.max(0, idx - 1)]?.id || next[0]?.id || "");
      }
      return next;
    });
  }, [activeWfId]);

  const renameWorkflowTab = useCallback((id: string, name: string) => {
    setWorkflowTabs((tabs) => tabs.map((t) => t.id === id ? { ...t, name } : t));
  }, []);

  /** Save the active workflow into localStorage so the user can
   *  reload it later as a starting point for a different image set. */
  const saveCurrentWorkflow = useCallback(() => {
    const name = window.prompt("Save workflow as:", activeWf.name);
    if (!name) return;
    // Strip output payloads / status — saved workflows are
    // templates, not run results.
    const sanitizedNodes = activeWf.nodes.map((n) => ({
      ...n,
      data: { ...n.data, outputs: [], status: "idle" as const, error: undefined },
    }));
    const wf: SavedWorkflow = {
      id: newId("saved"), name, createdAt: Date.now(),
      nodes: sanitizedNodes, edges: activeWf.edges,
    };
    setSavedWorkflows((cur) => {
      const next = [...cur, wf];
      saveSavedWorkflows(next);
      return next;
    });
  }, [activeWf]);

  /** Load a saved workflow as a NEW tab so the user can run it
   *  against the current insets without overwriting their work. */
  const loadSavedWorkflow = useCallback((id: string) => {
    const wf = savedWorkflows.find((w) => w.id === id);
    if (!wf) return;
    const newWfId = newId("wf");
    // Replace the saved source node's `sources` with the LIVE
    // current insets so newly-flagged regions are reachable.
    const refreshedNodes = wf.nodes.map((n) =>
      n.id === "source" ? { ...n, data: { ...n.data, sources: insetSources } } : n
    );
    setWorkflowTabs((tabs) => [...tabs, {
      id: newWfId, name: wf.name, nodes: refreshedNodes, edges: wf.edges,
    }]);
    setActiveWfId(newWfId);
    setSelectedNodeId(null);
  }, [savedWorkflows, insetSources]);

  const deleteSavedWorkflow = useCallback((id: string) => {
    setSavedWorkflows((cur) => {
      const next = cur.filter((w) => w.id !== id);
      saveSavedWorkflows(next);
      return next;
    });
  }, []);

  // ── Topological execution ──────────────────────────────────

  /** Build the dependency graph from edges and return node ids in
   *  topological order. Throws on cycles (shouldn't happen — the
   *  UI prevents loops). */
  const topoOrder = useCallback((): string[] => {
    const incoming = new Map<string, Set<string>>();
    const outgoing = new Map<string, Set<string>>();
    for (const n of nodes) {
      incoming.set(n.id, new Set());
      outgoing.set(n.id, new Set());
    }
    for (const e of edges) {
      incoming.get(e.target)?.add(e.source);
      outgoing.get(e.source)?.add(e.target);
    }
    const ready: string[] = [];
    for (const n of nodes) if ((incoming.get(n.id) || new Set()).size === 0) ready.push(n.id);
    const order: string[] = [];
    while (ready.length) {
      const id = ready.shift()!;
      order.push(id);
      for (const child of outgoing.get(id) || []) {
        incoming.get(child)?.delete(id);
        if ((incoming.get(child) || new Set()).size === 0) ready.push(child);
      }
    }
    return order;
  }, [nodes, edges]);

  /** Resolve a node's inputs by walking its incoming edges and
   *  pulling the named upstream output payload. Returns an array
   *  of { key, kind, image_b64 | csv } suitable for the backend
   *  `extra_inputs` parameter. */
  const collectInputs = useCallback((nodeId: string, nodeMap: Map<string, Node<NodeData>>) => {
    const result: Array<{ key: string; kind: DataKind; label: string; image_b64?: string; csv?: string }> = [];
    let imgCount = 0;
    let tblCount = 0;
    for (const e of edges) {
      if (e.target !== nodeId) continue;
      const upstream = nodeMap.get(e.source);
      if (!upstream) continue;
      const upstreamData = upstream.data;
      // Source node: output ports correspond to insets (by index) or
      // the measurements table.
      if (upstreamData.kind === "source") {
        if ((e.sourceHandle || "").startsWith("out_image_")) {
          const idx = parseInt((e.sourceHandle || "").replace("out_image_", ""), 10);
          const src = (upstreamData.sources || [])[idx];
          if (!src) continue;
          const key = `inset_${imgCount++}_${src.key}`;
          // Source insets get base64 PNG of their thumbnail — the
          // backend has the full image too via _extract_inset_image
          // but we use the thumbnail here for speed. The runner
          // sends the FULL image as a regular `sources` entry below.
          result.push({ key, kind: "image", label: src.label, image_b64: src.thumbnail });
        } else if (e.sourceHandle === "out_table_measurements") {
          const key = `measurements`;
          result.push({ key, kind: "table", label: "measurements", csv: measurementsCsv });
        }
        continue;
      }
      // Upstream process node: pull from its cached outputs.
      const outputs = upstreamData.outputs || [];
      const outId = e.sourceHandle || "";
      // Match by handle id — but process nodes only have generic
      // `out_image` / `out_table` / `out_plot` handles, so we pull
      // ALL outputs of that kind from the upstream node.
      const kind: DataKind = outId.includes("image") ? "image" : outId.includes("table") ? "table" : "plot";
      if (kind === "plot") continue;  // plots can't feed another node
      for (const out of outputs) {
        if (out.kind !== kind) continue;
        const key = kind === "image" ? `up_image_${imgCount++}_${out.name}` : `up_table_${tblCount++}_${out.name}`;
        if (kind === "image") {
          result.push({ key, kind, label: `${upstream.data.label}/${out.name}`, image_b64: out.payload });
        } else {
          result.push({ key, kind, label: `${upstream.data.label}/${out.name}`, csv: out.payload });
        }
      }
    }
    return result;
  }, [edges, measurementsCsv]);

  const runNode = useCallback(async (node: Node<NodeData>, nodeMap: Map<string, Node<NodeData>>) => {
    const engine = node.data.kind as EngineKind;
    if (engine !== "python" && engine !== "matlab" && engine !== "r") return;
    // Mark running
    setNodes((cur) => cur.map((n) => n.id === node.id ? { ...n, data: { ...n.data, status: "running", error: undefined } } : n));
    consoleRef.current += `\n=== Run ${node.data.label} (${engine}) ===\n`;
    setConsoleOut(consoleRef.current);

    try {
      const extra = collectInputs(node.id, nodeMap);
      // Source-fed images go in the `sources` array (full pixel
      // extract via inset-sources), upstream-fed images go in
      // `extra_inputs`. We currently route ALL through extra_inputs
      // for simplicity — both Py and MATLAB harnesses already
      // tolerate that.
      let result: { success: boolean; stdout: string; stderr: string;
                    plots: string[]; tables: { name: string; csv: string }[];
                    images: { name: string; image: string }[]; };
      if (engine === "python") {
        // Filter sources from extras: the backend's `sources` path
        // re-extracts insets at full resolution, which is what we
        // want for haze etc. Match the inset key back.
        const sources = extra
          .filter((x) => x.kind === "image" && x.key.startsWith("inset_"))
          .map((x) => {
            const insetKey = x.key.replace(/^inset_\d+_/, "");
            const src = insetSources.find((s) => s.key === insetKey);
            if (!src) return null;
            return { key: insetKey, row: src.row, col: src.col, inset_index: src.inset_index, label: src.label };
          })
          .filter((s): s is { key: string; row: number; col: number; inset_index: number; label: string } => !!s);
        // Strip the inset entries from extras so they don't double-count.
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        const r = await api.runPython(node.data.code || PY_DEFAULT, sources, 60);
        // Wire extras AFTER — backend supports it via runPython's
        // extra_inputs param. apiClient currently doesn't expose
        // it, so fall back to manual fetch when extras present.
        if (extras.length > 0) {
          const resp = await fetch(`${(import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765"}/api/analysis/run-python`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: node.data.code || PY_DEFAULT, sources, extra_inputs: extras, timeout_sec: 60 }),
          });
          result = await resp.json();
        } else {
          result = r;
        }
      } else if (engine === "matlab") {
        const sources = extra
          .filter((x) => x.kind === "image" && x.key.startsWith("inset_"))
          .map((x) => {
            const insetKey = x.key.replace(/^inset_\d+_/, "");
            const src = insetSources.find((s) => s.key === insetKey);
            if (!src) return null;
            return { key: insetKey, row: src.row, col: src.col, inset_index: src.inset_index, label: src.label };
          })
          .filter((s): s is { key: string; row: number; col: number; inset_index: number; label: string } => !!s);
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        if (extras.length > 0) {
          const resp = await fetch(`${(import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765"}/api/analysis/run-matlab`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: node.data.code || ML_DEFAULT, sources, extra_inputs: extras, timeout_sec: 90 }),
          });
          result = await resp.json();
        } else {
          result = await api.runMatlab(node.data.code || ML_DEFAULT, sources, 90);
        }
      } else {
        // R node: inputs are tables, presented as a list of data
        // frames named after their source. We assemble an inline
        // R script that defines `inputs <- list(...)` from the
        // upstream CSVs, then appends the user's code.
        const tables = extra.filter((x) => x.kind === "table");
        const inputsAssign = tables.map((t, i) =>
          `  ${JSON.stringify(t.label || `t${i}`)} = read.csv(text = ${JSON.stringify(t.csv || "")}, stringsAsFactors = FALSE)`
        ).join(",\n");
        const prelude = tables.length > 0
          ? `inputs <- list(\n${inputsAssign}\n)\n# Convenience: \`data\` aliases the first input table.\nif (length(inputs) > 0) data <- inputs[[1]] else data <- data.frame()\n\n`
          : `inputs <- list()\ndata <- data.frame()\n\n`;
        const fullCode = prelude + (node.data.code || R_DEFAULT);
        const r = await api.runR(fullCode, "Panel,Name,Group,Value,Unit\n", undefined);
        result = { ...r, images: [] };
      }

      // Convert result into NodeOutputs.
      const newOutputs: NodeOutput[] = [
        ...result.plots.map((b64, i) => ({ id: `out_plot_${i}`, kind: "plot" as DataKind, name: `plot_${i + 1}`, payload: b64 })),
        ...result.tables.map((t) => ({ id: `out_table_${t.name}`, kind: "table" as DataKind, name: t.name, payload: t.csv })),
        ...((result.images || []).map((im, i) => ({ id: `out_image_${i}`, kind: "image" as DataKind, name: im.name, payload: im.image }))),
      ];
      const stdout = (result.stdout || "") + (result.stderr ? `\n${result.stderr}` : "");
      consoleRef.current += stdout.trim() ? stdout + "\n" : "(no console output)\n";
      setConsoleOut(consoleRef.current);
      setNodes((cur) => cur.map((n) => n.id === node.id ? {
        ...n,
        data: { ...n.data, outputs: newOutputs, status: result.success ? "ok" : "error", error: result.success ? undefined : (result.stderr || "Run failed").slice(0, 200) },
      } : n));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      consoleRef.current += `error: ${msg}\n`;
      setConsoleOut(consoleRef.current);
      setNodes((cur) => cur.map((n) => n.id === node.id ? { ...n, data: { ...n.data, status: "error", error: msg.slice(0, 200) } } : n));
    }
  }, [collectInputs, insetSources]);

  const runGraph = useCallback(async () => {
    setRunningGraph(true);
    consoleRef.current += `\n──────────── RUN GRAPH ────────────\n`;
    setConsoleOut(consoleRef.current);
    // Take a fresh snapshot of nodes — we'll be updating state in
    // the loop and need to look up upstream outputs as they land.
    const order = topoOrder();
    // We run sequentially so each node sees fresh upstream state.
    for (const id of order) {
      // Re-read node from state on each iteration.
      const snapshot = await new Promise<Map<string, Node<NodeData>>>((resolve) => {
        setNodes((cur) => {
          resolve(new Map(cur.map((n) => [n.id, n])));
          return cur;
        });
      });
      const node = snapshot.get(id);
      if (!node) continue;
      if (node.data.kind === "source") continue;
      await runNode(node, snapshot);
    }
    setRunningGraph(false);
  }, [topoOrder, runNode]);

  const selectedNode = useMemo(() => nodes.find((n) => n.id === selectedNodeId) || null, [nodes, selectedNodeId]);

  // Aggregate outputs across all nodes for the drawer.
  const allOutputs = useMemo<AggregatedOutput[]>(() => {
    return nodes.flatMap((n) => (n.data.outputs || []).map((o) => ({
      nodeId: n.id,
      nodeLabel: n.data.label,
      outputId: o.id,
      kind: o.kind,
      name: o.name,
      payload: o.payload,
      pinned: o.pinned,
    })));
  }, [nodes]);
  // Notify parent (for save/load) — debounce by deferring to a microtask.
  useEffect(() => {
    if (onOutputsChanged) onOutputsChanged(allOutputs);
  }, [allOutputs, onOutputsChanged]);

  const togglePin = useCallback((nodeId: string, outputId: string) => {
    setNodes((cur) => cur.map((n) => n.id !== nodeId ? n : {
      ...n,
      data: {
        ...n.data,
        outputs: (n.data.outputs || []).map((o) => o.id === outputId ? { ...o, pinned: !o.pinned } : o),
      },
    }));
  }, []);

  // ── Render ─────────────────────────────────────────────────

  return (
    <Box sx={{ position: "relative", flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      {/* ── Layer-2: workflow tabs + saved-library dropdown ──────
          Each tab is an independent graph (one canvas). The user
          can run the same analysis on a different image set by
          adding a new tab, or load a saved template from the
          dropdown. Tabs persist for the session; saved workflows
          persist in localStorage across app restarts. */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.25, borderBottom: "1px solid", borderColor: "divider", bgcolor: "background.paper", flexShrink: 0 }}>
        <Tabs
          value={activeWf.id}
          onChange={(_, v) => { setActiveWfId(v); setSelectedNodeId(null); }}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ minHeight: 28, "& .MuiTab-root": { minHeight: 28, fontSize: "0.7rem", py: 0, px: 1, textTransform: "none" } }}
        >
          {workflowTabs.map((t) => (
            <Tab key={t.id} value={t.id} label={
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <span>{t.name}</span>
                {workflowTabs.length > 1 && (
                  <Box
                    component="span"
                    onClick={(e) => { e.stopPropagation(); removeWorkflowTab(t.id); }}
                    sx={{ fontSize: "0.7rem", cursor: "pointer", color: "text.disabled", "&:hover": { color: "error.main" } }}
                  >×</Box>
                )}
              </Box>
            } />
          ))}
        </Tabs>
        <IconButton size="small" onClick={addWorkflowTab} title="Add a new workflow tab">
          <AddIcon sx={{ fontSize: 16 }} />
        </IconButton>
        <Button
          size="small"
          onClick={() => {
            const newName = window.prompt("Rename this workflow:", activeWf.name);
            if (newName && newName !== activeWf.name) renameWorkflowTab(activeWf.id, newName);
          }}
          sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25, minWidth: 0 }}
        >
          ✎ Rename
        </Button>
        <Box sx={{ flex: 1 }} />
        <Button
          size="small"
          variant="outlined"
          onClick={saveCurrentWorkflow}
          sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25 }}
        >
          💾 Save workflow
        </Button>
        <Select
          size="small"
          displayEmpty
          value=""
          renderValue={() => `📂 Load saved (${savedWorkflows.length})`}
          onChange={(e) => loadSavedWorkflow(String(e.target.value))}
          disabled={savedWorkflows.length === 0}
          sx={{ fontSize: "0.6rem", height: 26, minWidth: 150, "& .MuiSelect-select": { py: 0.25, px: 1 } }}
        >
          {savedWorkflows.length === 0 ? (
            <MenuItem value="" disabled sx={{ fontSize: "0.7rem" }}>(no saved workflows)</MenuItem>
          ) : savedWorkflows.map((wf) => (
            <MenuItem key={wf.id} value={wf.id} sx={{ fontSize: "0.7rem", display: "flex", justifyContent: "space-between" }}>
              <span style={{ flex: 1 }}>{wf.name}</span>
              <Box
                component="span"
                onClick={(e) => { e.stopPropagation(); if (window.confirm(`Delete saved workflow '${wf.name}'?`)) deleteSavedWorkflow(wf.id); }}
                sx={{ fontSize: "0.85rem", color: "text.disabled", ml: 1, "&:hover": { color: "error.main" } }}
              >🗑</Box>
            </MenuItem>
          ))}
        </Select>
      </Box>

      {/* Canvas + side panel + drawer (flex row below the tabs) */}
      <Box sx={{ flex: 1, display: "flex", minHeight: 0, position: "relative" }}>
      {/* Left: graph canvas */}
      <Box sx={{ flex: 1, position: "relative", minWidth: 0, minHeight: 0 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          nodeTypes={nodeTypes}
          onNodeClick={(_, n) => setSelectedNodeId(n.id)}
          onPaneClick={() => setSelectedNodeId(null)}
          onInit={(inst) => { rfRef.current = inst as unknown as RFInstance; }}
          fitView
          deleteKeyCode={["Backspace", "Delete"]}
        >
          <Background />
          <Controls />
          <MiniMap pannable zoomable />
        </ReactFlow>
        {/* Toolbar overlay */}
        <Box sx={{ position: "absolute", top: 8, left: 8, display: "flex", gap: 0.5, zIndex: 5 }}>
          <Tooltip placement="bottom" title="Add a Python node">
            <Button size="small" variant="contained" startIcon={<AddIcon sx={{ fontSize: 14 }} />}
              onClick={() => addProcessNode("python")}
              sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25, bgcolor: KIND_COLOR.python, "&:hover": { bgcolor: KIND_COLOR.python, filter: "brightness(0.9)" } }}>
              🐍 Python
            </Button>
          </Tooltip>
          <Tooltip placement="bottom" title={matlabKind ? `Add a MATLAB node (uses ${matlabKind})` : "Octave / MATLAB not detected — install Octave to enable"}>
            <span>
              <Button size="small" variant="contained" startIcon={<AddIcon sx={{ fontSize: 14 }} />}
                disabled={!matlabKind}
                onClick={() => addProcessNode("matlab")}
                sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25, bgcolor: KIND_COLOR.matlab, "&:hover": { bgcolor: KIND_COLOR.matlab, filter: "brightness(0.9)" } }}>
                📐 MATLAB
              </Button>
            </span>
          </Tooltip>
          <Tooltip placement="bottom" title="Add an R node — receives tables, produces plots">
            <Button size="small" variant="contained" startIcon={<AddIcon sx={{ fontSize: 14 }} />}
              onClick={() => addProcessNode("r")}
              sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25, bgcolor: KIND_COLOR.r, "&:hover": { bgcolor: KIND_COLOR.r, filter: "brightness(0.9)" } }}>
              📊 R Plot
            </Button>
          </Tooltip>
          <Box sx={{ flex: 1 }} />
          <Button size="small" variant="contained" color="primary"
            startIcon={runningGraph ? <CircularProgress size={12} /> : <PlayArrowIcon sx={{ fontSize: 14 }} />}
            disabled={runningGraph}
            onClick={runGraph}
            sx={{ fontSize: "0.7rem", textTransform: "none", py: 0.25, ml: 1 }}>
            {runningGraph ? "Running…" : "Run graph"}
          </Button>
        </Box>
      </Box>

      {/* Right: selected-node detail panel */}
      {selectedNode && selectedNode.id !== "source" && (() => {
        const engine = selectedNode.data.kind as EngineKind;
        const builtin = BUILTIN_PRESETS[engine] || [];
        const userPresets = loadUserPresets(engine);
        return (
        <Box sx={{ width: 460, borderLeft: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", minWidth: 0, bgcolor: "background.paper" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 1, borderBottom: "1px solid", borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.75rem" }}>
              {KIND_ICON[engine]} {selectedNode.data.label}
            </Typography>
            <Box sx={{ flex: 1 }} />
            <Button size="small" variant="outlined"
              startIcon={selectedNode.data.status === "running" ? <CircularProgress size={11} /> : <PlayArrowIcon sx={{ fontSize: 13 }} />}
              disabled={selectedNode.data.status === "running"}
              onClick={async () => {
                const nm = new Map(nodes.map((n) => [n.id, n]));
                await runNode(selectedNode, nm);
              }}
              sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25 }}>
              Run this node
            </Button>
            <IconButton size="small" onClick={() => removeNode(selectedNode.id)} title="Remove node">
              <DeleteOutlineIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Box>
          {/* Layer-1 preset selector — built-in snippets + user-
              saved presets stored per-engine in localStorage. */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.5, borderBottom: "1px solid", borderColor: "divider", flexWrap: "wrap" }}>
            <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>Preset:</Typography>
            <Select
              size="small"
              value=""
              displayEmpty
              renderValue={() => "Load preset…"}
              onChange={(e) => {
                const val = String(e.target.value);
                let preset: CodePreset | undefined;
                if (val.startsWith("b:")) preset = builtin[parseInt(val.slice(2), 10)];
                else if (val.startsWith("u:")) preset = userPresets[parseInt(val.slice(2), 10)];
                if (preset) {
                  setNodes((cur) => cur.map((n) => n.id === selectedNode.id
                    ? { ...n, data: { ...n.data, code: preset!.code, status: "idle" as const } } : n));
                }
              }}
              sx={{ fontSize: "0.65rem", height: 26, minWidth: 180, "& .MuiSelect-select": { py: 0.4, px: 1 } }}
            >
              <MenuItem value="" disabled sx={{ fontSize: "0.7rem", fontWeight: 700 }}>Built-in</MenuItem>
              {builtin.map((p, i) => (
                <MenuItem key={`b${i}`} value={`b:${i}`} sx={{ fontSize: "0.7rem", pl: 2 }}>{p.name}</MenuItem>
              ))}
              {userPresets.length > 0 && (
                <MenuItem value="" disabled sx={{ fontSize: "0.7rem", fontWeight: 700, mt: 0.5 }}>Saved</MenuItem>
              )}
              {userPresets.map((p, i) => (
                <MenuItem key={`u${i}`} value={`u:${i}`} sx={{ fontSize: "0.7rem", pl: 2 }}>★ {p.name}</MenuItem>
              ))}
            </Select>
            <Button size="small"
              onClick={() => {
                const name = window.prompt("Save current code as preset (name):");
                if (!name) return;
                const code = selectedNode.data.code || "";
                const next = [...userPresets.filter((p) => p.name !== name), { name, code }];
                saveUserPresets(engine, next);
                setSelectedNodeId(selectedNode.id);  // force re-render
              }}
              sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25, minWidth: 0 }}
            >
              💾 Save
            </Button>
            {userPresets.length > 0 && (
              <Tooltip title="Delete a saved preset">
                <Select
                  size="small"
                  value=""
                  displayEmpty
                  renderValue={() => "🗑"}
                  onChange={(e) => {
                    const name = String(e.target.value);
                    if (!name) return;
                    if (!window.confirm(`Delete preset '${name}'?`)) return;
                    saveUserPresets(engine, userPresets.filter((p) => p.name !== name));
                    setSelectedNodeId(selectedNode.id);
                  }}
                  sx={{ fontSize: "0.6rem", height: 22, minWidth: 0, "& .MuiSelect-select": { py: 0.1, px: 0.5 } }}
                >
                  {userPresets.map((p) => (
                    <MenuItem key={p.name} value={p.name} sx={{ fontSize: "0.7rem" }}>{p.name}</MenuItem>
                  ))}
                </Select>
              </Tooltip>
            )}
          </Box>
          <Box sx={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <Box sx={{ flex: 1, minHeight: 200, "& .cm-editor": { height: "100%" }, "& .cm-scroller": { overflow: "auto", fontFamily: "monospace" } }}>
              <CodeMirror
                value={selectedNode.data.code || ""}
                onChange={(v) => setNodes((cur) => cur.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, code: v, status: n.data.status === "ok" ? "stale" : n.data.status } } : n))}
                theme={oneDark}
                height="100%"
                extensions={selectedNode.data.kind === "r"
                  ? [StreamLanguage.define(cmR)]
                  : [cmPython()]}
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLine: true,
                  foldGutter: false,
                  bracketMatching: true,
                  closeBrackets: true,
                }}
              />
            </Box>
            {/* Per-node outputs preview */}
            {(selectedNode.data.outputs?.length || 0) > 0 && (
              <Box sx={{ borderTop: "1px solid", borderColor: "divider", maxHeight: 140, overflowY: "auto", p: 0.5 }}>
                <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 700, color: "text.secondary", letterSpacing: 0.5, textTransform: "uppercase", display: "block", mb: 0.5 }}>
                  Outputs ({selectedNode.data.outputs?.length})
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                  {(selectedNode.data.outputs || []).map((o) => (
                    <Box key={o.id} sx={{ border: "1px solid", borderColor: "divider", borderRadius: 0.25, p: 0.25, minWidth: 70 }}>
                      <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600 }}>{o.kind}: {o.name}</Typography>
                      {(o.kind === "plot" || o.kind === "image") && (
                        <Box component="img" src={`data:image/png;base64,${o.payload}`} alt={o.name}
                          sx={{ display: "block", width: 60, height: 60, objectFit: "contain", mt: 0.25 }} />
                      )}
                      <IconButton size="small" onClick={() => togglePin(selectedNode.id, o.id)} sx={{ p: 0 }}>
                        {o.pinned ? <StarIcon sx={{ fontSize: 12, color: "warning.main" }} /> : <StarBorderIcon sx={{ fontSize: 12 }} />}
                      </IconButton>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        </Box>
        );
      })()}

      {/* Bottom drawer: aggregated outputs */}
      <Drawer
        variant="persistent"
        anchor="bottom"
        open={drawerOpen}
        PaperProps={{ sx: { position: "absolute", height: "32%", minHeight: 180 } }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.25, borderBottom: "1px solid", borderColor: "divider" }}>
          <Tabs value={drawerTab} onChange={(_, v) => setDrawerTab(v)} sx={{ minHeight: 28, "& .MuiTab-root": { minHeight: 28, fontSize: "0.65rem", py: 0, textTransform: "none" } }}>
            <Tab value="plot" label={`📊 Plots (${allOutputs.filter((o) => o.kind === "plot").length})`} />
            <Tab value="table" label={`📋 Tables (${allOutputs.filter((o) => o.kind === "table").length})`} />
            <Tab value="image" label={`🖼 Images (${allOutputs.filter((o) => o.kind === "image").length})`} />
          </Tabs>
          <Box sx={{ flex: 1 }} />
          <Button size="small" onClick={() => setDrawerOpen(false)} sx={{ fontSize: "0.55rem", textTransform: "none" }}>
            Hide
          </Button>
        </Box>
        <Box sx={{ flex: 1, overflow: "auto", p: 1 }}>
          {allOutputs.filter((o) => o.kind === drawerTab).length === 0 ? (
            <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.disabled", fontStyle: "italic" }}>
              No {drawerTab} outputs yet. Build a graph and click Run.
            </Typography>
          ) : (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
              {allOutputs.filter((o) => o.kind === drawerTab).map((o) => (
                <Box key={`${o.nodeId}-${o.outputId}`} sx={{ border: "1px solid", borderColor: o.pinned ? "warning.main" : "divider", borderRadius: 0.5, p: 0.5, width: 140, display: "flex", flexDirection: "column", gap: 0.25, position: "relative" }}>
                  <IconButton size="small" onClick={() => togglePin(o.nodeId, o.outputId)} sx={{ position: "absolute", top: 2, right: 2, p: 0 }}>
                    {o.pinned ? <StarIcon sx={{ fontSize: 14, color: "warning.main" }} /> : <StarBorderIcon sx={{ fontSize: 14 }} />}
                  </IconButton>
                  <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600, color: "text.secondary" }}>
                    {o.nodeLabel}
                  </Typography>
                  <Typography variant="caption" sx={{ fontSize: "0.65rem", fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {o.name}
                  </Typography>
                  {(o.kind === "plot" || o.kind === "image") && (
                    <Box component="img" src={`data:image/png;base64,${o.payload}`} alt={o.name}
                      sx={{ width: "100%", height: 80, objectFit: "contain", bgcolor: "background.default" }} />
                  )}
                  {o.kind === "table" && (
                    <Tooltip placement="top" title={
                      <pre style={{ margin: 0, fontSize: 10, maxHeight: 240, overflow: "auto", whiteSpace: "pre" }}>{o.payload}</pre>
                    }>
                      <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", cursor: "help" }}>
                        {(o.payload.split("\n").length - 1)} rows
                      </Typography>
                    </Tooltip>
                  )}
                  <Button size="small" component="a"
                    href={o.kind === "table"
                      ? `data:text/csv;base64,${btoa(unescape(encodeURIComponent(o.payload)))}`
                      : `data:image/png;base64,${o.payload}`}
                    download={`${o.name}.${o.kind === "table" ? "csv" : "png"}`}
                    startIcon={<DownloadIcon sx={{ fontSize: 12 }} />}
                    sx={{ fontSize: "0.55rem", textTransform: "none", py: 0, minWidth: 0 }}>
                    Download
                  </Button>
                </Box>
              ))}
            </Box>
          )}
        </Box>
      </Drawer>

      {/* Console out */}
      {!drawerOpen && (
        <Button size="small" variant="contained" onClick={() => setDrawerOpen(true)}
          sx={{ position: "absolute", bottom: 8, right: 8, fontSize: "0.6rem", textTransform: "none", py: 0.25, zIndex: 5 }}>
          Show outputs ({allOutputs.length})
        </Button>
      )}
      {consoleOut && (
        <Tooltip placement="top" title={<pre style={{ margin: 0, fontSize: 10, maxHeight: 320, overflow: "auto", whiteSpace: "pre" }}>{consoleOut}</pre>}>
          <Button size="small" variant="text" onClick={() => { consoleRef.current = ""; setConsoleOut(""); }}
            sx={{ position: "absolute", bottom: 8, left: 8, fontSize: "0.55rem", textTransform: "none", py: 0, zIndex: 5, color: "text.secondary" }}>
            console ({consoleOut.split("\n").length} lines)
          </Button>
        </Tooltip>
      )}
      </Box>{/* end canvas + side-panel row */}
    </Box>
  );
}
