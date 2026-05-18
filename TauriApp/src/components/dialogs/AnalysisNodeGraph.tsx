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

// ── Starter snippets ─────────────────────────────────────────

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
  // Graph state.
  const [nodes, setNodes] = useState<Node<NodeData>[]>(() => [
    {
      id: "source",
      type: "source",
      position: { x: 30, y: 30 },
      data: { label: "Source", kind: "source", sources: [], status: "ok" } as NodeData,
      draggable: true,
      deletable: false,
    },
  ]);
  const [edges, setEdges] = useState<Edge[]>([]);
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
        setInsetSources(r.sources || []);
        setNodes((cur) => cur.map((n) =>
          n.id === "source" ? { ...n, data: { ...n.data, sources: r.sources || [] } } : n
        ));
      })
      .catch(() => setInsetSources([]));
    api.checkMatlab().then((m) => setMatlabKind(m.kind || "")).catch(() => setMatlabKind(""));
  }, [open]);

  // ── Graph handlers ─────────────────────────────────────────

  const onNodesChange = useCallback((changes: NodeChange<Node<NodeData>>[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  }, []);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  }, []);

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

  const addProcessNode = useCallback((engine: EngineKind) => {
    const id = newId(engine);
    const defaultCode = engine === "python" ? PY_DEFAULT : engine === "matlab" ? ML_DEFAULT : R_DEFAULT;
    setNodes((cur) => [
      ...cur,
      {
        id,
        type: engine,
        position: { x: 350 + Math.random() * 100, y: 120 + Math.random() * 200 },
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
  }, []);

  const removeNode = useCallback((nodeId: string) => {
    if (nodeId === "source") return;
    setNodes((cur) => cur.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  }, [selectedNodeId]);

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
    <Box sx={{ position: "relative", flex: 1, display: "flex", minHeight: 0 }}>
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
      {selectedNode && selectedNode.id !== "source" && (
        <Box sx={{ width: 460, borderLeft: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", minWidth: 0, bgcolor: "background.paper" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 1, borderBottom: "1px solid", borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.75rem" }}>
              {KIND_ICON[selectedNode.data.kind as EngineKind]} {selectedNode.data.label}
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
      )}

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
    </Box>
  );
}
