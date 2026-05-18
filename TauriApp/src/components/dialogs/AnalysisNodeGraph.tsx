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

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
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
  Dialog,
  DialogTitle,
  DialogContent,
  Checkbox,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AddIcon from "@mui/icons-material/Add";
import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";
import DownloadIcon from "@mui/icons-material/Download";
import CloseIcon from "@mui/icons-material/Close";
import OpenInFullIcon from "@mui/icons-material/OpenInFull";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import HourglassEmptyIcon from "@mui/icons-material/HourglassEmpty";
import EditIcon from "@mui/icons-material/Edit";
import TextField from "@mui/material/TextField";
import DialogActions from "@mui/material/DialogActions";
import CodeMirror from "@uiw/react-codemirror";
import { python as cmPython } from "@codemirror/lang-python";
import { StreamLanguage } from "@codemirror/language";
import { r as cmR } from "@codemirror/legacy-modes/mode/r";
import { oneDark } from "@codemirror/theme-one-dark";
import { api } from "../../api/client";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";

// ── Helper: base64 PNG → File (for main-timeline + collage uploads) ──
function b64ToFile(b64: string, filename: string): File {
  const bin = atob(b64);
  const len = bin.length;
  const u8 = new Uint8Array(len);
  for (let i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
  return new File([u8], filename, { type: "image/png" });
}

// Natural image dimensions (used to scale the collage drop).
function imageDims(b64: string): Promise<{ w: number; h: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth || 400, h: img.naturalHeight || 300 });
    img.onerror = () => resolve({ w: 400, h: 300 });
    img.src = `data:image/png;base64,${b64}`;
  });
}

/** Parse a code blob for output-declaration calls — `mpfig_data(...)`,
 *  `mpfig_image(...)`, `mpfig_plot(...)`.  Returns the kind + name for
 *  each call we recognise.  Best-effort regex scan; tolerates Python,
 *  R, MATLAB and IJ-macro string-quoting conventions.
 *
 *  Examples that match:
 *      mpfig_data(rows, "haze_analysis")
 *      mpfig_data(rows, name="haze_analysis")
 *      mpfig_image(arr, "thresholded")
 *      mpfig_plot("bar.png")
 *      mpfig_data(struct(...), 'stats')
 *      mpfig_data("particles", labels, a, b)
 */
function parseDeclaredOutputs(code: string): DeclaredOutput[] {
  if (!code) return [];
  const seen = new Set<string>();
  const out: DeclaredOutput[] = [];
  // Run a global regex for each helper.  For mpfig_data / mpfig_image
  // we look for the FIRST string literal inside the parentheses (the
  // name arg lands there in every flavour we ship).
  const patterns: Array<{ kind: DataKind; re: RegExp }> = [
    { kind: "plot",  re: /mpfig_plot\s*\(\s*["']([^"']+?)\.?(?:png|pdf|svg)?["']/g },
    { kind: "table", re: /mpfig_data\s*\(\s*(?:[^()"',]+,\s*)*?(?:name\s*=\s*)?["']([^"']+)["']/g },
    { kind: "image", re: /mpfig_image\s*\(\s*(?:[^()"',]+,\s*)*?(?:name\s*=\s*)?["']([^"']+)["']/g },
  ];
  for (const { kind, re } of patterns) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(code)) !== null) {
      const raw = m[1].trim();
      const name = raw.replace(/\.(png|pdf|svg|csv)$/i, "");
      const key = `${kind}:${name}`;
      if (seen.has(key) || !name) continue;
      seen.add(key);
      out.push({ kind, name });
    }
  }
  return out;
}

// ── Per-node Console panel (under the code editor) ────────────
// Renders the most recent stdout / stderr / traceback for one node.
// Collapsible — collapsed by default for clean / no-output runs,
// auto-opens on error so the user sees the failure inline.
function NodeConsolePanel({ nodeId, text, status }: {
  nodeId: string;
  text: string;
  status?: NodeData["status"];
}) {
  // Per-node open state — keyed on nodeId via the key= prop in the
  // parent, so switching nodes always shows a fresh collapsed state
  // (unless the node is in error, see below).
  const isError = status === "error";
  const [open, setOpen] = useState(isError);
  // Auto-open whenever the node enters error after a run.
  useEffect(() => {
    if (isError) setOpen(true);
  }, [isError]);
  // Render even when empty so users can manually expand if they need
  // to confirm "no output".
  const hasOutput = text && text.trim().length > 0;
  const lineCount = hasOutput ? text.split("\n").length : 0;
  void nodeId;
  return (
    <Box sx={{ borderTop: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", flexShrink: 0 }}>
      <Box
        onClick={() => setOpen((v) => !v)}
        sx={{
          display: "flex", alignItems: "center", gap: 0.5,
          px: 1, py: 0.25, cursor: "pointer",
          bgcolor: isError ? "error.dark" : "action.hover",
          color: isError ? "error.contrastText" : "text.primary",
          "&:hover": { bgcolor: isError ? "error.main" : "action.selected" },
        }}
      >
        {open ? <ExpandMoreIcon sx={{ fontSize: 14 }} /> : <ExpandLessIcon sx={{ fontSize: 14, transform: "rotate(-90deg)" }} />}
        <Typography variant="caption" sx={{ fontSize: "0.6rem", fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase", flex: 1 }}>
          Console {isError ? "· ERROR" : ""}
        </Typography>
        <Typography variant="caption" sx={{ fontSize: "0.55rem", opacity: 0.75 }}>
          {hasOutput ? `${lineCount} line${lineCount === 1 ? "" : "s"}` : "(no output)"}
        </Typography>
      </Box>
      {open && (
        <Box sx={{
          maxHeight: 200, overflow: "auto",
          fontFamily: "monospace", fontSize: "0.65rem",
          whiteSpace: "pre-wrap", wordBreak: "break-word",
          p: 0.75, bgcolor: isError ? "rgba(244, 67, 54, 0.07)" : "background.default",
          color: isError ? "error.main" : "text.primary",
        }}>
          {hasOutput ? text : <span style={{ opacity: 0.6, fontStyle: "italic" }}>Run this node to see stdout / stderr here.</span>}
        </Box>
      )}
    </Box>
  );
}

// ── Drag-and-drop staging zone (drawer right column) ─────────
// Drop target for output cards from the left grid.  Lists the
// staged items as small chips with an × to unstage; a Commit
// button at the bottom pushes the whole batch to its destination.
function StagingZone({ title, hint, bucket, staged, allOutputs, onAdd, onRemove, onCommit }: {
  title: string;
  hint: string;
  bucket: "main" | "collage";
  staged: Set<string>;
  allOutputs: AggregatedOutput[];
  onAdd: (key: string) => void;
  onRemove: (key: string) => void;
  onCommit: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  const stagedList = allOutputs.filter((o) => staged.has(`${o.nodeId}-${o.outputId}`));
  return (
    <Box
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("application/x-mpfig-output")) {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
          if (!hovered) setHovered(true);
        }
      }}
      onDragLeave={() => setHovered(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHovered(false);
        const key = e.dataTransfer.getData("application/x-mpfig-output");
        if (key) onAdd(key);
      }}
      sx={{
        flex: 1, minHeight: 0,
        display: "flex", flexDirection: "column",
        p: 0.75,
        borderBottom: bucket === "main" ? "1px solid" : undefined,
        borderColor: "divider",
        bgcolor: hovered ? "action.selected" : "background.paper",
        transition: "background-color 120ms",
        outline: hovered ? "2px dashed" : "none",
        outlineColor: "primary.main",
        outlineOffset: -4,
      }}
    >
      <Typography variant="caption" sx={{ fontSize: "0.65rem", fontWeight: 700, display: "block" }}>
        {title} ({stagedList.length})
      </Typography>
      <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", display: "block", mt: 0.25, lineHeight: 1.3 }}>
        {hint}
      </Typography>
      <Box sx={{ flex: 1, overflow: "auto", mt: 0.5, minHeight: 40 }}>
        {stagedList.length === 0 ? (
          <Box sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", border: "1px dashed", borderColor: "divider", borderRadius: 0.5, p: 0.5 }}>
            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.disabled", fontStyle: "italic" }}>
              drop here
            </Typography>
          </Box>
        ) : (
          stagedList.map((o) => {
            const key = `${o.nodeId}-${o.outputId}`;
            return (
              <Box key={key} sx={{ display: "flex", alignItems: "center", gap: 0.4, py: 0.25, px: 0.4, mb: 0.25, borderRadius: 0.25, bgcolor: "background.default", border: "1px solid", borderColor: "divider" }}>
                <Box component="img" src={`data:image/png;base64,${o.payload}`} alt={o.name}
                  sx={{ width: 22, height: 22, objectFit: "contain", flexShrink: 0, border: "1px solid", borderColor: "divider", borderRadius: 0.25 }} />
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600, display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {o.name}
                  </Typography>
                  <Typography variant="caption" sx={{ fontSize: "0.45rem", color: "text.secondary", lineHeight: 1 }}>
                    {o.nodeLabel}
                  </Typography>
                </Box>
                <Box component="span" onClick={() => onRemove(key)}
                  sx={{ fontSize: "0.7rem", cursor: "pointer", color: "text.disabled", px: 0.3, "&:hover": { color: "error.main" } }}>×</Box>
              </Box>
            );
          })
        )}
      </Box>
      <Button size="small" variant="contained" color="primary"
        onClick={onCommit}
        disabled={stagedList.length === 0}
        sx={{ mt: 0.5, fontSize: "0.6rem", textTransform: "none", py: 0.25 }}>
        Send {stagedList.length || ""} to {bucket === "main" ? "main timeline" : "collage"}
      </Button>
    </Box>
  );
}

// ── User-defined source-library group ──────────────────────
// Drop-target accordion: drag an inset onto the header → adds to
// this group (removes from any prior group).  The × on each row
// pops the inset back to Ungrouped.  Local open state per group.
function UserSourceGroup({
  group, items, onAssignKey, onRemoveKey, onRename, onDelete, isUngrouped,
}: {
  group: SourceGroup;
  items: InsetSource[];
  onAssignKey: (sourceKey: string) => void;
  onRemoveKey: (sourceKey: string) => void;
  onRename?: () => void;
  onDelete?: () => void;
  isUngrouped?: boolean;
}) {
  const [open, setOpen] = useState(true);
  const [hovered, setHovered] = useState(false);
  return (
    <Box sx={{ mb: 0.5 }}>
      <Box
        onClick={() => setOpen((v) => !v)}
        onDragOver={(e) => {
          if (isUngrouped) return;
          if (e.dataTransfer.types.includes("application/x-mpfig-source")) {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            if (!hovered) setHovered(true);
          }
        }}
        onDragLeave={() => setHovered(false)}
        onDrop={(e) => {
          if (isUngrouped) return;
          e.preventDefault();
          setHovered(false);
          const key = e.dataTransfer.getData("application/x-mpfig-source");
          if (key) onAssignKey(key);
        }}
        sx={{
          display: "flex", alignItems: "center", gap: 0.25, cursor: "pointer",
          px: 0.5, py: 0.25, borderRadius: 0.25,
          bgcolor: hovered ? "warning.light" : isUngrouped ? "transparent" : "action.hover",
          border: hovered ? "1px dashed" : isUngrouped ? "none" : "1px solid",
          borderColor: hovered ? "warning.main" : "divider",
          transition: "background-color 100ms",
          "&:hover": { bgcolor: hovered ? "warning.light" : "action.selected" },
        }}
        title={isUngrouped ? undefined : "Drop an inset here to assign it to this group"}
      >
        {open ? <ExpandMoreIcon sx={{ fontSize: 14 }} /> : <ExpandLessIcon sx={{ fontSize: 14, transform: "rotate(-90deg)" }} />}
        <Typography variant="caption" sx={{ fontSize: "0.6rem", fontWeight: 700, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {isUngrouped ? group.name : `📁 ${group.name}`}
        </Typography>
        <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>{items.length}</Typography>
        {!isUngrouped && onRename && (
          <Box
            component="span"
            onClick={(e) => { e.stopPropagation(); onRename(); }}
            sx={{ fontSize: "0.55rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "primary.main" } }}
            title="Rename group"
          >✎</Box>
        )}
        {!isUngrouped && onDelete && (
          <Box
            component="span"
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            sx={{ fontSize: "0.7rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "error.main" } }}
            title="Delete group"
          >×</Box>
        )}
      </Box>
      {open && (
        <Box sx={{ pl: 0.5, mt: 0.25 }}>
          {items.length === 0 ? (
            <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.disabled", fontStyle: "italic", display: "block", px: 0.5, py: 0.5 }}>
              {isUngrouped ? "(no ungrouped sources)" : "Drop insets here from another group"}
            </Typography>
          ) : items.map((s) => (
            <Box
              key={s.key}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData("application/x-mpfig-source", s.key);
                e.dataTransfer.effectAllowed = "copy";
              }}
              sx={{
                display: "flex", alignItems: "center", gap: 0.5,
                px: 0.5, py: 0.3, mb: 0.3, borderRadius: 0.5,
                border: "1px solid", borderColor: "divider",
                bgcolor: "background.default", cursor: "grab",
                "&:hover": { borderColor: "primary.main", bgcolor: "action.hover" },
                "&:active": { cursor: "grabbing" },
              }}
            >
              {s.thumbnail && (
                <Box component="img" src={`data:image/png;base64,${s.thumbnail}`} alt={s.label}
                  sx={{ width: 26, height: 26, objectFit: "contain", borderRadius: 0.25, flexShrink: 0, border: "1px solid", borderColor: "divider" }}
                />
              )}
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <Typography variant="caption" sx={{ fontSize: "0.58rem", fontWeight: 700, display: "block", lineHeight: 1.15, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  R{s.row + 1}C{s.col + 1}·{s.inset_index + 1}
                </Typography>
                <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", lineHeight: 1 }}>
                  {s.natural_width}×{s.natural_height}
                </Typography>
              </Box>
              {!isUngrouped && (
                <Box
                  component="span"
                  onClick={(e) => { e.stopPropagation(); onRemoveKey(s.key); }}
                  sx={{ fontSize: "0.6rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "error.main" } }}
                  title="Move back to Ungrouped"
                >×</Box>
              )}
            </Box>
          ))}
        </Box>
      )}
    </Box>
  );
}

// ── Tauri-friendly prompt / confirm replacements ─────────────
// Tauri's webview disables window.prompt / window.confirm by
// default (no native shell to host the OS-level dialogs), so the
// node-graph dialog ships its own MUI-based equivalents.

interface PromptRequest {
  title: string;
  label?: string;
  defaultValue?: string;
  /** Custom OK label, e.g. "Save". */
  okLabel?: string;
  resolve: (val: string | null) => void;
}
interface ConfirmRequest {
  title: string;
  message: string;
  /** Custom OK label, e.g. "Delete". */
  okLabel?: string;
  resolve: (val: boolean) => void;
}

function usePromptDialog() {
  const [request, setRequest] = useState<PromptRequest | null>(null);
  const prompt = useCallback((opts: Omit<PromptRequest, "resolve">): Promise<string | null> => {
    return new Promise((resolve) => setRequest({ ...opts, resolve }));
  }, []);
  return { request, setRequest, prompt };
}
function useConfirmDialog() {
  const [request, setRequest] = useState<ConfirmRequest | null>(null);
  const confirm = useCallback((opts: Omit<ConfirmRequest, "resolve">): Promise<boolean> => {
    return new Promise((resolve) => setRequest({ ...opts, resolve }));
  }, []);
  return { request, setRequest, confirm };
}

function PromptDialogBody({ request, setRequest }: {
  request: PromptRequest | null;
  setRequest: (r: PromptRequest | null) => void;
}) {
  const [value, setValue] = useState("");
  useEffect(() => { setValue(request?.defaultValue ?? ""); }, [request]);
  if (!request) return null;
  const close = (v: string | null) => { request.resolve(v); setRequest(null); };
  return (
    <Dialog open={true} onClose={() => close(null)} maxWidth="xs" fullWidth>
      <DialogTitle sx={{ fontSize: "0.95rem", py: 1.25 }}>{request.title}</DialogTitle>
      <DialogContent sx={{ pt: "8px !important" }}>
        <TextField
          autoFocus
          fullWidth
          size="small"
          label={request.label}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); close(value); }
            else if (e.key === "Escape") { e.preventDefault(); close(null); }
          }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={() => close(null)}>Cancel</Button>
        <Button onClick={() => close(value)} variant="contained">{request.okLabel || "OK"}</Button>
      </DialogActions>
    </Dialog>
  );
}

function ConfirmDialogBody({ request, setRequest }: {
  request: ConfirmRequest | null;
  setRequest: (r: ConfirmRequest | null) => void;
}) {
  if (!request) return null;
  const close = (v: boolean) => { request.resolve(v); setRequest(null); };
  return (
    <Dialog open={true} onClose={() => close(false)} maxWidth="xs" fullWidth>
      <DialogTitle sx={{ fontSize: "0.95rem", py: 1.25 }}>{request.title}</DialogTitle>
      <DialogContent sx={{ pt: "8px !important" }}>
        <Typography variant="body2">{request.message}</Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => close(false)}>Cancel</Button>
        <Button onClick={() => close(true)} variant="contained" color={request.okLabel?.toLowerCase().includes("delete") ? "error" : "primary"}>
          {request.okLabel || "OK"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ── Cross-component callbacks for the custom node renderers ──
// React Flow renders our SourceNode / ProcessNode components without
// passing any extra props — they only receive the `data` object. To
// let those nodes call back into the parent (drop-to-attach, output
// chip clicks, etc.) we use a React context that the canvas wraps
// around its ReactFlow tree.
interface GraphCallbacks {
  /** Attach an inset (by InsetSource.key) to a SourceNode's list. */
  addSourceToNode: (nodeId: string, sourceKey: string) => void;
  /** Detach an inset (by index) from a SourceNode's list. */
  removeSourceFromNode: (nodeId: string, idx: number) => void;
  /** Open the output drawer focused on this output + briefly flash it. */
  navigateToOutput: (nodeId: string, outputId: string, kind: DataKind) => void;
  /** Open the large-preview modal for an output. */
  openPreview: (nodeId: string, outputId: string) => void;
  /** Live InsetSource list — SourceNodes look up labels / thumbnails
   *  by key when re-rendering after a drag-drop. */
  insetSources: InsetSource[];
  /** True when the host figure has measurements to expose — source
   *  nodes use this to decide whether to render the measurements port. */
  hasMeasurements: boolean;
  /** Node id that should currently render an upstream-highlight ring. */
  highlightedUpstreamIds: Set<string>;
  /** Per-node incoming-edge counts (image / table).  Derived from
   *  the live edge set via useMemo so node cards can render a
   *  non-zero in-count the moment the user wires an edge. */
  inputCountsByNode: Map<string, { image: number; table: number }>;
  /** Per-node declared outputs (parsed from code).  Surfaced as
   *  outlined chips on the node card BEFORE the first run lands. */
  declaredByNode: Map<string, DeclaredOutput[]>;
}
const GraphCallbacksContext = createContext<GraphCallbacks | null>(null);
function useGraphCallbacks(): GraphCallbacks | null {
  return useContext(GraphCallbacksContext);
}

// ── Types ────────────────────────────────────────────────────

export type DataKind = "image" | "table" | "plot";
export type EngineKind = "python" | "matlab" | "r" | "imagej";

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

/** Placeholder output entry — what the code DECLARES it will produce.
 *  Surfaced on the node card as outlined chips before a run lands so
 *  users can plan downstream wiring while still drafting their code. */
export interface DeclaredOutput {
  kind: DataKind;
  name: string;
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
  /** Per-node combined stdout / stderr from the most recent run.
   *  Rendered as a collapsible Console panel under the code editor
   *  in the detail panel — keeps run output close to the code that
   *  produced it instead of dumped into the (deprecated) global
   *  console at the bottom of the canvas. */
  consoleOut?: string;
  /** Outputs the user's code DECLARES it will produce — parsed from
   *  `mpfig_data(...)` / `mpfig_image(...)` / `mpfig_plot(...)` calls.
   *  Shown as outlined placeholder chips on the node card so the user
   *  can plan downstream wiring before they hit Run. */
  declaredOutputs?: DeclaredOutput[];
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

// Cohesive node-header palette — desaturated cool greys + warm
// accents tuned to read as a single family rather than four
// unrelated tools. Borders on the cards use the lighter tint
// (KIND_BORDER) so the colour wash on the header doesn't have to
// fight with the rest of the card.
const KIND_COLOR: Record<"source" | EngineKind, string> = {
  source: "#546e7a",   // blue-grey 600
  python: "#3f6790",   // muted indigo
  matlab: "#8a6d3b",   // bronze
  r:      "#5f7a4f",   // sage
  imagej: "#6e5a8a",   // muted violet
};
const KIND_BORDER: Record<"source" | EngineKind, string> = {
  source: "#90a4ae",
  python: "#6f8aa8",
  matlab: "#b69266",
  r:      "#8aa37b",
  imagej: "#9886b3",
};

const KIND_ICON: Record<"source" | EngineKind, string> = {
  source: "📥",
  python: "🐍",
  matlab: "📐",
  r: "📊",
  imagej: "🔬",
};

const PORT_COLOR: Record<DataKind, string> = {
  image: "#4a9d92",   // teal
  table: "#9b7bba",   // muted lavender
  plot:  "#c98a47",   // burnt sienna
};

// ── Node card components ─────────────────────────────────────

interface NodeCardProps {
  data: NodeData;
  id: string;
  selected?: boolean;
}

function StatusPip({ status }: { status?: NodeData["status"] }) {
  // Running → spinning hourglass; everything else → small colored dot.
  if (status === "running") {
    return (
      <Tooltip placement="top" title="running">
        <HourglassEmptyIcon
          sx={{
            fontSize: 14,
            color: "white",
            animation: "mpfig-spin 1s linear infinite",
            "@keyframes mpfig-spin": {
              "0%": { transform: "rotate(0deg)" },
              "100%": { transform: "rotate(360deg)" },
            },
          }}
        />
      </Tooltip>
    );
  }
  const colour = status === "ok" ? "#67c187"
    : status === "error" ? "#e57373"
    : status === "stale" ? "#bdbdbd"
    : "#eceff1";
  const label = status === "ok" ? "fresh"
    : status === "error" ? "error"
    : status === "stale" ? "stale"
    : "idle";
  return (
    <Tooltip placement="top" title={label}>
      <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: colour, border: "1px solid rgba(255,255,255,0.6)" }} />
    </Tooltip>
  );
}

/** Source node — exposes inset thumbnails as draggable output ports.
 *  Accepts inset drops from the left-hand library panel (drag a thumb
 *  onto a source node to attach it). Measurements port is exposed
 *  on every source node when the host figure has measurement data. */
function SourceNode({ data, id }: NodeCardProps) {
  const sources = data.sources || [];
  const cbs = useGraphCallbacks();
  const showMeasurements = !!cbs?.hasMeasurements;
  const isUpstreamHi = cbs?.highlightedUpstreamIds.has(id) ?? false;
  const [dragOver, setDragOver] = useState(false);

  const onDragOver = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes("application/x-mpfig-source")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      if (!dragOver) setDragOver(true);
    }
  };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const key = e.dataTransfer.getData("application/x-mpfig-source");
    if (key && cbs) cbs.addSourceToNode(id, key);
  };

  return (
    <Box
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      sx={{
        minWidth: 140,
        maxWidth: 200,
        bgcolor: "background.paper",
        border: "2px solid",
        borderColor: dragOver ? "warning.main" : isUpstreamHi ? "#ffd54f" : KIND_BORDER.source,
        borderRadius: 1,
        boxShadow: isUpstreamHi ? "0 0 0 4px rgba(255,213,79,0.45)" : 2,
        transition: "border-color 100ms, box-shadow 200ms",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 0.75, py: 0.25, bgcolor: KIND_COLOR.source, color: "white", borderRadius: "4px 4px 0 0" }}>
        <Typography variant="caption" sx={{ fontSize: "0.62rem", fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {KIND_ICON.source} {data.label || "Source"}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <StatusPip status={data.status} />
      </Box>
      <Box sx={{ p: 0.25 }}>
        {sources.length === 0 ? (
          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: dragOver ? "warning.main" : "text.secondary", fontStyle: "italic", display: "block", px: 0.5, py: 0.5, textAlign: "center", lineHeight: 1.3 }}>
            {dragOver ? "Drop to attach" : "Drag insets here from the left panel"}
          </Typography>
        ) : (
          sources.map((s, idx) => (
            <Box key={`${s.key}_${idx}`} sx={{ position: "relative", display: "flex", alignItems: "center", gap: 0.4, py: 0.15, pr: 1.5 }}>
              {s.thumbnail && (
                <Box component="img" src={`data:image/png;base64,${s.thumbnail}`} alt={s.label}
                  sx={{ width: 18, height: 18, objectFit: "contain", border: "1px solid", borderColor: "divider", borderRadius: 0.25, flexShrink: 0 }}
                />
              )}
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600, display: "block", lineHeight: 1.1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  R{s.row + 1}C{s.col + 1}·{s.inset_index + 1}
                </Typography>
                <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", lineHeight: 1 }}>
                  {s.natural_width}×{s.natural_height}
                </Typography>
              </Box>
              <Box
                component="span"
                onClick={(e) => { e.stopPropagation(); cbs?.removeSourceFromNode(id, idx); }}
                sx={{ fontSize: "0.55rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "error.main" } }}
              >
                ×
              </Box>
              <Handle
                type="source"
                position={Position.Right}
                id={`out_image_${idx}`}
                style={{ background: PORT_COLOR.image, width: 8, height: 8, border: "2px solid white", right: -4 }}
              />
            </Box>
          ))
        )}
        {/* Measurements port — every source node exposes it when the
            host figure has measurement data, so any source can feed
            an R node downstream. The runner dedups by sourceHandle
            so multiple sources wiring measurements is harmless. */}
        {showMeasurements && (
          <Box sx={{ position: "relative", display: "flex", alignItems: "center", gap: 0.4, py: 0.25, pr: 1.5, borderTop: "1px dashed", borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600, flex: 1 }}>📋 measurements</Typography>
            <Handle
              type="source"
              position={Position.Right}
              id="out_table_measurements"
              style={{ background: PORT_COLOR.table, width: 8, height: 8, border: "2px solid white", right: -4 }}
            />
          </Box>
        )}
      </Box>
    </Box>
  );
}

/** Process node — Python / MATLAB / R / ImageJ. */
function ProcessNode({ data, id }: NodeCardProps) {
  const engine = data.kind as EngineKind;
  const colour = KIND_COLOR[engine];
  const outputs = data.outputs || [];
  const cbs = useGraphCallbacks();
  const isUpstreamHi = cbs?.highlightedUpstreamIds.has(id) ?? false;
  // Live counts come from context (derived from the edge list) so we
  // don't have to keep `data.inputs` synced via a setNodes-from-
  // effect — that loop fed into React Flow's store and blew up.
  const inCounts = cbs?.inputCountsByNode.get(id) || { image: 0, table: 0 };
  // Allow plot outputs ONLY for R nodes (Python/MATLAB/ImageJ are
  // data-extraction by design).
  const allowedOutKinds: DataKind[] = engine === "r" ? ["plot", "table"] : ["image", "table"];
  void allowedOutKinds;

  // Kind glyph for the output indicator chips on the card.
  const kindGlyph: Record<DataKind, string> = { image: "🖼", table: "📋", plot: "📊" };

  return (
    <Box sx={{
      minWidth: 150,
      maxWidth: 220,
      bgcolor: "background.paper",
      border: "2px solid",
      borderColor: isUpstreamHi ? "#ffd54f" : KIND_BORDER[engine],
      borderRadius: 1,
      boxShadow: isUpstreamHi ? "0 0 0 4px rgba(255,213,79,0.45)" : 2,
      transition: "border-color 100ms, box-shadow 200ms",
    }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 0.75, py: 0.25, bgcolor: colour, color: "white", borderRadius: "4px 4px 0 0" }}>
        <Typography variant="caption" sx={{ fontSize: "0.62rem", fontWeight: 700 }}>
          {KIND_ICON[engine]} {data.label}
        </Typography>
        <Box sx={{ flex: 1 }} />
        <StatusPip status={data.status} />
      </Box>
      <Box sx={{ display: "flex" }}>
        {/* Input ports (left edge) */}
        <Box sx={{ position: "relative", width: 14, py: 0.25 }}>
          {/* image input */}
          <Handle
            type="target"
            position={Position.Left}
            id="in_image"
            style={{ background: PORT_COLOR.image, top: 18, width: 8, height: 8, border: "2px solid white", left: -4 }}
          />
          {/* table input */}
          <Handle
            type="target"
            position={Position.Left}
            id="in_table"
            style={{ background: PORT_COLOR.table, top: 40, width: 8, height: 8, border: "2px solid white", left: -4 }}
          />
        </Box>
        <Box sx={{ flex: 1, p: 0.25, minWidth: 0 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, py: 0.15 }}>
            <Box sx={{ width: 5, height: 5, borderRadius: "50%", bgcolor: PORT_COLOR.image }} />
            <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>image ({inCounts.image})</Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, py: 0.15 }}>
            <Box sx={{ width: 5, height: 5, borderRadius: "50%", bgcolor: PORT_COLOR.table }} />
            <Typography variant="caption" sx={{ fontSize: "0.55rem" }}>table ({inCounts.table})</Typography>
          </Box>
          {data.error && (
            <Tooltip placement="top" title={data.error}>
              <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "error.main", display: "block", mt: 0.25, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                ⚠ {data.error}
              </Typography>
            </Tooltip>
          )}
        </Box>
        {/* Output ports (right edge) — three handles per kind so
            the user can route plot/table/image separately. */}
        <Box sx={{ position: "relative", width: 14, py: 0.25 }}>
          {engine !== "r" && (
            <Handle
              type="source"
              position={Position.Right}
              id="out_image"
              style={{ background: PORT_COLOR.image, top: 18, width: 8, height: 8, border: "2px solid white", right: -4 }}
            />
          )}
          <Handle
            type="source"
            position={Position.Right}
            id="out_table"
            style={{ background: PORT_COLOR.table, top: engine === "r" ? 18 : 40, width: 8, height: 8, border: "2px solid white", right: -4 }}
          />
          {engine === "r" && (
            <Handle
              type="source"
              position={Position.Right}
              id="out_plot"
              style={{ background: PORT_COLOR.plot, top: 40, width: 8, height: 8, border: "2px solid white", right: -4 }}
            />
          )}
        </Box>
      </Box>
      {/* Output indicator chips — one per produced output, click to
          open the drawer at that output. Each chip is small (one glyph)
          so the node footprint stays compact.  When the code DECLARES
          an output (via mpfig_data / mpfig_image / mpfig_plot) but the
          node hasn't been run yet, we show an outlined PLACEHOLDER
          chip so the user can plan downstream wiring before the
          first run lands.  Real outputs win over placeholders with
          the same name. */}
      {(() => {
        const declared = cbs?.declaredByNode.get(id) || [];
        const realNames = new Set(outputs.map((o) => o.name));
        const placeholders = declared.filter((d) => !realNames.has(d.name));
        if (outputs.length === 0 && placeholders.length === 0) return null;
        return (
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.25, px: 0.5, pb: 0.4, pt: 0.1, borderTop: "1px dashed", borderColor: "divider" }}>
            {outputs.map((o) => (
              <Tooltip key={o.id} placement="bottom" title={`${o.kind}: ${o.name} — click to view`}>
                <Box
                  onClick={(e) => { e.stopPropagation(); cbs?.navigateToOutput(id, o.id, o.kind); }}
                  onDoubleClick={(e) => { e.stopPropagation(); cbs?.openPreview(id, o.id); }}
                  sx={{
                    display: "inline-flex", alignItems: "center", gap: 0.25,
                    fontSize: "0.55rem", lineHeight: 1,
                    px: 0.4, py: 0.2, borderRadius: 0.25,
                    bgcolor: "action.hover", cursor: "pointer",
                    border: "1px solid", borderColor: "divider",
                    "&:hover": { borderColor: PORT_COLOR[o.kind], bgcolor: "action.selected" },
                  }}
                >
                  <span>{kindGlyph[o.kind]}</span>
                  <span style={{ maxWidth: 64, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{o.name}</span>
                </Box>
              </Tooltip>
            ))}
            {placeholders.map((d, i) => (
              <Tooltip key={`p_${d.kind}_${d.name}_${i}`} placement="bottom" title={`Declared ${d.kind}: ${d.name} — will appear here after the node runs`}>
                <Box
                  sx={{
                    display: "inline-flex", alignItems: "center", gap: 0.25,
                    fontSize: "0.55rem", lineHeight: 1,
                    px: 0.4, py: 0.2, borderRadius: 0.25,
                    color: PORT_COLOR[d.kind],
                    border: "1px dashed", borderColor: PORT_COLOR[d.kind],
                    bgcolor: "transparent", opacity: 0.75,
                  }}
                >
                  <span>{kindGlyph[d.kind]}</span>
                  <span style={{ maxWidth: 64, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontStyle: "italic" }}>{d.name}</span>
                </Box>
              </Tooltip>
            ))}
          </Box>
        );
      })()}
    </Box>
  );
}

const nodeTypes = {
  source: SourceNode,
  python: ProcessNode,
  matlab: ProcessNode,
  r: ProcessNode,
  imagej: ProcessNode,
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

// ── ImageJ / Fiji presets (macro language) ─────────────────
// The backend stub will return a clear "Fiji not installed"
// error when there's no `ImageJ-*` or `fiji` binary on PATH;
// users who do install Fiji get headless macro execution.
const IJ_DEFAULT = `// ImageJ / Fiji macro — extracts metrics from each input image.
// @name: ImageJ analysis

setBatchMode(true);
n = lengthOf(inputs);
labels = newArray(n);
means  = newArray(n);
maxes  = newArray(n);
for (i = 0; i < n; i++) {
  open(inputs[i].path);
  labels[i] = inputs[i].label;
  getStatistics(area, mean, min, max);
  means[i] = mean;
  maxes[i] = max;
  close();
}
mpfig_data("stats", labels, means, maxes);
`;
const IJ_PARTICLES = `// PARTICLE ANALYSIS — threshold + count + size per image.
// @name: Particle analysis

setBatchMode(true);
n = lengthOf(inputs);
labels = newArray(n);
counts = newArray(n);
sizes  = newArray(n);
for (i = 0; i < n; i++) {
  open(inputs[i].path);
  run("8-bit");
  setAutoThreshold("Otsu dark");
  run("Convert to Mask");
  run("Analyze Particles...", "size=10-Infinity show=Outlines summarize");
  labels[i] = inputs[i].label;
  counts[i] = nResults;
  sizes[i]  = (nResults > 0) ? getResult("Area", 0) : 0;
  close();
  close();
}
mpfig_data("particles", labels, counts, sizes);
`;
const IMAGEJ_PRESETS: CodePreset[] = [
  { name: "Custom (starter)", code: IJ_DEFAULT },
  { name: "Particle analysis", code: IJ_PARTICLES },
];

const BUILTIN_PRESETS: Record<EngineKind, CodePreset[]> = {
  python: PYTHON_PRESETS,
  matlab: MATLAB_PRESETS,
  r: R_PRESETS,
  imagej: IMAGEJ_PRESETS,
};

// ── User-saved code presets (localStorage) ──────────────────
const USER_PRESET_KEY: Record<EngineKind, string> = {
  python: "mpfig.user_presets.python",
  matlab: "mpfig.user_presets.matlab",
  r: "mpfig.user_presets.r",
  imagej: "mpfig.user_presets.imagej",
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

// ── User-defined source-library groups (localStorage) ─────────
// Each group has a name and a list of inset-source `.key` strings.
// Sources not in any group fall into a synthetic "Ungrouped" section.
// Groups persist across sessions so a user's organisational scheme
// follows the figure they're working on.
const SOURCE_GROUPS_KEY = "mpfig.source_groups";
interface SourceGroup {
  id: string;
  name: string;
  sourceKeys: string[];
}
function loadSourceGroups(): SourceGroup[] {
  try { return JSON.parse(localStorage.getItem(SOURCE_GROUPS_KEY) || "[]"); }
  catch { return []; }
}
function saveSourceGroups(gs: SourceGroup[]) {
  try { localStorage.setItem(SOURCE_GROUPS_KEY, JSON.stringify(gs)); } catch { /* ignore */ }
}

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

function newSourceNode(sources: InsetSource[], opts?: { id?: string; label?: string; position?: { x: number; y: number } }): Node<NodeData> {
  const id = opts?.id ?? "source";
  return {
    id,
    type: "source",
    position: opts?.position ?? { x: 30, y: 30 },
    data: { label: opts?.label ?? "Source", kind: "source", sources, status: "ok" } as NodeData,
    draggable: true,
    // Only the primary source is undeletable — extra sources can be removed.
    deletable: id !== "source",
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
  // setNodes / setEdges must BAIL OUT when the updater returns the
  // SAME reference — otherwise the surrounding `.map(...)` always
  // produces a new tabs array, which forces a re-render even on a
  // no-op update. Combined with the parseDeclaredOutputs effect
  // (which runs on every node-state change), the lack of a bailout
  // produces an "infinite update" crash that blanks the dialog.
  const setNodes = useCallback((updater: (n: Node<NodeData>[]) => Node<NodeData>[]) => {
    setWorkflowTabs((tabs) => {
      let touched = false;
      const next = tabs.map((t) => {
        if (t.id !== activeWf.id) return t;
        const nn = updater(t.nodes);
        if (nn === t.nodes) return t;
        touched = true;
        return { ...t, nodes: nn };
      });
      return touched ? next : tabs;
    });
  }, [activeWf.id]);
  const setEdges = useCallback((updater: (e: Edge[]) => Edge[]) => {
    setWorkflowTabs((tabs) => {
      let touched = false;
      const next = tabs.map((t) => {
        if (t.id !== activeWf.id) return t;
        const ne = updater(t.edges);
        if (ne === t.edges) return t;
        touched = true;
        return { ...t, edges: ne };
      });
      return touched ? next : tabs;
    });
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
  // Set of currently-selected node ids + edge ids — populated by
  // React Flow's onSelectionChange.  Drives the visible "Delete
  // selected" affordance in the canvas toolbar (the only reliable
  // way to discover the delete action — Backspace works but isn't
  // a discoverable handle).
  const [rfSelection, setRfSelection] = useState<{ nodeIds: Set<string>; edgeIds: Set<string> }>(() => ({ nodeIds: new Set(), edgeIds: new Set() }));
  const [runningGraph, setRunningGraph] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [drawerTab, setDrawerTab] = useState<DataKind | "console">("plot");
  const consoleRef = useRef<string>("");
  const [consoleOut, setConsoleOut] = useState("");

  // Output viewer state — preview modal (single output blown up) and
  // multi-selection for the drawer's batch actions (download / send
  // to main timeline / send to collage).
  const [previewOutput, setPreviewOutput] = useState<{ nodeId: string; outputId: string } | null>(null);
  const [selectedOutputKeys, setSelectedOutputKeys] = useState<Set<string>>(new Set());
  // Output that's been "navigated to" by clicking a chip on a process
  // node — highlights the matching drawer card for a couple of
  // seconds so the user can find it amongst many.
  const [highlightOutputKey, setHighlightOutputKey] = useState<string | null>(null);
  // Node ids that are upstream of the currently-clicked output chip,
  // used by SourceNode / ProcessNode to render a flashing ring.
  const [highlightedUpstreamIds, setHighlightedUpstreamIds] = useState<Set<string>>(new Set());

  // Tauri-friendly prompt / confirm replacements.
  const { request: promptRequest, setRequest: setPromptRequest, prompt } = usePromptDialog();
  const { request: confirmRequest, setRequest: setConfirmRequest, confirm } = useConfirmDialog();

  // Width of the right detail panel — user-resizable via a drag
  // handle on the panel's left edge. Default wider than the old
  // 460 so there's more room for code without hiding the canvas.
  const [detailPanelWidth, setDetailPanelWidth] = useState<number>(620);
  const [resizingPanel, setResizingPanel] = useState(false);

  // User-saved code presets per engine — kept in state so the
  // preset dropdown re-renders the moment a new one lands, and
  // hydrated from localStorage on first mount so they survive
  // app restarts.
  const [userPresetsByEngine, setUserPresetsByEngine] = useState<Record<EngineKind, CodePreset[]>>(() => ({
    python: loadUserPresets("python"),
    matlab: loadUserPresets("matlab"),
    r: loadUserPresets("r"),
    imagej: loadUserPresets("imagej"),
  }));
  const persistUserPresets = useCallback((engine: EngineKind, next: CodePreset[]) => {
    saveUserPresets(engine, next);
    setUserPresetsByEngine((cur) => ({ ...cur, [engine]: next }));
  }, []);

  // Drawer staging — items the user has "queued" for the main
  // timeline or the collage. They're greyed-out in the main output
  // grid while in the staging zone, then committed via the staging
  // toolbar buttons.
  const [stagedForMain, setStagedForMain] = useState<Set<string>>(new Set());
  const [stagedForCollage, setStagedForCollage] = useState<Set<string>>(new Set());

  // ImageJ availability — surfaced like checkMatlab.
  const [imagejKind, setImagejKind] = useState<string>("");

  // Console / snippet runner state.
  const [consoleEngine, setConsoleEngine] = useState<EngineKind>("python");
  const [consoleSnippet, setConsoleSnippet] = useState<string>("");
  const [consoleSnippetRunning, setConsoleSnippetRunning] = useState(false);

  // Measurements present? Falsy CSV (empty / null) hides the
  // measurements port on every source node + any related hints.
  const hasMeasurements = !!(measurementsCsv && measurementsCsv.trim().length > 0);

  // ── User-defined source-library groups ───────────────────────
  // Hydrated from localStorage on mount; each mutation persists back.
  const [sourceGroups, setSourceGroups] = useState<SourceGroup[]>(() => loadSourceGroups());
  const persistSourceGroups = useCallback((next: SourceGroup[]) => {
    setSourceGroups(next);
    saveSourceGroups(next);
  }, []);
  const addSourceGroupHandler = useCallback(async () => {
    const name = await prompt({ title: "New source group", label: "Group name", okLabel: "Create" });
    if (!name) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    persistSourceGroups([
      ...sourceGroups,
      { id: newId("sg"), name: trimmed, sourceKeys: [] },
    ]);
  }, [sourceGroups, persistSourceGroups, prompt]);
  const removeSourceGroup = useCallback(async (groupId: string) => {
    const group = sourceGroups.find((g) => g.id === groupId);
    if (!group) return;
    const ok = await confirm({
      title: "Delete source group",
      message: `Delete '${group.name}'? Sources will move to Ungrouped.`,
      okLabel: "Delete",
    });
    if (!ok) return;
    persistSourceGroups(sourceGroups.filter((g) => g.id !== groupId));
  }, [sourceGroups, persistSourceGroups, confirm]);
  const renameSourceGroupHandler = useCallback(async (groupId: string) => {
    const group = sourceGroups.find((g) => g.id === groupId);
    if (!group) return;
    const newName = await prompt({ title: "Rename source group", label: "Name", defaultValue: group.name, okLabel: "Rename" });
    if (!newName) return;
    persistSourceGroups(sourceGroups.map((g) => g.id === groupId ? { ...g, name: newName.trim() || g.name } : g));
  }, [sourceGroups, persistSourceGroups, prompt]);
  /** Assign an inset (by key) to a group. Removes it from any other
   *  group first so each source lives in at most one group. */
  const assignSourceToGroup = useCallback((groupId: string, sourceKey: string) => {
    persistSourceGroups(sourceGroups.map((g) => ({
      ...g,
      sourceKeys: g.id === groupId
        ? (g.sourceKeys.includes(sourceKey) ? g.sourceKeys : [...g.sourceKeys, sourceKey])
        : g.sourceKeys.filter((k) => k !== sourceKey),
    })));
  }, [sourceGroups, persistSourceGroups]);
  /** Pop an inset out of a group → it falls back to Ungrouped. */
  const removeSourceFromGroup = useCallback((groupId: string, sourceKey: string) => {
    persistSourceGroups(sourceGroups.map((g) =>
      g.id === groupId ? { ...g, sourceKeys: g.sourceKeys.filter((k) => k !== sourceKey) } : g,
    ));
  }, [sourceGroups, persistSourceGroups]);
  // Filename canonicalisation when sending an output to the figure
  // image timeline / collage — the analysis dialog used a similar
  // pattern keyed on (tabName, plotType, statTest, idx). Here we use
  // (workflow name, node label, output name) which is more meaningful
  // for the node-graph flow.
  // Refs into the drawer card DOM so we can scroll-into-view when
  // navigating from a node chip.
  const outputCardRefs = useRef<Map<string, HTMLDivElement | null>>(new Map());
  // Live snapshot of the aggregated outputs — kept in sync via a
  // useEffect below. Lets the selection-action callbacks read the
  // latest output payloads without having to be re-bound on every
  // node update.
  const allOutputsRef = useRef<AggregatedOutput[]>([]);

  // Figure-store + collage-store wiring for the "Add to main timeline"
  // / "Add to collage" actions on selected outputs.
  const uploadImages = useFigureStore((s) => s.uploadImages);
  const removeImage = useFigureStore((s) => s.removeImage);
  const addCollageItem = useCollageStore((s) => s.addItem);
  void removeImage;  // wired below in discard flow

  // Load inset sources + MATLAB availability on open.
  useEffect(() => {
    if (!open) return;
    api.listInsetAnalysisSources()
      .then((r) => {
        const list = r.sources || [];
        setInsetSources(list);
        // Do NOT auto-populate the Source node — the user picks
        // which insets to attach by dragging from the library
        // panel. We still refresh any sources that are already
        // attached so saved workflows pick up the latest thumbnail
        // / pixel data when the figure changed underneath them.
        setWorkflowTabs((tabs) => tabs.map((t) => ({
          ...t,
          nodes: t.nodes.map((n) => {
            if (n.data.kind !== "source") return n;
            const refreshed = (n.data.sources || []).map((s) => {
              const live = list.find((l) => l.key === s.key);
              return live ?? s;
            });
            return { ...n, data: { ...n.data, sources: refreshed } };
          }),
        })));
      })
      .catch(() => setInsetSources([]));
    api.checkMatlab().then((m) => setMatlabKind(m.kind || "")).catch(() => setMatlabKind(""));
    // Optional ImageJ / Fiji detection. The endpoint may not exist
    // on older sidecars — treat failure as "not installed" silently
    // rather than surfacing a 404.
    api.checkImageJ?.()
      .then((m: { installed?: boolean; kind?: string }) => setImagejKind(m?.installed ? (m.kind || "imagej") : ""))
      .catch(() => setImagejKind(""));
  }, [open]);

  // ── Graph handlers ─────────────────────────────────────────

  const onNodesChange = useCallback((changes: NodeChange<Node<NodeData>>[]) => {
    setNodes((nds) => applyNodeChanges(changes, nds));
  }, [setNodes]);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    setEdges((eds) => applyEdgeChanges(changes, eds));
  }, [setEdges]);

  // Derived data — computed via useMemo and exposed through the
  // GraphCallbacksContext so the node renderers can read counts /
  // declared outputs WITHOUT us having to write them back into
  // `node.data`.  The previous implementation used setNodes-from-
  // useEffect, which leaked into React Flow's subscriber fanout
  // and produced a "Maximum update depth exceeded" crash inside
  // @xyflow/react's store the moment the dialog mounted.

  /** edges → per-target { image: count, table: count }. */
  const inputCountsByNode = useMemo(() => {
    const m = new Map<string, { image: number; table: number }>();
    for (const e of edges) {
      const cur = m.get(e.target) || { image: 0, table: 0 };
      const kind: DataKind = (e.targetHandle || "").includes("table") ? "table" : "image";
      cur[kind] += 1;
      m.set(e.target, cur);
    }
    return m;
  }, [edges]);

  /** code → declared outputs (placeholder chips before run lands). */
  const declaredByNode = useMemo(() => {
    const m = new Map<string, DeclaredOutput[]>();
    for (const n of nodes) {
      if (n.data.kind === "source") continue;
      const decl = parseDeclaredOutputs(n.data.code || "");
      if (decl.length > 0) m.set(n.id, decl);
    }
    return m;
  }, [nodes]);

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
      ?? (engine === "python" ? PY_DEFAULT
        : engine === "matlab" ? ML_DEFAULT
        : engine === "imagej" ? IJ_DEFAULT
        : R_DEFAULT);
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
          label: engine === "python" ? "Python" : engine === "matlab" ? "MATLAB" : engine === "imagej" ? "ImageJ" : "R Plot",
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
    if (nodeId === "source") return;  // primary source is undeletable
    setNodes((cur) => cur.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  }, [selectedNodeId, setNodes, setEdges]);

  /** Delete whatever is currently selected in the graph — both nodes
   *  (except the primary source, which is undeletable) and edges. The
   *  primary source is a safe singleton: it's the one node ReactFlow
   *  shouldn't let users orphan, since the workflow always needs at
   *  least one source container to drop insets into. */
  const deleteSelected = useCallback(() => {
    const { nodeIds, edgeIds } = rfSelection;
    if (nodeIds.size === 0 && edgeIds.size === 0) return;
    const deletableNodeIds = new Set(Array.from(nodeIds).filter((id) => id !== "source"));
    setNodes((cur) => cur.filter((n) => !deletableNodeIds.has(n.id)));
    setEdges((eds) => eds.filter((e) =>
      !edgeIds.has(e.id) &&
      !deletableNodeIds.has(e.source) &&
      !deletableNodeIds.has(e.target),
    ));
    if (selectedNodeId && deletableNodeIds.has(selectedNodeId)) setSelectedNodeId(null);
    setRfSelection({ nodeIds: new Set(), edgeIds: new Set() });
  }, [rfSelection, selectedNodeId, setNodes, setEdges]);

  // ── Source library ↔ source-node plumbing ─────────────────────
  // The left-hand library panel lists every flagged inset. Users
  // drag a thumbnail onto a source node card, which fires
  // `addSourceToNode`. They can detach via the × on each row inside
  // the source node, which fires `removeSourceFromNode`. They can
  // also spawn extra empty source nodes via `addEmptySourceNode`
  // to partition insets across multiple parallel pipelines on the
  // same canvas.

  /** Attach an inset (looked up by InsetSource.key) to a source node. */
  const addSourceToNode = useCallback((nodeId: string, sourceKey: string) => {
    setNodes((cur) => {
      const inset = insetSources.find((s) => s.key === sourceKey);
      if (!inset) return cur;
      return cur.map((n) => {
        if (n.id !== nodeId || n.data.kind !== "source") return n;
        const existing = n.data.sources || [];
        // Avoid duplicating the same inset into the same source.
        if (existing.some((s) => s.key === sourceKey)) return n;
        return { ...n, data: { ...n.data, sources: [...existing, inset] } };
      });
    });
  }, [insetSources, setNodes]);

  /** Detach the inset at `idx` from a source node. Also removes
   *  any edges that were wired off the now-defunct output handle. */
  const removeSourceFromNode = useCallback((nodeId: string, idx: number) => {
    setNodes((cur) => cur.map((n) => {
      if (n.id !== nodeId || n.data.kind !== "source") return n;
      const next = (n.data.sources || []).filter((_, i) => i !== idx);
      return { ...n, data: { ...n.data, sources: next } };
    }));
    // Drop any edges whose source handle no longer exists. The
    // handle id is `out_image_<idx>`; after the splice all indices
    // shift, so the simplest safe move is to invalidate any edge
    // off this source node that points to a missing handle. We
    // accept some collateral here — users can re-wire from the
    // node card directly.
    setEdges((eds) => eds.filter((e) => {
      if (e.source !== nodeId) return true;
      const h = e.sourceHandle || "";
      if (!h.startsWith("out_image_")) return true;
      const i = parseInt(h.slice("out_image_".length), 10);
      return i < idx;  // edges before the removed slot stay valid
    }));
  }, [setNodes, setEdges]);

  /** Spawn an empty source node beside the primary one. */
  const addEmptySourceNode = useCallback(() => {
    const id = newId("src");
    // Pick a viewport-aware position next to existing source nodes.
    let pos = { x: 30, y: 220 };
    const inst = rfRef.current;
    if (inst) {
      try {
        const container = document.querySelector(".react-flow") as HTMLElement | null;
        if (container) {
          const rect = container.getBoundingClientRect();
          pos = inst.screenToFlowPosition({
            x: rect.left + 60 + Math.random() * 30,
            y: rect.top + rect.height * 0.55 + Math.random() * 40,
          });
        }
      } catch { /* fall back to default pos */ }
    }
    const sourceCount = nodes.filter((n) => n.data.kind === "source").length;
    setNodes((cur) => [
      ...cur,
      newSourceNode([], { id, label: `Source ${sourceCount + 1}`, position: pos }),
    ]);
    setSelectedNodeId(id);
  }, [nodes, setNodes]);

  // ── Output viewer plumbing ─────────────────────────────────────

  /** Open the drawer at the right tab and flash-highlight the card.
   *  Also paints a ring on every upstream node so users can see at
   *  a glance which sources fed this output. */
  const navigateToOutput = useCallback((nodeId: string, outputId: string, kind: DataKind) => {
    setDrawerOpen(true);
    setDrawerTab(kind);
    const key = `${nodeId}-${outputId}`;
    setHighlightOutputKey(key);
    // Walk backwards through incoming edges to find every upstream
    // node — inlined here (vs a separate useCallback) to keep the
    // declaration order TDZ-safe.
    const upstream = new Set<string>([nodeId]);
    const queue = [nodeId];
    const adj = new Map<string, string[]>();
    for (const e of edges) {
      const arr = adj.get(e.target) || [];
      arr.push(e.source);
      adj.set(e.target, arr);
    }
    while (queue.length) {
      const cur = queue.shift()!;
      for (const u of adj.get(cur) || []) {
        if (!upstream.has(u)) { upstream.add(u); queue.push(u); }
      }
    }
    setHighlightedUpstreamIds(upstream);
    setTimeout(() => {
      const el = outputCardRefs.current.get(key);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 30);
    setTimeout(() => {
      setHighlightOutputKey((cur) => (cur === key ? null : cur));
      setHighlightedUpstreamIds(new Set());
    }, 2200);
  }, [edges]);

  /** Open the full-size preview modal for a single output. */
  const openPreview = useCallback((nodeId: string, outputId: string) => {
    setPreviewOutput({ nodeId, outputId });
  }, []);

  // ── Detail-panel resizer ───────────────────────────────────
  // Track resize-drag with window-level pointer events so the
  // cursor doesn't escape the splitter strip when the user moves
  // fast.  Clamped to a sensible range so the user can't trap
  // themselves outside the panel.
  useEffect(() => {
    if (!resizingPanel) return;
    const onMove = (e: MouseEvent) => {
      // Compute width from the right edge of the viewport.
      const next = Math.min(900, Math.max(320, window.innerWidth - e.clientX));
      setDetailPanelWidth(next);
    };
    const onUp = () => setResizingPanel(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [resizingPanel]);


  // ── Layer-2 workflow tab management ─────────────────────────

  const addWorkflowTab = useCallback(() => {
    const id = newId("wf");
    const idx = workflowTabs.length + 1;
    setWorkflowTabs((tabs) => [...tabs, {
      id, name: `Workflow ${idx}`,
      nodes: [newSourceNode([])], edges: [],
    }]);
    setActiveWfId(id);
    setSelectedNodeId(null);
  }, [workflowTabs.length]);

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
  const saveCurrentWorkflow = useCallback(async () => {
    const name = await prompt({ title: "Save workflow", label: "Name", defaultValue: activeWf.name, okLabel: "Save" });
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
  }, [activeWf, prompt]);

  /** Load a saved workflow as a NEW tab so the user can run it
   *  against the current insets without overwriting their work.
   *  We refresh each attached inset's payload against the LIVE
   *  inset list (by key) so a saved workflow re-runs with current
   *  pixel data, but we don't add or remove sources — the user's
   *  saved selection wins. */
  const loadSavedWorkflow = useCallback((id: string) => {
    const wf = savedWorkflows.find((w) => w.id === id);
    if (!wf) return;
    const newWfId = newId("wf");
    const refreshedNodes = wf.nodes.map((n) => {
      if (n.data.kind !== "source") return n;
      const refreshed = (n.data.sources || []).map((s) => {
        const live = insetSources.find((l) => l.key === s.key);
        return live ?? s;
      });
      return { ...n, data: { ...n.data, sources: refreshed } };
    });
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
    if (engine !== "python" && engine !== "matlab" && engine !== "r" && engine !== "imagej") return;
    // Parse a `# @name: <label>` / `// @name: <label>` marker from the
    // first 20 lines so users can name a node from within the code.
    const codeBody = node.data.code || "";
    const nameMatch = codeBody.split("\n").slice(0, 20).join("\n").match(/(?:^|\n)\s*(?:#|\/\/|%)\s*@name:\s*([^\n]+)/);
    if (nameMatch) {
      const newLabel = nameMatch[1].trim().slice(0, 40);
      if (newLabel && newLabel !== node.data.label) {
        setNodes((cur) => cur.map((n) => n.id === node.id ? { ...n, data: { ...n.data, label: newLabel } } : n));
      }
    }
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
      } else if (engine === "imagej") {
        // Headless Fiji macro — same `sources` shape as MATLAB; the
        // sidecar materialises each inset to a tempfile and exposes
        // `inputs[i].path` / `inputs[i].label` to the macro.
        const sources = extra
          .filter((x) => x.kind === "image" && x.key.startsWith("inset_"))
          .map((x) => {
            const insetKey = x.key.replace(/^inset_\d+_/, "");
            const src = insetSources.find((s) => s.key === insetKey);
            if (!src) return null;
            return { key: insetKey, row: src.row, col: src.col, inset_index: src.inset_index, label: src.label };
          })
          .filter((s): s is { key: string; row: number; col: number; inset_index: number; label: string } => !!s);
        try {
          result = await api.runImageJ(node.data.code || IJ_DEFAULT, sources, 120);
        } catch (err) {
          // Soft-fail when the endpoint is missing on the sidecar:
          // surface the install hint inline rather than killing the
          // entire run.
          result = {
            success: false,
            stdout: "",
            stderr: err instanceof Error ? err.message
              : "ImageJ / Fiji not detected. Install Fiji and ensure `ImageJ-*` is on PATH.",
            plots: [], tables: [], images: [],
          };
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
      // Per-node consoleOut: the detail panel renders this directly
      // under the code editor so users see the FULL run output
      // (including tracebacks) next to the code that produced it,
      // not buried in a separate Console tab.
      const nodeOut = stdout.trim() ? stdout : "(no console output)";
      consoleRef.current += stdout.trim() ? stdout + "\n" : "(no console output)\n";
      setConsoleOut(consoleRef.current);
      setNodes((cur) => cur.map((n) => n.id === node.id ? {
        ...n,
        data: {
          ...n.data,
          outputs: newOutputs,
          status: result.success ? "ok" : "error",
          error: result.success ? undefined : (result.stderr || "Run failed").slice(0, 200),
          consoleOut: nodeOut,
        },
      } : n));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      consoleRef.current += `error: ${msg}\n`;
      setConsoleOut(consoleRef.current);
      setNodes((cur) => cur.map((n) => n.id === node.id ? {
        ...n,
        data: { ...n.data, status: "error", error: msg.slice(0, 200), consoleOut: `error: ${msg}` },
      } : n));
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
    allOutputsRef.current = allOutputs;
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

  // ── Drawer selection + batch actions ──────────────────────────

  const toggleOutputSelected = useCallback((key: string) => {
    setSelectedOutputKeys((cur) => {
      const next = new Set(cur);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  /** Build a stable, descriptive filename for one output — used for
   *  downloads, main-timeline uploads, and collage labels. */
  const outputFilename = useCallback((o: AggregatedOutput) => {
    const wfName = activeWf.name.replace(/\s+/g, "_");
    const node = o.nodeLabel.replace(/\s+/g, "_");
    return `analysis_${wfName}_${node}_${o.name}`;
  }, [activeWf.name]);

  // List of currently-checked outputs (filtered to the active tab
  // so the action buttons only act on what's visible).
  const selectedOutputsList = useCallback((kind?: DataKind): AggregatedOutput[] => {
    return allOutputsRef.current.filter((o) => {
      const key = `${o.nodeId}-${o.outputId}`;
      if (!selectedOutputKeys.has(key)) return false;
      if (kind && o.kind !== kind) return false;
      return true;
    });
  }, [selectedOutputKeys]);

  /** Download every selected output to the user's disk — PNG for
   *  images / plots, CSV for tables. */
  const downloadSelected = useCallback(() => {
    const list = selectedOutputsList();
    list.forEach((o) => {
      const fname = outputFilename(o);
      const link = document.createElement("a");
      if (o.kind === "table") {
        link.href = `data:text/csv;base64,${btoa(unescape(encodeURIComponent(o.payload)))}`;
        link.download = `${fname}.csv`;
      } else {
        link.href = `data:image/png;base64,${o.payload}`;
        link.download = `${fname}.png`;
      }
      link.click();
    });
  }, [selectedOutputsList, outputFilename]);

  // The select-then-bulk-send buttons have been replaced by the
  // drawer's right-hand staging columns (drag a card into "Main
  // timeline" or "Collage", click Send).  Keeping `selectedOutputsList`
  // (used by `downloadSelected`) and removing the timeline / collage
  // helpers — they were dead code after the toolbar trim.

  // ── Console / snippet runner ──────────────────────────────
  // Fires off an ad-hoc snippet against the same Python / R /
  // MATLAB / ImageJ harnesses that nodes use. No persistent state
  // between runs (every call is a fresh process) — that's a
  // future enhancement; for now this is a "scratch pad" for
  // quick syntax checks and one-shot calculations.
  const runConsoleSnippet = useCallback(async () => {
    if (consoleSnippetRunning) return;
    const snippet = consoleSnippet.trim();
    if (!snippet) return;
    setConsoleSnippetRunning(true);
    consoleRef.current += `\n=== Console [${consoleEngine}] ===\n`;
    setConsoleOut(consoleRef.current);
    try {
      let res: { success: boolean; stdout: string; stderr: string };
      if (consoleEngine === "python") {
        res = await api.runPython(snippet, [], 60);
      } else if (consoleEngine === "matlab") {
        res = await api.runMatlab(snippet, [], 60);
      } else if (consoleEngine === "imagej") {
        try { res = await api.runImageJ(snippet, [], 60); }
        catch (e) { res = { success: false, stdout: "", stderr: e instanceof Error ? e.message : "ImageJ not available" }; }
      } else {
        res = await api.runR(snippet, "", undefined);
      }
      const merged = (res.stdout || "") + (res.stderr ? `\n${res.stderr}` : "");
      consoleRef.current += merged + (merged.endsWith("\n") ? "" : "\n");
      setConsoleOut(consoleRef.current);
    } catch (err) {
      consoleRef.current += `error: ${err instanceof Error ? err.message : String(err)}\n`;
      setConsoleOut(consoleRef.current);
    } finally {
      setConsoleSnippetRunning(false);
    }
  }, [consoleEngine, consoleSnippet, consoleSnippetRunning]);

  // ── Staging area: drop-to-queue + commit ──────────────────
  // The right column of the drawer hosts two drop zones — "Main
  // timeline" and "Collage". Drag-and-drop adds the output to the
  // respective staged Set; the destination button under each zone
  // commits the whole batch and clears the zone.
  const toggleStaged = useCallback((bucket: "main" | "collage", key: string) => {
    const setter = bucket === "main" ? setStagedForMain : setStagedForCollage;
    setter((cur) => {
      const next = new Set(cur);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);
  const commitStaged = useCallback(async (bucket: "main" | "collage") => {
    const keys = bucket === "main" ? stagedForMain : stagedForCollage;
    const list = allOutputsRef.current.filter(
      (o) => keys.has(`${o.nodeId}-${o.outputId}`) && (o.kind === "plot" || o.kind === "image")
    );
    if (list.length === 0) return;
    if (bucket === "main") {
      const stamp = Date.now();
      const files = list.map((o) => b64ToFile(o.payload, `${outputFilename(o)}_${stamp}.png`));
      try {
        const names = await uploadImages(files);
        consoleRef.current += `\n[${names.length} output(s) committed to main image timeline.]\n`;
      } catch (err) {
        consoleRef.current += `\n[Main-timeline commit failed: ${err instanceof Error ? err.message : String(err)}]\n`;
      }
      setStagedForMain(new Set());
    } else {
      let placed = 0;
      for (const o of list) {
        const { w, h } = await imageDims(o.payload);
        const scale = Math.min(1, 700 / Math.max(w, h));
        addCollageItem({
          kind: "image",
          fromAnalysis: true,
          src: `data:image/png;base64,${o.payload}`,
          name: outputFilename(o),
          x: 60 + placed * 40,
          y: 60 + placed * 40,
          w: Math.round(w * scale),
          h: Math.round(h * scale),
          naturalW: w,
          naturalH: h,
        });
        placed++;
      }
      consoleRef.current += `\n[${placed} output(s) committed to Collage Assembly.]\n`;
      setStagedForCollage(new Set());
    }
    setConsoleOut(consoleRef.current);
  }, [stagedForMain, stagedForCollage, outputFilename, uploadImages, addCollageItem]);

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
          onClick={async () => {
            const newName = await prompt({ title: "Rename workflow", label: "Name", defaultValue: activeWf.name, okLabel: "Rename" });
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
                onClick={async (e) => {
                  e.stopPropagation();
                  const ok = await confirm({ title: "Delete saved workflow", message: `Delete saved workflow '${wf.name}'?`, okLabel: "Delete" });
                  if (ok) deleteSavedWorkflow(wf.id);
                }}
                sx={{ fontSize: "0.85rem", color: "text.disabled", ml: 1, "&:hover": { color: "error.main" } }}
              >🗑</Box>
            </MenuItem>
          ))}
        </Select>
      </Box>

      {/* Canvas + side panel + drawer (flex row below the tabs) */}
      <GraphCallbacksContext.Provider value={{
        addSourceToNode, removeSourceFromNode, navigateToOutput, openPreview,
        insetSources, hasMeasurements, highlightedUpstreamIds,
        inputCountsByNode, declaredByNode,
      }}>
      <Box sx={{ flex: 1, display: "flex", minHeight: 0, position: "relative" }}>
      {/* Far left: source library panel.  Lists every flagged inset
          as a draggable thumbnail; users drag entries onto a source
          node on the canvas to attach. "+ Source" spawns an empty
          source container so users can partition insets across
          parallel pipelines on the same canvas. */}
      <Box sx={{ width: 168, flexShrink: 0, borderRight: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", bgcolor: "background.paper" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.5, borderBottom: "1px solid", borderColor: "divider" }}>
          <Typography variant="caption" sx={{ fontSize: "0.6rem", fontWeight: 700, color: "text.secondary", letterSpacing: 0.5, textTransform: "uppercase", flex: 1 }}>
            Sources ({insetSources.length})
          </Typography>
          <Tooltip placement="right" title="Create a custom group — drag insets onto it to organise them">
            <IconButton size="small" onClick={addSourceGroupHandler} sx={{ p: 0.25 }}>
              <span style={{ fontSize: 12, lineHeight: 1 }}>📁+</span>
            </IconButton>
          </Tooltip>
          <Tooltip placement="right" title="Add an empty source node to the canvas">
            <IconButton size="small" onClick={addEmptySourceNode} sx={{ p: 0.25 }}>
              <AddIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
        </Box>
        <Box sx={{ flex: 1, overflow: "auto", p: 0.5 }}>
          {insetSources.length === 0 ? (
            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.disabled", fontStyle: "italic", display: "block", px: 0.5, py: 1, lineHeight: 1.35 }}>
              No flagged insets yet. Open <strong>Edit Panel → Zoom Inset</strong> and tick <em>Include in Analysis</em>.
            </Typography>
          ) : (() => {
            // User-defined groups (each drag-drop target) followed by
            // an Ungrouped section for sources not assigned anywhere.
            const grouped = new Set<string>();
            for (const g of sourceGroups) for (const k of g.sourceKeys) grouped.add(k);
            const ungrouped = insetSources.filter((s) => !grouped.has(s.key));
            return (
              <>
                {sourceGroups.map((g) => (
                  <UserSourceGroup
                    key={g.id}
                    group={g}
                    items={g.sourceKeys
                      .map((k) => insetSources.find((s) => s.key === k))
                      .filter((s): s is InsetSource => !!s)}
                    onAssignKey={(key) => assignSourceToGroup(g.id, key)}
                    onRemoveKey={(key) => removeSourceFromGroup(g.id, key)}
                    onRename={() => renameSourceGroupHandler(g.id)}
                    onDelete={() => removeSourceGroup(g.id)}
                  />
                ))}
                {/* "Ungrouped" — anything not assigned to a group. */}
                {(ungrouped.length > 0 || sourceGroups.length === 0) && (
                  <UserSourceGroup
                    key="__ungrouped__"
                    group={{ id: "__ungrouped__", name: sourceGroups.length === 0 ? "All sources" : "Ungrouped", sourceKeys: [] }}
                    items={ungrouped}
                    onAssignKey={() => {/* drops on ungrouped are a no-op */}}
                    onRemoveKey={() => {/* not in a group → nothing to remove */}}
                    isUngrouped
                  />
                )}
              </>
            );
          })()}
          {/* Measurements aren't a draggable library item — they appear
              automatically as the 📋 port on every source node when
              the host figure has measurement data, so wire them by
              edge directly. The hint below tells the user. */}
          {hasMeasurements && (
            <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", display: "block", mt: 1, px: 0.5, py: 0.5, borderTop: "1px dashed", borderColor: "divider", lineHeight: 1.3, fontStyle: "italic" }}>
              📋 Measurements port is available on every source node — wire it to an R node downstream.
            </Typography>
          )}
        </Box>
      </Box>

      {/* Middle: graph canvas */}
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
          onSelectionChange={({ nodes: sn, edges: se }) => {
            // BAIL OUT when nothing actually changed.  React Flow fires
            // onSelectionChange on every internal store update — if
            // we blindly setState with a new Set each call, the parent
            // re-renders, RF re-syncs, fires onSelectionChange again,
            // and we hit "Maximum update depth exceeded" inside RF's
            // subscriber fanout.  The compare-then-skip pattern stops
            // the loop cold.
            setRfSelection((prev) => {
              const newNodeIds = sn.map((n) => n.id);
              const newEdgeIds = se.map((e) => e.id);
              const sameN = prev.nodeIds.size === newNodeIds.length && newNodeIds.every((nid) => prev.nodeIds.has(nid));
              const sameE = prev.edgeIds.size === newEdgeIds.length && newEdgeIds.every((eid) => prev.edgeIds.has(eid));
              if (sameN && sameE) return prev;
              return { nodeIds: new Set(newNodeIds), edgeIds: new Set(newEdgeIds) };
            });
          }}
          onInit={(inst) => { rfRef.current = inst as unknown as RFInstance; }}
          fitView
          fitViewOptions={{ maxZoom: 0.85, minZoom: 0.4, padding: 0.25 }}
          minZoom={0.15}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 0.7 }}
          deleteKeyCode={["Backspace", "Delete"]}
          edgesFocusable
          elementsSelectable
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
          <Tooltip placement="bottom" title={imagejKind
            ? `Add an ImageJ / Fiji macro node (detected: ${imagejKind})`
            : "ImageJ / Fiji not detected — install Fiji and ensure ImageJ-* is on PATH"}>
            <span>
              <Button size="small" variant="contained" startIcon={<AddIcon sx={{ fontSize: 14 }} />}
                disabled={!imagejKind}
                onClick={() => addProcessNode("imagej")}
                sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25, bgcolor: KIND_COLOR.imagej, "&:hover": { bgcolor: KIND_COLOR.imagej, filter: "brightness(0.9)" } }}>
                🔬 ImageJ
              </Button>
            </span>
          </Tooltip>
          <Box sx={{ flex: 1 }} />
          {/* Delete selected — operates on whatever React Flow has
              selected (click a node or an edge first; Shift+click for
              multi-select). Tooltip surfaces the Backspace shortcut
              for keyboard-driven users. */}
          {(rfSelection.nodeIds.size + rfSelection.edgeIds.size) > 0 && (
            <Tooltip placement="bottom" title="Delete selected node(s) and edge(s) — Backspace / Delete also works">
              <Button size="small" variant="outlined" color="error"
                startIcon={<DeleteOutlineIcon sx={{ fontSize: 14 }} />}
                onClick={deleteSelected}
                sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}>
                Delete ({rfSelection.nodeIds.size + rfSelection.edgeIds.size})
              </Button>
            </Tooltip>
          )}
          <Button size="small" variant="contained" color="primary"
            startIcon={runningGraph ? <CircularProgress size={12} /> : <PlayArrowIcon sx={{ fontSize: 14 }} />}
            disabled={runningGraph}
            onClick={runGraph}
            sx={{ fontSize: "0.7rem", textTransform: "none", py: 0.25, ml: 1 }}>
            {runningGraph ? "Running…" : "Run graph"}
          </Button>
        </Box>
      </Box>

      {/* Right: selected-node detail panel — source nodes get a
          rename + attached-source list; process nodes get the
          code editor.  Width is user-resizable via the drag strip
          on the left edge. */}
      {selectedNode && selectedNode.data.kind === "source" && (
        <Box sx={{ width: detailPanelWidth, position: "relative", borderLeft: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", minWidth: 0, bgcolor: "background.paper" }}>
          {/* Resizer strip */}
          <Box
            onMouseDown={(e) => { e.preventDefault(); setResizingPanel(true); }}
            sx={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 6, cursor: "col-resize", zIndex: 5, "&:hover": { bgcolor: "primary.main", opacity: 0.4 } }}
          />
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 1, borderBottom: "1px solid", borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.75rem", flexShrink: 0 }}>
              {KIND_ICON.source}
            </Typography>
            <TextField
              size="small"
              value={selectedNode.data.label}
              onChange={(e) => {
                const v = e.target.value.slice(0, 40);
                setNodes((cur) => cur.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, label: v } } : n));
              }}
              sx={{ flex: 1, "& .MuiInputBase-input": { fontSize: "0.75rem", py: 0.4 } }}
            />
            {selectedNode.id !== "source" && (
              <IconButton size="small" onClick={() => removeNode(selectedNode.id)} title="Remove source node">
                <DeleteOutlineIcon sx={{ fontSize: 14 }} />
              </IconButton>
            )}
          </Box>
          <Box sx={{ flex: 1, overflow: "auto", p: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 700, color: "text.secondary", letterSpacing: 0.5, textTransform: "uppercase", display: "block", mb: 0.5 }}>
              Attached insets ({(selectedNode.data.sources || []).length})
            </Typography>
            {(selectedNode.data.sources || []).length === 0 ? (
              <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.disabled", fontStyle: "italic" }}>
                Drag insets here from the source library on the left.
              </Typography>
            ) : (
              (selectedNode.data.sources || []).map((s, idx) => (
                <Box key={`${s.key}_${idx}`} sx={{ display: "flex", alignItems: "center", gap: 0.5, py: 0.4, borderBottom: "1px dashed", borderColor: "divider" }}>
                  {s.thumbnail && (
                    <Box component="img" src={`data:image/png;base64,${s.thumbnail}`} alt={s.label}
                      sx={{ width: 36, height: 36, objectFit: "contain", border: "1px solid", borderColor: "divider", borderRadius: 0.25 }}
                    />
                  )}
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 600, display: "block" }}>
                      R{s.row + 1}C{s.col + 1} · inset {s.inset_index + 1}
                    </Typography>
                    <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>
                      {s.label} · {s.natural_width}×{s.natural_height}
                    </Typography>
                  </Box>
                  <IconButton size="small" onClick={() => removeSourceFromNode(selectedNode.id, idx)} title="Detach this inset">
                    <DeleteOutlineIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </Box>
              ))
            )}
            {hasMeasurements && (
              <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", display: "block", mt: 1, fontStyle: "italic" }}>
                The 📋 measurements port (on the node card) is available on every source — wire it to an R node to plot panel measurements.
              </Typography>
            )}
          </Box>
        </Box>
      )}

      {selectedNode && selectedNode.data.kind !== "source" && (() => {
        const engine = selectedNode.data.kind as EngineKind;
        const builtin = BUILTIN_PRESETS[engine] || [];
        const userPresets = userPresetsByEngine[engine] || [];
        return (
        <Box sx={{ width: detailPanelWidth, position: "relative", borderLeft: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", minWidth: 0, bgcolor: "background.paper" }}>
          {/* Resizer strip */}
          <Box
            onMouseDown={(e) => { e.preventDefault(); setResizingPanel(true); }}
            sx={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 6, cursor: "col-resize", zIndex: 5, "&:hover": { bgcolor: "primary.main", opacity: 0.4 } }}
          />
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 1, borderBottom: "1px solid", borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.85rem", flexShrink: 0 }}>
              {KIND_ICON[engine]}
            </Typography>
            <Tooltip placement="top" title="Rename this node (you can also set the name from code via `# @name: …`)">
              <TextField
                size="small"
                value={selectedNode.data.label}
                onChange={(e) => {
                  const v = e.target.value.slice(0, 40);
                  setNodes((cur) => cur.map((n) => n.id === selectedNode.id ? { ...n, data: { ...n.data, label: v } } : n));
                }}
                sx={{ flex: 1, "& .MuiInputBase-input": { fontSize: "0.75rem", py: 0.4 } }}
              />
            </Tooltip>
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
            <Tooltip placement="top" title="Save the current code as a named preset (persists across sessions)">
              <Button size="small" variant="outlined"
                onClick={async () => {
                  const name = await prompt({
                    title: "Save code as preset",
                    label: `Preset name (${engine})`,
                    defaultValue: "",
                    okLabel: "Save",
                  });
                  if (!name) return;
                  const trimmed = name.trim();
                  if (!trimmed) return;
                  const code = selectedNode.data.code || "";
                  const next = [...userPresets.filter((p) => p.name !== trimmed), { name: trimmed, code }];
                  persistUserPresets(engine, next);
                }}
                sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25, minWidth: 0 }}
              >
                💾 Save as preset
              </Button>
            </Tooltip>
            {userPresets.length > 0 && (
              <Tooltip title="Delete a saved preset">
                <Select
                  size="small"
                  value=""
                  displayEmpty
                  renderValue={() => "🗑"}
                  onChange={async (e) => {
                    const name = String(e.target.value);
                    if (!name) return;
                    const ok = await confirm({ title: "Delete preset", message: `Delete preset '${name}'?`, okLabel: "Delete" });
                    if (!ok) return;
                    persistUserPresets(engine, userPresets.filter((p) => p.name !== name));
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
            {/* Per-node console — collapsible, sits between the code
                editor and the outputs preview so users can read the
                FULL stdout / stderr / traceback right next to the
                code that produced it.  Auto-opens on error. */}
            <NodeConsolePanel
              key={`console-${selectedNode.id}`}
              nodeId={selectedNode.id}
              text={selectedNode.data.consoleOut || ""}
              status={selectedNode.data.status}
            />
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

      {/* Bottom drawer: aggregated outputs.  Each card has a checkbox
          (drives the batch toolbar) and clicking the thumbnail opens
          the large-preview modal. */}
      <Drawer
        variant="persistent"
        anchor="bottom"
        open={drawerOpen}
        PaperProps={{ sx: { position: "absolute", height: "42%", minHeight: 240, display: "flex", flexDirection: "column" } }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.25, borderBottom: "1px solid", borderColor: "divider", flexWrap: "wrap" }}>
          <Tabs value={drawerTab} onChange={(_, v) => setDrawerTab(v)} sx={{ minHeight: 28, "& .MuiTab-root": { minHeight: 28, fontSize: "0.65rem", py: 0, textTransform: "none" } }}>
            <Tab value="plot" label={`📊 Plots (${allOutputs.filter((o) => o.kind === "plot").length})`} />
            <Tab value="table" label={`📋 Tables (${allOutputs.filter((o) => o.kind === "table").length})`} />
            <Tab value="image" label={`🖼 Images (${allOutputs.filter((o) => o.kind === "image").length})`} />
            <Tab value="console" label="💻 Console" />
          </Tabs>
          <Box sx={{ flex: 1 }} />
          {/* Batch action — download is the only one that doesn't have
              an equivalent staging zone, so it stays in the toolbar.
              Timeline / collage are now driven by the drop zones to
              the right of the output grid. */}
          <Tooltip placement="top" title="Download checked outputs to disk (PNG / CSV)">
            <span>
              <Button size="small" variant="outlined"
                disabled={selectedOutputKeys.size === 0}
                onClick={downloadSelected}
                startIcon={<DownloadIcon sx={{ fontSize: 14 }} />}
                sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25 }}>
                💾 Save checked ({selectedOutputKeys.size})
              </Button>
            </span>
          </Tooltip>
          {selectedOutputKeys.size > 0 && (
            <Button size="small" onClick={() => setSelectedOutputKeys(new Set())}
              sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.25, color: "text.secondary" }}>
              Clear
            </Button>
          )}
          <Button size="small" onClick={() => setDrawerOpen(false)} sx={{ fontSize: "0.55rem", textTransform: "none" }}>
            Hide
          </Button>
        </Box>
        {/* Console snippet runner — engine picker + free-text
            CodeMirror buffer + Run.  Output appended to the console
            tail at the bottom of the canvas. */}
        {drawerTab === "console" && (
          <Box sx={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 1, py: 0.5, borderBottom: "1px solid", borderColor: "divider" }}>
              <Typography variant="caption" sx={{ fontSize: "0.65rem", fontWeight: 600 }}>Engine:</Typography>
              <Select size="small" value={consoleEngine} onChange={(e) => setConsoleEngine(e.target.value as EngineKind)}
                sx={{ fontSize: "0.7rem", height: 26, "& .MuiSelect-select": { py: 0.25, px: 1 } }}>
                <MenuItem value="python" sx={{ fontSize: "0.7rem" }}>🐍 Python</MenuItem>
                <MenuItem value="r" sx={{ fontSize: "0.7rem" }}>📊 R</MenuItem>
                <MenuItem value="matlab" disabled={!matlabKind} sx={{ fontSize: "0.7rem" }}>📐 MATLAB {matlabKind ? "" : "(unavailable)"}</MenuItem>
                <MenuItem value="imagej" disabled={!imagejKind} sx={{ fontSize: "0.7rem" }}>🔬 ImageJ {imagejKind ? "" : "(unavailable)"}</MenuItem>
              </Select>
              <Box sx={{ flex: 1 }} />
              <Button size="small" variant="contained" color="primary"
                onClick={runConsoleSnippet}
                disabled={consoleSnippetRunning || !consoleSnippet.trim()}
                startIcon={consoleSnippetRunning ? <CircularProgress size={12} /> : <PlayArrowIcon sx={{ fontSize: 14 }} />}
                sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}>
                {consoleSnippetRunning ? "Running…" : "Run snippet"}
              </Button>
              <Button size="small" onClick={() => { consoleRef.current = ""; setConsoleOut(""); }}
                sx={{ fontSize: "0.6rem", textTransform: "none", py: 0.25 }}>
                Clear output
              </Button>
            </Box>
            <Box sx={{ flex: 1, display: "flex", overflow: "hidden" }}>
              <Box sx={{ flex: 1, borderRight: "1px solid", borderColor: "divider", "& .cm-editor": { height: "100%" }, "& .cm-scroller": { fontFamily: "monospace" } }}>
                <CodeMirror
                  value={consoleSnippet}
                  onChange={(v) => setConsoleSnippet(v)}
                  theme={oneDark}
                  height="100%"
                  extensions={consoleEngine === "r" ? [StreamLanguage.define(cmR)] : [cmPython()]}
                  basicSetup={{ lineNumbers: true, highlightActiveLine: true, foldGutter: false, bracketMatching: true, closeBrackets: true }}
                />
              </Box>
              <Box sx={{ width: "45%", overflow: "auto", p: 0.75, fontFamily: "monospace", fontSize: "0.7rem", whiteSpace: "pre-wrap", bgcolor: "background.default" }}>
                {consoleOut || <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.disabled", fontStyle: "italic" }}>Snippet output will appear here.  Each Run is a fresh process — no state persists between calls.</Typography>}
              </Box>
            </Box>
          </Box>
        )}

        {/* Output grid + staging column (default view) */}
        {drawerTab !== "console" && (
          <>
            {/* Select-all helper for the currently visible tab */}
            {allOutputs.filter((o) => o.kind === drawerTab).length > 0 && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.1, borderBottom: "1px solid", borderColor: "divider", bgcolor: "action.hover" }}>
                <Checkbox
                  size="small"
                  checked={(() => {
                    const visible = allOutputs.filter((o) => o.kind === drawerTab);
                    return visible.length > 0 && visible.every((o) => selectedOutputKeys.has(`${o.nodeId}-${o.outputId}`));
                  })()}
                  indeterminate={(() => {
                    const visible = allOutputs.filter((o) => o.kind === drawerTab);
                    const sel = visible.filter((o) => selectedOutputKeys.has(`${o.nodeId}-${o.outputId}`)).length;
                    return sel > 0 && sel < visible.length;
                  })()}
                  onChange={(e) => {
                    const visible = allOutputs.filter((o) => o.kind === drawerTab);
                    setSelectedOutputKeys((cur) => {
                      const next = new Set(cur);
                      if (e.target.checked) visible.forEach((o) => next.add(`${o.nodeId}-${o.outputId}`));
                      else visible.forEach((o) => next.delete(`${o.nodeId}-${o.outputId}`));
                      return next;
                    });
                  }}
                  sx={{ p: 0.25 }}
                />
                <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>
                  Select all {drawerTab}s
                </Typography>
                <Box sx={{ flex: 1 }} />
                <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.disabled", fontStyle: "italic" }}>
                  Drag a card → into the staging zones on the right to queue for main timeline / collage.
                </Typography>
              </Box>
            )}
            <Box sx={{ flex: 1, display: "flex", overflow: "hidden" }}>
              {/* Left: output cards */}
              <Box sx={{ flex: 1, overflow: "auto", p: 1 }}>
                {allOutputs.filter((o) => o.kind === drawerTab).length === 0 ? (
                  <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.disabled", fontStyle: "italic" }}>
                    No {drawerTab} outputs yet. Build a graph and click Run.
                  </Typography>
                ) : (
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                    {allOutputs.filter((o) => o.kind === drawerTab).map((o) => {
                      const key = `${o.nodeId}-${o.outputId}`;
                      const isSel = selectedOutputKeys.has(key);
                      const isHi = highlightOutputKey === key;
                      const inMain = stagedForMain.has(key);
                      const inCollage = stagedForCollage.has(key);
                      const isStaged = inMain || inCollage;
                      return (
                        <Box
                          key={key}
                          ref={(el: HTMLDivElement | null) => { outputCardRefs.current.set(key, el); }}
                          draggable={o.kind !== "table"}
                          onDragStart={(e) => {
                            e.dataTransfer.setData("application/x-mpfig-output", key);
                            e.dataTransfer.effectAllowed = "copy";
                          }}
                          sx={{
                            border: "2px solid",
                            borderColor: isHi ? "warning.main" : isSel ? "primary.main" : o.pinned ? "warning.light" : "divider",
                            borderRadius: 0.5, p: 0.5, width: 150,
                            display: "flex", flexDirection: "column", gap: 0.25,
                            position: "relative",
                            boxShadow: isHi ? "0 0 0 4px rgba(255,179,0,0.35)" : "none",
                            transition: "box-shadow 240ms, border-color 240ms, opacity 200ms",
                            opacity: isStaged ? 0.45 : 1,
                            filter: isStaged ? "grayscale(0.7)" : "none",
                            cursor: o.kind !== "table" ? "grab" : "default",
                            "&:active": { cursor: o.kind !== "table" ? "grabbing" : "default" },
                          }}>
                          <Checkbox
                            size="small"
                            checked={isSel}
                            onChange={() => toggleOutputSelected(key)}
                            sx={{ position: "absolute", top: 0, left: 0, p: 0.25 }}
                          />
                          <IconButton size="small" onClick={() => togglePin(o.nodeId, o.outputId)} sx={{ position: "absolute", top: 2, right: 2, p: 0 }}>
                            {o.pinned ? <StarIcon sx={{ fontSize: 14, color: "warning.main" }} /> : <StarBorderIcon sx={{ fontSize: 14 }} />}
                          </IconButton>
                          <Typography variant="caption" sx={{ fontSize: "0.55rem", fontWeight: 600, color: "text.secondary", pl: 2.5, lineHeight: 1.2 }}>
                            {o.nodeLabel}
                          </Typography>
                          <Typography variant="caption" sx={{ fontSize: "0.65rem", fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", pr: 2 }}>
                            {o.name}
                          </Typography>
                          {(o.kind === "plot" || o.kind === "image") && (
                            <Tooltip placement="top" title="Click to preview at full size">
                              <Box
                                onClick={() => openPreview(o.nodeId, o.outputId)}
                                component="div"
                                sx={{ position: "relative", cursor: "zoom-in", "&:hover .preview-icon": { opacity: 1 } }}
                              >
                                <Box component="img" src={`data:image/png;base64,${o.payload}`} alt={o.name}
                                  sx={{ display: "block", width: "100%", height: 80, objectFit: "contain", bgcolor: "background.default" }} />
                                <Box className="preview-icon" sx={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", opacity: 0, transition: "opacity 120ms", bgcolor: "rgba(0,0,0,0.35)", color: "white" }}>
                                  <OpenInFullIcon sx={{ fontSize: 22 }} />
                                </Box>
                              </Box>
                            </Tooltip>
                          )}
                          {o.kind === "table" && (
                            <Tooltip placement="top" title="Click for a full-size table preview">
                              <Typography variant="caption"
                                onClick={() => openPreview(o.nodeId, o.outputId)}
                                sx={{ fontSize: "0.6rem", color: "primary.main", cursor: "pointer", textDecoration: "underline" }}>
                                {(o.payload.split("\n").length - 1)} rows — preview
                              </Typography>
                            </Tooltip>
                          )}
                          {isStaged && (
                            <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "warning.main", fontWeight: 700 }}>
                              staged{inMain && inCollage ? " (main+collage)" : inMain ? " (main)" : " (collage)"}
                            </Typography>
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
                      );
                    })}
                  </Box>
                )}
              </Box>
              {/* Right: staging column with two drop zones */}
              <Box sx={{ width: 280, flexShrink: 0, borderLeft: "1px solid", borderColor: "divider", display: "flex", flexDirection: "column", bgcolor: "background.paper" }}>
                <StagingZone
                  title="📤 Main timeline"
                  hint="Drop plots / images here to queue for the main figure's image timeline."
                  bucket="main"
                  staged={stagedForMain}
                  allOutputs={allOutputs}
                  onAdd={(k) => toggleStaged("main", k)}
                  onRemove={(k) => toggleStaged("main", k)}
                  onCommit={() => commitStaged("main")}
                />
                <StagingZone
                  title="🖼 Collage"
                  hint="Drop plots / images here to queue for the Collage Assembly canvas."
                  bucket="collage"
                  staged={stagedForCollage}
                  allOutputs={allOutputs}
                  onAdd={(k) => toggleStaged("collage", k)}
                  onRemove={(k) => toggleStaged("collage", k)}
                  onCommit={() => commitStaged("collage")}
                />
              </Box>
            </Box>
          </>
        )}
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
      </GraphCallbacksContext.Provider>

      {/* Large-preview modal — opens via the chip / thumbnail click.
          Shows the full-resolution image/plot, or a scrollable CSV
          for tables. */}
      <Dialog
        open={previewOutput !== null}
        onClose={() => setPreviewOutput(null)}
        maxWidth="lg"
        fullWidth
      >
        {(() => {
          if (!previewOutput) return null;
          const o = allOutputs.find((x) => x.nodeId === previewOutput.nodeId && x.outputId === previewOutput.outputId);
          if (!o) return null;
          return (
            <>
              <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2 }}>
                <Box sx={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                  <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>
                    {o.nodeLabel} · {o.kind}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 700, fontSize: "0.85rem" }}>
                    {o.name}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
                  <Button size="small" component="a"
                    href={o.kind === "table"
                      ? `data:text/csv;base64,${btoa(unescape(encodeURIComponent(o.payload)))}`
                      : `data:image/png;base64,${o.payload}`}
                    download={`${outputFilename(o)}.${o.kind === "table" ? "csv" : "png"}`}
                    startIcon={<DownloadIcon sx={{ fontSize: 14 }} />}
                    sx={{ fontSize: "0.65rem", textTransform: "none" }}>
                    Download
                  </Button>
                  <IconButton size="small" onClick={() => setPreviewOutput(null)}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
              </DialogTitle>
              <DialogContent sx={{ p: 1, bgcolor: "background.default", minHeight: "70vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {(o.kind === "plot" || o.kind === "image") ? (
                  // Fill the dialog: width 100%, max-height 85vh,
                  // imageRendering pixelated so a small (e.g. 200px)
                  // raster doesn't blur when upscaled to fit.
                  <Box component="img" src={`data:image/png;base64,${o.payload}`} alt={o.name}
                    sx={{
                      display: "block",
                      width: "100%",
                      maxHeight: "85vh",
                      objectFit: "contain",
                      imageRendering: "auto",
                    }} />
                ) : (
                  // Pretty-print CSV as a real table so the preview
                  // doesn't degenerate into a single <pre> column.
                  <Box sx={{ width: "100%", maxHeight: "85vh", overflow: "auto", bgcolor: "background.paper", p: 1, border: "1px solid", borderColor: "divider", borderRadius: 0.5 }}>
                    {(() => {
                      const rows = o.payload.split("\n").filter((r) => r.length > 0).map((r) => r.split(","));
                      if (rows.length === 0) return <Typography variant="caption">(empty)</Typography>;
                      const header = rows[0];
                      const body = rows.slice(1);
                      return (
                        <Box component="table" sx={{ borderCollapse: "collapse", fontSize: "0.7rem", fontFamily: "monospace", width: "100%" }}>
                          <thead>
                            <tr>
                              {header.map((h, i) => (
                                <Box component="th" key={i} sx={{ borderBottom: "2px solid", borderColor: "divider", textAlign: "left", py: 0.4, px: 0.75, position: "sticky", top: 0, bgcolor: "background.paper" }}>{h}</Box>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {body.map((r, ri) => (
                              <tr key={ri}>
                                {r.map((c, ci) => (
                                  <Box component="td" key={ci} sx={{ borderBottom: "1px dashed", borderColor: "divider", py: 0.3, px: 0.75, whiteSpace: "nowrap" }}>{c}</Box>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </Box>
                      );
                    })()}
                  </Box>
                )}
              </DialogContent>
            </>
          );
        })()}
      </Dialog>

      {/* Tauri-friendly prompt + confirm dialogs */}
      <PromptDialogBody request={promptRequest} setRequest={setPromptRequest} />
      <ConfirmDialogBody request={confirmRequest} setRequest={setConfirmRequest} />
    </Box>
  );
}
