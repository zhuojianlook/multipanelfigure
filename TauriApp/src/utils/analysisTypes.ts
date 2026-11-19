/* ──────────────────────────────────────────────────────────
   analysisTypes — the Analysis graph's data model.
   Types only (every import below is type-only, so this module
   has no runtime). Shared by the graph component, the analysis
   workspace store and the pure persistence helpers
   (workflowOwnership / workflowPersist), none of which may pull
   in the 9.7k-line component. Moved from AnalysisNodeGraph,
   which re-exports every name.
   ────────────────────────────────────────────────────────── */

import type { Node, Edge } from "@xyflow/react";
import type { BandPickerConfig } from "../components/dialogs/BandPickerDialog";
import type { FluorIntensityConfig } from "../components/dialogs/IntensityPickerDialog";
import type { CellposeSegConfig } from "../components/dialogs/CellposeSegDialog";
import type { IntensityPlotConfig } from "./intensityPlotOptions";

export type DataKind = "image" | "table" | "plot";
export type EngineKind = "python" | "matlab" | "r" | "imagej" | "cellpose";

export interface InsetSource {
  /** Globally unique: `${figKey}::${rawKey}`. The backend's own key is bare
   *  grid coords and is IDENTICAL across figures, so an un-namespaced key
   *  silently matches another figure's inset — see the refresh in the
   *  sources effect, which rebinds attached sources by key equality. */
  key: string;
  /** The backend's key for this inset, as the run payload needs it. */
  rawKey?: string;
  /** The key the sidecar exposes this source under in `inputs` — see
   *  runtimeKeyFor. Filesystem-safe and bare for the active figure, so
   *  existing single-figure scripts keep working. Recomputed whenever the
   *  active doc changes, which is why it is carried rather than derived at
   *  each use site. */
  runKey?: string;
  /** Stable identity of the owning figure (figKey) — survives a restart for
   *  saved docs. This, not mpfId, is what binds a source to its figure.
   *  A source whose document is CLOSED is keyed `u:<docUid>` (+ `#<figureId>`)
   *  — see utils/workflowOwnership. */
  figKey?: string;
  /** Which MPF this source comes from (doc id) + its display name, so the
   *  Sources panel can group sources into a drawer per loaded MPF and the
   *  project save can tell which MPF a workflow's data belongs to.
   *  mpfId is SESSION-SCOPED (openDocs is not persisted) — never bind on it. */
  mpfId?: string;
  mpf?: string;
  /** Figure tab of that document the source belongs to (utils/figures.ts).
   *  Absent on sources saved before figure tabs = the project's first
   *  figure (DEFAULT_FIGURE_ID) — see figKey. */
  figureId?: string;
  figureName?: string;
  /** Set for STANDALONE analysis images uploaded directly into the Analysis
   *  tab (not a builder inset). When present, the backend resolves the WHOLE
   *  uploaded image (by this loaded-image name) at full bit depth, and row/col/
   *  inset_index are -1. */
  name?: string;
  row: number;
  col: number;
  inset_index: number;
  label: string;
  /** The panel's name from its row/column labels + headers ("1hr, Sample 1").
   *  Display only — the key stays grid coordinates. Absent on older sidecars. */
  panel_name?: string;
  natural_width: number;
  natural_height: number;
  thumbnail: string;
  // Parent panel preview — set by the backend when available.  Used
  // by the library's hover popover to show the inset's location on
  // the host panel so the user knows which region they're picking.
  // bbox is in `parent_natural_*` pixel coordinates.
  parent_thumbnail?: string;
  parent_natural_width?: number;
  parent_natural_height?: number;
  parent_bbox?: [number, number, number, number]; // [x, y, w, h]
  // Whole-figure preview — a tiled thumbnail of every panel in the
  // grid so the hover popover can show WHERE in the figure this
  // source comes from (not just within its own panel).  bbox + cell
  // are in the figure thumbnail's own pixel coordinates.
  figure_thumbnail?: string;
  figure_natural_width?: number;
  figure_natural_height?: number;
  figure_bbox?: [number, number, number, number];
  figure_cell_bbox?: [number, number, number, number];
}

/** A figure's measurements table attached to a SourceNode — carries the
 *  source figure id/name (for the port label + per-MPF folder) and the
 *  CSV the node feeds downstream when its 📋 port is wired. */
export interface MeasurementRef {
  /** Session doc id of the figure the table came from (dedup on attach). */
  mpfId: string;
  mpf: string;
  label: string;
  csv: string;
  /** Durable uid of that document (its `workflows/manifest.json` docUid) —
   *  what ownership and the saved file bind on. Set at attach time; older
   *  refs are back-filled from `mpfId` when serialized. */
  docUid?: string;
}

/** Per-slot ggplot text styling for an R-plot output — mirrors the
 *  collage's RTextOverride and the backend's _r_text_override_block.
 *  `text` is honoured only for the editable label slots (title, axes,
 *  legend title, caption); the tick/legend-label slots take styling
 *  only (they're data-driven). */
export interface RTextOverride {
  text?: string;
  size?: number;
  color?: string;
  bold?: boolean;
  italic?: boolean;
  font?: string;
}

export interface NodeOutput {
  /** Stable id within a node — `out_image_<idx>`, `out_table_<name>`,
   *  `out_plot_<idx>`. Used as the handle id. Archived previous-run
   *  outputs append a __v<N> suffix to keep them distinct from the
   *  newest run's outputs (which use the canonical id). */
  id: string;
  kind: DataKind;
  name: string;
  /** PNG bytes (base64) for `image` / `plot`, CSV text for `table`. */
  payload: string;
  /** Names this payload in the .mpf (`workflows/blobs/<blobId>.png|.csv`).
   *  Minted whenever the payload is produced or re-rendered, so an unchanged
   *  id means unchanged bytes. 1-128 chars of [A-Za-z0-9_-]. */
  blobId?: string;
  /** Whether the user pinned this output (sticky in the drawer's
   *  summary view; pinned items survive node re-runs). */
  pinned?: boolean;
  /** For R plot outputs: the self-contained R script that produced this
   *  plot (upstream input CSVs inlined into the prelude), so the collage
   *  can re-run it at a new font size via "Synchronize headers". */
  rCode?: string;
  rInterpreter?: string | null;
  /** Position of this plot in the run's returned `plots` (filename order) —
   *  run-r's override_plot_index. */
  rPlotIndex?: number;
  /** Destinations this output has already been committed to (one entry
   *  per "Send" click). Surfaced as small chips on the output card so
   *  the user can see at a glance "sent to → My Figure / Collage".
   *  The output stays in the drawer after a send so it can be re-used
   *  on a different destination later. */
  sentTo?: string[];
  /** Empty/undefined = current run. Set to "v1", "v2", ... when a node
   *  re-run pushes the previous output here so the user can compare
   *  old vs new without losing the old one. Capped at v2 per output
   *  slot (older versions drop on the next re-run). Image / plot only;
   *  tables aren't archived. */
  versionLabel?: string;
  /** Per-slot R text styling applied to THIS plot output (Edit text…).
   *  Re-applied on top of rCode via the override re-render so the edits
   *  survive in the drawer; cleared/replaced on the next node run. */
  rTextOverrides?: Record<string, RTextOverride>;
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
  /** Revision of each upstream image source when this node last ran
   *  successfully (store/sourceEditStore). A source edited in Edit Panel
   *  since then makes this node's results stale (utils/analysisStale). */
  ranWith?: Record<string, number>;
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
  /** Currently-loaded preset descriptor, or `"custom"` once the user
   *  has hand-edited the code away from any known preset.  Format:
   *  `"b:<idx>"` for built-in, `"u:<idx>"` for user-saved.  Lets the
   *  preset Select render the active preset's name instead of always
   *  reading "Load preset…". */
  currentPreset?: string;
  /** For source nodes only — list of inset sources to expose as
   *  output handles. */
  sources?: InsetSource[];
  /** For source nodes — the figure's measurements table attached by
   *  dragging the 📋 Measurements tile onto the node. When set, the node
   *  exposes its 📋 (out_table_measurements) port carrying this CSV. */
  measurementsRef?: MeasurementRef;
  /** Additional measurement tables attached to the same source node. A node
   *  used to hold exactly ONE table, so a second drop silently replaced the
   *  first — wiring e.g. "control readings" and "treated readings" into one
   *  script was impossible. `measurementsRef` is kept as the FIRST slot so
   *  saved workflows / sessions (which stored only that field) still load and
   *  keep their existing `out_table_measurements` edge valid. */
  measurementRefs?: MeasurementRef[];
  /** When true, the user has explicitly removed the node from
   *  the canvas — used to no-op stale auto-edges. */
  deleted?: boolean;
  /** Marks a specialised interactive node:
   *    "wb_bands"        → western-blot band picker (Python).
   *    "fluor_intensity" → fluorescence channel intensity picker (Python).
   *    "cellpose_seg"    → Cellpose segmentation config + preview (Cellpose).
   *  The node's `code` is auto-generated from the config (`roi` /
   *  `intensity` / `cellposeSeg`) — manual edits are overwritten on the
   *  next save. Unlike the two Python markers, "cellpose_seg" sits on a
   *  `kind:"cellpose"` node: it configures the JSON that node runs and does
   *  NOT change the engine or the downstream (labels → ImageJ → plot). */
  interactive?: "wb_bands" | "fluor_intensity" | "cellpose_seg";
  /** Band-picker ROI config (only when interactive === "wb_bands"). */
  roi?: BandPickerConfig;
  /** Fluorescence-intensity config (only when interactive === "fluor_intensity"). */
  intensity?: FluorIntensityConfig;
  /** Cellpose segmentation config (only when interactive === "cellpose_seg"). */
  cellposeSeg?: CellposeSegConfig;
  /** "Plot intensities" options; `code` is generated from them. */
  intensityPlot?: IntensityPlotConfig;
  /** hashCode of the generated `code`; a mismatch means it was edited by hand. */
  intensityPlotHash?: string;
  /** The hand-edited code that applying options last replaced. */
  intensityPlotBackup?: { code: string; savedAt: string };
  [key: string]: unknown;  // index signature for ReactFlow's Record
}

// ── Workflow tabs ───────────────────────────────────────────
// Each tab is an independent graph (nodes + edges). The user can
// add multiple to run the same analysis on different image sets.
export interface WorkflowTab {
  id: string;
  name: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
  /** Epoch ms of the last content change (utils/workflowPersist contentSig),
   *  bumped by the workspace store — never while the workflow is locked.
   *  When two files hold the same workflow id, the newer copy wins.
   *  Absent = never edited since it was created / loaded from a legacy save. */
  updatedAt?: number;
  /** Durable uid of the document this workflow is saved in when no figure
   *  data ties it to one (uploads only), and of the one that embeds its
   *  uploaded images when it is shared. Maintained by deriveOwnership
   *  (utils/workflowOwnership). */
  home?: string;
  /** The user picked `home` in the upload-owner prompt — don't ask again. */
  homeConfirmed?: boolean;
}
