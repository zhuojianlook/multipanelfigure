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

import { createContext, forwardRef, useCallback, useContext, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react";
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
  useUpdateNodeInternals,
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
  // parent, so switching nodes resets the open state.  Defaults to
  // OPEN so users see the run output (or "no output yet" hint)
  // immediately under the code editor — closed-by-default was the
  // common "where's my console?" complaint.
  const isError = status === "error";
  const [open, setOpen] = useState(true);
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

// ── Engine paths dialog ──────────────────────────────────────
// Edits the four binary-path overrides for Python / R / MATLAB /
// ImageJ.  Empty fields fall back to the sidecar's auto-detect.
function EngineSettingsDialog({ open, initial, onClose, onSave }: {
  open: boolean;
  initial: EnginePaths;
  onClose: () => void;
  onSave: (next: EnginePaths) => void;
}) {
  const [paths, setPaths] = useState<EnginePaths>(initial);
  useEffect(() => { setPaths(initial); }, [initial, open]);

  const Row = ({ label, glyph, k, hint }: { label: string; glyph: string; k: keyof EnginePaths; hint?: string }) => (
    <Box sx={{ mb: 1.5 }}>
      <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 700, display: "block", mb: 0.25 }}>
        {glyph} {label}
      </Typography>
      <TextField
        fullWidth
        size="small"
        value={paths[k]}
        onChange={(e) => setPaths((p) => ({ ...p, [k]: e.target.value }))}
        placeholder={placeholderForEngine(k)}
        sx={{ "& .MuiInputBase-input": { fontFamily: "monospace", fontSize: "0.75rem" } }}
      />
      {hint && (
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", display: "block", mt: 0.25, lineHeight: 1.3 }}>
          {hint}
        </Typography>
      )}
    </Box>
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem", py: 1.25 }}>Engine binary paths</DialogTitle>
      <DialogContent>
        <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary", display: "block", mb: 1.5, lineHeight: 1.4 }}>
          Pin a specific interpreter for each engine. Leave empty to use the sidecar's auto-detection.
          Paths are kept in localStorage and travel with every node-run request.
        </Typography>
        <Row label="Python" glyph="🐍" k="python"
          hint="Falls back to the running sidecar's `sys.executable` when empty." />
        <Row label="R (Rscript)" glyph="📊" k="r"
          hint="On Windows, point at `Rscript.exe` (NOT R.exe). On macOS / Linux, `Rscript`." />
        <Row label="MATLAB / Octave" glyph="📐" k="matlab"
          hint="MATLAB on Windows: matlab.exe. macOS GUI app: …/MATLAB.app/bin/matlab. Octave's `octave-cli` also works." />
        <Row label="ImageJ / Fiji" glyph="🔬" k="imagej"
          hint="Point at the launcher binary (ImageJ-win64.exe on Windows, ImageJ-macosx in Fiji.app/Contents/MacOS on macOS)." />
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setPaths(DEFAULT_ENGINE_PATHS)}>Clear all</Button>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={() => onSave(paths)} variant="contained">Save</Button>
      </DialogActions>
    </Dialog>
  );
}

// ── Bumps every node's handle registration on tab change ─────
// Lives inside <ReactFlow> so it can use useUpdateNodeInternals().
// When activeWfId flips (load template, switch tab), we schedule a
// burst of internal refreshes — once on the next frame and again
// 80 ms later, to cover the case where the new node tree hasn't
// fully painted by the first call.
/** Exposed up to the parent via a ref so it can imperatively
 *  trigger updateNodeInternals from places where the hook itself
 *  isn't reachable (e.g. addSourceToNode, onConnectStart). */
export interface RFInternalsHandle {
  refresh: (ids?: string[]) => void;
}

const NodeInternalsRefresher = forwardRef<RFInternalsHandle, {
  activeWfId: string;
  nodeIds: string[];
}>(function NodeInternalsRefresher({ activeWfId, nodeIds }, ref) {
  const update = useUpdateNodeInternals();
  // Fire on a staircase of delays after the active workflow tab
  // changes, so we cover both fast browsers (where 16 ms is enough)
  // and slow ones where layout settles only after several frames.
  useLayoutEffect(() => {
    const fire = () => nodeIds.forEach((id) => update(id));
    fire();
    const t1 = setTimeout(fire, 16);
    const t2 = setTimeout(fire, 80);
    const t3 = setTimeout(fire, 250);
    const t4 = setTimeout(fire, 600);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); clearTimeout(t4); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeWfId]);
  // Expose imperative refresh — handy when the parent mutates a
  // node's handle set (drag-drop a source into the source node)
  // and needs RF's internal map updated NOW.
  useImperativeHandle(ref, () => ({
    refresh: (ids?: string[]) => {
      const target = ids && ids.length ? ids : nodeIds;
      // Burst: immediate sync + a couple of post-paint frames.
      target.forEach((id) => update(id));
      setTimeout(() => target.forEach((id) => update(id)), 16);
      setTimeout(() => target.forEach((id) => update(id)), 80);
    },
  }), [nodeIds, update]);
  return null;
});

// ── One row in the source library ─────────────────────────────
// Renders the inset thumbnail + name (with ✎ rename) + an HTML-
// title fallback / popover that shows the parent panel with the
// inset's bounding box highlighted, so the user can see at a
// glance where on the source the region comes from.
function SourceLibraryRow({ source, isUngrouped, onRemoveFromGroup }: {
  source: InsetSource;
  isUngrouped: boolean;
  onRemoveFromGroup: () => void;
}) {
  const cbs = useGraphCallbacks();
  const overrides = cbs?.sourceNameOverrides || {};
  const name = displayName(source, overrides);
  const isCustomName = !!overrides[source.key];
  // Controlled tooltip: the popover can block drag-drops onto a
  // source node (especially when the source node card is positioned
  // along the same horizontal band as the library entry).  We force
  // the tooltip closed the moment the user mouses-down to start a
  // drag, and re-allow hover after the drag ends.
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const dismissForDrag = () => setTooltipOpen(false);
  return (
    <Tooltip
      placement="right"
      open={tooltipOpen}
      onOpen={() => setTooltipOpen(true)}
      onClose={() => setTooltipOpen(false)}
      title={<SourceHoverPreview source={source} name={name} />}
      enterDelay={350}
      enterNextDelay={120}
      // disableInteractive — keeps the popover from intercepting
      // mouse events on the canvas behind it.
      disableInteractive
      componentsProps={{
        tooltip: { sx: { p: 0.5, bgcolor: "background.paper", color: "text.primary", border: "1px solid", borderColor: "divider", boxShadow: 4, maxWidth: 360 } },
        popper: { sx: { pointerEvents: "none" } },
      }}
    >
      <Box
        draggable
        onMouseDown={dismissForDrag}
        onDragStart={(e) => {
          dismissForDrag();
          e.dataTransfer.setData("application/x-mpfig-source", source.key);
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
        {source.thumbnail && (
          <Box component="img" src={`data:image/png;base64,${source.thumbnail}`} alt={name}
            sx={{ width: 26, height: 26, objectFit: "contain", borderRadius: 0.25, flexShrink: 0, border: "1px solid", borderColor: "divider" }}
          />
        )}
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="caption" sx={{
            fontSize: "0.58rem",
            fontWeight: 700,
            display: "block",
            lineHeight: 1.15,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: isCustomName ? "primary.main" : "text.primary",
          }}>
            {name}
          </Typography>
          <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", lineHeight: 1 }}>
            R{source.row + 1}C{source.col + 1}·{source.inset_index + 1} · {source.natural_width}×{source.natural_height}
          </Typography>
        </Box>
        <Box
          component="span"
          onClick={(e) => { e.stopPropagation(); cbs?.renameSource(source); }}
          sx={{ fontSize: "0.55rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "primary.main" } }}
          title="Rename this source"
        >✎</Box>
        {!isUngrouped && (
          <Box
            component="span"
            onClick={(e) => { e.stopPropagation(); onRemoveFromGroup(); }}
            sx={{ fontSize: "0.6rem", cursor: "pointer", color: "text.disabled", px: 0.25, "&:hover": { color: "error.main" } }}
            title="Move back to Ungrouped"
          >×</Box>
        )}
      </Box>
    </Tooltip>
  );
}

// ── Hover popover: whole-figure context + highlighted region ──
// Two stacked previews so the user sees BOTH the source's host
// panel in the context of the whole figure AND the inset region
// inside its panel:
//   1) Figure thumbnail with the source's cell + region marked
//   2) Per-panel close-up with the inset's bbox highlighted
function SourceHoverPreview({ source, name }: { source: InsetSource; name: string }) {
  const fb = source.figure_bbox;
  const fcb = source.figure_cell_bbox;
  const fw = source.figure_natural_width || 0;
  const fh = source.figure_natural_height || 0;
  const showFigure = !!(source.figure_thumbnail && fw > 0 && fh > 0 && fb);

  const pw = source.parent_natural_width || 0;
  const ph = source.parent_natural_height || 0;
  const pb = source.parent_bbox;
  const showParent = !!(source.parent_thumbnail && pw > 0 && ph > 0 && pb);

  const pct = (v: number, denom: number) => (denom > 0 ? (v / denom) * 100 : 0);

  return (
    <Box sx={{ width: 340, p: 0.25 }}>
      <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 700, display: "block", mb: 0.5 }}>
        {name}
      </Typography>
      <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", mb: 0.75 }}>
        Panel R{source.row + 1} C{source.col + 1}
        {source.inset_index >= 0 ? ` · inset ${source.inset_index + 1}` : ""}
        {` · ${source.natural_width}×${source.natural_height} px`}
      </Typography>
      {showFigure && (
        <Box sx={{ mb: 0.75 }}>
          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", mb: 0.25, letterSpacing: 0.5, textTransform: "uppercase" }}>
            Position in figure
          </Typography>
          <Box sx={{ position: "relative", width: "100%", border: "1px solid", borderColor: "divider", borderRadius: 0.25, overflow: "hidden", bgcolor: "background.default" }}>
            <Box component="img" src={`data:image/png;base64,${source.figure_thumbnail}`} alt="figure"
              sx={{ display: "block", width: "100%", height: "auto" }} />
            {fcb && (
              <Box sx={{
                position: "absolute",
                left: `${pct(fcb[0], fw)}%`, top: `${pct(fcb[1], fh)}%`,
                width: `${pct(fcb[2], fw)}%`, height: `${pct(fcb[3], fh)}%`,
                border: "1.5px solid #2196f3",
                pointerEvents: "none",
              }} />
            )}
            <Box sx={{
              position: "absolute",
              left: `${pct(fb![0], fw)}%`, top: `${pct(fb![1], fh)}%`,
              width: `${pct(fb![2], fw)}%`, height: `${pct(fb![3], fh)}%`,
              border: "2px solid #ffa726",
              boxShadow: "0 0 0 1px rgba(0,0,0,0.4) inset",
              pointerEvents: "none",
            }} />
          </Box>
        </Box>
      )}
      {showParent && (
        <Box>
          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", mb: 0.25, letterSpacing: 0.5, textTransform: "uppercase" }}>
            Region in panel
          </Typography>
          <Box sx={{ position: "relative", width: "100%", border: "1px solid", borderColor: "divider", borderRadius: 0.25, overflow: "hidden", bgcolor: "background.default" }}>
            <Box component="img" src={`data:image/png;base64,${source.parent_thumbnail}`} alt="parent panel"
              sx={{ display: "block", width: "100%", height: "auto" }} />
            <Box sx={{
              position: "absolute",
              left: `${pct(pb![0], pw)}%`, top: `${pct(pb![1], ph)}%`,
              width: `${pct(pb![2], pw)}%`, height: `${pct(pb![3], ph)}%`,
              border: "2px solid #ffa726",
              boxShadow: "0 0 0 1px rgba(0,0,0,0.4) inset",
              pointerEvents: "none",
            }} />
          </Box>
        </Box>
      )}
      {!showFigure && !showParent && source.thumbnail && (
        <Box component="img" src={`data:image/png;base64,${source.thumbnail}`} alt={name}
          sx={{ display: "block", maxWidth: "100%", maxHeight: 200, objectFit: "contain", border: "1px solid", borderColor: "divider" }} />
      )}
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
            <SourceLibraryRow
              key={s.key}
              source={s}
              isUngrouped={!!isUngrouped}
              onRemoveFromGroup={() => onRemoveKey(s.key)}
            />
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
  /** Per-source display-name overrides (user renames). */
  sourceNameOverrides: Record<string, string>;
  /** Prompt the user to rename this inset + persist the override. */
  renameSource: (s: InsetSource) => void;
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
export type EngineKind = "python" | "matlab" | "r" | "imagej" | "cellpose";

export interface InsetSource {
  key: string;
  row: number;
  col: number;
  inset_index: number;
  label: string;
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
  /** Currently-loaded preset descriptor, or `"custom"` once the user
   *  has hand-edited the code away from any known preset.  Format:
   *  `"b:<idx>"` for built-in, `"u:<idx>"` for user-saved.  Lets the
   *  preset Select render the active preset's name instead of always
   *  reading "Load preset…". */
  currentPreset?: string;
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
  cellpose: "#3b7a78", // teal
};
const KIND_BORDER: Record<"source" | EngineKind, string> = {
  source: "#90a4ae",
  python: "#6f8aa8",
  matlab: "#b69266",
  r:      "#8aa37b",
  imagej: "#9886b3",
  cellpose: "#6fa5a2",
};

const KIND_ICON: Record<"source" | EngineKind, string> = {
  source: "📥",
  python: "🐍",
  matlab: "📐",
  r: "📊",
  imagej: "🔬",
  cellpose: "🧬",
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
  // React Flow caches a node's handle set from the first render —
  // since SourceNode emits a `<Handle id="out_image_<idx>" />` per
  // attached inset, every time the user adds / removes a source we
  // need to tell RF the handle map changed, otherwise the new dots
  // appear in the DOM but aren't connectable.
  //
  // CRITICAL: must be useEffect, NOT useLayoutEffect.  RF's <Handle>
  // registers itself with the store via its own useEffect (which
  // runs AFTER all useLayoutEffects in the commit phase).  If we
  // call updateNodeInternals from a useLayoutEffect, the new Handle
  // hasn't registered yet and RF measures stale handles only —
  // producing the "visible-but-dead port" bug where the user has to
  // close+reopen the dialog before the new source can be connected
  // to a downstream node.  Switching to useEffect lets the child
  // Handle's mount effect run first, then we trigger the re-measure.
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    // Immediate sync call now that all child effects (Handle
    // registration) have already fired.
    updateNodeInternals(id);
    // Belt-and-braces staircase to cover slow-frame edge cases —
    // some browsers don't fully settle layout until the next paint.
    const t1 = setTimeout(() => updateNodeInternals(id), 16);
    const t2 = setTimeout(() => updateNodeInternals(id), 120);
    const t3 = setTimeout(() => updateNodeInternals(id), 320);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [id, sources.length, showMeasurements, updateNodeInternals]);

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
                  {displayName(s, cbs?.sourceNameOverrides || {})}
                </Typography>
                <Typography variant="caption" sx={{ fontSize: "0.5rem", color: "text.secondary", lineHeight: 1 }}>
                  R{s.row + 1}C{s.col + 1}·{s.inset_index + 1}
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
  // Force React Flow to (re-)scan our handle set on mount + when
  // the engine kind changes (engine drives WHICH handles render —
  // R has out_plot, others have out_image).  Loading a template
  // mounted these nodes with handles that RF hadn't seen yet,
  // leaving connections to / from them dead until the dialog was
  // re-opened.  Sync-after-render + a delayed re-fire closes that
  // gap on slow layouts.
  const updateNodeInternals = useUpdateNodeInternals();
  useLayoutEffect(() => {
    updateNodeInternals(id);
    const t1 = setTimeout(() => updateNodeInternals(id), 16);
    const t2 = setTimeout(() => updateNodeInternals(id), 120);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [id, engine, updateNodeInternals]);
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
  cellpose: ProcessNode,
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
//
// Inputs are exposed as three parallel arrays:
//   input_paths[i]   absolute filesystem path (use with open())
//   input_labels[i]  human label (e.g. "R1C1·1")
//   input_keys[i]    stable key (e.g. "inset_0" or "out_image_0")
// CSV output via mpfig_dataN(name, "h1,h2,…", col1, col2, …).

setBatchMode(true);
n = lengthOf(input_paths);
labels = newArray(n);
means  = newArray(n);
maxes  = newArray(n);
for (i = 0; i < n; i++) {
  open(input_paths[i]);
  labels[i] = input_labels[i];
  getStatistics(area, mean, min, max);
  means[i] = "" + mean;
  maxes[i] = "" + max;
  close();
}
mpfig_data3("stats", "label,mean,max", labels, means, maxes);
`;
const IJ_PARTICLES = `// PARTICLE ANALYSIS — threshold + count + size per image.
// @name: Particle analysis

setBatchMode(true);
n = lengthOf(input_paths);
labels = newArray(n);
counts = newArray(n);
sizes  = newArray(n);
for (i = 0; i < n; i++) {
  open(input_paths[i]);
  run("8-bit");
  setAutoThreshold("Otsu dark");
  run("Convert to Mask");
  run("Analyze Particles...", "size=10-Infinity show=Outlines summarize");
  labels[i] = input_labels[i];
  counts[i] = "" + nResults;
  sizes[i]  = (nResults > 0) ? "" + getResult("Area", 0) : "0";
  close();
  close();
}
mpfig_data3("particles", "label,count,first_area", labels, counts, sizes);
`;
// Cell shape metrics from Cellpose label maps — designed for the
// "Cell characteristics" workflow.  Reads *_labels.png inputs (8-bit
// grayscale, pixel value = cell ID) and uses ImageJ's built-in
// measurement routines for area / perim / circularity / Feret /
// AR / Solidity etc.  Far more battle-tested than rolling our own
// in numpy.
const IJ_CELL_SHAPE_METRICS = `// @name: Cell shape metrics (ImageJ)
// Iterates input_paths, opens each as a Cellpose *_labels image
// (8-bit grayscale, value = cell ID), and measures every cell
// individually using ImageJ's built-in shape descriptors.
// Emits a per-cell CSV row.  Columns:
//   source         — image label (e.g. "Control_1")
//   group          — inferred biological group (e.g. "Control")
//   cell_id        — per-image sequential ID
//   area_px        — pixel area
//   perimeter_px   — pixel perimeter
//   circularity    — 4πA/P²        ∈ [0, 1]   (1 = perfect circle)
//   eq_diameter_px — 2·√(A/π)      circle of same area
//   feret_px       — Feret max caliper diameter
//   min_feret_px   — Feret min caliper diameter (cell "width")
//   aspect_ratio   — Feret / MinFeret (major/minor axis ratio)
//   roundness      — 4·A / (π·MajorAxis²)     ∈ [0, 1]
//   solidity       — A / ConvexA              ∈ [0, 1]

setBatchMode(true);

// Set Measurements: enable area, perimeter, shape descriptors
// (Circ./AR/Round/Solidity), and Feret's diameter (Feret/MinFeret/
// FeretAngle).  These map to getResult("…", row) lookups below.
run("Set Measurements...", "area perimeter shape feret's redirect=None decimal=3");

// Per-metric output arrays.  IJ macro has no struct/list type, so
// we use parallel arrays and emit them as columns at the end via
// mpfig_data8.  Array.concat appends one element at a time.
out_source       = newArray(0);
out_group        = newArray(0);
out_cell_id      = newArray(0);
out_area         = newArray(0);
out_perim        = newArray(0);
out_circ         = newArray(0);
out_eqdiam       = newArray(0);
out_feret        = newArray(0);
out_minferet     = newArray(0);
out_ar           = newArray(0);
out_round        = newArray(0);
out_solidity     = newArray(0);

n_inputs = lengthOf(input_paths);
for (i = 0; i < n_inputs; i++) {
  path  = input_paths[i];
  label = input_labels[i];

  // Skip inputs that aren't *_labels — outline / mask variants
  // also come down the pipe but only the labels image has the
  // grayscale per-cell IDs we need.
  if (indexOf(label, "_labels") < 0 && indexOf(input_keys[i], "_labels") < 0) {
    print("skipping non-labels input: " + label);
    continue;
  }

  open(path);
  run("8-bit");  // ensure single-channel

  // -- Clean the label so plots downstream show readable categories.
  // Upstream labels come through as "<NodeName>/<imageName>_labels"
  // (the runner prefixes the upstream node's display label).  Strip
  // BOTH the path prefix and the "_labels" suffix.
  clean = label;
  slash = lastIndexOf(clean, "/");
  if (slash >= 0) clean = substring(clean, slash + 1);
  clean = replace(clean, "_labels", "");

  // -- Infer biological group from the source name.
  // Strip trailing "_N", "-N", or " N" (single or multi-digit
  // numeric suffix) — turns "Control_1", "Control-2", "Control 3"
  // all into "Control".  If no numeric suffix, the whole cleaned
  // label is the group.  Mirrors the old Python harness's
  // re.match(r"^(.*?)[\\s_\\-]+\\d+$", label) regex.
  group = clean;
  // Walk from the end while the last char is a digit, then strip
  // an immediately-preceding separator [_- ].
  endIdx = lengthOf(group);
  while (endIdx > 0) {
    ch = substring(group, endIdx - 1, endIdx);
    if (ch >= "0" && ch <= "9") endIdx--;
    else break;
  }
  if (endIdx > 0 && endIdx < lengthOf(group)) {
    sep = substring(group, endIdx - 1, endIdx);
    if (sep == "_" || sep == "-" || sep == " ") {
      group = substring(group, 0, endIdx - 1);
    }
  }
  if (lengthOf(group) == 0) group = clean;

  // Find the max label value via image statistics.
  getStatistics(area_total, mean, min_v, max_v);
  n_cells_max = round(max_v);
  print("measuring '" + clean + "' (group='" + group + "') — max label value: " + n_cells_max);

  cell_counter = 0;
  for (lab = 1; lab <= n_cells_max; lab++) {
    // Threshold == lab to isolate one cell.
    setThreshold(lab, lab);
    run("Create Selection");
    // Selection is empty if this label value isn't present.
    if (selectionType() < 0) continue;

    // Read measurements via the Result table.  Result columns
    // ("Area", "Perim.", "Circ.", "AR", "Round", "Solidity",
    // "Feret", "MinFeret") are populated by Set Measurements
    // above; missing keys default to 0 so we read defensively.
    run("Measure");
    n = nResults;
    if (n == 0) continue;
    a    = getResult("Area",     n - 1);
    p    = getResult("Perim.",   n - 1);
    c    = getResult("Circ.",    n - 1);
    ar   = getResult("AR",       n - 1);
    rnd  = getResult("Round",    n - 1);
    sol  = getResult("Solidity", n - 1);
    fd   = getResult("Feret",    n - 1);
    mfd  = getResult("MinFeret", n - 1);
    run("Select None");

    if (a < 20) continue;  // discard sub-threshold blobs (matches Python harness)

    // Equivalent diameter — diameter of a circle with the same area.
    // R's "Plot cell characteristics" preset expects this column
    // verbatim ("eq_diameter_px").  Pre-compute it client-side so
    // the macro stays a single-pass pipeline.
    eqd = 2.0 * sqrt(a / PI);

    cell_counter++;
    out_source   = Array.concat(out_source,   clean);
    out_group    = Array.concat(out_group,    group);
    out_cell_id  = Array.concat(out_cell_id,  "" + cell_counter);
    out_area     = Array.concat(out_area,     "" + a);
    out_perim    = Array.concat(out_perim,    "" + p);
    out_circ     = Array.concat(out_circ,     "" + c);
    out_eqdiam   = Array.concat(out_eqdiam,   "" + eqd);
    out_feret    = Array.concat(out_feret,    "" + fd);
    out_minferet = Array.concat(out_minferet, "" + mfd);
    out_ar       = Array.concat(out_ar,       "" + ar);
    out_round    = Array.concat(out_round,    "" + rnd);
    out_solidity = Array.concat(out_solidity, "" + sol);
  }
  print("  measured " + cell_counter + " cell(s)");
  resetThreshold();
  close();
  // Clear Results so the next image starts from 0.
  if (isOpen("Results")) { selectWindow("Results"); run("Close"); }
}

print("total cells: " + lengthOf(out_source));
// 12-column emit — uses mpfig_data12 helper generated by the harness
// prelude (see api_server.py run-imagej prelude, which now generates
// dataN variants from 1 through 12).
mpfig_data12("cell_metrics",
  "source,group,cell_id,area_px,perimeter_px,circularity,eq_diameter_px,feret_px,min_feret_px,aspect_ratio,roundness,solidity",
  out_source, out_group, out_cell_id, out_area, out_perim, out_circ, out_eqdiam, out_feret, out_minferet, out_ar, out_round, out_solidity);
`;

const IMAGEJ_PRESETS: CodePreset[] = [
  { name: "Custom (starter)", code: IJ_DEFAULT },
  { name: "Particle analysis", code: IJ_PARTICLES },
  { name: "Cell shape metrics (from Cellpose)", code: IJ_CELL_SHAPE_METRICS },
];

// ── Cellpose 3 module ────────────────────────────────────────
// A first-class "module" engine alongside the free-form code
// engines.  The node body is a JSON config that the sidecar
// parses; the harness runs the chosen cellpose model against
// each upstream image, emits the labelled mask as an image, and
// produces a per-image CSV with cell counts + size stats.
const CELLPOSE_DEFAULT = `{
  "model": "cpsam",
  "diameter": null,
  "channels": [0, 0],
  "flow_threshold": 0.4,
  "cellprob_threshold": 0.0,
  "min_size": 15,
  "use_gpu": false
}
// model: Cellpose 4.x ships only "cpsam" (the SAM-based segmenter)
//        as a built-in.  Earlier names ("cyto3", "cyto2", "nuclei",
//        "livecell") are accepted for back-compat with Cellpose 3 —
//        the sidecar auto-falls back to "cpsam" when running on v4
//        and prints a warning in the Console.
// diameter: estimated cell diameter in pixels (null = auto-estimate)
// channels: [cyto, nuclei] — 0 = grayscale, 1 = red, 2 = green, 3 = blue
//           (deprecated under v4 but still accepted)
// flow_threshold: 0..3 — lower keeps fewer questionable cells
// cellprob_threshold: -6..6 — lower includes dimmer cells
// min_size: discard masks below this area (pixels)
// use_gpu: requires a working torch + CUDA / MPS install`;

const CELLPOSE_NUCLEI = `{
  "model": "cpsam",
  "diameter": 17,
  "channels": [0, 0],
  "flow_threshold": 0.4,
  "cellprob_threshold": 0.0,
  "min_size": 10,
  "use_gpu": false
}
// Tuned for DAPI / Hoechst nuclei: small diameter prior, lower
// min_size.  Under Cellpose 4 the model is always "cpsam".`;

const CELLPOSE_PRESETS: CodePreset[] = [
  { name: "Default (cpsam)", code: CELLPOSE_DEFAULT },
  { name: "Nuclei (DAPI)",   code: CELLPOSE_NUCLEI },
];

const BUILTIN_PRESETS: Record<EngineKind, CodePreset[]> = {
  python: PYTHON_PRESETS,
  matlab: MATLAB_PRESETS,
  r: R_PRESETS,
  imagej: IMAGEJ_PRESETS,
  cellpose: CELLPOSE_PRESETS,
};

// ── User-saved code presets (localStorage) ──────────────────
const USER_PRESET_KEY: Record<EngineKind, string> = {
  python: "mpfig.user_presets.python",
  matlab: "mpfig.user_presets.matlab",
  r: "mpfig.user_presets.r",
  imagej: "mpfig.user_presets.imagej",
  cellpose: "mpfig.user_presets.cellpose",
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

// ── Custom engine paths (localStorage) ─────────────────────────
// Lets the user point each engine at a specific interpreter binary
// when the auto-detected default doesn't match their install.
// Empty string ⇒ fall back to the sidecar's auto-detection
// (which probes PATH, well-known macOS/Linux dirs, and the Windows
// install locations enumerated below).
const ENGINE_PATHS_KEY = "mpfig.engine_paths";
interface EnginePaths {
  python: string;
  r: string;
  matlab: string;
  imagej: string;
}
const DEFAULT_ENGINE_PATHS: EnginePaths = { python: "", r: "", matlab: "", imagej: "" };
function loadEnginePaths(): EnginePaths {
  try {
    const raw = localStorage.getItem(ENGINE_PATHS_KEY);
    if (!raw) return DEFAULT_ENGINE_PATHS;
    return { ...DEFAULT_ENGINE_PATHS, ...(JSON.parse(raw) as Partial<EnginePaths>) };
  } catch { return DEFAULT_ENGINE_PATHS; }
}
function saveEnginePaths(p: EnginePaths) {
  try { localStorage.setItem(ENGINE_PATHS_KEY, JSON.stringify(p)); } catch { /* ignore */ }
}
/** Best-guess Windows install paths surfaced as TextField placeholders.
 *  Used purely as hints — the user still needs to confirm the path
 *  exists.  On macOS / Linux the placeholders shift to the common
 *  Unix locations. */
function placeholderForEngine(engine: keyof EnginePaths): string {
  const isWin = typeof navigator !== "undefined" && /windows/i.test(navigator.userAgent || "");
  if (isWin) {
    switch (engine) {
      case "python": return "C:\\Python312\\python.exe";
      case "r":      return "C:\\Program Files\\R\\R-4.4.0\\bin\\Rscript.exe";
      case "matlab": return "C:\\Program Files\\MATLAB\\R2024a\\bin\\matlab.exe";
      case "imagej": return "C:\\Fiji.app\\ImageJ-win64.exe";
    }
  }
  switch (engine) {
    case "python": return "/usr/local/bin/python3";
    case "r":      return "/usr/local/bin/Rscript";
    case "matlab": return "/Applications/MATLAB_R2024a.app/bin/matlab  (or octave)";
    case "imagej": return "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx";
  }
}

// ── Per-source display-name overrides (localStorage) ────────
// The library auto-generates labels like "R1C1·1"; users rename
// them in-place to things like "Control_1" / "Drug_2" so the
// downstream group-inference reads what they wrote.  Overrides
// are keyed by the inset's stable `.key` (panel coords + index)
// so renames survive figure changes that don't move the inset.
const SOURCE_NAMES_KEY = "mpfig.source_names";
function loadSourceNames(): Record<string, string> {
  try {
    const raw = localStorage.getItem(SOURCE_NAMES_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch { return {}; }
}
function saveSourceNames(m: Record<string, string>) {
  try { localStorage.setItem(SOURCE_NAMES_KEY, JSON.stringify(m)); } catch { /* ignore */ }
}
/** Display name for an inset — user override if present, else the
 *  backend-supplied label (`R{r}C{c}·{idx}`). */
function displayName(s: InsetSource, overrides: Record<string, string>): string {
  return (overrides[s.key] || s.label || s.key).trim();
}

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

// ── Built-in workflow templates ───────────────────────────────
// Shipped multi-node pipelines surfaced in the "Load saved"
// dropdown under a "Built-in" header.  IDs are prefixed with
// "builtin:" so loadSavedWorkflow can route them to the constant
// list instead of the user's localStorage saves.

// Python source bodies for each node in the Corneal Haze workflow.
// Kept as top-level constants so each one can be unit-tested by
// dropping into a Python REPL with mock `inputs`.
const HAZE_NORMALIZE = `# @name: Normalize
# Background-floor subtraction.  For each image we anchor its
# DARK background to zero by subtracting the 5th-percentile pixel
# value, then clip a 99.5th-percentile glare cap.  The relative
# bright-pixel signal (which is what carries corneal haze) is
# preserved — unlike a min-max stretch, which would equalise the
# histograms across groups and erase the very thing we're measuring.
# This is the right normalisation for slit-lamp / scheimpflug
# corneal photographs taken at consistent exposure: it removes
# camera dark-current and ambient-light offsets while keeping the
# haze-scattering intensity in absolute terms.
import numpy as np

imgs = [(k, v) for k, v in inputs.items() if isinstance(v, dict) and "image" in v]
if not imgs:
    raise SystemExit("No image inputs — drag your control + sample insets into the source node, then wire each into this node's image input.")
print(f"normalising {len(imgs)} image(s) (dark-floor subtract + glare cap)")

for key, src in imgs:
    arr = src["image"].astype(np.float32)
    floor = float(np.percentile(arr, 5))    # dark background level
    cap   = float(np.percentile(arr, 99.5)) # specular-glare cap
    # Subtract the dark floor and clip extreme glare, but DO NOT
    # rescale to fill 0-255 — that would re-equalise the histograms
    # across groups and erase the haze signal we're measuring.  The
    # downstream mean-grey metric reads absolute intensity in this
    # post-subtraction space, so hazier images stay genuinely brighter.
    norm = np.clip(arr - floor, 0, max(cap - floor, 1.0)).clip(0, 255).astype(np.uint8)
    label = src.get("label", key)
    # Keep the OUTPUT name identical to the source label so the
    # downstream "Group" inference reads the user's nice naming
    # ("Control_1", "Treatment_A_2") instead of "norm_…".
    mpfig_image(norm, name=label)
`;

const HAZE_GRAYMEAN = `# @name: Grayscale + ROI metrics
# Convert each upstream image to BT.601 luminance, extract the
# CENTRAL 80% as the region of interest (edges + eyelash artefacts
# get cropped), and emit a row per image with the ROI's mean /
# median / std grey value.  Group inference: split the source
# label on the LAST underscore / dash / space before a trailing
# replicate number ("Control_1" → group "Control"); falls back to
# the whole label when no replicate suffix is present.  Control
# group = the one literally named "control" (case-insensitive),
# else the FIRST group seen.
import numpy as np
import re

imgs = [(k, v) for k, v in inputs.items() if isinstance(v, dict) and "image" in v]
if not imgs:
    raise SystemExit("No image inputs — wire normalised images into the image input.")

def clean(label):
    # Upstream labels arrive as "PrevNode/Control_1"; strip the node prefix.
    if "/" in label:
        label = label.rsplit("/", 1)[-1]
    return label.strip()

def infer_group(label):
    m = re.match(r"^(.*?)[\s_\-]+\d+$", label)
    return (m.group(1) if m else label).strip() or label

rows = []
group_order = []
for key, src in imgs:
    img = src["image"].astype(np.float32)
    if img.ndim == 2:
        gray = img
    else:
        # BT.601 luminance.
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # Central 80% ROI — symmetric crop on each axis.
    h, w = gray.shape
    y0, y1 = int(h * 0.10), int(h * 0.90)
    x0, x1 = int(w * 0.10), int(w * 0.90)
    roi = gray[y0:y1, x0:x1]

    raw_label = clean(src.get("label", key))
    grp = infer_group(raw_label)
    if grp not in group_order:
        group_order.append(grp)

    rows.append({
        "source": raw_label,
        "group": grp,
        "mean_gray": float(roi.mean()),
        "median_gray": float(np.median(roi)),
        "std_gray": float(roi.std()),
        "n_pixels": int(roi.size),
    })

# Decide which group is the control.
ctrl_grp = next((g for g in group_order if g.lower() == "control"), None)
if ctrl_grp is None:
    ctrl_grp = group_order[0]
for r in rows:
    r["is_control"] = (r["group"] == ctrl_grp)

print(f"groups: {group_order} (control = {ctrl_grp!r})")
print(f"per-image: " + "; ".join(f"{r['source']} → {r['mean_gray']:.1f}" for r in rows))
mpfig_data(rows, name="gray_means")
`;

const HAZE_SCORE = `# @name: Haze score
# Compute every image's haze offset relative to the CONTROL group's
# mean grey value across all its replicates.  Robust to a
# single-replicate control (uses that one value) and to multi-
# replicate controls (uses their pooled mean).
import math

table = None
for v in inputs.values():
    if isinstance(v, dict) and "table" in v:
        table = v["table"]
        break
if not table:
    raise SystemExit("No upstream table — wire 'Grayscale + ROI metrics' into the table input.")

ctrl_rows = [r for r in table if str(r.get("is_control")).lower() == "true"]
if not ctrl_rows:
    # Fallback for tables that don't carry the flag.
    ctrl_rows = [table[0]]
ctrl_vals = [float(r["mean_gray"]) for r in ctrl_rows]
ctrl_mean = sum(ctrl_vals) / len(ctrl_vals)
ctrl_grp = ctrl_rows[0].get("group", ctrl_rows[0].get("source"))
print(f"control group = {ctrl_grp!r} (n={len(ctrl_vals)}, mean={ctrl_mean:.2f})")

rows = []
for r in table:
    m = float(r["mean_gray"])
    rows.append({
        **r,
        "haze_score": m - ctrl_mean,
        "ratio_to_control": (m / ctrl_mean) if ctrl_mean else math.nan,
    })
mpfig_data(rows, name="haze_scores")
`;

const HAZE_PLOT_R = `# @name: Plot haze
# Publication-quality plot of corneal haze scores.  Per group:
#   • Bar = mean haze_score (filled by control / treatment)
#   • Errorbar = ± SEM
#   • Jittered points = each replicate (alpha-blended)
# Dashed horizontal line marks the control baseline (0).  When
# rstatix is available we annotate the plot subtitle with the
# group-wise Wilcoxon (2 groups) or Kruskal–Wallis (3+ groups)
# p-value; falls back silently otherwise.
library(ggplot2); library(ggprism)

# First (and in practice only) upstream table.
data <- inputs[[1]]
data$group <- factor(data$group, levels = unique(data$group))
data$is_control <- as.logical(toupper(as.character(data$is_control)))
ctrl_grp <- as.character(unique(data$group[data$is_control])[1])
if (is.na(ctrl_grp)) ctrl_grp <- as.character(levels(data$group))[1]

# Per-group summary (mean + SEM) for the bar layer.
summ <- do.call(rbind, lapply(split(data, data$group), function(d) {
  data.frame(
    group = d$group[1],
    n = nrow(d),
    mean_haze = mean(d$haze_score, na.rm = TRUE),
    sem_haze = ifelse(nrow(d) > 1,
                      sd(d$haze_score, na.rm = TRUE) / sqrt(nrow(d)),
                      0),
    is_control = d$is_control[1]
  )
}))

# Compose the subtitle from non-parametric group stats.  Best
# effort — bail silently if rstatix isn't reachable.
subtitle <- ""
groups_n <- length(levels(data$group))
if (groups_n >= 2 && requireNamespace("rstatix", quietly = TRUE)) {
  tryCatch({
    if (groups_n == 2) {
      t <- rstatix::wilcox_test(data, haze_score ~ group)
      subtitle <- sprintf("Wilcoxon rank-sum: p = %.3g", t$p[1])
    } else {
      kw <- rstatix::kruskal_test(data, haze_score ~ group)
      subtitle <- sprintf("Kruskal–Wallis: H(%d) = %.2f, p = %.3g",
                          kw$df[1], kw$statistic[1], kw$p[1])
    }
  }, error = function(e) { subtitle <<- "" })
}

# X-axis labels carry the per-group N inline.  Use a single-line
# format ("Control · n=4") — multi-line labels via paste0 + line
# break are fiddly to escape across the TS → JSON → Python → R
# transport, and a single line wraps cleanly under theme_prism.
xlabs <- paste0(as.character(summ$group), " · n=", summ$n)
names(xlabs) <- as.character(summ$group)

# Set the output to publication dimensions: 5" × 4" at 300 DPI.
mpfig_plot("haze_score.png", width = 1500, height = 1200, res = 300)

# Build the plot with NO inherited aesthetics on the main ggplot()
# call — every geom carries its own data + aes.  This avoids any
# downstream overlay (e.g. add_pvalue) tripping on a column it
# doesn't have, and keeps each layer independent.
p <- ggplot() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey55",
             linewidth = 0.5) +
  geom_col(data = summ,
           aes(x = group, y = mean_haze, fill = is_control),
           width = 0.65, color = "grey25", linewidth = 0.4) +
  geom_errorbar(data = summ,
                aes(x = group, ymin = mean_haze - sem_haze,
                    ymax = mean_haze + sem_haze),
                width = 0.22, color = "grey25", linewidth = 0.5) +
  geom_jitter(data = data,
              aes(x = group, y = haze_score),
              width = 0.13, height = 0, size = 1.7, alpha = 0.75,
              color = "grey20", stroke = 0) +
  # Legend says just "Control" / "Treatment" — the control group's
  # actual name is already on the x-axis, so duplicating it
  # ("Control (Control)") would be redundant.  Only show the name
  # parenthetically when it's something other than literally
  # "Control".
  scale_fill_manual(values = c("TRUE" = "#cfd8dc", "FALSE" = "#c98a47"),
                    name = NULL,
                    labels = c("TRUE" = if (tolower(ctrl_grp) == "control")
                                          "Control"
                                        else
                                          paste0("Control (", ctrl_grp, ")"),
                               "FALSE" = "Treatment"),
                    breaks = c("TRUE", "FALSE")) +
  scale_x_discrete(labels = xlabs) +
  labs(x = NULL,
       y = expression(paste("Haze score (", Delta, " mean grey vs control)")),
       subtitle = subtitle) +
  theme_prism(base_size = 11) +
  theme(legend.position = "top",
        legend.text = element_text(size = 9),
        axis.text.x = element_text(size = 9.5, lineheight = 0.95),
        plot.subtitle = element_text(size = 9, color = "grey40",
                                     margin = margin(b = 4)),
        plot.margin = margin(10, 10, 6, 10))

print(p)
`;

/** Build the Corneal Haze workflow as a SavedWorkflow.  Source
 *  node starts empty — user drags their control inset FIRST,
 *  then the experimental insets, then wires each one's image
 *  port to the Normalize node's `in_image` input. */
function buildCornealHazeWorkflow(): SavedWorkflow {
  const sourceId = "source";
  const normId = "haze_normalize";
  const grayId = "haze_graymean";
  const scoreId = "haze_score";
  const plotId = "haze_plot";
  const nodes: Node<NodeData>[] = [
    {
      id: sourceId,
      type: "source",
      position: { x: 40, y: 60 },
      data: {
        label: "Source — name insets 'Control_1', 'Treatment_1' …",
        kind: "source",
        sources: [],
        status: "ok",
      } as NodeData,
      draggable: true,
      deletable: false,
    },
    {
      id: normId,
      type: "python",
      position: { x: 320, y: 60 },
      data: { label: "Normalize", kind: "python", code: HAZE_NORMALIZE,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: grayId,
      type: "python",
      position: { x: 580, y: 60 },
      data: { label: "Grayscale + ROI metrics", kind: "python", code: HAZE_GRAYMEAN,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: scoreId,
      type: "python",
      position: { x: 840, y: 60 },
      data: { label: "Haze score", kind: "python", code: HAZE_SCORE,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: plotId,
      type: "r",
      position: { x: 1100, y: 60 },
      data: { label: "Plot haze", kind: "r", code: HAZE_PLOT_R,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
  ];
  const mkEdge = (sId: string, sH: string, tId: string, tH: string, kind: DataKind): Edge => ({
    id: `e_${sId}_${sH}__${tId}_${tH}`,
    source: sId, sourceHandle: sH,
    target: tId, targetHandle: tH,
    type: "default",
    animated: false,
    style: { stroke: PORT_COLOR[kind], strokeWidth: 2 },
  });
  const edges: Edge[] = [
    // Normalize → Grayscale + mean (image stream)
    mkEdge(normId,  "out_image", grayId,  "in_image", "image"),
    // Grayscale + mean → Haze score (table stream)
    mkEdge(grayId,  "out_table", scoreId, "in_table", "table"),
    // Haze score → R plot (table stream)
    mkEdge(scoreId, "out_table", plotId,  "in_table", "table"),
  ];
  return {
    id: "builtin:corneal_haze",
    name: "Corneal haze (normalize → grey → score → plot)",
    nodes, edges, createdAt: 0,
  };
}

// ── Intensity per channel (fluorescence) ─────────────────────
// 3-node pipeline: per-image RGB-channel mean intensity → group
// summary → faceted bar plot.  Each image's label is parsed for
// group + (optional) channel suffix; if no suffix is given, all
// three channels are emitted for every image.
const INTENSITY_PYTHON = `# @name: Channel intensities
# Per-image mean intensity in each of the R / G / B channels.
# Group is inferred from the source label using the same
# "<group>_<replicate>" convention the Corneal Haze workflow uses
# (e.g. "GFP_1", "DAPI_3").  Falls back to the whole label when no
# replicate suffix is present.
import numpy as np, re

def infer_group(label):
    label = label.rsplit("/", 1)[-1].strip()
    m = re.match(r"^(.*?)[\s_\-]+\d+$", label)
    return (m.group(1) if m else label).strip() or label

imgs = [(k, v) for k, v in inputs.items() if isinstance(v, dict) and "image" in v]
if not imgs:
    raise SystemExit("No image inputs — wire your fluorescence panels into this node.")

rows = []
for key, src in imgs:
    arr = src["image"].astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w = arr.shape[:2]
    # Central 80% ROI — drop edges to suppress vignetting.
    roi = arr[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]
    label = src.get("label", key).rsplit("/", 1)[-1]
    grp = infer_group(label)
    for ch_idx, ch_name in [(0, "R"), (1, "G"), (2, "B")]:
        rows.append({
            "source": label,
            "group": grp,
            "channel": ch_name,
            "mean_intensity": float(roi[..., ch_idx].mean()),
            "max_intensity": float(roi[..., ch_idx].max()),
        })
print(f"computed channel means for {len(imgs)} image(s); groups = "
      + ", ".join(sorted({r['group'] for r in rows})))
mpfig_data(rows, name="channel_intensities")
`;

const INTENSITY_PLOT_R = `# @name: Plot intensities per channel
# Faceted bar plot — one facet per channel, bars = per-group mean
# intensity, jittered raw points overlaid.  Requires the "channel"
# column from the upstream Python step.
library(ggplot2); library(ggprism)

data <- inputs[[1]]
data$channel <- factor(data$channel, levels = c("R", "G", "B"))
data$group   <- factor(data$group, levels = unique(data$group))

summ <- aggregate(mean_intensity ~ group + channel, data = data,
                  FUN = function(v) c(mean = mean(v),
                                      sem  = if (length(v) > 1) sd(v)/sqrt(length(v)) else 0))
summ <- do.call(data.frame, summ)
names(summ) <- c("group", "channel", "mean_int", "sem_int")

mpfig_plot("channel_intensity.png", width = 1700, height = 900, res = 300)
p <- ggplot() +
  geom_col(data = summ, aes(x = group, y = mean_int, fill = channel),
           width = 0.65, color = "grey25", linewidth = 0.3) +
  geom_errorbar(data = summ,
                aes(x = group, ymin = mean_int - sem_int, ymax = mean_int + sem_int),
                width = 0.22, color = "grey25", linewidth = 0.4) +
  geom_jitter(data = data,
              aes(x = group, y = mean_intensity),
              width = 0.13, height = 0, size = 1.4, alpha = 0.7,
              color = "grey20", stroke = 0) +
  scale_fill_manual(values = c("R" = "#d35454", "G" = "#5fa566", "B" = "#5d80c0"),
                    name = NULL, guide = "none") +
  facet_wrap(~ channel, nrow = 1) +
  theme_prism(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
        strip.text  = element_text(size = 11, face = "bold"),
        plot.margin = margin(10, 10, 6, 10)) +
  labs(x = NULL, y = "Mean ROI intensity (0–255)")
print(p)
`;

function buildIntensityWorkflow(): SavedWorkflow {
  const srcId = "source", pyId = "intensity_py", rId = "intensity_plot";
  const nodes: Node<NodeData>[] = [
    {
      id: srcId, type: "source", position: { x: 40, y: 60 },
      data: { label: "Source — label as 'Group_replicate'", kind: "source", sources: [], status: "ok" } as NodeData,
      draggable: true, deletable: false,
    },
    {
      id: pyId, type: "python", position: { x: 360, y: 60 },
      data: { label: "Channel intensities", kind: "python", code: INTENSITY_PYTHON,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: rId, type: "r", position: { x: 680, y: 60 },
      data: { label: "Plot intensities", kind: "r", code: INTENSITY_PLOT_R,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
  ];
  const mkEdge = (s: string, sh: string, t: string, th: string, k: DataKind): Edge => ({
    id: `e_${s}_${sh}__${t}_${th}`, source: s, sourceHandle: sh,
    target: t, targetHandle: th, type: "default", animated: false,
    style: { stroke: PORT_COLOR[k], strokeWidth: 2 },
  });
  return {
    id: "builtin:intensity_per_channel",
    name: "Intensity per channel (fluorescence R / G / B)",
    nodes,
    edges: [mkEdge(pyId, "out_table", rId, "in_table", "table")],
    createdAt: 0,
  };
}

// ── Western blot quantification ──────────────────────────────
// 3-node pipeline: per-band integrated density (background-
// subtracted) → normalise to a control band → bar plot.  Source
// labels: `Sample_1`, `Sample_2`, …  First label = loading
// control / reference; everything else is normalised against it.
const WB_DENSITY_PY = `# @name: Band density (integrated, bg-subtracted)
# For each band image we invert (so dark protein bands become
# bright), subtract a per-image background floor (5th percentile
# of pixel intensity), then sum the remaining signal.  The
# integrated optical density (IOD) so produced is the canonical
# western-blot quantification metric.
import numpy as np, re

def clean_label(label):
    return label.rsplit("/", 1)[-1].strip()

def infer_group(label):
    m = re.match(r"^(.*?)[\s_\-]+\d+$", label)
    return (m.group(1) if m else label).strip() or label

imgs = [(k, v) for k, v in inputs.items() if isinstance(v, dict) and "image" in v]
if not imgs:
    raise SystemExit("No band images — wire each cropped band inset here in order (first = loading control / ref).")

rows = []
for i, (key, src) in enumerate(imgs):
    arr = src["image"].astype(np.float32)
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr
    inverted = 255.0 - gray            # bright bands on dark bg
    bg = float(np.percentile(inverted, 5))
    signal = np.clip(inverted - bg, 0, None)
    iod = float(signal.sum())
    label = clean_label(src.get("label", key))
    rows.append({
        "source": label,
        "group": infer_group(label),
        "is_reference": (i == 0),
        "iod": iod,
        "mean_signal": float(signal.mean()),
        "area_px": int(gray.size),
    })
print(f"quantified {len(rows)} band(s); reference = {rows[0]['source']!r}")
mpfig_data(rows, name="band_iod")
`;

const WB_NORMALIZE_PY = `# @name: Normalize to reference
# Divide every band's IOD by the reference band's IOD (the first
# input).  Output is "relative_density" — unitless ratio used in
# western-blot comparison plots.
table = None
for v in inputs.values():
    if isinstance(v, dict) and "table" in v:
        table = v["table"]; break
if not table:
    raise SystemExit("No upstream table — wire 'Band density' into the table input.")

ref = next((r for r in table if str(r.get("is_reference")).lower() == "true"), table[0])
ref_iod = float(ref["iod"]) or 1.0
print(f"reference = {ref.get('source')!r} (IOD = {ref_iod:.0f})")

rows = []
for r in table:
    rows.append({**r, "relative_density": float(r["iod"]) / ref_iod})
mpfig_data(rows, name="band_relative_density")
`;

const WB_PLOT_R = `# @name: Plot relative band density
library(ggplot2); library(ggprism)

data <- inputs[[1]]
data$is_reference <- as.logical(toupper(as.character(data$is_reference)))
data$source <- factor(data$source, levels = unique(data$source))

mpfig_plot("wb_relative_density.png", width = 1500, height = 1100, res = 300)
p <- ggplot(data, aes(x = source, y = relative_density, fill = is_reference)) +
  geom_col(width = 0.65, color = "grey25", linewidth = 0.3) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_fill_manual(values = c("TRUE" = "#cfd8dc", "FALSE" = "#4a7c59"),
                    name = NULL,
                    labels = c("TRUE" = "Reference (loading)", "FALSE" = "Sample")) +
  theme_prism(base_size = 11) +
  theme(legend.position = "top",
        axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
        plot.margin = margin(10, 10, 6, 10)) +
  labs(x = NULL, y = "Relative band density (× reference)")
print(p)
`;

function buildWesternBlotWorkflow(): SavedWorkflow {
  const srcId = "source", densId = "wb_density", normId = "wb_normalize", plotId = "wb_plot";
  const nodes: Node<NodeData>[] = [
    {
      id: srcId, type: "source", position: { x: 40, y: 60 },
      data: {
        label: "Source — drop each band in order (first = loading control)",
        kind: "source", sources: [], status: "ok",
      } as NodeData,
      draggable: true, deletable: false,
    },
    {
      id: densId, type: "python", position: { x: 360, y: 60 },
      data: { label: "Band density", kind: "python", code: WB_DENSITY_PY,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: normId, type: "python", position: { x: 620, y: 60 },
      data: { label: "Normalize to reference", kind: "python", code: WB_NORMALIZE_PY,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: plotId, type: "r", position: { x: 880, y: 60 },
      data: { label: "Plot relative density", kind: "r", code: WB_PLOT_R,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
  ];
  const mkEdge = (s: string, sh: string, t: string, th: string, k: DataKind): Edge => ({
    id: `e_${s}_${sh}__${t}_${th}`, source: s, sourceHandle: sh,
    target: t, targetHandle: th, type: "default", animated: false,
    style: { stroke: PORT_COLOR[k], strokeWidth: 2 },
  });
  return {
    id: "builtin:western_blot",
    name: "Western blot quantification (IOD → normalised → plot)",
    nodes,
    edges: [
      mkEdge(densId, "out_table", normId, "in_table", "table"),
      mkEdge(normId, "out_table", plotId, "in_table", "table"),
    ],
    createdAt: 0,
  };
}

// ── Cell characteristics (Cellpose → shape metrics → plot) ──
// 3-node pipeline that segments brightfield / fluorescence cell
// images with Cellpose 3 and then derives shape metrics from the
// labelled masks: per-cell area, perimeter, circularity (4πA/P²),
// equivalent diameter and per-image cell count.  Final R node
// emits a multi-panel comparison plot — one facet per metric,
// jittered points per cell + per-group mean ± SEM.
const CELLPOSE_CYTO_FOR_CELLCHAR = `{
  "model": "cpsam",
  "diameter": null,
  "channels": [0, 0],
  "flow_threshold": 0.4,
  "cellprob_threshold": 0.0,
  "min_size": 15,
  "use_gpu": false
}
// Drop your brightfield / phase / DAPI insets into the source
// node and wire them into this Cellpose node.  Use null diameter
// to let the network auto-estimate (recommended).  Model "cpsam"
// is the Cellpose 4 default and handles cyto + nuclei jointly.`;

const CELL_SHAPE_PY = `# @name: Cell shape metrics
# Take the labelled masks produced by Cellpose (one image per
# source) and emit a per-CELL row with area, perimeter,
# circularity, equivalent diameter and the source image label.
# Numpy + Pillow only — no scikit-image dependency.
import numpy as np, re

def clean(label):
    return label.rsplit("/", 1)[-1].strip()

def infer_group(label):
    m = re.match(r"^(.*?)[\s_\-]+\d+$", label)
    return (m.group(1) if m else label).strip() or label

def measure_label(mask, lab):
    """Return (area, perim) for one label in a 2D integer mask."""
    yx = np.argwhere(mask == lab)
    if yx.size == 0:
        return 0, 0
    area = int(yx.shape[0])
    # 4-connected boundary: a pixel is on the boundary if any of
    # its 4 neighbours is a different label (or off-image).
    h, w = mask.shape
    bool_mask = (mask == lab)
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = bool_mask
    edges = (
        (pad[1:-1, 1:-1] & ~pad[0:-2, 1:-1]) |
        (pad[1:-1, 1:-1] & ~pad[2:, 1:-1]) |
        (pad[1:-1, 1:-1] & ~pad[1:-1, 0:-2]) |
        (pad[1:-1, 1:-1] & ~pad[1:-1, 2:])
    )
    perim = int(edges.sum())
    return area, perim

imgs = [(k, v) for k, v in inputs.items() if isinstance(v, dict) and "image" in v]
mask_imgs = [(k, v) for k, v in imgs
             if (str(v.get("label", "")).endswith("_mask") or "mask" in str(k).lower())]
# If labels don't carry "_mask" (e.g. user wired a different node),
# fall back to ALL inputs.
if not mask_imgs:
    mask_imgs = imgs
if not mask_imgs:
    raise SystemExit("No mask inputs — wire Cellpose's *_mask outputs into this node's image input.")

rows = []
for key, src in mask_imgs:
    arr = src["image"]
    # Cellpose ships masks as palette-coloured RGB.  Reduce to a
    # single per-pixel label by hashing the RGB triplet — works
    # because the palette assigns a unique colour per cell.
    if arr.ndim == 3:
        flat = arr.astype(np.uint32)
        rgb_hash = (flat[..., 0] << 16) | (flat[..., 1] << 8) | flat[..., 2]
        labels, inv = np.unique(rgb_hash, return_inverse=True)
        labelled = inv.reshape(arr.shape[:2])
    else:
        labelled = arr.astype(np.int32)
        labels = np.unique(labelled)
    # The most common label is the background.
    bg = int(np.bincount(labelled.ravel()).argmax())
    img_label = clean(src.get("label", key)).replace("_mask", "").replace("_outlines", "")
    group = infer_group(img_label)
    cell_id = 0
    for lab in np.unique(labelled):
        if int(lab) == bg:
            continue
        area, perim = measure_label(labelled, int(lab))
        if area < 20:  # discard sub-threshold blobs
            continue
        circularity = (4.0 * np.pi * area / max(perim * perim, 1)) if perim > 0 else 0.0
        eqdiam = float(2.0 * np.sqrt(area / np.pi))
        cell_id += 1
        rows.append({
            "source": img_label,
            "group": group,
            "cell_id": cell_id,
            "area_px": int(area),
            "perimeter_px": int(perim),
            "circularity": float(min(circularity, 1.0)),  # clamp to [0,1]
            "eq_diameter_px": eqdiam,
        })
print(f"measured {len(rows)} cell(s) across {len(mask_imgs)} image(s)")
mpfig_data(rows, name="cell_metrics")
`;

const CELL_SHAPE_PLOT_R = `# @name: Plot cell characteristics
# Multi-panel comparison plot — one panel per shape metric.  Each
# panel shows:
#   • Bar of group mean (filled by group)
#   • Error bar = ±1 SD
#   • Scatter overlay (each point = one cell, slight jitter)
#   • Significance brackets between groups (Wilcoxon rank-sum,
#     pairwise; only annotated when p < 0.05).
# Metrics rendered when the upstream column is present:
#   area_px / perimeter_px / circularity / eq_diameter_px /
#   feret_px / min_feret_px / aspect_ratio / roundness / solidity.
# Plus a per-image cell-count bar.

library(ggplot2); library(ggprism); library(dplyr)

# Defensive: if the upstream Python "Cell shape metrics" node failed
# (e.g. Cellpose couldn't load) we'd otherwise crash with a cryptic
# "subscript out of bounds" on inputs[[1]].  Surface a clear message
# and produce a placeholder plot so the run reports something useful.
if (length(inputs) == 0 || is.null(inputs[[1]]) || nrow(inputs[[1]]) == 0) {
  message("No table inputs reached this node — did the Cellpose / shape-metrics node upstream succeed?")
  mpfig_plot("cell_characteristics.png", width = 900, height = 600, res = 150)
  ggplot() + theme_void() +
    annotate("text", x = 0.5, y = 0.55, size = 6,
             label = "No cell-shape table reached this node.") +
    annotate("text", x = 0.5, y = 0.40, size = 4, color = "grey40",
             label = "Wire an upstream Cellpose → 'Cell shape metrics' chain into this R node.") +
    xlim(0, 1) + ylim(0, 1)
} else {

data <- inputs[[1]]

# Coerce ALL known numeric metric columns — ImageJ's mpfig_data emits
# everything as strings via concat, and read.csv may or may not
# auto-promote depending on locale.  Force numeric so geom_bar /
# error bars work.
num_cols <- c("area_px", "perimeter_px", "circularity", "eq_diameter_px",
              "feret_px", "min_feret_px", "aspect_ratio", "roundness",
              "solidity", "cell_id")
for (cn in num_cols) {
  if (cn %in% names(data)) data[[cn]] <- suppressWarnings(as.numeric(data[[cn]]))
}
# Drop rows where the primary metric (area) failed to parse.
if ("area_px" %in% names(data)) data <- data[!is.na(data$area_px), , drop = FALSE]

if (nrow(data) == 0) {
  message("All rows were dropped after numeric coercion — check that the upstream node emitted a well-formed metrics CSV.")
  mpfig_plot("cell_characteristics.png", width = 900, height = 600, res = 150)
  print(ggplot() + theme_void() +
    annotate("text", x = 0.5, y = 0.5, size = 6,
             label = "Metrics table parsed empty.") +
    xlim(0, 1) + ylim(0, 1))
} else {

# group/source columns may be missing for hand-crafted upstream code.
if (!"group"  %in% names(data)) data$group  <- "all"
if (!"source" %in% names(data)) data$source <- "input"

data$group  <- factor(data$group,  levels = unique(data$group))
data$source <- factor(data$source, levels = unique(data$source))
group_levels <- levels(data$group)

# Per-image counts.
counts <- aggregate(rep(1, nrow(data)) ~ source + group, data = data, FUN = sum)
names(counts)[3] <- "n_cells"

# Pre-compute every pairwise Wilcoxon p-value per metric.  We use
# Wilcoxon (Mann-Whitney) rather than t-test because cell-shape
# metrics are often non-normal and we may have small N.  Annotate
# brackets only when p < 0.05; collapse non-significant comparisons
# to keep the plot readable.
sig_label <- function(p) {
  if (!is.finite(p)) return("")
  if (p < 0.0001) return("****")
  if (p < 0.001)  return("***")
  if (p < 0.01)   return("**")
  if (p < 0.05)   return("*")
  return("ns")
}
make_pvals <- function(yvar) {
  vals <- data[[yvar]]
  ok <- is.finite(vals)
  if (sum(ok) < 4) return(NULL)
  d2 <- data.frame(y = vals[ok], group = data$group[ok])
  pairs <- combn(group_levels, 2, simplify = FALSE)
  rows <- list()
  for (pp in pairs) {
    g1 <- d2$y[d2$group == pp[1]]; g2 <- d2$y[d2$group == pp[2]]
    if (length(g1) < 2 || length(g2) < 2) next
    pv <- tryCatch(suppressWarnings(wilcox.test(g1, g2)$p.value), error = function(e) NA)
    rows[[length(rows) + 1]] <- data.frame(g1 = pp[1], g2 = pp[2], p = pv, label = sig_label(pv))
  }
  if (length(rows) == 0) return(NULL)
  do.call(rbind, rows)
}

# Brackets are drawn manually so we don't require ggsignif/ggpubr.
# Each significant comparison gets a horizontal line + label above
# the data range, stacked vertically when there are >1.
add_brackets <- function(g, yvar, pvals) {
  if (is.null(pvals)) return(g)
  sig <- pvals[!is.na(pvals$p) & pvals$p < 0.05, , drop = FALSE]
  if (nrow(sig) == 0) return(g)
  ymax <- suppressWarnings(max(data[[yvar]], na.rm = TRUE))
  if (!is.finite(ymax)) return(g)
  step <- 0.08 * (ymax - suppressWarnings(min(data[[yvar]], na.rm = TRUE)))
  if (!is.finite(step) || step <= 0) step <- 0.05 * abs(ymax) + 1
  group_idx <- setNames(seq_along(group_levels), group_levels)
  for (i in seq_len(nrow(sig))) {
    x1 <- group_idx[[sig$g1[i]]]; x2 <- group_idx[[sig$g2[i]]]
    y  <- ymax + step * (i + 0.4)
    g <- g + annotate("segment", x = x1, xend = x2, y = y, yend = y, linewidth = 0.4) +
             annotate("segment", x = x1, xend = x1, y = y, yend = y - step * 0.25, linewidth = 0.4) +
             annotate("segment", x = x2, xend = x2, y = y, yend = y - step * 0.25, linewidth = 0.4) +
             annotate("text", x = (x1 + x2) / 2, y = y + step * 0.15, label = sig$label[i], size = 3.2)
  }
  # Expand y-axis to fit the topmost bracket.
  g + scale_y_continuous(expand = expansion(mult = c(0.05, 0.18 + 0.10 * nrow(sig))))
}

mpfig_plot("cell_characteristics.png", width = 2400, height = 1800, res = 300)
common_theme <- theme_prism(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
        plot.title  = element_text(size = 10, face = "bold", hjust = 0.5),
        legend.position = "none",
        plot.margin = margin(6, 8, 6, 8))

# Bar + scatter + SD error-bar + significance for one metric.
# Returns NULL when the column is missing or empty so the layout
# step can skip it.
metric_plot <- function(yvar, ylab, ytitle = ylab) {
  if (!(yvar %in% names(data))) return(NULL)
  vals <- data[[yvar]]
  if (all(is.na(vals))) return(NULL)
  pvals <- make_pvals(yvar)
  g <- ggplot(data, aes(x = group, y = .data[[yvar]], fill = group)) +
    stat_summary(fun = mean, geom = "bar", width = 0.65,
                 color = "grey25", linewidth = 0.3, alpha = 0.85) +
    stat_summary(fun.data = function(v) {
      m <- mean(v, na.rm = TRUE); s <- sd(v, na.rm = TRUE)
      data.frame(y = m, ymin = m - s, ymax = m + s)
    }, geom = "errorbar", width = 0.25, linewidth = 0.4, color = "grey20") +
    geom_jitter(width = 0.16, size = 0.9, alpha = 0.55,
                color = "grey20", show.legend = FALSE) +
    labs(x = NULL, y = ylab, title = ytitle) +
    common_theme
  add_brackets(g, yvar, pvals)
}

# All metric panels — generated only if the column exists & has data.
p_list <- list(
  metric_plot("area_px",         "Area (px)",          "Area"),
  metric_plot("perimeter_px",    "Perimeter (px)",     "Perimeter"),
  metric_plot("circularity",     "Circularity",        "Circularity (4πA/P²)"),
  metric_plot("eq_diameter_px",  "Eq. diameter (px)",  "Equivalent Ø"),
  metric_plot("feret_px",        "Feret (px)",         "Feret Ø (max)"),
  metric_plot("min_feret_px",    "MinFeret (px)",      "MinFeret Ø"),
  metric_plot("aspect_ratio",    "Aspect ratio",       "Aspect Ratio"),
  metric_plot("roundness",       "Roundness",          "Roundness"),
  metric_plot("solidity",        "Solidity",           "Solidity")
)

# Per-image cell count.
p_count <- ggplot(counts, aes(x = source, y = n_cells, fill = group)) +
  geom_col(width = 0.7, color = "grey25", linewidth = 0.3) +
  labs(x = NULL, y = "Cells per image", title = "Cell count per image") +
  common_theme +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 7))

# Filter out NULL panels, append the count chart at the end.
panels <- Filter(Negate(is.null), p_list)
panels <- c(panels, list(p_count))

# Lay out in a 3-column grid via patchwork if available, else
# gridExtra, else just the first metric.
ok <- FALSE
if (length(panels) >= 1) {
  tryCatch({
    if (requireNamespace("patchwork", quietly = TRUE)) {
      pp <- patchwork::wrap_plots(panels, ncol = 3) +
            patchwork::plot_annotation(title = "Cell shape characteristics",
                                       theme = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 13)))
      print(pp); ok <- TRUE
    }
  }, error = function(e) message("patchwork layout failed: ", conditionMessage(e)))
  if (!ok) {
    tryCatch({
      if (requireNamespace("gridExtra", quietly = TRUE)) {
        do.call(gridExtra::grid.arrange, c(panels, list(ncol = 3, top = "Cell shape characteristics")))
        ok <- TRUE
      }
    }, error = function(e) message("gridExtra layout failed: ", conditionMessage(e)))
  }
  if (!ok) print(panels[[1]])
}

}  # end else (numeric-rows-survived branch)
}  # end else (have-inputs branch)
`;

function buildCellCharacteristicsWorkflow(): SavedWorkflow {
  const srcId = "source", cpId = "cell_cellpose", shapeId = "cell_shape", plotId = "cell_plot";
  const nodes: Node<NodeData>[] = [
    {
      id: srcId, type: "source", position: { x: 40, y: 60 },
      data: {
        label: "Source — drop brightfield / DAPI images",
        kind: "source", sources: [], status: "ok",
      } as NodeData,
      draggable: true, deletable: false,
    },
    {
      id: cpId, type: "cellpose", position: { x: 360, y: 60 },
      data: { label: "Cellpose (cpsam)", kind: "cellpose", code: CELLPOSE_CYTO_FOR_CELLCHAR,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      // Switched from Python → ImageJ.  Cellpose emits a *_labels
      // grayscale image (pixel value = cell ID) which ImageJ can
      // iterate over directly using Set Measurements + Create
      // Selection + Measure — the canonical, battle-tested route
      // for cell shape morphometry.  The Python harness rolled its
      // own perimeter/circularity logic which was less robust.
      id: shapeId, type: "imagej", position: { x: 640, y: 60 },
      data: { label: "Cell shape metrics (ImageJ)", kind: "imagej",
              code: IJ_CELL_SHAPE_METRICS,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
    {
      id: plotId, type: "r", position: { x: 920, y: 60 },
      data: { label: "Plot cell characteristics", kind: "r", code: CELL_SHAPE_PLOT_R,
              outputs: [], inputs: [], status: "idle", currentPreset: "custom" } as NodeData,
    },
  ];
  const mkEdge = (s: string, sh: string, t: string, th: string, k: DataKind): Edge => ({
    id: `e_${s}_${sh}__${t}_${th}`, source: s, sourceHandle: sh,
    target: t, targetHandle: th, type: "default", animated: false,
    style: { stroke: PORT_COLOR[k], strokeWidth: 2 },
  });
  return {
    id: "builtin:cell_characteristics",
    name: "Cell characteristics (Cellpose → ImageJ → plot)",
    nodes,
    edges: [
      // Cellpose emits multiple per-image outputs (out_image_0
      // for the first input's mask, out_image_1 for labels, etc.
      // — actual handle indices depend on the order images_out is
      // built in the sidecar).  The ImageJ node filters on
      // "_labels" suffix in its label/key, so wiring "out_image"
      // (a kind-channel handle that aggregates all image outputs)
      // lets the macro pick out just the label images.
      mkEdge(cpId,    "out_image", shapeId, "in_image", "image"),
      mkEdge(shapeId, "out_table", plotId,  "in_table", "table"),
    ],
    createdAt: 0,
  };
}

const BUILTIN_WORKFLOWS: SavedWorkflow[] = [
  buildCornealHazeWorkflow(),
  buildIntensityWorkflow(),
  buildWesternBlotWorkflow(),
  buildCellCharacteristicsWorkflow(),
];

// ── Workflow tabs (in-session) ──────────────────────────────
// Each tab is an independent graph (nodes + edges). The user can
// add multiple to run the same analysis on different image sets.
interface WorkflowTab {
  id: string;
  name: string;
  nodes: Node<NodeData>[];
  edges: Edge[];
}

// ── In-session workflow persistence ─────────────────────────
// We persist the currently-open workflow tabs (including each
// node's code, label, attached sources, currentPreset and position)
// to localStorage on every change so closing and reopening the
// Analysis dialog doesn't blow away in-flight work.  Output
// payloads (PNG / CSV base64) are stripped before save — they're
// large, regenerable, and saving them blows past localStorage's
// ~5 MB per-origin quota.  Status is reset to "stale" so users see
// the cached payloads are no longer authoritative.
const SESSION_KEY = "mpfig.workflow_session";
interface WorkflowSession {
  tabs: WorkflowTab[];
  activeId: string;
}
function sanitizeNodeForSave(n: Node<NodeData>): Node<NodeData> {
  // Drop the huge `outputs[].payload` blobs; cap `consoleOut`.
  const co = n.data.consoleOut;
  const trimmedOut = (co && co.length > 4000) ? co.slice(0, 4000) + "\n…(truncated)" : co;
  return {
    ...n,
    data: {
      ...n.data,
      outputs: [],
      consoleOut: trimmedOut,
      status: n.data.status === "running" ? "stale" : (n.data.status === "ok" ? "stale" : n.data.status),
    } as NodeData,
  };
}
function loadWorkflowSession(): WorkflowSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<WorkflowSession>;
    if (!parsed || !Array.isArray(parsed.tabs) || parsed.tabs.length === 0) return null;
    // Light shape validation.
    const tabs = parsed.tabs.filter((t) =>
      t && typeof t.id === "string" && typeof t.name === "string" &&
      Array.isArray(t.nodes) && Array.isArray(t.edges),
    ) as WorkflowTab[];
    if (tabs.length === 0) return null;
    return { tabs, activeId: typeof parsed.activeId === "string" ? parsed.activeId : tabs[0].id };
  } catch { return null; }
}
function saveWorkflowSession(s: WorkflowSession) {
  try {
    const stripped: WorkflowSession = {
      activeId: s.activeId,
      tabs: s.tabs.map((t) => ({ ...t, nodes: t.nodes.map(sanitizeNodeForSave) })),
    };
    localStorage.setItem(SESSION_KEY, JSON.stringify(stripped));
  } catch { /* localStorage quota / disabled — give up silently */ }
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
  //
  // Hydrate from the in-session save (mpfig.workflow_session) so
  // closing and reopening the Analysis dialog doesn't blow away
  // the user's open tabs, attached insets, and code edits.  Falls
  // back to a single empty workflow when there's nothing saved.
  const [workflowTabs, setWorkflowTabs] = useState<WorkflowTab[]>(() => {
    const restored = loadWorkflowSession();
    if (restored && restored.tabs.length > 0) return restored.tabs;
    return [{
      id: newId("wf"),
      name: "Workflow 1",
      nodes: [newSourceNode([])],
      edges: [],
    }];
  });
  const [activeWfId, setActiveWfId] = useState<string>(() => {
    const restored = loadWorkflowSession();
    return restored?.activeId || "";
  });
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

  // Persist the open tabs + active id to localStorage on every
  // change so closing and reopening the Analysis dialog (or even
  // reloading the whole app) restores in-flight work.  Debounced
  // so a flurry of node-position drag updates doesn't hammer
  // JSON.stringify on a multi-megabyte tabs array.
  const sessionSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (sessionSaveTimerRef.current) clearTimeout(sessionSaveTimerRef.current);
    sessionSaveTimerRef.current = setTimeout(() => {
      saveWorkflowSession({ tabs: workflowTabs, activeId: activeWfId });
    }, 400);
    return () => {
      if (sessionSaveTimerRef.current) clearTimeout(sessionSaveTimerRef.current);
    };
  }, [workflowTabs, activeWfId]);

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

  // Monotonic counter bumped whenever a source node's handle set
  // changes (drop / detach an inset).  Used as part of the
  // <ReactFlow key=...> so the entire RF tree remounts and any
  // stale handle-cache is discarded — see the long comment on
  // the <ReactFlow> element below.
  const [handleRev, setHandleRev] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  // Set of currently-selected node ids + edge ids — populated by
  // React Flow's onSelectionChange.  Drives the visible "Delete
  // selected" affordance in the canvas toolbar (the only reliable
  // way to discover the delete action — Backspace works but isn't
  // a discoverable handle).
  const [rfSelection, setRfSelection] = useState<{ nodeIds: Set<string>; edgeIds: Set<string> }>(() => ({ nodeIds: new Set(), edgeIds: new Set() }));
  const [runningGraph, setRunningGraph] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [drawerTab, setDrawerTab] = useState<DataKind>("plot");
  const consoleRef = useRef<string>("");
  const [consoleOut, setConsoleOut] = useState("");
  // Togglable Console panel — collapsed by default so it doesn't
  // cover the canvas; auto-opens during long-running ops (Install
  // Cellpose, etc.) by setting setConsoleOpen(true) in those code
  // paths.  Click the right-edge "Console" handle to flip it.
  const [consoleOpen, setConsoleOpen] = useState(false);
  // Imperative handle to NodeInternalsRefresher (lives inside the
  // <ReactFlow> tree, where the useUpdateNodeInternals hook is
  // reachable).  We call rfInternalsRef.current?.refresh([id]) from
  // anywhere a node's handle set changes (drop an inset into a
  // source node, drag-start an edge connection) so RF doesn't keep
  // serving stale handle maps that block the very next connection.
  const rfInternalsRef = useRef<RFInternalsHandle | null>(null);

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
    cellpose: loadUserPresets("cellpose"),
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
  // Cellpose availability — `kind` carries "cellpose" + version if
  // the import succeeds in the sidecar's Python env, empty string
  // otherwise.  The Add-Cellpose-node button reads this.
  const [cellposeKind, setCellposeKind] = useState<string>("");
  const [cellposeInstalling, setCellposeInstalling] = useState(false);
  // Latches the last install's exit code so the button can surface
  // "Install failed — see console" instead of silently reverting to
  // the idle label.  Cleared when the user clicks Retry.
  const [cellposeInstallFailed, setCellposeInstallFailed] = useState(false);

  // (Drawer Console tab + ad-hoc snippet runner removed — per-node
  // Console under the code editor is the canonical place to read
  // stdout / stderr.  The state declarations went with it.)

  // Custom engine binary paths (Python / R / MATLAB / ImageJ).
  // Hydrated from localStorage on mount; the Settings dialog edits
  // them.  Each path travels along with the engine's API call so
  // the sidecar uses the user-pinned binary instead of its own
  // auto-detected default.
  const [enginePaths, setEnginePaths] = useState<EnginePaths>(() => loadEnginePaths());
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Measurements present? Falsy CSV (empty / null) hides the
  // measurements port on every source node + any related hints.
  const hasMeasurements = !!(measurementsCsv && measurementsCsv.trim().length > 0);

  // ── Per-source name overrides ────────────────────────────────
  // `sourceNameOverrides[key]` is the user's chosen display name for
  // that inset.  Falls back to the backend's auto-generated label
  // ("R1C1·1 (Standard Zoom)") when absent.
  const [sourceNameOverrides, setSourceNameOverrides] = useState<Record<string, string>>(() => loadSourceNames());
  const persistSourceName = useCallback((key: string, name: string | null) => {
    setSourceNameOverrides((cur) => {
      const next = { ...cur };
      const trimmed = (name || "").trim();
      if (trimmed) next[key] = trimmed;
      else delete next[key];
      saveSourceNames(next);
      return next;
    });
  }, []);
  const renameSourceHandler = useCallback(async (s: InsetSource) => {
    const currentName = displayName(s, sourceNameOverrides);
    const newName = await prompt({
      title: "Rename source",
      label: "Display name (used as the source label in plots)",
      defaultValue: currentName,
      okLabel: "Rename",
    });
    if (newName === null) return;
    persistSourceName(s.key, newName);
  }, [sourceNameOverrides, persistSourceName, prompt]);

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
    // Probe MATLAB / ImageJ with the user's pinned binary (if any).
    // The sidecar accepts a `?path=` query — empty falls back to
    // its auto-detection.
    const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
    const mlPath = enginePaths.matlab;
    const ijPath = enginePaths.imagej;
    fetch(`${apiBase}/api/analysis/check-matlab${mlPath ? `?path=${encodeURIComponent(mlPath)}` : ""}`)
      .then((r) => r.json())
      .then((m: { installed?: boolean; kind?: string }) => setMatlabKind(m?.installed ? (m.kind || "") : ""))
      .catch(() => setMatlabKind(""));
    fetch(`${apiBase}/api/analysis/check-imagej${ijPath ? `?path=${encodeURIComponent(ijPath)}` : ""}`)
      .then((r) => r.json())
      .then((m: { installed?: boolean; kind?: string }) => setImagejKind(m?.installed ? (m.kind || "imagej") : ""))
      .catch(() => setImagejKind(""));
    fetch(`${apiBase}/api/analysis/check-cellpose`)
      .then((r) => r.json())
      .then((m: { installed?: boolean; kind?: string }) => setCellposeKind(m?.installed ? (m.kind || "cellpose") : ""))
      .catch(() => setCellposeKind(""));
  }, [open, enginePaths]);

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
    const preset = BUILTIN_PRESETS[engine][presetIdx];
    const defaultCode = preset?.code
      ?? (engine === "python" ? PY_DEFAULT
        : engine === "matlab" ? ML_DEFAULT
        : engine === "imagej" ? IJ_DEFAULT
        : engine === "cellpose" ? CELLPOSE_DEFAULT
        : R_DEFAULT);
    const engineLabel = engine === "python" ? "Python" : engine === "matlab" ? "MATLAB" : engine === "imagej" ? "ImageJ" : engine === "cellpose" ? "Cellpose" : "R Plot";
    // Use the preset's name as the node label so the chip carries
    // the intent ("Haze analysis") rather than a generic engine label.
    // Fall back to the engine label when the preset is "Custom (starter)".
    const initialLabel = preset && !/^Custom/i.test(preset.name) ? preset.name : engineLabel;
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
          label: initialLabel,
          kind: engine,
          code: defaultCode,
          outputs: [],
          inputs: [],
          status: "idle",
          currentPreset: `b:${presetIdx}`,
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
    // Adding an inset spawns a new output handle on this source
    // node (`out_image_<idx>`).  RF v12's internal handle-cache
    // can serve stale data even after useUpdateNodeInternals — the
    // ONLY 100%-reliable fix we've found is to bump `handleRev`,
    // which is part of the <ReactFlow key=…> and forces a full
    // RF-tree remount with a clean store.  Heavy hammer, but it
    // definitively kills the "have to exit + re-enter analysis to
    // connect source → first node" bug.
    setHandleRev((r) => r + 1);
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
    // Detaching an inset shrinks the handle set — same staleness
    // hazard as addSourceToNode.  Bump handleRev to remount the RF
    // tree with the now-smaller handle map (and discard any cached
    // edges that referenced the dropped handle).
    setHandleRev((r) => r + 1);
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
    // Built-in templates live in a separate constant list — same
    // shape as user saves but indexed by a "builtin:<key>" id so
    // we don't accidentally write them into the user's localStorage.
    const wf = id.startsWith("builtin:")
      ? BUILTIN_WORKFLOWS.find((w) => w.id === id)
      : savedWorkflows.find((w) => w.id === id);
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
          // Pass the user's renamed label downstream (falls back to
          // the backend-supplied default).  Lets the corneal-haze
          // group inference key off "Control_1" instead of "R1C1·1".
          result.push({ key, kind: "image", label: displayName(src, sourceNameOverrides), image_b64: src.thumbnail });
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
  }, [edges, measurementsCsv, sourceNameOverrides]);

  const runNode = useCallback(async (node: Node<NodeData>, nodeMap: Map<string, Node<NodeData>>) => {
    const engine = node.data.kind as EngineKind;
    if (engine !== "python" && engine !== "matlab" && engine !== "r" && engine !== "imagej" && engine !== "cellpose") return;
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
      // All four engines now go through a direct fetch so we can
      // include the user's optional `interpreter_path` in the body
      // (the api.* wrappers don't expose it).  The sidecar treats
      // an empty/missing path as "auto-detect like before".
      const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
      const buildSources = () => extra
        .filter((x) => x.kind === "image" && x.key.startsWith("inset_"))
        .map((x) => {
          const insetKey = x.key.replace(/^inset_\d+_/, "");
          const src = insetSources.find((s) => s.key === insetKey);
          if (!src) return null;
          return { key: insetKey, row: src.row, col: src.col, inset_index: src.inset_index, label: displayName(src, sourceNameOverrides) };
        })
        .filter((s): s is { key: string; row: number; col: number; inset_index: number; label: string } => !!s);

      if (engine === "python") {
        const sources = buildSources();
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        const resp = await fetch(`${apiBase}/api/analysis/run-python`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            code: node.data.code || PY_DEFAULT,
            sources, extra_inputs: extras, timeout_sec: 60,
            interpreter_path: enginePaths.python || undefined,
          }),
        });
        result = await resp.json();
      } else if (engine === "cellpose") {
        // Cellpose module — node body is a JSON config + free-text
        // comments; the sidecar's run-cellpose handler strips
        // comments and feeds the parsed config to the cellpose
        // model.  Image inputs come through the same extras / source
        // path as Python.
        const sources = buildSources();
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        try {
          const resp = await fetch(`${apiBase}/api/analysis/run-cellpose`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              config: node.data.code || CELLPOSE_DEFAULT,
              sources, extra_inputs: extras, timeout_sec: 300,
            }),
          });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          result = await resp.json();
        } catch (err) {
          result = {
            success: false, stdout: "",
            stderr: err instanceof Error ? err.message
              : "Cellpose not available. Install via `pip install cellpose` in the sidecar's Python env.",
            plots: [], tables: [], images: [],
          };
        }
      } else if (engine === "imagej") {
        const sources = buildSources();
        // Pipe upstream node outputs (Cellpose masks, Python images,
        // etc.) into ImageJ as `extra_inputs`.  Without this, the
        // ImageJ node could only consume source insets — defeating
        // the whole purpose of putting it downstream of Cellpose for
        // shape metrics on the labelled output.
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        try {
          const resp = await fetch(`${apiBase}/api/analysis/run-imagej`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              code: node.data.code || IJ_DEFAULT,
              sources, extra_inputs: extras, timeout_sec: 120,
              interpreter_path: enginePaths.imagej || undefined,
            }),
          });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          result = await resp.json();
        } catch (err) {
          result = {
            success: false, stdout: "",
            stderr: err instanceof Error ? err.message
              : "ImageJ / Fiji not detected. Configure a path in Settings or install Fiji.",
            plots: [], tables: [], images: [],
          };
        }
      } else if (engine === "matlab") {
        const sources = buildSources();
        const extras = extra.filter((x) => !x.key.startsWith("inset_"));
        const resp = await fetch(`${apiBase}/api/analysis/run-matlab`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            code: node.data.code || ML_DEFAULT,
            sources, extra_inputs: extras, timeout_sec: 90,
            interpreter_path: enginePaths.matlab || undefined,
          }),
        });
        result = await resp.json();
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
        const resp = await fetch(`${apiBase}/api/analysis/run-r`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            code: fullCode,
            measurements_csv: "Panel,Name,Group,Value,Unit\n",
            interpreter_path: enginePaths.r || undefined,
          }),
        });
        const rr = await resp.json();
        result = { ...rr, images: rr.images || [] };
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
  }, [collectInputs, insetSources, enginePaths, setNodes, sourceNameOverrides]);

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
          renderValue={() => `📂 Load template / saved (${BUILTIN_WORKFLOWS.length}+${savedWorkflows.length})`}
          onChange={(e) => loadSavedWorkflow(String(e.target.value))}
          sx={{ fontSize: "0.6rem", height: 26, minWidth: 200, "& .MuiSelect-select": { py: 0.25, px: 1 } }}
        >
          {/* Built-in templates — shipped pipelines like "Corneal
              haze" that the user can drop in as a starting point. */}
          <MenuItem value="" disabled sx={{ fontSize: "0.7rem", fontWeight: 700 }}>Built-in templates</MenuItem>
          {BUILTIN_WORKFLOWS.map((wf) => (
            <MenuItem key={wf.id} value={wf.id} sx={{ fontSize: "0.7rem", pl: 2 }}>
              <Box component="span" sx={{ flex: 1 }}>{wf.name}</Box>
            </MenuItem>
          ))}
          <MenuItem value="" disabled sx={{ fontSize: "0.7rem", fontWeight: 700, mt: 0.5 }}>Your saved workflows</MenuItem>
          {savedWorkflows.length === 0 ? (
            <MenuItem value="" disabled sx={{ fontSize: "0.7rem", pl: 2, fontStyle: "italic", color: "text.disabled" }}>
              (none saved yet — click 💾 Save workflow)
            </MenuItem>
          ) : savedWorkflows.map((wf) => (
            <MenuItem key={wf.id} value={wf.id} sx={{ fontSize: "0.7rem", pl: 2, display: "flex", justifyContent: "space-between" }}>
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
        insetSources, sourceNameOverrides, renameSource: renameSourceHandler,
        hasMeasurements, highlightedUpstreamIds,
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
      <Box sx={{
        flex: 1, position: "relative", minWidth: 0, minHeight: 0,
        // Dark-theme the RF Controls buttons.  Keep RF's native SVG
        // icons (zoom-in, zoom-out, fit-view, lock) — they're set up
        // with `fill: currentColor` so setting `color` on the button
        // is enough to recolour them.  We only override the chrome
        // (background, separator lines) and shrink the buttons a bit
        // so the strip doesn't dominate the canvas corner.
        "& .react-flow__controls": {
          boxShadow: "none",
          border: "none",
        },
        "& .react-flow__controls-button": {
          backgroundColor: "transparent",
          color: "#cfd8dc",
          width: 22,
          height: 22,
          padding: 0,
          borderColor: "#2a3744",
          transition: "background-color 120ms, color 120ms",
        },
        "& .react-flow__controls-button svg": {
          width: 12,
          height: 12,
          fill: "currentColor",
        },
        "& .react-flow__controls-button:hover": {
          backgroundColor: "#2a3744",
          color: "#ffa726",
        },
      }}>
        <ReactFlow
          // FORCE REMOUNT on workflow change.  Despite multiple rounds
          // of useUpdateNodeInternals + position nudges, RF v12's
          // internal handle-map cache continued to serve stale data
          // after a template load, producing the "have to exit and
          // re-enter analysis to connect source→first-node" bug.
          // The cache lives on the <ReactFlow> instance itself; the
          // ONLY guaranteed way to wipe it without unmounting the
          // dialog is to remount the RF instance.  Keying it by the
          // active workflow id achieves exactly that — every template
          // load gets a fresh RF tree with a clean store.
          //
          // Also keying on `handleRev` (bumped in addSourceToNode /
          // removeSourceFromNode) so drops of new insets into a
          // source node also trigger a clean remount.  This is the
          // hammer that finally kills the bug.
          key={`rf-${activeWf.id}-${handleRev}`}
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          // Belt-and-braces: still refresh handle internals on
          // connect-start, in case any drag begins before the
          // remount has fully settled.
          onConnectStart={() => rfInternalsRef.current?.refresh()}
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
          // Hide the React Flow attribution badge in the bottom-right.
          // The link points to reactflow.dev which is blocked in the
          // preview (Tauri webview only allows localhost), producing
          // the "Link to reactflow.dev was blocked" error popup. We
          // honour the OSS spirit by keeping a one-liner credit in
          // our project README instead.
          proOptions={{ hideAttribution: true }}
          fitView
          fitViewOptions={{ maxZoom: 0.85, minZoom: 0.4, padding: 0.25 }}
          minZoom={0.15}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 0.7 }}
          deleteKeyCode={["Backspace", "Delete"]}
          edgesFocusable
          elementsSelectable
        >
          {/* Bring our own Background — RF default. */}
          <Background />
          {/* Controls — small horizontal strip pinned at top-right
              BELOW the MiniMap.  This keeps the "canvas dashboard"
              (zoom + overview) grouped in one corner.  The console
              handle sits at top:142px (down past this group), so
              the three elements stack cleanly: MiniMap → Controls
              → empty gap → Console.  Bottom-left would collide with
              the outputs drawer, so we avoid that corner. */}
          <Controls
            position="top-right"
            orientation="horizontal"
            showInteractive
            style={{
              // MiniMap is 110 px tall with margin: 8 → ends at 126.
              // Sit just below it (small 4 px gap looks deliberate).
              marginTop: 130,
              marginRight: 8,
              backgroundColor: "#1f2933",
              border: "1px solid #37474f",
              borderRadius: 6,
              boxShadow: "0 2px 6px rgba(0,0,0,0.35)",
              overflow: "hidden",
              zIndex: 9,
            }}
          />
          {/* MiniMap — dark theme to match the analysis canvas
              chrome.  Pinned top-right with z-index above the
              Controls + drawer so it's never occluded.  Mask shows
              the un-viewed region as a semi-transparent dark
              gradient; nodes punch through in their kind-colours. */}
          <MiniMap
            pannable
            zoomable
            position="top-right"
            style={{
              width: 168,
              height: 110,
              margin: 8,
              backgroundColor: "#1f2933",
              border: "1px solid #37474f",
              borderRadius: 6,
              boxShadow: "0 2px 6px rgba(0,0,0,0.35)",
              zIndex: 9,
            }}
            maskColor="rgba(8,12,16,0.55)"
            maskStrokeColor="#ffa726"
            maskStrokeWidth={1.5}
            nodeColor={(n) => KIND_BORDER[(n.data as NodeData | undefined)?.kind ?? "source"] || "#cfd8dc"}
            nodeStrokeColor="#e0e0e0"
            nodeStrokeWidth={0.5}
            nodeBorderRadius={2}
          />
          {/* Re-register every node's handle set whenever the
              active workflow tab changes (e.g. user clicked Load
              template).  Without this, freshly-mounted nodes
              that React Flow "remembers" by id from a previous
              tab keep their stale handle map and refuse drag-to-
              connect until the dialog is closed and reopened.
              The imperative ref lets us also force a refresh from
              `addSourceToNode` and `onConnectStart` (see above). */}
          <NodeInternalsRefresher
            ref={rfInternalsRef}
            activeWfId={activeWf.id}
            nodeIds={nodes.map((n) => n.id)}
          />
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
          {cellposeKind ? (
            <Tooltip placement="bottom" title={`Add a Cellpose node (detected: ${cellposeKind})`}>
              <span>
                <Button size="small" variant="contained" startIcon={<AddIcon sx={{ fontSize: 14 }} />}
                  onClick={() => addProcessNode("cellpose")}
                  sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25, bgcolor: KIND_COLOR.cellpose, "&:hover": { bgcolor: KIND_COLOR.cellpose, filter: "brightness(0.9)" } }}>
                  🧬 Cellpose
                </Button>
              </span>
            </Tooltip>
          ) : (
            <Tooltip placement="bottom" title={cellposeInstalling
              ? "Installing… (cellpose + torch download is ~500 MB, can take several minutes)"
              : cellposeInstallFailed
                ? "Previous install failed — see the Console panel for the pip error.  Click to retry."
                : "Install Cellpose 3 into the sidecar's Python (runs `pip install cellpose`)"}>
              <span>
                <Button size="small" variant={cellposeInstallFailed ? "contained" : "outlined"}
                  color={cellposeInstallFailed ? "error" : "primary"}
                  startIcon={
                    cellposeInstalling
                      ? <CircularProgress size={12} sx={{ color: "inherit" }} />
                      : <span style={{ fontSize: 12 }}>{cellposeInstallFailed ? "⚠" : "⤓"}</span>
                  }
                  disabled={cellposeInstalling}
                  onClick={async () => {
                    setCellposeInstalling(true);
                    setCellposeInstallFailed(false);
                    setConsoleOpen(true);  // surface the live log
                    consoleRef.current += `\n=== Install Cellpose 3 ===\n`;
                    setConsoleOut(consoleRef.current);
                    const apiBase = (import.meta as { env?: { VITE_API?: string } }).env?.VITE_API || "http://127.0.0.1:8765";
                    // The streaming endpoint emits a final `{"done":true,"returncode":N}`
                    // frame; we latch the exit code so the button can
                    // stay in an error state after the request completes.
                    let lastRc: number | null = null;
                    try {
                      const resp = await fetch(`${apiBase}/api/analysis/install-cellpose-stream`, {
                        method: "POST",
                        headers: { "Accept": "text/event-stream" },
                      });
                      if (!resp.ok || !resp.body) {
                        throw new Error(`HTTP ${resp.status}`);
                      }
                      const reader = resp.body.getReader();
                      const dec = new TextDecoder();
                      let buf = "";
                      // eslint-disable-next-line no-constant-condition
                      while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += dec.decode(value, { stream: true });
                        let nl: number;
                        while ((nl = buf.indexOf("\n\n")) !== -1) {
                          const frame = buf.slice(0, nl);
                          buf = buf.slice(nl + 2);
                          if (!frame.startsWith("data:")) continue;
                          try {
                            const payload = JSON.parse(frame.slice(frame.indexOf(":") + 1).trim());
                            if (payload.line) {
                              consoleRef.current += payload.line + "\n";
                              setConsoleOut(consoleRef.current);
                            }
                            if (payload.done) {
                              lastRc = typeof payload.returncode === "number" ? payload.returncode : -1;
                              consoleRef.current += `[done — rc=${lastRc}]\n`;
                              setConsoleOut(consoleRef.current);
                            }
                          } catch { /* ignore non-JSON */ }
                        }
                      }
                      // Re-probe after install (success or fail).
                      const probe = await fetch(`${apiBase}/api/analysis/check-cellpose`);
                      const pd: { installed?: boolean; kind?: string } = await probe.json();
                      setCellposeKind(pd?.installed ? (pd.kind || "cellpose") : "");
                      if (pd?.installed) {
                        consoleRef.current += `[Cellpose ready: ${pd.kind}]\n`;
                        setConsoleOut(consoleRef.current);
                        setCellposeInstallFailed(false);
                      } else {
                        // Latch the failure state until the user
                        // explicitly retries — keeps the cause
                        // visible instead of silently reverting.
                        setCellposeInstallFailed(true);
                        consoleRef.current += `\n[Install failed${lastRc !== null ? ` (pip exit ${lastRc})` : ""} — read the lines above for the actual error]\n`;
                        setConsoleOut(consoleRef.current);
                      }
                    } catch (e) {
                      consoleRef.current += `\n[Install request failed: ${e instanceof Error ? e.message : String(e)}]\n`;
                      setConsoleOut(consoleRef.current);
                      setCellposeInstallFailed(true);
                    } finally {
                      setCellposeInstalling(false);
                    }
                  }}
                  sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25,
                        ...(cellposeInstallFailed
                          ? {}
                          : { borderColor: KIND_BORDER.cellpose, color: KIND_COLOR.cellpose }) }}>
                  {cellposeInstalling
                    ? "Installing…"
                    : cellposeInstallFailed
                      ? "Install failed — Retry"
                      : "🧬 Install Cellpose 3"}
                </Button>
              </span>
            </Tooltip>
          )}
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
          <Tooltip placement="bottom" title="Custom paths for Python / R / MATLAB / ImageJ binaries">
            <Button size="small" variant="outlined"
              onClick={() => setSettingsOpen(true)}
              sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}>
              ⚙ Engines
            </Button>
          </Tooltip>
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
              saved presets stored per-engine in localStorage.  The
              Select is controlled by the node's currentPreset field
              so picking a preset surfaces the name in the dropdown
              (the value=""+renderValue trick of "Load preset…" was
              the older one-shot UX).  Code-editor changes flip the
              field to "custom" so the user sees it diverged. */}
          {(() => {
            const cp = selectedNode.data.currentPreset || `b:0`;
            const cpLabel: string = (() => {
              if (cp === "custom") return "Custom (edited)";
              if (cp.startsWith("b:")) return builtin[parseInt(cp.slice(2), 10)]?.name || "Built-in";
              if (cp.startsWith("u:")) return `★ ${userPresets[parseInt(cp.slice(2), 10)]?.name || "Saved"}`;
              return "Load preset…";
            })();
            return (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 1, py: 0.5, borderBottom: "1px solid", borderColor: "divider", flexWrap: "wrap" }}>
            <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary" }}>Preset:</Typography>
            <Select
              size="small"
              value={cp}
              displayEmpty
              renderValue={() => cpLabel}
              onChange={(e) => {
                const val = String(e.target.value);
                if (val === "custom") {
                  // Custom: don't touch code; just flip the marker.
                  setNodes((cur) => cur.map((n) => n.id === selectedNode.id
                    ? { ...n, data: { ...n.data, currentPreset: "custom" } } : n));
                  return;
                }
                let preset: CodePreset | undefined;
                if (val.startsWith("b:")) preset = builtin[parseInt(val.slice(2), 10)];
                else if (val.startsWith("u:")) preset = userPresets[parseInt(val.slice(2), 10)];
                if (preset) {
                  // Rename the node to the preset name unless it's
                  // the generic "Custom (starter)" — in which case
                  // keep the engine-default label so the chip stays
                  // meaningful.
                  const newLabel = /^Custom/i.test(preset.name)
                    ? (engine === "python" ? "Python" : engine === "matlab" ? "MATLAB" : engine === "imagej" ? "ImageJ" : engine === "cellpose" ? "Cellpose" : "R Plot")
                    : preset.name;
                  setNodes((cur) => cur.map((n) => n.id === selectedNode.id
                    ? { ...n, data: { ...n.data, code: preset!.code, label: newLabel, status: "idle" as const, currentPreset: val } } : n));
                }
              }}
              sx={{ fontSize: "0.65rem", height: 26, minWidth: 200, "& .MuiSelect-select": { py: 0.4, px: 1 } }}
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
              <MenuItem value="custom" sx={{ fontSize: "0.7rem", pl: 2, fontStyle: "italic", color: "text.secondary" }}>
                ◇ Custom (edited)
              </MenuItem>
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
          );
          })()}
          <Box sx={{ flex: 1, minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            {/* CodeMirror wrapper.
                Canonical "fill flex parent" recipe for CodeMirror 6:
                  • wrapper  → flex: 1, minHeight: 0, display: flex
                    column.  Takes remaining height in the properties
                    panel column, can shrink past content.
                  • .cm-editor → flex: 1, minHeight: 0, height: auto
                    (override the inline `height: 100%` that the
                    React wrapper sets, which would otherwise sample
                    the parent's height ONCE at layout and bake it
                    in — re-mounting on every panel resize).
                  • .cm-scroller → flex: 1, minHeight: 0, overflow:
                    auto.  This is the ACTUAL scroll container.
                The earlier `position:absolute` + `height:0` hack
                worked in Firefox but the Tauri WebKit view didn't
                propagate computed height down into CodeMirror's
                inner divs, leaving the editor stuck at 0 height
                and silently ignoring overflow.  Letting the flex
                cascade do the work fixes it across both engines. */}
            <Box sx={{
              flex: 1, minHeight: 0,
              display: "flex", flexDirection: "column",
              "& .cm-theme, & .cm-editor": {
                flex: 1, minHeight: 0,
                display: "flex", flexDirection: "column",
                height: "auto !important",
              },
              "& .cm-scroller": {
                flex: 1, minHeight: 0,
                overflow: "auto",
                fontFamily: "monospace",
              },
            }}>
              <CodeMirror
                value={selectedNode.data.code || ""}
                onChange={(v) => setNodes((cur) => cur.map((n) => {
                  if (n.id !== selectedNode.id) return n;
                  // Flip currentPreset → "custom" the moment the user
                  // edits the code so the dropdown stops claiming the
                  // node still matches a preset.  If the edit returns
                  // the buffer to exactly the preset's body, snap back.
                  const cp = n.data.currentPreset;
                  let nextPreset = cp;
                  if (cp && cp !== "custom") {
                    let presetCode = "";
                    if (cp.startsWith("b:")) presetCode = builtin[parseInt(cp.slice(2), 10)]?.code || "";
                    else if (cp.startsWith("u:")) presetCode = userPresets[parseInt(cp.slice(2), 10)]?.code || "";
                    if (presetCode && v !== presetCode) nextPreset = "custom";
                  }
                  return { ...n, data: { ...n.data, code: v, status: n.data.status === "ok" ? "stale" : n.data.status, currentPreset: nextPreset } };
                }))}
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
        {/* Output grid + staging column */}
        {true && (
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
      {/* Togglable Console panel — anchored to the RIGHT edge of the
          canvas so it doesn't collide with the bottom outputs
          drawer or the top-right MiniMap.  A small "Console" pill
          handle sticks out below the MiniMap when collapsed; click
          to slide the full log panel open.  Auto-opens during
          long-running ops (Install Cellpose etc.). */}
      <Box sx={{
        position: "absolute",
        right: 8,
        // Top-right corner stack: MiniMap (top 8 → ≈126), then
        // Controls strip (≈130 → ≈160).  The console handle starts
        // safely below both at 172, leaving a clean visual gap.
        top: 172,
        bottom: 8,        // grow to fill vertical space down to outputs drawer
        zIndex: 6,
        display: "flex", flexDirection: "row", alignItems: "stretch",
        pointerEvents: "none",
      }}>
        {/* Spine handle — vertical tab pinned flush to the right
            edge, matches the dark MiniMap chrome above it.  Glyph
            flips on hover; lines-count chip floats next to the
            label when there's output. */}
        <Box sx={{
          pointerEvents: "auto",
          display: "flex", flexDirection: "column",
          alignSelf: "flex-start",
        }}>
          <Tooltip
            placement="left"
            title={consoleOpen ? "Hide console" : "Show console output (graph runs, installs, dependency fetches)"}
          >
            <Box
              onClick={() => setConsoleOpen((v) => !v)}
              sx={{
                cursor: "pointer",
                userSelect: "none",
                display: "flex", flexDirection: "column", alignItems: "center",
                gap: 0.5,
                py: 1, px: 0.6,
                minHeight: 80,
                bgcolor: "#1f2933",
                color: consoleOpen ? "#ffa726" : "#cfd8dc",
                border: "1px solid #37474f",
                borderRadius: 0.75,
                boxShadow: "0 2px 6px rgba(0,0,0,0.35)",
                transition: "color 120ms, transform 120ms",
                "&:hover": { color: consoleOpen ? "#ffb74d" : "#ffffff", transform: "translateX(-1px)" },
              }}
            >
              <span style={{ fontSize: 13, lineHeight: 1 }}>{consoleOpen ? "▸" : "◂"}</span>
              <Typography variant="caption" sx={{
                fontSize: "0.6rem", fontWeight: 700, letterSpacing: 0.6,
                writingMode: "vertical-rl",
                transform: "rotate(180deg)",
                textTransform: "uppercase",
              }}>
                Console
              </Typography>
              {consoleOut && (
                <Box sx={{
                  fontSize: "0.55rem",
                  px: 0.5, py: 0.1, mt: 0.25,
                  bgcolor: "rgba(255,167,38,0.15)",
                  color: "#ffd180",
                  border: "1px solid rgba(255,167,38,0.45)",
                  borderRadius: 0.5,
                  fontFamily: "monospace",
                  lineHeight: 1,
                }}>
                  {consoleOut.split("\n").length}
                </Box>
              )}
            </Box>
          </Tooltip>
        </Box>
        {consoleOpen && (
          <Box sx={{
            pointerEvents: "auto",
            ml: 0.5,
            width: 440,
            maxWidth: "55vw",
            display: "flex", flexDirection: "column",
            bgcolor: "rgba(0,0,0,0.93)",
            color: "#e0e0e0",
            border: "1px solid", borderColor: "divider",
            borderRadius: 0.5,
            overflow: "hidden",
            boxShadow: 3,
          }}>
            <Box sx={{
              display: "flex", alignItems: "center", gap: 0.5,
              px: 0.75, py: 0.4,
              borderBottom: "1px solid rgba(255,255,255,0.12)",
              fontFamily: "monospace", fontSize: "0.6rem",
              color: "#bcd",
            }}>
              <span style={{ flex: 1, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase" }}>
                Console
              </span>
              <span style={{ opacity: 0.7 }}>{consoleOut ? `${consoleOut.split("\n").length} lines` : "—"}</span>
              <Button size="small" onClick={() => { consoleRef.current = ""; setConsoleOut(""); }}
                sx={{ fontSize: "0.55rem", color: "#9aa", textTransform: "none", minWidth: 0, ml: 0.5 }}>
                Clear
              </Button>
            </Box>
            <Box sx={{
              flex: 1, minHeight: 0, overflow: "auto", p: 0.75,
              fontFamily: "monospace", fontSize: "0.65rem",
              whiteSpace: "pre-wrap", wordBreak: "break-word",
            }}>
              {consoleOut || <span style={{ opacity: 0.55, fontStyle: "italic" }}>(no output yet — runs, installs, and other long ops stream here)</span>}
            </Box>
          </Box>
        )}
      </Box>
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

      {/* Engine paths — custom binary locations for the four
          interpreters.  Empty fields fall back to the sidecar's
          auto-detect.  Placeholders shift to the typical Windows /
          POSIX defaults so users have a starting point. */}
      <EngineSettingsDialog
        open={settingsOpen}
        initial={enginePaths}
        onClose={() => setSettingsOpen(false)}
        onSave={(next) => {
          setEnginePaths(next);
          saveEnginePaths(next);
          setSettingsOpen(false);
        }}
      />
    </Box>
  );
}
