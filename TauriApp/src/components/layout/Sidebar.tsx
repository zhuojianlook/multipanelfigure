/* ──────────────────────────────────────────────────────────
   Sidebar — left panel with property editors.
   Sections: GRID, SPACING, PROJECT.
   ────────────────────────────────────────────────────────── */

import { useState, useRef, useEffect } from "react";
import {
  Typography,
  Button,
  Slider,
  TextField,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Box,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  CircularProgress,
  Checkbox,
  Collapse,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import StraightenIcon from "@mui/icons-material/Straighten";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { api } from "../../api/client";
import { confirm as confirmDialog, alert as alertDialog } from "../shared/ConfirmDialog";
import { detectGlobalFont, describeDetectedFont } from "../../utils/detectGlobalFont";
import { openProjectIntoTab, enterAnalysis } from "../../utils/projectNav";
import { saveProjectDialog } from "../shared/SaveProjectDialog";

/* ── tiny reusable pieces ─────────────────────────────── */

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography
      variant="overline"
      sx={{ px: 1.5, pt: 2, pb: 0.5, display: "block", color: "text.secondary", fontSize: "0.625rem", letterSpacing: 1.5 }}
    >
      {children}
    </Typography>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", px: 1.5, py: 0.5 }}>
      <Typography variant="caption" sx={{ color: "text.secondary" }}>{label}</Typography>
      {children}
    </Box>
  );
}

function Spinner({
  value,
  min = 1,
  max = 50,
  onChange,
}: {
  value: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <IconButton
        size="small"
        onClick={() => onChange(Math.max(min, value - 1))}
        sx={{
          width: 28, height: 28,
          border: "1px solid", borderColor: "divider", borderRadius: 1,
          color: "text.primary",
          "&:hover": { bgcolor: "action.hover" },
        }}
      >
        <RemoveIcon sx={{ fontSize: 16 }} />
      </IconButton>
      <Typography variant="caption" sx={{ width: 28, textAlign: "center", fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
        {value}
      </Typography>
      <IconButton
        size="small"
        onClick={() => onChange(Math.min(max, value + 1))}
        sx={{
          width: 28, height: 28,
          border: "1px solid", borderColor: "divider", borderRadius: 1,
          color: "text.primary",
          "&:hover": { bgcolor: "action.hover" },
        }}
      >
        <AddIcon sx={{ fontSize: 16 }} />
      </IconButton>
    </Box>
  );
}

/* ── main component ───────────────────────────────────── */

/* CollageSidebar — collage-specific tools: Synchronize headers.
   A target pt size that, when applied, re-renders selected figure-kind
   items so their chosen text elements land at the same visual size after
   the collage scaling. Granularity:
     • pick which figures to apply to (checkbox per figure)
     • expand a figure to pick exactly which text elements get the size
   Builder-side hooks are NOT mounted here, so the builder doesn't thrash
   the API while the user is in collage mode. */

function CollageSidebar() {
  const items = useCollageStore((s) => s.items);
  const globalHeaderPt = useCollageStore((s) => s.globalHeaderPt);
  const setGlobalHeaderPt = useCollageStore((s) => s.setGlobalHeaderPt);
  const updateItem = useCollageStore((s) => s.updateItem);
  const [applyBusy, setApplyBusy] = useState(false);
  const [pendingPt, setPendingPt] = useState<number>(globalHeaderPt ?? 12);
  // Free-text draft for the pt field so the user can type intermediate
  // values without the controlled-number clamp coercing every keystroke.
  const [ptText, setPtText] = useState<string>(String(globalHeaderPt ?? 12));
  const commitPt = () => {
    const v = Math.max(1, Math.min(200, Math.round(Number(ptText) || pendingPt)));
    setPendingPt(v);
    setPtText(String(v));
    return v;
  };

  const figureItems = items.filter((it) => it.kind === "figure" && it.projectPath);
  // R/analysis plots can also be re-rendered at a target font size by
  // re-running their captured R code with an injected base size.
  const rItems = items.filter((it) => it.kind === "image" && it.fromAnalysis && it.rCode);
  // Collage text boxes — synced by setting their font size to the same pt.
  const textItems = items.filter((it) => it.kind === "text");

  // Inclusion set (which items the sync applies to). Default = all.
  const [excludedIds, setExcludedIds] = useState<Record<string, boolean>>({});
  const isIncluded = (id: string) => !excludedIds[id];
  const toggleIncluded = (id: string) => setExcludedIds((m) => ({ ...m, [id]: !m[id] }));
  const includedCount =
    figureItems.filter((it) => isIncluded(it.id)).length
    + rItems.filter((it) => isIncluded(it.id)).length
    + textItems.filter((it) => isIncluded(it.id)).length;
  const setAllIncluded = (on: boolean) => {
    const next: Record<string, boolean> = {};
    if (!on) for (const it of [...figureItems, ...rItems, ...textItems]) next[it.id] = true;
    setExcludedIds(next);
    // Also clear / restore the per-element (header) selection so the figure
    // hotspots match — "None" truly deselects everything (incl. headers),
    // "All" restores the default header/axis-label selection.
    applyAllElemSel(on ? "defaults" : "none");
  };

  // Per-element selection lives in the store so the canvas can overlay
  // clickable hotspots in sync with these checkboxes. Expand + loading are
  // UI-only local state.
  const elements = useCollageStore((s) => s.elemListByItem);
  const elemSel = useCollageStore((s) => s.elemSelByItem);
  const setElemList = useCollageStore((s) => s.setElemList);
  const toggleElemSel = useCollageStore((s) => s.toggleElemSel);
  const setAllElemSel = useCollageStore((s) => s.setAllElemSel);
  const applyAllElemSel = useCollageStore((s) => s.applyAllElemSel);
  const setElemSyncItem = useCollageStore((s) => s.setElemSyncItem);
  const setHoveredElem = useCollageStore((s) => s.setHoveredElem);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [elemLoading, setElemLoading] = useState<Record<string, boolean>>({});

  const loadElements = async (it: typeof figureItems[number]) => {
    if (elements[it.id] || elemLoading[it.id]) return;
    setElemLoading((m) => ({ ...m, [it.id]: true }));
    try {
      const { elements: els } = await api.getFigureElements(it.projectPath!);
      setElemList(it.id, els);
    } catch (e) {
      console.error("[collage] load elements failed for", it.name, e);
      await alertDialog({
        title: "Could not read figure",
        body: "Failed to list this figure's text elements — its .mpf may have moved.",
      });
    } finally {
      setElemLoading((m) => ({ ...m, [it.id]: false }));
    }
  };
  const toggleExpand = (it: typeof figureItems[number]) => {
    const open = !expanded[it.id];
    setExpanded((m) => ({ ...m, [it.id]: open }));
    // Show this figure's clickable element hotspots on the canvas while
    // it's expanded (clear them when collapsing).
    setElemSyncItem(open ? it.id : null);
    if (open) void loadElements(it);
  };
  const toggleElem = (itemId: string, elemId: string) => toggleElemSel(itemId, elemId);
  const setAllElems = (itemId: string, on: boolean) => setAllElemSel(itemId, on);

  const applyNow = async () => {
    const targetPt = commitPt();
    if (includedCount === 0) {
      await alertDialog({
        title: "Nothing selected",
        body: "Select at least one figure to synchronize. Add a figure (with a saved .mpf) first if the list is empty.",
      });
      return;
    }
    setApplyBusy(true);
    let succeeded = 0;
    let failed = 0;
    let skipped = 0;
    try {
      for (const it of useCollageStore.getState().items) {
        if (it.kind !== "figure" || !it.projectPath) continue;
        if (!isIncluded(it.id)) continue;
        // Element targeting: if the figure was expanded (selection known),
        // pass the explicitly-checked element ids; otherwise null = the
        // backend default set (column/row headers + axis labels).
        let elementIds: string[] | null = null;
        const sel = elemSel[it.id];
        if (sel) {
          const ids = Object.keys(sel).filter((k) => sel[k]);
          if (ids.length === 0) { skipped++; continue; }
          elementIds = ids;
        }
        const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
        const overrides = useCollageStore.getState().elemOverridesByItem[it.id] || null;
        try {
          const resp = await api.renderCollageFigure(
            it.projectPath, targetPt, Math.max(0.001, scale), it.w, elementIds,
            overrides as Record<string, unknown> | null,
          );
          if (resp?.image && resp.width && resp.height) {
            const newAspect = resp.width / resp.height;
            updateItem(it.id, {
              src: `data:image/png;base64,${resp.image}`,
              naturalW: resp.width,
              naturalH: resp.height,
              h: it.w / newAspect,
            });
            succeeded++;
          } else {
            failed++;
          }
        } catch (e) {
          console.error("[collage] header re-render failed for", it.name, e);
          failed++;
        }
      }
      // R/analysis plots: re-run their captured R code with the target font
      // size injected, then swap in the regenerated PNG. The base size is
      // compensated for the item's collage scale (approximate for R since
      // its text units differ from matplotlib's).
      for (const it of useCollageStore.getState().items) {
        if (!(it.kind === "image" && it.fromAnalysis && it.rCode)) continue;
        if (!isIncluded(it.id)) continue;
        const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
        const baseFs = Math.max(1, Math.round(targetPt / Math.max(0.001, scale)));
        try {
          const res = await api.runR(it.rCode, it.rDataCsv ?? "", it.rInterpreter ?? undefined, baseFs);
          const idx = it.rPlotIndex ?? 0;
          const png = res.plots?.[idx] ?? res.plots?.[0];
          if (res.success && png) {
            const dataUrl = `data:image/png;base64,${png}`;
            // Recompute height from the regenerated plot's aspect so text
            // isn't stretched (objectFit:"fill"); keep the user's width.
            const dims = await new Promise<{ w: number; h: number }>((resolve) => {
              const im = new window.Image();
              im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight });
              im.onerror = () => resolve({ w: it.naturalW, h: it.naturalH });
              im.src = dataUrl;
            });
            const newAspect = dims.h > 0 ? dims.w / dims.h : (it.naturalW / Math.max(1, it.naturalH));
            updateItem(it.id, {
              src: dataUrl,
              naturalW: dims.w,
              naturalH: dims.h,
              h: it.w / Math.max(0.001, newAspect),
            });
            succeeded++;
          } else {
            console.error("[collage] R re-run failed for", it.name, res.stderr);
            failed++;
          }
        } catch (e) {
          console.error("[collage] R re-run error for", it.name, e);
          failed++;
        }
      }
      // Text boxes: set the font size to the same physical pt. The canvas is
      // a 300-DPI virtual page, so 1 pt = 300/72 px. (Approximate for
      // non-default canvas DPIs.)
      const PT_TO_PX = 300 / 72;
      for (const it of useCollageStore.getState().items) {
        if (it.kind !== "text") continue;
        if (!isIncluded(it.id)) continue;
        updateItem(it.id, { fontSize: Math.round(targetPt * PT_TO_PX) });
        succeeded++;
      }
      setGlobalHeaderPt(targetPt);
    } finally {
      setApplyBusy(false);
    }
    if (failed > 0 || skipped > 0) {
      void alertDialog({
        title: "Finished",
        body: `Synchronized ${succeeded} figure${succeeded === 1 ? "" : "s"} to ${targetPt} pt`
          + (skipped ? ` · ${skipped} skipped (no elements selected)` : "")
          + (failed ? ` · ${failed} failed (the .mpf may have moved; check the console)` : "")
          + ".",
      });
    }
  };

  const clearOverride = () => {
    setGlobalHeaderPt(null);
    void alertDialog({
      title: "Lock cleared",
      body: "Synchronized size cleared. Existing collage items keep their last rendered preview — re-synchronize or re-add figures to revert to their saved sizes.",
    });
  };

  return (
    <Box sx={{ p: 2, display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Typography variant="caption" sx={{ display: "block", letterSpacing: 1.2, fontSize: "0.6rem", textTransform: "uppercase", color: "text.secondary" }}>
        Collage Tools
      </Typography>

      <Box>
        <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary", mb: 0.5, display: "block", fontWeight: 600 }}>
          Synchronize headers to size (pt)
        </Typography>
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", mb: 1, display: "block", lineHeight: 1.4 }}>
          Arrange + scale figures first, then Synchronize. Each selected figure
          is re-rendered so its chosen text elements land at this point size
          after collage scaling — uniform across differently-scaled figures.
          Expand a figure to pick exactly which text elements to apply to.
        </Typography>
        <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mb: 1 }}>
          <TextField
            type="number"
            size="small"
            value={ptText}
            onChange={(e) => setPtText(e.target.value)}
            onBlur={commitPt}
            onKeyDown={(e) => { if (e.key === "Enter") { commitPt(); (e.target as HTMLInputElement).blur(); } }}
            inputProps={{ min: 1, max: 200, step: 1 }}
            sx={{
              width: 64,
              "& input": { fontSize: "0.7rem", py: 0.5, textAlign: "center", colorScheme: "dark" },
              "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
                filter: "invert(1)", opacity: 1,
              },
            }}
          />
          <Button
            variant="contained"
            size="small"
            disabled={applyBusy || includedCount === 0}
            onClick={applyNow}
            sx={{ fontSize: "0.65rem", textTransform: "none", flex: 1 }}
          >
            {applyBusy ? "Synchronizing…" : `Synchronize headers${includedCount ? ` (${includedCount})` : ""}`}
          </Button>
        </Box>

        {/* Figure selection list */}
        {figureItems.length === 0 ? (
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", display: "block", fontStyle: "italic" }}>
            No figures in the collage yet. Use "Add to Collage" from the builder or "Import project".
          </Typography>
        ) : (
          <>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.25 }}>
              <Typography variant="caption" sx={{ fontSize: "0.58rem", color: "text.secondary" }}>
                Figures ({includedCount}/{figureItems.length})
              </Typography>
              <Box>
                <Button size="small" variant="text" onClick={() => setAllIncluded(true)} sx={{ fontSize: "0.55rem", textTransform: "none", minWidth: 0, p: "0 4px" }}>All</Button>
                <Button size="small" variant="text" onClick={() => setAllIncluded(false)} sx={{ fontSize: "0.55rem", textTransform: "none", minWidth: 0, p: "0 4px" }}>None</Button>
              </Box>
            </Box>
            <Box sx={{ border: "1px solid var(--c-border)", borderRadius: 1, maxHeight: 300, overflowY: "auto" }}>
              {figureItems.map((it) => {
                const els = elements[it.id];
                const sel = elemSel[it.id];
                const selCount = sel ? Object.values(sel).filter(Boolean).length : null;
                return (
                  <Box key={it.id} sx={{ borderBottom: "1px solid var(--c-border)", "&:last-child": { borderBottom: "none" } }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.25, px: 0.25 }}>
                      <Checkbox
                        size="small"
                        checked={isIncluded(it.id)}
                        onChange={() => toggleIncluded(it.id)}
                        sx={{ p: 0.25 }}
                      />
                      <Box
                        onClick={() => toggleExpand(it)}
                        sx={{ flex: 1, display: "flex", alignItems: "center", gap: 0.25, cursor: "pointer", overflow: "hidden", py: 0.5 }}
                      >
                        {expanded[it.id] ? <ExpandMoreIcon sx={{ fontSize: 14 }} /> : <ChevronRightIcon sx={{ fontSize: 14 }} />}
                        <Typography variant="caption" sx={{ fontSize: "0.62rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={it.name}>
                          {it.name}
                        </Typography>
                        {selCount !== null && (
                          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", ml: "auto", pl: 0.5, flexShrink: 0 }}>
                            {selCount} el
                          </Typography>
                        )}
                      </Box>
                    </Box>
                    <Collapse in={!!expanded[it.id]} unmountOnExit>
                      <Box sx={{ pl: 2.5, pr: 0.5, pb: 0.5 }}>
                        {elemLoading[it.id] && (
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 0.5 }}>
                            <CircularProgress size={12} />
                            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>Reading elements…</Typography>
                          </Box>
                        )}
                        {els && els.length === 0 && (
                          <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", fontStyle: "italic" }}>No text elements.</Typography>
                        )}
                        {els && els.length > 0 && (
                          <>
                            <Box sx={{ mb: 0.25 }}>
                              <Button size="small" variant="text" onClick={() => setAllElems(it.id, true)} sx={{ fontSize: "0.52rem", textTransform: "none", minWidth: 0, p: "0 4px" }}>All</Button>
                              <Button size="small" variant="text" onClick={() => setAllElems(it.id, false)} sx={{ fontSize: "0.52rem", textTransform: "none", minWidth: 0, p: "0 4px" }}>None</Button>
                            </Box>
                            {els.map((e) => (
                              <Box key={e.id}
                                onMouseEnter={() => {
                                  if (!e.geom) return;
                                  // Show this figure's hotspots on the canvas + highlight this one.
                                  setElemSyncItem(it.id);
                                  setHoveredElem({ itemId: it.id, elemId: e.id });
                                }}
                                onMouseLeave={() => setHoveredElem(null)}
                                sx={{ display: "flex", alignItems: "center", gap: 0.25, borderRadius: 0.5, "&:hover": { backgroundColor: "rgba(79,195,247,0.12)" } }}>
                                <Checkbox
                                  size="small"
                                  checked={!!sel?.[e.id]}
                                  onChange={() => toggleElem(it.id, e.id)}
                                  sx={{ p: 0.25 }}
                                />
                                <Typography variant="caption" sx={{ fontSize: "0.56rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={`${e.type}: ${e.text}`}>
                                  <Box component="span" sx={{ color: "text.secondary" }}>{e.type}: </Box>
                                  {e.text || "(empty)"}
                                  {e.font_size != null && <Box component="span" sx={{ color: "text.secondary" }}> · {e.font_size}pt</Box>}
                                </Typography>
                              </Box>
                            ))}
                          </>
                        )}
                      </Box>
                    </Collapse>
                  </Box>
                );
              })}
            </Box>
          </>
        )}

        {/* R / analysis plots — re-rendered by re-running their R code with
            the target font size injected. No per-element tree (raster). */}
        {rItems.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.58rem", color: "text.secondary", display: "block", mb: 0.25 }}>
              R / analysis figures
            </Typography>
            <Box sx={{ border: "1px solid var(--c-border)", borderRadius: 1, maxHeight: 160, overflowY: "auto" }}>
              {rItems.map((it) => (
                <Box key={it.id} sx={{ display: "flex", alignItems: "center", gap: 0.25, px: 0.25, borderBottom: "1px solid var(--c-border)", "&:last-child": { borderBottom: "none" } }}>
                  <Checkbox size="small" checked={isIncluded(it.id)} onChange={() => toggleIncluded(it.id)} sx={{ p: 0.25 }} />
                  <Typography variant="caption" sx={{ fontSize: "0.62rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", py: 0.5 }} title={it.name}>
                    {it.name}
                  </Typography>
                </Box>
              ))}
            </Box>
            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", lineHeight: 1.3, mt: 0.25 }}>
              R plots re-run their code with the size injected (best-effort for ggplot themes; requires R installed).
            </Typography>
          </Box>
        )}

        {/* Text boxes — synced by setting their font size to the same pt. */}
        {textItems.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.58rem", color: "text.secondary", display: "block", mb: 0.25 }}>
              Text boxes
            </Typography>
            <Box sx={{ border: "1px solid var(--c-border)", borderRadius: 1, maxHeight: 120, overflowY: "auto" }}>
              {textItems.map((it) => (
                <Box key={it.id} sx={{ display: "flex", alignItems: "center", gap: 0.25, px: 0.25, borderBottom: "1px solid var(--c-border)", "&:last-child": { borderBottom: "none" } }}>
                  <Checkbox size="small" checked={isIncluded(it.id)} onChange={() => toggleIncluded(it.id)} sx={{ p: 0.25 }} />
                  <Typography variant="caption" sx={{ fontSize: "0.62rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", py: 0.5 }} title={it.text || it.name}>
                    {it.text || it.name}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        )}

        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", display: "block", mt: 1 }}>
          {globalHeaderPt
            ? `Last applied: ${globalHeaderPt} pt`
            : "Not yet synchronized (each figure uses its own sizes)"}
        </Typography>
        {globalHeaderPt !== null && (
          <Button
            size="small"
            variant="text"
            color="warning"
            onClick={clearOverride}
            sx={{ fontSize: "0.6rem", textTransform: "none", mt: 0.5, p: 0, minWidth: 0 }}
          >
            Clear lock
          </Button>
        )}
      </Box>
    </Box>
  );
}

export function Sidebar() {
  const mode = useCollageStore((s) => s.mode);
  if (mode === "collage") return <CollageSidebar />;
  return <BuilderSidebar />;
}

function BuilderSidebar() {
  const config = useFigureStore((s) => s.config);
  const updateGridSize = useFigureStore((s) => s.updateGridSize);
  const setSpacing = useFigureStore((s) => s.setSpacing);
  const setConfig = useFigureStore((s) => s.setConfig);
  const saveProject = useFigureStore((s) => s.saveProject);
  const checkGridResizeConflict = useFigureStore((s) => s.checkGridResizeConflict);
  const fetchFonts = useFigureStore((s) => s.fetchFonts);
  const fonts = useFigureStore((s) => s.fonts);
  const fontInputRef = useRef<HTMLInputElement>(null);

  // Grid resize warning dialog
  const [resizeWarningOpen, setResizeWarningOpen] = useState(false);
  const [resizeWarningConflicts, setResizeWarningConflicts] = useState<string[]>([]);
  const [pendingResize, setPendingResize] = useState<{ rows: number; cols: number } | null>(null);
  // Global-font-apply override warning state
  const [fontWarningOpen, setFontWarningOpen] = useState(false);
  const [fontWarningConflicts, setFontWarningConflicts] = useState<string[]>([]);
  const [pendingGlobalFont, setPendingGlobalFont] = useState<string | null>(null);

  // Load project dialog (Save uses the shared saveProjectDialog helper)
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [projectPath, setProjectPath] = useState("");
  // Project-load progress. The backend's /api/project/load can take
  // many seconds for large .mpf files (zip decompression of all
  // images + matplotlib font cache warm-up + first preview render),
  // so we lock the dialog and show a spinner until the round-trip
  // completes. Indeterminate because the backend doesn't stream
  // progress events today; switching to determinate later just
  // means swapping CircularProgress's `value` prop.
  const [loadingProject, setLoadingProject] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Analysis dialog
  // When true, the user enters the scale-bar value as pixels per unit and we
  // invert it to units per pixel before storing.
  const [scaleInverted, setScaleInverted] = useState(false);
  // When set, the add-scalebar form is editing an existing preset; the
  // original key is removed on save so renames work cleanly.
  const [editingScaleKey, setEditingScaleKey] = useState<string | null>(null);

  // Computed measurements from backend
  const [computedMeasurements, setComputedMeasurements] = useState<Array<{ panel: string; name: string; type: string; value: string; numeric?: number; unit?: string }>>([]);
  useEffect(() => {
    // Fetch measurements whenever config changes (lines/areas may have been added)
    if (!config) return;
    const hasAny = config.panels.some((row: any[]) =>
      row.some((p: any) =>
        (p.lines?.some((l: any) => l.show_measure) || p.areas?.some((a: any) => a.show_measure))
      )
    );
    if (!hasAny) { setComputedMeasurements([]); return; }
    const t = setTimeout(() => {
      api.getMeasurements().then((r) => setComputedMeasurements(r.measurements)).catch(() => {});
    }, 500); // debounce
    return () => clearTimeout(t);
  }, [config]);

  if (!config) return null;

  // Use nullish coalescing so a genuine `0` spacing stays at 0 instead of
  // being snapped back to the default 0.02 (which displayed as 40 px).
  const spacingPx = Math.round((config.spacing ?? 0.02) * 2000);

  const handleGridResize = (newRows: number, newCols: number) => {
    const conflicts = checkGridResizeConflict(newRows, newCols);
    if (conflicts.length > 0) {
      setResizeWarningConflicts(conflicts);
      setPendingResize({ rows: newRows, cols: newCols });
      setResizeWarningOpen(true);
    } else {
      updateGridSize(newRows, newCols);
    }
  };

  const confirmResize = () => {
    if (pendingResize) {
      updateGridSize(pendingResize.rows, pendingResize.cols);
    }
    setResizeWarningOpen(false);
    setPendingResize(null);
    setResizeWarningConflicts([]);
  };

  // Apply the global font to every header / label, including clearing
  // any per-segment font_name overrides so the new global font wins
  // uniformly. Used by both the no-conflict path and the override
  // confirmation dialog.
  const applyGlobalFont = (fontName: string) => {
    if (!config) return;
    const stripSegFont = (segs: any[] | null | undefined) =>
      Array.isArray(segs) ? segs.map((s) => ({ ...s, font_name: undefined })) : segs;
    const colLabels = config.column_labels.map((l: any) => ({
      ...l,
      font_name: fontName,
      styled_segments: stripSegFont(l.styled_segments),
    }));
    const rowLabels = config.row_labels.map((l: any) => ({
      ...l,
      font_name: fontName,
      styled_segments: stripSegFont(l.styled_segments),
    }));
    const colHeaders = config.column_headers.map((level: any) => ({
      ...level,
      headers: level.headers.map((h: any) => ({
        ...h,
        font_name: fontName,
        styled_segments: stripSegFont(h.styled_segments),
      })),
    }));
    const rowHeaders = config.row_headers.map((level: any) => ({
      ...level,
      headers: level.headers.map((h: any) => ({
        ...h,
        font_name: fontName,
        styled_segments: stripSegFont(h.styled_segments),
      })),
    }));
    setConfig({ ...config, column_labels: colLabels, row_labels: rowLabels, column_headers: colHeaders, row_headers: rowHeaders });
  };

  const confirmGlobalFont = () => {
    if (pendingGlobalFont) {
      applyGlobalFont(pendingGlobalFont);
    }
    setFontWarningOpen(false);
    setPendingGlobalFont(null);
    setFontWarningConflicts([]);
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", pb: 2 }}>
      {/* App title */}
      <Box sx={{ px: 1.5, py: 1.5, borderBottom: 1, borderColor: "divider" }}>
        <Typography variant="subtitle2" fontWeight={600}>
          Multi-Panel Figure
        </Typography>
      </Box>

      {/* ── GRID ──────────────────────────────────────── */}
      <SectionTitle>Grid</SectionTitle>

      <Field label="Rows">
        <Spinner
          value={config.rows}
          onChange={(v) => handleGridResize(v, config.cols)}
        />
      </Field>

      <Field label="Columns">
        <Spinner
          value={config.cols}
          onChange={(v) => handleGridResize(config.rows, v)}
        />
      </Field>

      <Field label="Spacing">
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: 160 }}>
          <Slider
            value={spacingPx}
            min={0}
            max={250}
            step={1}
            onChange={(_, val) => setSpacing((val as number) / 2000)}
            sx={{ flex: 1 }}
          />
          <Typography variant="caption" sx={{ width: 44, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
            {spacingPx} px
          </Typography>
        </Box>
      </Field>

      {/* Renumber panels — manually re-letters every panel's auto label
          to match its current (row, col). Useful for projects from
          before grid-resize / swap auto-renumbering landed, or to clean
          up after a series of drag operations. Custom-typed labels
          (linked_to_header off, or text not /^[a-z]{1,2}$/) are skipped. */}
      <Box sx={{ px: 1.5, mt: 0.5 }}>
        <Button
          size="small"
          variant="outlined"
          fullWidth
          onClick={() => {
            const renumberPanels = useFigureStore.getState().renumberPanels;
            if (renumberPanels) renumberPanels();
          }}
          sx={{ fontSize: "0.65rem", textTransform: "none" }}
        >
          Renumber panels (a, b, c…)
        </Button>
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── SCALE BARS (Resolution Presets) ────────────── */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", pr: 1.5 }}>
        <SectionTitle>Scale Bars</SectionTitle>
        <Button
          size="small"
          variant="text"
          onClick={async () => {
            const ok = await confirmDialog({
              title: "Restore default scale bars",
              body: "Replace all scale-bar presets with the bundled microscope defaults? Custom entries will be lost.",
              confirmLabel: "Restore defaults",
              destructive: true,
            });
            if (!ok) return;
            try {
              const entries = await api.restoreDefaultResolutions();
              if (config) setConfig({ ...config, resolution_entries: entries });
            } catch (e) { console.error(e); }
          }}
          sx={{ fontSize: "0.55rem", textTransform: "none", py: 0, minWidth: 0, px: 0.75 }}
        >
          Restore defaults
        </Button>
      </Box>
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
        {/* Add new scale bar: Name, Value, Unit, direction toggle.
            Kept at the top so a long preset list (and it gets long
            once the bundled microscope defaults are loaded) doesn't
            push the input form below the fold. */}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, alignItems: "center" }}>
          <TextField
            placeholder="Name"
            size="small"
            sx={{ flex: "1 1 50px", minWidth: 50, "& input": { fontSize: "0.65rem", py: 0.25, px: 0.5 } }}
            id="scale-name-input"
          />
          <TextField
            placeholder={scaleInverted ? "px/unit" : "unit/px"}
            type="number"
            size="small"
            sx={{
              flex: "0 0 60px", width: 60,
              "& input": { fontSize: "0.65rem", py: 0.25, px: 0.5, MozAppearance: "textfield" },
              "& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button": { WebkitAppearance: "none", margin: 0 },
            }}
            id="scale-value-input"
            inputProps={{ step: 0.001, min: 0 }}
          />
          <select
            id="scale-unit-input"
            defaultValue="um"
            style={{
              fontSize: "0.6rem",
              width: 44,
              height: 22,
              backgroundColor: "var(--c-surface2)",
              color: "var(--c-text)",
              border: "1px solid var(--c-border)",
              borderRadius: 4,
              padding: "0 2px",
              flexShrink: 0,
              appearance: "auto",
            }}
          >
            <option value="km">km</option>
            <option value="m">m</option>
            <option value="cm">cm</option>
            <option value="mm">mm</option>
            <option value="um">{"\u00B5m"}</option>
            <option value="nm">nm</option>
            <option value="pm">pm</option>
          </select>
          <Button
            size="small"
            variant={scaleInverted ? "contained" : "outlined"}
            onClick={() => setScaleInverted(!scaleInverted)}
            title={scaleInverted ? "Input is pixels per unit — click to switch to units per pixel" : "Input is units per pixel — click to switch to pixels per unit"}
            sx={{ fontSize: "0.5rem", textTransform: "none", px: 0.5, py: 0, minWidth: 38, height: 22, flexShrink: 0 }}
          >
            {scaleInverted ? "px/u" : "u/px"}
          </Button>
          <IconButton size="small" title={editingScaleKey ? "Save changes to this scale bar" : "Add scale bar"} onClick={() => {
            const nameEl = document.getElementById("scale-name-input") as HTMLInputElement;
            const valEl = document.getElementById("scale-value-input") as HTMLInputElement;
            const unitEl = document.getElementById("scale-unit-input") as HTMLSelectElement;
            if (nameEl?.value && valEl?.value) {
              const unit = unitEl?.value || "um";
              const rawValue = Number(valEl.value);
              if (!isFinite(rawValue) || rawValue <= 0) return;
              // If user is in "px/unit" mode, invert to get "unit/px" for storage.
              const valuePerPx = scaleInverted ? 1 / rawValue : rawValue;
              // Convert to μm/px for internal storage, store unit in key suffix
              const conversionToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
              const valueInUm = valuePerPx * (conversionToUm[unit] || 1);
              const key = `${nameEl.value}|${unit}`;
              const entries = { ...config.resolution_entries };
              // Editing: drop the original key first so a rename / unit
              // change replaces the entry instead of leaving a duplicate.
              if (editingScaleKey && editingScaleKey !== key) delete entries[editingScaleKey];
              entries[key] = valueInUm;
              setConfig({ ...config, resolution_entries: entries });
              api.updateResolutions(entries).catch(console.error);
              nameEl.value = "";
              valEl.value = "";
              setEditingScaleKey(null);
              setScaleInverted(false);
            }
          }} sx={{ width: 22, height: 22 }}>
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
          {editingScaleKey && (
            <Button
              size="small" variant="text"
              onClick={() => {
                setEditingScaleKey(null);
                const nameEl = document.getElementById("scale-name-input") as HTMLInputElement;
                const valEl = document.getElementById("scale-value-input") as HTMLInputElement;
                if (nameEl) nameEl.value = "";
                if (valEl) valEl.value = "";
                setScaleInverted(false);
              }}
              sx={{ fontSize: "0.5rem", textTransform: "none", px: 0.5, py: 0, minWidth: 0, height: 22 }}
            >
              Cancel edit
            </Button>
          )}
        </Box>
        {/* Capped + scrollable list — long preset libraries shouldn't
            push every other sidebar section below the fold. */}
        <Box sx={{ maxHeight: 260, overflowY: "auto", display: "flex", flexDirection: "column", gap: 0.5, mt: 0.5 }}>
          {config.resolution_entries && Object.entries(config.resolution_entries).map(([name, val]) => {
            // Parse unit from name suffix if present, e.g. "Microscope 10x|um" or default to μm
            const parts = name.split("|");
            const displayName = parts[0];
            const unit = parts[1] || "um";
            const unitLabelMap: Record<string, string> = { km: "km", m: "m", cm: "cm", mm: "mm", um: "\u00B5m", nm: "nm", pm: "pm" };
            const unitLabel = unitLabelMap[unit] || "\u00B5m";
            // val is stored in µm/px internally — convert back to display unit
            const convFromUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
            const displayVal = (val as number) / (convFromUm[unit] || 1);
            return (
              <Box key={name} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                ...(editingScaleKey === name ? { outline: "1px solid #4FC3F7", borderRadius: 1, px: 0.5 } : {}) }}>
                <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>{displayName}: {Number(displayVal.toPrecision(6))} {unitLabel}/px</Typography>
                <Box sx={{ display: "flex", alignItems: "center" }}>
                  <IconButton size="small" title="Edit this scale bar" onClick={() => {
                    // Load this preset into the add-form for editing. Always
                    // populate in "unit/px" mode (the stored convention).
                    const nameEl = document.getElementById("scale-name-input") as HTMLInputElement;
                    const valEl = document.getElementById("scale-value-input") as HTMLInputElement;
                    const unitEl = document.getElementById("scale-unit-input") as HTMLSelectElement;
                    if (nameEl) nameEl.value = displayName;
                    if (valEl) valEl.value = String(Number(displayVal.toPrecision(6)));
                    if (unitEl) unitEl.value = unit;
                    setScaleInverted(false);
                    setEditingScaleKey(name);
                  }} sx={{ width: 20, height: 20 }}>
                    <EditIcon sx={{ fontSize: 12 }} />
                  </IconButton>
                  <IconButton size="small" title="Delete this scale bar" onClick={() => {
                    const entries = { ...config.resolution_entries };
                    delete entries[name];
                    setConfig({ ...config, resolution_entries: entries });
                    api.updateResolutions(entries).catch(console.error);
                    if (editingScaleKey === name) setEditingScaleKey(null);
                  }} sx={{ width: 20, height: 20 }}>
                    <DeleteIcon sx={{ fontSize: 12 }} />
                  </IconButton>
                </Box>
              </Box>
            );
          })}
        </Box>
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── FONTS ──────────────────────────────────────── */}
      <SectionTitle>Fonts</SectionTitle>
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 1 }}>
        {/* Global font selector */}
        <Box>
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", mb: 0.5, display: "block" }}>
            Global font
          </Typography>
          <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
          <select
            id="global-font-select"
            defaultValue={(() => {
              if (fonts.length === 0) return "";
              // Case-insensitive, extension-insensitive match against the
              // available fonts list (system fonts may be listed as
              // "Arial.ttf"/"Arial" with different casing than the stored
              // "arial.ttf").
              const stripExt = (f: string) => f.replace(/\.(ttf|otf|ttc)$/i, "");
              const matchInFonts = (target?: string | null) => {
                if (!target) return undefined;
                const t = stripExt(target).toLowerCase();
                return fonts.find((f) => stripExt(f).toLowerCase() === t);
              };
              // 1. Detected most-common font across labels/headers.
              const detected = detectGlobalFont(config);
              const dm = matchInFonts(detected?.font_name);
              if (dm) return dm;
              // 2. First-column-label fallback (brand-new figure).
              const em = matchInFonts(config?.column_labels?.[0]?.font_name);
              if (em) return em;
              // 3. Default to EXACT "Arial" (not "Arial Narrow"/"Arial
              //    Black" — the old /^arial\b/ matched those because of
              //    the word boundary after "Arial ").
              const exactArial = fonts.find((f) => /^arial$/i.test(stripExt(f)));
              if (exactArial) return exactArial;
              const arialFamily = fonts.find((f) => /^arial\b/i.test(stripExt(f)));
              if (arialFamily) return arialFamily;
              return fonts[0] || "";
            })()}
            // Key includes a fingerprint of the detected font so the
            // <select> re-mounts (picks up a new defaultValue) when
            // headers/labels are re-fonted.  Without this, the
            // <select>'s defaultValue is locked at first mount.
            key={(() => {
              const d = detectGlobalFont(config);
              const fp = d ? `${d.font_name}|${(d.font_style || []).join(",")}` : "none";
              return `${fonts.length > 0 ? "loaded" : "empty"}-${fonts.length}-${fp}`;
            })()}
            style={{
              fontSize: "0.65rem",
              flex: 1,
              height: 24,
              backgroundColor: "var(--c-surface2)",
              color: "var(--c-text)",
              border: "1px solid var(--c-border)",
              borderRadius: 4,
              padding: "0 4px",
              minWidth: 0,
            }}
          >
            {[...fonts].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })).map(f => (
              <option key={f} value={f}>{f.replace(/\.(ttf|otf|ttc)$/i, "")}</option>
            ))}
          </select>
          <Button
            variant="contained"
            size="small"
            sx={{ fontSize: "0.55rem", textTransform: "none", px: 1, py: 0.25, minWidth: 0, flexShrink: 0 }}
            onClick={() => {
              if (!config) return;
              const el = document.getElementById("global-font-select") as HTMLSelectElement;
              if (!el) return;
              const fontName = el.value;
              // Collect human-readable labels for any element whose
              // current font_name (or any seg's font_name) differs from
              // the new global font. If any are found, show the
              // override-warning dialog before committing.
              const conflicts: string[] = [];
              const isMismatch = (f: string | null | undefined) =>
                f && f !== fontName;
              const segMismatch = (segs: any[] | null | undefined) =>
                Array.isArray(segs) && segs.some((s) => isMismatch(s?.font_name));
              config.column_labels.forEach((l: any, i: number) => {
                if (isMismatch(l.font_name) || segMismatch(l.styled_segments)) {
                  conflicts.push(`Column ${i + 1} label`);
                }
              });
              config.row_labels.forEach((l: any, i: number) => {
                if (isMismatch(l.font_name) || segMismatch(l.styled_segments)) {
                  conflicts.push(`Row ${i + 1} label`);
                }
              });
              config.column_headers.forEach((level: any, li: number) => {
                level.headers.forEach((h: any, gi: number) => {
                  if (isMismatch(h.font_name) || segMismatch(h.styled_segments)) {
                    conflicts.push(`Column header tier ${li + 1} group ${gi + 1}`);
                  }
                });
              });
              config.row_headers.forEach((level: any, li: number) => {
                level.headers.forEach((h: any, gi: number) => {
                  if (isMismatch(h.font_name) || segMismatch(h.styled_segments)) {
                    conflicts.push(`Row header tier ${li + 1} group ${gi + 1}`);
                  }
                });
              });
              if (conflicts.length > 0) {
                setFontWarningConflicts(conflicts);
                setPendingGlobalFont(fontName);
                setFontWarningOpen(true);
              } else {
                applyGlobalFont(fontName);
              }
            }}
          >
            Apply
          </Button>
          </Box>
          {/* Inline summary of what the detector actually found.  Surfaces
              the font_style (bold / italic) which the <select> alone can't
              represent — fixes the user complaint that "Arial Narrow Italic"
              looked like just "Arial" in the picker. */}
          {(() => {
            const d = detectGlobalFont(config);
            if (!d) return null;
            const ambiguous = d.total > 0 && d.count < Math.max(2, d.total * 0.5);
            return (
              <Typography variant="caption" sx={{
                fontSize: "0.55rem",
                color: ambiguous ? "warning.light" : "text.secondary",
                mt: 0.25,
                display: "block",
                fontStyle: d.font_style.some((s) => /italic/i.test(s)) ? "italic" : "normal",
                fontWeight: d.font_style.some((s) => /bold/i.test(s)) ? 700 : 400,
              }}>
                Detected: {describeDetectedFont(d)} ({d.count}/{d.total})
                {ambiguous && " — mixed; pick one to unify"}
              </Typography>
            );
          })()}
        </Box>

        {/* Global header font size — applies one size across every header
            so they line up in a collage / poster builder context. */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", flex: 1 }}>
            Header size
          </Typography>
          <TextField
            type="number"
            size="small"
            id="global-header-size-input"
            defaultValue={config?.column_headers?.[0]?.headers?.[0]?.font_size ?? 14}
            key={`hdr-size-${config?.column_headers?.length ?? 0}-${config?.row_headers?.length ?? 0}`}
            inputProps={{ min: 4, max: 200, step: 1 }}
            sx={{
              width: 56,
              // colorScheme:dark switches the WebView's native form controls
              // (including the number-input spinner arrows) to a dark
              // palette → arrows render white instead of dark grey on the
              // dark sidebar.
              "& input": {
                fontSize: "0.65rem",
                py: 0.25,
                px: 0.5,
                textAlign: "center",
                colorScheme: "dark",
              },
              "& input[type=number]::-webkit-inner-spin-button, & input[type=number]::-webkit-outer-spin-button": {
                filter: "invert(1)",
                opacity: 1,
              },
            }}
          />
          <Button
            variant="contained"
            size="small"
            sx={{ fontSize: "0.55rem", textTransform: "none", px: 1, py: 0.25, minWidth: 0, flexShrink: 0 }}
            onClick={() => {
              if (!config) return;
              const el = document.getElementById("global-header-size-input") as HTMLInputElement;
              if (!el) return;
              const n = Number(el.value);
              if (!isFinite(n) || n < 4) return;
              // Apply to BOTH secondary headers (column_headers/row_headers)
              // AND primary labels (column_labels/row_labels). Previously
              // only the former were updated, so when a project had no
              // secondary header levels the button appeared to do nothing.
              const colHeaders = config.column_headers.map((level: any) => ({
                ...level,
                headers: level.headers.map((h: any) => ({
                  ...h,
                  font_size: n,
                  // Clear any per-segment size overrides so the global size
                  // actually wins for every character.
                  styled_segments: (h.styled_segments || []).map((seg: any) => ({ ...seg, font_size: undefined })),
                })),
              }));
              const rowHeaders = config.row_headers.map((level: any) => ({
                ...level,
                headers: level.headers.map((h: any) => ({
                  ...h,
                  font_size: n,
                  styled_segments: (h.styled_segments || []).map((seg: any) => ({ ...seg, font_size: undefined })),
                })),
              }));
              const colLabels = config.column_labels.map((l: any) => ({
                ...l,
                font_size: n,
                styled_segments: (l.styled_segments || []).map((seg: any) => ({ ...seg, font_size: undefined })),
              }));
              const rowLabels = config.row_labels.map((l: any) => ({
                ...l,
                font_size: n,
                styled_segments: (l.styled_segments || []).map((seg: any) => ({ ...seg, font_size: undefined })),
              }));
              setConfig({ ...config, column_headers: colHeaders, row_headers: rowHeaders, column_labels: colLabels, row_labels: rowLabels });
            }}
          >
            Apply
          </Button>
        </Box>
        {/* Upload custom font */}
        <input
          ref={fontInputRef}
          type="file"
          accept=".ttf,.otf"
          multiple
          style={{ display: "none" }}
          onChange={async (e) => {
            const files = e.target.files;
            if (!files || files.length === 0) return;
            try {
              await api.uploadFonts(Array.from(files));
              fetchFonts();
            } catch (err) {
              console.error("Font upload failed", err);
            }
            if (fontInputRef.current) fontInputRef.current.value = "";
          }}
        />
        <Button
          fullWidth
          variant="outlined"
          size="small"
          startIcon={<UploadFileIcon />}
          onClick={() => fontInputRef.current?.click()}
          sx={{ fontSize: "0.6rem", textTransform: "none" }}
        >
          Upload Font (.ttf/.otf)
        </Button>
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── DISPLAY ────────────────────────────────────── */}
      <SectionTitle>Display</SectionTitle>
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <Typography variant="caption" sx={{ color: "text.secondary" }}>Normalize widths</Typography>
          <input
            type="checkbox"
            checked={config.normalize_widths ?? false}
            onChange={(e) => {
              // When the user FIRST enables normalize (mode is
              // unset), default to "height" — Match-height is the
              // common case for grids of microscopy panels. Once
              // they pick a mode explicitly, preserve that choice
              // across toggles.
              const enabled = e.target.checked;
              const nextMode = config.normalize_mode ?? "height";
              const updated = { ...config, normalize_widths: enabled, normalize_mode: nextMode };
              setConfig(updated);
              api.updateConfig(updated).catch(console.error);
            }}
            style={{ cursor: "pointer" }}
          />
        </Box>
        {config.normalize_widths && (
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Typography variant="caption" sx={{ color: "text.secondary" }}>Mode</Typography>
            <select
              value={config.normalize_mode ?? "height"}
              onChange={(e) => {
                const updated = { ...config, normalize_mode: e.target.value };
                setConfig(updated);
                api.updateConfig(updated).catch(console.error);
              }}
              style={{ fontSize: "0.65rem", backgroundColor: "var(--c-surface2)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "2px 4px" }}
            >
              <option value="height">Match height</option>
              <option value="width">Match width</option>
            </select>
          </Box>
        )}
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── ANALYSIS ───────────────────────────────────── */}
      <SectionTitle>Analysis</SectionTitle>
      {(() => {
        // Count zoom insets flagged include_in_analysis — those are
        // pixel-analysis sources, an independent reason to enter the
        // Analysis dialog (the user might have no line / area
        // measurements but still want to run Python pipelines on a
        // marked inset).
        let insetCount = 0;
        if (config) {
          for (let r = 0; r < config.rows; r++) {
            for (let c = 0; c < config.cols; c++) {
              const p = config.panels[r]?.[c];
              if (!p?.add_zoom_inset) continue;
              const arr = (p.zoom_insets && p.zoom_insets.length > 0)
                ? p.zoom_insets
                : (p.zoom_inset ? [p.zoom_inset] : []);
              for (const zi of arr) {
                if (zi && (zi as { include_in_analysis?: boolean }).include_in_analysis) insetCount++;
              }
            }
          }
        }
        const hasAnything = computedMeasurements.length > 0 || insetCount > 0;
        return (
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
        {!hasAnything ? (
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled" }}>
            No measurements yet. Add lines / areas with "measure" enabled, or tick "Include in Analysis" on a zoom inset.
          </Typography>
        ) : (
          <>
          {/* Group measurements by panel */}
          {computedMeasurements.length > 0 && (
            <Box sx={{ maxHeight: 200, overflowY: "auto" }}>
              {(() => {
                const grouped = new Map<string, typeof computedMeasurements>();
                computedMeasurements.forEach(m => {
                  if (!grouped.has(m.panel)) grouped.set(m.panel, []);
                  grouped.get(m.panel)!.push(m);
                });
                return Array.from(grouped.entries()).map(([panel, measurements]) => (
                  <Box key={panel} sx={{ mb: 0.5 }}>
                    <Typography variant="caption" sx={{ fontSize: "0.6rem", fontWeight: 700, color: "text.primary", display: "block", mb: 0.25 }}>
                      {panel}
                    </Typography>
                    {measurements.map((m, i) => (
                      <Box key={i} sx={{ display: "flex", alignItems: "center", pl: 1, py: 0.1 }}>
                        <StraightenIcon sx={{ fontSize: 10, mr: 0.5, color: "text.secondary" }} />
                        <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>
                          {m.name}:&nbsp;
                        </Typography>
                        <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "primary.main", fontWeight: 600 }}>
                          {m.value}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                ));
              })()}
            </Box>
          )}
          {insetCount > 0 && (
            <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "secondary.main", fontStyle: "italic" }}>
              {insetCount} zoom inset{insetCount === 1 ? "" : "s"} marked for pixel analysis — open Analysis to run Python.
            </Typography>
          )}
          <Box sx={{ display: "flex", gap: 0.5 }}>
            {computedMeasurements.length > 0 && (
              <Button size="small" variant="text" sx={{ fontSize: "0.55rem", textTransform: "none", flex: 1 }}
                onClick={() => {
                  // Value + Unit as separate columns so the measurement
                  // unit is explicitly recorded, not just embedded in text.
                  const csv = ["Panel,Name,Type,Value,Unit",
                    ...computedMeasurements.map(m =>
                      `${m.panel},${m.name},${m.type},${m.numeric ?? m.value},${m.unit ?? ""}`)].join("\n");
                  const blob = new Blob([csv], { type: "text/csv" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url; a.download = "measurements.csv"; a.click();
                  URL.revokeObjectURL(url);
                }}
              >Export CSV</Button>
            )}
            <Button size="small" variant="outlined" sx={{ fontSize: "0.55rem", textTransform: "none", flex: 1 }}
              onClick={() => void enterAnalysis()}
            >Open Analysis</Button>
          </Box>
          </>
        )}
      </Box>
      );
      })()}

      <Divider sx={{ my: 1 }} />

      {/* ── PROJECT ───────────────────────────────────── */}
      <SectionTitle>Project</SectionTitle>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 1, px: 1.5 }}>
        <Button
          fullWidth
          variant="contained"
          color="primary"
          onClick={async () => {
            const now = new Date();
            const ts = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,"0")}${String(now.getDate()).padStart(2,"0")}_${String(now.getHours()).padStart(2,"0")}${String(now.getMinutes()).padStart(2,"0")}${String(now.getSeconds()).padStart(2,"0")}`;
            const picked = await saveProjectDialog({ defaultPath: `${ts}_project.mpf` });
            if (picked) await saveProject(picked);
          }}
        >
          Save Project
        </Button>
        <Button
          fullWidth
          variant="outlined"
          onClick={() => {
            setProjectPath("");
            setLoadDialogOpen(true);
          }}
        >
          Load Project
        </Button>
      </Box>

      {/* ── Grid Resize Warning Dialog ───────────────── */}
      <Dialog open={resizeWarningOpen} onClose={() => setResizeWarningOpen(false)}>
        <DialogTitle>Panels with images will be removed</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The following panels have images assigned and will be lost if you resize the grid:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, pl: 2, maxHeight: 200, overflowY: "auto" }}>
            {resizeWarningConflicts.map((c, i) => (
              <li key={i}>
                <Typography variant="body2">{c}</Typography>
              </li>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResizeWarningOpen(false)}>Cancel</Button>
          <Button onClick={confirmResize} color="error" variant="contained">
            Resize Anyway
          </Button>
        </DialogActions>
      </Dialog>

      {/* ── Global Font Override Warning Dialog ─────── */}
      <Dialog open={fontWarningOpen} onClose={() => setFontWarningOpen(false)}>
        <DialogTitle>Override custom fonts?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The following elements have a custom font set that will be overridden by the global font:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, pl: 2, maxHeight: 200, overflowY: "auto" }}>
            {fontWarningConflicts.map((c, i) => (
              <li key={i}>
                <Typography variant="body2">{c}</Typography>
              </li>
            ))}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setFontWarningOpen(false)}>Cancel</Button>
          <Button onClick={confirmGlobalFont} color="error" variant="contained">
            Apply Anyway
          </Button>
        </DialogActions>
      </Dialog>

      {/* ── Load Project Dialog ──────────────────────── */}
      <Dialog
        open={loadDialogOpen}
        onClose={(_, reason) => {
          // Don't allow backdrop or ESC dismissal mid-load — the user
          // would land in a half-applied state with no recourse.
          if (loadingProject) return;
          if (reason === "backdropClick" || reason === "escapeKeyDown") {
            setLoadDialogOpen(false);
          } else {
            setLoadDialogOpen(false);
          }
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Load Project</DialogTitle>
        <DialogContent>
          <Box>
            <Box sx={{ display: "flex", gap: 1, mt: 1, alignItems: "center" }}>
              <TextField
                autoFocus fullWidth size="small"
                label="File path"
                value={projectPath}
                disabled={loadingProject}
                onChange={(e) => setProjectPath(e.target.value)}
              />
              <Button
                variant="outlined" size="small"
                sx={{ minWidth: 80, flexShrink: 0 }}
                disabled={loadingProject}
                onClick={async () => {
                  try {
                    const { open } = await import("@tauri-apps/plugin-dialog");
                    const selected = await open({
                      filters: [{ name: "Project", extensions: ["mpf"] }],
                      multiple: false,
                    });
                    if (selected) { setProjectPath(typeof selected === "string" ? selected : (selected as any).path || ""); return; }
                  } catch { /* not in Tauri */ }
                  setProjectPath("~/Documents/");
                }}
              >Browse</Button>
            </Box>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: 1.5, mt: 0.25, display: "block", fontSize: "0.65rem" }}>
              Enter full path to .mpf file. In web preview, Browse pre-fills ~/Documents/.
            </Typography>
            {/* Loading indicator. Indeterminate because the backend's
                /api/project/load is one-shot — once we have streaming
                progress events for project load, swap to determinate. */}
            {loadingProject && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mt: 2, px: 1, py: 1.5, bgcolor: "action.hover", borderRadius: 1 }}>
                <CircularProgress size={20} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="caption" sx={{ display: "block", fontWeight: 600 }}>
                    Loading project…
                  </Typography>
                  <Typography variant="caption" sx={{ display: "block", color: "text.secondary", fontSize: "0.65rem" }}>
                    Decompressing images, restoring config, rendering preview.
                    Larger files take longer.
                  </Typography>
                </Box>
              </Box>
            )}
            {loadError && !loadingProject && (
              <Typography variant="caption" sx={{ display: "block", color: "error.main", mt: 1.5, fontSize: "0.65rem" }}>
                {loadError}
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            disabled={loadingProject}
            onClick={() => {
              setLoadDialogOpen(false);
              setProjectPath("");
              setLoadError(null);
            }}
          >Cancel</Button>
          <Button
            variant="contained"
            disabled={!projectPath || loadingProject}
            onClick={async () => {
              if (!projectPath) return;
              setLoadError(null);
              setLoadingProject(true);
              try {
                // Open into a (new or existing) document tab, guarding
                // unsaved changes in the current builder doc first.
                await openProjectIntoTab(projectPath);
                setLoadDialogOpen(false);
                setProjectPath("");
              } catch (e) {
                console.error("[load project] failed:", e);
                setLoadError(e instanceof Error ? e.message : String(e));
              } finally {
                setLoadingProject(false);
              }
            }}
          >
            {loadingProject ? "Loading…" : "Load"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
