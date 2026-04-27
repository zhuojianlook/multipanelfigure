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
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import DeleteIcon from "@mui/icons-material/Delete";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import StraightenIcon from "@mui/icons-material/Straighten";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { api } from "../../api/client";
import { AnalysisDialog } from "../dialogs/AnalysisDialog";

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

/* CollageSidebar — collage-specific tools. Currently:
     • Global header point size — a single pt setting that, when
       applied, re-renders every figure-kind item in the collage
       with header sizes compensated for that item's collage scale,
       so all figures show headers at the same visual size.
   Builder-side hooks are NOT mounted here, so the builder doesn't
   thrash the API while the user is in collage mode. */
function CollageSidebar() {
  const items = useCollageStore((s) => s.items);
  const globalHeaderPt = useCollageStore((s) => s.globalHeaderPt);
  const autoRenderOnResize = useCollageStore((s) => s.autoRenderOnResize);
  const setGlobalHeaderPt = useCollageStore((s) => s.setGlobalHeaderPt);
  const setAutoRenderOnResize = useCollageStore((s) => s.setAutoRenderOnResize);
  const updateItem = useCollageStore((s) => s.updateItem);
  const [applyBusy, setApplyBusy] = useState(false);
  const [pendingPt, setPendingPt] = useState<number>(globalHeaderPt ?? 12);
  const figureItemCount = items.filter((it) => it.kind === "figure" && it.projectPath).length;

  const applyNow = async () => {
    if (figureItemCount === 0) {
      window.alert("No figure items in the collage to re-render. Add a figure with a saved .mpf path first.");
      return;
    }
    setApplyBusy(true);
    let succeeded = 0;
    let failed = 0;
    try {
      for (const it of useCollageStore.getState().items) {
        if (it.kind !== "figure" || !it.projectPath) continue;
        const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
        try {
          const resp = await api.renderCollageFigure(it.projectPath, pendingPt, Math.max(0.001, scale), it.w);
          if (resp?.image && resp.width && resp.height) {
            // The new render's aspect ratio almost always differs
            // from the old one because matplotlib grows fig_h to
            // accommodate larger headers. Preserve item.w as the
            // user's chosen footprint, and recompute item.h from the
            // new aspect — otherwise objectFit:"fill" stretches the
            // PNG and the headers come out distorted (this was the
            // "headers don't end up the same size" bug).
            const newAspect = resp.width / resp.height;
            const newH = it.w / newAspect;
            updateItem(it.id, {
              src: `data:image/png;base64,${resp.image}`,
              naturalW: resp.width,
              naturalH: resp.height,
              h: newH,
            });
            succeeded++;
          } else {
            failed++;
          }
        } catch (e) {
          console.error("[collage] render with header override failed for", it.name, e);
          failed++;
        }
      }
      setGlobalHeaderPt(pendingPt);
    } finally {
      setApplyBusy(false);
    }
    window.alert(
      `Re-rendered ${succeeded} figure${succeeded === 1 ? "" : "s"} at ${pendingPt} pt header size` +
      (failed > 0 ? ` (${failed} failed — check console; the .mpf may have been moved or the path is wrong).` : "."),
    );
  };

  const clearOverride = () => {
    setGlobalHeaderPt(null);
    window.alert(
      "Global header size cleared. Existing collage items keep their last rendered preview — re-add or update individual figures to revert to their saved header sizes.",
    );
  };

  return (
    <Box sx={{ p: 2, display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Typography variant="caption" sx={{ display: "block", letterSpacing: 1.2, fontSize: "0.6rem", textTransform: "uppercase", color: "text.secondary" }}>
        Collage Tools
      </Typography>

      <Box>
        <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary", mb: 0.5, display: "block" }}>
          Global header size (pt)
        </Typography>
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", mb: 1, display: "block", lineHeight: 1.4 }}>
          Re-renders every figure-kind item so its headers / primary labels
          appear at this point size after the collage downscale, regardless
          of the figure's individual scale. Run again after resizing items.
        </Typography>
        <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", mb: 1 }}>
          <TextField
            type="number"
            size="small"
            value={pendingPt}
            onChange={(e) => setPendingPt(Math.max(1, Math.min(200, Number(e.target.value) || 12)))}
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
            disabled={applyBusy || figureItemCount === 0}
            onClick={applyNow}
            sx={{ fontSize: "0.65rem", textTransform: "none", flex: 1 }}
          >
            {applyBusy ? "Rendering…" : `Apply to ${figureItemCount} fig${figureItemCount === 1 ? "" : "s"}`}
          </Button>
        </Box>
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", display: "block" }}>
          {globalHeaderPt
            ? `Currently locked: ${globalHeaderPt} pt`
            : "Currently free (each figure uses its own header sizes)"}
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

        {/* Auto-re-render toggle. Meaningful only when a header lock
            is set; we leave it visible regardless so the user can
            pre-configure the behaviour before locking. */}
        <Box sx={{ display: "flex", alignItems: "center", mt: 1, gap: 0.5 }}>
          <input
            type="checkbox"
            id="collage-auto-rerender"
            checked={autoRenderOnResize}
            onChange={(e) => setAutoRenderOnResize(e.target.checked)}
            style={{ accentColor: "#4FC3F7", cursor: "pointer" }}
          />
          <Typography
            component="label"
            htmlFor="collage-auto-rerender"
            variant="caption"
            sx={{ fontSize: "0.65rem", cursor: "pointer", lineHeight: 1.3 }}
          >
            Re-render on resize
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary", display: "block", lineHeight: 1.3, mt: 0.25 }}>
          When the lock is set, dragging a figure's resize handle re-renders
          it automatically so the global pt size keeps holding through
          scale changes.
        </Typography>
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
  const loadProject = useFigureStore((s) => s.loadProject);
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

  // Save/Load project dialogs
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [projectPath, setProjectPath] = useState("");

  // Analysis dialog
  const [analysisOpen, setAnalysisOpen] = useState(false);
  // When true, the user enters the scale-bar value as pixels per unit and we
  // invert it to units per pixel before storing.
  const [scaleInverted, setScaleInverted] = useState(false);

  // Computed measurements from backend
  const [computedMeasurements, setComputedMeasurements] = useState<Array<{ panel: string; name: string; type: string; value: string }>>([]);
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

      <Divider sx={{ my: 1 }} />

      {/* ── SCALE BARS (Resolution Presets) ────────────── */}
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", pr: 1.5 }}>
        <SectionTitle>Scale Bars</SectionTitle>
        <Button
          size="small"
          variant="text"
          onClick={async () => {
            if (!window.confirm("Replace all scale-bar presets with the bundled microscope defaults? Custom entries will be lost.")) return;
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
          <IconButton size="small" onClick={() => {
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
              const entries = { ...config.resolution_entries, [key]: valueInUm };
              setConfig({ ...config, resolution_entries: entries });
              api.updateResolutions(entries).catch(console.error);
              nameEl.value = "";
              valEl.value = "";
            }
          }} sx={{ width: 22, height: 22 }}>
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
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
              <Box key={name} sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <Typography variant="caption" sx={{ fontSize: "0.65rem" }}>{displayName}: {Number(displayVal.toPrecision(6))} {unitLabel}/px</Typography>
                <IconButton size="small" onClick={() => {
                  const entries = { ...config.resolution_entries };
                  delete entries[name];
                  setConfig({ ...config, resolution_entries: entries });
                  api.updateResolutions(entries).catch(console.error);
                }} sx={{ width: 20, height: 20 }}>
                  <DeleteIcon sx={{ fontSize: 12 }} />
                </IconButton>
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
              const existing = config?.column_labels?.[0]?.font_name;
              if (existing && fonts.includes(existing)) return existing;
              // Case-insensitive Arial match (Windows "arial.ttf",
              // macOS "Arial.ttf", Linux variants — the installed
              // filename varies by OS).
              const arial = fonts.find((f) =>
                /^arial\b/i.test(f.replace(/\.(ttf|otf|ttc)$/i, "")),
              );
              if (arial) return arial;
              return fonts[0] || "";
            })()}
            key={fonts.length > 0 ? `loaded-${fonts.length}` : "empty"}
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
              const updated = { ...config, normalize_widths: e.target.checked };
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
              value={config.normalize_mode ?? "width"}
              onChange={(e) => {
                const updated = { ...config, normalize_mode: e.target.value };
                setConfig(updated);
                api.updateConfig(updated).catch(console.error);
              }}
              style={{ fontSize: "0.65rem", backgroundColor: "var(--c-surface2)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "2px 4px" }}
            >
              <option value="width">Match width</option>
              <option value="height">Match height</option>
            </select>
          </Box>
        )}
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── ANALYSIS ───────────────────────────────────── */}
      <SectionTitle>Analysis</SectionTitle>
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
        {computedMeasurements.length === 0 ? (
          <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled" }}>
            No measurements. Add lines/areas with "measure" enabled in panel settings.
          </Typography>
        ) : (
          <>
          {/* Group measurements by panel */}
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
          <Box sx={{ display: "flex", gap: 0.5 }}>
            <Button size="small" variant="text" sx={{ fontSize: "0.55rem", textTransform: "none", flex: 1 }}
              onClick={() => {
                const csv = ["Panel,Name,Type,Value", ...computedMeasurements.map(m => `${m.panel},${m.name},${m.type},${m.value}`)].join("\n");
                const blob = new Blob([csv], { type: "text/csv" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url; a.download = "measurements.csv"; a.click();
                URL.revokeObjectURL(url);
              }}
            >Export CSV</Button>
            <Button size="small" variant="outlined" sx={{ fontSize: "0.55rem", textTransform: "none", flex: 1 }}
              onClick={() => setAnalysisOpen(true)}
            >Open Analysis</Button>
          </Box>
          </>
        )}
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ── PROJECT ───────────────────────────────────── */}
      <SectionTitle>Project</SectionTitle>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 1, px: 1.5 }}>
        <Button
          fullWidth
          variant="contained"
          color="primary"
          onClick={() => {
            const now = new Date();
            const ts = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,"0")}${String(now.getDate()).padStart(2,"0")}_${String(now.getHours()).padStart(2,"0")}${String(now.getMinutes()).padStart(2,"0")}${String(now.getSeconds()).padStart(2,"0")}`;
            setProjectPath(`${ts}_project.mpf`);
            setSaveDialogOpen(true);
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

      {/* ── Save Project Dialog ──────────────────────── */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Save Project</DialogTitle>
        <DialogContent>
          <Box>
            <Box sx={{ display: "flex", gap: 1, mt: 1, alignItems: "center" }}>
              <TextField
                autoFocus fullWidth size="small"
                label="File path"
                value={projectPath}
                onChange={(e) => setProjectPath(e.target.value)}
              />
              <Button variant="outlined" size="small" sx={{ minWidth: 80, flexShrink: 0 }}
                onClick={async () => {
                  try {
                    const { save } = await import("@tauri-apps/plugin-dialog");
                    const selected = await save({
                      defaultPath: projectPath || "project.mpf",
                      filters: [{ name: "Project", extensions: ["mpf"] }],
                    });
                    if (selected) { setProjectPath(selected); return; }
                  } catch { /* not in Tauri — web fallback */ }
                  const fname = (projectPath || "project.mpf").split("/").pop() || "project.mpf";
                  setProjectPath(`~/Documents/${fname}`);
                }}
              >Browse</Button>
            </Box>
            <Typography variant="caption" sx={{ color: "text.secondary", ml: 1.5, mt: 0.25, display: "block", fontSize: "0.65rem" }}>
              Enter full path. In web preview, Browse pre-fills ~/Documents/.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setSaveDialogOpen(false); setProjectPath(""); }}>Cancel</Button>
          <Button
            variant="contained"
            disabled={!projectPath}
            onClick={async () => {
              if (projectPath) {
                await saveProject(projectPath);
                setSaveDialogOpen(false);
                setProjectPath("");
              }
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

      {/* ── Load Project Dialog ──────────────────────── */}
      <Dialog open={loadDialogOpen} onClose={() => setLoadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Load Project</DialogTitle>
        <DialogContent>
          <Box>
            <Box sx={{ display: "flex", gap: 1, mt: 1, alignItems: "center" }}>
              <TextField
                autoFocus fullWidth size="small"
                label="File path"
                value={projectPath}
                onChange={(e) => setProjectPath(e.target.value)}
              />
              <Button variant="outlined" size="small" sx={{ minWidth: 80, flexShrink: 0 }}
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
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setLoadDialogOpen(false); setProjectPath(""); }}>Cancel</Button>
          <Button
            variant="contained"
            disabled={!projectPath}
            onClick={async () => {
              if (projectPath) {
                await loadProject(projectPath);
                setLoadDialogOpen(false);
              }
            }}
          >
            Load
          </Button>
        </DialogActions>
      </Dialog>

      {/* Analysis Dialog */}
      <AnalysisDialog
        open={analysisOpen}
        onClose={() => setAnalysisOpen(false)}
        measurements={computedMeasurements}
      />
    </Box>
  );
}
