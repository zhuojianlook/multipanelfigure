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
  max = 20,
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

export function Sidebar() {
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

  // Save/Load project dialogs
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [projectPath, setProjectPath] = useState("");

  // Analysis dialog
  const [analysisOpen, setAnalysisOpen] = useState(false);

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

  const spacingPx = Math.round((config.spacing || 0.02) * 2000);

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
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: 140 }}>
          <Slider
            value={spacingPx}
            min={0}
            max={100}
            onChange={(_, val) => setSpacing((val as number) / 2000)}
            sx={{ flex: 1 }}
          />
          <Typography variant="caption" sx={{ width: 40, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
            {spacingPx} px
          </Typography>
        </Box>
      </Field>

      <Divider sx={{ my: 1 }} />

      {/* ── SCALE BARS (Resolution Presets) ────────────── */}
      <SectionTitle>Scale Bars</SectionTitle>
      <Box sx={{ px: 1.5, display: "flex", flexDirection: "column", gap: 0.5 }}>
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
        {/* Add new scale bar: Name, Value, Unit */}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, alignItems: "center" }}>
          <TextField
            placeholder="Name"
            size="small"
            sx={{ flex: "1 1 50px", minWidth: 50, "& input": { fontSize: "0.65rem", py: 0.25, px: 0.5 } }}
            id="scale-name-input"
          />
          <TextField
            placeholder="val/px"
            type="number"
            size="small"
            sx={{
              flex: "0 0 50px", width: 50,
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
          <IconButton size="small" onClick={() => {
            const nameEl = document.getElementById("scale-name-input") as HTMLInputElement;
            const valEl = document.getElementById("scale-value-input") as HTMLInputElement;
            const unitEl = document.getElementById("scale-unit-input") as HTMLSelectElement;
            if (nameEl?.value && valEl?.value) {
              const unit = unitEl?.value || "um";
              // Convert to μm/px for internal storage, store unit in key suffix
              const conversionToUm: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
              const valueInUm = Number(valEl.value) * (conversionToUm[unit] || 1);
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
            defaultValue={fonts.length > 0 ? (config?.column_labels?.[0]?.font_name || "arial.ttf") : ""}
            key={fonts.length > 0 ? "loaded" : "empty"}
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
              const colLabels = config.column_labels.map((l: any) => ({ ...l, font_name: fontName }));
              const rowLabels = config.row_labels.map((l: any) => ({ ...l, font_name: fontName }));
              const colHeaders = config.column_headers.map((level: any) => ({
                ...level,
                headers: level.headers.map((h: any) => ({ ...h, font_name: fontName })),
              }));
              const rowHeaders = config.row_headers.map((level: any) => ({
                ...level,
                headers: level.headers.map((h: any) => ({ ...h, font_name: fontName })),
              }));
              setConfig({ ...config, column_labels: colLabels, row_labels: rowLabels, column_headers: colHeaders, row_headers: rowHeaders });
            }}
          >
            Apply
          </Button>
          </Box>
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
