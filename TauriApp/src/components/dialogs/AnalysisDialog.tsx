/* ──────────────────────────────────────────────────────────
   AnalysisDialog — Full-screen dialog for R-based analysis.
   Three panels: Data | R Code | Plots
   ────────────────────────────────────────────────────────── */

import { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
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
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { api } from "../../api/client";

interface Measurement {
  panel: string;
  name: string;
  type: string;
  value: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  measurements: Measurement[];
}

// ── R Code Presets ────────────────────────────────────────

const R_PRESETS: Record<string, { label: string; code: string }> = {
  bar: {
    label: "Bar Graph",
    code: `library(ggplot2)

mpfig_plot("bar_chart.png")
ggplot(data, aes(x=Panel, y=as.numeric(gsub("[^0-9.]", "", Value)), fill=Name)) +
  geom_bar(stat="identity", position="dodge") +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white"),
    legend.position = "bottom"
  ) +
  labs(title="Measurements by Panel", y="Value", x="Panel")
`,
  },
  line: {
    label: "Line Graph",
    code: `library(ggplot2)

mpfig_plot("line_chart.png")
ggplot(data, aes(x=Panel, y=as.numeric(gsub("[^0-9.]", "", Value)), group=Name, color=Name)) +
  geom_line(linewidth=1) +
  geom_point(size=3) +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white"),
    legend.position = "bottom"
  ) +
  labs(title="Measurements", y="Value", x="Panel")
`,
  },
  box: {
    label: "Box Plot",
    code: `library(ggplot2)

mpfig_plot("box_plot.png")
ggplot(data, aes(x=Panel, y=as.numeric(gsub("[^0-9.]", "", Value)))) +
  geom_boxplot(fill="#4FC3F7", alpha=0.7) +
  geom_jitter(width=0.2, alpha=0.5) +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white")
  ) +
  labs(title="Distribution by Panel", y="Value", x="Panel")
`,
  },
  scatter: {
    label: "Scatter Plot",
    code: `library(ggplot2)

mpfig_plot("scatter.png")
data$NumValue <- as.numeric(gsub("[^0-9.]", "", data$Value))
ggplot(data, aes(x=seq_along(NumValue), y=NumValue, color=Panel)) +
  geom_point(size=3) +
  geom_smooth(method="lm", se=TRUE, alpha=0.2) +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white"),
    legend.position = "bottom"
  ) +
  labs(title="Scatter Plot", y="Value", x="Index")
`,
  },
  anova: {
    label: "ANOVA",
    code: `# One-way ANOVA: compare means across panels
data$NumValue <- as.numeric(gsub("[^0-9.]", "", data$Value))
model <- aov(NumValue ~ Panel, data=data)
cat("\\n=== ANOVA Summary ===\\n")
print(summary(model))
cat("\\n=== Tukey HSD Post-hoc ===\\n")
print(TukeyHSD(model))

# Plot
library(ggplot2)
mpfig_plot("anova.png")
ggplot(data, aes(x=Panel, y=NumValue)) +
  geom_boxplot(fill="#4FC3F7", alpha=0.7) +
  geom_jitter(width=0.2, alpha=0.5) +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white")
  ) +
  labs(title="ANOVA: Measurements by Panel", y="Value", x="Panel")
`,
  },
  ttest: {
    label: "T-Test",
    code: `# Welch's t-test between first two panels
data$NumValue <- as.numeric(gsub("[^0-9.]", "", data$Value))
panels <- unique(data$Panel)
if (length(panels) >= 2) {
  g1 <- data$NumValue[data$Panel == panels[1]]
  g2 <- data$NumValue[data$Panel == panels[2]]
  cat("\\n=== Welch's T-Test ===\\n")
  cat(paste("Group 1:", panels[1], "  Group 2:", panels[2], "\\n"))
  print(t.test(g1, g2))
} else {
  cat("Need at least 2 panels for t-test\\n")
}

library(ggplot2)
mpfig_plot("ttest.png")
ggplot(data, aes(x=Panel, y=NumValue, fill=Panel)) +
  geom_boxplot(alpha=0.7) +
  geom_jitter(width=0.2, alpha=0.5) +
  theme_minimal(base_size=14) +
  theme(
    panel.background = element_rect(fill="white"),
    plot.background = element_rect(fill="white"),
    legend.position = "none"
  ) +
  labs(title="T-Test Comparison", y="Value", x="Panel")
`,
  },
};

export function AnalysisDialog({ open, onClose, measurements }: Props) {
  // R status
  const [rInstalled, setRInstalled] = useState<boolean | null>(null);
  const [rVersion, setRVersion] = useState("");

  // Data
  const [selectedPanels, setSelectedPanels] = useState<Set<string>>(new Set());
  const [manualRows, setManualRows] = useState<string[][]>([]);
  const [manualCols, setManualCols] = useState<string[]>(["Panel", "Name", "Type", "Value"]);

  // R code
  const [preset, setPreset] = useState("bar");
  const [code, setCode] = useState(R_PRESETS.bar.code);
  const [running, setRunning] = useState(false);

  // Results
  const [stdout, setStdout] = useState("");
  const [stderr, setStderr] = useState("");
  const [plots, setPlots] = useState<string[]>([]);
  const [selectedPlot, setSelectedPlot] = useState<number | null>(null);

  // Check R on open
  useEffect(() => {
    if (open) {
      api.checkR().then((r) => {
        setRInstalled(r.installed);
        setRVersion(r.version);
      }).catch(() => setRInstalled(false));

      // Auto-select all panels
      const panels = new Set(measurements.map(m => m.panel));
      setSelectedPanels(panels);
    }
  }, [open, measurements]);

  // Build CSV from selected data
  const buildCsv = useCallback(() => {
    const rows: string[][] = [];
    // From annotations
    measurements.forEach(m => {
      if (selectedPanels.has(m.panel)) {
        rows.push([m.panel, m.name, m.type, m.value]);
      }
    });
    // From manual entries
    manualRows.forEach(row => {
      if (row.some(cell => cell.trim())) rows.push(row);
    });
    const header = manualCols.join(",");
    const body = rows.map(r => r.map(c => `"${c.replace(/"/g, '""')}"`).join(",")).join("\n");
    return `${header}\n${body}`;
  }, [measurements, selectedPanels, manualRows, manualCols]);

  const handleRun = async () => {
    setRunning(true);
    setStdout("");
    setStderr("");
    setPlots([]);
    setSelectedPlot(null);
    try {
      const csv = buildCsv();
      const result = await api.runR(code, csv);
      setStdout(result.stdout);
      setStderr(result.stderr);
      setPlots(result.plots);
    } catch (err) {
      setStderr(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const handlePresetChange = (key: string) => {
    setPreset(key);
    setCode(R_PRESETS[key]?.code ?? "");
  };

  const togglePanel = (panel: string) => {
    setSelectedPanels(prev => {
      const next = new Set(prev);
      if (next.has(panel)) next.delete(panel);
      else next.add(panel);
      return next;
    });
  };

  const addManualRow = () => {
    setManualRows(prev => [...prev, manualCols.map(() => "")]);
  };

  const updateManualCell = (rowIdx: number, colIdx: number, value: string) => {
    setManualRows(prev => {
      const next = [...prev];
      next[rowIdx] = [...next[rowIdx]];
      next[rowIdx][colIdx] = value;
      return next;
    });
  };

  const deleteManualRow = (rowIdx: number) => {
    setManualRows(prev => prev.filter((_, i) => i !== rowIdx));
  };

  const savePlot = async (plotB64: string, index: number) => {
    // Download via browser data URL
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${plotB64}`;
    link.download = `analysis_plot_${index + 1}.png`;
    link.click();
  };

  const panels = [...new Set(measurements.map(m => m.panel))];

  return (
    <Dialog open={open} onClose={onClose} fullScreen>
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", py: 1, px: 2, bgcolor: "background.paper" }}>
        <Typography variant="h6" sx={{ fontSize: "1rem", fontWeight: 700 }}>Analysis</Typography>
        <IconButton onClick={onClose} size="small"><CloseIcon /></IconButton>
      </DialogTitle>
      <DialogContent sx={{ p: 0, display: "flex", height: "100%", overflow: "hidden" }}>
        {/* ── Left: Data Panel ── */}
        <Box sx={{ width: 280, flexShrink: 0, borderRight: 1, borderColor: "divider", overflow: "auto", p: 1.5, display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Data Source</Typography>

          {/* Panel checkboxes */}
          <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>From Annotations:</Typography>
          {panels.length === 0 ? (
            <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled", fontStyle: "italic" }}>No measurements available</Typography>
          ) : (
            panels.map(p => (
              <FormControlLabel key={p} sx={{ ml: 0, "& .MuiTypography-root": { fontSize: "0.65rem" } }}
                control={<Checkbox size="small" checked={selectedPanels.has(p)} onChange={() => togglePanel(p)} sx={{ p: 0.25 }} />}
                label={`${p} (${measurements.filter(m => m.panel === p).length} items)`}
              />
            ))
          )}

          <Divider sx={{ my: 0.5 }} />

          {/* Manual data entry */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>Manual Data:</Typography>
            <IconButton size="small" onClick={addManualRow} title="Add row"><AddIcon sx={{ fontSize: 14 }} /></IconButton>
          </Box>

          {manualRows.length > 0 && (
            <Box sx={{ overflowX: "auto" }}>
              <table style={{ fontSize: "0.6rem", borderCollapse: "collapse", width: "100%" }}>
                <thead>
                  <tr>{manualCols.map((col, ci) => (
                    <th key={ci} style={{ padding: "2px 4px", borderBottom: "1px solid var(--c-border)", textAlign: "left", fontSize: "0.55rem" }}>{col}</th>
                  ))}<th></th></tr>
                </thead>
                <tbody>
                  {manualRows.map((row, ri) => (
                    <tr key={ri}>
                      {row.map((cell, ci) => (
                        <td key={ci} style={{ padding: "1px 2px" }}>
                          <input
                            value={cell}
                            onChange={(e) => updateManualCell(ri, ci, e.target.value)}
                            style={{ width: "100%", fontSize: "0.55rem", background: "var(--c-surface2)", color: "var(--c-text)", border: "1px solid var(--c-border)", borderRadius: 2, padding: "1px 3px" }}
                          />
                        </td>
                      ))}
                      <td><IconButton size="small" onClick={() => deleteManualRow(ri)} sx={{ p: 0.1 }}><DeleteOutlineIcon sx={{ fontSize: 10 }} /></IconButton></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Box>
          )}

          <Divider sx={{ my: 0.5 }} />

          {/* Data preview */}
          <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>Preview ({measurements.filter(m => selectedPanels.has(m.panel)).length + manualRows.filter(r => r.some(c => c.trim())).length} rows):</Typography>
          <Box sx={{ maxHeight: 120, overflowY: "auto", fontSize: "0.55rem", fontFamily: "monospace", bgcolor: "background.default", p: 0.5, borderRadius: 1 }}>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{buildCsv()}</pre>
          </Box>

          <Button size="small" variant="text" sx={{ fontSize: "0.55rem", textTransform: "none" }}
            onClick={() => {
              const csv = buildCsv();
              const blob = new Blob([csv], { type: "text/csv" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url; a.download = "analysis_data.csv"; a.click();
              URL.revokeObjectURL(url);
            }}
          >Export CSV</Button>
        </Box>

        {/* ── Center: R Code ── */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", borderRight: 1, borderColor: "divider", overflow: "hidden" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 1, borderBottom: 1, borderColor: "divider" }}>
            <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>R Code</Typography>
            <Select
              size="small"
              value={preset}
              onChange={(e) => handlePresetChange(e.target.value)}
              sx={{ fontSize: "0.65rem", minWidth: 120, "& .MuiSelect-select": { py: 0.25, px: 1 } }}
            >
              {Object.entries(R_PRESETS).map(([key, { label }]) => (
                <MenuItem key={key} value={key} sx={{ fontSize: "0.65rem" }}>{label}</MenuItem>
              ))}
            </Select>
            <Box sx={{ flex: 1 }} />
            {rInstalled === false && (
              <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "error.main" }}>R not installed</Typography>
            )}
            {rInstalled && rVersion && (
              <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "text.secondary" }}>{rVersion}</Typography>
            )}
            <Button
              size="small"
              variant="contained"
              startIcon={running ? <CircularProgress size={12} /> : <PlayArrowIcon sx={{ fontSize: 14 }} />}
              disabled={running || !rInstalled}
              onClick={handleRun}
              sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}
            >
              {running ? "Running..." : "Run"}
            </Button>
          </Box>

          {/* Code editor */}
          <TextField
            multiline
            fullWidth
            value={code}
            onChange={(e) => setCode(e.target.value)}
            sx={{
              flex: 1,
              "& .MuiOutlinedInput-root": { height: "100%", alignItems: "flex-start", borderRadius: 0 },
              "& .MuiOutlinedInput-notchedOutline": { border: "none" },
              "& textarea": { fontFamily: "monospace", fontSize: "0.7rem", lineHeight: 1.5 },
            }}
          />

          {/* Console output */}
          {(stdout || stderr) && (
            <Box sx={{ maxHeight: 200, overflowY: "auto", borderTop: 1, borderColor: "divider", p: 1, bgcolor: "background.default" }}>
              <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.6rem", mb: 0.5, display: "block" }}>Console Output</Typography>
              {stdout && <pre style={{ margin: 0, fontSize: "0.6rem", fontFamily: "monospace", whiteSpace: "pre-wrap", color: "var(--c-text)" }}>{stdout}</pre>}
              {stderr && <pre style={{ margin: 0, fontSize: "0.6rem", fontFamily: "monospace", whiteSpace: "pre-wrap", color: "#f44336" }}>{stderr}</pre>}
            </Box>
          )}
        </Box>

        {/* ── Right: Plots ── */}
        <Box sx={{ width: 350, flexShrink: 0, overflow: "auto", p: 1.5, display: "flex", flexDirection: "column", gap: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>Plots</Typography>

          {rInstalled === null && <CircularProgress size={20} />}
          {rInstalled === false && (
            <Alert severity="warning" sx={{ fontSize: "0.65rem" }}>
              R is not installed. Install R from <strong>https://cran.r-project.org/</strong> and ensure <code>Rscript</code> is on your PATH.
            </Alert>
          )}

          {plots.length === 0 && rInstalled && !running && (
            <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.disabled", fontStyle: "italic" }}>
              No plots yet. Select data and click Run.
            </Typography>
          )}

          {running && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <CircularProgress size={16} />
              <Typography variant="caption" sx={{ fontSize: "0.6rem" }}>Generating plots...</Typography>
            </Box>
          )}

          {/* Plot gallery */}
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
            {plots.map((plotB64, i) => (
              <Box key={i} sx={{ position: "relative", cursor: "pointer", border: 1, borderColor: selectedPlot === i ? "primary.main" : "divider", borderRadius: 1, overflow: "hidden" }}>
                <img
                  src={`data:image/png;base64,${plotB64}`}
                  alt={`Plot ${i + 1}`}
                  style={{ width: 150, height: "auto", display: "block" }}
                  onClick={() => setSelectedPlot(selectedPlot === i ? null : i)}
                />
                <IconButton
                  size="small"
                  sx={{ position: "absolute", top: 2, right: 2, bgcolor: "rgba(0,0,0,0.5)", "&:hover": { bgcolor: "rgba(0,0,0,0.7)" } }}
                  onClick={(e) => { e.stopPropagation(); savePlot(plotB64, i); }}
                  title="Save plot"
                >
                  <SaveAltIcon sx={{ fontSize: 12, color: "#fff" }} />
                </IconButton>
              </Box>
            ))}
          </Box>

          {/* Enlarged plot */}
          {selectedPlot !== null && plots[selectedPlot] && (
            <Box sx={{ mt: 1, border: 1, borderColor: "primary.main", borderRadius: 1, overflow: "hidden" }}>
              <img
                src={`data:image/png;base64,${plots[selectedPlot]}`}
                alt={`Plot ${selectedPlot + 1} (enlarged)`}
                style={{ width: "100%", height: "auto", display: "block" }}
              />
              <Box sx={{ display: "flex", gap: 0.5, p: 0.5 }}>
                <Button size="small" sx={{ fontSize: "0.55rem", textTransform: "none" }}
                  onClick={() => savePlot(plots[selectedPlot], selectedPlot)}
                  startIcon={<SaveAltIcon sx={{ fontSize: 12 }} />}
                >Save Plot</Button>
              </Box>
            </Box>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
}
