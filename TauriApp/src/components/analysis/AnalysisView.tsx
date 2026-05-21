/* ──────────────────────────────────────────────────────────
   AnalysisView — the Analysis workspace rendered inline as a
   permanent top-level tab (no longer a modal). Wraps the node-graph
   canvas so the user can build analysis workflows at any time, even
   with no measurements yet.

   Measurements are pulled from the active builder figure (and the
   backend computes them from the live config). When there are none,
   the canvas still renders so workflows can be authored from scratch.
   ────────────────────────────────────────────────────────── */

import { useEffect, useState, useCallback } from "react";
import { Box } from "@mui/material";
import { AnalysisNodeGraph } from "../dialogs/AnalysisNodeGraph";
import { useFigureStore } from "../../store/figureStore";
import { api } from "../../api/client";

export function AnalysisView() {
  // Re-fetch measurements whenever the active config changes so the
  // analysis source reflects the current figure.
  const config = useFigureStore((s) => s.config);
  const [measurementsCsv, setMeasurementsCsv] = useState<string>("Panel,Name,Group,Value,Unit\n");

  const refresh = useCallback(async () => {
    try {
      const { measurements } = await api.getMeasurements();
      const header = "Panel,Name,Group,Value,Unit";
      const body = (measurements || [])
        .map((m) => {
          const value = m.numeric != null ? String(m.numeric) : (m.value ?? "");
          // Group defaults to the panel so the stats blocks have something
          // to split on; the user re-assigns groups in their code.
          return [m.panel, m.name, m.panel, value, m.unit ?? ""]
            .map((c) => `"${String(c).replace(/"/g, '""')}"`)
            .join(",");
        })
        .join("\n");
      setMeasurementsCsv(body ? `${header}\n${body}` : `${header}\n`);
    } catch {
      setMeasurementsCsv("Panel,Name,Group,Value,Unit\n");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh, config]);

  return (
    <Box sx={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <AnalysisNodeGraph open measurementsCsv={measurementsCsv} />
    </Box>
  );
}
