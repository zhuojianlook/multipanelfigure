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
import { AnalysisNodeGraph, MEAS_CSV_HEADER, measArrToCsv, figureLabels } from "../dialogs/AnalysisNodeGraph";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { api } from "../../api/client";

export function AnalysisView() {
  // Re-fetch measurements whenever the active config changes so the
  // analysis source reflects the current figure.
  const config = useFigureStore((s) => s.config);
  // The active figure's label — /api/measurements only ever serves the active
  // doc and its rows carry no figure identifier, so attribute here, where the
  // doc IS in hand. Panel labels are bare grid coords ("R1C1"), so without this
  // two figures' rows are indistinguishable downstream. (The node graph fetches
  // every OTHER open doc itself and labels those the same way.)
  //
  // Label, not name: DocTab.name has no uniqueness constraint, so two tabs can
  // both be "fig1" and a group-by would merge unrelated experiments.
  const openDocs = useCollageStore((s) => s.openDocs);
  const activeDocId = useCollageStore((s) => s.activeDocId);
  const activeDocLabel = figureLabels(openDocs || [])[activeDocId || ""] || "";

  const [measurementsCsv, setMeasurementsCsv] = useState<string>(`${MEAS_CSV_HEADER}\n`);
  const [measurements, setMeasurements] = useState<Array<Record<string, unknown>>>([]);

  const refresh = useCallback(async () => {
    try {
      const { measurements: rows } = await api.getMeasurements();
      // Shared with the node graph's per-figure folders, so the active figure's
      // table and every other figure's are built by the same code and can't
      // drift apart in column order or quoting.
      setMeasurementsCsv(measArrToCsv(
        (rows || []) as unknown as Array<Record<string, unknown>>,
        activeDocLabel,
      ));
      setMeasurements((rows || []) as unknown as Array<Record<string, unknown>>);
    } catch {
      setMeasurementsCsv(`${MEAS_CSV_HEADER}\n`);
      setMeasurements([]);
    }
  }, [activeDocLabel]);

  useEffect(() => { void refresh(); }, [refresh, config]);

  // Swallow EXTERNAL file drops anywhere in the Analysis tab. Without this the
  // WebView's default behavior on a file drop is to NAVIGATE to the file,
  // which replaces the whole SPA → the Analysis tab appears to crash. We only
  // intercept "Files" drags; the internal inset drags (application/x-mpfig-*)
  // pass straight through to the source nodes that handle them.
  const blockFileDrop = useCallback((e: React.DragEvent) => {
    if (Array.from(e.dataTransfer.types).includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
    }
  }, []);

  return (
    <Box
      sx={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}
      onDragOver={blockFileDrop}
      onDrop={blockFileDrop}
    >
      <AnalysisNodeGraph open measurementsCsv={measurementsCsv} measurements={measurements} />
    </Box>
  );
}
