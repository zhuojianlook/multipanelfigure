/**
 * CellposeSegDialog — interactive configuration + live segmentation preview
 * for the Cellpose node used by the "Cell characteristics" workflow.
 *
 * Mirrors the *experience* of the fluorescence Intensity picker (a modal with
 * parameter controls and a live outline preview), but is deliberately its own,
 * far smaller component: the Intensity picker is channel/threshold/compartment
 * centric, whereas a Cellpose node just needs model + diameter + channel +
 * threshold knobs and a "does this segmentation look right?" preview.
 *
 * It does NOT change what the node emits. On save it serialises the config to
 * the exact JSON the `kind:"cellpose"` node runs (`/api/analysis/run-cellpose`
 * → cellpose_plugin), so the downstream Cellpose → ImageJ shape-metrics → plot
 * pipeline (per-cell area / perimeter / circularity / equivalent diameter) is
 * untouched. The live preview posts the SAME config to
 * `/api/analysis/cellpose-preview`, which feeds the SAME plugin — so preview
 * matches the run.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Dialog, DialogTitle, DialogContent, DialogActions, Button, Box, Stack,
  Typography, Select, MenuItem, FormControlLabel, Switch, Slider, TextField,
  CircularProgress, Alert, IconButton, Divider,
} from "@mui/material";

/** One wired input image the dialog can preview. `source` (when present) lets
 *  the sidecar re-extract the full-res inset losslessly; otherwise the base64
 *  thumbnail is used. Structurally compatible with the node graph's
 *  FluorPickerImage so the shared picker-image builder feeds both dialogs. */
export type CellposePickerImage = {
  id: string;
  label: string;
  image_b64: string;
  // Opaque source descriptor (the node graph's FluorImageSource) — passed
  // straight through to the preview endpoint for full-res re-extraction.
  // Typed as `object` so the shared FluorPickerImage[] builder feeds this
  // dialog without a structural-mismatch cast.
  source?: object;
};

/** Structured, UI-friendly config stored on `node.data.cellposeSeg`. Serialised
 *  to the node's run JSON by `cellposeSegToJson`. */
export type CellposeSegConfig = {
  version: 1;
  /** Which plugin venv to run — v4 = cpsam only; v3 = cyto3/cyto2/nuclei zoo. */
  cellposeVersion: "v3" | "v4";
  model: string;
  /** true → let the network auto-estimate the object diameter (recommended). */
  autoDiameter: boolean;
  diameter: number;
  /** 0 = grayscale (use all channels), 1 = R, 2 = G, 3 = B. */
  segChannel: 0 | 1 | 2 | 3;
  /** 0 = none. Only meaningful for cyto models that accept a nuclei channel. */
  nucleiChannel: 0 | 1 | 2 | 3;
  flowThreshold: number;
  cellprobThreshold: number;
  minSize: number;
  useGpu: boolean;
};

export function emptyCellposeSegConfig(): CellposeSegConfig {
  return {
    version: 1,
    cellposeVersion: "v4",
    model: "cpsam",
    autoDiameter: true,
    diameter: 30,
    segChannel: 0,
    nucleiChannel: 0,
    flowThreshold: 0.4,
    cellprobThreshold: 0.0,
    minSize: 15,
    useGpu: true,
  };
}

const V4_MODELS = ["cpsam"];
const V3_MODELS = ["cyto3", "cyto2", "nuclei"];
const CHANNEL_LABELS: Record<number, string> = { 0: "Grayscale (all)", 1: "Red", 2: "Green", 3: "Blue" };

/** Serialise to the exact JSON the `kind:"cellpose"` node runs. The trailing
 *  comment lines are stripped by the run/preview parsers (they drop `//`/`#`
 *  lines before json.loads), so they're safe and act as inline help. */
export function cellposeSegToJson(c: CellposeSegConfig): string {
  const obj = {
    model: c.model,
    diameter: c.autoDiameter ? null : Math.max(0, Math.round(c.diameter)),
    channels: [c.segChannel, c.nucleiChannel],
    flow_threshold: c.flowThreshold,
    cellprob_threshold: c.cellprobThreshold,
    min_size: Math.max(0, Math.round(c.minSize)),
    use_gpu: c.useGpu,
    cellpose_version: c.cellposeVersion,
  };
  return (
    JSON.stringify(obj, null, 2) +
    "\n// Interactive Cellpose config — edit via the node's \"Configure segmentation…\" button.\n" +
    "// diameter null = auto-estimate. channels [seg, nuclei], 0 = grayscale/none."
  );
}

/** Best-effort parse of a node's saved JSON `code` back into a structured
 *  config, so reopening the dialog restores the user's settings even for nodes
 *  that predate the structured `cellposeSeg` field. */
export function cellposeSegFromJson(code: string | undefined): CellposeSegConfig {
  const base = emptyCellposeSegConfig();
  if (!code) return base;
  try {
    const stripped = code
      .split("\n")
      .filter((l) => !/^\s*(\/\/|#)/.test(l))
      .join("\n");
    const o = JSON.parse(stripped || "{}") as Record<string, unknown>;
    const ch = Array.isArray(o.channels) ? (o.channels as number[]) : [0, 0];
    const ver = (o.cellpose_version === "v3" ? "v3" : "v4") as "v3" | "v4";
    const diaRaw = o.diameter;
    return {
      ...base,
      cellposeVersion: ver,
      model: typeof o.model === "string" ? o.model : ver === "v3" ? "cyto3" : "cpsam",
      autoDiameter: diaRaw === null || diaRaw === undefined,
      diameter: typeof diaRaw === "number" && diaRaw > 0 ? diaRaw : base.diameter,
      segChannel: (([0, 1, 2, 3].includes(Number(ch[0])) ? Number(ch[0]) : 0) as 0 | 1 | 2 | 3),
      nucleiChannel: (([0, 1, 2, 3].includes(Number(ch[1])) ? Number(ch[1]) : 0) as 0 | 1 | 2 | 3),
      flowThreshold: typeof o.flow_threshold === "number" ? o.flow_threshold : base.flowThreshold,
      cellprobThreshold: typeof o.cellprob_threshold === "number" ? o.cellprob_threshold : base.cellprobThreshold,
      minSize: typeof o.min_size === "number" ? o.min_size : base.minSize,
      useGpu: typeof o.use_gpu === "boolean" ? o.use_gpu : base.useGpu,
    };
  } catch {
    return base;
  }
}

function apiBase(): string {
  const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
  return env?.VITE_API_BASE || env?.VITE_API || "http://127.0.0.1:8765";
}

type Props = {
  open: boolean;
  images: CellposePickerImage[];
  initial: CellposeSegConfig | null;
  onClose: () => void;
  onSave: (cfg: CellposeSegConfig) => void;
};

export default function CellposeSegDialog(props: Props) {
  const { open, images, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<CellposeSegConfig>(initial || emptyCellposeSegConfig());
  const [imgIdx, setImgIdx] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [overlay, setOverlay] = useState<string>("");
  const [nCells, setNCells] = useState<number | null>(null);
  const [err, setErr] = useState<string>("");
  const [notInstalled, setNotInstalled] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Re-seed when (re)opened for a different node.
  useEffect(() => {
    if (open) {
      setCfg(initial || emptyCellposeSegConfig());
      setImgIdx(0);
      setOverlay("");
      setNCells(null);
      setErr("");
      setNotInstalled(false);
    }
  }, [open, initial]);

  // Warm the heavy scientific imports so the first preview isn't a cold start.
  useEffect(() => {
    if (!open) return;
    fetch(`${apiBase()}/api/analysis/warmup`, { method: "POST" }).catch(() => {});
  }, [open]);

  const models = cfg.cellposeVersion === "v3" ? V3_MODELS : V4_MODELS;

  const patch = useCallback((p: Partial<CellposeSegConfig>) => setCfg((c) => ({ ...c, ...p })), []);

  const selected = images[Math.min(imgIdx, Math.max(0, images.length - 1))];

  const runPreview = useCallback(async () => {
    if (!selected) { setErr("No input image — wire a source inset into this node first."); return; }
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setPreviewing(true);
    setErr("");
    setNotInstalled(false);
    try {
      const resp = await fetch(`${apiBase()}/api/analysis/cellpose-preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: cellposeSegToJson(cfg),
          source: selected.source || null,
          image_b64: selected.image_b64 || "",
          cellpose_version: cfg.cellposeVersion,
        }),
        signal: ac.signal,
      });
      const j = await resp.json();
      if (j.success) {
        setOverlay(j.overlay_b64 || "");
        setNCells(typeof j.n_cells === "number" ? j.n_cells : null);
        if (!j.overlay_b64) setErr("Segmentation ran but returned no overlay image.");
      } else {
        const msg = String(j.error || "Preview failed.");
        setErr(msg);
        if (/isn't installed|not installed|Install Cellpose/i.test(msg)) setNotInstalled(true);
      }
    } catch (e) {
      if ((e as { name?: string }).name !== "AbortError") setErr(String((e as Error).message || e));
    } finally {
      setPreviewing(false);
    }
  }, [cfg, selected]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const overlaySrc = useMemo(() => (overlay ? `data:image/png;base64,${overlay}` : ""), [overlay]);
  const baseSrc = useMemo(
    () => (selected?.image_b64 ? `data:image/png;base64,${selected.image_b64.split(",").pop()}` : ""),
    [selected],
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ pb: 1 }}>
        🔬 Configure Cellpose segmentation
        <Typography variant="body2" color="text.secondary">
          Pick the model + parameters and preview the outlines. The downstream shape metrics
          (area, perimeter, circularity) run on these masks.
        </Typography>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
          {/* ── Controls ── */}
          <Stack spacing={1.5} sx={{ flex: "1 1 320px", minWidth: 300 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ width: 130 }} variant="body2">Cellpose version</Typography>
              <Select size="small" value={cfg.cellposeVersion}
                onChange={(e) => {
                  const v = e.target.value as "v3" | "v4";
                  patch({ cellposeVersion: v, model: v === "v3" ? "cyto3" : "cpsam" });
                }}>
                <MenuItem value="v4">v4 (cpsam)</MenuItem>
                <MenuItem value="v3">v3 (cyto3 / nuclei)</MenuItem>
              </Select>
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ width: 130 }} variant="body2">Model</Typography>
              <Select size="small" value={cfg.model} onChange={(e) => patch({ model: String(e.target.value) })}>
                {models.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ width: 130 }} variant="body2">Segment channel</Typography>
              <Select size="small" value={cfg.segChannel}
                onChange={(e) => patch({ segChannel: Number(e.target.value) as 0 | 1 | 2 | 3 })}>
                {[0, 1, 2, 3].map((c) => <MenuItem key={c} value={c}>{CHANNEL_LABELS[c]}</MenuItem>)}
              </Select>
            </Stack>

            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ width: 130 }} variant="body2">Nuclei channel</Typography>
              <Select size="small" value={cfg.nucleiChannel}
                onChange={(e) => patch({ nucleiChannel: Number(e.target.value) as 0 | 1 | 2 | 3 })}>
                <MenuItem value={0}>None</MenuItem>
                {[1, 2, 3].map((c) => <MenuItem key={c} value={c}>{CHANNEL_LABELS[c]}</MenuItem>)}
              </Select>
            </Stack>

            <Divider flexItem />

            <FormControlLabel
              control={<Switch size="small" checked={cfg.autoDiameter}
                onChange={(e) => patch({ autoDiameter: e.target.checked })} />}
              label="Auto-estimate diameter" />
            {!cfg.autoDiameter && (
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography sx={{ width: 130 }} variant="body2">Diameter (px)</Typography>
                <TextField size="small" type="number" value={cfg.diameter}
                  onChange={(e) => patch({ diameter: Number(e.target.value) })}
                  inputProps={{ min: 0, step: 1, style: { width: 90 } }} />
              </Stack>
            )}

            <Box>
              <Typography variant="body2">Flow threshold: {cfg.flowThreshold.toFixed(2)}</Typography>
              <Slider size="small" min={0} max={3} step={0.05} value={cfg.flowThreshold}
                onChange={(_, v) => patch({ flowThreshold: v as number })} />
            </Box>
            <Box>
              <Typography variant="body2">Cell-probability threshold: {cfg.cellprobThreshold.toFixed(1)}</Typography>
              <Slider size="small" min={-6} max={6} step={0.5} value={cfg.cellprobThreshold}
                onChange={(_, v) => patch({ cellprobThreshold: v as number })} />
            </Box>

            <Stack direction="row" spacing={1} alignItems="center">
              <Typography sx={{ width: 130 }} variant="body2">Min size (px)</Typography>
              <TextField size="small" type="number" value={cfg.minSize}
                onChange={(e) => patch({ minSize: Number(e.target.value) })}
                inputProps={{ min: 0, step: 1, style: { width: 90 } }} />
            </Stack>

            <FormControlLabel
              control={<Switch size="small" checked={cfg.useGpu}
                onChange={(e) => patch({ useGpu: e.target.checked })} />}
              label="Use GPU when available (MPS / CUDA)" />
          </Stack>

          {/* ── Preview ── */}
          <Stack spacing={1} sx={{ flex: "1 1 360px", minWidth: 320 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Button variant="contained" size="small" onClick={runPreview} disabled={previewing || !selected}>
                {previewing ? "Segmenting…" : "Preview segmentation"}
              </Button>
              {previewing && <CircularProgress size={18} />}
              {nCells != null && !previewing && (
                <Typography variant="body2" color="text.secondary">{nCells} cell{nCells === 1 ? "" : "s"}</Typography>
              )}
              <Box sx={{ flex: 1 }} />
              {images.length > 1 && (
                <Stack direction="row" spacing={0.5} alignItems="center">
                  <IconButton size="small" onClick={() => { setImgIdx((i) => (i - 1 + images.length) % images.length); setOverlay(""); }}>‹</IconButton>
                  <Typography variant="caption">{Math.min(imgIdx, images.length - 1) + 1}/{images.length}</Typography>
                  <IconButton size="small" onClick={() => { setImgIdx((i) => (i + 1) % images.length); setOverlay(""); }}>›</IconButton>
                </Stack>
              )}
            </Stack>
            {selected && (
              <Typography variant="caption" color="text.secondary" noWrap title={selected.label}>
                {selected.label}
              </Typography>
            )}
            {notInstalled && (
              <Alert severity="warning" sx={{ py: 0 }}>
                Cellpose {cfg.cellposeVersion} isn't installed yet. Open <b>Plugins → Install Cellpose</b>, then preview again.
              </Alert>
            )}
            {err && !notInstalled && <Alert severity="error" sx={{ py: 0 }} onClose={() => setErr("")}>{err}</Alert>}
            <Box sx={{
              border: "1px solid", borderColor: "divider", borderRadius: 1, overflow: "hidden",
              minHeight: 240, display: "flex", alignItems: "center", justifyContent: "center",
              background: "#111",
            }}>
              {overlaySrc
                ? <img src={overlaySrc} alt="segmentation preview" style={{ maxWidth: "100%", maxHeight: 460, display: "block" }} />
                : baseSrc
                  ? <img src={baseSrc} alt="input" style={{ maxWidth: "100%", maxHeight: 460, display: "block", opacity: 0.65 }} />
                  : <Typography variant="body2" color="text.secondary" sx={{ p: 3 }}>
                      Wire a source inset into this node, then click “Preview segmentation”.
                    </Typography>}
            </Box>
          </Stack>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} sx={{ textTransform: "none" }}>Cancel</Button>
        <Button variant="contained" onClick={() => onSave(cfg)} sx={{ textTransform: "none" }}>
          Save configuration
        </Button>
      </DialogActions>
    </Dialog>
  );
}
