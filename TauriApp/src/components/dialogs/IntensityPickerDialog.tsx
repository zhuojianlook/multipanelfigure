// ─────────────────────────────────────────────────────────────
//  IntensityPickerDialog — fluorescence channel intensity picker
// ─────────────────────────────────────────────────────────────
//  Lets the user configure a fluorescence-intensity comparison
//  across multiple images:
//    • RENAME each channel from the default "Channel R / G / B"
//      to whatever stain the colour reports (DAPI, Anti-VegF, …).
//    • Assign each input image to an experimental GROUP (Control,
//      Treatment, …) — bands are then compared across groups, not
//      across individual files.
//    • Pick MODE:
//        Simple   — raw fluorescence: per-image mean intensity in
//                   each channel.
//        Advanced — Cellpose 3 segments cells in the chosen
//                   channel, then we compute per-CELL mean intensity
//                   in each channel. The R plot compares within-cell
//                   intensity distributions across groups.
//
//  The picker emits a Python code body (and saves the config back
//  onto the node) so a normal "Run" reproduces the configured run
//  on the full-resolution images.
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Box, Typography, Button, IconButton, Dialog, DialogTitle, DialogContent,
  DialogActions, TextField, Tooltip, ToggleButton, ToggleButtonGroup, MenuItem,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";

// ── Types (exported so the host can persist the config on the node) ──
export interface FluorChannels {
  r: string;
  g: string;
  b: string;
}
export interface FluorGroup {
  id: string;
  name: string;
  /** Image labels in this group (matched against each upstream input's `label`). */
  images: string[];
}
export interface FluorCellpose {
  /** Cellpose 4 model name. "cpsam" = the SAM-based default. */
  model: string;
  /** Object diameter prior in px. 0 = auto-estimate. */
  diameter: number;
  /** Which channel to segment ON (cells / nuclei). */
  segChannel: "r" | "g" | "b";
  /** Min object size in px² — filters out spurious tiny masks. */
  minSize: number;
}

/** Advanced quantification config — mirrors quantify_fluorescence.py.
 *  Pipeline: rolling-ball BG → mask (threshold OR Cellpose) → object-level
 *  per-channel metrics (raw + background-corrected). Renders an annotated
 *  overlay per source so the user can verify which cells were segmented. */
export interface FluorQuantify {
  /** "threshold" — percentile/otsu on the combined mask channels (script default).
   *  "cellpose" — use Cellpose 3+ to generate the mask (via the plugin venv). */
  maskSource: "threshold" | "cellpose";
  /** Which channels feed the segmentation image. Default = all three. */
  maskChannels: ("r" | "g" | "b")[];
  /** How those channels combine into the seg image. */
  segmentationMode: "sum" | "max" | "mean";
  /** Threshold rule (ignored when maskSource = "cellpose"). */
  thresholdMethod: "percentile" | "otsu";
  thresholdPercentile: number;
  /** Rolling-ball BG radius (px) — 0 disables. Matches --rolling-radius. */
  rollingRadius: number;
  /** Drop objects smaller than this px² area. Matches --min-object-area. */
  minObjectArea: number;
  /** Cellpose params (used only when maskSource = "cellpose"). */
  cellpose: FluorCellpose;
}

export interface FluorIntensityConfig {
  version: 1;
  channels: FluorChannels;
  /** "simple"  — central-80% per-image mean (fast, no segmentation)
   *  "advanced"— quantify_fluorescence.py pipeline (per-cell rows) */
  mode: "simple" | "advanced" | "cellpose";   // "cellpose" kept for back-compat
  /** Legacy field — preserved so older saved nodes keep loading. New picker
   *  UI writes to `quantify` instead. */
  cellpose: FluorCellpose;
  /** Advanced pipeline knobs. */
  quantify?: FluorQuantify;
  groups: FluorGroup[];
}

export function emptyFluorConfig(): FluorIntensityConfig {
  return {
    version: 1,
    channels: { r: "Channel R", g: "Channel G", b: "Channel B" },
    mode: "simple",
    cellpose: { model: "cpsam", diameter: 0, segChannel: "b", minSize: 30 },
    quantify: {
      maskSource: "threshold",
      // Defaults match the user's CLI invocation:
      //   --mask-channels 2,3 → green + blue
      //   --segmentation-mode sum
      //   --threshold-method percentile --threshold-percentile 99
      //   --rolling-radius 35 --min-object-area 120
      maskChannels: ["g", "b"],
      segmentationMode: "sum",
      thresholdMethod: "percentile",
      thresholdPercentile: 99,
      rollingRadius: 35,
      minObjectArea: 120,
      cellpose: { model: "cpsam", diameter: 0, segChannel: "b", minSize: 120 },
    },
    groups: [],
  };
}

/** Migrate legacy "cellpose" mode → "advanced" + maskSource="cellpose"
 *  so older saved nodes display sensibly in the new picker UI. */
function migrateLegacyMode(cfg: FluorIntensityConfig): FluorIntensityConfig {
  if (cfg.mode !== "cellpose") return cfg;
  const base = cfg.quantify || emptyFluorConfig().quantify!;
  return {
    ...cfg,
    mode: "advanced",
    quantify: {
      ...base,
      maskSource: "cellpose",
      cellpose: cfg.cellpose,
    },
  };
}

let _uid = 0;
const uid = (p = "g") => `${p}_${Date.now().toString(36)}_${(_uid++).toString(36)}_${Math.random().toString(36).slice(2, 5)}`;

// ── Python code generator ────────────────────────────────────
// Self-contained — runs inside the analysis sidecar's Python
// engine. Reads `inputs` (the upstream Source's images), follows
// the config's channel renames + group mapping + mode flag, and
// emits a `fluor_intensities` table.
export function generateFluorCode(cfg: FluorIntensityConfig): string {
  const json = JSON.stringify(cfg);
  return `# @name: Channel intensities (renameable channels)
# Auto-generated from the interactive Intensity picker. Edit channel
# names / groups / mode via the "Configure intensity…" button on the
# node — manual edits here are overwritten the next time the picker
# saves.
import numpy as np, json

CFG = json.loads(r'''${json}''')
mode = CFG.get("mode", "simple")
ch_name = {
    "R": (CFG.get("channels", {}).get("r") or "Channel R"),
    "G": (CFG.get("channels", {}).get("g") or "Channel G"),
    "B": (CFG.get("channels", {}).get("b") or "Channel B"),
}
img2group = {}
for g in CFG.get("groups", []) or []:
    nm = (g.get("name") or "").strip()
    if not nm: continue
    for im in g.get("images", []) or []:
        img2group[str(im)] = nm

# Gather the upstream images. Prefer FULL-BIT-DEPTH pixels (image_raw)
# when present — they preserve the native scientific dynamic range —
# but fall back to the 8-bit display image otherwise.
imgs = []
for k, v in inputs.items():
    if not isinstance(v, dict): continue
    if "image_raw" in v or "image" in v:
        imgs.append((k, v))
if not imgs:
    raise SystemExit("No image inputs — wire your fluorescence panels into this node.")

def _label_of(src, key):
    return str(src.get("label") or key).rsplit("/", 1)[-1]
def _pixels(src):
    arr = np.asarray(src.get("image_raw") if "image_raw" in src else src["image"]).astype(np.float32)
    if arr.ndim == 2: arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] < 3:
        pad = np.repeat(arr[..., :1], 3 - arr.shape[2], axis=2)
        arr = np.concatenate([arr, pad], axis=2)
    return arr[..., :3]

rows = []
if mode in ("advanced", "cellpose"):
    # Port of quantify_fluorescence.py — multichannel per-object pipeline:
    #   1. Per-channel rolling-ball BG subtraction.
    #   2. Build segmentation image by combining the chosen mask channels.
    #   3. Get the object mask either by a percentile/Otsu threshold OR
    #      via Cellpose (the frozen sidecar delegates to the plugin venv
    #      via /api/analysis/run-cellpose over loopback).
    #   4. Clean (open/close/fill/remove-small/clear-border).
    #   5. Label each object, then measure raw + background-corrected
    #      mean and integrated density per channel for every object.
    #   6. Emit one row per (object, channel) into fluor_intensities so
    #      the downstream R plot can aggregate by (group, channel) with
    #      SD across cells.
    #   7. Save an annotated overlay per source as a separate image
    #      output so the user can see what was segmented.
    import io as _io, base64 as _b64, json as _json, sys as _sys
    import urllib.request as _ur
    from PIL import Image as _Im, ImageDraw as _ImD
    try:
        import cv2 as _cv2
        from scipy import ndimage as _ndi
        _have_scipy = True
    except Exception as _e:
        print(f"[intensity] scipy/cv2 unavailable ({_e}); falling back to simple per-image means")
        _have_scipy = False

    qf = (CFG.get("quantify") or {})
    # Back-compat: legacy "cellpose" mode reads from the old cfg.cellpose
    # block; new "advanced" mode reads from cfg.quantify.cellpose.
    if mode == "cellpose":
        cp = CFG.get("cellpose", {}) or {}
        mask_source = "cellpose"
    else:
        cp = (qf.get("cellpose") or CFG.get("cellpose") or {})
        mask_source = qf.get("maskSource", "threshold")
    mask_chs_keys = (qf.get("maskChannels") or ["g", "b"])
    mask_chs_idx = [{"r": 0, "g": 1, "b": 2}.get(k, 2) for k in mask_chs_keys]
    seg_mode = qf.get("segmentationMode", "sum")
    thr_method = qf.get("thresholdMethod", "percentile")
    thr_pct = float(qf.get("thresholdPercentile", 99))
    rolling_radius = int(qf.get("rollingRadius", 35))
    min_object_area = int(qf.get("minObjectArea", cp.get("minSize") or 80))

    def _rolling_bg(img, radius):
        if radius <= 0 or not _have_scipy: return np.zeros_like(img, dtype=np.float64)
        k = int(radius) * 2 + 1
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
        return _cv2.morphologyEx(img.astype(np.float32), _cv2.MORPH_OPEN, kernel).astype(np.float64)

    def _disk(r):
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x * x + y * y) <= r * r

    def _make_threshold_mask(seg_img):
        if thr_method == "otsu" and _have_scipy:
            scaled = np.clip((seg_img - np.percentile(seg_img, 0.2)) /
                             max(1e-9, np.percentile(seg_img, 99.8) - np.percentile(seg_img, 0.2)), 0, 1)
            u8 = (scaled * 255).astype(np.uint8)
            thr_u8, _ = _cv2.threshold(u8, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
            lo, hi = np.percentile(seg_img, [0.2, 99.8])
            thr = float(lo + (thr_u8 / 255.0) * (hi - lo))
        else:
            thr = float(np.percentile(seg_img, thr_pct))
        m = seg_img > thr
        if _have_scipy:
            m = _ndi.binary_opening(m, structure=_disk(1))
            m = _ndi.binary_closing(m, structure=_disk(2))
            m = _ndi.binary_fill_holes(m)
        return m.astype(bool)

    def _cellpose_mask_for_image(label, a8_rgb):
        cfg_json = _json.dumps({
            "model": cp.get("model") or "cpsam",
            "diameter": float(cp.get("diameter") or 0) or None,
            "min_size": int(cp.get("minSize") or min_object_area),
            "channels": [{"r": 1, "g": 2, "b": 3}.get(cp.get("segChannel") or mask_chs_keys[0], 3), 0],
        })
        buf = _io.BytesIO()
        _Im.fromarray(a8_rgb).save(buf, format="PNG")
        payload = _json.dumps({
            "config": cfg_json,
            "extra_inputs": [{"kind": "image", "key": label, "label": label,
                              "image_b64": _b64.b64encode(buf.getvalue()).decode()}],
            "sources": [], "timeout_sec": 600,
        })
        try:
            req = _ur.Request("http://127.0.0.1:8765/api/analysis/run-cellpose",
                              data=payload.encode("utf-8"),
                              headers={"Content-Type": "application/json"})
            with _ur.urlopen(req, timeout=600) as resp:
                cp_out = _json.loads(resp.read().decode("utf-8"))
        except Exception as _e:
            return None, f"cellpose call failed: {_e}"
        if not cp_out.get("success"):
            return None, cp_out.get("stderr", "(no detail)").strip()
        lbl_b64 = next((im["image"] for im in (cp_out.get("images") or [])
                        if im.get("name") == f"{label}_labels"), None)
        if not lbl_b64:
            return None, "no labels image returned"
        try:
            return np.asarray(_Im.open(_io.BytesIO(_b64.b64decode(lbl_b64))).convert("L")), None
        except Exception as _e:
            return None, f"labels decode failed: {_e}"

    n_images_with_objects = 0
    overlay_count = 0
    for key, src in imgs:
        if not _have_scipy: break
        label = _label_of(src, key)
        grp = img2group.get(label)
        if not grp:
            print(f"[intensity] {label}: not assigned to any group — skipping")
            continue
        raw = _pixels(src)                                  # HxWx3 float32
        # Per-channel rolling-ball background subtraction.
        corrected = np.zeros_like(raw, dtype=np.float64)
        for ci in range(3):
            bg_ci = _rolling_bg(raw[..., ci], rolling_radius)
            corr = raw[..., ci].astype(np.float64) - bg_ci
            corr[corr < 0] = 0
            corrected[..., ci] = corr
        # Build the segmentation image.
        sel = corrected[..., mask_chs_idx]
        if seg_mode == "max":   seg_img = np.max(sel, axis=-1)
        elif seg_mode == "mean":seg_img = np.mean(sel, axis=-1)
        else:                   seg_img = np.sum(sel, axis=-1)
        # Get the mask.
        if mask_source == "cellpose":
            a8 = np.clip(raw, 0, 255).astype(np.uint8)
            lbl_arr, err = _cellpose_mask_for_image(label, a8)
            if err or lbl_arr is None:
                print(f"[intensity] {label}: {err} — open Plugins → Cellpose 3 → Install if needed; falling back to threshold")
                mask = _make_threshold_mask(seg_img)
                labels, _n = _ndi.label(mask)
            else:
                if lbl_arr.shape[:2] != raw.shape[:2]:
                    print(f"[intensity] {label}: label shape {lbl_arr.shape} ≠ image {raw.shape[:2]} — falling back to threshold")
                    mask = _make_threshold_mask(seg_img)
                    labels, _n = _ndi.label(mask)
                else:
                    labels = lbl_arr.astype(np.int32)
        else:
            mask = _make_threshold_mask(seg_img)
            labels, _n = _ndi.label(mask)
        # Drop sub-threshold-size objects + relabel sequentially.
        unique_ids = [int(x) for x in np.unique(labels) if x != 0]
        next_id = 1
        out_labels = np.zeros_like(labels, dtype=np.int32)
        for oid in unique_ids:
            obj_mask = labels == oid
            area = int(obj_mask.sum())
            if area < min_object_area: continue
            out_labels[obj_mask] = next_id
            next_id += 1
        labels = out_labels
        cell_ids = [int(x) for x in np.unique(labels) if x != 0]
        if not cell_ids:
            print(f"[intensity] {label}: 0 objects above area {min_object_area} — skipped")
            continue
        n_images_with_objects += 1
        # Per-object per-channel measurements (raw + bg-corrected).
        for cid in cell_ids:
            m = labels == cid
            area = int(m.sum())
            ys, xs = np.where(m)
            for ci, ck in enumerate(("R", "G", "B")):
                rv = raw[..., ci][m].astype(np.float64)
                cv = corrected[..., ci][m]
                rows.append({
                    "source": label,
                    "group": grp,
                    "channel": ch_name[ck],
                    "object_id": int(cid),
                    "area_px": area,
                    "centroid_x": float(np.mean(xs)) if xs.size else 0.0,
                    "centroid_y": float(np.mean(ys)) if ys.size else 0.0,
                    "raw_mean": float(np.mean(rv)),
                    "raw_integrated_density": float(np.sum(rv)),
                    "background_corrected_mean": float(np.mean(cv)),
                    "background_corrected_integrated_density": float(np.sum(cv)),
                    # Alias the bg-corrected mean as "mean_intensity" so the
                    # existing R plot (which keys off that column name) just
                    # works with the new per-object rows.
                    "mean_intensity": float(np.mean(cv)),
                    "max_intensity": float(np.max(rv)),
                })
        # Annotated overlay PNG — yellow object boundaries on a bright
        # composite. Saved as a separate image output per source so the
        # user can verify the segmentation visually.
        try:
            comp = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.float32)
            for ci in range(min(3, raw.shape[2])):
                ch = raw[..., ci].astype(np.float32)
                lo, hi = np.percentile(ch, [1, 99.5])
                if hi > lo: comp[..., ci] = np.clip((ch - lo) / (hi - lo), 0, 1)
            comp_u8 = (comp * 255).astype(np.uint8)
            boundaries = np.zeros(labels.shape, dtype=bool)
            boundaries[:-1, :] |= labels[:-1, :] != labels[1:, :]
            boundaries[:, :-1] |= labels[:, :-1] != labels[:, 1:]
            boundaries &= labels > 0
            comp_u8[boundaries] = (255, 255, 0)
            overlay = _Im.fromarray(comp_u8)
            d = _ImD.Draw(overlay)
            for cid in cell_ids[:300]:
                ys, xs = np.where(labels == cid)
                if xs.size == 0: continue
                d.text((float(np.mean(xs)), float(np.mean(ys))), str(cid), fill=(255, 255, 255))
            mpfig_image(overlay, name=f"{label}_overlay")
            overlay_count += 1
        except Exception as _e:
            print(f"[intensity] {label}: overlay render failed: {_e}", file=_sys.stderr)
        print(f"[intensity] {label}: {len(cell_ids)} object(s) measured (mask={mask_source}, BG radius={rolling_radius})")

    if rows:
        print(f"computed per-object intensities across {n_images_with_objects} image(s); "
              f"{overlay_count} overlay(s) emitted")
        mpfig_data(rows, name="fluor_intensities")
        raise SystemExit(0)
    print("[intensity] no objects measured — falling back to simple per-image means")

# Simple mode (and cellpose-fallback): per-image mean intensity in the
# central 80% ROI. Each image contributes ONE row per channel.
for key, src in imgs:
    label = _label_of(src, key)
    grp = img2group.get(label, label)  # unassigned → its own group (one bar)
    arr = _pixels(src)
    h, w = arr.shape[:2]
    roi = arr[int(h * 0.10):int(h * 0.90), int(w * 0.10):int(w * 0.90)]
    for ci, ck in enumerate(("R", "G", "B")):
        rows.append({
            "source": label,
            "group": grp,
            "channel": ch_name[ck],
            "mean_intensity": float(roi[..., ci].mean()),
            "max_intensity": float(roi[..., ci].max()),
        })

print(f"computed channel means for {len(imgs)} image(s); "
      f"groups = {sorted({r['group'] for r in rows})}")
mpfig_data(rows, name="fluor_intensities")
`;
}

// ── Dialog ────────────────────────────────────────────────────

interface IntensityPickerDialogProps {
  open: boolean;
  /** Distinct labels of the upstream images (one entry per Source's
   *  inset that's wired in). Used to populate the group-assignment chips. */
  imageLabels: string[];
  initial: FluorIntensityConfig | null;
  onClose: () => void;
  onSave: (cfg: FluorIntensityConfig) => void;
}

const CHANNEL_SUGGESTIONS = [
  "DAPI", "Hoechst", "GFP", "RFP", "YFP", "CFP",
  "mCherry", "FITC", "TRITC", "AF488", "AF555", "AF594", "AF647", "AF680",
  "Anti-VegF", "Anti-CD31", "Anti-Ki67", "α-SMA",
];

export default function IntensityPickerDialog(props: IntensityPickerDialogProps) {
  const { open, imageLabels, initial, onClose, onSave } = props;
  const [cfg, setCfg] = useState<FluorIntensityConfig>(initial ?? emptyFluorConfig());

  // Reset when the dialog (re)opens.
  useEffect(() => {
    if (open) setCfg(initial ? migrateLegacyMode(structuredClone(initial)) : emptyFluorConfig());
  }, [open, initial]);

  // Distinct images currently upstream — chips for group assignment.
  const images = useMemo(() => Array.from(new Set(imageLabels.map((l) => l.trim()))).filter(Boolean), [imageLabels]);
  // Per-image group lookup (last group wins if duplicated; the UI prevents that).
  const imgToGroup = useMemo(() => {
    const m = new Map<string, string>();
    for (const g of cfg.groups) for (const im of g.images) m.set(im, g.name);
    return m;
  }, [cfg.groups]);

  const setChannel = useCallback((k: keyof FluorChannels, v: string) => {
    setCfg((c) => ({ ...c, channels: { ...c.channels, [k]: v } }));
  }, []);

  const addGroup = useCallback(() => {
    const used = new Set(cfg.groups.map((g) => g.name));
    let n = 1; while (used.has(`Group ${n}`)) n++;
    setCfg((c) => ({ ...c, groups: [...c.groups, { id: uid("grp"), name: `Group ${n}`, images: [] }] }));
  }, [cfg.groups]);
  const renameGroup = useCallback((id: string, name: string) => {
    setCfg((c) => ({ ...c, groups: c.groups.map((g) => g.id === id ? { ...g, name } : g) }));
  }, []);
  const deleteGroup = useCallback((id: string) => {
    setCfg((c) => ({ ...c, groups: c.groups.filter((g) => g.id !== id) }));
  }, []);
  const toggleImageInGroup = useCallback((gid: string, label: string) => {
    setCfg((c) => ({
      ...c,
      groups: c.groups.map((g) => {
        if (g.id === gid) {
          const has = g.images.includes(label);
          return { ...g, images: has ? g.images.filter((x) => x !== label) : [...g.images, label] };
        }
        // Enforce single membership — remove from other groups when adding here.
        return { ...g, images: g.images.filter((x) => x !== label) };
      }),
    }));
  }, []);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ fontSize: "1rem", py: 1.25 }}>
        🌈 Configure intensity
        <Typography component="span" variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
          Rename channels · assign images to groups · choose simple vs Cellpose per-cell
        </Typography>
      </DialogTitle>
      <DialogContent dividers sx={{ display: "flex", flexDirection: "column", gap: 1.5, py: 1.5 }}>

        {/* Channel rename ──────────────────────────────────── */}
        <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.25 }}>
          <Tooltip title="Rename each colour channel to the stain it actually shows (e.g. DAPI for blue, Anti-VegF for green). The downstream R plot uses these names on the x-axis instead of R / G / B.">
            <Typography variant="caption" sx={{ fontWeight: 700, display: "block", mb: 0.75 }}>
              Channel names
            </Typography>
          </Tooltip>
          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 1.25 }}>
            {(["r", "g", "b"] as const).map((k) => {
              const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
              return (
                <Box key={k} sx={{ display: "flex", flexDirection: "column", gap: 0.3 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Box sx={{ width: 12, height: 12, borderRadius: 0.5, bgcolor: sw, border: "1px solid", borderColor: "divider" }} />
                    <Typography variant="caption" sx={{ fontWeight: 700, textTransform: "uppercase", fontSize: "0.62rem" }}>
                      {k === "r" ? "Red" : k === "g" ? "Green" : "Blue"}
                    </Typography>
                  </Box>
                  <TextField size="small" value={cfg.channels[k]}
                    onChange={(e) => setChannel(k, e.target.value)}
                    placeholder={`Channel ${k.toUpperCase()}`}
                    inputProps={{ list: `chsugg-${k}`, style: { fontSize: "0.78rem", padding: "5px 8px" } }} />
                  <datalist id={`chsugg-${k}`}>
                    {CHANNEL_SUGGESTIONS.map((s) => <option key={s} value={s} />)}
                  </datalist>
                </Box>
              );
            })}
          </Box>
        </Box>

        {/* Mode toggle ─────────────────────────────────────── */}
        <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.25 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75, flexWrap: "wrap" }}>
            <Typography variant="caption" sx={{ fontWeight: 700 }}>Quantification mode</Typography>
            <ToggleButtonGroup size="small" exclusive value={cfg.mode === "cellpose" ? "advanced" : cfg.mode}
              onChange={(_, v) => { if (v) setCfg((c) => ({ ...c, mode: v })); }}
              sx={{ ml: 1 }}>
              <ToggleButton value="simple" sx={{ textTransform: "none", fontSize: "0.7rem", py: 0.2 }}>
                Simple (per-image mean)
              </ToggleButton>
              <ToggleButton value="advanced" sx={{ textTransform: "none", fontSize: "0.7rem", py: 0.2 }}>
                Advanced (per-object)
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          {cfg.mode === "simple" ? (
            <Typography variant="caption" sx={{ color: "text.secondary", display: "block" }}>
              Mean intensity over the central 80% of each image, one number per channel per image.
              Fast and dependency-free — good for spot-checking expression differences across conditions.
            </Typography>
          ) : (() => {
            const qf = cfg.quantify ?? emptyFluorConfig().quantify!;
            const setQf = (patch: Partial<FluorQuantify>) =>
              setCfg((c) => ({ ...c, quantify: { ...(c.quantify ?? emptyFluorConfig().quantify!), ...patch } }));
            const setCp = (patch: Partial<FluorCellpose>) =>
              setQf({ cellpose: { ...(qf.cellpose), ...patch } });
            const toggleMaskChan = (k: "r" | "g" | "b") => {
              const cur = qf.maskChannels.includes(k)
                ? qf.maskChannels.filter((x) => x !== k)
                : [...qf.maskChannels, k];
              setQf({ maskChannels: cur.length ? cur : qf.maskChannels });   // never empty
            };
            return (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <Typography variant="caption" sx={{ color: "text.secondary" }}>
                  Mirrors <code>quantify_fluorescence.py</code>: per-channel rolling-ball background subtraction,
                  build a segmentation image from the chosen channels, get the object mask (percentile/Otsu OR
                  Cellpose), then measure raw + background-corrected mean and integrated density inside every
                  object. The R plot aggregates per (group, channel) with SD across cells. An annotated overlay
                  PNG is emitted per image so you can verify the segmentation.
                </Typography>
                {/* Mask source */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 90 }}>Mask source</Typography>
                  <ToggleButtonGroup size="small" exclusive value={qf.maskSource}
                    onChange={(_, v) => { if (v) setQf({ maskSource: v }); }}>
                    <ToggleButton value="threshold" sx={{ textTransform: "none", fontSize: "0.65rem", py: 0.15 }}>
                      Threshold
                    </ToggleButton>
                    <ToggleButton value="cellpose" sx={{ textTransform: "none", fontSize: "0.65rem", py: 0.15 }}>
                      Cellpose 3
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Box>
                {/* Mask channels — multi-select chips */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                  <Tooltip title="Channels combined into the segmentation image. Default = both biological signal channels (Ch2/Ch3 in the CLI).">
                    <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 90 }}>Mask channels</Typography>
                  </Tooltip>
                  {(["r", "g", "b"] as const).map((k) => {
                    const sw = k === "r" ? "#d35454" : k === "g" ? "#5fa566" : "#5d80c0";
                    const on = qf.maskChannels.includes(k);
                    return (
                      <Box key={k} onClick={() => toggleMaskChan(k)}
                        sx={{
                          fontSize: "0.66rem", px: 0.6, py: 0.15, borderRadius: 0.75,
                          cursor: "pointer", userSelect: "none",
                          bgcolor: on ? sw : "transparent",
                          color: on ? "common.white" : "text.secondary",
                          border: "1px solid", borderColor: on ? sw : "divider",
                          fontWeight: on ? 700 : 500,
                          display: "inline-flex", alignItems: "center", gap: 0.3,
                        }}>
                        {cfg.channels[k] || `Channel ${k.toUpperCase()}`}
                      </Box>
                    );
                  })}
                  <TextField select size="small" value={qf.segmentationMode}
                    onChange={(e) => setQf({ segmentationMode: e.target.value as "sum" | "max" | "mean" })}
                    inputProps={{ style: { fontSize: "0.7rem", padding: "3px 6px" } }}
                    sx={{ minWidth: 90, ml: "auto" }}>
                    <MenuItem value="sum" sx={{ fontSize: "0.78rem" }}>Sum</MenuItem>
                    <MenuItem value="max" sx={{ fontSize: "0.78rem" }}>Max</MenuItem>
                    <MenuItem value="mean" sx={{ fontSize: "0.78rem" }}>Mean</MenuItem>
                  </TextField>
                </Box>
                {/* Threshold params (active when maskSource = "threshold") */}
                {qf.maskSource === "threshold" && (
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1 }}>
                    <TextField select size="small" label="Threshold method" value={qf.thresholdMethod}
                      onChange={(e) => setQf({ thresholdMethod: e.target.value as "percentile" | "otsu" })}
                      inputProps={{ style: { fontSize: "0.78rem" } }}>
                      <MenuItem value="percentile">Percentile</MenuItem>
                      <MenuItem value="otsu">Otsu</MenuItem>
                    </TextField>
                    <TextField size="small" label="Percentile" type="number"
                      value={qf.thresholdPercentile}
                      onChange={(e) => setQf({ thresholdPercentile: Math.max(0, Math.min(100, Number(e.target.value) || 0)) })}
                      inputProps={{ min: 0, max: 100, step: 0.5, style: { fontSize: "0.78rem" } }}
                      disabled={qf.thresholdMethod !== "percentile"} />
                    <TextField size="small" label="Rolling BG (px)" type="number"
                      value={qf.rollingRadius}
                      onChange={(e) => setQf({ rollingRadius: Math.max(0, Number(e.target.value) || 0) })}
                      inputProps={{ min: 0, max: 200, step: 1, style: { fontSize: "0.78rem" } }} />
                    <TextField size="small" label="Min object area (px²)" type="number"
                      value={qf.minObjectArea}
                      onChange={(e) => setQf({ minObjectArea: Math.max(0, Number(e.target.value) || 0) })}
                      inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.78rem" } }} />
                  </Box>
                )}
                {/* Cellpose params */}
                {qf.maskSource === "cellpose" && (
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1 }}>
                    <TextField select size="small" label="Model" value={qf.cellpose.model}
                      onChange={(e) => setCp({ model: e.target.value })}
                      inputProps={{ style: { fontSize: "0.78rem" } }}>
                      <MenuItem value="cpsam">cpsam (default)</MenuItem>
                      <MenuItem value="cyto3">cyto3</MenuItem>
                      <MenuItem value="cyto2">cyto2</MenuItem>
                      <MenuItem value="nuclei">nuclei</MenuItem>
                    </TextField>
                    <TextField select size="small" label="Segment on" value={qf.cellpose.segChannel}
                      onChange={(e) => setCp({ segChannel: e.target.value as "r" | "g" | "b" })}
                      inputProps={{ style: { fontSize: "0.78rem" } }}>
                      <MenuItem value="r">{cfg.channels.r}</MenuItem>
                      <MenuItem value="g">{cfg.channels.g}</MenuItem>
                      <MenuItem value="b">{cfg.channels.b}</MenuItem>
                    </TextField>
                    <TextField size="small" label="Diameter (px, 0=auto)" type="number"
                      value={qf.cellpose.diameter}
                      onChange={(e) => setCp({ diameter: Math.max(0, Number(e.target.value) || 0) })}
                      inputProps={{ min: 0, max: 400, step: 5, style: { fontSize: "0.78rem" } }} />
                    <TextField size="small" label="Min object area (px²)" type="number"
                      value={qf.minObjectArea}
                      onChange={(e) => setQf({ minObjectArea: Math.max(0, Number(e.target.value) || 0) })}
                      inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.78rem" } }} />
                  </Box>
                )}
              </Box>
            );
          })()}
        </Box>

        {/* Groups (per-image assignment) ─────────────────────── */}
        <Box sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.25 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
            <Tooltip title="Group images into experimental conditions (Control, Treatment, …). The downstream R plot draws one bar per (group, channel) with mean ± SD and pairwise significance brackets.">
              <Typography variant="caption" sx={{ fontWeight: 700 }}>Groups</Typography>
            </Tooltip>
            <Button size="small" variant="outlined" onClick={addGroup} disabled={images.length === 0}
              sx={{ textTransform: "none", fontSize: "0.65rem", py: 0.1, px: 0.75, ml: "auto" }}>
              + Group
            </Button>
          </Box>
          {images.length === 0 ? (
            <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
              Wire image sources upstream first — then come back here to assign them to groups.
            </Typography>
          ) : cfg.groups.length === 0 ? (
            <Typography variant="caption" sx={{ color: "text.disabled", fontStyle: "italic", display: "block" }}>
              No groups yet — click <b>+ Group</b>, then click each image chip you want in that group.
              An image can only be in one group at a time.
            </Typography>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
              {cfg.groups.map((g) => (
                <Box key={g.id} sx={{ display: "flex", alignItems: "flex-start", gap: 0.75, py: 0.4, borderTop: "1px dashed", borderColor: "divider" }}>
                  <TextField variant="standard" value={g.name}
                    onChange={(e) => renameGroup(g.id, e.target.value)}
                    inputProps={{ style: { fontSize: "0.78rem", fontWeight: 700, width: 110 } }} />
                  <Box sx={{ flex: 1, display: "flex", flexWrap: "wrap", gap: 0.4 }}>
                    {images.map((im) => {
                      const on = g.images.includes(im);
                      const inOther = imgToGroup.has(im) && imgToGroup.get(im) !== g.name;
                      return (
                        <Tooltip key={im} title={inOther ? `Already in "${imgToGroup.get(im)}" — clicking will move it here.` : im}>
                          <Box onClick={() => toggleImageInGroup(g.id, im)}
                            sx={{
                              fontSize: "0.66rem", px: 0.55, py: 0.1, borderRadius: 0.75,
                              cursor: "pointer", userSelect: "none",
                              bgcolor: on ? "primary.main" : "transparent",
                              color: on ? "primary.contrastText" : (inOther ? "text.disabled" : "text.secondary"),
                              border: "1px solid", borderColor: on ? "primary.main" : "divider",
                              fontWeight: on ? 700 : 500,
                              opacity: inOther && !on ? 0.65 : 1,
                              maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                            }}>
                            {im}
                          </Box>
                        </Tooltip>
                      );
                    })}
                  </Box>
                  <IconButton size="small" onClick={() => deleteGroup(g.id)}>
                    <DeleteOutlineIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Box>
              ))}
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.3 }}>
                {cfg.groups.map((g) => (
                  <Typography key={g.id} variant="caption" sx={{ fontSize: "0.58rem", color: "text.disabled" }}>
                    {g.name}: {g.images.length} image(s)
                  </Typography>
                ))}
              </Box>
            </Box>
          )}
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
