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
export interface FluorIntensityConfig {
  version: 1;
  channels: FluorChannels;
  mode: "simple" | "cellpose";
  cellpose: FluorCellpose;
  groups: FluorGroup[];
}

export function emptyFluorConfig(): FluorIntensityConfig {
  return {
    version: 1,
    channels: { r: "Channel R", g: "Channel G", b: "Channel B" },
    mode: "simple",
    cellpose: { model: "cpsam", diameter: 0, segChannel: "b", minSize: 30 },
    groups: [],
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
if mode == "cellpose":
    # Per-cell intensity. The frozen sidecar can't import cellpose itself
    # (torch is too heavy to bundle), so we delegate to the sidecar's own
    # /api/analysis/run-cellpose endpoint over loopback — that endpoint
    # forwards to the user-installed Cellpose plugin venv (managed via the
    # in-app Plugins → Cellpose 3 panel). The endpoint returns each image's
    # labels mask as a base64 PNG, which we decode and use to slice per-cell
    # regions from the original pixels.
    import io as _io, base64 as _b64, json as _json
    import urllib.request as _ur
    from PIL import Image as _Im
    cp = CFG.get("cellpose", {}) or {}
    cfg_json = _json.dumps({
        "model": cp.get("model") or "cpsam",
        "diameter": float(cp.get("diameter") or 0) or None,
        "min_size": int(cp.get("minSize") or 30),
        "channels": [{"r": 1, "g": 2, "b": 3}.get(cp.get("segChannel") or "b", 3), 0],
    })
    # Build the request — encode each image as PNG, POST as extra_inputs.
    extras = []
    for key, src in imgs:
        label = _label_of(src, key)
        if not img2group.get(label):
            continue   # skip ungrouped — keeps the response small
        arr = _pixels(src)
        a8 = np.clip(arr, 0, 255).astype(np.uint8)
        buf = _io.BytesIO()
        _Im.fromarray(a8).save(buf, format="PNG")
        extras.append({"kind": "image", "key": key, "label": label,
                       "image_b64": _b64.b64encode(buf.getvalue()).decode()})
    if not extras:
        print("[intensity] no grouped images — assign images to groups in the Intensity picker first.")
    else:
        payload = _json.dumps({"config": cfg_json, "extra_inputs": extras, "sources": [], "timeout_sec": 600})
        try:
            req = _ur.Request(
                "http://127.0.0.1:8765/api/analysis/run-cellpose",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with _ur.urlopen(req, timeout=600) as resp:
                cp_out = _json.loads(resp.read().decode("utf-8"))
        except Exception as _e:
            cp_out = {"success": False, "stderr": f"cellpose endpoint failed: {_e}", "images": []}
        if not cp_out.get("success"):
            print(f"[intensity] cellpose unavailable — {cp_out.get('stderr','(no detail)').strip()}")
            print("[intensity] Open Plugins → Cellpose 3 → Install, then re-run this workflow.")
        else:
            label_imgs = {im["name"]: im["image"] for im in (cp_out.get("images") or [])}
            for key, src in imgs:
                label = _label_of(src, key)
                grp = img2group.get(label)
                if not grp:
                    print(f"[intensity] {label}: not assigned to any group — skipping")
                    continue
                lbl_b64 = label_imgs.get(f"{label}_labels")
                if not lbl_b64:
                    print(f"[intensity] {label}: no labels image returned by cellpose — skipping")
                    continue
                try:
                    lbl_arr = np.asarray(_Im.open(_io.BytesIO(_b64.b64decode(lbl_b64))).convert("L"))
                except Exception as _e:
                    print(f"[intensity] {label}: labels decode failed ({_e}) — skipping")
                    continue
                arr = _pixels(src)
                # Match labels size to the analysed image (Cellpose works at the
                # original resolution since we sent native PNG).
                if lbl_arr.shape[:2] != arr.shape[:2]:
                    print(f"[intensity] {label}: label shape {lbl_arr.shape} ≠ image {arr.shape[:2]} — skipping")
                    continue
                cell_ids = [int(c) for c in np.unique(lbl_arr) if c != 0]
                if not cell_ids:
                    print(f"[intensity] no cells segmented in {label}")
                    continue
                for cid in cell_ids:
                    m = lbl_arr == cid
                    area = int(m.sum())
                    if area < int(cp.get("minSize") or 30):
                        continue
                    for ci, ck in enumerate(("R", "G", "B")):
                        rows.append({
                            "source": label,
                            "group": grp,
                            "channel": ch_name[ck],
                            "cell_id": int(cid),
                            "mean_intensity": float(arr[m, ci].mean()),
                            "area_px": area,
                        })
                print(f"[intensity] {label}: {len(cell_ids)} cell(s) segmented")
            if rows:
                print(f"computed per-cell intensities for {len({(r['source']) for r in rows})} image(s)")
                mpfig_data(rows, name="fluor_intensities")
                raise SystemExit(0)
            print("[intensity] cellpose produced 0 cells across all images — falling back to simple mean")

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
    if (open) setCfg(initial ? structuredClone(initial) : emptyFluorConfig());
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
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
            <Typography variant="caption" sx={{ fontWeight: 700 }}>Quantification mode</Typography>
            <ToggleButtonGroup size="small" exclusive value={cfg.mode}
              onChange={(_, v) => { if (v) setCfg((c) => ({ ...c, mode: v })); }}
              sx={{ ml: 1 }}>
              <ToggleButton value="simple" sx={{ textTransform: "none", fontSize: "0.7rem", py: 0.2 }}>
                Simple (raw intensity)
              </ToggleButton>
              <ToggleButton value="cellpose" sx={{ textTransform: "none", fontSize: "0.7rem", py: 0.2 }}>
                Advanced (Cellpose per-cell)
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          {cfg.mode === "simple" ? (
            <Typography variant="caption" sx={{ color: "text.secondary", display: "block" }}>
              Mean intensity over the central 80% of each image, one number per channel per image.
              Fast and dependency-free — good for spot-checking expression differences across conditions.
            </Typography>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              <Typography variant="caption" sx={{ color: "text.secondary" }}>
                Cellpose 3+ segments cells using the channel you choose, then we compute the per-cell
                mean intensity in EVERY channel. The R plot compares the cell-level intensity distributions
                between groups — much more sensitive than image-level means.
              </Typography>
              <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 1 }}>
                <TextField select size="small" label="Model" value={cfg.cellpose.model}
                  onChange={(e) => setCfg((c) => ({ ...c, cellpose: { ...c.cellpose, model: e.target.value } }))}
                  inputProps={{ style: { fontSize: "0.78rem" } }}>
                  <MenuItem value="cpsam">cpsam (default)</MenuItem>
                  <MenuItem value="cyto3">cyto3</MenuItem>
                  <MenuItem value="cyto2">cyto2</MenuItem>
                  <MenuItem value="nuclei">nuclei</MenuItem>
                </TextField>
                <TextField select size="small" label="Segment on" value={cfg.cellpose.segChannel}
                  onChange={(e) => setCfg((c) => ({ ...c, cellpose: { ...c.cellpose, segChannel: e.target.value as "r" | "g" | "b" } }))}
                  inputProps={{ style: { fontSize: "0.78rem" } }}>
                  <MenuItem value="r">{cfg.channels.r}</MenuItem>
                  <MenuItem value="g">{cfg.channels.g}</MenuItem>
                  <MenuItem value="b">{cfg.channels.b}</MenuItem>
                </TextField>
                <TextField size="small" label="Diameter (px, 0 = auto)" type="number"
                  value={cfg.cellpose.diameter}
                  onChange={(e) => setCfg((c) => ({ ...c, cellpose: { ...c.cellpose, diameter: Math.max(0, Number(e.target.value) || 0) } }))}
                  inputProps={{ min: 0, max: 400, step: 5, style: { fontSize: "0.78rem" } }} />
                <TextField size="small" label="Min cell area (px²)" type="number"
                  value={cfg.cellpose.minSize}
                  onChange={(e) => setCfg((c) => ({ ...c, cellpose: { ...c.cellpose, minSize: Math.max(0, Number(e.target.value) || 0) } }))}
                  inputProps={{ min: 0, max: 10000, step: 5, style: { fontSize: "0.78rem" } }} />
              </Box>
            </Box>
          )}
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
