/* ──────────────────────────────────────────────────────────
   Toolbar — horizontal bar above the image strip.
   Load Images button, image count badge, Save Figure button,
   Help/About button.
   ────────────────────────────────────────────────────────── */

import { useRef, useState, useEffect } from "react";
import {
  Box,
  Button,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Select,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Tooltip,
  Divider,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate";
import SaveIcon from "@mui/icons-material/Save";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import SystemUpdateAltIcon from "@mui/icons-material/SystemUpdateAlt";
import DownloadIcon from "@mui/icons-material/Download";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import LibraryAddIcon from "@mui/icons-material/LibraryAdd";
import { useCollageStore, PT_TO_PX, DEFAULT_TEXT_PT, PANEL_LABEL_DEFAULT_PT, panelLabelTextMap } from "../../store/collageStore";
import type { CollageItem } from "../../store/collageStore";
import { api } from "../../api/client";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { getVersion } from "@tauri-apps/api/app";

// Dynamic changelog — fetched from GitHub releases on dialog open
interface ChangelogEntry {
  version: string;
  date: string;
  changes: string[];
}

let _changelogCache: ChangelogEntry[] | null = null;
// The About dialog auto-opens once per app session. Tracked at module scope
// so the Toolbar remounting (e.g. switching out of and back into the
// Analysis tab, which unmounts the toolbar) does NOT re-open it.
let _aboutAutoShown = false;

/** Fetch a URL via Rust proxy (bypasses WebView CORS), fallback to browser fetch */
async function proxyFetch(url: string): Promise<string> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke("fetch_url", { url }) as string;
  } catch {
    const resp = await fetch(url);
    return resp.text();
  }
}

async function fetchChangelog(): Promise<ChangelogEntry[]> {
  if (_changelogCache) return _changelogCache;
  try {
    const REPO = "zhuojianlook/multipanelfigure";
    const relText = await proxyFetch(`https://api.github.com/repos/${REPO}/releases?per_page=30`);
    const releases = JSON.parse(relText);
    if (!Array.isArray(releases)) throw new Error("Invalid releases response");

    // Fetch recent commits to extract commit messages (much more useful than release notes)
    let commitMessages: Record<string, string[]> = {};
    try {
      const commitsText = await proxyFetch(`https://api.github.com/repos/${REPO}/commits?per_page=100`);
      const commits = JSON.parse(commitsText);
      if (Array.isArray(commits)) {
        // Group commits by their closest tag (based on release dates)
        const tagDates = releases.map((r: { tag_name: string; published_at: string }) => ({
          tag: r.tag_name,
          date: new Date(r.published_at).getTime(),
        })).sort((a: { date: number }, b: { date: number }) => b.date - a.date);

        for (const commit of commits) {
          const msg = (commit.commit?.message || "").split("\n")[0].trim();
          if (!msg || msg.startsWith("Merge") || msg.includes("Co-Authored-By")) continue;
          // Clean up: remove conventional commit prefixes for readability
          const cleaned = msg.replace(/^(feat|fix|chore|docs|refactor|style|test|ci|build)(\(.+?\))?:\s*/i, "").trim();
          if (cleaned.length < 8) continue;
          // Find which release this commit belongs to
          const commitDate = new Date(commit.commit?.author?.date || "").getTime();
          let assignedTag = tagDates[0]?.tag || "";
          for (let i = 0; i < tagDates.length - 1; i++) {
            if (commitDate <= tagDates[i].date && commitDate > tagDates[i + 1].date) {
              assignedTag = tagDates[i].tag;
              break;
            }
          }
          if (assignedTag) {
            if (!commitMessages[assignedTag]) commitMessages[assignedTag] = [];
            if (commitMessages[assignedTag].length < 8) { // cap per release
              commitMessages[assignedTag].push(cleaned);
            }
          }
        }
      }
    } catch { /* commits fetch failed, fall back to release body */ }

    const entries: ChangelogEntry[] = [];
    for (const rel of releases) {
      const tagName = rel.tag_name || "";
      const version = tagName.replace(/^(v|exp-)/, "");
      const date = (rel.published_at || "").slice(0, 10);
      const isExp = tagName.startsWith("exp-");

      // Use commit messages if available, otherwise parse release body
      let changes: string[] = commitMessages[tagName] || [];

      if (changes.length === 0) {
        // Parse release body for meaningful lines
        const body = rel.body || "";
        for (const line of body.split("\n")) {
          const trimmed = line.trim();
          if (/^[*\-]\s+/.test(trimmed)) {
            let text = trimmed.replace(/^[*\-]\s+/, "").trim();
            text = text.replace(/\s+by\s+@\S+.*$/i, "").replace(/\s+in\s+https:\/\/\S+/g, "").trim();
            // Strip conventional-commit prefix (feat/fix/chore/...) so the
            // changelog reads cleanly. CI now writes release bodies from
            // git log including those prefixes; this is the corresponding
            // display-side cleanup.
            text = text.replace(/^(feat|fix|chore|docs|refactor|style|test|ci|build|perf)(\(.+?\))?:\s*/i, "").trim();
            // Skip "Full Changelog" links
            if (text.includes("Full Changelog") || text.includes("github.com/compare")) continue;
            if (text.length > 5) changes.push(text);
          }
        }
      }

      if (version) {
        const label = isExp ? `${version} (experimental)` : version;
        entries.push({
          version: label,
          date,
          changes: changes.length > 0 ? changes : [`Release ${tagName}`],
        });
      }
    }
    _changelogCache = entries;
    return entries;
  } catch (e) {
    console.error("Changelog fetch failed:", e);
    // Return a static fallback changelog
    return [
      { version: "0.1.70", date: "2026-04-16", changes: [
        "Stable/experimental update channels with channel switcher",
        "Dynamic changelog fetched from GitHub releases",
        "Z-stack TIFF slice selection",
        "Drag-and-drop files from OS into timeline",
        "Right-click to copy preview to clipboard",
      ]},
      { version: "0.1.57", date: "2026-04-08", changes: [
        "Preview pan & zoom with controls",
        "Header margin fixes for all positions",
        "Grid horizontal scrolling, 50 row/col limit",
        "R analysis integration with presets",
        "Media groups for organizing images",
      ]},
      { version: "0.1.0", date: "2026-03-25", changes: [
        "Initial release: multi-panel scientific figure builder",
      ]},
    ];
  }
}
import { useFigureStore } from "../../store/figureStore";
import { SaveFigureDialog } from "../dialogs/SaveFigureDialog";
import { confirm as confirmDialog, alert as alertDialog } from "../shared/ConfirmDialog";
import { ensureProjectSaved } from "../../utils/projectNav";

/** CRC-32 (PNG polynomial) over a byte array — used to checksum the pHYs
 *  chunk we splice into the exported PNG. */
function _crc32(buf: Uint8Array): number {
  let c = ~0;
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xEDB88320 & -(c & 1));
  }
  return (~c) >>> 0;
}

/** Splice a `pHYs` chunk (physical pixel dimensions) into a PNG data-URL so
 *  the exported image reports the chosen DPI — downstream tools (Word,
 *  Illustrator, journal portals) then place it at the correct physical size.
 *  The chunk goes right after IHDR. Returns the original URL on any error. */
function pngWithDpi(dataUrl: string, dpi: number): string {
  try {
    const b64 = dataUrl.split(",")[1];
    if (!b64) return dataUrl;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    // 8-byte signature, then IHDR: len(4)+type(4)+data(13)+crc(4) = 25 bytes.
    const ihdrEnd = 8 + 25;
    const ppu = Math.round(dpi / 0.0254); // pixels per metre
    const type = new Uint8Array([0x70, 0x48, 0x59, 0x73]); // "pHYs"
    const typeAndData = new Uint8Array(4 + 9);
    typeAndData.set(type, 0);
    const tdv = new DataView(typeAndData.buffer);
    tdv.setUint32(4, ppu); tdv.setUint32(8, ppu); typeAndData[12] = 1; // unit = metre
    const crc = _crc32(typeAndData);
    const chunk = new Uint8Array(4 + 4 + 9 + 4);
    const cv = new DataView(chunk.buffer);
    cv.setUint32(0, 9);            // data length
    chunk.set(typeAndData, 4);     // type + data
    cv.setUint32(17, crc);         // crc
    const out = new Uint8Array(bytes.length + chunk.length);
    out.set(bytes.subarray(0, ihdrEnd), 0);
    out.set(chunk, ihdrEnd);
    out.set(bytes.subarray(ihdrEnd), ihdrEnd + chunk.length);
    let s = "";
    const CH = 0x8000;
    for (let i = 0; i < out.length; i += CH) {
      s += String.fromCharCode.apply(null, Array.from(out.subarray(i, i + CH)));
    }
    return "data:image/png;base64," + btoa(s);
  } catch (e) {
    console.warn("[collage] pHYs inject failed", e);
    return dataUrl;
  }
}

/** One drawable text run, positioned by the browser's own layout engine.
 *  Coordinates are in page px RELATIVE to the item's top-left (it.x/it.y). */
type TextRun = {
  text: string;
  x: number;        // left edge of the run (page px, relative to it.x)
  baseline: number; // alphabetic baseline (page px, relative to it.y)
  font: string;     // canvas font shorthand
  sizePx: number;   // run font size in px (for underline/strike thickness)
  color: string;
  underline: boolean;
  strike: boolean;
  width: number;    // measured run width (for underline/strike length)
};

/** Custom fonts register under their file-name-without-extension. */
function _famCss(n?: string): string {
  return `"${(n ?? "Arial").replace(/\.(ttf|otf|ttc|woff2?)$/i, "")}", Arial, sans-serif`;
}

/** Lay a text item out OFF-SCREEN at 1:1 with the EXACT same CSS the canvas
 *  uses (CollageView), then read each word's real baseline + x straight from
 *  the browser. Drawing those measured positions makes the export positionally
 *  identical to the on-screen text — no analytical baseline approximation, so
 *  zero drift (wrap, alignment, per-segment fonts and super/subscript all come
 *  from the same layout engine). Returns [] if anything goes wrong so the
 *  caller can fall back to the analytical renderer. */
function layoutTextRuns(it: CollageItem): TextRun[] {
  const align = it.align ?? "left";
  const baseColor = it.fontColor ?? "#000000";
  const basePt = it.fontSize ?? DEFAULT_TEXT_PT;

  // Per-segment definition (plain text = a single synthetic segment).
  type SegDef = {
    text: string; fontFamily: string; sizePx: number; bold: boolean;
    italic: boolean; sup: boolean; sub: boolean; color: string;
    underline: boolean; strike: boolean; font: string;
  };
  const segs: SegDef[] = [];
  const makeFont = (italic: boolean, bold: boolean, px: number, fam: string) =>
    `${italic ? "italic" : "normal"} ${bold ? "bold" : "normal"} ${px}px ${fam}`;

  if (it.styledSegments?.length) {
    for (const s of it.styledSegments) {
      const st = s.font_style ?? [];
      const sup = st.includes("Superscript");
      const sub = st.includes("Subscript");
      const fam = _famCss(s.font_name);
      const px = (s.font_size ?? basePt) * (sup || sub ? 0.7 : 1) * PT_TO_PX;
      const bold = st.includes("Bold");
      const italic = st.includes("Italic");
      segs.push({
        text: s.text ?? "", fontFamily: fam, sizePx: px, bold, italic, sup, sub,
        color: s.color || baseColor,
        underline: st.includes("Underline"), strike: st.includes("Strikethrough"),
        font: makeFont(italic, bold, px, fam),
      });
    }
  } else {
    const fam = _famCss(it.fontFamily);
    const px = basePt * PT_TO_PX;
    segs.push({
      text: it.text ?? "", fontFamily: fam, sizePx: px,
      bold: !!it.fontBold, italic: !!it.fontItalic, sup: false, sub: false,
      color: baseColor, underline: !!it.fontUnderline, strike: false,
      font: makeFont(!!it.fontItalic, !!it.fontBold, px, fam),
    });
  }

  const container = document.createElement("div");
  Object.assign(container.style, {
    position: "absolute", left: "-99999px", top: "0px", width: `${it.w}px`,
    boxSizing: "border-box", margin: "0", padding: "0",
    lineHeight: "1.2", whiteSpace: "pre-wrap", wordBreak: "break-word",
    textAlign: align, visibility: "hidden",
  } as Partial<CSSStyleDeclaration>);

  // Tokenize each segment into word / space / newline tokens, each wrapped in
  // its own inline span (inline spans are transparent to layout, so wrap +
  // metrics match the real div exactly).
  const toks: { el: HTMLSpanElement; seg: SegDef; text: string }[] = [];
  for (const seg of segs) {
    const segSpan = document.createElement("span");
    Object.assign(segSpan.style, {
      fontFamily: seg.fontFamily, fontSize: `${seg.sizePx}px`,
      fontWeight: seg.bold ? "700" : "400", fontStyle: seg.italic ? "italic" : "normal",
      verticalAlign: seg.sup ? "super" : seg.sub ? "sub" : "baseline",
      whiteSpace: "pre-wrap",
    } as Partial<CSSStyleDeclaration>);
    for (const part of seg.text.split(/(\n| +)/)) {
      if (part === "") continue;
      const tk = document.createElement("span");
      tk.textContent = part;
      segSpan.appendChild(tk);
      toks.push({ el: tk, seg, text: part });
    }
    container.appendChild(segSpan);
  }

  document.body.appendChild(container);
  try {
    const c = container.getBoundingClientRect();
    const runs: TextRun[] = [];
    // A zero-size inline-block probe inherits the run's baseline (including any
    // super/sub shift); its top edge sits exactly on that baseline.
    const baselineOf = (host: HTMLElement): number => {
      const p = document.createElement("span");
      Object.assign(p.style, {
        display: "inline-block", width: "0", height: "0",
        overflow: "hidden", verticalAlign: "baseline",
      } as Partial<CSSStyleDeclaration>);
      host.appendChild(p);
      const top = p.getBoundingClientRect().top;
      host.removeChild(p);
      return top - c.top;
    };
    for (const { el, seg, text } of toks) {
      if (text === "\n" || /^\s+$/.test(text)) continue; // whitespace: layout only
      const rects = el.getClientRects();
      if (!rects.length) continue;
      if (rects.length === 1) {
        const r = rects[0];
        runs.push({
          text, x: r.left - c.left, baseline: baselineOf(el), font: seg.font,
          sizePx: seg.sizePx, color: seg.color, underline: seg.underline,
          strike: seg.strike, width: r.width,
        });
      } else {
        // A single token wrapped across lines (break-word on a long string):
        // measure each character so every fragment lands exactly.
        el.textContent = "";
        const chars: HTMLSpanElement[] = [];
        for (const ch of text) {
          const cs = document.createElement("span");
          cs.textContent = ch;
          el.appendChild(cs);
          chars.push(cs);
        }
        for (let i = 0; i < chars.length; i++) {
          const cr = chars[i].getBoundingClientRect();
          runs.push({
            text: text[i], x: cr.left - c.left, baseline: baselineOf(chars[i]),
            font: seg.font, sizePx: seg.sizePx, color: seg.color,
            underline: seg.underline, strike: seg.strike, width: cr.width,
          });
        }
      }
    }
    return runs;
  } finally {
    document.body.removeChild(container);
  }
}

/* ── SaveCollageButton ───────────────────────────────────────
   Renders the collage canvas to PNG client-side (compositing
   each item at its x/y/w/h on a single offscreen <canvas>),
   then writes the bytes to a user-chosen path via the
   existing save_base64_to_path Tauri command. In a non-Tauri
   browser preview, falls back to a download anchor.

   Export DPI: the canvas is a 300-DPI virtual page, so factor =
   exportDpi/300 scales the output pixels. Figures (.mpf) and R
   plots are re-rendered at the higher resolution first so they
   carry real detail; text/lines/guides re-rasterize crisply via
   the scaled context. The PNG is tagged with the chosen DPI. */
function SaveCollageButton() {
  const items = useCollageStore((s) => s.items);
  const canvasW = useCollageStore((s) => s.canvasW);
  const canvasH = useCollageStore((s) => s.canvasH);
  const background = useCollageStore((s) => s.background);
  const exportDpi = useCollageStore((s) => s.exportDpi);
  const setExportDpi = useCollageStore((s) => s.setExportDpi);
  const exportFormat = useCollageStore((s) => s.exportFormat);
  const setExportFormat = useCollageStore((s) => s.setExportFormat);
  const panelLabelUpper = useCollageStore((s) => s.panelLabelUpper);
  const panelLabelParen = useCollageStore((s) => s.panelLabelParen);
  const [exporting, setExporting] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const handleSave = async () => {
    if (items.length === 0) {
      await alertDialog({
        title: "Empty collage",
        body: "No items in the collage to save. Add a figure or image first.",
      });
      return;
    }
    const fmt = (exportFormat || "png").toLowerCase();
    const dlgExt = fmt === "jpeg" ? "jpg" : fmt;
    // Ask WHERE to save first, so the native dialog appears immediately — the
    // high-DPI render below can take several seconds and we don't want the user
    // to think nothing happened.
    let savePath: string | null = null;
    let inTauri = false;
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      savePath = await save({
        defaultPath: `collage.${dlgExt}`,
        filters: [{ name: dlgExt.toUpperCase(), extensions: [dlgExt] }],
      });
      inTauri = true;
      if (savePath === null) return; // user cancelled the dialog
    } catch {
      inTauri = false; // not running in Tauri — fall back to a browser download
    }

    // Output scale: the canvas is a 300-DPI page, so factor = DPI/300.
    const factor = Math.max(0.25, Math.min(4, exportDpi / 300));
    setExporting(true);
    try {
    const sorted = [...items].sort((a, b) => a.z - b.z);

    // ── Pre-render figures + R plots at the export resolution ──
    // Build a map of item-id → high-res data-URL. Figures re-render via
    // matplotlib at a DPI that targets the item's export pixel footprint;
    // R plots re-run ggplot at factor× their natural size. Best-effort:
    // any failure falls back to the existing (display-res) raster.
    const hiRes = new Map<string, string>();
    for (const it of sorted) {
      if (it.kind === "figure" && it.projectPath) {
        try {
          const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
          const pt = useCollageStore.getState().globalHeaderPt;
          const sel = useCollageStore.getState().elemSelByItem[it.id];
          const elementIds = sel ? Object.keys(sel).filter((k) => sel[k]) : null;
          const overrides = useCollageStore.getState().elemOverridesByItem[it.id] || null;
          // Supersample (2×) the figure render so raster text is as crisp as
          // the vector elements: target 2× the export footprint (it.w×factor),
          // then drawImage downsamples it. (Without this, figures render near
          // ~150 dpi while the page is 300 dpi, so MPF text looked soft.)
          const SS = 2;
          const dpi = Math.max(200, Math.min(1200, Math.round(150 * (it.w * factor * SS) / Math.max(1, it.naturalW))));
          const resp = await api.renderCollageFigure(
            it.projectPath, pt ?? null, Math.max(0.001, scale), it.w, elementIds,
            overrides as Record<string, unknown> | null, dpi,
          );
          if (resp?.image) hiRes.set(it.id, `data:image/png;base64,${resp.image}`);
        } catch (e) {
          console.warn("[collage] hi-res figure render failed for", it.name, e);
        }
      } else if (it.kind === "image" && it.rCode) {
        try {
          const ov = it.rTextOverrides || {};
          const pt = useCollageStore.getState().globalHeaderPt;
          const scale = it.naturalW > 0 ? it.w / it.naturalW : 1;
          const baseFs = pt ? Math.max(1, Math.round(pt / Math.max(0.001, scale))) : null;
          // R is raster — render at 2× the export footprint with res scaled in
          // lock-step (uniform supersample), then drawImage downsamples → crisp.
          const nW = it.naturalW || 640, nH = it.naturalH || 480;
          const k = Math.min(12, Math.max(1, (it.w * factor * 2) / nW));
          const res = await api.runR(it.rCode, it.rDataCsv ?? "", it.rInterpreter ?? undefined, baseFs, {
            textOverrides: ov as Record<string, unknown>,
            renderOverride: true,
            overrideOnly: true,
            overrideWidth: Math.round(nW * k),
            overrideHeight: Math.round(nH * k),
            overrideRes: Math.round(150 * k),
          });
          const png = res.plots?.[0];
          if (res.success && png) hiRes.set(it.id, `data:image/png;base64,${png}`);
        } catch (e) {
          console.warn("[collage] hi-res R render failed for", it.name, e);
        }
      }
    }

    // Compose items onto an offscreen canvas at factor× virtual resolution.
    // The context is scaled by `factor` so all the existing draw math stays
    // in canvas-pixel space; text/lines re-rasterize crisply, and the
    // high-res figure/R rasters map ~1:1.
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(canvasW * factor);
    canvas.height = Math.round(canvasH * factor);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      await alertDialog({
        title: "Canvas error",
        body: "Could not initialise a 2D canvas context.",
      });
      return;
    }
    ctx.scale(factor, factor);
    // High-quality downsampling for the supersampled figure/R rasters.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    // A "transparent" background leaves the canvas unfilled so the exported
    // PNG keeps its alpha channel; any other value fills with that color.
    if (background !== "transparent") {
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, canvasW, canvasH);
    }
    // Wrap a draw in a rotation transform around the item's centre when the
    // item has a rotation (matches the on-canvas CSS transform).
    const withRotation = (it: typeof sorted[number], draw: () => void) => {
      const rot = it.rotation || 0;
      if (!rot) { draw(); return; }
      const cx = it.x + it.w / 2, cy = it.y + it.h / 2;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate((rot * Math.PI) / 180);
      ctx.translate(-cx, -cy);
      draw();
      ctx.restore();
    };
    // Make sure custom fonts are loaded before measuring/drawing text, so the
    // off-screen layout uses the same glyph metrics as the on-screen canvas.
    try { await (document as Document & { fonts?: FontFaceSet }).fonts?.ready; } catch { /* no-op */ }
    // Draw browser-measured text runs at (baseX, baseY) — used for both text
    // items and panel labels so they share the same zero-drift positioning.
    const drawRuns = (runs: TextRun[], baseX: number, baseY: number) => {
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
      for (const r of runs) {
        const dx = baseX + r.x;
        const by = baseY + r.baseline;
        ctx.font = r.font;
        ctx.fillStyle = r.color;
        ctx.fillText(r.text, dx, by);
        if (r.underline || r.strike) {
          ctx.save();
          ctx.strokeStyle = r.color;
          ctx.lineWidth = Math.max(1, r.sizePx / 14);
          if (r.underline) { ctx.beginPath(); ctx.moveTo(dx, by + r.sizePx * 0.16); ctx.lineTo(dx + r.width, by + r.sizePx * 0.16); ctx.stroke(); }
          if (r.strike) { ctx.beginPath(); ctx.moveTo(dx, by - r.sizePx * 0.3); ctx.lineTo(dx + r.width, by - r.sizePx * 0.3); ctx.stroke(); }
          ctx.restore();
        }
      }
    };
    // Precompute each labeled item's letter (reading order + global format).
    const labelTextById = panelLabelTextMap(sorted, panelLabelUpper, panelLabelParen);
    for (const it of sorted) {
      if (it.kind === "text") {
        withRotation(it, () => {
          // PRIMARY: lay the text out with the browser's own engine off-screen
          // at 1:1 and draw each word at its measured baseline/x, so the export
          // is positionally identical to the on-screen text (zero drift).
          let runs: TextRun[] = [];
          try { runs = layoutTextRuns(it); }
          catch (e) { console.warn("[collage] DOM text layout failed; using analytical fallback", e); }
          if (runs.length) {
            drawRuns(runs, it.x, it.y);
            return;
          }
          // FALLBACK (analytical) — only if the DOM layout produced nothing.
          // Text font sizes are stored in POINTS; the canvas is a 300-DPI
          // page, so multiply by PT_TO_PX to get device pixels.
          const baseSizePx = (it.fontSize ?? DEFAULT_TEXT_PT) * PT_TO_PX;
          const baseColor = it.fontColor ?? "#000000";
          const align = it.align ?? "left";
          // Custom fonts register under their file-name-without-extension.
          const fam = (n?: string) => `"${(n ?? "Arial").replace(/\.(ttf|otf|ttc|woff2?)$/i, "")}", Arial, sans-serif`;

          if (it.styledSegments?.length) {
            // Rich text: lay out per-character segments with word wrap,
            // per-segment font/size/colour and bold/italic/underline/
            // strikethrough/super-subscript. Two passes: wrap into lines,
            // then draw each line with its alignment offset.
            type Tok = { text: string; seg: typeof it.styledSegments[number]; space: boolean };
            // Returns device px for a segment (point size × PT_TO_PX, ×0.7 for super/sub).
            const sizeOf = (seg: Tok["seg"]) => {
              const st = seg.font_style ?? [];
              const sub = st.includes("Superscript") || st.includes("Subscript");
              return (seg.font_size ?? (it.fontSize ?? DEFAULT_TEXT_PT)) * (sub ? 0.7 : 1) * PT_TO_PX;
            };
            const fontOf = (seg: Tok["seg"]) => {
              const st = seg.font_style ?? [];
              const weight = st.includes("Bold") ? "bold" : "normal";
              const ital = st.includes("Italic") ? "italic" : "normal";
              return `${ital} ${weight} ${sizeOf(seg)}px ${fam(seg.font_name)}`;
            };
            const measure = (t: Tok) => { ctx.font = fontOf(t.seg); return ctx.measureText(t.text).width; };
            // Tokenize: words, runs of spaces, and explicit newlines.
            const tokens: Tok[] = [];
            for (const seg of it.styledSegments) {
              for (const part of (seg.text ?? "").split(/(\n| +)/)) {
                if (part === "") continue;
                tokens.push({ text: part, seg, space: part === "\n" || /^ +$/.test(part) });
              }
            }
            // Pass 1: wrap.
            const lines: Tok[][] = [];
            let cur: Tok[] = [];
            let curW = 0;
            for (const tk of tokens) {
              if (tk.text === "\n") { lines.push(cur); cur = []; curW = 0; continue; }
              if (tk.space && cur.length === 0) continue; // drop leading spaces
              const w = measure(tk);
              if (!tk.space && curW + w > it.w && cur.length > 0) { lines.push(cur); cur = []; curW = 0; }
              cur.push(tk); curW += w;
            }
            if (cur.length) lines.push(cur);
            // Pass 2: draw. Match the on-screen HTML inline flow — segments are
            // baseline-aligned (not top-aligned), and the line box adds half its
            // leading above. So we draw on the shared baseline of each line.
            ctx.textBaseline = "alphabetic";
            ctx.textAlign = "left";
            let y = it.y;
            for (const line of lines) {
              if (line.length === 0) { y += baseSizePx * 1.2; continue; } // blank line
              const widths = line.map(measure);
              const lineW = widths.reduce((a, b) => a + b, 0);
              const maxSize = Math.max(baseSizePx, ...line.map((t) => sizeOf(t.seg)));
              const lineH = maxSize * 1.2;
              // Baseline via the CSS line-box model: leading = lineH − the
              // tallest run's content box (measured ascent+descent), split evenly
              // above/below, then drop by the ascent. Using asc+desc (not the em
              // size) matches the browser's inline flow — the em-size form left
              // text ~(asc+desc−size)/2 px too low.
              const tallest = line.reduce((a, b) => (sizeOf(b.seg) >= sizeOf(a.seg) ? b : a), line[0]);
              ctx.font = fontOf(tallest.seg);
              const fm = ctx.measureText("Mg");
              const ascent = fm.fontBoundingBoxAscent || (maxSize * 0.8);
              const descent = fm.fontBoundingBoxDescent || (maxSize * 0.2);
              const baselineY = y + (lineH - (ascent + descent)) / 2 + ascent;
              let x = align === "center" ? it.x + (it.w - lineW) / 2 : align === "right" ? it.x + (it.w - lineW) : it.x;
              line.forEach((tk, i) => {
                const st = tk.seg.font_style ?? [];
                const sz = sizeOf(tk.seg);
                // super/sub shift relative to the line baseline (em of the run).
                const dy = st.includes("Superscript") ? -sz * 0.45 : st.includes("Subscript") ? sz * 0.2 : 0;
                ctx.font = fontOf(tk.seg);
                ctx.fillStyle = tk.seg.color || baseColor;
                ctx.fillText(tk.text, x, baselineY + dy);
                if (st.includes("Underline") || st.includes("Strikethrough")) {
                  ctx.save();
                  ctx.strokeStyle = tk.seg.color || baseColor;
                  ctx.lineWidth = Math.max(1, sz / 14);
                  if (st.includes("Underline")) { ctx.beginPath(); ctx.moveTo(x, baselineY + dy + sz * 0.16); ctx.lineTo(x + widths[i], baselineY + dy + sz * 0.16); ctx.stroke(); }
                  if (st.includes("Strikethrough")) { ctx.beginPath(); ctx.moveTo(x, baselineY + dy - sz * 0.3); ctx.lineTo(x + widths[i], baselineY + dy - sz * 0.3); ctx.stroke(); }
                  ctx.restore();
                }
                x += widths[i];
              });
              y += lineH;
            }
            return;
          }

          // Plain text box (whole-box font props, incl. underline).
          const fs = baseSizePx;
          const weight = it.fontBold ? "bold" : "normal";
          const style = it.fontItalic ? "italic" : "normal";
          ctx.font = `${style} ${weight} ${fs}px ${fam(it.fontFamily)}`;
          ctx.fillStyle = baseColor;
          ctx.textBaseline = "alphabetic";
          ctx.textAlign = align === "center" ? "center" : align === "right" ? "right" : "left";
          const xBase = align === "center" ? it.x + it.w / 2 : align === "right" ? it.x + it.w : it.x;
          const lineHeight = fs * 1.2;
          // Baseline via the CSS line-box model (matches the on-screen HTML box):
          // leading = lineHeight − the font's content box (measured ascent+
          // descent), split evenly above/below, then drop by the ascent. Using
          // asc+desc instead of the em size removes the ~(asc+desc−fs)/2 px
          // downward drift the old "top"-baseline offset produced.
          const pfm = ctx.measureText("Mg");
          const pAsc = pfm.fontBoundingBoxAscent || fs * 0.8;
          const pDesc = pfm.fontBoundingBoxDescent || fs * 0.2;
          let y = it.y + (lineHeight - (pAsc + pDesc)) / 2 + pAsc;
          for (const para of (it.text ?? "").split("\n")) {
            const words = para.split(" ");
            let line = "";
            const flush = (ln: string) => {
              ctx.fillText(ln, xBase, y);
              if (it.fontUnderline && ln) {
                const w = ctx.measureText(ln).width;
                const x0 = align === "center" ? xBase - w / 2 : align === "right" ? xBase - w : xBase;
                ctx.save();
                ctx.strokeStyle = baseColor;
                ctx.lineWidth = Math.max(1, fs / 14);
                ctx.beginPath(); ctx.moveTo(x0, y + fs * 0.16); ctx.lineTo(x0 + w, y + fs * 0.16); ctx.stroke();
                ctx.restore();
              }
              y += lineHeight;
            };
            for (const word of words) {
              const test = line ? line + " " + word : word;
              if (ctx.measureText(test).width > it.w && line) {
                flush(line);
                line = word;
              } else {
                line = test;
              }
            }
            flush(line);
          }
        });
        continue;
      }
      if (it.kind === "line") {
        withRotation(it, () => {
          ctx.save();
          const th = it.lineThickness ?? 3;
          ctx.strokeStyle = it.lineColor ?? "#000000";
          ctx.lineWidth = th;
          if (it.lineStyle === "dashed") ctx.setLineDash([th * 3, th * 2]);
          else if (it.lineStyle === "dotted") ctx.setLineDash([th, th * 1.5]);
          else ctx.setLineDash([]);
          const ly = it.y + it.h / 2;
          ctx.beginPath();
          ctx.moveTo(it.x, ly);
          ctx.lineTo(it.x + it.w, ly);
          ctx.stroke();
          ctx.restore();
        });
        continue;
      }
      await new Promise<void>((resolve) => {
        const img = new window.Image();
        img.onload = () => {
          try {
            withRotation(it, () => {
              ctx.drawImage(img, it.x, it.y, it.w, it.h);
              // Panel label (a, b, c…) drawn ON the item so it rotates with it,
              // using the same zero-drift browser-measured text layout.
              const lt = labelTextById[it.id];
              if (it.panelLabel && lt) {
                const lbl = it.panelLabel;
                const synth = {
                  ...it, kind: "text", text: lt, styledSegments: undefined,
                  w: 100000, align: "left",
                  fontSize: lbl.fontSize ?? PANEL_LABEL_DEFAULT_PT,
                  fontColor: lbl.color ?? "#000000",
                  fontBold: lbl.bold !== false, fontItalic: false, fontUnderline: false,
                  fontFamily: lbl.fontFamily,
                } as CollageItem;
                try {
                  const runs = layoutTextRuns(synth);
                  if (runs.length) drawRuns(runs, it.x + lbl.offsetX, it.y + lbl.offsetY);
                } catch (e) { console.warn("[collage] panel label draw failed", e); }
              }
            });
          } catch (err) {
            console.warn("[collage] drawImage failed for", it.name, err);
          }
          resolve();
        };
        img.onerror = () => resolve();
        // Prefer the high-res raster rendered for export; fall back to display.
        img.src = hiRes.get(it.id) ?? it.src;
      });
    }

    // Compose to a DPI-tagged PNG. For non-PNG formats, hand that PNG to the
    // backend to convert (JPEG/TIFF/PDF) with the DPI embedded.
    const pngDataUrl = pngWithDpi(canvas.toDataURL("image/png"), exportDpi);
    let b64 = pngDataUrl.split(",")[1] ?? "";
    let ext = "png";
    let mime = "image/png";
    if (fmt !== "png") {
      try {
        const conv = await api.convertCollage(
          canvas.toDataURL("image/png").split(",")[1] ?? "",
          fmt, exportDpi, background === "transparent" ? "#FFFFFF" : background,
        );
        b64 = conv.image;
        ext = conv.ext || fmt;
        mime = ext === "pdf" ? "application/pdf" : ext === "tiff" ? "image/tiff" : ext === "jpg" ? "image/jpeg" : "image/png";
      } catch (e) {
        console.error("[collage] format convert failed", e);
        await alertDialog({ title: "Convert failed", body: `Could not export as ${fmt.toUpperCase()} (is the backend running?). Saved as PNG instead.` });
      }
    }

    if (inTauri && savePath) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("save_base64_to_path", { path: savePath, dataB64: b64 });
        await alertDialog({ title: "Collage saved", body: `Collage saved to ${savePath}\n${Math.round(canvasW * factor)}×${Math.round(canvasH * factor)} px @ ${exportDpi} DPI` });
      } catch (e) {
        console.error("[collage] save failed", e);
        await alertDialog({ title: "Save failed", body: `Could not write the file. ${e instanceof Error ? e.message : String(e)}` });
      }
      return;
    }
    // Browser preview — download.
    const a = document.createElement("a");
    a.href = `data:${mime};base64,${b64}`;
    a.download = `collage.${ext}`;
    a.click();
    } finally {
      setExporting(false);
    }
  };

  const selectSx = {
    fontSize: "0.8rem", height: 34, borderRadius: 1, px: 0.75, width: "100%",
    bgcolor: "var(--c-surface)", color: "var(--c-text)",
    border: "1px solid var(--c-border)", cursor: "pointer",
  } as const;

  return (
    <>
      <Button
        variant="contained"
        color="secondary"
        startIcon={<SaveIcon />}
        onClick={() => setModalOpen(true)}
        disabled={exporting}
      >
        {exporting ? "Saving…" : "Save Collage"}
      </Button>

      <Dialog open={modalOpen} onClose={() => !exporting && setModalOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>Save Collage</DialogTitle>
        <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
          <Box>
            <Typography variant="caption" sx={{ color: "text.secondary", mb: 0.5, display: "block" }}>Output format</Typography>
            <Box component="select" value={exportFormat}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setExportFormat(e.target.value)} sx={selectSx}>
              {[["png", "PNG (lossless, supports transparency)"], ["jpeg", "JPEG (smaller, flattened)"], ["tiff", "TIFF (lossless, journals)"], ["pdf", "PDF (raster page)"]].map(([v, l]) => (
                <option key={v} value={v}>{l}</option>
              ))}
            </Box>
          </Box>
          <Box>
            <Typography variant="caption" sx={{ color: "text.secondary", mb: 0.5, display: "block" }}>Resolution</Typography>
            <Box component="select" value={String(exportDpi)}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setExportDpi(Number(e.target.value))} sx={selectSx}>
              {[[150, "150 DPI (draft)"], [300, "300 DPI (standard)"], [600, "600 DPI (high)"], [1200, "1200 DPI (max)"]].map(([d, l]) => (
                <option key={d} value={d}>{l}</option>
              ))}
            </Box>
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary", lineHeight: 1.4 }}>
            Output: {Math.round(canvasW * Math.max(0.25, Math.min(4, exportDpi / 300)))}×{Math.round(canvasH * Math.max(0.25, Math.min(4, exportDpi / 300)))} px.
            Figures and R plots are re-rendered at this resolution; higher DPI takes longer.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setModalOpen(false)} disabled={exporting}>Cancel</Button>
          <Button variant="contained" color="secondary" startIcon={<SaveIcon />} disabled={exporting}
            onClick={async () => { setModalOpen(false); await handleSave(); }}>
            {exporting ? "Saving…" : "Choose file & save"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

/* ── CollageWorkspaceControls ────────────────────────────────
   Workspace toggle (Builder ↔ Collage) and the "Add to Collage"
   action that captures the currently-rendered figure preview
   and pushes it into the collage store. */
function CollageWorkspaceControls() {
  const mode = useCollageStore((s) => s.mode);
  const setMode = useCollageStore((s) => s.setMode);
  const addItem = useCollageStore((s) => s.addItem);
  const updateItem = useCollageStore((s) => s.updateItem);
  const itemCount = useCollageStore((s) => s.items.length);

  const handleAddToCollage = async () => {
    let projectPath: string | null;
    try {
      projectPath = await ensureProjectSaved();
    } catch (e) {
      console.error("[collage] save before add failed:", e);
      await alertDialog({
        title: "Save failed",
        body: "Could not save the project. Add to Collage cancelled.",
      });
      return;
    }
    if (!projectPath) return; // user cancelled the save dialog

    // Uniqueness check — the same .mpf can appear at most once in a
    // collage. If a duplicate exists, offer to refresh that item
    // (re-render with the latest state) instead of adding a second.
    const existing = useCollageStore.getState().items.find((i) => i.projectPath === projectPath);

    try {
      // Capture the figure as a baked raster (headers included), exactly
      // as the .mpf renders. Header sizing across figures is unified
      // later via the sidebar "Update headers" button, which re-renders
      // each figure at the right per-figure override pt.
      const resp = await api.getPreview();
      if (!resp?.image) {
        await alertDialog({
          title: "Empty preview",
          body: "Preview is empty — add some images to your panels first.",
        });
        return;
      }
      const naturalW = resp.width || 0;
      const naturalH = resp.height || 0;
      const aspect = naturalH > 0 ? naturalW / naturalH : 1;

      if (existing) {
        const ok = await confirmDialog({
          title: "Already in collage",
          body: `"${existing.name}" (${projectPath}) is already in the collage.\n\n`
            + "Update it with the latest rendered figure? (Position and size "
            + "stay where you put them.)",
          confirmLabel: "Update",
        });
        if (!ok) return;
        updateItem(existing.id, {
          src: `data:image/png;base64,${resp.image}`,
          naturalW,
          naturalH,
        });
        setMode("collage");
        return;
      }

      const targetMax = 600;
      const w = aspect >= 1 ? targetMax : targetMax * aspect;
      const h = aspect >= 1 ? targetMax / aspect : targetMax;
      const offset = itemCount * 24;
      addItem({
        kind: "figure",
        src: `data:image/png;base64,${resp.image}`,
        name: projectPath.split("/").pop()?.replace(/\.mpf$/i, "") || `Figure ${itemCount + 1}`,
        x: 40 + offset,
        y: 40 + offset,
        w,
        h,
        naturalW,
        naturalH,
        projectPath,
      });
      setMode("collage");
    } catch (e) {
      console.error("Add to collage failed:", e);
      await alertDialog({
        title: "Add to collage failed",
        body: "Failed to capture the figure preview. Check the console.",
      });
    }
  };

  return (
    <>
      {/* The Builder ↔ Collage toggle buttons were removed — navigation
          now lives in the DocumentTabs strip (Collage tab + one tab per
          open .mpf). "Add to Collage" stays: it renders the current
          builder figure into the collage. */}
      {mode === "builder" && (
        <Tooltip title="Render the current figure and add it to the Collage Assembly">
          <Button
            variant="outlined"
            color="primary"
            size="small"
            startIcon={<LibraryAddIcon />}
            onClick={handleAddToCollage}
            sx={{ textTransform: "none" }}
          >
            Add to Collage
          </Button>
        </Tooltip>
      )}
    </>
  );
}

export function Toolbar() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const uploadImages = useFigureStore((s) => s.uploadImages);
  const uploadImagesFromPaths = useFigureStore((s) => s.uploadImagesFromPaths);
  const mode = useCollageStore((s) => s.mode);
  const fileRef = useRef<HTMLInputElement>(null);
  const [saveDlgOpen, setSaveDlgOpen] = useState(false);
  const [newConfirmOpen, setNewConfirmOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(() => {
    // Auto-open only the first time the toolbar mounts this session.
    if (_aboutAutoShown) return false;
    _aboutAutoShown = true;
    return true;
  });
  const [updateStatus, setUpdateStatus] = useState<"idle" | "checking" | "up-to-date" | "available" | "downloading" | "ready" | "error">("idle");
  const [latestVersion, setLatestVersion] = useState<string | null>(null);
  const [releaseNotes, setReleaseNotes] = useState("");
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadTotal, setDownloadTotal] = useState<number | null>(null);
  const [updateRef, setUpdateRef] = useState<Awaited<ReturnType<typeof check>> | null>(null);
  const [citationCopied, setCitationCopied] = useState(false);
  const [appVersion, setAppVersion] = useState("...");
  const [updateChannel, setUpdateChannel] = useState<"stable" | "experimental">(() => {
    return (localStorage.getItem("mpfig_update_channel") as "stable" | "experimental") || "stable";
  });

  const toggleChannel = (channel: "stable" | "experimental") => {
    setUpdateChannel(channel);
    localStorage.setItem("mpfig_update_channel", channel);
    setUpdateStatus("idle");
  };

  const [changelog, setChangelog] = useState<ChangelogEntry[]>([]);

  useEffect(() => {
    getVersion().then((v) => setAppVersion(v)).catch(() => setAppVersion("unknown"));
  }, []);

  // The Help menu now lives in the always-visible DocumentTabs bar; it opens
  // this About dialog by firing a window event (avoids lifting all the update
  // state out of the toolbar).
  useEffect(() => {
    const onOpen = () => setAboutOpen(true);
    window.addEventListener("mpfig:open-about", onOpen);
    return () => window.removeEventListener("mpfig:open-about", onOpen);
  }, []);

  // Fetch changelog when About dialog opens
  useEffect(() => {
    if (aboutOpen) {
      fetchChangelog().then(setChangelog);
    }
  }, [aboutOpen]);

  const imageCount = Object.keys(loadedImages).length;

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const fileArr = Array.from(files);
    try {
      await uploadImages(fileArr);
    } catch (err) {
      console.error("Image upload failed:", err);
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleLoadMedia = async () => {
    try {
      // Try Tauri native file dialog — returns file paths, avoids base64/IPC limits
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: true,
        filters: [{
          name: "Images & Video",
          extensions: ["tif", "tiff", "png", "jpg", "jpeg", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "pef", "raf", "nd2", "mp4", "avi", "mov", "mkv", "webm", "wmv", "flv", "m4v", "mpg", "mpeg", "3gp", "ts", "mts"],
        }],
      });
      if (selected) {
        const items = Array.isArray(selected) ? selected : [selected];
        // open() may return strings or {path, name} objects depending on version
        const paths = items.map((item: unknown) =>
          typeof item === "string" ? item : (item as { path: string }).path
        ).filter(Boolean);
        if (paths.length > 0) {
          await uploadImagesFromPaths(paths);
        }
      }
    } catch {
      // If dialog import fails (dev mode), fall back to HTML file input
      fileRef.current?.click();
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1.5,
        px: 1.5,
        py: 0.75,
        borderBottom: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        flexShrink: 0,
        flexWrap: "wrap",
      }}
    >
      {/* The Load Media / file-count chip / New trio belongs to the
          multi-panel builder workflow only. In collage mode they're
          hidden — the collage has its own Import image / Import
          project buttons inside CollageView's toolbar. */}
      {mode === "builder" && (
        <>
          <Button
            variant="contained"
            startIcon={<AddPhotoAlternateIcon />}
            onClick={handleLoadMedia}
          >
            Load Media
          </Button>

          <input
            ref={fileRef}
            type="file"
            accept=".tif,.tiff,.png,.jpg,.jpeg,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf,.nd2,.mp4,.avi,.mov,.mkv,.webm,.wmv,.flv,.m4v,.mpg,.mpeg,.3gp,.ts,.mts"
            multiple
            style={{ display: "none" }}
            aria-label="Load image files"
            onChange={(e) => handleFiles(e.target.files)}
          />

          <Chip
            label={`${imageCount} file${imageCount !== 1 ? "s" : ""}`}
            size="small"
            variant="outlined"
          />

          <Tooltip title="New figure — clears all panels, images, and settings">
            <Button
              size="small"
              variant="outlined"
              color="error"
              startIcon={<RestartAltIcon />}
              onClick={() => {
                setNewConfirmOpen(true);
              }}
            >
              New
            </Button>
          </Tooltip>
        </>
      )}

      {/* New figure confirmation dialog */}
      <Dialog open={newConfirmOpen} onClose={() => setNewConfirmOpen(false)}>
        <DialogTitle>New Figure</DialogTitle>
        <DialogContent>
          <Typography>Start a new figure? All current images, settings, and panels will be cleared.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewConfirmOpen(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={async () => {
            setNewConfirmOpen(false);
            // "New" is a deliberate reset — clear the persisted collage
            // so any stash of analysis plots doesn't outlive this
            // session, and signal the beforeunload handler to skip
            // its analysis-plots warning so the upcoming reload
            // actually runs (otherwise the browser cancels it).
            try { useCollageStore.getState().clear(); } catch { /* store not ready */ }
            (window as unknown as { __mpfigAllowUnload?: boolean }).__mpfigAllowUnload = true;
            try {
              // Preserve user-defined scale bars
              const savedScales = await api.getResolutions().catch(() => ({}));
              // Reset backend to fresh 2x2 grid
              await api.updateConfig({
                rows: 2, cols: 2, spacing: 0.02, output_format: "TIFF", background: "White",
                panels: [[{} as never, {} as never], [{} as never, {} as never]],
                column_labels: [
                  { text: "Column 1", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Top", rotation: 0, styled_segments: [], visible: true },
                  { text: "Column 2", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Top", rotation: 0, styled_segments: [], visible: true },
                ] as never,
                row_labels: [
                  { text: "Row 1", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Left", rotation: 90, styled_segments: [], visible: true },
                  { text: "Row 2", font_size: 12, font_name: "arial.ttf", font_path: null, font_style: [], default_color: "#000000", distance: 0.01, position: "Left", rotation: 90, styled_segments: [], visible: true },
                ] as never,
                column_headers: [], row_headers: [],
                resolution_entries: savedScales, dpi: 300,
              });
              // Delete all loaded images
              const imgs = await api.listImages();
              for (const name of imgs.names) {
                await api.deleteImage(name).catch(() => {});
              }
            } catch (err) {
              console.error("Clear session failed", err);
            }
            // Full reload to reset frontend state
            window.location.reload();
          }}>Confirm</Button>
        </DialogActions>
      </Dialog>

      <Box sx={{ flex: 1 }} />

      {/* Collage workspace toggle + Add to Collage */}
      <CollageWorkspaceControls />

      {/* Save figure / Save collage — same button, two modes. The
          collage path renders the canvas client-side via <canvas>
          and ships the bytes through the existing save_base64_to_path
          Tauri command (or download fallback in browser). */}
      {mode === "builder" ? (
        <Button
          variant="contained"
          color="secondary"
          startIcon={<SaveIcon />}
          onClick={() => setSaveDlgOpen(true)}
        >
          Save Figure
        </Button>
      ) : (
        <SaveCollageButton />
      )}

      {/* Record app + Help moved to the DocumentTabs bar so they sit on one
          always-visible top level (and a recording keeps running across every
          mode). The About dialog below is opened from there via the
          "mpfig:open-about" window event. */}

      <SaveFigureDialog open={saveDlgOpen} onClose={() => setSaveDlgOpen(false)} />

      {/* About Dialog */}
      <Dialog open={aboutOpen} onClose={() => { setAboutOpen(false); setUpdateStatus("idle"); }} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ pb: 1 }}>About</DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: "center", py: 2 }}>
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
              Multi-Panel Figure Builder
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Version {appVersion}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Created by <strong>Zhuojian Look</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
              A tool for creating professional multi-panel scientific figures
              with full control over layout, annotations, scale bars, and image adjustments. For the benefit of scientists.
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Update Channel Toggle */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1, mb: 1 }}>
            <Typography variant="caption" sx={{ fontSize: "0.65rem", color: "text.secondary" }}>Update channel:</Typography>
            <Button
              size="small"
              variant={updateChannel === "stable" ? "contained" : "outlined"}
              onClick={() => toggleChannel("stable")}
              sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.1, px: 1, minWidth: 0 }}
            >Stable</Button>
            <Button
              size="small"
              variant={updateChannel === "experimental" ? "contained" : "outlined"}
              color="warning"
              onClick={() => toggleChannel("experimental")}
              sx={{ fontSize: "0.55rem", textTransform: "none", py: 0.1, px: 1, minWidth: 0 }}
            >Experimental</Button>
          </Box>
          {updateChannel === "experimental" && (
            <Typography variant="caption" sx={{ fontSize: "0.55rem", color: "warning.main", textAlign: "center", mb: 0.5 }}>
              Experimental updates may contain unstable features
            </Typography>
          )}

          {/* Check for Updates */}
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1, mb: 2 }}>
            <Button
              variant="outlined"
              size="small"
              startIcon={updateStatus === "checking" ? <CircularProgress size={14} /> : <SystemUpdateAltIcon />}
              disabled={updateStatus === "checking" || updateStatus === "downloading"}
              onClick={async () => {
                setUpdateStatus("checking");
                setLatestVersion(null);
                setUpdateRef(null);
                try {
                  // Fetch the correct manifest based on channel
                  const manifestFile = updateChannel === "experimental" ? "latest-experimental.json" : "latest.json";
                  const manifestUrl = `https://raw.githubusercontent.com/zhuojianlook/multipanelfigure/updater/${manifestFile}`;

                  // Fetch manifest via Rust proxy (WebView blocks cross-origin)
                  const manifestText = await proxyFetch(manifestUrl);
                  const manifest = JSON.parse(manifestText) as { version: string; notes: string };
                  const latestVer = manifest.version || "";

                  // Compare versions
                  const current = appVersion.split(".").map(Number);
                  const latest = latestVer.split(".").map(Number);
                  const isNewer = latest[0] > current[0] ||
                    (latest[0] === current[0] && latest[1] > current[1]) ||
                    (latest[0] === current[0] && latest[1] === current[1] && (latest[2] || 0) > (current[2] || 0));

                  if (isNewer) {
                    setLatestVersion(latestVer);
                    setReleaseNotes(manifest.notes || "");
                    // Try Tauri updater for the actual download
                    try {
                      const update = await check();
                      if (update) {
                        setUpdateRef(update);
                      }
                    } catch {
                      // check() may fail for experimental channel — that's ok,
                      // we'll still show the update is available
                    }
                    setUpdateStatus("available");
                  } else {
                    setUpdateStatus("up-to-date");
                  }
                } catch (e: unknown) {
                  console.error("Update check failed:", e);
                  const msg = e instanceof Error ? e.message : String(e);
                  setReleaseNotes(msg);
                  setUpdateStatus("error");
                }
              }}
            >
              {updateStatus === "checking" ? "Checking..." : "Check for Updates"}
            </Button>

            {updateStatus === "up-to-date" && (
              <Alert severity="success" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                You are running the latest {updateChannel} version ({appVersion}).
              </Alert>
            )}
            {updateStatus === "available" && (
              <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Typography sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
                  Version {latestVersion} is available!
                </Typography>
                {/* Show changelog of what's new since current version */}
                <Box sx={{ mt: 1, maxHeight: 160, overflowY: "auto" }}>
                  {changelog.filter((entry: ChangelogEntry) => {
                    // Show entries newer than current version
                    const current = appVersion.split(".").map(Number);
                    const entry_v = entry.version.split(".").map(Number);
                    for (let i = 0; i < 3; i++) {
                      if ((entry_v[i] || 0) > (current[i] || 0)) return true;
                      if ((entry_v[i] || 0) < (current[i] || 0)) return false;
                    }
                    return false;
                  }).map((entry: ChangelogEntry) => (
                    <Box key={entry.version} sx={{ mb: 1 }}>
                      <Typography sx={{ fontWeight: 600, fontSize: "0.7rem" }}>
                        v{entry.version} — {entry.date}
                      </Typography>
                      <Box component="ul" sx={{ m: 0, pl: 2, "& li": { fontSize: "0.65rem", color: "text.secondary", lineHeight: 1.4 } }}>
                        {entry.changes.map((change: string, i: number) => (
                          <li key={i}>{change}</li>
                        ))}
                      </Box>
                    </Box>
                  ))}
                </Box>
                <Button size="small" variant="contained" color="primary" sx={{ mt: 0.5, fontSize: "0.65rem", textTransform: "none" }}
                  startIcon={<DownloadIcon />}
                  onClick={async () => {
                    const { invoke } = await import("@tauri-apps/api/core");
                    const { listen } = await import("@tauri-apps/api/event");
                    try {
                      try { await invoke("kill_sidecar"); } catch { /* ignore */ }
                      setUpdateStatus("downloading");
                      setDownloadProgress(0);
                      setDownloadTotal(null);

                      if (updateChannel === "stable" && updateRef) {
                        let downloaded = 0;
                        await updateRef.downloadAndInstall((event) => {
                          if (event.event === "Started") {
                            setDownloadProgress(0);
                            setDownloadTotal(event.data.contentLength ?? null);
                          } else if (event.event === "Progress") {
                            downloaded += event.data.chunkLength;
                            setDownloadProgress(downloaded);
                          }
                        });
                      } else {
                        // Experimental: use Rust command with custom endpoint
                        // Listen for the same progress events the Rust side
                        // emits, so the UI can show "X MB / Y MB" exactly
                        // like the stable path.
                        const unlisten = await listen<{ downloaded: number; total: number | null }>(
                          "updater://progress",
                          (e) => {
                            setDownloadProgress(e.payload.downloaded);
                            if (e.payload.total) setDownloadTotal(e.payload.total);
                          }
                        );
                        const manifestFile = updateChannel === "experimental" ? "latest-experimental.json" : "latest.json";
                        const manifestUrl = `https://raw.githubusercontent.com/zhuojianlook/multipanelfigure/updater/${manifestFile}`;
                        const timeoutMs = 3 * 60 * 1000;
                        const timeoutPromise = new Promise((_, reject) =>
                          setTimeout(() => reject(new Error("Download timed out after 3 minutes")), timeoutMs)
                        );
                        try {
                          await Promise.race([
                            invoke("download_and_install_update", { manifestUrl }),
                            timeoutPromise,
                          ]);
                        } finally {
                          unlisten();
                        }
                      }
                      setUpdateStatus("ready");
                    } catch (e: unknown) {
                      console.error("Update failed:", e);
                      const errMsg = e instanceof Error ? e.message : String(e);
                      // On failure, offer browser download fallback
                      const tag = updateChannel === "experimental" ? `exp-${latestVersion}` : `v${latestVersion}`;
                      const releaseUrl = `https://github.com/zhuojianlook/multipanelfigure/releases/tag/${tag}`;
                      try {
                        const { open } = await import("@tauri-apps/plugin-shell");
                        await open(releaseUrl);
                        setReleaseNotes(`In-app update failed (${errMsg}). Opened browser for manual download.`);
                      } catch {
                        setReleaseNotes(errMsg);
                      }
                      setUpdateStatus("error");
                    }
                  }}
                >
                  Download & Install Update
                </Button>
              </Alert>
            )}
            {updateStatus === "downloading" && (() => {
              const dlMB = downloadProgress / 1024 / 1024;
              const totalMB = downloadTotal ? downloadTotal / 1024 / 1024 : null;
              const pct = downloadTotal && downloadTotal > 0
                ? Math.min(100, Math.round((downloadProgress / downloadTotal) * 100))
                : null;
              const text = downloadProgress > 0
                ? (totalMB != null
                    ? `(${dlMB.toFixed(1)} MB / ${totalMB.toFixed(1)} MB${pct != null ? ` — ${pct}%` : ""})`
                    : `(${dlMB.toFixed(1)} MB)`)
                : "";
              return (
                <Alert severity="info" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <CircularProgress size={16} />
                    <Typography sx={{ fontSize: "0.75rem" }}>
                      Downloading update... {text}
                    </Typography>
                  </Box>
                </Alert>
              );
            })()}
            {updateStatus === "ready" && (
              <Alert severity="success" sx={{ py: 0.5, fontSize: "0.75rem", width: "100%" }}>
                <Typography sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
                  Update installed! Restart to apply.
                </Typography>
                <Button size="small" variant="contained" color="success" sx={{ mt: 0.5, fontSize: "0.65rem", textTransform: "none" }}
                  onClick={async () => {
                    try {
                      const { invoke } = await import("@tauri-apps/api/core");
                      await invoke("kill_sidecar");
                    } catch { /* ignore */ }
                    await relaunch();
                  }}
                >
                  Restart Now
                </Button>
              </Alert>
            )}
            {updateStatus === "error" && (
              <Alert severity="warning" sx={{ py: 0, fontSize: "0.75rem", width: "100%" }}>
                Could not check for updates. {releaseNotes ? `Error: ${releaseNotes}` : "Please check your internet connection."}
              </Alert>
            )}
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Citation */}
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Citation</Typography>
          <Box sx={{
            bgcolor: "action.hover",
            borderRadius: 1,
            p: 1.5,
            mb: 2,
            position: "relative",
            fontFamily: "monospace",
            fontSize: "0.7rem",
            lineHeight: 1.5,
            color: "text.secondary",
          }}>
            <Typography sx={{ fontFamily: "inherit", fontSize: "inherit", lineHeight: "inherit", color: "inherit" }}>
              Look, Z. (2026). Multi-Panel Figure Builder (Version {appVersion}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure
            </Typography>
            <Tooltip title={citationCopied ? "Copied!" : "Copy citation"}>
              <IconButton
                size="small"
                sx={{ position: "absolute", top: 4, right: 4 }}
                onClick={() => {
                  navigator.clipboard.writeText(
                    `Look, Z. (2026). Multi-Panel Figure Builder (Version ${appVersion}) [Computer software]. https://github.com/zhuojianlook/multipanelfigure`
                  );
                  setCitationCopied(true);
                  setTimeout(() => setCitationCopied(false), 2000);
                }}
              >
                <ContentCopyIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Changelog — collapsible */}
          <Accordion disableGutters elevation={0} sx={{ bgcolor: "transparent", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0, minHeight: 32 }}>
              <Typography variant="subtitle2">Changelog</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0, pt: 0 }}>
              {changelog.length === 0 ? (
                <Typography variant="caption" sx={{ color: "text.disabled" }}>Loading changelog...</Typography>
              ) : changelog.map((entry) => (
                <Box key={entry.version} sx={{ mb: 1.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    v{entry.version} — {entry.date}
                  </Typography>
                  <Box component="ul" sx={{ m: 0, pl: 2.5, "& li": { fontSize: "0.75rem", color: "text.secondary" } }}>
                    {entry.changes.map((change, i) => (
                      <li key={i}>{change}</li>
                    ))}
                  </Box>
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setAboutOpen(false); setUpdateStatus("idle"); }}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// ── Developer options + screen recorder ──────────────────────
export const DEV_OPTIONS_KEY = "mpfig.dev_options_enabled";

/** Mounted ONCE in the always-present DocumentTabs bar (so the button — and an
 *  in-progress recording — persists across every mode/tab; the recorder used to
 *  live in the mode-gated Toolbar, which unmounted it when switching to
 *  Analysis and killed the capture). Shown only when developer options are on.
 *  Manages a getDisplayMedia + MediaRecorder (or native ffmpeg) session and
 *  writes the result to disk. */
export function RecordAppButton() {
  // Drive visibility from the persisted flag PLUS the in-window
  // event the Help menu fires so the button flips on/off without
  // a remount of the host shell.
  const [visible, setVisible] = useState<boolean>(() => {
    try { return localStorage.getItem(DEV_OPTIONS_KEY) === "1"; } catch { return false; }
  });
  useEffect(() => {
    const onChange = (e: Event) => {
      const detail = (e as CustomEvent).detail as { enabled?: boolean } | undefined;
      setVisible(!!detail?.enabled);
    };
    window.addEventListener("mpfig:dev-options-changed", onChange);
    return () => window.removeEventListener("mpfig:dev-options-changed", onChange);
  }, []);
  const [recording, setRecording] = useState(false);
  const [recError, setRecError] = useState<string>("");
  const mediaRecRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  // Native (ffmpeg) recording state — used on macOS, where the WKWebView
  // has no navigator.mediaDevices.getDisplayMedia.
  const nativeChildRef = useRef<{ write: (d: string) => Promise<void>; kill: () => Promise<void> } | null>(null);
  // Path of the temp file ffmpeg writes WHILE recording. We capture to a temp
  // file and only ask the user where to save AFTER they stop (matching the web
  // path's "record now, name later" UX).
  const nativeTmpRef = useRef<string>("");
  // True while WE are deliberately stopping the recorder. Lets the process
  // "close" handler tell an intentional Stop (handled entirely by stopNative)
  // apart from an unexpected exit (permission denied / crash → show an error).
  const nativeIntentionalStopRef = useRef<boolean>(false);

  // Elapsed-time indicator (drives the "● 00:12" recording chip). Counts from
  // recStartRef while `recording` is true (see the effect below).
  const [recElapsed, setRecElapsed] = useState(0);
  const recStartRef = useRef<number>(0);
  // After a native recording stops we open this modal instead of a bare save
  // dialog, so the user can Discard or Save As… in a chosen video format.
  const [saveModal, setSaveModal] = useState<{ tmp: string; durationSec: number } | null>(null);
  const [saveFormat, setSaveFormat] = useState<"mp4" | "mov" | "gif" | "webm">("mp4");
  const [saving, setSaving] = useState(false);

  // Run the elapsed-time clock purely off the `recording` flag so every start
  // path (native + web) and every stop path reset it consistently.
  useEffect(() => {
    if (!recording) { setRecElapsed(0); return; }
    recStartRef.current = Date.now();
    setRecElapsed(0);
    const id = setInterval(() => {
      setRecElapsed(Math.max(0, Math.floor((Date.now() - recStartRef.current) / 1000)));
    }, 500);
    return () => clearInterval(id);
  }, [recording]);

  const fmtDuration = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  // Jump to macOS Screen Recording settings (used from the permission-denied
  // dialog and the save modal's "looks black?" hint).
  const openScreenRecordingSettings = async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("open_url", { url: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture" });
    } catch { /* ignore */ }
  };

  /** True when the webview supports browser screen capture (Windows WebView2,
   *  most Linux WebKitGTK, and any plain-browser dev preview). macOS WKWebView
   *  returns false → we fall back to native ffmpeg capture. */
  const canWebCapture = () =>
    typeof navigator !== "undefined"
    && !!navigator.mediaDevices
    && typeof navigator.mediaDevices.getDisplayMedia === "function";

  // Best-effort detection of which container/codec the host webview
  // is willing to record. Falls back through a few standard ones.
  const pickMimeType = () => {
    const candidates = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ];
    for (const t of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  };

  const stopWeb = async (save: boolean) => {
    const rec = mediaRecRef.current;
    const stream = streamRef.current;
    if (!rec) return;
    return new Promise<void>((resolve) => {
      rec.onstop = async () => {
        try { stream?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
        streamRef.current = null;
        mediaRecRef.current = null;
        setRecording(false);
        if (!save) { chunksRef.current = []; resolve(); return; }
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "video/webm" });
        chunksRef.current = [];
        const ext = blob.type.includes("mp4") ? "mp4" : "webm";
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        // Try Tauri save flow first; fall back to a browser download.
        try {
          const { save: saveDialog } = await import("@tauri-apps/plugin-dialog");
          const { invoke } = await import("@tauri-apps/api/core");
          const path = await saveDialog({
            defaultPath: `mpfig-recording-${stamp}.${ext}`,
            filters: [{ name: "Video", extensions: [ext] }],
          });
          if (path) {
            const buf = await blob.arrayBuffer();
            const b64 = btoa(Array.from(new Uint8Array(buf), (b) => String.fromCharCode(b)).join(""));
            await invoke("save_base64_to_path", { path, dataB64: b64 });
          }
        } catch {
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `mpfig-recording-${stamp}.${ext}`;
          a.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
        resolve();
      };
      try { rec.stop(); } catch { resolve(); }
    });
  };

  const startWeb = async () => {
    setRecError("");
    try {
      // Ask the host for a screen / window stream.  In Tauri this
      // opens the system screen-share picker.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 } as MediaTrackConstraints,
        audio: false,
      });
      streamRef.current = stream;
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      rec.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data); };
      rec.start(1000);  // emit a chunk every 1s so we don't lose much on crash
      mediaRecRef.current = rec;
      setRecording(true);
      // If the user stops sharing via the OS-level UI, end gracefully.
      stream.getVideoTracks()[0]?.addEventListener("ended", () => { stopWeb(true); });
    } catch (e) {
      setRecError(e instanceof Error ? e.message : String(e));
    }
  };

  // ── Native (ffmpeg) screen capture for macOS WKWebView ──

  /** Query ffmpeg for the avfoundation index of the screen-capture device.
   *  Prefers "Capture screen 0" (the main display); else the first "Capture
   *  screen N". THROWS if none is found instead of guessing a numeric index —
   *  a wrong guess can land on a camera (e.g. a Continuity Camera at [1]) and
   *  silently record the wrong thing. */
  const getScreenDeviceIndex = async (): Promise<string> => {
    const { Command } = await import("@tauri-apps/plugin-shell");
    const out = await Command.sidecar("binaries/ffmpeg", [
      "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", "",
    ]).execute();
    const text = `${out.stderr || ""}\n${out.stdout || ""}`;
    const m = text.match(/\[(\d+)\]\s+Capture screen 0\b/)
      || text.match(/\[(\d+)\]\s+Capture screen\b/);
    if (m) return m[1];
    throw new Error(
      "No screen-capture device was found by ffmpeg. Make sure macOS Screen "
      + "Recording permission is enabled for this app (System Settings → "
      + "Privacy & Security → Screen Recording), then fully quit and reopen it.",
    );
  };

  const startNative = async () => {
    setRecError("");
    try {
      // Record straight to a temp file — we ask WHERE to save only after the
      // user stops (see stopNative + the close handler below). This makes the
      // button behave like a real recorder: press to start, press to stop &
      // save.
      const { tempDir, join } = await import("@tauri-apps/api/path");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const path = await join(await tempDir(), `mpfig-recording-${stamp}.mp4`);
      const idx = await getScreenDeviceIndex();
      // Crop the screen capture down to JUST this app window. avfoundation can
      // only grab a whole display, so we compute the window's rectangle as
      // fractions of its monitor (scale-independent — works on Retina/scaled
      // modes) and let ffmpeg's crop filter resolve them against the real
      // capture resolution at runtime. Falls back to full-screen if the
      // window/monitor geometry can't be read.
      let cropVf: string | null = null;
      try {
        const { getCurrentWindow, currentMonitor } = await import("@tauri-apps/api/window");
        const win = getCurrentWindow();
        const [pos, size, mon] = await Promise.all([win.outerPosition(), win.outerSize(), currentMonitor()]);
        if (mon && mon.size.width > 0 && mon.size.height > 0) {
          const fx = (pos.x - mon.position.x) / mon.size.width;
          const fy = (pos.y - mon.position.y) / mon.size.height;
          const fw = size.width / mon.size.width;
          const fh = size.height / mon.size.height;
          const cx = Math.max(0, Math.min(1, fx));
          const cy = Math.max(0, Math.min(1, fy));
          const cw = Math.max(0.02, Math.min(1 - cx, fw));
          const ch = Math.max(0.02, Math.min(1 - cy, fh));
          cropVf = `crop=floor(iw*${cw.toFixed(5)}/2)*2:floor(ih*${ch.toFixed(5)}/2)*2:floor(iw*${cx.toFixed(5)}/2)*2:floor(ih*${cy.toFixed(5)}/2)*2`;
        }
      } catch (e) {
        console.warn("[record] window geometry unavailable — recording full screen", e);
      }
      const { Command } = await import("@tauri-apps/plugin-shell");
      const cmd = Command.sidecar("binaries/ffmpeg", [
        "-hide_banner", "-y",
        "-f", "avfoundation", "-capture_cursor", "1", "-framerate", "30",
        "-i", `${idx}:none`,
        ...(cropVf ? ["-vf", cropVf] : []),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        // 1-second keyframe interval + fragmented mp4. This makes the output a
        // valid, playable file even if ffmpeg is force-killed (the moov atom is
        // written up front and each ~1s fragment is self-contained), so Stop
        // never depends on ffmpeg shutting down gracefully.
        "-g", "30",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        path,
      ]);
      let stderrTail = "";
      cmd.stderr.on("data", (line: string) => { stderrTail = (stderrTail + line).slice(-1200); });
      cmd.on("error", (err: string) => {
        nativeChildRef.current = null;
        setRecording(false);
        setRecError(`ffmpeg failed to start: ${err}`);
      });
      cmd.on("close", async () => {
        nativeChildRef.current = null;
        setRecording(false);
        // An intentional Stop is finalized by stopNative (kill → save dialog →
        // move). Don't double-handle it here.
        if (nativeIntentionalStopRef.current) return;
        // Otherwise ffmpeg exited on its own — almost always missing macOS
        // Screen Recording permission (it opens the output, fails to grab the
        // screen, and quits). Surface a clear dialog and discard the temp file.
        const tmp = nativeTmpRef.current;
        nativeTmpRef.current = "";
        if (tmp) { try { const { invoke } = await import("@tauri-apps/api/core"); await invoke("delete_file", { path: tmp }); } catch { /* ignore */ } }
        const perm = /denied|not authorized|permission|abort|Operation not permitted/i.test(stderrTail);
        if (perm) {
          setRecError("Screen Recording permission needed — see the dialog.");
          const go = await confirmDialog({
            title: "Screen Recording permission needed",
            body: "macOS hasn't granted Screen Recording permission to the recorder, so it "
              + "couldn't capture anything.\n\nEnable it in System Settings → Privacy & Security "
              + "→ Screen Recording, then fully quit and reopen the app and try again.",
            confirmLabel: "Open Settings",
            cancelLabel: "Close",
          });
          if (go) await openScreenRecordingSettings();
        } else {
          setRecError("Recording stopped unexpectedly.");
          await alertDialog({ title: "Recording failed", body: `Recording stopped unexpectedly.\n\n${stderrTail.slice(-400) || "No ffmpeg output."}` });
        }
      });
      const child = await cmd.spawn();
      nativeChildRef.current = child as unknown as { write: (d: string) => Promise<void>; kill: () => Promise<void> };
      nativeTmpRef.current = path;
      nativeIntentionalStopRef.current = false;
      setRecording(true);
    } catch (e) {
      setRecording(false);
      const msg = e instanceof Error ? e.message : String(e);
      setRecError(msg);
      // Surface the failure in a dialog — previously it only set the tooltip,
      // so a blocked spawn / missing device looked like "the save dialog did
      // nothing".
      await alertDialog({ title: "Couldn't start recording", body: msg });
    }
  };

  // Stop the native recorder and open the Save/Discard modal (instead of a bare
  // save dialog). Kill is reliable; fragmented-mp4 output stays valid on a hard
  // kill, so we don't depend on the "close" event or on ffmpeg reading "q".
  const stopNativeToModal = async () => {
    const child = nativeChildRef.current;
    if (!child) return;
    const durationSec = recElapsed;
    nativeIntentionalStopRef.current = true;
    setRecording(false);  // instant UI feedback (also stops the timer)
    try { await child.kill(); } catch { /* ignore */ }
    nativeChildRef.current = null;
    // Let the OS flush + close the file handle before we touch the file.
    await new Promise((r) => setTimeout(r, 400));
    nativeIntentionalStopRef.current = false;
    const tmp = nativeTmpRef.current;
    nativeTmpRef.current = "";
    if (!tmp) return;
    setSaveFormat("mp4");
    setSaveModal({ tmp, durationSec });
  };

  // Transcode/remux the recorded temp clip to the chosen container/codec via the
  // bundled ffmpeg. mp4/mov from our mp4 temp are a fast stream-copy; gif/webm
  // re-encode. Throws on a non-zero ffmpeg exit.
  const transcodeRecording = async (src: string, dest: string, fmt: string) => {
    const { Command } = await import("@tauri-apps/plugin-shell");
    let args: string[];
    if (fmt === "mp4") args = ["-y", "-i", src, "-c", "copy", "-movflags", "+faststart", dest];
    else if (fmt === "mov") args = ["-y", "-i", src, "-c", "copy", dest];
    else if (fmt === "webm") args = ["-y", "-i", src, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1", dest];
    else if (fmt === "gif") args = ["-y", "-i", src, "-vf", "fps=12,scale=720:-1:flags=lanczos", "-loop", "0", dest];
    else args = ["-y", "-i", src, dest];
    const out = await Command.sidecar("binaries/ffmpeg", args).execute();
    if (out.code !== 0) throw new Error((out.stderr || "").slice(-300) || "ffmpeg conversion failed");
  };

  const discardRecording = async () => {
    const m = saveModal;
    setSaveModal(null);
    if (!m) return;
    try { const { invoke } = await import("@tauri-apps/api/core"); await invoke("delete_file", { path: m.tmp }); } catch { /* ignore */ }
  };

  const saveRecordingAs = async () => {
    const m = saveModal;
    if (!m) return;
    const fmt = saveFormat;
    const { save: saveDialog } = await import("@tauri-apps/plugin-dialog");
    const { invoke } = await import("@tauri-apps/api/core");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const dest = await saveDialog({
      defaultPath: `mpfig-recording-${stamp}.${fmt}`,
      filters: [{ name: fmt.toUpperCase(), extensions: [fmt] }],
    });
    if (!dest) return;  // keep modal open so the user can pick again
    setSaving(true);
    try {
      try {
        await transcodeRecording(m.tmp, dest, fmt);
      } catch (convErr) {
        // If conversion failed but the user wanted mp4, fall back to a plain
        // move of the (already-mp4) temp so they don't lose the recording.
        if (fmt === "mp4") await invoke("move_file", { src: m.tmp, dest });
        else throw convErr;
      }
      try { await invoke("delete_file", { path: m.tmp }); } catch { /* ignore */ }
      setSaveModal(null);
      await alertDialog({ title: "Recording saved", body: `Saved to ${dest}` });
    } catch (err) {
      await alertDialog({ title: "Couldn't save recording", body: err instanceof Error ? err.message : String(err) });
    } finally {
      setSaving(false);
    }
  };

  // Dispatch to the web or native implementation.
  const startRecording = () => (canWebCapture() ? startWeb() : startNative());
  // Click on the recording chip → stop. Native opens the Save/Discard modal;
  // the web path keeps its existing save-on-stop flow.
  const onStopClick = () =>
    (mediaRecRef.current ? stopWeb(true) : nativeChildRef.current ? stopNativeToModal() : Promise.resolve());

  // If the host unmounts mid-recording, stop the stream / kill ffmpeg so the
  // OS recording indicator goes away.
  useEffect(() => {
    return () => {
      try { streamRef.current?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
      // Mark as intentional so the close handler stays quiet (no error dialog)
      // when we kill ffmpeg on unmount.
      if (nativeChildRef.current) nativeIntentionalStopRef.current = true;
      try { void nativeChildRef.current?.kill(); } catch { /* ignore */ }
    };
  }, []);

  if (!visible) return null;
  return (
    <>
      <Tooltip title={recording
        ? "Recording — click to stop"
        : (recError || "Record the app window (asks the OS for screen-share permission, then saves to a file)")}>
        <span>
          <Button
            variant={recording ? "contained" : "outlined"}
            color={recording ? "error" : "primary"}
            size="small"
            onClick={recording ? onStopClick : startRecording}
            startIcon={
              recording
                ? <Box sx={{ width: 9, height: 9, borderRadius: "50%", bgcolor: "white",
                            animation: "mpfig-rec-blink 1s linear infinite",
                            "@keyframes mpfig-rec-blink": { "50%": { opacity: 0.25 } } }} />
                : <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#e53935" }} />
            }
            sx={{
              textTransform: "none", fontVariantNumeric: "tabular-nums",
              minWidth: recording ? 84 : undefined,
              // Keep it compact so it sits inside the 30px DocumentTabs bar with
              // breathing room (was as tall as the bar and touched the border).
              height: 22, minHeight: 0, py: 0, fontSize: "0.72rem", lineHeight: 1,
            }}
          >
            {recording ? `Rec ${fmtDuration(recElapsed)}` : "Record app"}
          </Button>
        </span>
      </Tooltip>

      {/* Save / Discard modal shown after a native recording stops. */}
      <Dialog open={!!saveModal} onClose={() => { /* require an explicit Discard/Save choice */ }} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ pb: 0.5 }}>Recording finished</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ color: "text.secondary", mb: 2 }}>
            Captured {saveModal ? fmtDuration(saveModal.durationSec) : "0:00"}. Choose a format and save it, or discard it.
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
            <Typography variant="body2" sx={{ width: 60 }}>Format</Typography>
            <Select
              size="small"
              value={saveFormat}
              disabled={saving}
              onChange={(e) => setSaveFormat(e.target.value as typeof saveFormat)}
              sx={{ flex: 1 }}
            >
              <MenuItem value="mp4">MP4 (H.264) — best for video</MenuItem>
              <MenuItem value="mov">MOV (H.264) — QuickTime</MenuItem>
              <MenuItem value="webm">WebM (VP9) — web embed</MenuItem>
              <MenuItem value="gif">Animated GIF — slides/docs</MenuItem>
            </Select>
          </Box>
          {(saveFormat === "webm" || saveFormat === "gif") && (
            <Typography variant="caption" sx={{ color: "text.secondary", display: "block" }}>
              {saveFormat === "gif" ? "GIF is re-encoded at 12 fps / 720px wide." : "WebM is re-encoded with VP9 — this can take a while for long clips."}
            </Typography>
          )}
          <Typography variant="caption" sx={{ color: "text.secondary", display: "block", mt: 1.5 }}>
            Recording looks black?{" "}
            <Box component="span" onClick={() => void openScreenRecordingSettings()}
              sx={{ color: "primary.main", cursor: "pointer", textDecoration: "underline" }}>
              Grant Screen Recording permission
            </Box>{" "}then record again.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button color="error" disabled={saving} onClick={() => void discardRecording()} sx={{ textTransform: "none" }}>
            Discard
          </Button>
          <Button variant="contained" disabled={saving} onClick={() => void saveRecordingAs()}
            startIcon={saving ? <CircularProgress size={14} /> : <SaveIcon sx={{ fontSize: 16 }} />}
            sx={{ textTransform: "none" }}>
            {saving ? "Saving…" : "Save As…"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
