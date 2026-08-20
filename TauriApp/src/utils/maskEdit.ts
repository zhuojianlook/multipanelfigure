/**
 * Shared label-mask helpers for the interactive cellpose mask editor.
 *
 * A "label image" is an Int32Array with one integer per pixel: 0 = background,
 * N = the Nth cell. It round-trips to/from the backend as an RGBA-packed PNG
 * where R = label low byte and G = label high byte (the same encoding the
 * fluorescence Intensity picker and cellpose_plugin's `_labels16` use, so the
 * sidecar's `edited_label_paths` substitution accepts it unchanged).
 *
 * These are byte-for-byte the same as the copies inlined in
 * IntensityPickerDialog — kept here as the single source of truth for the
 * cellpose editor. The packing MUST match the backend contract, so change both
 * halves together if it ever moves.
 */

/** Decode an RGBA-packed label PNG (base64, no data: prefix) into an
 *  Int32Array of per-pixel IDs (R = low byte, G = high byte). */
export async function decodeRgbaLabels(
  b64: string,
): Promise<{ labels: Int32Array; w: number; h: number } | null> {
  return new Promise((resolve) => {
    const img = new window.Image();
    img.onload = () => {
      const cnv = document.createElement("canvas");
      cnv.width = img.naturalWidth;
      cnv.height = img.naturalHeight;
      const ctx = cnv.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, cnv.width, cnv.height).data;
      const n = cnv.width * cnv.height;
      const labels = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        labels[i] = data[i * 4] | (data[i * 4 + 1] << 8);
      }
      resolve({ labels, w: cnv.width, h: cnv.height });
    };
    img.onerror = () => resolve(null);
    img.src = b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
  });
}

/** Encode an Int32Array label image back to an RGBA-packed PNG data URL for
 *  round-tripping to the backend (R = lo, G = hi). */
export function encodeRgbaLabels(labels: Int32Array, w: number, h: number): string {
  const cnv = document.createElement("canvas");
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext("2d");
  if (!ctx) return "";
  const imageData = ctx.createImageData(w, h);
  const d = imageData.data;
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i] | 0;
    d[i * 4] = v & 0xFF;
    d[i * 4 + 1] = (v >>> 8) & 0xFF;
    d[i * 4 + 2] = 0;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return cnv.toDataURL("image/png");
}

/** Derive a 4-neighbour boundary mask from an Int32Array label image. */
export function deriveBoundary(labels: Int32Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = labels[i];
      if (v === 0) continue;
      const up = y > 0 ? labels[i - w] : -1;
      const dn = y < h - 1 ? labels[i + w] : -1;
      const lf = x > 0 ? labels[i - 1] : -1;
      const rt = x < w - 1 ? labels[i + 1] : -1;
      if (v !== up || v !== dn || v !== lf || v !== rt) out[i] = 1;
    }
  }
  return out;
}

/** Remap the distinct non-zero IDs of a label image to a contiguous 1..N so
 *  there are no gaps. Editing (delete/merge) leaves holes in the ID sequence;
 *  the backend counts cells as max(id), so a gap would over-count. Relabelling
 *  before send keeps the reported cell count honest. Returns a fresh array. */
export function relabelContiguous(labels: Int32Array): Int32Array {
  const remap = new Map<number, number>();
  let next = 1;
  const out = new Int32Array(labels.length);
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (v <= 0) continue;
    let m = remap.get(v);
    if (m === undefined) { m = next++; remap.set(v, m); }
    out[i] = m;
  }
  return out;
}

/** Count the number of distinct non-zero IDs in a label image. */
export function countNonZeroIds(labels: Int32Array): number {
  const seen = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (v > 0) seen.add(v);
  }
  return seen.size;
}

/** Render a boundary mask onto a transparent canvas in the given colour. */
export function renderBoundaryCanvas(
  boundary: Uint8Array, w: number, h: number, rgb: [number, number, number],
): HTMLCanvasElement {
  const cnv = document.createElement("canvas");
  cnv.width = w; cnv.height = h;
  const ctx = cnv.getContext("2d");
  if (!ctx) return cnv;
  const imageData = ctx.createImageData(w, h);
  const d = imageData.data;
  for (let i = 0; i < boundary.length; i++) {
    if (!boundary[i]) continue;
    d[i * 4] = rgb[0];
    d[i * 4 + 1] = rgb[1];
    d[i * 4 + 2] = rgb[2];
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return cnv;
}
