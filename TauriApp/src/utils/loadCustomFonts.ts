/* ──────────────────────────────────────────────────────────
   loadCustomFonts — register @font-face declarations for every
   font reported by /api/fonts so the browser can render them
   by their filename (or stripped-extension equivalent) in CSS.

   The backend's matplotlib / PIL renderer resolves a font like
   "ArialNarrowItalic.ttf" via a filesystem search — it loads the
   actual file.  The browser doesn't know about those files, so
   `font-family: "ArialNarrowItalic"` falls back to sans-serif
   unless we explicitly register a @font-face that ties the name
   to the same bytes.

   Tauri's WebView CSP blocks direct binary fetches at
   /api/fonts/file/<name>, so we go via the api client which
   proxies through Rust IPC: fetch base64, build a Blob, create
   an object URL, register @font-face with that URL.  This means
   the CSS overlay sees the EXACT font the backend uses, no
   fallback or guess.

   Idempotent — calling twice for the same font is a no-op.

   Usage:
     useEffect(() => { loadCustomFonts(fonts); }, [fonts.length]);
   ────────────────────────────────────────────────────────── */

import { api } from "../api/client";

const _loaded = new Set<string>();
const _inflight = new Map<string, Promise<void>>();
const _objectUrls = new Map<string, string>();

/** Decode a base64 string into a Uint8Array.  Stays browser-only —
 *  atob + manual byte fill avoids the deprecated Buffer API. */
function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function loadOne(name: string): Promise<void> {
  if (_loaded.has(name)) return;
  const inflight = _inflight.get(name);
  if (inflight) return inflight;

  const stripped = name.replace(/\.(ttf|otf|ttc)$/i, "");

  const promise = (async () => {
    try {
      const resp = await api.getFontFileB64(name);
      if (!resp || !resp.b64) {
        // Backend returned 404 / decode failure.  Fail silently — the
        // CSS overlay will fall back to system fonts by family name.
        return;
      }
      const bytes = b64ToBytes(resp.b64);
      // Cast to BlobPart — TypeScript's lib.dom typings narrowed
      // BlobPart to ArrayBufferView<ArrayBuffer>, but Uint8Array's
      // generic buffer can be SharedArrayBuffer.  The runtime accepts
      // any typed-array view; the cast is the standard workaround.
      const blob = new Blob([bytes as BlobPart], { type: resp.mime || "font/ttf" });
      const url = URL.createObjectURL(blob);
      _objectUrls.set(name, url);

      const FF = (window as unknown as { FontFace?: typeof FontFace }).FontFace;
      if (FF && document.fonts?.add) {
        // Register BOTH "ArialNarrowItalic.ttf" and "ArialNarrowItalic"
        // so callers can use either spelling in font-family CSS.  The
        // backend stores filenames with extensions and the CSS overlay
        // strips extensions before lookup; register both to be safe.
        const faces = [new FF(name, `url(${url})`)];
        if (stripped !== name) faces.push(new FF(stripped, `url(${url})`));
        for (const face of faces) {
          const loaded = await face.load();
          document.fonts.add(loaded);
        }
      } else {
        // Older browsers — append a <style> with @font-face declarations.
        const css = `
@font-face { font-family: ${JSON.stringify(name)}; src: url(${JSON.stringify(url)}); font-display: swap; }
${stripped !== name ? `@font-face { font-family: ${JSON.stringify(stripped)}; src: url(${JSON.stringify(url)}); font-display: swap; }` : ""}
`;
        const styleEl = document.createElement("style");
        styleEl.setAttribute("data-font", name);
        styleEl.textContent = css;
        document.head.appendChild(styleEl);
      }
      _loaded.add(name);
    } catch (err) {
      console.warn(`[loadCustomFonts] failed to load ${name}:`, err);
    } finally {
      _inflight.delete(name);
    }
  })();

  _inflight.set(name, promise);
  return promise;
}

/** Bulk-register every font in `names`.  Returns immediately; loads
 *  happen in the background — failures are logged and skipped.
 *  Subsequent calls with the same names are no-ops. */
export function loadCustomFonts(names: string[]): void {
  for (const name of names) {
    if (!_loaded.has(name) && !_inflight.has(name)) {
      void loadOne(name);
    }
  }
}

/** Hint check for callers that want to know whether a particular
 *  font is ready (FontFace .load() resolves async).  Returns true
 *  once the bytes are downloaded AND registered with document.fonts. */
export function isFontLoaded(name: string): boolean {
  return _loaded.has(name);
}
