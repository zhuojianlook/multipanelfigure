/* ──────────────────────────────────────────────────────────
   API client for the FastAPI backend (api_server.py).
   All methods return parsed JSON.
   Uses Tauri IPC proxy (invoke) to bypass WebView restrictions
   on localhost requests. Falls back to browser fetch in dev.
   ────────────────────────────────────────────────────────── */

import type {
  FigureConfig,
  UploadResponse,
  ImagesListResponse,
  PreviewResponse,
  ProjectLoadResponse,
} from "./types";

const DEFAULT_BASE = "http://127.0.0.1:8765";

/** Try to use Tauri invoke for HTTP requests; fall back to browser fetch */
let _invoke: ((cmd: string, args: Record<string, unknown>) => Promise<unknown>) | null = null;
let _invokeReady = false;

async function getInvoke() {
  if (_invokeReady) return _invoke;
  try {
    const mod = await import("@tauri-apps/api/core");
    _invoke = mod.invoke;
  } catch {
    _invoke = null;
  }
  _invokeReady = true;
  return _invoke;
}

/** Make an API request — uses Tauri proxy if available, else browser fetch */
async function apiRequest(path: string, method: string = "GET", body?: string): Promise<string> {
  const invoke = await getInvoke();
  if (invoke) {
    // Use Rust-side proxy to bypass WebView restrictions
    return invoke("proxy_request", { method, path, body: body ?? null }) as Promise<string>;
  }
  // Fallback: browser fetch (works in dev)
  const res = await fetch(`${DEFAULT_BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body,
  });
  return res.text();
}

/** Make a request and parse JSON response */
async function apiJson<T>(path: string, method: string = "GET", body?: string): Promise<T> {
  const text = await apiRequest(path, method, body);
  const parsed = JSON.parse(text);
  // Check for error responses
  if (parsed.detail) {
    throw new Error(`API error: ${JSON.stringify(parsed.detail)}`);
  }
  return parsed as T;
}

/** Last health check error for diagnostics */
export let lastHealthError = "";

/** Check if the sidecar API is reachable */
export async function checkHealth(): Promise<boolean> {
  try {
    const text = await apiRequest("/api/health");
    const data = JSON.parse(text);
    if (data.status === "ok") {
      lastHealthError = "";
      return true;
    }
    lastHealthError = `Health response: ${text.substring(0, 200)}`;
    return false;
  } catch (e) {
    lastHealthError = e instanceof Error ? e.message : String(e);
    return false;
  }
}

class ApiClient {
  // ── Config ─────────────────────────────────────────────

  async getConfig(): Promise<FigureConfig> {
    return apiJson<FigureConfig>("/api/config");
  }

  async updateConfig(config: FigureConfig): Promise<FigureConfig> {
    return apiJson<FigureConfig>("/api/config", "PUT", JSON.stringify({ config }));
  }

  async patchGrid(rows: number, cols: number, spacing: number): Promise<FigureConfig> {
    return apiJson<FigureConfig>("/api/config/grid", "PATCH", JSON.stringify({ rows, cols, spacing }));
  }

  async patchPanel(r: number, c: number, panel: Record<string, unknown>): Promise<unknown> {
    return apiJson(`/api/config/panel/${r}/${c}`, "PATCH", JSON.stringify({ panel }));
  }

  async patchColumnLabels(labels: unknown[]): Promise<unknown> {
    return apiJson("/api/config/column-labels", "PATCH", JSON.stringify({ labels }));
  }

  async patchRowLabels(labels: unknown[]): Promise<unknown> {
    return apiJson("/api/config/row-labels", "PATCH", JSON.stringify({ labels }));
  }

  async patchColumnHeaders(headers: unknown[]): Promise<unknown> {
    return apiJson("/api/config/column-headers", "PATCH", JSON.stringify({ headers }));
  }

  async patchRowHeaders(headers: unknown[]): Promise<unknown> {
    return apiJson("/api/config/row-headers", "PATCH", JSON.stringify({ headers }));
  }

  async patchBackground(background: string): Promise<unknown> {
    return apiJson("/api/config/background", "PATCH", JSON.stringify({ background }));
  }

  // ── Image upload / management ──────────────────────────

  async uploadImages(files: File[]): Promise<UploadResponse> {
    // File uploads use base64 through Tauri IPC proxy
    const invoke = await getInvoke();
    if (invoke) {
      const filesData: { name: string; data: string }[] = [];
      for (const f of files) {
        const buf = await f.arrayBuffer();
        const bytes = new Uint8Array(buf);
        const CHUNK = 8192;
        const chunks: string[] = [];
        for (let i = 0; i < bytes.length; i += CHUNK) {
          chunks.push(String.fromCharCode(...bytes.subarray(i, i + CHUNK)));
        }
        filesData.push({ name: f.name, data: btoa(chunks.join("")) });
      }
      const text = await invoke("proxy_upload", {
        path: "/api/images/upload",
        files: filesData,
        fieldName: "files",
      }) as string;
      return JSON.parse(text) as UploadResponse;
    }
    // Fallback: browser fetch with FormData
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch(`${DEFAULT_BASE}/api/images/upload`, { method: "POST", body: form });
    return res.json() as Promise<UploadResponse>;
  }

  /** Upload images from file paths — one at a time to avoid IPC size limits */
  async uploadImagesFromPaths(filePaths: string[]): Promise<UploadResponse> {
    const invoke = await getInvoke();
    if (!invoke) throw new Error("Path-based upload requires Tauri runtime");

    const allNames: string[] = [];
    const allThumbnails: Record<string, string> = {};

    // Upload one file at a time to avoid IPC response size limits
    for (const filePath of filePaths) {
      const text = await invoke("upload_files_from_paths", {
        apiPath: "/api/images/upload",
        filePaths: [filePath],
        fieldName: "files",
      }) as string;
      const result = JSON.parse(text);
      // Validate response has expected fields
      if (result.detail) {
        throw new Error(`Server error for ${filePath.split(/[/\\]/).pop()}: ${result.detail}`);
      }
      if (result.names && Array.isArray(result.names)) {
        allNames.push(...result.names);
      }
      if (result.thumbnails && typeof result.thumbnails === "object") {
        Object.assign(allThumbnails, result.thumbnails);
      }
    }

    return { names: allNames, thumbnails: allThumbnails };
  }

  async deleteImage(name: string): Promise<void> {
    await apiRequest(`/api/images/${encodeURIComponent(name)}`, "DELETE");
  }

  async listImages(): Promise<ImagesListResponse> {
    return apiJson<ImagesListResponse>("/api/images");
  }

  async getImageThumbnail(name: string): Promise<{ thumbnail: string }> {
    return apiJson<{ thumbnail: string }>(`/api/images/${encodeURIComponent(name)}/thumb`);
  }

  async getImageInfo(name: string): Promise<{ width: number; height: number }> {
    return apiJson<{ width: number; height: number }>(`/api/images/${encodeURIComponent(name)}/info`);
  }

  // ── Panel Preview ──────────────────────────────────────

  async getPanelPreview(r: number, c: number): Promise<{ image: string; processed_width?: number; processed_height?: number }> {
    return apiJson<{ image: string }>(`/api/panel-preview/${r}/${c}`);
  }

  async patchPanelAndPreview(
    r: number, c: number, panel: Record<string, unknown>,
  ): Promise<{ panel: Record<string, unknown>; image: string; processed_width?: number; processed_height?: number }> {
    return apiJson(`/api/panel-patch-preview/${r}/${c}`, "POST", JSON.stringify({ panel }));
  }

  // ── Auto-Adjust ───────────────────────────────────────

  async autoAdjust(
    r: number, c: number, type: "levels" | "contrast" | "white_balance",
  ): Promise<{ adjustments: Record<string, number> }> {
    return apiJson(`/api/auto-adjust/${r}/${c}`, "POST", JSON.stringify({ type }));
  }

  // ── Panel rendered preview (with matplotlib overlays) ──

  async getPanelRenderedPreview(row: number, col: number): Promise<{ image: string }> {
    return apiJson(`/api/panel-rendered-preview/${row}/${col}`);
  }

  // ── Preview ────────────────────────────────────────────

  async getPreview(): Promise<PreviewResponse> {
    return apiJson<PreviewResponse>("/api/preview", "POST");
  }

  // ── Save / Load project ────────────────────────────────

  async saveProject(path: string): Promise<{ ok: boolean }> {
    return apiJson("/api/project/save", "POST", JSON.stringify({ path }));
  }

  async loadProject(path: string): Promise<ProjectLoadResponse> {
    return apiJson<ProjectLoadResponse>("/api/project/load", "POST", JSON.stringify({ path }));
  }

  // ── Save final figure ──────────────────────────────────

  async saveFigure(
    path: string,
    format: string = "TIFF",
    background: string = "White",
    dpi: number = 300,
  ): Promise<{ ok: boolean; path: string }> {
    return apiJson("/api/figure/save", "POST", JSON.stringify({ path, format, background, dpi }));
  }

  // ── Fonts ──────────────────────────────────────────────

  async listFonts(): Promise<{ fonts: Record<string, string> }> {
    return apiJson<{ fonts: Record<string, string> }>("/api/fonts");
  }

  async uploadFonts(files: File[]): Promise<{ names: string[]; total: number }> {
    const invoke = await getInvoke();
    if (invoke) {
      const filesData: { name: string; data: string }[] = [];
      for (const f of files) {
        const buf = await f.arrayBuffer();
        const bytes = new Uint8Array(buf);
        const CHUNK = 8192;
        const chunks: string[] = [];
        for (let i = 0; i < bytes.length; i += CHUNK) {
          chunks.push(String.fromCharCode(...bytes.subarray(i, i + CHUNK)));
        }
        filesData.push({ name: f.name, data: btoa(chunks.join("")) });
      }
      const text = await invoke("proxy_upload", {
        path: "/api/fonts/upload",
        files: filesData,
        fieldName: "files",
      }) as string;
      return JSON.parse(text) as { names: string[]; total: number };
    }
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch(`${DEFAULT_BASE}/api/fonts/upload`, { method: "POST", body: form });
    return res.json() as Promise<{ names: string[]; total: number }>;
  }

  // ── Resolution presets ─────────────────────────────────

  async getResolutions(): Promise<Record<string, number>> {
    return apiJson<Record<string, number>>("/api/resolutions");
  }

  async updateResolutions(entries: Record<string, number>): Promise<void> {
    await apiJson("/api/resolutions", "PUT", JSON.stringify({ entries }));
  }

  // ── Measurements ────────────────────────────────────────

  async getMeasurements(): Promise<{ measurements: Array<{ panel: string; name: string; type: string; value: string }> }> {
    return apiJson("/api/measurements");
  }

  // ── Image Thumbnail ──────────────────────────────────────

  async getImageThumb(name: string): Promise<{ thumbnail: string }> {
    return apiJson(`/api/images/${encodeURIComponent(name)}/thumb`);
  }

  // ── Video ────────────────────────────────────────────────

  async getVideoInfo(name: string): Promise<{ frame_count: number; fps: number; width: number; height: number; duration_sec: number }> {
    return apiJson(`/api/video/${encodeURIComponent(name)}/info`);
  }

  async getVideoFrame(name: string, frameNum: number): Promise<{ frame: number; width: number; height: number; thumbnail: string }> {
    return apiJson(`/api/video/${encodeURIComponent(name)}/frame/${frameNum}`);
  }

  async listVideos(): Promise<{ videos: string[] }> {
    return apiJson("/api/video/list");
  }

  // ── Magic Wand Selection ────────────────────────────────

  async magicWandSelect(row: number, col: number, xPct: number, yPct: number, tolerance: number, overrides?: { rotation?: number; crop?: number[]; crop_image?: boolean }): Promise<{ points: number[][]; pixel_count: number; smooth?: boolean }> {
    return apiJson(`/api/magic-wand/${row}/${col}`, "POST", JSON.stringify({ x_pct: xPct, y_pct: yPct, tolerance, ...overrides }));
  }
}

export const api = new ApiClient();
export default ApiClient;
