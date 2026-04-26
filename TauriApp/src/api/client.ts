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
  // Detect actual Tauri runtime (window.__TAURI_INTERNALS__). In a plain
  // browser dev preview the @tauri-apps/api/core module still resolves via
  // Vite, but invoke() throws at runtime because there is no IPC bridge.
  // Fall through to browser fetch in that case.
  const inTauri =
    typeof window !== "undefined" &&
    (("__TAURI_INTERNALS__" in window) || ("__TAURI__" in window) || ("__TAURI_IPC__" in window));
  if (!inTauri) {
    _invoke = null;
    _invokeReady = true;
    return _invoke;
  }
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

  // ── Render figure as video (animates panels with play_range) ──────────

  async renderVideoFfmpegAvailable(): Promise<{ available: boolean }> {
    return apiJson("/api/figure/render-video/ffmpeg-available");
  }

  async renderVideoStart(
    path: string,
    format: "mp4" | "avi" = "mp4",
    fps: number = 30,
    background: string = "White",
    dpi: number = 150,
    audio_panel_image_name: string | null = null,
  ): Promise<{ job_id: string; total_frames: number }> {
    return apiJson(
      "/api/figure/render-video",
      "POST",
      JSON.stringify({ path, format, fps, background, dpi, audio_panel_image_name }),
    );
  }

  async renderVideoProgress(
    jobId: string,
  ): Promise<{ status: "running" | "done" | "error" | "cancelled"; current: number; total: number; output_path?: string; error?: string }> {
    return apiJson(`/api/figure/render-video/${encodeURIComponent(jobId)}/progress`);
  }

  async renderVideoCancel(jobId: string): Promise<{ ok: boolean }> {
    return apiJson(`/api/figure/render-video/${encodeURIComponent(jobId)}/cancel`, "POST");
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

  // ── R Analysis ──────────────────────────────────────────

  async checkR(customPath?: string): Promise<{ installed: boolean; version: string }> {
    const params = customPath ? `?rscript_path=${encodeURIComponent(customPath)}` : "";
    return apiJson(`/api/analysis/check-r${params}`);
  }

  async runR(code: string, dataCsv: string, rscriptPath?: string): Promise<{ success: boolean; stdout: string; stderr: string; plots: string[] }> {
    return apiJson("/api/analysis/run-r", "POST", JSON.stringify({ code, data_csv: dataCsv, rscript_path: rscriptPath || null }));
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

  // ── Z-Stack TIFF ─────────────────────────────────────────

  async getZStackInfo(name: string): Promise<{ frame_count: number; width: number; height: number }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/info`);
  }

  async getZStackFrame(name: string, frameNum: number, row?: number, col?: number): Promise<{ frame: number; width: number; height: number; thumbnail: string }> {
    const params = (row != null && col != null) ? `?row=${row}&col=${col}` : "";
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/frame/${frameNum}${params}`);
  }

  async listZStacks(): Promise<{ zstacks: string[] }> {
    return apiJson("/api/zstack/list");
  }

  async projectZStack(name: string, startFrame: number, endFrame: number, method: string): Promise<{ method: string; start_frame: number; end_frame: number; width: number; height: number; thumbnail: string }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/project`, "POST", JSON.stringify({ start_frame: startFrame, end_frame: endFrame, method }));
  }

  async getVolumeData(name: string, startFrame: number = 0, endFrame: number = -1, maxDim: number = 128): Promise<{ data: string; width: number; height: number; depth: number }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/volume`, "POST", JSON.stringify({ start_frame: startFrame, end_frame: endFrame, max_dim: maxDim }));
  }

  async saveCanvasAsPanel(imageName: string, row: number, col: number, dataB64: string): Promise<{ ok: boolean; image_name: string }> {
    return apiJson(`/api/save-canvas-as-panel`, "POST", JSON.stringify({
      image_name: imageName, row, col, data_b64: dataB64,
    }));
  }

  async getZStackNifti(name: string, opts: {
    startFrame?: number; endFrame?: number; maxDim?: number; zSpacing?: number;
  } = {}): Promise<{ data: string; width: number; height: number; depth: number }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/nifti`, "POST", JSON.stringify({
      start_frame: opts.startFrame ?? 0, end_frame: opts.endFrame ?? -1,
      max_dim: opts.maxDim ?? 256,
      z_spacing: opts.zSpacing ?? 1.0,
    }));
  }

  async getZStackMips(name: string, opts: {
    startFrame?: number; endFrame?: number; colormap?: string;
    rotationFrames?: number; includeRotation?: boolean; maxDim?: number;
  } = {}): Promise<{ mip_xy: string; mip_xz: string; mip_yz: string; rotation_frames: string[] }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/mips`, "POST", JSON.stringify({
      start_frame: opts.startFrame ?? 0, end_frame: opts.endFrame ?? -1,
      colormap: opts.colormap ?? "gray",
      rotation_frames: opts.rotationFrames ?? 36,
      include_rotation: opts.includeRotation ?? false,
      max_dim: opts.maxDim ?? 128,
    }));
  }

  async renderVolume(name: string, opts: {
    startFrame?: number; endFrame?: number; elev?: number; azim?: number;
    threshold?: number; zSpacing?: number; colormap?: string;
    width?: number; height?: number; method?: string;
    showAxes?: boolean; zoom?: number; fast?: boolean;
  } = {}): Promise<{ image: string; width: number; height: number }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/volume-render`, "POST", JSON.stringify({
      start_frame: opts.startFrame ?? 0, end_frame: opts.endFrame ?? -1,
      elev: opts.elev ?? 30, azim: opts.azim ?? -60,
      threshold: opts.threshold ?? 0.3, z_spacing: opts.zSpacing ?? 1.0,
      colormap: opts.colormap ?? "gray", width: opts.width ?? 800, height: opts.height ?? 600,
      method: opts.method ?? "surface",
      show_axes: opts.showAxes ?? true, zoom: opts.zoom ?? 1.0, fast: opts.fast ?? false,
    }));
  }

  async saveVolumeRenderAsImage(name: string, opts: {
    startFrame?: number; endFrame?: number; elev?: number; azim?: number;
    threshold?: number; zSpacing?: number; colormap?: string;
    width?: number; height?: number; method?: string;
    showAxes?: boolean; zoom?: number;
    format?: string; quality?: number; filePath: string;
  }): Promise<{ ok: boolean; path: string }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/volume-save`, "POST", JSON.stringify({
      start_frame: opts.startFrame ?? 0, end_frame: opts.endFrame ?? -1,
      elev: opts.elev ?? 30, azim: opts.azim ?? -60,
      threshold: opts.threshold ?? 0.3, z_spacing: opts.zSpacing ?? 1.0,
      colormap: opts.colormap ?? "gray", width: opts.width ?? 1600, height: opts.height ?? 1200,
      method: opts.method ?? "surface",
      show_axes: opts.showAxes ?? true, zoom: opts.zoom ?? 1.0,
      format: opts.format ?? "PNG", quality: opts.quality ?? 95, path: opts.filePath,
    }));
  }

  async useVolumeAsPanel(name: string, row: number, col: number, opts: {
    startFrame?: number; endFrame?: number; elev?: number; azim?: number;
    threshold?: number; zSpacing?: number; colormap?: string;
    width?: number; height?: number; method?: string; showAxes?: boolean; zoom?: number;
  } = {}): Promise<{ ok: boolean; image_name: string }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/volume-as-panel`, "POST", JSON.stringify({
      start_frame: opts.startFrame ?? 0, end_frame: opts.endFrame ?? -1,
      elev: opts.elev ?? 30, azim: opts.azim ?? -60,
      threshold: opts.threshold ?? 0.3, z_spacing: opts.zSpacing ?? 1.0,
      colormap: opts.colormap ?? "gray", width: opts.width ?? 1600, height: opts.height ?? 1200,
      method: opts.method ?? "surface",
      show_axes: opts.showAxes ?? true, zoom: opts.zoom ?? 1.0,
      row, col,
    }));
  }

  // ── Magic Wand Selection ────────────────────────────────

  async magicWandSelect(row: number, col: number, xPct: number, yPct: number, tolerance: number, overrides?: { rotation?: number; crop?: number[]; crop_image?: boolean }): Promise<{ points: number[][]; pixel_count: number; smooth?: boolean }> {
    return apiJson(`/api/magic-wand/${row}/${col}`, "POST", JSON.stringify({ x_pct: xPct, y_pct: yPct, tolerance, ...overrides }));
  }
}

export const api = new ApiClient();
export default ApiClient;
