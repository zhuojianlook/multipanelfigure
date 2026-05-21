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

  async saveProject(path: string, analysis?: import("./types").AnalysisPayload | null): Promise<{ ok: boolean }> {
    return apiJson(
      "/api/project/save",
      "POST",
      JSON.stringify({ path, analysis: analysis ?? null }),
    );
  }

  async loadProject(path: string): Promise<ProjectLoadResponse> {
    return apiJson<ProjectLoadResponse>("/api/project/load", "POST", JSON.stringify({ path }));
  }

  // ── In-session document snapshots (for seamless tab switching) ──
  // snapshotProject serializes the CURRENT backend builder state to a
  // base64 .mpf blob without writing to disk; restoreProject loads such a
  // blob back into the global state. Used by projectNav to preserve a
  // tab's unsaved edits when the user switches to another document tab.

  async snapshotProject(): Promise<{ blob: string }> {
    return apiJson<{ blob: string }>("/api/project/snapshot", "GET");
  }

  async restoreProject(
    blob: string,
  ): Promise<Omit<ProjectLoadResponse, "analysis">> {
    return apiJson<Omit<ProjectLoadResponse, "analysis">>(
      "/api/project/restore",
      "POST",
      JSON.stringify({ blob }),
    );
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

  /** Fetch a single font's bytes as base64.  Used by the @font-face
   *  loader so the CSS overlay can render labels with the same fonts
   *  the backend uses.  Returns null if the font isn't installed on
   *  this host (404). */
  async getFontFileB64(name: string): Promise<{ name: string; b64: string; mime: string } | null> {
    try {
      return await apiJson<{ name: string; b64: string; mime: string }>(
        `/api/fonts/file-b64/${encodeURIComponent(name)}`,
      );
    } catch {
      return null;
    }
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

  async restoreDefaultResolutions(): Promise<Record<string, number>> {
    return apiJson<Record<string, number>>("/api/resolutions/restore-defaults", "POST");
  }

  // ── Collage stash ──────────────────────────────────────
  async deleteCollageStash(itemId: string): Promise<void> {
    await apiJson(`/api/collage/stash/${encodeURIComponent(itemId)}`, "DELETE");
  }

  /** Stateless render of a saved .mpf with optional header-pt override.
   *  Doesn't touch the live builder state — the user can stay in
   *  collage mode while we re-render figures with normalised header
   *  sizes.
   *
   *  Pass `itemW` (collage-canvas pixel width of the item) to engage
   *  the two-pass iterative renderer. The backend measures the
   *  figure's post-override naturalW and compensates for it, so
   *  figures with row headers / labels (whose width grows with
   *  header pt) still come out at the right visual size. The legacy
   *  `scale` parameter is kept for the case where itemW isn't
   *  available — it's a one-pass approximation that's accurate only
   *  for figures without row labels. */
  async renderCollageFigure(
    projectPath: string,
    headerPt: number | null,
    scale: number,
    itemW?: number,
    elementIds?: string[] | null,
    elementOverrides?: Record<string, unknown> | null,
  ): Promise<{ image: string; width: number; height: number }> {
    return apiJson("/api/collage/render-figure", "POST", JSON.stringify({
      project_path: projectPath,
      header_pt: headerPt,
      scale,
      item_w: itemW ?? null,
      element_ids: elementIds && elementIds.length ? elementIds : null,
      element_overrides: elementOverrides && Object.keys(elementOverrides).length ? elementOverrides : null,
    }));
  }

  /** List the editable text elements of a saved .mpf (column/row headers,
   *  axis labels, panel labels, scale bars) so the collage UI can offer
   *  per-element font synchronization + customization. Includes each
   *  element's geometry (for on-figure hotspots) and current style. */
  async getFigureElements(
    projectPath: string,
  ): Promise<{ elements: Array<import("../store/collageStore").CollageFigElement & {
    font_name?: string | null; color?: string | null;
    font_style?: string[]; styled_segments?: import("./types").StyledSegment[];
  }> }> {
    return apiJson("/api/collage/figure-elements", "POST", JSON.stringify({
      project_path: projectPath,
    }));
  }

  /** Decompose an .mpf into a header-LESS body raster + header geometry
   *  for live overlay rendering. Headers come back in figure fractions
   *  (0..1, y from bottom). Lets the collage place + restyle headers
   *  instantly without re-rendering the figure through matplotlib. */
  async decomposeCollageFigure(
    projectPath: string,
  ): Promise<{
    image: string;
    width: number;
    height: number;
    headers: import("../store/collageStore").CollageHeader[];
  }> {
    return apiJson("/api/collage/decompose", "POST", JSON.stringify({
      project_path: projectPath,
    }));
  }

  // ── Measurements ────────────────────────────────────────

  async getMeasurements(): Promise<{ measurements: Array<{ panel: string; name: string; type: string; value: string; numeric?: number; unit?: string }> }> {
    return apiJson("/api/measurements");
  }

  // ── R Analysis ──────────────────────────────────────────

  async checkR(customPath?: string): Promise<{ installed: boolean; version: string }> {
    const params = customPath ? `?rscript_path=${encodeURIComponent(customPath)}` : "";
    return apiJson(`/api/analysis/check-r${params}`);
  }

  async runR(
    code: string,
    dataCsv: string,
    rscriptPath?: string,
    baseFontSize?: number | null,
  ): Promise<{
    success: boolean;
    stdout: string;
    stderr: string;
    plots: string[];
    /** CSV-encoded data.frames the R script wrote out via mpfig_data(df, name).
     *  Empty array when the script didn't call mpfig_data. */
    tables: { name: string; csv: string }[];
  }> {
    return apiJson("/api/analysis/run-r", "POST", JSON.stringify({
      code, data_csv: dataCsv, rscript_path: rscriptPath || null,
      base_font_size: baseFontSize && baseFontSize > 0 ? Math.round(baseFontSize) : null,
    }));
  }

  /** Run a raw R command (no data/plot boilerplate) — for the Analysis
   *  dialog's mini R console, mainly install.packages("..."). */
  async runRConsole(command: string, rscriptPath?: string): Promise<{ success: boolean; stdout: string; stderr: string }> {
    return apiJson("/api/analysis/run-console", "POST", JSON.stringify({ command, rscript_path: rscriptPath || null }));
  }

  /** List zoom insets across the grid that have `include_in_analysis`
   *  enabled. Used by the Analysis tab's Python / MATLAB pipeline
   *  source list. Includes a base64 thumbnail per inset so the
   *  Analysis dialog can show the user exactly which pixels its
   *  pipelines will operate on. */
  async listInsetAnalysisSources(): Promise<{
    sources: Array<{
      key: string;
      row: number;
      col: number;
      inset_index: number;
      inset_type: string;
      x: number; y: number; width: number; height: number;
      zoom_factor: number;
      label: string;
      natural_width: number;
      natural_height: number;
      /** Base64 PNG (no data: prefix). Up to 256 px on the long edge. */
      thumbnail: string;
    }>;
  }> {
    return apiJson("/api/analysis/inset-sources");
  }

  /** Like listInsetAnalysisSources but for a SPECIFIC .mpf (not the active
   *  figure) — lets the Analysis tab show a sources drawer per loaded MPF. */
  async listInsetAnalysisSourcesFor(projectPath: string): Promise<{
    sources: Array<Record<string, unknown>>;
  }> {
    return apiJson("/api/analysis/inset-sources-for", "POST", JSON.stringify({ project_path: projectPath }));
  }

  /** Detect whether a MATLAB-compatible interpreter (Octave or MATLAB)
   *  is available on the host. The Analysis dialog hides the Run MATLAB
   *  button when this returns `installed: false`. */
  async checkMatlab(): Promise<{ installed: boolean; kind: string; path: string }> {
    return apiJson("/api/analysis/check-matlab");
  }

  /** Detect whether ImageJ / Fiji is installed and reachable. The
   *  Analysis dialog hides the ImageJ button when this returns
   *  `installed: false`. The endpoint is optional on older sidecars
   *  so this is wrapped to soft-fail. */
  async checkImageJ(): Promise<{ installed: boolean; kind: string; path: string }> {
    return apiJson("/api/analysis/check-imagej");
  }

  /** Run an ImageJ / Fiji headless macro against the requested
   *  inset images. Output: same shape as runMatlab — plots, tables,
   *  and per-image outputs.  Requires a working Fiji install on PATH;
   *  the backend returns `success: false` with a clear stderr
   *  otherwise. */
  async runImageJ(
    code: string,
    sources: Array<{ key: string; row: number; col: number; inset_index: number; label?: string }>,
    timeoutSec: number = 120,
  ): Promise<{
    success: boolean;
    kind?: string;
    stdout: string;
    stderr: string;
    plots: string[];
    tables: { name: string; csv: string }[];
    images: { name: string; image: string }[];
  }> {
    return apiJson("/api/analysis/run-imagej", "POST", JSON.stringify({ code, sources, timeout_sec: timeoutSec }));
  }

  /** Run a MATLAB / Octave pipeline against the requested zoom-inset
   *  regions. The script can `load("inputs.mat")` to get
   *  `inputs.<safe_key>.image` (uint8 H×W×3 matrix), and call
   *  `mpfig_plot(name)`, `mpfig_data(table, name)`, `mpfig_image(arr, name)`
   *  to push outputs back into the Analysis timeline. */
  async runMatlab(
    code: string,
    sources: Array<{ key: string; row: number; col: number; inset_index: number; label?: string }>,
    timeoutSec: number = 60,
  ): Promise<{
    success: boolean;
    kind?: string;
    stdout: string;
    stderr: string;
    plots: string[];
    tables: { name: string; csv: string }[];
    images: { name: string; image: string }[];
  }> {
    return apiJson("/api/analysis/run-matlab", "POST", JSON.stringify({ code, sources, timeout_sec: timeoutSec }));
  }

  /** Run a Python pipeline against the requested zoom-inset regions.
   *  The harness exposes `inputs[key]` → { image: np.ndarray, ... } for
   *  each source, and `mpfig_plot()`, `mpfig_data()`, `mpfig_image()`
   *  helpers to push outputs back to the Analysis tab. */
  async runPython(
    code: string,
    sources: Array<{ key: string; row: number; col: number; inset_index: number; label?: string }>,
    timeoutSec: number = 30,
  ): Promise<{
    success: boolean;
    stdout: string;
    stderr: string;
    plots: string[];
    tables: { name: string; csv: string }[];
    images: { name: string; image: string }[];
  }> {
    return apiJson("/api/analysis/run-python", "POST", JSON.stringify({ code, sources, timeout_sec: timeoutSec }));
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

  async getZStackFrame(name: string, frameNum: number, row?: number, col?: number): Promise<{ frame: number; width: number; height: number; thumbnail: string; image_name?: string }> {
    const params = (row != null && col != null) ? `?row=${row}&col=${col}` : "";
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/frame/${frameNum}${params}`);
  }

  async listZStacks(): Promise<{ zstacks: string[] }> {
    return apiJson("/api/zstack/list");
  }

  // ── Multichannel TIFF (channel groups) ───────────────────
  //
  // Returns `{ is_multichannel: false }` for ordinary single-channel
  // images, so callers can probe any image name and conditionally
  // render the channel-tint UI.
  async getChannelInfo(name: string): Promise<{
    is_multichannel: boolean;
    axes?: string;
    num_channels?: number;
    num_z?: number;
    current_z?: number;
    tints?: string[];
    enabled?: boolean[];
    black_levels?: number[];
    white_levels?: number[];
    names?: string[];
  }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/channels/info`);
  }

  // Update per-channel tints / enabled / levels / current_z. Any subset
  // of fields may be provided; only present fields are applied. Returns
  // the new composite thumbnail + the post-update state for state sync.
  async updateChannels(name: string, body: {
    tints?: string[]; enabled?: boolean[];
    black_levels?: number[]; white_levels?: number[];
    current_z?: number;
    names?: string[];
    row?: number; col?: number;
  }): Promise<{
    thumbnail: string; width: number; height: number;
    current_z: number; tints: string[]; enabled: boolean[];
    black_levels: number[]; white_levels: number[];
    names?: string[];
  }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/channels`, "PATCH", JSON.stringify(body));
  }

  // Ask the server which z-stack alignment algorithms are usable.
  // SIFT needs ImageJ/Fiji installed; phase_correlation is always available.
  async getAlignAvailability(): Promise<{
    sift: { available: boolean; kind: string; path: string };
    phase_correlation: { available: boolean; kind: string };
  }> {
    return apiJson("/api/zstack/align/availability");
  }

  // Run alignment on a z-stack. Returns a new in-memory name (e.g.
  // "Composite.tif::aligned") that the caller can swap into the panel.
  async alignZStack(name: string, opts: {
    method: "sift" | "phase_correlation";
    startFrame?: number; endFrame?: number;
    alignChannel?: number;
    // SIFT params
    siftInitialGaussianBlur?: number;
    siftStepsPerScaleOctave?: number;
    siftMinimumImageSize?: number;
    siftMaximumImageSize?: number;
    siftFeatureDescriptorSize?: number;
    siftFeatureDescriptorOrientationBins?: number;
    siftClosestNextClosestRatio?: number;
    siftMaximalAlignmentError?: number;
    siftInlierRatio?: number;
    siftExpectedTransformation?: "Translation" | "Rigid" | "Similarity" | "Affine";
    siftInterpolate?: boolean;
    // Phase correlation
    pcWindow?: "hann" | "rect";
    pcMaxShiftFrac?: number;
    // Performance knobs
    alignMaxDim?: number;   // cap the per-frame size before alignment (default 1024 px)
    timeoutSec?: number;    // subprocess timeout for SIFT (default 1800)
    // Reference source — which image(s) to feed into SIFT/phase corr.
    // "max" (default) pools features across all enabled channels.
    alignmentSource?: "max" | "mean" | "sum" | "channel";
    // CLAHE pre-processing — local contrast normalization.
    useClahe?: boolean;
    claheClipLimit?: number;
    claheTileGrid?: number;
  }): Promise<{
    /** Job id — pass to `pollAlignStatus` to track progress. */
    job_id: string;
    status: string;
  }> {
    return apiJson(`/api/zstack/${encodeURIComponent(name)}/align`, "POST", JSON.stringify({
      method: opts.method,
      start_frame: opts.startFrame ?? 0,
      end_frame: opts.endFrame ?? -1,
      align_channel: opts.alignChannel ?? -1,
      sift_initial_gaussian_blur: opts.siftInitialGaussianBlur ?? 1.6,
      sift_steps_per_scale_octave: opts.siftStepsPerScaleOctave ?? 3,
      sift_minimum_image_size: opts.siftMinimumImageSize ?? 64,
      sift_maximum_image_size: opts.siftMaximumImageSize ?? 1024,
      sift_feature_descriptor_size: opts.siftFeatureDescriptorSize ?? 4,
      sift_feature_descriptor_orientation_bins: opts.siftFeatureDescriptorOrientationBins ?? 8,
      sift_closest_next_closest_ratio: opts.siftClosestNextClosestRatio ?? 0.92,
      sift_maximal_alignment_error: opts.siftMaximalAlignmentError ?? 25,
      sift_inlier_ratio: opts.siftInlierRatio ?? 0.05,
      sift_expected_transformation: opts.siftExpectedTransformation ?? "Rigid",
      sift_interpolate: opts.siftInterpolate ?? true,
      pc_window: opts.pcWindow ?? "hann",
      pc_max_shift_frac: opts.pcMaxShiftFrac ?? 0.25,
      align_max_dim: opts.alignMaxDim ?? 1024,
      timeout_sec: opts.timeoutSec ?? 1800,
      alignment_source: opts.alignmentSource ?? "max",
      use_clahe: opts.useClahe ?? false,
      clahe_clip_limit: opts.claheClipLimit ?? 2.0,
      clahe_tile_grid: opts.claheTileGrid ?? 8,
    }));
  }

  /** Poll the status of an alignment job started via `alignZStack`. */
  async pollAlignStatus(jobId: string): Promise<{
    status: "starting" | "running" | "done" | "error";
    progress: number;   // 0..1
    stage: string;      // human-readable
    result: null | {
      aligned_name: string;
      method: string;
      n_frames: number;
      n_channels: number;
      shifts: number[][];
      log: Record<string, string>;
    };
    error: string | null;
  }> {
    return apiJson(`/api/zstack/align/status/${encodeURIComponent(jobId)}`);
  }

  /** Convenience: start an alignment job and poll until done/error.
   *  `onProgress` is called with (progress 0..1, stage label) on each
   *  poll, throttled by `pollIntervalMs`. */
  async alignAndWait(
    name: string,
    opts: Parameters<ApiClient["alignZStack"]>[1],
    onProgress?: (progress: number, stage: string) => void,
    pollIntervalMs: number = 500,
  ): Promise<NonNullable<Awaited<ReturnType<ApiClient["pollAlignStatus"]>>["result"]>> {
    const start = await this.alignZStack(name, opts);
    for (;;) {
      const status = await this.pollAlignStatus(start.job_id);
      onProgress?.(status.progress, status.stage);
      if (status.status === "done" && status.result) return status.result;
      if (status.status === "error") {
        throw new Error(status.error || "Alignment failed");
      }
      await new Promise(r => setTimeout(r, pollIntervalMs));
    }
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
  } = {}): Promise<{
    data: string; width: number; height: number; depth: number;
    /** True when the server composited the source's per-channel tints
     *  into an RGB24 NIfTI volume (multichannel TIFFs only). NiiVue
     *  ignores the colormap dropdown for RGB volumes. */
    rgb?: boolean;
  }> {
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
