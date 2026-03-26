/* ──────────────────────────────────────────────────────────
   API client for the FastAPI backend (api_server.py).
   All methods return parsed JSON.
   Uses Tauri HTTP plugin to bypass WebView restrictions.
   ────────────────────────────────────────────────────────── */

import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import type {
  FigureConfig,
  UploadResponse,
  ImagesListResponse,
  PreviewResponse,
  ProjectLoadResponse,
} from "./types";

const DEFAULT_BASE = "http://127.0.0.1:8765";

// Use Tauri fetch if available, fallback to browser fetch
const httpFetch: typeof globalThis.fetch = typeof tauriFetch === "function"
  ? (tauriFetch as unknown as typeof globalThis.fetch)
  : globalThis.fetch;

class ApiClient {
  private base: string;

  constructor(baseUrl?: string) {
    this.base = (baseUrl ?? DEFAULT_BASE).replace(/\/+$/, "");
  }

  // ── helpers ──────────────────────────────────────────────

  private url(path: string): string {
    return `${this.base}${path}`;
  }

  private async json<T>(input: string, init?: RequestInit): Promise<T> {
    const res = await httpFetch(input, init);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`API ${res.status}: ${body}`);
    }
    return res.json() as Promise<T>;
  }

  // ── Config ─────────────────────────────────────────────

  async getConfig(): Promise<FigureConfig> {
    return this.json<FigureConfig>(this.url("/api/config"));
  }

  async updateConfig(config: FigureConfig): Promise<FigureConfig> {
    return this.json<FigureConfig>(this.url("/api/config"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config }),
    });
  }

  async patchGrid(rows: number, cols: number, spacing: number): Promise<FigureConfig> {
    return this.json<FigureConfig>(this.url("/api/config/grid"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows, cols, spacing }),
    });
  }

  async patchPanel(r: number, c: number, panel: Record<string, unknown>): Promise<unknown> {
    return this.json(this.url(`/api/config/panel/${r}/${c}`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ panel }),
    });
  }

  async patchColumnLabels(labels: unknown[]): Promise<unknown> {
    return this.json(this.url("/api/config/column-labels"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    });
  }

  async patchRowLabels(labels: unknown[]): Promise<unknown> {
    return this.json(this.url("/api/config/row-labels"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    });
  }

  async patchColumnHeaders(headers: unknown[]): Promise<unknown> {
    return this.json(this.url("/api/config/column-headers"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headers }),
    });
  }

  async patchRowHeaders(headers: unknown[]): Promise<unknown> {
    return this.json(this.url("/api/config/row-headers"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headers }),
    });
  }

  async patchBackground(background: string): Promise<unknown> {
    return this.json(this.url("/api/config/background"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ background }),
    });
  }

  // ── Image upload / management ──────────────────────────

  async uploadImages(files: File[]): Promise<UploadResponse> {
    const form = new FormData();
    for (const f of files) {
      form.append("files", f);
    }
    return this.json<UploadResponse>(this.url("/api/images/upload"), {
      method: "POST",
      body: form,
    });
  }

  async deleteImage(name: string): Promise<void> {
    await httpFetch(this.url(`/api/images/${encodeURIComponent(name)}`), {
      method: "DELETE",
    });
  }

  async listImages(): Promise<ImagesListResponse> {
    return this.json<ImagesListResponse>(this.url("/api/images"));
  }

  async getImageThumbnail(name: string): Promise<{ thumbnail: string }> {
    return this.json<{ thumbnail: string }>(
      this.url(`/api/images/${encodeURIComponent(name)}/thumb`),
    );
  }

  async getImageInfo(name: string): Promise<{ width: number; height: number }> {
    return this.json<{ width: number; height: number }>(
      this.url(`/api/images/${encodeURIComponent(name)}/info`),
    );
  }

  // ── Panel Preview ──────────────────────────────────────

  async getPanelPreview(r: number, c: number): Promise<{ image: string; processed_width?: number; processed_height?: number }> {
    return this.json<{ image: string }>(this.url(`/api/panel-preview/${r}/${c}`));
  }

  /** Atomically patch a panel AND get its processed preview in one request.
   *  Eliminates race conditions from separate PATCH + GET calls. */
  async patchPanelAndPreview(
    r: number, c: number, panel: Record<string, unknown>,
  ): Promise<{ panel: Record<string, unknown>; image: string; processed_width?: number; processed_height?: number }> {
    return this.json(this.url(`/api/panel-patch-preview/${r}/${c}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ panel }),
    });
  }

  // ── Auto-Adjust ───────────────────────────────────────

  /** Compute optimal adjustments from the original image. */
  async autoAdjust(
    r: number, c: number, type: "levels" | "contrast" | "white_balance",
  ): Promise<{ adjustments: Record<string, number> }> {
    return this.json(this.url(`/api/auto-adjust/${r}/${c}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });
  }

  // ── Panel rendered preview (with matplotlib overlays) ──

  async getPanelRenderedPreview(row: number, col: number): Promise<{ image: string }> {
    return this.json(this.url(`/api/panel-rendered-preview/${row}/${col}`));
  }

  // ── Preview ────────────────────────────────────────────

  async getPreview(): Promise<PreviewResponse> {
    return this.json<PreviewResponse>(this.url("/api/preview"), {
      method: "POST",
    });
  }

  // ── Save / Load project ────────────────────────────────

  async saveProject(path: string): Promise<{ ok: boolean }> {
    return this.json(this.url("/api/project/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  }

  async loadProject(path: string): Promise<ProjectLoadResponse> {
    return this.json<ProjectLoadResponse>(this.url("/api/project/load"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  }

  // ── Save final figure ──────────────────────────────────

  async saveFigure(
    path: string,
    format: string = "TIFF",
    background: string = "White",
    dpi: number = 300,
  ): Promise<{ ok: boolean; path: string }> {
    return this.json(this.url("/api/figure/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, format, background, dpi }),
    });
  }

  // ── Fonts ──────────────────────────────────────────────

  async listFonts(): Promise<{ fonts: Record<string, string> }> {
    return this.json<{ fonts: Record<string, string> }>(this.url("/api/fonts"));
  }

  async uploadFonts(files: File[]): Promise<{ names: string[]; total: number }> {
    const form = new FormData();
    for (const f of files) {
      form.append("files", f);
    }
    return this.json<{ names: string[]; total: number }>(this.url("/api/fonts/upload"), {
      method: "POST",
      body: form,
    });
  }

  // ── Resolution presets ─────────────────────────────────

  async getResolutions(): Promise<Record<string, number>> {
    return this.json<Record<string, number>>(this.url("/api/resolutions"));
  }

  async updateResolutions(entries: Record<string, number>): Promise<void> {
    await this.json(this.url("/api/resolutions"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries }),
    });
  }

  // ── Measurements ────────────────────────────────────────

  async getMeasurements(): Promise<{ measurements: Array<{ panel: string; name: string; type: string; value: string }> }> {
    return this.json(this.url("/api/measurements"));
  }

  // ── Image Thumbnail ──────────────────────────────────────

  async getImageThumb(name: string): Promise<{ thumbnail: string }> {
    return this.json(this.url(`/api/images/${encodeURIComponent(name)}/thumb`));
  }

  // ── Video ────────────────────────────────────────────────

  async getVideoInfo(name: string): Promise<{ frame_count: number; fps: number; width: number; height: number; duration_sec: number }> {
    return this.json(this.url(`/api/video/${encodeURIComponent(name)}/info`));
  }

  async getVideoFrame(name: string, frameNum: number): Promise<{ frame: number; width: number; height: number; thumbnail: string }> {
    return this.json(this.url(`/api/video/${encodeURIComponent(name)}/frame/${frameNum}`));
  }

  async listVideos(): Promise<{ videos: string[] }> {
    return this.json(this.url("/api/video/list"));
  }

  // ── Magic Wand Selection ────────────────────────────────

  async magicWandSelect(row: number, col: number, xPct: number, yPct: number, tolerance: number): Promise<{ points: number[][]; pixel_count: number; smooth?: boolean }> {
    return this.json(this.url(`/api/magic-wand/${row}/${col}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x_pct: xPct, y_pct: yPct, tolerance }),
    });
  }
}

export const api = new ApiClient();
export default ApiClient;
