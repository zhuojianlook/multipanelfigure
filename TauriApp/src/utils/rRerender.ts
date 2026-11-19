/* ──────────────────────────────────────────────────────────
   rRerender — reading the result of re-running an R plot (run-r with
   override_plot_index: collage Edit text, Synchronize headers, high-DPI
   export, SVG upgrade, and the Analysis drawer's Edit text).

   rPlotIndex is the plot's position in the `plots` list of a normal run
   (filename order). The sidecar re-renders exactly that plot or returns no
   image plus `override_error`; it never substitutes another plot, and these
   helpers never fall back to a different plot either.
   ────────────────────────────────────────────────────────── */

/** The run-r response fields read here. */
export interface RRerunResult {
  success?: boolean;
  stderr?: string | null;
  plots?: string[] | null;
  svg?: string | null;
  override_error?: string | null;
  /** Machine-readable override_error (see RAnalysisRequest in api_server.py). */
  override_code?: string | null;
}

/** Codes that are a property of the script itself: re-running the same code
 *  for the same plot cannot produce an image (the plot is base graphics /
 *  grid, was saved directly, or no longer exists). Script errors and render
 *  failures are not in here — they may be transient. */
const PERMANENT_CODES = new Set(["not_ggplot", "not_mpfig_plot", "no_such_plot"]);

/** True when this R plot can never be re-rendered from its code, so callers
 *  can stop re-running R for it (e.g. on every resize). */
export function rPlotNotRerenderable(res: RRerunResult | null | undefined): boolean {
  return !!res?.success && PERMANENT_CODES.has(res.override_code || "");
}

/** The PNG a re-run produced for one R plot item, or undefined.
 *  `override`: the request asked for the override image only (override_only),
 *  which is then the single returned plot. Otherwise the run returned every
 *  plot and the item's own is at `plotIndex` — never plot 0 in its place. */
export function rRerunPng(
  res: RRerunResult | null | undefined,
  opts: { override: boolean; plotIndex?: number },
): string | undefined {
  if (!res?.success) return undefined;
  const plots = res.plots || [];
  if (opts.override) return plots[0] || undefined;
  const i = Math.max(0, Math.floor(opts.plotIndex ?? 0));
  return plots[i] || undefined;
}

/** Why a re-render gave no image, for the user: the sidecar's reason, else
 *  the end of R's stderr (where the error is; the start is package noise),
 *  else `fallback`. */
export function rRerenderError(res: RRerunResult | null | undefined, fallback = "Re-render failed"): string {
  const reason = (res?.override_error || "").trim();
  if (reason) return reason;
  const err = (res?.stderr || "").trim();
  return err ? err.slice(-400) : fallback;
}
