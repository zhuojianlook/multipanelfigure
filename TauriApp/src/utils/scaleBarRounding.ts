/* ──────────────────────────────────────────────────────────
   scaleBarRounding — snap a derived (inherited) scale-bar
   length in µm to a human-readable value.

   The cascade math gives bar values like 15.7 µm or 1.57 µm,
   which look awkward in a figure. This helper snaps to:
     • multiples of 5 in the local decade for v ≥ 5
       (so 15.7 → 15, 153 → 150, 1234 → 1250)
     • integers for 1 ≤ v < 5
     • 0.1 increments for 0.1 ≤ v < 1
     • two decimals for v < 0.1 (preserves accuracy for very
       small bars on deep zooms)

   mpp (µm-per-pixel) is NOT touched — only the displayed bar
   length is snapped. The bar's pixel count adjusts accordingly,
   so it still represents the snapped physical length exactly.
   ────────────────────────────────────────────────────────── */

export function niceScaleBarUm(v: number, mode: "round" | "floor" = "round"): number {
  if (!isFinite(v) || v <= 0) return mode === "floor" ? 0.01 : 1;
  const snap = (x: number, step: number) => {
    const k = mode === "floor" ? Math.floor(x / step) : Math.round(x / step);
    return k * step;
  };
  if (v < 0.1) {
    return Math.max(0.01, snap(v, 0.01));
  }
  if (v < 1) {
    return Math.max(0.1, snap(v, 0.1));
  }
  if (v < 5) {
    return Math.max(1, snap(v, 1));
  }
  // v ≥ 5 — snap to multiples of 5 within the local decade.
  const exp = Math.floor(Math.log10(v));
  const base5 = 5 * Math.pow(10, exp - 1);
  return Math.max(base5, snap(v, base5));
}
