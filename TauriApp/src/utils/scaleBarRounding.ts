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

export function niceScaleBarUm(v: number): number {
  if (!isFinite(v) || v <= 0) return 1;
  if (v < 0.1) {
    return Math.max(0.01, Math.round(v * 100) / 100);
  }
  if (v < 1) {
    return Math.round(v * 10) / 10;
  }
  if (v < 5) {
    return Math.max(1, Math.round(v));
  }
  // v ≥ 5 — snap to multiples of 5 within the local decade.
  // For v in [5, 50): multiples of 5 (5, 10, 15, 20, ...)
  // For v in [50, 500): multiples of 50 (50, 100, 150, ...)
  // For v in [500, 5000): multiples of 500, etc.
  const exp = Math.floor(Math.log10(v));
  const base5 = 5 * Math.pow(10, exp - 1);
  return Math.max(base5, Math.round(v / base5) * base5);
}
