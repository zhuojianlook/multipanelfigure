/* ──────────────────────────────────────────────────────────
   Helpers for the font dropdowns shared by the collage text
   controls and the Synchronize panel: stable alphabetical
   ordering + an Arial-first default so every selector behaves
   consistently (instead of showing whatever the OS lists first).
   ────────────────────────────────────────────────────────── */

/** Display name = file name without its extension. */
export const fontDisplayName = (f: string): string =>
  f.replace(/\.(ttf|otf|ttc|woff2?)$/i, "");

/** Return the font list sorted case-insensitively by display name. */
export function sortFontList(fonts: string[]): string[] {
  return [...fonts].sort((a, b) =>
    fontDisplayName(a).toLowerCase().localeCompare(fontDisplayName(b).toLowerCase()),
  );
}

/** Pick a sensible default font: Arial if present, then Helvetica, else the
 *  first one alphabetically. Falls back to "arial.ttf" when the list is empty. */
export function pickDefaultFont(fonts: string[]): string {
  const sorted = sortFontList(fonts);
  const arial =
    sorted.find((f) => /^arial(\b|[ ._-]|$)/i.test(fontDisplayName(f))) ||
    sorted.find((f) => /arial/i.test(fontDisplayName(f)));
  const helvetica = sorted.find((f) => /^helvetica/i.test(fontDisplayName(f)));
  return arial || helvetica || sorted[0] || "arial.ttf";
}
