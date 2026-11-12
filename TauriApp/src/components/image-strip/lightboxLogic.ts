/* ──────────────────────────────────────────────────────────
   lightboxLogic — the pure decisions behind "double-click a media
   tile to view it large" (ImageTile + ImageLightbox). Kept out of
   the components so they can be unit-tested without a DOM.
   ────────────────────────────────────────────────────────── */

/* ── Single click vs double click on a tile ──────────────────

   A tile's single click toggles the click-to-assign selection, and it must
   keep doing so with NO added delay (no "wait and see if a second click
   comes" timer — that makes every ordinary click feel laggy). But a
   double-click is delivered as click(detail=1), click(detail=2), dblclick,
   so the first click has ALREADY toggled the selection by the time we know
   it was a double-click. Toggling again on the second click is not enough:
   if image B was selected and the user double-clicks A, click 1 selects A
   and a plain toggle on click 2 would leave NOTHING selected — B is lost.

   So click 1 records what the selection was before it (`before`) and click 2
   restores exactly that, making the pair net to no change whatever was
   selected. The snapshot names the tile it came from: a detail=2 click whose
   detail=1 click landed somewhere else (another tile, a gap) must not restore
   a stale snapshot, so it changes nothing. detail=0 is a synthesized click
   (element.click(), keyboard activation) — treated as an ordinary single
   click. detail>=3 (triple click) changes nothing: by then the lightbox is
   open over the tile anyway. */

export interface TileClickSnapshot {
  /** Name of the tile whose single click took the snapshot. */
  tile: string;
  /** The selection as it was immediately before that single click. */
  before: string | null;
}

export interface TileClickResult {
  /** The selection to apply after this click. */
  next: string | null;
  /** The snapshot to remember for a possible follow-up click. */
  snapshot: TileClickSnapshot | null;
}

export function resolveTileClick(
  detail: number,
  tile: string,
  current: string | null,
  snapshot: TileClickSnapshot | null,
): TileClickResult {
  if (detail <= 1) {
    return {
      next: current === tile ? null : tile,
      snapshot: { tile, before: current },
    };
  }
  if (detail === 2 && snapshot && snapshot.tile === tile) {
    // Undo click 1 and consume the snapshot so a later stray detail=2 can't
    // replay it.
    return { next: snapshot.before, snapshot: null };
  }
  return { next: current, snapshot: null };
}

/** The one shared snapshot slot. Module-level rather than per-tile so every
 *  single click on ANY tile overwrites it — a tile can then never act on a
 *  snapshot left over from its own click minutes ago (see resolveTileClick). */
export const tileClickMemory: { snapshot: TileClickSnapshot | null } = { snapshot: null };

/* ── How big a preview to ask the backend for ────────────────

   Enough device pixels to fill the lightbox's image box (LIGHTBOX_BOX_VW ×
   LIGHTBOX_BOX_VH of the viewport — the longer of the two, × devicePixelRatio)
   so it looks crisp on a Retina screen. Capped at 2048: the preview is a
   base64 PNG carried over Tauri IPC, and at the backend's 4096 limit a real
   microscopy image runs to many MB for detail the eye can't use in a modal.
   Never less than 512, so even a tiny window gets a real upgrade over the
   256 px strip thumbnail. Garbage inputs (0, NaN — e.g. jsdom or a window
   that hasn't laid out yet) fall back to 2048. */

export const LIGHTBOX_DEFAULT_MAX_SIDE = 2048;
export const LIGHTBOX_MIN_MAX_SIDE = 512;
export const LIGHTBOX_MAX_MAX_SIDE = 2048;
/** The image box in ImageLightbox.tsx: 92vw × 88vh. Keep in sync. */
export const LIGHTBOX_BOX_VW = 92;
export const LIGHTBOX_BOX_VH = 88;

export function lightboxMaxSide(viewportW: number, viewportH: number, devicePixelRatio: number): number {
  const box = Math.max(viewportW * LIGHTBOX_BOX_VW, viewportH * LIGHTBOX_BOX_VH) / 100;
  if (!Number.isFinite(box) || box <= 0) return LIGHTBOX_DEFAULT_MAX_SIDE;
  const dpr = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
  const want = Math.ceil(box * dpr);
  return Math.min(LIGHTBOX_MAX_MAX_SIDE, Math.max(LIGHTBOX_MIN_MAX_SIDE, want));
}
