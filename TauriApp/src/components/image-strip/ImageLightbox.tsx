/* ──────────────────────────────────────────────────────────
   ImageLightbox — full-window view of one loaded image, opened by
   double-clicking a media-strip tile (timeline or group drawer).

   Shows the tile's 256 px thumbnail at once (scaled up to fill the
   box, so the layout never jumps), asks the backend for a
   screen-sized version through the same /thumb endpoint, and swaps
   it in when it lands. Closes on the X button, Esc, or a click on
   the dark surround — not on a click on the image itself.
   ────────────────────────────────────────────────────────── */

import { useEffect, useState } from "react";
import type { SyntheticEvent } from "react";
import { Modal, Fade, IconButton, CircularProgress } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { api } from "../../api/client";
import { lightboxMaxSide } from "./lightboxLogic";

/** Above every layer this app uses: MUI drawers (1200), the busy overlay
 *  (1250), dialogs (1300), tooltips (1500) and the colour-picker / text
 *  poppers (1600). Only the 4 px top progress bar (9999) stays over it. */
const LIGHTBOX_Z = 1700;

/** React synthetic events bubble up the REACT tree, not the DOM tree — so
 *  even though the Modal portals into <body>, a click / double-click /
 *  drag inside it would otherwise climb back up into whatever rendered the
 *  lightbox: the ImageTile's selection toggle, its own onDoubleClick (which
 *  would re-open us), the group drawer's handlers. Swallow them all at the
 *  Modal root; everything inside has already run by then. */
const swallow = (e: SyntheticEvent) => e.stopPropagation();

export interface ImageLightboxProps {
  open: boolean;
  /** Backend image name (the loaded_images key; may contain non-ASCII). */
  name: string;
  /** The strip's already-loaded thumbnail, shown immediately. */
  thumbnailB64: string;
  onClose: () => void;
}

export function ImageLightbox({ open, name, thumbnailB64, onClose }: ImageLightboxProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}       // Esc (MUI's own keydown handler on the root)
      closeAfterTransition
      hideBackdrop            // our full-window surround IS the backdrop
      sx={{ zIndex: LIGHTBOX_Z }}
      onClick={swallow}
      onDoubleClick={swallow}
      onMouseDown={swallow}
      onPointerDown={swallow}
      onContextMenu={swallow}
      onDragStart={(e) => { e.preventDefault(); e.stopPropagation(); }}
      // Runs BEFORE MUI's Escape handling on the same element, so Esc still
      // closes; it just doesn't leak to the tile / drawer underneath.
      onKeyDown={swallow}
    >
      <Fade in={open}>
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Image preview: ${name}`}
          // A click anywhere that isn't the image / caption / X lands here.
          onClick={onClose}
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            // Always dark, whatever the app theme: a lightbox surround should
            // be neutral and recede so the image is judged on its own —
            // which is also why the caption / X use fixed light colours
            // rather than the theme's --c-text.
            backgroundColor: "rgba(0, 0, 0, 0.88)",
            outline: "none",
            overflow: "hidden",
          }}
        >
          <IconButton
            aria-label="Close"
            title="Close (Esc)"
            onClick={(e) => { e.stopPropagation(); onClose(); }}
            sx={{
              position: "absolute",
              top: 12,
              right: 12,
              color: "#ffffff",
              backgroundColor: "rgba(255, 255, 255, 0.12)",
              "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.24)" },
            }}
          >
            <CloseIcon />
          </IconButton>
          {/* Keyed by name so switching images starts from a clean slate
              (no previous image's large version or error note). */}
          <LightboxImage key={name} name={name} thumbnailB64={thumbnailB64} active={open} />
        </div>
      </Fade>
    </Modal>
  );
}

/** The image + caption. Mounted only while the Modal shows its content, so
 *  each opening starts with fresh state. */
function LightboxImage({ name, thumbnailB64, active }: {
  name: string;
  thumbnailB64: string;
  /** False while the Modal fades out — a response landing then is ignored. */
  active: boolean;
}) {
  const [largeB64, setLargeB64] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  // Natural width / height of whatever is displayed. Until it's known the
  // image renders invisibly at natural size; once known it's sized to fill
  // the 92vw × 88vh box at its true aspect, so the 256 px thumbnail and the
  // large version occupy the SAME box and the swap doesn't move anything.
  const [aspect, setAspect] = useState<number | null>(null);

  useEffect(() => {
    if (!active) return;
    // `alive` goes false when the lightbox closes, switches image or
    // unmounts — a late response must not touch state after any of those.
    let alive = true;
    const maxSide = lightboxMaxSide(window.innerWidth, window.innerHeight, window.devicePixelRatio);
    api.getImagePreview(name, maxSide).then(
      (res) => {
        if (!alive) return;
        if (res && typeof res.thumbnail === "string" && res.thumbnail.length > 0) setLargeB64(res.thumbnail);
        else setFailed(true);
      },
      // e.g. a 404 for an entry the backend doesn't hold as a still image.
      // The thumbnail stays up; the caption says why it may look soft.
      () => { if (alive) setFailed(true); },
    );
    return () => { alive = false; };
  }, [name, active]);

  const src = `data:image/png;base64,${largeB64 ?? thumbnailB64}`;
  const loading = !largeB64 && !failed;

  return (
    <>
      <img
        src={src}
        alt={name}
        draggable={false}
        // Clicking the picture itself shouldn't dismiss it.
        onClick={(e) => e.stopPropagation()}
        onLoad={(e) => {
          const im = e.currentTarget;
          if (im.naturalWidth > 0 && im.naturalHeight > 0) setAspect(im.naturalWidth / im.naturalHeight);
        }}
        onError={() => {
          // A large payload that won't decode: fall back to the thumbnail.
          if (largeB64) { setLargeB64(null); setFailed(true); }
        }}
        style={
          aspect
            ? {
                // Fit-inside the box at the image's own aspect, scaling UP
                // as well as down (this is the "see it large" view).
                width: `min(92vw, calc(88vh * ${aspect}))`,
                aspectRatio: String(aspect),
                height: "auto",
                objectFit: "contain",
                display: "block",
                boxShadow: "0 8px 40px rgba(0, 0, 0, 0.6)",
                backgroundColor: "rgba(255, 255, 255, 0.03)",
              }
            : { maxWidth: "92vw", maxHeight: "88vh", display: "block", opacity: 0 }
        }
      />
      <div
        className="flex items-center gap-2 text-[12px]"
        style={{ color: "rgba(255, 255, 255, 0.85)", maxWidth: "92vw" }}
        // Selecting / copying the filename shouldn't close the lightbox.
        onClick={(e) => e.stopPropagation()}
      >
        <span className="truncate" title={name}>{name}</span>
        {loading && (
          <CircularProgress
            size={12}
            thickness={6}
            sx={{ color: "rgba(255, 255, 255, 0.7)", flex: "none" }}
            aria-label="Loading full-size preview"
          />
        )}
        {failed && (
          <span className="flex-none text-[11px]" style={{ color: "rgba(255, 255, 255, 0.5)", fontStyle: "italic" }}>
            — full-size preview unavailable, showing thumbnail
          </span>
        )}
      </div>
    </>
  );
}
