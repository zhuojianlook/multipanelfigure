/* ──────────────────────────────────────────────────────────
   CollageStrip — horizontal "timeline" of all items currently
   in the collage. Acts like the multi-panel builder's
   ImageStrip but lives in the collage workspace and shows
   both imported images and rendered figures (badged so the
   user can tell them apart at a glance).

   Clicking a thumbnail selects the item on the canvas;
   clicking the trash icon removes that single item — no
   global wipe.
   ────────────────────────────────────────────────────────── */

import { Box, IconButton, Tooltip, Typography, Chip } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import VerticalAlignTopIcon from "@mui/icons-material/VerticalAlignTop";
import { useCollageStore, type CollageItem } from "../../store/collageStore";
import { api } from "../../api/client";

export function CollageStrip() {
  const items = useCollageStore((s) => s.items);
  const selectedId = useCollageStore((s) => s.selectedId);
  const setSelectedId = useCollageStore((s) => s.setSelectedId);
  const removeItem = useCollageStore((s) => s.removeItem);
  const bringToFront = useCollageStore((s) => s.bringToFront);

  /** Remove an item AND clean up its stash file. Stash deletion is
   *  best-effort — if the backend can't reach it we still drop the
   *  item from the collage. */
  const removeItemAndStash = async (it: CollageItem) => {
    try {
      await api.deleteCollageStash(it.id);
    } catch {
      /* ignore — orphaned stash will be cleaned up on next reset */
    }
    removeItem(it.id);
  };

  return (
    <Box
      sx={{
        flexShrink: 0,
        borderTop: "1px solid var(--c-border)",
        backgroundColor: "var(--c-surface)",
        px: 1.5,
        py: 1,
        minHeight: 96,
        display: "flex",
        alignItems: "center",
        gap: 1,
        overflowX: "auto",
      }}
    >
      <Typography
        variant="caption"
        sx={{ flexShrink: 0, color: "text.secondary", letterSpacing: 1.2, fontSize: "0.6rem", textTransform: "uppercase" }}
      >
        Collage items
      </Typography>
      {items.length === 0 ? (
        <Typography variant="caption" sx={{ color: "text.secondary", fontStyle: "italic" }}>
          Use "Add to Collage" from the Multi-Panel Builder, or "Import image" / "Import project" above.
        </Typography>
      ) : (
        items.map((it) => {
          const isSelected = it.id === selectedId;
          return (
            <Box
              key={it.id}
              onClick={() => {
                setSelectedId(it.id);
                bringToFront(it.id);
              }}
              sx={{
                position: "relative",
                width: 80,
                height: 70,
                flexShrink: 0,
                border: isSelected ? "2px solid #4FC3F7" : "1px solid var(--c-border)",
                borderRadius: 1,
                cursor: "pointer",
                backgroundColor: "var(--c-bg)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
              }}
              title={it.name}
            >
              <Box
                component="img"
                src={it.src}
                alt={it.name}
                sx={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", pointerEvents: "none" }}
                draggable={false}
              />
              <Chip
                label={it.kind === "figure" ? "Fig" : "Img"}
                size="small"
                sx={{
                  position: "absolute",
                  top: 2,
                  left: 2,
                  height: 14,
                  fontSize: "0.55rem",
                  backgroundColor: it.kind === "figure" ? "rgba(79,195,247,0.85)" : "rgba(0,0,0,0.6)",
                  color: "#fff",
                  "& .MuiChip-label": { px: 0.5 },
                }}
              />
              <Box
                sx={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  display: "flex",
                  justifyContent: "space-between",
                  px: 0.25,
                  py: 0.25,
                  background: "linear-gradient(to top, rgba(0,0,0,0.55), transparent)",
                }}
              >
                <Tooltip title="Bring to front">
                  <IconButton
                    size="small"
                    onClick={(e) => { e.stopPropagation(); bringToFront(it.id); }}
                    sx={{ width: 18, height: 18, color: "#fff" }}
                  >
                    <VerticalAlignTopIcon sx={{ fontSize: 12 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Remove this item">
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm(`Remove "${it.name}" from the collage?`)) {
                        removeItemAndStash(it);
                      }
                    }}
                    sx={{ width: 18, height: 18, color: "#ff8a80" }}
                  >
                    <DeleteIcon sx={{ fontSize: 12 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          );
        })
      )}
    </Box>
  );
}
