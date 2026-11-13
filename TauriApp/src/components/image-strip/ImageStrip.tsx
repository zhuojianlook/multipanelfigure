/* ──────────────────────────────────────────────────────────
   ImageStrip — horizontal scrollable strip of loaded images.
   Click an image to select it, then click a panel to assign.
   Also supports HTML5 drag-and-drop.
   Includes collapsible media groups below the timeline.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect, useRef, useMemo } from "react";
import { Tooltip, IconButton, CircularProgress } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import { useFigureStore } from "../../store/figureStore";
import type { LoadedImage } from "../../store/figureStore";
import type { ImageGroup } from "../../api/types";
import { sectionsOf, type SectionTarget } from "../../utils/imageGroups";
import { imageUsage, type FigureUse, type ImageUsage } from "../../utils/figures";
import { confirm } from "../shared/ConfirmDialog";
import { FigureMarker } from "../shared/FigureMarker";
import type { DragEvent as ReactDragEvent } from "react";
import { ImageLightbox } from "./ImageLightbox";
import { resolveTileClick, tileClickMemory } from "./lightboxLogic";
import { useDrawerState } from "./drawerState";

// Global selected-image state for click-to-assign
let _selectedImageName: string | null = null;
let _listeners: (() => void)[] = [];

export function getSelectedImageName() { return _selectedImageName; }
export function setSelectedImageName(name: string | null) {
  _selectedImageName = name;
  _listeners.forEach(fn => fn());
}
export function clearSelectedImage() { setSelectedImageName(null); }
export function useSelectedImage() {
  const [, setTick] = useState(0);
  useEffect(() => {
    const fn = () => setTick(t => t + 1);
    _listeners.push(fn);
    return () => { _listeners = _listeners.filter(f => f !== fn); };
  }, []);
  return _selectedImageName;
}

/** Placeholder tile shown while an image is still uploading/processing */
function LoadingTile({ name }: { name: string }) {
  return (
    <Tooltip title={`Loading ${name}…`} placement="bottom" arrow enterDelay={200}>
      <div
        className="relative flex-none flex flex-col items-center gap-0.5 rounded border p-1"
        style={{
          borderColor: "var(--c-border)",
          backgroundColor: "var(--c-surface)",
          borderWidth: 1,
          borderStyle: "dashed",
          opacity: 0.85,
        }}
      >
        <div
          className="w-12 h-12 flex items-center justify-center rounded"
          style={{ backgroundColor: "rgba(255,255,255,0.04)" }}
        >
          <CircularProgress size={22} thickness={5} />
        </div>
        <span
          className="text-[9px] max-w-[56px] truncate"
          style={{ color: "var(--c-text-dim)", fontStyle: "italic" }}
          title={name}
        >
          {name}
        </span>
      </div>
    </Tooltip>
  );
}

/** Reusable image thumbnail tile (timeline AND group drawers).
 *  Click toggles the click-to-assign selection; double-click opens the
 *  image large in a lightbox; drag carries it to a panel / group. */
/** How a tile shows where its image is used. An image can sit in panels of
 *  several figure tabs:
 *    - the green border highlights images used in the figure ON SCREEN
 *      (as before figure tabs) — and only those;
 *    - once the project has more than one figure, one numbered marker per
 *      figure that uses the image, numbered like the figure tabs (green =
 *      the figure on screen). The tooltip names them.
 *  A single-figure project looks exactly as it always did. */
interface TileUsageProps {
  usage: ImageUsage | undefined;
  /** The project has more than one figure tab. */
  multiFigure: boolean;
}

const panelsText = (n: number) => `${n} panel${n === 1 ? "" : "s"}`;

function usageLines(u: ImageUsage | undefined): string[] {
  if (!u) return [];
  return u.byFigure.map((f) =>
    `${f.index} · ${f.name} (${f.current ? "this figure, " : ""}${panelsText(f.count)})`);
}

/** Most markers a 56px tile can show; past that the last slot reads "+n". */
const MAX_TILE_MARKERS = 4;

/** The markers to draw: every figure if they fit, otherwise the figure on
 *  screen (if it uses the image) plus the first others, in tab order. */
function tileMarkers(byFigure: FigureUse[]): { shown: FigureUse[]; hidden: number } {
  if (byFigure.length <= MAX_TILE_MARKERS) return { shown: byFigure, hidden: 0 };
  const room = MAX_TILE_MARKERS - 1;
  const cur = byFigure.find((f) => f.current);
  const picked = new Set<FigureUse>(cur ? [cur] : []);
  for (const f of byFigure) {
    if (picked.size >= room) break;
    picked.add(f);
  }
  const shown = byFigure.filter((f) => picked.has(f));
  return { shown, hidden: byFigure.length - shown.length };
}

function ImageTile({ img, usage, multiFigure, isSelected, onRemove }: TileUsageProps & {
  img: LoadedImage;
  isSelected: boolean;
  onRemove: () => void;
}) {
  // Lightbox state is local to the tile: nothing else needs to know a
  // preview is open, so no global store change.
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const inCurrent = (usage?.inCurrent ?? 0) > 0;
  const where = usageLines(usage);
  const markers = multiFigure && usage ? tileMarkers(usage.byFigure) : { shown: [], hidden: 0 };
  const tooltip = multiFigure && where.length > 0 ? (
    <span style={{ display: "block" }}>
      <span style={{ display: "block", fontWeight: 600 }}>{img.name}</span>
      <span style={{ display: "block" }}>Used in: {where.join(" · ")}</span>
    </span>
  ) : img.name;
  return (
    <>
    {/* Empty title while the lightbox is up: MUI renders no tooltip for an
        empty title, and tooltips (z 1500) would otherwise float over the
        lightbox if the pointer was still over the tile when it opened. */}
    <Tooltip title={lightboxOpen ? "" : tooltip} placement="bottom" arrow enterDelay={200}>
      <div
        className="relative flex-none flex flex-col items-center gap-0.5 rounded border p-1 group"
        style={{
          borderColor: isSelected ? "#2196f3" : inCurrent ? "var(--c-green)" : "var(--c-border)",
          backgroundColor: isSelected ? "rgba(33,150,243,0.15)" : "var(--c-surface)",
          borderWidth: isSelected ? 3 : inCurrent ? 2 : 1,
          borderStyle: "solid",
          cursor: "pointer",
          transition: "border-color 0.15s, background-color 0.15s",
        }}
        // Single click toggles selection IMMEDIATELY (no wait-for-a-second-
        // click delay). The second click of a double-click (detail=2) then
        // puts the selection back exactly as it was before the first —
        // see resolveTileClick. Reads the live global rather than the
        // `isSelected` prop so the toggle never depends on a re-render
        // having landed between the two clicks.
        onClick={(e) => {
          const current = getSelectedImageName();
          const { next, snapshot } = resolveTileClick(e.detail, img.name, current, tileClickMemory.snapshot);
          tileClickMemory.snapshot = snapshot;
          if (next !== current) setSelectedImageName(next);
        }}
        onDoubleClick={(e) => {
          e.stopPropagation(); // the tile owns this gesture; containers shouldn't also react
          setLightboxOpen(true);
        }}
        draggable="true"
        onDragStart={(e) => {
          e.dataTransfer.setData("application/x-image-name", img.name);
          e.dataTransfer.setData("text/plain", img.name);
          e.dataTransfer.effectAllowed = "copyMove";
        }}
      >
        <img
          src={`data:image/png;base64,${img.thumbnailB64}`}
          alt={img.name}
          className="w-12 h-12 object-contain rounded"
          draggable={false}
        />
        <span
          className="text-[9px] max-w-[56px] truncate"
          style={{ color: isSelected ? "#2196f3" : "var(--c-text-dim)" }}
          title={img.name}
        >
          {img.name}
        </span>
        {markers.shown.length > 0 && (
          <div
            data-testid="tile-figure-markers"
            aria-label={`Used in ${where.join("; ")}`}
            className="absolute -bottom-1.5 left-0.5 flex gap-[1px] pointer-events-none"
          >
            {markers.shown.map((f) => (
              <FigureMarker key={f.id} label={f.index} current={f.current} />
            ))}
            {markers.hidden > 0 && <FigureMarker label={`+${markers.hidden}`} current={false} />}
          </div>
        )}
        {isSelected && (
          <div
            className="absolute -top-1 left-1/2 -translate-x-1/2 text-[8px] px-1 rounded-b"
            style={{ backgroundColor: "#2196f3", color: "#fff", whiteSpace: "nowrap" }}
          >
            Click panel to assign
          </div>
        )}
        <button
          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full flex items-center justify-center
                     text-[10px] font-bold leading-none opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
          style={{ backgroundColor: "var(--c-red)", color: "#ffffff" }}
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          // A double-click on × must not open the lightbox on the tile
          // it is removing.
          onDoubleClick={(e) => e.stopPropagation()}
          aria-label={`Remove image ${img.name}`}
          title="Remove image"
        >
          &times;
        </button>
      </div>
    </Tooltip>
    {/* Sibling of the Tooltip, not inside it: the lightbox portals to <body>
        but its React events still bubble through its React parents, and it
        must not drive the Tooltip's hover / focus listeners. */}
    <ImageLightbox
      open={lightboxOpen}
      name={img.name}
      thumbnailB64={img.thumbnailB64}
      onClose={() => setLightboxOpen(false)}
    />
    </>
  );
}

/** Inline editable group name */
function EditableName({ value, onChange, subgroup = false, autoEdit = false, onEditEnd }: {
  value: string;
  onChange: (v: string) => void;
  /** The smaller label style of a subgroup box's name tag (horizontal,
   *  ellipsized past ~18 characters) instead of the group-name style. */
  subgroup?: boolean;
  /** Mount straight into rename mode with the text selected — used for a
   *  divider the user just placed, so typing replaces "Subgroup N". */
  autoEdit?: boolean;
  /** Called once when an edit ends, whether committed or cancelled. */
  onEditEnd?: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const [editing, setEditing] = useState(autoEdit);
  const [draft, setDraft] = useState(value);
  // True only while an edit is open. Enter / Esc unmount the focused input,
  // and some engines then fire a trailing blur whose stale closure would
  // commit a second time — or commit the very draft Esc meant to discard.
  const activeRef = useRef(autoEdit);

  useEffect(() => {
    if (autoEdit) setTimeout(() => ref.current?.select(), 0);
    // Mount-only: autoEdit just picks the initial mode.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const begin = () => {
    activeRef.current = true;
    setDraft(value);   // never reopen on a stale draft from an earlier edit
    setEditing(true);
    setTimeout(() => ref.current?.select(), 0);
  };

  const finish = (save: boolean) => {
    if (!activeRef.current) return;
    activeRef.current = false;
    setEditing(false);
    const next = draft.trim();
    if (save && next && next !== value) onChange(next);
    else setDraft(value);
    onEditEnd?.();
  };

  if (!editing) {
    return subgroup ? (
      <span
        data-testid="subgroup-name"
        className="text-[9px] font-semibold cursor-pointer px-0.5 rounded hover:bg-white/10"
        style={{
          color: "var(--c-text)",
          maxWidth: 110,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          lineHeight: "14px",
        }}
        onDoubleClick={begin}
        title={`${value} — double-click to rename`}
      >
        {value}
      </span>
    ) : (
      <span
        className="text-[10px] font-semibold cursor-pointer px-1 rounded hover:bg-white/10"
        style={{ color: "var(--c-text)" }}
        onDoubleClick={begin}
        title="Double-click to rename"
      >
        {value}
      </span>
    );
  }

  return (
    <input
      ref={ref}
      maxLength={subgroup ? 60 : undefined}
      className={`${subgroup ? "text-[9px]" : "text-[10px]"} font-semibold px-1 rounded outline-none`}
      style={{ backgroundColor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-accent)", width: subgroup ? 110 : 120 }}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => finish(true)}
      onKeyDown={(e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); }}
      autoFocus
    />
  );
}

export function ImageStrip() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const config = useFigureStore((s) => s.config);
  const removeImage = useFigureStore((s) => s.removeImage);
  const imageGroups = useFigureStore((s) => s.imageGroups);
  const createImageGroup = useFigureStore((s) => s.createImageGroup);
  const renameImageGroup = useFigureStore((s) => s.renameImageGroup);
  const deleteImageGroup = useFigureStore((s) => s.deleteImageGroup);
  const moveImageToGroup = useFigureStore((s) => s.moveImageToGroup);
  const moveImageToTimeline = useFigureStore((s) => s.moveImageToTimeline);
  const addGroupDivider = useFigureStore((s) => s.addGroupDivider);
  const renameGroupDivider = useFigureStore((s) => s.renameGroupDivider);
  const deleteGroupDivider = useFigureStore((s) => s.deleteGroupDivider);
  const pendingUploads = useFigureStore((s) => s.pendingUploads);
  const figures = useFigureStore((s) => s.figures);
  const activeFigureId = useFigureStore((s) => s.activeFigureId);
  const selectedImage = useSelectedImage();
  // The drawer's and each group's collapsed state outlive this component
  // (see drawerState): reopening the drawer restores every group as it was.
  const groupsExpanded = useDrawerState((s) => s.drawerOpen);
  const setGroupsExpanded = useDrawerState((s) => s.setDrawerOpen);
  const [timelineDragOver, setTimelineDragOver] = useState(false);

  const entries = Object.values(loadedImages);
  const groupedNames = new Set(imageGroups.flatMap(g => g.imageNames));
  const timelineEntries = entries.filter(img => !groupedNames.has(img.name));
  // Only show a placeholder for uploads whose filename isn't already present
  // as a loaded image (handles both name collisions and race conditions).
  const loadedNames = new Set(entries.map((e) => e.name));
  const visiblePendingUploads = pendingUploads.filter((n) => !loadedNames.has(n));

  // Where each image is used — in this figure and in the project's other
  // figure tabs (an image can be in several figures at once).
  const usage = useMemo(() => imageUsage(figures, activeFigureId, config), [figures, activeFigureId, config]);
  const tileUsage = { multiFigure: figures.length > 1 };
  // Removing an image empties every panel that shows it, in EVERY figure —
  // so warn whenever any figure uses it (the one on screen included).
  const requestRemove = async (name: string) => {
    const u = usage.get(name);
    if (u && (u.inCurrent > 0 || u.others.length > 0)) {
      const where = u.byFigure.length > 0
        ? u.byFigure.map((f) => `${f.name} (${f.current ? "this figure, " : ""}${panelsText(f.count)})`)
        : [`this figure (${panelsText(u.inCurrent)})`];
      const several = u.byFigure.length > 1;
      const ok = await confirm({
        title: "Remove an image that is in use?",
        body: `"${name}" is used in ${where.join(", ")}. `
          + `Removing it empties ${several ? "those panels in every figure" : "those panels"}.`,
        confirmLabel: several ? "Remove from all figures" : "Remove image",
        destructive: true,
      });
      if (!ok) return;
    }
    await removeImage(name);
  };

  if (entries.length === 0 && visiblePendingUploads.length === 0) {
    return (
      <div
        className="flex items-center px-3 py-2 text-[11px] border-b flex-none"
        style={{ borderColor: "var(--c-border)", color: "var(--c-text-dim)", backgroundColor: "var(--c-bg)" }}
      >
        No images loaded. Use the toolbar to add images.
      </div>
    );
  }

  return (
    <div className="flex-none" style={{ backgroundColor: "var(--c-bg)" }}>
      {/* Main timeline */}
      <div
        className="flex items-center gap-2 px-3 py-2 overflow-x-auto border-b"
        style={{
          borderColor: "var(--c-border)",
          backgroundColor: timelineDragOver ? "rgba(33,150,243,0.08)" : undefined,
          minHeight: 72,
        }}
        onDragOver={(e) => {
          const imgName = e.dataTransfer.types.includes("application/x-image-name");
          if (imgName) { e.preventDefault(); e.dataTransfer.dropEffect = "move"; setTimelineDragOver(true); }
        }}
        onDragLeave={() => setTimelineDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setTimelineDragOver(false);
          const imgName = e.dataTransfer.getData("application/x-image-name");
          if (imgName && groupedNames.has(imgName)) {
            moveImageToTimeline(imgName);
          }
        }}
      >
        {timelineEntries.length === 0 && visiblePendingUploads.length === 0 ? (
          <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
            All images are in groups. Drag here to return to timeline.
          </span>
        ) : (
          <>
            {timelineEntries.map((img) => (
              <ImageTile
                key={img.name}
                img={img}
                usage={usage.get(img.name)}
                {...tileUsage}
                isSelected={selectedImage === img.name}
                onRemove={() => { void requestRemove(img.name); }}
              />
            ))}
            {visiblePendingUploads.map((name, i) => (
              <LoadingTile key={`pending-${name}-${i}`} name={name} />
            ))}
          </>
        )}
        {selectedImage && (
          <button
            className="flex items-center text-[10px] px-2 py-1 rounded border"
            style={{ color: "var(--c-text-dim)", borderColor: "var(--c-border)", backgroundColor: "transparent", cursor: "pointer" }}
            onClick={() => clearSelectedImage()}
          >
            Cancel
          </button>
        )}
      </div>

      {/* Groups section */}
      <div className="border-b" style={{ borderColor: "var(--c-border)" }}>
        {/* Groups header */}
        <div
          className="flex items-center gap-1 px-3 py-1"
          style={{ backgroundColor: "var(--c-surface)" }}
        >
          <IconButton size="small" onClick={() => setGroupsExpanded(!groupsExpanded)} sx={{ p: 0.25 }}>
            {groupsExpanded ? <ExpandLessIcon sx={{ fontSize: 14 }} /> : <ExpandMoreIcon sx={{ fontSize: 14 }} />}
          </IconButton>
          <span className="text-[10px] font-semibold tracking-wider uppercase" style={{ color: "var(--c-text-dim)" }}>
            Groups ({imageGroups.length})
          </span>
          <IconButton
            size="small"
            onClick={() => createImageGroup(`Group ${imageGroups.length + 1}`)}
            sx={{ p: 0.25, ml: 0.5 }}
            title="Create new group"
          >
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </div>

        {/* Group rows */}
        {groupsExpanded && imageGroups.map((group) => (
          <GroupRow
            key={group.id}
            group={group}
            loadedImages={loadedImages}
            usage={usage}
            tileUsage={tileUsage}
            selectedImage={selectedImage}
            onRename={(name) => renameImageGroup(group.id, name)}
            onDelete={() => deleteImageGroup(group.id)}
            onMoveImage={(imgName, target) => moveImageToGroup(imgName, group.id, target)}
            onAddDivider={(opts) => addGroupDivider(group.id, opts)}
            onRenameDivider={(dividerId, name) => renameGroupDivider(group.id, dividerId, name)}
            onDeleteDivider={(dividerId) => deleteGroupDivider(group.id, dividerId)}
            onRemoveImage={requestRemove}
          />
        ))}
      </div>
    </div>
  );
}

/** MIME type every image-tile drag carries (set in ImageTile's dragstart).
 *  Drawer drop targets react to nothing else, so OS file drops and other
 *  in-app drags pass straight through them. */
const IMAGE_DRAG_MIME = "application/x-image-name";
const isImageDrag = (e: ReactDragEvent) => e.dataTransfer.types.includes(IMAGE_DRAG_MIME);

/** Hover-revealed "+" in the gap between two tiles of a drawer section.
 *  Clicking it starts a new subgroup at that gap: the tile on its right and
 *  every tile after it in the section move into the new subgroup's box.
 *  It occupies only the gap itself (it IS the inter-tile spacing), never a
 *  tile, so it can't steal a tile's click, double-click or drag — and it is
 *  fully transparent until the pointer (or keyboard focus) is on it. */
function SplitHandle({ onSplit }: { onSplit: () => void }) {
  return (
    <div className="relative flex-none self-stretch flex items-center justify-center" style={{ width: 8 }}>
      <button
        type="button"
        className="relative flex items-center justify-center opacity-0 hover:opacity-100 focus-visible:opacity-100 transition-opacity"
        style={{ width: 8, height: 32, padding: 0, border: 0, background: "transparent", cursor: "pointer" }}
        onClick={(e) => { e.stopPropagation(); onSplit(); }}
        onDoubleClick={(e) => e.stopPropagation()}
        title="Start a new subgroup here"
        aria-label="Start a new subgroup here"
      >
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-1/2 -translate-x-1/2 rounded-full"
          style={{ width: 2, backgroundColor: "var(--c-accent)" }}
        />
        <span
          aria-hidden
          className="pointer-events-none relative flex items-center justify-center rounded-full text-[11px] font-bold leading-none"
          style={{ width: 14, height: 14, flex: "none", backgroundColor: "var(--c-accent)", color: "#ffffff" }}
        >
          +
        </span>
      </button>
    </div>
  );
}

/** The name tag on a subgroup box's top edge: the subgroup's name, read
 *  horizontally (double-click to rename), and a delete revealed while the
 *  box is hovered. Deleting never loses images — they merge into the
 *  section on the left (see deleteDividerFromGroup). */
function SubgroupTag({ name, autoEdit, onRename, onDelete, onEditEnd }: {
  name: string;
  autoEdit: boolean;
  onRename: (name: string) => void;
  onDelete: () => void;
  onEditEnd: () => void;
}) {
  return (
    <div
      className="flex items-center gap-0.5 self-start rounded px-1"
      style={{
        // Pulled up onto the box's top border, fieldset-legend style, but
        // kept in the flow so a long name widens the box instead of
        // overflowing it.
        marginTop: -8,
        marginBottom: 1,
        height: 16,
        backgroundColor: "var(--c-surface)",
        border: "1px solid var(--c-accent)",
      }}
    >
      <EditableName subgroup value={name} onChange={onRename} autoEdit={autoEdit} onEditEnd={onEditEnd} />
      <button
        type="button"
        className="flex items-center justify-center rounded opacity-0 group-hover/subgroup:opacity-100 focus-visible:opacity-100 transition-opacity hover:bg-white/10"
        style={{ width: 12, height: 12, padding: 0, border: 0, background: "transparent", color: "var(--c-text-dim)", cursor: "pointer" }}
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        title="Remove this subgroup (its images join the section on its left)"
        aria-label={`Delete subgroup ${name}`}
      >
        <DeleteOutlineIcon sx={{ fontSize: 11 }} />
      </button>
    </div>
  );
}

/** A single group row with drag-and-drop.
 *
 *  Expanded, the drawer is one horizontal run of SECTIONS: the unsorted
 *  images (no frame), then each subgroup as its own outlined box with its
 *  name on the top edge, so subgrouped images read apart from unsorted ones.
 *  Every section is its own drop target (drop = append to that section) and
 *  so is every tile (drop = insert before that tile, in its section). Those
 *  handlers stopPropagation: the row-level handler — still needed for drops
 *  on the name area and on collapsed rows (=> unsectioned) — must not ALSO
 *  fire, or the image would be moved twice. */
function GroupRow({
  group, loadedImages, usage, tileUsage, selectedImage, onRename, onDelete,
  onMoveImage, onAddDivider, onRenameDivider, onDeleteDivider, onRemoveImage,
}: {
  group: ImageGroup;
  loadedImages: Record<string, LoadedImage>;
  usage: Map<string, ImageUsage>;
  tileUsage: Omit<TileUsageProps, "usage">;
  selectedImage: string | null;
  onRename: (name: string) => void;
  onDelete: () => void;
  onMoveImage: (imageName: string, target?: SectionTarget) => void;
  onAddDivider: (opts?: { splitBeforeImage?: string }) => string;
  onRenameDivider: (dividerId: string, name: string) => void;
  onDeleteDivider: (dividerId: string) => void;
  onRemoveImage: (name: string) => Promise<void>;
}) {
  const [dragOver, setDragOver] = useState(false);
  // Each group is individually collapsible — when there are many groups,
  // collapsing the ones you're not working with keeps this strip short so
  // it doesn't squeeze the preview / panel planner below it. Collapsed
  // groups still accept drag-and-drop drops (the handler is on the row).
  // Stored by group id outside the row, so collapsing and reopening the
  // whole drawer (which unmounts the rows) keeps each group as it was.
  const collapsed = useDrawerState((s) => !!s.collapsedGroups[group.id]);
  const setGroupCollapsed = useDrawerState((s) => s.setGroupCollapsed);
  const setCollapsed = (c: boolean) => setGroupCollapsed(group.id, c);
  // Where a drag is hovering inside the expanded drawer: which section
  // (divider id, or "" for the unsectioned area) and, when over a tile,
  // which tile it would be inserted before. Drives the per-section
  // highlight and the insertion bar.
  const [dropAt, setDropAt] = useState<{ section: string; before: string | null } | null>(null);
  // A divider the user just placed opens straight into rename mode.
  const [renamingId, setRenamingId] = useState<string | null>(null);

  const sections = sectionsOf(group);
  const dividerCount = sections.length - 1;
  const imageCount = group.imageNames.length;
  const summary =
    (imageCount === 0 ? "empty" : `${imageCount} image${imageCount === 1 ? "" : "s"}`)
    + (dividerCount > 0 ? ` · ${dividerCount} subgroup${dividerCount === 1 ? "" : "s"}` : "");

  const placeDivider = (splitBeforeImage?: string) => {
    const id = onAddDivider(splitBeforeImage ? { splitBeforeImage } : undefined);
    if (!id) return;
    setCollapsed(false);   // a divider placed into a collapsed row would be invisible
    setRenamingId(id);
  };

  /** Drag handlers for a whole section (before = null → append to it) or
   *  for one tile in it (before = that tile → insert in front of it). */
  const dropHandlers = (dividerId: string | null, before: string | null) => {
    const section = dividerId ?? "";
    return {
      onDragOver: (e: ReactDragEvent) => {
        if (!isImageDrag(e)) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect = "move";
        // dragover fires continuously — bail out of the re-render when the
        // target hasn't changed.
        setDropAt((cur) => (cur && cur.section === section && cur.before === before ? cur : { section, before }));
      },
      onDragLeave: (e: ReactDragEvent) => {
        // Crossing onto a child of this same target isn't leaving it (the
        // child's own dragover re-targets anyway; this just avoids flicker).
        if (e.relatedTarget instanceof Node && e.currentTarget.contains(e.relatedTarget)) return;
        setDropAt(null);
      },
      onDrop: (e: ReactDragEvent) => {
        if (!isImageDrag(e)) return;
        e.preventDefault();
        e.stopPropagation();
        setDropAt(null);
        setDragOver(false);
        const imgName = e.dataTransfer.getData(IMAGE_DRAG_MIME);
        // A drop on the section itself (not on a tile) is its END zone: it
        // appends, and — via atEnd — also moves an image already in this
        // section to the end, which is the only way to do that.
        if (imgName) onMoveImage(imgName, { dividerId, beforeImage: before, atEnd: before === null });
      },
    };
  };

  return (
    <div
      // Expanded rows carry their vertical padding inside the sections
      // instead, so a drop anywhere in a section's column — right up to the
      // row's edges — lands in that section rather than the row fallback.
      className={`flex items-center gap-2 border-t ${collapsed ? "px-3 py-1.5" : "pl-3"}`}
      style={{
        borderColor: "var(--c-border)",
        backgroundColor: dragOver ? "rgba(33,150,243,0.08)" : undefined,
        minHeight: collapsed ? 30 : 56,
      }}
      onDragOver={(e) => {
        if (isImageDrag(e)) {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const imgName = e.dataTransfer.getData(IMAGE_DRAG_MIME);
        // Name area / collapsed row → the group's unsectioned area.
        if (imgName) onMoveImage(imgName);
      }}
    >
      {/* Collapse toggle + group name + add-divider + delete */}
      <div className="flex items-center gap-0.5 flex-none" style={{ minWidth: 80 }}>
        <IconButton
          size="small"
          onClick={() => setCollapsed(!collapsed)}
          sx={{ p: 0.25 }}
          title={collapsed ? "Expand group" : "Collapse group"}
        >
          {collapsed ? <ExpandMoreIcon sx={{ fontSize: 14 }} /> : <ExpandLessIcon sx={{ fontSize: 14 }} />}
        </IconButton>
        <EditableName value={group.name} onChange={onRename} />
        <IconButton
          size="small"
          onClick={() => placeDivider()}
          sx={{ p: 0.25, opacity: 0.5, "&:hover": { opacity: 1 } }}
          title="Add a subgroup (or hover the gap between two images and click + to start one there)"
        >
          <CropSquareIcon sx={{ fontSize: 12 }} />
        </IconButton>
        <IconButton
          size="small"
          onClick={onDelete}
          sx={{ p: 0.25, opacity: 0.5, "&:hover": { opacity: 1 } }}
          title="Delete group (images return to timeline)"
        >
          <DeleteOutlineIcon sx={{ fontSize: 12 }} />
        </IconButton>
      </div>

      {/* Group images by section — or, when collapsed, a one-line summary */}
      {collapsed ? (
        <span className="text-[9px] flex-1" style={{ color: "var(--c-text-dim)" }}>
          {summary}
        </span>
      ) : (
        <div className="flex items-stretch overflow-x-auto flex-1 self-stretch">
          {sections.map((sec, i) => {
            const sectionKey = sec.dividerId ?? "";
            const tiles = sec.imageNames.filter((n) => loadedImages[n]);
            const over = dropAt?.section === sectionKey;
            // With subgroups present, an empty unsectioned area shrinks to a
            // slim pad: no hint text, but still a drop target, so an image
            // can be dragged back out of a subgroup.
            const slimPad = sec.dividerId === null && dividerCount > 0 && tiles.length === 0;
            const isSub = sec.dividerId !== null;
            const last = i === sections.length - 1;
            const content = tiles.length === 0 ? (
              !slimPad && (
                <span className="text-[9px] italic whitespace-nowrap px-1" style={{ color: "var(--c-text-dim)" }}>
                  Drag images here
                </span>
              )
            ) : (
              tiles.map((name, j) => (
                <div
                  key={name}
                  className="relative flex-none flex items-center self-stretch"
                  {...dropHandlers(sec.dividerId, name)}
                >
                  {/* The gap before every tile but the first doubles as the
                      split handle — so it's the inter-tile spacing too. */}
                  {j > 0 && <SplitHandle onSplit={() => placeDivider(name)} />}
                  {over && dropAt?.before === name && (
                    <span
                      aria-hidden
                      className="pointer-events-none absolute rounded-full"
                      style={{ left: j > 0 ? 3 : -3, top: 6, bottom: 6, width: 2, backgroundColor: "var(--c-accent)" }}
                    />
                  )}
                  <ImageTile
                    img={loadedImages[name]}
                    usage={usage.get(name)}
                    {...tileUsage}
                    isSelected={selectedImage === name}
                    onRemove={() => onRemoveImage(name)}
                  />
                </div>
              ))
            );
            const endBar = over && dropAt?.before === null && tiles.length > 0 && (
              <span
                aria-hidden
                className="pointer-events-none flex-none self-stretch rounded-full"
                style={{ width: 2, margin: "6px 0 6px 4px", backgroundColor: "var(--c-accent)" }}
              />
            );
            return (
              <div
                key={isSub ? `d:${sec.dividerId}` : "u"}
                className="flex items-center pl-1 rounded"
                style={{
                  // Room above a subgroup box for its name tag on the border.
                  paddingTop: isSub ? 12 : 6,
                  paddingBottom: 6,
                  // A real trailing drop zone (appending to the end of a
                  // middle subgroup used to be a ~4px target), and the last
                  // section also owns the row's right gutter, so a drop
                  // there lands in it rather than in the unsectioned area.
                  // A subgroup box carries its trailing zone inside.
                  paddingRight: (isSub ? 4 : tiles.length > 0 ? 24 : 4) + (last ? 12 : 0),
                  // The last section soaks up the leftover width, so a drop
                  // to the right of everything appends to the LAST section
                  // rather than falling through to the row (=> unsectioned).
                  flex: last ? "1 0 auto" : "0 0 auto",
                  minWidth: slimPad ? 12 : undefined,
                  backgroundColor: !isSub && over ? "rgba(33,150,243,0.08)" : undefined,
                  transition: "background-color 0.1s",
                }}
                title={slimPad ? "Drop here to move an image out of its subgroup" : undefined}
                {...dropHandlers(sec.dividerId, null)}
              >
                {isSub ? (
                  <div
                    data-testid="subgroup-box"
                    className="group/subgroup flex flex-col rounded-md"
                    style={{
                      border: `1px solid ${over ? "var(--c-accent)" : "rgba(79,195,247,0.5)"}`,
                      backgroundColor: over ? "rgba(33,150,243,0.12)" : "rgba(79,195,247,0.05)",
                      padding: "0 4px 4px 6px",
                      minWidth: 64,
                      transition: "background-color 0.1s, border-color 0.1s",
                    }}
                  >
                    <SubgroupTag
                      name={sec.name ?? ""}
                      autoEdit={renamingId === sec.dividerId}
                      onRename={(name) => onRenameDivider(sec.dividerId!, name)}
                      onDelete={() => onDeleteDivider(sec.dividerId!)}
                      onEditEnd={() => setRenamingId(null)}
                    />
                    <div className="flex items-center" style={{ paddingRight: tiles.length > 0 ? 18 : 0, minHeight: 28 }}>
                      {content}
                      {endBar}
                    </div>
                  </div>
                ) : (
                  <>
                    {content}
                    {endBar}
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
