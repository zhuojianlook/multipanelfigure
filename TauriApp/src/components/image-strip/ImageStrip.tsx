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
import BorderVerticalIcon from "@mui/icons-material/BorderVertical";
import { useFigureStore } from "../../store/figureStore";
import type { LoadedImage } from "../../store/figureStore";
import type { ImageGroup } from "../../api/types";
import { sectionsOf, type SectionTarget } from "../../utils/imageGroups";
import { imageUsage, type ImageUsage } from "../../utils/figures";
import { confirm } from "../shared/ConfirmDialog";
import type { DragEvent as ReactDragEvent } from "react";
import { ImageLightbox } from "./ImageLightbox";
import { resolveTileClick, tileClickMemory } from "./lightboxLogic";

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
 *  several figure tabs, so "used" has two flavours:
 *    - in THIS figure → solid green border (as before figure tabs);
 *    - only in OTHER figures → dashed accent border;
 *  and, once the project has more than one figure, a corner pill with the
 *  number of figures that use it (filled = includes this one). The tooltip
 *  names them. A single-figure project looks exactly as it always did. */
interface TileUsageProps {
  usage: ImageUsage | undefined;
  /** The project has more than one figure tab. */
  multiFigure: boolean;
  currentFigureName: string;
}

function usageLines(u: ImageUsage | undefined, currentFigureName: string): string[] {
  if (!u) return [];
  const panels = (n: number) => `${n} panel${n === 1 ? "" : "s"}`;
  const parts: string[] = [];
  if (u.inCurrent > 0) parts.push(`${currentFigureName} (this figure, ${panels(u.inCurrent)})`);
  for (const o of u.others) parts.push(`${o.name} (${panels(o.count)})`);
  return parts;
}

function ImageTile({ img, usage, multiFigure, currentFigureName, isSelected, onRemove }: TileUsageProps & {
  img: LoadedImage;
  isSelected: boolean;
  onRemove: () => void;
}) {
  // Lightbox state is local to the tile: nothing else needs to know a
  // preview is open, so no global store change.
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const inCurrent = (usage?.inCurrent ?? 0) > 0;
  const inOthers = (usage?.others.length ?? 0) > 0;
  const figureCount = (inCurrent ? 1 : 0) + (usage?.others.length ?? 0);
  const where = usageLines(usage, currentFigureName);
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
          borderColor: isSelected ? "#2196f3" : inCurrent ? "var(--c-green)" : inOthers ? "var(--c-accent)" : "var(--c-border)",
          backgroundColor: isSelected ? "rgba(33,150,243,0.15)" : "var(--c-surface)",
          borderWidth: isSelected ? 3 : (inCurrent || inOthers) ? 2 : 1,
          borderStyle: !isSelected && !inCurrent && inOthers ? "dashed" : "solid",
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
        {multiFigure && figureCount > 0 && (
          <div
            aria-label={`Used in ${figureCount} figure${figureCount === 1 ? "" : "s"}`}
            className="absolute -bottom-1.5 -left-1.5 min-w-[14px] h-[14px] px-[3px] rounded-full flex items-center justify-center text-[8px] font-bold leading-none pointer-events-none"
            style={inCurrent
              ? { backgroundColor: "var(--c-green)", color: "#ffffff" }
              : { backgroundColor: "var(--c-surface)", color: "var(--c-accent)", border: "1px solid var(--c-accent)" }}
          >
            {figureCount}
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
function EditableName({ value, onChange, vertical = false, autoEdit = false, onEditEnd }: {
  value: string;
  onChange: (v: string) => void;
  /** Run the label up the side of a subgroup divider's rule (rotated text,
   *  ellipsized) instead of the horizontal group-name style. */
  vertical?: boolean;
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
    return vertical ? (
      <span
        className="text-[9px] font-semibold cursor-pointer py-1 rounded hover:bg-white/10"
        style={{
          color: "var(--c-text-dim)",
          // Reads bottom-to-top along the rule, like the panel grid's
          // rotated row labels.
          writingMode: "vertical-rl",
          transform: "rotate(180deg)",
          // 72px still fits inside a tile-height row (it doesn't grow the
          // drawer) and shows ~13 characters — enough for "Subgroup 10+";
          // longer names ellipsize, with the full name in the tooltip.
          maxHeight: 72,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
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
      maxLength={vertical ? 60 : undefined}
      className="text-[10px] font-semibold px-1 rounded outline-none"
      style={{ backgroundColor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-accent)", width: vertical ? 96 : 120 }}
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
  const [groupsExpanded, setGroupsExpanded] = useState(true);
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
  const tileUsage = {
    multiFigure: figures.length > 1,
    currentFigureName: figures.find((f) => f.id === activeFigureId)?.name ?? "This figure",
  };
  // Removing an image clears it from EVERY figure that shows it — say so
  // first when that includes figures other than the one on screen.
  const requestRemove = async (name: string) => {
    const others = usage.get(name)?.others ?? [];
    if (others.length > 0) {
      const ok = await confirm({
        title: "Remove image from the project?",
        body: `"${name}" is also used in ${others.map((o) => o.name).join(", ")}. `
          + "Removing it clears it from every figure in this project.",
        confirmLabel: "Remove from all figures",
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
 *  Clicking it drops a subgroup divider into that gap: the tile on its
 *  right and every tile after it in the section become the new subgroup.
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
        title="Insert a subgroup divider here"
        aria-label="Insert a subgroup divider here"
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

/** A subgroup divider's header inside a group drawer: a slim vertical rule
 *  with the subgroup's name running up beside it (double-click to rename)
 *  and a hover-revealed delete. Deleting never loses images — they merge
 *  into the section above (see deleteDividerFromGroup). */
function DividerHeader({ name, autoEdit, onRename, onDelete, onEditEnd }: {
  name: string;
  autoEdit: boolean;
  onRename: (name: string) => void;
  onDelete: () => void;
  onEditEnd: () => void;
}) {
  return (
    // Named group (group/divider) so only hovering THIS header reveals its
    // delete — a plain `group` would also light up any ImageTile's own
    // group-hover controls if the scopes ever nested.
    <div className="group/divider relative flex-none flex items-stretch self-stretch mr-1.5">
      <div className="flex-none rounded-full" style={{ width: 2, backgroundColor: "var(--c-accent)", opacity: 0.7 }} />
      <div className="flex flex-col items-center justify-between pl-0.5" style={{ minWidth: 14 }}>
        <EditableName vertical value={name} onChange={onRename} autoEdit={autoEdit} onEditEnd={onEditEnd} />
        <button
          type="button"
          className="flex items-center justify-center rounded opacity-0 group-hover/divider:opacity-100 focus-visible:opacity-100 transition-opacity hover:bg-white/10"
          style={{ width: 14, height: 14, padding: 0, border: 0, background: "transparent", color: "var(--c-text-dim)", cursor: "pointer" }}
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="Delete divider (images move to the section above)"
          aria-label={`Delete divider ${name}`}
        >
          <DeleteOutlineIcon sx={{ fontSize: 12 }} />
        </button>
      </div>
    </div>
  );
}

/** A single group row with drag-and-drop.
 *
 *  Expanded, the drawer is one horizontal run of SECTIONS: the unsectioned
 *  images, then each subgroup divider (rule + name) followed by its images.
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
  const [collapsed, setCollapsed] = useState(false);
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
          onClick={() => setCollapsed((c) => !c)}
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
          title="Add a subgroup divider (or hover the gap between two images and click +)"
        >
          <BorderVerticalIcon sx={{ fontSize: 12 }} />
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
            return (
              <div
                key={sec.dividerId === null ? "u" : `d:${sec.dividerId}`}
                className="flex items-center py-1.5 pl-1 rounded"
                style={{
                  // A real trailing drop zone (appending to the end of a
                  // middle subgroup used to be a ~4px target), and the last
                  // section also owns the row's right gutter, so a drop
                  // there lands in it rather than in the unsectioned area.
                  paddingRight: (tiles.length > 0 ? 24 : 4) + (i === sections.length - 1 ? 12 : 0),
                  // The last section soaks up the leftover width, so a drop
                  // to the right of everything appends to the LAST section
                  // rather than falling through to the row (=> unsectioned).
                  flex: i === sections.length - 1 ? "1 0 auto" : "0 0 auto",
                  minWidth: slimPad ? 12 : undefined,
                  backgroundColor: over ? "rgba(33,150,243,0.08)" : undefined,
                  transition: "background-color 0.1s",
                }}
                title={slimPad ? "Drop here to move an image out of its subgroup" : undefined}
                {...dropHandlers(sec.dividerId, null)}
              >
                {sec.dividerId !== null && (
                  <DividerHeader
                    name={sec.name ?? ""}
                    autoEdit={renamingId === sec.dividerId}
                    onRename={(name) => onRenameDivider(sec.dividerId!, name)}
                    onDelete={() => onDeleteDivider(sec.dividerId!)}
                    onEditEnd={() => setRenamingId(null)}
                  />
                )}
                {tiles.length === 0 ? (
                  !slimPad && (
                    <span className="text-[9px] italic whitespace-nowrap" style={{ color: "var(--c-text-dim)" }}>
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
                )}
                {over && dropAt?.before === null && tiles.length > 0 && (
                  <span
                    aria-hidden
                    className="pointer-events-none flex-none self-stretch rounded-full"
                    style={{ width: 2, margin: "6px 0 6px 4px", backgroundColor: "var(--c-accent)" }}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
