/* ──────────────────────────────────────────────────────────
   ImageStrip — horizontal scrollable strip of loaded images.
   Click an image to select it, then click a panel to assign.
   Also supports HTML5 drag-and-drop.
   Includes collapsible media groups below the timeline.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect, useRef } from "react";
import { Tooltip, IconButton } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import { useFigureStore } from "../../store/figureStore";
import type { LoadedImage } from "../../store/figureStore";

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

/** Reusable image thumbnail tile */
function ImageTile({ img, inUse, isSelected, onRemove }: {
  img: LoadedImage;
  inUse: boolean;
  isSelected: boolean;
  onRemove: () => void;
}) {
  return (
    <Tooltip title={img.name} placement="bottom" arrow enterDelay={200}>
      <div
        className="relative flex-none flex flex-col items-center gap-0.5 rounded border p-1 group"
        style={{
          borderColor: isSelected ? "#2196f3" : inUse ? "var(--c-green)" : "var(--c-border)",
          backgroundColor: isSelected ? "rgba(33,150,243,0.15)" : "var(--c-surface)",
          borderWidth: isSelected ? 3 : inUse ? 2 : 1,
          cursor: "pointer",
          transition: "border-color 0.15s, background-color 0.15s",
        }}
        onClick={() => setSelectedImageName(isSelected ? null : img.name)}
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
          aria-label={`Remove image ${img.name}`}
          title="Remove image"
        >
          &times;
        </button>
      </div>
    </Tooltip>
  );
}

/** Inline editable group name */
function EditableName({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  const commit = () => {
    setEditing(false);
    if (draft.trim() && draft !== value) onChange(draft.trim());
    else setDraft(value);
  };

  if (!editing) {
    return (
      <span
        className="text-[10px] font-semibold cursor-pointer px-1 rounded hover:bg-white/10"
        style={{ color: "var(--c-text)" }}
        onDoubleClick={() => { setEditing(true); setTimeout(() => ref.current?.select(), 0); }}
        title="Double-click to rename"
      >
        {value}
      </span>
    );
  }

  return (
    <input
      ref={ref}
      className="text-[10px] font-semibold px-1 rounded outline-none"
      style={{ backgroundColor: "var(--c-surface)", color: "var(--c-text)", border: "1px solid var(--c-accent)", width: 120 }}
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") { setDraft(value); setEditing(false); } }}
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
  const selectedImage = useSelectedImage();
  const [groupsExpanded, setGroupsExpanded] = useState(true);
  const [timelineDragOver, setTimelineDragOver] = useState(false);

  const entries = Object.values(loadedImages);
  const groupedNames = new Set(imageGroups.flatMap(g => g.imageNames));
  const timelineEntries = entries.filter(img => !groupedNames.has(img.name));

  const usedNames = new Set<string>();
  if (config) {
    for (let r = 0; r < config.rows; r++) {
      for (let c = 0; c < config.cols; c++) {
        const name = config.panels[r]?.[c]?.image_name;
        if (name) usedNames.add(name);
      }
    }
  }

  if (entries.length === 0) {
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
        {timelineEntries.length === 0 ? (
          <span className="text-[10px]" style={{ color: "var(--c-text-dim)" }}>
            All images are in groups. Drag here to return to timeline.
          </span>
        ) : (
          timelineEntries.map((img) => (
            <ImageTile
              key={img.name}
              img={img}
              inUse={usedNames.has(img.name)}
              isSelected={selectedImage === img.name}
              onRemove={() => removeImage(img.name)}
            />
          ))
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
            usedNames={usedNames}
            selectedImage={selectedImage}
            onRename={(name) => renameImageGroup(group.id, name)}
            onDelete={() => deleteImageGroup(group.id)}
            onDropImage={(imgName) => moveImageToGroup(imgName, group.id)}
            onRemoveImage={removeImage}
          />
        ))}
      </div>
    </div>
  );
}

/** A single group row with drag-and-drop */
function GroupRow({ group, loadedImages, usedNames, selectedImage, onRename, onDelete, onDropImage, onRemoveImage }: {
  group: { id: string; name: string; imageNames: string[] };
  loadedImages: Record<string, LoadedImage>;
  usedNames: Set<string>;
  selectedImage: string | null;
  onRename: (name: string) => void;
  onDelete: () => void;
  onDropImage: (imageName: string) => void;
  onRemoveImage: (name: string) => Promise<void>;
}) {
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 border-t"
      style={{
        borderColor: "var(--c-border)",
        backgroundColor: dragOver ? "rgba(33,150,243,0.08)" : undefined,
        minHeight: 56,
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("application/x-image-name")) {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const imgName = e.dataTransfer.getData("application/x-image-name");
        if (imgName) onDropImage(imgName);
      }}
    >
      {/* Group name + delete */}
      <div className="flex items-center gap-0.5 flex-none" style={{ minWidth: 80 }}>
        <EditableName value={group.name} onChange={onRename} />
        <IconButton
          size="small"
          onClick={onDelete}
          sx={{ p: 0.25, opacity: 0.5, "&:hover": { opacity: 1 } }}
          title="Delete group (images return to timeline)"
        >
          <DeleteOutlineIcon sx={{ fontSize: 12 }} />
        </IconButton>
      </div>

      {/* Group images */}
      <div className="flex items-center gap-1.5 overflow-x-auto flex-1">
        {group.imageNames.length === 0 ? (
          <span className="text-[9px] italic" style={{ color: "var(--c-text-dim)" }}>
            Drag images here
          </span>
        ) : (
          group.imageNames.map((name) => {
            const img = loadedImages[name];
            if (!img) return null;
            return (
              <ImageTile
                key={name}
                img={img}
                inUse={usedNames.has(name)}
                isSelected={selectedImage === name}
                onRemove={() => onRemoveImage(name)}
              />
            );
          })
        )}
      </div>
    </div>
  );
}
