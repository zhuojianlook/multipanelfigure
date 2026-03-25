/* ──────────────────────────────────────────────────────────
   ImageStrip — horizontal scrollable strip of loaded images.
   Click an image to select it, then click a panel to assign.
   Also supports HTML5 drag-and-drop.
   ────────────────────────────────────────────────────────── */

import { useState, useEffect } from "react";
import { Tooltip } from "@mui/material";
import { useFigureStore } from "../../store/figureStore";

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

export function ImageStrip() {
  const loadedImages = useFigureStore((s) => s.loadedImages);
  const config = useFigureStore((s) => s.config);
  const removeImage = useFigureStore((s) => s.removeImage);
  const selectedImage = useSelectedImage();

  const entries = Object.values(loadedImages);

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
    <div
      className="flex items-center gap-2 px-3 py-2 overflow-x-auto border-b flex-none"
      style={{ borderColor: "var(--c-border)", backgroundColor: "var(--c-bg)" }}
    >
      {entries.map((img) => {
        const inUse = usedNames.has(img.name);
        const isSelected = selectedImage === img.name;
        return (
          <Tooltip key={img.name} title={img.name} placement="bottom" arrow enterDelay={200}>
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
              onClick={(e) => { e.stopPropagation(); removeImage(img.name); }}
              aria-label={`Remove image ${img.name}`}
              title="Remove image"
            >
              &times;
            </button>
          </div>
          </Tooltip>
        );
      })}
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
  );
}
