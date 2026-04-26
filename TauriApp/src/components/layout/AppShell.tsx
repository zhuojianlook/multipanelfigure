/* ──────────────────────────────────────────────────────────
   AppShell — top-level layout.
   Left sidebar (260 px) | center: toolbar + image strip +
   horizontal split (grid LEFT | preview RIGHT).
   ────────────────────────────────────────────────────────── */

import { useState, useCallback, useEffect } from "react";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { Sidebar } from "./Sidebar";
import { Toolbar } from "./Toolbar";
import { ImageStrip } from "../image-strip/ImageStrip";
import { PanelGrid } from "../grid/PanelGrid";
import { PreviewPane } from "../preview/PreviewPane";
import { CollageView } from "../collage/CollageView";
import { Alert } from "@mui/material";

function CenterPane({
  splitPct, dragging, onSplitterDown,
}: { splitPct: number; dragging: boolean; onSplitterDown: () => void }) {
  const mode = useCollageStore((s) => s.mode);
  if (mode === "collage") {
    return (
      <div className="flex flex-1 flex-col min-h-0 min-w-0">
        <CollageView />
      </div>
    );
  }
  return (
    <>
      {/* Image strip */}
      <ImageStrip />

      {/* Horizontal splitter: grid (left) | divider | preview (right) */}
      <div className="flex flex-1 min-h-0 min-w-0 relative">
        <div
          className="overflow-auto min-w-0"
          style={{ width: `${splitPct}%`, minWidth: 200 }}
        >
          <PanelGrid />
        </div>
        <div
          className="flex-none cursor-col-resize flex items-center justify-center"
          style={{ width: 8, backgroundColor: "var(--c-border)" }}
          onMouseDown={onSplitterDown}
        >
          <div
            className="rounded-full"
            style={{
              width: 4, height: 40,
              backgroundColor: dragging ? "var(--c-accent)" : "var(--c-text-dim)",
              opacity: 0.6,
            }}
          />
        </div>
        <div
          className="overflow-auto min-w-0"
          style={{ width: `${100 - splitPct}%` }}
        >
          <PreviewPane />
        </div>
      </div>
    </>
  );
}

export function AppShell() {
  const fetchConfig = useFigureStore((s) => s.fetchConfig);
  const fetchFonts = useFigureStore((s) => s.fetchFonts);
  const fetchImages = useFigureStore((s) => s.fetchImages);
  const config = useFigureStore((s) => s.config);
  const requestPreview = useFigureStore((s) => s.requestPreview);
  const apiError = useFigureStore((s) => s.apiError);
  // uploadImagesFromPaths used by Tauri native dialog; uploadImages for File objects

  useEffect(() => {
    const init = async () => {
      await fetchConfig();
      await fetchImages();
      fetchFonts();
      setTimeout(() => requestPreview(), 200);
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle OS file drops via HTML5 drag-drop (works with dragDropEnabled: false)
  const [fileDragOver, setFileDragOver] = useState(false);
  const handleGlobalDragOver = useCallback((e: React.DragEvent) => {
    // Only show drop zone for files from OS (not internal drags)
    if (e.dataTransfer.types.includes("Files")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      setFileDragOver(true);
    }
  }, []);
  const handleGlobalDragLeave = useCallback((e: React.DragEvent) => {
    // Only reset if leaving the window entirely
    if (e.relatedTarget === null || !(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setFileDragOver(false);
    }
  }, []);
  const handleGlobalDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setFileDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    const imageExts = new Set(["tif","tiff","png","jpg","jpeg","cr2","cr3","nef","arw","dng","orf","rw2","pef","raf","nd2","mp4","avi","mov","mkv","webm","wmv","flv","m4v","mpg","mpeg","3gp","ts","mts"]);
    const validFiles = files.filter(f => {
      const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
      return imageExts.has(ext);
    });
    if (validFiles.length > 0) {
      // Use the File objects directly (browser fetch path)
      useFigureStore.getState().uploadImages(validFiles);
    }
  }, []);

  // Horizontal split: grid (left) | preview (right)
  // splitPct is the % width for the grid pane
  const [splitPct, setSplitPct] = useState(45);
  const [dragging, setDragging] = useState(false);

  const onMouseDown = useCallback(() => setDragging(true), []);

  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!dragging) return;
      const container = e.currentTarget;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const pct = Math.min(75, Math.max(25, (x / rect.width) * 100));
      setSplitPct(pct);
    },
    [dragging],
  );

  const onMouseUp = useCallback(() => setDragging(false), []);

  if (!config) {
    return (
      <div
        className="flex h-screen w-screen items-center justify-center flex-col gap-4"
        style={{
          backgroundColor: "var(--c-bg)",
          color: "var(--c-text-dim)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 40, height: 40, border: "3px solid rgba(255,255,255,0.15)",
            borderTopColor: "rgba(255,255,255,0.6)", borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }} />
          <span style={{ fontSize: 16, fontWeight: 500, letterSpacing: 0.5 }}>
            Multi-Panel Figure Builder
          </span>
          <span style={{ fontSize: 12, opacity: 0.5 }}>
            Loading...
          </span>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  return (
    <div
      className="flex h-screen w-screen overflow-hidden select-none"
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onDragOver={handleGlobalDragOver}
      onDragLeave={handleGlobalDragLeave}
      onDrop={handleGlobalDrop}
    >
      {/* ── File Drop Overlay ──────────────────────────── */}
      {fileDragOver && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center pointer-events-none"
          style={{ backgroundColor: "rgba(33, 150, 243, 0.15)", border: "3px dashed #2196f3" }}>
          <div className="px-6 py-3 rounded-lg text-lg font-semibold"
            style={{ backgroundColor: "rgba(33, 150, 243, 0.9)", color: "#fff" }}>
            Drop files to add to timeline
          </div>
        </div>
      )}
      {/* ── API Error Banner ────────────────────────────── */}
      {apiError && (
        <Alert severity="error" sx={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 9999, borderRadius: 0 }}>
          {apiError}
        </Alert>
      )}

      {/* ── Sidebar ─────────────────────────────────────── */}
      <aside
        className="flex-none overflow-y-auto overflow-x-hidden border-r"
        style={{
          width: 260,
          minWidth: 220,
          backgroundColor: "var(--c-surface)",
          borderColor: "var(--c-border)",
        }}
      >
        <Sidebar />
      </aside>

      {/* ── Center area ─────────────────────────────────── */}
      <main className="flex flex-1 flex-col min-w-0 min-h-0">
        {/* Toolbar */}
        <Toolbar />

        {/* Mode-specific body. Builder = image strip + grid|preview split.
            Collage = single-pane CollageView. The Toolbar is shared so
            users can flip between modes from one place. */}
        <CenterPane
          splitPct={splitPct}
          dragging={dragging}
          onSplitterDown={onMouseDown}
        />
      </main>
    </div>
  );
}
