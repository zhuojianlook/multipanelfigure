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
import { DocumentTabs } from "./DocumentTabs";
import { ImageStrip } from "../image-strip/ImageStrip";
import { PanelGrid } from "../grid/PanelGrid";
import { PreviewPane } from "../preview/PreviewPane";
import { CollageView } from "../collage/CollageView";
import { AnalysisView } from "../analysis/AnalysisView";
import { hasUnsavedSnapshots } from "../../utils/projectNav";
import { maybeRestoreSession, persistOpenDocs, getReopenPref, setReopenPref } from "../../utils/sessionRestore";
import { Alert, CircularProgress } from "@mui/material";

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
  if (mode === "analysis") {
    return (
      <div className="flex flex-1 flex-col min-h-0 min-w-0">
        <AnalysisView />
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
  const mode = useCollageStore((s) => s.mode);
  // uploadImagesFromPaths used by Tauri native dialog; uploadImages for File objects

  // True while the app is saving-and-quitting. Drives a blocking overlay so the
  // user can't keep editing during the (potentially slow) per-file .mpf saves
  // between clicking "Save all & close"/"Overwrite" and the window actually
  // closing. Sits BELOW MUI dialogs (z 1300) so the per-file save prompts stay
  // clickable above it.
  const [closing, setClosing] = useState(false);

  useEffect(() => {
    const init = async () => {
      await fetchConfig();
      await fetchImages();
      fetchFonts();
      // Reopen last session's .mpf tabs when the user has opted in. When it
      // loads a project the builder already has a preview pending, so only
      // kick our own preview when nothing was restored.
      const restored = await maybeRestoreSession().catch(() => false);
      if (!restored) setTimeout(() => requestPreview(), 200);
    };
    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mirror the set of open .mpf tabs (paths + active doc) to localStorage on
  // every change, so "Re-open tabs on relaunch" can rebuild them next launch.
  // openDocs is session-only state, so we subscribe to the collage store and
  // persist whenever the doc set / active doc / a tab name actually changes
  // (collage canvas edits fire far more often — diff a compact key to skip
  // those). Persisted regardless of the preference; only restore honours it.
  useEffect(() => {
    const docKey = () => {
      const { openDocs, activeDocId } = useCollageStore.getState();
      return openDocs.map((d) => `${d.path ?? ""}::${d.name}`).join("||")
        + ">>" + (activeDocId ?? "");
    };
    // No initial write here — maybeRestoreSession reads a boot-time snapshot,
    // and an eager persist would clobber the saved session before restore runs
    // (and overwrite it with the pre-restore blank doc). We only persist once
    // the doc set / active doc / a tab name actually changes.
    let prevKey = docKey();
    const unsub = useCollageStore.subscribe(() => {
      const key = docKey();
      if (key !== prevKey) { prevKey = key; persistOpenDocs(); }
    });
    return unsub;
  }, []);

  // Warn on app close if there are unsaved changes — the builder figure
  // (configDirty) and/or collage content. Offers Save / Don't save /
  // Cancel. The collage auto-persists to local storage, but analysis
  // plots moved in only live there (no disk copy yet), so we still flag
  // collage content as at-risk on close.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    const hasUnsaved = () => {
      const dirtyBuilder = useFigureStore.getState().unsaved;
      const collageItems = useCollageStore.getState().items.length > 0;
      // Other open tabs may hold unsaved edits parked in in-memory snapshots.
      return dirtyBuilder || collageItems || hasUnsavedSnapshots();
    };
    (async () => {
      try {
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        const win = getCurrentWindow();
        // Quit the whole app. On macOS, destroying the last window does NOT
        // terminate the process (it lingers windowless), so the red × would
        // appear to "not exit". Exit the process explicitly instead; that also
        // fires Rust's RunEvent::Exit which kills the Python sidecar.
        const quit = async () => {
          try {
            const { exit } = await import("@tauri-apps/plugin-process");
            await exit(0);
          } catch {
            await win.destroy();
          }
        };
        const off = await win.onCloseRequested(async (event) => {
          // Always take control of the close so we can guarantee a real quit.
          event.preventDefault();
          if (hasUnsaved()) {
            const dirtyBuilder = useFigureStore.getState().unsaved;
            const collageItems = useCollageStore.getState().items.length > 0;
            const otherTabs = hasUnsavedSnapshots();
            // How many .mpf documents (current + parked tabs) are dirty — drives
            // singular/plural wording and whether there's anything to save.
            const dirtyDocs = (dirtyBuilder ? 1 : 0)
              + useCollageStore.getState().snapshotDirtyDocIds.length;
            const parts: string[] = [];
            if (dirtyBuilder && otherTabs) parts.push(`${dirtyDocs} open figures have unsaved changes`);
            else if (dirtyBuilder) parts.push("the current figure has unsaved changes");
            else if (otherTabs) parts.push(
              dirtyDocs > 1 ? `${dirtyDocs} open figures have unsaved changes`
                : "an open figure has unsaved changes");
            if (collageItems) parts.push("your collage has unsaved content");
            const { confirmThree } = await import("../shared/ConfirmDialog");
            const { saveAllOpenDocs } = await import("../../utils/projectNav");
            const choice = await confirmThree({
              title: "Save before closing?",
              body: `Before you close, ${parts.join(" and ")}.\n\n`
                + "Save all open figures now? You'll be asked where to save each "
                + "one. (The collage auto-saves to this app's local storage; export "
                + "it via Save Collage if you need a file.)",
              confirmLabel: "Save all & close",
              tertiaryLabel: "Close without saving",
              cancelLabel: "Cancel",
              // Inline toggle so the session-restore preference is discoverable
              // right where the user is thinking about their open tabs.
              checkbox: {
                label: "Re-open my open tabs next time I launch",
                checked: getReopenPref(),
                onChange: (v) => setReopenPref(v),
              },
            });
            if (choice === "cancel") return;            // stay open
            if (choice === "confirm") {
              // Block the UI while we save — saving an .mpf re-encodes every
              // image, which can take seconds, and we must not let the user keep
              // editing (or lose those edits) in the gap before the window
              // closes. The overlay sits below the per-file save prompts so
              // those stay interactive.
              setClosing(true);
              let saved = false;
              try {
                // Save every dirty .mpf (current + parked tabs), prompting for a
                // location per file. Any cancelled save aborts the close.
                saved = await saveAllOpenDocs({ promptForLocation: true });
              } catch (e) {
                console.error("[close] save-all failed", e);
              }
              if (!saved) { setClosing(false); return; } // cancelled/failed → stay
            }
          }
          // Keep (or show) the overlay through the quit so nothing is clickable
          // in the final beat before the window tears down.
          setClosing(true);
          await quit();
        });
        if (cancelled) off();
        else unlisten = off;
      } catch {
        // Browser preview — generic beforeunload prompt. Skipped when an
        // intentional reload was just requested (window.__mpfigAllowUnload).
        const handler = (e: BeforeUnloadEvent) => {
          const w = window as unknown as { __mpfigAllowUnload?: boolean };
          if (w.__mpfigAllowUnload) {
            w.__mpfigAllowUnload = false;
            return;
          }
          if (hasUnsaved()) {
            e.preventDefault();
            e.returnValue = "";
          }
        };
        window.addEventListener("beforeunload", handler);
        if (cancelled) window.removeEventListener("beforeunload", handler);
        else unlisten = () => window.removeEventListener("beforeunload", handler);
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
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
      className="flex flex-col h-screen w-screen overflow-hidden select-none"
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onDragOver={handleGlobalDragOver}
      onDragLeave={handleGlobalDragLeave}
      onDrop={handleGlobalDrop}
    >
      {/* ── Saving-and-closing overlay ──────────────────── */}
      {/* Shown from the moment the user confirms "Save all & close" until the
          window tears down. Blocks all interaction (so edits can't be made — or
          lost — mid-save) at z 1250, BELOW MUI dialogs (z 1300) so the per-file
          save prompts remain clickable above it. */}
      {closing && (
        <div
          className="fixed inset-0 flex items-center justify-center"
          style={{ zIndex: 1250, backgroundColor: "rgba(0,0,0,0.55)", cursor: "wait" }}
        >
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
            <CircularProgress size={40} sx={{ color: "#fff" }} />
            <span style={{ fontSize: 14, fontWeight: 500, color: "#fff", letterSpacing: 0.3 }}>
              Saving and closing…
            </span>
          </div>
        </div>
      )}
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

      {/* ── Document tabs ───────────────────────────────── */}
      {/* Full-width top bar so the tabs (Collage / Analysis / open .mpf) keep
          a fixed position across every mode — switching tabs never shifts the
          navigation, even when the left tools column changes. */}
      <DocumentTabs />

      {/* ── Workspace row: left tools column | center area ── */}
      <div className="flex flex-1 min-h-0 min-w-0">
        {/* Left tools column. Builder + Collage render their tools here; the
            Analysis workspace provides its own equivalent column (Sources)
            inside its canvas, kept at the same width + styling so the left
            column reads consistently and never collapses. */}
        {mode !== "analysis" && (
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
        )}

        {/* ── Center area ─────────────────────────────────── */}
        <main className="flex flex-1 flex-col min-w-0 min-h-0">
          {/* Toolbar — builder/collage actions. Always MOUNTED (it hides its
              own bar via display:none in Analysis mode) so the About dialog it
              hosts stays reachable from the always-visible Help menu in every
              mode; the Analysis workspace provides its own toolbar separately. */}
          <Toolbar />

          {/* Mode-specific body. Builder = image strip + grid|preview split.
              Collage = single-pane CollageView. Analysis = node-graph. */}
          <CenterPane
            splitPct={splitPct}
            dragging={dragging}
            onSplitterDown={onMouseDown}
          />
        </main>
      </div>
    </div>
  );
}
