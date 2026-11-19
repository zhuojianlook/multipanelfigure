/* ──────────────────────────────────────────────────────────
   DocumentTabs — a stable tab strip across the top: the Collage
   Assembly plus one tab per open .mpf document. Switching never
   reorders tabs; dragging a document tab left / right does. Each doc
   tab has a close (×) button; a + button opens a new blank document.

   All navigation routes through projectNav so unsaved builder
   changes are guarded (Save / Don't save / Cancel).

   Drag-to-reorder uses POINTER events with pointer capture, not HTML5
   drag-and-drop: a native drag session that ends off a drop target
   soft-locks the Tauri WKWebView (0.1.409), and a tab strip has no
   use for a drag image or OS drop targets. Reordering is order-only
   (collageStore.docReorder) — no document is loaded, saved or switched,
   so it needs no doc-op serialization.
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { Box, Tooltip, CircularProgress, IconButton, Menu, MenuItem } from "@mui/material";
import ViewModuleIcon from "@mui/icons-material/ViewModule";
import DescriptionIcon from "@mui/icons-material/Description";
import InsightsIcon from "@mui/icons-material/Insights";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import SaveIcon from "@mui/icons-material/Save";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import InfoIcon from "@mui/icons-material/Info";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { useAnalysisWorkspaceStore, untitledAnalysisDocIds } from "../../store/analysisWorkspaceStore";
import { enterCollage, enterAnalysis, switchToDocument, newBlankDoc, closeDoc, saveDocument } from "../../utils/projectNav";
import { getReopenPref, setReopenPref } from "../../utils/sessionRestore";
import { dropSlot, moveToSlot } from "../../utils/tabReorder";
import { RecordAppButton, DEV_OPTIONS_KEY } from "./Toolbar";

/** Pixels the pointer must travel before a press on a tab becomes a drag
 *  (below it, the press is an ordinary click / double-click). */
const DRAG_THRESHOLD_PX = 5;
/** Within this many px of the strip's edge, a drag scrolls the strip. */
const EDGE_SCROLL_PX = 28;
const TAB_ACCENT = "#4FC3F7";

/* Help menu — lives in the always-visible tab bar (same level as the Record
   button) so it's reachable in every mode. "About" opens the toolbar's About
   dialog via a window event; the developer-options toggle is self-contained
   (localStorage + the broadcast event the Record button listens for). */
function HelpMenuButton() {
  const [anchor, setAnchor] = useState<null | HTMLElement>(null);
  const [devOn, setDevOn] = useState<boolean>(() => {
    try { return localStorage.getItem(DEV_OPTIONS_KEY) === "1"; } catch { return false; }
  });
  // "Re-open tabs on relaunch" — opt-in persistence of the open .mpf tabs
  // (handled by sessionRestore). Self-contained, like the dev-options toggle.
  const [reopenOn, setReopenOn] = useState<boolean>(() => getReopenPref());
  return (
    <>
      <Tooltip title="Help">
        <IconButton size="small" onClick={(e) => setAnchor(e.currentTarget)} sx={{ p: 0.5 }}>
          <HelpOutlineIcon sx={{ fontSize: 18, color: "var(--c-text-dim)" }} />
        </IconButton>
      </Tooltip>
      <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)}>
        <MenuItem onClick={() => { window.dispatchEvent(new CustomEvent("mpfig:open-about")); setAnchor(null); }}>
          <InfoIcon sx={{ mr: 1, fontSize: 18 }} /> About
        </MenuItem>
        <MenuItem
          onClick={(e) => {
            e.preventDefault();
            const next = !devOn;
            setDevOn(next);
            try { localStorage.setItem(DEV_OPTIONS_KEY, next ? "1" : "0"); } catch { /* ignore */ }
            window.dispatchEvent(new CustomEvent("mpfig:dev-options-changed", { detail: { enabled: next } }));
          }}
        >
          <Box component="span" sx={{ mr: 1, width: 18, display: "inline-flex", justifyContent: "center" }}>
            {devOn ? "☑" : "☐"}
          </Box>
          Enable developer options
        </MenuItem>
        <MenuItem
          onClick={(e) => {
            e.preventDefault();
            const next = !reopenOn;
            setReopenOn(next);
            setReopenPref(next);
          }}
        >
          <Box component="span" sx={{ mr: 1, width: 18, display: "inline-flex", justifyContent: "center" }}>
            {reopenOn ? "☑" : "☐"}
          </Box>
          Re-open tabs on relaunch
        </MenuItem>
      </Menu>
    </>
  );
}

export function DocumentTabs() {
  const mode = useCollageStore((s) => s.mode);
  const openDocs = useCollageStore((s) => s.openDocs);
  const activeDocId = useCollageStore((s) => s.activeDocId);
  const snapshotDirtyDocIds = useCollageStore((s) => s.snapshotDirtyDocIds);
  const docRename = useCollageStore((s) => s.docRename);
  const unsaved = useFigureStore((s) => s.unsaved);
  // Untitled documents holding Analysis work (only a file can save it).
  // Saved documents get no analysis dot: their analysis auto-saves. A string
  // so the strip re-renders only when the set changes, not on every graph edit.
  const untitledAnalysis = useAnalysisWorkspaceStore((s) => untitledAnalysisDocIds(s, openDocs).join("\n"));

  // Inline tab rename: id of the tab being renamed + its draft text.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  // id of the tab currently being saved (shows a spinner on its save button).
  const [savingId, setSavingId] = useState<string | null>(null);
  const commitRename = () => {
    if (editingId) docRename(editingId, editValue);
    setEditingId(null);
  };

  // ── Drag to reorder ──
  // The press being tracked (a ref: every pointermove reads the latest).
  // `active` flips once the pointer has moved DRAG_THRESHOLD_PX sideways;
  // until then the press is an ordinary click / double-click.
  const pressRef = useRef<{ id: string; pointerId: number; x0: number; el: HTMLElement; active: boolean } | null>(null);
  // Removes the window listeners of the press in flight.
  const unlistenRef = useRef<(() => void) | null>(null);
  // Set when a drag ends so the click the browser fires right after the
  // pointerup doesn't also switch to the dragged tab. Cleared by the next
  // press, so a release that produces no click can't swallow a later one.
  const swallowClickRef = useRef(false);
  const tabEls = useRef(new Map<string, HTMLElement>());
  const stripRef = useRef<HTMLDivElement | null>(null);
  // What the strip draws while dragging: which tab, and the slot it would
  // land in (null = nowhere new: dropping leaves the order as it is).
  const [drag, setDrag] = useState<{ id: string; slot: number | null } | null>(null);

  const currentIds = () => useCollageStore.getState().openDocs.map((d) => d.id);
  const slotAt = (clientX: number): number => {
    const mids = currentIds().map((id) => {
      const r = tabEls.current.get(id)?.getBoundingClientRect();
      return r ? r.left + r.width / 2 : Number.NEGATIVE_INFINITY;
    });
    return dropSlot(mids, clientX);
  };

  /** Finish the press in flight: commit the move (pointer released) or
   *  drop it (Esc, window lost, cancelled). Idempotent. */
  const endPress = (commit: boolean, clientX?: number) => {
    unlistenRef.current?.();
    unlistenRef.current = null;
    const p = pressRef.current;
    pressRef.current = null;
    if (!p) return;
    if (p.active) {
      swallowClickRef.current = true;
      document.body.style.removeProperty("cursor");
      try { if (p.el.hasPointerCapture?.(p.pointerId)) p.el.releasePointerCapture(p.pointerId); } catch { /* ignore */ }
      if (commit && clientX !== undefined) {
        const ids = currentIds();
        const next = moveToSlot(ids, p.id, slotAt(clientX));
        if (next !== ids) useCollageStore.getState().docReorder([...next]);
      }
      setDrag(null);
    }
  };
  const endPressRef = useRef(endPress);
  endPressRef.current = endPress;

  // Esc cancels a drag in progress; losing the window ends it too (the
  // pointerup may never arrive).
  const dragging = drag !== null;
  useEffect(() => {
    if (!dragging) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault(); e.stopPropagation();
      endPressRef.current(false);
    };
    const onBlur = () => endPressRef.current(false);
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("blur", onBlur);
    };
  }, [dragging]);

  // Unmounting mid-press: drop the listeners and the grabbing cursor.
  useEffect(() => () => endPressRef.current(false), []);

  const onTabPointerDown = (docId: string, editing: boolean) => (e: ReactPointerEvent<HTMLElement>) => {
    swallowClickRef.current = false;
    if (pressRef.current?.active) return;   // a drag is already running
    endPress(false);                         // a stale, never-released press
    if (e.button !== 0 || editing) return;
    // Not from the tab's own buttons (save / close): those are clicks.
    if ((e.target as HTMLElement).closest?.("[data-tab-action]")) return;
    const el = e.currentTarget;
    const pointerId = e.pointerId;
    pressRef.current = { id: docId, pointerId, x0: e.clientX, el, active: false };

    // Window-level, so a fast flick that leaves the tab in one move still
    // registers. Pointer capture is taken only once the press BECOMES a
    // drag: taking it on press would retarget the double-click that
    // renames a tab away from the name.
    const onMove = (ev: PointerEvent) => {
      const p = pressRef.current;
      if (!p || ev.pointerId !== pointerId) return;
      if (!(ev.buttons & 1)) { endPress(false); return; }   // released unseen
      if (!p.active) {
        if (Math.abs(ev.clientX - p.x0) < DRAG_THRESHOLD_PX) return;
        p.active = true;
        try { el.setPointerCapture(pointerId); } catch { /* ignore */ }
        document.body.style.cursor = "grabbing";
      }
      ev.preventDefault();
      // Scroll an overflowing strip when dragging near its ends.
      const strip = stripRef.current;
      if (strip && strip.scrollWidth > strip.clientWidth) {
        const r = strip.getBoundingClientRect();
        if (ev.clientX < r.left + EDGE_SCROLL_PX) strip.scrollLeft -= 12;
        else if (ev.clientX > r.right - EDGE_SCROLL_PX) strip.scrollLeft += 12;
      }
      const ids = currentIds();
      const slot = slotAt(ev.clientX);
      const nextSlot = moveToSlot(ids, p.id, slot) !== ids ? slot : null;
      setDrag((d) => (d && d.id === p.id && d.slot === nextSlot ? d : { id: p.id, slot: nextSlot }));
    };
    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      endPress(true, ev.clientX);
    };
    const onCancel = (ev: PointerEvent) => {
      if (ev.pointerId !== pointerId) return;
      endPress(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
    unlistenRef.current = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onCancel);
    };
  };

  // Seed tabs once from persisted collage figure items, so figures that
  // survived a reload still appear as openable document tabs. Only adds
  // missing paths; never reorders or removes.
  const seeded = useRef(false);
  useEffect(() => {
    if (seeded.current) return;
    seeded.current = true;
    const cs = useCollageStore.getState();
    const cur = useFigureStore.getState().currentProjectPath;
    // If the builder already has a project loaded, reflect it on the
    // initial Untitled tab rather than leaving a stale "Untitled".
    if (cur && cs.activeDocId) cs.docSetPath(cs.activeDocId, cur);
    for (const it of cs.items) {
      if (it.kind === "figure" && it.projectPath) cs.docEnsure(it.projectPath);
    }
  }, []);

  const tabSx = (active: boolean) => ({
    display: "flex",
    alignItems: "center",
    gap: 0.5,
    pl: 1.25,
    pr: 0.5,
    py: 0.5,
    fontSize: "0.72rem",
    lineHeight: 1.2,
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
    maxWidth: 200,
    borderRight: "1px solid var(--c-border)",
    color: active ? "var(--c-text)" : "var(--c-text-dim)",
    backgroundColor: active ? "var(--c-bg)" : "transparent",
    borderBottom: active ? "2px solid #4FC3F7" : "2px solid transparent",
    "&:hover": { backgroundColor: "var(--c-bg)" },
    userSelect: "none" as const,
  });

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "stretch",
        height: 30,
        flexShrink: 0,
        backgroundColor: "var(--c-surface)",
        borderBottom: "1px solid var(--c-border)",
      }}
    >
     {/* Scrollable tabs region (tabs can overflow without pushing the
         pinned controls off-screen). */}
     <Box
      ref={stripRef}
      sx={{
        display: "flex",
        alignItems: "stretch",
        flex: 1,
        minWidth: 0,
        overflowX: "auto",
        overflowY: "hidden",
        "&::-webkit-scrollbar": { height: 4 },
        // Every tab sets cursor: pointer, which beats the body's inherited
        // "grabbing" — override it for the strip while a drag runs.
        ...(drag ? { "&, & *": { cursor: "grabbing !important" } } : {}),
      }}
     >
      {/* Collage tab — always present, leftmost. */}
      <Tooltip title="Collage Assembly — arrange figures on a shared canvas">
        <Box sx={{ ...tabSx(mode === "collage"), pr: 1.25 }} onClick={() => void enterCollage()}>
          <ViewModuleIcon sx={{ fontSize: 14 }} />
          Collage
        </Box>
      </Tooltip>

      {/* Analysis tab — always present. Build data/image workflows; pulls
          sources from open MPFs. Available even with no data. */}
      <Tooltip title="Analysis — build data + image-analysis workflows">
        <Box sx={{ ...tabSx(mode === "analysis"), pr: 1.25 }} onClick={() => void enterAnalysis()}>
          <InsightsIcon sx={{ fontSize: 14 }} />
          Analysis
        </Box>
      </Tooltip>

      {/* One tab per open .mpf, in stable order (the user's drag order). */}
      {openDocs.map((doc, di) => {
        const active = mode === "builder" && activeDocId === doc.id;
        // Dirty when: the live active doc has unsaved edits, OR this tab
        // holds unsaved edits parked in an in-memory snapshot (the user
        // switched away from it without saving), OR it is Untitled and holds
        // Analysis work.
        const dirty = (activeDocId === doc.id && unsaved)
          || snapshotDirtyDocIds.includes(doc.id)
          || (!doc.path && untitledAnalysis.split("\n").includes(doc.id));
        const editing = editingId === doc.id;
        const dragged = drag?.id === doc.id;
        // Insertion line: on this tab's left edge for its own slot, and on
        // the last tab's right edge for the slot after it.
        const lineLeft = drag?.slot === di;
        const lineRight = drag?.slot === openDocs.length && di === openDocs.length - 1;
        return (
          <Tooltip
            key={doc.id}
            title={editing || drag ? "" : `${doc.path || "Unsaved document"} — double-click to rename, drag to reorder`}
          >
            <Box
              ref={(el: HTMLElement | null) => {
                if (el) tabEls.current.set(doc.id, el);
                else tabEls.current.delete(doc.id);
              }}
              data-doc-tab={doc.id}
              data-dragging={dragged ? "true" : undefined}
              data-drop-line={lineLeft ? "left" : lineRight ? "right" : undefined}
              sx={{
                ...tabSx(active),
                ...(dragged ? { opacity: 0.55 } : {}),
                ...(lineLeft || lineRight
                  ? { boxShadow: `inset ${lineLeft ? 2 : -2}px 0 0 ${TAB_ACCENT}` }
                  : {}),
              }}
              onPointerDown={onTabPointerDown(doc.id, editing)}
              onClick={() => {
                if (swallowClickRef.current) { swallowClickRef.current = false; return; }
                if (!editing) void switchToDocument(doc.id);
              }}
            >
              <DescriptionIcon sx={{ fontSize: 14 }} />
              {editing ? (
                <Box
                  component="input"
                  autoFocus
                  value={editValue}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditValue(e.target.value)}
                  onMouseDown={(e: React.MouseEvent) => e.stopPropagation()}
                  onClick={(e: React.MouseEvent) => e.stopPropagation()}
                  onFocus={(e: React.FocusEvent<HTMLInputElement>) => e.target.select()}
                  onBlur={commitRename}
                  onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                    if (e.key === "Enter") { e.preventDefault(); commitRename(); }
                    else if (e.key === "Escape") { e.preventDefault(); setEditingId(null); }
                  }}
                  sx={{
                    width: 110, fontSize: "0.72rem", lineHeight: 1.2, px: 0.25, py: 0,
                    color: "var(--c-text)", backgroundColor: "var(--c-bg)",
                    border: "1px solid #4FC3F7", borderRadius: "3px", outline: "none",
                  }}
                />
              ) : (
                <Box
                  component="span"
                  sx={{ overflow: "hidden", textOverflow: "ellipsis", maxWidth: 130 }}
                  onDoubleClick={(e) => { e.stopPropagation(); setEditValue(doc.name); setEditingId(doc.id); }}
                >
                  {doc.name}
                </Box>
              )}
              {/* Unsaved indicator + Save button in one: amber save icon means
                  "unsaved — click to save". Saved tabs show nothing here.
                  Overwrites the file in place if the tab has one; an Untitled
                  tab prompts for a location (defaulting to the tab's name). */}
              {dirty && (
                <Box
                  component="span"
                  data-tab-action="save"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (savingId) return;
                    setSavingId(doc.id);
                    void saveDocument(doc.id).finally(() => setSavingId(null));
                  }}
                  title={doc.path
                    ? `Save changes — overwrites ${doc.path.split(/[\\/]/).pop()}`
                    : "Save… — choose a location for this new project"}
                  sx={{
                    display: "flex", alignItems: "center", borderRadius: "50%",
                    ml: 0.25, p: "1px", color: "#FFB74D", cursor: "pointer",
                    "&:hover": { backgroundColor: "rgba(255,183,77,0.22)" },
                  }}
                >
                  {savingId === doc.id
                    ? <CircularProgress size={11} sx={{ color: "#FFB74D" }} />
                    : <SaveIcon sx={{ fontSize: 13 }} />}
                </Box>
              )}
              {/* Close button */}
              <Box
                component="span"
                data-tab-action="close"
                onClick={(e) => { e.stopPropagation(); void closeDoc(doc.id); }}
                title="Close document"
                sx={{
                  display: "flex", alignItems: "center", borderRadius: "50%",
                  ml: 0.25, p: "1px",
                  "&:hover": { backgroundColor: "rgba(255,255,255,0.15)" },
                }}
              >
                <CloseIcon sx={{ fontSize: 13 }} />
              </Box>
            </Box>
          </Tooltip>
        );
      })}

      {/* New document. */}
      <Tooltip title="New figure (opens a new tab)">
        <Box
          onClick={() => void newBlankDoc()}
          sx={{
            display: "flex", alignItems: "center", px: 0.75, cursor: "pointer",
            color: "var(--c-text-dim)",
            "&:hover": { backgroundColor: "var(--c-bg)", color: "var(--c-text)" },
          }}
        >
          <AddIcon sx={{ fontSize: 16 }} />
        </Box>
      </Tooltip>
     </Box>{/* end scrollable tabs region */}

     {/* Pinned right, never scrolls off, present in every mode: the Record
         button (only when dev options are on — handled inside RecordAppButton)
         and the Help menu, both on this one always-visible top level. */}
     <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, px: 0.5, flexShrink: 0 }}>
       <RecordAppButton />
       <HelpMenuButton />
     </Box>
    </Box>
  );
}
