/* ──────────────────────────────────────────────────────────
   DocumentTabs — a stable tab strip across the top: the Collage
   Assembly plus one tab per open .mpf document. Switching never
   reorders tabs. Each doc tab has a close (×) button; a + button
   opens a new blank document.

   All navigation routes through projectNav so unsaved builder
   changes are guarded (Save / Don't save / Cancel).
   ────────────────────────────────────────────────────────── */

import { useEffect, useRef, useState } from "react";
import { Box, Tooltip, CircularProgress } from "@mui/material";
import ViewModuleIcon from "@mui/icons-material/ViewModule";
import DescriptionIcon from "@mui/icons-material/Description";
import InsightsIcon from "@mui/icons-material/Insights";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import SaveIcon from "@mui/icons-material/Save";
import { useFigureStore } from "../../store/figureStore";
import { useCollageStore } from "../../store/collageStore";
import { enterCollage, enterAnalysis, switchToDocument, newBlankDoc, closeDoc, saveDocument } from "../../utils/projectNav";
import { RecordAppButton } from "./Toolbar";

export function DocumentTabs() {
  const mode = useCollageStore((s) => s.mode);
  const openDocs = useCollageStore((s) => s.openDocs);
  const activeDocId = useCollageStore((s) => s.activeDocId);
  const snapshotDirtyDocIds = useCollageStore((s) => s.snapshotDirtyDocIds);
  const docRename = useCollageStore((s) => s.docRename);
  const unsaved = useFigureStore((s) => s.unsaved);

  // Inline tab rename: id of the tab being renamed + its draft text.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  // id of the tab currently being saved (shows a spinner on its save button).
  const [savingId, setSavingId] = useState<string | null>(null);
  const commitRename = () => {
    if (editingId) docRename(editingId, editValue);
    setEditingId(null);
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
      sx={{
        display: "flex",
        alignItems: "stretch",
        flex: 1,
        minWidth: 0,
        overflowX: "auto",
        overflowY: "hidden",
        "&::-webkit-scrollbar": { height: 4 },
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

      {/* One tab per open .mpf, in stable order. */}
      {openDocs.map((doc) => {
        const active = mode === "builder" && activeDocId === doc.id;
        // Dirty when: the live active doc has unsaved edits, OR this tab
        // holds unsaved edits parked in an in-memory snapshot (the user
        // switched away from it without saving).
        const dirty = (activeDocId === doc.id && unsaved)
          || snapshotDirtyDocIds.includes(doc.id);
        const editing = editingId === doc.id;
        return (
          <Tooltip key={doc.id} title={editing ? "" : `${doc.path || "Unsaved document"} — double-click to rename`}>
            <Box sx={tabSx(active)} onClick={() => { if (!editing) void switchToDocument(doc.id); }}>
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

     {/* Record button — pinned right, never scrolls off, and present in
         every mode (builder / analysis / collage) so a full workflow can
         be filmed continuously across tab switches. Only renders when dev
         options are enabled (handled inside RecordAppButton). */}
     <Box sx={{ display: "flex", alignItems: "center", px: 0.5, flexShrink: 0 }}>
       <RecordAppButton />
     </Box>
    </Box>
  );
}
