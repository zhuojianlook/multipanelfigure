/* ──────────────────────────────────────────────────────────
   FigureTabs — the figures INSIDE the open document.

   One .mpf holds one set of uploaded images (and image groups) and any
   number of figures built from them. This row sits directly under the
   image strip's Groups section — the images above are shared by every
   figure listed here — and is deliberately unlike the document tab bar
   at the top (boxed file tabs, one per .mpf): small pills with a grid
   icon, labelled "Figures".

   Click switches figure, double-click renames, right-click (or the ⋯ on
   the active pill) opens Rename / Duplicate / Move / Delete. Reordering
   is menu-based, not drag-and-drop, given WKWebView's stuck-native-drag
   history (0.1.409). Every operation goes through utils/projectNav, which
   serializes it with document switch / save / open / close.
   ────────────────────────────────────────────────────────── */

import { useState, type MouseEvent as ReactMouseEvent } from "react";
import { IconButton, Menu, MenuItem, ListItemIcon, Tooltip } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import GridViewIcon from "@mui/icons-material/GridView";
import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
import EditIcon from "@mui/icons-material/Edit";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { useFigureStore } from "../../store/figureStore";
import {
  switchFigure, addFigure, duplicateFigure, deleteFigure, renameFigure, moveFigure,
} from "../../utils/projectNav";
import { confirm } from "../shared/ConfirmDialog";
import { FigureMarker } from "../shared/FigureMarker";

export function FigureTabs() {
  const figures = useFigureStore((s) => s.figures);
  const activeFigureId = useFigureStore((s) => s.activeFigureId);
  const hasConfig = useFigureStore((s) => s.config !== null);
  const [busy, setBusy] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null);

  if (!hasConfig) return null;

  /** Run one figure op at a time; the tabs ignore clicks meanwhile. */
  const run = async (op: () => Promise<unknown>) => {
    if (busy) return;
    setBusy(true);
    try { await op(); } catch (err) { console.error("Figure tab operation failed", err); }
    finally { setBusy(false); }
  };

  const beginRename = (id: string) => {
    const f = figures.find((x) => x.id === id);
    if (!f) return;
    setDraft(f.name);
    setEditingId(id);
  };
  const finishRename = (save: boolean) => {
    const id = editingId;
    setEditingId(null);
    if (!save || !id) return;
    const name = draft.trim();
    const f = figures.find((x) => x.id === id);
    if (f && name && name !== f.name) void run(() => renameFigure(id, name));
  };

  const requestDelete = async (id: string) => {
    const f = figures.find((x) => x.id === id);
    if (!f || figures.length <= 1) return;
    const ok = await confirm({
      title: `Delete "${f.name}"?`,
      body: "This removes the figure's layout, labels and annotations from the project. "
        + "The images stay in the project for the other figures. "
        + "Collage items made from this figure will no longer render.",
      confirmLabel: "Delete figure",
      destructive: true,
    });
    if (ok) void run(() => deleteFigure(id));
  };

  const openMenu = (e: ReactMouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    setMenu({ id, x: e.clientX, y: e.clientY });
  };

  const menuFig = menu ? figures.find((f) => f.id === menu.id) : undefined;
  const menuIdx = menuFig ? figures.indexOf(menuFig) : -1;
  const closeMenu = () => setMenu(null);

  return (
    <div
      className="flex items-center gap-1 px-3 border-b flex-none overflow-x-auto"
      style={{ borderColor: "var(--c-border)", backgroundColor: "var(--c-surface)", minHeight: 26 }}
      role="tablist"
      aria-label="Figures in this project"
    >
      <Tooltip title="Every figure in this project uses the images and groups above">
        <span
          className="text-[10px] font-semibold tracking-wider uppercase flex-none mr-1"
          style={{ color: "var(--c-text-dim)" }}
        >
          Figures ({figures.length})
        </span>
      </Tooltip>

      {figures.map((f, i) => {
        const active = f.id === activeFigureId;
        if (editingId === f.id) {
          return (
            <input
              key={f.id}
              autoFocus
              maxLength={80}
              className="text-[10px] font-semibold px-1.5 rounded outline-none flex-none"
              style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text)", border: "1px solid var(--c-accent)", width: 140, height: 20 }}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onFocus={(e) => e.target.select()}
              onBlur={() => finishRename(true)}
              onKeyDown={(e) => {
                if (e.key === "Enter") finishRename(true);
                if (e.key === "Escape") finishRename(false);
              }}
            />
          );
        }
        return (
          <Tooltip key={f.id} title={`${f.name} — double-click to rename, right-click for more`} enterDelay={500}>
            <div
              role="tab"
              aria-selected={active}
              className="flex items-center gap-1 pl-1.5 pr-1 rounded-full flex-none select-none"
              style={{
                height: 20,
                maxWidth: 200,
                fontSize: 10.5,
                fontWeight: active ? 600 : 500,
                cursor: busy ? "progress" : "pointer",
                color: active ? "var(--c-text)" : "var(--c-text-dim)",
                backgroundColor: active ? "rgba(79,195,247,0.14)" : "transparent",
                border: `1px solid ${active ? "var(--c-accent)" : "var(--c-border)"}`,
                opacity: busy && !active ? 0.6 : 1,
              }}
              onClick={() => { if (!active) void run(() => switchFigure(f.id)); }}
              onDoubleClick={() => beginRename(f.id)}
              onContextMenu={(e) => openMenu(e, f.id)}
            >
              {/* With several figures, the tab's number is the marker the
                  image tiles show for "used in this figure". */}
              {figures.length > 1
                ? <FigureMarker label={i + 1} current={active} />
                : <GridViewIcon sx={{ fontSize: 11, opacity: 0.8 }} />}
              <span className="truncate">{f.name}</span>
              {active && (
                <IconButton
                  size="small"
                  aria-label={`More actions for ${f.name}`}
                  onClick={(e) => openMenu(e, f.id)}
                  sx={{ p: 0, ml: 0.25, color: "inherit" }}
                >
                  <MoreHorizIcon sx={{ fontSize: 12 }} />
                </IconButton>
              )}
            </div>
          </Tooltip>
        );
      })}

      <Tooltip title="New figure in this project (uses the same images)">
        <span>
          <IconButton
            size="small"
            disabled={busy}
            onClick={() => void run(() => addFigure())}
            sx={{ p: 0.25, ml: 0.25 }}
            aria-label="New figure"
          >
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </span>
      </Tooltip>

      <Menu
        open={!!menu && !!menuFig}
        onClose={closeMenu}
        anchorReference="anchorPosition"
        anchorPosition={menu ? { top: menu.y, left: menu.x } : undefined}
        slotProps={{ paper: { sx: { fontSize: 12 } } }}
      >
        <MenuItem dense onClick={() => { const id = menu!.id; closeMenu(); beginRename(id); }}>
          <ListItemIcon><EditIcon sx={{ fontSize: 16 }} /></ListItemIcon>Rename
        </MenuItem>
        <MenuItem dense disabled={busy} onClick={() => { const id = menu!.id; closeMenu(); void run(() => duplicateFigure(id)); }}>
          <ListItemIcon><ContentCopyIcon sx={{ fontSize: 16 }} /></ListItemIcon>Duplicate
        </MenuItem>
        <MenuItem dense disabled={busy || menuIdx <= 0} onClick={() => { const id = menu!.id; closeMenu(); void run(() => moveFigure(id, -1)); }}>
          <ListItemIcon><ChevronLeftIcon sx={{ fontSize: 16 }} /></ListItemIcon>Move left
        </MenuItem>
        <MenuItem dense disabled={busy || menuIdx < 0 || menuIdx >= figures.length - 1} onClick={() => { const id = menu!.id; closeMenu(); void run(() => moveFigure(id, 1)); }}>
          <ListItemIcon><ChevronRightIcon sx={{ fontSize: 16 }} /></ListItemIcon>Move right
        </MenuItem>
        <MenuItem dense disabled={busy || figures.length <= 1} onClick={() => { const id = menu!.id; closeMenu(); void requestDelete(id); }}>
          <ListItemIcon><DeleteOutlineIcon sx={{ fontSize: 16 }} /></ListItemIcon>Delete
        </MenuItem>
      </Menu>
    </div>
  );
}
