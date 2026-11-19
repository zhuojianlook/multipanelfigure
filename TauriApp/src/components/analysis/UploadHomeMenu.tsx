/* The Analysis workflow tab's menu (right-click a tab, or click its owner
   chip) for where the workflow's uploaded images are saved:
     • shared (reads several documents' figures): "Store uploaded images
       in ▸" — one of its documents, the current one checked;
     • uploads only (reads no figure data): "Save this analysis in…" —
       reopens the choose-document dialog. */

import { ListSubheader, Menu, MenuItem } from "@mui/material";

export interface UploadHomeChoice {
  uid: string;
  label: string;
}

export interface UploadHomeMenuProps {
  /** Where the menu opens (client coordinates); null: closed. */
  anchor: { top: number; left: number } | null;
  onClose: () => void;
  /** Shared workflow: the documents that may store its uploads. */
  choices?: readonly UploadHomeChoice[];
  /** The document that stores them now. */
  home?: string;
  onPick?: (uid: string) => void;
  /** Uploads-only workflow: open the choose-document dialog. */
  onChooseDocument?: () => void;
  /** Read-only (locked) workflow: shown, not changeable. */
  disabled?: boolean;
}

export function UploadHomeMenu({ anchor, onClose, choices, home, onPick, onChooseDocument, disabled }: UploadHomeMenuProps) {
  const items = [];
  if (choices && choices.length) {
    items.push(
      <ListSubheader key="hdr" sx={{ fontSize: "0.7rem", lineHeight: "28px" }}>Store uploaded images in ▸</ListSubheader>,
    );
    for (const c of choices) {
      const on = c.uid === home;
      items.push(
        <MenuItem
          key={c.uid}
          dense
          disabled={disabled}
          selected={on}
          aria-checked={on}
          role="menuitemradio"
          onClick={() => { onClose(); if (!on) onPick?.(c.uid); }}
          sx={{ fontSize: "0.75rem", pl: 3 }}
        >
          <span style={{ width: 16, display: "inline-block" }}>{on ? "✓" : ""}</span>{c.label}
        </MenuItem>,
      );
    }
  }
  if (onChooseDocument) {
    items.push(
      <MenuItem
        key="choose"
        dense
        disabled={disabled}
        onClick={() => { onClose(); onChooseDocument(); }}
        sx={{ fontSize: "0.75rem" }}
      >
        Save this analysis in…
      </MenuItem>,
    );
  }
  return (
    <Menu
      open={!!anchor && items.length > 0}
      onClose={onClose}
      anchorReference="anchorPosition"
      anchorPosition={anchor ?? undefined}
    >
      {items}
    </Menu>
  );
}
