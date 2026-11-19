/* ──────────────────────────────────────────────────────────
   csvTable — read CSV text into rows of cells (RFC 4180).

   Tables from R (write.csv) and Python quote any text cell, and since
   0.1.420 panel names carry the row and column headers joined by ", "
   ("1hr, Sample 1"). Splitting such a line on every comma pushes the
   remaining cells one column to the right, under the wrong header. This
   reader keeps commas, doubled quotes and line breaks inside quoted cells.
   ────────────────────────────────────────────────────────── */

/** Rows of cells. Blank lines are skipped; a trailing newline adds no row. */
export function parseCsvTable(text: string): string[][] {
  const rows: string[][] = [];
  const s = text ?? "";
  let row: string[] = [];
  let cell = "";
  let inQ = false;
  let quoted = false;   // the current cell had quotes, so "" is a real empty cell
  const endRow = () => {
    row.push(cell);
    if (row.length > 1 || cell !== "" || quoted) rows.push(row);
    row = []; cell = ""; quoted = false;
  };
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inQ) {
      if (ch === '"') {
        if (s[i + 1] === '"') { cell += '"'; i++; } else inQ = false;
      } else cell += ch;
    } else if (ch === '"') {
      inQ = true; quoted = true;
    } else if (ch === ",") {
      row.push(cell); cell = ""; quoted = false;
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && s[i + 1] === "\n") i++;
      endRow();
    } else cell += ch;
  }
  if (cell !== "" || row.length > 0 || quoted) endRow();
  return rows;
}

/** Number of data rows (all rows but the header). */
export function csvDataRowCount(text: string): number {
  return Math.max(0, parseCsvTable(text).length - 1);
}
