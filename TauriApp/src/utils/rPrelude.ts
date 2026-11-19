/* ──────────────────────────────────────────────────────────
   rPrelude — the R text run before an R node's own code. It defines
   `inputs`, the node's upstream tables as data frames named after their
   source, and `data`, the first of them. The CSV text is inlined, so the
   whole script re-runs on its own (the collage re-renders plots from it).

   read.csv converts columns that look numeric or logical ("0.10" -> 0.1,
   "01" -> 1, "T" -> TRUE). Code that needs a table exactly as written
   reads it again from `.mpfig_csv_text[[i]]`, the text behind inputs[[i]]
   (the "Plot intensities" runtime does, for group and channel names).
   ────────────────────────────────────────────────────────── */

export interface RPreludeTable {
  label?: string;
  csv?: string;
}

// R is usually spawned without LANG/LC_ALL (a Tauri app launched from Finder
// inherits no locale), so it runs under LC_CTYPE=C. In that locale
// read.csv(text=) TRANSLITERATES any non-ASCII byte into literal escape text:
// "µm" becomes the 9-character string "<c2><b5>m", which then draws as
// garbage in every plot label. Marking the CSV string as UTF-8 *before*
// parsing is the only thing that survives: passing encoding="UTF-8" to
// read.csv has no effect on the text= path, and re-marking afterwards is too
// late because the character is already destroyed. The text is kept with
// that mark, so a second read of it behaves the same.
const CSV_HELPER =
  ".mpfig_csv_text <- list()\n"
  + ".mpfig_csv <- function(s) { Encoding(s) <- \"UTF-8\"; "
  + ".mpfig_csv_text[[length(.mpfig_csv_text) + 1L]] <<- s; "
  + "read.csv(text = s, stringsAsFactors = FALSE) }\n";

/** The prelude for an R node reading `tables` (in input order). */
export function rInputsPrelude(tables: readonly RPreludeTable[]): string {
  if (!tables.length) return "inputs <- list()\ndata <- data.frame()\n\n";
  const assign = tables.map((t, i) =>
    `  ${JSON.stringify(t.label || `t${i}`)} = .mpfig_csv(${JSON.stringify(t.csv || "")})`,
  ).join(",\n");
  return `${CSV_HELPER}inputs <- list(\n${assign}\n)\n`
    + "# Convenience: `data` aliases the first input table.\n"
    + "if (length(inputs) > 0) data <- inputs[[1]] else data <- data.frame()\n\n";
}
