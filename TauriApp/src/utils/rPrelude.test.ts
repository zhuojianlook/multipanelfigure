import { describe, it, expect } from "vitest";
import { rInputsPrelude } from "./rPrelude";

describe("rInputsPrelude", () => {
  it("defines empty inputs when nothing is wired", () => {
    expect(rInputsPrelude([])).toBe("inputs <- list()\ndata <- data.frame()\n\n");
  });

  it("inlines each table, in order, named after its source", () => {
    const p = rInputsPrelude([{ label: "Picker \"A\"", csv: "group\n0.10\n" }, { csv: "x\n1\n" }]);
    expect(p).toContain('  "Picker \\"A\\"" = .mpfig_csv("group\\n0.10\\n"),\n  "t1" = .mpfig_csv("x\\n1\\n")\n)');
    expect(p).toContain("if (length(inputs) > 0) data <- inputs[[1]] else data <- data.frame()");
  });

  it("reads as UTF-8 and keeps each table's text, for code that needs it as written", () => {
    const p = rInputsPrelude([{ label: "t", csv: "a\n1\n" }]);
    const helper = p.slice(0, p.indexOf("inputs <- list("));
    expect(helper).toBe(
      ".mpfig_csv_text <- list()\n"
      + '.mpfig_csv <- function(s) { Encoding(s) <- "UTF-8"; .mpfig_csv_text[[length(.mpfig_csv_text) + 1L]] <<- s; '
      + "read.csv(text = s, stringsAsFactors = FALSE) }\n",
    );
    // The text is stored after it is marked UTF-8, so a second read decodes it the same way.
    expect(helper.indexOf('Encoding(s) <- "UTF-8"')).toBeLessThan(helper.indexOf("<<- s"));
  });
});
