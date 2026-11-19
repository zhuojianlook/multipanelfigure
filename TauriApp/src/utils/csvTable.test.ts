import { describe, expect, it } from "vitest";
import { csvDataRowCount, parseCsvTable } from "./csvTable";

describe("parseCsvTable", () => {
  it("keeps a quoted group name with a comma in one cell (thickness ANOVA summary)", () => {
    const csv = '"group","mean","sd","n"\n"1hr, Sample 1",12.5,1.25,6\n"2hr, Sample 1",14,2,6\n';
    expect(parseCsvTable(csv)).toEqual([
      ["group", "mean", "sd", "n"],
      ["1hr, Sample 1", "12.5", "1.25", "6"],
      ["2hr, Sample 1", "14", "2", "6"],
    ]);
  });

  it("handles doubled quotes, line breaks inside quotes and CRLF", () => {
    const csv = 'a,b\r\n"say ""hi""","line 1\nline 2"\r\n';
    expect(parseCsvTable(csv)).toEqual([["a", "b"], ['say "hi"', "line 1\nline 2"]]);
  });

  it("keeps empty cells and skips blank lines", () => {
    expect(parseCsvTable("a,b,c\n1,,3\n\n,,\n")).toEqual([["a", "b", "c"], ["1", "", "3"], ["", "", ""]]);
    expect(parseCsvTable('x\n""\n')).toEqual([["x"], [""]]);
  });

  it("reads a last line without a newline, and empty text as no rows", () => {
    expect(parseCsvTable("a,b\n1,2")).toEqual([["a", "b"], ["1", "2"]]);
    expect(parseCsvTable("")).toEqual([]);
  });

  it("counts data rows without the header", () => {
    expect(csvDataRowCount('"g","v"\n"a, b",1\n"c",2\n')).toBe(2);
    expect(csvDataRowCount("")).toBe(0);
  });
});
