"""Which plot an R re-render targets (run-r `override_plot_index`).

The index a caller stores (rPlotIndex) is the plot's position in the
`plots` list a normal run returns: Python sorted() FILENAME order. These
tests run real R through `run_r_code` (no server) and check that a re-render,
text edit, SVG or label request for index i gets plot i — whatever order the
script called mpfig_plot() in, whether it built the ggplot before opening the
device, and that a plot which cannot be re-rendered yields no image (with
`override_error`) rather than a different plot.

Skipped unless Rscript and the harness packages are installed (otherwise the
generated prelude would try install.packages over the network).
"""

import base64
import io
import os
import struct
import subprocess
import sys
import tempfile

import pytest

if "api_server" not in sys.modules:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
from PIL import Image  # noqa: E402

RSCRIPT = S._find_rscript(None)


def _r_has(*pkgs) -> bool:
    if not RSCRIPT:
        return False
    vec = ", ".join(f'"{p}"' for p in pkgs)
    try:
        r = subprocess.run(
            [RSCRIPT, "-e", f"quit(status = !all(sapply(c({vec}), requireNamespace, quietly = TRUE)))"],
            capture_output=True, timeout=60, env=S._r_env(),
        )
    except Exception:
        return False
    return r.returncode == 0


pytestmark = pytest.mark.skipif(
    not _r_has("ggplot2", "ggprism", "rstatix", "dplyr"),
    reason="needs Rscript with ggplot2, ggprism, rstatix and dplyr",
)

PRE = (
    "suppressMessages(library(ggplot2))\n"
    "G <- function(t) ggplot(mtcars, aes(mpg, wt)) + geom_point() + labs(title = t)\n"
)

# Call order zeta, alpha, Mid; each ggplot is BUILT before its device opens
# (as the nuclei / western blot / thickness templates do). Python sorts
# "Mid.png" < "alpha.png" < "zeta.png". 1103 x 851 at res 300 is a size the
# macOS PNG device used to truncate by a pixel in both directions.
ORDER = (
    'for (n in c("zeta", "alpha", "Mid")) {\n'
    '  p <- G(n); mpfig_plot(paste0(n, ".png"), width = 1103, height = 851, res = 300); print(p)\n'
    "}\n"
)
ORDER_TITLES = ["Mid", "alpha", "zeta"]


def run(code, **kw):
    return S.run_r_code(S.RAnalysisRequest(data_csv="x\n1\n", code=PRE + code, **kw))


def rerender(code, idx, w=1103, h=851, res=300, **kw):
    kw.setdefault("render_override", True)
    return run(code, override_plot_index=idx, override_only=True, emit_labels=True,
               override_width=w, override_height=h, override_res=res, **kw)


def dims(b64):
    return struct.unpack(">II", base64.b64decode(b64)[16:24])


def pixels(b64):
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    return im.size, im.tobytes()


def assert_no_image(r, why, code=None):
    assert r["success"], r["stderr"][-800:]
    assert r["plots"] == [] and not r.get("svg"), "a plot came back for an index that must render nothing"
    assert r["labels"] == {}
    assert r["override_error"] and why in r["override_error"], r["override_error"]
    if code:
        assert r["override_code"] == code, (r["override_code"], r["override_error"])


def titles_by_index(code, n, **kw):
    out = []
    for i in range(n):
        r = rerender(code, i, w=800, h=600, res=150, **kw)
        out.append(r["labels"].get("title") if r["plots"] else r["override_code"])
    return out


NOT_GG = "is not a ggplot"
NOT_MPFIG = "was not drawn as a single image through mpfig_plot()"


def test_index_is_filename_order_not_call_order():
    base = run(ORDER)
    assert base["success"], base["stderr"][-800:]
    assert len(base["plots"]) == 3 and base["override_error"] is None
    # mpfig_plot() asks for 1103 x 851; the file is exactly that size.
    assert [dims(p) for p in base["plots"]] == [(1103, 851)] * 3
    px = [pixels(p) for p in base["plots"]]
    assert len({p[1] for p in px}) == 3
    for i, want in enumerate(ORDER_TITLES):
        r = rerender(ORDER, i)
        assert r["success"] and r["override_error"] is None, r["stderr"][-800:]
        assert r["labels"].get("title") == want
        assert len(r["plots"]) == 1 and dims(r["plots"][0]) == (1103, 851)
        # Same size and res -> the re-render IS plot i, pixel for pixel.
        got = pixels(r["plots"][0])
        assert [j for j, p in enumerate(px) if p == got] == [i]


def test_index_past_the_end_renders_nothing():
    r = rerender(ORDER, 3)
    assert_no_image(r, "There is no plot 4", "no_such_plot")


def test_auto_names_sort_as_strings():
    code = "".join(f'mpfig_plot(); print(G("T{i}"))\n' for i in range(1, 13))
    # plot_1, plot_10, plot_11, plot_12, plot_2, ... plot_9
    for idx, want in ((1, "T10"), (4, "T2"), (11, "T9")):
        r = rerender(code, idx, w=800, h=600, res=150)
        assert r["labels"].get("title") == want, (idx, r["labels"], r["override_error"])
        assert len(r["plots"]) == 1


def test_duplicate_filename_last_call_wins():
    code = 'mpfig_plot("a.png"); print(G("first")); mpfig_plot("b.png"); print(G("B")); mpfig_plot("a.png"); print(G("second"))\n'
    assert rerender(code, 0, w=800, h=600, res=150)["labels"].get("title") == "second"
    assert rerender(code, 1, w=800, h=600, res=150)["labels"].get("title") == "B"


MIX = (
    # a: a ggplot; the user then closes it and writes c_direct.png with png(),
    # which reuses the same device number.
    'mpfig_plot("a.png"); print(G("A")); dev.off()\n'
    'png(file.path(.plot_dir, "c_direct.png")); print(G("DIRECT")); dev.off()\n'
    # b: a ggplot and then base graphics on the same device -> the file holds the base plot.
    'mpfig_plot("b.png"); print(G("B_GG")); plot(1:10)\n'
    # not a .png: not returned, so it must not shift later indices.
    'mpfig_plot("noext"); print(G("N"))\n'
    'mpfig_plot("d.png"); print(G("D"))\n'
    # e: base graphics right after a ggplot device; f: a non-ggplot grid page.
    'mpfig_plot("e.png"); plot(1:3)\n'
    'mpfig_plot("f_grid.png"); grid::grid.newpage(); grid::grid.rect()\n'
)


def test_plots_that_cannot_be_rerendered_yield_no_image():
    base = run(MIX)
    assert base["success"], base["stderr"][-800:]
    assert len(base["plots"]) == 6  # a, b, c_direct, d, e, f_grid
    assert rerender(MIX, 0, w=800, h=600, res=150)["labels"].get("title") == "A"
    for idx in (1, 4, 5):
        assert_no_image(rerender(MIX, idx, w=800, h=600, res=150), NOT_GG, "not_ggplot")
    assert_no_image(rerender(MIX, 2, w=800, h=600, res=150), NOT_MPFIG, "not_mpfig_plot")
    assert rerender(MIX, 3, w=800, h=600, res=150)["labels"].get("title") == "D"


@pytest.mark.skipif(not _r_has("svglite"), reason="needs svglite")
def test_a_device_reusing_the_number_is_not_the_mpfig_device():
    # After the user's dev.off(), svglite (no "filepath" attribute) gets the
    # same device number; neither its ggplot nor its base plot is a.png's.
    code = ('mpfig_plot("a.png"); print(G("ONE")); dev.off()\n'
            'ggsave(file.path(tempdir(), "extra.svg"), G("TWO"), width = 4, height = 3)\n'
            'svglite::svglite(file.path(tempdir(), "extra2.svg")); plot(1:3); dev.off()\n'
            'mpfig_plot("b.png"); print(G("B"))\n'
            'ggsave(file.path(tempdir(), "extra3.svg"), G("THREE"), width = 4, height = 3)\n')
    assert titles_by_index(code, 2) == ["ONE", "B"]


def test_the_users_own_device_on_an_mpfig_file_updates_it():
    # The file now holds what the user's png() drew there, whatever spelling
    # of the path it was opened with.
    code = ('mpfig_plot("a.png"); print(G("A")); dev.off()\n'
            'png(paste0(.plot_dir, "/./a.png")); print(G("OVER")); dev.off()\n'
            'mpfig_plot("b.png"); print(G("B")); dev.off()\n'
            'png(file.path(.plot_dir, "b.png")); plot(1:3); dev.off()\n')
    assert titles_by_index(code, 2) == ["OVER", "not_ggplot"]


def test_case_variants_of_one_name():
    # Case-insensitive file systems (macOS, Windows): one file, holding the
    # later plot. Case-sensitive (Linux): two files, each its own plot.
    code = 'mpfig_plot("A.png"); print(G("first")); mpfig_plot("a.png"); print(G("second"))\n'
    n = len(run(code)["plots"])
    assert titles_by_index(code, n) == (["second"] if n == 1 else ["first", "second"])


def test_non_ascii_names():
    # The macOS PNG device stores "größe.png" decomposed (NFD); list.files()
    # returns it unmarked, which R's radix sort refuses. These names sort the
    # same composed or decomposed (Hangul decomposes to U+11xx, still after
    # "zeta"), so the expected order is plain sorted().
    names = {"zeta": "Z", "한글": "KO", "a": "A", "größe": "GROSS"}
    code = "".join(f'mpfig_plot("{k}.png"); print(G("{v}"))\n' for k, v in names.items())
    assert titles_by_index(code, len(names)) == [names[k] for k in sorted(names)]
    # A single non-ASCII file: nothing to compare it with in the sort.
    assert titles_by_index('mpfig_plot("größe.png"); print(G("GG"))\n', 1) == ["GG"]
    assert titles_by_index('mpfig_plot("图.png"); print(G("ZH")); mpfig_plot("b.png"); print(G("B"))\n', 2) == ["B", "ZH"]


def test_page_pattern_names():
    # png() fills "%02d" with the page number: one page is plot01.png.
    assert titles_by_index('mpfig_plot("plot%02d.png"); print(G("GG"))\n', 1) == ["GG"]
    # Two pages make two files from one call: neither can be mapped to a
    # single printed plot, the next call's file still can.
    code = 'mpfig_plot("pg%d.png"); print(G("P1")); print(G("P2"))\nmpfig_plot("z.png"); print(G("Z"))\n'
    assert titles_by_index(code, 3) == ["not_mpfig_plot", "not_mpfig_plot", "Z"]
    r = rerender(code, 0, w=800, h=600, res=150)
    assert_no_image(r, NOT_MPFIG, "not_mpfig_plot")


def test_override_png_is_returned_last():
    code = ('mpfig_plot("a.png", width = 1400, height = 900, res = 300); print(G("A"))\n'
            'mpfig_plot("zzz.png", width = 1103, height = 851, res = 300); print(G("Z"))\n')
    base = run(code)
    assert [dims(p) for p in base["plots"]] == [(1400, 900), (1103, 851)]
    # 401 px at res 150 is another size the macOS device truncated.
    r = run(code, render_override=True, override_plot_index=1, override_width=500, override_height=401, override_res=150)
    assert r["success"] and r["override_error"] is None
    assert len(r["plots"]) == 3 and dims(r["plots"][-1]) == (500, 401)
    assert [pixels(p) for p in r["plots"][:-1]] == [pixels(p) for p in base["plots"]]


def test_text_override_and_labels_follow_the_index():
    r = rerender(ORDER, 1, text_overrides={"title": {"text": "NEW"}}, render_override=False)
    assert r["success"] and r["override_error"] is None
    assert r["labels"].get("title") == "NEW"
    assert len(r["plots"]) == 1 and dims(r["plots"][0]) == (1103, 851)
    # Labels only (Edit text / Synchronize fetch them): that plot's, all plots returned.
    lab = run(ORDER, emit_labels=True, override_plot_index=2)
    assert lab["labels"].get("title") == "zeta"
    assert len(lab["plots"]) == 3 and lab["override_error"] is None


@pytest.mark.skipif(not _r_has("svglite"), reason="needs svglite")
def test_svg_is_the_indexed_plot():
    r = run(ORDER, render_svg=True, render_override=True, override_only=True, override_plot_index=1,
            override_width=1103, override_height=851, override_res=300)
    assert r["success"] and r["override_error"] is None
    svg = r["svg"] or ""
    assert ">alpha<" in svg and ">zeta<" not in svg and ">Mid<" not in svg


@pytest.mark.skipif(not _r_has("patchwork"), reason="needs patchwork")
def test_patchwork_is_captured_as_printed():
    # patchwork's print does not set last_plot(), which here is a later plot.
    code = ('suppressMessages(library(patchwork))\n'
            'pw <- G("L") / G("R"); z <- G("LATER")\n'
            'mpfig_plot("pw.png", width = 1103, height = 851, res = 300); print(pw)\n')
    base = run(code)
    r = rerender(code, 0)
    assert r["override_error"] is None and len(r["plots"]) == 1
    assert pixels(r["plots"][0]) == pixels(base["plots"][0])


PW = ('suppressMessages(library(patchwork))\n'
      'H <- function(t, y) ggplot(mtcars, aes(mpg, .data[[y]], colour = factor(cyl))) + geom_point() +\n'
      '  labs(title = t, colour = "Cyl")\n'
      'pw <- (H("L", "wt") | H("R", "hp")) + plot_annotation(title = "Overall",\n'
      '  theme = theme(plot.title = element_text(size = 13)))\n'
      'mpfig_plot("pw.png", width = 1200, height = 600); print(pw)\n')


@pytest.mark.skipif(not _r_has("patchwork"), reason="needs patchwork")
def test_patchwork_labels_are_the_annotation_and_shared_panel_labels():
    # Not the last panel's ("R", "hp"): the overall title, plus the axis /
    # legend titles every panel agrees on (y differs, so it is left out).
    lab = run(PW, emit_labels=True, override_plot_index=0)["labels"]
    assert lab.get("title") == "Overall" and lab.get("x") == "mpg" and lab.get("colour") == "Cyl"
    assert "y" not in lab


@pytest.mark.skipif(not (_r_has("patchwork") and _r_has("svglite")), reason="needs patchwork and svglite")
def test_patchwork_text_overrides_reach_every_panel():
    import re
    r = run(PW, emit_labels=True, override_plot_index=0, render_svg=True, render_override=True, override_only=True,
            override_width=1200, override_height=600,
            text_overrides={"title": {"text": "NEW", "size": 30}, "yaxis": {"size": 7, "color": "#ff0000"},
                            "legend_title": {"text": "Cylinders"}})
    assert r["success"] and r["override_error"] is None, r["stderr"][-800:]
    assert r["labels"].get("title") == "NEW" and r["labels"].get("colour") == "Cylinders"
    svg = r["svg"] or ""

    def styles(t):
        return re.findall(r"<text[^>]*style='([^']*)'[^>]*>" + re.escape(t) + "</text>", svg)

    # The overall title is renamed; the panel titles keep their text, and
    # every title, y axis title and legend title gets the style.
    assert not styles("Overall") and len(styles("NEW")) == 1
    for t in ("NEW", "L", "R"):
        assert len(styles(t)) == 1 and "font-size: 30.00px" in styles(t)[0], (t, styles(t))
    for t in ("wt", "hp"):
        assert len(styles(t)) == 1 and "font-size: 7.00px" in styles(t)[0] and "#FF0000" in styles(t)[0].upper()
    assert len(styles("Cylinders")) == 2 and not styles("Cyl")


def test_script_error_is_reported():
    r = rerender('mpfig_plot("a.png"); print(G("A"))\nstop("boom-from-r")\n', 0, w=800, h=600, res=150)
    assert r["plots"] == [] and not r["success"]
    assert r["override_error"].startswith("The R script stopped with an error") and "boom-from-r" in r["override_error"]
    assert r["override_code"] == "script_error"


def test_name_mismatch_returns_nothing(monkeypatch):
    # If Python's list ever disagreed with the name R resolved, the override
    # is dropped instead of being returned as plot i.
    real = S._r_plot_files

    def reversed_files(plot_dir):
        normal, ov = real(plot_dir)
        return normal[::-1], ov

    monkeypatch.setattr(S, "_r_plot_files", reversed_files)
    assert_no_image(rerender(ORDER, 0), "could not be matched", "mismatch")


def test_plot_file_listing(tmp_path):
    for name in ("b.png", "zzz.png", "zz_mpfig_override.png", "a.png", ".hidden.png", "c.PNG", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "dir.png").mkdir()
    normal, ov = S._r_plot_files(str(tmp_path))
    names = [os.path.basename(p) for p in normal]
    want = ["a.png", "b.png", "zzz.png"]
    if os.name == "nt":  # glob matches case-insensitively there (R mirrors it)
        want = ["a.png", "b.png", "c.PNG", "zzz.png"]
    assert names == want
    assert os.path.basename(ov) == "zz_mpfig_override.png"
