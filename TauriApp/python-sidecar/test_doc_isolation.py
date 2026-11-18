"""Regression tests: every open document keeps its OWN content and analysis
always reads the document a source belongs to (0.1.421).

Complements test_doc_gate.py (swap windows). Covers per-document raw 16-bit
arrays, a new blank tab not destroying a parked tab's z-stack files, document
ids on analysis sources, the document-independent Analysis upload store, and
the measurements listing never pausing the builder. In-process, no server.
"""

import io
import os
import sys
import tempfile
import threading

import numpy as np
import pytest

if "api_server" not in sys.modules:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
import tifffile  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from models import FigureConfig, _to_dict  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────

def ep(method, path):
    return next(r.endpoint for r in S.app.routes
                if isinstance(r, APIRoute) and r.path == path and method in r.methods)


def _clear_store():
    for v in S._analysis_store.values():
        if isinstance(v, (dict, set)):
            v.clear()


@pytest.fixture(autouse=True)
def _clean():
    S._doc_state_cache.clear()
    S._bind_refs(S._blank_doc_state({}))
    S._live_doc_id = ""
    _clear_store()
    yield
    S._doc_state_cache.clear()
    S._live_doc_id = ""
    _clear_store()
    assert S._doc_gate._writer is None and S._doc_gate._readers == 0, "gate left held"


def tiff_bytes(arr, **kw):
    b = io.BytesIO()
    tifffile.imwrite(b, arr, **kw)
    return b.getvalue()


def raw16(v):
    return tiff_bytes(np.full((16, 16), v, np.uint16))


def zstack(v=10):
    return tiff_bytes(np.full((7, 32, 32), v, np.uint8), photometric="minisblack")


def upload(name, data, path="/api/images/upload"):
    return ep("POST", path)(files=[UploadFile(file=io.BytesIO(data), filename=name)])


def stash(doc):
    return ep("POST", "/api/project/stash-state")(body=S.DocStateRequest(doc_id=doc))


def activate(doc):
    return ep("POST", "/api/project/activate-state")(body=S.DocStateRequest(doc_id=doc))


def new_blank(doc=None):
    return ep("POST", "/api/project/new-blank")(body=S.NewBlankRequest(doc_id=doc))


def save(path):
    return ep("POST", "/api/project/save")(body=S.ProjectSaveRequest(path=path))


def load(path, doc=None):
    return ep("POST", "/api/project/load")(body=S.ProjectLoadRequest(path=path, doc_id=doc))


def put_panel(r, c, image_name):
    S.cfg.panels[r][c].image_name = image_name


def panel_src(doc, r=0, c=0, **kw):
    return {"mpf_doc_id": doc, "key": f"r{r}c{c}_panel", "row": r, "col": c,
            "inset_index": -1, "inset_type": "Panel", **kw}


def two_docs_same_raw_name():
    """Doc A (blot.tif raw = 1000) parked, doc B (blot.tif raw = 50000) live —
    via the real new-tab flow."""
    new_blank("A")
    upload("blot.tif", raw16(1000))
    put_panel(0, 0, "blot.tif")
    stash("A")
    new_blank("B")
    upload("blot.tif", raw16(50000))
    put_panel(0, 0, "blot.tif")


# ── F6: raw 16-bit arrays belong to their document ──────────────────────────

def test_raw_arrays_travel_with_their_document():
    two_docs_same_raw_name()
    stash("B")
    activate("A")
    assert S.raw_analysis_images["blot.tif"].max() == 1000
    stash("A")
    activate("B")
    assert S.raw_analysis_images["blot.tif"].max() == 50000


def test_new_tab_keeps_parked_docs_raw_array():
    new_blank("A")
    upload("a_blot.tif", raw16(1234))
    stash("A")
    new_blank("B")
    stash("B")
    activate("A")
    assert S.raw_analysis_images["a_blot.tif"].max() == 1234


def test_analysis_reads_parked_docs_raw_pixels():
    two_docs_same_raw_name()                      # A parked, B live
    with S._source_ctx(panel_src("A")):
        a = S._extract_source_raw(panel_src("A"))
    with S._source_ctx(panel_src("B")):
        b = S._extract_source_raw(panel_src("B"))
    assert a.max() == 1000 and b.max() == 50000


def test_disk_document_never_uses_live_raw_array():
    new_blank("A")
    upload("blot.tif", raw16(1000))
    put_panel(0, 0, "blot.tif")
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "a.mpf")
    save(p)
    # The live doc keeps the same-named raw array; the saved file has none.
    src = {"project_path": p, "key": "r0c0_panel", "row": 0, "col": 0}
    S._live_doc_id = ""
    with S._source_ctx(src):
        assert S._extract_source_raw(src) is None


def test_state_captured_by_older_code_gets_its_own_raw_dict():
    S.raw_analysis_images["x"] = np.zeros((2, 2), np.uint16)
    st = S._capture_doc_state()
    st.pop("raw_analysis_images")                  # shape of a pre-0.1.421 stash
    S._apply_doc_state(st)
    assert S.raw_analysis_images == {} and S.raw_analysis_images is st["raw_analysis_images"]


def test_volume_cache_is_per_document():
    new_blank("A")
    upload("z.tif", zstack(0))
    va = S._load_volume("z.tif", 0, -1, 16)
    stash("A")
    new_blank("B")
    upload("z.tif", zstack(200))
    vb = S._load_volume("z.tif", 0, -1, 16)
    assert va.max() == 0 and vb.max() == 200


# ── F7: a new tab / delete never destroys another document's files ──────────

def test_new_blank_keeps_parked_doc_intact():
    new_blank("A")
    upload("z.tif", zstack(10))
    upload("h.png", tiff_bytes(np.zeros((8, 8, 3), np.uint8)))
    S.hidden_images.add("h.png")
    put_panel(0, 0, "z.tif")
    zpath = S.loaded_zstacks["z.tif"]
    stash("A")
    r = new_blank("B")
    assert r["ok"] and S._live_doc_id == "B"
    assert S.loaded_images == {} and S.loaded_zstacks == {} and S.hidden_images == set()
    assert S.raw_analysis_images == {}
    assert os.path.isfile(zpath)
    activate("A")
    S.cfg.panels[0][0].frame = 3
    assert S._get_panel_image(S.cfg.panels[0][0]) is not None   # frame seek works
    ep("POST", "/api/preview")()                                   # preview renders


def test_new_blank_keeps_app_wide_fonts():
    S.custom_fonts["MyFont.ttf"] = b"font-bytes"
    new_blank("B")
    assert "MyFont.ttf" in S.custom_fonts


def test_delete_image_keeps_file_a_stash_references():
    new_blank("A")
    upload("z.tif", zstack())
    zpath = S.loaded_zstacks["z.tif"]
    stash("A")                                      # A still live, stash shares the path
    ep("DELETE", "/api/images/{name}")(name="z.tif")
    assert os.path.isfile(zpath)
    ep("POST", "/api/project/drop-state")(body=S.DocStateRequest(doc_id="A"))
    assert not os.path.isfile(zpath)


def test_delete_image_unlinks_unreferenced_file():
    new_blank("A")
    upload("z.tif", zstack())
    zpath = S.loaded_zstacks["z.tif"]
    ep("DELETE", "/api/images/{name}")(name="z.tif")
    assert not os.path.isfile(zpath)


def test_drop_state_keeps_files_of_the_live_doc():
    new_blank("A")
    upload("z.tif", zstack())
    zpath = S.loaded_zstacks["z.tif"]
    stash("A")
    ep("POST", "/api/project/drop-state")(body=S.DocStateRequest(doc_id="A"))
    assert os.path.isfile(zpath)


def test_load_gives_the_loaded_doc_fresh_containers():
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "empty.mpf")
    save(p)                                       # an empty project on disk
    new_blank("A")
    upload("z.tif", zstack())
    upload("blot.tif", raw16(7))
    S.hidden_images.add("blot.tif")
    stash("A")
    load(p, "C")                                  # open it into a new tab
    assert S._live_doc_id == "C"
    assert S.loaded_zstacks == {} and S.zstack_counts == {} and S.hidden_images == set()
    assert S.raw_analysis_images == {}


def test_load_does_not_wipe_a_parked_docs_containers():
    new_blank("A")
    upload("z.tif", zstack())
    upload("blot.tif", raw16(9))
    stash("A")
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "x.mpf")
    save(p)
    load(p, "C")
    st = S._doc_state_cache["A"]
    assert "z.tif" in st["loaded_zstacks"] and st["raw_analysis_images"]["blot.tif"].max() == 9


# ── F5: analysis reads the document a source belongs to ────────────────────

def _docs_a_parked_b_live():
    new_blank("A")
    upload("a.png", tiff_bytes(np.full((8, 8, 3), (255, 0, 0), np.uint8)))
    put_panel(0, 0, "a.png")
    stash("A")
    new_blank("B")
    upload("b.png", tiff_bytes(np.full((8, 8, 3), (0, 0, 255), np.uint8)))
    put_panel(0, 0, "b.png")


def test_live_document_resolves_without_a_swap():
    _docs_a_parked_b_live()
    with S._doc_context("B", ""):
        assert sorted(S.loaded_images) == ["b.png"]
        assert S._doc_gate._writer is None
    with S._doc_context("A", ""):
        assert sorted(S.loaded_images) == ["a.png"]
    assert sorted(S.loaded_images) == ["b.png"]


def test_source_follows_its_document_across_a_tab_switch():
    _docs_a_parked_b_live()
    src = panel_src("B")
    with S._source_ctx(src):
        before = S._extract_source_image(src).getpixel((1, 1))
    stash("B")
    activate("A")                                  # user switched tabs mid-run
    with S._source_ctx(src):
        after = S._extract_source_image(src).getpixel((1, 1))
    assert before == after == (0, 0, 255)


def test_live_doc_wins_over_its_stale_stash():
    new_blank("A")
    upload("a.png", tiff_bytes(np.zeros((8, 8, 3), np.uint8)))
    stash("A")                                     # A stays live (switch failed)
    put_panel(0, 0, "a.png")                       # newer than the stash
    with S._doc_context("A", ""):
        assert S.cfg.panels[0][0].image_name == "a.png"


def test_closed_document_is_an_error_not_a_live_read():
    _docs_a_parked_b_live()
    with pytest.raises(HTTPException) as e:
        with S._doc_context("gone", ""):
            pass
    assert e.value.status_code == 409


def test_unknown_live_doc_keeps_legacy_behaviour():
    _docs_a_parked_b_live()
    S._live_doc_id = ""
    with S._doc_context("gone", ""):
        assert sorted(S.loaded_images) == ["b.png"]


def test_run_python_refuses_a_closed_documents_source():
    _docs_a_parked_b_live()
    body = S.PyAnalysisRequest(code="print(1)", sources=[panel_src("gone")])
    with pytest.raises(HTTPException) as e:
        ep("POST", "/api/analysis/run-python")(body=body)
    assert e.value.status_code == 409


def test_live_doc_id_is_declared_by_every_doc_op():
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "x.mpf")
    save(p)
    load(p, "X")
    assert S._live_doc_id == "X"
    load(p)
    assert S._live_doc_id == ""
    stash("Y")
    assert S._live_doc_id == "Y"
    new_blank("Z")
    assert S._live_doc_id == "Z"
    S._doc_state_cache["old"] = S._capture_doc_state()
    r = ep("POST", "/api/project/bind-live")(body=S.BindLiveRequest(doc_id="doc_initial", reset_cache=True))
    assert r["ok"] and S._live_doc_id == "doc_initial" and S._doc_state_cache == {}


def test_activate_keeps_the_stash_when_the_apply_fails(monkeypatch):
    _docs_a_parked_b_live()
    def boom(st):
        raise RuntimeError("apply failed")
    monkeypatch.setattr(S, "_apply_doc_state", boom)
    with pytest.raises(RuntimeError):
        activate("A")
    assert "A" in S._doc_state_cache


def test_tab_switch_waits_for_an_in_flight_reader():
    _docs_a_parked_b_live()
    held, done = threading.Event(), threading.Event()

    def reader():
        with S._doc_gate.shared():
            held.set()
            done.wait(5)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert held.wait(5)
    stash("B")
    sw = threading.Thread(target=lambda: activate("A"), daemon=True)
    sw.start()
    sw.join(0.3)
    assert sw.is_alive(), "activate-state replaced the live doc under a reader"
    assert sorted(S.loaded_images) == ["b.png"]
    done.set()
    sw.join(5)
    t.join(5)
    assert sorted(S.loaded_images) == ["a.png"]


# ── F8: standalone Analysis uploads belong to no document ──────────────────

def test_analysis_upload_is_document_independent():
    new_blank("A")
    r = upload("blot.tif", raw16(4242), path="/api/analysis/images/upload")
    assert r["names"] == ["blot.tif"] and r["sizes"]["blot.tif"] == [16, 16]
    assert "blot.tif" not in S.loaded_images
    assert "blot.tif" not in ep("GET", "/api/images")()["names"]
    src = {"name": "blot.tif", "row": -1, "col": -1, "key": "img:blot.tif"}
    # Survives a tab switch, a new tab and a project load, and is not shadowed
    # by a same-named image of the live doc.
    stash("A")
    new_blank("B")
    upload("blot.tif", raw16(7))
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "b.mpf")
    save(p)
    load(p, "C")
    with S._source_ctx(src):
        assert S._extract_source_image(src) is not None
        raw = S._extract_source_raw(src)
    assert raw.dtype == np.uint16 and raw.max() == 4242


def test_analysis_upload_leaves_the_live_doc_alone():
    new_blank("A")
    before = (S.cfg, S.loaded_images, S.raw_analysis_images)
    upload("x.png", tiff_bytes(np.zeros((8, 8, 3), np.uint8)), path="/api/analysis/images/upload")
    assert (S.cfg, S.loaded_images, S.raw_analysis_images) == before
    assert all(a is b for a, b in zip((S.cfg, S.loaded_images, S.raw_analysis_images), before))


def test_legacy_standalone_upload_still_resolves():
    upload("old.png", tiff_bytes(np.zeros((8, 8, 3), np.uint8)))   # 0.1.420 style: in the doc
    src = {"name": "old.png", "row": -1, "col": -1, "key": "img:old.png"}
    assert S._extract_source_image(src) is not None


def test_analysis_delete():
    upload("z.tif", zstack(), path="/api/analysis/images/upload")
    zpath = S._analysis_store["loaded_zstacks"]["z.tif"]
    ep("DELETE", "/api/analysis/images/{name}")(name="z.tif")
    assert "z.tif" not in S._analysis_store["loaded_images"]
    assert not os.path.isfile(zpath)
    with pytest.raises(HTTPException):
        ep("DELETE", "/api/analysis/images/{name}")(name="z.tif")


# ── F10: measurements never pause the builder ──────────────────────────────

def test_measurements_of_a_multi_figure_doc_take_only_the_shared_gate():
    c = S.cfg
    c.figures = [{"id": "f1", "name": "Fig 1"},
                 {"id": "f2", "name": "Fig 2", "config": _to_dict(FigureConfig(rows=1, cols=1))}]
    c.active_figure_id = "f1"
    held, done = threading.Event(), threading.Event()

    def reader():
        with S._doc_gate.shared():
            held.set()
            done.wait(5)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        out = {}
        m = threading.Thread(target=lambda: out.setdefault("r", ep("GET", "/api/measurements")()), daemon=True)
        m.start()
        m.join(2)
        assert not m.is_alive(), "measurements waited for a reader (exclusive gate)"
        assert S.cfg is c
    finally:
        done.set()
        t.join(5)


# ── F9: background jobs keep reading their own document ────────────────────

def test_job_doc_ctx_follows_the_job_document():
    _docs_a_parked_b_live()
    fig = S.active_figure_id(S.cfg)
    with S._job_doc_ctx("B", fig):
        assert S._doc_gate._writer is None and sorted(S.loaded_images) == ["b.png"]
    stash("B")
    activate("A")
    with S._job_doc_ctx("B", fig):
        assert sorted(S.loaded_images) == ["b.png"]
    assert sorted(S.loaded_images) == ["a.png"]
    ep("POST", "/api/project/drop-state")(body=S.DocStateRequest(doc_id="B"))
    with pytest.raises(HTTPException):
        with S._job_doc_ctx("B", fig):
            pass


def test_new_routes_are_classified():
    S._doc_gate_selfcheck()
    paths = {r.path for r in S.app.routes if isinstance(r, APIRoute)}
    for p in ("/api/project/new-blank", "/api/project/bind-live", "/api/analysis/images/upload",
              "/api/project/activate-state", "/api/project/load"):
        assert p in S._GATE_SELF and p in paths
    assert getattr(ep("DELETE", "/api/analysis/images/{name}"), "_doc_gate", None) == "shared"


# ── a failed replacement leaves the previous document live ────────────────

def test_failed_load_keeps_the_previous_document_live(monkeypatch):
    new_blank("A")
    upload("a.png", tiff_bytes(np.zeros((8, 8, 3), np.uint8)))
    put_panel(0, 0, "a.png")
    p = os.path.join(tempfile.mkdtemp(prefix="mpf-iso-"), "a.mpf")
    save(p)
    before = (S.cfg, S.loaded_images)

    def boom(*a, **k):
        raise RuntimeError("undecodable member")
    monkeypatch.setattr(S, "_restore_channel_groups", boom)   # fails AFTER the rebind
    with pytest.raises(RuntimeError):
        load(p, "B")
    assert (S.cfg, S.loaded_images) == before and S.cfg is before[0]
    assert sorted(S.loaded_images) == ["a.png"] and S._live_doc_id == "A"


def test_failed_activate_keeps_the_previous_document_live(monkeypatch):
    _docs_a_parked_b_live()
    before = S.loaded_images
    real = S._apply_doc_state

    def half_apply(st):
        S._bind_refs({"loaded_images": st["loaded_images"]})   # partial rebind…
        raise RuntimeError("apply failed")                     # …then failure
    monkeypatch.setattr(S, "_apply_doc_state", half_apply)
    with pytest.raises(RuntimeError):
        activate("A")
    monkeypatch.setattr(S, "_apply_doc_state", real)
    assert S.loaded_images is before and sorted(S.loaded_images) == ["b.png"]
    assert "A" in S._doc_state_cache and S._live_doc_id == "B"
