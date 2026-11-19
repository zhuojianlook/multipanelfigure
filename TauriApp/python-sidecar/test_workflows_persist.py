"""Saved Analysis workflows inside the .mpf (the workflows/ section).

Covers the atomic figure save that carries workflows/ (and legacy analysis/)
over from the previous file, the workflow-only rewrite
(/api/project/workflows/write: tail fast path == full rewrite), the reader
(/api/project/workflows/read) and the embedding / restore of standalone
Analysis uploads from their ORIGINAL bytes. In-process, no server: endpoints
are called through the registered routes (gate wrappers included).
"""

import base64
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import zipfile

import numpy as np
import pytest

if "api_server" not in sys.modules:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from models import load_project  # noqa: E402
from PIL import Image  # noqa: E402
from test_doc_isolation import (  # noqa: E402,F401  (_clean is an autouse fixture)
    _clean, _clear_store, ep, load, new_blank, put_panel, raw16, stash, tiff_bytes, upload, zstack,
)

AUP = "/api/analysis/images/upload"


@pytest.fixture(autouse=True)
def _drop_originals(_clean):
    """Delete the upload-original temp files a test left (before _clean
    clears the store's maps)."""
    yield
    for p in list(S._analysis_store.get("orig_files", {}).values()):
        try:
            os.unlink(p)
        except OSError:
            pass


# ── helpers ─────────────────────────────────────────────────────────────────

def png_bytes(color=(255, 0, 0), size=(8, 8)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return b.getvalue()


def b64(data):
    return base64.b64encode(data).decode("ascii")


def multichannel():
    arr = np.stack([np.full((32, 32), v, np.uint16) for v in (100, 2000, 30000)])
    return tiff_bytes(arr, imagej=True, metadata={"axes": "CYX"})


def save(path, carry_from=None, analysis=None):
    return ep("POST", "/api/project/save")(
        body=S.ProjectSaveRequest(path=path, carry_from=carry_from, analysis=analysis))


def write(path, manifest, blobs=None, uploads=None):
    return ep("POST", "/api/project/workflows/write")(body=S.WorkflowsWriteRequest(
        path=path, manifest=manifest, blobs=blobs or {}, uploads=uploads or []))


def read(path):
    return ep("POST", "/api/project/workflows/read")(body=S.WorkflowsReadRequest(path=path))


def man(tag="1", uid="du_a", **kw):
    return {"format": "mpf-workflows", "version": 1, "docUid": uid,
            "workflows": [{"id": f"wf_{tag}", "name": f"Workflow {tag}", "nodes": [{"id": "n1"}],
                           "edges": []}], **kw}


def blobs_v1():
    return {"p1": {"kind": "png", "data": b64(png_bytes((255, 0, 0)))},
            "t1": {"kind": "csv", "data": "a,b\n1,2\n"},
            "c1": {"kind": "txt", "data": "x <- 1  # Grüße\n"}}


def figure_doc(tmp_path, name="a.mpf", color=(255, 0, 0), doc="A"):
    """A saved document with images/, channels/ and fonts/ members."""
    new_blank(doc)
    upload("a.png", png_bytes(color))
    upload("multi.tif", multichannel())
    S.custom_fonts["F.ttf"] = b"font-bytes"
    put_panel(0, 0, "a.png")
    p = str(tmp_path / name)
    save(p)
    return p


def members(path):
    """[(name, CRC)] in central-directory order."""
    with zipfile.ZipFile(path) as z:
        return [(i.filename, i.CRC) for i in z.infolist()]


def figure_members(path):
    return [m for m in members(path) if not m[0].startswith("workflows/")]


def member_bytes(path, name):
    with zipfile.ZipFile(path) as z:
        return z.read(name)


def temp_leftovers(path):
    d = os.path.dirname(path)
    return [f for f in os.listdir(d) if f.endswith(".tmp")]


def append_members(path, items):
    with zipfile.ZipFile(path, "a") as z:
        for name, data in items:
            z.writestr(name, data)


# ── write / read round trip ─────────────────────────────────────────────────

def test_write_read_round_trip_keeps_the_figure_byte_identical(tmp_path):
    p = figure_doc(tmp_path)
    before = figure_members(p)
    names = [n for n, _ in before]
    assert "config.json" in names and "images/a.png" in names and "fonts/F.ttf" in names
    assert any(n.startswith("channels/") for n in names)

    r = write(p, man(), blobs_v1())
    assert r["ok"] and r["workflows_bytes"] > 0 and r["missing_uploads"] == []
    assert figure_members(p) == before
    assert zipfile.ZipFile(p).testzip() is None

    out = read(p)
    assert out["unsupported"] is False
    assert out["manifest"]["workflows"] == man()["workflows"]
    assert out["manifest"]["docUid"] == "du_a" and out["manifest"]["uploads"] == []
    assert base64.b64decode(out["blobs"]["p1"]["data"]) == png_bytes((255, 0, 0))
    assert out["blobs"]["t1"] == {"kind": "csv", "data": "a,b\n1,2\n"}
    assert out["blobs"]["c1"] == {"kind": "txt", "data": "x <- 1  # Grüße\n"}
    # Canonical order: blobs, then the manifest last.
    assert [n for n, _ in members(p)][-1] == "workflows/manifest.json"
    # PNG stored (already compressed), text deflated.
    with zipfile.ZipFile(p) as z:
        assert z.getinfo("workflows/blobs/p1.png").compress_type == zipfile.ZIP_STORED
        assert z.getinfo("workflows/blobs/t1.csv").compress_type == zipfile.ZIP_DEFLATED


def test_file_with_workflows_still_loads_as_a_figure(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    res = load_project(p)
    assert len(res) == 5
    cfg, imgs, fonts, analysis, channels = res
    assert "a.png" in imgs and fonts == {"F.ttf": b"font-bytes"} and analysis is None
    assert "multi.tif" in channels
    r = load(p, "X")
    assert "a.png" in r["image_names"]


def test_old_file_reads_as_no_workflows(tmp_path):
    p = figure_doc(tmp_path)
    out = read(p)
    assert out["manifest"] is None and out["blobs"] == {} and out["unsupported"] is False
    assert out["uploads"] == [] and out["renamed"] == {}


def test_newer_manifest_is_unsupported_and_never_overwritten(tmp_path):
    p = figure_doc(tmp_path)
    append_members(p, [("workflows/blobs/z.png", png_bytes()),
                       ("workflows/manifest.json", json.dumps({"format": "mpf-workflows", "version": 2}))])
    out = read(p)
    assert out["unsupported"] is True and out["manifest"] is None and out["blobs"] == {}
    assert out["version"] == 2
    before = open(p, "rb").read()
    with pytest.raises(HTTPException) as e:
        write(p, man(), blobs_v1())
    assert e.value.status_code == 409 and e.value.detail["unsupported"] is True
    assert open(p, "rb").read() == before


def test_unreadable_manifest_reads_as_unsupported(tmp_path):
    p = figure_doc(tmp_path)
    append_members(p, [("workflows/manifest.json", "{not json")])
    out = read(p)
    assert out["unsupported"] is True and out["manifest"] is None


def test_malformed_uploads_list_is_ignored(tmp_path):
    p = figure_doc(tmp_path)
    append_members(p, [("workflows/manifest.json", json.dumps({**man(), "uploads": 5}))])
    out = read(p)
    assert out["manifest"]["uploads"] == 5 and out["uploads"] == []
    r = write(p, man("2"), blobs_v1(), [])
    assert r["ok"] and read(p)["manifest"]["uploads"] == []


def test_write_to_a_missing_file_is_404(tmp_path):
    with pytest.raises(HTTPException) as e:
        write(str(tmp_path / "gone.mpf"), man())
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        read(str(tmp_path / "gone.mpf"))
    assert e.value.status_code == 404


def test_bad_blob_ids_and_kinds_are_rejected(tmp_path):
    p = figure_doc(tmp_path)
    before = open(p, "rb").read()
    for blobs in ({"../x": {"kind": "png", "data": ""}}, {"a/b": {"kind": "csv", "data": ""}},
                  {"ok": {"kind": "exe", "data": ""}}, {"ok": {"kind": "csv", "data": 5}}):
        with pytest.raises(HTTPException) as e:
            write(p, man(), blobs)
        assert e.value.status_code == 400
    assert open(p, "rb").read() == before


# ── kept blobs / 409 ────────────────────────────────────────────────────────

def test_blob_without_data_keeps_the_files_member(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man("1"), blobs_v1())
    r = write(p, man("2"), {"p1": {"kind": "png"}, "t1": {"kind": "csv"},
                            "n1": {"kind": "txt", "data": "new"}})
    assert r["ok"]
    out = read(p)
    assert out["manifest"]["workflows"][0]["id"] == "wf_2"
    assert base64.b64decode(out["blobs"]["p1"]["data"]) == png_bytes((255, 0, 0))
    assert out["blobs"]["t1"]["data"] == "a,b\n1,2\n" and out["blobs"]["n1"]["data"] == "new"
    assert "c1" not in out["blobs"]                  # not listed = dropped


def test_missing_kept_blob_is_409_and_leaves_the_file_untouched(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    before = open(p, "rb").read()
    with pytest.raises(HTTPException) as e:
        write(p, man("2"), {"p1": {"kind": "png"}, "zz": {"kind": "png"}, "yy": {"kind": "csv"}})
    assert e.value.status_code == 409 and sorted(e.value.detail["missing"]) == ["yy", "zz"]
    with pytest.raises(HTTPException) as e:
        write(p, man("2"), {"p1": {"kind": "csv"}})      # same id, other kind = other member
    assert e.value.status_code == 409 and e.value.detail == {"missing": ["p1"]}
    assert open(p, "rb").read() == before and temp_leftovers(p) == []


# ── tail rewrite == full rewrite ────────────────────────────────────────────

def _scenario(step):
    """Successive payloads exercising every planning branch."""
    if step == 0:
        return man("0"), blobs_v1(), []
    if step == 1:   # one blob changed, one kept by id, one re-sent unchanged
        return man("1"), {"p1": {"kind": "png"}, "t1": {"kind": "csv", "data": "a,b\n1,2\n"},
                          "c1": {"kind": "txt", "data": "changed"}}, []
    if step == 2:   # a new upload ahead of the kept blobs
        return man("2"), {"p1": {"kind": "png"}, "t1": {"kind": "csv"}, "c1": {"kind": "txt"}}, ["u.tif"]
    if step == 3:   # identical payload
        return man("2"), {"p1": {"kind": "png"}, "t1": {"kind": "csv"}, "c1": {"kind": "txt"}}, ["u.tif"]
    if step == 4:   # drop a blob in the middle, add a png
        return man("4"), {"p1": {"kind": "png"}, "c1": {"kind": "txt"},
                          "p2": {"kind": "png", "data": b64(png_bytes((0, 0, 255)))}}, ["u.tif"]
    return man("5"), {}, []                              # everything removed


def test_tail_rewrite_matches_full_rewrite(tmp_path, monkeypatch):
    upload("u.tif", zstack(10), AUP)
    p = figure_doc(tmp_path, "fast.mpf")
    q = str(tmp_path / "full.mpf")
    shutil.copyfile(p, q)
    figure = figure_members(p)
    for step in range(6):
        m, blobs, ups = _scenario(step)
        monkeypatch.setattr(S, "_WF_TAIL_REWRITE", True)
        ra = write(p, m, blobs, ups)
        monkeypatch.setattr(S, "_WF_TAIL_REWRITE", False)
        rb = write(q, m, blobs, ups)
        assert ra["fast_path"] is True and rb["fast_path"] is False, step
        assert members(p) == members(q), step
        assert figure_members(p) == figure, step
        manifests = []
        for f in (p, q):
            with zipfile.ZipFile(f) as z:
                assert z.testzip() is None
                manifests.append(z.read("workflows/manifest.json"))
        assert manifests[0] == manifests[1], step
        assert read(p)["blobs"] == read(q)["blobs"], step
    assert temp_leftovers(p) == []


def test_non_tail_workflows_members_take_the_full_rewrite(tmp_path):
    p = figure_doc(tmp_path)
    # A foreign layout: workflows/ in the middle of the figure members.
    q = str(tmp_path / "odd.mpf")
    with zipfile.ZipFile(p) as src, zipfile.ZipFile(q, "w", zipfile.ZIP_DEFLATED) as dst:
        infos = src.infolist()
        dst.writestr(infos[0], src.read(infos[0]))
        dst.writestr("workflows/manifest.json", json.dumps(man("old")))
        dst.writestr("workflows/blobs/old.csv", "x")
        for i in infos[1:]:
            dst.writestr(i, src.read(i))
    fig = figure_members(q)
    r = write(q, man("new"), blobs_v1())
    assert r["fast_path"] is False
    assert figure_members(q) == fig and zipfile.ZipFile(q).testzip() is None
    out = read(q)
    assert out["manifest"]["workflows"][0]["id"] == "wf_new" and "old" not in out["blobs"]
    # Now the section sits at the tail: the next write is the fast path.
    assert write(q, man("next"), {"p1": {"kind": "png"}})["fast_path"] is True
    assert figure_members(q) == fig


def test_size_is_stable_across_rewrites(tmp_path, monkeypatch):
    upload("u.tif", zstack(10), AUP)
    for fast in (True, False):
        monkeypatch.setattr(S, "_WF_TAIL_REWRITE", fast)
        p = figure_doc(tmp_path, f"s{int(fast)}.mpf")
        sizes = [write(p, man(), blobs_v1(), ["u.tif"])["size_bytes"] for _ in range(5)]
        assert len(set(sizes)) == 1, sizes
        assert os.path.getsize(p) == sizes[0]


# ── the write never touches the live document ──────────────────────────────

def test_write_leaves_live_doc_and_its_unsaved_edits_alone(tmp_path):
    p = figure_doc(tmp_path)                       # doc A, live
    with zipfile.ZipFile(p) as z:
        cfg_on_disk = z.read("config.json")
    put_panel(0, 1, "a.png")                       # an unsaved edit
    S._doc_state_cache["B"] = S._blank_doc_state({})
    refs = S._live_refs()
    cache = dict(S._doc_state_cache)
    live_id = S._live_doc_id
    write(p, man(), blobs_v1())
    after = S._live_refs()
    assert all(after[k] is refs[k] for k in refs)
    assert S._live_doc_id == live_id == "A"
    assert S._doc_state_cache.keys() == cache.keys()
    assert all(S._doc_state_cache[k] is cache[k] for k in cache)
    assert S.cfg.panels[0][1].image_name == "a.png"
    with zipfile.ZipFile(p) as z:
        assert z.read("config.json") == cfg_on_disk


# ── atomicity ───────────────────────────────────────────────────────────────

def test_failed_replace_leaves_the_file_intact(tmp_path, monkeypatch):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    before = open(p, "rb").read()

    def boom(*a, **k):
        raise OSError("replace failed")
    monkeypatch.setattr(S.os, "replace", boom)
    with pytest.raises(OSError):
        write(p, man("2"), {"x": {"kind": "csv", "data": "1"}})
    with pytest.raises(OSError):
        save(p)
    monkeypatch.undo()
    assert open(p, "rb").read() == before
    assert temp_leftovers(p) == []


def test_failure_mid_write_leaves_the_file_intact(tmp_path, monkeypatch):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    before = open(p, "rb").read()
    real = S.write_zip_members

    def half(zf, items, src_zf=None):
        real(zf, list(items)[:1], src_zf)
        raise RuntimeError("disk vanished")
    monkeypatch.setattr(S, "write_zip_members", half)
    with pytest.raises(RuntimeError):
        write(p, man("2"), {"x": {"kind": "csv", "data": "1"}, "y": {"kind": "csv", "data": "2"}})
    monkeypatch.undo()

    real_save = S.save_project

    def half_save(cfg, images, path, *a, **k):
        with open(path, "wb") as f:
            f.write(b"PK partial")
        raise RuntimeError("disk vanished")
    monkeypatch.setattr(S, "save_project", half_save)
    with pytest.raises(RuntimeError):
        save(p)
    monkeypatch.setattr(S, "save_project", real_save)
    assert open(p, "rb").read() == before
    assert temp_leftovers(p) == []
    assert read(p)["manifest"]["workflows"] == man()["workflows"]


def test_save_keeps_the_files_permissions(tmp_path):
    p = figure_doc(tmp_path)
    os.chmod(p, 0o640)
    save(p)
    write(p, man(), blobs_v1())
    assert (os.stat(p).st_mode & 0o777) == 0o640


# ── the figure save carries the saved Analysis ──────────────────────────────

LEGACY = {"manifest": {"tabs": [1]}, "plots": {"pl": b64(png_bytes((0, 255, 0)))},
          "tables": {"tb": "x,y\n"}}


def test_resave_carries_workflows_and_legacy_analysis(tmp_path):
    new_blank("A")
    upload("a.png", png_bytes())
    p = str(tmp_path / "a.mpf")
    save(p, analysis=LEGACY)                     # an older build's analysis/ section
    write(p, man(), blobs_v1())
    wf_before = [m for m in members(p) if m[0].startswith("workflows/")]
    put_panel(0, 0, "a.png")                     # a figure edit, then a plain save
    r = save(p)
    assert r["ok"] and "carry_error" not in r and r["workflows_carried"] is True
    assert [m for m in members(p) if m[0].startswith("workflows/")] == wf_before
    assert members(p)[-len(wf_before):] == wf_before          # still the tail
    *_, analysis, _ch = load_project(p)
    assert analysis["manifest"] == {"tabs": [1]} and analysis["tables"] == {"tb": "x,y\n"}
    assert analysis["plots"]["pl"] == png_bytes((0, 255, 0))
    assert read(p)["manifest"]["workflows"] == man()["workflows"]
    with zipfile.ZipFile(p) as z:
        assert json.loads(z.read("config.json"))["panels"][0][0]["image_name"] == "a.png"
    # A carried section is at the tail, so the next workflow write is fast.
    assert write(p, man("2"), {"p1": {"kind": "png"}})["fast_path"] is True


def test_save_with_analysis_payload_replaces_legacy_but_carries_workflows(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    save(p, analysis=LEGACY)
    write(p, man(), blobs_v1())
    save(p, analysis={"manifest": {"tabs": [2]}, "plots": {}, "tables": {}})
    *_, analysis, _ch = load_project(p)
    assert analysis["manifest"] == {"tabs": [2]} and analysis["plots"] == {}
    assert read(p)["manifest"]["workflows"] == man()["workflows"]


def test_save_as_carries_from_the_previous_file(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    p_bytes = open(p, "rb").read()
    q = str(tmp_path / "copy.mpf")
    assert save(q, carry_from=p)["workflows_carried"] is True
    assert open(p, "rb").read() == p_bytes
    assert read(q)["manifest"]["workflows"] == man()["workflows"]
    assert read(q)["blobs"] == read(p)["blobs"]


def test_save_over_another_file_without_a_previous_file_carries_nothing(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    new_blank("U")                                 # an Untitled doc saved over p
    assert save(p, carry_from="")["workflows_carried"] is False
    assert read(p)["manifest"] is None
    assert not any(n.startswith("workflows/") for n, _ in members(p))


def test_save_to_a_new_path_with_no_previous_file(tmp_path):
    new_blank("A")
    p = str(tmp_path / "new.mpf")
    r = save(p)
    assert r["ok"] and r["workflows_carried"] is False and read(p)["manifest"] is None


def test_save_reports_the_path_it_wrote(tmp_path):
    new_blank("A")
    r = save(str(tmp_path / "typed-without-ext"))
    assert r["path"] == str(tmp_path / "typed-without-ext.mpf") and os.path.isfile(r["path"])


def test_unreadable_previous_file_does_not_block_the_save(tmp_path):
    p = figure_doc(tmp_path)
    bad = str(tmp_path / "bad.mpf")
    with open(bad, "wb") as f:
        f.write(b"not a zip")
    r = save(p, carry_from=bad)
    assert r["ok"] and r["carry_error"] and r["workflows_carried"] is False
    assert read(p)["manifest"] is None and load_project(p)[1]


# ── upload originals ────────────────────────────────────────────────────────

def test_analysis_upload_keeps_its_original_and_delete_removes_it():
    data = zstack(10)
    upload("z.tif", data, AUP)
    st = S._analysis_store
    op = st["orig_files"]["z.tif"]
    assert open(op, "rb").read() == data and st["orig_sha"]["z.tif"] == hashlib.sha1(data).hexdigest()
    upload("z.tif", zstack(20), AUP)               # re-upload replaces the original
    assert not os.path.exists(op) and open(st["orig_files"]["z.tif"], "rb").read() == zstack(20)
    op2 = st["orig_files"]["z.tif"]
    ep("DELETE", "/api/analysis/images/{name}")(name="z.tif")
    assert not os.path.exists(op2)
    assert "z.tif" not in st["orig_files"] and "z.tif" not in st["orig_sha"]


def test_failed_analysis_upload_keeps_no_original():
    with pytest.raises(HTTPException):
        upload("broken.png", b"not an image", AUP)
    assert "broken.png" not in S._analysis_store["orig_files"]


# ── uploads embed and restore ───────────────────────────────────────────────

def _upload_three():
    upload("stack.tif", zstack(10), AUP)
    upload("raw.tif", raw16(4242), AUP)
    upload("multi.tif", multichannel(), AUP)
    return ["stack.tif", "raw.tif", "multi.tif"]


def test_uploads_embed_their_original_bytes(tmp_path):
    names = _upload_three()
    p = figure_doc(tmp_path)
    fig = figure_members(p)
    r = write(p, man(), blobs_v1(), names + ["stack.tif"])
    assert r["missing_uploads"] == []
    with zipfile.ZipFile(p) as z:
        mf = json.loads(z.read("workflows/manifest.json"))
        ents = {e["name"]: e for e in mf["uploads"]}
        assert sorted(ents) == sorted(names)                       # de-duplicated
        assert z.read(ents["stack.tif"]["member"]) == zstack(10)
        assert ents["raw.tif"]["sha1"] == hashlib.sha1(raw16(4242)).hexdigest()
        assert ents["multi.tif"]["size"] == len(multichannel())
        order = z.namelist()
        assert order.index(ents["stack.tif"]["member"]) < order.index("workflows/blobs/p1.png")
    assert figure_members(p) == fig


def test_uploads_restore_like_fresh_uploads(tmp_path):
    names = _upload_three()
    p = figure_doc(tmp_path)
    write(p, man(), {}, names)
    _clear_store()                                 # a new session
    live = S._live_refs()
    out = read(p)
    assert S._doc_gate._writer is None and S._doc_gate._readers == 0
    assert all(S._live_refs()[k] is live[k] for k in live)
    assert sorted(u["name"] for u in out["uploads"]) == sorted(names)
    assert out["renamed"] == {} and out["missing_uploads"] == [] and out["failed_uploads"] == []
    assert all(u["thumbnail"] and len(u["size"]) == 2 for u in out["uploads"])
    st = S._analysis_store
    assert st["zstack_counts"]["stack.tif"] == 7 and os.path.isfile(st["loaded_zstacks"]["stack.tif"])
    assert st["raw_analysis_images"]["raw.tif"].dtype == np.uint16
    assert st["raw_analysis_images"]["raw.tif"].max() == 4242
    g = st["channel_groups"]["multi.tif"]
    assert g.num_channels == 3 and g.arr.dtype == np.uint16
    for n in names:                                # restored uploads can be embedded again
        assert open(st["orig_files"][n], "rb").read() == member_bytes(
            p, next(e["member"] for e in read(p)["manifest"]["uploads"] if e["name"] == n))
    src = {"name": "raw.tif", "row": -1, "col": -1, "key": "img:raw.tif"}
    with S._source_ctx(src):
        assert S._extract_source_raw(src).max() == 4242
    assert "stack.tif" not in S.loaded_images      # never in the live document


def test_same_upload_already_loaded_is_reused(tmp_path, monkeypatch):
    upload("stack.tif", zstack(10), AUP)
    p = figure_doc(tmp_path)
    write(p, man(), {}, ["stack.tif"])
    st = S._analysis_store
    img, zpath, op = st["loaded_images"]["stack.tif"], st["loaded_zstacks"]["stack.tif"], st["orig_files"]["stack.tif"]
    streamed = []
    real = S._stream_upload_original
    monkeypatch.setattr(S, "_stream_upload_original", lambda *a: streamed.append(a[0]) or real(*a))
    out = read(p)
    assert streamed == []                          # held already: its bytes are never read
    assert out["renamed"] == {} and [u["name"] for u in out["uploads"]] == ["stack.tif"]
    assert st["loaded_images"]["stack.tif"] is img and st["loaded_zstacks"]["stack.tif"] == zpath
    assert st["orig_files"]["stack.tif"] == op and len(st["orig_files"]) == 1


def test_same_name_different_content_is_renamed(tmp_path):
    upload("stack.tif", zstack(10), AUP)
    p = figure_doc(tmp_path)
    write(p, man(), {}, ["stack.tif"])
    _clear_store()
    upload("stack.tif", zstack(99), AUP)           # another document's same-named upload
    out = read(p)
    assert out["renamed"] == {"stack.tif": "stack (2).tif"}
    assert [u["name"] for u in out["uploads"]] == ["stack (2).tif"]
    st = S._analysis_store
    assert S._extract_tiff_frame(st["loaded_zstacks"]["stack (2).tif"], 0).getpixel((0, 0))[0] == 10
    assert S._extract_tiff_frame(st["loaded_zstacks"]["stack.tif"], 0).getpixel((0, 0))[0] == 99
    # Opening it again finds its own copy under the new name — no third copy.
    out2 = read(p)
    assert out2["renamed"] == {"stack.tif": "stack (2).tif"}
    assert sorted(st["loaded_images"]) == ["stack (2).tif", "stack.tif"]


def test_missing_original_is_reported_and_files_copy_is_kept(tmp_path):
    upload("stack.tif", zstack(10), AUP)
    p = figure_doc(tmp_path)
    r = write(p, man(), {}, ["ghost.tif"])
    assert r["missing_uploads"] == ["ghost.tif"]
    write(p, man(), {}, ["stack.tif"])
    member = read(p)["manifest"]["uploads"][0]["member"]
    ep("DELETE", "/api/analysis/images/{name}")(name="stack.tif")   # original gone
    r = write(p, man("2"), {}, ["stack.tif"])
    assert r["missing_uploads"] == []
    ents = read(p)["manifest"]["uploads"]
    assert [e["member"] for e in ents] == [member] and member_bytes(p, member) == zstack(10)


def test_changed_upload_content_is_embedded_again(tmp_path):
    upload("stack.tif", zstack(10), AUP)
    p = figure_doc(tmp_path)
    write(p, man(), {}, ["stack.tif"])
    upload("stack.tif", zstack(30), AUP)           # re-uploaded with new content
    write(p, man(), {}, ["stack.tif"])
    ents = read(p)["manifest"]["uploads"]
    assert len(ents) == 1 and member_bytes(p, ents[0]["member"]) == zstack(30)
    assert sum(1 for n, _ in members(p) if n.startswith("workflows/uploads/")) == 1


def test_video_upload_embeds_and_restores(tmp_path):
    cv2 = pytest.importorskip("cv2")
    vp = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    if not w.isOpened():
        pytest.skip("no MJPG writer")
    for i in range(3):
        w.write(np.full((32, 32, 3), 40 * i, np.uint8))
    w.release()
    data = open(vp, "rb").read()
    upload("clip.avi", data, AUP)
    p = figure_doc(tmp_path)
    write(p, man(), {}, ["clip.avi"])
    with zipfile.ZipFile(p) as z:
        ent = json.loads(z.read("workflows/manifest.json"))["uploads"][0]
        assert z.getinfo(ent["member"]).compress_type == zipfile.ZIP_STORED
    _clear_store()
    out = read(p)
    st = S._analysis_store
    assert [u["name"] for u in out["uploads"]] == ["clip.avi"]
    assert open(st["loaded_videos"]["clip.avi"], "rb").read() == data


# ── gate ────────────────────────────────────────────────────────────────────

def test_workflow_routes_are_gated():
    S._doc_gate_selfcheck()
    assert "/api/project/workflows/read" in S._GATE_SELF
    assert "/api/project/workflows/write" not in S._GATE_SELF | S._GATE_EXEMPT
    assert getattr(ep("POST", "/api/project/workflows/write"), "_doc_gate", None) == "shared"
    assert getattr(ep("POST", "/api/project/workflows/read"), "_doc_gate", None) is None


def _with_reader_held(fn):
    """Run fn() in a thread while another thread holds the gate SHARED.
    Returns (finished_while_held, result)."""
    held, done = threading.Event(), threading.Event()

    def reader():
        with S._doc_gate.shared():
            held.set()
            done.wait(10)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert held.wait(5)
    out = {}
    w = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    w.start()
    w.join(1.0)
    finished = not w.is_alive()
    done.set()
    w.join(10)
    t.join(10)
    return finished, out.get("r")


def test_read_parses_outside_the_gate_and_ingests_exclusively(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(), blobs_v1())
    finished, r = _with_reader_held(lambda: read(p))          # nothing to ingest
    assert finished and r["manifest"] is not None
    upload("stack.tif", zstack(10), AUP)
    write(p, man(), {}, ["stack.tif"])
    _clear_store()
    finished, r = _with_reader_held(lambda: read(p))          # ingest waits for readers
    assert not finished and [u["name"] for u in r["uploads"]] == ["stack.tif"]


# ── documents never see each other's workflows ──────────────────────────────

def test_two_files_never_see_each_others_blobs_or_uploads(tmp_path):
    upload("a.tif", raw16(1), AUP)
    upload("b.tif", raw16(2), AUP)
    pa = figure_doc(tmp_path, "a.mpf", (255, 0, 0), "A")
    stash("A")
    pb = figure_doc(tmp_path, "b.mpf", (0, 0, 255), "B")
    write(pa, man("a", "du_a"), {"x": {"kind": "png", "data": b64(png_bytes((255, 0, 0)))}}, ["a.tif"])
    write(pb, man("b", "du_b"), {"x": {"kind": "png", "data": b64(png_bytes((0, 0, 255)))}}, ["b.tif"])
    ra, rb = read(pa), read(pb)
    assert ra["manifest"]["docUid"] == "du_a" and rb["manifest"]["docUid"] == "du_b"
    assert base64.b64decode(ra["blobs"]["x"]["data"]) == png_bytes((255, 0, 0))
    assert base64.b64decode(rb["blobs"]["x"]["data"]) == png_bytes((0, 0, 255))
    assert [u["name"] for u in ra["uploads"]] == ["a.tif"]
    assert [u["name"] for u in rb["uploads"]] == ["b.tif"]
    assert not any("b.tif" in n for n, _ in members(pa))
    assert not any("a.tif" in n for n, _ in members(pb))
    # A figure save of one never carries the other's section.
    save(pb)
    assert read(pb)["manifest"]["docUid"] == "du_b"
    assert read(pa)["manifest"]["docUid"] == "du_a"
