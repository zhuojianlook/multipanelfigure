"""A .mpf keeps its identity when the user renames or moves the file.

Every save writes the document's id ("du_…") into a first, stored `doc.json`
member; a Save As copy gets a new id in the same write; a document whose
previous file is gone (renamed or moved outside the app) keeps its id. The
workflows/ rewrite carries doc.json byte for byte (both strategies), the
workflows reader reports it, and /api/fs/mpf-resolve finds a document by id
when its path no longer holds it: candidates first, then a bounded,
non-recursive scan of a few folders that never opens a cloud placeholder.
In-process, no server: endpoints are called through the registered routes.
"""

import base64
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from types import SimpleNamespace

import pytest

if "api_server" not in sys.modules:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
from models import (  # noqa: E402
    DOC_META_MEMBER, FigureConfig, load_project, make_doc_meta, new_doc_uid, read_mpf_uid,
    save_project, valid_doc_uid,
)
from test_doc_isolation import (  # noqa: E402,F401  (_clean is an autouse fixture)
    _clean, ep, load, new_blank, put_panel, upload,
)
from test_workflows_persist import (  # noqa: E402
    append_members, figure_doc, man, member_bytes, members, png_bytes, read, temp_leftovers, write,
)

UID_SHAPE = re.compile(r"du_[0-9a-z]+")
T0 = 1_700_000_000 * 10**9                         # an mtime in ns (2023)


# ── helpers ─────────────────────────────────────────────────────────────────

def save(path, carry_from=None, doc_uid=None, analysis=None):
    return ep("POST", "/api/project/save")(body=S.ProjectSaveRequest(
        path=path, carry_from=carry_from, analysis=analysis, doc_uid=doc_uid))


def resolve(*refs, scan=True):
    return ep("POST", "/api/fs/mpf-resolve")(body=S.MpfResolveRequest(
        refs=[S.MpfRef(**r) for r in refs], scan=scan))


def one(ref, scan=True, file_id=False):
    """One ref's result; its `file_id` (an inode, different on every run) is
    left out unless asked for."""
    out = dict(resolve({"key": "k", **ref}, scan=scan)["results"]["k"])
    if not file_id:
        out.pop("file_id", None)
    return out


def doc_json(path):
    with zipfile.ZipFile(path) as z:
        info = z.infolist()[0]
        assert info.filename == DOC_META_MEMBER
        assert info.compress_type == zipfile.ZIP_STORED
        return json.loads(z.read(info)), info


def mk(path, uid=None, manifest_uid=None, pad=0):
    """A small .mpf straight from the model layer: doc.json with `uid`
    (none when None), optionally a workflows manifest carrying `manifest_uid`,
    and `pad` bytes of image data to change its size."""
    path = str(path)
    save_project(FigureConfig(), {"pad.bin": b"x" * pad} if pad else {}, path,
                 doc_meta=make_doc_meta(uid) if uid else None)
    if manifest_uid:
        append_members(path, [("workflows/manifest.json", json.dumps(man(uid=manifest_uid)))])
    return path


def strip_doc_json(path):
    """What an older build's re-save leaves: the file without doc.json."""
    tmp = path + ".strip"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename != DOC_META_MEMBER:
                dst.writestr(info, src.read(info))
    os.replace(tmp, path)


def set_mtime(path, mtime_ns):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, mtime_ns))


@pytest.fixture(autouse=True)
def _fresh_uid_cache():
    S._mpf_uid_cache.clear()
    yield
    S._mpf_uid_cache.clear()


# ── ids ─────────────────────────────────────────────────────────────────────

def test_new_ids_are_unique_and_shaped_like_the_frontends():
    ids = {new_doc_uid() for _ in range(200)}
    assert len(ids) == 200
    assert all(UID_SHAPE.fullmatch(u) and valid_doc_uid(u) for u in ids)
    assert valid_doc_uid("du_a") and valid_doc_uid("du_a.b-c")
    for bad in (None, "", 5, "a b", "du_a\n", "x" * 129, "../x"):
        assert not valid_doc_uid(bad), bad


def test_read_mpf_uid_never_raises(tmp_path):
    junk = tmp_path / "junk.mpf"
    junk.write_bytes(b"not a zip")
    assert read_mpf_uid(str(junk)) is None
    assert read_mpf_uid(str(tmp_path / "gone.mpf")) is None
    p = mk(tmp_path / "bad-meta.mpf")
    append_members(p, [(DOC_META_MEMBER, "{nope")])
    assert read_mpf_uid(p) is None


# ── the figure save writes doc.json ─────────────────────────────────────────

def test_save_writes_doc_json_first_and_stored_with_the_requested_id(tmp_path):
    new_blank("A")
    upload("a.png", png_bytes())
    put_panel(0, 0, "a.png")
    p = str(tmp_path / "a.mpf")
    r = save(p, carry_from="", doc_uid="du_tab1")
    assert r["ok"] and r["doc_uid"] == "du_tab1" and r["uid_minted"] is False
    meta, _ = doc_json(p)
    assert meta["format"] == "mpf-doc" and meta["version"] == 1 and meta["uid"] == "du_tab1"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["created"])
    assert read_mpf_uid(p) == "du_tab1"
    assert [n for n, _ in members(p)][:2] == ["doc.json", "config.json"]
    cfg, imgs, *_ = load_project(p)
    assert "a.png" in imgs and cfg.panels[0][0].image_name == "a.png"


def test_save_without_an_id_mints_one_then_keeps_it(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    r = save(p)
    uid = r["doc_uid"]
    assert UID_SHAPE.fullmatch(uid) and r["uid_minted"] is True
    first = member_bytes(p, DOC_META_MEMBER)
    r2 = save(p)                                    # a request without an id
    assert r2["doc_uid"] == uid
    assert member_bytes(p, DOC_META_MEMBER) == first    # same id, same "created"
    r3 = save(p, doc_uid=uid)
    assert r3["doc_uid"] == uid and r3["uid_minted"] is False
    assert member_bytes(p, DOC_META_MEMBER) == first


def test_plain_resave_writes_the_tabs_id(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    save(p, doc_uid="du_one")
    r = save(p, doc_uid="du_two")                   # e.g. the held rule gave the tab a new id
    assert r["doc_uid"] == "du_two" and r["uid_minted"] is False
    assert read_mpf_uid(p) == "du_two"


def test_invalid_requested_id_is_ignored(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    save(p, doc_uid="du_keep")
    r = save(p, doc_uid="not an id!")
    assert r["doc_uid"] == "du_keep" and r["uid_minted"] is True


def test_legacy_file_with_only_the_manifest_id(tmp_path):
    p = figure_doc(tmp_path)
    write(p, man(uid="du_legacy"))
    strip_doc_json(p)                               # an older build re-saved it
    assert DOC_META_MEMBER not in [n for n, _ in members(p)]
    assert read_mpf_uid(p) == "du_legacy"
    out = read(p)
    assert out["doc_uid"] is None and out["manifest"]["docUid"] == "du_legacy"
    r = save(p)                                     # no id sent: the file's is kept
    assert r["doc_uid"] == "du_legacy"
    assert doc_json(p)[0]["uid"] == "du_legacy"
    strip_doc_json(p)
    r = save(p, doc_uid="du_legacy")
    assert r["doc_uid"] == "du_legacy" and r["uid_minted"] is False
    assert read(p)["doc_uid"] == "du_legacy"


def test_save_as_with_the_source_present_gets_a_new_id(tmp_path):
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    write(p, man(uid=uid), {"t": {"kind": "csv", "data": "a\n"}})
    p_bytes = open(p, "rb").read()
    q = str(tmp_path / "copy.mpf")
    r = save(q, carry_from=p, doc_uid=uid)
    assert r["uid_minted"] is True and r["doc_uid"] != uid and UID_SHAPE.fullmatch(r["doc_uid"])
    assert r["workflows_carried"] is True
    assert open(p, "rb").read() == p_bytes          # the original is untouched
    out = read(q)
    assert out["doc_uid"] == r["doc_uid"] and out["manifest"]["docUid"] == uid
    assert read_mpf_uid(q) == r["doc_uid"] and read_mpf_uid(p) == uid
    # The same without an id in the request.
    q2 = str(tmp_path / "copy2.mpf")
    r2 = save(q2, carry_from=p)
    assert r2["doc_uid"] not in (uid, r["doc_uid"])


def test_save_as_from_an_id_less_source_is_a_copy(tmp_path):
    p = figure_doc(tmp_path)
    strip_doc_json(p)
    assert read_mpf_uid(p) is None
    r = save(str(tmp_path / "copy.mpf"), carry_from=p, doc_uid="du_tab")
    assert r["doc_uid"] != "du_tab" and r["uid_minted"] is True


def test_save_as_from_a_file_holding_another_id_keeps_the_tabs(tmp_path):
    """A Finder copy opened second got a fresh id in memory; its Save As
    writes that id (the file it came from belongs to the original)."""
    p = figure_doc(tmp_path)
    r = save(str(tmp_path / "copy.mpf"), carry_from=p, doc_uid="du_fresh")
    assert r["doc_uid"] == "du_fresh" and r["uid_minted"] is False


def test_carry_from_the_same_file_is_not_a_copy(tmp_path):
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    assert save(p, carry_from=p, doc_uid=uid)["doc_uid"] == uid
    link = str(tmp_path / "link.mpf")
    os.symlink(p, link)
    assert save(p, carry_from=link, doc_uid=uid)["doc_uid"] == uid


def test_save_to_a_new_path_with_the_source_gone_keeps_the_id(tmp_path):
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    os.rename(p, str(tmp_path / "renamed-outside.mpf"))
    q = str(tmp_path / "chosen.mpf")
    r = save(q, carry_from=p, doc_uid=uid)
    assert r["doc_uid"] == uid and r["uid_minted"] is False and r["workflows_carried"] is False
    assert read_mpf_uid(q) == uid
    assert r["source_kept"] is False                # moved: its labels move with it


def test_source_kept_says_whether_the_previous_file_is_another_one_still_there(tmp_path):
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    assert save(str(tmp_path / "copy.mpf"), carry_from=p, doc_uid=uid)["source_kept"] is True
    # A file holding another id (a Finder copy's tab keeps its own): still a
    # copy of a file that is there, whatever id the new file gets.
    r = save(str(tmp_path / "copy2.mpf"), carry_from=p, doc_uid="du_fresh")
    assert r["doc_uid"] == "du_fresh" and r["source_kept"] is True
    assert save(p, doc_uid=uid)["source_kept"] is False           # a plain re-save
    assert save(p, carry_from=p, doc_uid=uid)["source_kept"] is False
    new_blank("U")
    assert save(str(tmp_path / "u.mpf"), carry_from="", doc_uid="du_u")["source_kept"] is False


def test_a_volume_without_inode_numbers_never_counts_two_files_as_one(tmp_path, monkeypatch):
    """st_ino 0 (some SMB / WebDAV / cloud drives on Windows): samestat would
    call any two files on the volume one file — a Save As over another file
    would keep the original's id and count as a move."""
    p = mk(tmp_path / "a.mpf", "du_a")
    q = mk(tmp_path / "other.mpf", "du_other")
    link = str(tmp_path / "link.mpf")
    os.symlink(p, link)
    real_stat = os.stat

    class NoIno:
        st_ino = 0

        def __init__(self, st):
            self._st = st

        def __getattr__(self, k):
            return getattr(self._st, k)

    monkeypatch.setattr(S.os, "stat", lambda path, *a, **kw: NoIno(real_stat(path, *a, **kw)))
    assert S._same_file(p, q) is False
    assert S._same_file(p, link) is True and S._same_file(p, p) is True
    assert S._file_id(p) is None
    # A Save As from a.mpf over other.mpf: a copy (a new id), not a re-save of
    # other.mpf under a.mpf's id.
    uid, minted, _meta = S._save_identity("du_a", p, q, p)
    assert uid != "du_a" and minted is True


def test_writes_and_reads_report_the_file_they_left(tmp_path):
    """file_id: the physical file (dev:inode) — every write replaces it."""
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    before = S._file_id(p)
    assert before and read(p)["file_id"] == before
    r = save(p, doc_uid=uid)
    assert r["file_id"] == S._file_id(p) != before
    w = write(p, man(uid=uid))
    assert w["file_id"] == S._file_id(p) != r["file_id"]


def test_without_an_id_over_another_file_the_previous_one_gone_gets_a_new_id(tmp_path):
    """A tab whose file was never read (no id sent) and whose file is gone
    is saved over an unrelated file: the content is not that document, so
    it must not take that file's id."""
    other = mk(tmp_path / "other.mpf", "du_other")
    new_blank("A")
    r = save(other, carry_from=str(tmp_path / "gone.mpf"))
    assert r["doc_uid"] != "du_other" and r["uid_minted"] is True
    assert read_mpf_uid(other) == r["doc_uid"]
    # A re-save without an id still keeps the file's (carry_from omitted, or
    # naming the file itself).
    kept = r["doc_uid"]
    assert save(other)["doc_uid"] == kept
    assert save(other, carry_from=other)["doc_uid"] == kept
    # carry_from "" (a document that never had a file) over another file: new.
    assert save(other, carry_from="")["doc_uid"] != kept


def test_the_file_replaced_is_opened_only_for_an_id_it_can_keep(tmp_path, monkeypatch):
    """Its doc.json is read only to keep "created": never on a Save As copy
    (the id is new), never on a cloud placeholder the save doesn't read
    anyway (opening one downloads it); a plain re-save still keeps it."""
    p = figure_doc(tmp_path)
    uid = read_mpf_uid(p)
    target = mk(tmp_path / "target.mpf", "du_target")
    opened = []
    real_zip = S._zipfile.ZipFile

    def spy(f, *a, **kw):
        if isinstance(f, str):
            opened.append(os.path.basename(f))
        return real_zip(f, *a, **kw)

    monkeypatch.setattr(S._zipfile, "ZipFile", spy)
    r = save(target, carry_from=p, doc_uid=uid)              # Save As over target.mpf
    assert r["uid_minted"] is True and opened         # the source is read (the carry)
    assert "target.mpf" not in opened
    # A placeholder: not opened, whether an Untitled save or a Save As
    # keeping the tab's id lands on it.
    target2 = mk(tmp_path / "evicted.mpf", "du_evicted")
    evicted = lambda st: st.st_ino == os.stat(target2).st_ino     # noqa: E731 (whatever file is there now)
    monkeypatch.setattr(S, "_is_cloud_placeholder", evicted)
    opened.clear()
    new_blank("U")
    assert save(target2, carry_from="", doc_uid="du_evicted")["doc_uid"] == "du_evicted"
    assert "evicted.mpf" not in opened
    assert save(target2, carry_from=p, doc_uid="du_held")["doc_uid"] == "du_held"
    assert "evicted.mpf" not in opened
    # A plain re-save reads its own file anyway (the carry): "created" kept.
    monkeypatch.setattr(S, "_is_cloud_placeholder", lambda st: False)
    first = doc_json(p)[0]["created"]
    monkeypatch.setattr(S, "_is_cloud_placeholder", lambda st: st.st_ino == os.stat(p).st_ino)
    assert save(p, doc_uid=uid)["doc_uid"] == uid
    assert doc_json(p)[0]["created"] == first


def test_untitled_first_save_writes_the_tabs_id(tmp_path):
    p = figure_doc(tmp_path)                        # another document's file is overwritten
    new_blank("U")
    r = save(p, carry_from="", doc_uid="du_untitled")
    assert r["doc_uid"] == "du_untitled" and read_mpf_uid(p) == "du_untitled"


def test_carry_error_retry_still_writes_doc_json(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    with open(p, "wb") as f:
        f.write(b"not a zip")                       # a plain re-save of an unreadable file
    r = save(p, doc_uid="du_t")
    assert r["carry_error"] and r["doc_uid"] == "du_t"
    assert doc_json(p)[0]["uid"] == "du_t" and load_project(p)[0] is not None
    bad = str(tmp_path / "bad.mpf")
    with open(bad, "wb") as f:
        f.write(b"not a zip")
    q = str(tmp_path / "b.mpf")
    r = save(q, carry_from=bad, doc_uid="du_t")     # a Save As from it
    assert r["carry_error"] and r["doc_uid"] != "du_t"
    assert doc_json(q)[0]["uid"] == r["doc_uid"]
    assert temp_leftovers(q) == []


# ── the workflows/ section never disturbs doc.json ─────────────────────────

def test_workflows_write_keeps_doc_json_identical_on_both_strategies(tmp_path, monkeypatch):
    p = figure_doc(tmp_path, "fast.mpf")
    q = str(tmp_path / "full.mpf")
    with open(p, "rb") as a, open(q, "wb") as b:
        b.write(a.read())
    meta_bytes = member_bytes(p, DOC_META_MEMBER)
    with zipfile.ZipFile(p) as z:
        info0 = z.getinfo(DOC_META_MEMBER)
    for step in range(3):
        blobs = {"p1": {"kind": "png", "data": base64.b64encode(png_bytes((step, 0, 0))).decode()}}
        monkeypatch.setattr(S, "_WF_TAIL_REWRITE", True)
        assert write(p, man(str(step)), blobs)["fast_path"] is True
        monkeypatch.setattr(S, "_WF_TAIL_REWRITE", False)
        assert write(q, man(str(step)), blobs)["fast_path"] is False
        for f in (p, q):
            meta, info = doc_json(f)
            assert member_bytes(f, DOC_META_MEMBER) == meta_bytes, (step, f)
            assert (info.CRC, info.file_size, info.date_time) == \
                (info0.CRC, info0.file_size, info0.date_time)
        assert members(p) == members(q)
    # A figure save keeps it at the head, so the tail rewrite still applies.
    monkeypatch.setattr(S, "_WF_TAIL_REWRITE", True)
    save(p)
    assert member_bytes(p, DOC_META_MEMBER) == meta_bytes
    assert write(p, man("x"), {"p1": {"kind": "png"}})["fast_path"] is True


def write_as(path, manifest, doc_uid, blobs=None):
    return ep("POST", "/api/project/workflows/write")(body=S.WorkflowsWriteRequest(
        path=path, manifest=manifest, blobs=blobs or {}, uploads=[], doc_uid=doc_uid))


def test_workflows_write_under_another_id_rewrites_doc_json_with_it(tmp_path, monkeypatch):
    """A Finder copy opened beside its original keeps an id of its own (the
    held rule); its Analysis write makes the file that document, doc.json
    included — else reopening it alone would take the original's id."""
    monkeypatch.setattr(S, "_WF_TAIL_REWRITE", True)
    p = figure_doc(tmp_path)
    orig = read_mpf_uid(p)
    r = write_as(p, man("1", uid="du_copy"), "du_copy")
    assert r["ok"] and r["doc_uid"] == "du_copy" and r["fast_path"] is False
    meta, _ = doc_json(p)                           # still first and stored
    assert meta["uid"] == "du_copy" and meta["uid"] != orig
    assert [n for n, _ in members(p)].count(DOC_META_MEMBER) == 1
    assert read_mpf_uid(p) == "du_copy" and read(p)["doc_uid"] == "du_copy"
    cfg, imgs, *_ = load_project(p)
    assert "a.png" in imgs
    # The same id again: doc.json byte for byte, and the fast path.
    before = member_bytes(p, DOC_META_MEMBER)
    r = write_as(p, man("2", uid="du_copy"), "du_copy")
    assert r["doc_uid"] == "du_copy" and r["fast_path"] is True
    assert member_bytes(p, DOC_META_MEMBER) == before
    # Without doc_uid: left alone, and reported.
    assert write(p, man("3", uid="du_copy"))["doc_uid"] == "du_copy"
    assert member_bytes(p, DOC_META_MEMBER) == before
    # A file without doc.json is left without one (its id is the manifest's).
    strip_doc_json(p)
    r = write_as(p, man("4", uid="du_m"), "du_m")
    assert r["doc_uid"] is None and DOC_META_MEMBER not in [n for n, _ in members(p)]
    assert read_mpf_uid(p) == "du_m"
    assert temp_leftovers(p) == []


def write_expecting(path, manifest, expect_uid, doc_uid=None):
    return ep("POST", "/api/project/workflows/write")(body=S.WorkflowsWriteRequest(
        path=path, manifest=manifest, blobs={}, uploads=[], doc_uid=doc_uid, expect_uid=expect_uid))


def test_workflows_write_never_lands_in_another_documents_file(tmp_path):
    """The tab's file name now holds another document (renamed or replaced
    outside the app): 409 {other_document}, the file untouched — its doc.json
    is never re-identified either. A file naming no id still counts."""
    p = mk(tmp_path / "fig.mpf", "du_w", manifest_uid="du_w")
    before = open(p, "rb").read()
    for doc_uid in (None, "du_held"):
        with pytest.raises(S.HTTPException) as ei:
            write_expecting(p, man("1", uid=doc_uid or "du_a"), "du_a", doc_uid)
        assert ei.value.status_code == 409 and ei.value.detail == {"other_document": "du_w"}
        assert open(p, "rb").read() == before
    # Its id only in the manifest (an older build's re-save dropped doc.json).
    strip_doc_json(p)
    before = open(p, "rb").read()
    with pytest.raises(S.HTTPException) as ei:
        write_expecting(p, man("2"), "du_a")
    assert ei.value.detail == {"other_document": "du_w"} and open(p, "rb").read() == before
    # The expected document: written (and a held tab's id taken into doc.json).
    q = mk(tmp_path / "mine.mpf", "du_a")
    r = write_expecting(q, man("3", uid="du_held"), "du_a", "du_held")
    assert r["ok"] and r["doc_uid"] == "du_held" and read_mpf_uid(q) == "du_held"
    # A file naming no id: written.
    n = mk(tmp_path / "none.mpf")
    assert write_expecting(n, man("4"), "du_a")["ok"]
    assert temp_leftovers(p) == []


def test_workflows_read_reports_doc_uid_without_or_with_an_unsupported_manifest(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    save(p, doc_uid="du_r")
    out = read(p)
    assert out["manifest"] is None and out["doc_uid"] == "du_r"
    append_members(p, [("workflows/manifest.json",
                        json.dumps({"format": "mpf-workflows", "version": 9, "docUid": "du_r"}))])
    out = read(p)
    assert out["unsupported"] is True and out["manifest"] is None and out["doc_uid"] == "du_r"
    # The manifest's copy is still read, whatever its version.
    strip_doc_json(p)
    assert read(p)["doc_uid"] is None and read_mpf_uid(p) == "du_r"


# ── loading never depends on doc.json ──────────────────────────────────────

def test_load_and_snapshot_restore_with_and_without_doc_json(tmp_path):
    p = figure_doc(tmp_path)
    q = str(tmp_path / "legacy.mpf")
    with open(p, "rb") as a, open(q, "wb") as b:
        b.write(a.read())
    strip_doc_json(q)
    for f in (p, q):
        r = load(f, "X")
        assert "a.png" in r["image_names"]
        assert S.cfg.panels[0][0].image_name == "a.png"
    snap = ep("GET", "/api/project/snapshot")()
    blob = base64.b64decode(snap["blob"])
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert DOC_META_MEMBER not in z.namelist()  # snapshots are never user files
    restore = ep("POST", "/api/project/restore")
    for data in (blob, open(p, "rb").read()):
        new_blank("Y")
        r = restore(body=S.ProjectRestoreRequest(blob=base64.b64encode(data).decode()))
        assert "a.png" in r["image_names"]


# ── /api/fs/mpf-resolve ─────────────────────────────────────────────────────

def test_ok_when_the_path_holds_the_document(tmp_path):
    p = mk(tmp_path / "a.mpf", "du_a")
    assert one({"path": p, "uid": "du_a"}) == {"status": "ok", "path": p, "exists": True,
                                                "uid": "du_a"}


def test_renamed_in_the_same_folder_is_moved(tmp_path):
    p = mk(tmp_path / "a.mpf", "du_a")
    mk(tmp_path / "other.mpf", "du_o")
    q = str(tmp_path / "renamed.mpf")
    os.rename(p, q)
    assert one({"path": p, "uid": "du_a"}) == {"status": "moved", "path": q, "exists": False,
                                                "uid": "du_a"}


def test_moved_via_candidates_without_a_scan(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    q = mk(tmp_path / "y" / "b.mpf", "du_a")
    old = str(tmp_path / "x" / "a.mpf")
    assert one({"path": old, "uid": "du_a", "candidates": [str(tmp_path / "x" / "nope.mpf"), q]},
               scan=False)["path"] == q
    # Without the candidate and without a scan nothing is found.
    assert one({"path": old, "uid": "du_a"}, scan=False)["status"] == "missing"


def test_moved_via_extra_dirs(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    q = mk(tmp_path / "y" / "b.mpf", "du_a")
    old = str(tmp_path / "x" / "a.mpf")
    assert one({"path": old, "uid": "du_a"})["status"] == "missing"
    assert one({"path": old, "uid": "du_a", "dirs": [str(tmp_path / "y")]})["path"] == q


def test_missing_and_ambiguous(tmp_path):
    old = str(tmp_path / "a.mpf")
    assert one({"path": old, "uid": "du_a"}) == {"status": "missing", "path": None,
                                                  "exists": False, "uid": None}
    assert one({"path": old})["status"] == "missing"
    c1 = mk(tmp_path / "copy 1.mpf", "du_a")
    c2 = mk(tmp_path / "copy 2.mpf", "du_a")
    set_mtime(c1, T0)
    set_mtime(c2, T0 + 10**12)
    r = one({"path": old, "uid": "du_a"})
    assert r["status"] == "ambiguous" and r["path"] is None and r["matches"] == [c2, c1]
    # Not even a candidate among several hits is picked (a Finder copy and
    # its original: following the id into the wrong one writes one document
    # over the other).
    r = one({"path": old, "uid": "du_a", "candidates": [c1]})
    assert r["status"] == "ambiguous" and r["path"] is None and r["matches"] == [c2, c1]


def test_a_single_hit_is_followed_only_when_it_is_the_very_file(tmp_path):
    """file_id: a rename keeps the inode; a Finder / Dropbox copy of a file
    whose original is gone does not — never followed, the caller asks."""
    p = mk(tmp_path / "foo.mpf", "du_a")
    fid = one({"path": p, "uid": "du_a"}, file_id=True)["file_id"]
    assert fid == S._file_id(p)
    backup = str(tmp_path / "foo backup.mpf")
    shutil.copyfile(p, backup)                      # a copy: same bytes, same id
    os.unlink(p)                                    # the original moved away / trashed
    assert one({"path": p, "uid": "du_a", "file_id": fid}) == {
        "status": "ambiguous", "path": None, "exists": False, "uid": None, "matches": [backup]}
    # Without a known file_id: as before.
    assert one({"path": p, "uid": "du_a"})["status"] == "moved"
    # A real rename keeps the inode: followed.
    q = str(tmp_path / "renamed.mpf")
    os.rename(backup, q)
    bfid = S._file_id(q)
    r = one({"path": backup, "uid": "du_a", "file_id": bfid}, file_id=True)
    assert r == {"status": "moved", "path": q, "exists": False, "uid": "du_a", "file_id": bfid}


def test_among_several_hits_the_very_file_is_followed(tmp_path):
    """A backup copy left beside the original, then the original renamed:
    the renamed file keeps the inode the caller recorded — followed, not
    'ambiguous'. Without a known file_id, or one neither hit has: asked."""
    p = mk(tmp_path / "fig.mpf", "du_a")
    fid = S._file_id(p)
    copy = str(tmp_path / "fig copy.mpf")
    shutil.copyfile(p, copy)                        # a Finder duplicate: same id, new inode
    final = str(tmp_path / "fig_final.mpf")
    os.rename(p, final)                             # the original renamed: same inode
    assert one({"path": p, "uid": "du_a", "file_id": fid}, file_id=True) == {
        "status": "moved", "path": final, "exists": False, "uid": "du_a", "file_id": fid}
    assert one({"path": p, "uid": "du_a"})["status"] == "ambiguous"
    other = S._file_id(str(tmp_path))               # matches neither hit
    assert one({"path": p, "uid": "du_a", "file_id": other})["status"] == "ambiguous"


def test_path_with_another_id_moves_only_when_the_id_is_found(tmp_path):
    p = mk(tmp_path / "a.mpf", "du_b")              # the name was reused by another document
    assert one({"path": p, "uid": "du_a"}) == {"status": "ok", "path": p, "exists": True,
                                                "uid": "du_b"}
    q = mk(tmp_path / "renamed.mpf", "du_a")
    assert one({"path": p, "uid": "du_a"}) == {"status": "moved", "path": q, "exists": True,
                                                "uid": "du_a"}


def test_id_less_file_at_the_path_is_ok(tmp_path):
    p = mk(tmp_path / "legacy.mpf")
    mk(tmp_path / "elsewhere.mpf", "du_a")
    assert one({"path": p, "uid": "du_a"}) == {"status": "ok", "path": p, "exists": True,
                                                "uid": None}
    junk = tmp_path / "junk.mpf"
    junk.write_bytes(b"garbage")
    assert one({"path": str(junk), "uid": "du_a"})["status"] == "ok"


def test_manifest_only_id_is_found(tmp_path):
    q = mk(tmp_path / "renamed.mpf", manifest_uid="du_a")
    assert one({"path": str(tmp_path / "a.mpf"), "uid": "du_a"})["path"] == q
    assert one({"path": q})["uid"] == "du_a"


def test_ref_without_an_id_returns_the_files(tmp_path):
    p = mk(tmp_path / "a.mpf", "du_a")
    assert one({"path": p}) == {"status": "ok", "path": p, "exists": True, "uid": "du_a"}
    assert one({"path": p, "uid": "bad id!"})["uid"] == "du_a"   # an invalid id counts as none


def test_nulls_in_the_request_count_as_absent(tmp_path):
    q = mk(tmp_path / "renamed.mpf", "du_a")
    body = S.MpfResolveRequest(**json.loads(json.dumps({
        "refs": [{"key": "a", "path": str(tmp_path / "a.mpf"), "uid": "du_a",
                  "candidates": None, "dirs": None},
                 {"key": "b", "path": None, "uid": None}],
        "scan": None})))
    out = ep("POST", "/api/fs/mpf-resolve")(body=body)
    assert out["results"]["a"]["path"] == q
    assert out["results"]["b"] == {"status": "missing", "path": None, "exists": False, "uid": None}


def test_garbage_and_dot_files_are_skipped(tmp_path):
    (tmp_path / "junk.mpf").write_bytes(b"PK\x03\x04 broken")
    mk(tmp_path / ".hidden.mpf", "du_a")
    mk(tmp_path / ".a.mpf.0123456789ab.tmp", "du_a")
    mk(tmp_path / "not-an-mpf.zip", "du_a")
    assert one({"path": str(tmp_path / "a.mpf"), "uid": "du_a"})["status"] == "missing"


def test_a_symlink_to_the_same_file_counts_once(tmp_path):
    q = mk(tmp_path / "renamed.mpf", "du_a")
    os.symlink(q, str(tmp_path / "alias.mpf"))
    r = one({"path": str(tmp_path / "a.mpf"), "uid": "du_a"})
    assert r["status"] == "moved" and r["path"] == q


def test_a_hard_link_or_a_folder_spelled_in_another_case_counts_once(tmp_path):
    (tmp_path / "Folder").mkdir()
    q = mk(tmp_path / "Folder" / "bar.mpf", "du_a")
    # The same folder under two spellings (a case-insensitive volume — the
    # macOS default): the ref's own path in one, the extra folder in the other.
    lower = tmp_path / "folder"
    if os.path.isdir(str(lower)):
        r = one({"path": str(lower / "foo.mpf"), "uid": "du_a", "dirs": [str(tmp_path / "Folder")]})
        assert r["status"] == "moved", r
        assert r["path"].lower() == q.lower()
    # A second name for the same file (a hard link) is the same file.
    os.link(q, str(tmp_path / "Folder" / "baz.mpf"))
    r = one({"path": str(tmp_path / "Folder" / "foo.mpf"), "uid": "du_a"})
    assert r["status"] == "moved" and r["path"] in (q, str(tmp_path / "Folder" / "baz.mpf"))


def test_bare_name_resolves_in_documents(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    mk(home / "Documents" / "bare.mpf", "du_a")
    assert one({"path": "bare.mpf", "uid": "du_a"}) == {"status": "ok", "path": "bare.mpf",
                                                        "exists": True, "uid": "du_a"}
    os.rename(str(home / "Documents" / "bare.mpf"), str(home / "Documents" / "renamed.mpf"))
    r = one({"path": "bare.mpf", "uid": "du_a"})
    assert r["status"] == "moved" and r["path"] == str(home / "Documents" / "renamed.mpf")
    assert one({"path": "~/Documents/renamed.mpf"})["uid"] == "du_a"


def test_the_per_folder_cap_scans_the_newest_and_sets_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_RESOLVE_MAX_FILES", 3)
    for i in range(5):
        set_mtime(mk(tmp_path / f"o{i}.mpf", f"du_o{i}"), T0 + i * 10**12)
    target = mk(tmp_path / "t.mpf", "du_a")
    set_mtime(target, T0 - 10**12)                  # oldest: beyond the cap
    out = resolve({"key": "k", "path": str(tmp_path / "a.mpf"), "uid": "du_a"})
    assert out["results"]["k"]["status"] == "missing" and out["truncated"] is True
    set_mtime(target, T0 + 10**13)                  # newest: within it
    S._mpf_uid_cache.clear()
    out = resolve({"key": "k", "path": str(tmp_path / "a.mpf"), "uid": "du_a"})
    assert out["results"]["k"]["path"] == target and out["truncated"] is True
    monkeypatch.setattr(S, "_RESOLVE_MAX_FILES", 400)
    assert resolve({"key": "k", "path": str(tmp_path / "a.mpf"), "uid": "du_a"})["truncated"] is False


def test_time_budget_sets_truncated(tmp_path, monkeypatch):
    mk(tmp_path / "renamed.mpf", "du_a")
    monkeypatch.setattr(S, "_RESOLVE_BUDGET_S", -1.0)
    out = resolve({"key": "k", "path": str(tmp_path / "a.mpf"), "uid": "du_a"})
    assert out["results"]["k"]["status"] == "missing" and out["truncated"] is True


def test_folders_are_listed_once_per_request(tmp_path, monkeypatch):
    for i in range(3):
        mk(tmp_path / f"r{i}.mpf", f"du_{i}")
    calls = []
    real = S.os.scandir
    monkeypatch.setattr(S.os, "scandir", lambda p: calls.append(p) or real(p))
    out = resolve(*({"key": str(i), "path": str(tmp_path / f"gone{i}.mpf"), "uid": f"du_{i}"}
                    for i in range(3)))
    assert [out["results"][str(i)]["path"] for i in range(3)] == \
        [str(tmp_path / f"r{i}.mpf") for i in range(3)]
    assert len(calls) == 1


def test_peek_cache_hits_and_invalidates(tmp_path, monkeypatch):
    reads = []
    real = S.read_mpf_uid
    monkeypatch.setattr(S, "read_mpf_uid", lambda p: reads.append(p) or real(p))
    p = mk(tmp_path / "a.mpf", "du_aaaa")
    assert one({"path": p})["uid"] == "du_aaaa"
    assert one({"path": p})["uid"] == "du_aaaa"
    assert len(reads) == 1                          # second time from the cache
    old = os.stat(p).st_mtime_ns
    mk(p, "du_bbbb", pad=10)                        # other size, same mtime
    set_mtime(p, old)
    assert one({"path": p})["uid"] == "du_bbbb"
    size = os.stat(p).st_size
    mk(p, "du_cccc", pad=10)                        # same size, other mtime
    set_mtime(p, old + 1_000_000)
    assert os.stat(p).st_size == size
    assert one({"path": p})["uid"] == "du_cccc"


def test_our_own_writes_drop_the_cached_id(tmp_path):
    new_blank("A")
    p = str(tmp_path / "a.mpf")
    save(p, doc_uid="du_aaaa1")
    assert one({"path": p})["uid"] == "du_aaaa1"
    st = os.stat(p)
    save(p, doc_uid="du_bbbb2")                     # same length id: likely the same size
    set_mtime(p, st.st_mtime_ns)
    assert one({"path": p})["uid"] == "du_bbbb2"
    q = str(tmp_path / "legacy.mpf")
    save(q)
    strip_doc_json(q)
    write(q, man(uid="du_m1"))
    assert one({"path": q})["uid"] == "du_m1"
    st = os.stat(q)
    write(q, man(uid="du_m2"))                      # a workflows write sets a legacy file's id
    set_mtime(q, st.st_mtime_ns)
    assert one({"path": q})["uid"] == "du_m2"


def test_cloud_placeholders_are_never_opened(tmp_path, monkeypatch):
    assert S._is_cloud_placeholder(SimpleNamespace(st_flags=0x40000000))
    assert S._is_cloud_placeholder(SimpleNamespace(st_file_attributes=0x400000))
    assert S._is_cloud_placeholder(SimpleNamespace(st_file_attributes=0x1000))
    assert not S._is_cloud_placeholder(SimpleNamespace(st_flags=0, st_file_attributes=0x20))
    assert not S._is_cloud_placeholder(SimpleNamespace())
    cloud = mk(tmp_path / "renamed.mpf", "du_a")
    at_path = mk(tmp_path / "a.mpf", "du_b")
    placeholders = {os.stat(cloud).st_ino, os.stat(at_path).st_ino}
    monkeypatch.setattr(S, "_is_cloud_placeholder", lambda st: st.st_ino in placeholders)
    reads = []
    real = S.read_mpf_uid
    monkeypatch.setattr(S, "read_mpf_uid", lambda p: reads.append(p) or real(p))
    # A placeholder at the path is the document (its id is unknown).
    assert one({"path": at_path, "uid": "du_a"}) == {"status": "ok", "path": at_path,
                                                      "exists": True, "uid": None}
    assert one({"path": str(tmp_path / "gone.mpf"), "uid": "du_a",
                "candidates": [cloud]})["status"] == "missing"
    assert reads == []


def test_resolve_is_gate_exempt_and_runs_during_a_swap(tmp_path):
    S._doc_gate_selfcheck()
    assert "/api/fs/mpf-resolve" in S._GATE_EXEMPT
    assert getattr(ep("POST", "/api/fs/mpf-resolve"), "_doc_gate", None) is None
    p = mk(tmp_path / "renamed.mpf", "du_a")
    held, release = threading.Event(), threading.Event()

    def swapper():
        with S._doc_gate.exclusive():
            held.set()
            release.wait(10)

    t = threading.Thread(target=swapper, daemon=True)
    t.start()
    assert held.wait(5)
    out = {}
    w = threading.Thread(target=lambda: out.setdefault(
        "r", one({"path": str(tmp_path / "a.mpf"), "uid": "du_a"})), daemon=True)
    w.start()
    w.join(5)
    finished = not w.is_alive()
    release.set()
    t.join(5)
    w.join(5)
    assert finished and out["r"]["path"] == p


def test_resolve_never_touches_the_live_document(tmp_path):
    new_blank("A")
    upload("a.png", png_bytes())
    put_panel(0, 0, "a.png")
    S._doc_state_cache["B"] = S._blank_doc_state({})
    refs, cache, live = S._live_refs(), dict(S._doc_state_cache), S._live_doc_id
    mk(tmp_path / "renamed.mpf", "du_a")
    one({"path": str(tmp_path / "a.mpf"), "uid": "du_a"})
    after = S._live_refs()
    assert all(after[k] is refs[k] for k in refs) and S._live_doc_id == live == "A"
    assert S._doc_state_cache.keys() == cache.keys()


# ── collage cache ───────────────────────────────────────────────────────────

def test_collage_cache_notices_two_files_swapping_names(tmp_path):
    new_blank("A")
    upload("a.png", png_bytes((255, 0, 0), (8, 8)))
    put_panel(0, 0, "a.png")
    a = str(tmp_path / "a.mpf")
    save(a)
    new_blank("B")
    upload("b.png", png_bytes((0, 0, 255), (64, 64)))
    put_panel(0, 0, "b.png")
    b = str(tmp_path / "b.mpf")
    save(b)
    same = os.stat(a).st_mtime_ns
    set_mtime(b, same)
    _cfg, imgs = S._load_collage_mpf(a)
    assert list(imgs) == ["a.png"]
    tmp = str(tmp_path / "swap.mpf")
    os.rename(a, tmp)
    os.rename(b, a)
    os.rename(tmp, b)
    set_mtime(a, same)                              # a rename keeps mtime
    _cfg, imgs = S._load_collage_mpf(a)
    assert list(imgs) == ["b.png"]
