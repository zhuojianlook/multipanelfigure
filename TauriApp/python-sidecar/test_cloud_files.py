"""Files stored online only (Google Drive, Dropbox, OneDrive, iCloud).

On macOS a file its cloud app evicted to save space is a "dataless"
placeholder. A process may read one — the kernel then has the cloud app
download it — only when its I/O policy allows that; under the default policy
(which a sidecar spawned by the app gets) the read fails at once with
EDEADLK, "Resource deadlock avoided", and the tab showed "Load failed … may
have been moved or deleted". The sidecar opts in when it starts. A read that
still fails (cloud app not running, offline) is reported as a cloud download
failure (503), never as a missing (404) or broken file.
In-process, no server: endpoints are called through the registered routes.
"""

import ctypes
import errno
import os
import sys
import tempfile
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

if "api_server" not in sys.modules:
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
from test_doc_isolation import _clean, load  # noqa: E402,F401  (_clean is an autouse fixture)
from test_workflows_persist import read  # noqa: E402

IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS = 3, 0
OFF, ON = 1, 2


@pytest.mark.skipif(sys.platform != "darwin", reason="a macOS I/O policy")
def test_the_sidecar_may_download_online_only_files():
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    get = lambda: libc.getiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS)
    try:
        # As a sidecar spawned by the app starts: downloads not allowed.
        assert libc.setiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS, OFF) == 0
        assert get() == OFF
        S._allow_cloud_downloads()
        assert get() == ON
    finally:
        libc.setiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS, ON)


def _file(tmp_path):
    p = tmp_path / "DeepRed Staining.mpf"
    p.write_bytes(b"not read: the reader is patched")
    return str(p)


def _placeholder(_path):
    return SimpleNamespace(st_flags=S._SF_DATALESS, st_file_attributes=0, st_mtime_ns=0, st_size=1)


def _local(_path):
    return SimpleNamespace(st_flags=0, st_file_attributes=0, st_mtime_ns=0, st_size=1)


def _raise(err):
    def reader(*_a, **_k):
        raise err
    return reader


CASES = [
    # (stat before the read, the read's error) -> (status, detail starts with)
    (_local, OSError(errno.EDEADLK, "Resource deadlock avoided"), 503, S._CLOUD_READ_FAILED),
    (_placeholder, OSError(errno.EIO, "Input/output error"), 503, S._CLOUD_READ_FAILED),
    (_placeholder, OSError(errno.ETIMEDOUT, "Operation timed out"), 503, S._CLOUD_READ_FAILED),
    (_local, OSError(errno.EACCES, "Permission denied"), 500, "Could not read"),
    (_local, OSError(errno.ENOENT, "No such file or directory"), 404, "Project file not found"),
    (_local, zipfile.BadZipFile("File is not a zip file"), 400, "Not a readable .mpf"),
]


@pytest.mark.parametrize("stat, err, status, prefix", CASES)
def test_a_failed_load_says_why(tmp_path, monkeypatch, stat, err, status, prefix):
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", stat)
    monkeypatch.setattr(S, "load_project", _raise(err))
    live_before = S._live_doc_id
    with pytest.raises(HTTPException) as ei:
        load(path, "doc_cloud")
    assert ei.value.status_code == status
    assert str(ei.value.detail).startswith(prefix)
    if status == 503:
        assert path in ei.value.detail          # the frontend names the file and its cloud app
    assert S._live_doc_id == live_before        # nothing was replaced


@pytest.mark.parametrize("stat, err, status, prefix", CASES)
def test_a_failed_workflows_read_says_why(tmp_path, monkeypatch, stat, err, status, prefix):
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", stat)
    monkeypatch.setattr(S, "read_workflow_members", _raise(err))
    with pytest.raises(HTTPException) as ei:
        read(path)
    assert ei.value.status_code == status
    assert str(ei.value.detail).startswith(prefix)


def test_an_unexpected_error_is_not_dressed_up(tmp_path, monkeypatch):
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", _placeholder)
    monkeypatch.setattr(S, "load_project", _raise(ValueError("bad config")))
    with pytest.raises(ValueError):
        load(path, "doc_cloud")


class _UnreadableFile:
    """What an online-only file looks like to a process whose download
    failed: it opens and seeks, its first read fails."""
    def __init__(self, err):
        self.err, self.pos = err, 0

    def seek(self, offset, whence=0):
        self.pos = 4096 + offset if whence == 2 else offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, *_a):
        raise self.err


def test_zipfile_hiding_the_failed_read_still_says_cloud(tmp_path, monkeypatch):
    # zipfile reports the OSError of its first read as BadZipFile("File is
    # not a zip file"): the cloud failure must not read as a broken file.
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", _local)
    edeadlk = OSError(errno.EDEADLK, "Resource deadlock avoided")
    monkeypatch.setattr(S, "load_project", lambda _p: zipfile.ZipFile(_UnreadableFile(edeadlk)))
    with pytest.raises(HTTPException) as ei:
        load(path, "doc_cloud")
    assert ei.value.status_code == 503 and ei.value.detail.startswith(S._CLOUD_READ_FAILED)
    # A genuinely broken file is still reported as one.
    junk = tmp_path / "junk.mpf"
    junk.write_bytes(b"this is not a zip file at all" * 10)
    monkeypatch.undo()
    with pytest.raises(HTTPException) as ei:
        load(str(junk), "doc_junk")
    assert ei.value.status_code == 400 and "Not a readable .mpf" in ei.value.detail


def test_a_background_workflows_read_never_downloads(tmp_path, monkeypatch):
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", _placeholder)
    opened = []
    monkeypatch.setattr(S, "read_workflow_members", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError))
    with pytest.raises(HTTPException) as ei:
        ep_read(path, no_download=True)
    assert ei.value.status_code == 503 and ei.value.detail.startswith(S._CLOUD_ONLINE_ONLY)
    assert opened == []                               # the file was never opened
    # The same read for a document being opened downloads (reads) it.
    with pytest.raises(AssertionError):
        ep_read(path, no_download=False)
    assert len(opened) == 1


def test_a_local_file_is_read_by_a_background_read(tmp_path, monkeypatch):
    path = _file(tmp_path)
    monkeypatch.setattr(S, "_stat_or_none", _local)
    seen = []
    def reader(p, **_k):
        seen.append(p)
        raise zipfile.BadZipFile("File is not a zip file")
    monkeypatch.setattr(S, "read_workflow_members", reader)
    with pytest.raises(HTTPException) as ei:
        ep_read(path, no_download=True)
    assert seen == [path] and ei.value.status_code == 400


def ep_read(path, no_download):
    from test_doc_isolation import ep
    return ep("POST", "/api/project/workflows/read")(body=S.WorkflowsReadRequest(path=path, no_download=no_download))


def _gate_free() -> bool:
    """True when another thread can take the doc gate exclusively right now
    (so the caller holds no part of it)."""
    import threading
    ok = []

    def take():
        with S._doc_gate.exclusive():
            ok.append(True)
    t = threading.Thread(target=take, daemon=True)
    t.start()
    t.join(5)
    return ok == [True]


def test_collage_routes_read_the_file_before_taking_the_gate(tmp_path, monkeypatch):
    # A file stored online only downloads in that read: a tab switch (which
    # needs the gate exclusively) must not wait behind it.
    assert {"/api/collage/figure-elements", "/api/collage/render-figure",
            "/api/collage/decompose"} <= S._GATE_SELF
    path = _file(tmp_path)
    free_while_reading = []

    def loader(_p):
        free_while_reading.append(_gate_free())
        raise OSError(errno.EDEADLK, "Resource deadlock avoided")
    monkeypatch.setattr(S, "_stat_or_none", _placeholder)
    monkeypatch.setattr(S, "_load_collage_mpf", loader)
    from test_doc_isolation import ep
    for route, req in (("/api/collage/figure-elements", S.CollageFigureElementsRequest),
                       ("/api/collage/render-figure", S.CollageFigureRenderRequest),
                       ("/api/collage/decompose", S.CollageDecomposeRequest)):
        with pytest.raises(HTTPException) as ei:
            ep("POST", route)(body=req(project_path=path))
        assert ei.value.status_code == 503 and ei.value.detail.startswith(S._CLOUD_READ_FAILED)
    assert free_while_reading == [True, True, True]
