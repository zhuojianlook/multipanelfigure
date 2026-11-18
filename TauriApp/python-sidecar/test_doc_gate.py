"""Regression tests: open document tabs must never swap or lose content while
the Analysis workspace collects sources from another tab.

0.1.419 bug: with two document tabs open, visiting Analysis and switching tabs
left tab 1 holding tab 2's images and vice versa. Analysis lists another tab's
sources by temporarily swapping the sidecar's builder globals to that tab;
ordinary routes (stash-state / activate-state for tab switches, config writes)
ran inside that window and read or wrote the wrong document. The doc gate
(api_server._DocGate) makes the swap exclusive.

In-process and deterministic: no server. Endpoints are called through the
REGISTERED route endpoints (what FastAPI's threadpool runs, gate wrapper
included), and the swap window is held open by pausing the source collector.
Each race test fails on the pre-gate code (the intruding request completes
inside the window and corrupts state) and passes with the gate (it waits).
"""

import base64
import io
import os
import sys
import tempfile
import threading

import pytest

if "api_server" not in sys.modules:
    # Importing the sidecar writes under ~/.multipanelfigure — keep tests off
    # the user's real settings.
    os.environ["HOME"] = tempfile.mkdtemp(prefix="mpf-test-home-")
    os.environ.setdefault("MPLBACKEND", "Agg")

import api_server as S  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from models import FigureConfig, _to_dict  # noqa: E402
from PIL import Image  # noqa: E402

RED, BLUE, GREEN = (255, 0, 0), (0, 0, 255), (0, 255, 0)
COLOR_DOC = {RED: "A", BLUE: "B", GREEN: "A-fig2"}
_ORIG_COLLECT = S._collect_analysis_insets


# ── fixtures / helpers ─────────────────────────────────────────────────────

def _fig_cfg(image_name):
    c = FigureConfig(rows=1, cols=1)
    c.ensure_grid()
    c.panels[0][0].image_name = image_name
    c.panels[0][0].include_in_analysis = True
    return c


def _state(image_name, color, extra_images=None):
    imgs = {image_name: Image.new("RGB", (32, 32), color)}
    imgs.update(extra_images or {})
    return {
        "cfg": _fig_cfg(image_name), "loaded_images": imgs, "hidden_images": set(),
        "min_dims": (32, 32), "custom_fonts": {}, "loaded_videos": {},
        "video_frames": {}, "loaded_zstacks": {}, "zstack_frames": {},
        "zstack_counts": {}, "channel_groups": {},
    }


def _setup(a_two_figures=False):
    """Doc A (red a.png) live, doc B (blue b.png) parked — the backend after
    the frontend did: open A, open B, switch back to A."""
    S._collect_analysis_insets = _ORIG_COLLECT
    S._doc_state_cache.clear()
    extra = {"a2.png": Image.new("RGB", (32, 32), GREEN)} if a_two_figures else None
    st_a = _state("a.png", RED, extra)
    if a_two_figures:
        st_a["cfg"].figures = [
            {"id": "f1", "name": "Fig 1"},
            {"id": "f2", "name": "Fig 2", "config": _to_dict(_fig_cfg("a2.png"))},
        ]
        st_a["cfg"].active_figure_id = "f1"
    S._apply_doc_state(st_a)
    S._doc_state_cache["B"] = _state("b.png", BLUE)


@pytest.fixture(autouse=True)
def _clean():
    _setup()
    yield
    S._collect_analysis_insets = _ORIG_COLLECT
    gate = getattr(S, "_doc_gate", None)
    if gate is not None:
        assert gate._writer is None and gate._readers == 0, "gate left held"


def ep(method, path):
    """The endpoint FastAPI actually invokes for this route."""
    return next(r.endpoint for r in S.app.routes
                if isinstance(r, APIRoute) and r.path == path and method in r.methods)


def stash(doc):
    return ep("POST", "/api/project/stash-state")(body=S.DocStateRequest(doc_id=doc))


def activate(doc):
    return ep("POST", "/api/project/activate-state")(body=S.DocStateRequest(doc_id=doc))


def switch_to(target, current):
    """projectNav.switchToDocument: stash the outgoing doc, activate the target."""
    stash(current)
    return activate(target)


def for_doc(doc):
    return lambda: S.list_inset_sources_for_doc(S.DocStateRequest(doc_id=doc))


def live_images():
    return sorted(S.loaded_images)


def stash_images(doc):
    st = S._doc_state_cache.get(doc)
    return None if st is None else sorted(st["loaded_images"])


def owners(resp):
    """Which doc's PIXELS each analysis source shows (thumbnail colour)."""
    out = []
    for s in resp.get("sources", []):
        px = Image.open(io.BytesIO(base64.b64decode(s["thumbnail"]))).convert("RGB").getpixel((2, 2))
        out.append(COLOR_DOC.get(px, str(px)))
    return out


def race(collect_fn, intruder_fn, pause_on_call=1):
    """Run collect_fn on a worker and park it inside its swap window (the Nth
    _collect_analysis_insets call on that thread); run intruder_fn on a second
    thread meanwhile; then release. Returns (intruder result, collect result,
    whether the intruder was still blocked while the window was open)."""
    entered, release = threading.Event(), threading.Event()
    state = {"n": 0, "tid": None}

    def gated_collect(*a, **k):
        if threading.get_ident() == state["tid"]:
            state["n"] += 1
            if state["n"] == pause_on_call:
                entered.set()
                assert release.wait(20), "swap window never released"
        return _ORIG_COLLECT(*a, **k)

    S._collect_analysis_insets = gated_collect
    wbox, ibox = {}, {}

    def worker():
        state["tid"] = threading.get_ident()
        try:
            wbox["r"] = collect_fn()
        except BaseException as e:  # pragma: no cover
            wbox["e"] = e

    def intruder():
        try:
            ibox["r"] = intruder_fn()
        except BaseException as e:  # pragma: no cover
            ibox["e"] = e

    w = threading.Thread(target=worker, daemon=True)
    w.start()
    assert entered.wait(20), "worker never reached the swap window"
    i = threading.Thread(target=intruder, daemon=True)
    i.start()
    i.join(0.3)
    blocked = i.is_alive()
    release.set()
    w.join(20)
    i.join(20)
    S._collect_analysis_insets = _ORIG_COLLECT
    assert not w.is_alive() and not i.is_alive(), "deadlock"
    for box in (wbox, ibox):
        if "e" in box:
            raise box["e"]
    return ibox.get("r"), wbox.get("r"), blocked


# ── the reported bug: tab switch during Analysis ───────────────────────────

def test_tab_switch_during_analysis_listing_keeps_each_tab_its_own_content():
    """User sequence: Analysis lists parked tab B while the user clicks tab B
    (stash A ; activate B). Pre-gate: the tabs end up exchanged for good."""
    act, _, blocked = race(for_doc("B"), lambda: switch_to("B", "A"))
    assert blocked, "tab switch ran inside another document's swap window"
    assert act["image_names"] == ["b.png"]
    assert live_images() == ["b.png"]                  # tab B shows B
    assert stash_images("A") == ["a.png"]              # tab A parked with A
    # Analysis now: active (B) lists B's pixels, parked A lists A's.
    assert owners(S.list_inset_analysis_sources()) == ["B"]
    assert owners(for_doc("A")()) == ["A"]
    # And every later switch stays straight.
    assert switch_to("A", "B")["image_names"] == ["a.png"]
    assert switch_to("B", "A")["image_names"] == ["b.png"]


def test_activate_inside_window_does_not_lose_parked_doc():
    """Variant: only activate-state(B) lands in the window. Pre-gate B's state
    was popped, then overwritten by the swap's restore — B gone for good."""
    stash("A")
    race(for_doc("B"), lambda: activate("B"))
    assert live_images() == ["b.png"]
    assert stash_images("A") == ["a.png"]
    assert owners(S.list_inset_analysis_sources()) == ["B"]
    assert owners(for_doc("A")()) == ["A"]


def test_tab_switch_during_listing_of_a_saved_tab_from_disk():
    tmp = tempfile.mkdtemp(prefix="mpf-gate-")
    path_a = os.path.join(tmp, "docA.mpf")
    ep("POST", "/api/project/save")(body=S.ProjectSaveRequest(path=path_a))
    race(for_doc("B"), lambda: switch_to("B", "A"))
    assert owners(S.list_inset_analysis_sources()) == ["B"]
    assert owners(S.list_inset_sources_for(S.InsetSourcesForRequest(project_path=path_a))) == ["A"]
    assert live_images() == ["b.png"]
    assert switch_to("A", "B")["image_names"] == ["a.png"]


# ── each kind of intruding request ─────────────────────────────────────────

def test_stash_during_stashed_doc_listing():
    _, _, blocked = race(for_doc("B"), lambda: stash("A"))
    assert blocked
    assert stash_images("A") == ["a.png"]
    assert S._doc_state_cache["A"]["cfg"].panels[0][0].image_name == "a.png"


def test_stash_during_disk_listing():
    tmp = tempfile.mkdtemp(prefix="mpf-gate-")
    path_b = os.path.join(tmp, "docB.mpf")
    switch_to("B", "A")
    ep("POST", "/api/project/save")(body=S.ProjectSaveRequest(path=path_b))
    switch_to("A", "B")
    assert live_images() == ["a.png"]
    _, _, blocked = race(
        lambda: S.list_inset_sources_for(S.InsetSourcesForRequest(project_path=path_b)),
        lambda: stash("A"))
    assert blocked
    assert stash_images("A") == ["a.png"]


def test_config_put_during_swap_is_kept():
    edited = S.get_config()
    edited["spacing"] = 0.5
    resp, _, _ = race(for_doc("B"), lambda: ep("PUT", "/api/config")(body=S.ConfigUpdate(config=edited)))
    assert resp["spacing"] == 0.5
    assert S.get_config()["spacing"] == 0.5


def test_panel_patch_during_swap_lands_in_the_live_doc():
    panel = S.get_config()["panels"][0][0]
    panel["brightness"] = 1.7
    race(for_doc("B"), lambda: ep("PATCH", "/api/config/panel/{r}/{c}")(r=0, c=0, body=S.PanelUpdate(panel=panel)))
    assert S.cfg.panels[0][0].image_name == "a.png"
    assert S.cfg.panels[0][0].brightness == 1.7
    parked = S._doc_state_cache["B"]["cfg"].panels[0][0]
    assert (parked.image_name, parked.brightness) == ("b.png", 1.0)


def test_config_get_during_swap_returns_the_live_doc():
    resp, _, blocked = race(for_doc("B"), lambda: ep("GET", "/api/config")())
    assert blocked
    assert resp["panels"][0][0]["image_name"] == "a.png"


def test_stash_during_figure_tab_listing_keeps_figure_tabs():
    _setup(a_two_figures=True)
    _, listing, blocked = race(S.list_inset_analysis_sources, lambda: stash("A"), pause_on_call=2)
    assert blocked
    c = S._doc_state_cache["A"]["cfg"]
    assert c.panels[0][0].image_name == "a.png"
    assert [f.get("id") for f in c.figures] == ["f1", "f2"]
    assert sorted(owners(listing)) == ["A", "A-fig2"]


def test_single_figure_listing_does_not_revert_a_concurrent_config_put():
    """The active-doc listing of a single-figure doc has nothing to swap; it
    must neither block the builder nor restore a stale cfg over a PUT."""
    edited = S.get_config()
    edited["spacing"] = 0.4
    resp, _, blocked = race(S.list_inset_analysis_sources,
                            lambda: ep("PUT", "/api/config")(body=S.ConfigUpdate(config=edited)))
    assert not blocked, "builder blocked by a listing that needs no swap"
    assert resp["spacing"] == 0.4 and S.get_config()["spacing"] == 0.4


def test_sequential_control():
    for_doc("B")()
    assert switch_to("B", "A")["image_names"] == ["b.png"]
    assert switch_to("A", "B")["image_names"] == ["a.png"]


# ── the gate itself ─────────────────────────────────────────────────────────

def test_gate_semantics():
    g = S._DocGate()
    held = threading.Event()
    done = threading.Event()

    def reader():
        with g.shared():
            held.set()
            done.wait(5)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert held.wait(5)
    with g.shared():                                   # readers share
        pass
    got = threading.Event()

    def writer():
        with g.exclusive():
            got.set()

    w = threading.Thread(target=writer, daemon=True)
    w.start()
    assert not got.wait(0.2), "exclusive entered while a reader held the gate"
    done.set()
    assert got.wait(5)
    t.join(5)
    w.join(5)
    with g.exclusive():
        with g.exclusive():                            # re-entrant
            with g.shared():                           # shared nests in exclusive
                assert not g.swapped_elsewhere()
    with g.shared():
        with pytest.raises(RuntimeError):
            with g.exclusive():                        # upgrade refused, not a deadlock
                pass
    assert g._writer is None and g._readers == 0


def test_project_context_parked_figure_of_active_doc():
    _setup(a_two_figures=True)
    with S._project_context("", "", "f2"):
        assert S.cfg.panels[0][0].image_name == "a2.png"
    assert S.cfg.panels[0][0].image_name == "a.png"


def test_active_doc_extraction_does_not_block_on_readers():
    held, done = threading.Event(), threading.Event()

    def reader():
        with S._doc_gate.shared():
            held.set()
            done.wait(5)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        ok = threading.Event()

        def extract():
            with S._source_ctx({}):
                ok.set()

        x = threading.Thread(target=extract, daemon=True)
        x.start()
        assert ok.wait(2), "active-doc extraction waited for a reader"
        x.join(5)
    finally:
        done.set()
        t.join(5)


def test_ungated_stash_refuses_while_swapped():
    entered, release = threading.Event(), threading.Event()

    def hold():
        with S._doc_gate.exclusive():
            entered.set()
            release.wait(5)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert entered.wait(5)
    try:
        with pytest.raises(HTTPException) as e:
            S.stash_state(S.DocStateRequest(doc_id="A"))   # module name = unwrapped
        assert e.value.status_code == 503
    finally:
        release.set()
        t.join(5)


def test_every_route_is_classified():
    S._doc_gate_selfcheck()
    import inspect
    for r in S.app.routes:
        if not isinstance(r, APIRoute):
            continue
        in_sets = (r.path in S._GATE_EXEMPT) + (r.path in S._GATE_SELF)
        wrapped = getattr(r.endpoint, "_doc_gate", None) == "shared"
        assert in_sets + wrapped == 1, r.path
        if wrapped:
            assert not inspect.iscoroutinefunction(r.endpoint), r.path
    stash_route = next(r for r in S.app.routes
                       if isinstance(r, APIRoute) and r.path == "/api/project/stash-state")
    assert stash_route.dependant.body_params, "wrapper hid the request body"
