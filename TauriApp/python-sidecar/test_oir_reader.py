"""Tests for oir_reader.

Real-file tests use the user's FV-series samples and are skipped unless
OIR_SAMPLE_DIR points at a folder containing them (lab data is not committed).
Synthetic tests build .oir bytes with a small writer that follows the layout
documented in oir_reader.py, to cover what the real samples don't: several
channels, Z x T, 8-bit, out-of-order strips, a missing index, corrupt files.
"""

import os
import struct
import uuid

import numpy as np
import pytest

import oir_reader as R

SAMPLES = os.environ.get("OIR_SAMPLE_DIR")


# ── real samples ─────────────────────────────────────────────────

def _sample(name):
    if not SAMPLES:
        pytest.skip("set OIR_SAMPLE_DIR to run the real-file tests")
    p = os.path.join(SAMPLES, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not in OIR_SAMPLE_DIR")
    return p


def test_real_single_plane_magenta():
    im = R.read_oir(_sample("DeepRed_10uM_1.oir"))
    assert im.data.shape == (1, 1, 1, 800, 800) and im.data.dtype == np.uint16
    pl = im.data[0, 0, 0]
    assert (pl.min(), pl.max()) == (9, 4095)
    assert pl.mean() == pytest.approx(565.7, abs=0.05)
    (c,) = im.channels
    assert (c.name, c.color_hex, c.lut_name) == ("CH4", "#ff00ff", "Magenta")
    assert im.pixel_size_um == pytest.approx((1.590990257669732,) * 2, abs=1e-12)
    assert im.bit_count == 12 and im.magnification == 10.0 and im.z_step_um is None


def test_real_two_plane_zstack_red():
    im = R.read_oir(_sample("DeepRed-25uM.oir"))
    assert im.data.shape == (1, 2, 1, 800, 800)
    means = [im.data[0, z, 0].mean() for z in range(2)]
    assert means == pytest.approx([61.9, 72.0], abs=0.05)
    assert im.channels[0].name == "CH3" and im.channels[0].color_hex == "#ff0000"
    assert im.z_step_um == pytest.approx(2.2)


def test_real_bytes_and_path_agree():
    p = _sample("DeepRed-25uM_0001.oir")
    a, b = R.read_oir(p), R.read_oir(open(p, "rb").read())
    assert np.array_equal(a.data, b.data)
    assert a.data[0, 0, 0].max() == 3234


# ── synthetic writer ─────────────────────────────────────────────

def _xml(body: str) -> bytes:
    return ('<?xml version="1.0" encoding="UTF-8"?>' + body).encode()


def _doc(body: str) -> bytes:
    x = _xml(body)
    return struct.pack("<I", len(x)) + x


def _block(typ: int, payload: bytes) -> bytes:
    return struct.pack("<II", len(payload), typ) + payload


def make_oir(planes, *, width=6, height=4, depth=2, channels=None, colors=None,
             strip_rows=2, shuffle=False, with_index=True, dims_fmt="z{z:03d}t{t:03d}",
             px=0.5, drop_strip=None):
    """planes: dict[(t, z, c)] -> 2-D array (height, width). Returns .oir bytes."""
    C = 1 + max(c for (_, _, c) in planes)
    guids = [str(uuid.UUID(int=i + 1)) for i in range(C)]
    names = channels or [f"CH{i + 1}" for i in range(C)]
    colors = colors or [(1.0, 0.0, 0.0)] * C
    dtype = "<u2" if depth == 2 else "u1"

    blocks = []
    # Pixel strips, one plane at a time.
    items = list(planes.items())
    strips = []
    for (t, z, c), arr in items:
        raw = np.ascontiguousarray(arr, dtype=dtype).tobytes()
        step = width * depth * strip_rows
        for k, off in enumerate(range(0, len(raw), step)):
            if drop_strip == (t, z, c, k):
                continue
            uid = f"{dims_fmt.format(z=z, t=t)}_0_1_{guids[c]}_{k}".encode()
            data = raw[off:off + step]
            strips.append(_block(3, struct.pack("<III", off, len(data), len(uid)) + uid)
                          + _block(4, data))
    if shuffle:
        strips = strips[::-1]
    blocks.extend(strips)

    # Metadata bundle (type 0).
    chan_xml = "".join(
        f'<commonphase:channel id="{g}" order="{i + 1}"><commonphase:name>{n}</commonphase:name>'
        f"<commonphase:length><commonparam:x>{px}</commonparam:x><commonparam:y>{px}</commonparam:y>"
        f"<commonparam:z>1.5</commonparam:z></commonphase:length>"
        f"<commonphase:pixelUnit><commonphase:x>MICRO_METER</commonphase:x>"
        f"<commonphase:y>MICRO_METER</commonphase:y><commonphase:z>MICRO_METER</commonphase:z>"
        f"</commonphase:pixelUnit></commonphase:channel>"
        for i, (g, n) in enumerate(zip(guids, names)))
    bundle = _doc(f"<lsmimage:imageProperties>{chan_xml}"
                  f"<opticalelement:magnification>20.0</opticalelement:magnification>"
                  f"</lsmimage:imageProperties>")
    for g, (r, gr, b) in zip(guids, colors):
        lut = _xml(f"<lut:LUT><lut:name>X</lut:name>"
                   f"<lut:red><lut:contrast>{r}</lut:contrast></lut:red>"
                   f"<lut:green><lut:contrast>{gr}</lut:contrast></lut:green>"
                   f"<lut:blue><lut:contrast>{b}</lut:contrast></lut:blue></lut:LUT>")
        bundle += struct.pack("<I", 36) + g.encode() + struct.pack("<I", len(lut)) + lut
    bundle += _doc(f"<base:imageDefinition><base:width>{width}</base:width>"
                   f"<base:height>{height}</base:height><base:depth>{depth}</base:depth>"
                   f"<base:bitCounts>{8 * depth}</base:bitCounts></base:imageDefinition>")
    blocks.append(_block(0, bundle))

    body = b"".join(blocks)
    offsets, pos = [], R.HEADER_LEN
    while pos < R.HEADER_LEN + len(body):
        length, _ = struct.unpack_from("<II", body, pos - R.HEADER_LEN)
        offsets.append(pos)
        pos += 8 + length
    idx_off = R.HEADER_LEN + len(body)
    index = struct.pack("<i", -1) + struct.pack(f"<{len(offsets)}Q", *offsets)
    total = idx_off + (len(index) if with_index else 0)
    header = (R.MAGIC + struct.pack("<4I", 12, 0, 1, 2)
              + struct.pack("<QQQQQ", total, idx_off if with_index else 0,
                            len(offsets) if with_index else 0, 1, 0)
              + b"FLUOVIEW" + struct.pack("<II", 3, 2) + b"\xff" * 8)
    assert len(header) == R.HEADER_LEN
    return header + body + (index if with_index else b"")


def _rand(shape, hi, seed):
    return np.random.default_rng(seed).integers(0, hi, shape)


# ── synthetic tests ──────────────────────────────────────────────

def test_multichannel_order_and_colors():
    a, b, c = (_rand((4, 6), 4096, s) for s in (1, 2, 3))
    buf = make_oir({(1, 1, 0): a, (1, 1, 1): b, (1, 1, 2): c},
                   colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    im = R.read_oir(buf)
    assert im.data.shape == (1, 1, 3, 4, 6)
    for i, arr in enumerate((a, b, c)):
        assert np.array_equal(im.data[0, 0, i], arr)
    assert [ch.color_hex for ch in im.channels] == ["#ff0000", "#00ff00", "#0000ff"]
    assert [ch.name for ch in im.channels] == ["CH1", "CH2", "CH3"]
    assert im.pixel_size_um == (0.5, 0.5) and im.magnification == 20.0


def test_z_by_t_placement():
    planes = {(t, z, 0): _rand((4, 6), 4096, 10 * t + z) for t in (1, 2, 3) for z in (1, 2)}
    im = R.read_oir(make_oir(planes))
    assert im.data.shape == (3, 2, 1, 4, 6)
    for (t, z, _), arr in planes.items():
        assert np.array_equal(im.data[t - 1, z - 1, 0], arr)
    assert im.z_step_um == 1.5


def test_eight_bit():
    a = _rand((4, 6), 256, 7)
    im = R.read_oir(make_oir({(1, 1, 0): a}, depth=1))
    assert im.data.dtype == np.uint8 and np.array_equal(im.data[0, 0, 0], a)


def test_strips_out_of_order_and_no_index():
    a = _rand((4, 6), 4096, 4)
    for kw in ({"shuffle": True}, {"with_index": False}, {"shuffle": True, "with_index": False}):
        im = R.read_oir(make_oir({(1, 1, 0): a}, **kw))
        assert np.array_equal(im.data[0, 0, 0], a), kw


@pytest.mark.parametrize("mutate, msg", [
    (lambda b: b"NOTANOLYMPUSFILE" + b[16:], "not an Olympus"),
    (lambda b: b[:len(b) // 2], "truncated"),
    (lambda b: b[:40], "not an Olympus"),
])
def test_corrupt_files_raise_clear_errors(mutate, msg):
    buf = make_oir({(1, 1, 0): _rand((4, 6), 4096, 5)})
    with pytest.raises(R.OirError, match=msg):
        R.read_oir(mutate(buf))


def test_missing_strip_is_an_error_not_a_black_band():
    buf = make_oir({(1, 1, 0): _rand((4, 6), 4096, 6)}, drop_strip=(1, 1, 0, 1))
    with pytest.raises(R.OirError, match="incomplete"):
        R.read_oir(buf)


def test_missing_plane_is_an_error():
    # Channel 2 has a plane only at z=1 while channel 1 has z=1 and z=2.
    planes = {(1, 1, 0): _rand((4, 6), 9, 1), (1, 2, 0): _rand((4, 6), 9, 2),
              (1, 1, 1): _rand((4, 6), 9, 3)}
    with pytest.raises(R.OirError, match="incomplete"):
        R.read_oir(make_oir(planes))


def test_unknown_dimension_is_refused_not_guessed():
    planes = {(1, 1, 0): _rand((4, 6), 9, 1), (1, 2, 0): _rand((4, 6), 9, 2)}
    # Encode the second axis as 'l' (e.g. lambda) instead of 'z'.
    with pytest.raises(R.OirError, match="unsupported image dimension"):
        R.read_oir(make_oir(planes, dims_fmt="l{z:03d}t{t:03d}"))
