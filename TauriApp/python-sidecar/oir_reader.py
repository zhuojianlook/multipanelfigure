"""oir_reader — a small, pure-numpy reader for Olympus FluoView .oir files.

No Java (Bio-Formats can't be bundled into this PyInstaller sidecar) and no
new dependencies. The layout below was established by inspecting real
FV-series files, not copied from any other implementation. Everything is
little-endian.

  header (96 bytes)
    0   "OLYMPUSRAWFORMAT"
    32  u64 file size          40  u64 offset of the block index
    48  u64 number of blocks   (first block starts at 96)
  block index (at header@40): i32 -1, then nblocks x u64 absolute offsets
  every block: [u32 payload_len][u32 type][payload]
    type 3  pixel-strip header: u32 byte offset of the strip within its
            plane, u32 strip length, u32 uid length, uid (ASCII) — always
            followed immediately by a type-4 block holding the strip bytes
    type 4  raw pixel bytes of that strip (uncompressed)
    type 0  metadata bundle: XML documents, each preceded by its u32 length;
            rewritten after every plane, so the LAST one is current
    1 per-frame XML, 2 thumbnail, 5 separator (all ignored here)
  strip uid: "<dims>_<..>_<channel GUID>_<strip#>", e.g.
            "z002_0_1_872aadf8-...-8fb95721f688_14"; <dims> holds letter +
            number pairs (z001, t003, z002t003). "REF_..." strips are a
            reference image and are skipped.

Verified on real files: single-channel, Z = 1 and Z = 2, uint16 holding
12-bit data. NOT yet seen on real data: several channels, time series, 8-bit,
other dimension letters. The reader therefore refuses (OirError) anything it
cannot place unambiguously instead of guessing — a wrong channel/plane
assignment would silently corrupt a figure.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MAGIC = b"OLYMPUSRAWFORMAT"
HEADER_LEN = 96

_GUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_UID_RE = re.compile(r"(.+?)_(" + _GUID + r")_(\d+)")
# Pixel-length units we can convert to micrometres.
_UNIT_TO_UM = {"MICRO_METER": 1.0, "NANO_METER": 1e-3, "MILLI_METER": 1e3}


class OirError(ValueError):
    """A file that is not a readable .oir — the message is shown to the user."""


@dataclass
class OirChannel:
    id: str                         # channel GUID
    name: str                       # e.g. "CH3"
    order: int                      # acquisition order from the XML
    dye: Optional[str] = None       # e.g. "Alexa Fluor 594" (single-channel only)
    lut_name: Optional[str] = None  # e.g. "Red", "Magenta"
    color_hex: Optional[str] = None  # display tint, e.g. "#ff0000"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.dye})" if self.dye else self.name


@dataclass
class OirImage:
    data: np.ndarray                # (T, Z, C, Y, X), uint16 or uint8
    channels: List[OirChannel]
    width: int
    height: int
    bit_count: Optional[int] = None
    pixel_size_um: Optional[Tuple[float, float]] = None   # (x, y)
    z_step_um: Optional[float] = None
    magnification: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def read_oir(src) -> OirImage:
    """Read an .oir from bytes or a path. Raises OirError on anything it
    can't decode unambiguously."""
    buf = src if isinstance(src, (bytes, bytearray, memoryview)) else _read_path(src)
    buf = bytes(buf)
    try:
        return _read(buf)
    except OirError:
        raise
    except (struct.error, IndexError, ValueError, UnicodeDecodeError) as e:
        raise OirError(f"corrupt .oir file: {e}") from e


def _read_path(path) -> bytes:
    with open(os.fspath(path), "rb") as f:
        return f.read()


# ── container walk ───────────────────────────────────────────────

def _block_offsets(buf: bytes, idx_off: int, nblocks: int) -> List[int]:
    n = len(buf)
    if 0 < idx_off and nblocks > 0 and idx_off + 4 + 8 * nblocks <= n:
        offs = list(struct.unpack_from(f"<{nblocks}Q", buf, idx_off + 4))
        if all(HEADER_LEN <= o <= n - 8 for o in offs):
            return offs
    # No usable index: walk the blocks from the first one.
    offs, off = [], HEADER_LEN
    stop = idx_off if HEADER_LEN < idx_off <= n else n
    while off + 8 <= stop:
        length, typ = struct.unpack_from("<II", buf, off)
        if typ > 5 or off + 8 + length > n:
            break
        offs.append(off)
        off += 8 + length
    return offs


def _read(buf: bytes) -> OirImage:
    n = len(buf)
    if n < HEADER_LEN or buf[:16] != MAGIC:
        raise OirError("not an Olympus .oir file")
    size, idx_off, nblocks = struct.unpack_from("<QQQ", buf, 32)
    if size > n:
        raise OirError(f"file is truncated (header says {size} bytes, got {n})")

    # (t, z, channel GUID) -> [(offset in plane, data offset, length)]
    strips: Dict[Tuple[int, int, str], List[Tuple[int, int, int]]] = {}
    meta: Optional[Tuple[int, int]] = None
    for off in _block_offsets(buf, idx_off, nblocks):
        length, typ = struct.unpack_from("<II", buf, off)
        if off + 8 + length > n:
            raise OirError("file is truncated (a block runs past the end)")
        if typ == 0:
            meta = (off + 8, length)
        elif typ == 3:
            plane_off, dlen, uid_len = struct.unpack_from("<III", buf, off + 8)
            uid = buf[off + 20: off + 20 + uid_len].decode("ascii")
            nxt = off + 20 + uid_len
            if nxt + 8 > n or struct.unpack_from("<II", buf, nxt) != (dlen, 4):
                raise OirError("pixel data is missing or out of place")
            data_off = nxt + 8
            if data_off + dlen > n:
                raise OirError("file is truncated (pixel data runs past the end)")
            if uid.startswith("REF_"):
                continue
            t, z, guid = _parse_uid(uid)
            strips.setdefault((t, z, guid), []).append((plane_off, data_off, dlen))
    if meta is None:
        raise OirError("no image metadata found")
    if not strips:
        raise OirError("no image data found")

    info = _parse_metadata(buf[meta[0]: meta[0] + meta[1]])
    width, height, depth = info["width"], info["height"], info["depth"]
    if depth == 2:
        dtype = np.dtype("<u2")
    elif depth == 1:
        dtype = np.dtype("u1")
    else:
        raise OirError(f"unsupported sample depth ({depth} bytes)")

    # Channels = GUIDs that actually carry pixels, in acquisition order.
    present = sorted({g for (_, _, g) in strips})
    by_id = {c.id: c for c in info["channels"]}
    unknown = [g for g in present if g not in by_id]
    if unknown:
        raise OirError("pixel data refers to a channel missing from the metadata")
    chans = sorted((by_id[g] for g in present), key=lambda c: c.order)
    cidx = {c.id: i for i, c in enumerate(chans)}

    T = max(t for (t, _, _) in strips)
    Z = max(z for (_, z, _) in strips)
    C = len(chans)
    plane_bytes = width * height * depth
    out = np.zeros((T, Z, C, width * height), dtype=dtype)
    for (t, z, g), parts in strips.items():
        covered = 0
        flat = out[t - 1, z - 1, cidx[g]]
        for plane_off, data_off, dlen in parts:
            if plane_off % depth or dlen % depth or plane_off + dlen > plane_bytes:
                raise OirError("pixel data does not fit the image size")
            flat[plane_off // depth:(plane_off + dlen) // depth] = np.frombuffer(
                buf, dtype=dtype, count=dlen // depth, offset=data_off)
            covered += dlen
        if covered != plane_bytes:
            raise OirError("incomplete image data (a plane is missing pixels)")
    # Every (t, z, channel) plane must exist — no silent black planes.
    if len(strips) != T * Z * C:
        raise OirError("incomplete image data (some planes are missing)")

    if C == 1 and info["dyes"] and len(info["dyes"]) == 1:
        chans[0].dye = info["dyes"][0]
    return OirImage(
        data=out.reshape(T, Z, C, height, width),
        channels=chans, width=width, height=height,
        bit_count=info["bit_count"], pixel_size_um=info["pixel_size_um"],
        z_step_um=info["z_step_um"] if Z > 1 else None,
        magnification=info["magnification"],
    )


def _parse_uid(uid: str) -> Tuple[int, int, str]:
    m = _UID_RE.fullmatch(uid)
    if not m:
        raise OirError(f"unrecognised pixel block id {uid!r}")
    dims = m.group(1).split("_", 1)[0]
    t = z = 1
    pairs = re.findall(r"([A-Za-z])(\d+)", dims)
    if not pairs or "".join(a + b for a, b in pairs) != dims:
        raise OirError(f"unrecognised pixel block id {uid!r}")
    for letter, num in pairs:
        v = int(num)
        if letter in "zZ":
            z = v
        elif letter in "tT":
            t = v
        elif v != 1:
            # Another axis (e.g. lambda) with several positions would make
            # planes collide — refuse rather than mix them up.
            raise OirError(f"unsupported image dimension {letter!r} in this .oir")
    if t < 1 or z < 1:
        raise OirError(f"unrecognised pixel block id {uid!r}")
    return t, z, m.group(2).lower()


# ── metadata ─────────────────────────────────────────────────────

def _tag(s: str, name: str) -> Optional[str]:
    m = re.search(r"<" + re.escape(name) + r">\s*([^<]*?)\s*</" + re.escape(name) + r">", s)
    return m.group(1) if m else None


def _num(s: Optional[str]) -> Optional[float]:
    try:
        return float(s) if s is not None else None
    except ValueError:
        return None


def _parse_metadata(bundle: bytes) -> dict:
    info = {"width": None, "height": None, "depth": None, "bit_count": None,
            "channels": [], "pixel_size_um": None, "z_step_um": None,
            "magnification": None, "dyes": []}
    luts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    pos = bundle.find(b"<?xml")
    while pos != -1:
        if pos < 4:
            pos = bundle.find(b"<?xml", pos + 5)
            continue
        (xlen,) = struct.unpack_from("<I", bundle, pos - 4)
        # Peek at THIS document only — a short one (imageDefinition is ~320
        # bytes) is followed directly by the next, often a LUT.
        head = bundle[pos:pos + min(xlen, 700)].decode("utf-8", "replace")
        if "<lut:LUT" in head:
            # A LUT embeds a 65536-entry table (~0.5 MB); only the colour
            # header at the top is needed, preceded by [u32 36][GUID].
            guid = None
            if pos >= 44 and struct.unpack_from("<I", bundle, pos - 44)[0] == 36:
                guid = bundle[pos - 40:pos - 4].decode("ascii", "replace").lower()
            luts[guid or ""] = _lut_color(bundle[pos:pos + 4096].decode("utf-8", "replace"))
        else:
            doc = bundle[pos:pos + xlen].decode("utf-8", "replace")
            if "<base:imageDefinition" in doc[:700]:
                info["width"] = int(_num(_tag(doc, "base:width")) or 0)
                info["height"] = int(_num(_tag(doc, "base:height")) or 0)
                info["depth"] = int(_num(_tag(doc, "base:depth")) or 0)
                bc = _num(_tag(doc, "base:bitCounts"))
                info["bit_count"] = int(bc) if bc else None
            elif "lsmimage:imageProperties" in doc[:700]:
                _parse_image_properties(doc, info)
            elif "lsmimage:lsmChannel" in doc[:700]:
                dye = _tag(doc, "lsmimage:dyeName")
                if dye:
                    info["dyes"].append(dye)
        pos = bundle.find(b"<?xml", pos + 5)

    if not (info["width"] and info["height"] and info["depth"]):
        raise OirError("image size missing from the metadata")
    for c in info["channels"]:
        name_hex = luts.get(c.id)
        if name_hex:
            c.lut_name, c.color_hex = name_hex
    return info


def _parse_image_properties(doc: str, info: dict) -> None:
    for m in re.finditer(
            r'<commonphase:channel\s+id="(' + _GUID + r')"\s+order="(\d+)"\s*>(.*?)</commonphase:channel>',
            doc, re.S):
        guid, order, body = m.group(1).lower(), int(m.group(2)), m.group(3)
        name = _tag(body, "commonphase:name") or f"Ch {order}"
        info["channels"].append(OirChannel(id=guid, name=name, order=order))
        if info["pixel_size_um"] is None:
            length = re.search(r"<commonphase:length>(.*?)</commonphase:length>", body, re.S)
            unit = re.search(r"<commonphase:pixelUnit>(.*?)</commonphase:pixelUnit>", body, re.S)
            if length:
                x, y = _num(_tag(length.group(1), "commonparam:x")), _num(_tag(length.group(1), "commonparam:y"))
                z = _num(_tag(length.group(1), "commonparam:z"))
                ux = _tag(unit.group(1), "commonphase:x") if unit else "MICRO_METER"
                uy = _tag(unit.group(1), "commonphase:y") if unit else "MICRO_METER"
                uz = _tag(unit.group(1), "commonphase:z") if unit else "MICRO_METER"
                if x and y and ux in _UNIT_TO_UM and uy in _UNIT_TO_UM:
                    info["pixel_size_um"] = (x * _UNIT_TO_UM[ux], y * _UNIT_TO_UM[uy])
                if z and uz in _UNIT_TO_UM:
                    info["z_step_um"] = z * _UNIT_TO_UM[uz]
    mag = re.search(r"<opticalelement:magnification>\s*([0-9.]+)\s*<", doc)
    if mag:
        info["magnification"] = _num(mag.group(1))


def _lut_color(head: str) -> Tuple[Optional[str], Optional[str]]:
    name = _tag(head, "lut:name")
    comps = []
    for ch in ("red", "green", "blue"):
        m = re.search(r"<lut:" + ch + r">.*?<lut:contrast>\s*([0-9.]+)\s*</lut:contrast>", head, re.S)
        comps.append(_num(m.group(1)) if m else None)
    if any(v is None for v in comps):
        return name, None
    rgb = [max(0, min(255, round(v * 255))) for v in comps]
    if not any(rgb):
        return name, None
    return name, "#%02x%02x%02x" % tuple(rgb)
