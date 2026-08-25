#!/usr/bin/env python3
"""
rview_server.py -- lightweight viewer for PVD/VTI 2D time series over a slow link.

Reads .pvd collections of 2D vtkImageData (.vti) files, quantizes one field at a
time to 8 bit against a fixed value window, and serves ~25 KB encoded frames plus
a self-contained browser UI.  Meant to run on the machine that holds the data
(fast local disk) and be reached through an ssh tunnel, so that only the reduced
frames cross the network instead of multi-megabyte .vti files.

Dependencies: numpy, Pillow.  No VTK, no matplotlib, no web framework.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

WORKERS = min(32, (os.cpu_count() or 4))


def _build_id() -> str:
    """Hash of this source file: lets the launcher spot a stale remote server."""
    import hashlib
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


BUILD = _build_id()

# --------------------------------------------------------------------------
# colormaps: exact matplotlib tables, 256x3 uint8, base64 of the raw bytes
# --------------------------------------------------------------------------
CMAPS_B64: dict[str, str] = {}  # filled in below by _load_cmaps()


def _cmap_array(name: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(CMAPS_B64[name]), dtype=np.uint8).reshape(256, 3)


class ByteLRU:
    """LRU bounded by total bytes held, not entry count.

    Entry counts are the wrong unit here: a field is 0.6 MB on a 464x336 grid
    and 10 MB on a 1856x1344 one, so a fixed count silently turns into gigabytes.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                return None
            self._d.move_to_end(key)
            return hit[0]

    def put(self, key, value, size: int):
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._d[key] = (value, size)
            self._bytes += size
            while self._bytes > self.budget and len(self._d) > 1:
                _, (_, sz) = self._d.popitem(last=False)
                self._bytes -= sz

    def stats(self):
        with self._lock:
            return len(self._d), self._bytes


def file_id(path: str):
    """Identity of a file's current contents, for cache keys.

    Paths are not enough: a mount can be repointed at a different dataset, or
    files regenerated in place, leaving the same names on different data.
    """
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, -1)


# --------------------------------------------------------------------------
# .vti / .pvd parsing
# --------------------------------------------------------------------------
NP_DTYPES = {
    "Float32": "f4", "Float64": "f8",
    "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2",
    "Int32": "i4", "UInt32": "u4", "Int64": "i8", "UInt64": "u8",
}

# Attribute order inside <DataArray> is not guaranteed, so match with lookaheads.
_ARRAY_RE = re.compile(
    r'<DataArray\b'
    r'(?=[^>]*\bformat="appended")'
    r'(?=[^>]*\bName="(?P<name>[^"]+)")'
    r'(?=[^>]*\btype="(?P<type>\w+)")'
    r'(?=[^>]*\boffset="(?P<offset>\d+)")'
    r'[^>]*>'
)
_NCOMP_RE = re.compile(r'\bNumberOfComponents="(\d+)"')
_DATASET_RE = re.compile(
    r'<DataSet\b(?=[^>]*\btimestep="(?P<t>[^"]+)")(?=[^>]*\bfile="(?P<f>[^"]+)")[^>]*>'
)

HEADER_SCAN = 65536


class VtiHeader:
    """Geometry + appended-array directory for one .vti file."""

    __slots__ = ("path", "nx", "ny", "origin", "spacing", "arrays",
                 "data_start", "hdr_bytes", "byteorder")

    def __init__(self, path, nx, ny, origin, spacing, arrays, data_start, hdr_bytes, byteorder):
        self.path = path
        self.nx = nx
        self.ny = ny
        self.origin = origin
        self.spacing = spacing
        self.arrays = arrays          # name -> {type, ncomp, offset}
        self.data_start = data_start  # byte offset of the '_' payload
        self.hdr_bytes = hdr_bytes    # size prefix before each appended block
        self.byteorder = byteorder    # '<' or '>'

    @property
    def npoints(self) -> int:
        return self.nx * self.ny


def read_vti_header(path: str) -> VtiHeader:
    with open(path, "rb") as fh:
        head = fh.read(HEADER_SCAN)

    ap = head.find(b"<AppendedData")
    if ap < 0:
        raise ValueError(
            f"{os.path.basename(path)}: no <AppendedData> in the first {HEADER_SCAN} bytes; "
            "only appended-raw VTI is supported (not inline/base64)")
    us = head.find(b"_", ap)
    if us < 0:
        raise ValueError(f"{os.path.basename(path)}: malformed <AppendedData> (no '_' marker)")

    open_tag = head[ap:us].decode("utf-8", "replace")
    if 'encoding="raw"' not in open_tag:
        raise ValueError(
            f"{os.path.basename(path)}: appended data is not encoding=\"raw\" ({open_tag.strip()!r})")

    xml = head[:ap].decode("utf-8", "replace")
    if "compressor=" in xml:
        raise ValueError(
            f"{os.path.basename(path)}: compressed appended data is not supported "
            "(re-export without a compressor, or add a decompression path here)")

    m = re.search(r'\bheader_type="(\w+)"', xml)
    hdr_bytes = 8 if (m and m.group(1).endswith("64")) else 4
    byteorder = ">" if 'byte_order="BigEndian"' in xml else "<"

    m = re.search(r'\bWholeExtent="([^"]+)"', xml)
    if not m:
        raise ValueError(f"{os.path.basename(path)}: no WholeExtent")
    ext = [int(float(v)) for v in m.group(1).split()]
    nx, ny, nz = ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1
    if nz != 1:
        raise ValueError(f"{os.path.basename(path)}: only single-slice 2D ImageData is supported (nz={nz})")

    def _vec(attr, default):
        mm = re.search(r'\b%s="([^"]+)"' % attr, xml)
        return [float(v) for v in mm.group(1).split()] if mm else default

    origin = _vec("Origin", [0.0, 0.0, 0.0])
    spacing = _vec("Spacing", [1.0, 1.0, 1.0])

    # Only PointData arrays; FieldData (e.g. TimeValue) and CellData are ignored.
    ps, pe = xml.find("<PointData"), xml.find("</PointData>")
    if ps < 0:
        raise ValueError(f"{os.path.basename(path)}: no <PointData> block")
    block = xml[ps:pe if pe > 0 else len(xml)]

    arrays = {}
    for m in _ARRAY_RE.finditer(block):
        tag = m.group(0)
        nc = _NCOMP_RE.search(tag)
        typ = m.group("type")
        if typ not in NP_DTYPES:
            continue
        arrays[m.group("name")] = {
            "type": typ,
            "ncomp": int(nc.group(1)) if nc else 1,
            "offset": int(m.group("offset")),
        }
    if not arrays:
        raise ValueError(f"{os.path.basename(path)}: no appended PointData arrays found")

    return VtiHeader(path, nx, ny, origin, spacing, arrays, us + 1, hdr_bytes, byteorder)


# Header length varies between files in a collection (the ascii TimeValue grows),
# so data_start must be resolved per file -- never assume a fixed header size.
_hdr_cache: dict[tuple, VtiHeader] = {}
_hdr_lock = threading.Lock()


def header_for(path: str) -> VtiHeader:
    key = (path,) + file_id(path)
    with _hdr_lock:
        h = _hdr_cache.get(key)
    if h is None:
        h = read_vti_header(path)
        with _hdr_lock:
            if len(_hdr_cache) > 4000:
                _hdr_cache.clear()
            _hdr_cache[key] = h
    return h


def _component(arr: np.ndarray, comp: str) -> np.ndarray:
    """Reduce an (ny, nx, ncomp) array to (ny, nx) per the requested component."""
    if arr.ndim == 2:
        return arr
    if comp in ("", "mag", None):
        return np.sqrt(np.einsum("ijk,ijk->ij", arr, arr))
    return arr[:, :, int(comp)]


_field_cache = ByteLRU(2 << 30)      # 2 GiB of decoded fields


def read_field(path: str, name: str, comp: str = "") -> np.ndarray:
    """Return a read-only (ny, nx) float32 view of one field. Callers must not mutate."""
    key = (path,) + file_id(path) + (name, comp)
    arr = _field_cache.get(key)
    if arr is not None:
        return arr

    hdr = header_for(path)
    if name not in hdr.arrays:
        raise KeyError(f"{os.path.basename(path)}: no field {name!r} "
                       f"(have: {', '.join(sorted(hdr.arrays))})")
    spec = hdr.arrays[name]
    dt = np.dtype(hdr.byteorder + NP_DTYPES[spec["type"]])
    count = hdr.npoints * spec["ncomp"]
    nbytes = count * dt.itemsize

    with open(path, "rb") as fh:
        fh.seek(hdr.data_start + spec["offset"] + hdr.hdr_bytes)
        buf = fh.read(nbytes)
    if len(buf) != nbytes:
        raise ValueError(f"{os.path.basename(path)}: short read for {name} "
                         f"({len(buf)} of {nbytes} bytes)")

    raw = np.frombuffer(buf, dtype=dt, count=count)
    raw = raw.reshape(hdr.ny, hdr.nx, spec["ncomp"]) if spec["ncomp"] > 1 else raw.reshape(hdr.ny, hdr.nx)
    arr = np.ascontiguousarray(_component(raw.astype(np.float32, copy=False), comp))
    arr.setflags(write=False)
    _field_cache.put(key, arr, arr.nbytes)
    return arr


class Collection:
    """One .pvd file: an ordered list of timesteps sharing a geometry and field set."""

    def __init__(self, name: str, pvd_path: str):
        self.name = name
        self.pvd_path = pvd_path
        base = os.path.dirname(os.path.abspath(pvd_path))
        with open(pvd_path, "r", errors="replace") as fh:
            txt = fh.read()
        entries = [(float(m.group("t")), os.path.join(base, m.group("f")))
                   for m in _DATASET_RE.finditer(txt)]
        if not entries:
            raise ValueError(f"{pvd_path}: no <DataSet> entries")
        entries.sort(key=lambda e: e[0])
        # A running simulation writes the .pvd entry before (or while) the .vti
        # lands, so the tail can name files that are not there yet. Drop those
        # instead of letting one unfinished step sink the whole collection.
        listed = len(entries)
        while len(entries) > 1 and not os.path.exists(entries[-1][1]):
            entries.pop()
        # If we dropped a tail, the .pvd will not change again when those files
        # finally appear, so refresh() has to re-check rather than trust mtime.
        self._pending = len(entries) < listed
        self.times = [e[0] for e in entries]
        self.files = [e[1] for e in entries]
        if not os.path.exists(self.files[0]):
            raise ValueError(f"{pvd_path}: referenced file not found: {self.files[0]}")
        self._pvd_id = file_id(pvd_path)
        self._checked = time.time()
        self._ver = None
        self._ver_at = 0.0
        header_for(self.files[0])   # fail fast on an unreadable first file

    @property
    def head(self) -> str:
        """Identity of step 0 alone, restatted on every call.

        Unchanged head plus unchanged leading timestep values means the steps a
        viewer already holds are still the same data, so a refresh that only
        appended steps can keep its cache. A rerun rewrites step 0 and this
        changes, even when the .pvd itself does not.
        """
        return "%x-%x" % file_id(self.files[0])

    @property
    def header(self) -> VtiHeader:
        # Looked up rather than stored: the files behind a collection can change
        # (a remount, a rerun) and the geometry with them.
        return header_for(self.files[0])

    def refresh(self, force: bool = False) -> None:
        """Re-read the .pvd if it changed, so added or removed steps are picked up.

        A tail we dropped as unwritten needs re-checking even when the .pvd is
        untouched, but not on every frame request: force it when the listing is
        what was asked for, and rate-limit it otherwise.
        """
        if file_id(self.pvd_path) != self._pvd_id:
            self.__init__(self.name, self.pvd_path)
        elif self._pending and (force or time.time() - self._checked >= 1.0):
            self.__init__(self.name, self.pvd_path)

    def version(self, ttl: float = 2.0) -> str:
        """Short fingerprint of every file backing this collection.

        Goes into frame URLs so the browser's immutable caching stays safe: same
        bytes, same URL; different data, different URL.
        """
        now = time.time()
        if self._ver is not None and (now - self._ver_at) < ttl:
            return self._ver
        import hashlib
        h = hashlib.blake2b(digest_size=6)
        h.update(repr(self._pvd_id).encode())
        for f in self.files:
            h.update(repr(file_id(f)).encode())
        self._ver = h.hexdigest()
        self._ver_at = now
        return self._ver

    @property
    def nsteps(self) -> int:
        return len(self.files)

    def path(self, t: int) -> str:
        if not 0 <= t < len(self.files):
            raise IndexError(f"timestep {t} out of range 0..{len(self.files) - 1}")
        return self.files[t]

    def fields_meta(self) -> list:
        out = []
        for nm, spec in self.header.arrays.items():
            out.append({"name": nm, "ncomp": spec["ncomp"], "type": spec["type"]})
        return out

    def meta(self) -> dict:
        h = self.header
        return {
            "name": self.name,
            "version": self.version(),
            "head": self.head,
            "nsteps": self.nsteps,
            "times": self.times,
            "fields": self.fields_meta(),
            "nx": h.nx, "ny": h.ny,
            "origin": h.origin[:2],
            "spacing": h.spacing[:2],
        }


# --------------------------------------------------------------------------
# value ranges and frame encoding
# --------------------------------------------------------------------------
_range_cache: dict[tuple, dict] = {}
_range_lock = threading.Lock()
RANGE_SAMPLE_STRIDE = 41  # subsample within each frame when pooling percentiles


def field_range(coll: Collection, name: str, comp: str, max_frames: int = 0) -> dict:
    """Global min/max plus pooled percentiles for one field across the time series.

    Only that field's byte range is read from each file (offsets are fixed), so this
    touches ~nsteps * one-field bytes, not the whole dataset.
    """
    key = (coll.name, coll.version(), name, comp)
    with _range_lock:
        hit = _range_cache.get(key)
    if hit is not None:
        return hit

    idx = list(range(coll.nsteps))
    if max_frames and coll.nsteps > max_frames:
        idx = [int(round(i)) for i in np.linspace(0, coll.nsteps - 1, max_frames)]
        idx = sorted(set(idx))

    def one(i):
        a = read_field(coll.path(i), name, comp)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return None
        pos = finite[finite > 0]
        return (float(finite.min()), float(finite.max()),
                np.array(finite.ravel()[::RANGE_SAMPLE_STRIDE], dtype=np.float32),
                float(pos.min()) if pos.size else None)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = [r for r in ex.map(one, idx) if r is not None]
    if not res:
        info = {"vmin": 0.0, "vmax": 0.0, "p1": 0.0, "p99": 0.0, "vminpos": None,
                "constant": True, "scanned": 0, "seconds": 0.0}
    else:
        vmin = min(r[0] for r in res)
        vmax = max(r[1] for r in res)
        pool = np.concatenate([r[2] for r in res])
        p1, p99 = (float(v) for v in np.percentile(pool, [1.0, 99.0]))
        pos = [r[3] for r in res if r[3] is not None]
        info = {
            "vmin": vmin, "vmax": vmax,
            "p1": p1, "p99": p99,
            # smallest value a log window could sit on, for fields that touch zero
            "vminpos": min(pos) if pos else None,
            # A field that never varies (xi, p_bias, ... at early steps) must not
            # divide by zero downstream, and the UI says so instead of going black.
            "constant": bool(vmax <= vmin),
            "scanned": len(res),
            "seconds": round(time.time() - t0, 2),
        }
    with _range_lock:
        _range_cache[key] = info
    return info


def decimate(a: np.ndarray, d: int) -> np.ndarray:
    """Every d-th cell. Striding, not averaging, on purpose.

    Averaging would blur values across a boundary and smear the no-data floor,
    which then leaks past a "hide <=" threshold. Striding keeps every sent pixel
    an exact cell value, so thresholds and the hover readout stay truthful.
    """
    return a[::d, ::d] if d > 1 else a


def use_log(lo: float, hi: float, log: bool) -> bool:
    """Whether a log window is actually usable. The page applies the same test,
    so both sides agree on what the codes mean without having to negotiate."""
    return bool(log) and lo > 0 and hi > lo and np.isfinite(lo) and np.isfinite(hi)


def quantize(a: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    """Map a 2D field to uint8 codes over [lo, hi], flipped so image row 0 is +y.

    With log=True the 256 codes are spread evenly in log space, which is the
    whole point: quantizing linearly and then colouring logarithmically would
    leave the low decades sharing a handful of codes.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        codes = np.zeros(a.shape, dtype=np.uint8)
    elif use_log(lo, hi, log):
        with np.errstate(divide="ignore", invalid="ignore"):
            x = ((np.log(a.astype(np.float32)) - math.log(lo))
                 * (255.0 / (math.log(hi) - math.log(lo))) + 0.5)
        # zero -> -inf, negative -> nan: both sit below the window, i.e. code 0
        x = np.where(np.isfinite(x), x, 0.0)
        codes = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = (a.astype(np.float32) - lo) * (255.0 / (hi - lo)) + 0.5
        x = np.where(np.isfinite(x), x, 0.0)
        codes = np.clip(x, 0, 255).astype(np.uint8)
    # VTK row 0 is y=0 (bottom); image row 0 is the top of the picture.
    return np.flipud(codes)


def code_values(lo: float, hi: float, log: bool = False) -> np.ndarray:
    """The value each of the 256 codes decodes to, the way the viewer decodes it."""
    n = np.arange(256, dtype=np.float64) / 255.0
    if use_log(lo, hi, log):
        return np.exp(math.log(lo) + n * (math.log(hi) - math.log(lo)))
    return lo + n * (hi - lo)


MAX_DETAIL = 8


def clamp_detail(d) -> int:
    try:
        return max(1, min(MAX_DETAIL, int(d)))
    except (TypeError, ValueError):
        return 1


def hillshade(a: np.ndarray, sx: float, sy: float,
              azimuth: float = 315.0, altitude: float = 45.0,
              zf: float = 1.0) -> np.ndarray:
    """Lambertian relief shading of a 2D height field, returned in [0, 1].

    Done as a plain surface-normal / sun-vector dot product rather than the
    slope-aspect formula, which is easy to get 90 degrees wrong by mixing a
    compass azimuth with a math-convention aspect.

    Row 0 of `a` is y=0, so +row is north and +col is east.  `azimuth` is a
    compass bearing for the sun (clockwise from north, 315 = north-west) and
    `altitude` its height above the horizon.
    """
    a = a.astype(np.float32, copy=False)
    dzdy, dzdx = np.gradient(a, max(sy, 1e-9), max(sx, 1e-9))

    # upward surface normal, before normalising: (-dz/dx, -dz/dy, 1)
    nx = -dzdx * zf
    ny = -dzdy * zf
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)

    alt = np.radians(altitude)
    az = np.radians(azimuth)
    sun = (np.cos(alt) * np.sin(az),   # east component
           np.cos(alt) * np.cos(az),   # north component
           np.sin(alt))

    hs = (nx * sun[0] + ny * sun[1] + sun[2]) * inv
    return np.clip(hs, 0.0, 1.0)


_shade_cache = ByteLRU(512 << 20)   # 512 MiB of encoded relief


def encode_shade(coll: "Collection", name: str, t: int, azimuth: float,
                 altitude: float, zf: float, fmt: str, d: int = 1) -> tuple[bytes, str]:
    """Relief as its own 8-bit layer.

    Kept separate from the colour frame on purpose: the browser composites the
    two, so changing colormap, window or contour levels stays instant and costs
    no refetch.
    """
    fmt = fmt if fmt in ENCODERS else "webp"
    d = max(1, int(d))
    key = ((coll.name, name, t) + file_id(coll.path(t))
           + (round(azimuth, 3), round(altitude, 3), round(zf, 6), fmt, d))
    hit = _shade_cache.get(key)
    if hit is not None:
        return hit, ENCODERS[fmt][0]

    h = coll.header
    # Shade the decimated terrain rather than decimating the shading: sampling a
    # full-resolution hillshade aliases badly. Cell spacing grows with d.
    hs = hillshade(decimate(read_field(coll.path(t), name, ""), d),
                   h.spacing[0] * d, h.spacing[1] * d, azimuth, altitude, zf)
    codes = np.flipud((hs * 255.0 + 0.5).astype(np.uint8))
    ctype, kw = ENCODERS[fmt]
    buf = io.BytesIO()
    Image.fromarray(codes, mode="L").save(buf, **kw)
    data = buf.getvalue()
    _shade_cache.put(key, data, len(data))
    return data, ctype


_frame_cache = ByteLRU(512 << 20)   # 512 MiB of encoded frames

ENCODERS = {
    # lossless: exact 8-bit codes, ~23 KB   |   fast: lossy, ~6 KB, for long sweeps
    "webp": ("image/webp", dict(format="WEBP", lossless=True, quality=60, method=4)),
    "webpfast": ("image/webp", dict(format="WEBP", quality=90)),
    "png": ("image/png", dict(format="PNG", compress_level=6)),
    # relief is decoration, so lossy is fine here and 2.4x smaller than lossless
    "shade": ("image/webp", dict(format="WEBP", quality=85)),
}


def encode_frame(coll: Collection, name: str, comp: str, t: int,
                 lo: float, hi: float, fmt: str, d: int = 1,
                 log: bool = False) -> tuple[bytes, str]:
    fmt = fmt if fmt in ENCODERS else "webp"
    d = max(1, int(d))
    log = use_log(lo, hi, log)
    key = ((coll.name, name, comp, t) + file_id(coll.path(t))
           + (round(lo, 6), round(hi, 6), fmt, d, log))
    hit = _frame_cache.get(key)
    if hit is not None:
        return hit, ENCODERS[fmt][0]

    codes = quantize(decimate(read_field(coll.path(t), name, comp), d), lo, hi, log)
    ctype, kw = ENCODERS[fmt]
    buf = io.BytesIO()
    Image.fromarray(codes, mode="L").save(buf, **kw)
    data = buf.getvalue()
    _frame_cache.put(key, data, len(data))
    return data, ctype


# These two must match TERRAIN_GREY and the page background in the viewer, or an
# exported GIF will not look like what was on screen.
TERRAIN_GREY = 150
PAGE_BG = (11, 13, 18)


def hidden_codes(lo: float, hi: float, maskval: float, log: bool = False) -> np.ndarray:
    """Which of the 256 codes the viewer's "hide <=" would blank, by the same rule.

    The slack absorbs a threshold typed in decimal against a float32 stored value
    (0.1 vs 0.10000000149). A linear window scales it by the window; a log one
    scales it by the threshold, since near the floor the codes are much finer.
    """
    v = code_values(lo, hi, log)
    tol = abs(maskval if use_log(lo, hi, log) else hi - lo) * 1e-6
    return v <= maskval + tol


def level_lut(cmap: str, levels: int) -> np.ndarray:
    """256x3 colour table for an 8-bit code, optionally banded into `levels` steps.

    Must match mapNorm() in the page, or an exported GIF would not look like
    what was on screen.
    """
    cm = _cmap_array(cmap if cmap in CMAPS_B64 else "viridis")
    k = np.arange(256, dtype=np.float32) / 255.0
    if levels and levels > 0:
        k = (np.minimum(np.floor(k * levels), levels - 1) + 0.5) / levels
    return cm[np.clip(np.round(k * 255), 0, 255).astype(np.int32)]


def make_gif(coll: Collection, name: str, comp: str, lo: float, hi: float,
             cmap: str, stride: int, fps: float, levels: int = 0,
             relief: str = "", az: float = 315.0, alt: float = 45.0,
             zf: float = 1.0, static: bool = False, intensity: float = 0.55,
             mask: bool = False, maskval: float = 0.0, opacity: float = 1.0,
             d: int = 1, log: bool = False, max_frames: int = 600) -> bytes:
    """Animated GIF of the whole series, rendered here rather than in the browser."""
    idx = list(range(0, coll.nsteps, max(1, stride)))[:max_frames]
    log = use_log(lo, hi, log)
    lut = level_lut(cmap, levels)
    if relief and opacity < 1.0:
        lut = np.clip(opacity * lut.astype(np.float32)
                      + (1.0 - opacity) * TERRAIN_GREY, 0, 255).astype(np.uint8)
    hidden = hidden_codes(lo, hi, maskval, log) if mask else np.zeros(256, bool)

    if not relief:
        # No relief: the colour table *is* the GIF palette, so the 8-bit codes
        # can be used directly as palette indices. Hiding is just a palette edit.
        table = lut.copy()
        table[hidden] = PAGE_BG
        pal = table.flatten().tolist()

        def one(i):
            im = Image.fromarray(
                quantize(decimate(read_field(coll.path(i), name, comp), d), lo, hi, log), mode="P")
            im.putpalette(pal)
            return im

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            frames = list(ex.map(one, idx))
    else:
        h = coll.header
        shade_cache = {}

        def shade_for(i):
            key = 0 if static else i
            if key not in shade_cache:
                shade_cache[key] = hillshade(decimate(read_field(coll.path(key), relief, ""), d),
                                             h.spacing[0] * d, h.spacing[1] * d, az, alt, zf)
            return shade_cache[key]

        def one(i):
            codes = quantize(decimate(read_field(coll.path(i), name, comp), d), lo, hi, log)
            rgb = lut[codes].astype(np.float32)
            if mask:
                # hidden cells show the bare shaded ground, as in the viewer
                rgb[hidden[codes]] = TERRAIN_GREY
            m = (1.0 - intensity) + intensity * np.flipud(shade_for(i))
            rgb *= m[:, :, None]
            return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            rgbs = list(ex.map(one, idx))
        # One palette derived from the first frame and reused, so colours do not
        # crawl from frame to frame.
        base = rgbs[0].quantize(colors=256, method=Image.FASTOCTREE)
        frames = [base] + [f.quantize(palette=base) for f in rgbs[1:]]

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=max(20, int(round(1000.0 / max(0.1, fps)))), loop=0, optimize=False)
    return buf.getvalue()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Store:
    """All collections discovered under the served directory."""

    def __init__(self, directory: str, range_frames: int = 0):
        self.dir = os.path.abspath(directory)
        self.range_frames = range_frames
        self.colls: dict[str, Collection] = {}
        self.errors: list[str] = []
        self.rescan(strict=True)

    def rescan(self, strict: bool = False) -> None:
        """Re-list the directory so collections written since startup appear.

        A run in progress can drop in a whole new .pvd, not just extra steps in
        an existing one, so refreshing has to look at the directory again and
        not only at the collections already loaded.
        """
        try:
            pvds = sorted(f for f in os.listdir(self.dir) if f.lower().endswith(".pvd"))
        except OSError as exc:
            if strict:
                raise SystemExit(f"cannot read {self.dir}: {exc}")
            self.errors = [f"{self.dir}: {exc}"]
            return
        if strict and not pvds:
            raise SystemExit(f"no .pvd files in {self.dir}")
        keep, errors = {}, []
        for f in pvds:
            nm = os.path.splitext(f)[0]
            path = os.path.join(self.dir, f)
            have = self.colls.get(nm)
            if have is not None and have.pvd_path == path:
                keep[nm] = have          # kept as-is; refresh() re-reads it
                continue
            try:
                keep[nm] = Collection(nm, path)
            except Exception as exc:  # a broken collection must not sink the rest
                errors.append(f"{f}: {exc}")
                sys.stderr.write(f"[rview] skipping {f}: {exc}\n")
        if not keep:
            if strict:
                raise SystemExit(f"no readable collections in {self.dir}")
            # Transient: a remount in progress, say. Keep serving what we have.
            self.errors = errors or [f"no readable collections in {self.dir}"]
            return
        self.colls = keep
        self.errors = errors

    def get(self, name: str) -> Collection:
        if name not in self.colls:
            raise KeyError(f"unknown collection {name!r} (have: {', '.join(self.colls)})")
        c = self.colls[name]
        c.refresh()
        return c

    def meta(self) -> dict:
        self.rescan()
        for c in self.colls.values():
            c.refresh(force=True)
        return {
            "dir": self.dir,
            "build": BUILD,
            "host": os.uname().nodename,
            "collections": [c.meta() for c in self.colls.values()],
            "cmaps": CMAPS_B64,
            "errors": self.errors,
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive matters: prefetch reuses one tunnelled socket
    server_version = "rview"
    store: Store = None  # set by serve()

    def log_message(self, fmt, *args):
        pass  # quiet by default; errors are reported explicitly below

    # -- helpers ---------------------------------------------------------
    def _send(self, code, body: bytes, ctype: str, extra: dict = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200, extra=None):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8", extra)

    def _fail(self, code, msg):
        sys.stderr.write(f"[rview] {code} {self.path} :: {msg}\n")
        self._json({"error": msg}, code)

    def _q(self, q, key, default=None, cast=None):
        v = q.get(key, [default])[0]
        if v is None:
            return default
        return cast(v) if cast else v

    # -- routes ----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                page = HTML_PAGE.encode("utf-8")
                return self._send(200, page, "text/html; charset=utf-8",
                                  {"Cache-Control": "no-cache"})
            if u.path == "/api/meta":
                return self._json(self.store.meta(), extra={"Cache-Control": "no-cache"})
            if u.path == "/api/range":
                return self._route_range(q)
            if u.path == "/api/frame":
                return self._route_frame(q)
            if u.path == "/api/shade":
                return self._route_shade(q)
            if u.path == "/api/probe":
                return self._route_probe(q)
            if u.path == "/api/gif":
                return self._route_gif(q)
            return self._fail(404, f"no such endpoint: {u.path}")
        except (KeyError, IndexError, ValueError) as exc:
            return self._fail(400, str(exc))
        except BrokenPipeError:
            return  # client navigated away mid-prefetch
        except Exception as exc:
            sys.stderr.write(traceback.format_exc())
            return self._fail(500, f"{type(exc).__name__}: {exc}")

    def _target(self, q):
        coll = self.store.get(self._q(q, "coll", ""))
        field = self._q(q, "field", "")
        comp = self._q(q, "comp", "")
        if field not in coll.header.arrays:
            raise KeyError(f"unknown field {field!r} in {coll.name} "
                           f"(have: {', '.join(sorted(coll.header.arrays))})")
        return coll, field, comp

    def _route_range(self, q):
        coll, field, comp = self._target(q)
        info = field_range(coll, field, comp, self.store.range_frames)
        self._json(info, extra={"Cache-Control": "no-cache"})

    def _route_frame(self, q):
        coll, field, comp = self._target(q)
        t = self._q(q, "t", 0, int)
        lo = self._q(q, "lo", 0.0, float)
        hi = self._q(q, "hi", 1.0, float)
        fmt = self._q(q, "fmt", "webp")
        d = clamp_detail(self._q(q, "d", 1, int))
        log = self._q(q, "scale", "lin") == "log"
        data, ctype = encode_frame(coll, field, comp, t, lo, hi, fmt, d, log)
        # The URL fully determines the bytes, so let the browser keep them forever.
        self._send(200, data, ctype, {
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Vmin": repr(lo), "X-Vmax": repr(hi),
        })

    def _route_shade(self, q):
        coll, field, _ = self._target(q)
        if coll.header.arrays[field]["ncomp"] != 1:
            raise ValueError(f"relief needs a scalar field; {field!r} has "
                             f"{coll.header.arrays[field]['ncomp']} components")
        data, ctype = encode_shade(
            coll, field,
            self._q(q, "t", 0, int),
            self._q(q, "az", 315.0, float),
            self._q(q, "alt", 45.0, float),
            self._q(q, "zf", 1.0, float),
            self._q(q, "fmt", "shade"),
            clamp_detail(self._q(q, "d", 1, int)),
        )
        self._send(200, data, ctype, {
            "Cache-Control": "public, max-age=31536000, immutable",
        })

    def _route_probe(self, q):
        coll, field, comp = self._target(q)
        t = self._q(q, "t", 0, int)
        col = self._q(q, "col", 0, int)
        row = self._q(q, "row", 0, int)  # image row: 0 is the top of the picture
        h = coll.header
        if not (0 <= col < h.nx and 0 <= row < h.ny):
            raise ValueError(f"pixel ({col},{row}) outside {h.nx}x{h.ny}")
        a = read_field(coll.path(t), field, comp)
        j = h.ny - 1 - row  # back to VTK row order
        self._json({
            "value": float(a[j, col]),
            "col": col, "row": row,
            "x": h.origin[0] + col * h.spacing[0],
            "y": h.origin[1] + j * h.spacing[1],
            "time": coll.times[t],
        }, extra={"Cache-Control": "no-cache"})

    def _route_gif(self, q):
        coll, field, comp = self._target(q)
        lo = self._q(q, "lo", 0.0, float)
        hi = self._q(q, "hi", 1.0, float)
        cmap = self._q(q, "cmap", "viridis")
        stride = max(1, self._q(q, "stride", 1, int))
        fps = self._q(q, "fps", 10.0, float)
        relief = self._q(q, "relief", "")
        if relief and relief not in coll.header.arrays:
            raise ValueError(f"unknown relief field {relief!r}")
        data = make_gif(
            coll, field, comp, lo, hi, cmap, stride, fps,
            levels=self._q(q, "levels", 0, int),
            relief=relief,
            az=self._q(q, "az", 315.0, float),
            alt=self._q(q, "alt", 45.0, float),
            zf=self._q(q, "zf", 1.0, float),
            static=self._q(q, "static", "0") == "1",
            intensity=self._q(q, "intensity", 0.55, float),
            mask=self._q(q, "mask", "0") == "1",
            maskval=self._q(q, "maskval", 0.0, float),
            opacity=self._q(q, "opacity", 1.0, float),
            d=clamp_detail(self._q(q, "d", 1, int)),
            log=self._q(q, "scale", "lin") == "log",
        )
        tag = f"{coll.name}_{field}{('_' + comp) if comp else ''}"
        self._send(200, data, "image/gif", {
            "Content-Disposition": f'attachment; filename="{tag}.gif"',
            "Cache-Control": "no-cache",
        })


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rview</title>
<style>
  :root{
    --bg:#12141a; --panel:#1b1e26; --edge:#2b303b; --fg:#e6e9ef;
    --dim:#9aa3b2; --accent:#5aa9e6; --warn:#e6b25a;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{
    background:var(--bg);color:var(--fg);
    font:13px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    display:flex;flex-direction:column;overflow:hidden;
  }
  .bar{
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
    padding:7px 10px;background:var(--panel);border-bottom:1px solid var(--edge);
  }
  .bar.bottom{border-bottom:none;border-top:1px solid var(--edge)}
  label{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  select,input,button{
    background:#232733;color:var(--fg);border:1px solid var(--edge);
    border-radius:5px;padding:4px 7px;font:inherit;font-size:12px;
  }
  select:focus,input:focus{outline:1px solid var(--accent);border-color:var(--accent)}
  button{cursor:pointer;user-select:none}
  button:hover{background:#2c3140;border-color:#3d4356}
  button:active{background:#353b4c}
  input[type=number]{width:88px}
  input[type=range]{accent-color:var(--accent)}
  .grow{flex:1}
  .sep{width:1px;height:20px;background:var(--edge)}
  #stage{
    flex:1;position:relative;overflow:hidden;background:#0b0d12;
    display:flex;align-items:center;justify-content:center;
  }
  #cv{
    image-rendering:pixelated;transform-origin:0 0;position:absolute;left:0;top:0;
    cursor:crosshair;transition:opacity .12s;
  }
  #ax{position:absolute;left:0;top:0;pointer-events:none}
  #hud{
    position:absolute;left:10px;top:10px;padding:6px 9px;border-radius:6px;
    background:rgba(12,14,20,.82);border:1px solid var(--edge);
    font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;
    pointer-events:none;color:var(--dim);
  }
  #hud b{color:var(--fg);font-weight:600}
  #note{
    position:absolute;left:50%;top:14px;transform:translateX(-50%);
    padding:6px 12px;border-radius:6px;background:rgba(230,178,90,.14);
    border:1px solid rgba(230,178,90,.45);color:var(--warn);display:none;
  }
  #cbwrap{
    position:absolute;right:12px;top:50%;transform:translateY(-50%);
    display:flex;flex-direction:column;align-items:flex-start;gap:4px;
    background:rgba(12,14,20,.82);border:1px solid var(--edge);
    border-radius:6px;padding:8px;
  }
  #cb{display:block}
  .cblab{font:11px ui-monospace,Menlo,monospace;color:var(--dim)}
  #prog{
    position:relative;width:150px;height:6px;border-radius:3px;
    background:#232733;overflow:hidden;
  }
  #progfill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--accent);transition:width .15s}
  .mono{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
  #tslider{flex:1;min-width:160px}
  .pill{color:var(--dim);font-size:11px}
</style>
</head>
<body>

<div class="bar">
  <label>set</label><select id="coll"></select>
  <button id="refresh" title="re-read the .pvd listing (r): picks up timesteps written since this page loaded">&#8635;</button>
  <label>field</label><select id="field"></select>
  <div class="sep"></div>
  <label>range</label><select id="rmode">
    <option value="robust">1-99%</option>
    <option value="global">min/max</option>
    <option value="manual">manual</option>
  </select>
  <input type="number" id="lo" step="any" title="window minimum">
  <input type="number" id="hi" step="any" title="window maximum">
  <div class="sep"></div>
  <label title="hide cells at or below this value; with relief on they show as bare shaded ground"><input type="checkbox" id="maskOn"> hide &le;</label>
  <input type="number" id="maskVal" step="any" value="0" style="width:70px">
  <div class="sep"></div>
  <label title="send every d-th cell; 'auto' matches the grid to your screen">detail</label>
  <select id="detail" title="send every d-th cell; 'auto' matches the grid to your screen">
    <option value="auto">auto</option>
    <option value="1">full</option>
    <option value="2">1/2</option>
    <option value="3">1/3</option>
    <option value="4">1/4</option>
  </select>
  <label>quality</label><select id="fmt">
    <option value="webp" title="exact 8-bit codes (~50 kB/frame)">lossless</option>
    <option value="webpfast" title="lossy WebP: ~2.5x smaller, but values are approximate">fast (lossy)</option>
    <option value="png" title="lossless PNG, for browsers without WebP">png</option>
  </select>
  <div class="grow"></div>
  <button id="savepng">save PNG</button>
  <button id="savegif">save GIF</button>
</div>

<div class="bar">
  <label>color</label><select id="cmap"></select>
  <label title="0 = smooth; N = N filled contour bands">levels</label>
  <input type="number" id="levels" min="0" max="64" step="1" value="0" style="width:56px"
         title="0 = smooth; N = N filled contour bands">
  <label title="spread the colours over decades instead of evenly">scale</label>
  <select id="scale" title="spread the colours over decades instead of evenly">
    <option value="lin">linear</option>
    <option value="log">log</option>
  </select>
  <div class="sep"></div>
  <label title="tick-labelled x and y in the file's own coordinates">axes</label>
  <select id="axes" title="tick-labelled x and y in the file's own coordinates">
    <option value="ticks">on</option>
    <option value="grid">on + grid</option>
    <option value="off">off</option>
  </select>
  <div class="sep"></div>
  <label>relief</label><select id="relief"></select>
  <span id="reliefopts" style="display:none;align-items:center;gap:10px">
    <label title="shade every step with step 0's relief: one image for the whole series
instead of one per step"><input type="checkbox" id="rstatic"> static</label>
    <label>sun</label>
    <input type="number" id="az" value="315" min="0" max="360" step="15" style="width:62px"
           title="azimuth: compass bearing of the sun, 315 = north-west">
    <input type="number" id="alt" value="45" min="1" max="89" step="5" style="width:56px"
           title="altitude of the sun above the horizon, degrees">
    <label>exag</label>
    <input type="number" id="zf" value="1" min="0.1" max="20" step="0.5" style="width:60px"
           title="vertical exaggeration; 1 is true geometry">
    <label>strength</label>
    <input type="range" id="intensity" min="0" max="100" value="55" style="width:110px"
           title="how strongly the relief darkens the colours">
    <label>opacity</label>
    <input type="range" id="opacity" min="0" max="100" value="100" style="width:110px"
           title="opacity of the draped field: at 0 only the bare relief is left">
  </span>
  <div class="grow"></div>
  <span class="pill" id="status"></span>
</div>

<div id="stage">
  <canvas id="cv"></canvas>
  <canvas id="ax"></canvas>
  <div id="hud"></div>
  <div id="note"></div>
  <div id="cbwrap">
    <span class="cblab" id="cbhi">-</span>
    <canvas id="cb" width="16" height="190"></canvas>
    <span class="cblab" id="cblo">-</span>
  </div>
</div>

<div class="bar bottom">
  <button id="first" title="first step (Home)">&#9198;</button>
  <button id="prev" title="previous step (left arrow)">&#9194;</button>
  <button id="play" title="play / pause (space)" style="width:34px">&#9205;</button>
  <button id="next" title="next step (right arrow)">&#9193;</button>
  <button id="last" title="last step (End)">&#9197;</button>
  <input type="range" id="tslider" min="0" max="0" value="0">
  <span class="mono" id="tlabel">-</span>
  <div class="sep"></div>
  <label>fps</label><input type="number" id="fps" value="10" min="1" max="60" style="width:56px">
  <div class="sep"></div>
  <div id="prog" title="frames cached at the current window"><div id="progfill"></div></div>
  <span class="pill" id="progtxt">-</span>
  <button id="reset" title="reset zoom (double-click the image)">fit</button>
</div>

<script>
"use strict";

const CACHE_BYTES = 384 << 20;  // decoded codes held in the tab; one 1856x1344
                                // step is 2.5 MB, so a long series needs a bound
const TERRAIN_GREY = 150;       // bare ground under a "hide <=" cutout, before shading
                                // (mirrored by TERRAIN_GREY in the server, for GIF export)
const CONCURRENCY = 6;          // parallel frame fetches over the one tunnelled socket
const MAX_TRIES = 3;

const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d", {willReadFrequently:false});
const scratch = document.createElement("canvas");
const sctx = scratch.getContext("2d", {willReadFrequently:true});

const S = {
  meta:null, coll:null, field:"", comp:"", nx:0, ny:0, nsteps:0,
  origin:[0,0], spacing:[1,1], times:[],
  t:0, cmap:"viridis", fmt:"webp", version:"",
  detail:"auto", d:1, fullNx:0, fullNy:0,
  lo:0, hi:1, encLo:0, encHi:1, encLog:false, rmode:"robust", range:null, scale:"lin",
  frames:new Map(), inflight:new Set(), failed:new Map(), gen:0,
  playing:false, fps:10, mask:false, maskVal:0,
  zoom:1, panx:0, pany:0, base:1, hover:null,
  levels:0,
  axes:"ticks",
  relief:"off", rstatic:false, az:315, alt:45, zf:1, intensity:0.55, opacity:1,
  refnote:"",
  shades:new Map(),
};

let CM = {};                    // name -> Uint8Array(768)
let imgData = null, lutBuf = new Uint8ClampedArray(1024), lutKey = "";
let mulBuf = new Uint16Array(256), mulKey = "";   // 0..256, so 256 means "unchanged"

// ---------------------------------------------------------------- helpers
const clamp = (v,a,b) => v<a?a:(v>b?b:v);
const sleep = (ms) => new Promise(r=>setTimeout(r,ms));

// enough digits to be faithful, few enough to read in a spin box
function trim(v){
  if(!isFinite(v)) return "0";
  const r = parseFloat(v.toPrecision(6));
  return String(r);
}

function fmtNum(v){
  if(v === null || v === undefined || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if(a === 0) return "0";
  if(a < 1e-3 || a >= 1e6) return v.toExponential(3);
  return v.toFixed(a < 1 ? 4 : (a < 100 ? 2 : 1));
}

function note(msg){
  const n = $("note");
  if(!msg){ n.style.display = "none"; return; }
  n.textContent = msg; n.style.display = "block";
}

function fieldKey(){ return S.field + "|" + S.comp; }

// ------------------------------------------------------------- value scaling
// A window maps values to 0..1 either evenly or by decades. Every place that
// turns a value into a colour, or a code back into a value, goes through these,
// so the encoded frame, the colourbar and the readout cannot drift apart.
// Log is honoured only where it is meaningful; the server applies the same test
// to the same lo/hi, so the two sides never disagree about what a code means.
function canLog(lo, hi){ return lo > 0 && hi > lo && isFinite(lo) && isFinite(hi); }
function logOK(){ return S.scale === "log" && canLog(S.lo, S.hi); }

function normOf(v, lo, hi, log){
  return log ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
             : (v - lo) / (hi - lo);
}
function valueOf(n, lo, hi, log){
  return log ? Math.exp(Math.log(lo) + n * (Math.log(hi) - Math.log(lo)))
             : lo + n * (hi - lo);
}
// what one of a frame's 256 codes is worth, in the window it was encoded against
function codeValue(k, f){
  return (f.hi > f.lo) ? valueOf(k / 255, f.lo, f.hi, !!f.log) : f.lo;
}
// slack for "hide <=" typed in decimal against a float32 cell (0.1 vs 0.1000000015).
// Scaled by the window when linear; by the threshold when log, where the codes
// near the floor are far finer. Mirrored by hidden_codes() for the GIF.
function maskTol(){
  return Math.abs(logOK() ? S.maskVal : S.hi - S.lo) * 1e-6;
}

// ---------------------------------------------------------------- startup
async function boot(){
  const meta = await (await fetch("/api/meta")).json();
  S.meta = meta;
  for(const [name,b64] of Object.entries(meta.cmaps)){
    const bin = atob(b64), arr = new Uint8Array(768);
    for(let i=0;i<768;i++) arr[i] = bin.charCodeAt(i);
    CM[name] = arr;
  }
  const cmapSel = $("cmap");
  for(const name of Object.keys(CM)){
    const o = document.createElement("option"); o.value = o.textContent = name;
    cmapSel.appendChild(o);
  }
  cmapSel.value = S.cmap;

  const cs = fillCollections();
  if(meta.errors && meta.errors.length) console.warn("collection errors", meta.errors);

  const want = new URLSearchParams(location.search).get("coll");
  cs.value = (want && meta.collections.some(c=>c.name===want)) ? want : meta.collections[0].name;
  wire();
  await selectCollection(cs.value);
}

function collMeta(){ return S.meta.collections.find(c=>c.name===S.coll); }

// The step count lives in the option label, so this is also how a refresh shows
// that a set has grown without anything being selected.
function fillCollections(){
  const cs = $("coll");
  cs.innerHTML = "";
  for(const c of S.meta.collections){
    const o = document.createElement("option");
    o.value = c.name; o.textContent = c.name + "  (" + c.nsteps + ")";
    cs.appendChild(o);
  }
  return cs;
}

// ------------------------------------------------------ refreshing the listing
// A run in progress keeps appending steps. Re-reading /api/meta picks them up;
// the work here is deciding whether the frames already downloaded are still the
// same data, because dropping a 500-frame cache on every refresh would defeat
// the point of watching a run.
function refstat(msg){ S.refnote = msg || ""; updateStatus(); }

function sameFields(a, b){
  return a.length === b.length &&
         a.every((f,i) => f.name === b[i].name && f.ncomp === b[i].ncomp);
}

// Same step 0, same geometry, same fields, and the old timestep values still a
// prefix of the new ones: the steps we hold are untouched, only later ones added.
function appendedOnly(was, now){
  return !!was && now.head === was.head && now.nx === was.nx && now.ny === was.ny
      && now.nsteps >= was.nsteps && sameFields(was.fields, now.fields)
      && was.times.every((t,i) => t === now.times[i]);
}

// Rescan the value range. Adopting a new window re-encodes every frame, so say
// whether the new range actually needs more room than the current window.
async function rangeGrew(){
  const qs = new URLSearchParams({coll:S.coll, field:S.field, comp:S.comp});
  let r;
  try{ r = await (await fetch("/api/range?" + qs)).json(); }
  catch(e){ return false; }
  if(!r || r.error) return false;
  S.range = r;
  const lo = S.rmode === "global" ? r.vmin : r.p1;
  const hi = S.rmode === "global" ? r.vmax : r.p99;
  const tol = 0.01 * Math.max(S.hi - S.lo, Math.abs(S.hi) * 1e-9);
  return lo < S.lo - tol || hi > S.hi + tol;
}

async function refreshListing(){
  const btn = $("refresh");
  if(btn.disabled) return;              // one at a time; meta can take a moment
  btn.disabled = true;
  refstat("checking ...");
  try{
    const meta = await (await fetch("/api/meta", {cache:"no-store"})).json();
    const was = collMeta();
    S.meta = meta;
    if(meta.errors && meta.errors.length) console.warn("collection errors", meta.errors);
    const cs = fillCollections();

    if(!meta.collections.some(c => c.name === S.coll)){
      const gone = S.coll;
      cs.value = meta.collections[0].name;
      await selectCollection(cs.value);   // clears the note, so say it after
      refstat("'" + gone + "' is gone");
      return;
    }
    cs.value = S.coll;
    const now = collMeta();

    if(!appendedOnly(was, now)){
      // geometry, fields or the history itself changed: nothing cached survives
      await selectCollection(S.coll);
      refstat("dataset changed - reloaded (" + S.nsteps + ")");
      return;
    }

    const added = now.nsteps - was.nsteps;
    const wasAtEnd = S.t === was.nsteps - 1;
    if(added && was.nsteps){
      // whatever was last may have been caught half-written
      S.frames.delete(was.nsteps - 1);
      S.shades.delete(was.nsteps - 1);
    }
    S.version = now.version || "";
    S.nsteps = now.nsteps; S.times = now.times;
    $("tslider").max = String(S.nsteps - 1);
    refstat(added ? ("+" + added + " step" + (added > 1 ? "s" : "") + "  (" + S.nsteps + ")")
                  : ("no new steps (" + S.nsteps + ")"));

    if(added && S.rmode !== "manual" && await rangeGrew()) applyRangeMode();
    updateStatus();
    setT(wasAtEnd ? S.nsteps - 1 : S.t);   // parked on the end means follow the end
    startPrefetch(); updateProgress();
  }catch(e){
    refstat("refresh failed: " + e);
  }finally{
    btn.disabled = false;
  }
}

// Pick a decimation that lands the served grid near the size it will actually be
// drawn at. At "fit" a 1856x1344 grid shows at about 965x700 on a normal window,
// so sending every cell is roughly 4x more pixels than any of them can occupy.
function autoDetail(){
  const st = $("stage");
  const w = st.clientWidth || 1200, h = st.clientHeight || 700;
  const need = Math.max(S.fullNx / w, S.fullNy / h);
  return clamp(Math.round(need), 1, 4);
}

// Resize the canvas and drop cached layers: the served grid just changed shape.
function applyDetail(){
  S.d = S.detail === "auto" ? autoDetail() : clamp(parseInt(S.detail, 10) || 1, 1, 4);
  S.nx = Math.ceil(S.fullNx / S.d);
  S.ny = Math.ceil(S.fullNy / S.d);
  cv.width = S.nx; cv.height = S.ny;
  scratch.width = S.nx; scratch.height = S.ny;
  imgData = ctx.createImageData(S.nx, S.ny);
  // decimated cells are bigger than screen pixels, so let the browser smooth them
  cv.style.imageRendering = S.d > 1 ? "auto" : "pixelated";
  S.frames.clear(); S.shades.clear(); S.failed.clear();
  lutKey = "";
  detailNote();
}

function detailNote(){ updateStatus(); }

async function selectCollection(name){
  S.coll = name;
  S.refnote = "";        // whatever the last refresh said was about another set
  const c = collMeta();
  S.fullNx = c.nx; S.fullNy = c.ny; S.nsteps = c.nsteps; S.version = c.version || "";
  S.origin = c.origin; S.spacing = c.spacing; S.times = c.times;
  S.t = Math.min(S.t, S.nsteps-1);
  applyDetail();

  // one entry per scalar; vectors expand to magnitude + components
  const fs = $("field"); fs.innerHTML = "";
  const axes = ["x","y","z"];
  for(const f of c.fields){
    if(f.ncomp === 1){
      const o = document.createElement("option");
      o.value = f.name + "|"; o.textContent = f.name; fs.appendChild(o);
    } else {
      const m = document.createElement("option");
      m.value = f.name + "|mag"; m.textContent = "|" + f.name + "|"; fs.appendChild(m);
      for(let k=0;k<f.ncomp;k++){
        const o = document.createElement("option");
        o.value = f.name + "|" + k;
        o.textContent = f.name + "_" + (axes[k] || k);
        fs.appendChild(o);
      }
    }
  }
  // relief can be draped from any scalar in this collection (bed, srf, ...)
  const rs = $("relief"), had = S.relief;
  rs.innerHTML = "";
  const off = document.createElement("option");
  off.value = "off"; off.textContent = "off"; rs.appendChild(off);
  for(const f of c.fields){
    if(f.ncomp !== 1) continue;
    const o = document.createElement("option");
    o.value = o.textContent = f.name;
    rs.appendChild(o);
  }
  S.relief = c.fields.some(f => f.name === had && f.ncomp === 1) ? had : "off";
  rs.value = S.relief;
  $("reliefopts").style.display = S.relief === "off" ? "none" : "inline-flex";
  S.shades.clear();

  $("tslider").max = String(S.nsteps-1);
  $("tslider").value = String(S.t);
  fitView();
  await selectField(fs.value);
}

async function selectField(key){
  const [f, c] = key.split("|");
  S.field = f; S.comp = c || "";
  $("field").value = key;
  S.hover = null;   // a readout from the previous field must not linger
  S.frames.clear(); S.failed.clear(); S.gen++;
  reliefNote();
  updateProgress();
  note("scanning value range over " + S.nsteps + " steps ...");
  ctx.clearRect(0,0,S.nx,S.ny);

  const qs = new URLSearchParams({coll:S.coll, field:S.field, comp:S.comp});
  let r;
  try{
    r = await (await fetch("/api/range?" + qs)).json();
  }catch(e){ note("range scan failed: " + e); return; }
  if(r.error){ note(r.error); return; }
  S.range = r;
  S.maskVal = r.vmin;
  $("maskVal").value = String(r.vmin);
  note(r.constant ? ("'" + S.field + "' is constant at " + fmtNum(r.vmin) + " over every step")
                  : "");
  if(logImpossible()){
    note("'" + S.field + "' has no values above zero, so it is shown linearly");
    S.scale = "lin"; $("scale").value = "lin";
  }
  applyRangeMode();
}

// A log window needs a positive floor. A field that reaches zero (thk on bare
// ground, a signed field's negatives) gets the smallest positive value in the
// series instead of an impossible one, and is told so.
function fixLogFloor(){
  if(S.scale !== "log" || S.lo > 0) return false;
  const r = S.range || {};
  const from = [r.vminpos, S.hi * 1e-4].find(v => v > 0 && isFinite(v)) || 1e-6;
  S.lo = Math.min(from, S.hi > 0 ? S.hi / 2 : from);
  return true;
}

// Nothing positive at all, so there is nothing a log window could sit on.
function logImpossible(){
  return S.scale === "log" && !(S.range && S.range.vmax > 0);
}

function applyRangeMode(){
  const r = S.range; if(!r) return;
  if(S.rmode === "global"){ S.lo = r.vmin; S.hi = r.vmax; }
  else if(S.rmode === "robust"){ S.lo = r.p1; S.hi = r.p99; }
  if(!(S.hi > S.lo)){ S.hi = S.lo + (Math.abs(S.lo) || 1) * 1e-6; }
  if(fixLogFloor()){
    note("log needs a positive minimum; using " + fmtNum(S.lo)
         + ", the smallest above zero in this field");
  }
  $("lo").value = trim(S.lo); $("hi").value = trim(S.hi);
  S.encLo = S.lo; S.encHi = S.hi; S.encLog = logOK();
  drawColorbar(); setT(S.t); startPrefetch();
}

// ------------------------------------------------------- frames + prefetch
function frameURL(t){
  const qs = new URLSearchParams({
    coll:S.coll, field:S.field, comp:S.comp, t:String(t),
    lo:String(S.encLo), hi:String(S.encHi), fmt:S.fmt, v:S.version, d:String(S.d),
    scale:(S.encLog ? "log" : "lin"),
  });
  return "/api/frame?" + qs;
}

// Turns an encoded layer into raw 8-bit codes.
// The size check is not paranoia: an image of the wrong size drawn at natural
// size lands in the corner of the scratch canvas and the rest of the frame is
// whatever was there before, which reads as a small tile over stale pixels.
async function decodeLayer(res, what){
  if(!res.ok) throw new Error(what + ": HTTP " + res.status);
  const bmp = await createImageBitmap(await res.blob());
  const w = bmp.width, h = bmp.height;
  if(w !== S.nx || h !== S.ny){
    bmp.close && bmp.close();
    throw new Error(what + ": got " + w + "x" + h + ", expected " + S.nx + "x" + S.ny +
                    " (stale cache or changed dataset)");
  }
  sctx.clearRect(0, 0, S.nx, S.ny);
  sctx.drawImage(bmp, 0, 0);
  bmp.close && bmp.close();
  const d = sctx.getImageData(0, 0, S.nx, S.ny).data;
  const code = new Uint8Array(S.nx * S.ny);
  for(let p=0,q=0;p<code.length;p++,q+=4) code[p] = d[q];
  return code;
}

async function fetchFrame(t){
  const code = await decodeLayer(await fetch(frameURL(t)), "frame " + t);
  return {code, lo:S.encLo, hi:S.encHi, fmt:S.fmt, log:S.encLog};
}

async function fetchShade(t){
  const qs = new URLSearchParams({
    coll:S.coll, field:S.relief, comp:"", t:String(t),
    az:String(S.az), alt:String(S.alt), zf:String(S.zf), fmt:shadeFmt(),
    v:S.version, d:String(S.d),
  });
  const code = await decodeLayer(await fetch("/api/shade?" + qs), "relief " + t);
  return {code, key:shadeKey()};
}

// A cached frame stays usable after a window change as long as nothing was
// clamped away and we are not stretching so few codes that banding shows.
function stale(f){
  if(!f) return true;
  if(f.fmt !== S.fmt) return true;
  if(!!f.log !== logOK()) return true;   // the codes are spaced the other way
  // judge the window in the space the codes were spread over
  const tr = v => f.log ? Math.log(v) : v;
  const fl = tr(f.lo), fh = tr(f.hi), sl = tr(S.lo), sh = tr(S.hi);
  const ew = fh - fl;
  if(!(ew > 0)) return S.hi > S.lo;
  const eps = 1e-9 * (Math.abs(fl) + Math.abs(fh) + 1);
  if(!(sl >= fl - eps) || !(sh <= fh + eps)) return true;
  return (sh - sl) < 0.7 * ew;
}

// Walks outward from the step being viewed, so whatever the user is looking at
// (and its neighbours) loads first. Colour frame before relief for a given step.
// How many steps we can afford to hold at this grid size. When the whole series
// fits, this is just nsteps and everything behaves as before; when it does not,
// the cache becomes a window that follows the step being viewed.
function cacheWindow(){
  const perStep = S.nx * S.ny * ((S.relief !== "off" && !S.rstatic) ? 2 : 1);
  if(!perStep) return S.nsteps;
  return clamp(Math.floor(CACHE_BYTES / perStep), 8, S.nsteps);
}

function trimCache(){
  const win = cacheWindow();
  if(win >= S.nsteps) return;
  const reach = Math.floor(win / 2);
  for(const t of [...S.frames.keys()]) if(Math.abs(t - S.t) > reach) S.frames.delete(t);
  if(!S.rstatic){
    for(const t of [...S.shades.keys()]) if(Math.abs(t - S.t) > reach) S.shades.delete(t);
  }
}

function nextJob(){
  const win = cacheWindow();
  const reach = win >= S.nsteps ? S.nsteps : Math.floor(win / 2);
  for(let d=0; d<=reach && d<S.nsteps; d++){
    const cands = d === 0 ? [S.t] : [S.t + d, S.t - d];
    for(const t of cands){
      if(t < 0 || t >= S.nsteps) continue;
      const fk = "f" + t;
      if(!S.inflight.has(fk) && (S.failed.get(fk) || 0) < MAX_TRIES && stale(S.frames.get(t)))
        return {kind:"frame", t, key:fk};
      if(S.relief !== "off"){
        const st = shadeStep(t), sk = "s" + st;
        if(!S.inflight.has(sk) && (S.failed.get(sk) || 0) < MAX_TRIES && shadeStale(S.shades.get(st)))
          return {kind:"shade", t:st, key:sk};
      }
    }
  }
  return null;
}

function startPrefetch(){
  S.gen++;
  const gen = S.gen;
  S.failed.clear();
  for(let i=0;i<CONCURRENCY;i++) prefetchWorker(gen);
}

async function prefetchWorker(gen){
  while(gen === S.gen){
    const job = nextJob();
    if(!job){
      // Nothing claimable right now. That can mean genuinely finished, or that
      // the previous generation's requests are still draining and are holding
      // the keys we would take. Exiting in the second case would strand the
      // work, so only stop once nothing is in flight at all.
      if(S.inflight.size === 0) break;
      await sleep(120);
      continue;
    }
    S.inflight.add(job.key);
    try{
      if(job.kind === "frame"){
        const f = await fetchFrame(job.t);
        if(gen !== S.gen) return;
        S.frames.set(job.t, f);
      } else {
        const sh = await fetchShade(job.t);
        if(gen !== S.gen) return;
        S.shades.set(job.t, sh);
      }
      S.failed.delete(job.key);
      trimCache();
      if(job.t === S.t || (job.kind === "shade" && shadeStep(S.t) === job.t)) draw();
    }catch(err){
      S.failed.set(job.key, (S.failed.get(job.key) || 0) + 1);
      console.warn(err);
      await sleep(150);
    }finally{
      S.inflight.delete(job.key);
    }
    updateProgress();
  }
  updateProgress();
}

function reliefNote(){ updateStatus(); }

function updateStatus(){
  const el = $("status");
  if(!el) return;
  const bits = [S.d > 1 ? (S.nx + "x" + S.ny + " of " + S.fullNx + "x" + S.fullNy)
                        : (S.fullNx + "x" + S.fullNy)];
  if(S.relief !== "off"){
    bits.push(S.rstatic ? "relief: 1 image for the series"
                        : "relief: 1 image per step (" + S.nsteps + ")");
  }
  if(S.refnote) bits.push(S.refnote);
  el.textContent = bits.join("   \u00b7   ");
}

function updateProgress(){
  const win = cacheWindow(), windowed = win < S.nsteps;
  const reach = Math.floor(win / 2);
  const from = windowed ? Math.max(0, S.t - reach) : 0;
  const to   = windowed ? Math.min(S.nsteps - 1, S.t + reach) : S.nsteps - 1;
  let ok = 0, total = 0;
  for(let t=from;t<=to;t++){ total++; if(!stale(S.frames.get(t))) ok++; }
  if(S.relief !== "off"){
    if(S.rstatic){
      total += 1;
      if(!shadeStale(S.shades.get(0))) ok++;
    } else {
      for(let t=from;t<=to;t++){ total++; if(!shadeStale(S.shades.get(t))) ok++; }
    }
  }
  const pct = total ? (100*ok/total) : 0;
  $("progfill").style.width = pct.toFixed(1) + "%";
  $("progtxt").textContent = ok + "/" + total + " cached" + (windowed ? " (window)" : "");
  $("prog").title = windowed
    ? ("this grid is too big to hold the whole series, so " + win +
       " steps around the current one are kept")
    : "frames cached at the current window";
}

// ------------------------------------------------------------------ render
// Banding lives here so the image, the colourbar and an exported GIF agree.
// Mirrors level_lut() on the server.
function mapNorm(n){
  if(S.levels > 0) n = (Math.min(Math.floor(n * S.levels), S.levels - 1) + 0.5) / S.levels;
  return n;
}

function shadeFmt(){ return S.fmt === "png" ? "png" : "shade"; }
function shadeKey(){ return [S.relief, S.az, S.alt, S.zf, shadeFmt(), S.d].join("|"); }
function shadeStale(e){ return !e || e.key !== shadeKey(); }
function shadeStep(t){ return S.rstatic ? 0 : t; }
function shadeFor(t){
  if(S.relief === "off") return null;
  const e = S.shades.get(shadeStep(t));
  return shadeStale(e) ? null : e;
}

// relief multiplier per shade code, in 0..256
function shadeMul(){
  const key = String(S.intensity);
  if(key === mulKey) return mulBuf;
  for(let k=0;k<256;k++){
    const m = (1 - S.intensity) + S.intensity * (k / 255);
    mulBuf[k] = clamp(Math.round(m * 256), 0, 256);
  }
  mulKey = key;
  return mulBuf;
}

// Draping at less than full opacity mixes the colour toward the bare ground
// before any shading. Folding it into the 256-entry table keeps the per-pixel
// loop untouched, so opacity costs nothing to drag.
function drapeAlpha(){
  return (S.relief !== "off") ? S.opacity : 1;
}

function baseColor(cm, i, out, o){
  const a = drapeAlpha();
  if(a >= 1){ out[o] = cm[i]; out[o+1] = cm[i+1]; out[o+2] = cm[i+2]; return; }
  const g = (1 - a) * TERRAIN_GREY;
  out[o]   = Math.round(a * cm[i]   + g);
  out[o+1] = Math.round(a * cm[i+1] + g);
  out[o+2] = Math.round(a * cm[i+2] + g);
}

function buildLUT(f){
  const dlog = logOK();
  const key = [f.lo, f.hi, f.log?1:0, S.lo, S.hi, dlog?1:0, S.cmap,
               S.mask?1:0, S.maskVal, S.levels, drapeAlpha()].join("|");
  if(key === lutKey) return lutBuf;
  const cm = CM[S.cmap] || CM.viridis;
  const tol = maskTol();
  for(let k=0;k<256;k++){
    const v = codeValue(k, f);
    const n = mapNorm(clamp(normOf(v, S.lo, S.hi, dlog), 0, 1));
    const i = clamp(Math.round(n * 255), 0, 255) * 3;
    const o = k << 2;
    baseColor(cm, i, lutBuf, o);
    lutBuf[o+3] = (S.mask && v <= S.maskVal + tol) ? 0 : 255;
  }
  lutKey = key;
  return lutBuf;
}

function draw(){
  hud();
  const f = S.frames.get(S.t);
  // Keep the previous frame on screen while this one loads -- blanking would
  // strobe during a scrub -- but dim it, so stale pixels are never mistaken
  // for the step the label claims.
  cv.style.opacity = f ? "1" : "0.45";
  if(!f || !imgData) return;
  const lut = buildLUT(f), d = imgData.data, c = f.code;
  const sh = shadeFor(S.t);
  if(sh){
    const mul = shadeMul(), sc = sh.code;
    for(let p=0,q=0;p<c.length;p++,q+=4){
      const b = c[p] << 2, m = mul[sc[p]];
      if(lut[b+3] === 0){
        // hidden cell over relief: show the bare shaded ground
        const g = (TERRAIN_GREY * m) >> 8;
        d[q] = g; d[q+1] = g; d[q+2] = g; d[q+3] = 255;
      } else {
        d[q] = (lut[b]*m) >> 8; d[q+1] = (lut[b+1]*m) >> 8; d[q+2] = (lut[b+2]*m) >> 8;
        d[q+3] = 255;
      }
    }
  } else {
    for(let p=0,q=0;p<c.length;p++,q+=4){
      const b = c[p] << 2;
      d[q] = lut[b]; d[q+1] = lut[b+1]; d[q+2] = lut[b+2]; d[q+3] = lut[b+3];
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

const CB_H = 190, CB_W = 16;

// Round numbers on a log bar: whole decades, subdivided when there are too few
// and thinned when there are too many. Inside a single decade log is nearly
// linear, so ordinary nice steps read better than 1-2-5.
function cbTicks(){
  const lo = S.lo, hi = S.hi;
  const d0 = Math.floor(Math.log10(lo)), d1 = Math.ceil(Math.log10(hi));
  const gen = ms => {
    const out = [];
    for(let d = d0; d <= d1; d++)
      for(const m of ms){
        const v = m * Math.pow(10, d);
        if(v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
      }
    return out.sort((a, b) => a - b);
  };
  for(const ms of [[1], [1, 3], [1, 2, 5]]){
    const t = gen(ms);
    if(t.length >= 3 && t.length <= 9) return t;
  }
  let t = gen([1]);
  if(t.length > 9){
    for(const k of [2, 3, 5, 10]){
      const thin = t.filter((_, i) => i % k === 0);
      if(thin.length <= 9) return thin;
    }
  }
  if(t.length >= 2) return t;
  return ticksIn(lo, hi, niceStep(hi - lo, 5));
}

// short enough for a 16px-wide bar's margin; fmtNum pads 0.1 out to 0.1000
function cbLabel(v){
  const a = Math.abs(v);
  if(a === 0) return "0";
  if(a < 1e-2 || a >= 1e5) return v.toExponential(0).replace("e+", "e");
  return v.toFixed(Math.min(6, Math.max(0, -Math.floor(Math.log10(a)))));
}

function drawColorbar(){
  const cb = $("cb"), cbc = cb.getContext("2d");
  const cm = CM[S.cmap] || CM.viridis;
  const lg = logOK(), dpr = window.devicePixelRatio || 1;

  // a log bar carries its own ticks, so the two end labels come off
  let ticks = [], labels = [], wmax = 0;
  if(lg){
    cbc.setTransform(1, 0, 0, 1, 0, 0);
    cbc.font = AXFONT;
    ticks = cbTicks();
    labels = ticks.map(cbLabel);
    for(const t of labels) wmax = Math.max(wmax, cbc.measureText(t).width);
  }
  $("cbhi").style.display = $("cblo").style.display = lg ? "none" : "";

  const W = CB_W + 2 + (lg ? TICK + GAP + Math.ceil(wmax) : 0), H = CB_H;
  cb.width = Math.round(W * dpr); cb.height = Math.round(H * dpr);
  cb.style.width = W + "px"; cb.style.height = H + "px";
  cbc.setTransform(dpr, 0, 0, dpr, 0, 0);
  cbc.clearRect(0, 0, W, H);

  const GH = H - 2;                       // the gradient, inside its border
  for(let r = 0; r < GH; r++){
    const i = clamp(Math.round(mapNorm(1 - r / (GH - 1)) * 255), 0, 255) * 3;
    const px = [0, 0, 0];
    baseColor(cm, i, px, 0);
    cbc.fillStyle = "rgb(" + px[0] + "," + px[1] + "," + px[2] + ")";
    cbc.fillRect(1, 1 + r, CB_W, 1);
  }
  cbc.strokeStyle = "#2b303b"; cbc.lineWidth = 1;
  cbc.strokeRect(0.5, 0.5, CB_W + 1, H - 1);

  if(lg){
    const yOf = v => 1 + (1 - clamp(normOf(v, S.lo, S.hi, true), 0, 1)) * (GH - 1);
    cbc.font = AXFONT;
    cbc.fillStyle = "#9aa3b2"; cbc.strokeStyle = "#6b7486";
    cbc.textAlign = "left"; cbc.textBaseline = "middle";
    cbc.beginPath();
    for(let i = 0; i < ticks.length; i++){
      const y = Math.round(yOf(ticks[i])) + .5;
      cbc.moveTo(CB_W + 1, y); cbc.lineTo(CB_W + 1 + TICK, y);
      cbc.fillText(labels[i], CB_W + 1 + TICK + GAP, y);
    }
    cbc.stroke();
  } else {
    $("cblo").textContent = fmtNum(S.lo);
    $("cbhi").textContent = fmtNum(S.hi);
  }
}

function hud(){
  const f = S.frames.get(S.t);
  const lines = [];
  lines.push("<b>" + S.field + (S.comp === "" ? "" : (S.comp === "mag" ? " |mag|" : "[" + S.comp + "]")) + "</b>"
             + "   step <b>" + S.t + "</b>/" + (S.nsteps-1)
             + "   t = <b>" + fmtNum(S.times[S.t]) + "</b>");
  if(S.hover){
    lines.push("x " + fmtNum(S.hover.x) + "  y " + fmtNum(S.hover.y)
               + "   [" + S.hover.col + "," + S.hover.row + "]");
    const approx = !S.hover.exact && S.fmt === "webpfast";
    lines.push("value <b>" + (S.hover.v === null ? "-" : (approx ? "~" : "") + fmtNum(S.hover.v)) + "</b>"
               + (S.hover.exact ? "  (exact)" : (approx ? "  (lossy - click for exact)" : "  (click for exact)")));
  } else if(!f){
    lines.push("loading ...");
  }
  $("hud").innerHTML = lines.join("\n");
}

// ------------------------------------------------------------------ axes
// Drawn on an overlay canvas in screen pixels rather than on the image itself:
// the data canvas is CSS-scaled by the zoom, and text and hairlines scaled with
// it would be unreadable. The overlay also masks whatever the image has been
// panned into the margins, so the labels always sit on background.
const axc = $("ax").getContext("2d");
const AXFONT = "11px ui-monospace,SFMono-Regular,Menlo,monospace";
const STAGE_BG = "#0b0d12";     // matches #stage
const TICK = 5, GAP = 4, LINEH = 12;

// Served grid <-> the file's own coordinates. Deliberately the same arithmetic
// as hoverAt(), so the value you read at a labelled coordinate is that one.
function physX(col){ return S.origin[0] + col * S.spacing[0] * S.d; }
function physY(row){ return S.origin[1] + (S.ny - 1 - row) * S.spacing[1] * S.d; }
function colOf(x){ return (x - S.origin[0]) / (S.spacing[0] * S.d); }
function rowOf(y){ return S.ny - 1 - (y - S.origin[1]) / (S.spacing[1] * S.d); }

function niceStep(span, want){
  const raw = Math.abs(span) / Math.max(1, want);
  if(!(raw > 0)) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}

// The .vti says nothing about units, so labels never claim any. Long coordinates
// get a shared power of ten instead, quoted once on the axis.
function axisScale(lo, hi){
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  return m >= 1e5 ? Math.pow(10, 3 * Math.floor(Math.log10(m) / 3)) : 1;
}

function tickText(v, step, k){
  const dec = clamp(Math.ceil(-Math.log10(step / k)), 0, 6);
  const s = (v / k).toFixed(dec);
  return s === "-0" ? "0" : s;
}

function expLabel(k){
  if(k === 1) return "";
  const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  return "  ×10" + String(Math.round(Math.log10(k))).replace(/\d/g, d => sup[+d]);
}

// Margins are sized from the whole grid, not the visible part, so they do not
// twitch while panning.
function axisGutter(){
  if(S.axes === "off" || !S.nx) return {l:0, r:0, t:0, b:0};
  axc.font = AXFONT;
  const lo = physY(S.ny - 1), hi = physY(0);
  const k = axisScale(lo, hi), step = niceStep(hi - lo, 6);
  const w = Math.max(axc.measureText(tickText(lo, step, k)).width,
                     axc.measureText(tickText(hi, step, k)).width);
  return {l: Math.ceil(w) + TICK + GAP * 2 + LINEH, r: 12,
          t: 10, b: TICK + GAP * 2 + LINEH * 2};
}

function ticksIn(lo, hi, step){
  const out = [];
  const eps = step * 1e-6;
  for(let i = Math.ceil(lo / step - 1e-9); i * step <= hi + eps; i++) out.push(i * step);
  return out;
}

function drawAxes(){
  const st = $("stage"), ax = $("ax");
  const W = st.clientWidth, H = st.clientHeight, dpr = window.devicePixelRatio || 1;
  if(ax.width !== Math.round(W * dpr) || ax.height !== Math.round(H * dpr)){
    ax.width = Math.round(W * dpr); ax.height = Math.round(H * dpr);
    ax.style.width = W + "px"; ax.style.height = H + "px";
  }
  axc.setTransform(dpr, 0, 0, dpr, 0, 0);
  axc.clearRect(0, 0, W, H);
  if(S.axes === "off" || !S.nx) return;

  const g = axisGutter();
  const x0 = g.l, x1 = W - g.r, y0 = g.t, y1 = H - g.b;
  if(x1 - x0 < 40 || y1 - y0 < 40) return;

  // mask the margins: the image is free to be panned under them
  axc.fillStyle = STAGE_BG;
  axc.fillRect(0, 0, W, y0); axc.fillRect(0, y1, W, H - y1);
  axc.fillRect(0, y0, x0, y1 - y0); axc.fillRect(x1, y0, W - x1, y1 - y0);

  const s = S.base * S.zoom;
  // a cell's coordinate belongs to its centre, half a pixel in from its edge
  const sxOf = x => S.panx + (colOf(x) + 0.5) * s;
  const syOf = y => S.pany + (rowOf(y) + 0.5) * s;
  const xOf = sx => physX((sx - S.panx) / s - 0.5);
  const yOf = sy => physY((sy - S.pany) / s - 0.5);

  const dx = [physX(0), physX(S.nx - 1)].sort((a, b) => a - b);
  const dy = [physY(0), physY(S.ny - 1)].sort((a, b) => a - b);
  const vx = [Math.max(Math.min(xOf(x0), xOf(x1)), dx[0]),
              Math.min(Math.max(xOf(x0), xOf(x1)), dx[1])];
  const vy = [Math.max(Math.min(yOf(y0), yOf(y1)), dy[0]),
              Math.min(Math.max(yOf(y0), yOf(y1)), dy[1])];

  // The data's own rectangle. Ticks hang off it rather than off the window, so
  // the axes stay against the image; if a zoom pushes an edge out of view the
  // baseline clamps to the plot area and the labels fall into the margin.
  const dl = sxOf(dx[0]) - s / 2, dr = sxOf(dx[1]) + s / 2;
  const dt = syOf(dy[1]) - s / 2, db = syOf(dy[0]) + s / 2;
  const fl = clamp(dl, x0, x1), fr = clamp(dr, x0, x1);
  const ft = clamp(dt, y0, y1), fb = clamp(db, y0, y1);

  const showX = vx[1] > vx[0] && fr > fl, showY = vy[1] > vy[0] && fb > ft;
  const stepX = niceStep(vx[1] - vx[0], Math.max(2, Math.round((fr - fl) / 110)));
  const stepY = niceStep(vy[1] - vy[0], Math.max(2, Math.round((fb - ft) / 70)));
  const kX = axisScale(vx[0], vx[1]), kY = axisScale(vy[0], vy[1]);
  const tx = showX ? ticksIn(vx[0], vx[1], stepX) : [];
  const ty = showY ? ticksIn(vy[0], vy[1], stepY) : [];

  if(S.axes === "grid"){
    axc.strokeStyle = "rgba(230,233,239,.10)";
    axc.lineWidth = 1;
    axc.beginPath();
    for(const v of tx){ const X = Math.round(sxOf(v)) + .5; axc.moveTo(X, ft); axc.lineTo(X, fb); }
    for(const v of ty){ const Y = Math.round(syOf(v)) + .5; axc.moveTo(fl, Y); axc.lineTo(fr, Y); }
    axc.stroke();
  }

  axc.strokeStyle = "#4b5364";
  axc.lineWidth = 1;
  axc.strokeRect(Math.round(fl) + .5, Math.round(ft) + .5,
                 Math.max(0, Math.round(fr - fl)), Math.max(0, Math.round(fb - ft)));

  axc.font = AXFONT;
  axc.fillStyle = "#9aa3b2";
  axc.strokeStyle = "#6b7486";
  axc.beginPath();
  axc.textAlign = "center"; axc.textBaseline = "top";
  for(const v of tx){
    const X = Math.round(sxOf(v)) + .5;
    axc.moveTo(X, fb + .5); axc.lineTo(X, fb + TICK);
    axc.fillText(tickText(v, stepX, kX), X, fb + TICK + GAP);
  }
  let wmax = 0;
  axc.textAlign = "right"; axc.textBaseline = "middle";
  for(const v of ty){
    const Y = Math.round(syOf(v)) + .5, txt = tickText(v, stepY, kY);
    wmax = Math.max(wmax, axc.measureText(txt).width);
    axc.moveTo(fl + .5, Y); axc.lineTo(fl - TICK, Y);
    axc.fillText(txt, fl - TICK - GAP, Y);
  }
  axc.stroke();

  axc.fillStyle = "#c3cad6";
  if(showX){
    axc.textAlign = "center"; axc.textBaseline = "top";
    axc.fillText("x" + expLabel(kX), (fl + fr) / 2, fb + TICK + GAP + LINEH);
  }
  if(showY){
    axc.save();
    axc.translate(Math.max(GAP + 1, fl - TICK - GAP * 2 - wmax - LINEH), (ft + fb) / 2);
    axc.rotate(-Math.PI / 2);
    axc.textAlign = "center"; axc.textBaseline = "top";
    axc.fillText("y" + expLabel(kY), 0, 0);
    axc.restore();
  }
}

// -------------------------------------------------------------- view / zoom
function fitView(){
  const st = $("stage"), g = axisGutter();
  const w = Math.max(20, st.clientWidth - g.l - g.r);
  const h = Math.max(20, st.clientHeight - g.t - g.b - 4);
  S.base = Math.min(w / S.nx, h / S.ny) * 0.94;
  S.zoom = 1;
  S.panx = g.l + (w - S.nx * S.base) / 2;
  S.pany = g.t + (h - S.ny * S.base) / 2;
  applyView();
}

function applyView(){
  const s = S.base * S.zoom;
  cv.style.transform = "translate(" + S.panx + "px," + S.pany + "px) scale(" + s + ")";
  drawAxes();
}

function pixelAt(ev){
  const r = cv.getBoundingClientRect();
  const col = Math.floor((ev.clientX - r.left) / r.width * S.nx);
  const row = Math.floor((ev.clientY - r.top) / r.height * S.ny);
  if(col < 0 || row < 0 || col >= S.nx || row >= S.ny) return null;
  return {col, row};
}

function hoverAt(ev){
  const p = pixelAt(ev);
  if(!p){ S.hover = null; hud(); return; }
  const f = S.frames.get(S.t);
  let v = null;
  if(f) v = codeValue(f.code[p.row * S.nx + p.col], f);
  // physical coords: image row 0 is the top, i.e. the largest y. One served
  // pixel spans d cells, so step by d to get back to model space.
  S.hover = {
    col:p.col, row:p.row, v, exact:false,
    x: S.origin[0] + p.col * S.spacing[0] * S.d,
    y: S.origin[1] + (S.ny - 1 - p.row) * S.spacing[1] * S.d,
  };
  hud();
}

async function probeAt(ev){
  const p = pixelAt(ev);
  if(!p) return;
  // The probe reads the file, so it wants full-resolution indices. Columns just
  // scale by d, but rows are flipped for display, so scaling the image row
  // directly would land on a neighbouring cell.
  const dataRow = (S.ny - 1 - p.row) * S.d;               // row in the model, 0 = y min
  const fullRow = clamp(S.fullNy - 1 - dataRow, 0, S.fullNy - 1);   // back to image order
  const qs = new URLSearchParams({
    coll:S.coll, field:S.field, comp:S.comp, t:String(S.t),
    col:String(clamp(p.col * S.d, 0, S.fullNx - 1)),
    row:String(fullRow),
  });
  try{
    const r = await (await fetch("/api/probe?" + qs)).json();
    if(r.error) return;
    S.hover = {col:p.col, row:p.row, v:r.value, exact:true, x:r.x, y:r.y};
    hud();
  }catch(e){ console.warn(e); }
}

// ------------------------------------------------------------------ transport
function setT(t){
  S.t = clamp(t, 0, S.nsteps - 1);
  trimCache();
  $("tslider").value = String(S.t);
  $("tlabel").textContent = "step " + S.t + " / " + (S.nsteps-1) + "   t=" + fmtNum(S.times[S.t]);
  draw();
}

let lastAdvance = 0;
function playLoop(ts){
  if(!S.playing) return;
  if(ts - lastAdvance >= 1000 / S.fps){
    const nxt = (S.t + 1) % S.nsteps;
    if(S.frames.has(nxt)){ setT(nxt); lastAdvance = ts; }
  }
  requestAnimationFrame(playLoop);
}

function setPlaying(on){
  S.playing = on;
  $("play").innerHTML = on ? "&#9208;" : "&#9205;";
  if(on){ lastAdvance = 0; requestAnimationFrame(playLoop); }
}

// ----------------------------------------------------------------- wiring
let reTimer = null;
function windowChanged(){
  lutKey = "";
  drawColorbar(); draw();
  clearTimeout(reTimer);
  reTimer = setTimeout(() => {
    // Re-encode at the new window; stale() decides which frames actually need it.
    S.encLo = S.lo; S.encHi = S.hi; S.encLog = logOK();
    startPrefetch(); updateProgress();
  }, 450);
}

function wire(){
  $("coll").addEventListener("change", e => selectCollection(e.target.value));
  $("field").addEventListener("change", e => selectField(e.target.value));
  $("cmap").addEventListener("change", e => {
    S.cmap = e.target.value; lutKey = ""; drawColorbar(); draw();
  });
  $("rmode").addEventListener("change", e => {
    S.rmode = e.target.value;
    if(S.rmode === "manual"){ windowChanged(); } else { applyRangeMode(); }
  });
  for(const id of ["lo","hi"]){
    $(id).addEventListener("change", () => {
      const lo = parseFloat($("lo").value), hi = parseFloat($("hi").value);
      if(!isFinite(lo) || !isFinite(hi) || hi <= lo){ $(id).value = trim(S[id]); return; }
      if(S.scale === "log" && lo <= 0){
        note("a log scale needs a minimum above zero");
        $("lo").value = trim(S.lo);
        return;
      }
      S.lo = lo; S.hi = hi; S.rmode = "manual"; $("rmode").value = "manual";
      note("");
      windowChanged();
    });
  }

  $("scale").addEventListener("change", e => {
    S.scale = e.target.value;
    if(logImpossible()){
      note("'" + S.field + "' has no values above zero, so a log scale has nothing to sit on");
      S.scale = "lin"; e.target.value = "lin";
      return;
    }
    note("");
    // Re-derive rather than patch, so the window still follows the range mode
    // and a later field change picks its own window up again.
    if(S.rmode !== "manual"){ applyRangeMode(); updateProgress(); return; }
    if(fixLogFloor()){
      note("log needs a positive minimum; using " + fmtNum(S.lo)
           + ", the smallest above zero in this field");
      $("lo").value = trim(S.lo);
    }
    // the codes themselves are spaced differently now, so this is a refetch
    S.encLo = S.lo; S.encHi = S.hi; S.encLog = logOK();
    lutKey = ""; drawColorbar(); draw();
    startPrefetch(); updateProgress();
  });
  // banding is a pure recolour: no refetch, the cached codes are unchanged
  $("levels").addEventListener("change", e => {
    S.levels = clamp(parseInt(e.target.value, 10) || 0, 0, 64);
    e.target.value = String(S.levels);
    lutKey = ""; drawColorbar(); draw();
  });

  $("axes").addEventListener("change", e => {
    S.axes = e.target.value;
    fitView();          // the margins changed, so the image has to be re-placed
  });
  $("relief").addEventListener("change", e => {
    S.relief = e.target.value;
    $("reliefopts").style.display = S.relief === "off" ? "none" : "inline-flex";
    reliefNote();
    lutKey = ""; drawColorbar(); draw(); startPrefetch(); updateProgress();
  });
  $("rstatic").addEventListener("change", e => {
    S.rstatic = e.target.checked;
    reliefNote(); draw(); startPrefetch(); updateProgress();
  });
  for(const [id, key] of [["az","az"], ["alt","alt"], ["zf","zf"]]){
    $(id).addEventListener("change", e => {
      const v = parseFloat(e.target.value);
      if(!isFinite(v)) return;
      S[key] = v;
      startPrefetch(); updateProgress();   // the sun moved, so every shade is stale
    });
  }
  // strength only scales what is already downloaded
  $("intensity").addEventListener("input", e => {
    S.intensity = clamp(parseFloat(e.target.value) / 100, 0, 1);
    draw();
  });
  // opacity only rewrites the colour table: no refetch, nothing per-pixel
  $("opacity").addEventListener("input", e => {
    S.opacity = clamp(parseFloat(e.target.value) / 100, 0, 1);
    lutKey = ""; drawColorbar(); draw();
  });

  $("maskOn").addEventListener("change", e => { S.mask = e.target.checked; lutKey=""; draw(); });
  $("maskVal").addEventListener("change", e => {
    S.maskVal = parseFloat(e.target.value) || 0; lutKey=""; draw();
  });
  $("fmt").addEventListener("change", e => {
    S.fmt = e.target.value; startPrefetch(); updateProgress();
  });
  $("detail").addEventListener("change", e => {
    S.detail = e.target.value;
    applyDetail(); fitView(); drawColorbar(); draw();
    startPrefetch(); updateProgress();
  });

  $("tslider").addEventListener("input", e => setT(parseInt(e.target.value, 10)));
  $("first").onclick = () => setT(0);
  $("last").onclick  = () => setT(S.nsteps - 1);
  $("prev").onclick  = () => setT(S.t - 1);
  $("next").onclick  = () => setT(S.t + 1);
  $("play").onclick  = () => setPlaying(!S.playing);
  $("fps").addEventListener("change", e => { S.fps = clamp(parseFloat(e.target.value)||10, 1, 60); });
  $("reset").onclick = () => fitView();
  $("refresh").onclick = () => refreshListing();

  window.addEventListener("keydown", e => {
    if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const step = e.shiftKey ? 10 : 1;
    if(e.key === "ArrowRight"){ setT(S.t + step); e.preventDefault(); }
    else if(e.key === "ArrowLeft"){ setT(S.t - step); e.preventDefault(); }
    else if(e.key === "Home"){ setT(0); e.preventDefault(); }
    else if(e.key === "End"){ setT(S.nsteps - 1); e.preventDefault(); }
    else if(e.key === " "){ setPlaying(!S.playing); e.preventDefault(); }
    else if(e.key === "r" || e.key === "R"){ refreshListing(); e.preventDefault(); }
  });

  // pan / zoom
  let drag = null;
  cv.addEventListener("mousedown", e => { drag = {x:e.clientX, y:e.clientY, px:S.panx, py:S.pany, moved:false}; });
  window.addEventListener("mousemove", e => {
    if(drag){
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if(Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      S.panx = drag.px + dx; S.pany = drag.py + dy; applyView();
    }
  });
  window.addEventListener("mouseup", e => {
    if(drag && !drag.moved && e.target === cv) probeAt(e);
    drag = null;
  });
  cv.addEventListener("mousemove", hoverAt);
  cv.addEventListener("mouseleave", () => { S.hover = null; hud(); });
  cv.addEventListener("dblclick", () => fitView());
  $("stage").addEventListener("wheel", e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const before = S.base * S.zoom;
    S.zoom = clamp(S.zoom * (e.deltaY < 0 ? 1.15 : 1/1.15), 0.1, 40);
    const after = S.base * S.zoom;
    S.panx -= (after - before) * S.nx * fx;
    S.pany -= (after - before) * S.ny * fy;
    applyView();
  }, {passive:false});
  window.addEventListener("resize", () => fitView());

  // export
  $("savepng").onclick = () => {
    cv.toBlob(b => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = [S.coll, S.field, S.comp, "t" + S.t].filter(Boolean).join("_") + ".png";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    });
  };
  $("savegif").onclick = () => {
    const stride = Math.max(1, Math.ceil(S.nsteps / 300));
    const est = Math.round(S.nsteps / stride);
    if(!confirm("Build a " + est + "-frame GIF on the server (every " + stride +
                " step" + (stride>1?"s":"") + ")?\nThis renders remotely, then downloads a few MB.")) return;
    const qs = new URLSearchParams({
      coll:S.coll, field:S.field, comp:S.comp,
      lo:String(S.lo), hi:String(S.hi), cmap:S.cmap,
      stride:String(stride), fps:String(S.fps), levels:String(S.levels),
      relief:(S.relief === "off" ? "" : S.relief),
      az:String(S.az), alt:String(S.alt), zf:String(S.zf),
      static:(S.rstatic ? "1" : "0"), intensity:String(S.intensity),
      mask:(S.mask ? "1" : "0"), maskval:String(S.maskVal),
      opacity:String(S.opacity), d:String(S.d),
      scale:(logOK() ? "log" : "lin"),
    });
    window.location.href = "/api/gif?" + qs;
  };
}

boot().catch(err => { note("startup failed: " + err); console.error(err); });
</script>
</body>
</html>
"""


# Exact matplotlib colormap tables (256x3 uint8), embedded so the server has no
# matplotlib dependency and the browser shares the identical lookup tables.
CMAPS_B64.update({
    "viridis":
        "RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl",
    "magma":
        "AAAEAQAFAQEGAQEIAgEJAgILAgINAwMPAwMSBAQUBQQWBgUYBgUaBwYcCAceCQcgCggiCwkkDAkmDQopDgsrEAstEQwvEg0xEw00FA42FQ44Fg87GA89GRA/GhBCHBBEHRFHHhFJIBFLIRFOIhFQJBJTJRJVJxJYKRFaKhFcLBFfLRFhLxFjMRFlMxBnNBBpNhBrOBBsOQ9uOw9wPQ9xPw9yQA90Qg91RA92RRB3RxB4SRB4ShB5TBF6ThF7TxJ7URJ8UhN8VBN9VhR9VxV+WRV+WhZ+XBZ/XRd/Xxh/YBiAYhmAZBqAZRqAZxuAaByBahyBax2BbR2Bbh6BcB+Bch+BcyCBdSGBdiGBeCKBeSKCeyOCfCOCfiSCgCWCgSWBgyaBhCaBhieBiCeBiSiBiymBjCmBjiqBkCqBkSuBkyuAlCyAliyAmC2AmS2Amy5/nC5/ni9/oC9/oTB+ozB+pTF+pjF9qDJ9qjN9qzN8rTR8rjR7sDV7sjV7szZ6tTZ6tzd5uDd5ujh4vDl4vTl3vzp3wDp2wjt1xDx1xTx0xz1zyD5zyj5yzD9xzUBxz0Bw0EFv0kJv00Nu1URt1kVs2EVs2UZr20dq3Ehp3klo30po4Exn4k1m405l5E9k5VBk51Jj6FNi6VRi6lZh61dg7Fhg7Vpf7lte711e8F9e8WBd8mJd8mRc82Vc9Gdc9Glc9Wtc9mxc9m5c93Bc93Jc+HRc+HZc+Xhd+Xld+Xtd+n1e+n9e+oFf+4Nf+4Vg+4dh/Ilh/Ipi/Ixj/I5k/JBl/ZJm/ZRn/ZZo/Zhp/Zpq/Ztr/p1s/p9t/qFu/qNv/qVx/qdy/qlz/qp0/qx2/q53/rB4/rJ6/rR7/rZ8/rd+/rl//ruB/r2C/r+E/sGF/sKH/sSI/saK/siM/sqN/syP/s2Q/s+S/tGU/tOV/tWX/teZ/tia/dqc/dye/d6g/eCh/eKj/eOl/eWn/eep/emq/eus/Oyu/O6w/PCy/PK0/PS2/Pa4/Pe5/Pm7/Pu9/P2/",
    "inferno":
        "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k",
    "plasma":
        "DQiHEAeIEweJFgeKGQaMGwaNHQaOIAaPIgaQJAaRJgWRKAWSKgWTLAWULgWVLwWWMQWXMwWXNQSYNwSZOASaOgSaPASbPgScPwScQQSdQwOeRAOeRgOfSAOfSQOgSwOhTAKhTgKiUAKiUQKjUwKjVQKkVgGkWAGkWQGlWwGlXAGmXgGmYAGmYQCnYwCnZACnZgCnZwCoaQCoagCobACobgCobwCocQCocgGodAGodQGodwGoeAGoegKoewKofQOofgOogASogQSngwWnhAWnhgamhwemiAimigmliwqljQuljgykjw2kkQ6jkg+jlBCilRGhlhOhmBSgmRWfmhafnBeenRidnhmdoBqcoRuboh2aox6apR+ZpiCYpyGXqCKWqiOVqySUrCaUrSeTriiSsCmRsSqQsiuPsyyOtC6NtS+MtjCLtzGKuDKJujOIuzSIvDWHvTeGvjiFvzmEwDqDwTuCwjyBwz2AxD5/xUB+xkF9x0J8yEN7yUR6ykV6y0Z5zEd4zEl3zUp2zkt1z0x00E1z0U5y0k9x01Fx1FJw1VNv1VRu1lVt11Zs2Fdr2Vhq2lpq2ltp21xo3F1n3V5m3l9l3mFk32Jj4GNj4WRi4mVh4mZg42hf5Gle5Wpd5Wtd5mxc525b529a6HBZ6XFY6XJX6nRX63VW63ZV7HdU7XlT7XpS7ntR73xR735Q8H9P8IBO8YFN8YNM8oRL84VL84dK9IhJ9IlI9YtH9YxG9o1F9o9E95BE95FD95NC+JRB+JVA+Zc/+Zg++Zo++ps9+pw8+p47+586+6E5+6I4/KM4/KU3/KY2/Kg1/Kk0/asz/awz/a4y/a8x/bEw/bIv/bQv/bUu/rct/rgs/ros/rsr/r0q/r4q/sAp/cIp/cMo/cUn/cYn/cgn/com/csm/M0l/M4l/NAl/NIl+9Mk+9Uk+9ck+tgk+tok+dwk+d0l+N8l+OEl9+Il9+Ql9uYm9ugm9ekm9esn9O0n8+4n8/An8vIn8fQm8fUl8Pck8Pkh",
    "turbo":
        "MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD",
    "cividis":
        "ACJOACNPACRRACVTACVUACZWACdYAChZAChbACldACpfACphACtiACxkACxmAC1oAC5qAC5sAC9tADBvADBwADFwADFxATJxBTNxCDNwDDRwDzVwEjVwFDZwFjdwGDdvGjhvHDlvHjpvIDpvITtuIzxuJDxuJj1uJz5uKT9uKj9tK0BtLUFtLkFtL0JtMUNtMkNtM0RtNEVsNUVsNkZsOEdsOUhsOkhsO0lsPEpsPUpsPktsP0xsQExsQU1sQk5sQ05sRE9sRVBsRlFsR1FsSFJsSVNsSlNsS1RsTFVsTVVsTlZsT1dsUFdsUVhtUlltU1ptVFptVVttVVxtVlxtV11tWF5tWV5uWl9uW2BuXGFuXWFuXmJuXmNvX2NvYGRvYWVvYmVvY2ZwZGdwZWhwZWhwZmlwZ2pxaGpxaWtxamxxa21ybG1ybG5ybW9ybm9zb3BzcHFzcXJ0cnJ0cnN0c3R1dHR1dXV1dnZ2d3d2d3d3eHh3eXl3enp4e3p4fHt4fXx4fnx4fn14f354gH94gX94goB5g4F5hIJ5hYJ5hoN5h4R4iIV4iYV4ioZ4i4d4jIh4jYh4jol4j4p4kIt4kYt4kox4ko14k454lI53lY93lpB3l5F3mJJ3mZJ3mpN2m5R2nJV2nZV2npZ2n5d1oJh1oZl1opl1o5p0pJt0pZx0ppx0p51zqJ5zqZ9zqqBzq6ByrKFyraJyrqNxr6RxsKVxsaVws6ZwtKdvtahvtqlvt6luuKpuuattuqxtu61tvK5sva5svq9rv7BrwLFqwbJqwrNpw7NpxLRoxbVoxrZnx7dnyLhmyblly7llzLpkzbtjzrxjz71i0L5i0b9h0sBg08Bf1MFf1cJe1sNd18Rc2cVc2sZb28da3MhZ3chY3slY38pX4MtW4cxV4s1U5M5T5c9S5tBR59FQ6NJP6dNO6tNM69RL7dVK7tZJ79dI8NhG8dlF8tpE89tC9dxB9t0/994++N88+eA6++E4/OI2/eM0/uQ0/uU1/uY2/ug4",
    "coolwarm":
        "O0zAPE7CPVDDPlHFP1PGQFXIQlfJQ1jLRFrMRVzORl7PSF/RSWHSSmPTS2TVTGbWTmjYT2nZUGvaUW3bU27dVHDeVXLfVnPgWHXhWXfjWnjkW3rlXXzmXn3nX3/oYYDpYoLqY4TrZIXsZoftZ4juaIrvaovva43wbI/xbpDyb5LzcJPzcpX0c5b1dZf2dpn2d5r3eZz4ep34e5/5faD5fqH6gKP6gaT7gqb7hKf8haj8hqn8iKv9iaz9i639jK/+jbD+j7H+kLL+krT+k7X+lLb/lrf/l7j/mLn/mrv/m7z/nb3/nr7/n7//ocD/osH/o8L+pcP+psT+p8X+qcb9qsf9q8j9rcn9rsn8r8r8scv8ssz7s837tc36ts76t8/5udD5utD4u9H4vNL3vtL2v9P2wNT1wdT0w9X0xNXzxdbyxtbxx9fwydfwytjvy9juzNntzdnsztrrz9rq0drp0tvo09vn1Nvm1dvl1tzk19zj2Nzi2dzh2tzg29ze3N3d3dzc3tzb39vZ4NvY4drW4trV49nT5NnS5djR5tfP59fO6NbM6dXL6tXJ6tTI69PG7NPF7dLD7dHC7tDA78+/78698M278c268cy48su38sq18sm088iy88ex9Mav9MWt9cSs9cKq9cGp9cCn9r+m9r6k9r2i97yh97qf97me97ic97eb97WZ97SX97OW97GU97CT96+R962Q96yO96qM96mL96iJ96aI9qWG9qOF9qKD9aCB9Z+A9Z1+9Zx99Jp79Jh685d485V385R18pJ08pBy8Y9x8Y1v8Itu8Ips74hr7oZp7oRo7YNm7IFl7H9j631i6ntg6Xpf6Xhd6HZc53Rb5nJZ5XBY5G5W42xV42tU4mlS4WdR4GVP32NO3mFN3V9L3F1K2lpJ2VhH2FZG11RF1lJE1VBC1E5B0ktA0Uk/0Ec9z0U8zUI7zEA6yz44yjs3yDg2xzY1xTM0xDAywy4xwSswwCgvviQuvR8tuxssuhYruBIqtw0otQkntAQm",
    "RdBu_r":
        "BTBhBjJkBzRnCDZqCThtCjtwDD1zDT92DkF5D0N7EEV+EUeBEkmEE0yHFE6KFVCNF1KQGFSTGVaWGliZG1qcHFyfHV+iHmGlH2OoIGWrImesI2mtJGquJmyvJ26wKHCxKnGyK3OzLHW0Lne1L3m1MHq2Mny3M364NIC5NoG6N4O7OIW8Ooe9O4i+PIq+Poy/P47AQI/BQpHCQ5PDRpXESZfFTJnGT5vHUp3IVp/JWaHKXKPLX6XNYqfOZanPaKvQa6zRbq7ScbDTdbLUeLTVe7bWfrjXgbrYhLzZh77aisDbjcLckMTdk8belsffmMjgm8ngncvhoMzios3jpc7jp9DkqdHlrNLlrtPmsdXns9bottfouNjpu9rqvdvqwNzrwt3sxd/sx+DtyuHuzOLvz+Tv0eXw0ubw1Obx1efx1+jx2Onx2uny2+ry3evy3uvy4Ozz4e3z4+3z5O705u/05/D06fD06vH17PL17fL17/P18PT28vX28/X29fb39vf39/b29/X0+PTy+PPw+PLv+PHt+fDr+e/p+e7n+e3l+evj+urh+unf+uje+ufc++ba++XY++TW++PU/OLS/ODQ/N/P/N7N/d3L/dzJ/dvH/dnE/NfC/NW//NO8+9C5+863+8y0+sqx+siv+cas+cSp+cKn+L+k+L2h+Lue97mc97eZ97WW9rOU9rGR9q+O9ayL9aqJ9aiG9KaD86SB8qF/8Z598Jx775l57pZ37JN065Fy6o5w6Ytu6Ils5oZq5YNo5IBm435k4nti4Xhg33Ze3nNc3XBZ3G5X22tV2mhT2GVR12NP1mBN1V1M01pK0lhJ0FVIz1JGzk9FzExEy0lCyUdByERAxkE+xT49xDs8wjg6wTY5vzM4vjA2vS01uyo0uigyuCUxtyIwth8utBwtsxkssRgrrhcqqxYqqBUppRQpohMonxIonBEnmRAnlg8nkw4mkA0mjQwligslhwokhAkkgQgjfwgjfAcieQYidgUhcwQhcAMgbQIgagEfZwAf",
    "Blues":
        "9/v/9vr/9fr+9fn+9Pn+8/j+8vj98vf98ff98Pb97/b87vX87vX87fT87PT76/P76vP76vL76fL66PH65/H65/D65vD55e/55O/54+754+744u344e344Oz43+z33+v33uv33er33Or23On22+n22uj22ej12ef12Of11+b11ub01uX01eX01OT00+Tz0+Pz0uPz0eLz0OLy0OHyz+HyzuDyzeDxzd/xzN/xy97xyt7wyt3wyd3wyNzwx9zvx9vvxtvvxNruw9ruwtnuwdntv9jtvtjsvdfsvNfrutbrudbquNXqt9TqtdTptNPps9PostLosNLnr9HnrtHnrdDmq9Dmqs/lqc/lqM7kps7kpc3jpMzjo8zjocvioMvin8rhncrhnMnhmsjgmcfgl8bflcXflMTfksTekcPej8LejcHdjMDdir/dib7ch73chbzchLzbgrvbgbrbf7nafbjafLfaerbZebXZd7XZdbTYdLPYcrLYcbHXb7DXba/XbK7Waq7Waa3VaKzVZqvUZarUZKnTY6jTYafSYKfSX6bRXaXRXKTQW6PQWqLPWKHPV6DOVqDOVJ/NU57NUp3MUZzMT5vLTprLTZnKS5jKSpjJSZfJSJbIRpXIRZTHRJPHQpLGQZHGQJDFP4/FPo7EPY3EPIzDO4vCOorCOYnBOIjBN4fANobANYW/NIS/M4O+MoK+MYG9MIC9L3+8Ln68LX27LHy6K3u6Knq5KXm5J3e4Jna4JXW3JHS3I3O2InK2IXG1IHC0IG+0H26zHm2yHWyxHGuwHGqwG2mvGmiuGWetGWatGGWsF2SrFmOqFWKpFWGpFGCoE1+nEl6mEl2mEVylEFukD1qjDlmiDliiDVehDFagC1WfClSeClOeCVKdCFGcCFCbCE+ZCE6YCE2WCEyVCEuTCEqRCEmQCEiOCEeNCEaLCEWKCESICEOHCEKFCEGECECCCD6BCD1/CDx9CDt8CDp6CDl5CDh3CDd2CDZ0CDVzCDRxCDNwCDJuCDFtCDBr",
    "terrain":
        "MzOZMjacMDieLzuhLj6kLECmK0OpKkasKEiuJ0uxJk60JFC2I1O5Ila8IFi+H1vBHl7EHGDGG2PJGmbMGGjOF2vRFm7UFHDWE3PZEnbcEHjeD3vhDn7kDIDmC4PpCobsCIjuB4vxBo70BJD2A5P5Apb8AJj+AJv7AJz1AJ7vAKHpAKPjAKXdAKbXAKnRAKvLAK3FAK+/ALG5ALOzALWtALanALmhALubAL2VAL+PAMGJAMODAMV9AMd3AMlxAMtrAcxmBc1nCc5oDc9pEc9pFdBqGdFrHdJsIdNtJdNtKdRuLdVvMdZwNddxOddxPdhyQdlzRdp0Sdt1Tdt1Udx2Vd13Wd54Xd95Yd95ZeB6aeF7beJ8ceN9deN9eeR+feV/geaAheeBieeBjeiCkemDleqEmeuFneuFoeyGpe2Hqe6Ire+Jse+JtfCKufGLvfKMwfONxfONyfSOzfWP0faQ1feR2feR3fiS4fmT5fqU6fuV7fuV8fyW9f2X+f6Y/f+Z/v6Y/PuX+vmW+PaV9vOU9PGT8u6S8OyR7umQ7OeP6uSO6OKN5t+L5NyK4tqJ4NeI3tWH3NKG2tCF2M2E1suD1MiC0sWB0MOAzsB/zL59yrt8yLl7xrZ6xLN5wrF4wK53vqx2vKl1uqd0uKRztqJytJ9xspxvsJpurpdtrJVsqpJrqJBqpo1ppItooohnoIVmnoNlnIBkmn5imHthlnlglHZfknNekHFdjm5cjGxbimlaiGdZhmRYhGJXgl9WgFxUgV5Wg2BZhWNch2VeiWhhi2tkjW1mj3BpkXJsk3VulXdxl3p0mXx2m395nYJ8n4R+oYeBo4mEpYyGp46JqZGMq5OOrZaRr5mUsZuWs56ZtaCct6OfuaWhu6ikvaunv62pwbCsw7KvxbWxx7e0ybq3y7y5zb+8z8K/0cTB08fE1cnH18zJ2c7M29HP3dPR39bU4dnX49vZ5d7c5+Df6ePi6+Xk7ejn7+vq8e3s8/Dv9fLy9/X0+ff3+/r6/fz8////",
    "gray":
        "AAAAAQEBAgICAwMDBAQEBQUFBgYGBwcHCAgICQkJCgoKCwsLDAwMDQ0NDg4ODw8PEBAQEREREhISExMTFBQUFRUVFhYWFxcXGBgYGRkZGhoaGxsbHBwcHR0dHh4eHx8fICAgISEhIiIiIyMjJCQkJSUlJiYmJycnKCgoKSkpKioqKysrLCwsLS0tLi4uLy8vMDAwMTExMjIyMzMzNDQ0NTU1NjY2Nzc3ODg4OTk5Ojo6Ozs7PDw8PT09Pj4+Pz8/QEBAQUFBQkJCQ0NDRERERUVFRkZGR0dHSEhISUlJSkpKS0tLTExMTU1NTk5OT09PUFBQUVFRUlJSU1NTVFRUVVVVVlZWV1dXWFhYWVlZWlpaW1tbXFxcXV1dXl5eX19fYGBgYWFhYmJiY2NjZGRkZWVlZmZmZ2dnaGhoaWlpampqa2trbGxsbW1tbm5ub29vcHBwcXFxcnJyc3NzdHR0dXV1dnZ2d3d3eHh4eXl5enp6e3t7fHx8fX19fn5+f39/gICAgYGBgoKCg4ODhISEhYWFhoaGh4eHiIiIiYmJioqKi4uLjIyMjY2Njo6Oj4+PkJCQkZGRkpKSk5OTlJSUlZWVlpaWl5eXmJiYmZmZmpqam5ubnJycnZ2dnp6en5+foKCgoaGhoqKio6OjpKSkpaWlpqamp6enqKioqampqqqqq6urrKysra2trq6ur6+vsLCwsbGxsrKys7OztLS0tbW1tra2t7e3uLi4ubm5urq6u7u7vLy8vb29vr6+v7+/wMDAwcHBwsLCw8PDxMTExcXFxsbGx8fHyMjIycnJysrKy8vLzMzMzc3Nzs7Oz8/P0NDQ0dHR0tLS09PT1NTU1dXV1tbW19fX2NjY2dnZ2tra29vb3Nzc3d3d3t7e39/f4ODg4eHh4uLi4+Pj5OTk5eXl5ubm5+fn6Ojo6enp6urq6+vr7Ozs7e3t7u7u7+/v8PDw8fHx8vLy8/Pz9PT09fX19vb29/f3+Pj4+fn5+vr6+/v7/Pz8/f39/v7+////",
})


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def serve(directory: str, port: int, host: str = "127.0.0.1", range_frames: int = 0):
    store = Store(directory, range_frames)
    Handler.store = store
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise SystemExit(f"cannot bind {host}:{port}: {exc}")
    httpd.daemon_threads = True

    print(f"[rview] serving {store.dir} on http://{host}:{port}", file=sys.stderr)
    for c in store.colls.values():
        h = c.header
        print(f"[rview]   {c.name}: {c.nsteps} steps, {h.nx}x{h.ny}, "
              f"fields: {', '.join(sorted(c.header.arrays))}", file=sys.stderr)
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[rview] stopped", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Serve 2D VTI/PVD time series as small colormapped frames.")
    ap.add_argument("directory", help="directory holding the .pvd collections")
    ap.add_argument("-p", "--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default localhost only; reach it via an ssh tunnel)")
    ap.add_argument("--range-frames", type=int, default=0, metavar="N",
                    help="scan at most N evenly spaced timesteps when computing a field's "
                         "value range (0 = all; use a small value when the data is behind "
                         "a slow mount)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.directory):
        raise SystemExit(f"not a directory: {args.directory}")
    serve(args.directory, args.port, args.host, args.range_frames)


if __name__ == "__main__":
    main()
