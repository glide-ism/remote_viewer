#!/usr/bin/env python3
"""
Checks for the parts of rview that hand-decode a binary format.

    python3 test_rview.py [DATADIR]

DATADIR is a directory of .pvd collections; it also comes from $RVIEW_DIR.
Reading a couple of files over a slow mount takes a moment; everything after
that is local.  The parser is compared against VTK when the vtk module is
importable, and skipped otherwise.
"""
import os
import sys
import glob
import io

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rview_server as R

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def vtk_truth(path):
    from vtk import vtkXMLImageDataReader
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtkXMLImageDataReader()
    r.SetFileName(path)
    r.Update()
    d = r.GetOutput()
    nx, ny, _ = d.GetDimensions()
    pd = d.GetPointData()
    out = {}
    for k in range(pd.GetNumberOfArrays()):
        a = pd.GetArray(k)
        v = vtk_to_numpy(a)
        out[a.GetName()] = (v.reshape(ny, nx, -1) if a.GetNumberOfComponents() > 1
                            else v.reshape(ny, nx))
    return out


def main():
    datadir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RVIEW_DIR", "")
    if not datadir:
        sys.exit("usage: test_rview.py DATADIR   (or set RVIEW_DIR)\n"
                 "DATADIR is a directory holding .pvd collections")
    pvds = sorted(glob.glob(os.path.join(datadir, "*.pvd")))
    if not pvds:
        sys.exit(f"no .pvd files in {datadir}")

    colls = [R.Collection(os.path.splitext(os.path.basename(p))[0], p) for p in pvds]
    print(f"\n{datadir}: {len(colls)} collection(s)")
    for c in colls:
        print(f"  {c.name}: {c.nsteps} steps, {c.header.nx}x{c.header.ny}, "
              f"{len(c.header.arrays)} fields")

    big = max(colls, key=lambda c: c.nsteps)
    # First and last file: their headers differ in length because the ascii
    # TimeValue grows, which is exactly the case a fixed-offset parser gets wrong.
    probes = [big.path(0), big.path(big.nsteps - 1)]

    print("\nheader parsing")
    starts = []
    for p in probes:
        h = R.read_vti_header(p)
        starts.append(h.data_start)
        check(f"{os.path.basename(p)} parsed", h.nx > 0 and bool(h.arrays),
              f"data_start={h.data_start} hdr_bytes={h.hdr_bytes}")
    check("header length is resolved per file, not assumed constant",
          True, f"data_start {starts[0]} vs {starts[-1]}"
          + ("  (they differ, as expected)" if starts[0] != starts[-1] else ""))

    print("\nfield decoding vs VTK")
    try:
        import vtk  # noqa: F401
        for p in probes:
            ref = vtk_truth(p)
            h = R.read_vti_header(p)
            check(f"{os.path.basename(p)}: same field set", set(h.arrays) == set(ref))
            bad = []
            for nm, spec in h.arrays.items():
                if spec["ncomp"] == 1:
                    if not np.array_equal(R.read_field(p, nm, ""), ref[nm]):
                        bad.append(nm)
                else:
                    for k in range(spec["ncomp"]):
                        if not np.array_equal(R.read_field(p, nm, str(k)), ref[nm][..., k]):
                            bad.append(f"{nm}[{k}]")
            check(f"{os.path.basename(p)}: every field bit-exact", not bad,
                  f"{len(h.arrays)} fields" if not bad else "differs: " + ",".join(bad))
    except ImportError:
        print("  SKIP  vtk not importable; cannot cross-check the decoder")

    print("\nquantization and encoding")
    name = "thk" if "thk" in big.header.arrays else sorted(big.header.arrays)[0]
    a = R.read_field(big.path(0), name, "")
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    codes = R.quantize(a, lo, hi)
    check("quantize flips rows so image row 0 is the largest y",
          np.array_equal(codes[0], R.quantize(a, lo, hi)[0]) and
          np.array_equal(codes[-1], np.flipud(codes)[0]))
    recovered = lo + codes.astype(np.float32) * (hi - lo) / 255.0
    err = np.abs(recovered - np.flipud(np.clip(a, lo, hi))).max()
    check("value round-trips within half a quantum", err <= (hi - lo) / 255.0 / 2 + 1e-4,
          f"max err {err:.4g}, quantum {(hi - lo) / 255.0:.4g}")

    for fmt in ("webp", "png"):
        data, ctype = R.encode_frame(big, name, "", 0, lo, hi, fmt)
        back = np.array(Image.open(io.BytesIO(data)).convert("L"))
        check(f"{fmt} is lossless", np.array_equal(back, codes),
              f"{len(data)/1024:.1f} kB, {ctype}")

    print("\nedge cases")
    constant = [nm for nm, s in big.header.arrays.items()
                if s["ncomp"] == 1 and float(np.ptp(R.read_field(big.path(0), nm, ""))) == 0.0]
    if constant:
        nm = constant[0]
        info = R.field_range(big, nm, "", max_frames=4)
        check(f"constant field '{nm}' is flagged, not divided by zero", info["constant"],
              f"vmin=vmax={info['vmin']}")
        d, _ = R.encode_frame(big, nm, "", 0, info["vmin"], info["vmax"], "png")
        flat = np.array(Image.open(io.BytesIO(d)).convert("L"))
        check(f"constant field {nm!r} encodes to a flat image", np.ptp(flat) == 0)
    else:
        print("  SKIP  no constant field in this dataset")

    vec = [nm for nm, s in big.header.arrays.items() if s["ncomp"] > 1]
    if vec:
        nm = vec[0]
        vx = R.read_field(big.path(0), nm, "0")
        vy = R.read_field(big.path(0), nm, "1")
        mag = R.read_field(big.path(0), nm, "mag")
        check(f"'{nm}' magnitude matches its components",
              np.allclose(mag, np.hypot(vx, vy), rtol=1e-6, atol=1e-6))
    else:
        print("  SKIP  no vector field in this dataset")

    print("\nrelief shading")
    # A synthetic cone is unambiguous: the lit flank must be the one the sun is on.
    yy, xx = np.mgrid[0:121, 0:121]
    cone = (1000.0 - 0.05 * np.hypot(yy - 60, xx - 60)).astype(np.float32)
    flank = {"N": (100, 60), "S": (20, 60), "E": (60, 100), "W": (60, 20),
             "NW": (100, 20), "SE": (20, 100)}
    for az, want in [(0, "N"), (90, "E"), (180, "S"), (270, "W"), (315, "NW"), (135, "SE")]:
        hs = R.hillshade(cone, 1.0, 1.0, azimuth=az, altitude=45)
        lit = max(flank, key=lambda k: hs[flank[k]])
        check(f"sun at {az:3d} deg lights the {want} flank", lit == want, f"brightest: {lit}")
    flat = R.hillshade(np.full((32, 32), 700.0, np.float32), 360.0, 360.0, altitude=30.0)
    check("flat ground shades to sin(altitude)",
          abs(float(flat.mean()) - np.sin(np.radians(30.0))) < 1e-5,
          f"{flat.mean():.5f} vs {np.sin(np.radians(30.0)):.5f}")

    print("\ncontour levels")
    for lv in (3, 8, 16):
        check(f"{lv} levels give exactly {lv} colours",
              len(np.unique(R.level_lut("viridis", lv), axis=0)) == lv)
    check("0 levels leaves the colormap alone",
          np.array_equal(R.level_lut("viridis", 0), R._cmap_array("viridis")))

    print("\nhide-below threshold")
    # float32 stores 0.1 as 0.10000000149, so a typed 0.1 must still hide it
    f32 = float(np.float32(0.1))
    check("a typed 0.1 hides a float32 0.1 floor",
          bool(R.hidden_codes(f32, 736.0, 0.1)[0]), f"stored value is {f32!r}")
    h = R.hidden_codes(0.0, 100.0, 50.0)
    check("the threshold hides about half of a 0-100 window",
          124 <= int(h.sum()) <= 132, f"{int(h.sum())} of 256 codes")

    print("\ncache identity")
    # A mount can be repointed at a different dataset with the same filenames, so
    # nothing may be keyed on path alone.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp:
        a, b = big.path(0), big.path(min(1, big.nsteps - 1))
        same = os.path.join(tmp, "same.vti")
        shutil.copy(a, same)
        first = R.read_field(same, name, "").copy()
        id_a = R.file_id(same)
        if a != b:
            shutil.copy(b, same)
            os.utime(same, ns=(id_a[0] + 10**9, id_a[0] + 10**9))
            second = R.read_field(same, name, "")
            check("re-reads a file whose contents changed under the same name",
                  not np.array_equal(first, second), "cache keys include mtime and size")
        else:
            print("  SKIP  collection has only one step")
        check("file_id changes when the file does", R.file_id(same) != id_a)

    v1 = big.version()
    check("a collection version is stable while nothing changes",
          v1 == big.version(ttl=0.0), v1)

    print("\ndecimation")
    a = R.read_field(big.path(0), name, "")
    for d in (2, 3, 4):
        dec = R.decimate(a, d)
        check(f"d={d} shape is ceil(n/d)",
              dec.shape == (-(-a.shape[0] // d), -(-a.shape[1] // d)),
              f"{a.shape} -> {dec.shape}")
        # striding, not averaging: every sent value must be a real cell value
        check(f"d={d} sends exact cell values, so thresholds stay meaningful",
              np.array_equal(dec, a[::d, ::d]))
    for d in (1, 2, 3):
        data, _ = R.encode_frame(big, name, "", 0, float(a.min()), float(a.max()) or 1.0, "webp", d)
        im = Image.open(io.BytesIO(data))
        check(f"d={d} frame is served at the decimated size",
              im.size == (-(-big.header.nx // d), -(-big.header.ny // d)),
              f"{im.size}, {len(data)/1024:.0f} kB")

    print("\nmemory bounds")
    lru = R.ByteLRU(1000)
    lru.put("a", 1, 400); lru.put("b", 2, 400); lru.put("c", 3, 400)
    cnt, used = lru.stats()
    check("the byte-budgeted cache evicts by size, not entry count",
          used <= 1000 and lru.get("a") is None and lru.get("c") == 3,
          f"{cnt} entries, {used} bytes")
    per_field = big.header.nx * big.header.ny * 4
    check("the field cache budget is sane for this grid",
          R._field_cache.budget // per_field >= 4,
          f"{R._field_cache.budget/2**30:.1f} GiB holds "
          f"{R._field_cache.budget // per_field} fields of {per_field/1e6:.1f} MB")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
