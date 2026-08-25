# rview

A small viewer for `.pvd` collections of 2D `.vti` files, built for flipping
through timesteps over a slow link.

## Why

The problem was never ParaView's rendering, it was the wire. Measured on this
setup:

| | |
|---|---|
| link to the data host (raw `ssh`, and `sshfs` — identical) | **~0.64 MB/s** |
| one `.vti` | 8.7 MB → **10–21 s per timestep** |
| the whole 4.2 GB dataset | ~1.8 hours |

Any viewer that pulls whole `.vti` files is stuck at about one frame per 14
seconds. So `rview` does not move `.vti` files at all: it runs a small reducer on
the machine that owns the data, which reads one field from local disk, quantizes
it to 8 bit against a fixed value window, and sends a ~30–50 kB image. The
colormap is applied in the browser.

Measured against the live 500-step collection, through the tunnel:

| | |
|---|---|
| first frame on screen | **1.35 s** (includes scanning all 500 steps for the value range) |
| one frame | ~30–50 kB, ~80 ms |
| caching all 500 steps in the background | **~45 s** |
| scrubbing/playback once cached | **>2000 fps** (entirely local) |
| relief layer | 38–46 kB, same ~80 ms as a data frame |
| static relief for a whole series | one image, ~1.5 s |

At `level_0` resolution (1856x1344, 90 m — 16x the cells) the same link gives:

| | |
|---|---|
| first frame on screen | ~2.3 s |
| one frame | ~340 kB, ~0.5 s |
| relief layer | ~700 kB, ~13 s — relief is now the expensive layer, so prefer **static** |
| redraw (recolour, levels, opacity) | ~20 ms |
| the same at `auto` detail (d=2) | ~150 kB per layer vs ~357 kB, 2.4x less |
| held in the tab | 2.5 MB per step, so 16 steps ≈ 40 MB |

Beyond about 150 steps at that grid the cache becomes a window that follows the
current step rather than holding the whole series; the progress readout says
`(window)` when that happens.
| value range over all 500 steps | 0.11 s on the data host |

## Changing datasets

Frames are cached hard, in the browser and on the data host, so scrubbing a
series you have already seen is instant. Every cacheable URL therefore carries a
version token: a fingerprint of the size and mtime of every file behind the
collection, from `/api/meta`. Same bytes, same URL, served from cache; different
data, different URL, refetched. Repointing the mount, regenerating the files, or
adding steps all change it, and the server picks that up without a restart —
its own caches are keyed on each file's identity rather than its path, and a
`.pvd` is re-read when it changes.

Reload the page after swapping the data, or press **&#8635;** — see below.
As a backstop, a decoded layer whose size does not match the current grid is
rejected outright rather than drawn, so a stale image can never end up tiled
into a corner.

## Watching a run in progress

The **&#8635;** button beside the set selector (or `r`) re-reads the `.pvd`
listing, so steps written since the page loaded appear without a reload. It also
notices a whole new `.pvd` dropped into the directory, and a set that has gone
away.

What it avoids is throwing away the cache. If step 0 and the existing timestep
values are unchanged and only later steps were added, the frames already
downloaded are kept and only the new ones are fetched, and the status line on the
right reads `+7 steps  (507)`. If step 0 changed — a rerun over the same filenames — or
the geometry or field list changed, that cannot be true, and the collection is
reloaded from scratch instead.

Two details make this workable on a directory being written into:

- A `.pvd` entry whose `.vti` has not landed yet is dropped rather than failing
  the whole collection, and is picked up on a later refresh even though the
  `.pvd` itself never changes again.
- Refreshing rescans the value range, but adopting a new window re-encodes every
  cached frame, so the window only moves when the new data actually falls outside
  it. A run that stays within its existing range costs one `/api/meta` round trip
  and the new frames, nothing else.

If you were sitting on the last step, refresh follows the new last step, so
pressing `r` every so often tails a run.

## Use

```sh
rview /path/to/vti                        # local, or on an sshfs mount
rview --host box --remote-dir /data/vti   # remote, with nothing mounted locally
rview --coll icefield                     # open straight to one collection
rview --local                             # read through the mount, not over ssh (slow)
rview --show-config                       # where each setting came from, then exit
```

Settings resolve in this order: command line, environment (`RVIEW_DIR`,
`RVIEW_HOST`, `RVIEW_REMOTE_DIR`, `RVIEW_PORT`, `RVIEW_COLL`), config file,
built-in defaults. Given nothing at all, a single sshfs mount is used when there
is exactly one; with several, it lists them and asks you to pick.

The config file is the first of `$RVIEW_CONFIG`, `./rview.toml`,
`$XDG_CONFIG_HOME/rview/config.toml`, `~/.config/rview/config.toml`. Copy
`rview.toml.example`:

```toml
directory = "/path/to/vti"   # local path, may be an sshfs mount
port = 8777
coll = "icefield"
# only when the data is not reachable through a local mount:
# host = "mybox"
# remote_dir = "/data/vti"
```

`--show-config` prints each setting, where it came from, and what it resolves
to, which is the quickest way to see why it is looking somewhere unexpected.

When the directory sits on an sshfs mount, `rview` resolves the real host and
path from `/proc/mounts`, copies `rview_server.py` to `~/.cache/rview` there,
starts it behind an `ssh -L` tunnel and opens the browser. Ctrl-C stops
everything. Give `--host`/`--remote-dir` instead when there is no local mount. A server left running from a previous session is reused when it is
the same build serving the same directory, and replaced when it is not.

To run the reducer yourself, e.g. in tmux on the data host:

```sh
python3 rview_server.py /path/to/vti -p 8777          # then tunnel to it
```

## Controls

| | |
|---|---|
| ← → | step (hold shift for 10) |
| space | play / pause |
| Home / End | first / last step |
| drag, wheel | pan, zoom; double-click or **fit** to reset |
| hover | x, y and the value under the cursor |
| click | the exact float from the file, not the 8-bit approximation |
| r, **&#8635;** | re-read the file listing (see below) |

**levels** turns the colormap into N filled contour bands (0 = smooth). It is a
pure recolour of what is already downloaded, so it costs nothing and applies to
the image, the colourbar and an exported GIF alike.

**axes** puts tick-labelled x and y around the image, in the file's own
coordinates: `Origin` and `Spacing` read straight from the `.vti`, mapped with
the same arithmetic the hover readout uses, so the value you read at a labelled
coordinate is the one that belongs there. Ticks hang off the data's own frame
and follow pan and zoom, clamping into the margin when you zoom past an edge;
**on + grid** adds faint lines across the image. A `.vti` carries no units, so
the labels never claim any — long coordinates factor out a shared power of ten
(`x  ×10³`) rather than being relabelled km.

The axes are a screen overlay. **save PNG** still writes the data raster at its
served resolution and a GIF is rendered on the data host, so neither carries
them.

**relief** drapes the field over a hillshade computed on the data host from any
scalar in the collection, usually `bed` or `srf` — the equivalent of ParaView's
warp-by-scalar trick. The shading travels as its own 8-bit layer and is
composited in the browser, so colormap, window and level changes stay instant.
With **hide ≤** on, hidden cells show the bare shaded ground rather than page
background, which is the usual data-over-hillshade drape.

- **static** shades every step with step 0's relief: one image for the whole
  series instead of one per step. Use it whenever the relief source barely
  changes (`bed`), and leave it off to watch an evolving surface (`srf`).
  Per-step relief roughly doubles the prefetch; static costs one extra image.
- **sun** is a compass azimuth (315 = north-west) and an altitude above the
  horizon. **exag** is vertical exaggeration; 1 is true geometry.
- **strength** is how hard the relief darkens; **opacity** is how opaque the
  draped field is over it. At opacity 0 only the bare relief is left, which is
  a quick way to see the terrain you are draping onto. Both are folded into the
  256-entry colour table, so dragging either is free: no refetch, no extra
  per-pixel work. The colourbar follows the blend so it still matches the image.

The relief layer is encoded lossy (~40 kB, about 0.9% mean shading error) since
it modulates brightness rather than carrying values; the data layer stays
lossless.

**range** picks the value window: 1–99% (default, robust to outliers), full
min/max, or manual. Changing it recolors what is already cached immediately and
re-fetches in the background only when the new window would actually lose
detail. **hide ≤** makes everything at or below a threshold transparent, which
is what you want for `thk`, whose ice-free cells sit at 0.1 rather than 0.

**detail** sends every d-th cell. `auto` matches the served grid to the size it
will actually be drawn at, which on a 1856x1344 grid in a normal window means
half (928x672) — the full grid is about 4x more pixels than any of them can
occupy at "fit". That is roughly **2.4x less data for no visible difference**;
switch to `full` when zooming in. Decimation strides rather than averages, so
every pixel sent is still an exact cell value: thresholds stay sharp and the
hover readout stays truthful.

Prefer detail over **quality** for this kind of data. Lossy compression is
tempting — it is 4-10x smaller — but it rings at sharp edges, and on a field
with a no-data floor (`thk` sits at 0.1 where there is no ice) that ringing
lifts ice-free cells above a `hide ≤` threshold: measured at q90 on the 90 m
grid, 3.8% of them wrongly appear, which is ~130k speckled pixels scattered over
the bare ground. Decimation has no such failure mode. If the mask is off, lossy
is perfectly usable.

**quality** is lossless by default. *fast (lossy)* is about 2.5x smaller but
values are then approximate (up to ~30 codes on a busy field), so it is for
browsing, not reading — click a cell for the exact value either way.

**save PNG** writes the current frame from the browser. **save GIF** builds the
animation on the data host, using the colormap as the GIF palette. There is no
MP4: the data host here has no ffmpeg.

## Testing

```sh
python3 test_rview.py [DATADIR]
```

Checks the hand-written `.vti` decoder against VTK field by field (bit-exact on
all 13 fields), that the per-file header length is resolved rather than assumed,
that quantization round-trips within half a quantum and the image is the right
way up, that the encoders are lossless, and the constant-field and vector cases.
Also checks that relief is lit from the direction the sun is actually in (six
azimuths against a synthetic cone — this is easy to get 90° wrong), that N
levels yield exactly N colours, and that a typed threshold of `0.1` still hides
a field whose float32 floor is `0.10000000149`.

## Requirements

Data host: python3, numpy, Pillow. No VTK, no matplotlib, no web framework.
Local: python3 and ssh. The viewer is one HTML page with no external assets.

## Limitations

- Appended **raw, uncompressed** VTI only, `PointData`, single z-slice. Anything
  else fails at startup with a message saying which file and why. Compressed or
  inline/base64 arrays would need a decompression path added.
- Frames are 8-bit over the chosen window: ~0.4% of the range. The `/api/probe`
  endpoint (and clicking) gives the exact float.
- Collections are assumed to share one geometry across their timesteps (across
  *different* collections and across remounts the geometry may differ freely).
- Relief is Lambertian shading only: no cast shadows, no ambient occlusion.
- Detail is chosen once when a collection loads, from the window size then; it
  does not follow a resize or a zoom. Set it manually when zooming in.
- A GIF with relief cannot use the colormap as its palette, so it is quantized
  to 256 adaptive colours (a plain or banded GIF is exact).
