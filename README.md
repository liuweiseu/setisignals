# setisignals

A CLI for SETI@home Listen `.spike` hit files (narrow-band signal detections).
Converts `.spike` data to standard astronomical table formats (FITS/HDF5) and
reproduces figures from `Notes/setiathome_listen_data_analysis.pdf` for the
HIP63121 example dataset. Parsing and binning are parallelized across CPUs
(and optionally GPUs) via [Ray](https://www.ray.io/).

## Data format

`.spike` files are pipe-delimited ASCII text, one record per line, matching
the field order of `idl/spike__define.pro`:

```
id | result_id | peak_power | mean_power | time | ra | decl | q_pix |
freq | detection_freq | barycentric_freq | fft_len | chirp_rate |
rfi_checked | rfi_found | reserved |
```

`time` is a Julian date; `freq`/`detection_freq`/`barycentric_freq` are in
Hz. Note that `rfi_checked`/`rfi_found` are `0` for all raw hits in the
HIP63121 dataset — they are not populated by the client, so the RFI/Clean
figure below is computed from scratch (see below).

## Install

```
uv sync
```

## Commands

### `convert` — format conversion

```
setisignals convert HIP63121_data/HIP63121.spike --format fits -o hip63121.fits
setisignals convert HIP63121_data/HIP63121.spike --format hdf5 -o hip63121.h5
```

Column names/order match the struct field names verbatim. HDF5 output
stores the table under the fixed dataset name `spike` (i.e.
`h5py.File(path)["spike"]`).

### `merge` — combine on-source and off-source into one table

```
setisignals merge --on HIP63121_data/HIP63121.spike --off HIP63121_data/HIP63121_OFF.spike --format fits -o merged.fits
setisignals merge --on HIP63121_data/HIP63121.spike --off HIP63121_data/HIP63121_OFF.spike -o merged.spike
```

Concatenates the on-source and off-source hits into a single output (all
on-source rows first, then all off-source rows — a plain union, no
filtering) with one extra `target` field (`"on"` or `"off"`) identifying
which file each row came from. Useful for downstream tools that expect a
single file with the on/off distinction encoded as data rather than as two
separate files.

- With `--format fits|hdf5`: parses both inputs and writes a table, as in
  `convert`.
- **Without `--format`** (the default): writes a pipe-delimited `.spike`
  text file in the original format, with `target` appended as an extra
  field on each line. Every other field is copied byte-for-byte from the
  source files (no numeric reparsing/reformatting), so this mode is exact
  and considerably faster than round-tripping through the parsed
  representation.

### `plot power-hist` — power distribution histogram

```
setisignals plot power-hist HIP63121_data/HIP63121.spike -o power_hist.png
```

Log-log histogram of `peak_power/mean_power`, reproducing the paper's
Figure 2 style.

### `plot waterfall` — on/off frequency-time scatter

```
setisignals plot waterfall --on HIP63121_data/HIP63121.spike --off HIP63121_data/HIP63121_OFF.spike -o waterfall.png
```

Reproduces the paper's Figure 3: on-source hits in cyan, off-source in
magenta.

### `plot rfi` — RFI-vs-Clean density pair

```
setisignals plot rfi --on HIP63121_data/HIP63121.spike --off HIP63121_data/HIP63121_OFF.spike -o rfi_density.png
```

Grayscale, log-scaled 2D histograms (frequency x time) splitting hits into
RFI vs. Clean.

### `plot all` — generate all three figures

```
setisignals plot all --on HIP63121_data/HIP63121.spike --off HIP63121_data/HIP63121_OFF.spike --outdir figures/
```

All commands accept `--workers N` (default: CPU count) to control Ray
parallelism, and `plot rfi` accepts `--gpu` to bin frequencies with `cupy`
instead of NumPy inside each Ray task (requires the optional `gpu` extra:
`uv sync --extra gpu`; marginal benefit at the current ~5M-row scale, mainly
useful once the much larger `.pulse` files are supported).

## Approximate reproductions — read before trusting the plots

The paper doesn't give exact algorithms for two things this tool
reconstructs from first principles:

- **On/off dwell stacking (`waterfall`, `rfi`)**: real observing dwells for
  a target are interleaved in time (on, off, on, off, ...), not independent
  sequences. Dwell/session boundaries are detected via gaps in the
  *combined* on+off time series (`analysis/time_utils.py:detect_sessions`,
  `stack_combined_on_off`), then each dwell is stacked with a vertical
  offset so bands don't overlap. This is a best-effort visual match, not a
  literal decode of how the original figures were built.
- **RFI classification (`rfi`)**: a hit is marked RFI if its `detection_freq`
  falls in the same ~93 Hz-wide frequency bin as a hit from the opposite
  on/off dataset (`analysis/rfi.py:classify_rfi`). The paper states the
  window width and the ~1% false-coincidence rationale but not the exact
  binning/matching implementation.
- **Epoch restriction (`waterfall`, `rfi`)**: `HIP63121.spike` bundles hits
  from more than one observing epoch (a ~525-day-later re-observation at a
  different receiver band is mixed into the same file). Before pairing with
  the single-epoch off-source file, on-source hits are restricted to the
  epoch that overlaps the off-source time range
  (`analysis/time_utils.py:restrict_to_epoch`) — otherwise axis ranges and
  dwell counts are thrown off by the unrelated later observation.

## Scope

Only `.spike` is implemented. The schema layer (`schema.py`) is structured
so the sibling `.pulse`/`.triplet`/`.autocorr` formats — which share
`.spike`'s first 16 fields — can be added later as additional `SignalKind`
registry entries, without restructuring the reader/writer/CLI.

## Tests

```
uv run pytest
```

Tests use small hand-written fixture files (`tests/fixtures/`), not the
real multi-hundred-MB data files.
