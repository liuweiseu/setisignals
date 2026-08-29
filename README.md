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
setisignals convert HIP63121_data/HIP63121.spike -o hip63121.h5
setisignals convert HIP63121_data/HIP63121.spike --format fits -o hip63121.fits
```

`--format fits|hdf5` selects the output format (default: `hdf5`). Column
names/order match the struct field names verbatim. HDF5 output stores the
table under the fixed dataset name `spike` (i.e.
`h5py.File(path)["spike"]`).

`--targets <value>` adds a `target` column, two ways depending on the value:
- **An existing file path**: treated as a `target_time.txt`-style file
  (whitespace columns `target_name start_time end_time start_ra end_ra
  start_dec end_dec`, Julian dates) — each row's `time` is looked up
  against these windows and the matching target name is written (empty if
  no window contains that row's time). See the on/off ambiguity note below.
- **Anything else**: used as a literal label written into `target` for
  every row, e.g. `--targets HIP63121_OFF`.

### `merge` — combine two or more .spike files into one table

```
setisignals merge HIP63121_data/HIP63121.spike HIP63121_data/HIP63121_OFF.spike -o merged.h5
setisignals merge HIP63121_data/HIP63121.spike HIP63121_data/HIP63121_OFF.spike --format fits -o merged.fits
```

Takes **two or more** `.spike` files as positional arguments and
concatenates their hits into a single output (all of the first file's rows,
then the second's, and so on — a plain union, no filtering) with one extra
`target` field identifying which file each row came from. Useful for
downstream tools that expect a single file with the source distinction
encoded as data rather than as separate files.

- `--format fits|hdf5` selects the output format (default: `hdf5`).
- **`target` labeling** — three ways to control it:
  - **Default** (no `--targets`): each file's own name (stem) is used as
    its label, e.g. `"HIP63121"` / `"HIP63121_OFF"`.
  - **`--targets <target_time.txt>`** (a single existing file path):
    time-based lookup, same as `convert`'s `--targets` — resolves each
    row's actual observed target name and writes it into `target`,
    replacing the filename-derived default.
  - **`--targets <label1> --targets <label2> ...`** (one label per input
    file, repeat the flag once per file, same order as the files):
    explicit literal labels, e.g.
    `merge a.spike b.spike --targets X --targets Y` writes `"X"` for every
    row from `a.spike` and `"Y"` for every row from `b.spike`.

#### The `target_time.txt` on/off overlap

`target_time.txt` records the start/end of an entire observing block per
target, not individual dwell boundaries — so a target's on-source window and
its off-source window (e.g. `HIP63121` and `HIP63121_O`) routinely *overlap*
in time, since real observing alternates on/off/on/off/... within that span.
A naive "which window contains this time" lookup would misattribute a large
fraction of hits (verified on the real HIP63121 data: without correction,
7.5M/9.4M rows were mislabeled `HIP63121_O` including hits that were
genuinely on-source). To resolve this, `--targets <target_time.txt>` uses
each input file's name (via the same `_OFF`-suffix heuristic in both
`convert` and `merge`) to restrict candidate windows to off-variant names
(matched heuristically: case-insensitive suffix `_OFF`/`_OF`/`_O`, since
names are truncated to 10 characters at the source) for off-source files,
and non-off-variant names otherwise. On the real HIP63121 data this resolves
>99.9% of rows correctly (`HIP63121.spike` → `HIP63121`, `HIP63121_OFF.spike`
→ `HIP63121_O`); the remainder have no matching time window at all
(edge-of-window rows just outside the recorded start/end).

### `plot` — reproduce figures (FITS/HDF5 input only)

`plot` commands read the standard table formats this tool writes, not raw
`.spike` text, and every `plot` command takes exactly **one** input file
(`.fits`/`.fit`/`.h5`/`.hdf5`; passing a `.spike` file is rejected with a
clear error).

`plot power-hist` accepts any single `convert` or `merge` output. The other
three (`waterfall`, `rfi`, `all`) compare on-source against off-source, so
their one input must be a **`merge` output** whose `target` column
distinguishes on-source rows from off-source rows (an off-source label is
recognized if it ends in `_OFF`/`_OF`/`_O`, or is exactly `"off"`,
case-insensitive — matching `merge`'s own default filename-stem labels and
its `_OFF`-suffix convention) — a plain `convert` output of a single source
has no such column and is rejected with a clear error.

By default every `plot` command opens the figure in an interactive
matplotlib window instead of writing a file. Pass **`--save`** to write to
disk instead (to the path given by `-o`/`--outdir`) — without `--save`,
`-o`/`--outdir` is ignored.

```
setisignals convert HIP63121_data/HIP63121.spike --format fits -o on.fits
setisignals merge HIP63121_data/HIP63121.spike HIP63121_data/HIP63121_OFF.spike --format fits -o merged.fits
```

#### `plot power-hist` — power distribution histogram

```
setisignals plot power-hist on.fits                            # interactive window
setisignals plot power-hist on.fits --save -o power_hist.png   # save to disk
```

Log-log histogram of `peak_power/mean_power`, reproducing the paper's
Figure 2 style: a smooth power-law decline at low ratios giving way to a
sparse, visibly noisy tail at high ratios. `--n-bins` controls the log-bin
count (default: `2000`, chosen to match that granularity — a much coarser
value smooths away the noisy tail).

#### `plot waterfall` — on/off frequency-time scatter

```
setisignals plot waterfall merged.fits --save -o waterfall.png
```

Reproduces the paper's Figure 3: on-source hits in cyan, off-source in
magenta.

#### `plot rfi` — RFI-vs-Clean density pair

```
setisignals plot rfi merged.fits --save -o rfi_density.png
```

Grayscale, log-scaled 2D histograms (frequency x time) splitting hits into
RFI vs. Clean.

#### `plot all` — generate all three figures

```
setisignals plot all merged.fits                          # 3 windows at once
setisignals plot all merged.fits --save --outdir figures/  # save all 3 to figures/
```

All commands accept `--workers N` (default: CPU count) to control Ray
parallelism, and `plot rfi` accepts `--gpu` to bin frequencies with `cupy`
instead of NumPy inside each Ray task (requires the optional `gpu` extra:
`uv sync --extra gpu`; marginal benefit at the current ~5M-row scale, mainly
useful once the much larger `.pulse` files are supported).

### Global options

These go before the subcommand name, e.g. `setisignals --timestamp convert ...`:

- `--log-dir <dir>` — directory for log files (default: `./logs`, created
  if missing). Every run writes a plain-text `.log` and a structured
  `.jsonl` file there, alongside console output. Ray's own Python-side
  logging (cluster startup, `working_dir` packaging, etc.) is routed
  through these same handlers so its console/file output matches ours;
  the exception is raw subprocess output Ray forwards from its worker
  processes (lines prefixed `(raylet)`, e.g. runtime-env `uv` output),
  which bypasses Python logging entirely and can't be reformatted this way.
- `--timestamp` — on completion, log the command's total wall-clock runtime
  (high-resolution, includes Ray startup).
- `--version`/`-v` — print the installed version and exit.

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
