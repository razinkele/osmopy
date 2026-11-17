# Opt-in minimal outputs for calibration, 2026-11-17

Item 4 of `docs/osmopy-performance-optimization-plan.md`: "an opt-in mode that skips output
collectors a calibration objective does not read."

**Result: ships.** `output.minimal.enabled = true` makes a one-thread 10-yr Baltic run 8.7% faster
(28.63 → 26.13 s, three interleaved pairs, each between −8.4% and −9.5%). Biomass, abundance and
yield stay bit-identical, and every skipped output reads as absent. `scripts/calibrate_baltic.py`'s
`run_simulation` turns it on.

## Where the time was

A calibration evaluation needs per-species biomass, abundance and yield. Measured on a warm one-thread
10-yr Baltic run (29.4 s):

| cost | seconds | share |
|---|---:|---:|
| diet tracking (in the mortality kernel) and its per-step aggregation | ~1.6 | 5.4% |
| life-stage split (`_collect_by_life_stage`, also snapshotted before reproduction) | 0.59 | 2.0% |
| mortality-by-cause table | 0.21 | 0.7% (kept) |
| age/size distributions | 0.20 | 0.7% |
| mean TL, mean size, SSB, yield by number | 0.16 | 0.5% |
| biomass, abundance, yield | 0.07 | 0.2% (kept) |

The diet cost was measured by switching diet output off: the biomass and yield hashes were
identical, and the wall time fell from 28.66 to 27.10 s (three pairs). The other rows were timed in
process. Last week's tail profile put "all output collection" at 5.1% because it never saw the diet
tracking, which runs inside the kernel.

**Not worth doing: the disk path.** `run_simulation` writes every output file to a temp directory
and reads biomass and yield back. Against `run_in_memory` that cost nothing measurable (29.40 and
29.74 s against 29.51 and 29.63 s), so it stays as it is.

## The mode

- **`output.minimal.enabled`** (engine key, not in the schema; it is a calibration switch, not a
  UI option). `_parse_output_flags` turns every optional output flag off, so the existing gates
  skip the collectors, the writers and the diet tracking. `output_step0_include` is the exception:
  it decides which steps are recorded, so it is left as configured.
- **`_collect_outputs`** returns biomass, abundance, mortality by cause and yield only. It skips the
  life-stage split and the distributions, which have no flags of their own, and the loop no longer
  snapshots the stage split.
- **Absent, not zero.** In memory, a skipped output's getter raises `FileNotFoundError` ("has no
  'dietMatrix' output. Available: [...]"). On disk, its file is not written.
- **`calibrate_baltic.run_simulation`** sets it before the overrides, which can turn it off. It is
  left off when the recruitment diagnostic asks for abundance by age.
- **`OsmoseCalibrationProblem`** (the NSGA-II path) is not changed. Its objectives are
  user-supplied, and one built-in objective reads the diet.

## Tests (`tests/test_output_minimal.py`)

- The mode is off by default.
- It turns the optional outputs off but keeps the record frequency and step-0 setting.
- Biomass, abundance and yield are bit-identical to a full run. The fixture is the Baltic-derived
  tutorial config, with diet output on.
- Skipped outputs read as absent, both in memory and on disk.
- `run_simulation` runs minimal, its scored statistics equal a full run's, and an override turns
  the mode off.
