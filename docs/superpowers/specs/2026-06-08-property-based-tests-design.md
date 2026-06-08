# Property-based tests via Hypothesis — Design

**Date:** 2026-06-08
**Status:** Approved direction (brainstormed; codebase-grounded; breadth locked to a focused 4).

## Motivation

The OSMOSE Python suite (225 test files) is almost entirely **example-based**: each pure
function is exercised by a handful of hand-built fixtures. That leaves the *space between the
examples* unexplored — exactly where parser edge cases and aggregation-boundary bugs hide. This
adds **Hypothesis property-based tests** to four pure-Python targets where (a) an invariant is
crisp and checkable for *any* valid input, (b) current coverage is thin, and (c) a failure is
unambiguously a real bug (no Java-parity coupling). It is purely additive test infrastructure —
no production code changes (unless a property uncovers a real bug; see *If a property finds a bug*).

## Verified context (audit)

All four targets and their gotchas were read in the current tree before this spec:

- **Config round-trip** — `osmose/config/reader.py::OsmoseConfigReader.read(master) -> dict[str,str]`
  lowercases keys, records original case in `key_case_map`, strips trailing `";,:\t ="` from values,
  auto-detects the per-line separator, and **injects `flat["_osmose.config.dir"]`** (reader.py:87).
  `osmose/config/writer.py::OsmoseConfigWriter.write(config, output_dir, key_case_map=None)` buckets
  keys by routing prefix, writes sub-files + an `osm_all-parameters.csv` master, and **regenerates
  `osmose.configuration.{suffix}` reference keys** pointing at its own filenames (writer.py:88).
  → A naive `read(write(d)) == d` false-fails on `_osmose.config.dir` and the regenerated
  `osmose.configuration.*` references. The property must compare on the **substantive keys we put in**.
- **Preamble detection** — `osmose/results.py::_detect_preamble_lines(path) -> int` returns the index
  of the header = the first line whose field count equals the next line's **and is > 1**; falls back
  to the first `>1`-field line; caches on `(mtime_ns, size)`. `_read_output_csv` uses it. Currently
  tested only implicitly (via loading real output). The two-header mortality reader is a separate
  path (`_read_mortality_csv`) — **out of scope**.
- **Trophic aggregation** — `osmose/trophic_network.py::diet_network_at(output_dir, *, time,
  threshold=5.0, predator_level="species") -> df[predator,prey,proportion]` reads a wide diet CSV
  via `rglob("*_dietMatrix*.csv")`, so a property must **write a temp dietMatrix CSV** (matching the
  existing `_diet_fixture` test pattern) and call it on that dir. Invariants: prey-stage SUM (exact),
  predator unweighted mean over LIVE stages (0-sum dead stage excluded), threshold filter, NaN drop.
- **Size-spectrum** — `osmose/size_spectrum.py::compute_size_spectrum(output_dir, ...)` reads a CSV
  from disk, but the invariant-bearing logic is in pure helpers that take plain lists/frames:
  `_large_fish_indicator(edges, values, threshold) -> float` (∈[0,1]), `_mean_size(midpoints,
  values) -> float`, `_infer_bin_width(edges) -> float` (median of consecutive diffs),
  `_community_long(wide) -> long`, `_window_by_time(df, time_col, window_years) -> df`. **Fuzz the
  helpers directly** (no file-format scaffolding); `compute_size_spectrum` end-to-end stays covered by
  the existing real-EEC example test.
- **Dependency / settings** — `hypothesis` is **not** a dependency (no `from hypothesis` anywhere).
  `pyproject.toml` has `[project.optional-dependencies]` with a `dev`/test extra. There is a shared
  `tests/conftest.py`. CI lint targets `osmose/ ui/ tests/` (so the new test + strategies files **must
  be ruff-clean**). CI runs the full pytest suite.

## Architecture

Two components: a shared strategies module + four property test files (one per target, matching the
per-module test convention). Hypothesis is added as a test/dev optional dependency and configured for
**deterministic, CI-safe** runs.

### 0. Dependency + settings

- `pyproject.toml`: add `"hypothesis>=6.100"` to the existing dev/test optional-dependency extra
  (already installed in `.venv` will be confirmed by the plan; if absent the plan installs it).
- `tests/conftest.py`: register and load a Hypothesis **`"ci"` profile** with
  `max_examples=150`, `deadline=None` (no per-example timeout → no flaky timing failures under load),
  `derandomize=True` (a green run stays green; reproducible), and
  `suppress_health_check=[HealthCheck.too_slow]` only if needed. Load it via
  `settings.register_profile("ci", ...)` + `settings.load_profile("ci")` guarded by
  `pytest.importorskip("hypothesis")` so a missing dep skips rather than errors.
- `.gitignore`: add `.hypothesis/` (the example database directory).

### 1. Shared strategies — `tests/strategies.py` (new)

Hypothesis `@st.composite` strategies producing **valid-but-diverse** inputs (the discipline that
keeps failures meaningful, not "garbage in"):

- `config_keys()` → OSMOSE-shaped dotted lowercase keys (`species.linf.sp{i}`, `predation.*`,
  `grid.*`, `simulation.*`) drawn from a safe alphabet (`[a-z0-9.]`, `sp{0..9}` suffixes); never
  empty, never `_`-prefixed, never containing a separator char (`= ; , : tab`).
- `config_values()` → non-empty strings from a safe alphabet that **survive the reader's
  normalization**: no leading/trailing whitespace, no trailing `";,:\t ="`, no separator chars
  (so the value is not re-split). Includes digits, decimals, simple words, paths-without-separators.
- `config_kv_dicts()` → `dict(config_keys, config_values)` with ≥1 entry, unique keys.
- `csv_texts()` → tuples `(text, k, ncols)`: `k` (0..3) single-field preamble lines (realistic OSMOSE
  description rows, width 1), then a header + `1..4` data rows each with `ncols` (2..6) fields joined
  by the delimiter `_detect_preamble_lines` splits on (the plan confirms the exact delimiter and the
  strategy matches it). Single-field preamble lines guarantee the first equal-width-`>1` pair is the
  header/first-data-row, so the detector must return `k`.
- `diet_matrices()` → a wide `Time,Prey,<predator-stage cols>` DataFrame: 1..3 predator species each
  with 1..3 size-stages, a prey set including some predator species (self-loops) and a resource;
  cells are non-negative percents per stage (optionally a fully-zero "dead" stage; optionally NaN
  cells); a single Time value. Returned as a frame the test writes to `tmp/<x>_dietMatrix.csv`.
- `edges_and_values()` → `(edges, values)`: a sorted list of 1..8 distinct bin edges (≥0) and an
  equal-length list of non-negative values (some zero), for the size-spectrum helpers.
- `community_size_frames()` → small wide `Time,<size-bin cols>` frames for `_community_long` /
  `_window_by_time`.

### 2. Property test files (one per target)

**`tests/test_config_roundtrip_properties.py`**
- *Round-trip survives every key/value:* for `d = config_kv_dicts()`, `OsmoseConfigWriter().write(d,
  tmp)`, then `OsmoseConfigReader().read(tmp/"osm_all-parameters.csv")` → assert for every
  `(k, v) in d.items()`: `result[k] == v` (keys already lowercase; reader-injected `_osmose.config.dir`
  and writer-regenerated `osmose.configuration.*` are ignored by iterating `d`, not `result`).
- *Separator invariance:* a single `key<sep>value` line parses to the same `{key: value}` for each
  `sep in {=, ;, ,, :, tab}` (drives `read_file` on a one-line temp file). Asserts the auto-detect
  produces identical k/v regardless of which separator joined them.

**`tests/test_results_preamble_properties.py`**
- *Detects the planted header:* for `(text, k, ncols) = csv_texts()` written to a temp file,
  `_detect_preamble_lines(path) == k`.
- *Never raises on degenerate input:* for arbitrary small text (empty, single line, all width-1 rows,
  blank lines), `_detect_preamble_lines` returns an `int` and does not raise.
- *Idempotent / cache-stable:* two consecutive calls on an unmodified file return the same value.

**`tests/test_trophic_network_properties.py`** (uses `diet_matrices()` written to a temp CSV)
- *Proportion bounds & clean node names:* every returned `proportion` is `0 ≤ p ≤ 100 + 1e-9`; no
  node id (predator or prey) contains `" in ["`, at `predator_level="species"`.
- *Threshold monotonicity:* for `t_lo ≤ t_hi`, the edge set at `t_hi` is a **subset** of the edge set
  at `t_lo` (same `(predator,prey)` keys; `↑threshold ⇒ ⊆ edges`).
- *Prey-sum exactness:* at `predator_level="stage"`, each `(stage, prey-species)` proportion equals
  the exact sum of that prey's size-stage cells in the source (additive composition).
- *Dead stage never surfaces:* when `diet_matrices()` plants an all-zero predator size-stage, that
  stage label never appears as a predator at `predator_level="stage"`. (The subtler species-level
  "mean divides by live-stage count" stays an example test — it needs a known expected value that a
  generic fuzz can't assert cheaply.)

**`tests/test_size_spectrum_properties.py`** (pure helpers; lists/frames, no disk)
- *LFI bounded:* `_large_fish_indicator(edges, values, threshold)` ∈ `[0, 1]` for any
  `edges_and_values()` (or `NaN` only when total value ≤ 0, asserted explicitly).
- *Mean size bounded:* `_mean_size(midpoints, values)` is within `[min(midpoints), max(midpoints)]`
  when total value > 0 (else `NaN`).
- *Bin width = median of diffs, order-invariant:* `_infer_bin_width(edges)` equals the median of
  consecutive differences and is invariant to input ordering.
- *Window keeps only in-range rows:* `_window_by_time(df, "time", w)` returns only rows with
  `time > tmax - w`, and `n_rows(out) ≤ n_rows(in)`.

## Data flow

`strategy → (write temp file where the target reads from disk | pass lists/frames directly) → call
target → assert invariant`. Hypothesis drives many examples per property; `derandomize=True` makes
the example set reproducible; failures shrink to a minimal counterexample.

## Error handling / edge cases

- A property that legitimately allows `NaN` (LFI/mean on zero total) asserts the `NaN` branch
  explicitly rather than excluding it — so the zero-total path is *covered*, not skipped.
- Float comparisons use `== pytest.approx(...)` or an explicit `1e-9` tolerance for the additive/sum
  invariants; bound checks use `<=`/`>=` with a tiny epsilon where rounding applies.
- Strategies are constrained to **valid** inputs by construction (see §1) so a failure means a real
  invariant violation, not malformed input. Where a constraint is subtle (value normalization,
  single-field preamble), the strategy comment states *why* the constraint exists.

## If a property finds a real bug

Property tests routinely surface genuine defects. Policy for this PR:
- **Small + in-scope** (e.g. a value-normalization off-by-one in the reader, a boundary in
  `_detect_preamble_lines`): fix it in this PR with a focused regression test; note it in the
  CHANGELOG.
- **Larger / risky** (would expand scope or touch parity-adjacent code): do **not** silently pass —
  mark the property `@pytest.mark.xfail(strict=True, reason="…")` with a one-line written follow-up,
  so the discovered defect is recorded and the suite stays honest.

## Testing (meta)

The property tests *are* the tests. Verification: `pytest tests/test_*_properties.py -q` passes;
the four files + `tests/strategies.py` are ruff-clean (`check` + `format --check`); the full suite
stays green; running twice yields identical results (determinism check).

## Scope / YAGNI

- **In:** `hypothesis` dev dep + `"ci"` settings profile + `.hypothesis/` gitignore; `tests/strategies.py`;
  the four property files with the properties above.
- **Out:** scenarios/fisheries/shannon/schema/history fuzzing (lower marginal signal or fiddly
  valid-structure scaffolding — deferred, not rejected); the two-header mortality reader; engine
  kernels (parity-coupled); a nightly/large-`max_examples` fuzzing harness; stateful/`RuleBasedStateMachine`
  testing; fuzzing `compute_size_spectrum` end-to-end (its invariants live in the helpers we fuzz).

## Honest limitations

- Property tests assert invariants, not exact values — they complement, not replace, the example tests.
- The config strategy deliberately excludes value classes the reader normalizes away (trailing
  separators, multi-value `;` arrays, `_`-prefixed keys); those remain covered by targeted example
  tests, not the round-trip property.
- `derandomize=True` trades exploration breadth for reproducibility; raising `max_examples` or adding
  a non-derandomized nightly profile is a clean follow-on if deeper fuzzing is ever wanted.

## Delivery

Single additive PR: `pyproject.toml` (dep), `tests/conftest.py` (profile), `.gitignore`,
`tests/strategies.py`, four `tests/test_*_properties.py`, a CHANGELOG note. No production code
changes unless a property uncovers a real bug (handled per *If a property finds a real bug*).
