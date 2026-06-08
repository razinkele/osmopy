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
  `_large_fish_indicator(edges, values, threshold) -> float` (∈[0,1]; returns `0.0` not NaN on
  total≤0; compares lower `edge >= threshold`), `_mean_size(midpoints, values) -> float` (NaN on
  total≤0), `_infer_bin_width(edges) -> float` (median of consecutive diffs of `sorted(set(edges))`),
  `_window_by_time(df, time_col, window_years:int) -> df` (strict `time > tmax - w`). **Fuzz these
  helpers directly** (no file-format scaffolding); `compute_size_spectrum` end-to-end and
  `_community_long` (a plain melt) stay covered by the existing real-EEC example test.
- **Dependency / settings** — `hypothesis` is **not** a dependency and is **not installed in `.venv`**
  (verified by both spec reviewers: `import hypothesis` → ModuleNotFoundError). `pyproject.toml` has
  exactly one test/dev extra named **`dev`** (`[project.optional-dependencies].dev` — currently
  pytest, pytest-asyncio, pytest-cov, ruff, pre-commit, pyright, numba). There is a shared
  `tests/conftest.py`. CI lint targets `osmose/ ui/ tests/` (so the new test + strategies files **must
  be ruff-clean**). CI runs the full pytest suite. **The plan must `pip install hypothesis` into `.venv`.**

## Architecture

Two components: a shared strategies module + four property test files (one per target, matching the
per-module test convention). Hypothesis is added as a test/dev optional dependency and configured for
**deterministic, CI-safe** runs.

### 0. Dependency + settings

- `pyproject.toml`: add `"hypothesis>=6.100"` to the **`dev`** optional-dependency extra (the only
  test/dev extra). It is NOT currently installed in `.venv`, so the plan's first step is
  `.venv/bin/pip install "hypothesis>=6.100"`.
- `tests/conftest.py`: register and load a Hypothesis **`"ci"` profile** with
  `max_examples=150`, `deadline=None` (no per-example timeout → no flaky timing failures under load),
  `derandomize=True` (a green run stays green), and **`database=None`** (the `.hypothesis/examples`
  DB replays persisted examples on top of the derandomized set → a poisoned local DB and a fresh CI
  checkout would run *different* example sets; `database=None` makes every environment identical —
  verified). **Guard with `importlib.util.find_spec("hypothesis")`, NOT
  `pytest.importorskip`** — `importorskip` at conftest module scope raises `Skipped` → pytest
  treats it as a collection error and **the entire suite fails to run** (BLOCKER, prototype-confirmed
  exit 1). The `find_spec` guard mirrors the existing playwright guard in this conftest:
  ```python
  if importlib.util.find_spec("hypothesis") is not None:
      from hypothesis import settings
      settings.register_profile("ci", max_examples=150, deadline=None,
                                derandomize=True, database=None)
      settings.load_profile("ci")
  ```
  Each property-test *file* uses `hypothesis = pytest.importorskip("hypothesis")` at module top
  (that skips just that file cleanly when the dep is absent — verified exit 0).
- `.gitignore`: add `.hypothesis/`.
- **Temp-file mechanics (load-bearing):** three targets write a file per example. The pytest
  `tmp_path` fixture **cannot** be used under `@given` — it triggers
  `HealthCheck.function_scoped_fixture` (hard error, prototype-confirmed). Each disk-writing property
  uses `with tempfile.TemporaryDirectory() as td:` **inside the test body**. Measured @150 examples:
  preamble 0.3s, config round-trip 0.4–0.7s (fine); the diet-matrix property is ~2.2–3.3s (the cost is
  the intrinsic `diet_network_at` pandas pipeline, not I/O) → give the **diet file a per-test
  `@settings(max_examples=75)`** to keep it snappy.

### 1. Shared strategies — `tests/strategies.py` (new)

Hypothesis `@st.composite` strategies producing **valid-but-diverse** inputs (the discipline that
keeps failures meaningful, not "garbage in"):

- `config_keys()` → OSMOSE-shaped dotted lowercase keys (`species.linf.sp{i}`, `predation.*`,
  `grid.*`, `simulation.*`) drawn from a safe alphabet (`[a-z0-9.]`, `sp{0..9}` suffixes); never
  empty, never `_`-prefixed, never containing a separator char (`= ; , : tab`). **MUST exclude the
  whole `osmose.configuration.` prefix** — the writer *regenerates* those reference keys from its own
  routing, silently clobbering any user value at the 9 known suffixes (verified counterexample:
  `osmose.configuration.species` → comes back as `osm_param-species.csv`). The four listed families
  are collision-free.
- `config_values()` → non-empty strings that **survive the reader's normalization**. The reader
  splits each line on the writer's framing separator at `maxsplit=1`, so **internal separator chars
  are SAFE** (`a;b`, `x=y`, `1;2;3` all round-trip). The breakers are **leading/trailing whitespace**
  (reader `.strip()`) and a **trailing char in `";,:\t ="`** (reader `.rstrip(";,:\t =")`). So:
  non-empty, no leading/trailing whitespace, last char not in `;,:\t =`. **Also: the FIRST char must
  not be in `;,:\t =`** — required by the separator-invariance property, where a one-line
  `key<tab>value` whose value starts with a separator (e.g. `:0`) lets the reader's
  `\s*[=;,:\t]\s*` regex greedily swallow `<tab>:` together and drop the leading char
  (prototype-confirmed false-fail). Allowing internal separators still makes the test diverse.
- `config_kv_dicts()` → built with **`st.dictionaries(config_keys(), config_values(), min_size=1)`**
  (NOT a draw-a-list-then-filter-for-uniqueness construction, which risks `HealthCheck.filter_too_much`
  on the small 40-value keyspace). `st.dictionaries` gives unique keys by construction.
- `csv_texts()` → tuples `(text, k, ncols)`: `k` (0..3) single-field preamble lines (realistic OSMOSE
  description rows, width 1), then a header + `1..4` data rows each with `ncols` (2..6) fields.
  **`_detect_preamble_lines` counts fields with `csv.reader` — i.e. the COMMA delimiter** (NOT the
  config separator, NOT whitespace; a tab-joined strategy false-fails, verified). Rows are
  comma-joined and every field/preamble token is drawn from a **comma-free, double-quote-free**
  alphabet (a comma or a quoted field changes the csv field count and breaks the width-1 preamble
  assumption). Single-field preamble lines guarantee the first equal-width-`>1` pair is the
  header/first-data-row, so the detector must return `k`.
- `diet_matrices()` → a wide `Time,Prey,<predator-stage cols>` DataFrame: 1..3 predator species each
  with 1..3 size-stages, a prey set including some predator species (self-loops) and a resource;
  a single Time value. Cells are **non-negative** (`min_value=0`), and **each predator-stage column
  is normalized so its (non-NaN) sum is ≤ 100** (draw raw weights, scale the column to a drawn target
  in `(0, 100]`) — otherwise `diet_network_at` legitimately returns a proportion > 100 and the bound
  property false-fails (verified: column-sum 250 → proportion 150). Optionally a fully-zero "dead"
  stage (excluded from normalization); optionally NaN cells (dropped, not normalized). Returned as a
  frame the test writes to `tmp/<x>_dietMatrix.csv`.
- `edges_and_values()` → `(edges, values)`: a sorted list of 1..8 **distinct** bin edges (≥0) and an
  equal-length list of values where each value is **either exactly `0.0` or in `[1e-3, 1e6]`** (drawn
  `st.one_of(st.just(0.0), st.floats(min_value=1e-3, max_value=1e6, allow_nan=False,
  allow_infinity=False))`). The bounds are LOAD-BEARING and the `1e-3` floor is too: default
  `st.floats()` yields `inf`/`NaN`/`1e308` (overflow the sums to inf/NaN → false-fail), AND a
  **denormal** like `5e-324` keeps `sum>0` true while underflowing the weighted numerator to `0.0`
  → mean-size drops below `min(midpoints)` (prototype-confirmed false-fail). Keeping `0.0` in the set
  still exercises the zero-total branch.
- `time_value_frames()` → small long `time,value` frames (a few distinct integer-ish times, finite
  non-negative values) for the `_window_by_time` property.

### 2. Property test files (one per target)

**`tests/test_config_roundtrip_properties.py`**
- *Round-trip survives every key/value, with no drops or spurious keys:* for `d = config_kv_dicts()`,
  `OsmoseConfigWriter().write(d, tmp)`, then `OsmoseConfigReader().read(tmp/"osm_all-parameters.csv")`
  → (a) for every `(k, v) in d.items()`, `result[k] == v`; **(b)** the *substantive* key set is
  preserved exactly: `set(result) - {"_osmose.config.dir"} - {k for k in result if
  k.startswith("osmose.configuration.")} == set(d)`. Part (b) is the teeth: it catches a routing
  change that **silently drops** a key or **invents** an extra substantive one — invisible to the
  per-key survival check and to every existing example test (`test_roundtrip_basic` uses 5 clean
  keys; `test_roundtrip_value_with_semicolon` deliberately declines to pin behavior).
- *Separator invariance:* a single `key<sep>value` line parses to the same `{key: value}` for each
  `sep in {=, ;, ,, :, tab}` (drives `read_file` on a one-line temp file). Asserts the auto-detect
  produces identical k/v regardless of which separator joined them.

**`tests/test_results_preamble_properties.py`**
- *Detects the planted header:* for `(text, k, ncols) = csv_texts()` written to a temp file,
  `_detect_preamble_lines(path) == k`. (Highest-teeth target — the detector has near-zero direct unit
  coverage today; this catches someone dropping the `count > 1` guard or flipping the
  first-match fallback.)
- *Never raises on degenerate input:* for arbitrary small text (empty, single line, all width-1 rows,
  blank lines), `_detect_preamble_lines` returns an `int` and does not raise. (A robustness smoke
  test — catches a crash, not a wrong answer.)
- *Cache invalidates on file change:* write file A (preamble `k_a`) → call (returns `k_a`) →
  **overwrite the same path** with a different file B (different preamble `k_b` and a different byte
  size) → call again → assert it returns `k_b`. This exercises the `(mtime_ns, size)` cache **key**
  (the only interesting cache bug — a stale result after the file changes); a plain "two calls on an
  unmodified file agree" check is tautological (the function is pure) and is dropped.

**`tests/test_trophic_network_properties.py`** (uses `diet_matrices()` written to a temp CSV)
- *Non-negative & clean node names:* every returned `proportion` is `≥ 0`; no node id (predator or
  prey) contains `" in ["`, at `predator_level="species"` — across random structures (self-loops,
  resources, multi-stage predators), broader than the single EEC fixture. (The `≤ 100` upper bound is
  NOT asserted: the strategy normalizes each predator column ≤100 specifically so it holds, which
  makes the check circular — dropped per the teeth review.)
- *Threshold monotonicity:* for `t_lo ≤ t_hi`, the edge set at `t_hi` is a **subset** of the edge set
  at `t_lo` (same `(predator,prey)` keys; `↑threshold ⇒ ⊆ edges`).
- *Prey-sum exactness:* at `predator_level="stage"`, each `(stage, prey-species)` proportion equals
  the exact sum of that prey's size-stage cells in the source (additive composition).
- *Dead stage never surfaces:* when `diet_matrices()` plants an all-zero predator size-stage, that
  stage label never appears as a predator at `predator_level="stage"`. (The subtler species-level
  "mean divides by live-stage count" stays an example test — it needs a known expected value that a
  generic fuzz can't assert cheaply.)

**`tests/test_size_spectrum_properties.py`** (pure helpers; lists/frames, no disk)
- *Mean size convexity (the teeth-y one):* `_mean_size(midpoints, values)` is within
  `[min(midpoints), max(midpoints)]` when total value > 0, and is `NaN` when total ≤ 0. This is a real
  value-weighted-average convexity invariant (NOT arithmetically forced) — it catches a
  midpoints/values zip misalignment, a sign error, or using edges where midpoints are intended.
- *LFI threshold boundary:* `_large_fish_indicator(edges, values, threshold)` compares the bin
  **lower edge** with `edge >= threshold`, so a bin whose edge is **exactly** `threshold` IS counted;
  a bin just below is not. Assert this boundary directly (off-by-one risk). On `total ≤ 0` it returns
  **`0.0`** (verified — not NaN). *(The trivial `LFI ∈ [0,1]` bound is NOT a separate property: with
  non-negative values it's `0 ≤ sum(subset) ≤ sum(all)` by construction, and the zero-total `== 0.0`
  case is already in `test_large_fish_indicator` — dropped per the teeth review.)*
- *Bin width is order-invariant:* `_infer_bin_width(shuffled_edges) == _infer_bin_width(sorted(set(...)))`
  — feed shuffled-with-duplicates edges and assert the result equals the sorted-unique computation
  (catches someone removing the internal `sorted(set(edges))` and assuming pre-sorted input). *(The
  "equals the median of consecutive diffs" clause is dropped — re-implementing the formula to assert
  `median == median` has no teeth.)*
- *Window keeps only in-range rows:* `_window_by_time(df, "time", w)` (with `w = st.integers(min_value=1, …)`
  to match the `int` signature; `w < 1` raises) returns only rows with `time > tmax - w` (strict `>`),
  and `n_rows(out) ≤ n_rows(in)`.

## Data flow

`strategy → (write temp file where the target reads from disk | pass lists/frames directly) → call
target → assert invariant`. Hypothesis drives many examples per property; `derandomize=True` makes
the example set reproducible; failures shrink to a minimal counterexample.

## Error handling / edge cases

- A property that legitimately allows `NaN` (`_mean_size` on zero total — but NOT LFI, which returns
  `0.0`) asserts the branch explicitly rather than excluding it — so the zero-total path is *covered*.
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
- **Out:** **scenarios `fork`-independence — EVALUATED in round-2 review and REJECTED:** `Scenario.config`
  is a flat `dict[str, str]` and `fork` does `config=dict(source.config)`; a shallow copy of a
  flat dict with immutable string values is already fully independent (no nested mutable state to
  alias), and the only regression it could catch — dropping the `dict()` wrapper — isn't observable
  through the `fork()` API (the on-disk source is untouched). The property would be tautological, so
  it is not added. (R3's recommendation assumed a *nested* mutable config, which doesn't exist here.)
- **Out (deferred, not rejected):** fisheries `annual_rate` windowing, `analysis.shannon_diversity`
  bounds, schema/history fuzzing; the two-header mortality reader; engine kernels (parity-coupled); a
  nightly/large-`max_examples` fuzzing harness; stateful/`RuleBasedStateMachine` testing; fuzzing
  `compute_size_spectrum` end-to-end (its invariants live in the helpers we fuzz).
- **Cut in round-2 review (tautological):** the `LFI ∈ [0,1]` bound (arithmetically forced + already
  example-covered) and the bin-width `== median-of-diffs` re-implementation check — both replaced by
  higher-teeth variants (the `edge == threshold` boundary and order-invariance respectively). The
  preamble `idempotent` check was replaced by a `mutate-then-recall` cache-invalidation property.

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
