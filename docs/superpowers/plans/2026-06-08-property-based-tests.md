# Property-Based Tests (Hypothesis) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hypothesis property-based tests over four pure-Python targets (config round-trip, output-preamble detection, trophic diet aggregation, size-spectrum helpers) whose invariants are crisp and whose failures are real bugs.

**Architecture:** A new `tests/strategies.py` provides shared, valid-but-diverse Hypothesis strategies; four new `tests/test_*_properties.py` files each `@given`-drive one target. `hypothesis` is added to the `dev` extra and configured via a deterministic `"ci"` profile in `tests/conftest.py` (guarded by `find_spec`, `database=None`, `derandomize=True`). Disk-reading targets write a per-example `tempfile.TemporaryDirectory()` — never the `tmp_path` fixture (illegal under `@given`).

**Tech Stack:** Python 3.12, pytest, Hypothesis ≥6.100, pandas, numpy, ruff (line-length 100).

**Reference spec:** `docs/superpowers/specs/2026-06-08-property-based-tests-design.md` (reviewed clean across 3 in-loop rounds: invariants verified, teeth mutation-tested, strategies de-vacuumed, operations de-risked, internal consistency checked).

---

## Conventions for every task

- Run everything with `.venv/bin/python` and `.venv/bin/ruff` (NOT bare `python`).
- Shell rules: use `git -C /home/razinka/osmose/osmose-python ...` (never `cd && git`); no `>`/`>>` redirection (use the Write/Edit tools for file content); no `$(...)`/backticks/`${}`; quote paths; commit messages END with the `Co-Authored-By` trailer shown in each task.
- **RUFF FORMAT-FIRST** (a CI gate runs both `ruff check` AND `ruff format --check` on `osmose/ ui/ tests/`): before every "verify" step run `.venv/bin/ruff format <files>` first, then `ruff check`, then `ruff format --check`.
- Hypothesis is installed by Task 1; later tasks assume it is importable in `.venv`.

## File Structure

- **Modify** `pyproject.toml` — add `hypothesis>=6.100` to the `dev` extra (Task 1).
- **Modify** `tests/conftest.py` — register+load the `"ci"` Hypothesis profile, `find_spec`-guarded (Task 1).
- **Modify** `.gitignore` — add `.hypothesis/` (Task 1).
- **Create** `tests/strategies.py` — all shared strategies (Task 2): `config_keys`, `config_values`, `config_kv_dicts`, `csv_texts`, `csv_text_pairs`, `diet_matrices`, `edges_and_values`, `shuffled_bin_edges`, `time_value_frames`.
- **Create** `tests/test_config_roundtrip_properties.py` (Task 3).
- **Create** `tests/test_results_preamble_properties.py` (Task 4).
- **Create** `tests/test_trophic_network_properties.py` (Task 5).
- **Create** `tests/test_size_spectrum_properties.py` (Task 6).
- **Modify** `CHANGELOG.md` + full verification (Task 7).

---

## Task 1: Dependency + Hypothesis `"ci"` profile + gitignore

**Files:**
- Modify: `pyproject.toml` (the `[project.optional-dependencies].dev` list)
- Modify: `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add the dependency to the `dev` extra**

In `pyproject.toml`, inside `[project.optional-dependencies]` → `dev = [ ... ]`, add this line (alphabetical-ish, near the other test deps):
```
    "hypothesis>=6.100",
```

- [ ] **Step 2: Install it into the venv**

Run: `.venv/bin/pip install "hypothesis>=6.100"`
Expected: installs `hypothesis` (+ `sortedcontainers`, `attrs`). Confirm with
`.venv/bin/python -c "import hypothesis; print(hypothesis.__version__)"` → prints a 6.x version.

- [ ] **Step 3: Register the deterministic profile in conftest**

In `tests/conftest.py`, `find_spec` is already imported (`from importlib.util import find_spec`) and there is an existing playwright guard block. Immediately AFTER that playwright `if find_spec("playwright") is None: ...` block, add:
```python
# Register a deterministic Hypothesis profile for the property-based tests
# (tests/test_*_properties.py). Guarded by find_spec so the whole suite still
# COLLECTS when hypothesis is absent — a bare top-level `import hypothesis`
# would fail collection of every test. database=None + derandomize=True keep CI
# and local runs byte-identical; deadline=None avoids flaky timing failures.
if find_spec("hypothesis") is not None:
    from hypothesis import settings

    settings.register_profile(
        "ci", max_examples=150, deadline=None, derandomize=True, database=None
    )
    settings.load_profile("ci")
```

- [ ] **Step 4: Gitignore the example database**

In `.gitignore`, add a line:
```
.hypothesis/
```

- [ ] **Step 5: Verify conftest still imports + suite still collects**

Run: `.venv/bin/python -c "import tests.conftest; print('conftest ok')"` → `conftest ok`.
Run: `.venv/bin/python -m pytest tests/test_schema.py -q` → passes (a sanity that collection + the new profile load don't break an unrelated module).
Run: `.venv/bin/ruff check tests/conftest.py && .venv/bin/ruff format --check tests/conftest.py` → clean. (If `ruff format --check` flags it, run `.venv/bin/ruff format tests/conftest.py` and re-run.)

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add pyproject.toml tests/conftest.py .gitignore
```
Commit message body:
```
test(hypothesis): add hypothesis dev dep + deterministic ci profile

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Shared strategies (`tests/strategies.py`)

**Files:**
- Create: `tests/strategies.py`

> This task has no standalone test; it is validated by a smoke import + ruff, and exercised by Tasks 3–6. (It is pure strategy code — the property tests that consume it are the real tests.)

- [ ] **Step 1: Write the strategies module**

Create `tests/strategies.py` with EXACTLY this content:
```python
"""Hypothesis strategies for OSMOSE property-based tests.

Each strategy is constrained to VALID-but-diverse inputs by construction, so a
property failure means a real invariant violation, not malformed input. The
comments explain WHY each constraint exists (they were all derived from in-loop
review counterexamples).
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
from hypothesis import strategies as st

# --- config -----------------------------------------------------------------

# Family prefixes known to round-trip cleanly (route to distinct sub-files or
# master; never the writer-regenerated `osmose.configuration.*` reference keys).
_FAMILY_PREFIXES = [
    "species.linf",
    "species.k",
    "species.lifespan",
    "predation.accessibility.stage",
    "grid.ncolumn",
    "simulation.time.ndtperyear",
    "movement.distribution.method",
]

# Printable, non-whitespace ASCII for config values.
_ANY = string.digits + string.ascii_letters + string.punctuation
# Same minus the separator chars `= ; , :` — used for the FIRST and LAST char of
# a value (a value that starts/ends with a separator is eaten by the reader's
# `\s*[=;,:\t]\s*` split / `.strip().rstrip(";,:\t =")` normalization). Internal
# separators are SAFE (the reader splits maxsplit=1 on the writer's framing ` ; `).
_NONSEP = "".join(c for c in _ANY if c not in "=;,:")

# CSV field alphabet for preamble texts: NO comma, NO double-quote (either would
# change the csv.reader field count and break the width-1 preamble assumption).
_CSV_FIELD = string.ascii_letters + string.digits + "_-."


@st.composite
def config_keys(draw) -> str:
    """OSMOSE-shaped lowercase dotted key that round-trips (no separators, never
    `osmose`-prefixed). family[.leaf][.spN]."""
    parts = [draw(st.sampled_from(_FAMILY_PREFIXES))]
    if draw(st.booleans()):
        parts.append(draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6)))
    if draw(st.booleans()):
        parts.append(f"sp{draw(st.integers(min_value=0, max_value=9))}")
    return ".".join(parts)


@st.composite
def config_values(draw) -> str:
    """Non-empty value that survives reader normalization: no leading/trailing
    whitespace (alphabet excludes it), and first+last char are non-separators.
    Internal separators are allowed (and prove they round-trip)."""
    first = draw(st.sampled_from(_NONSEP))
    middle = draw(st.text(alphabet=_ANY, max_size=18))
    if not middle:
        return first
    last = draw(st.sampled_from(_NONSEP))
    return first + middle + last


def config_kv_dicts() -> st.SearchStrategy:
    """Flat config dict (unique keys by construction via st.dictionaries)."""
    return st.dictionaries(config_keys(), config_values(), min_size=1, max_size=8)


# --- preamble CSV text ------------------------------------------------------


def _build_csv(draw, k: int, ncols: int, ndata: int) -> str:
    """k width-1 preamble lines, then a header + ndata rows of `ncols` fields,
    comma-joined. Single-field preamble lines guarantee the first equal-width->1
    pair is the header / first data row."""
    field = st.text(alphabet=_CSV_FIELD, min_size=1, max_size=8)
    lines = [draw(field) for _ in range(k)]
    for _ in range(ndata + 1):  # header + ndata data rows
        lines.append(",".join(draw(field) for _ in range(ncols)))
    return "\n".join(lines) + "\n"


@st.composite
def csv_texts(draw):
    """(text, k, ncols) — a CSV with `k` (0..3) preamble lines before the header."""
    k = draw(st.integers(min_value=0, max_value=3))
    ncols = draw(st.integers(min_value=2, max_value=6))
    ndata = draw(st.integers(min_value=1, max_value=4))
    return _build_csv(draw, k, ncols, ndata), k, ncols


@st.composite
def csv_text_pairs(draw):
    """(text_a, k_a, text_b, k_b) with k_a != k_b AND different byte size.

    The byte-size guarantee is LOAD-BEARING (plan-review BLOCKER): _detect_preamble_lines
    caches on (mtime_ns, size), and a same-size in-place rewrite within one mtime_ns tick
    (coarse tmpfs clocks) would NOT invalidate the cache — so the property would falsely fail
    on the ~1% of same-size pairs. Forcing the sizes to differ tests invalidation honestly.
    """
    k_a = draw(st.integers(min_value=0, max_value=3))
    k_b = draw(st.integers(min_value=0, max_value=3))
    if k_b == k_a:
        k_b = (k_a + 1) % 4
    text_a = _build_csv(draw, k_a, draw(st.integers(2, 6)), draw(st.integers(1, 4)))
    text_b = _build_csv(draw, k_b, draw(st.integers(2, 6)), draw(st.integers(1, 4)))
    if len(text_b.encode()) == len(text_a.encode()):
        # Trailing line -> changes byte size (cache key) without changing k_b
        # (detection scans top-down and already settled on the header at line k_b).
        text_b = text_b + "zz\n"
    return text_a, k_a, text_b, k_b


# --- diet matrices ----------------------------------------------------------

_DIET_SPECIES = ["cod", "herring", "sprat"]
_RESOURCE = "Diatoms"
_STAGE_EDGES = [0, 10, 30, 1000]
_PREY_EDGES = [0, 5, 1000]


@st.composite
def diet_matrices(draw) -> pd.DataFrame:
    """Wide Time,Prey,<predator-stage cols> diet matrix. Non-negative cells; each
    live predator-stage column normalized so its non-NaN sum <= 100. Interesting
    cases are biased in EXPLICITLY (round-3 vacuous-pass review): per-stage dead
    flag, per-cell NaN flag, and the first predator species ALWAYS appears as a
    2-size-stage prey (so prey-sum-exactness has a multi-stage case)."""
    n_pred = draw(st.integers(min_value=1, max_value=3))
    pred_species = _DIET_SPECIES[:n_pred]

    pred_cols: list[str] = []
    dead_col: dict[str, bool] = {}
    for sp in pred_species:
        n_stage = draw(st.integers(min_value=1, max_value=3))
        for i in range(n_stage):
            col = f"{sp} in [{_STAGE_EDGES[i]}, {_STAGE_EDGES[i + 1]}["
            pred_cols.append(col)
            dead_col[col] = draw(st.booleans())

    prey_labels: list[str] = []
    for idx, sp in enumerate(pred_species):
        # First species always spans 2 prey size-stages (guarantees the multi-
        # stage prey case the prey-sum-exactness property needs to bite).
        n_prey_stage = 2 if idx == 0 else draw(st.integers(min_value=1, max_value=2))
        for i in range(n_prey_stage):
            prey_labels.append(f"{sp} in [{_PREY_EDGES[i]}, {_PREY_EDGES[i + 1]}[")
    prey_labels.append(_RESOURCE)

    n_rows, n_cols = len(prey_labels), len(pred_cols)
    data = np.zeros((n_rows, n_cols), dtype=float)
    nan_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        for c in range(n_cols):
            data[r, c] = draw(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
            )
            nan_mask[r, c] = draw(st.integers(min_value=0, max_value=3)) == 0  # ~25% NaN

    for c, col in enumerate(pred_cols):
        if dead_col[col]:
            data[:, c] = 0.0
            continue
        live = ~nan_mask[:, c]
        raw = float(data[live, c].sum())
        if raw > 0:
            target = draw(
                st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
            )
            data[live, c] = data[live, c] * (target / raw)

    df = pd.DataFrame(data, columns=pred_cols)
    df.insert(0, "Prey", prey_labels)
    df.insert(0, "Time", 1.0)
    for c, col in enumerate(pred_cols):
        if dead_col[col]:
            continue
        for r in range(n_rows):
            if nan_mask[r, c]:
                df.iat[r, c + 2] = float("nan")  # +2 for the Time, Prey columns
    return df


# --- size-spectrum ----------------------------------------------------------


@st.composite
def edges_and_values(draw):
    """(edges, values): sorted distinct edges (>=0); each value is exactly 0.0 or
    in [1e-3, 1e6] (the 1e-3 floor avoids denormal underflow that false-fails the
    mean-size bound; 0.0 exercises the zero-total branch)."""
    n = draw(st.integers(min_value=1, max_value=8))
    edges = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    value_st = st.one_of(
        st.just(0.0),
        st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    values = [draw(value_st) for _ in range(n)]
    return edges, values


@st.composite
def shuffled_bin_edges(draw):
    """(shuffled, canonical): >=2 distinct base edges + injected duplicates,
    shuffled; canonical = sorted(set(...)). For the bin-width order-invariance
    property (edges_and_values is sorted+distinct and can't exercise dups)."""
    base = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2,
                max_size=8,
                unique=True,
            )
        )
    )
    with_dupes = base + draw(st.lists(st.sampled_from(base), max_size=4))
    shuffled = draw(st.permutations(with_dupes))
    return list(shuffled), sorted(set(with_dupes))


@st.composite
def time_value_frames(draw) -> pd.DataFrame:
    """Long time,value frame (distinct integer-ish times) for _window_by_time."""
    times = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=6, unique=True))
    )
    rows = [
        {
            "time": float(t),
            "value": draw(
                st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
            ),
        }
        for t in times
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Smoke-import + sample each strategy + ruff**

Run: `.venv/bin/ruff format tests/strategies.py`
Run (one example per strategy, proves they generate without error):
```
.venv/bin/python -c "
from hypothesis import strategies as st, find
import tests.strategies as s
for name in ['config_kv_dicts','csv_texts','csv_text_pairs','diet_matrices','edges_and_values','shuffled_bin_edges','time_value_frames']:
    strat = getattr(s, name)()
    print(name, type(strat).__name__)
print('config_keys', s.config_keys().example() is not None)
"
```
Expected: prints each strategy name + a type, no exception. (If `.example()` warns, that is fine — it is a one-off smoke check, not used in tests.)
Run: `.venv/bin/ruff check tests/strategies.py && .venv/bin/ruff format --check tests/strategies.py` → clean.

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add tests/strategies.py
```
Commit message body:
```
test(hypothesis): shared property-test strategies

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 3: Config round-trip properties

**Files:**
- Create: `tests/test_config_roundtrip_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/test_config_roundtrip_properties.py` with EXACTLY this content:
```python
"""Property-based tests: OSMOSE config writer->reader round-trip."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from tests.strategies import config_keys, config_kv_dicts, config_values


@given(d=config_kv_dicts())
def test_roundtrip_survives_and_keyset(d):
    with tempfile.TemporaryDirectory() as td:
        OsmoseConfigWriter().write(d, Path(td))
        result = OsmoseConfigReader().read(Path(td) / "osm_all-parameters.csv")
    # (a) every substantive key/value survives (exact STRING equality, no approx).
    for k, v in d.items():
        assert result[k] == v
    # (b) the substantive key set is preserved exactly — catches a routing change
    # that INVENTS a spurious substantive key (part (a) is blind to that).
    substantive = (
        set(result)
        - {"_osmose.config.dir"}
        - {k for k in result if k.startswith("osmose.configuration.")}
    )
    assert substantive == set(d)


@given(key=config_keys(), value=config_values(), sep=st.sampled_from(["=", ";", ",", ":", "\t"]))
def test_separator_invariance(key, value, sep):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub.csv"
        p.write_text(f"{key}{sep}{value}\n")
        result = OsmoseConfigReader().read_file(p)
    assert result[key] == value
```

- [ ] **Step 2: Run to verify it passes (green)**

Run: `.venv/bin/python -m pytest tests/test_config_roundtrip_properties.py -q`
Expected: PASS (2 property tests). (There is no red→green step here — the targets already exist; the property simply asserts their invariant. If it FAILS, a property has found a real bug: STOP and apply the "If a property finds a real bug" policy from the spec — fix small/in-scope with a regression test, else `@pytest.mark.xfail(strict=True, reason=...)` + report.)

- [ ] **Step 3: Verify (format-first) + commit**

Run: `.venv/bin/ruff format tests/test_config_roundtrip_properties.py`
Run: `.venv/bin/ruff check tests/test_config_roundtrip_properties.py && .venv/bin/ruff format --check tests/test_config_roundtrip_properties.py` → clean.
```bash
git -C /home/razinka/osmose/osmose-python add tests/test_config_roundtrip_properties.py
```
Commit message body:
```
test(hypothesis): config writer->reader round-trip properties

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 4: Output-preamble detection properties

**Files:**
- Create: `tests/test_results_preamble_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/test_results_preamble_properties.py` with EXACTLY this content:
```python
"""Property-based tests: _detect_preamble_lines header/preamble detection."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import event, given
from hypothesis import strategies as st

from osmose.results import _detect_preamble_lines
from tests.strategies import csv_text_pairs, csv_texts


@given(tk=csv_texts())
def test_detects_planted_header(tk):
    text, k, _ncols = tk
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        p.write_text(text)
        assert _detect_preamble_lines(p) == k


@given(text=st.text(max_size=40))
def test_never_raises_on_degenerate_input(text):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.csv"
        p.write_text(text)
        assert isinstance(_detect_preamble_lines(p), int)


@given(pair=csv_text_pairs())
def test_cache_invalidates_on_file_change(pair):
    text_a, k_a, text_b, k_b = pair
    event(f"k_a={k_a} k_b={k_b}")  # confirm differing-k cases are generated
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.csv"
        p.write_text(text_a)
        assert _detect_preamble_lines(p) == k_a
        # Overwrite the SAME path; the (mtime_ns, size) cache key must change so a
        # stale cached value is not returned. (k_a != k_b AND byte size differs by
        # construction — see csv_text_pairs; size alone flips the cache key.)
        p.write_text(text_b)
        assert _detect_preamble_lines(p) == k_b
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_results_preamble_properties.py -q`
Expected: PASS (3 property tests). If it fails, apply the spec's bug policy (STOP/report or xfail).

- [ ] **Step 3: Verify (format-first) + commit**

Run: `.venv/bin/ruff format tests/test_results_preamble_properties.py`
Run: `.venv/bin/ruff check tests/test_results_preamble_properties.py && .venv/bin/ruff format --check tests/test_results_preamble_properties.py` → clean.
```bash
git -C /home/razinka/osmose/osmose-python add tests/test_results_preamble_properties.py
```
Commit message body:
```
test(hypothesis): preamble-detection properties (detect-k, no-raise, cache-invalidation)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 5: Trophic diet-aggregation properties

**Files:**
- Create: `tests/test_trophic_network_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/test_trophic_network_properties.py` with EXACTLY this content:
```python
"""Property-based tests: diet_network_at per-timestep aggregation."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings

from osmose.trophic_network import _split_species, diet_network_at
from tests.strategies import diet_matrices

# The diet pipeline (melt + groupbys + read_csv) is ~20 ms/example; cap examples
# so the file stays snappy (plan-review measured ~6s at 75; 50 keeps it ~4s while
# the strategies still hit their interesting cases >70% of the time).
DIET = settings(max_examples=50)


def _write(df, td):
    (Path(td) / "x_dietMatrix.csv").write_text(df.to_csv(index=False))
    return Path(td)


@DIET
@given(df=diet_matrices())
def test_proportions_nonneg_and_clean_names(df):
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0)
    assert (net["proportion"] >= 0).all()
    names = set(net["predator"]) | set(net["prey"])
    assert not any(" in [" in s for s in names)


@DIET
@given(df=diet_matrices())
def test_threshold_monotonic(df):
    with tempfile.TemporaryDirectory() as td:
        d = _write(df, td)
        lo = diet_network_at(d, time=1.0, threshold=10.0)
        hi = diet_network_at(d, time=1.0, threshold=40.0)
    lo_edges = {(r.predator, r.prey) for r in lo.itertuples()}
    hi_edges = {(r.predator, r.prey) for r in hi.itertuples()}
    assert hi_edges <= lo_edges


@DIET
@given(df=diet_matrices())
def test_prey_sum_exactness_stage_level(df):
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0, predator_level="stage")
    # ~28% of matrices are all-dead/all-NaN -> empty net; don't let this (the
    # reorder-catching property) pass vacuously on those.
    assume(len(net) > 0)
    prey_species = df["Prey"].map(_split_species)
    for r in net.itertuples():
        # stage label is r.predator; prey species is r.prey
        expected = df.loc[prey_species == r.prey, r.predator].sum(skipna=True)
        assert r.proportion == pytest.approx(expected, rel=1e-9, abs=1e-12)


@DIET
@given(df=diet_matrices())
def test_dead_stage_never_surfaces(df):
    pred_cols = [c for c in df.columns if c not in ("Time", "Prey")]
    dead = [c for c in pred_cols if df[c].fillna(0.0).sum() == 0]
    with tempfile.TemporaryDirectory() as td:
        net = diet_network_at(_write(df, td), time=1.0, threshold=0.0, predator_level="stage")
    preds = set(net["predator"])
    for dcol in dead:
        assert dcol not in preds
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_trophic_network_properties.py -q`
Expected: PASS (4 property tests). If it fails, apply the spec's bug policy.

- [ ] **Step 3: Verify (format-first) + commit**

Run: `.venv/bin/ruff format tests/test_trophic_network_properties.py`
Run: `.venv/bin/ruff check tests/test_trophic_network_properties.py && .venv/bin/ruff format --check tests/test_trophic_network_properties.py` → clean.
```bash
git -C /home/razinka/osmose/osmose-python add tests/test_trophic_network_properties.py
```
Commit message body:
```
test(hypothesis): diet_network_at properties (bounds, monotonicity, prey-sum, dead-stage)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 6: Size-spectrum helper properties

**Files:**
- Create: `tests/test_size_spectrum_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/test_size_spectrum_properties.py` with EXACTLY this content:
```python
"""Property-based tests: size-spectrum pure helpers."""

import math

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from osmose.size_spectrum import (
    _infer_bin_width,
    _large_fish_indicator,
    _mean_size,
    _window_by_time,
)
from tests.strategies import edges_and_values, shuffled_bin_edges, time_value_frames


@given(ev=edges_and_values())
def test_mean_size_convexity(ev):
    edges, values = ev  # use edges as the midpoint positions
    m = _mean_size(edges, values)
    if sum(values) > 0:
        assert min(edges) - 1e-9 <= m <= max(edges) + 1e-9
    else:
        assert math.isnan(m)


@given(ev=edges_and_values(), data=st.data())
def test_lfi_threshold_boundary(ev, data):
    edges, values = ev
    if sum(values) <= 0:
        assert _large_fish_indicator(edges, values, edges[0]) == 0.0
        return
    # Draw the threshold from an edge whose bin has POSITIVE value: the `edge >=
    # threshold` comparator counts that bin at thr==edge, so dropping it (thr just
    # above) strictly lowers LFI. (Same total denominator in both calls.)
    positive_edges = [e for e, v in zip(edges, values) if v > 0]
    thr = data.draw(st.sampled_from(positive_edges))
    incl = _large_fish_indicator(edges, values, thr)
    excl = _large_fish_indicator(edges, values, thr + 1e-6)
    assert incl > excl


@given(se=shuffled_bin_edges())
def test_bin_width_order_invariant(se):
    shuffled, canonical = se
    assert _infer_bin_width(shuffled) == _infer_bin_width(canonical)


@given(tv=time_value_frames(), w=st.integers(min_value=1, max_value=10))
def test_window_keeps_in_range(tv, w):
    out = _window_by_time(tv, "time", w)
    tmax = tv["time"].max()
    assert (out["time"] > tmax - w).all()
    assert len(out) <= len(tv)
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_size_spectrum_properties.py -q`
Expected: PASS (4 property tests). If it fails, apply the spec's bug policy.

- [ ] **Step 3: Verify (format-first) + commit**

Run: `.venv/bin/ruff format tests/test_size_spectrum_properties.py`
Run: `.venv/bin/ruff check tests/test_size_spectrum_properties.py && .venv/bin/ruff format --check tests/test_size_spectrum_properties.py` → clean.
```bash
git -C /home/razinka/osmose/osmose-python add tests/test_size_spectrum_properties.py
```
Commit message body:
```
test(hypothesis): size-spectrum helper properties (convexity, LFI boundary, bin-width, window)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 7: CHANGELOG + full verification (determinism)

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG note**

In `CHANGELOG.md`, under `## [Unreleased]` → `### Added` (create the `### Added` subsection if absent), add:
```markdown
- **tests (property-based):** Hypothesis property tests for four pure-Python targets — config
  writer↔reader round-trip (every key/value survives + no spurious keys; separator-invariance),
  output-preamble detection (planted-header, no-raise, cache-invalidation), diet_network_at
  (non-negativity, threshold monotonicity, prey-sum exactness, dead-stage exclusion), and the
  size-spectrum helpers (mean-size convexity, LFI threshold boundary, bin-width order-invariance,
  time-window). New `tests/strategies.py` + a deterministic Hypothesis `ci` profile.
```

- [ ] **Step 2: Full verification of the property suite**

Run: `.venv/bin/python -m pytest tests/test_config_roundtrip_properties.py tests/test_results_preamble_properties.py tests/test_trophic_network_properties.py tests/test_size_spectrum_properties.py -q`
Expected: PASS (13 property tests total: 2 + 3 + 4 + 4). Report the count.

- [ ] **Step 3: Determinism check (run twice, identical outcome)**

Run the same command a SECOND time:
`.venv/bin/python -m pytest tests/test_*_properties.py -q`
Expected: identical PASS result (derandomize=True + database=None → byte-identical example sets across runs). If the two runs differ, STOP and report (the profile is not deterministic).

- [ ] **Step 4: Lint the whole touched surface**

Run: `.venv/bin/ruff check osmose/ ui/ tests/ && .venv/bin/ruff format --check osmose/ ui/ tests/`
Expected: clean. (If `format --check` flags a touched file, run `.venv/bin/ruff format <file>` and re-run the property suite.)

- [ ] **Step 5: Broader sanity (no collateral breakage)**

Run: `.venv/bin/python -m pytest tests/test_config_reader.py tests/test_config_writer.py tests/test_trophic_network.py tests/test_size_spectrum.py tests/test_results.py -q`
Expected: the existing example tests for the same targets still pass (the new tests are purely additive). Report counts.

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add CHANGELOG.md
```
Commit message body:
```
docs(changelog): property-based tests via Hypothesis

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- [ ] **Step 7: Final whole-feature review + finish**

Then use superpowers:requesting-code-review (final whole-feature review over `git diff master...HEAD`), then superpowers:finishing-a-development-branch.

---

## Self-Review (plan author)

**Spec coverage:**
- Dep + `ci` profile (`find_spec` guard, `database=None`, `derandomize`, `deadline=None`) + `.hypothesis/` gitignore → Task 1. ✓
- `tests/strategies.py` with all strategies incl. the round-2/3 fixes (config first+last char non-sep, `st.dictionaries`, comma-delimited width-1 preamble, diet column-≤100 normalization + explicit dead/NaN/multi-stage-prey biasing, edges floor 1e-3, `shuffled_bin_edges`, `time_value_frames`) → Task 2. ✓ (`csv_text_pairs` added so the cache property isn't `assume`-heavy.)
- Config round-trip (survival + key-set part (b) + separator-invariance, exact string compare) → Task 3. ✓
- Preamble (detect-k, never-raise, mutate-then-recall cache-invalidation with `event`) → Task 4. ✓
- Trophic (non-neg + clean names, threshold monotonicity, prey-sum exactness at stage level w/ `approx(rel=1e-9, abs=1e-12)` + `assume(len(net) > 0)`, dead-stage) at `max_examples=50` → Task 5. ✓
- Size-spectrum (mean-size convexity ±1e-9, LFI boundary with `threshold = st.sampled_from(positive edges)` + strict `incl > excl`, bin-width order-invariance, window strict `>`) → Task 6. ✓
- CHANGELOG + determinism (twice) + full lint + additive-sanity → Task 7. ✓
- Operational mandates: `tempfile.TemporaryDirectory()`-in-body (never `tmp_path`) used in every disk-writing property (Tasks 3/4/5); `pytest.importorskip("hypothesis")` then imports (ruff-exempt from E402, verified against `test_ui_state.py`). ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code + exact commands. The "If it fails, apply the spec's bug policy" notes are deliberate (property tests can find real bugs), not placeholders. ✓

**Type/name consistency:** strategy names defined in Task 2 (`config_keys`, `config_values`, `config_kv_dicts`, `csv_texts`, `csv_text_pairs`, `diet_matrices`, `edges_and_values`, `shuffled_bin_edges`, `time_value_frames`) are exactly the names imported in Tasks 3–6. `config_kv_dicts()` is a plain function returning `st.dictionaries(...)` (NOT `@st.composite`) — imported and called as `config_kv_dicts()` in Task 3 ✓. Helper signatures match the real code: `_mean_size(midpoints, values)`, `_large_fish_indicator(edges, values, threshold)`, `_infer_bin_width(edges)`, `_window_by_time(df, time_col, window_years)`, `diet_network_at(output_dir, *, time, threshold, predator_level)`, `OsmoseConfigWriter().write(d, Path)`, `OsmoseConfigReader().read(path)`/`.read_file(path)`. ✓
