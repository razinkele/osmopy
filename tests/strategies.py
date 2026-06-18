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

# Family prefixes that round-trip cleanly. They must be canonicalization-STABLE:
# the reader canonicalizes every loaded config to NEW OSMOSE 4.4.0 keys on read
# (canonicalize_config -> migrate_config), so any LEGACY/renamable key (e.g. the
# pre-4.3.3 `grid.ncolumn` -> `grid.nlon` rename) would NOT survive the exact-key
# round-trip. Production `state.config` only ever holds canonical keys, so the
# property must feed canonical keys too. They also route to distinct sub-files or
# master; never the writer-regenerated `osmose.configuration.*` reference keys.
_FAMILY_PREFIXES = [
    "species.linf",
    "species.k",
    "species.lifespan",
    "predation.accessibility.stage",
    "grid.nlon",  # canonical form of legacy `grid.ncolumn` (renamed on read)
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
