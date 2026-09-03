"""Equivalence gates: the Numba mortality kernels vs the pure-Python reference.

Task 1 of ``docs/superpowers/plans/2026-08-31-bioen-numba-kernel.md``. Tasks 2-4 build
their behavioural tests on ``run_cell_both_paths`` / ``run_batch_both_paths`` and
``assert_arms_equal`` from this module.

WHY THE ``_HAS_NUMBA`` TOGGLE IS THE POINT OF THIS FILE
-------------------------------------------------------
``_mortality_in_cell`` is not "the Python path". It computes

    use_full_numba = (_HAS_NUMBA and inst_abd is not None and rsc_size_min is not None
                      and eff_starv is not None and not config.bioen_enabled)

and, when true, *dispatches into* ``_mortality_in_cell_numba``. Plan Task 4 removes the
``not config.bioen_enabled`` term. A harness that simply called ``_mortality_in_cell``
twice would, after that flip, run the kernel on BOTH arms and compare the kernel with
itself -- green forever, pinning nothing. So the reference arm here sets
``M._HAS_NUMBA = False`` for the duration of its call and restores it afterwards, which
is the only thing that guarantees the reference arm executes ``_apply_*_for_school``.

``ArmResult.kernel_calls`` closes the loop from the other side: the reference arm must
show **exactly 0** kernel entries and the candidate arm **at least 1**. ``require_kernel``
therefore turns "the candidate silently fell back to Python" (which is what happens under
bioen today, before Task 4's flip) into a loud failure instead of a vacuous pass.

Because ``_HAS_NUMBA`` is toggled rather than the RNG replayed, both arms can simply be
handed a freshly seeded ``np.random.default_rng(seed)``: the dispatch branch draws five
permutations then ``n_local`` shuffles of a 4-element cause list, and so does the Python
fallback (Task 0 verified no ``_apply_*_for_school`` consumes ``rng``). No replay
machinery exists here and none should be added. ``np.random.Generator`` is a C extension
type -- monkeypatching ``permutation``/``shuffle`` on it raises, so that route is closed
anyway.

THE BATCH HARNESS RUNS THE OTHER WAY ROUND
------------------------------------------
Production does NOT call ``_mortality_in_cell_numba`` -- ``mortality()`` dispatches to
``_mortality_all_cells_parallel`` (or ``_mortality_all_cells_numba``). Those generate
their RNG *inside* the kernel from ``rng_seed`` via Numba's ``np.random``, so they cannot
be driven from a caller-supplied ``Generator``. ``run_batch_both_paths`` instead seeds the
legacy NumPy global RNG with the same seed, replicates the kernel's draws exactly
(``_replicate_batch_kernel_rng``), and feeds them to ``_mortality_in_cell_numba`` cell by
cell. Verified on this machine: Numba's ``np.random.permutation``/``shuffle`` reproduce
NumPy's legacy MT19937 stream bit-for-bit.

EXP LIBRARY HAZARD (measured 2026-09-03, see ``test_exp_library_divergence_...``)
---------------------------------------------------------------------------------
Numba lowers ``np.exp`` to libm; NumPy uses its own SIMD polynomial. They disagree in the
last ULP on roughly 5-9% of arguments. Every non-predation cause is
``dead = abd * (1 - exp(-D))``, and that subtraction *amplifies* a 1-ULP error in
``exp(-D)`` by ~1/D. At engine-scale rates (D ~ 1e-3..1e-2) that is a ~1e-13 relative
error in the death count; for D ~ 1e-6 it reaches ~1e-10. It is a numerical-library
difference, not a kernel-logic defect, but it means bit-exact equivalence is only
attainable on rates where the two ``exp`` implementations happen to agree.
``assert_exp_library_agreement`` checks exactly that for a fixture's own rates, so a
libm/NumPy change fails with a message naming the cause instead of an opaque numeric diff
that a future reader would blame on Task 2's kernel edits.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes import mortality as M
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import SimulationContext
from osmose.engine.state import MortalityCause, SchoolState

numba = pytest.importorskip("numba")

pytestmark = pytest.mark.skipif(
    not M._HAS_NUMBA, reason="the whole file compares against the Numba kernels"
)


@numba.njit(cache=True)
def _nb_exp(x):
    """Numba's ``np.exp`` -- libm, NOT NumPy's SIMD polynomial. See EXP LIBRARY HAZARD."""
    return np.exp(x)


# ---------------------------------------------------------------------------
# What gets compared, and which path writes it
# ---------------------------------------------------------------------------

#: Written by BOTH arms today (bioen off). These are what actually discriminate.
LIVE_FIELDS: frozenset[str] = frozenset(
    {
        "n_dead",  # _apply_predation_* + _kill / _apply_single_cause
        "preyed_biomass",  # predation Phase 3, both arms
        "pred_success_rate",  # predation Phase 3, both arms
        "inst_abd",  # every death site in both arms
        "tl_weighted_sum",  # predation prey loop, both arms (ctx accumulator)
        "diet_matrix",  # predation prey loop, both arms (ctx accumulator)
        "resource_biomass",  # resource depletion in the predation prey loop, both arms
    }
)

#: Written by the Python reference under bioen only; the kernel learns them in Task 2.
#: Identically zero / absent on both arms bioen-OFF, so they discriminate NOTHING yet --
#: do not read a green Task 1 run as evidence these are covered.
DORMANT_FIELDS: frozenset[str] = frozenset(
    {
        "e_net",  # _consume survivor rescale + bioen_starvation_substep repayment
        "gonad_weight",  # bioen_starvation_substep flush
        "raw_preyed",  # _apply_predation_for_school's un-rescaled TL denominator
    }
)

#: NEITHER path writes these inside the cell/batch loop (``trophic_level`` is read by
#: predation and written by ``mortality()`` post-loop; ``abundance`` is written by
#: ``mortality()`` post-loop from ``n_dead``). Compared as no-write guards: if a future
#: kernel edit starts writing one in a single arm, this catches it for free. The brief's
#: field tuple lists ``abundance`` and then says the first draft wrongly included it --
#: keeping it costs nothing and closes that contradiction in the strict direction.
GUARD_FIELDS: frozenset[str] = frozenset({"abundance", "trophic_level"})

COMPARED_STATE_FIELDS: tuple[str, ...] = (
    "n_dead",
    "preyed_biomass",
    "pred_success_rate",
    "e_net",
    "gonad_weight",
    "abundance",
    "trophic_level",
)

#: Not ``SchoolState`` fields -- they live on ``inst_abd`` / the ``SimulationContext`` /
#: ``ResourceState``. This is why the harness returns ``ArmResult`` rather than the
#: brief's ``tuple[SchoolState, SchoolState]``: a bare SchoolState cannot carry them.
COMPARED_SIDE_ARRAYS: tuple[str, ...] = (
    "inst_abd",
    "raw_preyed",
    "tl_weighted_sum",
    "diet_matrix",
    "resource_biomass",
)

COMPARED: tuple[str, ...] = COMPARED_STATE_FIELDS + COMPARED_SIDE_ARRAYS

assert set(COMPARED) == LIVE_FIELDS | DORMANT_FIELDS | GUARD_FIELDS, (
    "every COMPARED field must be classified live / dormant / guard"
)


@dataclass
class ArmResult:
    """One arm's post-run state plus every side array the mortality loop writes."""

    label: str
    state: SchoolState
    input_state: SchoolState
    inst_abd: NDArray[np.float64]
    raw_preyed: NDArray[np.float64] | None
    tl_weighted_sum: NDArray[np.float64] | None
    diet_matrix: NDArray[np.float64] | None
    resource_biomass: NDArray[np.float64] | None
    kernel_calls: int
    bioen: bool = False
    eff_rates: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def get(self, name: str) -> NDArray[np.float64] | None:
        """Fetch a COMPARED field from wherever it actually lives."""
        if name in COMPARED_SIDE_ARRAYS:
            return getattr(self, name)
        return getattr(self.state, name)


def assert_arms_equal(
    reference: ArmResult,
    candidate: ArmResult,
    *,
    fields: Sequence[str] = COMPARED,
    witness_fields: Sequence[str] = ("n_dead",),
    witness_causes: Sequence[int] = (),
    label: str = "",
) -> None:
    """Assert bit-exact equality of every COMPARED field, and that the run did work.

    Args:
        reference: the arm that ran ``_apply_*_for_school`` (or the per-cell kernel).
        candidate: the arm that ran the kernel under test.
        fields: which COMPARED entries to check.
        witness_fields: fields that must be non-trivially non-zero on BOTH arms. Without
            these, a fixture in which nothing happens passes every equality check. Tasks
            2-4 must extend this per behaviour, not rely on the default.
        witness_causes: ``MortalityCause`` codes whose ``n_dead`` column must be > 0 on
            both arms (e.g. FORAGING for plan behaviour 4).
        label: prefix for assertion messages.

    Raises:
        AssertionError: under bioen, if ``witness_fields`` names none of
            ``DORMANT_FIELDS``. ``run_batch_both_paths`` bypasses ``use_full_numba``
            entirely -- both its arms call kernels directly -- so it has no
            ``require_kernel`` analogue and, under bioen today, happily compares two
            kernels that BOTH ignore bioen and agree perfectly. The witness set is the
            only thing standing between Task 2 and a green-but-vacuous bioen batch test.
    """
    prefix = f"[{label}] " if label else ""
    if reference.bioen and not (set(witness_fields) & DORMANT_FIELDS):
        raise AssertionError(
            f"{prefix}a bioen comparison must witness at least one of {sorted(DORMANT_FIELDS)} "
            "-- otherwise two arms that both ignore bioen agree perfectly and the test is "
            "vacuous. Plan Task 2's full set is witness_fields=('n_dead', 'e_net', "
            "'gonad_weight', 'raw_preyed') with witness_causes including STARVATION and "
            "FORAGING (the latter needs species.bioen.forage.k_for > 0, which "
            "BIOEN_OVERLAY deliberately does not set)."
        )
    for name in fields:
        a = reference.get(name)
        b = candidate.get(name)
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            raise AssertionError(
                f"{prefix}{name}: present on one arm only "
                f"({reference.label}={a is not None}, {candidate.label}={b is not None})"
            )
        np.testing.assert_array_equal(
            a,
            b,
            err_msg=f"{prefix}{name} differs between {reference.label} and {candidate.label}",
            strict=True,
        )

    for name in witness_fields:
        for arm in (reference, candidate):
            arr = arm.get(name)
            assert arr is not None, f"{prefix}witness field {name} is None on {arm.label}"
            assert np.any(arr != 0.0), (
                f"{prefix}witness field {name} is identically zero on {arm.label} -- "
                "the fixture exercised nothing and the equality above is vacuous"
            )

    for cause in witness_causes:
        for arm in (reference, candidate):
            total = float(arm.state.n_dead[:, int(cause)].sum())
            assert total > 0.0, (
                f"{prefix}no deaths recorded for cause {int(cause)} on {arm.label} -- "
                "the behaviour under test never fired"
            )


def assert_exp_library_agreement(*rate_arrays: NDArray[np.float64], label: str = "") -> None:
    """Guard: Numba's ``exp`` (libm) and NumPy's ``exp`` agree on every rate in play.

    See EXP LIBRARY HAZARD in the module docstring. When this fires, the fixture's rates
    have landed on one of the ~5-9% of arguments where the two implementations differ in
    the last ULP; ``1 - exp(-D)`` then amplifies that by ~1/D. It is NOT a kernel-logic
    defect -- pick a neighbouring rate value instead of loosening the comparison.
    """
    prefix = f"[{label}] " if label else ""
    for arr in rate_arrays:
        for d in np.unique(np.asarray(arr, dtype=np.float64)):
            if d <= 0.0:
                continue
            np_val = float(np.exp(-d))
            nb_val = float(_nb_exp(-d))
            assert np_val == nb_val, (
                f"{prefix}numpy.exp(-{d!r}) = {np_val!r} but numba's libm exp gives "
                f"{nb_val!r}. Bit-exact arm equality is unattainable at this rate; the "
                f"relative error in (1 - exp(-D)) is ~{abs(np_val - nb_val) / (1 - np_val):.3e}. "
                "This is a numerical-library difference, not a kernel defect."
            )


# ---------------------------------------------------------------------------
# Shared per-arm setup (mirrors mortality()'s prologue for the parts the loop reads)
# ---------------------------------------------------------------------------


class _KernelCounter:
    """Wraps a Numba dispatcher so the harness can prove which arm entered it."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._fn(*args, **kwargs)


@dataclass
class _ArmSetup:
    state: SchoolState
    resources: ResourceState | None
    inst_abd: NDArray[np.float64]
    eff_starv: NDArray[np.float64]
    eff_additional: NDArray[np.float64]
    eff_fishing: NDArray[np.float64]
    fishing_discard: NDArray[np.float64]
    rsc_arrays: tuple
    ctx: SimulationContext
    cap_fish: NDArray[np.float64] | None
    raw_preyed: NDArray[np.float64] | None


def _prepare_arm(
    state: SchoolState,
    config: EngineConfig,
    resources: ResourceState | None,
    *,
    n_subdt: int,
    step: int,
    diet_tracking: bool,
    fleet_state=None,
) -> _ArmSetup:
    """Deep-copy the inputs and rebuild what ``mortality()`` hands the cell loop."""
    st = copy.deepcopy(state)
    st = st.replace(feeding_stage=compute_feeding_stages(st, config))
    rsc = copy.deepcopy(resources)

    inst_abd = st.abundance.copy()
    eff_s, eff_a, eff_f, f_disc = M._precompute_effective_rates(
        st, config, n_subdt, step, fleet_state=fleet_state
    )
    rsc_arrays = M._precompute_resource_arrays(config, rsc)

    n = len(st)
    ctx = SimulationContext()
    if diet_tracking:
        ctx.diet_tracking_enabled = True
        ctx.diet_matrix = np.zeros((n, config.n_species + max(1, rsc_arrays[4])))
    ctx.tl_weighted_sum = np.zeros(n, dtype=np.float64)
    ctx.fleet_state = fleet_state

    cap_fish = None
    raw_preyed = None
    if config.bioen_enabled:
        from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap

        cap_fish = per_fish_ingestion_cap(
            st.weight,
            st.species_id,
            st.age_dt,
            config.bioen_i_max_all,
            config.bioen_beta,
            config.bioen_larvae_thres_dt,
            config.bioen_theta,
            config.bioen_c_rate,
            config.n_species,
            config.n_dt_per_year,
            n_subdt,
        )
        raw_preyed = np.zeros(n, dtype=np.float64)

    return _ArmSetup(
        state=st,
        resources=rsc,
        inst_abd=inst_abd,
        eff_starv=eff_s,
        eff_additional=eff_a,
        eff_fishing=eff_f,
        fishing_discard=f_disc,
        rsc_arrays=rsc_arrays,
        ctx=ctx,
        cap_fish=cap_fish,
        raw_preyed=raw_preyed,
    )


def _arm_result(
    label: str, setup: _ArmSetup, input_state: SchoolState, calls: int, *, bioen: bool
) -> ArmResult:
    return ArmResult(
        label=label,
        state=setup.state,
        input_state=input_state,
        inst_abd=setup.inst_abd,
        raw_preyed=setup.raw_preyed,
        tl_weighted_sum=setup.ctx.tl_weighted_sum,
        diet_matrix=setup.ctx.diet_matrix,
        resource_biomass=(setup.resources.biomass if setup.resources is not None else None),
        kernel_calls=calls,
        bioen=bioen,
        eff_rates={
            "starvation": setup.eff_starv,
            "additional": setup.eff_additional,
            "fishing": setup.eff_fishing,
        },
    )


def _cell_indices_for(state: SchoolState, cell_y: int, cell_x: int) -> NDArray[np.int32]:
    mask = (state.cell_y == cell_y) & (state.cell_x == cell_x)
    return np.where(mask)[0].astype(np.int32)


# ---------------------------------------------------------------------------
# Harness 1: per-cell reference vs per-cell kernel
# ---------------------------------------------------------------------------


def run_cell_both_paths(
    state: SchoolState,
    config: EngineConfig,
    *,
    seed: int,
    n_subdt: int = 10,
    resources: ResourceState | None = None,
    grid: Grid | None = None,
    step: int = 0,
    cell_y: int = 0,
    cell_x: int = 0,
    sub_steps: int = 1,
    diet_tracking: bool = True,
    access_matrix: NDArray[np.float64] | None = None,
    has_access: bool = False,
    use_stage_access: bool = False,
    require_kernel: bool = True,
    fleet_state=None,
) -> tuple[ArmResult, ArmResult]:
    """Run one cell through the Python reference and through the per-cell Numba kernel.

    Returns ``(python_arm, numba_arm)``. Both arms start from deep copies of ``state`` and
    ``resources`` and are driven by a freshly seeded ``np.random.default_rng(seed)``, so
    their ``seq_*`` permutations and cause orders coincide exactly.

    The Python arm forces ``M._HAS_NUMBA = False`` around its call -- see the module
    docstring. With ``require_kernel`` (the default) the Python arm must show 0 kernel
    entries and the Numba arm at least 1; that is what stops the comparison degenerating
    into kernel-vs-kernel (after plan Task 4) or Python-vs-Python (under bioen today).
    """
    grid = grid or Grid.from_dimensions(
        ny=int(state.cell_y.max()) + 1, nx=int(state.cell_x.max()) + 1
    )
    cell_indices = _cell_indices_for(state, cell_y, cell_x)
    assert len(cell_indices) > 0, "no schools in the requested cell -- the harness would no-op"

    def _run(label: str, force_python: bool) -> ArmResult:
        setup = _prepare_arm(
            state,
            config,
            resources,
            n_subdt=n_subdt,
            step=step,
            diet_tracking=diet_tracking,
            fleet_state=fleet_state,
        )
        rng = np.random.default_rng(seed)
        counter = _KernelCounter(M._mortality_in_cell_numba)
        prev_has_numba = M._HAS_NUMBA
        prev_kernel = M._mortality_in_cell_numba
        M._mortality_in_cell_numba = counter
        if force_python:
            M._HAS_NUMBA = False
        try:
            for _ in range(sub_steps):
                M._mortality_in_cell(
                    cell_indices,
                    setup.state,
                    config,
                    setup.resources,
                    cell_y,
                    cell_x,
                    rng,
                    n_subdt,
                    access_matrix if access_matrix is not None else M._DUMMY_ACCESS,
                    has_access,
                    use_stage_access,
                    np.zeros(len(setup.state), dtype=np.int32),
                    np.zeros(len(setup.state), dtype=np.int32),
                    inst_abd=setup.inst_abd,
                    step=step,
                    rsc_size_min=setup.rsc_arrays[0],
                    rsc_size_max=setup.rsc_arrays[1],
                    rsc_tl=setup.rsc_arrays[2],
                    rsc_access_rows=setup.rsc_arrays[3],
                    n_rsc=setup.rsc_arrays[4],
                    grid_nx=grid.nx,
                    eff_starv=setup.eff_starv,
                    eff_additional=setup.eff_additional,
                    eff_fishing=setup.eff_fishing,
                    fishing_discard=setup.fishing_discard,
                    ctx=setup.ctx,
                    egg_retained=setup.state.egg_retained,
                    cap_fish=setup.cap_fish,
                    raw_preyed=setup.raw_preyed,
                )
        finally:
            M._HAS_NUMBA = prev_has_numba
            M._mortality_in_cell_numba = prev_kernel
        return _arm_result(label, setup, state, counter.calls, bioen=config.bioen_enabled)

    python_arm = _run("python", force_python=True)
    numba_arm = _run("numba-cell-kernel", force_python=False)

    assert python_arm.kernel_calls == 0, (
        "the reference arm entered _mortality_in_cell_numba "
        f"{python_arm.kernel_calls}x despite _HAS_NUMBA=False -- the toggle no longer "
        "controls dispatch and this comparison is kernel-vs-kernel"
    )
    if require_kernel:
        assert numba_arm.kernel_calls >= 1, (
            "the candidate arm never reached _mortality_in_cell_numba -- it fell back to "
            "the Python path, so this comparison is Python-vs-Python and proves nothing. "
            "(Expected under bioen until plan Task 4 flips the `not config.bioen_enabled` "
            "term in `use_full_numba`.)"
        )
    return python_arm, numba_arm


# ---------------------------------------------------------------------------
# Harness 2: per-cell kernel vs the batch kernels production actually runs
# ---------------------------------------------------------------------------

#: The batch kernels' inline draw pattern, per non-empty cell, in order:
#: ``seq_pred, seq_starv, seq_fish, seq_nat`` then ``n_local`` shuffles of ``[0,1,2,3]``.
#: Plan Task 2 adds a fifth permutation (``seq_for``) and a fifth cause code under bioen;
#: when it does, ``n_perms``/``cause_codes`` here must be updated in the same commit or
#: this harness silently desynchronises from the kernel it is meant to pin.
_BATCH_N_PERMS_BIOEN_OFF = 4
_BATCH_PARALLEL_SEED_STRIDE = 7919


def _replicate_batch_kernel_rng(
    seed: int,
    boundaries: NDArray[np.int64],
    n_cells: int,
    *,
    parallel: bool,
    cause_codes: Sequence[int],
    n_perms: int,
) -> dict[int, tuple[list[NDArray[np.int32]], NDArray[np.int32]]]:
    """Reproduce the batch kernels' internal RNG using the legacy NumPy global stream.

    Numba's ``np.random`` is a MT19937 seeded exactly like NumPy's legacy ``RandomState``;
    ``permutation(int)`` and ``shuffle(int32[:])`` reproduce bit-for-bit (verified by
    ``test_numba_and_numpy_legacy_rng_streams_match``). The sequential kernel seeds once
    before the cell loop and lets cells consume the stream in order; the parallel kernel
    re-seeds per cell with ``rng_seed + cell * 7919``. Empty cells ``continue`` before any
    draw, so they consume nothing -- replicated here by the same early skip.
    """
    per_cell: dict[int, tuple[list[NDArray[np.int32]], NDArray[np.int32]]] = {}
    if not parallel:
        np.random.seed(seed)
    for cell in range(n_cells):
        start = int(boundaries[cell])
        end = int(boundaries[cell + 1])
        if end <= start:
            continue
        if parallel:
            np.random.seed(seed + cell * _BATCH_PARALLEL_SEED_STRIDE)
        n_local = end - start
        seqs = [np.random.permutation(n_local).astype(np.int32) for _ in range(n_perms)]
        causes = np.array(list(cause_codes), dtype=np.int32)
        cause_orders = np.empty((n_local, len(cause_codes)), dtype=np.int32)
        for ii in range(n_local):
            np.random.shuffle(causes)
            cause_orders[ii, :] = causes
        per_cell[cell] = (seqs, cause_orders)
    return per_cell


def _cell_groups(state: SchoolState, grid: Grid):
    """Reproduce ``mortality()``'s argsort/searchsorted cell grouping verbatim."""
    cell_ids = state.cell_y * grid.nx + state.cell_x
    valid = (state.cell_x >= 0) & (state.cell_y >= 0)
    valid_indices = np.where(valid)[0].astype(np.int32)
    n_cells = grid.ny * grid.nx
    if len(valid_indices) == 0:
        return np.zeros(0, dtype=np.int32), np.array([0, 0]), 0
    valid_cell_ids = cell_ids[valid_indices]
    order = np.argsort(valid_cell_ids, kind="mergesort")
    sorted_indices = valid_indices[order]
    boundaries = np.searchsorted(valid_cell_ids[order], np.arange(n_cells + 1))
    return sorted_indices, boundaries, n_cells


def run_batch_both_paths(
    state: SchoolState,
    config: EngineConfig,
    *,
    seed: int,
    parallel: bool,
    grid: Grid,
    n_subdt: int = 10,
    resources: ResourceState | None = None,
    step: int = 0,
    diet_tracking: bool = True,
    access_matrix: NDArray[np.float64] | None = None,
    has_access: bool = False,
    use_stage_access: bool = False,
    min_non_empty_cells: int = 2,
    n_threads: int = 2,
) -> tuple[ArmResult, ArmResult]:
    """Compare the per-cell kernel against the batch kernel production actually runs.

    Returns ``(per_cell_arm, batch_arm)``. The per-cell arm is driven with the batch
    kernel's own RNG draws, replicated in Python (see ``_replicate_batch_kernel_rng``) --
    the batch kernels generate RNG internally, so the comparison cannot be driven from a
    shared ``Generator`` the way ``run_cell_both_paths`` is.

    ``min_non_empty_cells`` defaults to 2 so ``prange`` iterates more than once, and the
    parallel arm asserts at least ``n_threads`` live Numba threads. Both are checked, not
    assumed: a single-cell fixture or a single-threaded runtime would make this the same
    check as ``run_cell_both_paths`` with extra steps.
    """
    sorted_indices, boundaries, n_cells = _cell_groups(state, grid)
    non_empty = [c for c in range(n_cells) if boundaries[c + 1] > boundaries[c]]
    assert len(non_empty) >= min_non_empty_cells, (
        f"fixture has {len(non_empty)} non-empty cells, need >= {min_non_empty_cells} so "
        "the batch kernels' cell loop (prange) iterates more than once"
    )
    # NumPy's legacy seed() takes [0, 2**32); the parallel kernel adds cell*7919 to it.
    assert 0 <= seed and seed + n_cells * _BATCH_PARALLEL_SEED_STRIDE < 2**32, (
        f"seed {seed} + {n_cells} cells * {_BATCH_PARALLEL_SEED_STRIDE} must stay inside "
        "the legacy RandomState seed range or the two sides wrap differently"
    )

    cause_codes = M._get_mortality_causes(config)
    n_perms = 5 if config.bioen_enabled else _BATCH_N_PERMS_BIOEN_OFF
    draws = _replicate_batch_kernel_rng(
        seed,
        boundaries,
        n_cells,
        parallel=parallel,
        cause_codes=cause_codes,
        n_perms=n_perms,
    )

    acc = access_matrix if access_matrix is not None else M._DUMMY_ACCESS

    # --- Arm A: the per-cell kernel, fed the batch kernel's own draws ---
    ref = _prepare_arm(
        state, config, resources, n_subdt=n_subdt, step=step, diet_tracking=diet_tracking
    )
    n = len(ref.state)
    rsc_bio_ref = ref.resources.biomass if ref.resources is not None else M._DUMMY_RSC_2D
    calls = 0
    for cell, (seqs, cause_orders) in draws.items():
        start = int(boundaries[cell])
        end = int(boundaries[cell + 1])
        M._mortality_in_cell_numba(
            sorted_indices[start:end],
            seqs[0],
            seqs[1],
            seqs[2],
            seqs[3],
            cause_orders,
            ref.inst_abd,
            ref.state.n_dead,
            ref.eff_starv,
            ref.eff_additional,
            ref.eff_fishing,
            ref.fishing_discard,
            ref.state.species_id,
            ref.state.length,
            ref.state.weight,
            ref.state.age_dt,
            ref.state.first_feeding_age_dt,
            ref.state.feeding_stage,
            ref.state.pred_success_rate,
            ref.state.preyed_biomass,
            ref.state.trophic_level,
            config.size_ratio_min,
            config.size_ratio_max,
            config.ingestion_rate,
            config.fr_shape,
            config.fr_halfsat,
            config.n_dt_per_year,
            n_subdt,
            acc,
            has_access,
            use_stage_access,
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            rsc_bio_ref,
            ref.rsc_arrays[0],
            ref.rsc_arrays[1],
            ref.rsc_arrays[2],
            ref.rsc_arrays[3],
            ref.rsc_arrays[4],
            config.n_species,
            cell,
            ref.ctx.tl_weighted_sum,
            True,
            ref.ctx.diet_matrix if ref.ctx.diet_matrix is not None else M._DUMMY_DIET,
            ref.ctx.diet_matrix is not None,
            ref.state.egg_retained,
        )
        calls += 1
    per_cell_arm = _arm_result("per-cell-kernel", ref, state, calls, bioen=config.bioen_enabled)

    # --- Arm B: the batch kernel mortality() actually dispatches to ---
    cand = _prepare_arm(
        state, config, resources, n_subdt=n_subdt, step=step, diet_tracking=diet_tracking
    )
    rsc_bio_cand = cand.resources.biomass if cand.resources is not None else M._DUMMY_RSC_2D
    prev_threads: int | None = None
    if parallel:
        # NUMBA_NUM_THREADS is read at IMPORT time, so setting the env var here would
        # silently leave prange serial. set_num_threads works post-import -- but it is
        # process-global, so it must be put back or every later test in the session
        # (test_engine_parity.py's engine runs included) inherits the reduced count.
        prev_threads = numba.get_num_threads()
        numba.set_num_threads(n_threads)
        assert numba.get_num_threads() >= n_threads, (
            f"only {numba.get_num_threads()} Numba threads available; the parallel kernel "
            "would run its prange serially and the cross-thread check would be vacuous"
        )
        batch_fn = M._mortality_all_cells_parallel
        label = "batch-parallel"
    else:
        batch_fn = M._mortality_all_cells_numba
        label = "batch-sequential"

    try:
        batch_fn(
            seed,
            sorted_indices,
            boundaries,
            n_cells,
            cand.inst_abd,
            cand.state.n_dead,
            cand.eff_starv,
            cand.eff_additional,
            cand.eff_fishing,
            cand.fishing_discard,
            cand.state.species_id,
            cand.state.length,
            cand.state.weight,
            cand.state.age_dt,
            cand.state.first_feeding_age_dt,
            cand.state.feeding_stage,
            cand.state.pred_success_rate,
            cand.state.preyed_biomass,
            cand.state.trophic_level,
            config.size_ratio_min,
            config.size_ratio_max,
            config.ingestion_rate,
            config.fr_shape,
            config.fr_halfsat,
            config.n_dt_per_year,
            n_subdt,
            acc,
            has_access,
            use_stage_access,
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.int32),
            rsc_bio_cand,
            cand.rsc_arrays[0],
            cand.rsc_arrays[1],
            cand.rsc_arrays[2],
            cand.rsc_arrays[3],
            cand.rsc_arrays[4],
            config.n_species,
            cand.ctx.tl_weighted_sum,
            True,
            cand.ctx.diet_matrix if cand.ctx.diet_matrix is not None else M._DUMMY_DIET,
            cand.ctx.diet_matrix is not None,
            cand.state.egg_retained,
        )
    finally:
        if prev_threads is not None:
            numba.set_num_threads(prev_threads)
    batch_arm = _arm_result(label, cand, state, 1, bioen=config.bioen_enabled)
    return per_cell_arm, batch_arm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SPECIES = (("Pred", 90.0), ("Prey", 20.0), ("Forage", 30.0))


def base_config(**overrides: str) -> EngineConfig:
    """Three focal species, every non-predation cause switched ON.

    Selectivity is left at the length-cutoff default with ``l50 = 0`` so the per-school
    path and ``_precompute_effective_rates`` compose the fishing rate in the SAME
    multiplication order (``selectivity`` is exactly 1.0, and multiplying by 1.0 is exact
    in IEEE-754). ``test_fishing_rate_composition_order_...`` deliberately breaks that
    symmetry to measure the residual.
    """
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": str(len(_SPECIES)),
        "simulation.nresource": "0",
        "mortality.subdt": "2",
    }
    for i, (name, linf) in enumerate(_SPECIES):
        cfg.update(
            {
                f"species.name.sp{i}": name,
                f"species.linf.sp{i}": str(linf),
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.egg.weight.sp{i}": "0.0005",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "10",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"species.first.feeding.age.sp{i}": "0.0417",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                f"predation.predprey.sizeratio.min.sp{i}": "2.0",
                f"predation.predprey.sizeratio.max.sp{i}": "20.0",
                f"mortality.natural.rate.sp{i}": ["0.3", "0.45", "0.2"][i],
                f"mortality.fishing.rate.sp{i}": ["0.4", "0.25", "0.0"][i],
                f"mortality.fishing.recruitment.age.sp{i}": "0",
                f"simulation.nschool.sp{i}": "1",
                f"species.zlayer.sp{i}": "0",
            }
        )
    cfg.update(overrides)
    config = EngineConfig.from_dict(cfg)
    # Discards need a CSV on the real loader; set the parsed array directly so the
    # n_dead[:,FISHING] / n_dead[:,DISCARDS] split is exercised (plan Task 2 Step 2).
    config.fishing_discard_rate = np.array([0.15, 0.4, 0.0])
    return config


def base_state(*, n_cells_used: int = 2) -> SchoolState:
    """Schools spread over ``n_cells_used`` cells, with a predator/prey pair in each.

    School 5 is a background school -- ``_apply_*_for_school`` skips it via
    ``is_background`` while the kernel relies on ``_zero_exempt`` having zeroed its rates,
    so the two arms only agree if that indirection holds.
    """
    n = 8
    st = SchoolState.create(n)
    cell_x = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int32)
    if n_cells_used == 1:
        cell_x = np.zeros(n, dtype=np.int32)
    is_bkg = np.zeros(n, dtype=np.bool_)
    is_bkg[5] = True
    return st.replace(
        species_id=np.array([0, 1, 1, 0, 1, 2, 2, 1], dtype=np.int32),
        abundance=np.array([1.0e5, 1.2e7, 5.0e6, 8.0e4, 9.0e6, 4.0e6, 3.0e6, 2.0e6]),
        weight=np.array([1.0e-3, 1.0e-6, 2.0e-6, 9.0e-4, 1.5e-6, 3.0e-6, 4.0e-6, 8.0e-7]),
        length=np.array([44.0, 4.0, 5.0, 39.0, 4.5, 6.0, 7.0, 3.5]),
        length_start=np.array([44.0, 4.0, 5.0, 39.0, 4.5, 6.0, 7.0, 3.5]),
        age_dt=np.array([60, 30, 30, 48, 30, 24, 24, 1], dtype=np.int32),
        first_feeding_age_dt=np.ones(n, dtype=np.int32),
        trophic_level=np.array([3.4, 2.0, 2.2, 3.1, 2.1, 2.6, 2.7, 2.05]),
        starvation_rate=np.array([0.2, 0.1, 0.15, 0.22, 0.05, 0.12, 0.3, 0.08]),
        is_background=is_bkg,
        cell_y=np.zeros(n, dtype=np.int32),
        cell_x=cell_x,
    )


def resource_config_and_state(grid: Grid) -> tuple[dict[str, str], ResourceState]:
    """One LTL resource with hand-set per-cell biomass (no NetCDF forcing)."""
    cfg = {
        "simulation.nresource": "1",
        "ltl.name.rsc0": "Plankton",
        "ltl.size.min.rsc0": "0.1",
        "ltl.size.max.rsc0": "20.0",
        "ltl.tl.rsc0": "1.8",
        "ltl.accessibility2fish.rsc0": "0.8",
    }
    rsc = ResourceState(cfg, grid)
    rsc.biomass[:] = 0.0
    rsc.biomass[0, :] = 12.0
    return cfg, rsc


# ---------------------------------------------------------------------------
# 0. The numerical ground the whole file stands on
# ---------------------------------------------------------------------------


def test_numba_and_numpy_legacy_rng_streams_match():
    """``run_batch_both_paths`` replicates the kernel's draws with NumPy's legacy RNG."""

    @numba.njit(cache=True)
    def _kernel_draws(seed, n_local, reps):
        np.random.seed(seed)
        perms = np.empty((reps, n_local), dtype=np.int32)
        causes = np.array([0, 1, 2, 3], dtype=np.int32)
        orders = np.empty((reps, 4), dtype=np.int32)
        for r in range(reps):
            perms[r] = np.random.permutation(n_local).astype(np.int32)
            np.random.shuffle(causes)
            for c in range(4):
                orders[r, c] = causes[c]
        return perms, orders

    for seed in (1, 4242, 999983):
        nb_perms, nb_orders = _kernel_draws(seed, 6, 5)
        np.random.seed(seed)
        py_perms = np.empty((5, 6), dtype=np.int32)
        py_orders = np.empty((5, 4), dtype=np.int32)
        causes = np.array([0, 1, 2, 3], dtype=np.int32)
        for r in range(5):
            py_perms[r] = np.random.permutation(6).astype(np.int32)
            np.random.shuffle(causes)
            py_orders[r, :] = causes
        np.testing.assert_array_equal(nb_perms, py_perms)
        np.testing.assert_array_equal(nb_orders, py_orders)


def test_exp_library_divergence_is_real_and_amplified_by_the_mortality_form():
    """Characterisation, not a gate: why bit-exactness is rate-dependent.

    Numba lowers ``np.exp`` to libm; NumPy uses its own SIMD polynomial. The two disagree
    in the last ULP on a few percent of arguments, and ``dead = abd * (1 - exp(-D))``
    amplifies that by ~1/D. Recorded here so that a future exact-equality failure is
    diagnosed as a library difference rather than blamed on the kernel.
    """
    draws = np.abs(np.random.default_rng(11).uniform(0.0, 1.0e-2, 20000))
    np_vals = np.exp(-draws)
    nb_vals = np.array([float(_nb_exp(-float(d))) for d in draws])
    mismatches = int((np_vals != nb_vals).sum())
    assert mismatches > 0, (
        "numba and numpy exp now agree everywhere sampled -- if that is genuinely true "
        "this file's rate-selection caveats can be dropped, but verify before doing so"
    )
    # Amplification: relative error in (1 - exp(-D)) is ~eps/D, not ~eps.
    rel = np.abs(np_vals - nb_vals) / np.maximum(1.0 - np_vals, 1e-300)
    assert rel.max() > 1e-13, f"expected >=1e-13 amplification, saw {rel.max():.3e}"


# ---------------------------------------------------------------------------
# 1. The per-cell harness
# ---------------------------------------------------------------------------


def test_python_arm_never_enters_the_kernel_and_the_numba_arm_always_does():
    """The single load-bearing property of this file (see the module docstring)."""
    config = base_config()
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(state, config, seed=7, n_subdt=2)
    assert python_arm.kernel_calls == 0
    assert numba_arm.kernel_calls == 1


def test_cell_arms_agree_bit_exactly_bioen_off():
    config = base_config()
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(state, config, seed=7, n_subdt=2, sub_steps=2)
    assert_exp_library_agreement(*python_arm.eff_rates.values(), label="cell/bioen-off")
    assert_arms_equal(
        python_arm,
        numba_arm,
        witness_fields=("n_dead", "preyed_biomass", "pred_success_rate", "tl_weighted_sum"),
        witness_causes=(
            MortalityCause.PREDATION,
            MortalityCause.STARVATION,
            MortalityCause.ADDITIONAL,
            MortalityCause.FISHING,
            MortalityCause.DISCARDS,
        ),
        label="cell/bioen-off",
    )


def test_cell_arms_agree_with_resource_depletion_and_diet_tracking():
    grid = Grid.from_dimensions(ny=1, nx=2)
    rsc_cfg, resources = resource_config_and_state(grid)
    config = base_config(**rsc_cfg)
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(
        state, config, seed=3, n_subdt=2, resources=resources, grid=grid, sub_steps=2
    )
    assert_arms_equal(
        python_arm,
        numba_arm,
        witness_fields=("n_dead", "preyed_biomass", "diet_matrix", "tl_weighted_sum"),
        label="cell/resources",
    )
    # The resource must actually have been eaten, or `resource_biomass` is a dead field.
    assert python_arm.resource_biomass is not None
    assert float(python_arm.resource_biomass.sum()) < 12.0 * grid.ny * grid.nx


def test_cell_path_writes_neither_abundance_nor_trophic_level_on_either_arm():
    """Justifies the two GUARD_FIELDS: they are no-write sentinels, not live witnesses.

    ``mortality()`` derives ``abundance`` from ``n_dead`` and updates ``trophic_level``
    AFTER the interleaved loop, so an arm that started writing either inside the loop
    would be double-counting. The equality check on them is free; this test records what
    it is guarding.
    """
    config = base_config()
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(state, config, seed=5, n_subdt=2)
    for arm in (python_arm, numba_arm):
        for name in sorted(GUARD_FIELDS):
            np.testing.assert_array_equal(
                getattr(arm.state, name),
                getattr(arm.input_state, name),
                err_msg=f"{arm.label} wrote {name} inside the cell loop",
            )


def test_dormant_fields_are_untouched_bioen_off_so_task_2_cannot_read_this_as_coverage():
    """``e_net``/``gonad_weight``/``raw_preyed`` discriminate nothing until Task 2."""
    config = base_config()
    state = base_state()
    python_arm, _ = run_cell_both_paths(state, config, seed=5, n_subdt=2)
    assert python_arm.raw_preyed is None
    assert not np.any(python_arm.state.e_net != 0.0)
    assert not np.any(python_arm.state.gonad_weight != 0.0)


@pytest.mark.parametrize("seed", [0, 13, 20250903])
def test_cell_arms_agree_across_seeds(seed):
    """Different seeds shuffle the cause order differently -- the arms must track it."""
    config = base_config()
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(state, config, seed=seed, n_subdt=3)
    assert_arms_equal(python_arm, numba_arm, label=f"cell/seed={seed}")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="`use_full_numba` still carries `not config.bioen_enabled`, so the candidate "
    "arm falls back to Python and `require_kernel` fires with `assert 0 >= 1`. Plan Task 4 "
    "flips it; this must then XPASS and the marker be removed deliberately rather than the "
    "bioen gate quietly never being written. `raises=AssertionError` narrows the marker so "
    "an unrelated breakage (a config-parsing error, a None bioen array) cannot keep the "
    "xfail satisfied after the flip and leave the gate permanently unwritten -- verify with "
    "`pytest --runxfail` that the message is still the require_kernel one.",
)
def test_cell_arms_agree_under_bioen():
    from tests._bioen_overlay import BIOEN_OVERLAY

    overrides = {k: v for k, v in BIOEN_OVERLAY.items() if k != "temperature.value"}
    overrides.update(
        {
            "species.maturity.m0.sp0": "30.0",
            "species.maturity.m1.sp0": "0",
            "species.maturity.r.sp0": "0.2",
            "species.maturity.eta.sp0": "1",
            "species.beta.sp0": "0.8",
            "species.bioen.assimilation.sp0": "0.7",
            "species.bioen.mobilized.tp.sp0": "10",
            "species.bioen.mobilized.e.mobi.sp0": "0.65",
            "species.bioen.mobilized.e.d.sp0": "1.5",
            "species.bioen.maint.e.maint.sp0": "0.65",
            "species.bioen.maint.energy.c_m.sp0": "1.0e12",
        }
    )
    for i in (1, 2):
        for key, value in list(overrides.items()):
            if key.endswith(".sp0"):
                overrides[key[:-4] + f".sp{i}"] = value
    config = base_config(**overrides)
    assert config.bioen_enabled
    state = base_state().replace(
        e_net=np.array([-2.0, 0.0, -0.5, -1.0, 0.0, 0.0, -0.2, 0.0]),
        gonad_weight=np.array([0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    python_arm, numba_arm = run_cell_both_paths(state, config, seed=7, n_subdt=2, sub_steps=2)
    # Witnesses are mandatory under bioen (assert_arms_equal enforces it): without them
    # two arms that both ignore bioen agree perfectly. FORAGING is absent here because
    # BIOEN_OVERLAY deliberately leaves k_for at 0 -- Task 2 must set it and add the
    # `n_dead[:, FORAGING] > 0` witness itself.
    assert_arms_equal(
        python_arm,
        numba_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed"),
        witness_causes=(MortalityCause.PREDATION, MortalityCause.STARVATION),
        label="cell/bioen-on",
    )


# ---------------------------------------------------------------------------
# 2. The batch harness -- the kernels production actually runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_kernel_matches_the_per_cell_kernel(parallel):
    grid = Grid.from_dimensions(ny=1, nx=2)
    config = base_config()
    state = base_state()
    per_cell_arm, batch_arm = run_batch_both_paths(
        state, config, seed=4242, parallel=parallel, grid=grid, n_subdt=2
    )
    assert per_cell_arm.kernel_calls == 2, "both cells must have been visited"
    assert_arms_equal(
        per_cell_arm,
        batch_arm,
        witness_fields=("n_dead", "preyed_biomass", "pred_success_rate", "tl_weighted_sum"),
        witness_causes=(
            MortalityCause.PREDATION,
            MortalityCause.STARVATION,
            MortalityCause.ADDITIONAL,
            MortalityCause.FISHING,
            MortalityCause.DISCARDS,
        ),
        label=f"batch/parallel={parallel}",
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_kernel_matches_the_per_cell_kernel_with_resources(parallel):
    grid = Grid.from_dimensions(ny=1, nx=2)
    rsc_cfg, resources = resource_config_and_state(grid)
    config = base_config(**rsc_cfg)
    state = base_state()
    per_cell_arm, batch_arm = run_batch_both_paths(
        state, config, seed=4242, parallel=parallel, grid=grid, n_subdt=2, resources=resources
    )
    assert_arms_equal(
        per_cell_arm,
        batch_arm,
        witness_fields=("n_dead", "diet_matrix", "tl_weighted_sum"),
        label=f"batch/resources/parallel={parallel}",
    )
    assert per_cell_arm.resource_biomass is not None
    assert float(per_cell_arm.resource_biomass.sum()) < 12.0 * grid.ny * grid.nx


def test_batch_harness_refuses_a_single_cell_fixture():
    """A one-cell fixture would make `prange` iterate once -- no cross-cell coverage."""
    grid = Grid.from_dimensions(ny=1, nx=2)
    config = base_config()
    state = base_state(n_cells_used=1)
    with pytest.raises(AssertionError, match="non-empty cells"):
        run_batch_both_paths(state, config, seed=1, parallel=True, grid=grid, n_subdt=2)


def test_a_bioen_comparison_must_witness_a_dormant_field():
    """The batch harness has no ``require_kernel`` analogue -- this is its anti-vacuity.

    ``run_batch_both_paths`` bypasses ``use_full_numba`` entirely: both arms call kernels
    directly. Under bioen today that means it compares two kernels which BOTH ignore bioen
    and which therefore agree perfectly. Nothing about the harness can detect that; only
    the witness set can. So ``assert_arms_equal`` refuses a bioen comparison that does not
    witness at least one of ``e_net`` / ``gonad_weight`` / ``raw_preyed``.
    """
    config = base_config()
    state = base_state()
    py, nb = run_cell_both_paths(state, config, seed=7, n_subdt=2)
    py.bioen = nb.bioen = True  # pretend, so the guard is testable without Task 4's flip
    with pytest.raises(AssertionError, match="must witness at least one of"):
        assert_arms_equal(py, nb, label="bioen-guard")


# ---------------------------------------------------------------------------
# 3. Plan Task 1 Step 4: the FISHING rate-composition risk, measured now
# ---------------------------------------------------------------------------


def _logistic_selectivity_config(*, seasonality: bool, spatial: bool) -> EngineConfig:
    """Force the two paths to compose the fishing rate in DIFFERENT orders.

    Per-school (``_apply_fishing_for_school``): ``f_rate *= cell_factor`` first, then
    ``F = f_rate * season * selectivity / n_subdt``.
    Precomputed (``_precompute_effective_rates``):
    ``eff = f_rate * selectivity * spatial * mpa * season / n_subdt``.

    With the default length-cutoff selectivity of exactly 1.0 the reorderings are all
    multiplications by 1.0 and therefore exact in IEEE-754. A logistic selectivity is a
    generic double, so ``(f*c)*s`` and ``(f*s)*c`` genuinely differ.
    """
    config = base_config()
    config.fishing_selectivity_type = np.array([1, 1, 1], dtype=np.int32)
    config.fishing_selectivity_l50 = np.array([30.0, 3.0, 5.0])
    config.fishing_selectivity_slope = np.array([0.3, 0.7, 0.5])
    config.fishing_rate = np.array([0.4, 0.25, 0.31])
    if seasonality:
        config.fishing_seasonality = np.tile(np.array([0.37, 1.63]), (3, config.n_dt_per_year // 2))
    if spatial:
        config.fishing_spatial_maps = [np.array([[0.83, 1.29]]) for _ in range(3)]
    return config


def _python_path_fishing_rate(
    idx: int, state: SchoolState, config: EngineConfig, n_subdt: int, step: int
) -> float:
    """Transcribe ``_apply_fishing_for_school``'s rate composition ORDER, for attribution.

    Deliberately a separate transcription rather than a call into the reference: its
    purpose is to say *which* of the two hazards moved a number, and it must therefore be
    readable next to ``_precompute_effective_rates``'s one-line product. Covers only the
    knobs these fixtures use (constant rate, logistic selectivity, spatial map,
    seasonality); no MPA, no fleet effort, no by-year override.
    """
    sp = int(state.species_id[idx])
    f_rate = float(config.fishing_rate[sp])
    l50 = config.fishing_selectivity_l50[sp]
    sel_type = int(config.fishing_selectivity_type[sp])
    if sel_type == 1:
        slope = config.fishing_selectivity_slope[sp]
        selectivity = 1.0 / (1.0 + np.exp(-slope * (state.length[idx] - l50)))
    elif sel_type == 0:
        a50 = config.fishing_selectivity_a50[sp]
        selectivity = 0.0 if state.age_dt[idx] / config.n_dt_per_year < a50 else 1.0
    else:  # length cutoff (-1 legacy): exactly 1.0, or the school is exempt
        selectivity = 0.0 if (l50 > 0 and state.length[idx] < l50) else 1.0
    sp_map = config.fishing_spatial_maps[sp]
    if sp_map is not None:
        f_rate = f_rate * sp_map[int(state.cell_y[idx]), int(state.cell_x[idx])]
    if config.fishing_seasonality is not None:
        season = config.fishing_seasonality[sp, step % config.n_dt_per_year]
        return float(f_rate * season * selectivity / n_subdt)
    return float(f_rate * selectivity / (config.n_dt_per_year * n_subdt))


def _fishing_rate_report(config: EngineConfig, state: SchoolState, n_subdt: int, step: int = 0):
    """Per-school (F_python, F_kernel) with both hazards evaluated. Returns a list."""
    st = state.replace(feeding_stage=compute_feeding_stages(state, config))
    _, _, eff_fishing, _ = M._precompute_effective_rates(st, config, n_subdt, step)
    rows = []
    for idx in range(len(st)):
        if st.is_background[idx] or eff_fishing[idx] <= 0.0:
            continue
        f_py = _python_path_fishing_rate(idx, st, config, n_subdt, step)
        f_kernel = float(eff_fishing[idx])
        rows.append(
            {
                "idx": idx,
                "f_python": f_py,
                "f_kernel": f_kernel,
                "rate_bit_equal": f_py == f_kernel,
                "rate_ulp": abs(f_py - f_kernel) / np.spacing(abs(f_kernel)),
                "exp_bit_equal": float(np.exp(-f_py)) == float(_nb_exp(-f_kernel)),
            }
        )
    return rows


def test_gate_fixture_composes_the_fishing_rate_identically_on_both_paths():
    """Why ``base_config`` is exactness-safe, asserted rather than asserted-by-comment.

    The gate fixtures leave selectivity at the length-cutoff default (exactly 1.0 for
    every school here) with no spatial map, no seasonality, no MPA and no fleet, so
    ``_apply_fishing_for_school``'s ``f * sel / (ndt*subdt)`` and
    ``_precompute_effective_rates``'s ``f * sel * 1.0 * 1.0 / denom`` are the same
    floating-point expression. If a future edit adds any of those knobs to ``base_config``
    this test goes red BEFORE the equality gates start failing for a reason nobody can
    attribute.
    """
    config = base_config()
    rows = _fishing_rate_report(config, base_state(), n_subdt=2)
    assert rows, "the gate fixture must actually fish, or the check is vacuous"
    offenders = [r for r in rows if not r["rate_bit_equal"]]
    assert not offenders, f"gate fixture rate composition already diverges: {offenders}"
    exp_offenders = [r for r in rows if not r["exp_bit_equal"]]
    assert not exp_offenders, (
        "gate fixture lands on rates where numba's libm exp and numpy's exp disagree: "
        f"{exp_offenders} -- pick neighbouring rates, do not loosen the comparison"
    )


@pytest.mark.parametrize(
    ("seasonality", "spatial"),
    [(True, False), (False, True), (True, True)],
)
def test_fishing_composition_order_costs_exactly_one_ulp(seasonality, spatial):
    """Plan Task 1 Step 4's named risk: measured, attributed and bounded.

    ``_precompute_effective_rates`` composes ``f * sel * spatial * mpa * season`` while
    ``_apply_fishing_for_school`` composes ``(f * spatial) * season * sel``. Once
    selectivity is a generic double the two are genuinely different products and differ by
    at most one ULP. This is NOT a reason to loosen ``assert_arms_equal``; the two
    available fixes are (a) reorder the precompute's product to match the per-school one,
    which costs nothing at runtime, or (b) keep gate fixtures on unit selectivity as
    ``base_config`` does. Neither is Task 1's to make -- Task 1's job is to prove the
    hazard is bounded at 1 ULP so a later task cannot mistake it for a logic defect.
    """
    config = _logistic_selectivity_config(seasonality=seasonality, spatial=spatial)
    rows = _fishing_rate_report(config, base_state(), n_subdt=2)
    diverging = [r for r in rows if not r["rate_bit_equal"]]
    assert diverging, (
        "expected the reordered product to move at least one rate; if IEEE-754 now makes "
        "these orders agree the hazard is gone and this test should be deleted"
    )
    assert max(r["rate_ulp"] for r in diverging) <= 1.0, (
        f"composition-order divergence exceeded 1 ULP: {diverging}"
    )


def test_fishing_composition_hazard_stays_below_1e_12_relative_in_n_dead():
    """The downstream size of both hazards together, on the worst fixture found.

    ``seasonality=False, spatial=True`` is the combination where ``assert_arms_equal``
    actually fails: two schools pick up a 1-ULP rate difference from the composition order
    AND one school lands on an argument where numba's libm ``exp`` and numpy's ``exp``
    disagree. ``1 - exp(-D)`` amplifies both by ~1/D. This test records the resulting
    magnitude so the report's number is reproducible; the exact gates elsewhere in this
    file stay exact.
    """
    config = _logistic_selectivity_config(seasonality=False, spatial=True)
    grid = Grid.from_dimensions(ny=1, nx=2)
    state = base_state()
    python_arm, numba_arm = run_cell_both_paths(
        state, config, seed=19, n_subdt=2, grid=grid, sub_steps=2
    )
    rows = _fishing_rate_report(config, state, n_subdt=2)
    assert any(not r["rate_bit_equal"] for r in rows), "composition hazard absent"
    assert any(not r["exp_bit_equal"] for r in rows), "exp-library hazard absent"

    a = python_arm.state.n_dead
    b = numba_arm.state.n_dead
    assert not np.array_equal(a, b), (
        "this fixture is the documented non-exact case; if it is now exact, re-measure "
        "and simplify the caveats in this module's docstring"
    )
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    assert rel.max() < 1e-12, f"divergence grew beyond a last-ULP effect: {rel.max():.3e}"
