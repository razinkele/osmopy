"""Equivalence gates: the Numba mortality kernels vs the pure-Python reference.

Task 1 of ``docs/superpowers/plans/2026-08-31-bioen-numba-kernel.md``. Tasks 2-4 build
their behavioural tests on ``run_cell_both_paths`` / ``run_batch_both_paths`` and
``assert_arms_equal`` from this module.

WHY THE ``_HAS_NUMBA`` TOGGLE IS THE POINT OF THIS FILE
-------------------------------------------------------
``_mortality_in_cell`` is not "the Python path". It computes

    use_full_numba = (_HAS_NUMBA and inst_abd is not None and rsc_size_min is not None
                      and eff_starv is not None)

and, when true, *dispatches into* ``_mortality_in_cell_numba``. **Task 3 Step 2 (done)**
removed the ``not config.bioen_enabled`` term that used to force this branch off under
bioen (Task 4 is whole-run sanity, not the flip). A harness that simply called
``_mortality_in_cell`` twice would, post-flip, run the kernel on BOTH arms and compare the
kernel with itself -- green forever, pinning nothing. So the reference arm here sets
``M._HAS_NUMBA = False`` for the duration of its call and restores it afterwards, which is
the only thing that guarantees the reference arm executes ``_apply_*_for_school``.

``ArmResult.kernel_calls`` closes the loop from the other side: the reference arm must
show **exactly 0** kernel entries and the candidate arm **at least 1**. ``require_kernel``
therefore turns "the candidate silently fell back to Python" (what happened under bioen
before the Task 3 flip) into a loud failure instead of a vacuous pass.

THE BIOEN TRIPWIRE *WAS* ``test_bioen_still_falls_back_to_python_until_the_dispatch_flip``
--------------------------------------------------------------------------------------------
Historical note, kept because the reasoning still matters for anyone re-deriving a similar
gate: an earlier revision of this file claimed the ``xfail`` on
``test_cell_arms_agree_under_bioen`` would XPASS at the flip and that ``strict=True`` would
force someone to remove it deliberately. **That was false and was disproved by review**:
applying the real dispatch flip with ZERO bioen work in the kernel still printed
``1 xfailed`` -- the test merely stopped failing on ``require_kernel`` and started failing
on ``n_dead`` (17/64 mismatched, max rel. diff 0.75). ``raises=AssertionError`` narrowed
nothing, because ``require_kernel``, every equality exit and every witness check all raise
``AssertionError``. So "dispatch flipped" and "bioen kernel absent or partial" were the
same colour, and only an unmarked test that asserted the dispatch mechanism *positively*
could tell them apart.

That unmarked test has done its job: the dispatch is flipped (Task 3 Step 2) AND the
kernel carries all five bioen behaviours (Task 2), so it has been inverted into
``test_bioen_dispatch_reaches_the_cell_kernel`` (``kernel_calls >= 1``) and now serves as
the permanent positive pin on the INNER gate. ``test_cell_arms_agree_under_bioen`` no
longer carries the ``xfail`` marker -- it is the real per-cell equivalence gate under
bioen, and it passes bit-exact on ``bioen_fixture()``. The equivalent positive pin on the
OUTER gate (`mortality()`'s own batch dispatch) is
``test_mortality_reaches_batched_numba_under_bioen`` in
``tests/test_engine_bioen_mortality_parity.py``.

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

GLOBAL STATE THIS FILE TOUCHES, AND PUTS BACK
---------------------------------------------
Three process-global handles are mutated and restored in ``finally`` blocks, because this
file sorts before several suites that would silently inherit the change:
``M._HAS_NUMBA`` / ``M._mortality_in_cell_numba`` (per arm), ``numba.set_num_threads``
(parallel batch arm), and **NumPy's legacy global RNG** (``_replicate_batch_kernel_rng``
must use the legacy stream -- it is the only one that matches Numba's -- but must not
leave it mutated; ``tests/conftest.py`` does no global seeding and at least six modules
using ``np.random``'s global functions sort after this one).
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import re
from collections.abc import Iterator, Sequence
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
    allow_missing_bioen_witnesses: Sequence[str] = (),
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
        allow_missing_bioen_witnesses: dormant fields this particular bioen comparison is
            allowed to skip. Naming one is a deliberate, greppable statement that the
            behaviour under test does not write it (e.g. a per-fish-cap test in which
            starvation never fires and ``gonad_weight`` stays zero). Silence is not.
        label: prefix for assertion messages.

    Raises:
        AssertionError: under bioen, if ``witness_fields`` omits any of ``DORMANT_FIELDS``
            without naming it in ``allow_missing_bioen_witnesses``.
            ``run_batch_both_paths`` bypasses ``use_full_numba`` entirely -- both its arms
            call kernels directly -- so it has no ``require_kernel`` analogue and, under
            bioen, would happily compare two kernels that BOTH ignore bioen and agree
            perfectly. The witness set is the only thing standing between Task 2 and a
            green-but-vacuous bioen batch test. Note this remains a floor, not a proof:
            ``np.any(arr != 0.0)`` accepts a single non-zero element, so pair it with
            ``assert_bioen_changes_the_answer`` for a real sensitivity check.
    """
    prefix = f"[{label}] " if label else ""
    if reference.bioen:
        missing = sorted(DORMANT_FIELDS - set(witness_fields) - set(allow_missing_bioen_witnesses))
        if missing:
            raise AssertionError(
                f"{prefix}a bioen comparison must witness {missing} (or name them in "
                "allow_missing_bioen_witnesses) -- otherwise two arms that both ignore "
                "bioen agree perfectly and the test is vacuous. Plan Task 2's full set is "
                "witness_fields=('n_dead', 'e_net', 'gonad_weight', 'raw_preyed') with "
                "witness_causes including STARVATION and FORAGING (the latter needs "
                "species.bioen.forage.k_for > 0, which BIOEN_OVERLAY deliberately does "
                "not set)."
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


def assert_bioen_changes_the_answer(
    state: SchoolState,
    config: EngineConfig,
    *,
    seed: int,
    n_subdt: int = 10,
    label: str = "",
    **kwargs,
) -> None:
    """Negative control: turning bioen OFF on this fixture must move ``n_dead``.

    Witness fields are a floor, not a proof -- ``np.any(arr != 0.0)`` is satisfied by a
    single non-zero element, and a fixture can witness ``e_net`` while the bioen behaviour
    under test never actually binds. This asks the sharper question directly: if flipping
    ``bioen_enabled`` off does not change the answer, the comparison cannot be testing
    bioen, whatever field is witnessed.

    Runs only the PYTHON reference arm on both settings (cheap, and it is the fixture being
    interrogated, not the kernel), so it is usable today, before the dispatch flip.
    """
    prefix = f"[{label}] " if label else ""
    assert config.bioen_enabled, f"{prefix}negative control needs a bioen-enabled config"
    config_off = copy.copy(config)
    config_off.bioen_enabled = False

    on, _ = run_cell_both_paths(
        state, config, seed=seed, n_subdt=n_subdt, require_kernel=False, **kwargs
    )
    off, _ = run_cell_both_paths(
        state, config_off, seed=seed, n_subdt=n_subdt, require_kernel=False, **kwargs
    )
    if np.array_equal(on.state.n_dead, off.state.n_dead):
        raise AssertionError(
            f"{prefix}n_dead is identical with bioen ON and OFF, so this fixture does not "
            "exercise bioen at all and any equality gate built on it is vacuous. Check that "
            "the cap binds, that e_net is negative on a school past first feeding, and that "
            "c_m is material (tests/_bioen_overlay.py's C_M) rather than the ~1e-8 that "
            "switches starvation off entirely."
        )


@contextlib.contextmanager
def legacy_rng_preserved() -> Iterator[None]:
    """Save/restore NumPy's LEGACY global RNG state around a block that seeds it.

    Replicating the batch kernels' draws *requires* the legacy global stream -- it is the
    only one that reproduces Numba's MT19937 (``Generator``/PCG64 does not). Leaving it
    mutated is a different matter: ``tests/conftest.py`` does no global seeding, and
    several suites that call ``np.random``'s module-level functions sort after this file
    (``test_engine_physical_data``, ``test_ensemble``, ``test_grid_creation``,
    ``test_reporting``, ``test_results``, ``test_study_workflows``, ``test_ui_results``).
    This mirrors what the parallel batch arm already does for ``numba.set_num_threads``.
    """
    saved = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(saved)


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
    eff_foraging: NDArray[np.float64]
    eta_school: NDArray[np.float64]


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
        # Built by the same helpers ``mortality()`` uses, so a fixture cannot disagree
        # with production about the FORAGING rate or eta.
        eff_foraging=M._precompute_foraging_rates(st, config, n_subdt),
        eta_school=M._precompute_bioen_eta(st, config),
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
    into kernel-vs-kernel (after plan Task 3 Step 2) or Python-vs-Python (under bioen
    today).
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
                    eta_school=setup.eta_school,
                    eff_foraging=setup.eff_foraging,
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
            "(Expected under bioen until plan Task 3 Step 2 flips the "
            "`not config.bioen_enabled` "
            "term in `use_full_numba`.)"
        )
    return python_arm, numba_arm


# ---------------------------------------------------------------------------
# Harness 2: per-cell kernel vs the batch kernels production actually runs
# ---------------------------------------------------------------------------

#: The batch kernels' inline draw pattern, per non-empty cell, in order:
#: ``seq_pred, seq_starv, seq_fish, seq_nat`` then ``n_local`` shuffles of ``[0,1,2,3]``.
#: Plan Task 2 adds a fifth permutation (``seq_for``) and a fifth cause code under bioen.
#: This is NOT left to a comment and a careful reader: ``_assert_batch_rng_contract``
#: MEASURES what the kernel actually draws at runtime and refuses to run if the harness's
#: expectation has drifted, so a one-sided update fails with a message naming both files
#: instead of surfacing as an RNG desync that looks exactly like a logic bug.
_BATCH_N_PERMS_BIOEN_OFF = 4
_BATCH_PARALLEL_SEED_STRIDE = 7919

#: (cell occupancy, seed) for the runtime draw probes. Several sizes are required: a single
#: probe only pins the kernel's stream POSITION, and different (n_perms, width) pairs can
#: land on the same position because permutation/shuffle consume a size-dependent number of
#: raw 32-bit words. Measured: n_local=8 alone admits both (0, 7) and (4, 4); intersecting
#: four probes leaves exactly (4, 4).
_RNG_PROBES: tuple[tuple[int, int], ...] = ((8, 4242), (5, 4242), (3, 11), (7, 99))
_MAX_PROBED_PERMS = 9


@numba.njit(cache=True)
def _nb_next_random():
    """Next draw from NUMBA's thread-local PRNG -- i.e. where a kernel left the stream.

    Numba's ``np.random`` state is shared across ``njit`` functions on a thread, so calling
    this straight after a kernel reads that kernel's own stream position. Verified against
    NumPy's legacy MT19937 in ``test_numba_prng_state_is_shared_across_njit_functions``.
    """
    return np.random.random()


def _candidates_from_stream_position(
    next_draw: float, n_local: int, seed: int
) -> set[tuple[int, int]]:
    """Every ``(n_perms, cause_width)`` whose replay leaves the stream at ``next_draw``.

    Replays ``n_perms`` permutations of ``n_local`` then ``n_local`` shuffles of a
    ``width``-element array on NumPy's legacy stream (bit-identical to Numba's) and keeps
    the shapes that land on the same next value. A single call is AMBIGUOUS -- see
    ``_RNG_PROBES`` -- so callers intersect several.
    """
    candidates: set[tuple[int, int]] = set()
    with legacy_rng_preserved():
        for n_perms in range(_MAX_PROBED_PERMS):
            for width in range(_MAX_PROBED_PERMS):
                np.random.seed(seed)
                for _ in range(n_perms):
                    np.random.permutation(n_local)
                causes = np.arange(width, dtype=np.int32)
                for _ in range(n_local):
                    np.random.shuffle(causes)
                if np.random.random() == next_draw:
                    candidates.add((n_perms, width))
    return candidates


def _probe_kernel_draw_shape(
    state: SchoolState,
    config: EngineConfig,
    *,
    n_local: int,
    seed: int,
    n_subdt: int,
) -> set[tuple[int, int]]:
    """Run the sequential batch kernel on one cell and report every ``(n_perms, width)``
    consistent with how far it advanced Numba's PRNG.

    RUNTIME measurement, not source introspection. Source counting was tried first and
    rejected: it reports a bioen-CONDITIONAL draw (``if bioen: seq_for = permutation(...)``
    -- the natural way to write Task 2's fifth permutation) as unconditional, and it
    degrades silently to ``([], -1)`` if the triplicated cause loop is ever hoisted into a
    shared helper. Measuring the real stream advance is immune to both, and it measures the
    branch THIS config actually takes.
    """
    grid = Grid.from_dimensions(ny=1, nx=2)
    cell_x = np.full(len(state), -1, dtype=np.int32)
    cell_x[:n_local] = 0
    probe_state = state.replace(cell_x=cell_x, cell_y=np.zeros(len(state), dtype=np.int32))
    sorted_indices, boundaries, n_cells = _cell_groups(probe_state, grid)
    assert int(boundaries[1] - boundaries[0]) == n_local, "probe fixture built wrong"

    arm = _prepare_arm(probe_state, config, None, n_subdt=n_subdt, step=0, diet_tracking=True)
    n = len(arm.state)
    M._mortality_all_cells_numba(
        seed,
        sorted_indices,
        boundaries,
        n_cells,
        arm.inst_abd,
        arm.state.n_dead,
        arm.eff_starv,
        arm.eff_additional,
        arm.eff_fishing,
        arm.fishing_discard,
        arm.state.species_id,
        arm.state.length,
        arm.state.weight,
        arm.state.age_dt,
        arm.state.first_feeding_age_dt,
        arm.state.feeding_stage,
        arm.state.pred_success_rate,
        arm.state.preyed_biomass,
        arm.state.trophic_level,
        config.size_ratio_min,
        config.size_ratio_max,
        config.ingestion_rate,
        config.fr_shape,
        config.fr_halfsat,
        config.n_dt_per_year,
        n_subdt,
        M._DUMMY_ACCESS,
        False,
        False,
        np.zeros(n, dtype=np.int32),
        np.zeros(n, dtype=np.int32),
        M._DUMMY_RSC_2D,
        arm.rsc_arrays[0],
        arm.rsc_arrays[1],
        arm.rsc_arrays[2],
        arm.rsc_arrays[3],
        arm.rsc_arrays[4],
        config.n_species,
        arm.ctx.tl_weighted_sum,
        True,
        arm.ctx.diet_matrix,
        True,
        arm.state.egg_retained,
        config.bioen_enabled,
        arm.cap_fish if arm.cap_fish is not None else M._DUMMY_RSC_1D,
        arm.raw_preyed if arm.raw_preyed is not None else M._DUMMY_RSC_1D,
        arm.state.e_net,
        arm.state.gonad_weight,
        arm.eta_school,
        arm.state.is_background,
        arm.eff_foraging,
    )
    return _candidates_from_stream_position(float(_nb_next_random()), n_local, seed)


def measure_batch_kernel_rng_shape(
    state: SchoolState, config: EngineConfig, *, n_subdt: int
) -> tuple[int, int]:
    """``(n_perms, cause_order_width)`` the sequential batch kernel ACTUALLY draws per cell."""
    probes = [
        _probe_kernel_draw_shape(state, config, n_local=n_local, seed=seed, n_subdt=n_subdt)
        for n_local, seed in _RNG_PROBES
        if n_local <= len(state)
    ]
    assert len(probes) >= 3, (
        f"need >= 3 usable probe sizes but the fixture has only {len(state)} schools; add a "
        "smaller entry to _RNG_PROBES -- a single probe pins only the stream position and "
        "admits several (n_perms, width) pairs"
    )
    common = set.intersection(*probes)
    assert len(common) == 1, (
        f"runtime probes did not identify a unique draw shape (candidates {sorted(common)}). "
        "Add another (n_local, seed) entry to _RNG_PROBES: distinct cell sizes break the "
        "position aliasing."
    )
    return common.pop()


def _source_shape_hint() -> str:
    """Best-effort source read, used ONLY to enrich a failure message.

    Never asserts. Source counting cannot see a bioen-conditional draw and returns nothing
    useful if the cause loop is hoisted into a shared helper -- which is exactly why the
    contract above is measured at runtime. Kept because when the runtime check DOES fire,
    naming the two likely causes saves the reader a bisect.
    """
    try:
        body = inspect.getsource(M._mortality_all_cells_numba.py_func)
    except OSError:  # pragma: no cover - source unavailable
        return "source unavailable"
    literal = re.search(r"causes = np\.array\(\[([^\]]*)\]", body)
    conditional = re.search(r"if\s+\w+\s*:\s*\n\s+\w+\s*=\s*np\.random\.permutation", body)
    return (
        f"source shows {len(re.findall(r'np[.]random[.]permutation[(]', body))} literal "
        f"permutation call(s), cause literal "
        f"{literal.group(1) if literal else '<none found -- loop may have been hoisted>'}"
        f"{'; at least one permutation is inside a conditional' if conditional else ''}"
    )


def _assert_batch_rng_contract(
    state: SchoolState, config: EngineConfig, cause_codes: Sequence[int], n_perms: int, n_subdt: int
) -> None:
    """Refuse to replicate draws the kernels do not actually make.

    ``run_batch_both_paths`` reproduces the batch kernels' internal RNG in Python, which is
    only valid while the harness's idea of the draw pattern matches the kernels' -- and the
    two live in different files. Today both batch kernels draw FOUR permutations
    unconditionally over ``[0, 1, 2, 3]``, with no bioen branch. When Task 2 adds
    ``seq_for`` and the five-cause order, this fires first and says so, instead of letting a
    desynchronised stream masquerade as a behavioural divergence in the thing under test.

    Only the WIDTH of the cause order affects RNG consumption, so the runtime probe pins
    ``n_perms`` and ``len(cause_codes)``. The cause CODES cannot be measured this way; a
    wrong code set applies the wrong cause to the wrong school and shows up as an equality
    failure in the batch comparison itself.
    """
    measured_perms, measured_width = measure_batch_kernel_rng_shape(state, config, n_subdt=n_subdt)
    assert (n_perms, len(cause_codes)) == (measured_perms, measured_width), (
        f"harness replicates {n_perms} permutations and a {len(cause_codes)}-wide cause "
        f"order {list(cause_codes)}, but the batch kernel actually draws {measured_perms} "
        f"permutations and a {measured_width}-wide order "
        f"(measured at runtime; {_source_shape_hint()}). Update run_batch_both_paths's "
        "n_perms/cause_codes in the SAME commit as the mortality.py kernel change -- a "
        "one-sided update desynchronises the replicated stream and the resulting mismatch "
        "is indistinguishable from a logic bug."
    )


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

    The legacy global stream is restored on exit (``legacy_rng_preserved``): using it is
    unavoidable, leaving it mutated is not.
    """
    per_cell: dict[int, tuple[list[NDArray[np.int32]], NDArray[np.int32]]] = {}
    with legacy_rng_preserved():
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
    _assert_batch_rng_contract(state, config, cause_codes, n_perms, n_subdt)
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
            seqs[4] if len(seqs) > 4 else seqs[0],
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
            config.bioen_enabled,
            ref.cap_fish if ref.cap_fish is not None else M._DUMMY_RSC_1D,
            ref.raw_preyed if ref.raw_preyed is not None else M._DUMMY_RSC_1D,
            ref.state.e_net,
            ref.state.gonad_weight,
            ref.eta_school,
            ref.state.is_background,
            ref.eff_foraging,
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
            config.bioen_enabled,
            cand.cap_fish if cand.cap_fish is not None else M._DUMMY_RSC_1D,
            cand.raw_preyed if cand.raw_preyed is not None else M._DUMMY_RSC_1D,
            cand.state.e_net,
            cand.state.gonad_weight,
            cand.eta_school,
            cand.state.is_background,
            cand.eff_foraging,
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
        with legacy_rng_preserved():
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


def test_numba_prng_state_is_shared_across_njit_functions():
    """The premise of the runtime draw measurement, pinned separately from its use.

    ``_probe_kernel_draw_shape`` reads a kernel's stream position by calling
    ``_nb_next_random`` right after it. That only works because Numba's ``np.random`` state
    is per-thread, not per-function -- and it is only *interpretable* because that state is
    an MT19937 matching NumPy's legacy stream.
    """

    @numba.njit(cache=True)
    def _seed_and_draw(seed, n, k):
        np.random.seed(seed)
        out = np.empty((k, n), dtype=np.int32)
        for i in range(k):
            out[i] = np.random.permutation(n).astype(np.int32)
        return out

    for seed, k in ((7, 2), (4242, 4), (999983, 5)):
        _seed_and_draw(seed, 6, k)
        nb_next = float(_nb_next_random())
        with legacy_rng_preserved():
            np.random.seed(seed)
            for _ in range(k):
                np.random.permutation(6)
            py_next = float(np.random.random())
        assert nb_next == py_next, (
            "Numba's PRNG state is no longer shared across njit functions (or no longer "
            "matches NumPy's legacy stream); the runtime draw measurement rests on both"
        )


def test_the_batch_harness_measures_the_kernel_draw_shape_at_runtime():
    """The RNG-shape contract, so a Task 2 one-sided update cannot look like a logic bug.

    ``run_batch_both_paths`` reproduces the batch kernels' internal draws in Python. Today
    both kernels draw FOUR permutations unconditionally over ``[0, 1, 2, 3]``, with no
    bioen branch -- so the harness's ``n_perms = 5 if bioen`` is a promise about code that
    does not exist yet, and Task 2 is the commit that must make it true.

    Measured at runtime rather than read out of the source: source counting reports a
    bioen-CONDITIONAL fifth permutation as unconditional, which is the natural way Task 2
    will write it.
    """
    config, state = base_config(), base_state()
    assert measure_batch_kernel_rng_shape(state, config, n_subdt=2) == (4, 4)

    # The contract holds for what the harness asks for today...
    _assert_batch_rng_contract(state, config, [0, 1, 2, 3], 4, 2)
    # ...and refuses both halves of a bioen-shaped request while the kernel is unchanged.
    with pytest.raises(AssertionError, match="actually draws 4 permutations"):
        _assert_batch_rng_contract(state, config, [0, 1, 2, 3], 5, 2)
    with pytest.raises(AssertionError, match="4-wide order"):
        _assert_batch_rng_contract(state, config, [0, 1, 2, 3, 5], 4, 2)


def test_runtime_measurement_sees_a_bioen_conditional_draw_that_source_counting_misses():
    """Why the contract is measured, not read (fix round 2, NEW-3).

    Task 2's fifth permutation (``seq_for``) will naturally be written as a CONDITIONAL
    draw. A source-occurrence count reports such a kernel as drawing five permutations
    whether or not the branch is taken, so it would have misreported the bioen-OFF shape
    and broken the currently-green bioen-off batch tests the moment Task 2 landed. The
    runtime probe measures the branch actually executed.
    """

    @numba.njit(cache=True)
    def _task2_shaped_kernel(seed, n_local, bioen):
        """Stands in for what Task 2 will write into the batch kernels."""
        np.random.seed(seed)
        np.random.permutation(n_local)
        np.random.permutation(n_local)
        np.random.permutation(n_local)
        np.random.permutation(n_local)
        width = 4
        if bioen:
            np.random.permutation(n_local)
            width = 5
        causes = np.arange(width).astype(np.int32)
        for _ in range(n_local):
            np.random.shuffle(causes)

    def _measure(bioen: bool) -> set[tuple[int, int]]:
        sets = []
        for n_local, seed in _RNG_PROBES:
            _task2_shaped_kernel(seed, n_local, bioen)
            sets.append(_candidates_from_stream_position(float(_nb_next_random()), n_local, seed))
        return set.intersection(*sets)

    assert _measure(False) == {(4, 4)}, "runtime probe misread the untaken branch"
    assert _measure(True) == {(5, 5)}, "runtime probe misread the taken branch"

    # ...whereas counting occurrences in the source says 5 for BOTH, which is the defect.
    body = inspect.getsource(_task2_shaped_kernel.py_func)
    assert len(re.findall(r"np\.random\.permutation\(", body)) == 5


def test_a_single_runtime_probe_would_not_have_identified_the_draw_shape():
    """Why ``_RNG_PROBES`` has four entries: one probe pins only the stream POSITION.

    ``permutation(n)`` and ``shuffle(width)`` consume a size-dependent number of raw 32-bit
    words, so distinct ``(n_perms, width)`` pairs can leave the stream in the same place.
    At ``n_local=8`` the kernel's advance is equally consistent with ``(0, 7)`` and the true
    ``(4, 4)``; intersecting probes at different cell sizes removes the aliasing.
    """
    config, state = base_config(), base_state()
    single = _probe_kernel_draw_shape(state, config, n_local=8, seed=4242, n_subdt=2)
    assert (4, 4) in single
    assert len(single) > 1, (
        "a single probe now identifies the shape uniquely -- if that is genuinely true "
        "_RNG_PROBES could shrink, but verify at several sizes before trusting it"
    )
    assert measure_batch_kernel_rng_shape(state, config, n_subdt=2) == (4, 4)


def test_no_site_in_this_file_leaks_the_legacy_global_rng_stream():
    """This file seeds NumPy's legacy global RNG; seven later-sorting suites use it.

    ``tests/conftest.py`` does no global seeding, so an unrestored seed would make
    ``test_engine_physical_data`` / ``test_ensemble`` / ``test_grid_creation`` /
    ``test_reporting`` / ``test_results`` / ``test_study_workflows`` / ``test_ui_results``
    consume a stream this file chose. Same discipline the parallel batch arm already
    applies to ``numba.set_num_threads``.
    """
    # This test seeds the global stream too, so it must sit inside the same guard it
    # verifies -- the round-1 version did not, and was itself the only unguarded seeding
    # site left in the file (fix round 2, NEW-1).
    with legacy_rng_preserved():
        np.random.seed(123456)
        before = np.random.get_state()
        expected_next = np.random.random()

        np.random.set_state(before)
        grid = Grid.from_dimensions(ny=1, nx=2)
        run_batch_both_paths(
            base_state(), base_config(), seed=4242, parallel=False, grid=grid, n_subdt=2
        )
        assert np.random.random() == expected_next, (
            "run_batch_both_paths left NumPy's legacy global RNG advanced"
        )

        np.random.set_state(before)
        test_numba_and_numpy_legacy_rng_streams_match()
        assert np.random.random() == expected_next, (
            "test_numba_and_numpy_legacy_rng_streams_match left the legacy global RNG advanced"
        )


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


def bioen_fixture() -> tuple[EngineConfig, SchoolState]:
    """``base_config``/``base_state`` plus a material bioen budget on all three species.

    Uses ``BIOEN_OVERLAY``'s values (Task 0), not ``apply_overlay`` itself: that helper is
    authored for the Baltic demo config -- it copies ``species.maturity.size`` into ``m0``
    and rewrites background ingestion rates -- neither of which this synthetic 3-species
    fixture has. ``temperature.value`` is dropped for the same reason.
    """
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
    # School 2's gonad (5.0) is deliberately far above its deficit (|e_net|/n_subdt = 0.25
    # at n_subdt=2, eta=1) so it lands in the SUFFICIENT branch (absorb + repay, no kill,
    # no flush) and stays there across `sub_steps=2` and any survivor rescale from another
    # cause landing on it first (a rescale only shrinks |e_net| toward 0, which shrinks the
    # deficit, never grows it) -- see Task 3 report for the derivation. Without this,
    # `test_cell_arms_agree_under_bioen` witnesses `gonad_weight` as identically zero on
    # BOTH arms (school 0's insufficient gonad and school 2's undersized one both flush to
    # 0.0), which is vacuous, not evidence of a kernel defect.
    state = base_state().replace(
        e_net=np.array([-2.0, 0.0, -0.5, -1.0, 0.0, 0.0, -0.2, 0.0]),
        gonad_weight=np.array([0.5, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    return config, state


def test_bioen_dispatch_reaches_the_cell_kernel():
    """THE positive pin on the INNER gate (`_mortality_in_cell`'s `use_full_numba`).

    Formerly ``test_bioen_still_falls_back_to_python_until_the_dispatch_flip``, the
    tripwire that was supposed to go red the moment Task 3 Step 2 flipped the dispatch
    (see the module docstring's "THE BIOEN TRIPWIRE" section for why the ``xfail`` on
    ``test_cell_arms_agree_under_bioen`` could not serve that role). It has now done its
    job -- inverted here into the permanent regression gate. Paired with
    ``test_mortality_never_enters_batched_numba_under_bioen`` (renamed
    ``test_mortality_reaches_batched_numba_under_bioen``,
    ``tests/test_engine_bioen_mortality_parity.py``), which is the equivalent positive
    pin on the OUTER gate (`mortality()`'s own batch dispatch). Together they are what
    stands guard against either gate's `bioen_enabled` exclusion silently coming back.
    """
    config, state = bioen_fixture()
    _, numba_arm = run_cell_both_paths(state, config, seed=7, n_subdt=2, require_kernel=False)
    assert numba_arm.kernel_calls >= 1, (
        "the bioen dispatch gate reverted -- `use_full_numba` once again carries "
        "`not config.bioen_enabled` (or an equivalent exclusion), so the candidate arm "
        "fell back to the Python path under bioen instead of reaching "
        "_mortality_in_cell_numba."
    )


def test_the_bioen_fixture_actually_exercises_bioen():
    """Negative control on the fixture the bioen gate will use (CF-4).

    A witness field only proves something is non-zero. This proves bioen is what made it
    so: with ``bioen_enabled`` flipped off, the same fixture must give a different answer.
    """
    config, state = bioen_fixture()
    assert_bioen_changes_the_answer(state, config, seed=7, n_subdt=2, label="bioen-fixture")


def test_cell_arms_agree_under_bioen():
    """THE real bioen gate (promoted from an ``xfail`` placeholder by Task 3 Step 2).

    Exercises `_mortality_in_cell`'s own kwarg-to-positional threading into
    `_mortality_in_cell_numba` -- the "inner" half of the Task 2 review's P4 gap (an
    ordering bug at either bioen call site would ship silently because both harnesses
    call the kernels directly with their own hand-built argument lists). The "outer"
    half (`mortality()`'s own threading into the batch kernel) is closed by
    ``test_end_to_end_bioen_run_reaches_kernel_with_arrays_threaded_correctly`` in
    ``tests/test_engine_bioen_mortality_parity.py``, which drives a real
    `PythonEngine().run_in_memory` bioen simulation instead of a hand-built harness call.
    """
    config, state = bioen_fixture()
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
    py.bioen = nb.bioen = True  # pretend, so the guard is testable before the dispatch flip
    with pytest.raises(AssertionError, match="must witness"):
        assert_arms_equal(py, nb, label="bioen-guard")
    # Every dormant field must be named, not just one...
    with pytest.raises(AssertionError, match="gonad_weight"):
        assert_arms_equal(py, nb, witness_fields=("n_dead", "e_net"), label="bioen-guard")
    # ...but naming one in the opt-out is a deliberate, greppable exemption.
    assert_arms_equal(
        py,
        nb,
        witness_fields=("n_dead",),
        allow_missing_bioen_witnesses=("e_net", "gonad_weight", "raw_preyed"),
        label="bioen-guard",
    )


# ---------------------------------------------------------------------------
# 2b. Plan Task 2: the five bioen behaviours, against the Python reference
# ---------------------------------------------------------------------------

#: FORAGING rate for the gate fixture. ``eff_foraging = (k_for / 24) / 2 = 0.0125``, a
#: value on which numba's libm ``exp`` and numpy's agree (checked by
#: ``assert_exp_library_agreement`` in every test that uses it). The hazard is real at this
#: scale, not theoretical: ``k_for = 1.2`` and ``0.48`` both land on arguments where the
#: two implementations differ in the last ULP.
GATE_K_FOR = 0.6


def bioen_gate_fixture(*, k_for: float = GATE_K_FOR) -> tuple[EngineConfig, SchoolState]:
    """The Task 2 fixture: all five bioen behaviours observable at once.

    Distinct from ``bioen_fixture`` (Task 1's, which the dispatch pin
    ``test_bioen_dispatch_reaches_the_cell_kernel`` and the real per-cell equivalence gate
    ``test_cell_arms_agree_under_bioen`` use). Each requirement of
    plan Task 2 Step 2 is met by a named school, so a reviewer can check them one at a
    time -- ``test_the_gate_fixture_makes_every_behaviour_observable`` asserts every one
    of them mechanically, because "the gate passed because the code was never reached"
    has happened five times on this branch.

    ============ ==============================================================
    school       what it is for
    ============ ==============================================================
    0 (sp0, c0)  predator; the per-fish cap BINDS; gonad 0.5 does NOT cover its
                 deficit 1.0, so starvation kills and flushes the gonad
    1 (sp1, c0)  prey that dies inside the sub-step; gonad 0 < deficit
    2 (sp1, c0)  gonad 1.0 COVERS its deficit 0.25 -> repayment branch, no death
    3 (sp0, c1)  second predator, second cell (so ``prange`` has two cells)
    4 (sp1, c1)  prey
    5 (sp2, c1)  BACKGROUND -- eaten by school 3, so the ``_consume`` rescale must
                 skip it; its seeded ``preyed_biomass``/``e_net`` must come back
                 untouched
    6 (sp2, c1)  ``deficit/weight`` = 1.25e7 > ``inst_abd`` 3e6: the death count is
                 deliberately NOT clamped (Java's factor goes negative there), while
                 the survivor FACTOR is clamped to 0
    7 (sp1, c0)  ``ageDt == firstFeedingAgeDt`` exactly -- the STRICT bioen boundary,
                 so starvation must skip it although ``e_net < 0``
    ============ ==============================================================

    ``base_config`` already supplies ``additional > 0`` AND ``fishing > 0`` with a
    non-zero discard rate for sp0/sp1, on schools that eat -- the case the natural bioen
    fixture (``tests/test_engine_bioen_mortality_parity.py``'s ``_bioen_config``) cannot
    see, because it zeroes both rates for order-independence.

    ``preyed_biomass`` is SEEDED non-zero on every school except the two predators. The
    survivor rescale multiplies ``preyed_biomass`` and ``e_net``; relying on ``e_net``
    alone would not catch a rescale that was applied to only one of the two, and a prey
    school's ``preyed_biomass`` is otherwise zero at its first death unless PREDATION
    happened to precede that cause in its shuffled order. The two predators keep
    ``preyed_biomass = 0`` so that ``raw_preyed[p] > preyed_biomass[p]`` is a clean joint
    witness for behaviours 2 and 5.
    """
    config = base_config(**_bioen_overrides(k_for=k_for))
    assert config.bioen_enabled
    return config, bioen_gate_state()


def _bioen_overrides(k_for: float = GATE_K_FOR) -> dict[str, str]:
    """``BIOEN_OVERLAY``'s keys for the three synthetic species, plus a live ``k_for``.

    Split out of ``bioen_gate_fixture`` so the resource-bearing variants can compose it
    with ``resource_config_and_state``'s keys without rebuilding the list by hand.

    Uses ``BIOEN_OVERLAY``'s values rather than ``apply_overlay`` itself: that helper is
    authored for the Baltic demo config (it copies ``species.maturity.size`` into ``m0``
    and rewrites background ingestion rates), neither of which this synthetic fixture has.
    ``temperature.value`` is dropped for the same reason.
    """
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
            # BIOEN_OVERLAY deliberately leaves k_for at 0 (its module docstring says so),
            # which makes FORAGING inert and behaviour 4 untestable. Set it here.
            "species.bioen.forage.k_for.sp0": repr(k_for),
        }
    )
    for i in (1, 2):
        for key, value in list(overrides.items()):
            if key.endswith(".sp0"):
                overrides[key[:-4] + f".sp{i}"] = value
    return overrides


def bioen_gate_state() -> SchoolState:
    """``base_state`` with the bioen budget fields seeded -- see ``bioen_gate_fixture``."""
    return base_state().replace(
        e_net=np.array([-2.0, -0.4, -0.5, -1.0, -0.3, -0.7, -100.0, -1.5]),
        gonad_weight=np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        preyed_biomass=np.array([0.0, 3.0, 2.0, 0.0, 1.0, 5.0, 4.0, 6.0]),
    )


def _bioen_gate_rates(config: EngineConfig, state: SchoolState, n_subdt: int):
    """Every rate array the gate fixture drives, for ``assert_exp_library_agreement``."""
    st = state.replace(feeding_stage=compute_feeding_stages(state, config))
    eff_s, eff_a, eff_f, _ = M._precompute_effective_rates(st, config, n_subdt, 0)
    eff_for = M._precompute_foraging_rates(st, config, n_subdt)
    return eff_s, eff_a, eff_f, eff_for


def run_cell_bioen_both_paths(
    state: SchoolState,
    config: EngineConfig,
    *,
    seed: int,
    n_subdt: int = 2,
    resources: ResourceState | None = None,
    grid: Grid | None = None,
    step: int = 0,
    sub_steps: int = 2,
    diet_tracking: bool = True,
    access_matrix: NDArray[np.float64] | None = None,
    has_access: bool = False,
    use_stage_access: bool = False,
    min_non_empty_cells: int = 2,
) -> tuple[ArmResult, ArmResult]:
    """Python reference vs the per-cell Numba kernel UNDER BIOEN, walking EVERY cell.

    A GATE ON BIOEN CORRECTNESS ALONGSIDE ``test_cell_arms_agree_under_bioen``.
    ``run_batch_both_paths`` has a kernel on BOTH arms, so it can only prove the three
    inlined interleaved loops received the same edit -- it is structurally blind to a
    behaviour that is wrong in all three. ``run_cell_both_paths`` (used by
    ``test_cell_arms_agree_under_bioen``, post Task 3's dispatch flip) drives a single
    named cell through the real ``_mortality_in_cell`` dispatch and is a real bioen gate
    in its own right now, but only for one cell at a time.

    This harness's job is complementary, not a pre-flip workaround: it walks EVERY
    non-empty cell per sub-step in ``mortality()``'s own cell order, so a fixture whose
    behaviours are spread across multiple cells (e.g. ``bioen_gate_fixture``'s background
    school and no-death-clamp school, deliberately placed in the second cell so the batch
    kernels' ``prange`` has something to iterate) gets every one of them exercised on the
    gating arm, not just whichever single cell a caller happened to name. It calls
    ``_mortality_in_cell_numba`` DIRECTLY (independent of ``use_full_numba``'s gate,
    exactly as ``run_batch_both_paths`` does for its reference arm) and drives it with the
    draws the reference itself would have made (``_pre_generate_cell_rng``, which Task 0
    fixed to emit five permutations and a cause list from ``_get_mortality_causes``).

    Both arms walk EVERY non-empty cell per sub-step, in ``mortality()``'s own cell order
    (``_cell_groups``' argsort/searchsorted), rather than a single hand-picked cell: with
    one cell the fixture's background school and its no-death-clamp school -- which live
    in the second cell so the batch kernels' ``prange`` has something to iterate -- would
    never be visited, and three of the five behaviours would go unexercised on the arm
    that is supposed to gate them.

    Draw replication is VERIFIED, not assumed: both arms take a freshly seeded
    ``default_rng(seed)`` and the two generators' bit-generator states are compared after
    every sub-step. Without it a drift between ``_pre_generate_cell_rng`` and
    ``_mortality_in_cell``'s own draws would surface as an ``n_dead`` mismatch that reads
    exactly like a starvation or rescale bug -- the failure mode Task 1 built
    ``_assert_batch_rng_contract`` to avoid for the other harness.

    Returns ``(python_arm, kernel_arm)``.
    """
    grid = grid or Grid.from_dimensions(
        ny=int(state.cell_y.max()) + 1, nx=int(state.cell_x.max()) + 1
    )
    sorted_indices, boundaries, n_cells = _cell_groups(state, grid)
    non_empty = [c for c in range(n_cells) if boundaries[c + 1] > boundaries[c]]
    assert len(non_empty) >= min_non_empty_cells, (
        f"fixture has {len(non_empty)} non-empty cells, need >= {min_non_empty_cells} so "
        "every school in the fixture is actually visited"
    )
    acc = access_matrix if access_matrix is not None else M._DUMMY_ACCESS

    # --- Arm A: the reviewed Python reference, driven exactly as mortality() drives it ---
    ref = _prepare_arm(
        state, config, resources, n_subdt=n_subdt, step=step, diet_tracking=diet_tracking
    )
    rng_ref = np.random.default_rng(seed)
    ref_rng_states = []
    counter = _KernelCounter(M._mortality_in_cell_numba)
    prev_has_numba = M._HAS_NUMBA
    prev_kernel = M._mortality_in_cell_numba
    M._mortality_in_cell_numba = counter
    M._HAS_NUMBA = False
    try:
        for _ in range(sub_steps):
            for cell in non_empty:
                lo, hi = int(boundaries[cell]), int(boundaries[cell + 1])
                M._mortality_in_cell(
                    sorted_indices[lo:hi],
                    ref.state,
                    config,
                    ref.resources,
                    cell // grid.nx,
                    cell % grid.nx,
                    rng_ref,
                    n_subdt,
                    acc,
                    has_access,
                    use_stage_access,
                    np.zeros(len(ref.state), dtype=np.int32),
                    np.zeros(len(ref.state), dtype=np.int32),
                    inst_abd=ref.inst_abd,
                    step=step,
                    rsc_size_min=ref.rsc_arrays[0],
                    rsc_size_max=ref.rsc_arrays[1],
                    rsc_tl=ref.rsc_arrays[2],
                    rsc_access_rows=ref.rsc_arrays[3],
                    n_rsc=ref.rsc_arrays[4],
                    grid_nx=grid.nx,
                    eff_starv=ref.eff_starv,
                    eff_additional=ref.eff_additional,
                    eff_fishing=ref.eff_fishing,
                    fishing_discard=ref.fishing_discard,
                    ctx=ref.ctx,
                    egg_retained=ref.state.egg_retained,
                    cap_fish=ref.cap_fish,
                    raw_preyed=ref.raw_preyed,
                    eta_school=ref.eta_school,
                    eff_foraging=ref.eff_foraging,
                )
            ref_rng_states.append(rng_ref.bit_generator.state)
    finally:
        M._HAS_NUMBA = prev_has_numba
        M._mortality_in_cell_numba = prev_kernel
    assert counter.calls == 0, (
        f"the reference arm entered _mortality_in_cell_numba {counter.calls}x despite "
        "_HAS_NUMBA=False -- this comparison would be kernel-vs-kernel"
    )

    # --- Arm B: the per-cell kernel, on the reference's own draws ---
    cand = _prepare_arm(
        state, config, resources, n_subdt=n_subdt, step=step, diet_tracking=diet_tracking
    )
    n = len(cand.state)
    rsc_bio = cand.resources.biomass if cand.resources is not None else M._DUMMY_RSC_2D
    rng_cand = np.random.default_rng(seed)
    cand_rng_states = []
    calls = 0
    for _ in range(sub_steps):
        seq_bufs, cause_orders_buf = M._pre_generate_cell_rng(rng_cand, boundaries, n_cells, config)
        cand_rng_states.append(rng_cand.bit_generator.state)
        for cell in non_empty:
            lo, hi = int(boundaries[cell]), int(boundaries[cell + 1])
            M._mortality_in_cell_numba(
                sorted_indices[lo:hi],
                seq_bufs[0][lo:hi],
                seq_bufs[1][lo:hi],
                seq_bufs[2][lo:hi],
                seq_bufs[3][lo:hi],
                seq_bufs[4][lo:hi],
                cause_orders_buf[lo:hi],
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
                rsc_bio,
                cand.rsc_arrays[0],
                cand.rsc_arrays[1],
                cand.rsc_arrays[2],
                cand.rsc_arrays[3],
                cand.rsc_arrays[4],
                config.n_species,
                cell,
                cand.ctx.tl_weighted_sum,
                True,
                cand.ctx.diet_matrix if cand.ctx.diet_matrix is not None else M._DUMMY_DIET,
                cand.ctx.diet_matrix is not None,
                cand.state.egg_retained,
                config.bioen_enabled,
                cand.cap_fish if cand.cap_fish is not None else M._DUMMY_RSC_1D,
                cand.raw_preyed if cand.raw_preyed is not None else M._DUMMY_RSC_1D,
                cand.state.e_net,
                cand.state.gonad_weight,
                cand.eta_school,
                cand.state.is_background,
                cand.eff_foraging,
            )
            calls += 1

    assert ref_rng_states == cand_rng_states, (
        "_pre_generate_cell_rng no longer reproduces _mortality_in_cell's own per-cell "
        "draws: the two generators are in different states after a sub-step, so the two "
        "arms saw different school sequences and/or cause orders. Fix the replication "
        "before reading any field difference below as a bioen logic defect."
    )
    return (
        _arm_result("python", ref, state, 0, bioen=True),
        _arm_result("numba-cell-kernel", cand, state, calls, bioen=True),
    )


def test_the_gate_fixture_makes_every_behaviour_observable():
    """Each plan Task 2 Step 2 requirement, asserted on the fixture itself.

    "A gate that passes because the code under test is never reached" is the failure this
    branch has produced five times. These assertions are on the FIXTURE, so they hold
    whatever the kernel does, and they fail loudly if a later edit to ``base_config`` /
    ``base_state`` quietly removes one of the conditions.
    """
    config, state = bioen_gate_fixture()
    n_subdt = 2
    st = state.replace(feeding_stage=compute_feeding_stages(state, config))

    # (a) the per-fish cap BINDS for both predators: cap * N < the prey available to them
    setup = _prepare_arm(state, config, None, n_subdt=n_subdt, step=0, diet_tracking=True)
    assert setup.cap_fish is not None
    for p_idx, prey in ((0, (1, 2, 7)), (3, (4, 5, 6))):
        max_eatable = float(setup.cap_fish[p_idx] * st.abundance[p_idx])
        available = float(sum(st.abundance[q] * st.weight[q] for q in prey))
        assert 0.0 < max_eatable < available, (
            f"school {p_idx}: cap {max_eatable} does not bind against {available} of prey "
            "-- behaviour 1 would be untested"
        )
        standard = float(
            st.abundance[p_idx]
            * st.weight[p_idx]
            * config.ingestion_rate[st.species_id[p_idx]]
            / (config.n_dt_per_year * n_subdt)
        )
        assert abs(max_eatable - standard) / standard > 0.1, (
            "the bioen cap coincides with the bioen-OFF cap, so behaviour 1 is invisible"
        )

    # (b) additional AND fishing are both live, with a non-zero discard rate, on the
    #     schools that eat
    _, eff_a, eff_f, eff_for = _bioen_gate_rates(config, state, n_subdt)
    for p_idx in (0, 3):
        assert eff_a[p_idx] > 0 and eff_f[p_idx] > 0
        assert config.fishing_discard_rate[st.species_id[p_idx]] > 0

    # (c) FORAGING is not inert
    assert config.bioen_k_for is not None and float(config.bioen_k_for[0]) == GATE_K_FOR
    assert eff_for[0] > 0.0

    # (d) the four starvation cases
    eta = M._precompute_bioen_eta(st, config)
    assert st.e_net[0] < 0 and st.gonad_weight[0] < eta[0] * abs(st.e_net[0]) / n_subdt, (
        "school 0 must be the gonad-does-NOT-cover case"
    )
    assert st.e_net[2] < 0 and st.gonad_weight[2] >= eta[2] * abs(st.e_net[2]) / n_subdt, (
        "school 2 must be the gonad-COVERS case"
    )
    assert st.age_dt[7] == st.first_feeding_age_dt[7] and st.e_net[7] < 0, (
        "school 7 must sit exactly on the strict ageDt > firstFeedingAgeDt boundary"
    )
    toll_6 = abs(st.e_net[6]) / n_subdt / st.weight[6]
    assert toll_6 > st.abundance[6], (
        "school 6 must have deficit/weight > inst_abd so the absent death-count clamp is "
        f"pinned (toll {toll_6} vs abundance {st.abundance[6]})"
    )

    # (e) exactly one background school, and it is eaten by school 3
    assert int(st.is_background.sum()) == 1 and bool(st.is_background[5])
    ratio = float(st.length[3] / st.length[5])
    sp3, stage3 = int(st.species_id[3]), int(st.feeding_stage[3])
    assert config.size_ratio_min[sp3, stage3] <= ratio < config.size_ratio_max[sp3, stage3], (
        "the background school is not inside school 3's prey size window, so the "
        "rescale-skips-background guard is never exercised"
    )

    # (f) n_subdt >= 2 (the deficit is |e_net| / n_subdt) and the fixture is bit-exact-safe
    assert n_subdt >= 2
    assert_exp_library_agreement(eff_a, eff_f, eff_for, label="bioen-gate-fixture")


def test_the_bioen_gate_fixture_actually_exercises_bioen():
    """Negative control: with ``bioen_enabled`` off, this fixture must answer differently."""
    config, state = bioen_gate_fixture()
    assert_bioen_changes_the_answer(state, config, seed=7, n_subdt=2, label="bioen-gate")


def test_cell_kernel_matches_the_python_reference_under_bioen():
    """The Task 2 gate: all five behaviours, per-cell kernel vs the reviewed reference."""
    config, state = bioen_gate_fixture()
    assert_exp_library_agreement(*_bioen_gate_rates(config, state, 2), label="cell/bioen")
    python_arm, kernel_arm = run_cell_bioen_both_paths(state, config, seed=7, n_subdt=2)
    assert_arms_equal(
        python_arm,
        kernel_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed", "preyed_biomass"),
        witness_causes=(
            MortalityCause.PREDATION,
            MortalityCause.STARVATION,
            MortalityCause.ADDITIONAL,
            MortalityCause.FISHING,
            MortalityCause.DISCARDS,
            MortalityCause.FORAGING,
        ),
        label="cell/bioen",
    )


@pytest.mark.parametrize("seed", [0, 13, 20250903])
def test_cell_kernel_matches_the_python_reference_under_bioen_across_seeds(seed):
    """Different seeds interleave the five causes differently; the arms must track it."""
    config, state = bioen_gate_fixture()
    python_arm, kernel_arm = run_cell_bioen_both_paths(
        state, config, seed=seed, n_subdt=3, sub_steps=2
    )
    assert_arms_equal(
        python_arm,
        kernel_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed"),
        witness_causes=(MortalityCause.STARVATION, MortalityCause.FORAGING),
        label=f"cell/bioen/seed={seed}",
    )


def test_cell_kernel_matches_the_python_reference_under_bioen_with_resources():
    """Resource depletion and diet tracking on the bioen path.

    Adds a second ingestion source, so ``raw_preyed`` and the TL accumulator have a
    contribution the schools-only fixture cannot produce.
    """
    grid = Grid.from_dimensions(ny=1, nx=2)
    rsc_cfg, resources = resource_config_and_state(grid)
    config = base_config(**{**_bioen_overrides(), **rsc_cfg})
    state = bioen_gate_state()
    python_arm, kernel_arm = run_cell_bioen_both_paths(
        state, config, seed=3, n_subdt=2, resources=resources, grid=grid
    )
    assert_arms_equal(
        python_arm,
        kernel_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed", "diet_matrix"),
        witness_causes=(MortalityCause.PREDATION, MortalityCause.FORAGING),
        label="cell/bioen/resources",
    )
    assert python_arm.resource_biomass is not None
    assert float(python_arm.resource_biomass.sum()) < 12.0 * grid.ny * grid.nx


def test_bioen_behaviours_are_each_visible_in_the_gate_result():
    """The five behaviours, read off the reference arm's OUTPUT (not just its inputs).

    ``assert_arms_equal``'s witness check is a floor -- ``np.any(arr != 0.0)`` is happy
    with one non-zero element. These are the sharp statements, and each names the
    behaviour it would lose.
    """
    config, state = bioen_gate_fixture()
    python_arm, kernel_arm = run_cell_bioen_both_paths(state, config, seed=7, n_subdt=2)

    for arm in (python_arm, kernel_arm):
        raw = arm.raw_preyed
        pb = arm.state.preyed_biomass
        assert raw is not None

        # Behaviour 5 + behaviour 2 together: the raw ingestion total and the
        # survivor-rescaled one must have come apart, and raw must be the LARGER.
        movers = [p for p in (0, 3) if raw[p] > 0.0]
        assert movers, f"{arm.label}: neither predator ate -- behaviour 5 is untested"
        assert any(raw[p] > pb[p] for p in movers), (
            f"{arm.label}: raw_preyed never exceeded preyed_biomass, so the survivor "
            "rescale of the ingestion accumulator (behaviour 2) never bit"
        )

        # Behaviour 2 at a NON-predation death: school 7 is skipped by starvation
        # (strict boundary) but still dies to additional / fishing / foraging, so its
        # seeded preyed_biomass must have been rescaled down.
        assert pb[7] < 6.0, (
            f"{arm.label}: school 7's seeded preyed_biomass was not rescaled -- the "
            "rescale is missing from at least one of ADDITIONAL / FISHING / FORAGING"
        )
        assert arm.state.e_net[7] > -1.5, (
            f"{arm.label}: school 7's e_net was not rescaled towards zero"
        )

        # Behaviour 2's background skip: school 5 dies to school 3's predation, and must
        # come back with its seeded values EXACTLY.
        assert arm.state.n_dead[5, int(MortalityCause.PREDATION)] > 0.0, (
            f"{arm.label}: the background school was never eaten, so the skip is untested"
        )
        assert pb[5] == 5.0 and arm.state.e_net[5] == -0.7, (
            f"{arm.label}: the background school was rescaled ({pb[5]}, "
            f"{arm.state.e_net[5]}) -- AbstractSchool.incrementNdead does not rescale"
        )

        # Behaviour 3: the gonad buffer, both branches, and the strict boundary.
        assert arm.state.gonad_weight[0] == 0.0, (
            f"{arm.label}: school 0's gonad was not flushed by the insufficient branch"
        )
        assert 0.0 < arm.state.gonad_weight[2] < 1.0, (
            f"{arm.label}: school 2's gonad did not absorb its deficit "
            f"({arm.state.gonad_weight[2]})"
        )
        assert arm.state.n_dead[2, int(MortalityCause.STARVATION)] == 0.0, (
            f"{arm.label}: school 2's gonad covered its deficit, so nobody should starve"
        )
        assert arm.state.n_dead[7, int(MortalityCause.STARVATION)] == 0.0, (
            f"{arm.label}: school 7 sits at ageDt == firstFeedingAgeDt and the bioen "
            "eligibility is STRICT -- it must not starve"
        )
        assert arm.state.n_dead[0, int(MortalityCause.STARVATION)] > 0.0

        # Behaviour 3's no-clamp: school 6's toll exceeds its abundance, so inst_abd goes
        # negative and the survivor FACTOR (not the death count) is what gets clamped.
        assert arm.state.n_dead[6, int(MortalityCause.STARVATION)] > arm.input_state.abundance[6]
        assert arm.inst_abd[6] < 0.0, (
            f"{arm.label}: school 6's inst_abd is {arm.inst_abd[6]}, so the absent "
            "death-count clamp is not pinned"
        )
        assert pb[6] == 0.0 and arm.state.e_net[6] == 0.0, (
            f"{arm.label}: the survivor factor was not clamped to 0 for school 6"
        )

        # Behaviour 4: FORAGING actually killed somebody.
        assert float(arm.state.n_dead[:, int(MortalityCause.FORAGING)].sum()) > 0.0
        assert float(arm.state.n_dead[5, int(MortalityCause.FORAGING)]) == 0.0, (
            f"{arm.label}: the background school foraged to death"
        )

        # Behaviour 1: ingestion is capped by the per-fish cap, not by biomass * Imax.
        setup_cap = _prepare_arm(state, config, None, n_subdt=2, step=0, diet_tracking=True)
        assert setup_cap.cap_fish is not None
        bioen_cap = float(setup_cap.cap_fish[0] * arm.input_state.abundance[0])
        standard_cap = float(
            arm.input_state.abundance[0]
            * arm.input_state.weight[0]
            * config.ingestion_rate[0]
            / (config.n_dt_per_year * 2)
        )
        assert raw[0] <= 2.0 * bioen_cap < standard_cap, (
            f"{arm.label}: raw ingestion {raw[0]} is not consistent with the per-fish cap "
            f"({bioen_cap} per sub-step, 2 sub-steps) versus the bioen-OFF cap "
            f"{standard_cap} -- behaviour 1 is not what limited it"
        )


def test_a_zero_abundance_school_gets_no_gonad_or_enet_write():
    """Behaviour 3's `inst_abd <= 0` guard, placed BEFORE any gonad/e_net write.

    Added after a revert probe found the guard unpinned by the main gate fixture, and the
    reason is worth recording: with the survivor rescale correct, a school whose
    `inst_abd` is driven non-positive by its own starvation toll has its `e_net` clamped
    to 0 by the SAME `_consume` call, so a later visit exits on the `e_net >= 0` test
    instead and the guard never decides anything. Removing the guard was therefore
    invisible -- 34 passed.

    A school that starts at abundance 0 makes it decidable: `_consume` never runs for it
    (`before > 0.0` is false), so its seeded `e_net = -0.3` and `gonad_weight = 0.4`
    survive, and only the guard stands between them and the gonad-absorbs branch, which
    would write `gonad 0.4 -> 0.25` and `e_net -0.3 -> -0.15`. Zero-abundance schools are
    an ordinary engine state, not a contrivance -- `mortality()` clamps abundance at 0 and
    `n_dead` routinely takes a school there.
    """
    config, state = bioen_gate_fixture()
    abundance = state.abundance.copy()
    abundance[4] = 0.0
    gonad = state.gonad_weight.copy()
    gonad[4] = 0.4
    state = state.replace(abundance=abundance, gonad_weight=gonad)
    assert float(state.e_net[4]) == -0.3

    python_arm, kernel_arm = run_cell_bioen_both_paths(state, config, seed=7, n_subdt=2)
    assert_arms_equal(
        python_arm,
        kernel_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed"),
        witness_causes=(MortalityCause.STARVATION, MortalityCause.FORAGING),
        label="cell/bioen/zero-abundance",
    )
    for arm in (python_arm, kernel_arm):
        assert arm.inst_abd[4] == 0.0, f"{arm.label}: school 4 should still be at zero"
        assert arm.state.gonad_weight[4] == 0.4, (
            f"{arm.label}: school 4's gonad moved to {arm.state.gonad_weight[4]} -- the "
            "`inst_abd <= 0` guard is missing or sits AFTER the gonad write"
        )
        assert arm.state.e_net[4] == -0.3, (
            f"{arm.label}: school 4's e_net moved to {arm.state.e_net[4]} -- the "
            "`inst_abd <= 0` guard is missing or sits AFTER the repayment"
        )
        assert arm.state.n_dead[4].sum() == 0.0, (
            f"{arm.label}: a school with zero abundance recorded deaths"
        )


def test_bioen_off_keeps_exactly_four_causes_and_its_rng_consumption():
    """Behaviour 4's other half -- what protects ``tests/test_engine_parity.py``.

    The batch kernels' fifth permutation and fifth cause code are inside ``if bioen:``.
    Measured at runtime on the same fixture with the flag both ways: 4 permutations and a
    4-wide order when off, 5 and 5 when on. A regression that hoisted either draw out of
    the branch would shift every bioen-OFF stream and move the committed fixed-seed
    baselines.
    """
    off_config, off_state = base_config(), base_state()
    assert measure_batch_kernel_rng_shape(off_state, off_config, n_subdt=2) == (4, 4)
    assert M._get_mortality_causes(off_config) == [0, 1, 2, 3]

    on_config, on_state = bioen_gate_fixture()
    assert measure_batch_kernel_rng_shape(on_state, on_config, n_subdt=2) == (5, 5)
    assert M._get_mortality_causes(on_config) == [0, 1, 2, 3, int(MortalityCause.FORAGING)]


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_kernel_matches_the_per_cell_kernel_under_bioen(parallel):
    """Cross-kernel agreement under bioen -- the ONLY check that all three inlined loops
    received the same edits.

    Parametrised over ``parallel`` because ``mortality()`` dispatches to
    ``_mortality_all_cells_parallel`` in production, and because Task 1's runtime
    RNG-shape probe can only instrument the SEQUENTIAL kernel (a ``prange`` iteration may
    run on a worker thread whose PRNG state the main thread cannot read). At
    ``parallel=True`` this equality IS the three-way draw-width agreement check.

    It cannot see a behaviour that is wrong in all three loops -- both arms are kernels.
    ``test_cell_kernel_matches_the_python_reference_under_bioen`` is what pins them to the
    reviewed reference.
    """
    grid = Grid.from_dimensions(ny=1, nx=2)
    config, state = bioen_gate_fixture()
    per_cell_arm, batch_arm = run_batch_both_paths(
        state, config, seed=4242, parallel=parallel, grid=grid, n_subdt=2
    )
    assert per_cell_arm.kernel_calls == 2, "both cells must have been visited"
    assert_arms_equal(
        per_cell_arm,
        batch_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed", "preyed_biomass"),
        witness_causes=(
            MortalityCause.PREDATION,
            MortalityCause.STARVATION,
            MortalityCause.ADDITIONAL,
            MortalityCause.FISHING,
            MortalityCause.DISCARDS,
            MortalityCause.FORAGING,
        ),
        label=f"batch/bioen/parallel={parallel}",
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_kernel_matches_the_per_cell_kernel_under_bioen_with_resources(parallel):
    grid = Grid.from_dimensions(ny=1, nx=2)
    rsc_cfg, resources = resource_config_and_state(grid)
    config = base_config(**{**_bioen_overrides(), **rsc_cfg})
    state = bioen_gate_state()
    per_cell_arm, batch_arm = run_batch_both_paths(
        state, config, seed=4242, parallel=parallel, grid=grid, n_subdt=2, resources=resources
    )
    assert_arms_equal(
        per_cell_arm,
        batch_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed", "diet_matrix"),
        witness_causes=(MortalityCause.FORAGING,),
        label=f"batch/bioen/resources/parallel={parallel}",
    )
    assert per_cell_arm.resource_biomass is not None
    assert float(per_cell_arm.resource_biomass.sum()) < 12.0 * grid.ny * grid.nx


def test_genetic_foraging_is_precomputed_not_dropped():
    """The genetic ``ForagingMortality`` variant reaches the kernel too.

    ``_apply_foraging_for_school`` switches on a four-way predicate
    (``foraging_k1_for`` / ``foraging_k2_for`` / ``foraging_I_max`` /
    ``state.imax_trait``). Rather than inline ``k1 * exp(k2 * (imax - I_max))`` in the
    kernel -- where numba's libm ``exp`` would disagree with the reference's numpy ``exp``
    in the last ULP -- ``_precompute_foraging_rates`` evaluates it in NumPy for BOTH
    paths. So the branch is supported rather than dropped or shunted back to Python; this
    test is what stops that claim being untested (plan Task 2 Step 4, behaviour 4).
    """
    config, state = bioen_gate_fixture()
    # Chosen by search so every resulting rate is one on which numba's libm ``exp`` and
    # numpy's agree -- the EXP LIBRARY HAZARD bites here exactly as it does elsewhere (the
    # first parameter set tried put school 4 on 0.0073541408..., where the two differ in
    # the last ULP and ``1 - exp(-D)`` amplifies that to ~1.5e-14).
    config.foraging_k1_for = np.array([0.71, 0.32, 0.80])
    config.foraging_k2_for = np.array([0.32, 0.22, 0.27])
    config.foraging_I_max = np.array([3.5, 3.5, 3.5])
    state = state.replace(imax_trait=np.array([2.8, 3.0, 3.9, 3.8, 3.8, 3.4, 4.4, 4.4]))

    st = state.replace(feeding_stage=compute_feeding_stages(state, config))
    eff_for = M._precompute_foraging_rates(st, config, 2)
    # The genetic branch was taken, and it is not the constant one.
    constant = np.full(len(st), (GATE_K_FOR / config.n_dt_per_year) / 2)
    assert not np.allclose(eff_for[~st.is_background], constant[~st.is_background])
    assert float(eff_for[0]) > 0.0
    assert_exp_library_agreement(eff_for, label="genetic-foraging")

    python_arm, kernel_arm = run_cell_bioen_both_paths(state, config, seed=7, n_subdt=2)
    assert_arms_equal(
        python_arm,
        kernel_arm,
        witness_fields=("n_dead", "e_net", "gonad_weight", "raw_preyed"),
        witness_causes=(MortalityCause.FORAGING,),
        label="cell/bioen/genetic-foraging",
    )


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
