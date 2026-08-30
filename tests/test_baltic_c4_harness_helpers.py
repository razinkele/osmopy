"""C4 harness helper tests (spec 2026-08-30, Task 3): pure functions only --
`arm_overlays`, `assert_no_salinity_constant`, `assert_arm_frame_count`, `ramp_ordering_ok`,
`salinity_load_through_ok` -- on synthetic fixtures (task-3-brief.md, verbatim; B2's proven
test set adapted for the salinity field's NaN-land convention). `run_c4` itself (the 5-arm x
5-seed x 50yr run) is NOT invoked here -- see the module docstring of
scripts/baltic_c4_salinity_ab.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "baltic_c4_salinity_ab",
    Path(__file__).resolve().parent.parent / "scripts" / "baltic_c4_salinity_ab.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


# Task 2's synthetic salinity field (tests/test_build_baltic_c4_forcing.py::_sal_field),
# reused verbatim so the ramp-ordering / load-through checks are exercised against the same
# NaN-land convention the offset/ramp functions were proven against.
def _sal_field():
    sal = np.full((24, 4, 4), 20.0)
    sal[:, 0, 0] = 1.5  # near-floor wet cell
    sal[:, 3, 3] = np.nan  # land
    wet = np.ones((4, 4), dtype=bool)
    wet[3, 3] = False
    return sal, wet


# ---------------------------------------------------------------------------
# module-level constants (spec binding facts)
# ---------------------------------------------------------------------------


def test_arms_tuple_matches_spec_order():
    assert m.ARMS == ("baseline", "zero", "ds_m1", "ds_m2", "ds_m3")


def test_seeds_tuple_matches_spec():
    assert m.SEEDS == (42, 123, 7, 999, 2024)


def test_zero_arm_def_is_imported_from_the_builder():
    assert m.ZERO_ARM_DEF is m._c4.ZERO_ARM_DEF
    assert m.ZERO_ARM_DEF == {"name": "zero", "dS_PSU": 0.0}


# ---------------------------------------------------------------------------
# arm_overlays (spec Design §3 / binding facts): overlays ONLY
# movement.salinity.field.file at the arm's absolute path -- nothing else.
# ---------------------------------------------------------------------------


def test_arm_overlays_baseline_has_no_salinity_key_override():
    artifacts = {"sal_nc": Path("/tmp/unused/salinity_offset.nc")}
    overlays = m.arm_overlays("baseline", artifacts)
    assert overlays == {}


def test_arm_overlays_arm_carries_only_the_file_key_absolute_path():
    sal_nc = Path("/tmp/c4_harness_x/ds_m1/salinity_offset.nc")
    artifacts = {"sal_nc": sal_nc}
    overlays = m.arm_overlays("ds_m1", artifacts)
    assert overlays == {"movement.salinity.field.file": str(sal_nc)}
    assert sal_nc.is_absolute()


def test_arm_overlays_zero_arm_also_carries_only_the_file_key():
    sal_nc = Path("/tmp/c4_harness_x/zero/salinity_offset.nc")
    artifacts = {"sal_nc": sal_nc}
    overlays = m.arm_overlays("zero", artifacts)
    assert overlays == {"movement.salinity.field.file": str(sal_nc)}


# ---------------------------------------------------------------------------
# assert_no_salinity_constant (spec decision 4, gate order item 2): the loader prefers
# movement.salinity.field.constant over .file -- a stray key would silently discard the
# arm's only lever.
# ---------------------------------------------------------------------------


def test_assert_no_salinity_constant_raises_on_poisoned_cfg():
    cfg = {"movement.salinity.field.file": "/tmp/arm.nc", "movement.salinity.field.constant": "5.0"}
    with pytest.raises(ValueError, match="constant"):
        m.assert_no_salinity_constant(cfg, "ds_m1")


def test_assert_no_salinity_constant_passes_on_clean_cfg():
    cfg = {"movement.salinity.field.file": "/tmp/arm.nc"}
    m.assert_no_salinity_constant(cfg, "ds_m1")  # must not raise


def test_assert_no_salinity_constant_passes_when_key_present_but_empty():
    # An empty-string value (the reader's own "absent" convention) must not trip the guard.
    cfg = {"movement.salinity.field.file": "/tmp/arm.nc", "movement.salinity.field.constant": ""}
    m.assert_no_salinity_constant(cfg, "ds_m1")  # must not raise


# ---------------------------------------------------------------------------
# assert_arm_frame_count (spec decision 4, gate order item 3): the salinity loader has NO
# frame validation of its own -- silent step % frames wrap.
# ---------------------------------------------------------------------------


def test_assert_arm_frame_count_raises_on_23_frame_synthetic():
    field = np.zeros((23, 4, 4))
    with pytest.raises(ValueError, match="24"):
        m.assert_arm_frame_count(field, "ds_m2")


def test_assert_arm_frame_count_passes_on_24_frame_synthetic():
    field = np.zeros((24, 4, 4))
    m.assert_arm_frame_count(field, "ds_m2")  # must not raise


# ---------------------------------------------------------------------------
# ramp_ordering_ok (spec decision 4, gate order item 5): ramp_w(arm) <= ramp_w(base) per wet
# cell for negative dS, equal for zero dS.
# ---------------------------------------------------------------------------


def test_ramp_ordering_ok_negative_delta_everywhere_le():
    sal, wet = _sal_field()
    arm_sal = m._c4.offset_salinity(sal, wet, -2.0)
    assert m.ramp_ordering_ok(arm_sal, sal, wet, dS=-2.0) is True


def test_ramp_ordering_ok_zero_delta_everywhere_equal():
    sal, wet = _sal_field()
    arm_sal = m._c4.offset_salinity(sal, wet, 0.0)
    assert m.ramp_ordering_ok(arm_sal, sal, wet, dS=0.0) is True


def test_ramp_ordering_ok_detects_violation():
    # dS says "negative" but the arm field is actually higher everywhere wet -- a real
    # wiring bug (e.g. the wrong file loaded) must be caught, not rubber-stamped.
    sal, wet = _sal_field()
    arm_sal = m._c4.offset_salinity(sal, wet, 2.0)
    assert m.ramp_ordering_ok(arm_sal, sal, wet, dS=-2.0) is False


def test_ramp_ordering_ok_ignores_land_cells():
    sal, wet = _sal_field()
    arm_sal = m._c4.offset_salinity(sal, wet, -2.0)
    assert np.isnan(arm_sal[0, 3, 3])
    assert np.isnan(sal[0, 3, 3])
    assert m.ramp_ordering_ok(arm_sal, sal, wet, dS=-2.0) is True


# ---------------------------------------------------------------------------
# salinity_load_through_ok (spec decision 4, gate order item 4, three-way load-through) --
# B2's strengthened three-way assert adapted verbatim: engine == disk == recomputed-expected
# via the builder's own offset_salinity. The no-op-write pathological case is the actual
# detector this check exists for -- a written file byte-identical to production despite
# dS != 0 would pass a plain engine==disk check trivially.
# ---------------------------------------------------------------------------


def test_salinity_load_through_ok_correctly_applied_offset_passes():
    sal, wet = _sal_field()
    dS = -2.0
    disk_sal = m._c4.offset_salinity(sal, wet, dS)  # correctly written
    engine_sal = disk_sal  # engine faithfully loaded the (correct) file
    assert m.salinity_load_through_ok(engine_sal, disk_sal, sal, wet, dS) is True


def test_salinity_load_through_ok_zero_delta_passes():
    sal, wet = _sal_field()
    disk_sal = m._c4.offset_salinity(sal, wet, 0.0)  # value-identical copy (the zero arm)
    engine_sal = disk_sal
    assert m.salinity_load_through_ok(engine_sal, disk_sal, sal, wet, 0.0) is True


def test_salinity_load_through_ok_silent_no_op_write_fails():
    # The pathological case (task-3-brief.md): dS != 0, but the "written" file is
    # byte-identical to the untouched production field, as if write_arm_dir/offset_salinity
    # silently did nothing. The engine loads that (buggy) file faithfully, so a plain
    # engine==disk check would have PASSED here -- the strengthened three-way check must FAIL.
    sal, wet = _sal_field()
    dS = -2.0
    disk_sal = sal.copy()  # no-op: never actually offset despite dS != 0
    engine_sal = disk_sal
    assert m.salinity_load_through_ok(engine_sal, disk_sal, sal, wet, dS) is False


def test_salinity_load_through_ok_engine_disk_mismatch_still_fails():
    # The original failure mode (a genuine loader bug): disk is correct, but the engine held
    # something else entirely.
    sal, wet = _sal_field()
    dS = -2.0
    disk_sal = m._c4.offset_salinity(sal, wet, dS)
    engine_sal = sal.copy()  # engine never actually loaded the offset file
    assert m.salinity_load_through_ok(engine_sal, disk_sal, sal, wet, dS) is False
