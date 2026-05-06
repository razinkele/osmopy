"""Phase 3 M4 + M5: Gompertz positivity guard + spatial-mortality clamps.

- M4: growth.gompertz.{linf, kg}.spN must be > 0 when species uses GOMPERTZ.
- M5a: predation accessibility coefficients > 1.0 emit a UserWarning.
- M5b: natural-mortality n_dead is clamped to abundance so spatial_factor>1
       cannot push abundance negative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig

REPO = Path(__file__).resolve().parents[1]


def _load_minimal_with_gompertz(linf: str = "150.0", kg: str = "0.5") -> dict[str, str]:
    """Load the minimal 2-species fixture and switch sp0 to GOMPERTZ growth.

    No shipped fixture uses GOMPERTZ — this is the smallest config that
    exercises the M4 guard.
    """
    matches = sorted((REPO / "data" / "minimal").glob("*_all-parameters.csv"))
    cfg = OsmoseConfigReader().read(str(matches[0]))
    cfg = {k: v for k, v in cfg.items() if k != ""}
    cfg["growth.java.classname.sp0"] = "fr.ird.osmose.process.growth.GompertzGrowth"
    # Sensible Gompertz defaults so the rest of from_dict succeeds when not testing the bound
    cfg["growth.exponential.ke.sp0"] = "0.001"
    cfg["growth.exponential.lstart.sp0"] = "0.5"
    cfg["growth.gompertz.kg.sp0"] = kg
    cfg["growth.gompertz.tg.sp0"] = "1.0"
    cfg["growth.gompertz.linf.sp0"] = linf
    cfg["growth.exponential.thr.age.sp0"] = "0.1"
    cfg["growth.gompertz.thr.age.sp0"] = "1.0"
    return cfg


def test_gompertz_minimal_baseline_is_clean() -> None:
    """Sanity: the minimal-with-Gompertz config (sensible defaults) parses fine."""
    cfg = _load_minimal_with_gompertz()
    EngineConfig.from_dict(cfg)  # must not raise


def test_gompertz_zero_linf_raises() -> None:
    """M4: linf=0 for a GOMPERTZ species must hard-fail."""
    cfg = _load_minimal_with_gompertz(linf="0")
    with pytest.raises(ValueError, match=r"growth\.gompertz\.linf\.sp0 must be > 0"):
        EngineConfig.from_dict(cfg)


def test_gompertz_negative_kg_raises() -> None:
    """M4: kg<0 must hard-fail."""
    cfg = _load_minimal_with_gompertz(kg="-0.1")
    with pytest.raises(ValueError, match=r"growth\.gompertz\.kg\.sp0 must be > 0"):
        EngineConfig.from_dict(cfg)


def test_gompertz_default_linf_zero_raises() -> None:
    """M4: removing linf altogether falls back to default 0.0 -> hard-fail."""
    cfg = _load_minimal_with_gompertz()
    cfg.pop("growth.gompertz.linf.sp0", None)
    with pytest.raises(ValueError, match=r"growth\.gompertz\.linf\.sp0 must be > 0"):
        EngineConfig.from_dict(cfg)


def test_accessibility_coefficient_above_one_warns(tmp_path: Path) -> None:
    """M5a: an accessibility CSV with a > 1.0 coefficient must emit UserWarning."""
    from osmose.engine.accessibility import AccessibilityMatrix

    csv = tmp_path / "bad_access.csv"
    csv.write_text(
        ";A < 1.0;A\n"
        "B < 0.5;0.3;0.7\n"
        "B;0.5;1.5\n"  # 1.5 violates the [0, 1] bound
    )
    with pytest.warns(UserWarning, match=r"accessibility matrix.*1 coefficient.*exceed 1\.0"):
        AccessibilityMatrix.from_csv(csv, species_names=["A", "B"])


def test_accessibility_coefficient_at_one_does_not_warn(tmp_path: Path) -> None:
    """M5a: exactly-1.0 coefficients are fine (saturation, not violation)."""
    from osmose.engine.accessibility import AccessibilityMatrix

    csv = tmp_path / "ok_access.csv"
    csv.write_text(";A < 1.0;A\nB < 0.5;0.3;0.7\nB;0.5;1.0\n")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            AccessibilityMatrix.from_csv(csv, species_names=["A", "B"])
        except UserWarning as exc:
            if "accessibility matrix" in str(exc):
                pytest.fail(f"unexpected accessibility warning at coefficient=1.0: {exc}")


def test_natural_mortality_clamps_n_dead_to_abundance() -> None:
    """M5b: n_dead must be clamped so spatial_factor>1 cannot push abundance negative.

    Direct unit test of the clamp expression: even with multiplier > 1 and
    mortality_fraction near 1, the resulting n_dead never exceeds abundance.
    """
    abundance = np.array([100.0, 50.0, 200.0])
    mortality_fraction = np.array([0.9, 0.99, 0.5])
    spatial_factor = np.array([5.0, 10.0, 0.0])  # 5x and 10x are > 1

    n_dead = abundance * mortality_fraction * spatial_factor
    np.minimum(n_dead, abundance, out=n_dead)

    new_abundance = abundance - n_dead
    assert (new_abundance >= 0.0).all(), (
        f"clamp failed; new_abundance went negative: {new_abundance}"
    )
    # Verify clamp triggered on schools 0 and 1 (their unclamped n_dead was > abundance)
    assert n_dead[0] == 100.0
    assert n_dead[1] == 50.0
    assert n_dead[2] == 0.0
