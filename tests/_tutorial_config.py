"""Canonical source of truth for the 30-min tutorial (Baltic substrate).

The tutorial uses the data/baltic/ calibrated config and highlights total cod,
sprat, and stickleback for the trophic cascade narrative.

The reader's workflow: copy baltic/ to a workdir, override nyear for speed,
run the engine, plot biomass for the 3 highlighted species, perturb
cod-on-sprat accessibility, re-run, observe the cascade.

Cascade mechanics (Baltic-substrate finding):
  Dropping cod's accessibility to sprat (cod_west 0.4, cod_east 0.5 -> 0.05) reduces
  predation pressure on stickleback (cod eats stickleback as well as sprat; with less
  sprat food, cod is less abundant -> fewer cod -> stickleback UP). The sprat signal is
  weak because cod biomass is small relative to the sprat population. This is
  ecologically realistic: Baltic cod is in an overfished, bottom-up-controlled state.

"cod" here means BOTH stocks (#129). The Baltic cod stock is disaggregated into
cod_west (sp0) and cod_east (sp8), so the tutorial reports their sum via
``osmose.results.total_cod`` and perturbs both predator columns. Following a single
stock would silently present the smaller half of the population as "cod".
"""

from __future__ import annotations

import shutil
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader

# Three series highlighted in the tutorial narrative. "total_cod" is synthesised by
# add_total_cod() below; the other two are engine species.
FOCAL_SPECIES = ["total_cod", "sprat", "stickleback"]

# The two cod stocks the tutorial sums and perturbs together.
COD_STOCKS = ("cod_west", "cod_east")

# Path to the canonical Baltic accessibility CSV (within data/baltic/).
# The tutorial copies the whole baltic/ directory to a workdir; the
# perturbation in Beat 6 edits the workdir's copy.
BALTIC_DIR = Path(__file__).resolve().parents[1] / "data" / "baltic"
ACCESSIBILITY_CSV_RELPATH = "predation-accessibility.csv"

# Beat-6 perturbation: drop BOTH cod stocks' accessibility to sprat to this value.
PERTURBED_ACCESSIBILITY = 0.05


def apply_cod_sprat_perturbation(acc_csv_path: Path, value: float = PERTURBED_ACCESSIBILITY) -> dict:
    """Set cod_west and cod_east accessibility to sprat to ``value``; return the old values.

    Edits by column NAME, not by string offset. The previous positional replace
    ("sprat;0.4;" -> "sprat;0.05;") could only ever reach the first predator column, so it
    silently missed cod_east once cod was split — and cod_east has the HIGHER accessibility
    to sprat (0.5 vs 0.4), so the perturbation was leaving the larger effect in place.

    The file is semicolon-separated (see CLAUDE.md: the config reader auto-detects the
    separator per line, so a comma-written file is read as one column and fails obscurely).
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(acc_csv_path, sep=";", index_col=0)
    missing = [c for c in COD_STOCKS if c not in df.columns]
    if missing or "sprat" not in df.index:
        raise AssertionError(
            f"Accessibility CSV layout changed: missing predator column(s) {missing} "
            f"or the 'sprat' prey row. Columns: {list(df.columns)[:10]}"
        )
    before = {c: float(df.loc["sprat", c]) for c in COD_STOCKS}
    for c in COD_STOCKS:
        df.loc["sprat", c] = value
    df.to_csv(acc_csv_path, sep=";")
    return before


def add_total_cod(bio_wide):
    """Add a ``total_cod`` column (cod_west + cod_east) to a wide per-species frame.

    Uses osmose.results.total_cod, which falls back to an aggregate ``cod`` column so this
    keeps working on undisaggregated configs.
    """
    from osmose.results import total_cod  # noqa: PLC0415

    out = bio_wide.copy()
    out["total_cod"] = total_cod(bio_wide)
    return out


def build_baltic_workdir(work_dir: Path, n_year: int = 30) -> Path:
    """Copy data/baltic/ into work_dir/baltic/, return the path.

    n_year is stored separately in the config override; this function only
    copies the directory tree.  Default 30 years is enough for the cascade
    demo (full canonical run is 50 yr, which is too slow for a tutorial).
    """
    target = work_dir / "baltic"
    shutil.copytree(BALTIC_DIR, target)
    return target


def build_config(work_dir: Path, n_year: int = 30) -> dict:
    """Load the Baltic config from work_dir/baltic/ and apply tutorial overrides.

    Returns a config dict ready for PythonEngine.run_in_memory().
    The accessibility CSV is read from work_dir/baltic/predation-accessibility.csv;
    tests that exercise the perturbation edit that file before calling build_config.
    """
    baltic_dir = build_baltic_workdir(work_dir, n_year=n_year)
    reader = OsmoseConfigReader()
    cfg = reader.read(str(baltic_dir / "baltic_all-parameters.csv"))
    # Override nyear for tutorial pacing (canonical = 50, tutorial = 30).
    cfg["simulation.time.nyear"] = str(n_year)
    return cfg
