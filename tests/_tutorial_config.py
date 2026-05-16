"""Canonical source of truth for the 30-min tutorial (Baltic substrate).

The tutorial uses the data/baltic/ 8-species calibrated config and highlights
cod, sprat, and stickleback for the trophic cascade narrative.

The reader's workflow: copy baltic/ to a workdir, override nyear for speed,
run the engine, plot biomass for the 3 highlighted species, perturb
cod-on-sprat accessibility, re-run, observe the cascade.

Cascade mechanics (Baltic-substrate finding):
  Dropping cod's accessibility to sprat (0.4 -> 0.05) slightly reduces
  predation pressure on stickleback (cod eats stickleback as well as sprat;
  with less sprat food, cod is less abundant -> fewer cod -> stickleback UP).
  The sprat signal is weak (<1 %) because cod biomass is small relative to the
  sprat population.  This is ecologically realistic: Baltic cod is in an
  overfished, bottom-up-controlled state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from osmose.config.reader import OsmoseConfigReader

# Three species highlighted in the tutorial narrative.
FOCAL_SPECIES = ["cod", "sprat", "stickleback"]

# Path to the canonical Baltic accessibility CSV (within data/baltic/).
# The tutorial copies the whole baltic/ directory to a workdir; the
# perturbation in Beat 6 edits the workdir's copy.
BALTIC_DIR = Path(__file__).resolve().parents[1] / "data" / "baltic"
ACCESSIBILITY_CSV_RELPATH = "predation-accessibility.csv"

# Beat-6 perturbation: drop cod's accessibility to sprat from 0.4 to 0.05.
# In the accessibility CSV rows = prey, cols = predators.
# The sprat row begins "sprat;0.4;" where 0.4 is cod's column (first predator).
# We replace it with 0.05 to simulate reduced predation opportunity.
BASELINE_PERTURBATION: tuple[str, str] = ("sprat;0.4;", "sprat;0.05;")


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
