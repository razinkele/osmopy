# Run and Perturb a Real Ecosystem in 30 Minutes

> Tutorial validated against OSMOSE Python **0.12.0** (commit `3db8d5f`).
> Results may differ slightly on other versions.
> The underlying Java engine is not required — this tutorial uses the Python
> in-memory engine exclusively.

This tutorial runs the **calibrated Baltic Sea 8-species OSMOSE configuration**,
zooms in on three focal species (cod, sprat, stickleback), then perturbs
cod's predation accessibility to sprat and observes the trophic cascade.

You do **not** need to know marine ecology to follow along.  Every ecological
term is defined on first use, and the tutorial is structured so you see
results before explanations.

**Audience:** Python-fluent developers or scientists new to OSMOSE.  You should
be comfortable with `pip`, virtual environments, and running Python scripts.
No fisheries background required.

**Time:** approximately 30 minutes (including ~25 s Numba JIT warmup on first
run; subsequent runs complete in under 5 s).

**What you will produce:**

- `tutorial-work/biomass.html` — interactive Plotly time-series of the three
  focal species over 30 simulated years.
- `tutorial-work/biomass_perturbed.html` — the same plot after dropping cod's
  access to sprat, showing the cascade signal.
- Terminal output of equilibrium biomass means for both runs.

---

## Before you start

### 1. Install OSMOSE Python

From the repository root:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode including Plotly and all
development dependencies.  Verify the installation with:

```bash
python -c "import osmose; print(osmose.__version__)"
```

Expected output: `0.12.0`

### 2. Numba JIT warning

OSMOSE's Python engine uses Numba-JIT-compiled kernels.  **The first
invocation compiles and caches the kernels (~25 s); every subsequent call
is fast (<5 s for a 30-year Baltic run).**  The compilation is silent by
default.  If you see messages like `numba: compiling …`, that is normal.

To pre-warm the cache before running the tutorial:

```bash
python -c "from osmose.engine import PythonEngine; PythonEngine()._warmup()"
```

### 3. Version stamp

The version shipped with this tutorial:

```text
0.12.0
```

If your installed version differs, equilibrium biomass values in Beats 2-4
may differ from the printed reference numbers — but the qualitative story
(cod << stickleback << sprat) will be the same.

---

## Beat 1 — Paste, run, observe

**Goal:** run the 30-year Baltic simulation and open the biomass plot in your
browser.  No editing required — paste the entire block below into a file
called `tutorial.py` and run it.

```python
"""30-minute OSMOSE tutorial: run and perturb a Baltic ecosystem.

See docs/tutorials/30-minute-ecosystem.md for the narrative.
Run from any directory:
    python tutorial.py
"""
from pathlib import Path
import shutil

import osmose
import plotly.express as px
from osmose.engine import PythonEngine
from osmose.config.reader import OsmoseConfigReader

# -- [Beat 1] Locate Baltic source data inside the installed package ----------
BALTIC_SRC = Path(osmose.__file__).resolve().parent.parent / "data" / "baltic"
if not BALTIC_SRC.exists():
    raise FileNotFoundError(
        f"Baltic data not found at {BALTIC_SRC}.\n"
        "Install the package from the repository root: pip install -e '.[dev]'"
    )

# -- [Beat 1] Set up a working directory next to wherever you run from --------
WORK = Path("tutorial-work").absolute()
WORK.mkdir(exist_ok=True)
BALTIC = WORK / "baltic"

# Copy the calibrated Baltic config into our workdir (idempotent on re-runs)
shutil.copytree(BALTIC_SRC, BALTIC, dirs_exist_ok=True)

# -- [Beat 2] Load the config; override nyear=30 for tutorial pacing ----------
# The canonical Baltic calibration runs 50 years.  30 years is enough to see
# the ecosystem reach a quasi-equilibrium state (years 5-25) while keeping
# the tutorial fast.
cfg = OsmoseConfigReader().read(str(BALTIC / "baltic_all-parameters.csv"))
cfg["simulation.time.nyear"] = "30"

# -- [Beat 3] Run the simulation -----------------------------------------------
# First run: ~25 s (Numba JIT compilation).  Subsequent runs: <5 s.
# seed=42 makes the stochastic run reproducible.
print("Running Baltic simulation (30 years) ...")
result = PythonEngine().run_in_memory(config=cfg, seed=42)
print("Simulation complete.")

# -- [Beat 4] Reshape biomass to tidy long form --------------------------------
# result.biomass() returns a wide DataFrame:
#   columns = [Time, cod, herring, sprat, ..., species]
# where "species" is a constant "all" sentinel.  We drop it before melting.
bio_wide = result.biomass()
drop_cols = [c for c in ["species"] if c in bio_wide.columns]
bio_long = bio_wide.drop(columns=drop_cols).melt(
    id_vars="Time", var_name="species", value_name="biomass"
)

# -- [Beat 5] Filter to the three focal species --------------------------------
# Cod (Gadus morhua):  top-level predator, overfished in the Baltic.
# Sprat (Sprattus sprattus):  abundant planktivore; key forage fish.
# Stickleback (Gasterosteus aculeatus):  small invertivore; indicator species.
focal = ["cod", "sprat", "stickleback"]
bio_focal = bio_long[bio_long["species"].isin(focal)].copy()

# -- [Beat 1] Write the interactive Plotly chart -------------------------------
fig = px.line(
    bio_focal,
    x="Time",
    y="biomass",
    color="species",
    title="Baltic Sea -- 3 focal species, biomass over 30 years (seed=42)",
    labels={"Time": "Year", "biomass": "Biomass (tonnes)"},
    template="plotly_white",
)
html_path = WORK / "biomass.html"
fig.write_html(html_path)
print(f"\nOpen in your browser: {html_path}")

# -- Equilibrium summary (years 25-30 mean) ------------------------------------
eq = bio_focal[bio_focal["Time"] >= 25].groupby("species")["biomass"].mean()
print("\nEquilibrium biomass (years 25-30 mean):")
print(eq.to_string())
```

**Run it:**

```bash
python tutorial.py
```

**Expected terminal output (values approximate):**

```text
Running Baltic simulation (30 years) ...
Simulation complete.

Open in your browser: /path/to/tutorial-work/biomass.html

Equilibrium biomass (years 25-30 mean):
species
cod             ~1 000 t
sprat        ~5 500 000 t
stickleback    ~550 000 t
```

Open `tutorial-work/biomass.html` in your browser.  You should see three
time-series trajectories diverging rapidly in the first 5 years as the
ecosystem burns off its initial-population state, then settling into a
quasi-equilibrium.

**Sanity check before continuing:**

- All three species are visible in the chart (cod, sprat, stickleback).
- Sprat biomass is roughly 1000x larger than cod biomass — this reflects the
  real Baltic food pyramid (sprat dominates the mid-trophic level; cod is
  a depleted top predator).
- The plot is interactive: hover to read exact values, click a legend label
  to toggle a species.

If the script fails, see the **Troubleshooting** section at the end of this
document.

---

## What just happened? (Beat 1 debrief)

### The Baltic Sea ecosystem

The Baltic Sea is one of the most studied marine ecosystems in the world.
It has eight modelled species in this OSMOSE configuration:

| Code        | Common name         | Trophic level |
|-------------|---------------------|---------------|
| cod         | Atlantic cod        | ~4.0          |
| herring     | Atlantic herring    | ~3.2          |
| sprat       | European sprat      | ~3.0          |
| flounder    | European flounder   | ~3.5          |
| stickleback | 3-spine stickleback | ~3.2          |
| perch       | European perch      | ~3.5          |
| pikeperch   | Pikeperch           | ~4.0          |
| smelt       | European smelt      | ~3.0          |

We highlight three species because they illustrate a **trophic cascade**
(Beat 6): cod eats sprat, cod eats stickleback — so changes in cod's feeding
opportunity ripple through the food web.

### How OSMOSE works

OSMOSE is an **Individual-Based Model** (IBM): every school (cohort) of fish
is tracked separately through time.  At each time step (24 per year in the
Baltic config), the engine:

1. **Grows** each school according to the von Bertalanffy growth equation.
2. **Feeds** each school: schools overlap spatially and predators consume
   accessible prey proportionally to encounter probability.
3. **Reproduces** stock-recruit relationships produce new age-0 cohorts.
4. **Applies mortality**: fishing, starvation, out-of-domain losses.

The **Python engine** (`osmose.engine.PythonEngine`) re-implements the Java
reference engine in NumPy/Numba and runs fully in-process, enabling fast
parameter sweeps and notebook integration.

---

## Next beats (Task 10 will add these)

- **Beat 2** — Inspect the raw output object (`result.biomass()`, `result.mortality()`).
- **Beat 3** — Compare the 8-species full plot to identify which species drive biomass.
- **Beat 4** — Read the equilibrium bands and compare to ICES reference points.
- **Beat 5** — Understand the accessibility matrix (what can eat what).
- **Beat 6** — Perturb cod-sprat accessibility, re-run, observe the cascade.

---

## Troubleshooting

### `FileNotFoundError: Baltic data not found`

You are not running from an editable install of the repository.  Install with:

```bash
pip install -e ".[dev]"
```

### `ModuleNotFoundError: No module named 'osmose'`

Activate your virtual environment before running:

```bash
source .venv/bin/activate
python tutorial.py
```

On Windows:

```bash
.venv\Scripts\activate
python tutorial.py
```

### `ModuleNotFoundError: No module named 'plotly'`

Plotly is included in the `[dev]` extras.  Install with:

```bash
pip install plotly
```

### Simulation raises an exception immediately

If the Baltic data directory is incomplete, re-clone the repository and
reinstall.  Verify with:

```bash
python -c "from pathlib import Path; import osmose; p = Path(osmose.__file__).resolve().parent.parent / 'data' / 'baltic'; print(len(list(p.iterdir())), 'files')"
```

Expected: 20+ files.

### The chart opens but is blank

Confirm `tutorial-work/biomass.html` is larger than 100 KB.  A very small
file indicates Plotly failed silently; check for import errors in the
terminal output.

### First run is very slow (> 5 minutes)

If the Numba JIT cache directory is not writable (e.g., in a read-only
container), Numba recompiles every run.  Set `NUMBA_CACHE_DIR` to a writable
path:

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache
python tutorial.py
```

---

*Tutorial maintained in `docs/tutorials/30-minute-ecosystem.md`.
Report issues at https://github.com/razinkele/osmopy.*
