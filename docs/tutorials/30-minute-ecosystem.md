# Run and Perturb a Real Ecosystem in 30 Minutes

> Tutorial validated against OSMOSE Python **0.12.0** (commit `3db8d5f`).
> Results may differ slightly on other versions.
> The underlying Java engine is not required — this tutorial uses the Python
> in-memory engine exclusively.

This tutorial runs the **calibrated Baltic Sea 8-species OSMOSE configuration**,
zooms in on three focal series (total cod, sprat, stickleback), then perturbs
cod's predation accessibility to sprat and observes the trophic cascade.

Baltic cod is modelled as two separately-assessed stocks (`cod_west`, `cod_east`);
"cod" throughout this tutorial means their **sum**, and the Beat-6 perturbation is
applied to **both**.

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
(sprat outweighs cod, which outweighs stickleback) will be the same.

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

# Copy the calibrated Baltic config into our workdir on first run.
# Subsequent runs preserve any reader edits (e.g., Beat 6 perturbation).
if not BALTIC.exists():
    shutil.copytree(BALTIC_SRC, BALTIC)

# -- [Beat 2] Load the config; override nyear=30 for tutorial pacing ----------
# The canonical Baltic calibration runs 50 years.  30 years captures the key
# transient dynamics (sprat plateau ~yr 20, stickleback boom-bust yr 4-15,
# cod near baseline throughout) while keeping the tutorial fast.
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
#   columns = [Time, cod_west, herring, sprat, ..., cod_east, ..., species]
# where "species" is a constant "all" sentinel.  We drop it before melting.
#
# Baltic cod is modelled as TWO stocks -- cod_west (western, sp0) and cod_east
# (eastern, sp8).  They are assessed separately by ICES and behave differently,
# so the model keeps them apart.  For this tutorial we follow total cod, i.e.
# their sum; osmose.results.total_cod does that (and falls back to a single
# "cod" column on configs that were never disaggregated).
from osmose.results import total_cod

bio_wide = result.biomass()
bio_wide["total_cod"] = total_cod(bio_wide)
drop_cols = [c for c in ["species"] if c in bio_wide.columns]
bio_long = bio_wide.drop(columns=drop_cols).melt(
    id_vars="Time", var_name="species", value_name="biomass"
)

# -- [Beat 5] Filter to the three focal series ---------------------------------
# Total cod (Gadus morhua):  top-level predator, overfished in the Baltic.
#   Both stocks summed -- see Beat 4.
# Sprat (Sprattus sprattus):  abundant planktivore; key forage fish.
# Stickleback (Gasterosteus aculeatus):  small invertivore; indicator species.
focal = ["total_cod", "sprat", "stickleback"]
bio_focal = bio_long[bio_long["species"].isin(focal)].copy()

# -- [Beat 1] Write the interactive Plotly chart -------------------------------
fig = px.line(
    bio_focal,
    x="Time",
    y="biomass",
    color="species",
    log_y=True,  # cod is invisible against sprat (~7M) on a linear scale
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

Biomass over the tutorial window (years 5-25 mean, seed=42):
species
total_cod        ~156 000 t     (cod_west ~10 500 + cod_east ~145 400)
sprat            ~914 000 t
stickleback       ~75 000 t
```

Note how lopsided the cod split is: the eastern stock carries ~93% of the total, so
following `cod_west` alone would track only a fourteenth of Baltic cod.

Open `tutorial-work/biomass.html` in your browser.

**What you should see:**

- **Sprat** is the largest of the three focal series, averaging ~914 000 t over the
  window.  (Herring, not plotted here, is larger still at ~2.55 Mt — the three focal
  series are a slice of the food web, not a census of it.)
- **Total cod** averages ~156 000 t, roughly **6× below sprat**.  That ordering — a
  forage fish outweighing its predator — is the biomass pyramid the tutorial is about.
- **Stickleback** averages ~75 000 t, below cod.  It is a scarce species in this
  configuration rather than a lower trophic level, so its position under cod is not a
  pyramid inversion; it sits at roughly sprat's trophic level.

**The shapes on the plot** (measured, seed=42):

- **Sprat** overshoots early — up from 71 kt in year 1 to a peak of **1.56 Mt at year 4** — then
  falls back and settles around 0.85–0.99 Mt for the rest of the run, ending at 950 kt.
- **Stickleback** does boom and bust: a sharp peak of **201 kt at year 2**, down to ~62 kt by
  year 7, then a noisy plateau between 60 and 94 kt. It does not go to zero.
- **Total cod** starts near nothing (the seeding bootstrap: 71 t in year 1), climbs steeply to
  **183 kt at year 10**, falls to 125 kt by year 20, climbs back to 180 kt by year 25, and is
  falling again at year 30 (134 kt).

**That last one is the most interesting line on the chart, and it is not noise.** Cod and sprat
cycle roughly *out of phase*: cod peaks at year 10 just after sprat bottoms out (822 kt, year 9);
cod then declines through year 20 while sprat climbs back to 973 kt; then cod recovers to another
peak at year 25 as sprat falls again. That is a textbook predator–prey lag — the predator's peak
trails its prey's, and each overshoot sows the next decline.

It also means **30 years is not long enough for this configuration to settle**. The "equilibrium
window" used below (years 5–25) happens to span roughly one full cycle, so its mean is a reasonable
summary — but do not read it as a steady state. The certified Baltic runs are 50 years and score the
final decade; there `cod_east` sits near 83 kt, well below the 145 kt this shorter window reports.

The plot uses a **log y-axis** because biomass spans five or more orders of
magnitude.  On a linear scale, cod would be a flat line indistinguishable from
zero and stickleback's boom-bust would be invisible against sprat's plateau.

**If your plot looks substantially different** — all-zero lines, runaway
exponentials, or a species missing entirely — stop and consult the
troubleshooting table at the bottom of this tutorial before reading on.

**The plot is interactive:** hover to read exact values; click a legend label
to toggle a species on or off.

If the script fails outright, see the **Troubleshooting** section at the end of
this document.

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

## Beat 2 — The grid and the clock

The script's `# -- [Beat 2]` comment anchors the config-loading step.  Here
is what each piece of the configuration means.

**The master config file.**  `data/baltic/baltic_all-parameters.csv` is the
single entry point.  It does not contain all parameters directly — instead it
`include`s every other CSV in the directory via lines like:

```text
include;data/baltic/baltic_species-parameters.csv
include;data/baltic/predation-accessibility.csv
```

`OsmoseConfigReader().read(...)` resolves those includes recursively,
merges them into a single flat dict, and returns a value the Python engine
(`PythonEngine`) can consume without touching the file system again.

**The Baltic grid.**  The simulation domain is a 50 × 40 cell grid covering
the Baltic Sea basin: longitude 10°–30°E, latitude 54°–66°N.  Each cell is
approximately 40 km wide × 33 km tall (~1 320 km²).  Of the 2 000 cells,
612 are ocean — the remainder are land masked out by the Baltic coastline.

**The clock.**  The tutorial overrides `simulation.time.nyear` to 30:

```bash
# in the script, after reading the config:
cfg["simulation.time.nyear"] = "30"
```

The canonical calibrated run uses 50 years.  30 years is enough to capture the system's transient dynamics: sprat reaches
its growth plateau by year 20; stickleback boom-busts in the first decade; cod
remains near its post-collapse baseline throughout.  The cascade signal we care
about in Beat 6 is clearly present at year 25, where the equilibrium-window
comparison (years 25–30 mean) captures the system's transient mean for each
species.

Within each year the clock ticks 24 times (`simulation.time.ndtperyear=24`),
giving a fortnightly time step.  A 30-year run therefore executes 720 time
steps.

---

## Beat 3 — The 8 species, with 3 highlighted

**Baltic food web in brief.**  The eight focal species form a complete
Baltic food web spanning four trophic levels:

- **Apex predators:** cod (*Gadus morhua*), flounder (*Platichthys flesus*),
  perch (*Perca fluviatilis*), pikeperch (*Sander lucioperca*).  These eat
  smaller fish and invertebrates.
- **Mid-trophic planktivores:** herring (*Clupea harengus*), sprat
  (*Sprattus sprattus*), stickleback (*Gasterosteus aculeatus*), smelt
  (*Osmerus eperlanus*).  These eat zooplankton and small invertebrates.
- **Lower trophic level (LTL) forcing:** six groups of phytoplankton and
  zooplankton (see Beat 4) provide the food chain's foundation.

We highlight **cod / sprat / stickleback** because they form a clean
three-level cascade chain that is easy to perturb and measure:

| Species     | Linf   | Lifespan | Role in the cascade |
|-------------|--------|----------|---------------------|
| cod         | 110 cm | 20 yr    | apex predator; eats herring, sprat, stickleback, smelt, and cannibal |
| sprat       | 16 cm  | 8 yr     | small pelagic planktivore; main forage fish for cod |
| stickleback | 8 cm   | 4 yr     | tiny coastal forage fish; under some cod predation pressure |

Baltic dynamics over 30 years are transient rather than steady-state: sprat
reaches its growth plateau by year 20; stickleback boom-busts in the first
decade; cod remains at its post-collapse baseline throughout.  The
equilibrium-window comparison used in Beat 6 (years 25–30 mean) captures the
system's transient mean — which is the meaningful quantity to compare between
baseline and perturbed runs.

**A note on real Baltic dynamics.**  Grey-seal predation on cod is a major
ecological force in the present-day Baltic.  The model proxies it via an
additional mortality rate (`mortality.additional.rate.sp0=0.2` yr⁻¹) on
cod, which is why cod biomass stays low even though cod is not heavily fished
in the simulation.  Post-2015 Baltic cod also shows documented growth
impairment linked to hypoxia and energy deficits — this is NOT currently
represented in the model, so the cod in the simulation is a healthy-growth cod
under extra mortality, not an impaired one.

For the other five species (herring, flounder, perch, pikeperch, smelt) and
full parameter provenance, see `docs/baltic_example.md`.

---

## Beat 4 — LTL forcing: where the food chain starts

OSMOSE does not dynamically model phytoplankton or zooplankton.  Instead it
reads **Lower Trophic Level (LTL) biomass** from a NetCDF file at each
time step.  In the Baltic configuration:

**The data source.**  LTL forcing comes from the CMEMS Baltic biogeochemistry
reanalysis product (`cmems_mod_bal_bgc_anfc_P1M-m`).  Depth-integrated
0–50 m monthly fields are extracted, regridded onto the 50 × 40 Baltic grid,
and stored in `data/baltic/baltic_ltl_biomass.nc` (12 monthly frames,
one per LTL group, units: g wet weight m⁻²).

**Six LTL groups.**  The NetCDF contains six variables:

```text
diatoms           (phytoplankton, large)
dinoflagellates   (phytoplankton, small)
microzooplankton  (< 0.2 mm)
mesozooplankton   (0.2 – 2 mm)  ← most important forage for sprat/herring
macrozooplankton  (> 2 mm)
benthos           (combined benthic invertebrates)
```

**How it is generated.**  The tutorial's canonical 2024-vintage NetCDF is
already present in the repository.  If you need to refresh it (e.g., for a
different year or domain), the generator is at
`mcp_servers/copernicus/server.py::generate_osmose_ltl()` — it requires
CMEMS credentials stored in `.env`.

**Why this matters.**  The foundation of the food chain is real ocean
observation data, refreshed annually from satellite and in-situ reanalysis.
When sprat collapses in the model, it is because the model's zooplankton
abundance — drawn from real data — cannot support that sprat biomass.  The
cascade you will induce in Beat 6 therefore propagates through a real LTL
substrate, not an arbitrary constant.

---

## Beat 5 — Who eats whom: the predation accessibility matrix

Open `tutorial-work/baltic/predation-accessibility.csv` (it was copied from
`data/baltic/predation-accessibility.csv` when you ran `tutorial.py`).  The
file has 15 rows × 16 columns:

- **Rows:** 9 focal prey species + 6 LTL groups (total 15 potential prey types)
- **Columns:** 9 focal predators + background/resource columns (no LTL predators)

Note the **two cod columns**: `cod_west` (sp0) and `cod_east` (sp8). ICES assesses the
western and eastern Baltic cod stocks separately, and the model follows suit.

**What accessibility means.**  Each cell value is the fraction of *encountered*
prey biomass that is actually consumed by that predator.  It is NOT an
encounter probability — encounter is determined separately by the size-ratio
kernel (predator length / prey length must fall within configured bounds) and
by spatial overlap (both must be in the same grid cell at the same time step).
Accessibility is the post-encounter consumption efficiency.

**The cod columns.**  Both cod stocks have nonzero entries; they overlap heavily but
are not identical:

```text
prey species          cod_west   cod_east
herring                   0.4       0.3
sprat                     0.4       0.5    ← you will change BOTH in Beat 6
flounder                  0.1       0.1
perch                     0.15      0.1
pikeperch                 0.1       0.05
smelt                     0.6       0.5
stickleback               0.3       0.2
cod (cannibal)            0.05      0.05
mesozooplankton           0.5       0.5
macrozooplankton          0.6       0.6
benthos                   0.6       0.7
```

Eastern cod has the **higher** accessibility to sprat (0.5 vs 0.4), so a perturbation that
touched only `cod_west` would miss the larger part of the effect.

**The cascade triangle.**  Cod (both stocks, predator) → sprat (prey).  Sprat in turn
suppresses stickleback via competition for zooplankton (cross-trophic indirect
effect).  When cod's access to sprat drops, sprat population stays higher, but
cod also starves slightly — reducing cod's direct predation on stickleback
as well.  Both pathways nudge stickleback biomass upward.

**Provenance.**  The matrix is hand-coded from published Baltic stomach-content
studies and audited twice against the literature (see `docs/baltic_example.md`
provenance section).  Real Baltic trophic cascades are subtler than a synthetic
3-species model would produce because the current ecosystem is bottom-up
controlled: cod biomass is far below its 1970s historical level, so cod's
top-down control signal is weak but still detectable.

---

## Beat 6 — Perturb and watch the cascade

Now you will make a single-line edit, re-run the script, and compare the
equilibrium biomass to the baseline.

**Step 1 — Edit the accessibility matrix.**

Open `tutorial-work/baltic/predation-accessibility.csv` in any text editor. Find the
**`sprat` row** and change the values in **both cod columns** — `cod_west` and `cod_east`
— to `0.05`:

```text
before:  sprat;0.4;0;0;0;0.2;0.3;0;0;0.5;...
                 ^^^ cod_west                ^^^ cod_east
after:   sprat;0.05;0;0;0;0.2;0.3;0;0;0.05;...
```

The file is **semicolon-separated**; keep it that way when you save.

Edit by *column position*, not by search-and-replace on `0.4` — that value appears
elsewhere in the matrix. If you prefer to do it in code:

```python
import pandas as pd
path = "tutorial-work/baltic/predation-accessibility.csv"
df = pd.read_csv(path, sep=";", index_col=0)
df.loc["sprat", ["cod_west", "cod_east"]] = 0.05
df.to_csv(path, sep=";")
```

This reduces western cod's accessibility to sprat by a factor of 8 and eastern cod's by
10.

**Step 2 — Re-run the script.**

```bash
python tutorial.py
```

The second run uses the cached Numba JIT and completes in under 2 s.  The
script reads `tutorial-work/baltic/predation-accessibility.csv` (your edited
copy, not the original in `data/baltic/`), runs the simulation, writes
`tutorial-work/biomass_perturbed.html`, and prints a new equilibrium summary.

**Step 3 — Compare the two equilibrium summaries.**

Side by side in the terminal:

```text
                    baseline          perturbed
total_cod        ~155 900 t       ~148 300 t       (-4.9 %)
sprat            ~914 100 t     ~1 134 300 t      (+24.1 %)
stickleback       ~74 900 t        ~74 800 t       (-0.1 %)
```

Exact values vary with environment; the **direction** is the claim.  **Sprat is the
clear signal: ~+24 %.**  Take both cod stocks off sprat and sprat is released — the
direct first link of a trophic cascade.  Cod itself declines ~5 %, having lost a food
source.  Stickleback does not measurably move.

> **This reverses what earlier editions of the tutorial taught.**  They reported sprat
> moving ~+4 % and stickleback ~+13 %, and described the sprat signal as negligible.  Two
> things changed (#129, 2026-08-02).  The perturbation instruction only ever edited the
> *first* cod column, so after Baltic cod was split into two stocks it silently left
> `cod_east` untouched — and `cod_east` holds the **higher** sprat accessibility (0.5 vs
> 0.4), so most of the intended effect was never applied.  Separately, the Baltic
> recalibration moved cod up by two orders of magnitude.  With both stocks actually
> perturbed, the direct release dominates and the indirect stickleback pathway is not
> detectable.

**Why this matters.**  A ~24 % shift in a forage-fish stock is squarely the
order-of-magnitude effect that ecosystem-based fisheries management (EBFM) cares about:
it is detectable in stock-assessment surveys, it changes sustainable yield estimates, and
it can trigger gear-restriction decisions.  The model reveals the mechanism even when
the magnitude is modest.

---

## Where next

- **Full Baltic provenance and parameter sources:** `docs/baltic_example.md`
  contains the literature citations behind every growth parameter, the ICES
  reference-point priors used for calibration, and the provenance of the
  spawning distribution maps.
- **Calibrate Baltic to ICES stock advice:** `docs/baltic_ices_validation_2026-04-18.md`
  describes the multi-phase calibration workflow; `scripts/calibrate_baltic.py`
  is the entry point (supports `--optimizer {de,cmaes,surrogate-de}`).
- **Engine internals and Java parity:** `docs/parity-roadmap.md` documents the
  14-point parity test suite (bit-exact within 1 OoM across all EEC and Baltic
  fixtures); the Shiny UI is launched with
  `shiny run app.py --host 0.0.0.0 --port 8000`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: osmose` | venv not activated, or repo not installed | `source .venv/bin/activate && pip install -e ".[dev]"` |
| `FileNotFoundError: ...baltic_all-parameters.csv` | OSMOSE not pip-installed (script uses `osmose.__file__` to locate `data/baltic/`) | `pip install -e ".[dev]"` from the repo root |
| First run appears to hang (~25–30 s) | Numba JIT compilation | Wait. Subsequent runs complete in under 2 s. |
| Re-run gives identical results after Beat 6 edit | Edited the wrong file | Edits go in `tutorial-work/baltic/predation-accessibility.csv`, NOT `data/baltic/...`. If you edited the source, reset: `rm -rf tutorial-work/` then re-run `python tutorial.py`. |
| Plot file path printed but the browser won't open it | Headless server with no display | The terminal equilibrium summary contains all the numbers you need; open the HTML on a machine with a browser. |
| `ModuleNotFoundError: No module named 'plotly'` | Plotly not installed | `pip install plotly` or reinstall with `pip install -e ".[dev]"` |
| Want to start completely fresh | Stale `tutorial-work/` from a previous run | `rm -rf tutorial-work/` then re-run. The script re-copies the calibrated config on every run. |
| Beat 6 perturbation didn't produce a visible change in the plot | The cascade signal is ~13 % — easy to miss visually | Compare the printed equilibrium numbers directly: `mean(stickleback_perturbed) / mean(stickleback_baseline)` should be ≈ 1.13. |
| First run is very slow (> 5 minutes) | Numba JIT cache directory not writable (e.g., read-only container) | `export NUMBA_CACHE_DIR=/tmp/numba_cache` then re-run. |
| Simulation raises an exception immediately | Incomplete Baltic data directory | Verify: `python -c "from pathlib import Path; import osmose; p = Path(osmose.__file__).resolve().parent.parent / 'data' / 'baltic'; print(len(list(p.iterdir())), 'files')"` — expect 20+ files. |

---

*Tutorial maintained in `docs/tutorials/30-minute-ecosystem.md`.
Report issues at https://github.com/razinkele/osmopy.*
