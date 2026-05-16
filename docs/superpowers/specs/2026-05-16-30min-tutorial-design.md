# 30-Minute Tutorial — Design Spec

> Status: **DRAFT r3** (2026-05-16). Converged through two in-loop multi-angle review rounds (architect, red-team, newcomer-UX, domain, internal-consistency, test-design, tech-writing). Implementation plan to follow via `superpowers:writing-plans`.

## Purpose

Fill the gap between the README's `## Quick start` (which stops at "open localhost:8000") and the working knowledge a newcomer needs to actually *use* OSMOSE. A 30-minute, self-paced, single-session tutorial that takes a reader from "I installed it" to "I built and perturbed a working 3-species food web."

This is the **front door** of the project for anyone who isn't already familiar with ecosystem models. After it ships, the docs index in README.md leads with it.

## Audience and success criteria

**Audience:** Python-fluent, broadly scientific, *not necessarily* an ecologist. Someone who has read the README, run `pip install -e ".[dev]"`, and wants to know what the simulator actually does.

**Reader succeeds when they have:**
1. A `tutorial.py` they wrote (by paste) that runs `PythonEngine().run_in_memory(...)` end-to-end.
2. A `biomass.html` plot showing three biomass trajectories converging into a stable food chain.
3. Made one parameter change (a single number in `accessibility.csv`), re-run, and *observed* the trophic cascade in the plot.
4. A mental model of OSMOSE's core surface area: grid, time, focal species, LTL resource forcing, predation accessibility.
5. Pointers to the next surfaces (`baltic_example.md`).

**Reader is NOT expected to:** know fisheries vocabulary (Linf, K, accessibility, dt, trophic level are defined inline at first use); have a Java install (Python engine only); have an internet connection (no Copernicus / no ICES API).

## Format and substrate decisions

| Decision | Value | Why |
|---|---|---|
| Format | Markdown walkthrough with code blocks | Cheapest; matches project docs style; no new deps. |
| Substrate | Synthetic 3-species toy model from scratch | "Build" framing; no literature-sourcing tax. |
| Arc | Demo-first: paste → run → unpack → perturb | Engagement matters in a 30-min budget; reader sees a plot at minute 5. |
| Config surface | Python dict literal using **real engine keys** | Pythonic, self-contained, matches what the reader will see later in `data/baltic/`. |
| Engine API | `PythonEngine().run_in_memory(config=cfg, seed=42)` → `OsmoseResults` | `.run()` returns `RunResult` (disk-shape, no DataFrame). `.run_in_memory()` returns `OsmoseResults`; `.biomass()` returns a wide-form `["Time", <sp_names>, "species"]` DataFrame which we must `melt` into tidy form for plotly. Refs in appendix. |
| Spatial | `grid.nlon=4, grid.nlat=4` — engine creates a rectangular all-ocean 4×4 grid | No mask CSV, no NetCDF for grid. |
| Movement | `movement.distribution.method.sp{i}=random`, `movement.randomwalk.range.sp{i}=1` per focal species | Skips indexed-map authoring; same shortcut `data/minimal/` uses. |
| LTL | One resource species (`species.type.sp3=resource`), variable matches `species.name.sp3` (case-sensitive), dims `(time, latitude, longitude)`, **24** timesteps (engine wraps `step % n_dt_per_year`) | Per the engine's resource discovery path. |
| Accessibility | A CSV the tutorial writes; pointed at by `predation.accessibility.file` | No dotted-dict-key surface exists for accessibility entries. |
| Plot | `plotly.express.line` → `fig.write_html()` | Matches Shiny UI idiom; works headless; no kaleido. |
| Companion script | None | Markdown is the only ship surface. |
| New `data/tutorial/` directory | No | Reader-created `tutorial-work/`; nothing in-tree. |

## File layout (ship-list)

```
docs/tutorials/
  30-minute-ecosystem.md      ← the tutorial (~700–900 lines)
  README.md                   ← one paragraph; index for future tutorials
tests/
  _tutorial_config.py         ← canonical config dict + CSV + NetCDF builders (single source of truth)
  test_tutorial_3species.py   ← regression test
README.md                     ← MODIFIED: top-of-page "New here?" pointer + new row at top of doc index
```

Deferred to a post-merge bookkeeping step (not in the plan): a `project_30min_tutorial_shipped.md` memory entry and a one-line update to `MEMORY.md`.

## Single source of truth

`tests/_tutorial_config.py` exports four objects:
- `build_config(work_dir: Path) -> dict` — returns the engine config (~80 keys) with `species.file.sp3` and `predation.accessibility.file` resolved against `work_dir`. **Must be a function, not a bare dict**, because path values vary between the tutorial's `tutorial-work/` and the test's `tmp_path`.
- `ACCESSIBILITY_CSV: str` — the canonical `;`-separated CSV string. Written verbatim to `work_dir / "accessibility.csv"` by the test fixture and the tutorial.
- `build_ltl(work_dir: Path) -> Path` — writes `ltl.nc` to `work_dir`, returns the path.
- `BASELINE_PERTURBATION: tuple[str, str]` — the exact substring the tutorial tells the reader to find + the exact string they replace it with. The test substring-checks both.

The tutorial markdown's main code block is a **transcribed copy** of these objects' contents (paths inlined for `tutorial-work/`). The sync contract: the test fixture imports `_tutorial_config` directly; a separate fast test (`test_markdown_block_parses`) regex-extracts the first ```python fence from `30-minute-ecosystem.md` and runs `ast.parse()` on it to catch syntactic drift. Behavioural drift in the markdown's actual values is caught by manual reconciliation at PR review time — discipline cue in the test docstring.

## The 30-minute narrative (six beats)

"30 minutes" is the brand-marketing time, not a measured stopwatch. Wall-clock for an attentive Python-fluent reader on a warm laptop is plausibly 25-40 minutes depending on JIT compile (~25 s first run) and reading speed.

### Preamble — "Before you start" (~1 min)

Three lines, not a beat:
1. `pip install -e ".[dev]"` from the repo root (verifies `osmose`, `xarray`, `plotly`, `h5netcdf` are importable). The implementer confirms `h5netcdf` is in the `[dev]` extras; if not, that's a one-line pyproject addition.
2. "First run takes ~25 s for Numba to JIT-compile native code. Subsequent runs are <2 s."
3. "Tutorial validated against OSMOSE Python `<version>` (commit `<sha>`)." Stamp the version at ship time.

### Beat 1 — "Run something"

A single self-contained code block (~100 lines, dense comments in the dict, sparse comments in the I/O). Reader pastes into `tutorial.py`, runs `python tutorial.py`, sees `tutorial-work/biomass.html` plus `print()` output for headless servers.

End-of-beat check: *"You should see three lines diverging from similar starts and stabilizing by year 50. PlanktonEater is the tallest, Forager mid, Predator lowest. **If your plot looks substantially different — flat zeros, runaway exponentials — STOP and consult the troubleshooting table at the bottom before reading on.**"*

### Beat 2 — "The grid and the clock"

Refer back to **anchor comments** in the dict (`# [Beat 2] grid + time ↓`), not line numbers. Explain `grid.nlon=4`, `grid.nlat=4`, `simulation.time.nyear=50`, `simulation.time.ndtperyear=24`, `mortality.subdt=10`. Define `dt` (time step), `ndtperyear` (24 = fortnightly), and `quasi-equilibrium` (biomass trajectories that have stopped trending and only fluctuate).

### Beat 3 — "The three fish"

Anchor `# [Beat 3] the four keys you should understand ↓` groups `linf`, `k`, `lifespan`, `egg.size` visibly. Brief von Bertalanffy aside: "`L∞` is asymptotic max length; `K` is how fast the fish approaches `L∞`. Large + slow fish (Predator: Linf=150 cm, K=0.12) live long; small + fast (PlanktonEater: Linf=8 cm, K=0.7) live short." Remaining ~16 keys per species ("growth-curve and allometry parameters using OSMOSE-EEC defaults") get one paragraph and are not walked through.

### Beat 4 — "Plankton: the bottom of the food chain"

Define LTL (low-trophic-level) and TL=1.0 (primary producers). Why this layer exists: OSMOSE doesn't dynamically model phytoplankton; the bottom of the food chain is exogenous forcing.

Show the ~10-line xarray Dataset construction → `.to_netcdf()`:
- Dims: `(time, latitude, longitude)` — **exactly these names**.
- Time length: **24** (one annual cycle; engine wraps).
- Variable name: `"Plankton"` — case-sensitive match against `species.name.sp3`.
- Constant: **10000 t/cell**. Lower values starve the food chain on a 16-cell grid.

**Toy-model framing:** "Real OSMOSE configs use spatiotemporal CMEMS-derived forcing on 50×40 grids. This tutorial uses a constant, single-group, 4×4 forcing because we're teaching the *shape*, not the science. For the science, see `docs/baltic_example.md`."

### Beat 5 — "Who eats whom"

The predation accessibility matrix is loaded from a CSV the tutorial writes (`tutorial-work/accessibility.csv`):

```
;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
```

Rows = prey labels; columns = predator labels (focal species only). Define `accessibility`: "the fraction of *encountered* prey biomass that a predator can actually consume given overlap, gut hardware, and behavioural rejection. Not encounter probability — that comes from the size-ratio kernel + spatial overlap."

Explain the **size-ratio kernel**: even at accessibility 1.0, predator length must be ≥ `sizeratio.min` × prey length to eat. We pin `predation.predprey.sizeratio.min.sp{i}=2.0` per focal species so both predation legs (Predator→Forager and Forager→PlanktonEater) transmit reliably. Note the Predator-on-PlanktonEater entry is 0 — Predator is a Forager specialist; this avoids the cascade short-circuit a generalist Predator would create.

### Beat 6 — "Perturb and watch the cascade"

Reader edits ONE substring in `tutorial-work/accessibility.csv`: the canonical line `Forager;0.8;0;0` — change `0.8` → `0.1`. Save. Re-run `python tutorial.py`. Compare biomass plots side by side.

Observable: Forager biomass climbs (no longer eaten); PlanktonEater biomass falls (more Foragers eating them). One paragraph framing: *"This is what ecosystem models do — they reveal indirect effects that wouldn't fall out of three independent single-species models."*

### Closing — "Where next" (2 sentences)

1. *"Real-world Baltic example with 8 species, CMEMS-derived forcing, and full literature provenance: `docs/baltic_example.md`."*
2. *"For engine internals, Java parity, and calibration: `docs/parity-roadmap.md` and the Shiny UI (`shiny run app.py`)."*

The earlier r2 promise that Shiny UI Setup → Load could open this tutorial's config was **dropped** — the UI accepts CSV/properties on disk, not Python dict literals; the model section section's `to_files()` serialization step is also out of scope (a follow-up tutorial).

### Troubleshooting (table at the bottom)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: osmose` | venv not activated or repo not installed | `source .venv/bin/activate && pip install -e ".[dev]"` |
| `ValueError: NetCDF backend not found` | `h5netcdf` missing | `pip install h5netcdf` |
| First run "hangs" ~25 s (much longer on slow VMs / cold laptops) | Numba JIT compilation | Wait. Subsequent runs are <2 s. |
| `ValueError: unknown configuration keys` | Dict mis-typed | Compare your dict against the canonical block at the top of this tutorial. |
| Plot file path printed but won't open | Headless server | The script also prints the equilibrium values; read those. |
| Want to start over from scratch | Stale artifacts in `tutorial-work/` | `rm -rf tutorial-work/` then re-run. |
| `FileNotFoundError: ltl.nc` after editing accessibility CSV | Re-ran from a different CWD | `cd` back to the directory where `tutorial.py` lives. Paths are absolute under `tutorial-work/`. |
| Beat 6 perturbation didn't change anything | Edited the wrong `0.8` | The CSV contains *two* `0.8`s — one on `Forager;0.8;0;0` (target) and one on `Plankton;0;0.2;0.8` (last column; do NOT edit). Search for the literal string `Forager;0.8` to find the right line. |

## The synthetic model (concrete values — normative)

This section is normative. Beats above may paraphrase; if they conflict with this section, this section wins.

### Three focal species

| sp  | name             | Linf (cm) | K     | t0    | lifespan (yr) | egg size (cm) | maturity len (cm) |
|-----|------------------|----------:|------:|------:|--------------:|--------------:|------------------:|
| sp0 | Predator         | 150       | 0.12  | -0.3  | 25            | 0.25          | 50                |
| sp1 | Forager          | 25        | 0.40  | -0.5  | 8             | 0.15          | 12                |
| sp2 | PlanktonEater    | 8         | 0.70  | -0.3  | 4             | 0.10          | 4                 |

Size ratios with these values: Predator/Forager = 6×, Forager/PlanktonEater = 3.1× — both safely above the `sizeratio.min=2.0` floor.

### Reproduction + mortality

- `mortality.additional.larva.rate.sp{i}=5.0` per focal species — without it, populations explode (see `project_phase12_cod_floor.md`).
- `population.seeding.biomass.sp{i}`: **200 / 1000 / 4000 t** for Predator/Forager/PlanktonEater.
- `population.seeding.year.max=20`.

### One LTL group

- `species.name.sp3=Plankton`, `species.type.sp3=resource`, `species.file.sp3=tutorial-work/ltl.nc`
- `species.size.min.sp3=0.0002`, `species.size.max.sp3=0.5`
- `species.trophic.level.sp3=1.0`
- `species.accessibility2fish.sp3=0.8` — overrides the engine default (~0.01). **Note:** this acts as a multiplier on the raw NetCDF biomass when fish encounter the resource (`osmose/engine/resources.py:231`), so the *effective* plankton biomass available to predators is 0.8 × 10 000 = 8 000 t/cell. If you want exactly 10 000 t/cell effective, set this to 1.0 (capped at 0.99 by the engine).
- `simulation.nresource=1`
- NetCDF: `Plankton` variable, dims `(time, latitude, longitude)`, shape `(24, 4, 4)`, constant **10000.0**.

### Grid + movement

- `grid.nlon=4`, `grid.nlat=4` — engine creates the rectangular ocean grid.
- `movement.distribution.method.sp{i}=random`, `movement.randomwalk.range.sp{i}=1` for each focal species.

### Predation accessibility (CSV)

`tutorial-work/accessibility.csv`:
```
;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
```

Loaded by `osmose/engine/accessibility.py:72-131` via `pd.read_csv(sep=';', index_col=0)`.

### Other mandatory keys

- `simulation.nspecies=3`
- `simulation.fishing.mortality.enabled=false`, `simulation.nfisheries=0`
- `mortality.subdt=10`
- `predation.efficiency.critical.sp{i}=0.57` per focal species (Shin & Cury 2004)
- `predation.ingestion.rate.max.sp{i}`: 4.0 / 5.0 / 6.0 g/g/yr for Predator/Forager/PlanktonEater
- `predation.predprey.sizeratio.min.sp{i}=2.0`, `.max.sp{i}=1000.0` (all-lowercase keys; the engine ignores mixed-case)
- `species.length2weight.condition.factor.sp{i}=0.0070` per species (OSMOSE-EEC default)
- `species.length2weight.allometric.power.sp{i}=3.05` per species (OSMOSE-EEC default)
- `species.vonbertalanffy.threshold.age.sp{i}=1.0` per species
- `species.maturity.size.sp{i}`: 50 / 12 / 4 cm for sp0/sp1/sp2 (matches the table at the top of this section)
- `species.relativefecundity.sp{i}=500` per species (the engine default at `osmose/engine/config.py:499`; reference Baltic configs use 300–1200, EEC uses 0.14–2228 with very different unit conventions, so the engine default is the right anchor for a toy)
- `species.sexratio.sp{i}=0.5` per species (engine default; declared explicitly to survive `validation.strict.enabled=error`)

If the engine has stage-structured predation thresholds enabled, `predation.predprey.stage.threshold.sp{i}` may also be required; check `osmose/engine/config.py:572-639`. Since this synthetic model has no stage structure, leave the global `predation.predprey.stage.structure=size` at its engine default (line 572) and skip the per-species variants + thresholds.

The implementer enumerates the complete set against `osmose/engine/config.py:1428-1500` (the `_get` / `_species_float` walk under `EngineConfig.from_dict`); expected total ~80 keys.

### Run economics

50 yr × 24 dt/yr × 3 focal + 1 LTL on a 4×4 grid: warm-Numba ≤ 2 s per run; first run pays ~25 s of JIT compile.

## Code shape

One file, one paste. Anchor comments mark each beat's domain. The skeleton below is **illustrative** — the implementer authors the final source against `tests/_tutorial_config.py` and verifies it runs.

```python
"""3-species OSMOSE ecosystem tutorial — see docs/tutorials/30-minute-ecosystem.md."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
from osmose.engine import PythonEngine

WORK = Path("tutorial-work").absolute(); WORK.mkdir(exist_ok=True)
N_LON, N_LAT, N_YR, N_DT = 4, 4, 50, 24

# [Beat 5] Predation accessibility CSV — Beat 6 perturbs the Forager;0.8 entry ↓
ACCESSIBILITY_CSV = """;Predator;Forager;PlanktonEater
Predator;0;0;0
Forager;0.8;0;0
PlanktonEater;0;0.8;0
Plankton;0;0.2;0.8
"""
(WORK / "accessibility.csv").write_text(ACCESSIBILITY_CSV)

# [Beat 4] LTL plankton forcing — constant 10000 t/cell over one annual cycle ↓
xr.Dataset(
    {"Plankton": (("time", "latitude", "longitude"), np.full((N_DT, N_LAT, N_LON), 10000.0))},
    coords={"time": np.arange(N_DT), "latitude": np.arange(N_LAT), "longitude": np.arange(N_LON)},
).to_netcdf(WORK / "ltl.nc")

# [Beat 2-5] The model: grid, clock, three focal species + one LTL group, no fishing ↓
config = {
    # [Beat 2] grid + time
    "grid.nlon": N_LON, "grid.nlat": N_LAT,
    "simulation.time.nyear": N_YR, "simulation.time.ndtperyear": N_DT,
    "simulation.nspecies": 3, "simulation.nresource": 1,
    "simulation.fishing.mortality.enabled": "false", "simulation.nfisheries": 0,
    "mortality.subdt": 10,

    # [Beat 3] the four keys you should understand: linf, k, lifespan, egg.size ↓
    "species.name.sp0": "Predator",
    "species.linf.sp0": 150.0, "species.k.sp0": 0.12, "species.lifespan.sp0": 25,
    "species.egg.size.sp0": 0.25,
    # ... t0, allometry, threshold age, ingestion, efficiency, size-ratio, movement,
    #     larva mortality, seeding biomass for sp0 — ~16 more keys ...
    # ... then sp1 (Forager) and sp2 (PlanktonEater) with their own values ...

    # [Beat 4] LTL Plankton (sp3)
    "species.name.sp3": "Plankton", "species.type.sp3": "resource",
    "species.file.sp3": (WORK / "ltl.nc").as_posix(),
    "species.size.min.sp3": 0.0002, "species.size.max.sp3": 0.5,
    "species.trophic.level.sp3": 1.0, "species.accessibility2fish.sp3": 0.8,

    # [Beat 5] Predation accessibility CSV
    "predation.accessibility.file": (WORK / "accessibility.csv").as_posix(),

    "population.seeding.year.max": 20,
}

# Run + reshape biomass to tidy form for plotly
result = PythonEngine().run_in_memory(config=config, seed=42)
bio_wide = result.biomass()                                            # ["Time", <sp_names>, "species"="all"]
bio_long = (bio_wide.drop(columns=["species"])                          # drop vestigial "all" column
                    .melt(id_vars="Time", var_name="species", value_name="biomass"))
bio_long = bio_long[bio_long["species"].isin(["Predator", "Forager", "PlanktonEater"])]

# Plot + headless fallback
fig = px.line(bio_long, x="Time", y="biomass", color="species",
              title=f"3-species ecosystem biomass over {N_YR} years",
              template="plotly_white")
fig.write_html(WORK / "biomass.html")
print(f"Open: {WORK / 'biomass.html'}")
print("Equilibrium biomass (years 45-50 mean):")
print(bio_long[bio_long["Time"] >= 45].groupby("species")["biomass"].mean())   # headless fallback
```

### Key shape decisions (post r1+r2 review)

- **Real engine keys**, real APIs — `run_in_memory()` returns `OsmoseResults`; `.biomass()` returns wide-form; we `melt` to tidy.
- **All-lowercase keys** including `predation.predprey.sizeratio.{min,max}.spN` — the engine ignores mixed-case silently.
- **`as_posix()`** on path values — Windows backslashes survive `str(Path)` but `:` in drive letters can collide with config-reader separator heuristics.
- **Anchor comments** for each beat — robust to paste-formatting and line-number drift.
- **CSV edit is the perturbation** — Beat 6 is "open `accessibility.csv`, find `Forager;0.8;0;0`, change `0.8` → `0.1`." This is how OSMOSE configs actually work.
- **No try/except** — engine errors surface raw; readers learn the real failure modes.

## Regression test

### File: `tests/test_tutorial_3species.py`

Imports `build_config`, `ACCESSIBILITY_CSV`, `build_ltl`, `BASELINE_PERTURBATION` from `tests/_tutorial_config.py`. The fixture writes the accessibility CSV + LTL NetCDF to a pytest `tmp_path`, calls `cfg = build_config(tmp_path)`, then `PythonEngine().run_in_memory(config=cfg, seed=42)`. Uses a session-scoped `numba_warmup` fixture (runs a single-year throwaway sim before any timed assertions) so the per-test budget is warm-only. Total session budget: **under 45 seconds** including warmup; per-test budget after warmup: under 15 s.

### Assertions

1. **The script runs to completion.** `run_in_memory(config=cfg, seed=42)` returns a non-empty `OsmoseResults`; `.biomass()` returns a wide-form DataFrame; the post-melt tidy DataFrame contains exactly `{"Predator", "Forager", "PlanktonEater"}` in its `species` column and has more than 100 rows per species. (Engine emits one biomass row per dt by default — `output.recordfrequency.ndt=1` in `osmose/engine/config.py:833` — so 50 yr × 24 dt = 1200 rows per species. Assert *more than 100* rather than exact equality to tolerate engine output-frequency changes.) Test sets `cfg["validation.strict.enabled"]="error"` to surface unknown-key drift loudly.
2. **Biomass pyramid holds at the windowed mean over years 45–50.** `mean(PlanktonEater) > mean(Forager) > mean(Predator)`. Margins are *measured during authorship* — the implementer runs the model, records the actual equilibrium ratios, and codes them into the test with a ±20 % slack band. (Hand-picked margins risked tuning-to-pass; measured margins anchor the test to empirical reality.)
3. **Trophic cascade is visible at the windowed mean over years 45–50.** Re-running with `Forager;0.1` (instead of `Forager;0.8`):
   - `mean(Forager_perturbed) > 2.0 × mean(Forager_baseline)` — release-from-predation produces a strong response (an 8× drop in accessibility should not just barely move the needle).
   - `mean(PlanktonEater_perturbed) < 0.6 × mean(PlanktonEater_baseline)` — the cascade reached the bottom with a strong signal.
4. **Markdown code block parses.** A separate fast test extracts the first ```python fence from `30-minute-ecosystem.md` and runs `ast.parse()` on it. Costs ~50 ms; catches syntactic drift between the helper and the markdown.
5. **The perturbation instruction is findable.** Substring check: `"Forager;0.8;0;0" in ACCESSIBILITY_CSV`. Catches column-reorder / row-rename drift that wouldn't break the main test but *would* break the tutorial's instructions.
6. **Headless fallback produces meaningful output.** Asserts the equilibrium summary `bio_long[bio_long["Time"] >= 45].groupby("species")["biomass"].mean()` returns 3 finite values (one per focal species).

### What the test does NOT assert

- Exact biomass values. Trophic *ordering* and *direction-of-change* are the load-bearing claims; absolute magnitudes can shift with engine vectorisation.
- Run time as a hard gate (Numba warmup variance is large on CI). 45 s is the session budget, not a per-assertion gate.
- Plotly output (test doesn't import plotly).

### Authorship discipline (not enforced — discipline cue)

Implementer authors `_tutorial_config.py` first, runs the regression test, then transcribes the helper's contents into `30-minute-ecosystem.md`. Test docstring says: *"If `CONFIG`, `ACCESSIBILITY_CSV`, or `build_ltl` change, update `docs/tutorials/30-minute-ecosystem.md` to match."*

## Cross-references

### `README.md` changes (two edits)

1. **Top-of-page "New here?" callout** — between the one-paragraph intro and the `## Status` table (not buried in the doc index):
   > *🚀 New here? Build a 3-species ecosystem in 30 minutes: [Tutorial](docs/tutorials/30-minute-ecosystem.md).*
2. **New row at the top of the `## Documentation index` table:**
   > *| Learn OSMOSE by building a 3-species ecosystem in 30 min | [`docs/tutorials/30-minute-ecosystem.md`](docs/tutorials/30-minute-ecosystem.md) |*

### `docs/tutorials/README.md` (one paragraph)

> Hands-on tutorials. Self-contained, single-session. Each tutorial is a markdown walkthrough you read top-to-bottom with code you copy-paste into one Python file.
>
> - [30-minute-ecosystem.md](30-minute-ecosystem.md) — build, run, and perturb a 3-species food web. Start here.

## Out of scope

- New dependencies. `xarray`, `plotly`, `numpy`, `numba`, `h5netcdf` are already in `[dev]`.
- Jupyter notebook (`.ipynb`).
- CI matrix changes — regression test runs in the existing pytest suite.
- `data/tutorial/` directory in-tree.
- Screenshots or embedded images.
- Separate `docs/getting-started/` landing page — the tutorial *is* the landing page.
- Follow-up tutorials in this round.
- Translations, PDF export.
- Shiny UI tour, dict-to-CSV serialisation for UI Setup → Load (`EngineConfig.from_dict().to_files()` is a follow-up tutorial).
- Companion `scripts/tutorial.py`.
- `kaleido` (only needed for `write_image()`; we ship `write_html()` only).
- Windows-specific CI; `as_posix()` is the mitigation but it's untested on Windows. Reader reports welcome.

## Implementation discipline for the writing-plans agent

This section is normative for how the plan should be structured.

### Task ordering (test-first, with stubs)

1. **Author `_tutorial_config.py` as stubs**: `build_config` returns `{}`; `build_ltl` raises `NotImplementedError`; `ACCESSIBILITY_CSV = ""`; `BASELINE_PERTURBATION = ("", "")`. Commit.
2. **Author `test_tutorial_3species.py`** in its final form (all 6 assertions). Run it — confirm it goes RED in the expected way (helper stubs fail). Commit.
3. **Probe `result.biomass()` column shape** by writing a 10-line script that runs `data/minimal/` end-to-end and prints `df.columns, df.head()`. Lock the melt + filter logic against the observed shape.
4. **Enumerate every mandatory key** in `EngineConfig.from_dict` (`osmose/engine/config.py:1428-1500`; follow `_get` and `_species_float` calls; also scan `_SUPPLEMENTARY_ALLOWLIST` for reader-injected keys). Build the complete `build_config` dict.
5. **Use a 5-year smoke** (`N_YR=5`) for parameter shake-down. Only run the 50-yr config once values are pinned. Each cold-Numba iteration on the 50-yr config is ~25 s — the 5-yr smoke is ~5 s after JIT and gives you direction-of-change for most parameter probes.
6. **Measure the equilibrium ratios**: with the locked `build_config`, run `N_YR=50, seed=42`, then `bio_long[bio_long["Time"] >= 45].groupby("species")["biomass"].mean()`. Encode those numbers into the test's pyramid assertion as `±20%` margins. **If the measured ordering doesn't match the spec's narrative** (`PlanktonEater > Forager > Predator`), the parameters need adjusting — the *narrative* is the load-bearing design promise, not the parameter values. Adjust seeding biomass + larva mortality until measurement matches narrative; do not flip the narrative.
7. **Implement `numba_warmup` as a session-scoped pytest fixture** that runs one `N_YR=1` simulation before any test class. After warmup, each timed test sees a warm JIT and runs in <2 s.
8. **Confirm `h5netcdf` is in `[dev]` extras** in `pyproject.toml`. If not, add one line.
9. **Verify `species.accessibility2fish.sp{N}` engine path** for resource species in `osmose/engine/resources.py`; lock the 0.8 override.
10. **Verify `simulation.fishing.mortality.enabled=false` + `simulation.nfisheries=0` is sufficient** to disable fishing without stub selectivity keys (an open question — if extras are needed, add them at step 4).
11. **Write `30-minute-ecosystem.md`** with the dict transcribed from the (now-locked) `build_config`. Set version stamp.
12. **Add the markdown `ast.parse` test + the perturbation-substring test**.
13. **Write `docs/tutorials/README.md`**, edit `README.md` (top-of-page pointer + doc-index row), and ship.

### Measurement protocol for the pyramid assertion (step 6 above)

**Prerequisite:** this script requires step 4 complete (`build_config` returning a real dict, not the stub). If you try to run it earlier you'll get `KeyError` from the engine's mandatory-key check.

```python
from pathlib import Path
from tests._tutorial_config import build_config, ACCESSIBILITY_CSV, build_ltl
from osmose.engine import PythonEngine

work = Path("/tmp/measure"); work.mkdir(exist_ok=True)
(work / "accessibility.csv").write_text(ACCESSIBILITY_CSV)
build_ltl(work)
cfg = build_config(work)

result = PythonEngine().run_in_memory(config=cfg, seed=42)
bio_long = (result.biomass().drop(columns=["species"])
                  .melt(id_vars="Time", var_name="species", value_name="biomass"))
focal = bio_long[bio_long["species"].isin(["Predator", "Forager", "PlanktonEater"])]
equilibrium = focal[focal["Time"] >= 45].groupby("species")["biomass"].mean()
print(equilibrium)
```

Take these numbers, encode them with ±20 % into the test. Commit the numbers in a comment so future drift can be detected via PR diff.

### `work_dir` contract

`build_config(work_dir: Path) -> dict` returns a dict where the `species.file.sp3` and `predation.accessibility.file` values are resolved to `work_dir / "ltl.nc"` and `work_dir / "accessibility.csv"` respectively (via `.as_posix()`). The tutorial passes `Path("tutorial-work").absolute()`; the test passes `tmp_path`. Helper has no global state.

## Acceptance criteria

1. **Follow the tutorial end-to-end** in a fresh venv with `pip install -e ".[dev]"`. Produces `tutorial-work/biomass.html` showing three diverging biomass lines that converge into a stable food chain by year 50 — the qualitative ordering matches whatever the measurement protocol pinned in step 6 above (if measurement says `Forager > PlanktonEater > Predator`, the tutorial narrative reflects that ordering).
2. **Perform the perturbation** (`Forager;0.8` → `Forager;0.1` in `accessibility.csv`); re-run produces a visibly different plot with Forager up and PlanktonEater down.
3. **Run `.venv/bin/python -m pytest tests/test_tutorial_3species.py -v`** — passes in under 45 seconds (session-level).
4. **All README links resolve.**
5. **Regression test sets `validation.strict.enabled=error` and completes without raising on unknown keys.** (Error mode raises `ValueError` listing every unknown key; success = no raise.) This enforces dict completeness.
6. **Troubleshooting table covers** every error a reader is likely to encounter in their first paste: missing venv, missing NetCDF backend, Numba cold start, dict typo, headless display, stale state, wrong CWD, wrong-`0.8`-edit.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Numba JIT compile is slow; reader thinks it hung | Preamble + troubleshooting table both warn explicitly. |
| Engine raises on a missing key | Open Item #1 forces enumeration; AC #5 enforces strict-key check. |
| Reader on a headless server can't open `biomass.html` | Script always prints equilibrium values; troubleshooting table covers headless. |
| Stale state in `tutorial-work/` | Troubleshooting table: `rm -rf tutorial-work/`. |
| Markdown drifts from `_tutorial_config.py` | Discipline cue + `ast.parse` test + substring-find test cover syntactic + perturbation-instruction drift. Magnitude drift is caught at PR review. |
| Trophic-pyramid assertion fails on a stochastic transient | Windowed mean over years 45-50; margins from *measured* equilibrium. |
| Trophic-cascade assertion gives a weak signal | 2.0× / 0.6× margins force a strong cascade, not just-above-noise. |
| Implementer tunes parameters to pass the test | Open Item #5 commits the implementer to *measuring* first; parameter values trace back to OSMOSE-EEC defaults + Baltic precedent. |
| Tutorial rots as engine API evolves | Version stamp + regression test fail loud + CI catches it. |
| Single-seed fragility on 16-cell grid | seed=42 + windowed mean + ±20 % slack. A 3-seed `@pytest.mark.slow` test for nightly is a follow-up if seed bias proves real. |

## What the tutorial does NOT teach (deferred to follow-up docs)

- The multi-file CSV / properties format of real OSMOSE configs (→ `baltic_example.md`).
- Calibration (→ `scripts/calibrate_baltic.py` + calibration UI).
- Sensitivity analysis (→ `scripts/sensitivity_phase12.py`).
- Multi-fishery setups, discards, selectivity (→ `baltic_example.md` fishing section).
- Java engine usage and parity (→ `docs/parity-roadmap.md`).
- Spatial movement maps from real ecology (→ Baltic `maps/`).
- LTL forcing from real data (→ `mcp_servers/copernicus/`).
- Reproduction seasonality, density-dependent recruitment (B-H), genetics.
- Loading a model in the Shiny UI (→ follow-up tutorial covering `EngineConfig.from_dict().to_files()` → UI Setup → Load).

## Appendix — Engine reference points

Citations used in the spec, consolidated:

- `osmose/engine/__init__.py:87-115` — `run()` returns `RunResult`.
- `osmose/engine/__init__.py:117-149` — `run_in_memory()` returns `OsmoseResults`; no `output_dir` arg; fully in-process.
- `osmose/engine/__init__.py:66-68` — grid-key resolution.
- `osmose/engine/grid.py:118-122` — `Grid.from_dimensions(ny, nx)` creates all-ocean rectangular grid.
- `osmose/engine/config.py:1428-1500` — `EngineConfig.from_dict` mandatory-key walk.
- `osmose/engine/config.py:572-639` — predation pred/prey lowercase keys.
- `osmose/engine/config.py:1437-1438` — `simulation.time.nyear`, `simulation.time.ndtperyear` (lowercase).
- `osmose/engine/config.py:1908-1911` — `simulation.fishing.mortality.enabled` toggle.
- `osmose/engine/resources.py:109-220` — resource-species discovery + LTL NetCDF loader; variable name match at `:216`.
- `osmose/engine/accessibility.py:72-131` — accessibility CSV loader (`pd.read_csv(sep=';', index_col=0)`).
- `osmose/engine/output.py:94-116` — `_build_species_dataframes` (wide-form biomass).
- `osmose/results.py:185, 218-231` — `_build_dataframes_from_outputs` adds `species="all"` to wide-form cross-species outputs.
- `osmose/results.py:342-348` — `OsmoseResults.biomass()`.
- `data/baltic/predation-accessibility.csv` + `data/minimal/predation/accessibility_matrix.csv` — canonical accessibility CSV examples.
- `data/minimal/osm_all-parameters.csv` — canonical minimal-viable parameter set.
