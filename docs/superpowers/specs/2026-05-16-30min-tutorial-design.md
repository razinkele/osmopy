# 30-Minute Tutorial — Design Spec

> Status: **DRAFT**. Brainstormed 2026-05-16. Implementation plan to follow via `superpowers:writing-plans`.

## Purpose

Fill the gap between the README's `## Quick start` (which stops at "open localhost:8000") and the working knowledge a newcomer needs to actually *use* OSMOSE. A 30-minute, self-paced, single-session tutorial that takes a reader from "I installed it" to "I built and perturbed a working 3-species food web."

This is the **front door** of the project for anyone who isn't already familiar with ecosystem models. After it ships, the docs index in README.md will lead with it.

## Audience and success criteria

**Audience:** Python-fluent, broadly scientific, *not necessarily* an ecologist. Someone who has read the README, run `pip install -e ".[dev]"`, and wants to know what the simulator actually does.

**Reader is successful when they have:**
1. A `tutorial.py` they wrote (by paste) that runs `PythonEngine().run(...)` end-to-end.
2. A `biomass.html` plot showing three biomass trajectories converging into a stable food chain.
3. Made one parameter change (predator-on-Forager accessibility 0.8 → 0.1), re-run, and *observed* the trophic cascade in the plot.
4. A mental model of OSMOSE's core surface area: grid, time, species, LTL forcing, predation accessibility.
5. Pointers to the next surfaces (`baltic_example.md`, Shiny UI, calibration roadmap).

**Reader is NOT expected to:**
- Know fisheries science vocabulary (Linf, K, Q/B are explained inline once).
- Have a Java install (Python engine only).
- Have an internet connection during the tutorial (no Copernicus calls; no ICES API).

## Format and substrate

| Decision | Value | Rationale |
|---|---|---|
| Format | Markdown walkthrough with code blocks | Cheapest to write, matches project docs style, no new deps. |
| Substrate | Synthetic 3-species toy model authored from scratch | "Build" framing matches the backlog entry; avoids the literature-sourcing tax that took Baltic weeks. |
| Arc | Demo-first: paste → run → unpack → perturb | Engagement matters in a 30-min budget; the reader sees a plot at minute 5. |
| Config surface | Python dict literal | Self-contained, Pythonic, matches the `PythonEngine().run(config=cfg)` API. |
| Spatial scope | Tiny 4×4 grid + 1 LTL group via a tiny NetCDF the tutorial writes | Reader sees the full forcing pipeline including NetCDF I/O without paying for a 50×40 grid. |
| Plot library | plotly | Matches the Shiny UI's chart idiom; transfers when the reader later opens the UI. |
| Companion script | None | One markdown file. No `scripts/tutorial.py` parallel. |
| New `data/tutorial/` directory | No | All on-disk artifacts go to a reader-created `tutorial-work/` directory; nothing in-tree. |

## File layout (ship-list)

```
docs/tutorials/
  30-minute-ecosystem.md      ← the tutorial itself (~600–800 lines)
  README.md                   ← one-paragraph index for future tutorials
tests/
  _tutorial_config.py         ← canonical config dict (source of truth for the test)
  test_tutorial_3species.py   ← regression test pinning the tutorial's three promised observations
README.md                     ← MODIFIED: new "Getting started" pointer + new row at top of doc index table
```

**Out-of-band, not part of the implementation plan:**
- After merge: a `project_30min_tutorial_shipped.md` memory entry and a line in `MEMORY.md` under "Status + roadmap."

## The 30-minute narrative (six beats)

Wall-clock targets are aspirational for an attentive reader on a warm laptop. Numba JIT compile on first run adds ~20 s; the tutorial warns about this so the reader doesn't think it hung.

### Beat 1 — "Run something" (0–5 min)

A single self-contained code block (~70 lines, heavily commented in the dict, sparse comments in the I/O). The reader pastes it into `tutorial.py`, runs `python tutorial.py`, and sees `tutorial-work/biomass.html` they can open in a browser. Two sentences of framing before, two sentences after. No conceptual exposition yet — the reader's first contact with OSMOSE is *successful execution*, not theory.

### Beat 2 — "The grid and the clock" (5–10 min)

Refer back to specific lines of the script. Explain `grid.ncolumns=4`, `grid.nrows=4`, `simulation.nyear=30`, `simulation.ndt.per.year=24`. Why 4×4 is enough to learn from. Why 24 timesteps/year is the OSMOSE default. Why 30 years is enough to reach quasi-equilibrium on a model this small.

### Beat 3 — "The three fish" (10–15 min)

Walk through the species block. For each of sp0/sp1/sp2: `species.linf.spN`, `species.k.spN`, `species.lifespan.spN`, plus the egg-size and maturity fields the engine requires. Round-number values chosen so the reader sees the *shape*, not the literature. One paragraph aside on von Bertalanffy growth — what Linf and K mean.

### Beat 4 — "Plankton: the bottom of the food chain" (15–20 min)

Why LTL (low-trophic-level) forcing exists at all — OSMOSE doesn't model phytoplankton dynamically; the lowest trophic level has to come from exogenous forcing. Show the ~8-line xarray Dataset construction → `.to_netcdf()`. Constant biomass over space + time (100 t/cell). Note that real configs use CMEMS-derived forcing (pointer to `baltic_example.md`).

### Beat 5 — "Who eats whom" (20–25 min)

The predation accessibility matrix, the ecologically richest concept. Show the 4×4 table:

| prey \ predator | Predator | Forager | PlanktonEater |
|---|---:|---:|---:|
| Predator | 0 | 0 | 0 |
| Forager | **0.8** | 0 | 0 |
| PlanktonEater | **0.3** | **0.8** | 0 |
| Plankton (LTL) | 0 | 0.2 | **0.8** |

Bold entries form the trophic-chain backbone Plankton → PlanktonEater → Forager → Predator. The 0.3 entry models the Predator as a partial generalist. One paragraph on what accessibility *is* (not "encounter probability" — it's the fraction of prey biomass available to a predator given the encounter, factoring in vertical overlap, gut hardware, etc.).

### Beat 6 — "Perturb and watch the cascade" (25–30 min)

Reader changes ONE value: predator-on-Forager accessibility 0.8 → 0.1. Re-runs. Compares biomass plots side by side. Observes: Forager biomass climbs, PlanktonEater biomass falls. One paragraph framing this as *what ecosystem models do* — they reveal indirect effects that wouldn't fall out of three independent single-species models.

### Closing — "Where next" (~5 sentences)

Pointers to: `docs/baltic_example.md` (a real-world 8-species config with full provenance), the Shiny UI (`shiny run app.py`) where this same model could be loaded and explored interactively, and `docs/parity-roadmap.md` for readers curious about engine internals and Java parity.

## The synthetic model (concrete numbers)

### Three focal species

| sp  | name             | Linf (cm) | K     | lifespan (yr) | egg size (cm) | role |
|-----|------------------|----------:|------:|--------------:|--------------:|------|
| sp0 | Predator         | 100       | 0.15  | 20            | 0.20          | apex |
| sp1 | Forager          | 25        | 0.40  | 8             | 0.15          | mid-trophic |
| sp2 | PlanktonEater    | 10        | 0.70  | 4             | 0.10          | small + fast |

Function-named (not "cod/herring/sprat") so the reader's focus is the *role* in the food web, not real-world biology that the round-number parameters don't represent.

### One LTL group

- Name: `Plankton`
- Biomass: constant 100 t per ocean cell across all 24 × 30 = 720 timesteps
- File: `tutorial-work/ltl.nc` written by the tutorial's xarray block
- Trophic level: 1.0
- Size range: standard plankton range (0.0002–0.5 cm; the predator-prey size-ratio kernel reads this)

### Grid + movement

- 4 × 4 ocean cells, all habitable (mask = all 1s)
- Grid mask handed to the engine as a CSV (`tutorial-work/mask.csv`) referenced via `grid.mask.file`
- Movement maps: one per species, each a 4×4 array of 1s ("species can be anywhere"). Each map written as a CSV the tutorial writes alongside the mask.

### Other parameters

The dict carries every OSMOSE-required key with round, defensible values:
- `simulation.nyear=30`, `simulation.ndt.per.year=24`, `mortality.subdt=10`
- `fisheries.nfisheries=0` (no fishing — the tutorial's perturbation is predation, not extraction)
- Critical predation efficiency: 0.57 (the OSMOSE default from Shin & Cury 2004)
- Predator-prey size ratios: standard OSMOSE-EEC values per species
- Initial seeding biomass: 1000 t, 5000 t, 20 000 t for Predator/Forager/PlanktonEater (rough trophic-pyramid anchor)
- All other engine-required keys: minimum defensible values, comments only where non-obvious

## Code shape

### One script, one paste

The tutorial's main code block lives at the top of `30-minute-ecosystem.md` and is ~70 lines. The reader pastes it once into `tutorial.py`. Subsequent beats refer to specific *lines* of that script ("look at line 32 where we set `species.linf.sp0=100`...") rather than asking the reader to re-paste.

### Skeleton (illustrative — implementer authors the final source)

```python
# tutorial.py — 3-species OSMOSE ecosystem in 30 minutes
from pathlib import Path
import numpy as np
import xarray as xr
import plotly.express as px
from osmose.engine import PythonEngine

WORK = Path("tutorial-work"); WORK.mkdir(exist_ok=True)

# --- 1. Grid mask: 4×4 of all ocean ---
mask = np.ones((4, 4), dtype=int)
np.savetxt(WORK / "mask.csv", mask, fmt="%d", delimiter=";")

# --- 2. Movement maps: one per species, all "anywhere" ---
for name in ("predator", "forager", "planktoneater"):
    np.savetxt(WORK / f"map_{name}.csv", mask, fmt="%d", delimiter=";")

# --- 3. LTL forcing: 1 group, constant 100 t/cell ---
ltl = xr.Dataset(
    {"plankton": (("time", "y", "x"), np.full((30 * 24, 4, 4), 100.0))},
    coords={"time": np.arange(30 * 24), "y": np.arange(4), "x": np.arange(4)},
)
ltl.to_netcdf(WORK / "ltl.nc")

# --- 4. Build the config dict ---
config = {
    "grid.ncolumns": 4, "grid.nrows": 4,
    "grid.mask.file": str(WORK / "mask.csv"),
    "simulation.nyear": 30, "simulation.ndt.per.year": 24,
    "simulation.nspecies": 3,
    "species.name.sp0": "Predator",
    "species.linf.sp0": 100.0, "species.k.sp0": 0.15, "species.lifespan.sp0": 20,
    "species.name.sp1": "Forager",
    "species.linf.sp1": 25.0,  "species.k.sp1": 0.40, "species.lifespan.sp1": 8,
    "species.name.sp2": "PlanktonEater",
    "species.linf.sp2": 10.0,  "species.k.sp2": 0.70, "species.lifespan.sp2": 4,
    # ... predation accessibility, LTL wiring, movement maps wiring ...
    "predation.accessibility.matrix.sp1.sp0": 0.8,   # Forager eaten by Predator
    "predation.accessibility.matrix.sp2.sp0": 0.3,   # PlanktonEater eaten by Predator
    "predation.accessibility.matrix.sp2.sp1": 0.8,   # PlanktonEater eaten by Forager
    "predation.accessibility.matrix.plankton.sp1": 0.2,
    "predation.accessibility.matrix.plankton.sp2": 0.8,
    # ... ~25 more keys: efficiency, size ratios, init biomass, mortality, output ...
}

# --- 5. Run ---
result = PythonEngine().run(config=config, output_dir=WORK / "output", seed=42)

# --- 6. Plot ---
df = result.biomass.to_dataframe().reset_index()
fig = px.line(df, x="year", y="biomass", color="species",
              title="3-species ecosystem biomass over 30 years",
              template="plotly_white")
fig.write_html(WORK / "biomass.html")
print(f"Open: {WORK / 'biomass.html'}")
```

### Key shape decisions

- **Single file, single paste** — reader never manages state across rebuilds.
- **All on-disk artifacts under `tutorial-work/`** — easy to delete; nothing pollutes cwd.
- **Real config keys**, not pretend-API — reader who later opens `data/baltic/` recognizes the shape.
- **`fig.write_html()` not `fig.show()`** — works in headless terminals; print the path so the reader knows where to look.
- **Dense comments in the dict, sparse comments in the I/O** — the dict is what the reader *learns from*.
- **No try/except** — let the engine raise; if the tutorial breaks, the reader sees the real error.

## Regression test

### File: `tests/test_tutorial_3species.py`

Single test class, ~80 lines. Imports the canonical config from `tests/_tutorial_config.py`. Writes the LTL NetCDF + mask CSV + movement CSVs to a pytest `tmp_path`. Runs the engine. Asserts:

1. **The script runs to completion** — engine returns a result, output files exist, no warnings about unknown config keys (catches `validation.strict.enabled=warn` regressions).
2. **Trophic pyramid emerges** — at year 30: `biomass(PlanktonEater) > biomass(Forager) > biomass(Predator)`. The chain is structured, not random.
3. **Trophic cascade is visible** — re-running with predator-on-Forager accessibility = 0.1 yields *higher* Forager biomass than the baseline run.

### What the test does NOT assert

- Exact biomass values (engine vectorisation can shift these; trophic *ordering* is the load-bearing claim).
- Run time.
- Plotly output (test doesn't import plotly — keeps test deps minimal).

### Sync discipline

`tests/_tutorial_config.py` is the **source of truth** for the dict. The tutorial markdown's code block is a *copy* of it. When the implementer ships, they author the helper first, then transcribe into the markdown. The test file's docstring says: *"If the dict here changes, update docs/tutorials/30-minute-ecosystem.md to match."* Discipline cue, not enforcement. Alternative (parse markdown + `exec` the block) is fragile — rejected.

### Runtime budget

The test runs two simulations (baseline + perturbed) on a 4×4 × 30 yr × 24 dt × 3 species + 1 LTL config. Single run ~3 s once Numba is warm; two runs ~6 s plus pytest fixture overhead. Budget: **under 10 seconds** total on the CI box. The acceptance criteria allow up to 15 s of slack for slow runners. No `@pytest.mark.slow` marker — first-class regression test, not a slow integration test.

## Cross-references

### `README.md` changes (two edits)

1. After `## Quick start` section, add a one-sentence "Getting started" pointer:
   > *New to OSMOSE? The [30-minute ecosystem tutorial](docs/tutorials/30-minute-ecosystem.md) walks you through building, running, and perturbing a 3-species food web from scratch.*
2. Add a new row at the **top** of the `## Documentation index` table:
   > *| Learn OSMOSE by building a 3-species ecosystem in 30 min | [`docs/tutorials/30-minute-ecosystem.md`](docs/tutorials/30-minute-ecosystem.md) |*

### `docs/tutorials/README.md`

One paragraph:
> Hands-on tutorials. Self-contained, single-session. Each tutorial is a markdown walkthrough you read top-to-bottom with code you copy-paste into one Python file.
>
> - [30-minute-ecosystem.md](30-minute-ecosystem.md) — build, run, and perturb a 3-species food web. Start here.

### Memory (deferred — not part of implementation plan)

After merge: a new memory file `project_30min_tutorial_shipped.md` + one line in `MEMORY.md` under the existing "Status + roadmap" pointers. Recorded as a manual post-merge step, not a planned task.

## Out of scope (explicit)

- No new dependencies. `xarray`, `plotly`, `numpy`, `numba` are already in the project.
- No Jupyter notebook (`.ipynb`).
- No CI matrix changes — the regression test runs in the existing pytest suite.
- No `data/tutorial/` directory in-tree.
- No screenshots or embedded images.
- No `docs/getting-started/` separate landing page (the tutorial *is* the landing page).
- No follow-up tutorials in this round.
- No translations, no PDF export.
- No Shiny UI screenshots or walkthrough (the closing paragraph mentions the UI as a next step but doesn't tour it).
- No companion `scripts/tutorial.py` — Markdown is the only ship surface for the reader.

## Open items for implementer (verify during planning, not blockers)

1. **Verify the exact `predation.accessibility.matrix.{prey}.{predator}` key format** against `data/baltic/predation-accessibility.csv` and `osmose/engine/processes/predation.py`. The spec uses dot-separated keys for illustration; the canonical format may differ.
2. **Verify `OsmoseConfigReader` / `EngineConfig.from_dict` accepts `grid.mask.file` pointing at a CSV without a companion NetCDF.** Inspect `osmose/engine/grid.py`. If not, fall back to writing a tiny grid NetCDF alongside the LTL NetCDF.
3. **Verify the LTL NetCDF variable name + dimension order** (`time, y, x` vs `time, x, y` vs `time, lat, lon`) by reading `data/baltic/baltic_ltl_biomass.nc` or `osmose/engine/resources.py`. Adjust the xarray Dataset construction accordingly.
4. **Verify `result.biomass` is the API** — check `osmose/results.py` and the `PythonEngine.run()` return shape. May need to be `result.get_biomass()` or read from disk.
5. **Verify Numba JIT compile message** — exact wording the reader sees on first run, so the warning in the tutorial matches reality.
6. **Verify `config_validation.strict.enabled` defaults** — the spec assumes silent-default. Confirm against `osmose/engine/config_validation.py` so the tutorial doesn't get an unexpected warning storm.
7. **Verify `fisheries.nfisheries=0` is actually accepted.** The engine may require certain fishing-related keys (selectivity, F rates) to exist even when set to zero. If so, the dict must include them as no-op stubs. Inspect `osmose/engine/processes/fishing.py` and `osmose/engine/config_validation.py`.
8. **Verify 30 yr is long enough** on a 4×4 grid with these parameter values to (a) reach quasi-equilibrium and (b) make the perturbation cascade visible. The regression test's assertions catch failures; if 30 yr is insufficient, bump to 50.

## Acceptance criteria

A reviewer should be able to verify the tutorial shipped correctly by:

1. **Following the tutorial end-to-end** in a fresh venv with `pip install -e ".[dev]"`. Must produce `tutorial-work/biomass.html` showing three diverging biomass lines that converge into a stable food chain by year 30.
2. **Performing the perturbation** in beat 6 and observing the cascade in the re-run plot.
3. **Running the regression test** — `.venv/bin/python -m pytest tests/test_tutorial_3species.py -v` passes in under 15 seconds.
4. **Following every README link** — all hrefs resolve.
5. **Reading time:** prose alone (excluding paste-and-run wait + reading the dict) should be ~20 minutes for a Python-fluent reader. The 30-minute total includes the runs.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Numba JIT compile is slow on first run; reader thinks it hung | Tutorial explicitly warns: "First run is slow because Numba JIT compiles ~20 s of native code. Subsequent runs are <2 s." |
| Engine raises on a missing config key (e.g., the dict skeleton doesn't cover everything OSMOSE requires) | "Open items" #1–6 above force the implementer to verify against the real engine before writing the dict. The regression test catches it on every commit. |
| Reader on a headless server can't open `biomass.html` | Tutorial includes one fallback: `print(df.tail())` to confirm trajectories numerically. |
| Reader's `tutorial-work/` collides with an existing directory | Tutorial says "delete `tutorial-work/` before re-running if you want a clean slate." Not enforced; the engine writes into `output/` subdir which is overwritten anyway. |
| Markdown code block drifts from `tests/_tutorial_config.py` | Discipline cue in test docstring. Drift will manifest as the tutorial failing to run; CI catches the test side. Manual reconciliation cost: 5 min on any change. |
| Tutorial-promised parameter (e.g., accessibility 0.8) doesn't actually produce a visible cascade with our random-number values | The regression test's "trophic cascade is visible" assertion catches this. Implementer iterates the values during authorship until the test passes. |

## What the tutorial does NOT teach (deferred to follow-up docs)

- Reading multiple CSVs as the canonical OSMOSE config format (pointer to `baltic_example.md`).
- Calibration (pointer to `scripts/calibrate_baltic.py` + the calibration UI).
- Sensitivity analysis (pointer to `scripts/sensitivity_phase12.py`).
- Multi-fishery setups, discards, selectivity (pointer to baltic_example.md fishing section).
- Java engine usage and parity testing (pointer to parity-roadmap.md).
- Spatial movement maps with actual ecology (pointer to baltic maps/).
- LTL forcing from real data (pointer to `mcp_servers/copernicus/` + baltic_example.md LTL section).
