# Fishery-Induced Evolution on Baltic Cod (Plumbing Demo)

> **Implementation status (2026-05-21).** The genetics plumbing — bioen
> integration, trait expression, per-step output, paired-scenario demo
> script, regression test — is shipped and tested. The `baltic_ev/`
> fixture sustains a cod population but exhibits boom-bust dynamics that
> currently prevent a measurable FIE signal from emerging in 50 model-years.
> The Task 11 regression test stands as a guard for future calibration
> sprints. Read this tutorial as a guide to the *mechanism* and how to
> *run* the demo; the *result* shown by the chart should be interpreted
> as a working plumbing test with deferred calibration, not as a peer-
> ready scientific finding.

**Time:** ~30 minutes (15 min reading + 15 min compute)

## What is FIE?

Fisheries-Induced Evolution (FIE) is the heritable response of fish
populations to selective harvesting. When large, fast-growing fish are
removed preferentially, the surviving population skews toward slower-growing
phenotypes — and because growth has a heritable component, that skew
propagates to offspring.

Classic evidence for *maturation-trait* FIE: Northern cod (Olsen et al.
2004). Direct evidence for *growth-rate* FIE is rarer — Heino, Pauli &
Dieckmann (2015) note that in the wild, most reported growth changes in
cod are confounded with concurrent maturation evolution; growth-FIE has
only been cleanly isolated in lab common-garden experiments on silversides
(Conover & Munch 2002; Walsh et al. 2006). This demo isolates the
growth-rate pathway because the dominant maturation pathway is held
constant (see Caveat #3).

## What the demo does

Two paired scenarios on the `baltic_ev/` fixture (a clone of the calibrated
Baltic config with bioenergetics + Ev-OSMOSE genetics enabled for cod):

| Scenario | Cod fishing mortality |
|---|---|
| `baltic_ev_high_f` | F = 0.6/yr (modern Baltic level) |
| `baltic_ev_low_f` | F = 0.1/yr (low-but-not-unfished reference) |

Each scenario runs 200 model-years × 3 seeds. The demo collects the mean
cod `imax` trait per timestep, then plots both trajectories with a ±1 SD
ribbon.

An optional third arm at F=0 is available via `--with-zero-f-control`
for a true drift-only baseline (doubles wall-clock).

## Running

```bash
.venv/bin/python scripts/run_fie_demo.py
```

Takes ~15–25 min on commodity hardware. Outputs:

- `outputs/fie_demo/fie_imax_trajectory.png` — the chart
- `outputs/fie_demo/<scenario>/seed<n>/osm_genetic_trait_means_Simu0.csv` — raw per-step trait stats

The script also prints an "imax-binding fraction" diagnostic per scenario.
If the binding fraction is < 30%, the trait is not the limiting constraint
on cod growth and the FIE signal will be drift-dominated.

### CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--n-years` | 200 | Simulation length in model-years |
| `--seeds` | 3 | Number of replicate seeds per scenario |
| `--output-dir` | `outputs/fie_demo` | Root directory for outputs |
| `--with-zero-f-control` | off | Add F=0 drift-only arm (doubles run time) |

## Interpretation

When you regenerate the chart, you should expect (under the current
fixture) that the high-F and low-F trajectories are essentially
indistinguishable — both stay close to the initial mean of `imax=3.0`.
This is **not** a scientific result; it is a known limitation of the
present `baltic_ev` fixture (see Caveats #7 and #8 below). The
*plumbing* — trait expression, inheritance, per-step output, selective
fishing — is exercised end-to-end and produces correct CSV outputs.

For comparison, the literature expects:

- **Olsen et al. (2004):** Northern cod showed measurable **maturation** trait
  shifts over ~3 cod generations under intense fishing (age + size at
  maturation declined; this is a different trait surface than `imax`
  growth-rate evolution).
- **Conover & Munch (2002):** silverside lab experiments measured a ~25%
  reduction in length-at-age (and ~40% reduction in weight-at-age, per
  Audzijonyte et al. 2013) over 4 generations under intense (90%/generation)
  size-selective removal. That is an experimental upper bound; Audzijonyte
  et al. (2013, https://doi.org/10.1111/eva.12044) puts modelled FIE
  growth-rate responses at moderate F (≈0.5–1.0/yr) at 0.02–0.93% per year.

## Exercise

Edit `data/baltic_ev/baltic_ev_param-genetics.csv` and try:

- `evolution.trait.imax.var.sp0=0.072` (h² ≈ 0.57, near the upper end of
  Otterå et al. 2018's cod body-weight range) — how much faster does the
  trait respond?
- `evolution.trait.imax.nlocus.sp0=5` (halve the loci count from Marty
  et al. 2015's 10) — does reduced polygenicity speed up or slow down the
  response? (Compare to Diaz Pauli & Heino 2014 on architecture sensitivity.)

You can also add a true unfished baseline:

```bash
.venv/bin/python scripts/run_fie_demo.py --with-zero-f-control
```

This adds a third F=0 arm (drift-only) at the cost of doubling wall-clock.

## Caveats

This demo emphasizes **direction of response, not absolute biomass**.
The bioenergetics parameters in `baltic_ev_param-bioen.csv` are
literature-default placeholders — the fixture is not calibrated against
ICES assessments. See `data/baltic_ev/README.md` for parameter provenance.

Eight modelling choices to be explicit about:

**Caveat 1: Size-selective fishing is the selection knob.**
Baltic's default `baltic_param-fishing.csv` uses age-knife-edge selectivity
for the cod fishery; this demo overrides it to length-sigmoidal (l50 = 35 cm,
matching the EU minimum landing size). Without size-selectivity, imax-FIE
cannot emerge: cod of any growth rate at a given age are equally vulnerable.

**Caveat 2: No thermal forcing.**
`simulation.bioen.phit.enabled` is explicitly set to `false` — bioen runs
thermally neutral. Adding temperature forcing (e.g., a Copernicus SST
time-series) is a future extension.

**Caveat 3: Ingestion-cap pathway only.**
This demo evolves only `imax` (intake-rate cap) targeting `bioen_i_max`.
Maturation is set to a flat threshold (m0 = 30 cm for cod, between the
Radtke & Grygiel 2013 + Svedäng 2024 estimates; see `data/baltic_ev/README.md`),
so FIE does **not** operate through age/size-at-maturation evolution — only
through the indirect "fast growers cross the gear's size threshold sooner"
pathway. The dominant FIE pathway documented in real cod stocks IS maturation
evolution (Olsen et al. 2004; Heino, Pauli & Dieckmann 2015) — this demo
intentionally isolates the secondary (growth-rate) pathway. The multi-trait
extension targeting `bioen_m0` is listed in the spec's out-of-scope
follow-ups.

**Caveat 4: h² ≈ 0.25 is borrowed from cod body-weight studies.**
Nielsen et al. (2014) is the source. No published h² estimate exists for fish
ingestion-rate as a standalone trait. A referee will fairly ask whether
body-weight h² (which integrates intake + assimilation + metabolic efficiency
+ activity) overstates ingestion-rate h². The demo's deliverable is the
direction of trait response, not the magnitude — a future sensitivity sweep
over h² ∈ {0.05, 0.15, 0.25, 0.40} would bracket this uncertainty.

**Caveat 5: Selection window is ~8 generations across 50 y, not 10.**
`evolution.seeding.year=10` configures the first 10 years as a "seed phase"
where offspring genotypes are randomly redrawn from population donors (per
`inheritance.py:61–68`), erasing any selection signature. Real allele-frequency
response only starts at year 10, leaving ~40 y / ~5 y-per-generation ≈ 8
selecting generations for the FIE-direction test, and ~38 generations for the
200 y demo.

**Caveat 6: F=0.1 "low-F" arm still applies meaningful selection.**
Per Audzijonyte et al. (2013, https://doi.org/10.1111/eva.12044) and Andersen
& Brander (2009, https://doi.org/10.1073/pnas.0901690106), modelled growth-rate
FIE response at moderate F (0.5–1.0/yr) clusters at 0.02–0.93% per year (mean
0.25%/yr). Over ~8 selecting generations the cumulative response is 1–4% in
the high-F arm, ~one-third of that in the low-F arm, so the paired contrast is
~0.7–2.7%. The cleanest reference would be F=0.0; add it via
`--with-zero-f-control` (doubles wall-clock). This demo's 2-arm default trades
the cleanest contrast for keeping low-F closer to a real management target.

**Caveat 7: Population dynamics are boom-bust under the current calibration.**
Cod biomass over 50 y under no fishing oscillates substantially (observed
year 41: 16 M t, year 44: 40 k t, year 49: 39 M t). At any given evaluation
year, the population may be in a peak or a trough phase. With fishing applied,
the cycle phase shifts and the year-50 evaluation can land on a
low-adult-biomass trough — producing a structurally-null FIE selection
differential regardless of trait variance. The current
`test_fie_demo_direction.py` reports `drop_pct ≈ 0.06%` at year 50 (well
below the 2% threshold). This is **a calibration limitation, not a plumbing
failure** — every genetics-pipeline component (expression → inheritance →
bioen override → fishing → output) is verified independently by the unit
tests added in Tasks 1–9. Producing a measurable FIE signal will require a
separate calibration sprint, likely involving one or more of:

- Damping the boom-bust cycle by tuning `seeding_biomass`,
  `population.seeding.year.max`, and the predation-accessibility matrix
  together for trophic balance
- Sourcing real Brander (1995) / Mehner & Wieser (1994) bioen parameter
  values for Baltic cod to replace the current placeholders
- Evaluating FIE on a windowed average (years 40–50) rather than a single
  year, to reduce cycle-phase sensitivity

**Caveat 8: Four engine bugs were found and fixed during this work.**
Bioen seeding fallback, egg-starvation guard, inheritance allele-pool fallback
for empty seed-phase populations, and `is_egg` recomputation in
`_bioen_reproduction`. All 99/99 pre-existing bioen + Java-parity tests still
pass with these fixes. See the commit history for details.

## Further reading

- Olsen, E. M. et al. (2004). Maturation trends indicative of rapid evolution
  preceded the collapse of northern cod. *Nature*, 428, 932–935.
  https://doi.org/10.1038/nature02430
- Heino, M., Pauli, B. D., & Dieckmann, U. (2015). Fisheries-induced evolution.
  *Annual Review of Ecology, Evolution, and Systematics*, 46, 461–480.
  https://doi.org/10.1146/annurev-ecolsys-112414-054339
- Conover, D. O., & Munch, S. B. (2002). Sustaining fisheries yields over
  evolutionary time scales. *Science*, 297, 94–96.
  https://doi.org/10.1126/science.1074085
- Walsh, M. R. et al. (2006). Maladaptive changes in multiple traits caused by
  fishing: impediments to population recovery. *Ecology Letters*, 9, 142–148.
  https://doi.org/10.1111/j.1461-0248.2005.00858.x
- Andersen, K. H., & Brander, K. (2009). Expected rate of fisheries-induced
  evolution is slow. *PNAS*, 106, 11657–11660.
  https://doi.org/10.1073/pnas.0901690106
- Audzijonyte, A. et al. (2013). Ecological consequences of body size
  decline in harvested fish species: positive feedback loops in trophic
  interactions amplify human impact. *Evolutionary Applications*, 6, 585–595.
  https://doi.org/10.1111/eva.12044
- Marty, L., Dieckmann, U., & Ernande, B. (2015). Fisheries-induced neutral
  and adaptive evolution in exploited fish populations and consequences for
  their adaptive potential. *Evolutionary Applications*, 8, 47–63.
  https://doi.org/10.1111/eva.12220
- Brander, K. M. (1995). The effect of temperature on growth of Atlantic cod
  (*Gadus morhua* L.). *ICES Journal of Marine Science*, 52, 1–10.
  https://doi.org/10.1016/1054-3139(95)80010-7
