# RV recruitment gate — mechanism A/B and the Phase 0 tolerance it implies

**Date:** 2026-08-02
**Config:** 9-species Baltic master, `--params current`, 50 yr, Python engine, seeds (42, 123)
**Motivated by:** the Phase 0 success criterion (#145, spec §5), which requires a mechanism
demonstration rather than an assumption from configuration.

## Question

Is the RV recruitment gate that is already on master for `sp8` (cod_east) doing anything, and if so
how much? This engine has twice shipped switches that silently no-op'd, so the gate's effect was
measured, not inferred.

## The gate is not inert — it is the dominant control on cod_east's certified equilibrium

Prescribed-series gate, `raw_cap`, `ref = 150`, offset 0. Factor by run window:

| window | mean factor | note |
|---|---|---|
| years 0–11 | **1.000** | inert; also the seeding bootstrap, where `reproduction.py:171` skips the gate anyway |
| years 12–39 | 0.695 | |
| years 40–49 | **0.438** | pinned at **0.320** from year 47 — the 2020 series minimum |

The gate bites *hardest* in the final decade, which is exactly the window certification scores.

Measured effect on cod_east biomass (ICES envelope **60,000–85,000 t**):

| arm | final-decade mean | whole-run mean | envelope |
|---|---|---|---|
| gate ON (master) | **83,135 t** | 113,941 t | **IN** |
| gate OFF | **167,377 t** | 149,601 t | **OUT** — 1.97× over the upper bound |

Ratio 0.497 against a mean factor of 0.438: the gate passes through to biomass close to
proportionally. Density dependence does **not** absorb it.

## The consequence for Phase 0: this is a high-risk swap, not a cosmetic one

**cod_east's in-envelope PASS is load-bearing on the gate.** Without it the stock sits at nearly
twice the envelope ceiling. Phase 0 proposes replacing the prescribed series with an RV *computed
from forcing* — so whatever that computation returns lands almost directly on cod_east's envelope
status.

~~Fitting the two measured points (`B ≈ 17,480 + 149,898 · factor`) gives the admissible band
0.284–0.450 for the final-decade mean factor, with the current 0.438 sitting 2.8% below the upper
breach point.~~

> **Retracted 2026-08-02, same day — the fit was invalid.** It regressed biomass on "mean factor"
> using two points that are not on the same curve: **0.438 is a time-average of a varying trajectory**
> (1.000 → 0.695 → 0.438, pinned at 0.320 from year 47), while **1.000 is a constant** — the gate-off
> arm sits at 1.0 in *every* year, including years 12–39 where the gate-on arm was at 0.695. Final-decade
> biomass reflects accumulated cohort history across years 12–49, not the final-decade mean factor
> alone, so the x-axis was not the variable being interpolated over. The earlier caveat ("two-point
> interpolation over a mildly sublinear response") named the wrong problem: the issue is not curvature.
>
> A real tolerance requires varying the factor directly — sweep `reproduction.rv.gate.ref`, which
> scales the whole trajectory. **Done; see the next section.** The valid band lands at 0.331–0.449,
> near the retracted 0.450 upper edge — which does not redeem the original fit. Landing close by luck
> is not deriving it correctly, and had the response been less linear over that span the retracted
> number would have been wrong.

What survives without any fit, and is sufficient for the Phase 0 conclusions:

* **cod_east's PASS is load-bearing on the gate** — gate-off puts it 1.97× over the ceiling.
* cod_east sits just under the envelope ceiling with the gate on (direct measurement, no fit).
  **Canonical figure: 82,968 t, 2.4% under, from the 3-seed sweep below.** The 83,135 t in the table
  above is the *same arm* measured with 2 seeds; the two differ by well under the 236 t seed spread.
* That headroom is **≈8.6 sd**, so the PASS is comfortable — see the retraction of my "marginal"
  claim under Limits.

A dynamic RV that runs *higher* than the observed series therefore risks failing non-regression on the
high side. That remains the reason the Phase 0 criterion demands **in-sample agreement** rather than
merely "the arms differ" — the direction is established even though the magnitude is not.

## The valid tolerance: sweep `reproduction.rv.gate.ref`

Varying `ref` generates a **one-parameter family** of trajectories, all built the same way
(`factor(y) = clip(rv[y]/ref, 0, 1)`). Within that family `ref → biomass` and `ref → final-decade mean
factor` are both well defined and monotone, so biomass-against-factor along it is a genuine curve —
unlike the retracted fit, which mixed a time-average against a constant. 3 seeds, per-seed values
retained.

| `ref` | final-decade mean factor | cod_east B | seed sd | envelope |
|---|---|---|---|---|
| 110 | 0.5791 | 109,023 t | 1.0% | OVER |
| 130 | 0.5054 | 95,913 t | 0.6% | OVER |
| **150 (shipped)** | **0.4380** | **82,968 t** | **0.3%** | **IN** |
| 170 | 0.3865 | 72,350 t | 0.3% | IN |
| 220 | 0.2986 | 52,843 t | 2.3% | UNDER |

Monotone in factor, and **both envelope edges are bracketed by actual runs** — this is interpolation,
not extrapolation.

**Admissible final-decade mean factor: ≈0.33 – 0.449.** Local slope near the shipped point is
≈1,920 t per 0.01 of factor.

**The two edges are not equally resolved.** The *upper* edge (0.449) interpolates between `ref=130` and
`ref=150`, both at 0.3–0.6% seed spread — it is tight, and it is the one that binds. The *lower* edge
is anchored by `ref=220`, whose 2.3% spread (51,192–54,106 t) straddles a 2,914 t range against a
60,000 t target, so **do not quote it to three decimals**: read it as "≥20% of margin below the shipped
point," not as 0.331.

### The constraint is strongly asymmetric — that is the Phase 0 headline

The shipped 0.438 sits **+2.4% below the upper breach** (0.449) but roughly **25% above the lower**.
A dynamic RV has ~10× more room to run *stronger* than the observed series than *weaker*. So a computed
RV that overestimates reproductive volume by even a few percent breaches the ceiling, while a
substantial underestimate is tolerated. **In-sample agreement matters mostly on the high side**, and a
Phase 0 implementation should be checked there first.

## Limits — read before citing

* **~~The headroom is within seed noise, so the PASS is marginal.~~ Retracted — measured, and it is
  not.** Seed spread for cod_east at the shipped `ref=150` is **0.3%** (236 t on 82,968 t, 3 seeds), so
  the 2,032 t headroom to the ceiling is **≈8.6 sd**. The PASS is comfortable. The ~1.9% figure came
  from earlier session notes on seed noise generally and does not hold for this species on this config.
  Noise does grow at the sweep extremes (1.0% at `ref=110`, 2.3% at `ref=220`), so the band edges are
  softer than its middle.
* The **2-seed A/B** above reported only across-seed means; the 5-point sweep uses 3 seeds and retains
  per-seed values.
* **Scope: the 9-species master only** (`cod_east` = sp8, envelope 60,000–85,000 t). The 8-species
  config is a different species and a different envelope (cod, 60,931–68,364 t); nothing here
  transfers to it.
* Single config, single parameter set. `abundance_by_age` is **unavailable on the in-memory path**
  (`biomassByAge` is available; `abundanceByAge` is not), so the recruitment-vs-RV correlation stated
  in the criterion was not evaluated here — only the biomass consequence. That is a capability gap,
  not a script bug: the correlation check needs either `biomass_by_age` or a disk-backed run.

## Note on the clamp

The clamp to the series terminal value (0.320, the 2020 minimum) is **intentional and its rationale
holds** — `recruitment_gate.py` documents that post-series years stay low (no major Baltic inflows
since) rather than cycling back to 1970s highs, and that clamping keeps the scored tail consistent
across run horizons. It is recorded here as context for Phase 0, **not as a defect**.
