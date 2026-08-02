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
> A real tolerance requires varying the factor directly — sweep `reproduction.rv.gate.ref`
> (150 → 130, 170, 190), which scales the whole trajectory — at ~3–4 additional runs. **Not yet done.**

What survives without any fit, and is sufficient for the Phase 0 conclusions:

* **cod_east's PASS is load-bearing on the gate** — gate-off puts it 1.97× over the ceiling.
* cod_east sits **2.2% under the envelope ceiling** with the gate on (direct measurement, no fit).
* That 2.2% is *inside* the ~1.9% seed-to-seed noise, so the PASS at the ceiling is **marginal**.

A dynamic RV that runs *higher* than the observed series therefore risks failing non-regression on the
high side. That remains the reason the Phase 0 criterion demands **in-sample agreement** rather than
merely "the arms differ" — the direction is established even though the magnitude is not.

## Limits — read before citing

* **The headroom is within seed noise.** 2.2% biomass headroom against ~1.9% seed-to-seed noise means
  cod_east's PASS at the ceiling is **marginal**, not comfortable. Any Phase 0 comparison at this
  boundary needs more than 2 seeds to resolve.
* **Two seeds, and the per-seed spread was not retained** — only the across-seed mean. The headline
  (≈2× effect) is far larger than seed noise and is safe; the 2.2% ceiling headroom is not.
* **No tolerance band is claimed** — see the retraction above. The two arms differ by an entire
  trajectory, not by a scalar factor, so biomass cannot be regressed on "mean factor" from these runs.
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
