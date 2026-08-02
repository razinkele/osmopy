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

Fitting the two measured points (`B ≈ 17,480 + 149,898 · factor`) gives the admissible band for the
final-decade mean factor:

* **0.284 – 0.450** keeps cod_east in envelope.
* The current prescribed factor, **0.438, sits 2.8% below the upper breach point** — and cod_east
  sits **2.2% under the envelope ceiling**.

A dynamic RV that returns a final-decade mean factor even a few percent *higher* than the prescribed
series pushes cod_east out of envelope on the high side. This is the quantitative reason the Phase 0
criterion demands **in-sample agreement** with the observed series rather than merely "the arms
differ."

## Limits — read before citing

* **The headroom is within seed noise.** 2.2% biomass headroom against ~1.9% seed-to-seed noise means
  cod_east's PASS at the ceiling is **marginal**, not comfortable. Any Phase 0 comparison at this
  boundary needs more than 2 seeds to resolve.
* **Two seeds, and the per-seed spread was not retained** — only the across-seed mean. The headline
  (≈2× effect) is far larger than seed noise and is safe; the tolerance band is not measured at that
  precision.
* **The band is a two-point linear interpolation** between factor 0.438 and 1.000. The true response
  is mildly sublinear over that span (proportional scaling from the gate-off arm would predict
  73.3 kt at factor 0.438; the measured value is 83.1 kt), so treat 0.284–0.450 as indicative, not
  exact. The upper edge is the decision-relevant one and is the better-anchored of the two.
* Single config, single parameter set. `abundance_by_age` is unavailable in-memory, so the
  recruitment-vs-RV correlation stated in the criterion was not evaluated here — only the biomass
  consequence.

## Note on the clamp

The clamp to the series terminal value (0.320, the 2020 minimum) is **intentional and its rationale
holds** — `recruitment_gate.py` documents that post-series years stay low (no major Baltic inflows
since) rather than cycling back to 1970s highs, and that clamping keeps the scored tail consistent
across run horizons. It is recorded here as context for Phase 0, **not as a defect**.
