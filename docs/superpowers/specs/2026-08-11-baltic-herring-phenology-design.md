# Herring spawning phenology under warming winters (spec C1, herring arm)

**Date:** 2026-08-11
**Status:** design, for review
**Grounded on:** `docs/baltic_recruitment_pathway_2026-08-10.md` (what the code does) and
`docs/baltic_thermal_recruitment_shape_2026-08-10.md` (what the literature supports). Every
mechanism claim below cites one of them or a verified file:line — the three specs withdrawn on
2026-08-10 all failed by asserting mechanisms instead.

## 1. Why phenology, not a thermal gate

Spec C1 originally meant "temperature-dependent stock–recruitment for cod and herring". The
literature check ruled out the obvious implementation:

* the existing `thermal_gate` applies a logistic **increasing** in temperature (built for percids);
  Voss & Quaas 2026 report warming **reducing** cod and herring productivity — no parameterisation
  flips a monotone curve;
* herring's documented mechanism is **not** a response curve at all. Polte et al. 2021
  (doi:10.3389/fmars.2021.589242): a 3.5–4.5 °C in-situ threshold triggers spawning onset; warming
  winters move it earlier, elongate the hatching window, and reduce larval production;
* cod's documented drivers (reproductive volume, egg predation, cannibalism) are already in the
  model, so a cod thermal gate would stack a weakly-evidenced multiplicative modifier on the
  already-dominant RV gate. **Cod is out of scope here.**

## 2. The insertion point exists and needs no engine change

`reproduction()` weights egg production by a per-step seasonality vector
(`reproduction.py:113-118`):

```
n_cols       = spawning_season.shape[1]
season_idx   = step % n_cols          # WRAPS (contrast: the RV gate CLAMPS)
season_factor = spawning_season[:, season_idx]
```

`_load_spawning_seasons` (`config.py:1022`) already accepts **multi-year** files — "n_columns
equals n_dt_per_year for single-year data, or n_dt_per_year * n_years for multi-year time series"
— read from `reproduction.season.file.sp{i}`, one value per step, expected to sum to 1.0 per year
(a non-unit sum warns, or is auto-rescaled under `reproduction.normalisation.enabled`).

Herring's current vector (`reproduction/reproduction-seasonality-sp1.csv`, 24 rows, sum 1.0) is
**bimodal**: a dominant spring window at steps 4–11 (peak 0.163 at steps 7–8, total weight 0.75)
and a minor autumn window at steps 16–21 (total 0.25). This matches the Baltic herring complex
(spring spawners dominant, autumn spawners minor) — and Polte's mechanism applies to the **spring**
component only.

So the feature is a **multi-year seasonality CSV**: pure data, exactly like C2(b) claimed to be,
except that here the insertion point has been read rather than assumed.

## 3. The mechanism must be shown to bite before a driver is built

Shifting the spring window earlier is only meaningful if the model *penalises* it. The candidate
pathway is match–mismatch: eggs hatch before the plankton bloom → low realised feeding → starvation.
That pathway exists — `update_starvation_rate` (`processes/starvation.py:41-59`) computes
`M_starv = M_max·(1 − S_R/C_SR)` from **this step's realised predation success**, with
`mortality.starvation.rate.max.sp1 = 0.3` — and the LTL forcing carries a real seasonal cycle.

**But herring also carries `mortality.additional.larva.rate.sp1 = 98.56`**, a calibrated pre-pass
mortality applied to eggs (`mortality.py:1827-1829`). If that term dominates, a food-timing signal
may be undetectable. **This is the pivotal uncertainty of the whole design, and it is cheap to
resolve.**

Hence the phased acceptance below: the mechanism test comes *first*, and a negative result ends the
work before any temperature series is derived.

## 4. Acceptance

* **A0 — mechanism detectability (first, cheap, decisive).** Construct three single-year variants
  of the herring vector: spring window shifted **−2 steps** (one month earlier), unshifted, and
  **+2 steps**, each renormalised to sum 1.0 with the autumn window untouched. Run 50 yr × 5 seeds
  via the harness as three arms. Report herring final-decade biomass, and — importantly — the
  realised **starvation mortality share** for herring from the mortality-by-cause output.
  **If the −2 and +2 arms differ from the unshifted arm by less than the 5-seed spread, the
  mechanism is not detectable in this configuration and the work STOPS**, recorded as a negative
  result. No temperature series is derived, no further compute is spent.
* **A1 — direction.** If detectable, the earlier-spawning arm must show *lower* herring biomass or
  *higher* larval starvation than the later arm. Polte's mechanism predicts earlier = worse. If the
  model says otherwise, the emergent behaviour contradicts the literature and that is the finding —
  report it rather than proceeding.
* **A2 — driver series.** Only then derive the multi-year vector: per year, shift the spring window
  by a step count driven by observed winter SST relative to the 3.5–4.5 °C onset threshold, using
  the CMEMS reanalysis already cached in-repo (`data/cmems_cache/`). State the shift rule
  explicitly (steps per °C) and its provenance. **The autumn window does not move** — Polte's
  finding is about the spring spawners.
* **A3 — series sanity, offline.** Every year sums to 1.0 (±1e-6); the shift stays within the
  window's support (no wrap into the autumn component); interannual variance in onset timing is
  non-zero — a constant series would silently reduce to the status quo, the same degeneracy that
  killed the computed-RV design.
* **A4 — certification.** 50 yr × 5 seeds, identity-pinned gate (5 assessed + perch +
  stickleback), off-arm PASS as precondition. Note herring sits 17.8% from its ceiling and sprat
  21.9% from its floor, so a herring change has room in both directions but propagates to sprat via
  competition — report both.

## 5. Out of scope

* **Cod** — see §1.
* The `thermal_gate` machinery: not used, not modified, stays disabled. If it is ever wanted for
  percids it is unaffected by this work.
* Autumn-spawning herring phenology (no comparable evidence retrieved).
* Any change to `mortality.additional.larva.rate.sp1` — if A0 shows it swamps the signal, that is
  the reportable finding, not a licence to lower it. It is a calibrated parameter that the whole
  configuration rests on.

## 6. Deliverables

1. `scripts/make_herring_phenology_seasons.py` — builds the A0 shifted variants and, on A1 PASS,
   the A2 driver series.
2. A0/A1 report (biomass + starvation-share deltas), committed either way.
3. On A1 PASS: the multi-year `reproduction/reproduction-seasonality-sp1-phenology.csv`, its
   config repoint, tests for A3, and the A4 certification record.
4. On A0 or A1 FAIL: the negative record, and nothing else.

## 7. Known risk, stated up front

The wrap-vs-clamp difference matters. A 30-row multi-year season file in a 50-year run replays
years 0–29, then 0–19 again (`step % n_cols`), unlike the RV gate which clamps to its final value.
That is defensible for a cyclic climate signal but means the scored final decade samples
**mid-series** years, not the warm tail. Any claim about "warming" behaviour in the scored window
must account for which years the wrap actually lands on — compute and state them explicitly rather
than assuming the run ends on the last series year.
