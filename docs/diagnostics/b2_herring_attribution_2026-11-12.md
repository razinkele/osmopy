# B2 herring attribution A/B: result, 2026-11-12

**Question:** B2's `rcp85_ref` arm (+2.9 °C through the C1 reproduction thermal gate) cut herring's
final-decade biomass by 62.27% in August and by 26.47% on v1.9.0. Which change halved the
response? The criterion was committed before any run:
`b2_herring_attribution_prestated_2026-11-11.md` (`f5774b66`).

**Answer: the halving happened at v1.9.0's data change (#236), and no tested single revert
inside it explains it.**
- **Restoring all of #236 restores 97% of the gap** between v1.9.0 and August: A4 gives −61.25%
  against August's −62.27%, which is 98.4% of the August response itself. A4 is v1.8.8 as
  released (identical code apart from an opt-in module, identical data). So the response was
  already at the August level at v1.8.8, and the August → v1.8.8 changes moved it by about
  1 pp net. August's noise is not known.
- **No tested single revert restores more than 27%.** The four sum to 0.26. Note that A5
  reverts two cod changes together, so offsetting effects inside it are not excluded.
- **That is an interaction, by the pre-stated rule.** The response belongs to the jointly fitted
  state, not to any one input.

**Post-hoc path.** Two arms, added after the pre-stated run, walk from A0 to A4:

| step | arm | decline | f | step vs previous |
|---|---|---:|---:|---|
| v1.9.0 as shipped | A0 | −26.47% | 0 | — |
| + herring larval rate **and** LTL file, together | A13 | −49.98% | **0.66** | z = −17.8 vs A0 |
| + herring's 25% autumn spawning share | A123 | −59.87% | **0.93** | z = −8.4 vs A13 |
| + the cod pair (= all of #236) | A4 | −61.25% | 0.97 | z = −2.6 vs A123 |

- **The two changes that were fitted together carry two thirds of the gap together.** Herring's
  larval rate and the food field it was re-fitted onto give 0.66. Apart, they give 0.27 + 0.06.
- **The autumn fold adds a quarter on top of that,** although alone it moves the response the
  other way (f = −0.09).
- **The cod pair adds 0.04.**
- **The shares are order-dependent.** With interactions this strong, they describe this path
  through the four changes and are not unique. The firm statements are the pre-stated ones: the
  gap is all in #236 (f = 0.97), and no single change explains it.

Harness: `scripts/b2_herring_attribution.py`. Raw output:
- `b2_herring_attribution_2026-11-11.json` (pre-stated arms, 60 runs, 2042 s);
- `b2_herring_attribution_posthoc_2026-11-12.json` (adds A13 and A123; 80 runs in all).

Each run stores every species' final-decade mean and herring's annual series. The full series
were dropped after the runs to keep the files small, and the analysis recomputed from the
compact form is identical.

## Gate

A0, the unchanged v1.9.0 config, gives **−26.469846771759054%**. That equals B2's `rcp85_ref`
decline to the last bit (difference 0.0). The harness runs one Numba thread per worker; B2 ran
serially and multithreaded. The two produce identical results, so the instrument is B2's own.
**PASS.**

## Pre-stated arms

The metric is B2's: the ratio of the 5-seed means of the final-decade herring mean, `rcp85_ref`
over the arm's own baseline. f = (D_arm − D_A0) / (D_Aug − D_A0). z is the difference from A0 over
the combined SE of the per-seed paired declines.

| arm | reverted to v1.8.8 | decline | SE | f | z vs A0 | baseline herring |
|---|---|---:|---:|---:|---:|---:|
| A0 | nothing | −26.47% | 0.73% | — | — | 1.53 Mt |
| A1 | herring larval rate sp1 | −36.07% | 0.20% | **+0.27** | −12.6 | 0.80 Mt |
| A2 | herring seasonality (25% autumn) | −23.38% | 0.69% | **−0.09** | +3.1 | 2.05 Mt |
| A3 | LTL forcing file | −28.66% | 0.51% | +0.06 | −2.4 | 3.97 Mt |
| **A4** | **all of #236** | **−61.25%** | 0.30% | **+0.97** | −43.9 | **2.70 Mt** |
| A5 | cod_east SD 25 map + cod_west larval sp0 | −27.26% | 0.37% | +0.02 | −1.0 | 1.53 Mt |

Per-seed paired declines are in the JSON. The largest within-arm range is 4.3 pp (A2, −21.25% to
−25.53%).

- **The pivot passes: f(A4) = 0.97 ≥ 0.8.** On the current code, reverting #236 restores the
  August-level response: the halving is the v1.8.8 → v1.9.0 step. This run does not test whether
  the August → v1.8.8 changes condition that step, for example by interacting with #236. A4's
  baseline herring, 2.70 Mt, is also the pre-recalibration level.
- **A1 moves the response** (z = −12.6): restoring herring's old larval rate alone restores 27%
  of the gap.
- **A2 moves it the other way** (z = +3.1). Restoring the 25% autumn spawning share makes herring
  *less* sensitive to warming: f = −0.09.
- **A3 moves it only slightly** (z = −2.4): restoring the old food field alone restores 6%.
- **A5 does not move it** (z = −1.0). The cod-only changes are inert for herring's warming
  response.
- **Interaction.** f(A1) + f(A2) + f(A3) + f(A5) = 0.26, against f(A4) = 0.97. The SEs are 0.2–0.7 pp
  on a 35.8 pp gap, so a shortfall of 0.71 is not noise.

## Not measured

- **Why the joint state is so much more sensitive.** The baselines show that a simple level effect
  is not the explanation. Herring's sensitivity does not track its biomass across the arms: −36%
  at 0.80 Mt, −26% at 1.53 Mt, −23% at 2.05 Mt, −61% at 2.70 Mt and −29% at 3.97 Mt. The mechanism
  was not measured here. The breakdown that followed (`herring_compensation_2026-11-12.md`) found
  the channel:
  - both states compensate through spawning biomass per age-1+ fish, about 4.6×;
  - survival per recruit is flat in both;
  - A0 reaches that rise at half the density reduction A4 needs (post-hoc secant 2.18 vs 1.00).
  Why remains open.
- **The noise on D_Aug.** The August report has no per-seed values, so the gap is treated as
  exact.

## What it means for B2

The halving is a consequence of the v1.9.0 recalibration as a whole. Along the tested path it
takes three steps:
- the corrected food field together with herring's larval re-fit onto it (0.66 of the gap);
- the autumn-spawning fold (a further 0.28);
- the cod pair (0.04).

These shares are order-dependent, and no tested single revert explains it. On the current, certified
model, herring's +2.9 °C response is −26.5%. The August −62% belonged to a herring stock calibrated
on a food field that overstated plankton about 2.5×. It is not the better number, only the older
one.
