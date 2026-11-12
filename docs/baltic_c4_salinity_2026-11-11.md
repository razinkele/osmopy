# Baltic C4 re-run on v1.9.0: salinity-gate sensitivity arms, 2026-11-11

> **Superseded 2026-11-12** by `docs/baltic_c4_salinity_2026-11-12.md` (v1.10.0, herring
> recalibration). cod_east now falls 12.9% at −3 PSU (12.9× SD) and cod_west rises 8.6% (7.7× SD); the
> July chain is still absent; the instruments are unchanged.

This re-runs `docs/baltic_c4_salinity_2026-08-30.md` on **v1.9.0** (`95baf49b`) with the same
harness, arms (ΔS = −1, −2, −3 PSU), seeds and labels. It is still a **mechanism
characterization, not a projection**. The August note's labels apply unchanged.

**Verdict: all gates pass again.**
- Builder zero-check, three-way load-through and ramp ordering pass on every arm.
- The zero arm is bit-identical to baseline: 0 violations across 5 seeds.
- There are 0 all-zero (map, frame) events, including at −3 PSU.

**The two August headlines, re-read:**
1. **The July regime-shift chain still does not reproduce.** Stickleback and the percids stay at
   seed-noise level at every lever.
2. **cod_west is still the largest *percentage* responder, but no longer the clearest signal.**
   It moves +1.5% / +7.0% / +18.7%, monotone, against +5.9% / +11.0% / +27.9% in August. In
   noise units that is 2.6× its baseline seed SD, down from 5.3×. **cod_east is now the clearest
   responder:** −6.5% at −3 PSU, 3.9× its SD, against −2.3% and 1.8× in August.

**Run:** 5 arms × seeds `[42, 123, 7, 999, 2024]` × 50 yr, in 3113 s (52 min; it was about 87 min).
Report: `docs/diagnostics/baltic_c4_salinity_report_2026-11-11.json`, written by
`scripts/baltic_c4_salinity_ab.py` with its report path redirected.

## Chain table: final-decade mean vs baseline

The distinguishability measure is August's: max |Δ| across the three arms over the baseline
5-seed SD (population SD, as in August). It is reproduced exactly from the August report: cod_west
5.33×, sprat 3.51×, cod_east 1.84×. The ~1.8× line is August's post-hoc convention.

| species | −1 PSU | −2 PSU | −3 PSU | baseline SD | max \|Δ\|/SD | August max \|Δ\|/SD |
|---|---:|---:|---:|---:|---:|---:|
| **cod_east** | +0.83% | +0.61% | **−6.45%** | 1.7% | **3.87×** | 1.84× |
| **cod_west** | +1.51% | +6.96% | **+18.66%** | 7.2% | **2.59×** | 5.33× |
| sprat | −0.50% | −0.75% | −0.98% | 0.7% | 1.34× | 3.51× |
| pikeperch | −0.82% | −1.82% | −0.44% | 1.5% | 1.21× | 0.88× |
| smelt | +0.83% | +0.92% | +0.69% | 1.1% | 0.85× | 0.46× |
| perch | +0.65% | +1.37% | +0.91% | 1.7% | 0.81× | 1.62× |
| stickleback | −0.21% | −2.38% | −3.24% | 4.7% | 0.69× | 0.84× |
| flounder | +0.28% | +0.54% | +0.16% | 1.5% | 0.35× | 0.86× |
| herring | +0.35% | +0.53% | +0.38% | 1.6% | 0.34× | 0.62× |

- **cod_west** is still monotone and still the largest relative change. Its baseline is noisier
  now (seed SD 7.2%, against 5.2%), and its −1 PSU response is inside the noise.
- **cod_east** moves only at −3 PSU. That is where exclusion from its juvenile and adult maps
  jumps, from 0.06% / 0.23% of cells at −1 and −2 PSU to 3.4% / 5.2% (instruments below). It is
  the same lever as in August, but the population response is 2.8× larger.
- **The secondary sprat signal is gone.** Sprat is still monotone (−0.5% / −0.75% / −1.0%) but at
  1.34× its SD, below the ~1.8× line. August had inferred, without measuring, that it came through
  cod_west predation. With cod_west's response smaller, a smaller sprat echo is consistent with
  that inference, but this run does not test it either.
- **Everything else stays under 1.3×.** That includes stickleback (−3.2% at −3 PSU, 0.69×) and
  the percids, so headline 1 stands.

## Instruments: identical to August except cod_east spawning

The map instruments come from the builder and the engine loader, with no simulation. They are
byte-identical to August for cod_west at every stage, and for cod_east's juvenile and adult maps.
For example, at −3 PSU cod_east's juvenile and adult `mean_dw` is −0.276 / −0.304, and its
`excluded_fraction` is 3.4% / 5.2%.

The one change is **cod_east spawning**. On every arm it now reads `mean_w_base` 1.0, `mean_dw`
0.0, `tv` 0.0, `excluded_fraction` 0.0 and `saturated_fraction` 1.0. In August, at −3 PSU, it read
`mean_dw` −0.280 and `excluded_fraction` 0.4%. The cause is C2, shipped in v1.9.0: it moved
cod_east's spawning map to 14 SD 25 cells, and every one sits where the gate is saturated
(w = 1) even at −3 PSU. **On the current config, freshening up to 3 PSU does not touch cod_east's
spawning distribution at all.** The gate reaches cod_east directly only through its juvenile and
adult maps. The same arms also shift cod_west's maps, and the two stocks share prey, so part of
cod_east's response can come through that interaction. This run does not separate the two.

**Not measured:** why cod_east's population response grew 2.8× while its juvenile and adult
instruments are unchanged and its spawning pathway dropped out. Its baseline also moved, from
65 kt to 78 kt, and every v1.9.0 input change touches it.

## What changes for anyone citing C4

- **cod_west at −3 PSU:** +18.7%, not +27.9%.
- **cod_east:** a clear −6.5% at −3 PSU, where August had a borderline −2.3%.
- **Sprat:** no longer a distinguishable secondary signal.
- **The salinity lever cannot move cod_east's spawning map** on the C2 config.
- **Unchanged:** "the July chain is absent", the instruments for every other map, and both
  loader-gap Stage-2 items.

The August note now carries a superseded line.
