# Baltic C4 re-run on v1.10.1: salinity-gate sensitivity arms, 2026-11-14

This re-runs `docs/baltic_c4_salinity_2026-11-12.md` (v1.10.0) on **v1.10.1** (`8a578f9e`), after
#255 changed sprat's maturity (age only, 1 yr) and re-solved SP1b's cod larval rates. Harness, arms
(ΔS = −1, −2, −3 PSU), seeds and labels are unchanged. Label 6 is still the pre-registered "cod_west
is an inert saturated null control", kept as a record and refuted again below. It is still a
mechanism characterisation, not a projection.

**Verdict: all gates pass, and both cod responses are again the clearest signals.**
- **cod_east** falls **13.5%** at −3 PSU (14.5× its seed SD). On v1.10.0 it fell 12.9% (12.9×).
- **cod_west** rises **9.1%** at −3 PSU (10.6×). On v1.10.0, +8.6% (7.7×).
- **The July regime-shift chain is still absent.** Stickleback and the percids stay at noise level.

**Run:** 5 arms × 5 seeds × 50 yr, in 3,959 s. Report:
`docs/diagnostics/baltic_c4_salinity_report_2026-11-14.json`.

**Gates:**
- builder zero-check, three-way load-through and ramp ordering pass on every arm;
- zero arm: 0 identity violations;
- 0 all-zero frames.

## Chain table: final-decade mean vs baseline

The distinguishability measure is max |Δ| over the baseline 5-seed SD (population SD), the same as
before.

| species | −1 PSU | −2 PSU | −3 PSU | baseline SD | max \|Δ\|/SD | v1.10.0 max \|Δ\|/SD |
|---|---:|---:|---:|---:|---:|---:|
| **cod_east** | −0.04% | −3.56% | **−13.45%** | 0.9% | **14.52×** | 12.91× |
| **cod_west** | −0.95% | +1.25% | **+9.13%** | 0.9% | **10.57×** | 7.72× |
| herring | +1.75% | +0.63% | +0.48% | 0.6% | 2.79× | 0.52× |
| sprat | −0.57% | +0.11% | −0.17% | 0.4% | 1.33× | 1.27× |
| flounder | −0.23% | −1.34% | −0.72% | 1.1% | 1.25× | 0.62× |
| perch | +0.43% | +2.27% | +1.53% | 2.2% | 1.03× | 0.20× |
| stickleback | −2.92% | −2.22% | +0.19% | 3.3% | 0.88× | 0.48× |
| pikeperch | −0.25% | −0.28% | +0.74% | 1.1% | 0.68× | 0.66× |
| smelt | −0.14% | +0.44% | −0.22% | 0.9% | 0.50× | 0.75× |

- **cod_east** responds mainly at −3 PSU, where exclusion from its juvenile and adult maps jumps (the
  instruments, identical to v1.10.0's, record the jump).
- **cod_west** is monotone and positive from −2 PSU.
- **Herring is the one new row above the ~1.8× line, and these runs do not establish a dose-response
  for it.** Its largest shift is +1.75% at −1 PSU, falling to +0.5% at −3 PSU: not monotone in ΔS.
  Its baseline seed SD is small (0.6%), so a small shift reads as large in SD units. Whether this is
  seed noise or an indirect effect of the salinity arms through the food web is not resolved here.
- **Everything else is below the ~1.8× line.**

## Instruments: identical to v1.10.0

The map instruments are **byte-identical** to the v1.10.0 run. #255 changed no map. So the cod
changes are not a change in what the salinity lever does to the maps; they come from something #255
changed (sprat's maturity, or the SP1b cod rates it re-solved), by a pathway not measured here.

## What changes for anyone citing C4

- **cod_east:** −13.5% at −3 PSU (14.5× SD); previously −12.9%.
- **cod_west:** +9.1% at −3 PSU (10.6× SD); previously +8.6%.
- **Unchanged:** the July chain is absent; the salinity lever does not touch cod_east's spawning map
  on the C2 config; label 6's inert-control prediction stays refuted.

The v1.10.0 note now carries a superseded line.
