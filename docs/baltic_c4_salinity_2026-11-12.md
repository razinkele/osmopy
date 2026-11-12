# Baltic C4 re-run on v1.10.0: salinity-gate sensitivity arms, 2026-11-12

This re-runs `docs/baltic_c4_salinity_2026-11-11.md` (v1.9.0) on **v1.10.0** (`05b517f3`), after
#250's herring recalibration (maturity 14.5 cm, adult additional mortality 0.2 /yr). Harness, arms
(ΔS = −1, −2, −3 PSU), seeds and labels are unchanged. It is still a mechanism characterisation, not
a projection.

**Verdict: all gates pass, and both cod responses are now much clearer signals.**
- **cod_east** falls **12.9%** at −3 PSU (12.9× its seed SD). On v1.9.0 it fell 6.5% (3.9×).
- **cod_west** rises **8.6%** at −3 PSU (7.7×). That is smaller in percentage than v1.9.0's +18.7%,
  but its baseline seed noise fell from 7.2% to 1.1%.
- **The July regime-shift chain is still absent.** Stickleback and the percids stay at noise level.

**Run:** 5 arms × 5 seeds × 50 yr, in 4,434 s. Report:
`docs/diagnostics/baltic_c4_salinity_report_2026-11-12.json`.

**Gates:**
- builder zero-check, three-way load-through and ramp ordering pass on every arm;
- zero arm: 0 identity violations;
- 0 all-zero frames.

## Chain table: final-decade mean vs baseline

The distinguishability measure is max |Δ| over the baseline 5-seed SD (population SD), the same as
before.

| species | −1 PSU | −2 PSU | −3 PSU | baseline SD | max \|Δ\|/SD | v1.9.0 max \|Δ\|/SD |
|---|---:|---:|---:|---:|---:|---:|
| **cod_east** | +0.78% | −2.34% | **−12.94%** | 1.0% | **12.91×** | 3.87× |
| **cod_west** | +0.60% | +2.88% | **+8.56%** | 1.1% | **7.72×** | 2.59× |
| sprat | −0.96% | −0.96% | +0.05% | 0.8% | 1.27× | 1.34× |
| smelt | −0.01% | +0.04% | −0.29% | 0.4% | 0.75× | 0.85× |
| pikeperch | +0.51% | +0.45% | +0.58% | 0.9% | 0.66× | 1.21× |
| flounder | +0.04% | −0.84% | −0.78% | 1.4% | 0.62× | 0.35× |
| herring | −0.68% | +0.87% | +0.46% | 1.7% | 0.52× | 0.34× |
| stickleback | +3.11% | −0.52% | +1.83% | 6.5% | 0.48× | 0.69× |
| perch | −0.19% | +0.35% | +0.49% | 2.5% | 0.20× | 0.81× |

- **cod_east** responds mainly at −3 PSU, where exclusion from its juvenile and adult maps jumps.
  Its response doubled relative to v1.9.0.
- **cod_west** is still monotone and positive.
- **Sprat and everything else are below the ~1.8× line.**

## Instruments: identical to v1.9.0

The map instruments are **byte-identical** to the v1.9.0 run. #250 changed no map, and cod_east's
SD 25 spawning map is still fully saturated, un-gated at every offset. So the stronger cod responses
come from the food web after the herring recalibration, not from a change in what the salinity lever
does to the maps.

**Why they changed is not measured.** Herring is cod food, and its biomass structure is now mostly
adult. The two stocks also share prey, and the arms shift both cod maps, so part of either stock's
response can come through the other.

## What changes for anyone citing C4

- **cod_east:** −12.9% at −3 PSU, a strong signal; previously −6.5%.
- **cod_west:** +8.6% at −3 PSU, a clearer signal; previously +18.7% with more noise.
- **Unchanged:** the July chain is absent; the salinity lever does not touch cod_east's spawning map
  on the C2 config; and both loader-gap Stage-2 items.

The v1.9.0 note now carries a superseded line.
