# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✗ | ✓ | 4.71e+01 | [13623.167490394462, 15063.505577609434] |
| cod_east | ✗ | ✓ | 1.74e+01 | [82635.55664288499, 83364.89294014715] |
| herring | ✓ | ✓ | 1.00e+06 | [2565436.9653721577, 2616576.243130738] |
| sprat | ✗ | ✓ | 7.04e+04 | [1051155.6795747548, 1070012.1610680171] |
| flounder | ✗ | ✓ | 5.18e+02 | [39813.85153578645, 41189.69514252036] |
| perch | ✗ | ✓ | 1.58e+02 | [43541.69155599967, 46636.13512026587] |
| pikeperch | ✓ | ✗ | 3.01e+05 | [1339137.6211135923, 1461749.51076667] |
| smelt | ✓ | ✗ | 6.02e+05 | [676793.0815886308, 688088.2102327935] |
| stickleback | ✓ | ✓ | 1.04e+04 | [74742.00553012146, 80413.6770304197] |

**Python verdict: 2/9 persistent & in-envelope.** Not 9/9 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).

**Java cross-check: 1/8 persistent (single seed).** Survivor sets DIFFER with Python — Python ['herring', 'pikeperch', 'smelt', 'stickleback'], Java ['flounder', 'herring', 'perch', 'pikeperch', 'smelt', 'stickleback']. Coarse consistency check only (Baltic is not bit-equal cross-engine); a DIFFER is a flag to inspect, not an automatic failure.

### Java 4.4.1 per-species (single seed 42, final-decade mean)

| species | Java mean (t) | vs envelope | note |
|---|---|---|---|
| cod_west | **0** | this draw | seed-sensitive — alive (0.39 M) in the other Java run |
| cod_east | **0** | this draw | seed-sensitive — alive (0.89 M) in the other Java run |
| herring | 1,959,470 | in-envelope | holds cross-engine |
| sprat | **0** | extinct | collapses on Java (both runs) |
| flounder | 3,007,264 | ~30–150× over | prey release |
| perch | 2,236,237 | ~45–280× over | prey release |
| pikeperch | 8,291,256 | ~330× over | prey release |
| smelt | 3,983,369 | ~33× over | prey release |
| stickleback | 4,815,401 | ~10× over | prey release |

## What the cross-check establishes

**Engineering goal achieved.** Java 4.4.1 now *loads and runs the disaggregated 9-species config to
completion* (previously it aborted in `Simulation.init` on the underscored species name `cod_west`).
This required the Java-staging reconciliation pass (`osmose/java_config_reconcile.py`): strip `_`/`-`
from species/fishery names to match Java's own `Species.java` sanitization, dedup the duplicated
background-predator column, and make the fishery catchability/discards matrices structurally
consistent. The `--java` cross-check path is now unblocked for the disaggregated config.

**Scientific verdict: the mid-trophic prey explode on Java, robustly — and cod's fate is a
seed-sensitive knife-edge.** Two Java runs of this config (the production cross-check above, seed
un-pinned, and an earlier reconciliation smoke-run) agree within ~10 % on *every* species except cod:
flounder 3.01 M / 2.93 M, perch 2.24 M / 2.14 M, pikeperch 8.29 M / 8.60 M, smelt 3.98 M / 4.39 M,
stickleback 4.82 M / 4.71 M, herring 1.96 M / 1.88 M, sprat 0 / 0. Cod is the sole disagreement — 0 / 0
in this run, but 0.39 M (cod_west) and 0.89 M (cod_east) *alive* in the other. So the prey release is
**independent of cod** (identical in the run where cod is alive at ~0.9 Mt), and "both cods extinct on
Java" is **not a settled finding — it is one non-reproducible draw.**

*Reproducibility caveat.* `certify_java` builds the Java command without a seed parameter, so the
single Java column is one draw; the two runs differing only on cod is consistent with an unpinned
Java RNG. Pinning the Java seed is a follow-up (the Java column is currently indicative, not
reproducible).

**The robust driver of the prey explosion is the background predators, which the staging represents
far more weakly than the Python engine does** — a background-staging *fidelity* gap, not an artifact
of the name/matrix reconciliation pass:

1. **Access coefficient.** The committed `data/baltic/predation-accessibility.csv` has a Cormorant
   column but **no GreySeal column**, so the Python engine predates GreySeal at its `access_coeff=1.0`
   fallback (see `osmose/engine/processes/predation.py`). The C2 staging instead hands Java the
   authored `BG_ACCESS` values (0.1–0.4). Java's seals are much weaker consumers.
2. **Standing biomass — the forcing itself is ineffective on Java, not merely un-multiplied.** Both
   engines force background biomass from `species.file.spN` = `baltic_predator_biomass.nc` via the
   `ResourceForcing` path (`species.biomass.total.spN` / NetCDF; `osmose/engine/background.py:270`
   mirrors `ResourceForcing.java`). On Java the seal biomass is **not held at the forced ~4500 t —
   it decays monotonically (GreySeal 1503 → 30 t over the run)**, so the schools are eaten down to
   near-zero. Java logs no forcing warning, so the root cause is non-obvious (a multi-hour debug of
   the Java NetCDF/`ResourceForcing` background path). Compounding it, the Python-side
   `species.biomass.multiplier.sp16 = 2.0` (Cormorant) is keyed differently from Java's
   `ResourceForcing` multiplier (`species.multiplier.spN`), so even a working forcing would under-scale
   Cormorant on Java.

With top-down control from the background predators collapsing on Java, the prey field cascades upward
regardless of cod — the same apex-predator-release mechanism the disaggregation experiment documented.

## Ceiling: this cross-check cannot validate cod_east

Even with a perfect background-forcing fix, the cross-check **cannot validate its own subject.** The
cod_east fix is three levers — M 2.5→0.9, `reproduction.rv.gate.ref` 250→150, and the RV-gate
wrap→clamp — and **two of the three are the RV recruitment gate, a Python-engine-only feature Java
4.4.1 has no counterpart for** (`osmose/engine/processes/recruitment_gate.py`; Java silently ignores
`reproduction.rv.gate.*`). So Java runs materially different cod dynamics than the config being
certified; cod's Java fate is additionally seed-sensitive (see above). A *faithful* cross-check of
cod_east is therefore not achievable at any level of staging effort — the most a fully-faithful
background fix could buy is a **prey-only** comparison, explicitly conceding cod.

**Bottom line (decision recorded 2026-07-29).** The engineering goal is delivered: Java 4.4.1 runs the
disaggregated config to completion and the `--java` path is unblocked. Making the *result* faithful
would require debugging the broken Java background-biomass forcing, re-keying the multiplier, matching
the access-coef, and pinning a seed OSMOSE may not expose — multi-hour work whose ceiling is a
prey-only comparison, since cod_east is structurally unvalidatable cross-engine. **Per user decision
that work is not pursued; the cross-check is accepted as coarse/indicative, and the Python 5-seed
certification (cod_east 82.6–83.4 kt, in-envelope) remains the authoritative result.**
