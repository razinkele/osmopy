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
2. **Standing biomass.** The staging inlines the raw NetCDF biomass; the Python-side
   `species.biomass.multiplier.*` scaling is not applied on Java, so GreySeal/Cormorant stand at
   ~5–31 t in the Java run vs the multiplier-scaled levels Python uses. Weaker still.

With top-down control from the background predators largely removed, the prey field on Java cascades
upward regardless of cod — the same apex-predator-release mechanism the disaggregation experiment
documented. Two further Python-vs-Java differences compound the mismatch but are not the primary
driver here: the **RV recruitment gate is Python-engine-only** (`reproduction.rv.gate.*` has no Java
counterpart, so the cod_east recruitment lever does not exist on Java), and the RNGs differ (PCG64 vs
MT19937) — Baltic was never bit-equal cross-engine.

**Bottom line.** The engineering goal is delivered: Java 4.4.1 runs the disaggregated config to
completion, and the `--java` cross-check path is unblocked. But the cross-check is not yet a faithful
apples-to-apples comparison — the background-predator staging under-represents Python's seal/cormorant
control, and the Java seed is un-pinned. Closing both (match the access-coef + apply the biomass
multiplier in staging; pin the Java seed) is the path to a trustworthy cross-engine number. **Until
then the Python 5-seed certification remains the authoritative result.**
