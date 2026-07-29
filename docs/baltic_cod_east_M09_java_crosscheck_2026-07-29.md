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
| cod_west | **0** | extinct | collapses on Java |
| cod_east | **0** | extinct | the Python-tuned fix does NOT transfer |
| herring | 1,959,470 | in-envelope | holds cross-engine |
| sprat | **0** | extinct | collapses on Java |
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

**Scientific verdict: the disaggregated, cod-tuned config is strongly Python-engine-specific — it does
not replicate on Java.** On Java both cod stocks and sprat collapse to zero while the mid-trophic prey
(flounder, perch, pikeperch, smelt, stickleback) explode 10–330× over envelope — a textbook trophic
release. Three inherent Python-vs-Java differences drive this, none of them artifacts of the staging
pass:

1. **The RV recruitment gate is Python-engine-only.** `reproduction.rv.gate.*` (the lever the cod_east
   fix tunes, `osmose/engine/processes/recruitment_gate.py`) has no counterpart in Java 4.4.1, which
   silently ignores those keys and uses its native recruitment. The cod_east dynamics that were
   hand-tuned on Python simply are not the dynamics Java runs.
2. **Background-predator forcing transfers at reduced strength.** The C2 staging inlines the raw
   NetCDF standing biomass; the Python-side `species.biomass.multiplier.*` scaling is not applied on
   Java, so GreySeal/Cormorant stand at ~tens of tonnes on Java and exert little top-down control.
3. **RNG + process implementation.** NumPy PCG64 vs Java MT19937 diverge on the first draw; Baltic was
   never bit-equal cross-engine (documented "within ~1 OoM" for the *calibrated aggregate* config —
   this disaggregated, cod-tuned config diverges far more).

With neither the Python-tuned cod nor the (barely-forced) background predators controlling the prey
field on Java, the prey cascade upward — the same apex-predator-release mechanism the disaggregation
experiment documented. **The Python 5-seed certification above remains the authoritative result;** the
Java run is a coarse consistency check, and here it flags — correctly — that this config's stability
is an artifact of the Python engine's calibration levers, not an engine-robust equilibrium.
