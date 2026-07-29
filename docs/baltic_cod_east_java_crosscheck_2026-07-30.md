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

**Java cross-check: 1/9 persistent (single seed).** Survivor sets DIFFER with Python — Python ['herring', 'pikeperch', 'smelt', 'stickleback'], Java ['flounder', 'herring', 'perch', 'pikeperch', 'smelt', 'stickleback']. Coarse consistency check only (Baltic is not bit-equal cross-engine); a DIFFER is a flag to inspect, not an automatic failure.

> Denominators above were emitted as `/8` by a hardcoded count in `_print_table` and the Java
> line (stale since the 8→9 disaggregation); corrected to `/9` here and fixed in the script.
> All biomass values and survivor sets are the run's own output, unmodified.

## Purpose: re-run after the `trawlcodeast` fix (#139)

The 2026-07-29 cross-check ran with `fisheries.movement.*.map8` absent, so Java deactivated
`fsh8` — the only fishery targeting `cod_east` — at every timestep. Eastern cod was therefore
**unfished on the Java side** while fished on the Python side, a confound in any survivor-set
comparison. `ff9dc48` added the missing map; this run is the re-derivation.

### Java 4.4.1 per-species, before vs after (seed 42, final-decade mean, tonnes)

| species | 2026-07-29 (`fsh8` inert) | 2026-07-30 (`fsh8` active) | change |
|---|---|---|---|
| cod_west | 0 | 0 | — |
| cod_east | **0** | **0** | — |
| herring | 1,959,470 | 1,874,062 | −4.4% |
| sprat | 0 | 0 | — |
| flounder | 3,007,264 | 3,080,151 | +2.4% |
| perch | 2,236,237 | 2,198,008 | −1.7% |
| pikeperch | 8,291,256 | 8,940,872 | +7.8% |
| smelt | 3,983,369 | 4,319,593 | +8.4% |
| stickleback | 4,815,401 | 4,994,603 | +3.7% |

## Verdict: the 2026-07-29 conclusions stand

**Every qualitative finding is unchanged.** `cod_east` still collapses to 0 on Java, the
survivor sets still DIFFER in exactly the same way (Python `[herring, pikeperch, smelt,
stickleback]` vs Java `[flounder, herring, perch, pikeperch, smelt, stickleback]`), and the
mid-trophic prey release persists. Quantitative shifts are a few percent — within the noise of a
single-seed coarse check, and not in a direction that changes any interpretation.

**The confound was real but pointed away from the discrepancy.** Missing `fsh8` biased Java's
`cod_east` *toward* survival by removing its fishing mortality, and it collapsed to zero anyway.
Restoring the fishery can only push it further down, so the gap between Python's in-envelope
`cod_east` (~83 kt) and Java's extinction was never explained by the missing map. This was
predicted before the run and the numbers bear it out.

**`fisheries.rate.base.fsh8` is 0.01**, so the small magnitude is expected — the fix corrects a
real structural error in the Java config without materially moving this particular comparison.

**Mid-trophic species rose slightly** (pikeperch +7.8%, smelt +8.4%) rather than falling, which
is mechanistically coherent: fishing the eastern cod stock releases a little more prey.

**The Python column is byte-identical to 2026-07-29**, independently confirming that the new
`map8` keys are inert on the Python engine (`osmose/engine/config.py` reads only
`fisheries.movement.file.map0`, as a shared map) — the fix is Java-only by construction.

### Still open

The Python↔Java `cod_east` disagreement is now a clean open question with the fishery confound
eliminated: Python holds it in-envelope at ~83 kt while Java drives it extinct on the same
config. Candidate causes remain the ones recorded on 2026-07-29 — forcing decay and the
RV recruitment gate being absent from the Java path — neither of which this fix touches.
