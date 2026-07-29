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

## Java 4.4.1 cross-check: FAILED to initialize (disaggregated-config staging incompatibility)

The single-seed Java 4.4.1 cross-check (staged via the C2 background recipe) **could not run** on the
disaggregated 9-species config. Java aborts during `Simulation.init`, before stepping, with:

```
osmose[severe] Wrong species name in spatial map series 'movement.map*'
java.io.IOException: Parameter movement.species.map0 = cod_west does not match any predefined species name.
    at fr.ird.osmose.background.BackgroundMapSet.loadMapsCsv(BackgroundMapSet.java:250)
    at fr.ird.osmose.background.BackgroundMapSet.loadMaps(BackgroundMapSet.java:218)
    at fr.ird.osmose.background.BackgroundMapSet.init(BackgroundMapSet.java:150)
    at fr.ird.osmose.background.BackgroundProcess.init(BackgroundProcess.java:29)
```

**Root cause — Java-side, NOT the M=0.9 tuning and NOT the Python engine.** The staged config is
internally consistent: `species.name.sp0 = cod_west`, `species.name.sp8 = cod_east`, and the shared
34-entry `movement.species.map*` series (focal maps map0–map29, background GreySeal/Cormorant maps
map30–map33) all reference valid, defined names. But Java 4.4.1's `BackgroundMapSet` map loader
validates every `movement.species.mapN` against a name registry that does not recognize the renamed
`cod_west` — it rejects the very first map (`map0 = cod_west`). The C2 background-staging path
(`stage_background_for_java`, which here emitted only `output.cutoff.enabled=false`) was built and
validated for the **aggregate 8-species** config; it does not reconcile the disaggregation rename
(cod → cod_west + appended cod_east) or the +1-shifted background indices (Cormorant sp16) with the
Java background map machinery. See [[cod-ew-disaggregation]]: the disaggregated config is a
Python-engine experiment; a Java cross-check would require extending the staging layer to re-register
the renamed/shifted species with Java's `BackgroundMapSet`.

**Bottom line.** The Python 5-seed certification above is the authoritative result and reproduces
exactly (cod_east in-envelope at 82.6–83.4 kt). Cross-engine bit-equivalence was never expected for
Baltic anyway (NumPy PCG64 vs Java MT19937 diverge on the first draw; the check is a coarse
survivor-set consistency test) — and on the disaggregated config that coarse check simply cannot be
staged for Java without further interop work.
