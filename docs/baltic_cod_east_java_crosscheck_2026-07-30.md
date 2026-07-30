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

---

# CORRECTION 2026-07-30 — the cod rows above are invalid

**Everything below supersedes the cod_west/cod_east rows and the entire "Verdict" section above.**

## What was wrong

`certify_java` read Java's biomass CSV by the RAW config species name. But
`reconcile_config_for_java` rewrites `species.name.sp*` to Java's stripped internal form, so Java
writes its columns as `codwest`/`codeast`. The lookup missed and fell through to
`series.get(sp, [0.0])`, fabricating a zero series that scored as min 0, late-mean 0, "COLLAPSE".

**Java never drove either cod stock extinct.** Fixed in `2250824`, which resolves names through
`sanitize_java_name` and now RAISES on a missing column instead of substituting zero.

This also voids the reasoning in the superseded Verdict section. The cod zeros were byte-identical
across the 2026-07-29 and 2026-07-30 runs because a failed dict lookup is deterministic and
independent of any config change — so the `fsh8` fix could not have moved them, and reading that
invariance as "the confound points away from the discrepancy" was a false confirmation. Only
species whose names contain no `_`/`-` (herring, sprat, flounder, perch, pikeperch, smelt,
stickleback) were ever read correctly; their values above stand.

## The real Java 4.4.1 result (50 yr, Java's own RNG, `--params current`)

| species | min (t) | final-decade mean (t) | envelope (t) | vs envelope |
|---|---|---|---|---|
| cod_west | 3,353 | **353,850** | 4,000–25,000 | **~14× OVER** |
| cod_east | 1,844 | **948,866** | 60,000–85,000 | **~11× OVER** |
| herring | 1,430,341 | 1,916,430 | 800,000–3,000,000 | in-envelope |
| sprat | 0 | **0** | 800,000–2,500,000 | extinct (real) |
| flounder | 49,287 | 3,011,791 | 20,000–100,000 | ~30× over |
| perch | 22,501 | 2,279,394 | 8,000–50,000 | ~46× over |
| pikeperch | 660,508 | 8,139,264 | 4,000–25,000 | ~326× over |
| smelt | 1,654,999 | 4,019,851 | 200,000–1,500,000 | ~2.7× over |
| stickleback | 392,845 | 5,177,973 | 20,000–200,000 | ~26× over |

`cod_east` still trips the `persists` flag, but for an unrelated reason: its **minimum** (1,844 t)
falls below `0.1 × envelope-lower` during the early seeding transient. The stock then grows to
~949 kt. A transient dip is not a collapse, and the flag conflates the two.

## The corrected diagnosis

**The old "cod goes extinct → prey release → mid-trophic explosion" narrative is dead.** Cod is not
extinct; it is 11–14× ABOVE envelope. Predation release cannot explain prey inflation when the
predator is itself inflated an order of magnitude.

The true pattern is **system-wide biomass inflation on Java — every focal species except sprat ends
2.7–326× above envelope, cod included** — against a Python run of the same config that sits near or
inside envelope for most species. That is a different problem with a different candidate set
(total-mortality or forcing scaling on the Java path), and it is not addressed by anything in this
session's commits.

**Sprat is the sole genuine Java collapse** and is now the one real cross-engine contradiction worth
chasing: Python holds it at ~1.06 Mt, Java drives it to zero. `sprat` contains no underscore, so it
was always read correctly — this finding predates and survives the harness bug.

## Also corrected

The "single seed 42" attribution in this note and in the 2026-07-29 note was unfounded.
`certify_java` accepted a `seed` argument and never passed it to Java; Java 4.4.1 exposes only a
`simulation.fixedseed.enabled` toggle, not a numeric seed. The parameter has been removed rather
than left as a silent no-op.
