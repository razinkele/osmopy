# Jar-swap 4.4.1 — Phase 2 cross-engine parity (EEC) results

2026-06-26. Harness: `scripts/cross_engine_parity_440.py`. N=16 replicates/engine, 10yr,
2yr spin-up, equivalence margin Δ=log10(3) (factor 3×). Metrics: biomass, yield, abundance.
Java seeds time-based (varied); Python seeds 1000+. Stats per species/metric on log10 of the
final-year mean: formal TOST (two one-sided t-tests vs ±Δ), two-sample KS, variance ratio,
collapse frequency; community skill = MEF (Nash-Sutcliffe) + Spearman on the per-species vector;
relative gate = |Python−4.4.1| ≤ |Python−4.3.3| + Δ.

## Verdict: PASS — 4.4.1 is statistically indistinguishable from 4.3.3 against the Python engine

Community modelling efficiency (Python predicting each Java version's per-species pattern):

| metric | Py~4.4.1 (MEF / Spearman) | Py~4.3.3 (MEF / Spearman) |
|---|---|---|
| biomass   | 0.95 / 0.96 | 0.95 / 0.97 |
| yield     | 0.97 / 0.94 | 0.97 / 0.94 |
| abundance | 0.97 / 0.97 | 0.97 / 0.99 |

- **Relative gate (441 ≤ 433): Y for all 14 species on all 3 metrics** — the swap does not degrade
  Python↔Java agreement.
- **Formal TOST: 12/14 species equivalent** on each metric. The two outside the strict 3× band —
  `lesserSpottedDogfish` (d≈0.59) and `mackerel` (d≈0.46) — are off by the same amount against
  BOTH Java versions → a pre-existing Python-port-vs-Java disagreement (low-biomass, high-variance
  predators), NOT introduced by the jar swap.
- **Zero biomass collapses** (0/0/0 all species/engines). The earlier single-seed sardine collapse
  + mackerel outlier were stochastic — gone over the ensemble. (dragonet/poorCod yield = 16/16/16
  is "unfished → zero yield in all engines", not a collapse.)
- N=16 achieved 90% CI half-widths ≈ 0.02–0.32 log (factor ~1.05–2.1×): well-powered for most
  species; the high-variance pelagics (sardine/mackerel/horseMackerel) have the coarsest MDD.

## Honest caveats (vs the full Phase-2 spec)
- **Metrics: biomass/yield/abundance only.** F (fishing mortality), mean trophic level, and
  size-structure need output-flag enablement (not in EEC's default output set) + Java-side support;
  deferred.
- **2 species (dogfish, mackerel) are outside strict equivalence** on biomass/yield — a Python-port
  limitation present in 4.3.3 too, surfaced here, worth a separate look but orthogonal to the swap.
- **KS p-values are often low** (the engines are genuinely different stochastic implementations,
  PCG64 vs MT19937) — true for 4.3.3 too; the meaningful signals are MEF/Spearman + the relative gate.
- **EEC only.** BoB Java-blocked (365-step forcing); Baltic Python-engine-only. The resource-forcing
  migration is inert for Baltic (default 4.3.3 write target; no file-forced resources) — covered by
  `test_non_resource_species_get_no_forcing_keys` + `test_reverse_to_4_3_3_does_not_add_forcing_keys`.
  Task 2.4 (Baltic round-trip) is moot without a cutover.
