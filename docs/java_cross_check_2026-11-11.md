# Python ↔ Java cross-engine check, EEC refresh: 2026-11-11

Re-run of `docs/java_cross_check_2026-09-27.md` on **v1.9.0** (`95baf49b`), with the same
harness, flags and config.

**Verdict: `REVIEW`, at 66 of 70 cells equivalent** (59/70 on 2026-09-27).

- All four remaining cells are **underpowered**: their 90% confidence interval is wider than
  their effect.
- The largest deviation anywhere is **0.21 log10 (1.6×)**. It was 0.52 (3.3×).
- The two species that carried a real signal on 2026-09-27, `lesserSpottedDogfish` and
  `mackerel`, are now equivalent on biomass and yield.

This is also the full 16-rep confirmation of the #237 yield fix. Before this run it had been
checked with 2 reps only.

## Method

```
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py \
  --config data/eec_full/eec_all-parameters.csv \
  --engines python,4.4.1 --n 16 --years 10 --spinup-years 2
```

- **Run:** 16 varied-seed reps per engine, 10 years, the first 2 years dropped as spin-up.
- **Test:** TOST on log10 post-spin-up means. The margin is Δ = 0.48 (3.0×) for biomass, yield
  and abundance, and 0.176 (1.5×) for mean weight and mean size.
- **Sign:** `d = log10(Python) − log10(Java)`, so a positive `d` means Python runs higher.
- **Wall time:** the simulations took 477 s; the 2026-09-27 run took 830 s in all.
- **Python determinism check:** passed.

**Jar:** `osmose-java/osmose-4.4.1-jar-with-dependencies.jar`.
- sha256 `8ae7c519…f697cee`, file date 2026-06-26, manifest `Implementation-Version: 4.4.1`.
- The jar carries no source commit. So it cannot be tied byte-for-byte to the `008f74b` build
  that the 2026-09-27 note describes, which was made in a different session.
- This is the same local jar that the 2026-11-08 re-runs used.

## Result by metric

| Metric | Δ | Equivalent | Not established |
|---|---|---|---|
| biomass | 3.0× | 13/14 | sardine |
| yield | 3.0× | 13/14 | sardine |
| abundance | 3.0× | 12/14 | mackerel, sardine |
| mean_weight | 1.5× | **14/14** | — |
| mean_size | 1.5× | **14/14** | — |
| **Total** | | **66/70** | 4 |

**Two yield cells are vacuous.** The EEC config sets `fisheries.rate.base.fsh4` (poorCod) and
`fsh6` (dragonet) to 0.0. Both engines therefore report zero yield for those species, and the
harness prints `d = 0.00, ci90 = ±0.00, KS = nan`. That is agreement about an unfished stock, not
evidence about the fishery code. Of the **12 fished species**, 11 are equivalent on yield, and
sardine is the underpowered exception below.

## The four non-equivalent cells

| Metric | Species | d | fold | ci90 | \|d\|+ci90 | Δ | Reading |
|---|---|---|---|---|---|---|---|
| abundance | mackerel | −0.21 | 0.62× | ±0.28 | 0.49 | 0.48 | underpowered: ci > \|d\| |
| biomass | sardine | −0.14 | 0.72× | ±0.42 | 0.56 | 0.48 | underpowered |
| yield | sardine | −0.15 | 0.71× | ±0.41 | 0.56 | 0.48 | underpowered |
| abundance | sardine | −0.12 | 0.76× | ±0.44 | 0.56 | 0.48 | underpowered |

None of the four point estimates is near the margin. What fails is the bound: the inter-seed
variance is high for these species on this config. The same was true of sardine on 2026-09-27,
when its ci90 was ±0.37–0.38. More reps could settle them; 16 does not. Read these rows neither as
divergence nor as agreement.

## What moved since 2026-09-27

| | 2026-09-27 | 2026-11-11 |
|---|---|---|
| equivalent cells | 59/70 | **66/70** |
| largest \|d\| | 0.52 (3.3×, dogfish yield) | **0.21** (1.6×, mackerel abundance) |
| lesserSpottedDogfish biomass / yield / mean weight | +0.45 / +0.52 / +0.30 | −0.01 / −0.06 / +0.02 |
| mackerel biomass / yield / abundance | +0.36 / +0.34 / +0.47 | −0.16 / −0.18 / −0.21 |
| whiting mean weight | −0.15 (not established) | +0.00 (equivalent) |
| horseMackerel mean weight | +0.08 (underpowered) | +0.02 (equivalent) |

**Which arm changed.** The Java arm's staged inputs were diffed between the two staging paths on
current code: `write_temp_config`, which the 2026-09-27 run used, and `stage_config_for_java`,
which every Java path has used since #223. They differ only in `fishery-catchability.csv` and
`fishery-discards.csv`, and there only in two ways:
- the trailing spaces Java trims anyway (`pouting `, `squids `) are gone;
- each resource group gets an all-zero row.

Every focal row is numerically identical. So the Java arm read the same food web and the same
fishery as on 2026-09-27, apart from the jar-build caveat above.

The Python arm did change. Engine commits since `d3ee111b` that touch dynamics on this config:
- #214 (`ddcc7414`): random-method schools are placed over their whole domain;
- #216 (`674b1014`): the random walk deals a uniform accessible cell;
- #233: T3 (`d40305e7`) zeroes any school below one fish, and T1 (`6bd4a6b3`) removes the
  length cap at L∞ when `species.lmax` is unset.

All four move Python toward Java's behaviour. **No per-commit attribution was measured.** Do not
credit the dogfish and mackerel recovery to any single one of them without an A/B.

## The 2026-11-08 interim run

`docs/baltic_ltl_recalibration_2026-11-08.md` reports a post-#233 EEC re-run: biomass within 1 OoM
and worst 2.6× (mackerel). That run's Java arm still carried the #223 fishery-staging regression,
fixed in #237. Java caught about 10⁻⁴ of Python's yield, so its biomass figures compared Python
against a nearly unfished Java. This refresh supersedes them.

## Relationship to the "within 1 OoM" claim

Every cell is within 1 OoM, and now within 1.6×. The `REVIEW` verdict comes only from the stricter
3× TOST bound, and here only from statistical power. CLAUDE.md's RNG gotcha now cites this note.

**Bay of Biscay was not re-run here.** Its 7/8 figure (Hake about 129× Java, Anchovy about 3.2×
below) comes from the 2026-11-08 run, which predates #237.

The #223 regression did not reach it. On the pre-#237 code (`99d2e61c`), staging `data/examples`
through `stage_config_for_java` and through `write_temp_config` gives byte-identical file trees.
So BoB's Java arm read the same inputs that night as it does now.

The BoB demo is not self-sustaining anyway: see CLAUDE.md.
