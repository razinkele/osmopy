# Baltic cod-predation mortality (M2) vs the WGSAM SMS key run — first comparison, 2026-10-03

**Issue #136.** First run of `scripts/validate_outputs_vs_ices.py --sms-m2` on the production
Baltic config (`data/baltic/baltic_all-parameters.csv`, 15 yr, fixed seeds, Python engine, output
prefix `osm`), trailing 5-yr model window against the SMS 2018–2022 window. **Report-only; nothing
is gated on this.** Method, provenance and caveats: `data/baltic/reference/ices_snapshots/README.md`
§ WGSAM SMS and the `sms_m2` block of `index.json`. Artifacts: `baltic_sms_m2_validation_2026-10-03.json`
(the validator's JSON output for this run) beside this note.

## WGSAM SMS cod-predation mortality (M2) — report-only, not gated

SMS M2 = annual instantaneous predation mortality BY COD on the prey stock, at age, weighted here across ages by biomass at age from the same key run (effectively ages 1+). Model = cod-ATTRIBUTED M2 on a biomass basis: annual tonnes of the prey eaten by the cod stocks (predatorPressure) over the prey's mean biomass (young-of-year excluded by output.cutoff.age). The two bases differ only through age structure. The model's TOTAL predation rate (all predators, per stage) is shown beside it so the non-cod residual is visible. SMS domain: ICES SD 25-32 excl. Gulf of Riga; the model grid also holds the western basin.

Model window: last 5 years of run · SMS window: 2018-2022

| prey | SMS stock | SMS M2 mean [min, max] | model cod M2 mean [min, max] | model/SMS | by cod stock | model total predation (Juvenil / Adult) | note |
|---|---|---:|---:|---:|---|---|---|
| herring | `her.27.25-2932` | 0.069 [0.060, 0.083] | 0.035 [0.032, 0.036] | 0.51 | cod_west 0.004, cod_east 0.031 | 0.612 / 0.567 | — |
| sprat | `spr.27.22-32` | 0.087 [0.079, 0.094] | 0.277 [0.252, 0.299] | 3.19 | cod_west 0.005, cod_east 0.272 | 3.126 / 0.603 | — |

## Reading it

- **Herring.** Cod-attributed M2 in the model is 0.035 yr⁻¹ against the SMS
  0.069 (ratio 0.51) — half, with cod_east carrying nearly all of it.
  The model's **total** predation rate on adult herring from `mortalityRate` is 0.57 yr⁻¹ — a
  stage rate on a different basis (numbers, adults only), so no cod share is read off the pair;
  the same-basis budget below is what attributes the predation.
- **Sprat.** Cod-attributed M2 is 0.277 yr⁻¹, **3.2×** the SMS
  0.087, again almost entirely cod_east. Total predation on juvenile sprat is
  3.13 yr⁻¹, an order of magnitude above anything SMS
  attributes to cod. (*2026-11-05: that mean is one year — see the correction at the end. The
  annual series is 0.71, 0.85, 0.81, 1.03, **12.23**; without the final year it is ~0.85 yr⁻¹.*)
- **The residual is pikeperch.** On the SAME basis as the cod figure — each focal predator's annual
  consumption of the prey over the prey's mean biomass — the shares are: herring — pikeperch 0.530,
  cod_east 0.031, perch 0.020, cod_west 0.004 (focal total 0.585 yr⁻¹); sprat — pikeperch 0.298,
  cod_east 0.272, perch 0.007, cod_west 0.005 (focal total 0.582). As a consistency check only (the
  bases differ), those focal totals sit close to the adult-stage `mortalityRate` predation rates
  (0.567 and 0.603 yr⁻¹), which suggests the focal food web accounts for most of the predation on
  ≥0.5-yr fish, and
  **pikeperch, at 1.35 Mt in this run (the known percid overshoot,
  `docs/baltic_percid_overshoot_conclusion_2026-07-05.md`), is the predator the SMS does not have.**
  The background predators — grey seal (no accessibility column, hence full access; CLAUDE.md gotcha)
  and cormorant — are absent from `predatorPressure` by construction (*no longer since #198 —
  closed with them in the 2026-11-05 addendum below: together 2.5% of the herring budget and 7% of
  sprat's*); the consistency check above
  leaves little room for them on ≥0.5-yr fish, but it is not a stage-matched budget. The juvenile sprat
  term (3.1 yr⁻¹) is not budgeted by this table and is where egg/YOY predation by clupeids, percids
  and the background species would sit. (*2026-11-05: the 3.1 is a single-year spike, not a
  standing juvenile term — see the correction at the end.*)
- **What this does and does not say.** The two rates are not equivalent: they differ through age
  structure (biomass basis vs numbers at age), through domain (SMS: SD 25–32 excl. Gulf of Riga; the
  grid also holds the western basin) and through the prey pool each is taken on (SSB table above:
  herring 3.3× the ICES envelope, sprat 0.85×). Each of those shifts the ratio on its own, so the
  comparison is indicative, not a like-for-like skill score. The comparison does its job: the predation engine's cod term is within a
  factor 2–3 of SMS on both stocks, and what the validator surfaces is the percid predation the SMS
  food web lacks — a calibration finding already on record, now with a number against an external
  multispecies benchmark.

## Next

1. Re-run after any percid or seal-matrix recalibration — this comparison is the cheap external check
   for both (the seal column fix is a recalibration decision: one cell moved cod_west 15 → 75 cm).
2. ~~If the JUVENILE term needs attributing, extend predatorPressure to background predators.~~
   Done in #198; budget closed in the addendum below. predatorPressure is tonnes, so it still
   cannot attribute egg/YOY predation by NUMBERS — that needs an age-resolved instrument.
3. The SAG config validator is stale after the cod split — issue #182.

## Addendum 2026-11-05 — the budget with the background predators

`predatorPressure` gained `GreySeal` and `Cormorant` columns in #198, so the same-basis budget above
can now include them. Fresh run: production config at master `a862a064` (which includes the sprat
F change 0.1637 → 0.369 of 2026-10-29, so these are not the October run's numbers), 15 yr, seed 0,
determinism keys on, diet output on; same trailing 5-yr window.

Each entry is that predator's annual consumption of the prey over the prey's mean biomass (yr⁻¹):

| prey | pikeperch | cod_east | Cormorant | GreySeal | perch | cod_west | all 11 predators | adult predation rate (`mortalityRate`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| herring | 0.560 | 0.033 | 0.010 | 0.006 | 0.020 | 0.004 | 0.632 | 0.566 |
| sprat | 0.297 | 0.292 | 0.028 | 0.018 | 0.008 | 0.005 | 0.648 | 0.620 |

- **The background predators are minor on ≥0.5-yr herring and sprat.** Seal + cormorant take
  0.016 yr⁻¹ of herring (2.5% of the budget) and 0.046 of sprat (7%). The October reading stands:
  the residual against SMS is **pikeperch**, not the seal.
- **The all-predator budget slightly exceeds the adult stage rate** (112% herring, 105% sprat). The
  two are on different bases (biomass vs numbers; the numerator counts every eaten tonne, including
  young-of-year, while the denominator excludes them by `output.cutoff.age`), so this is the same
  consistency check as above, not a closure to the percent.
- **Cod against SMS on this run:** herring 0.036 vs 0.069 (0.53×), sprat 0.298 vs 0.087 (3.43×) —
  unchanged in kind from October.
- **Juvenile sprat total predation is now 0.97 yr⁻¹** (October: 3.13). The sprat F change is the
  only production-config change in between, but the October run's seed was not recorded, so seed
  differences are not ruled out. The cause of that drop is not investigated here. (*Resolved the
  same day — next section.*)

## Correction 2026-11-05 — the October juvenile sprat rate was one year

Re-running current master with ONLY `fisheries.rate.base.fsh2` set back to 0.16365… (seed 0,
15 yr, determinism keys on) reproduces every October stage rate exactly: sprat 3.126 / 0.603,
herring 0.612 / 0.567. So the October run was seed 0, and seed is ruled out. But the annual
`mortalityRate` series shows the 3.13 is not a level:

| arm | sprat juvenile predation, model years 11–15 (yr⁻¹) | 5-yr mean |
|---|---|---:|
| sprat F 0.1637 (= October) | 0.71, 0.85, 0.81, 1.03, **12.23** | 3.13 |
| sprat F 0.369 (current) | 0.83, 0.92, 1.13, 1.07, 0.88 | 0.97 |

From model year 6 on, every year sits at 0.71–1.15 yr⁻¹ in **both** arms except the October arm's
final year; the October mean is carried by that one year alone. Sprat age-0 standing stock is the same in both arms (by-age bin 0, last
5 yr: 227 kt vs 233 kt). So:

- **There was no 3× drop in juvenile predation.** The sprat F change perturbed the trajectory and
  the one-year spike did not recur. Juvenile sprat predation is ~0.8–1.0 yr⁻¹ on either config.
- **The October readings built on 3.1 are withdrawn**: it is not "an order of magnitude above"
  cod's attributed M2 as a standing term, and it does not mark a large unbudgeted juvenile
  pathway. (A sum of per-step rates can spike when a stage holds few fish in some step — CLAUDE.md
  on mortality-rate outputs; the spike's step-level cause is not traced here.) (*Traced later the
  same day: 11.36 of the 12.23 is ONE step, t = 14.25, in which sprat's Juvenil stage held 7×10⁻¹⁹
  fish — every real juvenile had just matured and the new cohort was still eggs — so the rate is
  taken on an empty stage. Java's `MortalityOutput` uses the same formula; the validator now
  reports the annual median for these stage rates (October run: 0.85, current config: 0.92), with
  the mean kept in the JSON.*)
- **For the validator:** a 5-yr *mean* of an annual stage rate is not robust to one spike year. The
  cod-attributed M2 and the adult rates in this note are steady year to year and unaffected.

## Re-run 2026-11-11: v1.9.0, and the 15-yr protocol is not at equilibrium

Follow-up 1 above asks for a re-run after any seal-matrix or percid recalibration. v1.9.0 carries
both: the GreySeal accessibility column (#227) and the corrected-LTL recalibration (#236). This is
production `data/baltic` at `95baf49b`, seed 0, the same trailing 5-yr window, and the same
`validate_outputs_vs_ices.py --sms-m2`. Artifacts: `baltic_sms_m2_validation_2026-11-11_15yr.json`
and `…_50yr.json` beside this note.

The protocol's 15-yr run (model years 11–15) gives cod_east **155 kt**. Its certified equilibrium
is **76–80 kt** (50 yr, final decade, 5 seeds). So the 15-yr window still sits in cod_east's
transient. The same seed was therefore also run to 50 yr (window years 46–50):

| | 15 yr: model cod M2 | model / SMS | 50 yr: model cod M2 | model / SMS | 50-yr / 15-yr |
|---|---:|---:|---:|---:|---:|
| cod_east biomass (window mean) | 155,180 t | | 72,274 t | | **0.47** |
| herring, by cod_east | 0.069 | | 0.029 | | 0.42 |
| herring, by cod_west | 0.005 | | 0.015 | | 3.0 |
| **herring, cod total** | 0.074 | **1.08** | 0.044 | **0.64** | 0.59 |
| sprat, by cod_east | 0.381 | | 0.156 | | 0.41 |
| sprat, by cod_west | 0.004 | | 0.012 | | 3.0 |
| **sprat, cod total** | 0.386 | **4.44** | 0.168 | **1.93** | 0.44 |

SMS: herring 0.069 and sprat 0.087 (2018–2022), unchanged.

- **cod_east's attributed M2 falls in step with its biomass.** Biomass falls 0.47×; its
  predation on herring falls 0.42× and on sprat 0.41×. cod_west's share triples, as its biomass
  does (6.0 kt → 17.4 kt). So on the current config the 15-yr protocol overstates cod-attributed
  M2 about 2×, through the cod_east transient alone.
- **At equilibrium, cod sits within 2× of SMS on both prey**: herring 0.64×, sprat 1.93×. That is
  closer than either the October run (0.51× and 3.19×) or this config's own 15-yr reading. The
  October and 2026-11-05 runs used the 15-yr protocol too. Their cod_east trajectories were not
  recorded, so how much of their sprat excess was transient is not known.
- **Total predation, annual medians (Juvenil / Adult):**
  - herring: 0.38 / 0.68 at 50 yr, against 0.41 / 0.79 at 15 yr;
  - sprat: 0.51 / 0.42 at 50 yr, against 0.73 / 0.64 at 15 yr.
  These are context only, as above.
- **For the protocol:** compare the SMS key run against the model's equilibrium. Use a 50-yr run
  (the certification horizon), not 15 yr. The validator itself is unchanged; this is a statement
  about which run to feed it.

## Re-run 2026-11-12 on v1.10.0 (herring recalibration), 50-yr protocol

v1.10.0 (#250) recalibrated herring: maturity 14.5 cm, adult additional mortality 0.2 /yr, larval
rate re-fitted. Herring is one of the two prey stocks here, so the comparison is re-run with the
50-yr protocol the previous section recommends. Setup: seed 0, final 5 years (years 46–50). The
artifact is `baltic_sms_m2_validation_2026-11-12_v1.10.0_50yr.json`.

| | v1.9.0, 50 yr | **v1.10.0, 50 yr** | SMS |
|---|---:|---:|---:|
| herring, cod-attributed M2 | 0.044 (0.64×) | **0.036 (0.53×)** | 0.069 |
| sprat, cod-attributed M2 | 0.168 (1.93×) | **0.157 (1.81×)** | 0.087 |
| herring total predation, Juvenil / Adult (annual median) | 0.38 / 0.68 | **0.64 / 0.26** | — |
| sprat total predation, Juvenil / Adult | 0.51 / 0.42 | 0.65 / 0.37 | — |
| cod_east biomass (window mean) | 72,274 t | 67,198 t | — |

- **Cod stays within 2× of SMS on both prey.** It is 0.53× on herring and 1.81× on sprat.
- **Herring's adult total predation rate fell from 0.68 to 0.26 /yr.** v1.10.0's herring has many
  more adults (SSB 93% of biomass), and the predators' consumption is spread over that larger adult
  stock. The cod-attributed M2 on herring fell less, from 0.044 to 0.036. The mechanism is not
  measured.
- **The validator's SSB table still compares herring's ≥0.5-yr biomass** (1.58 Mt) **with an ICES
  SSB envelope.** `output.ssb.enabled` is on since v1.10.0, so switching it to compare SSB with SSB
  is now a small follow-up (#247, option 3). It is not done here.
