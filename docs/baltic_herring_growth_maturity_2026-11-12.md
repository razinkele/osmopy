# Baltic herring growth and maturity against data (issue #247): 2026-11-12

**Question:** on v1.9.0, herring's spawning stock is 1.5% of its biomass (#247). Is that because
model herring grow too slowly, or because the 18 cm maturity size is too high? Both were checked
against ICES survey and assessment data.

**Answer: neither is the main cause. The main cause is mortality.** Model herring hardly survive
past age 2. The maturity size is also clearly too high, and young-fish growth is somewhat slow.
Ranked by size:

1. **Age structure (largest).**
   - 95% of model herring by number are age 1 (Q1). In the WGSAM SMS assessment 32% are; 68% are
     age 2+.
   - The model's age-1+ mortality hazard is about 2.9 /yr. SMS cohort mortality is 0.32–0.68 /yr
     at ages 1–6, fishing included.
   - Most of the model's rate is herring's configured additional mortality,
     `mortality.additional.rate.sp1` = **2.25 /yr**, the highest of any focal species. SMS
     predation mortality on herring is only 0.03–0.15 /yr.
2. **Maturity size.**
   - BITS gives **L50 = 14.3 cm** (Q1; interquartile 12.9–15.6 cm; 6,227 fish) and **15.0 cm in
     Q4**. About 90% of age-2 herring are mature.
   - The model matures herring knife-edge at **18 cm**.
   - On the model's own size structure, lowering it to the observed L50 would raise the mature
     share of numbers about 11×, from 0.17% to 1.9%. Even then it stays far below the assessed
     stock.
3. **Growth (smallest, and partly a reference question).**
   - At ages 1–2, v1.9.0 herring are about 3.5 cm shorter than BITS means. Against SMS stock
     weights converted to length they are only about 1 cm short, and BITS's bottom trawl may
     favour larger young fish.
   - Older model herring overshoot badly (model L∞ 27 cm against a BITS fit of 23.5 cm).
   - The v1.8.8 state, on the richer food field, grew faster at young ages.

The data back changes to two parameters: herring's adult additional mortality and its maturity
size. Either change is a recalibration, so it is the maintainer's decision. Nothing was changed
here.

## Data

- **BITS individual records:** DATRAS catch-at-age (CA) records, quarters 1 and 4, 2018–2022,
  herring (AphiaID 126417). Only Poland and Russia report herring CA data in BITS, from SD 25–26
  (rectangles 37–41, 15–20 °E). That is 10,945 fish.
  - Maturity follows the standard BITS key: 61 immature, 62–65 mature, 66 excluded.
  - National Roman-numeral records are added under the BITS herring rule of Maioli et al. (2026):
    stages I–II immature, III+ mature.
  - The L50 is robust to that choice: it is 14.4 cm (Q1) and 14.9 cm (Q4) with the standard key
    alone.
  - Script: `scripts/datras_herring_growth_maturity.py`. Output:
    `docs/diagnostics/herring_bits_growth_maturity_2018-2022.json`.
- **Assessment numbers and weights:** the WGSAM SMS 2025 key run, central Baltic, already
  snapshotted in `data/baltic/reference/ices_snapshots/`. It provides quarter-1 stock numbers and
  mean weight at age, and predation mortality (M2) at age, for 2018–2022.
- **Model:**
  - **Runs:** production v1.9.0, and the v1.8.8 state (all of #236 reverted), seed 42, 50 yr.
  - **Hook:** a read-only hook records each herring school's length and its ICES-convention age
    (calendar years since birth) in the same Q1 (Feb–Mar) and Q4 (Oct–Nov) windows.
  - Script: `scripts/model_herring_length_at_age.py`. Output:
    `docs/diagnostics/herring_model_length_at_age_2026-11-12.json`.

## 1. Age structure and mortality

| ICES age (Q1 numbers) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMS 2018–22 share | 32.1% | 24.4% | 15.6% | 12.1% | 7.4% | 3.9% | 2.1% | 2.4% |
| model v1.9.0 share | **95.3%** | 4.5% | 0.17% | 0.005% | ~0 | ~0 | ~0 | ~0 |
| model v1.8.8 share | 94.7% | 5.1% | 0.21% | 0.008% | ~0 | ~0 | ~0 | ~0 |

**SMS cohort total mortality,** Z = ln(N_a,y / N_a+1,y+1) for the 2018–2021 cohorts:

| ages | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 |
|---|---:|---:|---:|---:|---:|---:|
| Z (/yr) | 0.32 | 0.36 | 0.43 | 0.53 | 0.68 | 0.68 |

**SMS M2 on herring** (2018–22 mean) is 0.15 at age 1, falling to 0.03 at age 8.

**Model age-1+ hazard** (`herring_compensation_2026-11-12.md`, v1.9.0 baseline, per year):

| cause | additional | predation | fishing | starvation | total |
|---|---:|---:|---:|---:|---:|
| hazard (/yr) | 2.11 | 0.36 | 0.36 | 0.06 | ≈ 2.9 |

That gives annual survival of about 6% past age 1, against e^−0.32 to e^−0.68, or 51–73%, in the
assessment. Both model states have the same truncated structure. The additional rate (2.25 /yr) is
not part of #236. It has been in the Baltic config since the example was built, and it is a
calibrated parameter. For comparison, the other focal species' additional rates are 0.006–1.25 /yr.

## 2. Maturity

| | L50 | L25–L75 | fish |
|---|---|---|---:|
| BITS Q1, all (incl. national keys) | **14.3 cm** | 12.9–15.6 | 6,227 |
| BITS Q1, standard key only | 14.4 cm | 13.2–15.7 | 2,418 |
| BITS Q4, all | 15.0 cm | 13.8–16.1 | 4,652 |
| **model** | **18 cm, knife-edge** (`species.maturity.size.sp1`) | — | — |

| ICES age | 1 | 2 | 3 | 4+ |
|---|---:|---:|---:|---:|
| BITS Q1, proportion mature | 0.18 | **0.90** | 0.98 | ≥ 0.99 |
| model v1.9.0 Q1, share ≥ 18 cm | 0.00 | **0.02** | 0.31 | 0.92 (age 4) |

- **An 18 cm herring is 95% mature in the data** (Q1; 94% in Q4), where maturity happens at age 2.
- **In the model only 2% of age-2 herring are mature**, and almost none survive to age 3–4,
  where most would be.
- **The mature share of numbers on the model's own lengths,** a static count with no feedback,
  depends on the maturity size used:

| state | at 18 cm | at 14.4 cm |
|---|---:|---:|
| v1.9.0 | 0.17% | 1.9% |
| v1.8.8 | 2.3% | 11.9% |

## 3. Growth

**Mean length at age** (cm) in Q1, ICES age:

| | age 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BITS (SD 25–26) | 13.0 | 16.7 | 18.5 | 19.5 | 20.8 | 20.8 | 21.2 |
| SMS weight → length* | 10.7 | 14.1 | 15.6 | 16.6 | 17.3 | 18.0 | 18.8 |
| model v1.9.0 | **9.5** | **13.1** | 17.2 | 20.8 | 23.5 | 25.8 | 27.2 |
| model v1.8.8 | 10.3 | 16.1 | 20.6 | 24.5 | 27.4 | 29.6 | 31.4 |

\* SMS mean weight in stock at age (7.0, 16.5, 22.6, 27.5, 31.3, 35.1, 40.0 g), converted with
the BITS length–weight fit. This is a derived estimate.

- **The von Bertalanffy curves differ.** BITS ages 0–11, both quarters, on post-hatch age, fit to
  L∞ 23.5 cm and K 0.28 /yr. The model's potential curve is L∞ 27 cm, K 0.35 /yr, t₀ −0.8.
- **Young fish:** v1.9.0 is 3.5 cm below BITS at ages 1–2, but about 1 cm below the SMS-derived
  lengths. The two references disagree by about 2.5 cm at age 1–2 themselves. BITS covers only
  SD 25–26, its trawl selectivity favours bigger young fish, and SMS's Q1 weights are lean.
- **Old fish:** the model overshoots both references from age 4.
- **v1.8.8 vs v1.9.0.** v1.8.8 had richer food and was 3 cm longer at age 2, which is why it
  reached 19% mature. The corrected food field slowed young growth.
- **Length–weight:** model a 0.0056, b 3.13; BITS a 0.0046, b 3.09. The model's herring are about
  34% heavier at 15 cm.

## Caveats

- **Coverage.** BITS herring age data come from SD 25–26 only. Northern Baltic herring grow more
  slowly (Vainikka et al., 2009), and the model's herring are a basin-wide aggregate, including
  the Gulf of Bothnia and the Gulf of Riga. Basin-wide size at age is therefore probably somewhat
  below the BITS values, which would narrow the growth gap and leave the maturity and mortality
  findings intact.
- **Age conventions.** ICES ages count 1 January birthdays. The model comparison uses the same
  convention, in the same months.
- **The counterfactual percentages are static.** They apply a different maturity size to the
  model's current lengths and numbers, with no feedback. A real change would shift recruitment,
  density and growth.

## What this means for #247

- **The thin spawning stock is mainly a mortality artefact.** Herring's additional mortality of
  2.25 /yr removes fish before they reach maturing ages. The 18 cm maturity size then excludes
  most of the few age-2 fish that do survive.
- **A data-backed fix is a recalibration with two changes**, and needs the maintainer's word:
  - herring's maturity size to about 14.3–15 cm;
  - herring's adult additional mortality brought toward the assessment's level, with the larval
    rate re-fitted to the biomass target, as #236 did.
- **Growth parameters are a secondary question.** A smaller L∞, closer to 23.5 cm, would match
  older fish.

## References

- Maioli, F., Vitale, F., Lindmark, M., et al. (2026). Empirical length-at-maturity for Northeast
  Atlantic marine fishes [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10856339/v1
- Vainikka, A., Casini, M., Gårdmark, A., et al. (2009). Spatial variation in growth, condition
  and maturation reaction norms of the Baltic herring *Clupea harengus membras*. *Marine Ecology
  Progress Series*, 383, 285–294. https://doi.org/10.3354/meps07970
- ICES DATRAS, Baltic International Trawl Survey (BITS), catch-at-age records 2018–2022,
  https://datras.ices.dk (accessed 2026-11-12).
- ICES WGSAM SMS Baltic 2025 key run (`ices-eg/wg_WGSAM@f690d4ff`), snapshot in
  `data/baltic/reference/ices_snapshots/`.
