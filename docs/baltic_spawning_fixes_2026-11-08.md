# Baltic spawning-map fixes: certified one change at a time against the GreySeal base (2026-11-08)

**Outcome:** of the three literature corrections from the 2026-11-07 review (C1, C2, S3), only one part
certifies: **C1a**, which removes herring's 10-cell autumn packing. It ships. The rest fail
certification, and the reason recurs: the certified Baltic configuration is a calibration fitted around
these data defects, so correcting one removes something the fit was leaning on. This note records every
run, so the remaining decisions rest on numbers.

Base for every run: master plus the GreySeal accessibility column (PR #227, certified 5/5). All runs are
50 yr × 5 seeds (42, 123, 7, 999, 2024) on the Python engine, read from the final-decade means.

## 1. The changes (review C1, C2, `docs/reviews/2026-11-07-temporal-spatial-dynamics-review.md`)

- **C1, herring autumn spawning.** The config released 25% of herring eggs in Sep–Nov and, for those six
  steps, moved *every* adult herring onto a 10-cell autumn map (5 cells in the Gulf of Riga, 5 around
  Rügen). Baltic autumn spawners are depressed or collapsed (Bekkevold et al. 2016; MacKenzie & Ojaveer
  2018), the 25% was unsourced, and Rügen is a spring ground. Full C1 folds the autumn eggs into the
  spring bell *and* keeps adults on their 593-cell adult map (archived on `exp/baltic-c1-full`).
- **C1a, the packing alone.** This removes the 10-cell relocation and keeps the 25% autumn egg share,
  which is then released from the adult distribution. It splits C1 into its two parts.
- **C2, eastern cod spawning.** "SD 25 has been basically the only area supporting successful
  reproduction of the eastern Baltic cod since the mid-1980s" (Eero et al. 2012). The old map had 144
  cells, none in the Bornholm Basin and 89 at or above 57 °N. The new one has 14 cells: the Bornholm Basin
  and the Słupsk Furrow, defined as the adult map ∩ the SD 25 box ∩ bottom salinity ≥ 11 PSU
  (`scripts/build_cod_east_sd25_spawning.py`, parked on `fix/baltic-cod-east-sd25-spawning`).

## 2. Results (final-decade mean range over the 5 seeds; envelope in brackets)

| species | master | GreySeal base | + C1 | **+ C1a** | + C2 | + C1a + C2 |
|---|---|---|---|---|---|---|
| cod_west [4–25 kt] | 12.3–14.3 kt | 12.8–13.7 kt | **0.006–5.4 kt ✗** | 11.8–13.1 kt | 12.1–13.3 kt | 11.7–13.0 kt |
| cod_east [60–85 kt] | 64.2–66.3 kt | 61.6–63.0 kt | 59.1–62.7 kt ✗ | 61.2–64.4 kt | **58.9–61.5 kt ✗** | **58.5–63.2 kt ✗** |
| herring [0.8–3.0 Mt] | 2.47–2.61 Mt | 2.48–2.53 Mt | 1.89–1.92 Mt | 2.65–2.74 Mt | 2.50–2.58 Mt | 2.63–2.74 Mt |
| sprat [0.8–2.5 Mt] | 0.91–0.92 Mt | 0.89–0.91 Mt | 0.92–0.95 Mt | 0.898–0.904 Mt | 0.90–0.92 Mt | 0.89–0.91 Mt |
| flounder [20–100 kt] | 32.6–33.6 kt | 32.4–33.6 kt | 32.6–37.1 kt | 32.9–34.6 kt | 30.5–32.3 kt | 31.3–33.9 kt |
| **verdict (assessed)** | 5/5 | 5/5 | **3/5** | **5/5** | **4/5** | **4/5** |

For reference, the LTL hole fill (S3, `docs/baltic_ltl_coastal_holes_2026-11-08.md`) gives 4/5 on the same
base, with herring at 3.84–3.95 Mt.

## 3. Reading

- **Full C1 collapses cod_west, and it is not seed noise.** The best seed's final-decade mean (5.4 kt) lies
  below the base's worst (12.8 kt), so every seed lost at least 60%, and some went to near zero (minimum
  2.6 t). Herring falls 24%.
- **The packing was not what propped cod_west up.** C1a keeps cod_west at 11.8–13.1 kt, and herring rises
  7% to 2.65–2.74 Mt. The pre-stated test was "cod_west ≥ 12 kt and herring ≈ 2.5 Mt means the egg timing
  killed cod_west, not the packing". cod_west's lowest seed misses 12 kt by 2%, and herring sits 7% high.
  The reading therefore holds: what collapsed cod_west in full C1 was folding the autumn eggs into
  spring. The mechanism (the herring decline, through which part of cod_west's diet?) has not been
  measured.
- **C2 is a near miss, twice.** Alone it moves only cod_east, by about −3%. Every C2 seed lies below every
  base seed (complete separation, p ≈ 0.008 for 5 against 5), so the shift is real. It crosses the 60 kt
  floor because the base sat only 2.7% above it. cod_east was 64–66 kt on master, and the GreySeal column
  already took it to 62–63 kt. C1a lifts cod_east slightly, but not enough: C1a + C2 still has a seed
  below the floor (58.5 kt).

## 4. What ships, and the open decisions

- **Shipped:** C1a (`fix/baltic-herring-autumn-packing`), with the SP1b rates re-solved for it and
  guard tests, which are mutation-checked.
- **C2, eastern cod spawning in SD 25.** It is the better-sourced map, but cod_east falls about 3% under
  its floor. The options: (a) keep the old map and document it as a known error; (b) ship C2 with a
  bounded cod_east recalibration (its larval mortality, or the RV gate's scale) to restore the 2–3%;
  (c) treat the cod_east floor as the binding constraint it now is, and revisit it with the stock's own
  assessment.
- **The autumn egg share (25%, unsourced).** Removing it is literature-correct but collapses cod_west, so
  any future attempt needs cod_west recalibrated with it.
- **S3, the LTL holes.** These belong with the LTL column bias (the file is about 2.5× a correct
  integral) as a single project. See `docs/baltic_ltl_coastal_holes_2026-11-08.md`.
