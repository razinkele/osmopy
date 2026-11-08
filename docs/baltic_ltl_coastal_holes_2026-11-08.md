# Baltic LTL: the 65 empty coastal cells, and why filling them is not yet shippable (2026-11-08)

**Status:** fix built and verified on branch `fix/baltic-ltl-coastal-holes`, **not merged**. Filling the
holes breaks certification. The cause is a larger, pre-existing bias in how the LTL file is built, so the
decision belongs with that bias rather than with the holes.

## 1. The defect (review S3)

65 of the 616 Baltic ocean cells (10.6%) were exactly 0 in all six LTL groups and all 24 frames of
`data/baltic/baltic_ltl_biomass.nc`. The 2026-04 mask rebuild (912 → 616 cells, `d61ea1c1`) added coastal
cells without regenerating the file. Plain nearest-neighbour regridding snapped them to land pixels of the
native ~1 nm ERGOM grid. Fish mapped there found no plankton or benthos (herring adult map: 65 of 593
cells). Source: `docs/reviews/2026-11-07-temporal-spatial-dynamics-review.md` §3.

## 2. The fix, and that it touches only the holes

`regrid()` gained an opt-in masked fallback: a wet target cell whose nearest source pixel is invalid takes
the nearest valid pixel, using the same rule and metric as the oxygen forcing. `scripts/build_baltic_ltl.py`
rebuilds the file from its cached source (CMEMS ERGOM monthly 2024), and its `--check` against the old
file shows:

- all 551 cells that had forcing are **bit-identical**;
- 65/65 holes are filled, and no ocean cell is left empty;
- the 3 land cells that still carried forcing from the old mask are now 0;
- domain-total LTL rises about 20% (Diatoms 0.960 → 1.146 Mt, mean over frames).

## 3. It fails certification

50 yr × 5 seeds, Python engine, on top of the GreySeal accessibility column (PR #227):

| species | envelope | GreySeal base, final decade | + hole fill | change |
|---|---|---|---|---|
| herring | 0.8–3.0 Mt | 2.48–2.53 Mt | **3.84–3.95 Mt ✗** | +54% |
| cod_west | 4–25 kt | 12.8–13.7 kt | 23.2–23.9 kt | +77% |
| cod_east | 60–85 kt | 61.6–63.0 kt | 73.6–74.8 kt | +19% |
| flounder | 20–100 kt | 32.4–33.6 kt | 48.0–49.1 kt | +46% |
| sprat | 0.8–2.5 Mt | 0.89–0.91 Mt | 0.95–0.98 Mt | +7% |
| perch (indicative) | 8–50 kt | 46.1–48.6 kt | 63.4–64.9 kt ✗ | +35% |

The verdict falls from **5/5 to 4/5 assessed**, with herring above its ceiling. The calibration was
fitted with 10% of the ocean empty of food, so restoring that food overfeeds the system. The size of the
jump, though, comes from the next section.

## 4. Why the filled cells carry so much food: the LTL column bias

`bgc_to_ltl` averages a native pixel's concentration over its depth **levels** in 0–50 m and multiplies
by a full **50 m** column. Two biases stack:

- **The levels crowd near the surface.** There are 22 levels in 0–50 m, about 1.0 m thick at the top and
  5.8 m at depth. An unweighted mean over levels therefore over-weights surface plankton, which overstates
  even deep cells.
- **A shallow pixel is credited with a 50 m column it does not have.**

`scripts/baltic_ltl_depth_bias.py` compares a thickness-weighted 0–50 m integral with the current product:

| | phyc | zooc |
|---|---|---|
| domain total, correct / current | **0.443** | **0.388** |
| per-cell median ratio, cells that had forcing | 0.761 | 0.759 |
| per-cell median ratio, the 65 filled cells | **0.083** | **0.083** |
| the hole fill adds, current integration | +19.7% | +23.0% |
| the hole fill adds, correct integration | **+4.7%** | **+6.2%** |

The holes are the shallowest cells in the domain, so the current integration inflates them about 12×. The
whole file is overstated about 2.5×. The calibration has absorbed that factor, so it is not by itself a
reason to distrust the certified config, but it does mean that any change touching shallow cells is read
through a 12× lens.

## 5. Options

1. **Leave the holes empty for now** (recommended). Keep the branch as the record, and treat the LTL as
   one project: a depth-correct integral (the whole file at about 0.4×) plus the hole fill, followed by a
   recalibration. Filling the holes alone fixes a 5–6% food gap through a method that turns it into 20%.
2. **Ship the fill and recalibrate** around it (at least herring). This bakes the column bias deeper into
   the calibration.
3. **Fill the holes with depth-correct values only.** That adds about 5% food, but the filled cells would
   then follow a different rule from their neighbours. Not recommended.

The C1 and C2 spawning-map fixes (branch `fix/baltic-spawning-maps`) do not depend on this, and they
are certified separately, without the hole fill.
