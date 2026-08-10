# Repair the cod spawning maps: give the Bornholm Deep back to cod_east

**Date:** 2026-08-10
**Status:** WITHDRAWN 2026-08-10, before implementation — three independent fatal flaws found by
> adversarial review (11 confirmed findings, 8 critical). See the WITHDRAWAL section at the end.
> The underlying defect is real and remains filed; this *fix* was wrong.
**Defect record:** `docs/baltic_cod_bornholm_spawning_defect_2026-08-10.md` (verified, unfixed)

## 1. The defect, restated

`cod_east_spawning.csv` has **zero** cells in the Bornholm Deep band (55.05–55.85 °N,
14.5–17.0 °E); its five nearest cells sit at 55.95 °N, the Hanö Bight rim.
`cod_west_spawning.csv` has **12** cells squarely in the Deep. So the eastern Baltic stock's
principal spawning ground — and since the late 1980s effectively its only productive one
(MacKenzie et al. 2000, doi:10.3354/meps193143; Köster et al. 2005,
doi:10.1016/j.icesjms.2005.05.004) — is attached to the western stock.

Root cause (traced, not inferred): `scripts/build_cod_ew_maps.py` builds the two maps from
**different sources** — east from the aggregate `cod_spawning.csv` masked to eastern columns, west
from the *adult* footprint masked west. The aggregate spawning map itself contains no Deep cells,
so the eastern map inherits an upstream gap while the western map acquired the Deep as a
side-effect of using a different source. The column masking is not at fault.

## 2. The decision, and why it is not a judgement call

**The Bornholm Deep belongs to cod_east, exclusively.** ICES assigns SD 25 to cod.27.24-32
(eastern) — the same stock definition `data/baltic/reference/biomass_targets.csv` already cites for
`cod_east`. SD 24 (Arkona) is the documented mixing zone and is *already* shared by the builder
(`WEST_COLS = 0..14`, `EAST_COLS = 13..49`); Bornholm is SD 25, east of that transition, so the
sharing precedent does not extend to it. No new convention is being invented.

`cod_west` retains SD 22–24: it loses 12 of its 126 spawning cells and keeps 114, all in the Belt
Sea and Arkona where the western stock actually spawns.

## 3. Design

### 3.1 Fix upstream, then rebuild

Three edits, in dependency order:

1. **`data/baltic/maps/cod_spawning.csv`** — add the Bornholm Deep cells. This is the aggregate
   map every downstream map inherits from, and the true location of the gap.
2. **`scripts/build_cod_ew_maps.py`** — build `cod_west_spawning` from the same aggregate spawning
   map as the east (masked west), not from the adult footprint, so the split becomes a genuine
   partition of one documented footprint. Record in the docstring why the adult-footprint
   shortcut was wrong.
3. **Regenerate** `cod_east_spawning.csv` and `cod_west_spawning.csv` from the builder.

The values written into the aggregate map must match the map's existing convention (inspect it —
presence/absence vs weights — and follow it rather than assuming binary).

### 3.2 Expected direction, stated before measuring

cod_east gains productive spawning habitat, so its recruitment should **rise**. That is the
favourable direction: cod_east currently sits at 65,209 t with its **floor** the binding edge
(8.0% margin) and 23.3% of headroom to the ceiling
(`docs/baltic_rv_ref_sweep_2026-08-09.md`). cod_west loses ~10% of its spawning cells and should
fall slightly from 12,875 t, against a 68.9% margin to its own floor — ample.

Stating this in advance so the A/B can contradict it. If cod_east *falls*, the mechanism is not
what this design assumes and adoption should not proceed on the gate result alone.

### 3.3 Interaction with the RV gate — the reason this matters beyond tidiness

The RV gate is enabled for **cod_east only** and is the dominant control on that stock (gate-off →
137,302 t, 1.61× over ceiling). It represents Bornholm-driven reproductive success. Today it is
applied to a stock that cannot spawn in Bornholm; after this fix, gate and spawning ground refer to
the same water for the first time.

Consequence for the plan: the gate's *effect size* may change, so the post-fix configuration must
be re-measured against the admissible `ref` band (~115–161) rather than assuming 150 still sits
where the sweep put it. This is a **report-and-check**, not a licence to re-tune: `ref` stays at
150 unless the A/B shows the fix pushes cod_east outside the envelope, in which case the result is
reported and the decision escalated rather than absorbed by a parameter change.

## 4. Acceptance

* **A1 — geometry (offline, seconds).** After regeneration: `cod_east_spawning` has ≥ 8 cells in
  the Deep band; `cod_west_spawning` has 0; neither map's total cell count changes by more than
  the Deep transfer plus whatever the aggregate repair adds; both remain non-empty and land-free
  (cross-check against the grid mask from `baltic_grid.nc`).
* **A2 — provenance (offline).** Both spawning maps are reproducible by running
  `scripts/build_cod_ew_maps.py` from the committed aggregate map — i.e. no hand-edited outputs.
* **A3 — certification A/B.** 50 yr × 5 seeds via the harness, identity-pinned gate (5 assessed +
  perch + stickleback), off-arm PASS as precondition. Report cod_east and cod_west deltas plus the
  realised RV-gate factor trajectory.
* **A4 — direction check.** cod_east must not *fall* (see §3.2). A fall means the assumed mechanism
  is wrong; stop and report rather than adopting a PASS whose cause is not understood.

## 5. Out of scope

* Re-tuning `ref`, mortalities, or accessibility — see §3.3.
* Enabling `reproduction.rv.spatial.*` (unused; would read the aggregate map, so it benefits from
  the repair but is a separate mechanism).
* The juvenile/adult cod maps — only the *spawning* maps are defective.
* Anything about the RV series' content, which is settled:
  `docs/baltic_rv_divergence_explained_2026-08-10.md`.

## 6. Deliverables

1. Repaired `data/baltic/maps/cod_spawning.csv` and regenerated east/west spawning maps.
2. Amended `scripts/build_cod_ew_maps.py` (single-source split + docstring rationale).
3. `tests/test_baltic_cod_spawning_maps.py` — A1 geometry assertions and the A2 reproducibility
   check, so the defect cannot silently return.
4. A/B report and, on PASS + direction check, a certification record; on FAIL or a wrong-direction
   result, the negative record.


---

# WITHDRAWAL (2026-08-10)

Reviewed before any code was written. Three independent flaws, each sufficient on its own.

## 1. §3.1 item 2 would have destroyed cod_west's spawning grounds

I called the builder's use of the adult footprint for `cod_west_spawning` a "shortcut [that] was
wrong" and told the implementer to rebuild it from the aggregate map masked west. Measured
consequence: `cod_west_spawning` collapses from **126 cells to 3** — and all three sit at 55.95 °N
× 15.0–15.8 °E, i.e. in SD 25, *eastern* water. The western stock would have been left with no
Belt Sea and no Øresund spawning at all, for 8 of 24 timesteps every year (`map2`, all
spawning-age fish).

The builder's docstring note I overrode — "the aggregate spawning map is eastern-biased" — is
**correct**, and has been true for the file's entire git history (141 cells from col 12 at
`396e496`; 68 in the pre-mask backup, also from col 12). The aggregate is an *eastern-stock*
spawning map, not a two-stock aggregate, so it cannot be partitioned into a western stock. The two
maps have different sources because the two stocks have different spawning grounds — biology, not
a shortcut. Literature: western Baltic cod spawn in SD 22/23 (Belt Sea, Øresund), eastern in the
deep basins of SD 25+ (Weist et al. 2019, doi:10.1371/journal.pone.0218127; Schade et al. 2022,
doi:10.1371/journal.pone.0274476).

## 2. My SD-to-column premise was inverted, making A1 unreachable

§2 argued "Bornholm is SD 25, east of that transition, so the sharing precedent does not extend to
it." In this grid that is false. `lon = 10.2 + 0.4·col`, so the Deep band (14.5–17.0 °E) is
cols 11–17 — and cols **13–14 ARE the shared transition** (`WEST_COLS = 0..14`,
`EAST_COLS = 13..49`), while cols 11–12 are west-only and never reach `cod_east` at all.

So a column-masked split of one source cannot give the Deep to `cod_east` exclusively: cells at
cols 13–14 land in *both* maps, cells at cols 11–12 land only in the west. **A1 ("cod_west has 0
Deep-band cells") is unsatisfiable under every option in §3.1**, including the reviewer's own
fallback — and the spec explicitly declined to change the masks.

## 3. The benefit I predicted does not exist — egg production is spatially blind

§3.2's mechanism ("cod_east gains productive spawning habitat, so recruitment should rise") has no
pathway in this engine. `processes/reproduction.py:141-147` computes eggs as
`sex_ratio · fecundity · SSB · seasonality · 1e6` — no spatial term; new schools are created with
`cell_x/cell_y = -1` and placed the following step by the map covering age_dt 0, which is
`cod_east_juvenile` (`map26`, all 24 steps) — and that map *already* contains the Bornholm cells.

The spawning map only relocates age ≥ 4 adults for 10 of 24 steps, and `cod_east_adult` already
places them in Bornholm for the other 14. Worse, presence/absence maps distribute schools
uniformly among presence cells (`movement.py:487-515`), so adding cells **dilutes** occupancy.
A4 ("cod_east must not fall") therefore gates on a pathway the code does not implement and could
abort a legitimate result for the wrong reason.

## What survives

* **The defect is still real** (`docs/baltic_cod_bornholm_spawning_defect_2026-08-10.md`): the
  eastern stock's principal spawning ground is in the western stock's spawning map. But its
  practical consequence is **smaller and different** than that record implies — it misplaces adult
  *distribution* during the spawning window, not recruitment volume. That record should be read
  with this correction attached.
* **Any future fix must** (a) leave `cod_west_spawning` adult-derived or build a genuine two-stock
  spawning footprint first; (b) change the column masks, or abandon exclusivity, since the Deep
  straddles the transition; (c) justify itself by adult distribution and predation exposure, not
  by recruitment; and (d) replace A4 with a check tied to a mechanism that exists.
* The cheapest correct action may be **none**: with egg production spatially blind and
  `cod_east_adult` already covering Bornholm, the model consequence of the map defect is a
  10-of-24-step adult distribution difference whose sign is not obvious.
