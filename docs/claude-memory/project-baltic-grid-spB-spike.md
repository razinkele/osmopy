---
name: project-baltic-grid-spb-spike
description: SP-B grid-refinement spike DONE — finer grid does NOT cure the Baltic percid boom/bust (NO-GO qualified); real habitat detail is the untested lever
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**SP-B grid-refinement spike — DONE 2026-07-02, VERDICT = NO-GO (qualified). Branch `worktree-spec+baltic-grid-spB` (tip `3b3cde5`, NOT merged; has SP-A branch merged in for its machinery). data/baltic UNTOUCHED.** Follows the SP-A gate [[project-baltic-stability-recalibration-spA]]. Full brainstorm→feasibility(Explore agent)→spec→in-loop-review→plan→subagent-driven exec (T1-T4).

**Question:** does a finer Baltic grid cure the collapse-or-explode overshoot SP-A hit (0/8 in-ICES, percids ×44-68)? **Approach:** upsample-everything to 2× (80×100) — cheapest test of "more cells within the same footprint → lower per-cell density → less boom/bust", before any expensive real-map rebuild.

**RESULT (Stage 1: SP-A ε=0.2 params on the 2× grid, 50yr×3seed):** 8/8 persist (up from 7/8 — sprat recovers), **2/8 in-envelope (up from 0/8) BUT the 2 are herring+sprat which were UNDERshooting on coarse and grew into range — NOT the target.** The percid/cod boom/bust is **NOT cured**: cod ×64→45, perch ×47→38, **pikeperch ×69→96 WORSE, flounder ×13→65 WORSE**, smelt ×10→6, stickleback ×2.6→2.4. Fails both gate thresholds (≥3/8 in-env OR percids→single digits). Stage 2 (finer-grid re-calibration) SKIPPED — SP-A proved params can't fix percids, not worth hours of ~10×-slower sims.

**KEY FINDING: more cells within the same footprint does NOT cure the piscivore explosion → it's population-level, not a per-cell-density artifact.** Combined with SP-A: the Baltic percid boom/bust resists BOTH parameters (SP-A) AND cell density (SP-B).

**CONFOUND (why qualified, not definitive):** upsample preserves the blocky coarse coastline + exact footprints — adds cells, NO new habitat detail. Rules out "finer cells", NOT "a real finer grid" with genuine estuary/lagoon habitat for the percids.

**▶▶ RECOMMENDED NEXT (banked, not started):** do NOT commit to the full 25-map SP-B build. Cheapest deciding step = a small follow-up building a REAL bathymetry mask + REAL finer habitat maps for the estuary/lagoon percids (perch/pikeperch) specifically → does real habitat DETAIL (not just more cells) cure the overshoot? If that also fails → structural; pivot (accept Baltic as Python-engine short/medium-horizon config, or a sub-basin-structured model). **Engineering facts (reusable):** grid is fully parameterized (GridSpec); OSMOSE aligns forcing NetCDFs by CELL INDEX not coordinate (shapes matching suffices); CMEMS BAL source ~2km (12× finer than the 25km grid) so LTL auto-regrids; the manual blocker in a real build = the 4 coastal species (perch/pikeperch/smelt/stickleback) have NO ICES point source. `scripts/baltic_grid_upsample.py` (block_replicate + block_conserve_total ÷4) + `baltic_grid_spike_stage.py` are reusable.
