---
name: project-baltic-salinity-spawning-followup
description: "Salinity-correct spawning areas do NOT fix the Baltic overshoot (3rd spatial lever ruled out); habitat follow-up = dynamic reproductive-volume recruitment gate, not maps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 18a62785-d85c-4be4-8f3a-e164e19add6a
---

**Salinity-dependent spawning-area attempt — DONE 2026-07-02, NEGATIVE. Branch `worktree-fix+baltic-salinity-spawning` (base = SP-A branch, tip `c602717`, NOT merged; data/baltic on master UNTOUCHED).** Third and final spatial lever in the Baltic long-term-stability arc [[project-baltic-stability-recalibration-spA]] [[project-baltic-grid-spB-spike]].

**What was tried:** refine spawning MAPS by a geographic salinity proxy (`scripts/baltic_salinity_spawning.py`) — cod → the saline deep reproductive volume (`cod_spawning.csv` 145→81 cells, lat≤57.5 Bornholm/Gdansk/S.Gotland ≥~11 PSU), percids → freshwater (`perch` 42→42 already fresh; `pikeperch` 19→3, dropped saline SW cells). `.pre-salinity.bak` backups (gitignored). Ran SP-A ε=0.2 params, 50yr×3seed via `baltic_stability_certify.py`.

**RESULT: NO effect on the overshoot.** 0/8 in-envelope (same as SP-A coarse baseline); **cod overshoot UNCHANGED 63.6×→63.7×** despite halving its spawning area; pikeperch slightly WORSE (68.5→72.2×) at 3 cells; others ~unchanged.

**WHY (decisive):** a spawning map controls only WHERE eggs are placed, not HOW MANY recruits the Beverton-Holt SR produces. The overshoot is a **population-level quantity (recruitment magnitude × mortality balance)** → NO spatial lever can touch it. This is why all three failed identically: **parameters (SP-A), cell density (SP-B spike), spawning location (this) — all spatial, all null.**

**▶▶ HABITAT FOLLOW-UP (written, `docs/baltic_habitat_followup_2026-07-02.md`):** the ONLY mechanism left = a **dynamic cod REPRODUCTIVE-VOLUME recruitment GATE** — wire CMEMS bottom salinity `so` + oxygen `o2b` forcing into the engine (`osmose/forcing/` is GridSpec-driven, already handles `so`), compute per-step deep-basin volume where salinity≥11 & O2≥2, and multiply cod's B-H recruitment by `RV/RV_ref` (clip [0,1]) in `osmose/engine/processes/reproduction.py`. Cod-first; a forcing+reproduction addition (~weeks), NOT a map edit. Closes the gap documented in `docs/baltic-fish-lifecycle.md:386-406`. Percids = accepted residual (weight 0.2, freshwater — no reproductive-volume mechanism; their limit is coarse-grid habitat under-resolution / basin-scale SR, [[project_percid_overshoot_diagnostic]]).

**RECOMMENDED DEFAULT:** accept Baltic as a ~15yr short-horizon Python-engine config (`nyear`=15 shipped `7d77862`); build the RV gate only for a specific long-horizon cod study. **Do NOT re-attempt params / grid density / spawning-map edits** — three experiments prove the overshoot is not spatial. **Reusable facts:** spawning grounds = per-species presence maps in `data/baltic/maps/*_spawning.csv` (`;`-sep 40×50, -99/0/1, south-first storage, np.flipud for engine orientation); wired via `baltic_param-movement.csv`; engine places eggs on the map (no engine change for spatial edits); edit pattern = `scripts/apply_ices_validation_fixes.py`.
