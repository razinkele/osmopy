---
name: Herring spring vs autumn spawning differentiated
description: 2026-04-21 — herring_spawning and herring_spawning_autumn were identical (98 cells each). Replaced: spring 51 cells (basin-wide coastal), autumn 10 cells (Rügen + Gulf of Riga BAS).
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
`data/baltic/maps/herring_spawning.csv` and `herring_spawning_autumn.csv` previously shared an identical 98-cell footprint that was SW-biased (Bay of Mecklenburg 33, Belt/Sound 15). This was ecologically wrong — spring-spawning is Baltic-wide across WBSS + CBH + Gulf of Riga + Bothnian stocks; autumn-spawning is much more restricted.

Replaced on 2026-04-21:

- **herring_spawning.csv (spring)**: 51 cells — Kiel Bay+Mecklenburg (7), Arkona+Rügen/Greifswalder Bodden (6), Bornholm+Gdansk (6), E Gotland coast (4), W Gotland coast (3), N Baltic Proper (1), Archipelago+Åland (4), Gulf of Finland coast (5), Gulf of Riga (6), S Bothnian Sea coast (5), Bothnian Bay/Quark (4). Matches Gröger et al. (2014) WBSS distribution, Bekkevold et al. (2023) CBH southern + northern components, plus ICES stocks `her.27.28` (Gulf of Riga) and `her.27.3031` (Bothnian).
- **herring_spawning_autumn.csv**: 10 cells — Greifswalder Bodden + Rügen (3), Pomeranian Bay (2), Gulf of Riga BAS (4), central historic (1). Matches "low fisheries importance but potentially temporally increasing" W Baltic autumn-spawners + Gulf of Riga Baltic Autumn Spawners (BAS) from Bekkevold et al. (2023). Cell count ratio ≈ 20% of spring, aligned with literature estimate of 20-30% reproductive share.
- **Overlap**: 7 cells in SW Baltic (Rügen, Pomeranian) + Gulf of Riga where both seasons do spawn — biologically correct, not a bug.
- Distinct MD5s; UI overlay pipeline loads both.

**Why:** part of the "13/26 maps share footprints" problem. Identical maps produced model behaviour where the autumn-spawning cohort (Sep-Nov, steps 16-21 in `baltic_param-movement.csv`) used the same spatial distribution as spring-spawning (Mar-Jun, steps 4-11) — biologically invalid.

**How to apply:** no config change needed; `movement.file.map6` + `movement.file.map25` already point at these filenames.

**Remaining duplicate groups still unfixed:**
- `cod_adult` + `flounder_adult` (431 cells — demersal generalist template)
- `pikeperch_adult` + `pikeperch_juvenile` (still share obsolete 88-cell footprint)
- `perch_adult` + `perch_juvenile` (still share 98-cell footprint — same MD5 `3d242ad033`)
- `herring_adult` + `herring_juvenile` + `sprat_juvenile` (544 cells — "everywhere pelagic" template)
- `stickleback_adult` + `stickleback_spawning` (147 cells)

**References:**
- Gröger, J. P., Hinrichsen, H.-H., & Polte, P. (2014). Broad-Scale Climate Influences on Spring-Spawning Herring Recruitment in the Western Baltic Sea. PLoS ONE 9(2): e87525. https://doi.org/10.1371/journal.pone.0087525
- Bekkevold, D., Berg, F., Polte, P., et al. (2023). Mixed-stock analysis of Atlantic herring: a tool for identifying management units and complex migration dynamics. ICES J Mar Sci 80(1): 173-184. https://doi.org/10.1093/icesjms/fsac223
- Gröger, J. P., & Gröhsler, T. (2001). Comparative analysis of alternative statistical models for differentiation of herring stocks based on meristic characters. J Appl Ichthyol 17(5): 207-219. https://doi.org/10.1046/j.1439-0426.2001.00254.x
