---
name: Clupeid + stickleback maps differentiated (final duplicate fix)
description: 2026-04-21 — resolved the last two duplicate groups. All 25 Baltic distribution maps now have unique footprints.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Closed the final duplicate groups from the 2026-04-21 map-validation audit:

**Clupeid pelagic template (544 cells, was 3 maps identical):**
- `herring_adult.csv` → 575 cells, basin-scale pelagic, all sub-basins (Bothnian Sea 97, Bornholm/Gdansk 94, Gulf of Riga 68, N Baltic Proper 49, …). Reflects adult herring's summer-feeding migration across the whole Baltic.
- `herring_juvenile.csv` → 267 cells, **coastal-only (hard filter via 8-connectivity land-neighbour check)**. Matches nursery-habitat biology: Greifswalder Bodden is WBSS's key nursery (Polte et al. 2021), coastal archipelago for CBH, Pärnu Bay for `her.27.28`.
- `sprat_juvenile.csv` → 278 cells, **central-basin focused** (Bornholm/Gdansk 86, Gulf of Riga 51, N Baltic Proper 42, E Gotland 31). Zero in Gulf of Bothnia (sprat absent — salinity constraint). Coastal suitability reduced 30%.

**Stickleback (147 cells, was 2 maps identical):**
- `stickleback_adult.csv` → 543 cells, basin-scale pelagic. Reflects the post-2010 Baltic three-spined stickleback population explosion with massive offshore summer aggregations (Olsson et al. 2019).
- `stickleback_spawning.csv` → 213 cells, coastal-only (hard filter). Concentrated in freshened shallow bays: Gulf of Riga 38, Bothnian Sea 33, Gulf of Finland 33, Bothnian Bay 27. Matches shallow vegetated-bay reproductive requirement.

**Final fingerprint audit:** 25/25 maps unique. Zero duplicates. Closes the "13/26 share footprints" finding from today's validation.

**Why:** completing the duplicate-template cleanup so every focal-species life stage has ecologically-defensible spatial distribution rather than a handful of habitat templates applied uniformly.

**How to apply:** no config changes — canonical filenames preserved. All `movement.file.map*` entries unchanged.

**Method note:** coastal cells defined as ocean cells with ≥1 land neighbour in 8-connectivity; used as a hard filter for herring_juvenile + stickleback_spawning (biological nursery/spawning require shoreline proximity at this resolution). Deterministic PRNG (seed=42) for partial-coverage regions so maps are reproducible.

**References:**
- Polte, P. et al. (2021) cited via Bekkevold et al. (2023) https://doi.org/10.1093/icesjms/fsac223 — Greifswalder Bodden as WBSS herring nursery
- Olsson, J. et al. (2019) — Baltic three-spined stickleback population explosion
- Lappalainen et al. (2003), Bučas et al. (2022), Ivanauskas et al. (2025) — Curonian Lagoon pikeperch (earlier memos)

Session summary: 13 maps fixed across 5 duplicate groups + 1 standalone (smelt). Total refactor over 2026-04-21. Project memory "Smelt spawning correction" + "Perch vs pikeperch differentiated" + "Herring spring vs autumn spawning" + "Cod/flounder + percid life-stages" + "Clupeid + stickleback maps" document the full cleanup trail.
