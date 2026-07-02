---
name: Baltic sp6 whitefish → smelt swap
description: sp6 replaced Coregonus lavaretus (whitefish) with Osmerus eperlanus (smelt) — more abundant, forage-fish role. All parameters, maps, and ecology swapped on 2026-04-17.
type: project
originSessionId: bb1fa3c0-cc39-4f4a-973f-2ecd872f3159
---
sp6 in `data/baltic/` is now European smelt (Osmerus eperlanus, AphiaID 126736) instead of whitefish (Coregonus lavaretus, AphiaID 127180). Smelt is the dominant small forage fish in the Baltic (Gulf of Bothnia, Gulf of Finland, Gulf of Riga), ecologically more important than whitefish for food-web dynamics.

**Parameter changes** (scripted at `scripts/swap_whitefish_to_smelt.py`):
- `baltic_param-species.csv` sp6: lifespan 20→7, Linf 60→25 cm, K 0.12→0.35, t0 -0.5→-0.3, condition factor 0.0075→0.005, L50 35→10 cm, egg size 0.25→0.09 cm, egg weight 0.001→0.0005 g, relative fecundity 250→1000 eggs/g. Linf from BITS 2021-2023 (AphiaID 126736): max 26 cm, p99=24 cm, modal 16-18 cm.
- `baltic_param-movement.csv`: species names + file paths updated; adult map steps switched to Jun-Jan (non-spawning); spawning map steps 4-11 (Feb-May, spring spawner); `movement.lastAge.map20/21` = 8 = lifespan+1 (OSMOSE convention — covers ages [initial, lastAge)).
- `reproduction/reproduction-seasonality-sp6.csv`: autumn-spawner weights replaced with spring peaks at steps 4-11 (Feb-May, mode around Mar-Apr).
- `predation-accessibility.csv`: smelt prey row set to `0.6;0;0;0.1;0.5;0.6;0;0;0;0;0;0;0;0` (heavy access to cod/perch/pikeperch); smelt predator column set to `cod 0.05, stickleback 0.05, Meso 0.8, Macro 0.6, rest minimal` (zooplanktivore).
- `fishery-catchability.csv` + `fishery-discards.csv`: `gill_whitefish` → `gill_smelt`; species row label whitefish → smelt.
- `reference/biomass_targets.csv`: whitefish 15 kt → smelt 60 kt (mid estimate; smelt is ~2-4× more abundant).
- Movement maps renamed: `whitefish_{juvenile,adult,spawning}.csv` → `smelt_*.csv` (same spatial footprint retained — Gulf of Bothnia + Gulf of Finland + Baltic proper N to ~58°N; matches smelt distribution per BITS).

**Status:** 2411 tests pass, lint clean, Baltic 1-year engine runs cleanly; smelt tracked in all biomass/abundance output files. Backups at `.pre-smelt-swap.bak`.

**Known follow-ups (not blocking):**
- Initial population (`baltic_param-init-pop.csv`) still uses whitefish-era biomass; calibration target now 60 kt so current 1-yr run produces ~28 t smelt biomass — needs recalibration against updated target.
- Calibration artifacts in `data/baltic/calibration_results/*.json` still reference "whitefish" — historical output files, no functional impact; leave as-is or regenerate via calibration.
- Spatial maps were kept identical to whitefish (Gulf of Bothnia–centric) for simplicity; could extend to Gulf of Riga and eastern Baltic for broader smelt habitat if needed. BITS itself undersamples smelt (demersal trawl, smelt is pelagic/coastal) — Baltic coastal-lagoon surveys would be needed for a comprehensive footprint.

**How to apply:**
- Rerun the swap script (idempotent via backup-suffix check) if data is re-imported: `.venv/bin/python scripts/swap_whitefish_to_smelt.py`
- When tuning smelt: it's small-bodied, fast-growing, short-lived, highly fecund forage fish. Key validation: BITS modal length 16-18 cm, max 26 cm, spring spawner.
