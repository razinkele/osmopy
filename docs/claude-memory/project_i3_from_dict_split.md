---
name: I-3 from_dict monolith split — COMPLETE
description: Split EngineConfig.from_dict into coordinator + 5 subsystem parsers (config.py refactor) — all 5 tasks done
type: project
originSessionId: a98c4c79-5e0f-4e75-b693-503237310f59
---
I-3 plan: split `EngineConfig.from_dict` (611 → 417 lines) into coordinator + 5 helpers.

**Why:** Monolithic parser was hard to test/maintain; each subsystem parser is now independently readable.

**How to apply:** Plan is complete. All 5 helpers are module-level functions in `osmose/engine/config.py`.

## Completed Tasks

| Task | Helper | Commit |
|------|--------|--------|
| 1 | `_parse_growth_params` | `de71cde` |
| 2 | `_parse_reproduction_params` | `a21c8c7` — also removed duplicate `focal_lmax` |
| 3 | `_parse_predation_params` | `52c65a4` — largest block (feeding stages + post-predation) |
| 4 | `_merge_focal_background` | `3dd4551` — focal/background concatenation via dict |
| 5 | `_parse_output_flags` | `05d1d03` — output recording, diet, cutoff, bins, bioen |

Branch: `refactor/from-dict-split-2026-04-12`
Final gate: 2169 tests passed, 15 skipped, 12/12 parity bit-exact, ruff clean.
`from_dict` reduced from 611 → 417 lines (32% reduction).
