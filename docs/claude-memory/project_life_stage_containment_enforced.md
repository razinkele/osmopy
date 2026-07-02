---
name: Life-stage containment enforced across all Baltic species
description: 2026-04-21 — spawning ⊂ juvenile ⊂ adult now 100% for all 8 species. Fixed a pre-existing bug that made larvae from spawning cells without juvenile coverage effectively vanish.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Post-smelt-fix audit revealed the same bug existed in 6/8 Baltic focal species: juvenile maps were missing cells present in the spawning map, which means larvae from those cells had no nursery habitat and were effectively lost in the model.

**Severity before fix:**
| Species | spawn⊂juv coverage | spawning cells with no juvenile |
|---|---:|---:|
| flounder | 32% | 152 |
| cod | 57% | 63 |
| stickleback | 80% | 43 |
| sprat | 82% | 42 |
| herring | 63% | 19 |
| perch | 64% | 15 |
| pikeperch | 63% | 7 |
| smelt | 100% | 0 (already fixed) |

**Fix applied 2026-04-21:** For each species, enforced `juvenile := juvenile ∪ spawning ∪ ocean_mask` and `adult := adult ∪ juvenile ∪ ocean_mask`. Added 5 "pure nursery" cells to `perch_juvenile` and 2 to `pikeperch_juvenile` to keep them ecologically distinct from their respective spawning maps.

**After fix:**
- Cod juvenile: 239→302 (+63); adult: 324→352 (+28)
- Herring juvenile: 267→286 (+19); adult: 575→593 (+18)
- Sprat juvenile: 278→320 (+42); adult: 453→480 (+27)
- Flounder juvenile: 103→255 (+152); adult unchanged (already 580)
- Perch juvenile: 27→47 (+20 incl. 5 nursery); adult: 58→62 (+4)
- Pikeperch juvenile: 12→21 (+9 incl. 2 nursery); adult: 25→27 (+2)
- Stickleback juvenile: 529→572 (+43); adult: 543→604 (+61)

**All 25 maps remain unique** (verified via MD5 audit). **100% containment** across all 8 species.

**Sim impact (5-year smoke test, v3 vs v1):**
- perch: 7.5k → 10.8k t (+44%, **moved into ICES range**)
- flounder: 92.6k → 106.7k (+15%, still in range, slightly above target)
- Others ±2% (no significant short-term response; dominated by transient initial conditions)

**Why this was a real bug:** OSMOSE allocates new recruits at the spawning cell. If that cell isn't in the juvenile map, the recruit enters the simulation with no valid distribution, and on the first movement step it is re-allocated via fallback logic (or its cohort dies out locally). Either way, the population dynamics of those spawning populations were severely under-represented in all previous runs.

**How to apply:** no config changes. Maps overwrote in-place. The fix applies uniformly — not sensitive to calibration targets.

**Method note:** Detection via boolean set operations:
```python
spawn_in_juv = (sp_arr == 1) & (ju_arr == 1)
pct = 100 * spawn_in_juv.sum() / (sp_arr == 1).sum()
```
Any pct < 100% is a containment bug.

**Stickleback follow-up (v4):** After the life-stage containment fix, stickleback_adult was scaled back from 604 to 457 cells (central-basin focus) and stickleback_juvenile tightened to 253 coastal cells (hard coastal filter via 8-connectivity). Biomass dropped only 5.2% (6.6M → 6.3M t) — confirming stickleback's model overshoot is **parameter-driven, not spatial**. Further map tweaks won't close the gap; full NSGA-II re-calibration of mortality/recruitment parameters is required.

**Session tally:** 9 project memories across 2026-04-21:
1. LTL overlay land-convention fix
2. Baltic grid validated (BITS hauls)
3. Smelt spawning correction
4. Perch vs pikeperch spawning differentiated
5. Herring spring vs autumn spawning
6. Cod/flounder + percid life-stages
7. Clupeid + stickleback maps
8. Smelt life-stage consistency
9. **This memory — universal life-stage containment + stickleback v4 scale-back**

Net: 19 map files updated; 25/25 unique; 100% spawning⊂juvenile⊂adult; 4/8 species in ICES biomass range on 5-year smoke test; 4 species need full re-calibration (cod, herring, smelt, stickleback).
