---
name: Phase 12 calibration with REAL predators (post length-fix)
description: 2026-04-27 — re-ran joint 24-param DE after fixing background.py length=0 bug. f=6.18 single-seed / 7.22±0.91 multi-seed. 1/8 species in ICES range (sprat) — first time on a real-predator basis. Cod paradoxically overshoots because DE picked low natural mortality assuming seal would compensate.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Re-ran phase 12 calibration on master after the 2026-04-27 length-fix landed in `osmose/engine/background.py:get_schools()`. Predators are now ACTUALLY predating focal species (verified by mult=0/1/10 ramp test pre-calibration).

**Run stats:**
- DE objective: 12.89 (gen 1) → **6.18 single-seed** → **7.22 ± 0.91 multi-seed mean**
- Wall-clock: 215 min DE + 10 min multi-seed = **~3.75h total** (vs prior 7.6h)
- Why faster: many trial sims terminate early when species crash under combined active-predation + adult mortality + fishing
- Multi-seed std: **0.91** (vs prior 0.04 with inert predators) — optimum sits in noisy basin

**Comparison across phase 12 variants:**
| Run | Single-seed f | Multi-seed mean ± std | Predator state |
|---|---:|---:|---|
| Phase 12 no-predators | 5.24 | (n/a) | inactive |
| Phase 12 inert-predators (length=0 bug) | 3.53 | 3.55 ± 0.04 | injected, but length=0 → no predation |
| **Phase 12 real-predators** | **6.18** | **7.22 ± 0.91** | **fully active** |

The prior 3.53 was an artifact: predators had length=0 so the size-ratio gate rejected all prey; DE saw a free degree of freedom. With predators truly biting, the effective parameter space is harder, and DE lands at f≈6-7.

**50-y validation biomasses (3 seeds, multi-seed mean):**

| Species | Biomass | Target | ICES range | Ratio | Verdict |
|---|---:|---:|---:|---:|---|
| **sprat** | **1,740,865** | 1,500,000 | 0.8-2.5M | **1.16** | **IN RANGE ✓** |
| herring | 566,468 | 1,500,000 | 0.8-3M | 0.38 | undershoot (close, +110% needed) |
| flounder | 14,751 | 50,000 | 20-100k | 0.30 | undershoot (close, +35% needed) |
| stickleback | 3,508,651 | 200,000 | 50-500k | 17.5 | structural overshoot |
| smelt | 1,996,039 | 60,000 | 20-120k | 33 | structural overshoot |
| cod | 5,590,264 | 120,000 | 60-250k | 46.6 | structural overshoot (paradox) |
| perch | 6,396,776 | 20,000 | 8-50k | 320 | structural overshoot |
| pikeperch | 10,137,671 | 10,000 | 4-25k | 1014 | structural overshoot |

**1/8 in ICES range (sprat) — first time on a real-predator basis.** Two more close to range: herring (×0.38) and flounder (×0.30).

**Cod paradox:**
DE picked `mortality.additional.rate.sp0 = 0.265` (log10 −0.577) — far below prior phase 12's 5.00 (at upper bound). DE assumed seal predation would compensate. It cannot:
- Seal standing biomass: ~50,000 t × ingestion 13/yr = ~650,000 t/yr aggregate consumption
- Cod stock 5.6M t means even 100% directed seal predation removes ~12% per year
- DE under-set adult mortality without realizing predator capacity is too small
- Action item: lock cod adult mortality at biologically-realistic floor (e.g., 0.3-0.5/yr) in next calibration round to prevent DE from under-mortalizing it

**Why herring + flounder undershoot:**
With predators biting, top-down pressure on prey species (cod ate herring; seal eats flounder + cod) is now real. DE didn't compensate enough on the larval/recruitment side. Workaround: bump up reproduction or reduce larval mortality bound for sp1 (herring) and sp3 (flounder).

**Persistent structural overshoots (perch, pikeperch, smelt, stickleback):**
Same structural diagnosis as prior phase 12:
- Cormorant 5,000 t × 40 ingestion = ~200,000 t/yr aggregate, vs perch 6.4M t and pikeperch 10.1M t — order of magnitude too small
- No density-dependent recruitment in OSMOSE — populations grow until food-limited only
- Cannot be fixed by parameter tuning; needs Beverton-Holt or Ricker structural change

**Optimized parameters:**

```
Larval mortality (sp0..sp7):
  cod=11.98, herring=0.59, sprat=0.93, flounder=9.46,
  perch=1.42, pikeperch=1.90, smelt=1.17, stickleback=0.15

Adult mortality (sp0..sp7):
  cod=0.265 (LOW — see paradox), herring=4.57 (HIGH — at upper bound), 
  sprat=0.0011, flounder=0.036, perch=0.116, pikeperch=0.432, 
  smelt=0.0014, stickleback=0.926

Fishing (fsh0..fsh7):
  cod=0.075, herring=0.97, sprat=0.051, flounder=0.011,
  perch=0.023, pikeperch=0.078, smelt=0.0051, stickleback=0.0070
```

**Recommended follow-up paths:**

1. **Cod adult-mortality lower-bound** — set `mortality.additional.rate.sp0` minimum to ~0.3/yr to prevent DE from under-mortalizing cod. Re-run with this bound. Cheapest improvement.

2. **Herring + flounder larval-mortality lower-bound** — drop the lower bound for sp1 + sp3 to allow DE to find larger biomass. Or bump fecundity bounds.

3. **Density-dependent recruitment for percids** — single biggest structural change for the order-of-magnitude perch/pikeperch overshoot. ~1 engineer-week. Required for percids to ever come into range.

4. **Multi-seed objective during DE** — the high seed variance (std 0.91) suggests the single-seed optimum overfits noise. Use mean-of-3-seeds as the DE objective. ~3× slower but more robust.

5. **Accept the f=7.22 calibration as the new baseline** — first real-predator result. Document Baltic-OSMOSE as a qualitative trophic-coupling tool.

**Session artifacts (master branch, 2026-04-27):**
- `data/baltic/calibration_results/phase12_results.json` — 24 optimized params (this run)
- `data/baltic/calibration_results/phase12_results.predators_inert_bug.json` — prior phase 12 (3.53 single-seed, predators inert)
- `data/baltic/calibration_results/phase12_results.no-predators.json` — phase 12 with predators disabled
- `/tmp/osmose_calibration_phase12_real_predators.log` — full 3.75h DE log
- `/tmp/launch_phase12_real_predators.sh` — wrapper script (workers=8)

**Master state at end (2026-04-27):**
- Length-fix shipped, predators truly active, calibration baseline established
- Net: f=7.22 multi-seed, 1/8 strict in-range (sprat), 2/8 close-to-range (herring + flounder)
- Cod overshoot is a parameter-tuning fix; perch/pikeperch overshoot is structural
