---
name: Phase 12 calibration with predators active (engine fix follow-up)
description: 2026-04-25/26 — re-ran joint 24-param DE after fixing reproduction.py and activating seal+cormorant. Best objective ever (3.55 multi-seed mean) but still 0/8 species in formal ICES range. Perch/pikeperch need structural model changes, not parameters.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Re-ran phase 12 calibration on master after the 2026-04-25 engine fix landed (`37bc1d1` reproduction.py slicing) and predators were activated (`f54bb26` baltic_all-parameters.csv `osmose.configuration.background` line). 7.6h wall-clock.

**Result summary:**
- DE objective: 11.10 (gen 1) → **3.53 single-seed** → **3.55 ± 0.04 multi-seed mean**
- f frozen at 3.53 from gen 6 onwards (9 gens unchanged) — strong convergence
- Best result of the entire calibration arc:
  - Phase 1 alone (20-y eval, no predators): f = 5.17
  - Phase 2 stacked (50-y, no predators): f = 3.93
  - Phase 12 (50-y, no predators, pivot bounds): f = 5.24
  - **Phase 12 with predators: f = 3.53** (−33% from prior best phase 12)

**50-y validation biomasses (3 seeds, multi-seed mean):**

| Species | Biomass | Target | Range | Ratio | Verdict |
|---|---:|---:|---:|---:|---|
| flounder | 126,825 | 50,000 | 20-100k | 2.54 | HIGH ×1.27 (just outside upper) |
| sprat | 3,964,499 | 1.5M | 0.8-2.5M | 2.64 | HIGH ×1.59 (just outside upper) |
| smelt | 362,029 | 60,000 | 20-120k | 6.03 | HIGH ×3.0 |
| herring | 7,833,774 | 1.5M | 0.8-3M | 5.22 | HIGH ×2.6 |
| stickleback | 1,594,477 | 200,000 | 50-500k | 7.97 | HIGH ×3.2 |
| cod | 1,725,905 | 120,000 | 60-250k | 14.38 | HIGH ×6.9 |
| perch | 6,731,449 | 20,000 | 8-50k | 336.57 | HIGH ×134 |
| pikeperch | 5,208,224 | 10,000 | 4-25k | 520.82 | HIGH ×208 |

**0/8 in formal ICES range** — strict scorecard worse than phase 2 stacked (1/8). But:
- f = 3.55 is the lowest objective ever achieved
- 3 species (flounder, sprat, smelt) within ×3 of upper bound
- 5 species within ×7 — distribution mass is closer to target than ever before

**Why DE picked the params it did:**
The optimizer recognized that seal predation (~78,500 t/yr aggregate consumption) handles meaningful natural mortality on cod/herring/sprat/flounder. So:
- DE backed off `mortality.additional.larva.rate.sp1` (herring larval): 2.17 → 0.13 (−94%) — seal does it
- DE backed off `fisheries.rate.base.fsh3` (flounder fishing): 1.94 → 0.12 (−94%) — cod/seal does it
- DE backed off `mortality.additional.rate.sp4` (perch adult): 0.046 → 0.0085 (−82%) — cormorant does it
- DE pushed `mortality.additional.rate.sp0` (cod adult): 1.75 → 5.00 (+186%, at upper bound)

**The persistent perch + pikeperch overshoot is structural, not parameter-tunable:**
- Perch standing biomass 6.7M t in the model
- Cormorant total predation ~20,000 t/yr (consumption-equivalent; 500 t × 40/yr)
- Even 100% directed at perch would remove 0.3% per year — far too small to constrain a 6.7M t population
- To suppress perch by 100× via cormorant alone would need cormorant standing biomass ~1.7M t (3000× actual)
- **Conclusion:** density-dependent recruitment, carrying capacity, or fishing-effort feedback is the missing structural piece for percids. No parameter tuning will fix this.

**The cod overshoot (×6.9) is also borderline structural:**
- DE pushed cod adult mortality to the upper bound (5.0/yr at log10=0.7)
- DE wants more — could try widening the bound further
- But realistic Baltic cod natural mortality is ~0.3/yr; pushing to 5/yr is biologically implausible

**What this run validates:**
- ✅ Engine fix (37bc1d1) works end-to-end through 7.6h × 768 evals
- ✅ Background-species predation pathway is functional
- ✅ DE benefits from realistic top-down mortality (lower larval-mortality rates for prey species)
- ✅ Stickleback in-range threshold is approachable when cormorant predation is real

**What this run doesn't fix:**
- ❌ Perch/pikeperch overpopulation requires model structure changes
- ❌ Strict 0/8 species in ICES range is unchanged

**Optimized parameters (for the record):**

```
Larval mortality (sp0..sp7):
  cod=1.15, herring=0.13, sprat=3.87, flounder=11.78,
  perch=2.81, pikeperch=3.09, smelt=4.73, stickleback=1.06

Adult mortality (sp0..sp7):
  cod=5.00 (at upper bound), herring=0.84, sprat=0.12, flounder=0.045,
  perch=0.0085, pikeperch=0.24, smelt=0.12, stickleback=0.15

Fishing (fsh0..fsh7):
  cod=0.45, herring=0.11, sprat=0.10, flounder=0.12,
  perch=0.014, pikeperch=1.96 (near upper), smelt=0.078, stickleback=0.045
```

**Parameters at bounds (DE wanted to go further):**
- `mortality.additional.rate.sp0` (cod adult) = 5.00 at upper log10=0.7
- `mortality.additional.larva.rate.sp3` (flounder larval) = 11.78 near upper log10=2.0 (=100)
- `fisheries.rate.base.fsh5` (pikeperch fishing) = 1.96 near upper log10=0.5 (=3.16)

Widening these bounds further may give DE more room but is biologically implausible.

**Recommended follow-up paths:**

1. **Add density-dependent recruitment for perch + pikeperch** — single biggest structural change to address the order-of-magnitude overshoot. Beverton-Holt or Ricker. ~1 engineer-week.

2. **Increase predator standing biomass artificially (×10)** — bump cormorant from 500 t to 5000 t to test if stronger top-down control closes the perch gap. Biologically dubious but methodologically informative.

3. **Multi-stock cod** — split cod into Western (cod.27.22-24) + Eastern (cod.27.24-32) per ICES. Each gets independent parameters.

4. **Accept and ship the f=3.55 calibration** — best we have. The objective improvement from predators is real. Document Baltic-OSMOSE as a qualitative trophic-coupling tool, not a target-matching one.

**Session artifacts (master branch, 2026-04-26):**
- `data/baltic/calibration_results/phase12_results.json` — 24 optimized params
- `data/baltic/calibration_results/phase12_results.no-predators.json` — prior phase 12 (5.24 single-seed)
- `/tmp/osmose_calibration_phase12_predators.log` — full 7.6h DE log
- `/tmp/launch_phase12_with_predators.sh` — wrapper with PYTHONUNBUFFERED=1 + OSMOSE_DE_WORKERS=8
- `scripts/compare_phase12_runs.py` — comparison utility

**Master state at end (2026-04-26):**
- Engine fix shipped, predators active, calibration result preserved
- All Tier 1 plan tasks effectively complete (T3-T5 unblocked, T8 re-run with predators)
- Net: best objective achieved (3.55), but 0/8 strict in-range. Structural model changes are the next lever.
