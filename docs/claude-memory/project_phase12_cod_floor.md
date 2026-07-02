---
name: Phase 12 calibration with cod adult-mortality floor 0.3/yr
description: 2026-04-27 — added 0.3/yr lower bound on cod adult mortality. f=6.01 single / 6.15 ± 0.19 multi-seed (5× tighter than prior 0.91 std). Stickleback moved into ICES range. But cod stayed at 6M t — DE compensated by dropping cod larval mortality 24× to keep recruitment high. Confirms density-dependent recruitment is the structural fix needed.
type: project
originSessionId: 1234202a-3436-4b5c-8595-2206a969a1ef
---
Re-ran phase 12 calibration with `mortality.additional.rate.sp0` lower bound raised from log10(0.001) = −3.0 to log10(0.3) ≈ −0.523 (effective bound 0.3/yr to 5.0/yr). All other params unchanged.

**Run stats:**
- DE objective: 145.2 (gen 1) → 9.12 (gen 2) → **6.01 single-seed** → **6.15 ± 0.19 multi-seed mean**
- Per-seed: 6.0083, 5.9680, 6.1973, 6.0671, 6.4848 — much tighter than prior run
- Wall-clock: 296 min DE + 17 min multi-seed = **~5h total** (vs prior 3.75h, slower because fewer early-extinction shortcuts)
- Multi-seed std: **0.19 (vs prior 0.91)** — 5× more stable optimum

**Comparison vs prior phase 12 real-predators (no cod floor):**

| Run | Single-seed | Multi-seed mean ± std | Cod adult mortality | Cod biomass |
|---|---:|---:|---:|---:|
| Phase 12 real-predators no-floor | 6.18 | 7.22 ± 0.91 | 0.265/yr | 5,590,264 |
| **Phase 12 cod-floor (this run)** | **6.01** | **6.15 ± 0.19** | **3.72/yr** | **6,030,849** |

**The escape-route lesson:**
The cod-floor forced `mortality.additional.rate.sp0` to 3.72/yr (14× higher than prior 0.265). But cod biomass barely moved (5.59M → 6.03M, +8%). Why?

DE compensated by dropping cod **larval** mortality:
- Prior phase 12: `mortality.additional.larva.rate.sp0` = 11.98 (log10 1.08)
- This run: `mortality.additional.larva.rate.sp0` = 0.50 (log10 −0.30)
- That's a **24× reduction** in cod larval mortality

So DE found an equivalent recruitment level: kill more adults, but let more juveniles survive. Net cod production unchanged. **OSMOSE has no density-dependent recruitment**, so any single-axis mortality constraint can be compensated for by adjusting the other axis.

**Implication:** Constrains alone won't fix cod, perch, pikeperch, flounder structural overshoots. The ONLY real fix is adding stock-recruitment compensation (Beverton-Holt or Ricker). Without that, DE will always find an escape route around any parameter floor we set.

**50-y validation biomasses (3 seeds, multi-seed mean):**

| Species | Biomass | Target | ICES range | Ratio | Verdict |
|---|---:|---:|---:|---:|---|
| **stickleback** | **366,969** | 200,000 | 50-500k | **1.83** | **IN RANGE ✓** (was 17.5×) |
| herring | 3,558,605 | 1,500,000 | 0.8-3M | 2.37 | just over upper (×1.19) |
| sprat | 4,756,278 | 1,500,000 | 0.8-2.5M | 3.17 | out (×1.9 of upper); was IN range |
| smelt | 733,673 | 60,000 | 20-120k | 12 | over (was 33×) |
| flounder | 1,504,818 | 50,000 | 20-100k | 30 | overshoot (flipped from 0.30 undershoot) |
| cod | 6,030,849 | 120,000 | 60-250k | 50 | overshoot (was 46.6) |
| pikeperch | 2,865,509 | 10,000 | 4-25k | 287 | structural (was 1014) |
| perch | 20,307,536 | 20,000 | 8-50k | 1015 | structural (worse than 320) |

**1/8 in ICES range (stickleback).** Same in-range count as prior run, but different species. Herring close to range (×1.19 of upper).

**Notable shifts vs prior run:**
- **Stickleback** dropped 90% (3.5M → 367k) → moved into range. Why: higher cod adult mortality + cod ate stickleback.
- **Herring** rose 6× (566k → 3.56M). Less seal pressure with cod competing for seal predation budget.
- **Sprat** rose 2.7× (1.74M → 4.76M). Lost in-range status.
- **Flounder** rose 100× (14.7k → 1.5M). DE backed off seal pressure on flounder.
- **Perch** got 3× worse (6.4M → 20.3M). Cod no longer suppressing perch as effectively (cod biomass distribution shifted).
- **Pikeperch** dropped 3.5× (10.1M → 2.87M). Better ratio, still ×287 of target.

**Optimized parameters:**

```
Larval mortality (sp0..sp7):
  cod=0.50 (vs prior 11.98 — 24× DROP), herring=4.20, sprat=0.36, flounder=9.65,
  perch=0.11, pikeperch=3.00, smelt=0.48, stickleback=0.14

Adult mortality (sp0..sp7):
  cod=3.72 (vs prior 0.265 — 14× UP, near new floor), herring=0.156, 
  sprat=0.022, flounder=0.022, perch=0.027, pikeperch=1.85, smelt=0.74, 
  stickleback=0.65

Fishing (fsh0..fsh7):
  cod=0.077, herring=0.238, sprat=0.0036, flounder=0.154,
  perch=0.018, pikeperch=0.010, smelt=0.0061, stickleback=0.031
```

**Recommended follow-up paths:**

1. **Density-dependent recruitment** — single biggest structural change. Without it, all constraint-tightening will be defeated by DE compensation. Beverton-Holt SR for cod, perch, pikeperch is the highest-leverage change. ~1 engineer-week.

2. **Joint larval+adult mortality bounds** — if SR can't be added, lock both larval and adult cod mortality to a tight range (e.g., sp0 larval ∈ [0.5, 2.0], sp0 adult ∈ [0.3, 0.8]). This removes the escape route. Test before committing.

3. **Multi-seed objective during DE** — the cod-floor run already showed std 0.19; even tighter multi-seed objective could push it lower.

4. **Accept the f=6.15 calibration as best stable baseline** — most robust calibration achieved. Document that cod/perch/pikeperch are persistently out of range due to engine-level lack of recruitment compensation.

**Session artifacts (master branch, 2026-04-27):**
- `data/baltic/calibration_results/phase12_results.json` — 24 optimized params (this run, cod-floor)
- `data/baltic/calibration_results/phase12_results.real_predators_no_cod_floor.json` — prior run (6.18 single-seed)
- `data/baltic/calibration_results/phase12_results.predators_inert_bug.json` — pre-fix run (3.55 multi-seed)
- `/tmp/osmose_calibration_phase12_cod_floor.log` — full 5h DE log
- `/tmp/launch_phase12_real_predators.sh` — wrapper (workers=8, env)
- Code change: `scripts/calibrate_baltic.py:312` — sp0 cod adult mortality bound `(-0.523, 0.7)` instead of `(-3.0, 0.7)`

**Master state at end (2026-04-27):**
- Cod-floor in calibrate_baltic.py committed (preserved for future runs)
- 1/8 strict in-range (stickleback), 1 close (herring ×1.19 over)
- Multi-seed std 5× tighter than prior — most robust calibration to date
- Density-dependent recruitment is the next milestone if Baltic-OSMOSE wants to clear ICES targets
