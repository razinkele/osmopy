# Depensation placement sweep (SP1 Task 8) — NO-GO: bistability confirmed, but the healthy basin can't be placed realistically-and-stably

**Date:** 2026-07-17 · **Status:** NO-GO (honest negative) · **Kind:** placement analysis run (Task 8 of SP1)
**Harness:** `scripts/calibrate_depensation_bistability.py` (merged PR #118); staged drivers run out-of-tree.

## Question

SP1 shipped the depensation gate (PR #118). Task 8 runs the placement sweep: does a `(larval-M scale,
S50, θ)` operating point exist where the model is **bistable** with a **healthy cod SSB basin that is
O(100kt) AND stable** (the spec GO band `[40k, 300k] t`)? That is the empirical open question SP1 exists
to answer; the spec flagged its central risk as a possibly "narrow-or-empty window."

## What was run (θ=4, the sharpest trap)

The full 48-point grid projected to ~10–12h (each warm-start point ~26 min), so — per the spec's
guidance to cut breadth (never seeds/years) — a staged scan at 3 seeds + 50yr screen + 175yr arbiter:
1. **Scale-scan** at S50=60k: scale ∈ {0.8, 0.85, 0.9, 0.95, 1.0}.
2. **S50-refine** at scale 0.9: S50 ∈ {150k, 250k, 350k, 450k} (+ the 60k baseline).
3. **Fine S50** at scale 0.9: S50 ∈ {275k, 300k, 325k} — probing the fold gap.

## Results

**The gate works: every point shows a clean bistable split** — the cod-rich IC settles high and the
cod-poor IC collapses to ~0 (gap 1.0) at identical parameters. The merged depensation gate reliably
manufactures the alternative-stable-state structure. The problem is the *healthy* basin's magnitude.

**Scale-scan (S50=60k, θ=4) — healthy basin (rich):**
| scale | healthy | regime |
|---|---|---|
| 0.80 | 741k | overshoot |
| 0.85 | 1.19M | overshoot (non-monotonic — larval-M is a community-wide lever) |
| 0.90 | 450k | overshoot (stable, nearest band) |
| 0.95 | 49k | **in-band magnitude but transient (declining) → culled** |
| 1.00 | 0 | collapsed (deployed larval-M collapses cod) |

**S50-refine (scale 0.9, θ=4) — raising S50 pulls the healthy basin down, then FOLDS to collapse:**
| S50 | healthy |
|---|---|
| 60k | 450k |
| 150k | 439k |
| 250k | 333k (overshoot, just above band) |
| 350k | **215 t (collapsed)** |
| 450k | ~0 |

**Fine S50 (scale 0.9, θ=4) — the band is a transient fold-crossing, not a stable branch:**
| S50 | healthy | stable? |
|---|---|---|
| 275k | 175k (IN-BAND magnitude) | **NO — declining/transient (screen-culled)** |
| 300k | 24k (below band) | NO — essentially collapsing |
| 325k | 2,029 t | NO — collapsed |

Consolidated healthy-basin curve at scale 0.9, θ=4 (SSB vs S50):
`60k→450k · 150k→439k · 250k→333k · 275k→175k · 300k→24k · 325k→2.0k · 350k→215t · 450k→~0`.

## Conclusion — NO-GO

**No `(scale, S50)` at θ=4 places the healthy basin at a realistic AND stable magnitude.** The *stable*
healthy branch **skips the `[40k, 300k]` target band**: it is overshoot (≥333k) or collapsed, and where
the magnitude *is* in-band (S50 ≈ 275k → 175k; scale 0.95 → 49k), it is a **transient slide toward
collapse**, not a stable equilibrium — exactly the critical-slowing-down/fold-crossing the arbiter is
built to reject. The spec's flagged "narrow-or-empty window" is **empty** (at θ=4). The healthy basin
plummets from 333k (S50=250k) through the whole band to ~2k (S50=325k) over a 75k S50 step — an abrupt
fold, no plateau.

Mechanistically: the healthy equilibrium sits on the upper stable branch of the Allee fold. A weak Allee
(S50 ≪ healthy SSB) leaves that branch at the no-Allee carrying capacity (overshoot); strengthening it
(raising S50) lowers the branch until, at a critical S50, it meets the unstable middle branch and
vanishes (collapse). The branch passes *through* `[40k,300k]` only in the immediate fold neighborhood,
where it is fragile/transient — there is no wide, stable in-band plateau to sit a realistic cod stock on.

### Caveats (so the negative isn't oversold)
- **Only θ=4 (sharpest trap) + scale 0.9 refined.** A gentler θ=2 bites *even less* at the healthy SSB,
  so it needs an even higher S50 to pull the basin down and hits the same fold — no reason to expect a
  qualitatively wider, stable in-band plateau. Other scales have a higher no-Allee healthy basin,
  requiring more S50 and the same fold. The fold is generic, not a grid artifact.
- **This is a PLACEMENT negative, not a MECHANISM negative.** The depensation gate genuinely creates
  bistability (robustly, at every point) — the finding is that its *healthy* attractor cannot be tuned to
  a realistic-and-stable cod SSB in this model. The gate ships and remains available (PR #118).

## Implication for SP2

SP2 (historical-F hindcast) needs a placed overlay whose healthy basin is a realistic cod SSB that
collapses to a realistic remnant under historical F. Since a realistic-and-stable healthy basin does not
exist, **SP2 as scoped is BLOCKED** — the model's cod either overshoots (unrealistically high) or
collapses; it cannot sit near ~Bpa and be driven across a fold at historically-reachable F. Reaching an
SSB-trajectory capability would need a different framing (relative regime dynamics from the overshoot
branch, or a different structural mechanism — cf. the cultivation-depensation predator-pit that Chunk C
tested negative), not a straightforward historical-F hindcast on this overlay.

The `data/baltic_depensation` overlay keeps its placeholder operating point: the gate is a valid, tested
feature, but the overlay is **not** a calibrated bistable-at-realistic-SSB config, because none exists.
Related: `docs/diagnostics/2026-07-16-depensation-bistability-spike.md`,
`docs/diagnostics/2026-07-15-ssb-f-hindcast-spike.md`;
spec `docs/superpowers/specs/2026-07-16-depensation-gate-bistability-design.md`.
