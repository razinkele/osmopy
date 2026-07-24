# SP-A — Baltic stability recalibration (parameters-only, current grid) — Design

**Date:** 2026-07-01
**Status:** design, pending approval
**Parent project:** Recalibrate Baltic for long-term dynamic stability. This is **SP-A**, the first of
three sub-projects (SP-A params-only on current grid → decision gate → SP-B grid re-resolution →
SP-C full recalibration on new grid). SP-A is the decisive, lowest-risk piece and gates SP-B/C.

## Goal

Recalibrate the **free parameters** of `data/baltic` (no grid change) so the ecosystem reaches a
**bounded equilibrium** — all 8 focal species persist *within* their ICES envelopes over a 50-yr+
horizon — instead of collapsing to a herring+sprat 2-species state by ~yr30. The decisive
deliverable is either (a) a stable, in-ICES `data/baltic`, or (b) a Pareto-front + named list of
which species cannot be stabilised by parameters alone — the evidence that gates SP-B.

## Background

- **Root cause (refined after review):** the current `calibrate_baltic.py` objective is envelope-aware
  on biomass **and already carries CV/trend "stability" penalties** (`w_stability·(cv−0.2)² +
  w_stability·(trend−0.05)²`, default `w_stability=5.0`) — but (a) it has **no persistence/extinction
  term** (a species crashing to ~0 isn't directly penalised), and (b) the calibration eval horizon is
  too short to *see* the late collapse, and the CV/trend window is the last 10 yr of that short run. So
  the existing stability terms never bite on the slow drift, and the config collapses. SP-A's new value
  is therefore the **persistence term + a long-horizon, commensurately-weighted** stability objective —
  not "adding stability from scratch". To avoid double-counting, the legacy `w_stability` CV/trend
  terms are **zeroed when `λ>0`** and the new commensurate term owns stability.
- Empirically: on the Python engine (seeds 0/42/7), Baltic holds ~10 yr then collapses to 2/8 by
  ~yr30; Java collapses 3/8. Seed-independent; predator-collapse → forage-release cascade.
- The **percid structural limit** (perch/pikeperch, weight 0.2) is a coarse-grid boom/bust within
  27–62 of 616 ocean cells — the expected residual that SP-A's gate will quantify.

## Approach: ① stability objective + Pareto recalibration, with a diagnostic front-end

### Phase 0 — Diagnostic front-end (cheap; ~hours)
Pin down the collapse driver and confirm the free-parameter set *before* spending calibration compute.
- Run the current `data/baltic` 50 yr × 3 seeds; record per-species annual biomass **and** the
  per-species mortality decomposition over time (predation / starvation / additional / fishing) from
  the `mortalityRate-{sp}` frames `OsmoseResults` builds (available on the in-memory path via
  `_build_mortality_dataframes`, verified to exist).
- Identify the keystone: which species declines **first**, and which mortality term dominates its
  decline (e.g. cod lost to juvenile predation vs starvation vs recruitment shortfall).
- **Output:** a short diagnostic note + the confirmed free-parameter set. The current
  `configure.py` patterns (`mortality.additional.rate`, `…additional.larva.rate`,
  `…starvation.rate.max`, `predation.ingestion.rate.max`) are the baseline; **add the recruitment
  parameters** — `stock.recruitment.ssbhalf.sp{i}` (Beverton-Holt half-saturation SSB =
  density-dependence strength, all 8 use B-H), `stock.recruitment.shape.sp{i}` (Shepherd β
  over-compensation, percid-type species only), and/or `species.relativefecundity.sp{i}`
  (recruitment magnitude) — if Phase 0 confirms recruitment over/under-compensation is a driver. The
  β-probe in the percid diagnostic already showed the Shepherd exponent is a stability lever.

### Components

1. **`StabilityObjective`** (new, picklable, `osmose/calibration/objectives.py`).
   Module-level class (crosses the ProcessPool boundary, mirroring `BiomassRMSEObjective`). From the
   simulated WIDE biomass trajectory over the eval window `[t_warmup, t_end]`, per focal species *s*:
   - **persistence penalty** — **smooth** log10-distance of the window-minimum *below* the floor
     `φ · lower_ices_s`: 0 if `min ≥ φ·lower`, else `(log10(φ·lower / min))²`. Smooth (not a flat step)
     so it is **commensurate with the ICES `log10²` error** and trades off continuously instead of
     swamping it. `φ` default 0.1.
   - **envelope penalty** — fraction of the window the species spends outside `[lower_ices_s,
     upper_ices_s]` (over- or under-shoot), measured on the time-mean and the **final-decade mean**
     (the last ~10 yr of whatever horizon is run — *relative*, so it is well-defined for both the
     35-yr proxy and the 50-yr certification; never an absolute year).
   - **trend penalty** — `|slope|` of `log10(biomass_s)`, taken as the **max of the full-window slope
     and the late-window (final ~third) slope**, so a config that holds flat then tips in the last
     years is not averaged out into a near-zero slope. The discriminating signal for incipient collapse.
   - **variability penalty** — CV over the window, but **down-weighted/clipped for documented
     boom-bust species** (stickleback) so natural oscillation isn't punished as instability.
   Per-species penalties are combined with the **existing ICES species weights** (cod/herring/sprat
   1.0 … percids 0.2) into one scalar (0 = perfectly bounded-stable). The persistence + trend terms
   dominate; envelope/variability shape the basin.

2. **Free-parameter set** — `configure.py` baseline patterns + (conditionally) recruitment params
   (`stock.recruitment.ssbhalf` / `…shape` / `species.relativefecundity`), per Phase 0.

3. **Pareto via ε-constraint over surrogate-DE.** `surrogate_assisted_de` optimises a **single scalar**.
   Trace the front by the **ε-constraint** method — minimise `ICES_loss` subject to `Stability ≤ ε`,
   sweeping ε loose→tight (implemented as a hinge `ICES_loss + Λ·max(0, Stability − ε)` with a large
   fixed Λ inside the one scalar DE minimises). **ε-constraint is preferred over a plain weighted sum**
   `ICES + λ·Stability`: weighted-sum provably recovers only the *convex hull* of the front and would
   miss non-convex points — likely here, since persistence is a near-threshold effect. The **ICES term
   reuses the existing envelope-aware Baltic objective** (zero inside `[lower, upper]`, `log10²` outside
   — already in `calibrate_baltic.py`; *not* the point-only `objectives.biomass_rmse`). Per-candidate
   results record `ICES_loss` and `Stability` **separately** (not just the summed score) so the sweep
   can read the true front. Selected config = the smallest-ICES front point whose certification keeps
   all 8 species persistent and in-envelope.

4. **Eval protocol (2-tier).**
   - *In-loop (cheap):* 35-yr eval × 3 seeds. The **trend** penalty is what makes the shorter proxy
     valid — it scores incipient decline (downward log-slope) *before* extinction, so a candidate that
     would collapse at yr45 already reads as drifting at yr35. Per-objective seed aggregation via
     `validate_multiseed`: **worst-seed** for the StabilityObjective (robustness — stable only if it
     survives every seed), **mean** for the ICES term (one unlucky seed shouldn't dominate the match).
     Python `run_in_memory`.
   - *Compute (honest):* ~4–5 min/candidate (35 yr × 3 seeds); a surrogate sweep is a few hundred real
     evals per λ × ~5 λ ≈ **1–3 days locally**, *not* a single night. Reducible toward overnight
     (30 yr / 2 seeds / tighter eval budget / fewer λ) or run the full sweep on the HPC Apptainer
     container.
   - *Final certification (full):* re-run each selected front point at **50 yr+ × ≥5 seeds on both
     engines** (Python + Java) before writing anything to `data/baltic`.

### Data flow

```
candidate ─▶ overrides ─▶ run_in_memory(35yr, seed∈{42,123,7}) ─▶ biomass + mortality traj
   ─▶ ICES_loss (mean over seeds) + λ·Stability (worst seed) ─▶ surrogate-DE   [one solve per λ]
λ-sweep ─▶ Pareto front ─▶ best in-envelope point ─▶ cert: 50yr × 5 seeds × {Python, Java}
   ─▶ all 8 persist & in-envelope & bounded ─▶ write data/baltic (parity-gated commit)
   ─▶ else ─▶ SP-B GATE: named failing species + structural-vs-tunable evidence
```

### Success metric (SP-A "done")

A Pareto-front config where, at **50 yr × 5 seeds (Python)**, every focal species (a) **persists**
(`min biomass > φ · lower_ices`), and (b) the **late-window (yr40–50) mean** biomass falls **within**
its ICES envelope (single-year endpoints are too noisy), with bounded variability (CV below
threshold, except boom-bust stickleback). If `<8/8`,
SP-A still *succeeds as a gate*: it outputs the named failing species + evidence (does cranking their
params move them, or are they grid-pinned?) → the decision input for SP-B.

### Validation / testing

- **Unit** (`StabilityObjective`): synthetic trajectories — flat-in-envelope → ~0; monotonic decline
  → high trend; boom-bust within envelope → low (not punished); extinction → huge; explosion → high.
  Plus a **picklability** test (round-trips a ProcessPool).
- **Integration:** a tiny-budget calibration smoke (one λ, tiny eval budget, 5-yr eval) runs the
  scalarized surrogate-DE end-to-end and returns a front point without error.
- **Regression:** existing calibration tests stay green; the new objective doesn't alter existing
  objective outputs.
- **Certification:** the winner re-run at 50 yr multiseed on Python **and** a Java cross-check, with
  the per-species persist/in-ICES table recorded in the SP-A outcome note.

### Risks / out of scope

- **Risk (expected, not a failure):** parameters-only may not reach 8/8 — the percids are the known
  structural suspect. That outcome *is* the SP-A gate deliverable for SP-B, not a defeat.
- **Compute:** ~1–3 days locally via the λ-swept surrogate-DE (reducible: 30 yr / 2 seeds / fewer λ);
  HPC Apptainer container is the escalation path.
- **Out of scope:** grid/spatial-resolution changes (SP-B), `baltic_ev`, fishing-policy changes, and
  any change that breaks the bundled-config round-trip parity (the final write is parity-gated).
- **Engine:** calibrate on Python (fast, deterministic per seed); stability should transfer to Java
  (both collapse the same direction) and the certification confirms it.
