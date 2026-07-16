# Depensation gate + bistability placement (SP1) — Design

**Status:** design approved 2026-07-16; revised after in-loop review round 1. **Branch:** `feat/depensation-gate`.

## Goal

Give the Baltic OSMOSE model a **recruitment depensation / Allee** mechanism that can create
**bistability** (two alternative stable cod states — a healthy basin and a collapsed basin), and
**place** that bistability at a realistic operating point (healthy basin O(100kt) SSB, stable). This is
sub-project 1 of the regime-shift effort; SP2 (a committed follow-on, not in this spec) will drive the
resulting bistable model with the historical annual F to attempt to reproduce the observed eastern
Baltic cod collapse-and-no-recovery.

## Background / why

The deployed model is robustly **monostable**: its four stock-recruitment forms (`beverton_holt`,
`ricker`, `hockey_stick`, `shepherd`) are all *compensatory* — per-capita recruitment is maximal as
SSB→0, so a single attractor. The three-mechanism exploration (2026-07-16) concluded that **recruitment
depensation/Allee is the root lever** (fishing hysteresis and historical-state init are downstream
diagnostics that need bistability to exist first), and the de-risk spike
(`docs/diagnostics/2026-07-16-depensation-bistability-spike.md`, PR #117) proved a monkeypatched cod
Allee factor manufactures bistability: at cod-viable larval scales, the warm-start cod-rich IC
overshoots while the cod-poor IC collapses, at identical parameters, where the no-Allee control is
monostable. This spec turns that proof-of-mechanism into a config-plumbed feature and searches for a
*realistic* operating point.

## Scope decisions (locked during brainstorming)

- **Deliverable:** the gate feature **+** a validated bistable config overlay **+** a demonstrated
  hysteresis-loop-vs-control (Unit 3 is mandatory per Success Criterion #2, not optional polish).
  Reproducing the historical trajectory is SP2.
- **Search method:** deterministic **grid sweep** using the warm-start reciprocal-invasion classifier
  (not an optimizer — bistability is an emergent binary property).
- **Healthy-basin target:** **realistic magnitude** — O(100kt) SSB, stable (non-transient), with a
  distinct collapsed basin. NOT a strict ICES-band match (that is SP2's concern).
- **Functional form:** Hill / Liermann-Hilborn `A(SSB)=SSB^θ/(S50^θ+SSB^θ)` applied as a **post-hoc
  multiplicative gate** on egg production (the spike-validated form; composes with any base SR type).
- **Species:** the gate is built **per-species-configurable** (general), but only **cod (sp0)** is
  calibrated/validated in SP1.
- **Overlay, not default:** the bistable config is a separate overlay; the deployed default is
  untouched.

## Architecture — four units

```
1. Depensation GATE (engine feature)   osmose/engine/processes/depensation_gate.py  (+ config plumbing + Java guard + schema)
2. Placement HARNESS (analysis)        scripts/calibrate_depensation_bistability.py
3. VALIDATION (analysis)               warm-start basin split + fishing-hysteresis F-ramp (with control)
4. Config OVERLAY (deliverable)        data/baltic_depensation/  (DRY overlay, explicit new Java guard)
```

Unit 1 is unit-tested production code (a CI gate). Units 2–4 are emergent analysis + a deliverable
config, documented in a diagnostics doc — NOT CI gates (long-running, seed/core-sensitive, per the
fragile-emergent-tests rule).

Key architectural point: unlike the RV/thermal gates (which read `step` → an environmental field), the
depensation factor depends on the **current SSB** — state-dependent, not time-driven. So the gate is a
pure function of `(ssb, s50, theta)` computed inside `reproduction()` where SSB is already in hand.

---

## Unit 1 — Depensation gate (engine feature)

### Module: `osmose/engine/processes/depensation_gate.py`

One pure, engine-state-free function (mirrors `thermal_gate.py`'s pure-helpers style):

```python
def depensation_factor(
    ssb: NDArray[np.float64],
    s50: NDArray[np.float64],
    theta: NDArray[np.float64],
    enabled: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per-species Allee multiplier A(SSB)=SSB^θ/(S50^θ+SSB^θ), in (0, 1].

    1.0 where disabled. A→0 as SSB→0, A=0.5 at SSB==S50, A→1 as SSB→∞.
    ssb, s50, theta, enabled are all length n_sp.
    """
    out = np.ones(ssb.shape[0], dtype=np.float64)
    for sp in range(ssb.shape[0]):
        if not enabled[sp]:
            continue
        s = ssb[sp]
        if s <= 0.0:
            out[sp] = 0.0  # SSB=0 → full suppression; harmless (n_eggs already 0) + skipped-when-seeded
            continue
        out[sp] = s ** theta[sp] / (s50[sp] ** theta[sp] + s ** theta[sp])
    return out
```

### Config keys (namespace mirrors the RV/thermal gates: `reproduction.<name>.gate.<attr>`)

- `reproduction.depensation.gate.enabled` — global bool (default false)
- `reproduction.depensation.gate.species.enabled.sp{i}` — per-species bool (default false)
- `reproduction.depensation.gate.s50.sp{i}` — half-suppression SSB in tonnes (float > 0)
- `reproduction.depensation.gate.theta.sp{i}` — Allee exponent (float ≥ 1; **θ>1 gives a genuine
  sigmoidal trap; θ=1 is the weak-Allee boundary — SP1's grid uses θ∈{2,4}**)

### Config loader: `_load_depensation_gate(cfg, n_sp)` in `osmose/engine/config.py`

Mirrors `_load_thermal_gate`'s **structure** but is simpler (per-species scalars, no CSV/time-series,
no normalization mode). **Returns a 3-tuple `(enabled, s50, theta)` where each element is `None` when
the global flag is off/absent** — NOT a bare `None` (every sibling returns an N-tuple of Nones, e.g.
`_load_thermal_gate` returns `None, None, 0`; a bare `None` would crash the `dep = ...` unpack at the
call site). **Eager validation, fail-fast to match all four sibling loaders**
(`_load_rv_gate`, `_load_salinity_gate`, `_load_recruitment_ceiling`, `_load_thermal_gate` all raise
`ValueError` in these cases — do the same, do NOT return inert `None`):
- global flag on but **no species enabled** → `raise ValueError` (fail-fast, like the siblings).
- any enabled species with `theta[sp] < 1.0` → `raise ValueError` (θ<1 is not an Allee trap).
- any enabled species with `s50[sp] <= 0.0` → `raise ValueError`.

### EngineConfig plumbing — mirror `thermal_gate`, which BYPASSES `_merge_focal_background`

**Correction from review:** `thermal_gate_*` is NOT plumbed through the `_merge_focal_background`
blocks (those handle `shepherd_beta`/`recruitment_type`, which need per-species background defaults).
`thermal_gate` is loaded once, directly, *after* the merge, on focal-only `n_sp`, and passed straight
to the constructor. Follow that exactly:
- **Dataclass fields** (near `config.py:1684-1687`, beside `thermal_gate_*`):
  ```python
  depensation_gate_enabled: NDArray[np.bool_] | None
  depensation_s50: NDArray[np.float64] | None
  depensation_theta: NDArray[np.float64] | None
  ```
  (set together — all `None` when off).
- **Load call site** (near `config.py:2429-2431`, right beside the `_load_thermal_gate(...)` call,
  after `_merge_focal_background` has run, using focal-only `n_sp`):
  `dep = _load_depensation_gate(cfg, n_sp)` → unpack to three locals (or `None,None,None`).
- **Constructor kwargs** (near `config.py:2506-2508`, beside the `thermal_gate_*=` kwargs).
- Do **NOT** touch the `_merge_focal_background` blocks (config.py ~607/~825/~877/~2050/~2156/~2473)
  — those are `shepherd_beta`'s path, not this one. Because it's focal-only length `n_sp`, the wiring
  needs no `[:n_sp]` slice (unlike the merged `shepherd_beta`/`sex_ratio` arrays).

### Wiring in `osmose/engine/processes/reproduction.py`

A new guarded block **after** the thermal-gate block (ends line 190), before "Create new schools from
eggs" (line 192). **Include the `assert ... is not None` narrowing** — all three sibling gates do this
(reproduction.py:161,171,186); omitting it fails the required pyright CI leg with `reportArgumentType`
(verified):

```python
# Recruitment depensation / Allee gate (SSB-dependent, not step-dependent). Inert unless
# enabled; skipped on seeded steps so the SSB=0 bootstrap can't be trapped, like the other gates.
if config.depensation_gate_enabled is not None:
    from osmose.engine.processes.depensation_gate import depensation_factor

    assert config.depensation_s50 is not None  # invariant: set together in _load_depensation_gate
    assert config.depensation_theta is not None
    dfac = depensation_factor(
        ssb, config.depensation_s50, config.depensation_theta, config.depensation_gate_enabled
    )
    for sp in range(n_sp):
        if config.depensation_gate_enabled[sp] and not seeded_this_step[sp]:
            n_eggs[sp] *= dfac[sp]
```

(`ssb` is already length `n_sp` — no slice needed.)

### Schema registration: `osmose/schema/species.py`

Add `OsmoseField(key_pattern=...)` entries for the four keys, mirroring the `reproduction.thermal.gate.*`
entries (species.py ~594/~655/~666), so the UI Setup panel renders them and `validate_config` enforces
`theta≥1` / `s50>0` bounds pre-run. (Not CI-blocking — the strict-key AST walker auto-discovers the
`cfg.get("...sp{sp}", ...)` literals — but the schema is the complete pattern all four siblings follow.)

### Determinism

Default-off → `depensation_gate_enabled is None` → block skipped → **byte-identical** to current output
(the same mechanism `tests/test_reproduction_thermal_gate.py::test_gate_off_is_bit_identical_to_baseline`
verifies). Plain Python (no Numba). **Caveat:** the gate is wired only into `reproduction()`; the
bioenergetic path `_bioen_reproduction` (used by `data/baltic_ev`, `config.bioen_enabled=true`) does not
call it and would be silently inert there. SP1's target `data/baltic` has no bioen keys, so this is a
documented non-issue for SP1; a follow-up would be needed to extend depensation to the bioen path.

---

## Unit 2 — Placement harness: `scripts/calibrate_depensation_bistability.py`

Deterministic grid sweep using the **real config-plumbed gate** via overrides (not the spike's
monkeypatch), reusing `scripts/baltic_bistability_chunk0.py` helpers.

### Measure SSB (not biomass), over the FULL horizon (two review corrections)

- **SSB, not total biomass.** `run_simulation`'s `{sp}_mean` is mean *total biomass*, but the target
  "healthy basin ~cod Bpa (~120kt)" is an **SSB** reference point. The harness must enable
  `output.ssb.enabled=true` and read the **cod SSB** series (via `PythonEngine().run_in_memory(raw,
  seed).ssb()["cod"]`, as `scripts/spikes/ssb_f_hindcast_spike.py` does) — a consistent SSB-vs-SSB
  comparison.
- **Full-horizon stability, not `run_simulation`'s trailing-10-year window.** `run_simulation` hardcodes
  `n_eval_years=10` (calibrate_baltic.py:254) — a 50-yr run there evaluates only years 41–50, which
  merely *relocates* the window, it does not widen it. The harness must extract the **full annual SSB
  trajectory** itself (from `.ssb()`) and apply a stability discriminator strong enough for a
  near-bifurcation regime (see next).

### Stability discriminator (critical-slowing-down guard)

SP1's target sits *near* an Allee/fold bifurcation, where trajectories can flatten (low CV, low trend)
while still slowly sliding toward the other basin — a "ghost attractor" that a single trailing-window
CV+trend cannot distinguish from genuine stability (this is exactly the spike's transient-185kt
confound). **But note the cod-rich IC is seeded at 300kt while the target healthy basin is O(100kt), so
a legitimate GO candidate must DECLINE from 300kt toward equilibrium during warm-up** — a naive
"non-decreasing across decades" check would wrongly reject it. The discriminator must separate
*converging down to a plateau* (OK) from *making ever-new lows toward collapse* (fail). This is
genuinely hard in a short window because settling to within a few percent of equilibrium takes ~3τ ≈
60–75 yr (τ ≈ cod's 20–25-yr relaxation time), longer than a 50-yr screen. So the **long confirmatory
re-run is the ARBITER**, and the 50-yr pass is only a cheap coarse filter:
1. **Screen (50 yr, coarse, permissive):** shortlist a point if its healthy basin is in the GO
   magnitude band AND its decline is *decelerating* — the final-decade mean is within a tolerance
   **≤10%** below the prior decade's mean (a still-steeply-declining trajectory is culled; a
   converging-toward-plateau trajectory passes). Deliberately loose — it only drops the obvious slides.
2. **Arbiter (confirmatory 150–200-yr re-run):** at each shortlisted point, the healthy basin is judged
   **genuinely stable iff it persists above the collapse threshold for the full 150–200 yr** — a true
   attractor settles to a plateau; a slow ghost-attractor slide eventually collapses within 150–200 yr
   (many multiples of τ) and is rejected. This long horizon, not the 50-yr screen, resolves
   converging-down-to-plateau vs sliding-to-collapse.

Reuse `baltic_bistability_chunk0.py`'s `basins_differ`/`classify_state` for the rich-vs-poor split, but
**do not rely on `is_stationary`'s original thresholds alone** (`cv_max=0.30`, `trend_max=0.05` were
tuned for the monostable, non-bifurcating investigation) — add the per-decade-monotonicity + long-re-run
checks above.

### Grid

- `S50 ∈ {30_000, 60_000, 90_000, 120_000}` × `θ ∈ {2.0, 4.0}` × `larval-M scale ∈
  {0.6, 0.75, 0.85, 0.90, 0.95, 1.0}`. The scale grid is **densified near 0.85–1.0** because the spike
  showed healthy-basin magnitude swings ~20× between scale 0.7 (overshoot) and 1.0 (transient) — the
  O(100kt) target plausibly lives at scale ≈0.90–0.97, between the original nodes. **Note:** the spike
  only tested scales {0.3,0.5,0.7,1.0}; 0.6/0.75/0.85/0.9/0.95 are extrapolation, so a structurally
  empty result must be checked for under-resolution (see the ambiguous outcome below) before being
  called a negative. If a promising-but-between-nodes point appears, do **one documented refinement
  pass** (finer scale/S50 around it) — still a grid, not an optimizer.
- **Overrides per point**: `reproduction.depensation.gate.enabled=true`,
  `reproduction.depensation.gate.species.enabled.sp0=true`,
  `reproduction.depensation.gate.s50.sp0=<S50>`, `reproduction.depensation.gate.theta.sp0=<θ>`,
  `output.ssb.enabled=true`, plus `warmstart_override(True)` + `cod_rich_seeding()`/`cod_poor_seeding()`
  + `larva_scale_override(scale, base_rates)`.
- **Per point**: run cod-rich and cod-poor warm-start ICs over a **50-yr horizon, 3–5 seeds**.

### Operating-point selection (three-way outcome)

Per point classify: `{bistable?, healthy_ssb_mean, healthy_stable?, collapsed_ssb_mean, det_frac}`.
- **GO** — ≥1 point is bistable (rich vs poor differ, `basins_differ` gap ≥ 0.5) AND healthy basin
  **SSB ∈ [40_000, 300_000] t** (the concrete "O(100kt)" gate — ~⅓Bpa to ~2.5×Bpa, deliberately wider
  than the canonical cod SSB band in `data/baltic/reference/biomass_targets.csv` [lower 60k … upper
  250k], per the realistic-magnitude scope; comfortably brackets Bpa while excluding the spike's
  collapse and overshoot basins — those spike figures were total *biomass*, cited only as an
  order-of-magnitude sanity check) AND healthy_stable (coarse 50-yr screen + the
  150–200-yr arbiter re-run) AND collapsed basin distinctly lower (same `gap_thresh=0.5`). **Selection when multiple
  qualify:** healthy basin closest to Bpa (~120kt), stable, lowest CV tie-break.
- **Instrument-limited / AMBIGUOUS** — if the determinate fraction is low (`det_frac < 0.5`: many
  seed-splits/undetermined points) OR a candidate falls between grid nodes, report **ambiguous /
  under-resolved**, NOT a structural negative. Preserve this branch in the machine-readable output — do
  not collapse it into GO or NO-GO. **Note:** only the *concept* of `det_frac` carries over from
  `baltic_bistability_chunk0.py` — its concrete verdict functions are tied to `run_simulation`'s
  `cod_mean`/`cod_cv`/`cod_trend` stats dict, which Unit 2 deliberately does NOT reuse (it needs the
  full-horizon SSB trajectory via `.ssb()`). The harness defines its own "determinate outcome"
  bookkeeping (per-seed: clean split / seed-split / undetermined) against the new per-decade SSB data.
- **NO-GO** — the grid is determinate (`det_frac ≥ 0.5`) and no point satisfies the GO criteria →
  documented negative (ships the gate feature only; see Success criteria).

Report `healthy_ssb_mean` **only when the aggregate state is determinate** — do not surface
`_median_valid`'s `0.0` fallback (returned when no seed is stationary) as a "collapsed" measurement.

### Compute budget (explicit)

6 scales × 4 S50 × 2 θ = 48 grid points × 2 ICs × 3–5 seeds × 50 yr ≈ **288–480 multi-decade
Python-engine runs**, plus confirmatory 80–100-yr re-runs at candidates — far heavier than the 15-yr
spike. This must run with the engine's parallel-run path and an explicit runtime budget; **do NOT
silently trim seeds or years to fit** (that reintroduces the exact transient-vs-stable confound the
long horizon is there to prevent). If the full grid is infeasible, cut grid *breadth* (fewer S50/θ),
never horizon or seeds.

## Unit 3 — Validation (analysis → diagnostics doc)

At the chosen operating point:
1. **Warm-start basin split** — rerun cod-rich vs cod-poor with extra seeds + the long horizon;
   confirm a robust, non-transient split (healthy stable per the Unit-2 discriminator, collapsed stays
   low).
2. **Fishing-hysteresis F-ramp** — from a healthy warm-start, sweep cod F via the validated `byyear`-F
   tooling (`mortality.fishing.rate.byyear.file.sp0`, per `scripts/spikes/ssb_f_hindcast_spike.py`):
   - **Quasi-static, stepped ramp**: **~8 F levels** spanning F_low→F_high (e.g. 0.5×→~8× base). **Hold
     each level until SSB EQUILIBRATES, not for a fixed dwell** — a per-level convergence check (SSB
     slope over the last ~2 decades ≈ 0) with a generous cap of **≥3τ (~75 yr)**. Fixed dwell is
     wrong here: relaxation time **inflates near the fold points** (F_collapse/F_recover) via critical
     slowing down — the very levels the ramp must cross — so ~30–40 yr (only ~1.3–2τ off-fold) is too
     short exactly where it matters. Go up then symmetrically back down.
   - **Critical-slowing-down caveat:** the no-depensation control is monostable and cannot reproduce
     fold-adjacent slowing, so "loop in depensation, none in control" is **necessary but not
     sufficient** — additionally require that every depensation-arm level actually passed its per-level
     convergence check, so the loop reflects genuine alternative equilibria rather than unconverged lag
     near the fold. If a level hits the cap without converging, flag it (do not silently treat the
     capped state as an equilibrium).
   - **Compute budget (explicit, like Unit 2):** ~8 levels × up-to-~75-yr equilibration × 2 directions
     ≈ up to 1,200 simulated yr per seed-arm × 3–5 seeds × 2 arms (depensation + control) ≈
     **~7,000–12,000 simulated years**. Same rule as Unit 2: never trim the equilibration cap to fit
     (that reintroduces the lag artifact this test exists to rule out); cut level count first.
   - **3–5 seeds** (a single realization near a saddle is weak evidence).
   - **No-depensation CONTROL ramp**: run the identical stepped F-ramp on the gate-off config;
     expect **no** comparable loop. The loop counts as hysteresis only if it appears with depensation
     and NOT in the control (rules out the relaxation-lag artifact both prior spikes flagged).
   - Plot cod SSB **parametrically vs F**; confirm the depensation legs form a loop (collapse at
     F_collapse up-leg; recovery only at F_recover < F_collapse down-leg) while the control legs overlap.

## Unit 4 — Config overlay: `data/baltic_depensation/`

DRY overlay on `data/baltic` (like the `baltic_a2` preset): only the changed keys — gate enabled for
cod + chosen S50/θ + adjusted larval-M (the operating point). Registered as a loadable demo/preset like
`baltic_a2`. Two review-mandated requirements:
- **`mortality.additional.larva.rate.sp{i}` MUST be stored as `engine_value × ndtperyear` (×24).**
  The reader divides by ndt on load (`osmose/config/aliases.py::_migrate_larva_rate`, v≥4.4.0) — the
  named `feedback-larval-rate-ndt-migration-gotcha`. `baltic_a2`'s test encodes `STORED = CONVERGED ×
  24`; carry that forward or the overlay silently applies a 1/24 larval mortality.
- **Java guard needs an explicit NEW check** — the existing `nbackground>0` guard does NOT block this
  overlay (Baltic bg species have staging + jar 4.4.1, so `java_engine_block_reason` returns `None`;
  verified via `tests/test_baltic_a2_demo.py`). Add a check to `osmose/runner.py::java_engine_block_reason`
  that returns a Python-only reason when `reproduction.depensation.gate.enabled=true` (the gate is a
  Python-engine feature the Java engine ignores → would silently run without depensation). Otherwise the
  Unit-1 "Java-guard rejects the overlay" test cannot pass.

## Testing

**Unit 1 (CI gate)** — new `tests/test_depensation_gate.py` (+ config-parse cases), mirroring
`tests/test_reproduction_thermal_gate.py`, `tests/test_recruitment_ceiling.py`, and
`tests/test_engine_stock_recruitment.py`:
- `depensation_factor` math: `A(S50)=0.5`; `A→0` as SSB→0 (`=0` at SSB=0); `A→1` at large SSB; `=1.0`
  where disabled; **θ=1 boundary case**; **≥2 species enabled simultaneously** (isolation + correct
  per-species values).
- **Off → byte-identical**: a short Baltic run with the gate off == the current baseline
  (`np.testing.assert_array_equal`), the determinism guarantee.
- **Seeded-step skip**: integration-level (the skip lives in the wiring, not the pure fn) — copy
  `tests/test_recruitment_ceiling.py::test_reproduction_ceiling_skips_seeded_step` (empty `SchoolState`
  → seeded SSB → assert not gated).
- Config parse: keys → EngineConfig fields; **fail-fast** on `θ<1`, `s50≤0`, and global-on/no-species
  (all `ValueError`, matching the siblings).
- Java guard: `java_engine_block_reason` returns a reason for a depensation-enabled config (mirror
  `tests/test_baltic_a2_demo.py::test_a2_blocks_java_engine`).
- Overlay: loads + runs on the Python engine; **passes strict validation** (mirror
  `test_baltic_a2_demo.py::test_a2_passes_strict_validation`).
- **Fixture note:** adding 3 new required `EngineConfig` fields breaks
  `test_engine_config_validation.py::_minimal_config`, which constructs `EngineConfig(**cfg)` directly —
  update that fixture (self-revealing via test failure). (Tests that go through `EngineConfig.from_dict`
  / `osmose_demo()` auto-populate the new fields via the loader and are unaffected.)
- Integration "gate-on changes cod recruitment": **mark `@pytest.mark.skipif(CI)`** — real-engine
  Baltic rel-change is non-reproducible across runner core counts (`feedback-ci-fragile-emergent-tests`;
  both the thermal and RV gate on-effect tests carry this marker).

**Units 2–3** — emergent analysis, **NOT CI gates** (long-running, seed/core-sensitive). Deliverable
is the diagnostics doc (mapped bistable region + chosen operating point + hysteresis loop vs control).
A light 1-point smoke test may live behind a skip-CI marker.

## Success criteria

SP1 succeeds when **both**:
1. The gate feature is shipped — config-plumbed, default-off byte-identical, Java-guarded, schema-
   registered, unit tests green.
2. A documented operating point exists that is **bistable + healthy-O(100kt)-SSB + stable** (per the
   critical-slowing-down discriminator), delivered as the `data/baltic_depensation` overlay, with the
   warm-start split and the hysteresis-loop-vs-control demonstrated in a diagnostics doc.

### Three-way outcome / honest-negative fallback

- **GO** — as above.
- **AMBIGUOUS / instrument-limited** — `det_frac<0.5` or a between-nodes candidate → ship the gate
  feature + a documented *ambiguous* result and a proposed refinement, NOT a structural claim.
- **NO-GO** — determinate grid, no qualifying point → ship the **gate feature** (valuable, tested,
  config-plumbed) + a documented negative ("a bistable region exists but the healthy basin cannot be
  placed at realistic-and-stable magnitude"), which reframes SP2.

We do not report a null (or an under-powered/ambiguous sweep) as a success.

## Out of scope (SP2 and beyond)

- Driving the bistable overlay with historical annual F to reproduce the observed cod
  collapse-and-no-recovery (SP2 — committed follow-on, its own spec).
- Strict ICES-band match / full multi-species biomass-band recalibration of the healthy basin.
- Depensation for species other than cod (the gate is general; only cod is calibrated here).
- Extending depensation to the bioenergetic reproduction path (`_bioen_reproduction`).
- Any change to the deployed Baltic default config.

## Key references

- Spike (GO): `docs/diagnostics/2026-07-16-depensation-bistability-spike.md`, script
  `scripts/spikes/depensation_bistability_spike.py` (PR #117).
- Warm-start harness: `scripts/baltic_bistability_chunk0.py` (classifier + seeding/warmstart helpers,
  incl. the `det_frac`/instrument-limited verdict branch).
- byyear-F tooling: `scripts/spikes/ssb_f_hindcast_spike.py`, `mortality.fishing.rate.byyear.file.sp{i}`.
- Gate pattern to mirror: `osmose/engine/processes/thermal_gate.py`,
  `osmose/engine/config.py::_load_thermal_gate` (+ its direct-load call site ~2429 and fields ~1684),
  `osmose/schema/species.py` (thermal gate `OsmoseField`s), `tests/test_reproduction_thermal_gate.py`,
  `tests/test_recruitment_ceiling.py` (seeded-step test).
- Overlay + Java guard: `data/baltic_a2`, `tests/test_baltic_a2_demo.py`,
  `osmose/runner.py::java_engine_block_reason`; `feedback-larval-rate-ndt-migration-gotcha`.
- SR wiring: `osmose/engine/processes/reproduction.py:15-190`; eval-window `calibrate_baltic.py:254`.
- Science: Casini et al. 2009 (PNAS 10.1073/pnas.0906620106); Köster & Möllmann 2000
  (10.1006/jmsc.1999.0528); Möllmann tipping-points (10.1111/nrm.12336) — cultivation-depensation /
  predator-pit basis for Baltic-cod depensation.
