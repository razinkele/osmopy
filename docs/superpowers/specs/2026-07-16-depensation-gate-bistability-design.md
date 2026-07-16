# Depensation gate + bistability placement (SP1) — Design

**Status:** design approved 2026-07-16. **Branch:** `feat/depensation-gate`.

## Goal

Give the Baltic OSMOSE model a **recruitment depensation / Allee** mechanism that can create
**bistability** (two alternative stable cod states — a healthy basin and a collapsed basin), and
**place** that bistability at a realistic operating point (healthy basin O(100kt), stable). This is
sub-project 1 of the regime-shift effort; SP2 (a committed follow-on, not in this spec) will drive the
resulting bistable model with the historical annual F to attempt to reproduce the observed eastern
Baltic cod collapse-and-no-recovery.

## Background / why

The deployed model is robustly **monostable**: its four stock-recruitment forms (`beverton_holt`,
`ricker`, `hockey_stick`, `shepherd`) are all *compensatory* — per-capita recruitment is maximal as
SSB→0, so a single attractor. The warm-start / Chunk-C / Chunk-A2 investigations never found a second
basin. The three-mechanism exploration (2026-07-16) concluded that **recruitment depensation/Allee is
the root lever** (fishing hysteresis and historical-state init are downstream diagnostics that need
bistability to exist first), and the de-risk spike
(`docs/diagnostics/2026-07-16-depensation-bistability-spike.md`, PR #117) proved a monkeypatched cod
Allee factor manufactures bistability: at cod-viable larval scales, the warm-start cod-rich IC
overshoots while the cod-poor IC collapses, at identical parameters, where the no-Allee control is
monostable. This spec turns that proof-of-mechanism into a config-plumbed feature and searches for a
*realistic* operating point.

## Scope decisions (locked during brainstorming)

- **Deliverable:** the gate feature **+** a validated bistable config overlay (SP1). Reproducing the
  historical trajectory is SP2.
- **Search method:** deterministic **grid sweep** using the warm-start reciprocal-invasion classifier
  (not an optimizer — bistability is an emergent binary property).
- **Healthy-basin target:** **realistic magnitude** — O(100kt), stable (non-transient over a long
  horizon), with a distinct collapsed basin. NOT a strict ICES-band match (that is SP2's concern).
- **Functional form:** Hill / Liermann-Hilborn `A(SSB)=SSB^θ/(S50^θ+SSB^θ)` applied as a **post-hoc
  multiplicative gate** on egg production (the spike-validated form; composes with any base SR type).
- **Species:** the gate is built **per-species-configurable** (general), but only **cod (sp0)** is
  calibrated/validated in SP1.
- **Overlay, not default:** the bistable config is a separate overlay; the deployed default is
  untouched.

## Architecture — four units

```
1. Depensation GATE (engine feature)   osmose/engine/processes/depensation_gate.py
2. Placement HARNESS (analysis)        scripts/calibrate_depensation_bistability.py
3. VALIDATION (analysis)               warm-start basin split + fishing-hysteresis F-ramp
4. Config OVERLAY (deliverable)        data/baltic_depensation/  (DRY overlay, Java-guarded)
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
    enabled: list[bool],
) -> NDArray[np.float64]:
    """Per-species Allee multiplier A(SSB)=SSB^θ/(S50^θ+SSB^θ), in (0, 1].

    1.0 where disabled. A→0 as SSB→0, A=0.5 at SSB==S50, A→1 as SSB→∞.
    ssb, s50, theta are all length n_sp; enabled is length n_sp.
    """
    out = np.ones(ssb.shape[0], dtype=np.float64)
    for sp in range(ssb.shape[0]):
        if not enabled[sp]:
            continue
        s = ssb[sp]
        if s <= 0.0:
            out[sp] = 0.0  # SSB=0 → full suppression (the trap); guarded by seeded-step skip upstream
            continue
        out[sp] = s ** theta[sp] / (s50[sp] ** theta[sp] + s ** theta[sp])
    return out
```

### Config keys (namespace mirrors the RV/thermal gates)

- `reproduction.depensation.gate.enabled` — global bool (default false)
- `reproduction.depensation.gate.species.enabled.sp{i}` — per-species bool (default false)
- `reproduction.depensation.gate.s50.sp{i}` — half-suppression SSB in tonnes (float > 0)
- `reproduction.depensation.gate.theta.sp{i}` — Allee exponent (float ≥ 1)

### Config loader: `_load_depensation_gate(cfg, n_sp)` in `osmose/engine/config.py`

Mirrors `_load_thermal_gate` but simpler (per-species scalars, no CSV/time-series, no normalization
mode). Returns `(enabled: list[bool], s50: NDArray, theta: NDArray)` or `None` when the global flag is
off/absent. **Eager validation** (like the thermal gate's mode/floor checks):
- `theta[sp] >= 1.0` for every enabled species (θ<1 is not a real Allee trap — reject with a clear
  message).
- `s50[sp] > 0.0` for every enabled species.
- If the global flag is on but no species is enabled → return `None` (inert), do not error.

### EngineConfig fields (dataclass, `config.py` ~1600 block)

```python
depensation_gate_enabled: list[bool] | None
depensation_s50: NDArray[np.float64] | None
depensation_theta: NDArray[np.float64] | None
```

Set together (all `None` when off). Plumbed through the same focal/background merge blocks as
`thermal_gate_enabled` / `shepherd_beta` (the merge points near config.py ~605, ~821, ~875, ~2048,
~2154, ~2471 — follow the existing `thermal_gate_*` pattern exactly).

### Wiring in `osmose/engine/processes/reproduction.py`

A new guarded block **after** the thermal-gate block (currently ends ~line 190), before "Create new
schools from eggs":

```python
# Recruitment depensation / Allee gate (SSB-dependent, not step-dependent). Inert unless
# enabled; skipped on seeded steps so the SSB=0 bootstrap can't be trapped, like the other gates.
if config.depensation_gate_enabled is not None:
    from osmose.engine.processes.depensation_gate import depensation_factor

    dfac = depensation_factor(
        ssb[:n_sp], config.depensation_s50, config.depensation_theta, config.depensation_gate_enabled
    )
    for sp in range(n_sp):
        if config.depensation_gate_enabled[sp] and not seeded_this_step[sp]:
            n_eggs[sp] *= dfac[sp]
```

### Determinism

Default-off → `depensation_gate_enabled is None` → block skipped → **byte-identical** to current output.
Guard identical in spirit to the RV/thermal gates. Plain Python (no Numba); reproduction is not on a
compiled path.

---

## Unit 2 — Placement harness: `scripts/calibrate_depensation_bistability.py`

Deterministic grid sweep using the **real config-plumbed gate** via overrides (not the spike's
monkeypatch), reusing `scripts/baltic_bistability_chunk0.py` helpers.

- **Grid**: `S50 ∈ {30_000, 60_000, 90_000, 120_000}` × `θ ∈ {2.0, 4.0}` ×
  `larval-M scale ∈ {0.6, 0.75, 0.85, 1.0}`. Weighted to the higher larval scales — the only regime
  where cod is viable-but-not-overshooting (low scales overshoot to millions per the spike).
- **Overrides per point**: `reproduction.depensation.gate.enabled=true`,
  `reproduction.depensation.gate.species.enabled.sp0=true`,
  `reproduction.depensation.gate.s50.sp0=<S50>`, `reproduction.depensation.gate.theta.sp0=<θ>`,
  plus `warmstart_override(True)` + `cod_rich_seeding()` / `cod_poor_seeding()` +
  `larva_scale_override(scale, base_rates)`.
- **Per point**: run cod-rich and cod-poor warm-start ICs over a **long horizon (40–50 yr)**,
  **3–5 seeds**. Classify with `baltic_bistability_chunk0.py`'s `classify_state` / `basins_differ`
  and compute healthy-basin magnitude + a stability check (CV + linear trend over the eval window —
  a stable basin has low CV and ~0 trend; a transient shows a decaying trend).
- **Output** (to a diagnostics doc + a machine-readable table): `(S50, θ, scale) →
  {bistable?, healthy_mean, healthy_stable?, collapsed_mean}`. **Select** the operating point =
  bistable AND healthy_mean ~O(100kt) AND healthy_stable AND collapsed_mean distinctly lower.
  **Selection rule when multiple points qualify:** prefer the point whose healthy basin is closest to
  cod Bpa (~120kt) among the stable ones (lowest CV as the tie-breaker) — this leaves SP2 the least
  distance to travel toward the ICES band.
- Runner: `run_simulation` from `calibrate_baltic` (in-process; the gate is now real config, so no
  monkeypatch). Long horizon = the fix for the spike's 15-yr instrument-limit.

## Unit 3 — Validation (analysis → diagnostics doc)

At the chosen operating point:
1. **Warm-start basin split** — rerun cod-rich vs cod-poor with extra seeds + the long horizon;
   confirm a robust, non-transient split (healthy stable, collapsed stays low).
2. **Fishing-hysteresis F-ramp** — from a healthy warm-start, ramp cod F low→high→low over N years
   using the validated `byyear`-F tooling (`mortality.fishing.rate.byyear.file.sp0`, per
   `scripts/spikes/ssb_f_hindcast_spike.py`), extract annual SSB, and plot SSB **parametrically vs F**.
   Confirm a **hysteresis loop**: collapse at F_collapse on the up-leg, recovery only at
   F_recover < F_collapse on the down-leg (the two legs do not overlap). This is the payoff — the
   bistability manifesting as genuine fishing hysteresis.

## Unit 4 — Config overlay: `data/baltic_depensation/`

DRY overlay on `data/baltic` (like the `baltic_a2` preset): only the changed keys — gate enabled for
cod + chosen S50/θ + adjusted larval-M (the operating point). **Python-engine-only, Java-guarded**
(the existing `nbackground>0` Java guard applies). Registered as a loadable demo/preset like
`baltic_a2`. This is the bistable Baltic variant SP2 consumes.

## Testing

**Unit 1 (CI gate)** — new `tests/test_depensation_gate.py` (+ config-parse cases), mirroring
`tests/test_reproduction_thermal_gate.py` and `tests/test_engine_stock_recruitment.py`:
- `depensation_factor` math: `A(S50)=0.5`; `A→0` as SSB→0 (and `=0` at SSB=0); `A→1` at large SSB;
  `=1.0` where disabled.
- Per-species isolation: only enabled species' factor differs from 1.0.
- **Off → byte-identical**: a short Baltic run with the gate off produces output identical to the
  current baseline (the determinism guarantee).
- Seeded-step skip: on a step where SSB is seeded, the factor is not applied.
- Config parse: keys → EngineConfig fields; `θ<1` rejected; `s50≤0` rejected; global-on/no-species →
  inert (`None`).
- Integration: gate-on measurably changes cod recruitment; the overlay config loads and runs; the
  Java-guard rejects the overlay on the Java engine (mirrors the existing Baltic Java-guard test).

**Units 2–3** — emergent analysis, **NOT CI gates** (long-running, seed/core-sensitive). Deliverable
is the diagnostics doc (mapped bistable region + chosen operating point + hysteresis loop). A light
1-point smoke test may live behind a skip-CI marker.

## Success criteria

SP1 succeeds when **both**:
1. The gate feature is shipped — config-plumbed, default-off byte-identical, unit tests green.
2. A documented operating point exists that is **bistable + healthy-O(100kt) + stable**, delivered as
   the `data/baltic_depensation` overlay, with the warm-start split and the hysteresis loop
   demonstrated in a diagnostics doc.

### Honest-negative fallback

The spike's healthy basins were either overshoot (millions, low larval-M) or transient (185kt at
deployed larval-M). Whether a point exists that is bistable **and** healthy-O(100kt) **and** stable is
the empirical question SP1 answers — it may not. **If the grid comes up empty**, SP1 still ships the
**gate feature** (valuable, tested, config-plumbed) plus a **documented negative** ("a bistable region
exists but the healthy basin cannot be placed at realistic-and-stable magnitude"), which reframes SP2.
We do not report a null as a success.

## Out of scope (SP2 and beyond)

- Driving the bistable overlay with historical annual F to reproduce the observed cod
  collapse-and-no-recovery (SP2 — committed follow-on, its own spec).
- Strict ICES-band match / full multi-species biomass-band recalibration of the healthy basin.
- Depensation for species other than cod (the gate is general; only cod is calibrated here).
- Any change to the deployed Baltic default config.

## Key references

- Spike (GO): `docs/diagnostics/2026-07-16-depensation-bistability-spike.md`, script
  `scripts/spikes/depensation_bistability_spike.py` (PR #117).
- Warm-start harness: `scripts/baltic_bistability_chunk0.py` (classifier + seeding/warmstart helpers).
- byyear-F tooling: `scripts/spikes/ssb_f_hindcast_spike.py`,
  `mortality.fishing.rate.byyear.file.sp{i}`.
- Gate pattern to mirror: `osmose/engine/processes/thermal_gate.py`,
  `osmose/engine/config.py::_load_thermal_gate`, `tests/test_reproduction_thermal_gate.py`.
- SR wiring: `osmose/engine/processes/reproduction.py:15-190`.
- Science: Casini et al. 2009 (PNAS 10.1073/pnas.0906620106); Köster & Möllmann 2000
  (10.1006/jmsc.1999.0528); Möllmann tipping-points (10.1111/nrm.12336) — cultivation-depensation /
  predator-pit basis for Baltic-cod depensation.
