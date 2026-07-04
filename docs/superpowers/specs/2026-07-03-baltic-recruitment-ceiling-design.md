# Baltic recruitment ceiling (unfished-level cap) — design

**Date:** 2026-07-03
**Status:** approved design, scientifically validated (2026-07-03, scite + primary
sources: McGregor 2019 quote verified verbatim & non-retracted, Bossier 2018
verified, framing tightened), pre-implementation
**Author:** brainstormed with the user
**Related:** `docs/baltic_recruitment_literature_review_2026-07-03.md` (candidate #1),
`osmose/engine/processes/reproduction.py` (`apply_stock_recruitment`),
`osmose/engine/processes/recruitment_gate.py` + `osmose/engine/config.py:_load_rv_gate`
(inert-by-default pattern to mirror),
`docs/superpowers/specs/2026-06-25-model-internal-reference-points-design.md`
(sidecar-from-a-reference-run pattern to mirror)

## 1. Motivation

The OSMOSE Baltic config exhibits population-level cod boom/bust overshoot and
percid overshoot (×38–96). Five levers have been ruled out — parameter
recalibration (SP-A), grid refinement (SP-B), salinity-correct spawning areas,
per-cell spatial egg-survival (SP1/SP1b), and a per-year reproductive-volume
recruitment gate (RV gate, both `mean_preserving` and `raw_cap`). The RV gate
**worsened** overshoot because it *modulated* recruitment, injecting variance
the model amplifies.

The 2026-07-03 literature review identified **one untried lever that is not a
gate**: a recruitment *ceiling*. McGregor, Fulton & Dunn (2019, *PeerJ* 7:e7308)
show that a Beverton–Holt stock-recruitment curve produces "sudden and excessive
increase when the population expands" and recommend **limiting recruitment to its
unfished level** (verbatim: "To avoid implausible increases in biomass, we
propose limiting recruitment to its unfished level"). The Baltic config added an
explicit Beverton–Holt / Shepherd SR curve during phase-12 calibration, so this
warning applies directly.

**Framing precision (from the 2026-07-03 scientific validation).** McGregor's
demonstration system is myctophids (a *prey* species) on the Chatham Rise, NZ,
and their trigger is *fishing-induced predation release* — prey booming once
their predators are fished down. Cod is a top predator, so the modeled cod boom
is **not** classic predation release; it is the *general* B-H failure mode
McGregor also proves — excessive recruitment upside whenever the population
expands, for any reason (a good-recruitment run, reduced fishing, or predation
release). The ceiling is justified by that general result, which covers the cod
case directly. We keep the causal language precise: "expansion," not "predation
release," is what drives the cod overshoot.

Note the OSMOSE B-H curve already saturates (recruitment → `k·ssb_half` as
SSB→∞, `reproduction.py:69`), but its asymptote is the curve's
maximum-productivity ceiling, not the *unfished-equilibrium* level. McGregor's
fix is a ceiling *below* that asymptote, at the level the stock produces at its
natural unfished equilibrium. This spec measures that level and clamps to it.

## 2. Goals and non-goals

**Goals**
- Cap each enabled focal species' per-step recruitment (egg output) at its
  unfished-equilibrium level, removing the SR-curve's runaway upside whenever the
  population expands.
- Derive the ceiling by *measurement*, not by a guessed constant: an F=0
  reference run's late-window mean recruitment, recorded per within-year season
  index (preserving spawning-season shape).
- Keep the feature config-gated and **inert by default** — all existing configs
  (Baltic, EEC, Bay of Biscay) produce bit-identical output unless the ceiling
  is explicitly enabled.
- Quantify the effect with an A/B diagnostic on the cod boom/bust overshoot
  ratio (the same go/no-go signal used to judge the RV gate).

**Non-goals**
- No hand-tuned absolute egg ceiling and no re-derivation inline on every
  production run (see §7 alternatives).
- No percid ceiling. Percids overshoot for different, unrepresented reasons
  (thermal year-class gating + density-dependent cannibalism, lit-review
  candidates #2/#3). Out of scope here.
- No recalibration of `ssb_half` / larval mortality in this change.
- No new in-engine environmental state. The ceiling is a precomputed per-season
  vector; spatial egg placement is unchanged.
- No hindcast / calendar-anchored validation — this tests the mechanism's
  stabilizing effect, not a specific year's cod stock.

## 3. Concept

"Unfished level" is measured from an F=0 reference run: with fishing mortality
zeroed, cod's predators/competitors stay intact (top-down control preserved), so
the reference run should be *more* stable than the fished one — consistent with
McGregor's premise that the boom is a predation-release artifact. Read off the
equilibrium egg production at each within-year timestep, average over a late
window, and use it as a per-season clamp in production.

Because recruitment is a monotone-increasing function of SSB and the clamp only
binds when production recruitment *exceeds* the unfished per-season level,
behavior below the reference is unchanged (bit-identical). Recording per season
index (not one annual number) preserves the natural within-year spawning timing
that a flat per-step cap would distort.

## 4. Architecture — two parts

### Part A — Derivation tool (offline, writes a sidecar)

A CLI modeled on the model-internal reference-points sweep:

1. Load the target config; build an in-memory variant with fishing mortality
   zeroed — set whichever fishing mode is active to 0 (rate-based
   `mortality.fishing.rate.sp{i}` and/or v4 fisheries `fisheries.rate.base.fsh{j}`).
2. Run the engine deterministically for the config's horizon, with an
   egg-recording hook capturing per-step, per-species egg counts (the output of
   `apply_stock_recruitment`, *before* any ceiling clamp).
3. Group recorded eggs by season index (`step % n_cols`, where `n_cols` matches
   `spawning_season`'s column count), average over a **late window** (default:
   last third of model years; seeded steps excluded) → `ceiling[sp][season_idx]`.
4. Assert the reference run reached a usable stationary window; warn if the
   late-window per-season means are not reasonably stable (the unfished run is
   expected to be well-behaved — if it is not, the ceiling is ill-defined and
   the tool says so rather than emitting a garbage cap).
5. Write a sidecar CSV: one row per season index, one column per species.

**Sidecar CSV format** (wide, shaped like `spawning_season`):
```
season_idx,ceiling_sp0,ceiling_sp1,...,ceiling_sp{n-1}
0,<eggs>,<eggs>,...
1,<eggs>,<eggs>,...
...
{n_cols-1},...
```
`ceiling_sp{i}` is the mean unfished equilibrium egg count at that season index,
in the same units as `n_eggs` in `reproduction()` (post-`apply_stock_recruitment`,
tonnes→grams already applied).

### Part B — Engine clamp (production, inert-by-default)

Mirrors the RV-gate load/apply pattern exactly.

- `config.py:_load_recruitment_ceiling(cfg, n_species, ...)` returns
  `(ceiling_by_season, enabled_mask)` — a `(n_cols, n_species)` float array and a
  `(n_species,)` bool mask — or `(None, None)` when the master switch is off.
  Fail-fast (ValueError / FileNotFoundError) on any invalid config: missing
  file, wrong columns, non-contiguous season indices, NaN/negative ceilings, or
  master-on with no species enabled.
- New `EngineConfig` fields: `recruitment_ceiling_by_season` and
  `recruitment_ceiling_enabled` (both `None` when off), populated in `from_dict`
  alongside the RV-gate load.
- In `reproduction()`, immediately after `apply_stock_recruitment` and the RV
  gate (orthogonal to both), for each enabled species not seeded this step:
  ```python
  if config.recruitment_ceiling_by_season is not None:
      col = step % config.recruitment_ceiling_by_season.shape[0]
      for sp in range(n_sp):
          if config.recruitment_ceiling_enabled[sp] and not seeded_this_step[sp]:
              cap = config.recruitment_ceiling_by_season[col, sp]
              if n_eggs[sp] > cap:
                  n_eggs[sp] = cap
  ```
  The clamp block only executes when the loaded ceiling is not `None`, so
  master-off runs never touch `n_eggs` and stay bit-identical.

## 5. Config keys (mirroring RV-gate naming)

```
reproduction.recruitment.ceiling.enabled               = false   # master switch, default off
reproduction.recruitment.ceiling.series.file           = <sidecar.csv relative path>
reproduction.recruitment.ceiling.species.enabled.sp{i} = true|false   # per-species; Baltic: cod only
```

- Master switch default `false` → feature inert; no other keys read.
- `series.file` resolved via the same `_require_file` / `_cfg_dir` helper the RV
  gate uses.
- Per-species enable mask selects which species are clamped even though the
  sidecar contains all species. For Baltic, only cod (`sp{cod_idx}`) is enabled.
- All three key patterns added to the `config_validation` allowlist (via the AST
  walker or `_SUPPLEMENTARY_ALLOWLIST`) so
  `test_from_dict_warn_mode_clean_on_example_configs` stays warning-free.

## 6. Key behaviors & assumptions

- **Inert by default:** the clamp is skipped entirely when the loaded ceiling is
  `None` (master off). Existing configs are bit-identical.
- **Seeding-safe:** steps where SSB was seeded are skipped — the bootstrap must
  not be clipped, exactly as the RV gate skips `seeded_this_step`.
- **Season-index alignment:** the sidecar's row count must equal the config's
  `spawning_season` column count (`n_cols`); `_load_recruitment_ceiling`
  validates this so the `step % n_cols` lookup is always in range.
- **Unfished-run assumption:** the F=0 reference is expected to be more stable
  than the fished run (predators intact). The derivation tool checks this and
  warns rather than silently emitting an inflated ceiling if the reference run
  itself booms.
- **Orthogonality:** ceiling and RV gate compose — RV gate multiplies, ceiling
  clamps; either, both, or neither can be enabled.
- **Conservative and non-distorting (scientific strength):** the real Baltic cod
  stock currently recruits *below* its unfished level, so an unfished-level cap
  rarely binds in realistic low-recruitment regimes — it only clips the model's
  implausible boom. The fix corrects the artifact without distorting realistic
  low-recruitment dynamics; this is the main reason it is preferable to
  re-tuning the SR curve.
- **Per-season vector is an extension, not a literal McGregor quantity:**
  McGregor's R₀ is a single (annual) unfished-recruitment value; the per-season
  vector is this project's adaptation to OSMOSE's sub-annual spawning, chosen to
  preserve the within-year spawning shape. The `n_cols`-row late-window mean is
  our empirical estimator of R₀ per season index. The stationarity check on the
  F=0 run (above) guards the one real risk — an ill-defined R₀ if the reference
  run does not settle.

## 7. Alternatives considered (derivation orchestration)

- **(A) Standalone CLI → sidecar CSV** — *chosen.* Reproducible, inspectable,
  matches the model-internal reference-points pattern; derivation is decoupled
  from production.
- **(B) Lazy auto-derive on first production run + cache** — fewer manual steps
  but couples derivation to a production run and hides the reference from
  inspection.
- **(C) Re-derive inline every run** — simplest wiring but doubles runtime and
  re-runs the F=0 reference every time.

Mechanism alternatives also weighed and rejected during brainstorming:
cap effective SSB at a hand-set reference SSB (loses the "measured" fidelity the
user wanted), and cap the final egg count at a hand-set absolute constant
(flattens the within-year spawning peak).

## 8. Testing / validation

- **Unit (TDD):**
  - below ceiling → `n_eggs` unchanged (bit-identical);
  - above ceiling → clamped exactly to `ceiling[sp][season_idx]`;
  - monotonic (larger input never yields a larger output past the cap);
  - per-species mask respected (disabled species never clamped);
  - master-off → no-op;
  - `_load_recruitment_ceiling` fail-fast on each invalid-config case.
- **Parity:** bit-identical vs master when disabled, using the determinism keys
  `movement.randomseed.fixed` + `stochastic.mortality.randomseed.fixed`.
- **A/B diagnostic:** a script (like the SP1/SP1b diagnostics) running Baltic cod
  with the ceiling off vs on, reporting the boom/bust overshoot ratio. Damping
  the ratio is the go/no-go signal; no damping means the ceiling is ruled out
  like the previous levers (and we record that, per the "don't retry" discipline).

## 9. Deliverables

- `osmose/engine/processes/reproduction.py` — ceiling clamp after
  `apply_stock_recruitment`.
- `osmose/engine/config.py` — `_load_recruitment_ceiling` + two new
  `EngineConfig` fields, wired in `from_dict`.
- `osmose/engine/config_validation.py` — allowlist entries for the three keys.
- Derivation CLI `scripts/derive_recruitment_ceiling.py` (mirrors the existing
  `scripts/baltic_*` diagnostics) + the egg-recording hook it needs (optional,
  `None` in production → zero overhead).
- Sidecar CSV for Baltic, committed alongside the Baltic config's data dir (it
  is a small file, like the RV-gate series CSV).
- A/B diagnostic script.
- Tests covering §8.
