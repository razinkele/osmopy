# Run-Page Engine-Capability Transparency — Design

**Date:** 2026-06-20
**Status:** Approved (brainstorming complete)

## Goal

Make the Run page honestly communicate **what the selected engine will and won't produce** for the loaded config, and remove the misleading engine "chooser" on the page. Today the two engines (pure-Python primary, Java jar subprocess) write different output families and gate different downstream pages, but the Run page is silent — so a user runs an engine and later finds empty Results/Diagnostics/Genetics panels with no explanation. The page's Java/Python tabs also *look* like the engine chooser but are a read-only mirror of the global header toggle (clicking them does nothing).

This is **surfacing + a chooser-clarity fix only** — no change to engine/runner execution behavior.

## Scope

**In scope:**
- A pure capability core describing, per engine + config: can-it-run, which result pages will populate, and the notable output-family differences.
- Run-page changes: replace the misleading Java/Python tabs with a clear "active engine" indicator + the active engine's settings + a live capability panel.

**Out of scope (deferred; noted from the Run-page investigation):**
Python progress streaming to the console, the timeout-vs-failed status label fix, a post-run "View Results" affordance, real run-history `duration_sec`/`summary`, and an engine-filtered Results dropdown. None are needed for the transparency goal the user chose.

## Background — the engine output divergence (the thing being surfaced)

One reader (`OsmoseResults`) serves both engines transparently, but each engine writes a different set of output families, so panels silently go empty:

- **Java-only families** (empty after a *Python* run; the Results dropdown still lists them): `biomassByTL`, `meanSize`/`meanSizeByAge`, `meanTLByAge`/`meanTLBySize`, `yieldN*`, `yieldByAge`/`yieldBySize`, `dietByAge`/`dietBySize`, `fisheryYield*`, `sizeSpectrum`.
- **Python-only** (empty after a *Java* run): genetics (`genetic_trait_means`), economics (`econ_*`), community Sheldon/MTL (relies on Python-written `*DistribBySize`+realized `meanTL`), spatial NetCDF (`osm_spatial_*`).
- **Config gating:** background-species configs (`simulation.nbackground>0`, e.g. Baltic) are **Java-blocked** by `java_engine_block_reason`; bioen/genetics/economics are Python-only.
- **Cross-engine is not bit-equal** (PCG64 vs MT19937 — "within ~1 OoM" parity), so comparisons are statistical-equivalence, not exact.

Downstream pages already gate on `state.engine_mode == "python"` (e.g. `genetics.py:27`, `economic.py:27`); the capability panel makes the *reason* explicit up front.

## Component 1 — `osmose/engine_capabilities.py` (pure, browser-free, tested)

Single source of truth. No Shiny, no engine imports beyond the existing `java_engine_block_reason`.

```python
from dataclasses import dataclass

@dataclass
class EngineCapability:
    engine: str                  # "python" | "java"
    can_run: bool                # for THIS config
    block_reason: str | None     # why not (e.g. Java + background species)
    pages_populated: list[str]   # result/diagnostic pages that WILL have data
    pages_empty: list[str]       # pages that will NOT, for this engine+config
    notable_outputs: str         # one concise line of family-level differences

def describe_engine(engine: str, config: dict[str, str]) -> EngineCapability: ...
```

**Curated capability data (the single map):**

- The candidate pages the panel reasons about (the real nav pages, `app.py`): `"Results"`, `"Spatial Results"`, `"Diagnostics"`, `"Genetics"`, `"Economic"`. (Community/Sheldon metrics live *inside* the Diagnostics page, which gates on `engine_mode=="python"` at `diagnostics.py:57` — so they're covered by `Diagnostics`, not a separate entry.)
- **Python** populates: `Results` and `Diagnostics` always; `Spatial Results` iff `output.spatial.enabled` truthy; `Genetics` iff `module.genetics.enabled` truthy; `Economic` iff `module.bioeconomics.enabled` truthy. (Module toggles are the canonical 4.4.0 keys; `state.config` is canonicalized, so read them directly.)
- **Java** populates: `Results` only — the Results dropdown carries the rich Java-only families (sizeSpectrum, byTL, fishery-yield, …). The dedicated pages `Diagnostics`, `Genetics`, `Economic`, and `Spatial Results` are all Python-gated (verified: `diagnostics.py:57`, `genetics.py:27`, `economic.py:27`) → they go in `pages_empty` with the reason "Python-engine only."
- `notable_outputs` strings (curated, short — not a 20-row mirror of every getter):
  - python: *"Not produced on the Python engine: sizeSpectrum, meanSize, meanTLByAge, yieldN, fishery-yield (run these on the Java engine)."*
  - java: *"Java run: no genetics, economics, or community size-spectrum outputs; cross-engine results are statistically equivalent, not bit-identical."*
- `can_run`/`block_reason`: for `engine == "java"`, call `java_engine_block_reason(config)`; if non-None → `can_run=False`, `block_reason=<that>`. Python always `can_run=True` for supported configs.

**Config-flag truthiness helper:** treat a key as enabled when its value lower-cases to `true`/`1` (mirror the engine's `cfg.get(k, "false").lower() == "true"` convention); missing/empty → disabled.

This module is fully unit-testable with plain dicts.

## Component 2 — Run-page changes (`ui/pages/run.py`)

**Remove** (chooser fix, option b):
- the `run_engine_tabs` `navset_tab` (lines ~187–234) and its one-way mirror observer `_sync_engine_tab` (lines ~577–581).

**Add three reactive render slots** (all reading `state.engine_mode`; the capability panel also reads `state.config`):

1. **Active-engine indicator** — `@render.ui`: "Active engine: **Python** — change in the header toggle ↗" (or Java). Honest label; the header toggle remains the single source of truth for `engine_mode` (no new writers).
2. **Engine-settings slot** — `@render.ui`: renders only the *active* engine's existing inputs — Java: jar path display, `java_opts`, `run_timeout`; Python: threads, verbosity — with the SAME input ids and behavior as today. (Switching engines re-renders this slot and resets these rarely-touched fields — accepted.)
3. **Capability panel** — `@render.ui`: builds `describe_engine(state.engine_mode.get(), state.config.get())` and renders:
   - if `not can_run`: a prominent warning with `block_reason` (e.g. "This config has background species — not supported by the Java engine; switch to Python in the header.");
   - else: a "Will populate: <pages_populated>" list, a muted "Won't populate (this engine): <pages_empty>" list, and the `notable_outputs` line;
   - if `config` is empty/unloaded: a neutral "Load a configuration to see engine capabilities." state.

`handle_run` is unchanged — it already reads the active engine's inputs on the matching branch, which now match the rendered settings slot.

## Data flow

```
header toggle ──> state.engine_mode ─┐
                                     ├─> indicator (render.ui)
                                     ├─> engine-settings slot (render.ui)
state.config ────────────────────────┴─> capability panel (render.ui) ─> describe_engine(engine, config)
```
No new writers to `engine_mode`; single source of truth preserved. Pure `describe_engine` is the only new logic.

## Error handling

The capability panel is read-only and total: any engine/config combination returns a valid `EngineCapability` (empty config → neutral state; Java + unsupported config → `can_run=False` + reason). No new failure paths; no exceptions surfaced to the user.

## Testing

**Pure (`tests/test_engine_capabilities.py`):**
- `describe_engine("java", {background-species cfg})` → `can_run=False`, `block_reason` mentions background/Java.
- `describe_engine("java", {plain cfg})` → `can_run=True`; `Diagnostics`/`Genetics`/`Economic`/`Spatial Results` in `pages_empty`; `Results` in `pages_populated`.
- `describe_engine("python", {module.genetics.enabled=true})` → `Genetics` in `pages_populated`; with it absent/false → in `pages_empty`. `Results`/`Diagnostics` always populated for Python.
- `describe_engine("python", {module.bioeconomics.enabled=true})` → `Economic` populated; `output.spatial.enabled=true` → `Spatial Results` populated.
- `notable_outputs` contains the engine-appropriate families (e.g. python mentions `sizeSpectrum`; java mentions genetics/economics + "statistically equivalent").
- truthiness helper: `"true"/"1"/"True"` enabled; `""/"false"/missing` disabled.

**Page (`tests/test_ui_run.py` or new `tests/test_ui_run_capability.py`):**
- `import app` clean; `ui.pages.run` exposes the new render functions / the page renders without the removed navset.
- a smoke test that the capability render returns engine-appropriate text (call `describe_engine` via the page path or assert the helper is wired).

**e2e (optional, viztest-gated):** load a config, toggle engine in the header, assert the Run page's capability panel text changes (Python ↔ Java).

## Files

- **Create:** `osmose/engine_capabilities.py`, `tests/test_engine_capabilities.py`
- **Modify:** `ui/pages/run.py` (remove `run_engine_tabs` navset + `_sync_engine_tab`; add indicator + settings slot + capability panel), and the Run page test file if present.
- **No `app.py` change** (the header toggle is untouched) → no nav / visual-baseline change.

## Reused infrastructure

`osmose.runner.java_engine_block_reason` (block reason), `ui.state` (`engine_mode`, `config`), the existing per-engine input ids in `run.py`, and the canonical `module.*.enabled` / `output.spatial.enabled` config keys.
