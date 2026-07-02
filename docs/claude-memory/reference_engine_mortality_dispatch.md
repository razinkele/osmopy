---
name: reference-engine-mortality-dispatch
description: "OSMOSE engine has TWO parallel mortality implementations — production uses the Numba batch path, NOT the per-cell Python path. Where to put mortality fixes."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9b0fdf28-14de-4aef-af1e-c62c175d61a7
---

# Engine mortality: two dispatch paths (the one that bites)

`osmose/engine/processes/mortality.py` has **two parallel mortality implementations**, and the per-cell Python one is NOT the production path. Editing the wrong one ships a fix that is **dead code in production**.

- **Production / CI hot path** (`_HAS_NUMBA=True`, the default install): `mortality()` outer dispatch (`if _HAS_NUMBA and len(valid_indices) > 0:`, ~`:1923`) → `_mortality_all_cells_numba` / `_mortality_all_cells_parallel` → `_apply_predation_numba`. Per-school rates come from `_precompute_effective_rates` (`eff_starv`/`eff_additional`/`eff_fishing`). **No bioen check in the outer dispatch** — bioen runs go through Numba too.
- **Pure-Python fallback** (`_HAS_NUMBA=False` ONLY): `mortality()` `else:` → `_mortality_in_cell` → `_get_mortality_causes` + `_apply_{starvation,additional,fishing,predation}_for_school`. The `if _HAS_NUMBA and not config.bioen_enabled` branch and the shuffled interleaved cause-loop live HERE — reachable only when numba is absent.

**Rule:** a mortality fix that must affect production goes in the **Numba path** (`_precompute_effective_rates` for rates; `_apply_predation_numba` for predation) AND, for parity, the Python fallback. Fixing only `_get_mortality_causes`/`_apply_*_for_school` touches the fallback alone. Verify which path a bug is on with `python -c "from osmose.engine.processes import mortality as m; print(m._HAS_NUMBA)"` (True here).

**Worked example (2026-06-24):** the bioen double-starvation "critical" was first fixed only in `_get_mortality_causes` (fallback) — dead code for production. The real production double-count was standard `eff_starv` (Numba kernel) applied ON TOP of `_bioen_step`'s bioen starvation; `update_starvation_rate` (`:2043`) runs ungated so `state.starvation_rate` is non-zero for under-fed bioen schools. Complete fix needed BOTH: exclude STARVATION from `_get_mortality_causes` (fallback) AND zero `eff_starv` when `config.bioen_enabled` in `_precompute_effective_rates` (production). Empirically confirmed: `eff_starv=[0.0208,0.0125]` for under-fed bioen schools pre-fix.

`@njit` kernels are positional — adding a param (e.g. `egg_retained`) means matching arg order at the kernel def, all 3 driver kernels (`_mortality_in_cell_numba`/`_mortality_all_cells_numba`/`_mortality_all_cells_parallel`), every call site, AND the existing direct kernel-call test at `tests/test_engine_functional_response.py:780`. See [[project-high-findings-remediation-2026-06]].
