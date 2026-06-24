# Task 4 Report: C Leaf Transcription + cffi Build

## Status: COMPLETE

## Files created
- `scripts/spikes/native_predation/kernel.c`
- `scripts/spikes/native_predation/build_ffi.py`

## Wrapper signatures authored

### `apply_predation_once` (parity target)
```c
void apply_predation_once(
    int p_idx, const int* cell_indices, int n_local,
    double* inst_abd, double* n_dead,
    const int* species_id, const double* length, const double* weight,
    const int* age_dt, const int* first_feeding_age_dt, const int* feeding_stage,
    double* pred_success_rate, double* preyed_biomass, const double* trophic_level,
    const double* size_ratio_min, const double* size_ratio_max,
    const double* ingestion_rate, const int* fr_shape, const double* fr_halfsat,
    double n_dt_per_year, double n_subdt,
    const double* access_matrix,
    int has_access, int use_stage_access,
    const int* prey_access_idx, const int* pred_access_idx,
    double* rsc_biomass, const double* rsc_size_min,
    const double* rsc_size_max, const double* rsc_tl, const int* rsc_access_rows,
    int n_resources, int n_species, int cell_id,
    double* tl_weighted_sum, int tl_tracking,
    double* diet_matrix, int diet_enabled,
    int* prey_type_buf, int* prey_id_buf, double* prey_eligible_buf,
    const double* egg_retained,
    /* 7 auxiliary shape args */
    int srm_ncol, int acc_nrow, int acc_ncol,
    int n_cells, int n_causes, int diet_nrow, int diet_ncol);
```

### `apply_predation_bench` (timing harness)
Same 41+7 args as `apply_predation_once`, plus bench-only tail:
```c
    int n_iter, int n_schools,
    const double* snap_inst_abd,
    const double* snap_n_dead,
    const double* snap_pred_success_rate,
    const double* snap_preyed_biomass,
    const double* snap_rsc_biomass,
    const double* snap_tl_weighted_sum,
    const double* snap_diet_matrix
```
Each iteration: memcpy all 7 mutated arrays from their pristine snapshots, then call `leaf`. Byte sizes:
- 1D arrays (inst_abd, pred_success_rate, preyed_biomass, tl_weighted_sum): `n_schools * sizeof(double)`
- n_dead: `n_schools * n_causes * sizeof(double)`
- rsc_biomass: `n_resources * n_cells * sizeof(double)`
- diet_matrix: `diet_nrow * diet_ncol * sizeof(double)`

### `noop`
Identical signature to `apply_predation_once`. Empty body with `(void)` suppressions for all args (no unused-parameter warnings).

## CDEF
The CDEF in `build_ffi.py` exactly matches all three wrapper signatures above — verified by successful compilation.

## Build command and output
```
PYTHONPATH=. .venv/bin/python -m scripts.spikes.native_predation.build_ffi
```
Output:
```
portable -> .../scripts/spikes/native_predation/_leaf_portable.cpython-312-x86_64-linux-gnu.so
native -> .../scripts/spikes/native_predation/_leaf_native.cpython-312-x86_64-linux-gnu.so
```
Both variants compiled with zero errors. gcc 13.3, cffi 2.0.0.

## Deviations from brief

1. **`n_causes` position**: The brief's `leaf` takes `n_causes` inline after `n_dead`. The public wrappers move it to the 7 auxiliary shape args block at the end (alongside `srm_ncol`, `acc_nrow`, etc.) — consistent with the cross-cutting contract. The `leaf` itself is verbatim from the brief.

2. **`n_local` is explicit in wrapper signatures**: The brief implied the caller slices `cell_indices` and provides `n_local`. It's kept as an explicit `int n_local` arg (consistent with `leaf`'s signature).

3. **No `__init__.py` change needed**: The compiled `.so` files land in `scripts/spikes/native_predation/` and are importable as `scripts.spikes.native_predation._leaf_portable` / `_leaf_native`.

## ABI compliance
- `n_dt_per_year` and `n_subdt` are C `double` (not int) — per contract.
- All 7 auxiliary shape ints (`srm_ncol`, `acc_nrow`, `acc_ncol`, `n_cells`, `n_causes`, `diet_nrow`, `diet_ncol`) appear in all three wrappers and the CDEF.
- `diet_nrow`+`diet_ncol` used for both diet bounds in `leaf` — correct.
- `n_local` passed as-is; scratch buffer (`prey_type_buf`, `prey_id_buf`, `prey_eligible_buf`) allocated by caller.

## Fix (post-review): modules were compiled but NOT importable

**Bug:** `ffi.set_source` was given a DOTTED module name
`f"scripts.spikes.native_predation._leaf_{variant}"` together with
`tmpdir=str(HERE)`. cffi interprets dots as subdirectory components under
`tmpdir`, so the `.so` was written to a double-nested path
`scripts/spikes/native_predation/scripts/spikes/native_predation/_leaf_*.so`
instead of directly into the package dir. The Step-3 smoke test only checked
that compilation succeeded (it printed the nested paths) and never attempted an
actual import, so `from scripts.spikes.native_predation import _leaf_portable`
raised `ImportError` — which would have blocked Task 5.

**Fix:** pass a NON-dotted module name `f"_leaf_{variant}"` to
`ffi.set_source` and keep `tmpdir=str(HERE)`. `HERE` is already the package
directory, so the `.so` lands directly in `scripts/spikes/native_predation/`
and is importable as `scripts.spikes.native_predation._leaf_<variant>`. Also
deleted the wrongly-created nested `scripts/spikes/native_predation/scripts/`
junk directory and stale `.so` artifacts.

**Verification (rebuild + actual import):**

```
$ PYTHONPATH=. .venv/bin/python -m scripts.spikes.native_predation.build_ffi
portable -> .../scripts/spikes/native_predation/_leaf_portable.cpython-312-x86_64-linux-gnu.so
native -> .../scripts/spikes/native_predation/_leaf_native.cpython-312-x86_64-linux-gnu.so
```
(both `.so` directly under `scripts/spikes/native_predation/`, no nested `scripts/`)

```
$ PYTHONPATH=. .venv/bin/python -c "from scripts.spikes.native_predation import _leaf_portable as L, _leaf_native as N; print(sorted(s for s in dir(L.lib) if not s.startswith('_'))); print(sorted(s for s in dir(N.lib) if not s.startswith('_')))"
['apply_predation_bench', 'apply_predation_once', 'noop']
['apply_predation_bench', 'apply_predation_once', 'noop']
```
Both modules import and expose all three symbols. No `.so` artifacts are
git-tracked (build outputs left untracked).
