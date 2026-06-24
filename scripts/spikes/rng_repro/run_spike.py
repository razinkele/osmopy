"""RNG-reproduction feasibility spike orchestrator.

End-to-end:
1. Assert provenance (worktree + numba path).
2. Build portable / native .so if absent.
3. Parity grid: compare_cell over GRID_N x GRID_SEEDS → PASS or STOP.
4. Speed: bench_rng at n in {12, 33} on the portable build.
5. Write docs/perf/2026-06-25-rng-repro-feasibility-spike.md.
6. Print PASS/STOP verdict.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Provenance guard — must come first.
# ---------------------------------------------------------------------------
worktree_root = Path(__file__).resolve().parents[3]

from scripts.spikes.native_predation.provenance import assert_provenance  # noqa: E402

prov = assert_provenance(worktree_root)
print(f"[provenance] OK — mortality @ {prov['mortality_file']}, numba {prov['numba_version']}")

# ---------------------------------------------------------------------------
# Build .so if absent, then import.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent


def _so_present(variant: str) -> bool:
    return any(HERE.glob(f"_rng_{variant}.cpython-*.so"))


for variant in ("portable", "native"):
    if not _so_present(variant):
        print(f"[build] compiling _rng_{variant} …")
        from scripts.spikes.rng_repro.build_ffi import build
        build(variant)
        print(f"[build] _rng_{variant} done")
    else:
        print(f"[build] _rng_{variant} already built, skipping")

# cffi-compiled modules land in scripts/spikes/rng_repro/ — add to sys.path so
# they can be imported by name.
rng_dir = str(HERE)
if rng_dir not in sys.path:
    sys.path.insert(0, rng_dir)

import _rng_portable  # noqa: E402
import _rng_native    # noqa: E402

lib_portable = _rng_portable.lib
ffi_portable = _rng_portable.ffi
lib_native = _rng_native.lib
ffi_native = _rng_native.ffi

print("[import] _rng_portable and _rng_native imported OK")

# ---------------------------------------------------------------------------
# Parity grid.
# ---------------------------------------------------------------------------
from scripts.spikes.rng_repro.parity import GRID_N, GRID_SEEDS, compare_cell  # noqa: E402

total_combos = len(GRID_N) * len(GRID_SEEDS)
print(f"\n[parity] running {total_combos} combos (GRID_N={GRID_N}, GRID_SEEDS len={len(GRID_SEEDS)})")

first_divergence = None
all_results: list[dict] = []

for seed in GRID_SEEDS:
    for n in GRID_N:
        result = compare_cell(seed, n, lib_portable, ffi_portable)
        all_match = all(result.values())
        all_results.append({"seed": seed, "n": n, "match": all_match, "arrays": result})
        if not all_match and first_divergence is None:
            # find the first non-matching array and a sample index
            for arr_name, ok in result.items():
                if not ok:
                    first_divergence = {"seed": seed, "n": n, "array": arr_name, "index": 0}
                    break

parity_pass = first_divergence is None
mismatch_count = sum(1 for r in all_results if not r["match"])
print(f"[parity] {'PASS' if parity_pass else 'FAIL'} — {total_combos - mismatch_count}/{total_combos} bit-identical")
if first_divergence:
    print(f"[parity] first divergence: {first_divergence}")

# ---------------------------------------------------------------------------
# Speed benchmarks.
# ---------------------------------------------------------------------------
from scripts.spikes.rng_repro.speed import bench_rng  # noqa: E402

N_ITER = 200
N_SAMPLES = 30
bench_results: dict[int, dict] = {}

print("\n[speed] warming JIT (n=12, 1 iter) …")
bench_rng(12, 1, 1, lib_portable, ffi_portable)
print("[speed] JIT warm")

for n in (12, 33):
    print(f"[speed] bench n={n} …")
    r = bench_rng(n, N_ITER, N_SAMPLES, lib_portable, ffi_portable)
    bench_results[n] = r
    print(
        f"  n={n}: Numba {r['numba_med_ns']:.0f} ±{r['numba_iqr_ns']:.0f} ns  "
        f"C {r['c_med_ns']:.0f} ±{r['c_iqr_ns']:.0f} ns  "
        f"ratio(Numba/C)={r['ratio']:.2f}x"
    )

# ---------------------------------------------------------------------------
# Build artifact text.
# ---------------------------------------------------------------------------
VERDICT = "PASS" if parity_pass else "STOP"

# Parity table rows
parity_rows = []
for r in all_results:
    status = "match" if r["match"] else "MISMATCH"
    parity_rows.append(f"| {r['seed']:>22} | {r['n']:>4} | {status} |")
parity_table_body = "\n".join(parity_rows)

divergence_section = ""
if first_divergence:
    divergence_section = textwrap.dedent(f"""
    ### First divergence

    | field | value |
    |---|---|
    | seed | `{first_divergence['seed']}` |
    | n    | `{first_divergence['n']}` |
    | array | `{first_divergence['array']}` |
    | index | `{first_divergence['index']}` |

    **Classification:** MT19937 seeding or draw-order mismatch in `mt19937.c`.
    """)

b12 = bench_results[12]
b33 = bench_results[33]

artifact_text = f"""\
# RNG Reproduction Feasibility Spike — 2026-06-25

**Verdict: {VERDICT}**

---

## 0. Premise (spec §0a)

The cell-loop kernel in `osmose/engine/processes/mortality.py` (production Numba path,
`_mortality_all_cells_numba`) drives per-cell randomness via `np.random.seed(seed)` and
`np.random.permutation` / `np.random.shuffle` inside `@njit`-compiled code.  Numba's
`np.random` in JIT context is the **NumPy legacy MT19937 (Mersenne Twister)**, seeded by a
scalar `uint32` seed via the `init_genrand` path — identical to `numpy.random.RandomState`
seeded the same way.  The spike confirms this by checking that the `@njit` oracle and the
CPython `RandomState` reference produce bit-identical output across the full parity grid.

The C port (`mt19937.c`) targets the same algorithm: `init_genrand(seed & 0xFFFFFFFF)`
(scalar seed, NOT `init_by_array`) followed by the same draw sequence (4× `genrand_perm(n)` +
`n`× `genrand_shuffle(causes)`).

---

## 1. Parity grid

Portable build (`-O3`, no `-march=native`).  Every `(seed, n)` pair must be bit-identical
across all 5 output arrays (`seq_pred`, `seq_starv`, `seq_fish`, `seq_nat`, `cause_orders`).

Grid: `GRID_N = {GRID_N}`
Seeds (count {len(GRID_SEEDS)}): `{GRID_SEEDS}`

| seed | n | result |
|---|---|---|
{parity_table_body}

**Summary:** {total_combos - mismatch_count}/{total_combos} combos bit-identical.
{divergence_section}
---

## 2. C-vs-Numba RNG-gen speed (portable build)

`n_iter={N_ITER}`, `n_samples={N_SAMPLES}`, interleaved A/B to cancel drift.
Metric: median ± IQR (ns per iteration).

| n  | Numba median (ns) | Numba IQR (ns) | C median (ns) | C IQR (ns) | ratio Numba/C |
|----|-------------------|----------------|---------------|------------|---------------|
| 12 | {b12['numba_med_ns']:.0f} | {b12['numba_iqr_ns']:.0f} | {b12['c_med_ns']:.0f} | {b12['c_iqr_ns']:.0f} | {b12['ratio']:.2f}x |
| 33 | {b33['numba_med_ns']:.0f} | {b33['numba_iqr_ns']:.0f} | {b33['c_med_ns']:.0f} | {b33['c_iqr_ns']:.0f} | {b33['ratio']:.2f}x |

Note: `-march=native` is NOT the feasibility gate.  The portable build ratio already answers
whether a C RNG-gen call is a meaningful fraction of per-cell cost.

---

## 3. Verdict

**{VERDICT}**

{'Bit-exact RNG reproduction is FEASIBLE.' if parity_pass else 'Parity FAILED — bit-exact RNG reproduction is NOT demonstrated.'}
{'The C implementation (`mt19937.c`) reproduces the Numba MT19937 stream bit-for-bit across the full GRID_N × GRID_SEEDS grid (portable build).' if parity_pass else f'First divergence at seed={first_divergence["seed"]}, n={first_divergence["n"]}, array={first_divergence["array"]}.'}

{'### What a PASS authorises' if parity_pass else '### What a STOP means'}

{'A PASS authorises **designing** Stage 1 — the cell-loop port: porting `_mortality_all_cells_parallel`, `_apply_single_cause`, and the leaf integration, adding `prange`/OpenMP parallelism, and then **measuring** end-to-end `eec_full` wall-time before deciding whether to ship.' if parity_pass else 'A STOP means the C MT19937 port has a seeding or draw-order defect. Do not proceed to Stage 1 design until the first divergence is diagnosed and `mt19937.c` is corrected.'}

A PASS does **not** authorise building Stage 1 blind.  The full design review must address:

- **Maintenance caveat (permanent):** a compiled OpenMP C extension is a second implementation
  of the mortality kernel that must be kept in sync with every future change to the Numba
  production path.  It adds CI/wheel/packaging complexity for what is currently ~40% of a 2.5 s
  benchmark with no workload actively blocked on it.
- **RNG-gen cost share:** if the C RNG-gen speed ratio (Numba/C ≈ {b12['ratio']:.1f}x at n=12,
  {b33['ratio']:.1f}x at n=33) is a large fraction of per-cell cost, it would erode the
  integration win even if the loop + predation kernel itself is fast.  This must be quantified
  in the Stage 1 design before committing to the implementation.
- **Stage 1 scope:** port `_apply_single_cause` + leaf integration, wire `prange`/OpenMP,
  gate behind an env-var or config flag, and measure end-to-end on `eec_full` before any
  merge decision.

---

*Spike committed: `scripts/spikes/rng_repro/`*
*Harness retained for reproducibility.*
*Build artifacts (`.so`, `.o`, `_rng_*.c`) not tracked in git.*
"""

# ---------------------------------------------------------------------------
# Write artifact.
# ---------------------------------------------------------------------------
artifact_path = worktree_root / "docs" / "perf" / "2026-06-25-rng-repro-feasibility-spike.md"
artifact_path.write_text(artifact_text)
print(f"\n[artifact] written → {artifact_path}")

# ---------------------------------------------------------------------------
# Final verdict.
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"VERDICT: {VERDICT}")
if parity_pass:
    print("Bit-exact RNG reproduction confirmed across the full parity grid.")
    print("Stage 1 (cell-loop port) design is AUTHORISED.")
else:
    print(f"First divergence: seed={first_divergence['seed']}, n={first_divergence['n']}, "
          f"array={first_divergence['array']}")
    print("Stage 1 is NOT authorised until the divergence is fixed.")
print(f"Speed n=12: Numba/C ratio = {b12['ratio']:.2f}x")
print(f"Speed n=33: Numba/C ratio = {b33['ratio']:.2f}x")
print(f"{'=' * 60}")
