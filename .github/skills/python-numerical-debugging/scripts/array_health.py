#!/usr/bin/env python3
"""
array_health.py — Diagnostic scanner for numerical arrays in a running Python process.

Usage as a module:
    from array_health import array_health, scan_namespace

Usage as a script (scans a pickled dict of arrays):
    python3 array_health.py snapshot.pkl

Exit codes:
    0  All arrays healthy
    1  Issues detected
"""

import sys
import numpy as np
from typing import Any, Optional


def array_health(
    name: str,
    arr: Any,
    *,
    expect_finite: bool = True,
    expect_nonneg: bool = False,
    bounds: Optional[tuple[float, float]] = None,
    verbose: bool = True,
) -> dict:
    """Diagnose a single numerical array. Returns a dict of findings.

    Parameters
    ----------
    name : str
        Label for this array in output.
    arr : array-like
        The array to inspect.
    expect_finite : bool
        If True, flag NaN and Inf as issues.
    expect_nonneg : bool
        If True, flag negative values as issues.
    bounds : tuple[float, float] | None
        If set, flag values outside [lo, hi] as issues.
    verbose : bool
        If True, print the diagnostic summary.

    Returns
    -------
    dict with keys: shape, dtype, min, max, mean, nan_count, inf_count,
    zero_count, neg_count, finite, issues (list of strings).
    """
    a = np.asarray(arr)
    is_numeric = np.issubdtype(a.dtype, np.number)

    result = {
        "name": name,
        "shape": a.shape,
        "dtype": str(a.dtype),
        "size": a.size,
        "issues": [],
    }

    if not is_numeric:
        result["issues"].append(f"Non-numeric dtype: {a.dtype}")
        if verbose:
            print(f"--- {name} ---")
            print(f"  shape={a.shape}  dtype={a.dtype}  NON-NUMERIC")
        return result

    af = a.astype(float) if not np.issubdtype(a.dtype, np.floating) else a

    nan_count = int(np.isnan(af).sum())
    inf_count = int(np.isinf(af).sum())
    zero_count = int(np.count_nonzero(af == 0))
    neg_count = int(np.count_nonzero(af < 0))

    result.update({
        "min": float(np.nanmin(af)) if af.size > 0 else None,
        "max": float(np.nanmax(af)) if af.size > 0 else None,
        "mean": float(np.nanmean(af)) if af.size > 0 else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_count": zero_count,
        "neg_count": neg_count,
        "finite": bool(np.isfinite(af).all()),
    })

    if expect_finite and nan_count > 0:
        pct = 100 * nan_count / af.size
        result["issues"].append(f"Contains {nan_count} NaN ({pct:.1f}%)")
        # Find first NaN location
        idx = np.argwhere(np.isnan(af))
        if len(idx) > 0:
            result["first_nan_at"] = tuple(idx[0].tolist())

    if expect_finite and inf_count > 0:
        result["issues"].append(f"Contains {inf_count} Inf")
        idx = np.argwhere(np.isinf(af))
        if len(idx) > 0:
            result["first_inf_at"] = tuple(idx[0].tolist())

    if expect_nonneg and neg_count > 0:
        result["issues"].append(
            f"Contains {neg_count} negative values (min={np.nanmin(af):.6g})"
        )

    if bounds is not None:
        lo, hi = bounds
        oob = np.count_nonzero((af < lo) | (af > hi))
        if oob > 0:
            result["issues"].append(
                f"{oob} values outside [{lo}, {hi}]"
            )

    if a.size == 0:
        result["issues"].append("Empty array (size=0)")

    if verbose:
        print(f"--- {name} ---")
        print(f"  shape={a.shape}  dtype={a.dtype}  size={a.size}")
        if a.size > 0:
            print(f"  min={result['min']:.6g}  max={result['max']:.6g}  mean={result['mean']:.6g}")
            print(f"  NaN={nan_count}  Inf={inf_count}  Zero={zero_count}  Negative={neg_count}")
        if a.ndim >= 2 and a.shape[1] > 1:
            row_sums = np.nansum(af, axis=1)
            print(f"  Row sums: [{np.min(row_sums):.6g}, {np.max(row_sums):.6g}]")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  ⚠ {issue}")
        else:
            print("  ✓ Healthy")

    return result


def scan_namespace(
    namespace: dict[str, Any],
    *,
    expect_finite: bool = True,
    min_size: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Scan all array-like objects in a namespace (e.g. locals()).

    Parameters
    ----------
    namespace : dict
        Variable name → value mapping (typically from locals() or vars(obj)).
    expect_finite : bool
        Flag NaN/Inf as issues.
    min_size : int
        Skip arrays smaller than this.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    List of result dicts, one per array found. Only arrays with issues are included.
    """
    results = []
    for name, val in sorted(namespace.items()):
        if name.startswith("_"):
            continue
        try:
            a = np.asarray(val)
        except (TypeError, ValueError):
            continue
        if not np.issubdtype(a.dtype, np.number) or a.size < min_size:
            continue
        r = array_health(name, a, expect_finite=expect_finite, verbose=verbose)
        if r["issues"]:
            results.append(r)
    return results


def compare_arrays(
    name: str,
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    verbose: bool = True,
) -> dict:
    """Compare two arrays element-wise and report differences.

    Returns
    -------
    dict with keys: match (bool), max_abs_diff, max_rel_diff, mismatch_count,
    first_mismatch_at, issues.
    """
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)

    result = {"name": name, "issues": []}

    if a.shape != b.shape:
        result["issues"].append(f"Shape mismatch: {a.shape} vs {b.shape}")
        result["match"] = False
        if verbose:
            print(f"--- {name} COMPARE ---")
            print(f"  ⚠ Shape mismatch: {a.shape} vs {b.shape}")
        return result

    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    mismatch = ~close
    mismatch_count = int(mismatch.sum())

    abs_diff = np.abs(a - b)
    # Avoid NaN in relative diff
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(b != 0, abs_diff / np.abs(b), 0.0)

    max_abs = float(np.nanmax(abs_diff))
    max_rel = float(np.nanmax(rel_diff[np.isfinite(rel_diff)])) if np.any(np.isfinite(rel_diff)) else 0.0

    result.update({
        "match": mismatch_count == 0,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatch_count": mismatch_count,
    })

    if mismatch_count > 0:
        idx = np.argwhere(mismatch)
        result["first_mismatch_at"] = tuple(idx[0].tolist())
        loc = result["first_mismatch_at"]
        result["issues"].append(
            f"{mismatch_count} mismatches (max |Δ|={max_abs:.6g}, max rel={max_rel:.6g})"
        )

    if verbose:
        print(f"--- {name} COMPARE (rtol={rtol}, atol={atol}) ---")
        print(f"  shape={a.shape}  mismatches={mismatch_count}/{a.size}")
        print(f"  max |Δ|={max_abs:.6g}  max rel Δ={max_rel:.6g}")
        if mismatch_count > 0:
            loc = result["first_mismatch_at"]
            print(f"  First mismatch at {loc}: actual={a[loc]:.10g}  expected={b[loc]:.10g}")
            print(f"  ⚠ MISMATCH")
        else:
            print("  ✓ Match")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 array_health.py <snapshot.pkl>")
        print("  Scans all arrays in a pickled dict.")
        sys.exit(1)

    import pickle
    path = sys.argv[1]
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Error: Expected dict, got {type(data).__name__}")
        sys.exit(1)

    print(f"Scanning {len(data)} items from {path}\n")
    issues = scan_namespace(data, verbose=True)

    if issues:
        print(f"\n{'='*40}")
        print(f"ISSUES FOUND in {len(issues)} array(s)")
        sys.exit(1)
    else:
        print(f"\n{'='*40}")
        print("All arrays healthy")
        sys.exit(0)
