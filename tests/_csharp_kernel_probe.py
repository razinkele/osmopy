"""Kernel-spike probe (plan task 0.6): run the C# kernel on a fixture, compare with the oracle.

Run only as a subprocess of ``tests/test_csharp_kernel_spike.py`` (the CLR never enters the
pytest process). For every layout x prefilter, with the oracle's injected draws, prints the C#
``KernelSpike.Compare()`` table; plus ``LibmCheck()``. Output is one JSON line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not the main one

import json

from osmose.engine_cs import bridge

fixture = sys.argv[1]
engine = bridge.load()
spike = engine.Spike.KernelSpike.Load(fixture)
result: dict = {"libm": str(spike.LibmCheck()).split(), "runs": {}, "digests": {}}
try:
    for layout in ("regrouped", "indexed"):
        for filt in ("full", "mask", "lists"):
            spike.Reset()
            if layout == "regrouped":
                spike.Regroup()
            spike.RunStep(layout, filt, "injected", 0)
            rows = {}
            for line in str(spike.Compare()).strip().splitlines():
                name, ndiff, max_abs, max_rel, max_scaled = line.split()
                rows[name] = {
                    "n_diff": int(ndiff),
                    "max_abs": float(max_abs),
                    "max_rel": float(max_rel),
                    "max_scaled": float(max_scaled),
                }
            result["runs"][f"{layout}/{filt}"] = rows
            result["digests"][f"{layout}/{filt}"] = str(spike.ResultDigest())
finally:
    spike.Dispose()
print(json.dumps(result))
