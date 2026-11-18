"""Digests of the C# engine's result buffers on fixed configs: the "off is identical" check.

Each Phase 2 task adds a feature behind a capability bit. With the feature absent from the
config, the C# outputs must be byte-identical to the build before the change (plan, Phase 2
acceptance). Run this on the base ref and on the branch, each with its own built engine, and
compare the two outputs (``--json``): every line must match.

Covers EEC (2 years) and Bay of Biscay (2 years) with minimal outputs, at two seeds and two
thread counts. Hosts the CLR in this process, so never import it from a test.

Usage: PYTHONPATH=. python scripts/csharp/result_digests.py [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = [("eec_full", 2), ("bay_of_biscay", 2)]
SEEDS = (1, 7)
THREADS = (1, 3)


def digests() -> dict[str, str]:
    import numpy as np

    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    from osmose.engine import PythonEngine
    from osmose.engine.config import EngineConfig
    from osmose.engine_cs import bridge
    from osmose.engine_cs.build import allocate_results, build_bundle

    out: dict[str, str] = {}
    for demo, years in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            raw = dict(OsmoseConfigReader().read(Path(osmose_demo(demo, Path(tmp))["config_file"])))
            raw["simulation.time.nyear"] = str(years)
            raw["output.minimal.enabled"] = "true"
            cfg = EngineConfig.from_dict(raw)
            grid = PythonEngine()._resolve_grid(raw)
            for seed in SEEDS:
                for threads in THREADS:
                    bundle = build_bundle(cfg, raw, grid, seed=seed, threads=threads)
                    buffers = allocate_results(bundle, cfg)
                    summary = bridge.run(bundle, buffers.control)
                    h = hashlib.sha256()
                    for f in fields(buffers):
                        a = getattr(buffers, f.name)
                        if isinstance(a, np.ndarray) and f.name != "control":
                            h.update(f.name.encode())
                            h.update(np.ascontiguousarray(a).tobytes())
                    out[f"{demo} {years}y seed={seed} threads={threads}"] = (
                        f"{h.hexdigest()[:24]} records={summary[2]}"
                    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    result = digests()
    for k, v in result.items():
        print(f"{k:<40} {v}")
    by_case: dict[str, set[str]] = {}
    for k, v in result.items():
        by_case.setdefault(k.rsplit(" threads=", 1)[0], set()).add(v)
    if any(len(v) != 1 for v in by_case.values()):  # thread-count invariance, task 1.9
        print("ERROR: a case differs across thread counts", file=sys.stderr)
        sys.exit(1)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
