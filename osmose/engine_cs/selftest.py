"""Self-test of the C# engine for a deployed environment (plan task 1.13, spec section 9.7).

Run it as its own process, with the interpreter and environment that will host the engine (the
service's, the image's)::

    python -m osmose.engine_cs.selftest

It loads the .NET runtime and runs one year of the bundled Bay of Biscay demo through the public
``CSharpEngine.run_in_memory`` path, then checks the frames. Exit 0 with one ``OK`` line, or exit
1 with a ``FAIL`` line naming the reason. It must not be imported into a process that forks or
that may host the CLR for something else (spec section 8.5): it is a command, not a library.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = "bay_of_biscay"
YEARS = 1


def problems(frames: dict[str, pd.DataFrame], expected_rows: int) -> list[str]:
    """What is wrong with a run's frames: each must have ``expected_rows`` rows, a ``Time`` column
    that increases, finite non-negative species values, and some biomass left at the end."""
    found: list[str] = []
    for name, frame in frames.items():
        if len(frame) != expected_rows:
            found.append(f"{name}: {len(frame)} rows, expected {expected_rows}")
            continue
        if "Time" not in frame or not np.all(np.diff(frame["Time"].to_numpy()) > 0):
            found.append(f"{name}: no increasing Time column")
        values = frame.drop(columns=["Time"], errors="ignore").select_dtypes("number").to_numpy()
        if values.size == 0:
            found.append(f"{name}: no species columns")
        elif not np.all(np.isfinite(values)):
            found.append(f"{name}: non-finite values")
        elif (values < 0).any():
            found.append(f"{name}: negative values")
        elif name == "biomass" and not (values[-1] > 0).any():
            found.append("biomass: every species is at zero in the last record")
    return found


def main() -> int:
    from osmose.engine_cs import csharp_engine_available, resolve_dotnet_root

    try:  # the probe itself can raise (an unreadable capabilities file): report that too
        ok, why = csharp_engine_available()
    except Exception as exc:
        ok, why = False, f"{type(exc).__name__}: {exc}"
    if not ok:
        print(f"FAIL: the C# engine is not available: {why}")
        return 1
    try:
        from osmose.config.reader import OsmoseConfigReader
        from osmose.demo import osmose_demo
        from osmose.engine.config import EngineConfig
        from osmose.engine_cs import CSharpEngine
        from osmose.engine_cs.build import n_records

        with tempfile.TemporaryDirectory(prefix="osmose_cs_selftest_") as tmp:
            raw = dict(OsmoseConfigReader().read(Path(osmose_demo(DEMO, Path(tmp))["config_file"])))
            raw["simulation.time.nyear"] = str(YEARS)
            raw["output.minimal.enabled"] = "true"
            raw["output.start.year"] = "0"  # a 1-year check needs its records (BoB sets 20)
            cfg = EngineConfig.from_dict(raw)
            n_steps = int(cfg.n_steps)
            # The engine's own recording rule (trailing partial window, output.step0.include).
            records = n_records(n_steps, cfg.output_record_frequency, cfg.output_step0_include)
            start = time.perf_counter()
            result = CSharpEngine().run_in_memory(raw, seed=1)
            elapsed = time.perf_counter() - start
            frames = {"biomass": result.biomass(), "abundance": result.abundance()}
    except Exception as exc:  # the point of a self-test: report any failure, then exit 1
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    found = problems(frames, expected_rows=records)
    if found:
        print("FAIL: " + "; ".join(found))
        return 1
    print(
        f"OK: C# engine ran {YEARS} year ({n_steps} steps) of {DEMO} in {elapsed:.2f} s, "
        f"first call included (.NET at {resolve_dotnet_root()})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
