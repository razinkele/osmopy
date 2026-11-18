"""Cross-language bundle probe (plan task 0.3), run only as a subprocess of
``tests/test_csharp_bundle.py`` so the CLR never enters the pytest process.

Writes ``python.bin`` with the Python writer, then has C#:
1. load it and save it back (``csharp_roundtrip.bin``);
2. build a ``SimulationConfig`` from the same bundle's NumPy addresses and save it
   (``csharp_from_addresses.bin``).
Prints the values C# read, as JSON.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout, not the main one
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

from test_csharp_bundle import sample

from osmose.engine_cs import bridge

out = Path(sys.argv[1])
engine = bridge.load()
from System import Double  # importable only once the CLR is hosted

bundle = sample()
bundle.save(out / "python.bin")

loaded = engine.SimulationConfig.Load(str(out / "python.bin"))
try:
    loaded.Save(str(out / "csharp_roundtrip.bin"))
    ok, linf = loaded.TryGet("growth.linf", None)
    _, steps = loaded.TryGet("time.n_steps", None)
    _, deadline = loaded.TryGet("run.deadline_seconds", None)
    _, mask = loaded.TryGet("grid.ocean_mask", None)
    values = {
        "time.n_steps": int(steps.IntValue),
        "run.deadline_seconds": float(deadline.FloatValue),
        "growth.linf": [float(x) for x in linf.ToArray[Double]()],
        "grid.ocean_mask.shape": [int(x) for x in mask.GetShape()],
    }
finally:
    loaded.Dispose()

config = bundle.to_csharp(engine)  # borrows bundle's buffers; bundle stays alive below
config.Save(str(out / "csharp_from_addresses.bin"))
del config
print(json.dumps({"values": values}))
