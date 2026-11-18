"""Bridge-spike probe (plan task 0.2): runs checks 1-6 in ONE fresh process, prints JSON.

Run only as a subprocess of ``tests/test_csharp_bridge_spike.py``. The CLR is hosted here and
never in the pytest process, because CoreCLR's threads do not survive ``fork`` and later tests
may fork. The parent strips ``DOTNET_ROOT`` from this process's environment, so check 1 also
proves the bridge resolves the runtime itself (spec section 9.4).
"""

import sys
from pathlib import Path

# First, before any osmose import: a script's own directory is sys.path[0], so without this a
# probe launched from a worktree would import the main checkout's osmose (CLAUDE.md gotcha).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import threading
import time

import numpy as np

from osmose import engine_cs
from osmose.engine_cs import bridge

out: dict = {"osmose_file": engine_cs.__file__}

# 1. pythonnet hosts .NET through clr_loader and the method is callable.
engine = bridge.load()
spike = engine.Spike.BridgeSpike
out["load"] = {
    "framework": str(spike.FrameworkDescription()),
    "dotnet_root": str(engine_cs.resolve_dotnet_root()),
}

# 2. A NumPy buffer's address, passed as Int64, is modified in place with no copy.
a = np.arange(1.0, 6.0)
addr_before = a.ctypes.data
spike.ScaleInPlace(a.ctypes.data, a.size, 2.0)
out["in_place"] = {"values": a.tolist(), "same_buffer": a.ctypes.data == addr_before}

# 3. The GIL is released during a call: a Python thread keeps running through a 2 s C# call.
counter = [0]
stop = threading.Event()


def _tick():
    while not stop.is_set():
        counter[0] += 1
        time.sleep(0.001)


t = threading.Thread(target=_tick)
t.start()
time.sleep(0.05)
before = counter[0]
spike.Spin(2.0)
during = counter[0] - before
stop.set()
t.join()
out["gil"] = {"ticks_during_call": during}

# 4. Writing the control block from Python stops a running C# loop within one iteration.
control = np.zeros(8, dtype=np.int64)
result = {}
worker = threading.Thread(
    target=lambda: result.setdefault(
        "done", int(spike.LoopUntilCancelled(control.ctypes.data, 100_000, 2))
    )
)
worker.start()
deadline = time.monotonic() + 10
while control[1] < 50 and time.monotonic() < deadline:
    time.sleep(0.005)
control[0] = 1
seen_after_cancel = int(control[1])
worker.join(10)
out["cancel"] = {
    "seen_after_cancel": seen_after_cancel,
    "returned": result.get("done"),
    "joined": not worker.is_alive(),
}

# 5. A .NET exception arrives as a Python exception carrying its message.
try:
    spike.Throw("spike-message-7f3a")
    out["exception"] = {"raised": False}
except Exception as exc:  # pythonnet maps it to the .NET type
    out["exception"] = {"raised": True, "type": type(exc).__name__, "text": str(exc)}

# 6. Two Python threads call at once; both overlap in C# and both return correct sums.
xs = [np.arange(1.0, 100_001.0), np.arange(1.0, 200_001.0)]
sums: dict = {}


def _call(i):
    sums[i] = float(spike.SumAfterDelay(xs[i].ctypes.data, xs[i].size, 500))


threads = [threading.Thread(target=_call, args=(i,)) for i in range(2)]
t0 = time.monotonic()
for th in threads:
    th.start()
for th in threads:
    th.join()
out["concurrent"] = {
    "sums": [sums.get(0), sums.get(1)],
    "expected": [float(x.sum()) for x in xs],
    "wall_s": time.monotonic() - t0,
}

print(json.dumps(out))
