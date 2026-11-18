"""Where does the C# kernel's speed-up come from? (G0 follow-up to bench_kernel.py)

One process, one thread, the arms interleaved with the order rotated each round:

- **numba-arm:** ``bench_kernel.NumbaStep`` (the hand-assembled production kernel, the gate);
- **mortality():** the REAL ``osmose.engine.processes.mortality.mortality`` on deep copies of a
  snapshot captured with its run's ``ctx``. It does more than the arm (larva pre-pass, feeding
  stages, accessibility indices, rate precompute, post-loop updates), so it should take at
  least as long. If it is faster, the arm is a mis-assembled baseline;
- **numba-draws:** only the parallel kernel's draws, per sub-step and cell: the same per-cell
  ``np.random.seed(seed + cell * 7919)``, four ``permutation`` calls and one ``shuffle`` per slot;
- **cs-philox:** C# regrouped + mask with production Philox draws;
- **cs-injected:** the same with the oracle's recorded draws, which cost almost nothing.

Kernel-minus-draws is then ``numba-arm - numba-draws`` against ``cs-injected``.

    NUMBA_NUM_THREADS=1 .venv/bin/python scripts/csharp/decompose_kernel.py \\
        --fixture tmp/fx/baltic_step129.bin --demo baltic --step 129
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import copy
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from bench_kernel import NumbaStep
from make_fixtures import load_config
from numba import njit

import osmose.engine.processes.mortality as mort
import osmose.engine.simulate as sim
from osmose.engine import PythonEngine
from osmose.engine_cs import bridge
from osmose.engine_cs.bundle import Bundle


@njit(cache=False)
def numba_draws_only(boundaries, n_cells, seeds):
    """The parallel kernel's draws (bioen off), nothing else; returns a checksum."""
    total = 0
    for s in range(seeds.size):
        for cell in range(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue
            np.random.seed(seeds[s] + np.int64(cell) * np.int64(7919))
            n_local = end - start
            a = np.random.permutation(n_local).astype(np.int32)
            b = np.random.permutation(n_local).astype(np.int32)
            c = np.random.permutation(n_local).astype(np.int32)
            d = np.random.permutation(n_local).astype(np.int32)
            causes = np.array([0, 1, 2, 3], dtype=np.int32)
            orders = np.empty((n_local, 4), dtype=np.int32)
            for ii in range(n_local):
                np.random.shuffle(causes)
                for cc in range(4):
                    orders[ii, cc] = causes[cc]
            total += a[0] + b[0] + c[0] + d[0] + orders[n_local - 1, 0]
    return total


def capture_with_ctx(demo: str, step: int, seed: int) -> dict:
    snap: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(demo, None, Path(tmp))
        ndt = int(cfg.get("simulation.time.ndtperyear", "24"))
        cfg["simulation.time.nyear"] = str(step // ndt + 1)
        cfg["output.minimal.enabled"] = "true"  # the calibration regime: no diet tracking
        config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg, seed)
        real = sim._mortality

        class Done(Exception):
            pass

        def spy(state, resources, config_, rng_, grid_, step=0, species_rngs=None, ctx=None):
            if step == target:
                snap.update(
                    state=copy.deepcopy(state), resources=copy.deepcopy(resources),
                    config=config_, grid=grid_, step=step, ctx=copy.deepcopy(ctx),
                )  # fmt: skip
                raise Done
            return real(
                state,
                resources,
                config_,
                rng_,
                grid_,
                step=step,
                species_rngs=species_rngs,
                ctx=ctx,
            )

        target = step
        sim._mortality = spy
        try:
            sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)
        except Done:
            pass
        finally:
            sim._mortality = real
    return snap


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--fixture", required=True)
    p.add_argument("--demo", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=9)
    a = p.parse_args()

    fx = Bundle.load(a.fixture)
    arm = NumbaStep(fx)
    starts = np.asarray(fx.get("cell.start")).astype(np.int64)
    n_cells = int(fx.get("fixture.n_cells"))
    n_subdt = int(fx.get("fixture.n_subdt"))
    snap = capture_with_ctx(a.demo, a.step, a.seed)
    spike = bridge.load().Spike.KernelSpike.Load(a.fixture)

    def t_arm(r):
        arm.reset()
        t0 = time.perf_counter()
        arm.run(r, parallel=True)
        return time.perf_counter() - t0

    def t_mortality(r):
        st, rs, cx = (
            copy.deepcopy(snap["state"]),
            copy.deepcopy(snap["resources"]),
            copy.deepcopy(snap["ctx"]),
        )
        g = np.random.default_rng(r)
        t0 = time.perf_counter()
        mort.mortality(
            st, rs, snap["config"], g, snap["grid"], step=snap["step"], parallel=True, ctx=cx
        )
        return time.perf_counter() - t0

    def t_draws(r):
        seeds = np.random.default_rng(r).integers(0, 2**62, size=n_subdt).astype(np.int64)
        t0 = time.perf_counter()
        numba_draws_only(starts, n_cells, seeds)
        return time.perf_counter() - t0

    def t_cs(draws):
        def f(r):
            spike.Reset()
            t0 = time.perf_counter()
            spike.Regroup()
            spike.RunStep("regrouped", "mask", draws, r)
            return time.perf_counter() - t0

        return f

    arms = [("numba-arm", t_arm), ("mortality()", t_mortality), ("numba-draws", t_draws),
            ("cs-philox", t_cs("philox")), ("cs-injected", t_cs("injected"))]  # fmt: skip
    for _, f in arms:
        for _ in range(12):
            f(0)
    times = {n: [] for n, _ in arms}
    for r in range(a.rounds):
        k = r % len(arms)
        for n, f in arms[k:] + arms[:k]:
            times[n].append(f(r + 1))
    med = {n: statistics.median(v) for n, v in times.items()}
    print(f"{'arm':14s} {'median ms':>10s}  rounds (ms)")
    for n, _ in arms:
        print(f"{n:14s} {med[n] * 1e3:10.2f}  {', '.join(f'{x * 1e3:.1f}' for x in times[n])}")
    nk = med["numba-arm"] - med["numba-draws"]
    ck = med["cs-injected"]
    cd = med["cs-philox"] - med["cs-injected"]
    print(
        f"\nmortality() / numba-arm: {med['mortality()'] / med['numba-arm']:.3f}  (>= 1 expected)"
    )
    print(
        f"draws:            numba {med['numba-draws'] * 1e3:.2f} ms   C# philox {cd * 1e3:.2f} ms"
    )
    print(f"kernel w/o draws: numba {nk * 1e3:.2f} ms   C# {ck * 1e3:.2f} ms   ratio {nk / ck:.3f}")
    print(f"whole:            numba-arm / cs-philox {med['numba-arm'] / med['cs-philox']:.3f}")
    spike.Dispose()


if __name__ == "__main__":
    main()
