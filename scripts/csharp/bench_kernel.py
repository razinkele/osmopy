"""Kernel spike benchmark (plan task 0.6, check 5): one mortality step, Numba vs C#, one thread.

Interleaved in ONE process, alternating the order each round. This is the performance plan's A/B
protocol: medians, at least seven rounds, and a pair is valid only if its signs agree. Run it
only on an otherwise idle machine. Both sides do the same work from the same captured step.

- **Numba:** cell grouping (``argsort(kind="mergesort")`` + ``searchsorted``), then
  ``_cell_prey_candidates``, then for each sub-step the egg release plus
  ``_mortality_all_cells_numba``, drawing its own MT19937 numbers. The arrays sit at their real
  Python state indices, so the memory layout is Python's.
- **C#:** for the regrouped layouts, ``Regroup()`` (counting sort) and then ``RunStep``; for the
  indexed layouts, ``RunStep`` alone. Production Philox draws. Four variants:
  {regrouped, indexed} x {mask, lists}.

Rate precomputation, larva mortality and feeding stages are excluded on both sides.

    NUMBA_NUM_THREADS=1 .venv/bin/python scripts/csharp/bench_kernel.py FIXTURE.bin --rounds 9
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import osmose.engine.processes.mortality as mort
from osmose.engine_cs import bridge
from osmose.engine_cs.bundle import Bundle

VARIANTS = [
    ("regrouped", "mask"),
    ("regrouped", "lists"),
    ("indexed", "mask"),
    ("indexed", "lists"),
]


class NumbaStep:
    """The captured step at Python state indices, runnable repeatedly from a pristine copy."""

    def __init__(self, fx: Bundle) -> None:
        g = fx.get
        self.n_state = int(g("fixture.n_state"))
        self.n_subdt = int(g("fixture.n_subdt"))
        self.ndt = int(g("fixture.n_dt_per_year"))
        self.n_species = int(g("fixture.n_species"))
        self.n_cells = int(g("fixture.n_cells"))
        self.n_rsc = int(g("fixture.n_resources"))
        self.has_access = bool(g("fixture.has_access"))
        self.use_stage = bool(g("fixture.use_stage_access"))
        idx = np.asarray(g("school.python_index"))
        starts = np.asarray(g("cell.start"))
        self.cell_of = np.full(self.n_state, -1, dtype=np.int64)
        for c in range(self.n_cells):
            self.cell_of[idx[starts[c] : starts[c + 1]]] = c

        def scatter(name, dtype, fill=0):
            out = np.full(self.n_state, fill, dtype=dtype)
            out[idx] = np.asarray(g(name))
            return out

        f8, i4 = np.float64, np.int32
        self.pristine = {
            "abundance": scatter("school.abundance", f8),
            "egg_retained": scatter("school.egg_retained", f8),
            "psr": scatter("school.pred_success_rate", f8),
            "preyed": scatter("school.preyed_biomass", f8),
        }
        self.static = {
            "species_id": scatter("school.species_id", i4),
            "length": scatter("school.length", f8),
            "weight": scatter("school.weight", f8),
            "age_dt": scatter("school.age_dt", i4),
            "ffa": scatter("school.first_feeding_age_dt", i4, 1),
            "stage": scatter("school.feeding_stage", i4),
            "tl": scatter("school.trophic_level", f8),
            "pred_idx": scatter("school.pred_access_idx", i4),
            "prey_idx": scatter("school.prey_access_idx", i4),
            "eff_s": scatter("school.rate_starvation", f8),
            "eff_a": scatter("school.rate_additional", f8),
            "eff_f": scatter("school.rate_fishing", f8),
            "disc": scatter("school.discard_fraction", f8),
            "is_egg": scatter("school.is_egg", np.uint8).astype(bool),
            "is_bkg": scatter("school.is_background", np.uint8).astype(bool),
        }
        self.tables = {
            k: np.asarray(g(f"species.{k}"))
            for k in (
                "size_ratio_min",
                "size_ratio_max",
                "ingestion_rate",
                "fr_shape",
                "fr_halfsat",
            )
        }
        self.access = np.asarray(g("access.matrix"))
        self.rsc0 = np.asarray(g("resources.biomass")).copy()
        self.rsc_meta = [
            np.asarray(g(f"resources.{k}"))
            for k in ("size_min", "size_max", "trophic_level", "access_rows")
        ]
        self.zeros = np.zeros(self.n_state)
        self.reset()

    def reset(self) -> None:
        self.w = {k: v.copy() for k, v in self.pristine.items()}
        self.n_dead = np.zeros((self.n_state, 8))
        self.tl_ws = np.zeros(self.n_state)
        self.rsc = self.rsc0.copy()

    def run(self, seed: int, parallel: bool = True) -> None:
        s, w, t = self.static, self.w, self.tables
        cell_ids = self.cell_of
        valid = np.where(cell_ids >= 0)[0].astype(np.int32)
        order = np.argsort(cell_ids[valid], kind="mergesort")
        sorted_idx = valid[order]
        boundaries = np.searchsorted(cell_ids[valid][order], np.arange(self.n_cells + 1))
        cand_ptr, cand_pos = mort._cell_prey_candidates(
            sorted_idx, boundaries, self.n_cells, s["length"], s["species_id"], s["stage"],
            t["size_ratio_min"], t["size_ratio_max"], s["age_dt"], s["ffa"],
        )  # fmt: skip
        rng = np.random.default_rng(seed)
        egg = w["egg_retained"]
        inst = w["abundance"].copy()
        for _ in range(self.n_subdt):
            release = np.where(s["is_egg"] & (egg > 0), w["abundance"] / self.n_subdt, 0.0)
            egg = np.maximum(0, egg - release)
            # mortality() defaults to parallel=True and simulate never overrides it, so the
            # production kernel is the prange one, even at one thread (per-cell MT reseed).
            kernel = (
                mort._mortality_all_cells_parallel if parallel else mort._mortality_all_cells_numba
            )
            kernel(
                int(rng.integers(0, 2**63)), sorted_idx, boundaries, self.n_cells, inst,
                self.n_dead, s["eff_s"], s["eff_a"], s["eff_f"], s["disc"], s["species_id"],
                s["length"], s["weight"], s["age_dt"], s["ffa"], s["stage"], w["psr"], w["preyed"],
                s["tl"], t["size_ratio_min"], t["size_ratio_max"], t["ingestion_rate"],
                t["fr_shape"], t["fr_halfsat"], self.ndt, self.n_subdt, self.access,
                self.has_access, self.use_stage, s["prey_idx"], s["pred_idx"], self.rsc,
                self.rsc_meta[0], self.rsc_meta[1], self.rsc_meta[2], self.rsc_meta[3], self.n_rsc, self.n_species, self.tl_ws, True, mort._DUMMY_DIET,
                False, egg, False, mort._DUMMY_RSC_1D, mort._DUMMY_RSC_1D, self.zeros,
                self.zeros, self.zeros, s["is_bkg"], self.zeros, cand_ptr, cand_pos,
            )  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("fixture")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--json", help="write the raw timings here")
    a = p.parse_args()

    fx = Bundle.load(a.fixture)
    numba_step = NumbaStep(fx)
    engine = bridge.load()
    spike = engine.Spike.KernelSpike.Load(a.fixture)

    totals: dict[str, list[tuple[float, float]]] = {}

    def time_arm(name: str, v, seed: int) -> float:
        if v is None or isinstance(v, bool):
            numba_step.reset()
            t0 = time.perf_counter()
            numba_step.run(seed, parallel=v is None)
            dt = time.perf_counter() - t0
            work = (float(numba_step.n_dead[:, 0].sum()), float(numba_step.w["preyed"].sum()))
        else:
            spike.Reset()
            t0 = time.perf_counter()
            if v[0] == "regrouped":
                spike.Regroup()
            spike.RunStep(v[0], v[1], "philox", seed)
            dt = time.perf_counter() - t0
            work = tuple(float(x) for x in str(spike.Totals()).split())
        totals.setdefault(name, []).append(work)  # type: ignore[arg-type]
        return dt

    # numba-parallel is the gate: production's kernel at one thread. numba-serial is shown too.
    arms: list[tuple[str, object]] = [("numba-parallel", None), ("numba-serial", False)]
    arms += [(f"cs-{lay}-{f}", (lay, f)) for lay, f in VARIANTS]
    warmup: dict[str, list[float]] = {}
    for name, v in arms:  # tiered JIT / PGO on .NET, compile + cache load on Numba
        warmup[name] = [time_arm(name, v, 0) for _ in range(12 if name.startswith("cs") else 3)]
    times: dict[str, list[float]] = {name: [] for name, _ in arms}
    for r in range(a.rounds):
        rotated = arms[r % len(arms) :] + arms[: r % len(arms)]
        for name, v in rotated:
            times[name].append(time_arm(name, v, r + 1))

    med = {k: statistics.median(v) for k, v in times.items()}
    gate = "numba-parallel"
    print(f"{'arm':24s} {'median s':>9s} {'gate/arm':>9s} {'serial/arm':>10s}  gate/arm per round")
    for name, _ in arms:
        ratios = [n / c for n, c in zip(times[gate], times[name], strict=True)]
        print(
            f"{name:24s} {med[name]:9.4f} {med[gate] / med[name]:9.3f} "
            f"{med['numba-serial'] / med[name]:10.3f}  [{min(ratios):.3f}, {max(ratios):.3f}]"
        )
    print("work per arm, median over timed runs (predation deaths, preyed biomass):")
    for name, _ in arms:
        d = [w[0] for w in totals[name][-a.rounds :]]
        b = [w[1] for w in totals[name][-a.rounds :]]
        print(f"  {name:24s} {statistics.median(d):.6g} {statistics.median(b):.6g}")
    env = {k: v for k, v in os.environ.items() if k.startswith(("DOTNET_", "NUMBA_"))}
    print("env:", env)
    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {"rounds": a.rounds, "times": times, "median": med, "warmup": warmup,
                 "totals": totals, "env": env, "fixture": a.fixture},
                indent=1,
            )
        )  # fmt: skip
    spike.Dispose()


if __name__ == "__main__":
    main()
