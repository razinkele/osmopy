"""``CSharpEngine``: the third in-process engine (spec section 9.2).

Same two methods and signatures as :class:`osmose.engine.PythonEngine`. Python resolves the
config into a bundle (:mod:`osmose.engine_cs.build`), allocates the result buffers, and the C#
engine runs the whole simulation and writes the records into them; this module turns those rows
into the ``StepOutput`` list that ``OsmoseResults.from_outputs`` and ``write_outputs`` already
consume. Nothing here imports pythonnet: :mod:`osmose.engine_cs.bridge` does, on first run.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from osmose.results import OsmoseResults
from osmose.runner import RunResult

logger = logging.getLogger("osmose")

#: ``RunStatus`` values (Osmose.Engine.RunStatus).
COMPLETED, CANCELLED, TIMED_OUT = 0, 1, 2


class CSharpEngineError(RuntimeError):
    """The C# engine refused the config or failed; the message says why."""


class CSharpEngineTimeout(CSharpEngineError):
    """The run passed its ``deadline_seconds`` (``RunStatus.TimedOut``)."""


class CSharpEngine:
    """The C# engine. Holds no run state: concurrent calls from several threads are allowed.

    ``deadline_seconds`` (0 = none) makes the engine stop on its own, for callers that are not on
    the main thread; ``checks`` turns on the engine's debug invariants (``run.checks``).
    """

    def __init__(
        self, *, threads: int = 1, deadline_seconds: float = 0.0, checks: bool = False
    ) -> None:
        self.threads = int(threads)
        self.deadline_seconds = float(deadline_seconds)
        self.checks = bool(checks)

    def _execute(self, config: dict[str, str], seed: int, cancel_token: threading.Event | None):
        from osmose.engine import PythonEngine, SimulationCancelled
        from osmose.engine.config import EngineConfig
        from osmose.engine.config_validation import warn_unread_java_only_keys
        from osmose.engine_cs import bridge
        from osmose.engine_cs.build import allocate_results, build_bundle
        from osmose.engine_cs.capabilities import csharp_engine_block_reason

        warn_unread_java_only_keys(config)
        engine_config = EngineConfig.from_dict(config)
        grid = PythonEngine()._resolve_grid(config)
        reason = csharp_engine_block_reason(config)
        if reason is not None:
            raise CSharpEngineError(f"the C# engine cannot run this config: {reason}")
        bundle = build_bundle(
            engine_config, config, grid, seed=seed, threads=self.threads,
            deadline_seconds=self.deadline_seconds, checks=self.checks,
        )  # fmt: skip
        buffers = allocate_results(bundle, engine_config)
        try:
            status, steps, records, peak, elapsed, scheme, phases = bridge.run(
                bundle, buffers.control, cancel_token
            )
        except bridge.CSharpEngineUnavailable as exc:
            raise CSharpEngineError(str(exc)) from exc
        except Exception as exc:
            # pythonnet surfaces a .NET exception as a Python one whose module is its namespace
            # (System.*, Osmose.Engine.*): bundle validation, unsupported features, invariants.
            if type(exc).__module__.split(".")[0] in ("System", "Osmose"):
                raise CSharpEngineError(f"{type(exc).__name__}: {exc}") from exc
            raise
        logger.info(
            "C# engine: %d steps, %d records, peak %d schools, %.2f s (rng scheme %d); phases %s",
            steps, records, peak, elapsed, scheme,
            ", ".join(f"{name} {sec:.2f}s" for name, sec in phases.items()),
        )  # fmt: skip
        if status == CANCELLED:
            raise SimulationCancelled(f"cancelled at step {steps}/{engine_config.n_steps}")
        if status == TIMED_OUT:
            raise CSharpEngineTimeout(
                f"stopped at step {steps}/{engine_config.n_steps}: past the {self.deadline_seconds} s deadline"
            )
        return engine_config, grid, _step_outputs(buffers, records)

    def run(
        self,
        config: dict[str, str],
        output_dir: Path,
        seed: int = 0,
        *,
        cancel_token: threading.Event | None = None,
        step_observer: Any = None,
    ) -> RunResult:
        if step_observer is not None:
            raise CSharpEngineError(
                "step_observer receives the Python engine's school state, which the C# engine "
                "does not have: the live map is a Python-engine feature"
            )
        from osmose.engine.output import write_outputs

        engine_config, grid, outputs = self._execute(config, seed, cancel_token)
        write_outputs(outputs, output_dir, engine_config, grid=grid)
        return RunResult(returncode=0, output_dir=output_dir, stdout="", stderr="")

    def run_in_memory(
        self, config: dict[str, str], seed: int = 0, *, cancel_token: threading.Event | None = None
    ) -> OsmoseResults:
        engine_config, grid, outputs = self._execute(config, seed, cancel_token)
        return OsmoseResults.from_outputs(outputs, engine_config, grid)


def _step_outputs(buffers: Any, records: int) -> list:
    """One ``StepOutput`` per written record, as views of the result rows (spec 8.4)."""
    from osmose.engine.simulate import StepOutput

    families = buffers.families
    offsets = buffers.age_bin_offsets

    def row(name, i):
        """A family's record as the array ``_collect_outputs`` gives; None when it is off."""
        return families[name][i] if name in families else None

    def by_age(name, i):
        """``{species: its age classes}``, cut from the concatenated row."""
        if name not in families:
            return None
        r = families[name][i]
        return {sp: r[offsets[sp] : offsets[sp + 1]] for sp in range(len(offsets) - 1)}

    def by_size(name, i):
        """``{species: its length classes}``."""
        return dict(enumerate(families[name][i])) if name in families else None

    def diet(i):
        """The window's diet. The initial record has none: no mortality step ran before it, and
        ``simulate`` leaves the field None there, which the diet frames skip."""
        if "diet_by_species" not in families or buffers.record_step[i] < 0:
            return None
        return families["diet_by_species"][i]

    def present(name, i):
        """``{species: value}`` without the species absent from the record (NaN in the row)."""
        if name not in families:
            return None
        r = families[name][i]
        return {sp: float(v) for sp, v in enumerate(r) if not np.isnan(v)}

    return [
        StepOutput(
            step=int(buffers.record_step[i]),
            biomass=buffers.biomass[i],
            abundance=buffers.abundance[i],
            mortality_by_cause=buffers.mortality[i],
            yield_by_species=buffers.yield_[i],
            window_steps=int(buffers.window_steps[i]),
            abundance_by_stage=row("abundance_by_stage", i),
            mortality_by_cause_stage=row("mortality_by_cause_stage", i),
            mortality_rate_by_cause_stage=row("mortality_rate_by_cause_stage", i),
            biomass_by_age=by_age("biomass_by_age", i),
            abundance_by_age=by_age("abundance_by_age", i),
            biomass_by_size=by_size("biomass_by_size", i),
            abundance_by_size=by_size("abundance_by_size", i),
            mean_tl=present("mean_tl", i),
            yield_n=row("yield_n", i),
            mean_size=present("mean_size", i),
            ssb=row("ssb", i),
            diet_by_species=diet(i),
        )
        for i in range(records)
    ]
