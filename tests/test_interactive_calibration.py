"""Tests for the interactive Baltic calibration launcher."""

from __future__ import annotations

import sys
from argparse import Namespace


def _args(**overrides):
    base = {
        "phase": "12",
        "optimizer": "de",
        "maxiter": 80,
        "popsize": 15,
        "popsize_mult": 10,
        "seeds": 3,
        "years": 40,
        "tol": 0.005,
        "checkpoint_every": 5,
        "patience": 20,
        "wall_clock_cap_h": 12.0,
        "warm_start": None,
        "skip_warm_start_keys": "",
        "validate": False,
        "report_seeds": 3,
        "report_years": 50,
    }
    base.update(overrides)
    return Namespace(**base)


def test_build_calibration_command_defaults():
    from scripts.interactive_calibration import CALIBRATE_SCRIPT, build_calibration_command

    cmd = build_calibration_command(_args())

    assert cmd[:2] == [sys.executable, str(CALIBRATE_SCRIPT)]
    assert "--phase" in cmd
    assert cmd[cmd.index("--phase") + 1] == "12"
    assert "--optimizer" in cmd
    assert cmd[cmd.index("--optimizer") + 1] == "de"
    assert "--validate" not in cmd


def test_build_calibration_command_optional_flags(tmp_path):
    from scripts.interactive_calibration import build_calibration_command

    warm_start = tmp_path / "prior.json"
    cmd = build_calibration_command(
        _args(
            phase="2",
            optimizer="cmaes",
            warm_start=warm_start,
            skip_warm_start_keys="mortality.additional.rate.sp0",
            validate=True,
        )
    )

    assert cmd[cmd.index("--phase") + 1] == "2"
    assert cmd[cmd.index("--optimizer") + 1] == "cmaes"
    assert cmd[cmd.index("--warm-start") + 1] == str(warm_start)
    assert cmd[cmd.index("--skip-warm-start-keys") + 1] == "mortality.additional.rate.sp0"
    assert "--validate" in cmd


def test_parse_progress_line_updates_generation_and_best():
    from scripts.interactive_calibration import ProgressState, parse_progress_line

    state = ProgressState(phase="12", optimizer="de", maxiter=80)

    parse_progress_line("differential_evolution step 17: f(x)= 1.2345", state)

    assert state.generation == 17
    assert state.best_objective == "1.2345"


def test_parse_progress_line_updates_final_metrics():
    from scripts.interactive_calibration import ProgressState, parse_progress_line

    state = ProgressState(phase="12", optimizer="de", maxiter=80)

    parse_progress_line("Function evaluations: 1234", state)
    parse_progress_line("Best objective (single-seed): 0.9876", state)

    assert state.nfev == 1234
    assert state.best_objective == "0.9876"


def test_render_progress_contains_key_fields():
    from scripts.interactive_calibration import ProgressState, render_progress

    state = ProgressState(phase="12", optimizer="de", maxiter=100)
    state.generation = 25
    state.best_objective = "3.14"
    state.nfev = 500

    rendered = render_progress(state, width=10)

    assert "25.0%" in rendered
    assert "gen 25/100" in rendered
    assert "best=3.14" in rendered
    assert "nfev=500" in rendered


def test_render_progress_indeterminate_for_cmaes():
    from scripts.interactive_calibration import ProgressState, render_progress

    state = ProgressState(phase="12", optimizer="cmaes", maxiter=80)
    state.nfev = 42

    rendered = render_progress(state)

    assert "running" in rendered
    assert "opt=cmaes" in rendered
    assert "0.0%" not in rendered
    assert "gen 0/80" not in rendered


def test_render_progress_indeterminate_for_validate():
    from scripts.interactive_calibration import ProgressState, render_progress

    state = ProgressState(phase="12", optimizer="de", maxiter=80, validate=True)

    rendered = render_progress(state)

    assert "validate" in rendered
    assert "0.0%" not in rendered


def test_parse_args_accepts_unknown_phase_string():
    from scripts.interactive_calibration import parse_args

    args = parse_args(["--phase", "experimental-3"])

    assert args.phase == "experimental-3"


def test_main_dry_run_prints_env(capsys):
    from scripts.interactive_calibration import main

    rc = main(["--phase", "12", "--dry-run", "--yes", "--workers", "4"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "OSMOSE_DE_WORKERS=4" in out
    assert "calibrate_baltic.py" in out


def test_run_with_progress_streams_and_exits_zero(tmp_path):
    """Smoke test: run a tiny child that emits a fake DE step line."""
    import argparse
    import sys as _sys

    from scripts.interactive_calibration import run_with_progress

    rc = run_with_progress(
        [
            _sys.executable,
            "-c",
            "import sys; print('differential_evolution step 1: f(x)= 0.5'); "
            "print('Function evaluations: 7'); "
            "print('Best objective (single-seed): 0.5'); sys.exit(0)",
        ],
        argparse.Namespace(
            phase="12",
            optimizer="de",
            maxiter=10,
            workers=1,
            show_log=True,
            validate=False,
        ),
    )

    assert rc == 0
