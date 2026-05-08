#!/usr/bin/env python3
"""Interactive Baltic calibration launcher with live progress display.

This is a terminal wrapper around ``scripts/calibrate_baltic.py``. It does not
change the optimizer; it builds a calibration command, optionally prompts for
common settings, and parses the child process output to show an updating status
line while preserving the underlying calibration log.

Examples:
    .venv/bin/python scripts/interactive_calibration.py
    .venv/bin/python scripts/interactive_calibration.py --phase 12 --maxiter 80 --yes
    .venv/bin/python scripts/interactive_calibration.py --phase 12 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

CLEAR_EOL = "\033[K"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATE_SCRIPT = PROJECT_ROOT / "scripts" / "calibrate_baltic.py"
REPORT_SCRIPT = PROJECT_ROOT / "scripts" / "report_calibration.py"

PHASE_CHOICES = ("1", "1b", "1c", "1d", "1e", "1f", "1g", "2", "12")
OPTIMIZER_CHOICES = ("de", "cmaes", "surrogate-de")
STEP_RE = re.compile(r"differential_evolution step\s+(\d+):\s+f\(x\)=\s+([^\s]+)")
BEST_RE = re.compile(r"Best objective \(single-seed\):\s+([^\s]+)")
NFEV_RE = re.compile(r"Function evaluations:\s+(\d+)")


@dataclass
class ProgressState:
    """Calibration progress parsed from the child process output."""

    phase: str
    optimizer: str
    maxiter: int
    validate: bool = False
    started_at: float = field(default_factory=time.time)
    generation: int = 0
    best_objective: str | None = None
    nfev: int | None = None
    last_line: str = ""
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=6))

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def fraction(self) -> float:
        if self.maxiter <= 0:
            return 0.0
        return min(1.0, self.generation / self.maxiter)

    @property
    def has_de_progress(self) -> bool:
        return not self.validate and self.optimizer == "de"


def parse_progress_line(line: str, state: ProgressState) -> None:
    """Update progress state from one calibration output line."""
    state.last_line = line.rstrip()
    if state.last_line:
        state.tail.append(state.last_line)

    step_match = STEP_RE.search(line)
    if step_match:
        state.generation = max(state.generation, int(step_match.group(1)))
        state.best_objective = step_match.group(2)
        return

    best_match = BEST_RE.search(line)
    if best_match:
        state.best_objective = best_match.group(1)
        return

    nfev_match = NFEV_RE.search(line)
    if nfev_match:
        state.nfev = int(nfev_match.group(1))


def _fmt_elapsed(seconds: float) -> str:
    seconds_i = int(seconds)
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def render_progress(state: ProgressState, *, width: int = 30, final: bool = False) -> str:
    """Return a single-line progress summary.

    For DE the line includes a generation-based progress bar. For CMA-ES,
    surrogate-DE, and ``--validate`` runs the optimizer does not emit
    generation markers, so we render an indeterminate status instead of a
    misleading 0% bar. Pass ``final=True`` for end-of-run summaries to swap
    the indeterminate ``[running]`` prefix for ``[done]``.
    """
    best = state.best_objective or "?"
    nfev = "?" if state.nfev is None else str(state.nfev)
    elapsed = _fmt_elapsed(state.elapsed)
    if state.has_de_progress:
        filled = int(round(width * state.fraction))
        bar = "#" * filled + "-" * (width - filled)
        pct = 100.0 * state.fraction
        return (
            f"[{bar}] {pct:5.1f}% "
            f"gen {state.generation}/{state.maxiter} "
            f"best={best} nfev={nfev} elapsed={elapsed} "
            f"phase={state.phase} opt={state.optimizer}"
        )
    mode = "validate" if state.validate else f"opt={state.optimizer}"
    prefix = "[done]" if final else "[running]"
    return f"{prefix} {mode} phase={state.phase} best={best} nfev={nfev} elapsed={elapsed}"


def _prompt_str(label: str, current: str, choices: tuple[str, ...] | None = None) -> str:
    suffix = f" ({'/'.join(choices)})" if choices else ""
    raw = input(f"{label}{suffix} [{current}]: ").strip()
    if not raw:
        return current
    if choices and raw not in choices:
        print(f"  Invalid choice {raw!r}; keeping {current!r}.")
        return current
    return raw


def _prompt_int(label: str, current: int, minimum: int = 0) -> int:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    try:
        value = int(raw)
    except ValueError:
        print(f"  Invalid integer {raw!r}; keeping {current}.")
        return current
    if value < minimum:
        print(f"  Value must be >= {minimum}; keeping {current}.")
        return current
    return value


def _prompt_float(label: str, current: float, minimum: float = 0.0) -> float:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    try:
        value = float(raw)
    except ValueError:
        print(f"  Invalid number {raw!r}; keeping {current}.")
        return current
    if value < minimum:
        print(f"  Value must be >= {minimum}; keeping {current}.")
        return current
    return value


def _prompt_phase(label: str, current: str) -> str:
    """Prompt for a phase string. Suggested values are shown but not enforced."""
    suggested = "/".join(PHASE_CHOICES)
    raw = input(f"{label} (suggested: {suggested}) [{current}]: ").strip()
    return raw or current


def maybe_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for common options when attached to an interactive terminal."""
    if args.yes or not sys.stdin.isatty():
        return args

    print("\nBaltic interactive calibration")
    print("Press Enter to accept defaults; pass --yes to skip prompts.\n")
    args.phase = _prompt_phase("Phase", args.phase)
    args.optimizer = _prompt_str("Optimizer", args.optimizer, OPTIMIZER_CHOICES)
    args.maxiter = _prompt_int("Max optimizer iterations", args.maxiter, minimum=1)
    args.years = _prompt_int("Simulation years per evaluation", args.years, minimum=1)
    args.seeds = _prompt_int("Validation seeds after optimization", args.seeds, minimum=1)
    args.workers = _prompt_int("Parallel workers", args.workers, minimum=1)
    args.patience = _prompt_int("DE stale-generation patience", args.patience, minimum=0)
    args.wall_clock_cap_h = _prompt_float("DE wall-clock cap hours", args.wall_clock_cap_h)

    report_default = "y" if args.report else "n"
    report_raw = input(
        f"Run report after successful calibration? [y/N, current {report_default}]: "
    )
    if report_raw.strip().lower() in {"y", "yes"}:
        args.report = True
    elif report_raw.strip().lower() in {"n", "no"}:
        args.report = False

    return args


def build_calibration_command(args: argparse.Namespace) -> list[str]:
    """Build the child command for calibrate_baltic.py."""
    cmd = [
        sys.executable,
        str(CALIBRATE_SCRIPT),
        "--phase",
        args.phase,
        "--maxiter",
        str(args.maxiter),
        "--popsize",
        str(args.popsize),
        "--popsize-mult",
        str(args.popsize_mult),
        "--seeds",
        str(args.seeds),
        "--years",
        str(args.years),
        "--tol",
        str(args.tol),
        "--optimizer",
        args.optimizer,
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--patience",
        str(args.patience),
        "--wall-clock-cap-h",
        str(args.wall_clock_cap_h),
    ]
    if args.warm_start:
        cmd.extend(["--warm-start", str(args.warm_start)])
    if args.skip_warm_start_keys:
        cmd.extend(["--skip-warm-start-keys", args.skip_warm_start_keys])
    if args.validate:
        cmd.append("--validate")
    return cmd


def build_report_command(args: argparse.Namespace) -> list[str]:
    """Build the post-calibration report command."""
    return [
        sys.executable,
        str(REPORT_SCRIPT),
        "--phase",
        args.phase,
        "--seeds",
        str(args.report_seeds),
        "--years",
        str(args.report_years),
    ]


def _print_command(title: str, cmd: list[str], env_extra: dict[str, str] | None = None) -> None:
    print(f"{title}:")
    if env_extra:
        env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_extra.items())
        print(f"  env: {env_str}")
    print("  " + " ".join(shlex.quote(part) for part in cmd))


def _terminate_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the child's process group, ignoring missing-process errors."""
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def run_with_progress(cmd: list[str], args: argparse.Namespace) -> int:
    """Run a child process while displaying parsed progress."""
    env = os.environ.copy()
    env["OSMOSE_DE_WORKERS"] = str(args.workers)
    env.setdefault("PYTHONUNBUFFERED", "1")

    state = ProgressState(
        phase=args.phase,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        validate=args.validate,
    )
    is_tty = sys.stdout.isatty()
    show_log = bool(args.show_log) or not is_tty
    print("\nStarting calibration. Press Ctrl-C once to stop the child process.\n")
    _print_command("Command", cmd, env_extra={"OSMOSE_DE_WORKERS": env["OSMOSE_DE_WORKERS"]})

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    rc: int = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            parse_progress_line(line, state)
            if show_log:
                if is_tty:
                    print("\r" + CLEAR_EOL + line, end="")
                else:
                    print(line, end="")
            if is_tty:
                print("\r" + CLEAR_EOL + render_progress(state), end="", flush=True)
        rc = process.wait()
    except KeyboardInterrupt:
        print("\nStopping calibration child process group...")
        _terminate_group(process, signal.SIGTERM)
        try:
            rc = process.wait(timeout=30)
        except KeyboardInterrupt:
            print("Second interrupt received; sending SIGKILL...")
            _terminate_group(process, signal.SIGKILL)
            rc = process.wait()
        except subprocess.TimeoutExpired:
            _terminate_group(process, signal.SIGKILL)
            rc = process.wait()

    if is_tty:
        print("\r" + CLEAR_EOL, end="")
        print()
    print("Final status: " + render_progress(state, final=True))
    if state.tail:
        print("\nRecent calibration output:")
        for item in state.tail:
            print(f"  {item}")
    return int(rc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="12",
        help=f"Calibration phase (e.g. {', '.join(PHASE_CHOICES)}). Forwarded as-is to "
        "calibrate_baltic.py, which validates the phase value.",
    )
    parser.add_argument("--optimizer", default="de", choices=OPTIMIZER_CHOICES)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--popsize-mult", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--years", type=int, default=40)
    parser.add_argument("--tol", type=float, default=0.005)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("OSMOSE_DE_WORKERS", "8"))
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--wall-clock-cap-h", type=float, default=12.0)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--skip-warm-start-keys", default="")
    parser.add_argument(
        "--validate", action="store_true", help="Run calibrate_baltic.py --validate"
    )
    parser.add_argument(
        "--report", action="store_true", help="Run report_calibration.py after success"
    )
    parser.add_argument("--report-seeds", type=int, default=3)
    parser.add_argument("--report-years", type=int, default=50)
    parser.add_argument(
        "--show-log", action="store_true", help="Print full child log while updating status"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running them")
    parser.add_argument("--yes", action="store_true", help="Skip interactive prompts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = maybe_prompt(parse_args(argv))
    cmd = build_calibration_command(args)
    env_extra = {"OSMOSE_DE_WORKERS": str(args.workers)}

    if args.dry_run:
        _print_command("Calibration command", cmd, env_extra=env_extra)
        if args.report:
            _print_command("Report command", build_report_command(args))
        return 0

    rc = run_with_progress(cmd, args)
    if rc != 0:
        print(f"\nCalibration failed with exit code {rc}.")
        return rc

    if args.report:
        report_cmd = build_report_command(args)
        print("\nCalibration completed; running report...")
        _print_command("Report command", report_cmd)
        return subprocess.call(report_cmd, cwd=PROJECT_ROOT)

    print("\nCalibration completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
