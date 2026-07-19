"""Run control page - execute OSMOSE simulations."""

import logging
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

from shiny import ui, reactive, render
from shiny_deckgl import (  # type: ignore[import-untyped]
    CARTO_DARK,
    CARTO_POSITRON,
    MapWidget,
    compass_widget,
    fullscreen_widget,
    scale_widget,
    zoom_widget,
)

from osmose.config.validator import summarize_config_validation
from osmose.engine import PythonEngine, SimulationCancelled, reset_run_warnings
from osmose.engine.thread_policy import apply_single_run_threads
from osmose.engine_capabilities import describe_engine
from osmose.live_movement import (
    STAGE_LABELS,
    format_progress_label,
    make_run_observer,
    make_step_observer,
)
from osmose.logging import setup_logging
from osmose.runner import (
    OsmoseRunner,
    RunResult,
    java_engine_block_reason,
    validate_java_opts,
)
from ui.components.collapsible import body_collapse_header
from ui.pages.live_movement_render import choose_live_layer, live_legend_widget
from ui.state import get_theme_mode
from ui.styles import STYLE_CONSOLE

_log = setup_logging("osmose.run")

JAR_DIR = Path("osmose-java")


def _set_run_buttons(disabled: bool, session) -> None:
    """Toggle both Run buttons (top-of-page + the Live Movement pane) together."""
    ui.update_action_button("btn_run", disabled=disabled, session=session)
    ui.update_action_button("btn_run_live", disabled=disabled, session=session)


def _species_choices(config: dict[str, str]) -> dict[str, str]:
    """Live-movement species dropdown choices from a flat config dict (focal species)."""
    choices = {"__all__": "All species"}
    try:
        n = int(float(config.get("simulation.nspecies", 0) or 0))
    except (ValueError, TypeError):
        n = 0
    for i in range(n):
        name = config.get(f"species.name.sp{i}")
        if name:
            choices[name] = name
    return choices


def parse_overrides(text: str) -> dict[str, str]:
    """Parse a text area of key=value lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def copy_data_files(config: dict[str, str], source_dir: Path, dest_dir: Path) -> list[str]:
    """Copy ancillary data files referenced in config from source_dir to dest_dir.

    Returns list of file paths that were missing or failed to copy.
    """
    skipped: list[str] = []
    source_resolved = source_dir.resolve()
    dest_resolved = dest_dir.resolve()
    for key, value in config.items():
        if key.startswith("osmose.configuration."):
            continue
        if "/" not in value and not value.endswith(
            (".csv", ".nc", ".txt", ".dat", ".json", ".properties")
        ):
            continue
        src = (source_dir / value).resolve()
        if not src.is_relative_to(source_resolved):
            _log.warning("Skipping path traversal in config key %s: %s", key, value)
            skipped.append(value)
            continue
        if not src.exists():
            _log.warning("Referenced data file not found: %s (key: %s)", src, key)
            skipped.append(value)
            continue
        dst = (dest_dir / value).resolve()
        if not dst.is_relative_to(dest_resolved):
            _log.warning("Skipping path traversal in dest for key %s: %s", key, value)
            skipped.append(value)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.is_file():
                shutil.copy2(src, dst)
        except OSError as exc:
            _log.error("Failed to copy %s -> %s: %s", src, dst, exc)
            skipped.append(value)
    return skipped


def _inject_random_movement_ncell(config: dict[str, str]) -> dict[str, str]:
    """Return a NEW dict with movement.distribution.ncell.spN injected.

    Works around a Java engine off-by-one bug in RandomDistribution.createRandomMap()
    where the engine accesses grid cell at index == grid size when ncell is not set.

    H3 (2026-05-06): refactored from in-place mutation to pure-return. The
    previous version mutated the caller's dict, which was surprising when
    invoked on a reactive value (the mutation could be observed mid-flight
    by other reactives). Callers now do `config = _inject_random_movement_ncell(config)`.
    """
    try:
        nlon = int(config.get("grid.nlon", "0"))
        nlat = int(config.get("grid.nlat", "0"))
    except ValueError:
        _log.warning(
            "Cannot inject random movement ncell: grid dimensions invalid (nlon=%r, nlat=%r)",
            config.get("grid.nlon"),
            config.get("grid.nlat"),
        )
        return dict(config)
    if nlon <= 0 or nlat <= 0:
        _log.warning(
            "Cannot inject random movement ncell: grid dimensions non-positive (nlon=%d, nlat=%d)",
            nlon,
            nlat,
        )
        return dict(config)
    total_cells = nlon * nlat
    out = dict(config)
    for key, value in config.items():
        if key.startswith("movement.distribution.method.sp") and value.strip() == "random":
            sp_suffix = key.split("movement.distribution.method.")[-1]
            ncell_key = f"movement.distribution.ncell.{sp_suffix}"
            if ncell_key not in out:
                out[ncell_key] = str(total_cells)
    return out


def write_temp_config(
    config: dict[str, str],
    output_dir: Path,
    source_dir: Path | None = None,
    key_case_map: dict[str, str] | None = None,
    target_version: str = "4.3.3",  # bare default 4.3.3 (string-faithful); the run path passes
    # target_version_for_jar(jar_path) -> 4.4.1 (C1)
) -> Path:
    """Write config to a directory, copy data files, and return the master file path.

    If source_dir is provided, copies the entire directory tree first so that
    all ancillary files (NetCDF grids, movement maps, etc.) are available to
    the Java engine.  Then writes a single flat master config containing ALL
    parameters — without ``osmose.configuration.*`` sub-file references — so
    the Java engine reads only this one file and ignores any copied sub-configs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_dir and source_dir.is_dir():
        shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    # Work around Java off-by-one bug in RandomDistribution.createRandomMap():
    # when movement.distribution.ncell.spN is absent and method is "random",
    # the engine tries to access grid cell at index == grid size (out of bounds).
    # Auto-inject ncell = nlon * nlat - 1 for any species using random movement.
    # H3: now returns a new dict (was in-place mutation).
    config = _inject_random_movement_ncell(config)

    # Reverse-map canonical 4.4.0 keys to the target engine's spellings
    # (default 4.3.3, the bundled jar). The injected movement.* key is not
    # renamed, so it passes through untouched.
    from osmose.config.aliases import to_target_keys

    config = to_target_keys(config, target_version=target_version)

    # Write a single flat master file with all params, stripping sub-config
    # references to avoid the Java engine loading duplicate parameters from
    # both the master and the copied sub-config files.
    # Restore original key case so Java's case-sensitive parser works.
    case_map = key_case_map or {}
    master = output_dir / "osm_all-parameters.csv"
    lines = []
    for key, value in sorted(config.items()):
        if key.startswith(("osmose.configuration.", "_")):
            continue
        original_key = case_map.get(key, key)
        lines.append(f"{original_key} ; {value}\n")
    master.write_text("".join(lines))
    return master


def run_ui():
    live_map = MapWidget("live_map", style=CARTO_POSITRON)
    return ui.div(
        ui.layout_columns(
            # Left: Run controls with engine tabs
            ui.card(
                body_collapse_header("Run Configuration", "run_config"),
                ui.output_ui("engine_indicator"),
                ui.panel_conditional(
                    "input.engine_mode !== 'python'",
                    ui.output_ui("jar_selector"),
                    ui.input_text(
                        "java_opts",
                        "Java options",
                        value="-Xmx2g",
                        placeholder="-Xmx4g -Xms1g",
                    ),
                    ui.input_numeric(
                        "run_timeout", "Timeout (seconds)", value=3600, min=60, max=86400
                    ),
                    ui.input_text_area(
                        "param_overrides",
                        "Parameter overrides (key=value, one per line)",
                        rows=4,
                    ),
                ),
                ui.panel_conditional(
                    "input.engine_mode === 'python'",
                    ui.input_numeric(
                        "py_threads",
                        "Threads (Numba; 0 = auto — physical cores)",
                        value=0,
                        min=0,
                        max=32,
                    ),
                    ui.input_text_area(
                        "py_param_overrides",
                        "Parameter overrides (key=value, one per line)",
                        rows=4,
                    ),
                ),
                ui.output_ui("engine_capability"),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run", "Start Run", class_="btn-success btn-lg w-100"
                    ),
                    ui.input_action_button(
                        "btn_cancel", "Cancel", class_="btn-danger btn-lg w-100"
                    ),
                    col_widths=[6, 6],
                ),
                ui.hr(),
                ui.h5("Run Status"),
                ui.output_ui("run_progress"),
                ui.output_text("run_status"),
            ),
            # Right: Console output
            ui.card(
                body_collapse_header("Console Output", "run_console"),
                ui.output_ui("run_console"),
            ),
            col_widths=[4, 8],
        ),
        ui.card(
            body_collapse_header(
                "Live Movement (Python engine) — expand to stream during a run",
                "run_live_movement",
            ),
            ui.layout_columns(
                ui.input_action_button("btn_run_live", "▶ Run", class_="btn-success"),
                ui.input_radio_buttons(
                    "live_movement_mode",
                    "Mode",
                    {"heatmap": "Heatmap", "dots": "Dots"},
                    selected="heatmap",
                    inline=True,
                ),
                ui.input_select(
                    "live_movement_species", "Species", choices={"__all__": "All species"}
                ),
                ui.input_select(
                    "live_movement_stage",
                    "Stage",
                    choices={
                        "__all__": "All stages",
                        **{str(k): v for k, v in STAGE_LABELS.items()},
                    },
                ),
                col_widths=[2, 2, 4, 4],
            ),
            ui.output_ui("live_movement_status"),
            live_map.ui(height="420px"),
        ),
        class_="osm-run-root",
        id="run_page",
    )


class _QueueLogHandler(logging.Handler):
    """Bridge osmose WARNING+ logs from ONE run's thread into that run's console queue (live, like
    the Java jar console). Thread-filtered so a concurrent session's run cannot leak in."""

    def __init__(self, log_q: "queue.Queue", thread_id: int, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._log_q = log_q
        self._thread_id = thread_id  # only records emitted on THIS run's thread are forwarded
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread_id:
            return  # a different session's run (shared global 'osmose' logger) — not ours
        try:
            self._log_q.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001 — a log line must never break a run
            self.handleError(record)


def _python_engine_thread(
    run_config, output_dir, cancel_token, step_observer, done_q, n_threads=0, log_q=None
):
    """Run the Python engine in a background thread; post the outcome to ``done_q``.

    Fire-and-forget (the calibration-dashboard pattern): runs OFF the main thread so the event
    handler that launched it returns immediately, letting the reactive poll flush live movement
    frames AND run_log/status during the run.

    When ``log_q`` is given, a thread-filtered ``_QueueLogHandler`` is attached to the ``osmose``
    logger for the run's duration, so the engine's WARNING+ logs (the #120/#123 warnings) stream
    live into the run console via ``_drain_run_log`` — mirroring the Java jar-console stream.

    Touches NO reactive state. Posts ``(kind, result_or_None, message)``.
    """
    apply_single_run_threads(n_threads)
    osmose_logger = logging.getLogger("osmose")
    handler = None
    try:
        if log_q is not None:
            handler = _QueueLogHandler(log_q, threading.get_ident())
            osmose_logger.addHandler(handler)  # first, so the finally always covers it
        engine = PythonEngine()
        result = engine.run(
            run_config,
            output_dir,
            seed=0,
            cancel_token=cancel_token,
            step_observer=step_observer,
        )
        done_q.put(("done", result, ""))
    except SimulationCancelled as exc:
        _log.info("Python engine cancelled: %s", exc)
        done_q.put(("cancelled", None, str(exc) or "user cancelled"))
    except Exception as exc:  # noqa: BLE001
        _log.error("Python engine failed: %s", exc, exc_info=True)
        done_q.put(("failed", None, str(exc)))
    finally:
        if handler is not None:
            osmose_logger.removeHandler(handler)


def _java_engine_setup(input, state, config, work_dir, source_dir):
    """Sync setup for a Java run: jar check, write + stage the config, build the runner.

    Returns a params dict for the run thread, or an error STRING on a known failure (jar missing,
    background-staging failure, bad java opts). No reactive side effects, so handle_run can call it
    once on the main thread before launching the off-thread run.
    """
    jar_path = Path(state.jar_path.get())
    if not jar_path.exists():
        return f"Error: JAR not found at {jar_path}"

    from osmose.config.aliases import _numeric_version, target_version_for_jar

    config_path = write_temp_config(
        config,
        work_dir,
        source_dir,
        key_case_map=state.key_case_map.get(),
        target_version=target_version_for_jar(jar_path),  # write keys matching the selected jar
    )
    overrides = parse_overrides(input.param_overrides() or "")
    # C2: stage background species for a >=4.4.0 jar (e.g. Baltic). The run gate already confirmed
    # the config is staging-supported; merge the staging's -P overrides (cutoff workaround).
    try:
        _n_bg = int(float(config.get("simulation.nbackground", 0) or 0))
    except (TypeError, ValueError):
        _n_bg = 0
    if _n_bg > 0 and _numeric_version(target_version_for_jar(jar_path)) >= _numeric_version(
        "4.4.0"
    ):
        from osmose.java_background_staging import stage_background_for_java

        try:
            overrides = {**overrides, **stage_background_for_java(config_path.parent, config)}
        except Exception as exc:  # noqa: BLE001 — surface staging failures, never silently
            _log.error("Java background staging failed", exc_info=True)
            return f"Background staging failed: {exc}"
    java_opts_text = input.java_opts() or ""
    java_opts = java_opts_text.split() if java_opts_text.strip() else []
    try:
        validate_java_opts(java_opts)  # type: ignore[arg-type]
    except ValueError as exc:
        return f"Error: {exc}"
    return {
        "runner": OsmoseRunner(jar_path=jar_path),
        "config_path": config_path,
        "output_dir": work_dir / "output",
        "java_opts": java_opts or None,
        "overrides": overrides,
        "timeout_sec": input.run_timeout(),
    }


def _java_engine_thread(
    runner, config_path, output_dir, java_opts, overrides, timeout_sec, log_q, done_q
):
    """Run the Java jar OFF the main thread; stream output lines to ``log_q``, post the outcome to
    ``done_q`` as ``(kind, result_or_None, message)``.

    Fire-and-forget (mirrors ``_python_engine_thread``): handle_run launches this and returns, so
    Shiny keeps flushing — ``_drain_run_log`` streams the jar console live and ``_drain_run_done``
    finishes the run. Touches NO reactive state.
    """
    import asyncio

    def on_progress(line: str) -> None:
        try:
            log_q.put_nowait(line)
        except queue.Full:
            pass

    try:
        result = asyncio.run(
            runner.run(
                config_path=config_path,
                output_dir=output_dir,
                java_opts=java_opts,
                overrides=overrides,
                on_progress=on_progress,
                timeout_sec=timeout_sec,
            )
        )
        done_q.put(("done", result, ""))  # _handle_result handles returncode 0 or non-zero
    except Exception as exc:  # noqa: BLE001
        _log.error("Java engine failed: %s", exc, exc_info=True)
        done_q.put(("failed", None, str(exc)))


def _handle_result(result, config, state, run_log, status, start_monotonic=None):
    """Process a RunResult from either engine.

    Pre-C4, state.output_dir was set unconditionally; on a failed or
    cancelled run, the Results page would then auto-load from a partial /
    nonexistent directory and surface stale or broken data. C4 (Phase A,
    2026-05-05) gates state.output_dir.set on returncode == 0 and clears
    it on failure or cancellation, so downstream reactives (notably
    _auto_load_results) re-fire with a None signal they can short-circuit on.
    """
    state.run_result.set(result)

    if result.returncode == 0:
        state.output_dir.set(result.output_dir)
        status.set(f"Complete. Output: {result.output_dir}")
        try:
            from osmose.history import RunRecord, default_run_history

            history = default_run_history()
            duration_sec = (
                max(0.0, time.monotonic() - start_monotonic) if start_monotonic is not None else 0.0
            )
            record = RunRecord(
                config_snapshot=config,
                duration_sec=duration_sec,
                output_dir=str(result.output_dir),
                summary={},
            )
            history.save(record)
        except (OSError, ValueError) as exc:
            _log.warning("Failed to save run history: %s", exc)
        return

    # Failure or cancellation — invalidate the output dir so dependent
    # reactives (Results page _auto_load_results) short-circuit instead of
    # loading a partial / missing directory.
    state.output_dir.set(None)
    if result.status == "cancelled":
        status.set(f"Cancelled: {result.message or 'user cancelled'}")
    else:
        status.set(f"Failed (exit code {result.returncode})")
    # Surface BOTH streams on failure: the Java engine writes its `osmose[severe] ...` errors to
    # STDOUT and the stacktrace to STDERR, so showing only stderr can read as "no output".
    extra: list[str] = []
    if result.stdout:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        extra.append(f"--- OUTPUT (last 40 lines) ---\n{tail}")
    if result.stderr:
        extra.append(f"--- STDERR ---\n{result.stderr}")
    if extra:
        run_log.set(list(run_log.get()) + extra)


def run_server(input, output, session, state):
    run_log = reactive.value([])
    status = reactive.value("Idle")
    runner_ref = reactive.value(None)

    # ── Live movement state (Python engine only) ─────────────────
    _live_map = MapWidget("live_map", style=CARTO_POSITRON)
    _live_queue: queue.Queue = queue.Queue(maxsize=4)
    _live_snapshot: reactive.Value = reactive.Value(None)  # MovementSnapshot | None
    _live_status_val: reactive.Value = reactive.Value(
        ""
    )  # "" | running | done | cancelled | failed
    _live_framed = [False]  # plain mutable flag (NOT reactive — render effect reads+writes it)
    _live_layer_id: list[str | None] = [None]  # last rendered deck.gl layer id (heatmap vs dots)
    _live_widget_sig: list[tuple | None] = [None]  # (layer_id, species_filter, stage_filter)
    _last_live_species: list[list[str] | None] = [
        None
    ]  # plain flag for the species-selector changed-only guard
    _live_note: reactive.Value = reactive.Value(None)  # heatmap-fallback note | None

    # ── Run completion (fire-and-forget thread → main-thread poll), both engines ──
    _run_done_q: queue.Queue = queue.Queue(maxsize=1)  # (kind, result|None, message)
    _run_config_cell: list = [None]  # config captured at run start, for _handle_result
    _run_start_cell: list = [None]  # run start (time.monotonic) for duration_sec

    # Java console lines streamed off-thread → drained to run_log by _drain_run_log (live console).
    _run_log_q: queue.Queue = queue.Queue()

    _progress_q: queue.Queue = queue.Queue(maxsize=1)  # (done, n_steps, elapsed_s)
    _progress: reactive.Value = reactive.Value(None)  # None | (done, n_steps, elapsed_s)

    # ── Session-teardown hardening ───────────────────────────────
    # When the browser tab/session dies, Shiny tears down the session; reactive
    # consumers that still touch it raise DestroyedReactiveError / CancelledError,
    # cascading. We flip _session_alive on session end and cancel the running
    # engine thread, then guard each consumer below.
    _session_alive = [True]
    _active_cancel_token: list = [None]  # plain ref so on_ended needn't read a reactive

    def _on_session_end():
        _session_alive[0] = False
        tok = _active_cancel_token[0]
        if tok is not None:
            tok.set()  # stop the daemon engine thread instead of running against a dead session

    session.on_ended(_on_session_end)

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_run_done():
        if not _session_alive[0]:
            return
        try:
            try:
                kind, result, msg = _run_done_q.get_nowait()
            except queue.Empty:
                return
            lines = list(run_log.get())
            if kind == "cancelled":
                lines.append(f"--- CANCELLED ---\n{msg}")
                _live_status_val.set("cancelled")
                result = RunResult(
                    returncode=-1,
                    output_dir=Path(""),
                    stdout="",
                    stderr=msg,
                    status="cancelled",
                    message=msg or "user cancelled",
                )
            elif kind == "failed":
                lines.append(f"--- ERROR ---\n{msg}")
                _live_status_val.set("failed")
                result = RunResult(
                    returncode=1,
                    output_dir=Path(""),
                    stdout="",
                    stderr=msg,
                    status="failed",
                    message=msg,
                )
            else:  # done
                _live_status_val.set("done")
            run_log.set(lines)
            state.busy.set(None)
            _set_run_buttons(False, session)
            ui.update_action_button("btn_cancel", disabled=True, session=session)
            _handle_result(result, _run_config_cell[0], state, run_log, status, _run_start_cell[0])
            while True:
                try:
                    _progress_q.get_nowait()
                except queue.Empty:
                    break
            _progress.set(None)
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("run-done poll skipped (session ending)", exc_info=True)

    @reactive.effect
    def _consume_run_done():
        _drain_run_done()

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_run_log():
        """Stream a run's console lines (Java jar console, or Python engine WARNING+ logs posted
        by _QueueLogHandler) from _run_log_q into run_log on the main thread."""
        if not _session_alive[0]:
            return
        new_lines: list[str] = []
        while True:
            try:
                new_lines.append(_run_log_q.get_nowait())
            except queue.Empty:
                break
        if new_lines:
            lines = list(run_log.get()) + new_lines
            if len(lines) > 500:
                lines = lines[-500:]
            run_log.set(lines)

    @reactive.effect
    def _consume_run_log():
        _drain_run_log()

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_live_queue():
        if not _session_alive[0]:
            return
        try:
            latest = None
            while True:
                try:
                    latest = _live_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                _live_snapshot.set(latest)
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("live poll skipped (session ending)", exc_info=True)

    @reactive.effect
    def _consume_live_poll():
        _drain_live_queue()

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_progress():
        if not _session_alive[0]:
            return
        try:
            latest = None
            while True:
                try:
                    latest = _progress_q.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                _progress.set(latest)
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("progress poll skipped (session ending)", exc_info=True)

    @reactive.effect
    def _consume_progress():
        _drain_progress()

    @reactive.effect
    def _populate_live_species():
        if not _session_alive[0]:
            return
        try:
            snap = _live_snapshot.get()
            if snap is None:
                return
            if snap.species == _last_live_species[0]:
                return
            _last_live_species[0] = list(snap.species)
            choices = {"__all__": "All species"}
            choices.update({name: name for name in snap.species})
            ui.update_select("live_movement_species", choices=choices)
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("species populate skipped (session ending)", exc_info=True)

    @reactive.effect
    def _populate_species_from_config():
        if not _session_alive[0]:
            return
        ui.update_select("live_movement_species", choices=_species_choices(state.config.get()))

    @render.ui
    def live_movement_status():
        status_v = _live_status_val.get()
        snap = _live_snapshot.get()
        if not status_v:
            if state.engine_mode.get() != "python":
                return ui.p("Live view available for the Python engine.", class_="text-muted")
            return ui.div()
        prog = f"step {snap.step + 1}/{snap.n_steps}" if snap is not None else ""
        extra = ""
        if snap is not None and snap.truncated:
            extra = f" — showing {snap.sp_id.size} of {snap.n_total} schools"
        note = _live_note.get()
        suffix = f" · {note}" if note else ""
        date = f" · {snap.date_label}" if snap is not None and snap.date_label else ""
        return ui.p(f"{status_v}{date} {prog}{extra}{suffix}".strip())

    @reactive.effect
    async def _render_live_map():
        if not _session_alive[0]:
            return
        try:
            if not input.live_view_expanded():
                return
        except Exception:
            return  # input unset -> collapsed -> nothing to render
        try:
            snap = _live_snapshot.get()
            mode = input.live_movement_mode()
            sel = input.live_movement_species()
            species_filter = None if sel in ("__all__", None) else sel
            stage_sel = input.live_movement_stage()
            stage_filter = None if stage_sel in ("__all__", None) else int(stage_sel)
            style = CARTO_DARK if get_theme_mode(input) == "dark" else CARTO_POSITRON
            if style != _live_map.style:
                _live_map.style = style
                await _live_map.set_style(session, style)
            if snap is None:
                return
            layer, note = choose_live_layer(snap, species_filter, mode, stage_filter=stage_filter)
            _live_note.set(note)
            legend = live_legend_widget(snap, species_filter, stage_filter, layer["id"])
            widgets = [
                fullscreen_widget(placement="top-left"),
                zoom_widget(placement="top-right"),
                compass_widget(placement="top-right"),
                scale_widget(placement="bottom-left"),
                legend,
            ]
            sig = (layer["id"], species_filter, stage_filter)
            if not _live_framed[0]:
                await _live_map.update(
                    session,
                    layers=[layer],
                    view_state={
                        "latitude": (snap.lat_min + snap.lat_max) / 2,
                        "longitude": (snap.lon_min + snap.lon_max) / 2,
                        "zoom": 5,
                    },
                    widgets=widgets,
                )
                _live_framed[0] = True
            elif layer["id"] != _live_layer_id[0]:
                # The active representation switched (heatmap <-> dots), distinct layer ids.
                # deck.gl cannot swap a layer's class under one id; a full update (no view_state,
                # to keep the camera) removes the old id and carries the fresh legend in one message.
                await _live_map.update(session, layers=[layer], widgets=widgets)
            else:
                await _live_map.partial_update(session, layers=[layer])
                if sig != _live_widget_sig[0]:
                    # same layer id, species/stage changed -> refresh the legend only.
                    await _live_map.set_widgets(session, widgets)
            _live_layer_id[0] = layer["id"]
            _live_widget_sig[0] = sig
        except BaseException:  # noqa: BLE001
            if _session_alive[0]:
                raise  # genuine bug during a live session — surface it
            _log.debug("live map render skipped (session ending)", exc_info=True)

    @render.ui
    def jar_selector():
        jars = sorted(JAR_DIR.glob("*.jar")) if JAR_DIR.is_dir() else []
        if jars:
            choices = {str(j): j.name for j in jars}
            default = str(jars[0])
        else:
            choices = {"": "— No JAR files found in osmose-java/ —"}
            default = ""
        return ui.input_select("jar_path", "OSMOSE JAR file", choices=choices, selected=default)

    @reactive.effect
    def sync_jar_path():
        val = input.jar_path()
        if val:
            state.jar_path.set(val)

    @render.ui
    def engine_indicator():
        mode = state.engine_mode.get()
        label = "Python" if mode == "python" else "Java"
        return ui.p(
            ui.tags.strong("Active engine: "),
            label,
            ui.tags.span(" — change in the header toggle ↗", class_="text-muted"),
            class_="mb-2",
        )

    @render.ui
    def engine_capability():
        config = state.config.get()
        if not config:
            return ui.p("Load a configuration to see engine capabilities.", class_="text-muted")
        from osmose.config.aliases import target_version_for_jar

        cap = describe_engine(
            state.engine_mode.get(), config, target_version_for_jar(Path(state.jar_path.get()))
        )
        if not cap.can_run:
            return ui.div(
                ui.tags.strong("This engine can't run this configuration. "),
                cap.block_reason or "",
                class_="alert alert-warning",
            )
        populated = ", ".join(cap.pages_populated) or "—"
        empty = ", ".join(cap.pages_empty) or "—"
        return ui.div(
            ui.p(ui.tags.strong("Will populate: "), populated),
            ui.p(ui.tags.strong("Won't populate (this engine): "), empty, class_="text-muted"),
            ui.p(cap.notable_outputs, class_="small text-muted"),
        )

    @render.text
    def run_status():
        return status.get()

    @render.ui
    def run_progress():
        prog = _progress.get()
        if prog is None:
            return None
        done, n, _elapsed = prog
        try:
            ndt = int(float(state.config.get().get("simulation.time.ndtperyear", "0") or "0"))
        except (TypeError, ValueError):
            ndt = 0
        label = format_progress_label(done, n, ndt)
        pct = round(done / n * 100) if n else 0
        # The label must be a SIBLING of the .progress track, not a child: Bootstrap 5
        # .progress is display:flex; overflow:hidden, so a nested <small> gets clipped.
        return ui.div(
            ui.div(
                ui.div(
                    f"{pct}%",
                    class_="progress-bar",
                    role="progressbar",
                    style=f"width: {pct}%",
                ),
                class_="progress mb-1",
            ),
            ui.tags.small(label, class_="text-muted"),
        )

    @render.ui
    def run_console():
        lines = run_log.get()
        prog = _progress.get()
        text = "\n".join(lines[-200:]) if lines else ""
        if prog is not None:
            done, n, _elapsed = prog
            pct = round(done / n * 100) if n else 0
            prog_line = f"running · step {done}/{n} ({pct}%)"
            text = f"{text}\n{prog_line}" if text else prog_line
        if not text:
            text = "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(text, style=STYLE_CONSOLE)

    @reactive.effect
    @reactive.event(input.btn_run, input.btn_run_live)
    async def handle_run():
        _progress.set(None)
        engine_mode = state.engine_mode.get()

        # Validate config before run (common to both engines)
        config = state.config.get()

        # Engine-compatibility guard: configs with background species (e.g. Baltic's
        # GreySeal/Cormorant) are Python-engine-only — the Java reference engine crashes
        # at year 0 because their entries are missing from the (comma-separated) fishery/
        # predation matrices. Block early with a clear message instead of launching a
        # doomed Java subprocess.
        if engine_mode != "python":
            from osmose.config.aliases import target_version_for_jar

            block = java_engine_block_reason(
                config, target_version_for_jar(Path(state.jar_path.get()))
            )
            if block:
                run_log.set(["--- RUN BLOCKED (engine not supported) ---", block])
                status.set("Java engine not supported for this configuration")
                ui.notification_show(block, type="error", duration=20)
                return
        errors, warnings = summarize_config_validation(
            config, state.registry, state.config_dir.get()
        )

        if errors:
            log_lines = ["--- VALIDATION ERRORS (run blocked) ---"]
            log_lines.extend(errors)
            if warnings:
                log_lines.append("--- WARNINGS ---")
                log_lines.extend(warnings)
            run_log.set(log_lines)
            status.set(f"Validation failed: {len(errors)} error(s)")
            return

        if warnings:
            log_lines = ["--- WARNINGS (continuing anyway) ---"]
            log_lines.extend(warnings)
            run_log.set(log_lines)
        else:
            run_log.set([])

        status.set("Writing config...")
        _set_run_buttons(True, session)
        ui.update_action_button("btn_cancel", disabled=False, session=session)

        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        source_dir = state.config_dir.get()

        live_observer = None
        try:
            _live_expanded = bool(input.live_view_expanded())
        except Exception:
            _live_expanded = False  # input unset (card never toggled) -> collapsed
        if _live_expanded and engine_mode == "python":
            while True:
                try:
                    _live_queue.get_nowait()
                except queue.Empty:
                    break
            _live_snapshot.set(None)
            _live_framed[0] = False
            _live_layer_id[0] = None
            _live_widget_sig[0] = None
            _last_live_species[0] = None
            _live_status_val.set("running")
            live_observer = make_step_observer(_live_queue, throttle_s=0.5)  # ≤2 fps (was 0.2)

        run_t0 = time.monotonic()
        if engine_mode == "python":
            # Fire-and-forget: launch the engine in a background thread and RETURN, so the
            # reactive polls (_drain_live_queue + _drain_run_done) flush live frames and the
            # final result on the main thread. Awaiting the run here would suspend handle_run
            # and Shiny would defer every flush until the run finished (no live updates).
            overrides = parse_overrides(input.py_param_overrides() or "")
            run_config = dict(config)
            run_config.update(overrides)
            if source_dir:
                run_config["_osmose.config.dir"] = str(source_dir)
            output_dir = work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            cancel_token = threading.Event()
            state.run_cancel_token.set(cancel_token)
            _active_cancel_token[0] = cancel_token
            state.busy.set("Running simulation (Python)...")
            status.set("Running (Python engine)...")
            _run_config_cell[0] = config
            _run_start_cell[0] = run_t0
            run_observer = make_run_observer(_progress_q, live_observer)
            n_threads = int(input.py_threads() or 0)
            reset_run_warnings()  # per-UI-run: clear the engine warning-dedup so this run re-emits
            threading.Thread(
                target=_python_engine_thread,
                args=(
                    run_config,
                    output_dir,
                    cancel_token,
                    run_observer,
                    _run_done_q,
                    n_threads,
                    _run_log_q,
                ),
                daemon=True,
            ).start()
            # handle_run returns now; _drain_run_done finishes the run on the main thread.
        else:
            # Java: set up synchronously (jar check + stage), then run OFF the main thread so
            # handle_run returns and Shiny keeps flushing — _drain_run_log streams the jar console
            # live and _drain_run_done finishes the run (mirrors the Python fire-and-forget path).
            try:
                params = _java_engine_setup(input, state, config, work_dir, source_dir)
            except Exception as exc:  # noqa: BLE001 — surface setup/config-write errors, never silently
                import traceback

                _log.error("Java run setup failed", exc_info=True)
                status.set(f"Java run failed: {exc}")
                run_log.set(
                    list(run_log.get()) + [f"--- JAVA SETUP ERROR ---\n{traceback.format_exc()}"]
                )
                _set_run_buttons(False, session)
                ui.update_action_button("btn_cancel", disabled=True, session=session)
                return
            if isinstance(params, str):  # known setup error (jar missing / staging / bad java opts)
                status.set(params)
                run_log.set([params])
                _set_run_buttons(False, session)
                ui.update_action_button("btn_cancel", disabled=True, session=session)
                return
            runner_ref.set(params["runner"])  # type: ignore[arg-type]
            _run_config_cell[0] = config
            _run_start_cell[0] = run_t0
            state.busy.set("Running simulation (Java)...")
            status.set("Running (Java engine)...")
            run_log.set([])  # fresh console for the new run
            threading.Thread(
                target=_java_engine_thread,
                args=(
                    params["runner"],
                    params["config_path"],
                    params["output_dir"],
                    params["java_opts"],
                    params["overrides"],
                    params["timeout_sec"],
                    _run_log_q,
                    _run_done_q,
                ),
                daemon=True,
            ).start()
            # handle_run returns now; _drain_run_log streams the console + _drain_run_done finishes.

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        # C4 Phase B: cancel both engine paths.
        # 1. Java engine: signal the OsmoseRunner subprocess.
        runner = runner_ref.get()
        if runner:
            runner.cancel()
            status.set("Cancelled")
        # 2. Python engine: set the cancellation token so simulate.py's
        #    outer step loop raises SimulationCancelled on next iteration.
        token = state.run_cancel_token.get()
        if token is not None:
            token.set()
            status.set("Cancelling Python engine — finishing current step…")
