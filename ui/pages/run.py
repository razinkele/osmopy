"""Run control page - execute OSMOSE simulations."""

import asyncio
import shutil
import tempfile
from pathlib import Path

from shiny import ui, reactive, render

from osmose.config.validator import (
    check_file_references,
    check_species_consistency,
    validate_config,
)
from osmose.engine import PythonEngine, SimulationCancelled
from osmose.logging import setup_logging
from osmose.runner import OsmoseRunner, RunResult, validate_java_opts
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_CONSOLE

_log = setup_logging("osmose.run")

JAR_DIR = Path("osmose-java")


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
    return ui.div(
        expand_tab("Run Configuration", "run"),
        ui.layout_columns(
            # Left: Run controls with engine tabs
            ui.card(
                collapsible_card_header("Run Configuration", "run"),
                ui.navset_tab(
                    ui.nav_panel(
                        "Java",
                        ui.output_ui("jar_selector"),
                        ui.input_text(
                            "java_opts",
                            "Java options",
                            value="-Xmx2g",
                            placeholder="-Xmx4g -Xms1g",
                        ),
                        ui.input_numeric(
                            "run_timeout",
                            "Timeout (seconds)",
                            value=3600,
                            min=60,
                            max=86400,
                        ),
                        ui.input_text_area(
                            "param_overrides",
                            "Parameter overrides (key=value, one per line)",
                            rows=4,
                        ),
                        value="run_java_tab",
                    ),
                    ui.nav_panel(
                        "Python",
                        ui.input_numeric(
                            "py_threads",
                            "Threads (Numba prange)",
                            value=1,
                            min=1,
                            max=32,
                        ),
                        ui.input_select(
                            "py_verbosity",
                            "Verbosity",
                            choices={"0": "Quiet", "1": "Normal", "2": "Verbose"},
                            selected="1",
                        ),
                        ui.input_text_area(
                            "py_param_overrides",
                            "Parameter overrides (key=value, one per line)",
                            rows=4,
                        ),
                        value="run_python_tab",
                    ),
                    id="run_engine_tabs",
                ),
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
                ui.output_text("run_status"),
            ),
            # Right: Console output
            ui.card(
                ui.card_header("Console Output"),
                ui.output_ui("run_console"),
            ),
            col_widths=[4, 8],
        ),
        class_="osm-split-layout",
        id="split_run",
    )


async def _run_python_engine(
    input,
    state,
    session,
    config,
    work_dir,
    source_dir,
    run_log,
    status,
    runner_ref,
):
    """Run the simulation using the in-process Python engine."""
    overrides = parse_overrides(input.py_param_overrides() or "")
    run_config = dict(config)
    run_config.update(overrides)

    # Store config_dir so PythonEngine can find grid NetCDF files
    if source_dir:
        run_config["_osmose.config.dir"] = str(source_dir)

    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = PythonEngine()
    status.set("Running (Python engine)...")

    # C4 Phase B: fresh cancellation token per run. The cancel button
    # handler (handle_cancel below) sets this to interrupt the simulation.
    import threading as _threading
    cancel_token = _threading.Event()
    state.run_cancel_token.set(cancel_token)

    state.busy.set("Running simulation (Python)...")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.run(run_config, output_dir, seed=0, cancel_token=cancel_token),
        )
    except SimulationCancelled as exc:
        # Phase B will trigger this from a UI Cancel button; Phase A defines
        # the path so the result-handling code is uniform across exit modes.
        _log.info("Python engine cancelled: %s", exc)
        lines = list(run_log.get())
        lines.append(f"--- CANCELLED ---\n{exc}")
        run_log.set(lines)
        result = RunResult(
            returncode=-1,
            output_dir=Path(""),
            stdout="",
            stderr=str(exc),
            status="cancelled",
            message=str(exc) or "user cancelled",
        )
    except Exception as exc:
        # Pre-C4 this returned early without calling _handle_result, leaving
        # state.output_dir pointing at the partial / missing output dir from
        # the previous run. C4 (Phase A) routes failures through
        # _handle_result so state.output_dir gets cleared.
        _log.error("Python engine failed: %s", exc)
        lines = list(run_log.get())
        lines.append(f"--- ERROR ---\n{exc}")
        run_log.set(lines)
        result = RunResult(
            returncode=1,
            output_dir=Path(""),
            stdout="",
            stderr=str(exc),
            status="failed",
            message=str(exc),
        )
    finally:
        state.busy.set(None)
        ui.update_action_button("btn_run", disabled=False, session=session)
        ui.update_action_button("btn_cancel", disabled=True, session=session)

    _handle_result(result, config, state, run_log, status)


async def _run_java_engine(
    input,
    state,
    session,
    config,
    work_dir,
    source_dir,
    run_log,
    status,
    runner_ref,
):
    """Run the simulation using the Java JAR subprocess."""
    jar_path = Path(state.jar_path.get())
    if not jar_path.exists():
        status.set(f"Error: JAR not found at {jar_path}")
        ui.update_action_button("btn_run", disabled=False, session=session)
        ui.update_action_button("btn_cancel", disabled=True, session=session)
        return

    config_path = write_temp_config(
        config, work_dir, source_dir, key_case_map=state.key_case_map.get()
    )

    overrides = parse_overrides(input.param_overrides() or "")
    java_opts_text = input.java_opts() or ""
    java_opts = java_opts_text.split() if java_opts_text.strip() else []
    try:
        validate_java_opts(java_opts)  # type: ignore[arg-type]
    except ValueError as exc:
        ui.notification_show(str(exc), type="error", duration=15)
        ui.update_action_button("btn_run", disabled=False, session=session)
        ui.update_action_button("btn_cancel", disabled=True, session=session)
        status.set(f"Error: {exc}")
        return
    java_opts = java_opts or None

    runner = OsmoseRunner(jar_path=jar_path)
    runner_ref.set(runner)  # type: ignore[arg-type]

    status.set("Running (Java engine)...")

    def on_progress(line: str):
        with reactive.isolate():
            lines = list(run_log.get())
        lines.append(line)
        if len(lines) > 500:
            lines = lines[-500:]
        run_log.set(lines)

    timeout_sec = input.run_timeout()

    state.busy.set("Running simulation (Java)...")
    try:
        result = await runner.run(
            config_path=config_path,
            output_dir=work_dir / "output",
            java_opts=java_opts,  # type: ignore[arg-type]
            overrides=overrides,
            on_progress=on_progress,
            timeout_sec=timeout_sec,
        )
    finally:
        state.busy.set(None)
        ui.update_action_button("btn_run", disabled=False, session=session)
        ui.update_action_button("btn_cancel", disabled=True, session=session)

    _handle_result(result, config, state, run_log, status)


def _handle_result(result, config, state, run_log, status):
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
            from osmose.history import RunRecord, RunHistory

            history = RunHistory(Path("data/history"))
            record = RunRecord(
                config_snapshot=config,
                duration_sec=0,
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
    if result.stderr:
        lines = list(run_log.get())
        lines.append(f"--- STDERR ---\n{result.stderr}")
        run_log.set(lines)


def run_server(input, output, session, state):
    run_log = reactive.value([])
    status = reactive.value("Idle")
    runner_ref = reactive.value(None)

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

    @reactive.effect
    def _sync_engine_tab():
        mode = state.engine_mode.get()
        tab = "run_java_tab" if mode == "java" else "run_python_tab"
        ui.update_navset("run_engine_tabs", selected=tab, session=session)

    @render.text
    def run_status():
        return status.get()

    @render.ui
    def run_console():
        lines = run_log.get()
        text = "\n".join(lines[-200:]) if lines else "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(
            text,
            style=STYLE_CONSOLE,
        )

    @reactive.effect
    @reactive.event(input.btn_run)
    async def handle_run():
        engine_mode = state.engine_mode.get()

        # Validate config before run (common to both engines)
        config = state.config.get()
        errors, warnings = validate_config(config, state.registry)
        source_dir = state.config_dir.get()
        if source_dir:
            file_errors = check_file_references(config, str(source_dir), state.registry)
            errors.extend(file_errors)
        species_warnings = check_species_consistency(config)
        warnings.extend(species_warnings)

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
        ui.update_action_button("btn_run", disabled=True, session=session)
        ui.update_action_button("btn_cancel", disabled=False, session=session)

        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        source_dir = state.config_dir.get()

        if engine_mode == "python":
            await _run_python_engine(
                input,
                state,
                session,
                config,
                work_dir,
                source_dir,
                run_log,
                status,
                runner_ref,
            )
        else:
            await _run_java_engine(
                input,
                state,
                session,
                config,
                work_dir,
                source_dir,
                run_log,
                status,
                runner_ref,
            )

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
