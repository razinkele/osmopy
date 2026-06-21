"""Run control page - execute OSMOSE simulations."""

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
from osmose.engine import PythonEngine, SimulationCancelled
from osmose.engine_capabilities import describe_engine
from osmose.live_movement import (
    config_is_spatial,
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
from ui.pages.live_movement_render import dots_layer_from_points, heatmap_layer_from_points
from ui.state import get_theme_mode
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
    target_version: str = "4.3.3",
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
                        "Threads (Numba; 0 = auto/all cores)",
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
            body_collapse_header("Live Movement (Python engine)", "run_live_movement"),
            ui.input_switch("live_movement_view", "Stream movement during run", value=False),
            ui.input_radio_buttons(
                "live_movement_mode",
                "Mode",
                {"heatmap": "Heatmap", "dots": "Dots"},
                selected="heatmap",
                inline=True,
            ),
            ui.input_select("live_movement_species", "Species", choices={"__all__": "All species"}),
            ui.output_ui("live_movement_status"),
            live_map.ui(height="420px"),
        ),
        class_="osm-run-root",
        id="run_page",
    )


def _python_engine_thread(run_config, output_dir, cancel_token, step_observer, done_q, n_threads=0):
    """Run the Python engine in a background thread; post the outcome to ``done_q``.

    Fire-and-forget (the calibration-dashboard pattern): runs OFF the main thread so the
    event handler that launched it returns immediately, letting the reactive poll flush
    live movement frames AND run_log/status during the run. (The previous
    ``await loop.run_in_executor(whole run)`` kept ``handle_run`` suspended, so Shiny
    deferred every flush until the run finished — the live view never updated mid-run.)

    Touches NO reactive state — the main-thread completion poll
    (``run_server._drain_run_done``) turns the posted outcome into run_log / status /
    button / ``_handle_result`` updates. Posts ``(kind, result_or_None, message)`` where
    ``kind`` is ``"done" | "cancelled" | "failed"``.
    """
    try:
        import numba  # type: ignore[import-untyped]  # optional extra; engine has a pure-Python fallback

        cap = numba.config.NUMBA_NUM_THREADS  # type: ignore[attr-defined]
        numba.set_num_threads(min(n_threads, cap) if n_threads >= 1 else cap)  # n<1 = auto/all cores
    except Exception:  # noqa: BLE001 — never block a run on numba absence/bad count
        _log.warning("could not apply py_threads; using Numba default", exc_info=True)
    engine = PythonEngine()
    try:
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
            from osmose.history import RunRecord, default_run_history

            history = default_run_history()
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

    # ── Live movement state (Python engine only) ─────────────────
    _live_map = MapWidget("live_map", style=CARTO_POSITRON)
    _live_queue: queue.Queue = queue.Queue(maxsize=4)
    _live_snapshot: reactive.Value = reactive.Value(None)  # MovementSnapshot | None
    _live_status_val: reactive.Value = reactive.Value(
        ""
    )  # "" | running | done | cancelled | failed
    _live_framed = [False]  # plain mutable flag (NOT reactive — render effect reads+writes it)
    _last_live_species: list[list[str] | None] = [
        None
    ]  # plain flag for the species-selector changed-only guard

    # ── Python-run completion (fire-and-forget thread → main-thread poll) ─────────
    _run_done_q: queue.Queue = queue.Queue(maxsize=1)  # (kind, result|None, message)
    _run_config_cell: list = [None]  # config captured at run start, for _handle_result

    _progress_q: queue.Queue = queue.Queue(maxsize=1)  # (done, n_steps, elapsed_s)
    _progress: reactive.Value = reactive.Value(None)  # None | (done, n_steps, elapsed_s)

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_run_done():
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
        ui.update_action_button("btn_run", disabled=False, session=session)
        ui.update_action_button("btn_cancel", disabled=True, session=session)
        _handle_result(result, _run_config_cell[0], state, run_log, status)
        while True:
            try:
                _progress_q.get_nowait()
            except queue.Empty:
                break
        _progress.set(None)

    @reactive.effect
    def _consume_run_done():
        _drain_run_done()

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_live_queue():
        latest = None
        while True:
            try:
                latest = _live_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _live_snapshot.set(latest)

    @reactive.effect
    def _consume_live_poll():
        _drain_live_queue()

    @reactive.poll(lambda: time.time(), interval_secs=0.2)
    def _drain_progress():
        latest = None
        while True:
            try:
                latest = _progress_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            _progress.set(latest)

    @reactive.effect
    def _consume_progress():
        _drain_progress()

    @reactive.effect
    def _populate_live_species():
        snap = _live_snapshot.get()
        if snap is None:
            return
        if snap.species == _last_live_species[0]:
            return
        _last_live_species[0] = list(snap.species)
        choices = {"__all__": "All species"}
        choices.update({name: name for name in snap.species})
        ui.update_select("live_movement_species", choices=choices)

    _last_spatial: list[bool | None] = [None]  # changed-only guard for auto-enable

    @reactive.effect
    def _auto_enable_live_for_spatial():
        config = state.config.get()
        if not config:
            return
        spatial = config_is_spatial(config)
        if spatial == _last_spatial[0]:
            return
        _last_spatial[0] = spatial
        ui.update_switch("live_movement_view", value=spatial, session=session)

    @render.ui
    def live_movement_status():
        status_v = _live_status_val.get()
        snap = _live_snapshot.get()
        if not status_v:
            if state.engine_mode.get() != "python":
                return ui.p("Live view available for the Python engine.", class_="text-muted")
            return ui.p("Enable the toggle before running to stream movement.", class_="text-muted")
        prog = f"step {snap.step + 1}/{snap.n_steps}" if snap is not None else ""
        extra = ""
        if snap is not None and snap.truncated:
            extra = f" — showing {snap.sp_id.size} of {snap.n_total} schools"
        return ui.p(f"{status_v} {prog}{extra}".strip())

    @reactive.effect
    async def _render_live_map():
        snap = _live_snapshot.get()
        mode = input.live_movement_mode()
        sel = input.live_movement_species()
        species_filter = None if sel in ("__all__", None) else sel
        style = CARTO_DARK if get_theme_mode(input) == "dark" else CARTO_POSITRON
        if style != _live_map.style:
            _live_map.style = style
            await _live_map.set_style(session, style)
        if snap is None:
            return
        layer = (
            dots_layer_from_points(snap, species_filter)
            if mode == "dots"
            else heatmap_layer_from_points(snap, species_filter)
        )
        if not _live_framed[0]:
            await _live_map.update(
                session,
                layers=[layer],
                view_state={
                    "latitude": (snap.lat_min + snap.lat_max) / 2,
                    "longitude": (snap.lon_min + snap.lon_max) / 2,
                    "zoom": 5,
                },
                widgets=[
                    fullscreen_widget(placement="top-left"),
                    zoom_widget(placement="top-right"),
                    compass_widget(placement="top-right"),
                    scale_widget(placement="bottom-left"),
                ],
            )
            _live_framed[0] = True
        else:
            await _live_map.partial_update(session, layers=[layer])

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
        cap = describe_engine(state.engine_mode.get(), config)
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
    @reactive.event(input.btn_run)
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
            block = java_engine_block_reason(config)
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
        ui.update_action_button("btn_run", disabled=True, session=session)
        ui.update_action_button("btn_cancel", disabled=False, session=session)

        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        source_dir = state.config_dir.get()

        live_observer = None
        if input.live_movement_view() and engine_mode == "python":
            while True:
                try:
                    _live_queue.get_nowait()
                except queue.Empty:
                    break
            _live_snapshot.set(None)
            _live_framed[0] = False
            _last_live_species[0] = None
            _live_status_val.set("running")
            live_observer = make_step_observer(_live_queue)

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
            state.busy.set("Running simulation (Python)...")
            status.set("Running (Python engine)...")
            _run_config_cell[0] = config
            run_observer = make_run_observer(_progress_q, live_observer)
            n_threads = int(input.py_threads() or 0)
            threading.Thread(
                target=_python_engine_thread,
                args=(run_config, output_dir, cancel_token, run_observer, _run_done_q, n_threads),
                daemon=True,
            ).start()
            # handle_run returns now; _drain_run_done finishes the run on the main thread.
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
