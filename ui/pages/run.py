"""Run control page - execute OSMOSE simulations."""

import logging
import shutil
import tempfile
from pathlib import Path

from shiny import ui, reactive, render

JAR_DIR = Path("osmose-java")

from osmose.config.writer import OsmoseConfigWriter
from osmose.runner import OsmoseRunner
from ui.styles import STYLE_CONSOLE

_log = logging.getLogger("osmose.run")


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


def write_temp_config(
    config: dict[str, str], output_dir: Path, source_dir: Path | None = None
) -> Path:
    """Write config to a directory, copy data files, and return the master file path."""
    writer = OsmoseConfigWriter()
    writer.write(config, output_dir)
    if source_dir and source_dir.is_dir():
        copy_data_files(config, source_dir, output_dir)
    return output_dir / "osm_all-parameters.csv"


def run_ui():
    return ui.layout_columns(
        # Left: Run controls
        ui.card(
            ui.card_header("Run Configuration"),
            ui.output_ui("jar_selector"),
            ui.input_text("java_opts", "Java options", value="-Xmx2g", placeholder="-Xmx4g -Xms1g"),
            ui.input_numeric("run_timeout", "Timeout (seconds)", value=3600, min=60, max=86400),
            ui.input_text_area(
                "param_overrides", "Parameter overrides (key=value, one per line)", rows=4
            ),
            ui.hr(),
            ui.layout_columns(
                ui.input_action_button("btn_run", "Start Run", class_="btn-success btn-lg w-100"),
                ui.input_action_button("btn_cancel", "Cancel", class_="btn-danger btn-lg w-100"),
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
    )


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
        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            status.set(f"Error: JAR not found at {jar_path}")
            return

        status.set("Writing config...")
        run_log.set([])

        # Write config to temp directory, copying data files from source
        config = state.config.get()
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        source_dir = state.config_dir.get()
        config_path = write_temp_config(config, work_dir, source_dir)

        # Parse overrides and java opts
        overrides = parse_overrides(input.param_overrides() or "")
        java_opts_text = input.java_opts() or ""
        java_opts = java_opts_text.split() if java_opts_text.strip() else None

        # Create runner
        runner = OsmoseRunner(jar_path=jar_path)
        runner_ref.set(runner)

        status.set("Running...")

        def on_progress(line: str):
            lines = list(run_log.get())
            lines.append(line)
            run_log.set(lines)

        timeout_sec = int(input.run_timeout()) if input.run_timeout() else None

        result = await runner.run(
            config_path=config_path,
            output_dir=work_dir / "output",
            java_opts=java_opts,
            overrides=overrides,
            on_progress=on_progress,
            timeout_sec=timeout_sec,
        )

        state.run_result.set(result)
        state.output_dir.set(result.output_dir)

        if result.returncode == 0:
            status.set(f"Complete. Output: {result.output_dir}")
        else:
            status.set(f"Failed (exit code {result.returncode})")
            if result.stderr:
                lines = list(run_log.get())
                lines.append(f"--- STDERR ---\n{result.stderr}")
                run_log.set(lines)

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        runner = runner_ref.get()
        if runner:
            runner.cancel()
            status.set("Cancelled")
