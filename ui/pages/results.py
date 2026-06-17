"""Results visualization page."""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.logging import setup_logging
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.pages.scenario_diff import scenario_diff_nav_panel, scenario_diff_server
from ui.state import AppState
from ui.styles import STYLE_EMPTY, STYLE_MONO_KEY

_log = setup_logging("osmose.results.ui")


_OSMOSE_TMP_PREFIXES: tuple[str, ...] = (
    "osmose_run_",
    "osmose_demo_",
    "osmose_export_",
    "osmose_cal_",
    "osmose_val_",
    "osmose_sens_",
)


def _safe_output_dir(raw: str) -> Path | None:
    """Validate a user-supplied output directory; return resolved Path or None.

    Closes C3 (path-traversal): previously the `..`-substring check accepted
    absolute paths like `/etc`, letting users read directories outside the
    working tree. We now resolve the path and accept it only when:

    1. It lives under `Path.cwd()` (the normal case for in-tree run outputs), OR
    2. It lives under `tempfile.gettempdir()` AND its first path component
       matches one of the osmose-owned tmpdir prefixes (`osmose_run_`,
       `osmose_demo_`, ...) used by the Run / Calibration / Demo / Export
       tabs when they call `tempfile.mkdtemp(prefix=...)`. This carve-out
       restores the natural Run → Results flow, which broke when the C3 fix
       overreached — a user-typed `/etc/passwd` is still rejected, but the
       engine-created `/tmp/osmose_run_abc123/output/` is accepted.

    Symlinks are resolved before the prefix / is-relative-to check, so a
    symlink that escapes the allowlist is rejected while a symlink pointing
    into the allowlist is accepted (common for calibration scenario forks).
    """
    if not raw:
        return None
    try:
        p = Path(raw).resolve(strict=False)
    except OSError:
        return None
    if not p.is_dir():
        return None

    cwd = Path.cwd().resolve()
    if p == cwd or p.is_relative_to(cwd):
        return p

    tmp_root = Path(tempfile.gettempdir()).resolve()
    if p.is_relative_to(tmp_root):
        rel = p.relative_to(tmp_root)
        if rel.parts and rel.parts[0].startswith(_OSMOSE_TMP_PREFIXES):
            return p

    return None


def _compare_run_choices(runs) -> dict[str, str]:
    """Build the compare_runs_select choices dict from RunHistory.list_runs() records.

    {timestamp: "<first 19 chars of timestamp> (<duration>s)"} — the exact label
    format the Compare Runs selector has always used.
    """
    return {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}


# ---------------------------------------------------------------------------
# Pure chart-generation functions (testable without Shiny)
# ---------------------------------------------------------------------------


def _tpl(input=None) -> str:
    """Return the Plotly template name for the current theme."""
    if input is None:
        return "osmose"
    from ui.state import get_theme_mode

    mode = get_theme_mode(input)
    return "osmose" if mode == "dark" else "osmose-light"


def make_timeseries_chart(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    species: str | None = None,
    template: str = "osmose",
) -> go.Figure:
    """Create a time series line chart from OSMOSE output.

    Accepts both shapes the engine emits:
    - Long/tidy form: columns ``[time, species, <value_col>]``.
    - Wide form (what ``OsmoseResults.biomass()`` returns for cross-species
      output types like biomass/abundance/yield): columns
      ``[Time, <sp_a>, <sp_b>, ..., species]`` where the ``species`` column is
      a constant ``"all"``. Wide form is detected and melted internally before
      plotting.
    """
    if df.empty:
        return go.Figure().update_layout(title=title, template=template)

    cols = list(df.columns)
    # Time-column casing varies (the engine emits "Time"; older fixtures use "time").
    if "Time" in cols:
        time_col = "Time"
    elif "time" in cols:
        time_col = "time"
    else:
        return go.Figure().update_layout(title=title, template=template)

    # Detect wide form: has a "species" column but the requested value column
    # is NOT present, while >=1 non-time, non-species columns exist (one per
    # species). Melt those species columns into rows.
    if "species" in cols and value_col not in cols:
        species_cols = [c for c in cols if c not in (time_col, "species")]
        if species_cols:
            df = df.drop(columns=["species"]).melt(
                id_vars=time_col,
                value_vars=species_cols,
                var_name="species",
                value_name=value_col,
            )

    # "all" is the UI's sentinel for "show every species" — treat as no filter.
    if species and species != "all" and "species" in df.columns:
        df = df[df["species"] == species]  # type: ignore[assignment]
    if df.empty:
        return go.Figure().update_layout(title=title, template=template)

    import plotly.express as px

    fig = px.line(df, x=time_col, y=value_col, color="species", title=title)
    fig.update_layout(template=template)
    return fig


# Columns that are metadata, never a `<predator>_<prey>` diet pair.
_DIET_META_COLS = {"Time", "time", "Step", "step", "species", "Simu", "simu", "replicate"}


def make_diet_heatmap(df: pd.DataFrame, template: str = "osmose") -> go.Figure:
    """Create a diet composition heatmap (predator rows x prey columns).

    Handles two input layouts:
    - legacy ``prey_<name>`` columns (optionally with a ``species`` predator column);
    - the engine's wide diet matrix, ``<predator>_<prey>`` columns (predator-major,
      values = biomass eaten; see osmose/engine/output.py) plus a ``Time`` column and a
      constant ``species`` column added by the reader. This layout is averaged over time
      and normalised per predator row to diet proportions.
    """
    if df.empty:
        return go.Figure().update_layout(title="Diet Composition", template=template)
    import plotly.express as px

    # Legacy layout: explicit `prey_<name>` columns.
    prey_cols = [c for c in df.columns if c.startswith("prey_")]
    if prey_cols:
        if "species" in df.columns:
            matrix = df.groupby("species")[prey_cols].mean()
        else:
            matrix = df[prey_cols].mean().to_frame().T  # type: ignore[union-attr]
        prey_names = [c.replace("prey_", "") for c in prey_cols]
        fig = px.imshow(
            matrix.values,  # type: ignore[union-attr]
            x=prey_names,
            y=list(matrix.index),  # type: ignore[arg-type,union-attr]
            title="Diet Composition",
            color_continuous_scale="YlOrRd",
            labels={"x": "Prey", "y": "Predator", "color": "Proportion"},
        )
        fig.update_layout(template=template)
        return fig

    # Engine wide layout: `<predator>_<prey>` columns. Split on the first "_"
    # (predator names carry no underscore; the remainder is the prey name).
    pair_cols = [c for c in df.columns if "_" in c and c not in _DIET_META_COLS]
    if not pair_cols:
        return go.Figure().update_layout(title="Diet Composition (no prey data)", template=template)
    means = df[pair_cols].mean(numeric_only=True)
    predators: list[str] = []
    prey_names = []
    cells: dict[tuple[str, str], float] = {}
    for col, val in means.items():
        pred, _, prey = str(col).partition("_")
        if pred not in predators:
            predators.append(pred)
        if prey not in prey_names:
            prey_names.append(prey)
        cells[(pred, prey)] = float(val)
    z = np.array([[cells.get((p, q), 0.0) for q in prey_names] for p in predators], dtype=float)
    # Per-predator-row normalisation to diet proportions (rows with no diet stay 0).
    row_sums = z.sum(axis=1, keepdims=True)
    z = np.divide(z, row_sums, out=np.zeros_like(z), where=row_sums > 0)
    fig = px.imshow(
        z,
        x=prey_names,
        y=predators,
        title="Diet Composition",
        color_continuous_scale="YlOrRd",
        labels={"x": "Prey", "y": "Predator", "color": "Proportion"},
    )
    fig.update_layout(template=template)
    return fig


def _delta_for_selected(records, metric: str, window_years: int):
    """Per-species output delta between exactly two runs (baseline=records[0], variant=records[1]).

    Reconstructs each run as OsmoseResults(rec.output_dir) and returns run_delta's
    list[SpeciesDelta]. Raises ValueError unless exactly 2 records are given.
    """
    if len(records) != 2:
        raise ValueError("need exactly 2 runs for a pairwise output delta")
    from pathlib import Path

    from osmose.analysis import run_delta
    from osmose.results import OsmoseResults

    baseline = OsmoseResults(Path(records[0].output_dir), strict=False)
    variant = OsmoseResults(Path(records[1].output_dir), strict=False)
    return run_delta(baseline, variant, metric=metric, window_years=window_years)


# ---------------------------------------------------------------------------
# Result method mapping — explicit lookup replaces fragile getattr
# ---------------------------------------------------------------------------

_RESULT_METHODS: dict[str, str] = {
    "biomass": "biomass",
    "abundance": "abundance",
    "yield": "yield_biomass",
    "mortality": "mortality",
    "diet": "diet_matrix",
    "trophic": "mean_trophic_level",
    "biomass_by_age": "biomass_by_age",
    "biomass_by_size": "biomass_by_size",
    "biomass_by_tl": "biomass_by_tl",
    "abundance_by_age": "abundance_by_age",
    "abundance_by_size": "abundance_by_size",
    "yield_by_age": "yield_by_age",
    "yield_by_size": "yield_by_size",
    "yield_n": "yield_abundance",
    "mortality_rate": "mortality_rate",
    "size_spectrum": "size_spectrum",
}


# ---------------------------------------------------------------------------
# Shiny UI
# ---------------------------------------------------------------------------


def results_ui():
    return ui.div(
        ui.div(
            expand_tab("Output Controls", "results"),
            ui.layout_columns(
                # Sidebar: Controls
                ui.card(
                    collapsible_card_header("Output Controls", "results"),
                    ui.output_ui("output_dir_input"),
                    ui.input_action_button(
                        "btn_load_results", "Load Results", class_="btn-primary w-100"
                    ),
                    ui.hr(),
                    ui.input_select(
                        "result_species",
                        "Species filter",
                        choices={"all": "All species"},
                        selected="all",
                    ),
                    ui.input_select(
                        "result_type",
                        "Output type",
                        choices={
                            "biomass": "Biomass",
                            "abundance": "Abundance",
                            "yield": "Yield",
                            "mortality": "Mortality",
                            "diet": "Diet Matrix",
                            "trophic": "Trophic Level",
                            "biomass_by_age": "Biomass by Age",
                            "biomass_by_size": "Biomass by Size",
                            "biomass_by_tl": "Biomass by TL",
                            "abundance_by_age": "Abundance by Age",
                            "abundance_by_size": "Abundance by Size",
                            "yield_by_age": "Yield by Age",
                            "yield_by_size": "Yield by Size",
                            "yield_n": "Catch Numbers",
                            "mortality_rate": "Mortality by Source",
                            "size_spectrum": "Size Spectrum",
                            "sheldon_spectrum": "Sheldon (Mass) Spectrum",
                            "abc_curve": "ABC (W-statistic)",
                        },
                        selected="biomass",
                    ),
                    ui.output_ui("ensemble_toggle"),
                    ui.hr(),
                    ui.download_button(
                        "download_results_csv", "Download CSV", class_="btn-outline-primary w-100"
                    ),
                ),
                # Main: Time Series visualization
                ui.card(
                    ui.card_header("Time Series"),
                    output_widget("results_chart"),
                ),
                col_widths=[3, 9],
            ),
            class_="osm-split-layout",
            id="split_results",
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Diet Composition",
                output_widget("diet_chart"),
            ),
            ui.nav_panel(
                "Trophic Network",
                # NB: this is an INDEX into the discrete diet-matrix Time list (see Step-3b/3c),
                # NOT the raw Time value — so fractional/sub-annual Time steps are addressable.
                ui.input_slider("trophic_time", "Timestep", min=0, max=0, value=0, step=1),
                ui.input_radio_buttons(
                    "trophic_predator_level",
                    "Predator level",
                    {"species": "Species", "stage": "Size stage"},
                    selected="species",
                    inline=True,
                ),
                ui.input_slider("trophic_threshold", "Min diet %", min=0, max=50, value=5, step=1),
                ui.output_ui("trophic_network"),
            ),
            ui.nav_panel(
                "Community Metrics",
                ui.output_ui("community_metrics_panel"),
            ),
            ui.nav_panel(
                "Compare Runs",
                ui.layout_columns(
                    ui.div(
                        ui.input_selectize(
                            "compare_runs_select",
                            "Select runs to compare",
                            choices={},
                            multiple=True,
                        ),
                        ui.input_select(
                            "compare_metric",
                            "Metric",
                            choices={
                                "biomass": "Biomass",
                                "yield": "Yield",
                                "abundance": "Abundance",
                            },
                        ),
                        ui.input_slider(
                            "compare_window_years", "Window (years)", min=1, max=30, value=10
                        ),
                    ),
                    col_widths=[12],
                ),
                output_widget("comparison_chart"),
                ui.output_ui("config_diff_table"),
                output_widget("run_delta_chart"),
                ui.output_ui("run_delta_table"),
            ),
            scenario_diff_nav_panel(),
        ),
    )


# ---------------------------------------------------------------------------
# Shiny Server
# ---------------------------------------------------------------------------


def results_server(input, output, session, state: AppState):
    results_obj: reactive.Value = reactive.Value(None)
    results_data: reactive.Value[dict[str, pd.DataFrame]] = reactive.Value({})
    rep_dirs: reactive.Value[list[Path]] = reactive.Value([])
    _prev_output_dir: reactive.Value[str] = reactive.Value("")
    _last_compare_choices: reactive.Value[dict[str, str]] = reactive.Value({})

    @render.ui
    def output_dir_input():
        with reactive.isolate():
            out = state.output_dir.get()
        default_val = str(out) if out else ""
        return ui.div(
            ui.input_text("output_dir", "Output directory", value=default_val),
            ui.output_ui("output_dir_status"),
        )

    @render.ui
    def output_dir_status():
        path_str = input.output_dir()
        if not path_str:
            return ui.div()
        p = Path(path_str)
        if not p.is_dir():
            return ui.tags.small("Directory not found", style="color: #e74c3c;")
        csvs = list(p.glob("*.csv"))
        # Also search engine subdirectories (Mortality/, Bioen/)
        for subdir in ("Mortality", "Bioen"):
            sub = p / subdir
            if sub.is_dir():
                csvs.extend(sub.glob("*.csv"))
        csvs = csvs[:200]
        if not csvs:
            return ui.tags.small("No results found in this directory", style="color: #e67e22;")
        return ui.tags.small(f"Found {len(csvs)} output files", style="color: #2ecc71;")

    def _do_load_results(out_dir: Path):
        state.busy.set("Loading results\u2026")
        try:
            from osmose.results import OsmoseResults

            # Pre-register this dir so _reset_results_loaded won't fire when
            # state.output_dir.set(out_dir) is called later in this function.
            _prev_output_dir.set(str(out_dir))

            old_res = results_obj.get()
            if old_res is not None and hasattr(old_res, "close_cache"):
                old_res.close_cache()

            res = OsmoseResults(out_dir, strict=False)
            results_obj.set(res)

            # Load only biomass eagerly (needed for species discovery)
            data: dict[str, pd.DataFrame] = {}
            data["biomass"] = res.biomass()
            results_data.set(data)

            # Detect ensemble replicate directories
            reps = sorted(out_dir.glob("rep_*"))
            rep_dirs.set([r for r in reps if r.is_dir()])

            # Update output dir in shared state
            if state is not None:
                state.output_dir.set(out_dir)

            # Discover species from biomass data, falling back to state
            species_choices: dict[str, str] = {"all": "All species"}
            bio_df = data.get("biomass", pd.DataFrame())
            if not bio_df.empty and "species" in bio_df.columns:
                for sp in sorted(bio_df["species"].unique()):
                    species_choices[sp] = sp
            elif state is not None:
                with reactive.isolate():
                    for sp in state.species_names.get():
                        species_choices[sp] = sp
            ui.update_select("result_species", choices=species_choices)

            # Trophic-network time slider = an INDEX into the discrete diet-matrix Time list
            # (0 .. n-1), so fractional/sub-annual Time values are addressable (the raw Time
            # is shown as a caption by the render fn). Leave at default if there's no diet output.
            try:
                from osmose.trophic_network import available_times

                _times = available_times(out_dir)
                if _times:
                    ui.update_slider("trophic_time", min=0, max=len(_times) - 1, value=0)
            except (FileNotFoundError, OSError, ValueError):
                pass  # no diet output -> leave the slider at its default

            ui.notification_show("Results loaded successfully.", type="message", duration=3)

            state.results_loaded.set(True)
        except (
            OSError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            _log.error("Failed to load results: %s", exc, exc_info=True)
            ui.notification_show(f"Error loading results: {exc}", type="error", duration=15)
        finally:
            state.busy.set(None)

    def _get_result_data(output_type: str) -> pd.DataFrame:
        """Load result data lazily — only when requested.

        ``results_obj`` is the SOLE reactive dependency (read first, unconditionally):
        outputs recompute exactly once when a new result loads. ``results_data`` is a
        pure memo and is read/written under ``reactive.isolate()`` so a lazy cache-fill
        does NOT re-invalidate the very output that triggered it. Without this isolation,
        a render fn (e.g. diet_chart) depends on results_data, fills it on first run, and
        that write re-invalidates the fn → a second recalc whose overlapping
        recalculating/progress messages desync shiny's OutputProgressReporter (the
        user-visible "diet_chart … is in an unexpected state" client-error panel).
        """
        res = results_obj.get()  # the live dependency — recompute on new results
        if res is None:
            return pd.DataFrame()
        with reactive.isolate():
            data = results_data.get()
        if output_type in data:
            return data[output_type]
        # Use explicit method mapping; fall back to export_dataframe for unknowns.
        method_name = _RESULT_METHODS.get(output_type)
        if method_name:
            method = getattr(res, method_name, None)
            df = method() if method is not None else res.export_dataframe(output_type)
        else:
            df = res.export_dataframe(output_type)
        # Memo-cache it WITHOUT touching the reactive graph (isolate the write).
        with reactive.isolate():
            cached = dict(results_data.get())
            cached[output_type] = df
            results_data.set(cached)
        return df

    @reactive.effect
    @reactive.event(input.btn_load_results)
    def _load_results():
        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            ui.notification_show(
                "Invalid output directory: must be inside the working directory.",
                type="error",
                duration=15,
            )
            return
        _do_load_results(out_dir)

    @reactive.effect
    def _auto_load_results():
        """Auto-load results when navigating to Results tab after a run."""
        nav = input.main_nav()
        if nav != "results":
            return
        with reactive.isolate():
            out = state.output_dir.get()
            loaded = state.results_loaded.get()
        if out and not loaded and Path(str(out)).is_dir():
            ui.update_text("output_dir", value=str(out))
            _do_load_results(Path(str(out)))

    @reactive.effect
    def _populate_compare_runs():
        """Populate the Compare Runs selector from run history whenever the user is on
        the Results tab — independent of any loaded output dir. Changed-only guard avoids
        re-tearing-down the widget (and clobbering the selection) on every re-navigation;
        selection is preserved across a real refresh (a newly recorded run)."""
        if input.main_nav() != "results":
            return
        from osmose.history import default_run_history

        try:
            runs = default_run_history().list_runs()
        except Exception:  # noqa: BLE001 — never crash the page on a history-read error
            return
        choices = _compare_run_choices(runs)
        with reactive.isolate():
            if choices == _last_compare_choices.get():
                return
            current = input.compare_runs_select()
        _last_compare_choices.set(choices)
        keep = [ts for ts in (current or ()) if ts in choices]
        ui.update_selectize("compare_runs_select", choices=choices, selected=keep)

    @reactive.effect
    def _reset_results_loaded():
        """Reset loaded flag only when output directory changes to a new, unloaded path.

        _do_load_results pre-sets _prev_output_dir before calling state.output_dir.set(),
        so this effect is a no-op during the load cycle (new_dir == prev).
        """
        new_dir = str(state.output_dir.get() or "")
        with reactive.isolate():
            prev = _prev_output_dir.get()
        if new_dir == prev:
            return  # same dir — either no change or triggered by _do_load_results
        _prev_output_dir.set(new_dir)
        state.results_loaded.set(False)

    @render.ui
    def ensemble_toggle():
        dirs = rep_dirs.get()
        if dirs:
            return ui.input_switch(
                "ensemble_mode", f"Ensemble view ({len(dirs)} replicates)", value=True
            )
        return ui.div()

    @render_plotly
    def results_chart():
        rtype = input.result_type()
        species_filter = input.result_species()
        tmpl = _tpl(input)

        # Map result types to their value column names
        col_map = {
            "biomass": "biomass",
            "abundance": "abundance",
            "yield": "yield",
            "mortality": "mortality",
            "trophic": "meanTL",
            "biomass_by_age": "value",
            "biomass_by_size": "value",
            "biomass_by_tl": "value",
            "abundance_by_age": "value",
            "abundance_by_size": "value",
            "yield_by_age": "value",
            "yield_by_size": "value",
            "yield_n": "yieldN",
            "mortality_rate": "value",
            "size_spectrum": "abundance",
        }
        title_map = {
            "biomass": "Biomass",
            "abundance": "Abundance",
            "yield": "Yield (Catch)",
            "mortality": "Mortality",
            "trophic": "Mean Trophic Level",
            "biomass_by_age": "Biomass by Age",
            "biomass_by_size": "Biomass by Size",
            "biomass_by_tl": "Biomass by Trophic Level",
            "abundance_by_age": "Abundance by Age",
            "abundance_by_size": "Abundance by Size",
            "yield_by_age": "Yield by Age",
            "yield_by_size": "Yield by Size",
            "yield_n": "Catch Numbers",
            "mortality_rate": "Mortality by Source",
            "size_spectrum": "Size Spectrum",
            "sheldon_spectrum": "Sheldon (Mass) Spectrum",
            "abc_curve": "ABC (W-statistic)",
        }

        sp = species_filter if species_filter != "all" else None

        # Ensemble mode: show CI bands for 1D types
        from osmose.ensemble import ENSEMBLE_OUTPUT_TYPES

        ensemble_on = False
        try:
            ensemble_on = bool(input.ensemble_mode()) and bool(rep_dirs.get())
        except (SilentException, AttributeError):
            pass

        if ensemble_on and rtype in ENSEMBLE_OUTPUT_TYPES:
            from osmose.ensemble import aggregate_replicates
            from osmose.plotting import make_ci_timeseries

            agg = aggregate_replicates(rep_dirs.get(), rtype, species=sp)
            if agg["time"]:
                title = title_map.get(rtype, rtype.title())
                fig = make_ci_timeseries(
                    agg["time"],
                    agg["mean"],
                    agg["lower"],
                    agg["upper"],
                    title=f"{title} (ensemble)",
                    y_label=col_map.get(rtype, rtype) or "",
                )
                fig.update_layout(template=tmpl)
                return fig

        # If diet is selected, show a placeholder message in time series
        if rtype == "diet":
            return go.Figure().update_layout(
                title="Diet data shown in heatmap below",
                template=tmpl,
            )

        # Structured output types use stacked area charts
        structured_types = {
            "biomass_by_age",
            "biomass_by_size",
            "biomass_by_tl",
            "abundance_by_age",
            "abundance_by_size",
            "yield_by_age",
            "yield_by_size",
        }
        if rtype in structured_types:
            from osmose.plotting import make_stacked_area

            df = _get_result_data(rtype)
            fig = make_stacked_area(df, title=title_map.get(rtype, rtype), species=sp)  # type: ignore[arg-type]
            fig.update_layout(template=tmpl)
            return fig

        if rtype == "mortality_rate":
            from osmose.plotting import make_mortality_breakdown

            df = _get_result_data(rtype)
            fig = make_mortality_breakdown(df, species=sp)
            fig.update_layout(template=tmpl)
            return fig

        if rtype == "size_spectrum":
            from osmose.plotting import make_size_spectrum_plot

            df = _get_result_data(rtype)
            fig = make_size_spectrum_plot(df)
            fig.update_layout(template=tmpl)
            return fig

        # NOTE: this branch intentionally BYPASSES _get_result_data — it computes directly from
        # state.output_dir / state.config. It MUST stay ABOVE the catch-all that calls
        # _get_result_data(rtype) below; if moved below, these rtypes would fall through to
        # res.export_dataframe(rtype) and raise a confusing error instead of rendering.
        if rtype in ("sheldon_spectrum", "abc_curve"):
            from osmose.community_metrics import compute_abc, compute_sheldon_spectrum
            from osmose.plotting import make_abc_plot, make_sheldon_spectrum_plot

            out_dir = state.output_dir.get()
            if not out_dir:
                return go.Figure().update_layout(
                    title="Run a simulation to see community diagnostics", template=tmpl
                )
            if rtype == "sheldon_spectrum":
                cfg = state.config.get()
                if not cfg:
                    return go.Figure().update_layout(
                        title="Sheldon (mass) spectrum — load a config for length-weight a,b",
                        template=tmpl,
                    )
                try:
                    fig = make_sheldon_spectrum_plot(compute_sheldon_spectrum(out_dir, cfg))
                except FileNotFoundError:
                    return go.Figure().update_layout(
                        title="Sheldon (mass) spectrum — no by-size output for this run",
                        template=tmpl,
                    )
            else:  # abc_curve
                try:
                    fig = make_abc_plot(compute_abc(out_dir))
                except FileNotFoundError:
                    return go.Figure().update_layout(
                        title="ABC — needs biomass and abundance outputs", template=tmpl
                    )
            fig.update_layout(template=tmpl)
            return fig

        df = _get_result_data(rtype)
        value_col = col_map.get(rtype, rtype)
        title = title_map.get(rtype, rtype.title())

        # NOTE: a former fallback here silently reassigned `value_col` to the
        # first numeric column when it wasn't a column of `df`. That broke the
        # wide-form code path in `make_timeseries_chart` (which needs to detect
        # `value_col not in df.columns` to trigger the melt). The chart helper
        # now handles the missing-value-col case via melt, so the fallback was
        # removed.

        return make_timeseries_chart(df, value_col, title, species=sp, template=tmpl)  # type: ignore[arg-type]

    @render_plotly
    def diet_chart():
        tmpl = _tpl(input)
        df = _get_result_data("diet")
        return make_diet_heatmap(df, template=tmpl)

    @reactive.calc
    def _trophic_cache():
        """(loaded-dir-keyed) cached (dir, times, {level: layout}) so slider ticks are cheap.

        Keys off the LOADED output dir (results_obj + state.output_dir), not the live
        output_dir text box, so it stays consistent with the rest of the Results page
        (e.g. diet_chart) and doesn't re-read the diet CSV on every keystroke.
        """
        if results_obj.get() is None:
            return None
        out_dir = _safe_output_dir(str(state.output_dir.get() or ""))
        if out_dir is None:
            return None
        from osmose.trophic_network import available_times, network_node_universe, species_layout

        try:
            times = available_times(out_dir)  # probes existence; raises if no diet matrix
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not times:
            return None
        layouts = {
            lvl: species_layout(network_node_universe(out_dir, lvl)) for lvl in ("species", "stage")
        }
        return {"dir": out_dir, "times": times, "layouts": layouts}

    @render.ui
    def trophic_network():
        cache = _trophic_cache()
        if cache is None:
            return ui.div("No diet-matrix output found.", style=STYLE_EMPTY)
        try:
            from osmose.trophic_network import diet_network_at, make_trophic_network_html
        except ImportError:
            return ui.div("Install pyvis to view the trophic network.", style=STYLE_EMPTY)
        level = input.trophic_predator_level()
        # The slider holds an INDEX into cache["times"]; map it to the actual Time (clamped),
        # so a fractional/sub-annual Time is addressable and we never pass an absent time value.
        times = cache["times"]
        idx = max(0, min(int(input.trophic_time()), len(times) - 1))
        t = times[idx]
        try:
            net = diet_network_at(
                cache["dir"],
                time=t,
                threshold=float(input.trophic_threshold()),
                predator_level=level,
            )
            # net is already filtered to the user's threshold by diet_network_at; pass 0.0 so
            # make_trophic_network_html's default (5.0) doesn't silently re-clamp sub-5% sliders.
            html = make_trophic_network_html(net, positions=cache["layouts"][level], threshold=0.0)
        except (FileNotFoundError, OSError, ValueError) as e:
            return ui.div(f"Could not build trophic network: {e}", style=STYLE_EMPTY)
        return ui.div(
            ui.tags.small(f"Time {t:g}", style=STYLE_MONO_KEY),
            ui.tags.iframe(
                srcdoc=html, style="width:100%; height:640px; border:0;", sandbox="allow-scripts"
            ),
        )

    @render.ui
    def community_metrics_panel():
        from osmose.community_metrics import community_report, format_community_report

        out_dir = state.output_dir.get()
        if not out_dir:
            return ui.markdown("_Community metrics unavailable — run a simulation first._")
        diag = community_report(out_dir, state.config.get() or None)
        return ui.markdown(format_community_report(diag))

    @render_plotly
    def comparison_chart():
        tmpl = _tpl(input)
        selected = input.compare_runs_select()
        if not selected or len(selected) < 1:
            return go.Figure().update_layout(title="Select runs to compare", template=tmpl)

        from osmose.history import default_run_history
        from osmose.plotting import make_run_comparison

        history = default_run_history()
        try:
            records = [history.load_run(ts) for ts in selected]
        except Exception:  # noqa: BLE001 — stale/missing run file: degrade, don't crash the render
            return go.Figure().update_layout(title="No run history found", template=tmpl)
        metric = input.compare_metric()
        fig = make_run_comparison(records, metrics=[metric])
        fig.update_layout(template=tmpl)
        return fig

    @render.ui
    def config_diff_table():
        selected = input.compare_runs_select()
        if not selected or len(selected) < 2:
            return ui.div("Select 2+ runs to see config differences.", style=STYLE_EMPTY)

        from osmose.history import default_run_history

        history = default_run_history()
        try:
            diffs = history.compare_runs_multi(list(selected))
        except Exception:  # noqa: BLE001 — stale/missing run file: degrade, don't crash the render
            return ui.div("No run history found.")

        if not diffs:
            return ui.div("No config differences found.", style=STYLE_EMPTY)

        # Build table header: Parameter | Run 1 | Run 2 | ...
        headers = [ui.tags.th("Parameter")]
        for i in range(len(selected)):
            headers.append(ui.tags.th(f"Run {i + 1}"))

        rows = []
        for diff in diffs:
            cells = [ui.tags.td(diff["key"], style=STYLE_MONO_KEY)]
            for val in diff["values"]:
                cells.append(ui.tags.td(str(val) if val is not None else "—"))
            rows.append(ui.tags.tr(*cells))

        return ui.tags.table(
            ui.tags.thead(ui.tags.tr(*headers)),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped",
            style="font-size: 13px;",
        )

    @render_plotly
    def run_delta_chart():
        tmpl = _tpl(input)
        selected = input.compare_runs_select()
        if not selected or len(selected) != 2:
            return go.Figure().update_layout(
                title="Select exactly 2 runs (1st = baseline, 2nd = variant)", template=tmpl
            )
        from osmose.history import default_run_history
        from osmose.plotting import make_run_delta_chart

        metric = input.compare_metric()
        try:
            records = [default_run_history().load_run(ts) for ts in selected]
            deltas = _delta_for_selected(records, metric, int(input.compare_window_years()))
        except Exception as e:  # noqa: BLE001 — UI guard: degrade to an error title, never crash the page
            return go.Figure().update_layout(title=f"Could not compute delta: {e}", template=tmpl)
        fig = make_run_delta_chart(deltas, metric=metric)
        fig.update_layout(template=tmpl)
        return fig

    @render.ui
    def run_delta_table():
        selected = input.compare_runs_select()
        if not selected or len(selected) != 2:
            return ui.div(
                "Select exactly 2 runs to see the per-species output delta "
                "(1st = baseline, 2nd = variant). The config diff above supports more than 2.",
                style=STYLE_EMPTY,
            )
        from osmose.history import default_run_history
        from osmose.analysis import format_delta_report

        metric = input.compare_metric()
        try:
            records = [default_run_history().load_run(ts) for ts in selected]
            window_years = int(input.compare_window_years())
            deltas = _delta_for_selected(records, metric, window_years)
        except Exception as e:  # noqa: BLE001 — UI guard: degrade to an error div, never crash the page
            return ui.div(f"Could not load run outputs: {e}")
        return ui.markdown(format_delta_report(deltas, metric=metric, window_years=window_years))

    @render.download(
        filename=lambda: (  # type: ignore[arg-type]
            f"osmose_{input.result_type()}"
            + (f"_{input.result_species()}" if input.result_species() != "all" else "")
            + ".csv"
        )
    )
    def download_results_csv():
        from osmose.results import OsmoseResults
        import tempfile

        out_dir = _safe_output_dir(input.output_dir())
        if out_dir is None:
            ui.notification_show(
                "No valid output directory: must be inside the working directory. "
                "Load results first.",
                type="warning",
                duration=5,
            )
            return

        res = OsmoseResults(out_dir)
        sp = input.result_species()
        species = sp if sp != "all" else None
        df = res.export_dataframe(input.result_type(), species=species)

        if df.empty:
            ui.notification_show(
                "No data available for the selected output type and species filter.",
                type="warning",
                duration=5,
            )
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        csv_path = tmp_dir / "export.csv"
        df.to_csv(csv_path, index=False)
        atexit.register(shutil.rmtree, str(tmp_dir), True)
        return str(csv_path)

    scenario_diff_server(input, output, session, state)
