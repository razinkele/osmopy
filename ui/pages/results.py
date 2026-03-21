"""Results visualization page."""

from __future__ import annotations

import atexit
import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.logging import setup_logging
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.styles import STYLE_EMPTY, STYLE_MONO_KEY

_log = setup_logging("osmose.results.ui")


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
    """Create a time series line chart from OSMOSE output."""
    if df.empty:
        return go.Figure().update_layout(title=title, template=template)
    if species and "species" in df.columns:
        df = df[df["species"] == species]  # type: ignore[assignment]
    if df.empty:
        return go.Figure().update_layout(title=title, template=template)
    import plotly.express as px

    fig = px.line(df, x="time", y=value_col, color="species", title=title)
    fig.update_layout(template=template)
    return fig


def make_diet_heatmap(df: pd.DataFrame, template: str = "osmose") -> go.Figure:
    """Create a diet composition heatmap."""
    if df.empty:
        return go.Figure().update_layout(title="Diet Composition", template=template)
    prey_cols = [c for c in df.columns if c.startswith("prey_")]
    if not prey_cols:
        return go.Figure().update_layout(title="Diet Composition (no prey data)", template=template)
    import plotly.express as px

    if "species" in df.columns:
        matrix = df.groupby("species")[prey_cols].mean()
    else:
        matrix = df[prey_cols].mean().to_frame().T  # type: ignore[union-attr]
    prey_names = [c.replace("prey_", "") for c in prey_cols]
    fig = px.imshow(
        matrix.values,
        x=prey_names,
        y=list(matrix.index),
        title="Diet Composition",
        color_continuous_scale="YlOrRd",
        labels={"x": "Prey", "y": "Predator", "color": "Proportion"},
    )
    fig.update_layout(template=template)
    return fig


def make_spatial_map(
    ds,
    var_name: str,
    time_idx: int = 0,
    title: str | None = None,
    template: str = "osmose",
) -> go.Figure:
    """Create a spatial heatmap from NetCDF data."""
    import plotly.express as px

    data = ds[var_name].isel(time=time_idx).values
    lat = ds["lat"].values
    lon = ds["lon"].values
    fig = px.imshow(
        data,
        x=lon,
        y=lat,
        origin="lower",
        color_continuous_scale="Viridis",
        labels={"x": "Longitude", "y": "Latitude", "color": var_name},
        title=title or f"{var_name} (t={time_idx})",
    )
    fig.update_layout(template=template)
    return fig


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
                "Spatial Distribution",
                ui.input_slider(
                    "spatial_time_idx",
                    "Time step",
                    min=0,
                    max=1,
                    value=0,
                    step=1,
                    animate=ui.AnimationOptions(
                        interval=1000,
                        loop=True,
                        play_button="Play",
                        pause_button="Pause",
                    ),
                ),
                output_widget("spatial_chart"),
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
                    ),
                    col_widths=[12],
                ),
                output_widget("comparison_chart"),
                ui.output_ui("config_diff_table"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Shiny Server
# ---------------------------------------------------------------------------


def results_server(input, output, session, state):
    results_obj: reactive.Value = reactive.Value(None)
    results_data: reactive.Value[dict[str, pd.DataFrame]] = reactive.Value({})
    spatial_ds: reactive.Value = reactive.Value(None)
    rep_dirs: reactive.Value[list[Path]] = reactive.Value([])
    _prev_output_dir: reactive.Value[str] = reactive.Value("")

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
        if not csvs:
            return ui.tags.small("No results found in this directory", style="color: #e67e22;")
        return ui.tags.small(f"Found {len(csvs)} output files", style="color: #2ecc71;")

    def _do_load_results(out_dir: Path):
        try:
            from osmose.results import OsmoseResults

            # Pre-register this dir so _reset_results_loaded won't fire when
            # state.output_dir.set(out_dir) is called later in this function.
            _prev_output_dir.set(str(out_dir))

            old_res = results_obj.get()
            if old_res is not None and hasattr(old_res, "close_cache"):
                old_res.close_cache()

            res = OsmoseResults(out_dir)
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

            # Look for NetCDF files for spatial data
            nc_files = [f for f in res.list_outputs() if f.endswith(".nc")]
            if nc_files:
                spatial_ds.set(res.read_netcdf(nc_files[0]))
                max_t = spatial_ds.get().sizes.get("time", 1) - 1
                ui.update_slider("spatial_time_idx", max=max(max_t, 0))

            # Populate run comparison choices from history
            from osmose.history import RunHistory

            history_dir = out_dir.parent / ".osmose_history"
            if history_dir.is_dir():
                history = RunHistory(history_dir)
                runs = history.list_runs()
                choices = {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}
                ui.update_selectize("compare_runs_select", choices=choices)

            ui.notification_show("Results loaded successfully.", type="message", duration=3)

            state.results_loaded.set(True)
        except (OSError, ValueError, KeyError, ImportError, AttributeError, pd.errors.ParserError) as exc:
            _log.error("Failed to load results: %s", exc, exc_info=True)
            ui.notification_show(f"Error loading results: {exc}", type="error", duration=10)

    def _get_result_data(output_type: str) -> pd.DataFrame:
        """Load result data lazily — only when requested."""
        data = results_data.get()
        if output_type in data:
            return data[output_type]
        res = results_obj.get()
        if res is None:
            return pd.DataFrame()
        # Use explicit method mapping; fall back to export_dataframe for unknowns.
        method_name = _RESULT_METHODS.get(output_type)
        if method_name:
            method = getattr(res, method_name, None)
            df = method() if method is not None else res.export_dataframe(output_type)
        else:
            df = res.export_dataframe(output_type)
        # Cache it
        data = dict(data)
        data[output_type] = df
        results_data.set(data)
        return df

    @reactive.effect
    @reactive.event(input.btn_load_results)
    def _load_results():
        out_dir = Path(input.output_dir())
        if ".." in out_dir.parts:
            ui.notification_show("Invalid output directory path.", type="error")
            return
        if not out_dir.is_dir():
            ui.notification_show(f"Directory not found: {out_dir}", type="error", duration=5)
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
                    y_label=col_map.get(rtype, rtype),
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

        df = _get_result_data(rtype)
        value_col = col_map.get(rtype, rtype)
        title = title_map.get(rtype, rtype.title())

        # If the expected value column doesn't exist, try first numeric column
        if not df.empty and value_col not in df.columns:
            numeric_cols = df.select_dtypes(include="number").columns
            non_time = [c for c in numeric_cols if c != "time"]
            if non_time:
                value_col = non_time[0]

        return make_timeseries_chart(df, value_col, title, species=sp, template=tmpl)  # type: ignore[arg-type]

    @render_plotly
    def diet_chart():
        tmpl = _tpl(input)
        df = _get_result_data("diet")
        return make_diet_heatmap(df, template=tmpl)

    @render_plotly
    def spatial_chart():
        tmpl = _tpl(input)
        ds = spatial_ds.get()
        if ds is None:
            return go.Figure().update_layout(
                title="No spatial data loaded",
                template=tmpl,
            )
        time_idx = input.spatial_time_idx()
        # Find a suitable variable (prefer 'biomass')
        var_names = [v for v in ds.data_vars if "lat" in ds[v].dims and "lon" in ds[v].dims]
        if not var_names:
            return go.Figure().update_layout(
                title="No spatial variables found",
                template=tmpl,
            )
        var_name = "biomass" if "biomass" in var_names else var_names[0]
        max_t = ds.sizes.get("time", 1) - 1
        safe_idx = min(time_idx, max_t)
        return make_spatial_map(ds, var_name, time_idx=safe_idx, template=tmpl)

    @render_plotly
    def comparison_chart():
        tmpl = _tpl(input)
        selected = input.compare_runs_select()
        if not selected or len(selected) < 1:
            return go.Figure().update_layout(title="Select runs to compare", template=tmpl)

        from osmose.history import RunHistory
        from osmose.plotting import make_run_comparison

        out_dir = Path(input.output_dir())
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return go.Figure().update_layout(title="No run history found", template=tmpl)

        history = RunHistory(history_dir)
        records = [history.load_run(ts) for ts in selected]
        metric = input.compare_metric()
        fig = make_run_comparison(records, metrics=[metric])
        fig.update_layout(template=tmpl)
        return fig

    @render.ui
    def config_diff_table():
        selected = input.compare_runs_select()
        if not selected or len(selected) < 2:
            return ui.div("Select 2+ runs to see config differences.", style=STYLE_EMPTY)

        from osmose.history import RunHistory

        out_dir = Path(input.output_dir())
        history_dir = out_dir.parent / ".osmose_history"
        if not history_dir.is_dir():
            return ui.div("No run history found.")

        history = RunHistory(history_dir)
        diffs = history.compare_runs_multi(list(selected))

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

    @render.download(
        filename=lambda: (
            f"osmose_{input.result_type()}"
            + (f"_{input.result_species()}" if input.result_species() != "all" else "")
            + ".csv"
        )
    )
    def download_results_csv():
        from osmose.results import OsmoseResults
        import tempfile

        out_dir = Path(input.output_dir())
        if ".." in out_dir.parts:
            ui.notification_show("Invalid output directory path.", type="error")
            return
        if not out_dir.is_dir():
            ui.notification_show(
                "No output directory selected. Load results first.",
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
