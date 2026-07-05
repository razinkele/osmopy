"""Fisheries stock-status page: F/M bars, F/Fmsy and B/Bmsy time-series, Kobe quadrant."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import output_widget, render_plotly

from osmose.logging import setup_logging
from osmose.validation.fisheries_reference import (
    ReferencePoint,
    ecosystem_of,
    load_reference_points,
    save_reference_points,
)
from osmose.validation.stock_status import StockStatus, compute_stock_status
from ui.state import AppState
from ui.styles import STYLE_EMPTY

_log = setup_logging("osmose.fisheries.ui")


def _ices_snapshot_dir(ecosystem: str) -> Path | None:
    """Return the bundled ICES snapshot dir for *ecosystem* if it exists, else None."""
    candidate = Path("data") / ecosystem / "reference" / "ices_snapshots"
    return candidate if candidate.is_dir() else None


_DISCLAIMER = (
    "**Disclaimer:** These are indicative stock-status metrics relative to user-supplied "
    "or ICES-auto-filled reference points. They are **not** a formal stock assessment. "
    "No uncertainty or retrospective pattern analysis is performed."
)

_SOURCE_NOTE = (
    "Reference points may be model-internal — run "
    "`scripts/compute_model_reference_points.py --config <cfg>` to (re)generate; "
    "ICES-auto-filled; or user-entered. "
    "Precedence: user > model > ICES."
)


# ---------------------------------------------------------------------------
# Pure view-assembly helper (unit-tested; no Shiny dependencies)
# ---------------------------------------------------------------------------


def build_fisheries_view(
    results,
    config,
    ecosystem: str,
    *,
    ices_snapshot_dir: Path | None = None,
) -> dict:
    """Assemble what the Fisheries page renders.

    Parameters
    ----------
    results:
        An OsmoseResults-like object (has ``.ssb(species)``), or None when
        no run has been loaded yet.
    config:
        An EngineConfig-like object (has ``.species_names``), or None.
    ecosystem:
        Ecosystem basename string (e.g. ``"baltic"``).
    ices_snapshot_dir:
        Optional path to an ICES SAG snapshot directory.

    Returns
    -------
    dict with keys:

    - ``lead`` – which panel to show first: always ``"fm_bars"``.
    - ``kobe_ready`` – True only when ≥1 species has both B/Bmsy and F/Fmsy ratios.
    - ``kobe_cta`` – call-to-action text to show when ``kobe_ready`` is False.
    - ``statuses`` – list of :class:`StockStatus`.
    - ``unmatched`` – species keys in the sidecar JSON with no matching config species.
    - ``save_target`` – string path (``data/<ecosystem>/reference``), or None when
      results is None.
    """
    view: dict = {
        "lead": "fm_bars",
        "kobe_ready": False,
        "kobe_cta": ("Enter a Bmsy for >=1 species in the table to populate the Kobe quadrant."),
        "statuses": [],
        "unmatched": [],
        "save_target": None,
    }
    if results is None or config is None:
        return view

    ref_dir = Path("data") / ecosystem / "reference"
    view["save_target"] = str(ref_dir)

    species = list(config.species_names)
    refs, unmatched = load_reference_points(ref_dir, species, ices_snapshot_dir=ices_snapshot_dir)
    view["unmatched"] = unmatched

    statuses = compute_stock_status(results, refs, config, species_list=species)
    view["statuses"] = statuses
    view["kobe_ready"] = any(s.latest_quadrant is not None for s in statuses)
    return view


# ---------------------------------------------------------------------------
# Shiny UI
# ---------------------------------------------------------------------------


def fisheries_ui():
    """Return the Fisheries page UI layout."""
    return ui.div(
        ui.h4("Fisheries Stock Status"),
        ui.p(ui.markdown(_DISCLAIMER), class_="text-muted small"),
        ui.p(
            ui.markdown(
                "**Tip:** For community-level size-spectrum and mean trophic level indicators, "
                "see the **Results** page (select *Size Spectrum* or *Mean Trophic Level* "
                "from the output type dropdown)."
            ),
            class_="text-muted small",
        ),
        ui.hr(),
        # Top row: F/M bars (zero-config) + F/Fmsy timeseries
        ui.layout_columns(
            ui.card(
                ui.card_header("F / M — Fishing vs Natural Mortality"),
                output_widget("fisheries_fm_bars"),
            ),
            ui.card(
                ui.card_header("F / Fmsy time-series"),
                output_widget("fisheries_f_ratio_ts"),
            ),
            col_widths=[6, 6],
        ),
        # Kobe panel (gated) or CTA
        ui.output_ui("fisheries_kobe_panel"),
        # Reference-point editor
        ui.card(
            ui.card_header("Reference Points (per species)"),
            ui.output_ui("fisheries_ref_table"),
            ui.output_ui("fisheries_save_btn"),
        ),
        # Unmatched warning
        ui.output_ui("fisheries_unmatched_warn"),
    )


# ---------------------------------------------------------------------------
# Shiny Server
# ---------------------------------------------------------------------------


def fisheries_server(input, output, session, state: AppState):
    """Fisheries page server: wires reactive computations to the UI outputs."""

    # ------------------------------------------------------------------
    # Reactive calc: build view when navigating to this tab
    # ------------------------------------------------------------------

    @reactive.calc
    def _view() -> dict:
        if input.main_nav() != "fisheries":
            return build_fisheries_view(None, None, "unknown")

        with reactive.isolate():
            raw_config = state.config.get()
            out_dir = state.output_dir.get()
            config_dir = state.config_dir.get()

        if not raw_config or not out_dir:
            return build_fisheries_view(None, None, "unknown")

        try:
            from osmose.engine.config import EngineConfig
            from osmose.results import OsmoseResults

            config = EngineConfig.from_dict(raw_config)
            results = OsmoseResults(Path(out_dir), strict=False)
            eco = ecosystem_of(config_dir)
            snapshot_dir = _ices_snapshot_dir(eco)
            return build_fisheries_view(results, config, eco, ices_snapshot_dir=snapshot_dir)
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            _log.warning("Fisheries view error: %s", exc)
            return build_fisheries_view(None, None, "unknown")

    # ------------------------------------------------------------------
    # F/M bar chart (zero-config: only needs mortalityRate CSV output)
    # ------------------------------------------------------------------

    @render_plotly
    def fisheries_fm_bars():
        if input.main_nav() != "fisheries":
            return go.Figure().update_layout(title="Navigate to Fisheries tab to load")

        with reactive.isolate():
            raw_config = state.config.get()
            out_dir = state.output_dir.get()

        if not raw_config or not out_dir:
            return go.Figure().update_layout(title="Run a simulation first", template="osmose")

        try:
            from osmose.engine.config import EngineConfig
            from osmose.plotting import make_fm_ratio_bars
            from osmose.validation.fisheries import compute_mortality_balance

            config = EngineConfig.from_dict(raw_config)
            ndt = int(raw_config.get("simulation.time.ndtperyear", "12"))
            freq = int(raw_config.get("output.recordfrequency.ndt", "1"))
            steps_per_year = max(1, ndt // freq)
            species = list(config.species_names)
            balances = compute_mortality_balance(
                Path(out_dir),
                prefix="osm",
                species_list=species,
                steps_per_year=steps_per_year,
            )
            return make_fm_ratio_bars(balances)
        except Exception as exc:  # noqa: BLE001
            _log.warning("F/M chart error: %s", exc)
            return go.Figure().update_layout(title=f"F/M unavailable: {exc}", template="osmose")

    # ------------------------------------------------------------------
    # F/Fmsy time-series
    # ------------------------------------------------------------------

    @render_plotly
    def fisheries_f_ratio_ts():
        view = _view()
        statuses: list[StockStatus] = view["statuses"]
        if not statuses:
            return go.Figure().update_layout(
                title="F/Fmsy — supply Fmsy reference points", template="osmose"
            )
        from osmose.plotting import make_ratio_timeseries

        return make_ratio_timeseries(statuses, "f")

    # ------------------------------------------------------------------
    # Kobe panel (gated)
    # ------------------------------------------------------------------

    @render.ui
    def fisheries_kobe_panel():
        view = _view()
        if not view["kobe_ready"]:
            return ui.card(
                ui.card_header("Kobe Quadrant"),
                ui.p(view["kobe_cta"], style=STYLE_EMPTY),
            )
        return ui.card(
            ui.card_header("Kobe Quadrant"),
            output_widget("fisheries_kobe"),
            output_widget("fisheries_b_ratio_ts"),
        )

    @render_plotly
    def fisheries_kobe():
        view = _view()
        statuses = view["statuses"]
        if not statuses:
            return go.Figure()
        from osmose.plotting import make_kobe_plot

        return make_kobe_plot(statuses)

    @render_plotly
    def fisheries_b_ratio_ts():
        view = _view()
        statuses = view["statuses"]
        if not statuses:
            return go.Figure()
        from osmose.plotting import make_ratio_timeseries

        return make_ratio_timeseries(statuses, "b")

    # ------------------------------------------------------------------
    # Reference-point editor table
    # ------------------------------------------------------------------

    @render.ui
    def fisheries_ref_table():
        view = _view()
        statuses: list[StockStatus] = view["statuses"]
        if not statuses:
            return ui.p("Load a run to edit reference points.", style=STYLE_EMPTY)

        with reactive.isolate():
            raw_config = state.config.get()
            out_dir = state.output_dir.get()
            config_dir = state.config_dir.get()

        # Build mean SSB for each species as a scale hint
        ssb_means: dict[str, str] = {}
        if raw_config and out_dir:
            try:
                from osmose.results import OsmoseResults

                results = OsmoseResults(Path(out_dir), strict=False)
                for st in statuses:
                    sdf = results.ssb(st.species)
                    if st.species in sdf.columns:
                        mean_ssb = float(sdf[st.species].mean())
                        ssb_means[st.species] = f"{mean_ssb:,.0f} t"
            except Exception as e:  # noqa: BLE001
                _log.warning("SSB scale-hint unavailable: %s", e)

        # Collect reference points to show their source in the table
        with reactive.isolate():
            _rp_dir = Path("data") / ecosystem_of(config_dir) / "reference"
        try:
            _refs, _ = load_reference_points(_rp_dir, [st.species for st in statuses])
        except Exception:  # noqa: BLE001
            _refs = {}

        rows = []
        for st in statuses:
            sid = st.species.replace(".", "_")
            bmsy_id = f"bmsy_{sid}"
            fmsy_id = f"fmsy_{sid}"
            ssb_hint = ssb_means.get(st.species, "—")
            rp = _refs.get(st.species)
            source_label = rp.source if rp is not None else "none"
            rows.append(
                ui.tags.tr(
                    ui.tags.td(ui.tags.strong(st.species)),
                    ui.tags.td(f"mean SSB ≈ {ssb_hint}"),
                    ui.tags.td(ui.input_numeric(bmsy_id, label=None, value=None, min=0, step=1000)),
                    ui.tags.td(ui.input_numeric(fmsy_id, label=None, value=None, min=0, step=0.01)),
                    ui.tags.td(
                        ui.tags.small(
                            st.takeaway or ("—" if not st.caveats else "; ".join(st.caveats)),
                            class_="text-muted",
                        )
                    ),
                    ui.tags.td(ui.tags.small(source_label, class_="text-muted font-monospace")),
                )
            )

        eco = ecosystem_of(config_dir)
        return ui.div(
            ui.p(
                ui.tags.small(ui.markdown(_SOURCE_NOTE), class_="text-muted"),
            ),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Species"),
                        ui.tags.th("Scale hint"),
                        ui.tags.th("Bmsy (t)"),
                        ui.tags.th("Fmsy (yr⁻¹)"),
                        ui.tags.th("Status"),
                        ui.tags.th("Source"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm table-striped",
                style="font-size: 13px;",
            ),
            ui.p(
                ui.tags.small(
                    f"Shared across all '{eco}' runs — "
                    f"saved to {view.get('save_target', 'data/<ecosystem>/reference')}",
                    class_="text-muted",
                )
            ),
        )

    # ------------------------------------------------------------------
    # Save button
    # ------------------------------------------------------------------

    @render.ui
    def fisheries_save_btn():
        view = _view()
        save_target = view.get("save_target")
        if not save_target:
            return ui.div()
        return ui.div(
            ui.input_action_button(
                "btn_save_ref_points",
                "Save Reference Points",
                class_="btn-primary",
            ),
        )

    @reactive.effect
    @reactive.event(input.btn_save_ref_points)
    def _save_ref_points():
        view = _view()
        save_target = view.get("save_target")
        statuses: list[StockStatus] = view["statuses"]
        if not save_target or not statuses:
            return

        # Collect user values from numeric inputs
        # Input IDs use sanitized species names (dots → underscores); map back to real names.
        updated_refs: dict[str, ReferencePoint] = {}
        for st in statuses:
            sid = st.species.replace(".", "_")
            bmsy_val = None
            fmsy_val = None
            try:
                bmsy_val = float(input[f"bmsy_{sid}"]())
            except Exception:  # noqa: BLE001
                pass
            try:
                fmsy_val = float(input[f"fmsy_{sid}"]())
            except Exception:  # noqa: BLE001
                pass

            rp = ReferencePoint(species=st.species)
            if bmsy_val and bmsy_val > 0:
                rp.bmsy = bmsy_val
                rp.b_ref_kind = "bmsy_user"
            if fmsy_val and fmsy_val > 0:
                rp.fmsy = fmsy_val
                rp.source = "user"
            updated_refs[st.species] = rp

        try:
            save_reference_points(Path(save_target), updated_refs)
            ui.notification_show(
                f"Reference points saved to {save_target}.", type="message", duration=4
            )
        except OSError as exc:
            ui.notification_show(f"Save failed: {exc}", type="error", duration=10)

    # ------------------------------------------------------------------
    # Unmatched sidecar keys warning
    # ------------------------------------------------------------------

    @render.ui
    def fisheries_unmatched_warn():
        view = _view()
        unmatched: list[str] = view.get("unmatched", [])
        if not unmatched:
            return ui.div()
        return ui.div(
            ui.tags.small(
                ui.tags.strong("Warning: "),
                f"The reference sidecar JSON contains keys not in the current config: "
                f"{', '.join(unmatched)}. These may be stale or mis-spelled.",
                class_="text-warning",
            )
        )
