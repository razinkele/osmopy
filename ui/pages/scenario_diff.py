"""Scenario Diff tab — side-by-side biomass + spatial maps for two runs.

Embedded as a tab in the Results page: ``scenario_diff_nav_panel()`` is added to the
Results navset and ``scenario_diff_server(...)`` is called once from ``results_server``.
This sub-server-in-a-page pattern is new (other pages are top-level), chosen to keep the
already-large ``results_server`` from growing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from shiny import reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly

from osmose.analysis import biomass_long, run_delta
from osmose.history import default_run_history
from osmose.logging import setup_logging
from osmose.plotting import make_biomass_overlay
from osmose.results import OsmoseResults
from osmose.spatial_series import grid_latlon, spatial_diff_2d
from ui.pages.grid_helpers import make_diff_map, make_spatial_map

_log = setup_logging("osmose.scenario_diff")

_SPATIAL_VAR_HINT = "spatial_biomass"  # preferred diff variable when present

# Display priority for the config-diff panel: changed group first, then added, then
# removed (NOT the change string's alphabetical order, which would put "added" first).
_CHANGE_ORDER = {"changed": 0, "added": 1, "removed": 2}


def _classify_config_diffs(diffs: list[dict]) -> list[dict]:
    """Tag each {key, value_a, value_b} row with a change type and sort for display.

    change is "added"   when value_a is None (key only in B),
              "removed" when value_b is None (key only in A),
              "changed" otherwise (both present, differ — incl. an empty-string value,
              since only None means a missing key).
    Sorted changed-group-first, then added, then removed; alphabetical by key within
    each group. Deterministic and independent of input order. Pure (no I/O).
    """
    rows: list[dict] = []
    for d in diffs:
        va = d.get("value_a")
        vb = d.get("value_b")
        if va is None:
            change = "added"
        elif vb is None:
            change = "removed"
        else:
            change = "changed"
        rows.append({"key": d["key"], "value_a": va, "value_b": vb, "change": change})
    rows.sort(key=lambda r: (_CHANGE_ORDER[r["change"]], r["key"]))
    return rows


def _run_choices(runs) -> dict[str, str]:
    return {r.timestamp: f"{r.timestamp[:19]} ({r.duration_sec:.0f}s)" for r in runs}


def _tpl(input) -> str:
    from ui.state import get_theme_mode

    return "osmose" if get_theme_mode(input) == "dark" else "osmose-light"


def scenario_diff_nav_panel():
    """The 'Scenario Diff' nav panel for embedding in the Results navset."""
    return ui.nav_panel(
        "Scenario Diff",
        ui.layout_columns(
            ui.div(
                ui.input_select("diff_run_a", "Baseline (A)", choices={}),
                ui.input_select("diff_run_b", "Variant (B)", choices={}),
                ui.input_selectize("diff_species", "Biomass species", choices={}, multiple=True),
                ui.input_slider(
                    "diff_window_years", "Caption window (years)", min=1, max=30, value=10
                ),
            ),
            col_widths=[12],
        ),
        output_widget("diff_biomass_chart"),
        ui.output_ui("diff_biomass_caption"),
        ui.hr(),
        ui.output_ui("diff_spatial_controls"),
        ui.output_ui("diff_cadence_warning"),
        ui.output_ui("diff_spatial_status"),
        # Map widgets are declared STATICALLY (shinywidgets binds on DOM insertion;
        # injecting output_widget inside a @render.ui leaves them blank). Each render
        # function returns an empty-state figure when its dataset is absent.
        ui.layout_columns(
            output_widget("diff_map_a"),
            output_widget("diff_map_b"),
            output_widget("diff_map_delta"),
            col_widths=[4, 4, 4],
        ),
    )


def scenario_diff_server(input, output, session, state):
    """Reactives for the Scenario Diff tab. Called once from results_server."""
    _ds_a: reactive.Value = reactive.Value(None)  # xarray Dataset | None
    _ds_b: reactive.Value = reactive.Value(None)
    _shared_ds: reactive.Value[bool] = reactive.Value(False)  # A and B are the SAME handle
    _last_choices: reactive.Value[dict] = reactive.Value({})
    _last_species: reactive.Value[dict] = reactive.Value({})

    # NB: render functions use BARE @render_plotly / @render.ui (no @output) to match
    # the host pages (ui/pages/results.py, ui/pages/spatial_results.py). The `output`
    # server param is unused but kept in the signature (the caller passes it positionally).

    def _safe(getter, default=None):
        # Catch AttributeError too: dynamic inputs (diff_spatial_species, diff_time)
        # are created inside a @render.ui, so a static map render can read them before
        # they register. This matches ui/pages/spatial_results.py:100,316,457.
        try:
            return getter()
        except (SilentException, AttributeError):
            return default

    # ── Populate run selectors from history when on the Results tab ──
    @reactive.effect
    def _populate_diff_runs():
        if input.main_nav() != "results":
            return
        try:
            runs = default_run_history().list_runs()
        except Exception:  # noqa: BLE001 — never crash the page on a history-read error
            return
        choices = _run_choices(runs)
        with reactive.isolate():
            if choices == _last_choices.get():
                return
        _last_choices.set(choices)
        ui.update_select("diff_run_a", choices=choices)
        ui.update_select("diff_run_b", choices=choices)

    # ── Resolve a selected timestamp → OsmoseResults ──
    def _results_for(ts):
        if not ts:
            return None
        try:
            rec = default_run_history().load_run(ts)
        except Exception:  # noqa: BLE001
            return None
        return OsmoseResults(Path(rec.output_dir), strict=False)

    # ── Spatial NetCDF lifecycle ──
    # ONE effect opens both runs, deduping the same-run case: opening the SAME .nc path
    # twice would trigger the documented HDF5-locking error, and selecting one run for
    # both A and B is a supported case (the "identical runs" caption). When A == B we
    # share a single handle and never double-close it. Handles are opened with
    # xr.open_dataset (the holder owns the only reference — no cache-then-close smell).
    def _close_one(ds):
        if ds is not None:
            try:
                ds.close()
            except Exception:  # noqa: BLE001
                _log.warning("Failed to close scenario-diff dataset", exc_info=True)

    def _close_handles():
        """Close held handles WITHOUT touching reactive values (safe at session end)."""
        a = _ds_a.get()
        _close_one(a)
        if not _shared_ds.get():
            _close_one(_ds_b.get())

    def _open_one(res):
        if res is None or res.output_dir is None:
            return None
        try:
            nc = [f for f in res.list_outputs() if f.endswith(".nc")]
        except (OSError, ValueError, KeyError):
            return None
        if not nc:
            return None
        try:
            return xr.open_dataset(Path(res.output_dir) / nc[0])
        except (OSError, ValueError, KeyError) as exc:
            _log.error("Failed to open spatial output: %s", exc, exc_info=True)
            return None

    @reactive.effect
    def _load_spatial():
        ts_a = _safe(input.diff_run_a)
        ts_b = _safe(input.diff_run_b)
        with reactive.isolate():  # close prior handles without depending on them
            _close_handles()
        ds_a = _open_one(_results_for(ts_a))
        if ts_a and ts_a == ts_b:
            ds_b = ds_a  # same run → share the single handle (no double-open)
            shared = True
        else:
            ds_b = _open_one(_results_for(ts_b))
            shared = False
        _shared_ds.set(shared)
        _ds_a.set(ds_a)
        _ds_b.set(ds_b)

    # Close handles when the session ends (we may hold two long-lived datasets).
    def _on_session_end():
        with reactive.isolate():
            _close_handles()

    session.on_ended(_on_session_end)

    # ── Biomass long frames (reactive) ──
    @reactive.calc
    def _long_a():
        res = _results_for(_safe(input.diff_run_a))
        return biomass_long(res) if res is not None else None

    @reactive.calc
    def _long_b():
        res = _results_for(_safe(input.diff_run_b))
        return biomass_long(res) if res is not None else None

    # ── Biomass species selector population (common to both runs) ──
    @reactive.effect
    def _populate_diff_species():
        la, lb = _long_a(), _long_b()
        if la is None or lb is None:
            return
        common = sorted(set(la["species"]) & set(lb["species"]))
        choices = {s: s for s in common}
        with reactive.isolate():
            # Changed-only guard: don't re-run update_selectize (and clobber the user's
            # manual selection) when the common species set hasn't changed.
            if choices == _last_species.get():
                return
            current = _safe(input.diff_species, ()) or ()
        _last_species.set(choices)
        keep = [s for s in current if s in choices]
        ui.update_selectize("diff_species", choices=choices, selected=keep or common[:3])

    # ── Biomass overlay chart ──
    @render_plotly
    def diff_biomass_chart():
        la, lb = _long_a(), _long_b()
        if la is None or lb is None:
            return go.Figure().update_layout(
                title="Select two runs to compare", template=_tpl(input)
            )
        species = list(_safe(input.diff_species, ()) or ())
        fig = make_biomass_overlay(la, lb, species)
        fig.update_layout(template=_tpl(input))
        return fig

    # ── Biomass caption (mean B−A over the trailing window) ──
    @render.ui
    def diff_biomass_caption():
        ts_a, ts_b = _safe(input.diff_run_a), _safe(input.diff_run_b)
        ra, rb = _results_for(ts_a), _results_for(ts_b)
        if ra is None or rb is None:
            return ui.div()
        if ts_a == ts_b:
            return ui.p("Identical runs (A = B).", class_="text-muted")
        try:
            deltas = run_delta(
                ra, rb, metric="biomass", window_years=int(input.diff_window_years())
            )
        except (ValueError, KeyError):
            return ui.div()
        species = set(_safe(input.diff_species, ()) or ())
        rows = [d for d in deltas if not species or d.species in species]
        if not rows:
            return ui.div()
        items = [ui.tags.li(f"{d.species}: ΔB = {d.abs_delta:+.3g}") for d in rows]
        return ui.div(ui.p("Mean B−A over trailing window:"), ui.tags.ul(*items))

    # ── Spatial: common species + variable helpers ──
    def _has_latlon(ds, v):
        dims = {str(d) for d in ds[v].dims}
        return "lat" in dims and "lon" in dims

    def _spatial_var(ds):
        candidates = [v for v in ds.data_vars if _has_latlon(ds, v)]
        if _SPATIAL_VAR_HINT in candidates:
            return _SPATIAL_VAR_HINT
        return candidates[0] if candidates else None

    def _common_species(ds_a, ds_b):
        sa = {str(s) for s in ds_a["species"].values} if "species" in ds_a.coords else set()
        sb = {str(s) for s in ds_b["species"].values} if "species" in ds_b.coords else set()
        return sorted(sa & sb)

    def _has_species_dim(ds, var):
        return "species" in {str(d) for d in ds[var].dims}

    def _spatial_empty_reason(a, b):
        """Message if the spatial maps can't be rendered comparably, else None.

        Crucially guards the summed path when there is NO common species: feeding
        ``ds.sel(species=[])`` into ``spatial_slice_2d``'s ``sum(skipna=False)`` over a
        zero-length dim yields 0.0 everywhere (silently destroying the land-NaN mask),
        so we short-circuit to an empty state instead.
        """
        if a is None or b is None:
            return "No spatial output — enable output.spatial in both configs."
        var_a, var_b = _spatial_var(a), _spatial_var(b)
        if var_a is None or var_b is None:
            return "No spatial variable in one of the runs."
        if _spatial_species() is None:  # "All (summed)" path
            if (_has_species_dim(a, var_a) or _has_species_dim(b, var_b)) and not _common_species(
                a, b
            ):
                return "No common species for spatial maps."
        return None

    # ── Spatial controls (species + time), rendered dynamically ──
    # (Dynamic input_select/input_slider inside @render.ui is fine — only shinywidgets
    # output_widget must be static; see the nav panel.)
    @render.ui
    def diff_spatial_controls():
        a, b = _ds_a.get(), _ds_b.get()
        if a is None or b is None:
            return ui.div()
        common = _common_species(a, b)
        choices = {"__sum__": "All (summed)"}
        choices.update({s: s for s in common})
        # Overlapping time range by VALUE; slider indexes into a shared 0..N-1 fraction
        n_a = int(a.sizes.get("time", 1))
        n_b = int(b.sizes.get("time", 1))
        n = min(n_a, n_b)
        return ui.div(
            ui.input_select(
                "diff_spatial_species", "Map species", choices=choices, selected="__sum__"
            ),
            ui.input_slider("diff_time", "Time step", min=0, max=max(n - 1, 0), value=0, step=1),
        )

    # ── Cadence-mismatch warning (spec: warn if n_dt_per_year differs) ──
    @render.ui
    def diff_cadence_warning():
        a, b = _ds_a.get(), _ds_b.get()
        if a is None or b is None:
            return ui.div()
        na = a.attrs.get("n_dt_per_year")
        nb = b.attrs.get("n_dt_per_year")
        if na is not None and nb is not None and na != nb:
            return ui.p(
                f"⚠ Runs have different time cadence ({na} vs {nb} steps/year); "
                "maps are aligned by nearest time value.",
                class_="text-warning",
            )
        return ui.div()

    # ── Spatial empty-state message (maps render empty figures alongside) ──
    @render.ui
    def diff_spatial_status():
        reason = _spatial_empty_reason(_ds_a.get(), _ds_b.get())
        return ui.p(reason, class_="text-muted") if reason else ui.div()

    def _spatial_species():
        sel = _safe(input.diff_spatial_species, "__sum__")
        return None if sel in ("__sum__", None) else sel

    def _time_indices(a, b):
        """Nearest indices in A and B for the chosen overlapping-time value."""
        ta = np.asarray(a["time"].values)
        tb = np.asarray(b["time"].values)
        lo = max(float(ta.min()), float(tb.min()))
        hi = min(float(ta.max()), float(tb.max()))
        idx = int(_safe(input.diff_time, 0) or 0)
        # Map the integer slider position onto the overlapping value range, then snap.
        n = min(len(ta), len(tb))
        frac = idx / max(n - 1, 1)
        v = lo + frac * (hi - lo)
        return int(np.abs(ta - v).argmin()), int(np.abs(tb - v).argmin())

    # ── Three spatial maps (static widgets; empty-state figure when no data) ──
    def _subset_for_sum(ds, var, common):
        """Narrow a dataset to the common species (so 'All (summed)' is comparable).

        Only reached when ``common`` is non-empty — ``_spatial_empty_reason`` short-
        circuits the empty case before any map computes, so ``ds.sel(species=[])``
        (which would zero the grid and destroy the land-NaN mask) never runs.
        """
        if _has_species_dim(ds, var):
            return ds.sel(species=common)
        return ds

    def _state_fig(msg):
        return go.Figure().update_layout(title=msg, template=_tpl(input))

    @render_plotly
    def diff_map_a():
        return _one_side_map(_ds_a.get(), _ds_b.get(), which="a")

    @render_plotly
    def diff_map_b():
        return _one_side_map(_ds_a.get(), _ds_b.get(), which="b")

    def _one_side_map(a, b, *, which):
        reason = _spatial_empty_reason(a, b)
        if reason:
            return _state_fig(reason)
        var = _spatial_var(a)
        if var is None:
            return _state_fig("No spatial variable in one of the runs.")
        common = _common_species(a, b)
        sp = _spatial_species()
        ti_a, ti_b = _time_indices(a, b)
        ds = a if which == "a" else b
        ti = ti_a if which == "a" else ti_b
        label = "A" if which == "a" else "B"
        # Real time value for this side at the aligned index (species subsetting
        # leaves the time dim untouched, so look it up on the original a/b).
        t_val = float((a if which == "a" else b)["time"].values[ti])
        if sp is None:
            ds = _subset_for_sum(ds, var, common)
        try:
            fig = make_spatial_map(
                ds, var, time_idx=ti, species=sp, title=f"{label}: {var} (t={t_val:.3g})"
            )
        except (ValueError, KeyError) as exc:
            return _state_fig(f"Cannot render {label}: {exc}")
        fig.update_layout(template=_tpl(input))
        return fig

    @render_plotly
    def diff_map_delta():
        a, b = _ds_a.get(), _ds_b.get()
        reason = _spatial_empty_reason(a, b)
        if reason:
            return _state_fig(reason)
        var = _spatial_var(a)
        if var is None:
            return _state_fig("No spatial variable in one of the runs.")
        common = _common_species(a, b)
        sp = _spatial_species()
        ti_a, ti_b = _time_indices(a, b)
        ds_a, ds_b = a, b
        if sp is None:
            ds_a = _subset_for_sum(a, var, common)
            ds_b = _subset_for_sum(b, var, common)
        try:
            diff = spatial_diff_2d(ds_a, ds_b, var, time_a=ti_a, time_b=ti_b, species=sp)
        except (ValueError, TypeError, KeyError) as exc:
            return _state_fig(f"Cannot diff: {exc}")
        lat, lon = grid_latlon(ds_a, var)
        t_b_val = float(b["time"].values[ti_b])
        return make_diff_map(
            diff,
            lat,
            lon,
            var_name=var,
            title=f"Δ {var} (B−A, t={t_b_val:.3g})",
            template=_tpl(input),
        )
