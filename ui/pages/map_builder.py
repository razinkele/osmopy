"""Map Builder page — draw/paint spatial maps (distribution / mask / zone) onto the grid.

The pure map-editing core lives in ``osmose.maps.builder`` (GridSpec / MapGrid / save_map);
this module is the Shiny wiring only.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
from shiny import reactive, render, ui
from shiny.types import SilentException

from shiny_deckgl import (  # type: ignore[import-untyped]
    CARTO_DARK,
    CARTO_POSITRON,
    MapWidget,
    compass_widget,
    fullscreen_widget,
    polygon_layer,
    scale_widget,
    zoom_widget,
)

from osmose.logging import setup_logging
from osmose.maps.builder import (
    GridSpec,
    MapGrid,
    from_csv_text,
    lonlat_to_cell,
    save_map,
    validate,
)
from ui.components.collapsible import collapsible_card_header, expand_tab
from ui.components.renderer_badge import renderer_badge
from ui.pages.grid_helpers import (
    _find_config_file,
    _zoom_for_span,
    build_grid_layers,
    load_mask,
)
from ui.state import get_theme_mode

_log = setup_logging("osmose.map_builder")

_DEFAULT_VIEW_STATE = {"latitude": 46.0, "longitude": -4.5, "zoom": 5, "pitch": 0, "bearing": 0}

# MapWidget id — determines input names input.mb_map_map_click / input.mb_map_drawn_features.
_MAP_ID = "mb_map"


def _species_choices(cfg: dict[str, str]) -> list[str]:
    """Return the ordered list of species names from a config dict."""
    try:
        n = int(float(cfg.get("simulation.nspecies", "0") or "0"))
    except (ValueError, TypeError):
        return []
    out: list[str] = []
    for i in range(n):
        name = cfg.get(f"species.name.sp{i}")
        if name:
            out.append(name)
    return out


def _existing_maps(cfg: dict[str, str]) -> list[tuple[str, str]]:
    """Discover existing on-disk map files referenced by a config.

    Returns ``(label, rel_path)`` pairs for movement distribution maps
    (``movement.file.map{N}``, N=0,1,...) and, if set, the land mask
    (``grid.mask.file``).  Returns ``[]`` if none are referenced.
    """
    out: list[tuple[str, str]] = []
    n = 0
    while True:
        rel = cfg.get(f"movement.file.map{n}")
        if not rel:
            break
        species = cfg.get(f"movement.species.map{n}", "?")
        out.append((f"{species} — {rel} (map{n})", rel))
        n += 1
    mask = cfg.get("grid.mask.file")
    if mask:
        out.append((f"land mask — {mask}", mask))
    return out


def map_builder_ui():
    builder_map = MapWidget(
        f"{_MAP_ID}",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
        tooltip={"html": "Value: {properties.value}", "style": {"fontSize": "12px"}},
        controls=[],
    )

    tools_card = ui.card(
        collapsible_card_header("Map Builder", "map_builder"),
        ui.output_ui("mb_hint"),
        ui.input_radio_buttons(
            "map_type",
            "Map type",
            choices={
                "distribution": "Distribution (movement)",
                "mask": "Land mask",
                "zone": "Generic zone",
            },
            selected="distribution",
        ),
        ui.input_radio_buttons(
            "tool_mode",
            "Tool",
            choices={
                "polygon": "Polygon draw",
                "brush": "Brush",
                "eraser": "Eraser",
                "mask": "Mask edit",
            },
            selected="polygon",
        ),
        ui.input_numeric("paint_value", "Paint value", value=1, step=1),
        ui.div(
            ui.input_action_button("mb_new_blank", "New blank map", class_="btn-sm"),
            ui.output_ui("mb_load_existing"),
            class_="d-flex flex-column gap-2",
        ),
        ui.output_ui("mb_staged_indicator"),
        ui.input_action_button("apply_polygons", "Apply polygon(s)", class_="btn-sm btn-primary"),
        ui.output_ui("mb_applicability"),
        ui.hr(),
        ui.input_text("mb_filename", "Filename", placeholder="my_map.csv"),
        ui.input_action_button("mb_save", "Save map", class_="btn-sm btn-success"),
    )

    return ui.div(
        expand_tab("Map Builder", "map_builder"),
        ui.layout_columns(
            tools_card,
            ui.div(
                builder_map.ui(height="100%"),
                renderer_badge(),
                class_="osm-grid-map-container",
            ),
            col_widths=[5, 7],
        ),
        class_="osm-split-layout",
        id="split_map_builder",
    )


def map_builder_server(input, output, session, state):
    _map = MapWidget(
        f"{_MAP_ID}",
        view_state=_DEFAULT_VIEW_STATE,
        style=CARTO_POSITRON,
    )

    # --- editing state -----------------------------------------------------
    grid_array: reactive.Value[MapGrid | None] = reactive.Value(None)
    base_mask: reactive.Value[np.ndarray | None] = reactive.Value(None)
    dirty: reactive.Value[int] = reactive.Value(0)
    staged: reactive.Value[dict | None] = reactive.Value(None)

    def _grid_spec() -> GridSpec | None:
        """Build a GridSpec from the active config, or None if no usable grid."""
        with reactive.isolate():
            cfg = state.config.get()
        if not cfg:
            return None
        try:
            return GridSpec.from_config(cfg)
        except (KeyError, ValueError, TypeError):
            return None

    def _paint_value() -> float:
        try:
            return float(input.paint_value() or 0)
        except (SilentException, ValueError, TypeError):
            return 0.0

    def _new_blank() -> None:
        grid = _grid_spec()
        if grid is None:
            return
        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()
        mask = load_mask(cfg, config_dir=cfg_dir)
        base_mask.set(mask)
        grid_array.set(MapGrid.blank(grid, base_mask=mask))
        staged.set(None)
        dirty.set(dirty.get() + 1)

    @render.ui
    def mb_hint():
        state.load_trigger.get()
        grid = _grid_spec()
        if grid is None:
            return ui.p(
                "Load a configuration with a regular lon/lat grid to build maps.",
                style="color: var(--osm-text-muted); padding: 8px;",
            )
        return ui.div()

    @render.ui
    def mb_load_existing():
        state.load_trigger.get()
        with reactive.isolate():
            cfg = state.config.get()
        maps = _existing_maps(cfg) if cfg else []
        if not maps:
            return ui.p(
                "No existing maps in this config.",
                style="color: var(--osm-text-muted); font-size: 12px; margin: 0;",
            )
        return ui.div(
            ui.input_select(
                "mb_existing_choice",
                None,
                choices={rel: label for label, rel in maps},
            ),
            ui.input_action_button("mb_load_existing_btn", "Load into editor", class_="btn-sm"),
            class_="d-flex flex-column gap-1",
        )

    @render.ui
    def mb_staged_indicator():
        fc = staged.get()
        n = len(fc.get("features", [])) if isinstance(fc, dict) else 0
        if not n:
            return ui.div()
        return ui.p(
            f"{n} polygon(s) staged — click 'Apply polygon(s)'.",
            style="color: var(--osm-accent); font-size: 12px; margin: 4px 0;",
        )

    @render.ui
    def mb_applicability():
        """Distribution applicability form — only shown for distribution maps."""
        try:
            mtype = input.map_type()
        except SilentException:
            mtype = "distribution"
        if mtype != "distribution":
            return ui.div()

        with reactive.isolate():
            cfg = state.config.get()
        species = _species_choices(cfg)
        ndt = 0
        try:
            ndt = int(float(cfg.get("simulation.time.ndtperyear", "0") or "0"))
        except (ValueError, TypeError):
            ndt = 0
        all_steps = ";".join(str(i) for i in range(ndt)) if ndt else ""

        return ui.div(
            ui.input_select(
                "mb_species",
                "Species",
                choices={s: s for s in species} if species else {},
            ),
            ui.layout_columns(
                ui.input_numeric("mb_initialage", "Initial age", value=0, min=0),
                ui.input_numeric("mb_lastage", "Last age", value=None, min=0),  # type: ignore[arg-type]
                col_widths=[6, 6],
            ),
            ui.input_text("mb_steps", "Season steps", value=all_steps, placeholder="all"),
            ui.layout_columns(
                ui.input_numeric("mb_initialyear", "Initial year", value=None, min=0),  # type: ignore[arg-type]
                ui.input_numeric("mb_lastyear", "Last year", value=None, min=0),  # type: ignore[arg-type]
                col_widths=[6, 6],
            ),
            class_="osm-applicability-form",
        )

    # --- new-blank trigger -------------------------------------------------
    @reactive.effect
    @reactive.event(input.mb_new_blank, ignore_none=False)
    def _on_new_blank():
        _new_blank()

    # --- load-existing trigger ---------------------------------------------
    @reactive.effect
    @reactive.event(input.mb_load_existing_btn)
    def _on_load_existing():
        grid = _grid_spec()
        if grid is None:
            ui.notification_show("Load a config with a usable grid first.", type="warning")
            return
        try:
            rel = input.mb_existing_choice()
        except SilentException:
            rel = None
        if not rel:
            return
        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()
        if cfg_dir is None:
            ui.notification_show(
                "Load the config from disk first so map files can be resolved.",
                type="warning",
            )
            return
        full_path = _find_config_file(rel, cfg_dir)
        if full_path is None:
            ui.notification_show(f"Map file not found: {rel}", type="error")
            return
        try:
            mg = from_csv_text(full_path.read_text(), grid)
        except (OSError, ValueError) as exc:
            ui.notification_show(f"Failed to load map {rel}: {exc}", type="error")
            return
        # Re-derive base mask so the editor reflects the active config's land cells.
        base_mask.set(load_mask(cfg, config_dir=cfg_dir))
        grid_array.set(mg)
        staged.set(None)
        dirty.set(dirty.get() + 1)
        ui.notification_show(f"Loaded {rel}.", type="message")

    # --- initial render + draw enablement, gated on deckgl_ready -----------
    @reactive.effect
    @reactive.event(input.deckgl_ready)
    async def _on_deckgl_ready():
        if grid_array.get() is None:
            _new_blank()
        await _render_full()
        try:
            mode = input.tool_mode()
        except SilentException:
            mode = "polygon"
        if mode == "polygon":
            await _map.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")

    def _view_state(grid: GridSpec) -> dict:
        center_lat = (grid.upleft_lat + grid.lowright_lat) / 2
        center_lon = (grid.upleft_lon + grid.lowright_lon) / 2
        span = max(
            abs(grid.upleft_lat - grid.lowright_lat),
            abs(grid.lowright_lon - grid.upleft_lon),
        )
        return {
            "latitude": center_lat,
            "longitude": center_lon,
            "zoom": _zoom_for_span(span),
        }

    def _value_cells_layer(mg: MapGrid, grid: GridSpec) -> dict | None:
        """Value-colored cells layer for painted (non-zero, non-land) cells."""
        arr = mg.array
        rs, cs = np.where((arr != 0) & (arr != -99))
        if len(rs) == 0:
            return None
        vals = arr[rs, cs]
        vmin, vmax = float(vals.min()), float(vals.max())
        vrange = vmax - vmin if vmax != vmin else 1.0
        cells = []
        for i in range(len(rs)):
            r, c = int(rs[i]), int(cs[i])
            v = float(vals[i])
            t = (v - vmin) / vrange
            r_ch = int(np.clip(68 + 187 * t * t, 0, 255))
            g_ch = int(np.clip(1 + 209 * t, 0, 255))
            b_ch = int(np.clip(84 + 86 * t - 170 * t * t, 0, 255))
            a_ch = int(np.clip(150 + 80 * t, 0, 255))
            cells.append(
                {
                    "polygon": grid.cell_polygon(r, c),
                    "value": v,
                    "fill": [r_ch, g_ch, b_ch, a_ch],
                }
            )
        return polygon_layer(
            "mb-value-cells",
            data=cells,
            getPolygon="@@=d.polygon",
            getFillColor="@@=d.fill",
            getLineColor=[0, 0, 0, 0],
            filled=True,
            stroked=False,
            pickable=True,
        )

    def _build_layers(mg: MapGrid, grid: GridSpec, is_dark: bool) -> list[dict]:
        layers = build_grid_layers(
            grid.upleft_lat,
            grid.upleft_lon,
            grid.lowright_lat,
            grid.lowright_lon,
            grid.nlon,
            grid.nlat,
            is_dark,
            base_mask.get(),
        )
        cells_layer = _value_cells_layer(mg, grid)
        if cells_layer is not None:
            layers.append(cells_layer)
        return layers

    async def _render_full() -> None:
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        is_dark = get_theme_mode(input) == "dark"
        style = CARTO_DARK if is_dark else CARTO_POSITRON
        if style != _map.style:
            _map.style = style
            await _map.set_style(session, style)
        widgets = [
            fullscreen_widget(placement="top-left"),
            zoom_widget(placement="top-right"),
            compass_widget(placement="top-right"),
            scale_widget(placement="bottom-right"),
        ]
        await _map.update(
            session,
            layers=_build_layers(mg, grid, is_dark),
            view_state=_view_state(grid),
            transition_duration=600,
            widgets=widgets,
        )

    # --- tool-mode toggle: enable polygon draw only in polygon mode --------
    @reactive.effect
    @reactive.event(input.tool_mode)
    async def _on_tool_mode():
        try:
            mode = input.tool_mode()
        except SilentException:
            return
        if mode == "polygon":
            await _map.enable_draw(session, modes=["draw_polygon"], default_mode="draw_polygon")
        else:
            await _map.disable_draw(session)

    # --- single-cell paint via map click (brush/eraser/mask) ---------------
    @reactive.effect
    @reactive.event(getattr(input, f"{_MAP_ID}_map_click"))
    def _on_map_click():
        try:
            mode = input.tool_mode()
        except SilentException:
            return
        if mode not in ("brush", "eraser", "mask"):
            return
        click = getattr(input, f"{_MAP_ID}_map_click")()
        if not isinstance(click, dict):
            return
        lon = click.get("longitude")
        lat = click.get("latitude")
        if lon is None or lat is None:
            return
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        cell = lonlat_to_cell(grid, float(lon), float(lat))
        if cell is None:
            return
        r, c = cell
        # Block painting on a -99 (land) cell unless we are in mask mode.
        if mode != "mask" and mg.array[r, c] == -99:
            return
        if mode == "brush":
            mg.apply_cells([cell], _paint_value())
        elif mode == "eraser":
            mg.erase([cell])
        elif mode == "mask":
            mg.set_mask([cell], True)
        dirty.set(dirty.get() + 1)

    # --- polygon staging + apply -------------------------------------------
    @reactive.effect
    @reactive.event(getattr(input, f"{_MAP_ID}_drawn_features"))
    def _on_drawn_features():
        fc = getattr(input, f"{_MAP_ID}_drawn_features")()
        if not isinstance(fc, dict) or not fc.get("features"):
            return
        staged.set(fc)

    @reactive.effect
    @reactive.event(input.apply_polygons)
    async def _on_apply_polygons():
        fc = staged.get()
        if not isinstance(fc, dict):
            return
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        try:
            mode = input.tool_mode()
        except SilentException:
            mode = "polygon"
        mask_edit = mode == "mask"
        value = _paint_value()
        painted = False
        for feature in fc.get("features", []):
            geom = feature.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if not coords:
                continue
            ring = coords[0]
            mg.apply_polygon(grid, ring, value, mask_edit=mask_edit)
            painted = True
        await _map.delete_drawn_features(session)
        staged.set(None)
        if painted:
            dirty.set(dirty.get() + 1)

    # --- coalesced render: rebuild only the value-cells layer on dirty -----
    @reactive.effect
    async def _on_dirty():
        dirty.get()
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            return
        cells_layer = _value_cells_layer(mg, grid)
        if cells_layer is None:
            # Nothing painted — send an empty cells layer to clear prior fills.
            cells_layer = polygon_layer(
                "mb-value-cells",
                data=[],
                getPolygon="@@=d.polygon",
                getFillColor="@@=d.fill",
                filled=True,
                stroked=False,
            )
        await _map.partial_update(session, layers=[cells_layer])

    # --- save --------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.mb_save)
    def _on_save():
        grid = _grid_spec()
        mg = grid_array.get()
        if grid is None or mg is None:
            ui.notification_show("No map to save — start a new blank map first.", type="warning")
            return
        try:
            mtype = input.map_type()
        except SilentException:
            mtype = "distribution"
        try:
            filename = (input.mb_filename() or "").strip()
        except SilentException:
            filename = ""
        if not filename:
            ui.notification_show("Enter a filename.", type="warning")
            return

        with reactive.isolate():
            cfg = state.config.get()
            cfg_dir = state.config_dir.get()
        if cfg_dir is None:
            tmp = tempfile.mkdtemp(prefix="osmose_maps_")
            cfg_dir = Path(tmp)
            state.config_dir.set(cfg_dir)
            ui.notification_show(
                f"No config dir set — saving into a session temp dir: {cfg_dir}",
                type="message",
            )

        applicability: dict | None = None
        if mtype == "distribution":
            try:
                species = input.mb_species()
            except SilentException:
                species = None
            if not species:
                ui.notification_show("Select a species for a distribution map.", type="warning")
                return
            applicability = {"species": species}
            try:
                applicability["initialage"] = float(input.mb_initialage() or 0)
            except (SilentException, ValueError, TypeError):
                pass
            try:
                lastage = input.mb_lastage()
                if lastage is not None:
                    applicability["lastage"] = float(lastage)
            except (SilentException, ValueError, TypeError):
                pass
            try:
                steps_txt = (input.mb_steps() or "").strip()
                if steps_txt:
                    applicability["steps"] = [
                        int(s) for s in steps_txt.replace(",", ";").split(";") if s.strip()
                    ]
            except (SilentException, ValueError, TypeError):
                pass
            try:
                iy = input.mb_initialyear()
                if iy is not None:
                    applicability["initialyear"] = int(iy)
            except (SilentException, ValueError, TypeError):
                pass
            try:
                ly = input.mb_lastyear()
                if ly is not None:
                    applicability["lastyear"] = int(ly)
            except (SilentException, ValueError, TypeError):
                pass

        # Validate — warn but DON'T block on the land-overlap warning.
        problems = validate(mg, grid, map_type=mtype, base_mask=base_mask.get())
        for p in problems:
            ui.notification_show(f"Warning: {p}", type="warning")

        # Confirm-style warning on overwrite (we proceed but flag it).
        subdir = "grid" if mtype == "mask" else "maps"
        dest = (
            Path(cfg_dir) / subdir / (filename if filename.endswith(".csv") else filename + ".csv")
        )
        if dest.exists():
            ui.notification_show(f"Overwriting existing file {dest.name}.", type="warning")

        # Warn on duplicate species/age/step overlap for distribution maps.
        if applicability is not None:
            _warn_on_overlap(cfg, applicability)

        try:
            new_cfg, summary, dest_path = save_map(
                mg,
                grid,
                mtype,
                filename,
                cfg,
                cfg_dir,
                applicability=applicability,
            )
        except (ValueError, OSError) as exc:
            ui.notification_show(f"Save failed: {exc}", type="error")
            return

        # Apply the new config via the real state API.
        with reactive.isolate():
            case_map = dict(state.key_case_map.get())
        state.load_config(new_cfg, case_map=case_map)
        state.dirty.set(True)
        ui.notification_show(f"{summary} → {dest_path}", type="message")


def _warn_on_overlap(cfg: dict[str, str], appl: dict) -> None:
    """Warn if an existing movement map already covers the same species/overlapping steps."""
    species = str(appl.get("species", ""))
    steps = set(appl.get("steps") or [])
    pat = re.compile(r"^movement\.species\.map(\d+)$")
    for k, v in cfg.items():
        m = pat.match(k)
        if not m or v != species:
            continue
        n = m.group(1)
        existing = cfg.get(f"movement.steps.map{n}", "")
        existing_steps = {
            int(s) for s in existing.replace(",", ";").split(";") if s.strip().isdigit()
        }
        if steps and existing_steps and (steps & existing_steps):
            ui.notification_show(
                f"map{n} already covers species '{species}' on overlapping steps.",
                type="warning",
            )
            return
