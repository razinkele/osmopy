"""OSMOPY - Python Interface for OSMOSE marine ecosystem simulator."""

from pathlib import Path

from shiny import App, render, ui

from ui.state import AppState
from ui.components.help_modal import about_modal, help_modal
from ui.theme import THEME
from ui.styles import STYLE_CONFIG_HEADER
import ui.charts as _charts  # noqa: F401 — registers custom plotly template

from shiny_deckgl import head_includes as _deckgl_head

from ui.pages.setup import setup_ui, setup_server
from ui.pages.grid import grid_ui, grid_server
from ui.pages.forcing import forcing_ui, forcing_server
from ui.pages.fishing import fishing_ui, fishing_server
from ui.pages.movement import movement_ui, movement_server
from ui.pages.run import run_ui, run_server
from ui.pages.results import results_ui, results_server
from ui.pages.calibration import calibration_ui, calibration_server
from ui.pages.scenarios import scenarios_ui, scenarios_server
from ui.pages.advanced import advanced_ui, advanced_server

_WWW = Path(__file__).parent / "www"


def _nav_section(label: str):
    """Render a section header in the pill list sidebar."""
    return ui.nav_control(
        ui.tags.span(label, class_="osmose-section-label"),
    )


app_ui = ui.page_fillable(
    # ── Custom CSS + theme toggle JS ────────────────────────────
    ui.head_content(ui.include_css(_WWW / "osmose.css")),
    ui.head_content(ui.include_css(_WWW / "deckgl-widgets.css")),
    ui.head_content(ui.include_css(_WWW / "maplibre-gl-opacity.css")),
    ui.head_content(ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1")),
    ui.head_content(ui.tags.link(rel="icon", type="image/svg+xml", href="favicon.svg")),
    # ── deck.gl JS/CSS dependencies (needed for grid map) ──────
    _deckgl_head(),
    ui.head_content(
        ui.tags.script("""
        function toggleTheme() {
            var html = document.documentElement;
            var current = html.getAttribute('data-theme');
            var next = current === 'light' ? 'dark' : 'light';
            html.setAttribute('data-theme', next);
            localStorage.setItem('osmose-theme', next);
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('theme_mode', next);
            }
        }
        // Restore saved theme on page load
        (function() {
            var saved = localStorage.getItem('osmose-theme') || 'light';
            document.documentElement.setAttribute('data-theme', saved);
            // Notify Shiny once connected
            if (typeof Shiny !== 'undefined') {
                Shiny.addCustomMessageHandler('_noop', function(){});
            }
            document.addEventListener('shiny:connected', function() {
                var theme = localStorage.getItem('osmose-theme') || 'light';
                Shiny.setInputValue('theme_mode', theme);
            });
        })();
        window.addEventListener('beforeunload', function(e) {
            if (typeof Shiny !== 'undefined' && Shiny.shinyapp &&
                Shiny.shinyapp.$inputValues && Shiny.shinyapp.$inputValues.is_dirty) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
        // Initialize Bootstrap popovers on dynamic content
        document.addEventListener('shiny:value', function() {
            document.querySelectorAll('[data-bs-toggle="popover"]').forEach(function(el) {
                if (!el._bsPopover) new bootstrap.Popover(el);
            });
        });

        // Show Help toggle
        function toggleHelpMode() {
            document.body.classList.toggle('show-all-help');
            var active = document.body.classList.contains('show-all-help');
            localStorage.setItem('osmose-show-help', active ? '1' : '0');
            var btn = document.getElementById('helpToggle');
            if (btn) btn.textContent = active ? 'Hide Help' : 'Show Help';
        }
        // Restore help mode
        (function() {
            if (localStorage.getItem('osmose-show-help') === '1') {
                document.body.classList.add('show-all-help');
            }
        })();
    """)
    ),
    # ── App header ──────────────────────────────────────────────
    ui.div(
        ui.tags.h4(
            "OSMOPY",
            ui.tags.span(" | Marine Ecosystem Simulator", class_="subtitle"),
            class_="osmose-logo",
        ),
        ui.tags.span(
            ui.tags.span(class_="dot"),
            "Marine Ecosystem Simulator",
            class_="osmose-badge",
        ),
        ui.div(
            ui.tags.button(
                ui.tags.span("\u2600\ufe0f", class_="icon-sun"),
                ui.tags.span("\u263e", class_="icon-moon"),
                class_="osmose-theme-toggle",
                id="themeToggle",
                title="Toggle light/dark theme",
                onclick="toggleTheme()",
                **{"aria-label": "Toggle light/dark theme"},
            ),
            ui.tags.button(
                "Show Help",
                class_="osmose-header-btn",
                id="helpToggle",
                onclick="toggleHelpMode()",
            ),
            ui.tags.a(
                "About",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#aboutModal"},
            ),
            ui.tags.a(
                "Help",
                class_="osmose-header-btn",
                href="#",
                **{"data-bs-toggle": "modal", "data-bs-target": "#helpModal"},
            ),
            class_="osmose-header-actions",
        ),
        class_="osmose-header",
    ),
    # ── Global loading overlay ───────────────────────────────────
    ui.output_ui("config_header"),
    ui.output_ui("loading_overlay"),
    # ── Left pill navigation with grouped sections ──────────────
    ui.navset_pill_list(
        # Configure
        _nav_section("Configure"),
        ui.nav_panel("Setup", setup_ui(), value="setup"),
        ui.nav_panel("Grid & Maps", grid_ui(), value="grid"),
        ui.nav_panel("Forcing", forcing_ui(), value="forcing"),
        ui.nav_panel("Fishing", fishing_ui(), value="fishing"),
        ui.nav_panel("Movement", movement_ui(), value="movement"),
        # Execute
        _nav_section("Execute"),
        ui.nav_panel("Run", run_ui(), value="run"),
        ui.nav_panel("Results", results_ui(), value="results"),
        # Optimize
        _nav_section("Optimize"),
        ui.nav_panel("Calibration", calibration_ui(), value="calibration"),
        # Manage
        _nav_section("Manage"),
        ui.nav_panel("Scenarios", scenarios_ui(), value="scenarios"),
        ui.nav_panel("Advanced", advanced_ui(), value="advanced"),
        id="main_nav",
        selected="setup",
        widths=(2, 10),
        well=False,
    ),
    # ── Modals (static HTML, triggered client-side) ─────────────
    about_modal(),
    help_modal(),
    # ── deck.gl init fallback ─────────────────────────────────────
    # CDN scripts may finish loading after shiny:connected fires,
    # leaving MapWidget divs uninitialized.  Poll until deps are
    # ready, then re-dispatch the event so deckgl-init.js catches up.
    ui.tags.script("""
    (function() {
        var t = setInterval(function() {
            if (typeof maplibregl === 'undefined' || typeof deck === 'undefined') return;
            if (!window.__deckgl_instances) return;
            var maps = document.querySelectorAll('.deckgl-map');
            if (!maps.length) { clearInterval(t); return; }
            var need = false;
            maps.forEach(function(el) {
                if (!window.__deckgl_instances[el.id]) need = true;
            });
            if (!need) { clearInterval(t); return; }
            document.dispatchEvent(new Event('shiny:connected'));
            // Nudge server to re-send map updates now that widgets are ready
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('deckgl_ready', Date.now(), {priority: 'event'});
            }
            clearInterval(t);
        }, 300);
    })();
    """),
    theme=THEME,
)


def server(input, output, session):
    state = AppState()
    state.reset_to_defaults()

    @render.ui
    def config_header():
        name = state.config_name.get()
        if not name:
            return ui.div()
        cfg = state.config.get()
        n_species = int(cfg.get("simulation.nspecies", "0"))
        n_params = len(cfg)
        is_dirty = state.dirty.get()
        return ui.div(
            ui.div(
                ui.tags.span(name, style="color: #d4a017; font-weight: 600;"),
                ui.tags.span(
                    f" {n_species} species \u2022 {n_params} parameters",
                    style="color: #5a6a7a; font-size: 12px; margin-left: 8px;",
                ),
            ),
            ui.tags.span(
                "modified" if is_dirty else "",
                style="color: #e67e22; font-size: 11px; font-style: italic;",
            ),
            style=STYLE_CONFIG_HEADER,
        )

    @render.ui
    def loading_overlay():
        msg = state.busy.get()
        if msg is None:
            return ui.div()
        return ui.div(
            ui.div(ui.div(class_="osm-spinner"), ui.p(msg), class_="osm-loading-content"),
            class_="osm-loading-overlay",
        )

    setup_server(input, output, session, state)
    grid_server(input, output, session, state)
    forcing_server(input, output, session, state)
    fishing_server(input, output, session, state)
    movement_server(input, output, session, state)
    run_server(input, output, session, state)
    results_server(input, output, session, state)
    calibration_server(input, output, session, state)
    scenarios_server(input, output, session, state)
    advanced_server(input, output, session, state)


app = App(app_ui, server, static_assets=_WWW)
