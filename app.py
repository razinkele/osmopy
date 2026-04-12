"""OSMOPY - Python Interface for OSMOSE marine ecosystem simulator."""

from pathlib import Path

from osmose import __version__

from shiny import App, reactive, render, ui

from ui.state import AppState
from ui.components.help_modal import about_modal, help_modal
from ui.theme import THEME
import ui.charts as _charts  # noqa: F401 — registers custom plotly template

from shiny_deckgl import head_includes as _deckgl_head

from ui.pages.setup import setup_ui, setup_server
from ui.pages.grid import grid_ui, grid_server
from ui.pages.forcing import forcing_ui, forcing_server
from ui.pages.fishing import fishing_ui, fishing_server
from ui.pages.movement import movement_ui, movement_server
from ui.pages.run import run_ui, run_server
from ui.pages.results import results_ui, results_server
from ui.pages.spatial_results import spatial_results_ui, spatial_results_server
from ui.pages.calibration import calibration_ui, calibration_server
from ui.pages.scenarios import scenarios_ui, scenarios_server
from ui.pages.advanced import advanced_ui, advanced_server
from ui.pages.map_viewer import map_viewer_ui, map_viewer_server
from ui.pages.genetics import genetics_ui, genetics_server
from ui.pages.economic import economic_ui, economic_server
from ui.pages.diagnostics import diagnostics_ui, diagnostics_server

from osmose.cleanup import cleanup_old_temp_dirs, register_cleanup

# Clean stale osmose temp dirs from previous sessions and register atexit
# handler so current-session dirs are removed on normal shutdown.
cleanup_old_temp_dirs()
register_cleanup()

_WWW = Path(__file__).parent / "www"


def _nav_section(label: str):
    """Render a section header in the pill list sidebar."""
    return ui.nav_control(
        ui.tags.span(label, class_="osmose-section-label"),
    )


app_ui = ui.page_fillable(
    # ── Skip link for keyboard users ────────────────────────────
    ui.tags.a(
        "Skip to content",
        href="#main-content",
        class_="visually-hidden-focusable",
    ),
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
        // Bootstrap popover init is at end of body (after dependencies load)

        // ── Nav collapse (same pattern as panel collapse) ─
        function toggleNav() {
            var row = document.querySelector('body > .row');
            if (!row) return;
            var navCol = row.children[0];
            var collapsed = navCol.classList.toggle('nav-col-collapsed');
            // Show/hide the expand tab
            var tab = document.getElementById('nav-expand-tab');
            if (tab) tab.classList.toggle('visible', collapsed);
            // Sync ARIA on both nav collapse and expand buttons
            var collapseBtn = document.querySelector('.osm-nav-collapse-btn');
            if (collapseBtn) collapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            if (tab) tab.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
            localStorage.setItem('osmose-nav-collapsed', collapsed ? '1' : '0');
        }
        // Nav expand tab is created at end of body (after DOM renders)

        // ── Panel collapse ────────────────────────────────
        function togglePanel(pageId) {
            var container = document.getElementById('split_' + pageId);
            if (!container) return;
            var row = container.querySelector('.row') || container.querySelector('.bslib-grid');
            if (!row) { console.warn('togglePanel: no .row or .bslib-grid in', pageId); return; }
            var left = row.children[0];
            var tab = document.getElementById('expand_' + pageId);

            var collapsed = left.classList.toggle('collapsed');
            if (tab) tab.classList.toggle('visible', collapsed);
            // Sync ARIA states on both collapse and expand buttons
            var collapseBtn = container.querySelector('.osm-collapse-btn');
            if (collapseBtn) collapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            if (tab) tab.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
            localStorage.setItem('osmose-panel-collapsed-' + pageId, collapsed ? '1' : '0');
        }

        // ── Restore panel states on tab activation ────────
        (function() {
            var restoredPanels = {};
            function restorePanelIfNeeded(pageId) {
                if (restoredPanels[pageId]) return;
                restoredPanels[pageId] = true;
                if (localStorage.getItem('osmose-panel-collapsed-' + pageId) === '1') {
                    setTimeout(function() { togglePanel(pageId); }, 100);
                }
            }
            var pageIds = ['setup','grid','forcing','fishing','movement','genetics','economic',
                           'run','results','spatial_results','diagnostics','calibration','scenarios','advanced','map_viewer'];

            document.addEventListener('DOMContentLoaded', function() {
                // Restore the initially active tab's panel
                var activeLink = document.querySelector('.nav-pills .nav-link.active');
                if (activeLink) {
                    var val = activeLink.getAttribute('data-value') || '';
                    pageIds.forEach(function(pid) {
                        if (val === pid) restorePanelIfNeeded(pid);
                    });
                }
                // Restore panels as tabs are activated (lazy rendering)
                document.addEventListener('shown.bs.tab', function(e) {
                    var val = e.target.getAttribute('data-value') ||
                              e.target.getAttribute('href') || '';
                    pageIds.forEach(function(pid) {
                        if (val === pid) restorePanelIfNeeded(pid);
                    });
                });
            });
        })();

        // Engine mode toggle — syncs with Shiny input and localStorage
        window.setEngineMode = function(mode) {
            localStorage.setItem('osmose-engine', mode);
            var jBtn = document.getElementById('engineBtnJava');
            var pBtn = document.getElementById('engineBtnPython');
            if (mode === 'java') {
                jBtn.classList.add('active');
                pBtn.classList.remove('active');
            } else {
                pBtn.classList.add('active');
                jBtn.classList.remove('active');
            }
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('engine_mode', mode);
            }
            // Toggle disabled state on Python-only nav items
            document.querySelectorAll('.osm-engine-gated').forEach(function(el) {
                if (mode === 'python') {
                    el.classList.remove('osm-disabled');
                    el.removeAttribute('title');
                } else {
                    el.classList.add('osm-disabled');
                    el.setAttribute('title', 'Requires Python engine');
                }
            });
        };
        // Restore engine mode from localStorage on page load
        // (deferred to consolidated init in end-of-body script)
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
            f"v{__version__}",
            class_="osmose-badge",
        ),
        # Engine mode toggle
        ui.div(
            ui.tags.button(
                "Java",
                class_="osm-engine-btn active",
                id="engineBtnJava",
                onclick="setEngineMode('java')",
            ),
            ui.tags.button(
                "Python",
                class_="osm-engine-btn",
                id="engineBtnPython",
                onclick="setEngineMode('python')",
            ),
            class_="osm-engine-toggle",
        ),
        ui.div(
            ui.output_ui("config_header"),
            ui.tags.button(
                ui.tags.span("\u2600\ufe0f", class_="icon-sun"),
                ui.tags.span("\u263e", class_="icon-moon"),
                class_="osmose-theme-toggle",
                id="themeToggle",
                title="Toggle light/dark theme",
                onclick="toggleTheme()",
                **{"aria-label": "Toggle light/dark theme"},
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
    ui.output_ui("loading_overlay"),
    # ── Left pill navigation with grouped sections ──────────────
    ui.navset_pill_list(
        # Collapse button for nav sidebar
        ui.nav_control(
            ui.tags.button(
                "\u00ab",
                class_="osm-collapse-btn osm-nav-collapse-btn",
                onclick="toggleNav()",
                title="Collapse menu",
                **{
                    "aria-label": "Collapse navigation",
                    "aria-expanded": "true",
                    "aria-controls": "main_nav",
                },
            ),
        ),
        # Configure
        _nav_section("Configure"),
        ui.nav_panel("Setup", setup_ui(), value="setup"),
        ui.nav_panel("Grid", grid_ui(), value="grid"),
        ui.nav_panel("Forcing", forcing_ui(), value="forcing"),
        ui.nav_panel("Fishing", fishing_ui(), value="fishing"),
        ui.nav_panel("Movement", movement_ui(), value="movement"),
        ui.nav_panel(
            ui.span("Genetics", class_="osm-engine-gated osm-disabled"),
            genetics_ui(),
            value="genetics",
        ),
        ui.nav_panel(
            ui.span("Economic", class_="osm-engine-gated osm-disabled"),
            economic_ui(),
            value="economic",
        ),
        # Execute
        _nav_section("Execute"),
        ui.nav_panel("Run", run_ui(), value="run"),
        ui.nav_panel("Results", results_ui(), value="results"),
        ui.nav_panel("Spatial Results", spatial_results_ui(), value="spatial_results"),
        ui.nav_panel(
            ui.span("Diagnostics", class_="osm-engine-gated osm-disabled"),
            diagnostics_ui(),
            value="diagnostics",
        ),
        # Optimize
        _nav_section("Optimize"),
        ui.nav_panel("Calibration", calibration_ui(), value="calibration"),
        # Manage
        _nav_section("Manage"),
        ui.nav_panel("Scenarios", scenarios_ui(), value="scenarios"),
        ui.nav_panel("Advanced", advanced_ui(), value="advanced"),
        ui.nav_panel("Map Viewer", map_viewer_ui(), value="map_viewer"),
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
    # End-of-body scripts — must run after Shiny renders the DOM.
    # Consolidated initialization: nav expand tab, popovers, spatial pill, skip link
    ui.tags.script("""
    (function() {
        // ── Consolidated one-shot DOM setup ──────────────────────
        function initOnceElements() {
            // 1. Create nav expand tab (if not already created)
            var row = document.querySelector('body > .row');
            if (row && row.children.length >= 2 && !document.getElementById('nav-expand-tab')) {
                var tab = document.createElement('button');
                tab.id = 'nav-expand-tab';
                tab.className = 'osm-expand-tab';
                tab.textContent = 'Menu';
                tab.setAttribute('aria-label', 'Expand navigation');
                tab.setAttribute('aria-expanded', 'false');
                tab.onclick = function() { toggleNav(); };
                row.insertBefore(tab, row.children[1]);
                if (localStorage.getItem('osmose-nav-collapsed') === '1') {
                    toggleNav();
                }
            }

            // 2. Disable spatial results pill initially
            var pill = document.querySelector('.nav-link[data-value="spatial_results"]');
            if (pill) pill.classList.add('osm-disabled');

            // 3. Set id/role on main tab-content for skip link
            var mainNav = document.getElementById('main_nav');
            if (mainNav) {
                var tabContent = mainNav.closest('.row');
                if (tabContent) tabContent = tabContent.querySelector('.tab-content');
                if (tabContent) {
                    tabContent.id = 'main-content';
                    tabContent.setAttribute('role', 'main');
                    tabContent.setAttribute('tabindex', '-1');
                }
            }

            // 4. Restore engine mode from localStorage
            if (document.getElementById('engineBtnJava')) {
                var savedEngine = localStorage.getItem('osmose-engine') || 'java';
                setEngineMode(savedEngine);
            }
        }

        // Try immediately, then retry via requestAnimationFrame
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            initOnceElements();
            // Retry once more in case DOM isn't fully rendered
            requestAnimationFrame(function() { initOnceElements(); });
        } else {
            document.addEventListener('DOMContentLoaded', function() {
                initOnceElements();
                requestAnimationFrame(function() { initOnceElements(); });
            });
        }

        // ── Popover init via MutationObserver ────────────────────
        function initNewPopovers(root) {
            if (typeof bootstrap === 'undefined' || !bootstrap.Popover) return;
            var els = (root || document).querySelectorAll('[data-bs-toggle="popover"]:not([data-osm-init])');
            els.forEach(function(el) {
                new bootstrap.Popover(el);
                el.setAttribute('data-osm-init', '1');
            });
        }
        // Initial scan
        initNewPopovers();
        // Watch for dynamically added popovers
        var observer = new MutationObserver(function(mutations) {
            var needScan = false;
            for (var i = 0; i < mutations.length; i++) {
                if (mutations[i].addedNodes.length) { needScan = true; break; }
            }
            if (needScan) initNewPopovers();
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // ── Spatial pill toggle (server-driven) ──────────────────
        var _pillRegistered = false;
        document.addEventListener('shiny:connected', function() {
            if (_pillRegistered) return;
            _pillRegistered = true;
            Shiny.addCustomMessageHandler('toggle-spatial-pill', function(msg) {
                var pill = document.querySelector('.nav-link[data-value="spatial_results"]');
                if (pill) {
                    if (msg.action === 'add') {
                        pill.classList.add('osm-disabled');
                    } else {
                        pill.classList.remove('osm-disabled');
                    }
                }
            });
        });

        // ── Pause caustic animation when tab is hidden ───────────
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                document.documentElement.classList.add('paused');
            } else {
                document.documentElement.classList.remove('paused');
            }
        });

        // ── Keyboard shortcuts ───────────────────────────────────
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd+S → save scenario
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue('shortcut_save', Date.now(), {priority: 'event'});
                }
                return;
            }
            // ? → open help modal (only when not typing in an input)
            if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                var tag = e.target.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' ||
                    e.target.contentEditable === 'true') return;
                var helpModal = document.getElementById('helpModal');
                if (helpModal && typeof bootstrap !== 'undefined') {
                    var modal = bootstrap.Modal.getOrCreateInstance(helpModal);
                    modal.show();
                }
            }
        });
    })();
    """),
    theme=THEME,
)


def server(input, output, session):
    state = AppState()
    state.reset_to_defaults()

    @reactive.effect
    @reactive.event(input.engine_mode)
    def _sync_engine_mode():
        mode = input.engine_mode()
        if mode in ("java", "python"):
            state.engine_mode.set(mode)

    @render.ui
    def config_header():
        name = state.config_name.get()
        if not name:
            return ui.div()
        cfg = state.config.get()
        try:
            n_species = int(float(cfg.get("simulation.nspecies", "0")))
        except (ValueError, TypeError):
            n_species = 0
        n_params = len(cfg)
        is_dirty = state.dirty.get()
        return ui.div(
            ui.tags.span(name, class_="osm-config-name"),
            ui.tags.span(
                f" {n_species} species \u2022 {n_params} params",
                class_="osm-config-stats",
            ),
            ui.tags.span(
                " modified" if is_dirty else "",
                class_="osm-config-dirty",
            ),
            class_="osm-config-info",
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
    spatial_results_server(input, output, session, state)
    calibration_server(input, output, session, state)
    scenarios_server(input, output, session, state)
    advanced_server(input, output, session, state)
    map_viewer_server(input, output, session, state)
    genetics_server(input, output, session, state)
    economic_server(input, output, session, state)
    diagnostics_server(input, output, session, state)


app = App(app_ui, server, static_assets=_WWW)
