---
name: Shiny UI architecture — nav, panels, theme, tooltips
description: Non-obvious UI shape: navset_pill_list sidebar, collapsible-panel pattern, shiny-deckgl 1.9 quirks, tooltip init order. Read before touching ui/.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
**Navigation:** `page_fillable` + `navset_pill_list` (left sidebar, 4 groups). IMPORTANT: `navset_pill_list` doesn't accept bare strings as section headers — use `ui.nav_control()` wrapper.

**Collapsible panels:** Both nav column and page panels share one pattern:

- `«` collapse button in header/pill list.
- Vertical expand tab ("Menu" / "Simulation Settings" / "Grid Type") appears when collapsed.
- CSS uses `.nav-col-collapsed` for nav, `.collapsed` for panels.
- JS `toggleNav()` and `togglePanel(pageId)` with localStorage persistence.
- Nav expand tab is created by end-of-body JS (polling until DOM ready).
- Panels use `.osm-split-layout` flex container + `.bslib-grid` CSS Grid override.
- When panel collapses: `grid-column: 1 / -1` on sibling for full-width expansion.

**Grid preview:** shiny-deckgl `MapWidget` + `polygon_layer` (plain map, no card wrapper). Overlay selector lives in Grid Type card (moved from preview card). Non-spatial files filtered from overlay dropdown (accessibility, season, catchability, discards).

**shiny_deckgl 1.9 compat:** `_make_legend()` wrapper in `grid.py` dispatches via `hasattr` to handle both old and new API shapes — see `project_shiny_deckgl_api.md`.

**Theme:** shinyswatch `superhero` + Nautical Observatory CSS overlay + ambient enhancements (caustic light atmosphere, sonar pulse spinner, polished notifications, nav micro-interactions — all respect `prefers-reduced-motion`). Light/dark toggle via JS, localStorage persistence.

**Tooltips (Bootstrap 5):** Popovers initialized via end-of-body `setInterval(500ms)` polling for new `[data-bs-toggle="popover"]` elements. Uses `bootstrap.Popover.getInstance(el)` (BS5 API), NOT `el._bsPopover` (BS4). Show Help button removed; tooltips work via hover only. Inline `<head>` scripts run before Bootstrap loads, so head-scripts don't work here.

**IMPORT RULE:** `import ui.charts` must only be in `app.py` — importing in page modules shadows `from shiny import ui`.
