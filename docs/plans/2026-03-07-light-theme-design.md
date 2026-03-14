# Light Theme Toggle Design

**Date:** 2026-03-07
**Goal:** Add a sun/moon toggle in the header that switches between dark (Nautical Observatory) and light (Ocean Blue-White) themes.

## Approach

CSS custom properties override. The existing dark CSS uses `--osm-*` variables throughout. A `[data-theme="light"]` selector overrides these variables. No duplicate CSS, no stylesheet swapping.

## Light Palette

- Backgrounds: `#f0f4f8` (page), `#ffffff` (cards), `#e8edf2` (card headers)
- Text: `#1a2a3a` (primary), `#4a5a6a` (secondary), `#8a95a0` (muted)
- Accents: amber `#d4942e` (darkened for contrast on light), teal `#2ba89e`
- Borders: `rgba(0, 0, 0, 0.08)`

## Components

1. **CSS** — `[data-theme="light"]` block in `osmose.css` overriding all `--osm-*` variables + hardcoded rgba values
2. **JS toggle** — Sun/moon button in header, sets `data-theme` on `<html>`, persists to `localStorage`
3. **Plotly** — `osmose-light` template in `ui/charts.py`, server switches via reactive input
4. **styles.py** — Replace hardcoded dark colors with CSS-class-based approach where needed
