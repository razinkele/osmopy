---
name: shiny_deckgl API migration (v1.9)
description: Server uses shiny_deckgl 1.9.1 which renamed deck_legend_control to layer_legend_widget
type: project
---

Server env has `shiny_deckgl` 1.9.1 (Python 3.13 at `/opt/micromamba/envs/shiny/`). Key API change: `deck_legend_control` replaced by `layer_legend_widget` (same `entries` format, uses `placement` instead of `position`). The old `legend_control` function exists but has a completely different API (MapLibre-based, uses `targets` dict, no `entries`).

**Why:** The server's shiny_deckgl package was upgraded, breaking the legend import.

**How to apply:** `grid.py` uses `_make_legend()` wrapper with `hasattr`-based dispatch to support both old and new API. If adding new deck.gl widgets, check the server's version first: `/opt/micromamba/envs/shiny/bin/python3 -c "import shiny_deckgl; print(dir(shiny_deckgl))"`.
