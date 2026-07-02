---
name: OSMOSE Python dev environment
description: System paths, venv locations, Playwright quirks, app/test commands. Read when reproducing a dev setup or debugging tool-level issues.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
# Development Environment

## System
- Linux Mint (Ubuntu-compatible)
- Python command: `.venv/bin/python` (not `python` — system python not available)
- Venv at: `/home/razinka/osmose/osmose-python/.venv/`
- Server Python (for deployed app): `/opt/micromamba/envs/shiny/bin/python3` (3.13)
- No sudo access in Claude Code sessions

## Playwright / Browser
- Google Chrome Beta installed at `/opt/google/chrome-beta/google-chrome-beta`
- Playwright expects Chrome at `/opt/google/chrome/chrome` — symlink needed but requires sudo
- Playwright's `browser_install` fails on Linux Mint (OS detection rejects non-Ubuntu/Debian)
- Workaround: Playwright `browser_run_code` / `browser_navigate` work if browser is already launched

## Running Tests
- Current: `.venv/bin/python -m pytest` → 2485 passing, 15 skipped (check `project_current_status.md` for up-to-date count — grows each release).
- Test fixtures in `tests/fixtures/`; integration configs in `data/examples/` (Bay of Biscay, EEC) and `data/baltic/`.

## Running the App
- Dev: `.venv/bin/shiny run app.py --host 0.0.0.0 --port 8000`
- Production: `osmose-shiny.service` (standalone Uvicorn port 8838), nginx proxies `https://laguna.ku.lt/osmose/`.

## Dependency surface (high-level)
- Runtime: shiny, shinyswatch, shiny-deckgl, pandas, xarray, netCDF4, plotly, jinja2, pymoo, scikit-learn, SALib, numpy, numba, pyyaml.
- Dev: pytest, ruff, pyright.
- Exact versions: read `pyproject.toml` — don't pin memory to them.
