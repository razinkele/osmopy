---
name: OSMOSE Python port architecture
description: Schema-driven design, parameter registry layout, config I/O mechanics, core-library module map. Read before adding a new OSMOSE parameter or touching the config reader/writer.
type: project
originSessionId: 12d091cb-241b-45e7-82e3-906f4025f88b
---
# OSMOSE Python Port — Architecture Notes

## Schema-Driven Design
- Every OSMOSE parameter defined once as `OsmoseField` in `osmose/schema/`
- `OsmoseField` has: key_pattern, param_type, default, min/max, description, category, unit, choices, indexed, required, advanced
- UI auto-generates forms from schema metadata via `ui/components/param_form.py`
- Adding a new OSMOSE parameter = adding one field to the schema
- Category counts drift as schema evolves — if you need an exact count, read `osmose/schema/` directly (CLAUDE.md says 153 as of 2026-04-17)

## Config I/O
- Reader: auto-detects separator (`;`, `=`, `,`, `:`, tab), recursive sub-file loading via `osmose.configuration.*` keys
- Writer: routes params by key prefix to category sub-files, writes master file with references
- All keys lowercase, dot-separated hierarchical (e.g., `species.linf.sp0`)

## Core Library (`osmose/`)
- `schema/` — Parameter definitions + registry
- `config/reader.py` — Parse OSMOSE .properties/.csv configs
- `config/writer.py` — Generate OSMOSE config directory
- `config/validator.py` — Cross-parameter validation
- `runner.py` — Async Java subprocess manager
- `results.py` — CSV/NetCDF output reader (xarray)
- `scenarios.py` — Save/load/compare/fork named configs (JSON)
- `calibration/` — pymoo NSGA-II, GP surrogate, SALib sensitivity

## UI (`ui/`)
- `page_fillable` + `navset_pill_list` left-sidebar (4 groups) — see [project_ui_architecture.md](project_ui_architecture.md)
- Each page in `ui/pages/` exports `*_ui()` and `*_server(input, output, session)`
- Theme: shinyswatch superhero + Nautical Observatory CSS overlay

## Known Issues / Gotchas
- `render_field()` generates input IDs by replacing dots with underscores in key_pattern
- Watch for overlapping prefix filters (e.g., `grid.netcdf.file` starts with `grid.n` AND contains `netcdf`) — caused duplicate ID bug, fixed in ca4a04c
- numpy ndarray size warning is benign (version mismatch in venv)
- SALib uses modern `SALib.sample.sobol` API (not deprecated `saltelli`)
- Runner tests use mock Python scripts as fake JARs via `_ScriptRunner` subclass
- pyproject.toml needs `[tool.setuptools.packages.find]` with `include = ["osmose*"]`

## Deployment
- Dockerfile: multi-stage (eclipse-temurin:17-jre + python:3.12-slim)
- Non-root user `osmose`, exposes port 8000
- `COPY osmose-java*` (wildcard) avoids build failure on empty directory
- `pip install .` runs AFTER copying application code (not before)
