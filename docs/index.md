# OSMOSE Python

Orchestration layer, simulation engine, and Shiny web interface for the OSMOSE
marine ecosystem simulator. OSMOSE Python provides two engine backends — a pure
Python engine (NumPy/Numba) and the original Java engine via subprocess — plus
config I/O, calibration, output reading, and visualization.

Install the project in editable mode with `pip install -e ".[dev]"`. New to the
model? Start with the 30-minute tutorial in the Guides section below.

:::{toctree}
:maxdepth: 2
:caption: Guides

usage-guide
r-to-python-migration
tutorials/30-minute-ecosystem
tutorials/fie-on-baltic-cod
:::

:::{toctree}
:maxdepth: 2
:caption: API Reference

api/index
:::
