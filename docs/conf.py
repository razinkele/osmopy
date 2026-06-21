"""Sphinx configuration for the OSMOSE Python documentation site.

Import-safe WITHOUT the [docs] extra installed: every setting below is plain
data so tests/test_docs_build.py can exec this file in the [dev] CI leg. Do NOT
import sphinx/furo at module top level -- defer any extension/theme imports into
a setup(app) hook if ever needed.
"""

from osmose.__version__ import __version__

# -- Project information -----------------------------------------------------
project = "OSMOSE Python"
author = "OSMOSE Python contributors"
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]

# Whitelist: ONLY these pages are part of the build. Everything else in docs/
# (diagnostics, plans/, perf/, superpowers/, the _build output) is ignored, so
# there are no "document isn't included in any toctree" warnings under -W.
include_patterns = [
    "index.md",
    "usage-guide.md",
    "tutorials/30-minute-ecosystem.md",
    "tutorials/fie-on-baltic-cod.md",
    "api/**",
]

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
add_module_names = False
nitpicky = False

# -- MyST --------------------------------------------------------------------
# colon_fence enables ::: directive blocks (used by index.md toctrees).
# heading_anchors=3 makes same-document #anchor links (e.g. usage-guide's
# [§6](#...)) resolve instead of raising myst.xref_missing under -W.
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

# -- intersphinx -------------------------------------------------------------
# nitpicky=False suppresses missing-reference warnings but NOT failed-download
# warnings; intersphinx_timeout bounds a CI network hiccup so -W is not flaky.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
intersphinx_timeout = 10

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"OSMOSE Python {release}"
