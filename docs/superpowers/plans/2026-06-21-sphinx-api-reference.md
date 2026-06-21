# Sphinx API Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a Sphinx + myst-parser documentation site for the `osmose/` library (recursive API reference + existing narrative guides), built with warnings-as-errors and auto-deployed to GitHub Pages.

**Architecture:** A single `conf.py` rooted at the existing `docs/` directory uses an `include_patterns` whitelist so only curated pages build. `sphinx.ext.autosummary` with `:recursive:` auto-discovers every `osmose` module via one `module.rst` template (zero per-module maintenance). The narrative markdown (`usage-guide.md`, both tutorials) is folded in. A dedicated `docs.yml` workflow builds with `-W` on PRs and builds+deploys to Pages on `master`.

**Tech Stack:** Sphinx ≥7, myst-parser ≥2, furo, GitHub Actions Pages (`upload-pages-artifact` / `deploy-pages`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-21-sphinx-api-reference-design.md`

**Conventions (from CLAUDE.md):** use `.venv/bin/python` / `.venv/bin/<tool>` (system `python` may not exist); OSMOSE GitHub repo is `razinkele/osmopy`, default branch `master`.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | add `[project.optional-dependencies] docs` extra; later add `[project.urls]` |
| `.gitignore` | ignore generated build output + autosummary stubs |
| `docs/conf.py` | Sphinx configuration (data-only, import-safe without `[docs]`) |
| `docs/_templates/autosummary/module.rst` | recursive autosummary template (public-filtered) |
| `docs/api/index.rst` | autosummary `:recursive:` root over `osmose` |
| `docs/index.md` | landing page + toctrees (the build's root document) |
| `docs/usage-guide.md` | EXISTING — rewrite 6 outbound links to absolute URLs |
| `.github/workflows/docs.yml` | build (`-W`) on PR; build + deploy to Pages on `master` |
| `tests/test_docs_build.py` | fast guard: all `osmose` modules import + conf.py data globals |

---

## Task 1: Add the `docs` extra and gitignore the build output

**Files:**
- Modify: `pyproject.toml` (the `[project.optional-dependencies]` table)
- Modify: `.gitignore`

- [ ] **Step 1: Add the `docs` extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a new `docs` key. Insert it immediately after the `numba = [...]` line (before `dev = [`):

```toml
docs = ["sphinx>=7", "myst-parser>=2", "furo>=2024.1"]
```

- [ ] **Step 2: Gitignore generated docs artifacts**

Append to `.gitignore` (the repo's `build/` rule does NOT match `docs/_build/` because that dir is named `_build`):

```gitignore
# Sphinx docs build output + generated autosummary stubs
docs/_build/
docs/api/_autosummary/
```

- [ ] **Step 3: Install and verify the extra resolves**

Run: `.venv/bin/pip install -e ".[docs]"`
Expected: installs sphinx, myst-parser, furo (and deps) without error.

Run: `.venv/bin/python -c "import sphinx, myst_parser, furo; print(sphinx.__version__)"`
Expected: prints a version ≥ 7, no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "build: add docs extra (sphinx + myst-parser + furo) and gitignore build output"
```

---

## Task 2: Guard test — all `osmose` modules import + conf.py data globals

This test is the fast `[dev]`-leg signal. The import-guard half passes immediately (autodoc-importability is already clean); the conf.py half FAILS until Task 3 creates `docs/conf.py` — that failure drives Task 3.

**Files:**
- Create: `tests/test_docs_build.py`

- [ ] **Step 1: Write the test**

Create `tests/test_docs_build.py`:

```python
"""Fast guard tests for the Sphinx docs build (run in the normal [dev] CI leg).

These are the cheap early signal; the authoritative check is the `-W` Sphinx
build in .github/workflows/docs.yml.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import osmose

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONF = _REPO_ROOT / "docs" / "conf.py"


def test_all_osmose_modules_import():
    """autodoc imports every module it documents; an unimportable module breaks
    the docs build. Catch it here in the fast CI leg, before the slow Sphinx job.

    Caveat: this leg runs under the [dev] extra (a superset of the docs job's
    [docs] + numba). It cannot catch a module that gains a top-level import of a
    dev-only dep (e.g. httpx, pillow) -- that is caught only by the -W docs build.
    """
    failures: list[str] = []
    # walk_packages imports each sub-PACKAGE itself (to read __path__ for
    # recursion); that import is outside the loop's try, so a raising package
    # __init__ would abort the walk with a raw traceback. onerror collects it.
    walker = pkgutil.walk_packages(
        osmose.__path__,
        "osmose.",
        onerror=lambda name: failures.append(f"{name}: package import error during walk"),
    )
    for mod in walker:
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - collect all, not just the first
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "osmose modules failed to import:\n" + "\n".join(failures)


def test_conf_py_is_import_safe_and_defines_data_globals():
    """conf.py must exec WITHOUT the [docs] extra installed (no top-level sphinx/
    furo import) and define the core data globals."""
    namespace: dict[str, object] = {}
    exec(compile(_CONF.read_text(), str(_CONF), "exec"), namespace)
    for key in ("project", "author", "release", "html_theme", "extensions"):
        assert key in namespace, f"conf.py missing data global: {key}"
    assert namespace["html_theme"] == "furo"
    assert "myst_parser" in namespace["extensions"]  # type: ignore[operator]
```

- [ ] **Step 2: Run the test — import guard passes, conf.py test fails**

Run: `.venv/bin/python -m pytest tests/test_docs_build.py -v`
Expected: `test_all_osmose_modules_import` PASSES; `test_conf_py_is_import_safe_and_defines_data_globals` FAILS with `FileNotFoundError` (docs/conf.py does not exist yet).

- [ ] **Step 3: Commit**

```bash
git add tests/test_docs_build.py
git commit -m "test: guard that all osmose modules import + conf.py defines data globals"
```

---

## Task 3: Sphinx configuration (`docs/conf.py`)

**Files:**
- Create: `docs/conf.py`
- Test: `tests/test_docs_build.py` (from Task 2)

- [ ] **Step 1: Create `docs/conf.py`**

```python
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
```

- [ ] **Step 2: Run the guard test — both tests now pass**

Run: `.venv/bin/python -m pytest tests/test_docs_build.py -v`
Expected: both tests PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/conf.py
git commit -m "docs: add Sphinx conf.py (furo, recursive autosummary, myst, whitelist)"
```

---

## Task 4: Recursive autosummary template

The standard recursive recipe. It iterates the PUBLIC-filtered template variables
(`functions`/`classes`/`attributes`/`exceptions`/`modules`), never `all_*`, so
underscore-leaf modules (`osmose.engine._netcdf`, `osmose.__version__`) stay out
of the reference. A Python package has objtype `module`, so this single template
renders both modules and packages; there is no `package` template.

**Files:**
- Create: `docs/_templates/autosummary/module.rst`

- [ ] **Step 1: Create the template**

```jinja
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
```

- [ ] **Step 2: Sanity-check the file exists at the path autosummary expects**

Run: `test -f docs/_templates/autosummary/module.rst && echo OK`
Expected: prints `OK` (templates_path in conf.py points at `_templates`; autosummary looks under `autosummary/`).

- [ ] **Step 3: Commit**

```bash
git add docs/_templates/autosummary/module.rst
git commit -m "docs: add recursive autosummary module template (public-filtered)"
```

---

## Task 5: API root + landing page; first (non-strict) build generates stubs

**Files:**
- Create: `docs/api/index.rst`
- Create: `docs/index.md`

- [ ] **Step 1: Create `docs/api/index.rst`**

```rst
API Reference
=============

The full ``osmose`` library API, generated from source. Submodules and
subpackages are discovered recursively.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   osmose
```

- [ ] **Step 2: Create `docs/index.md`**

The landing page and root document. Its toctrees list EVERY whitelisted page —
the narrative docs AND `api/index` — because `api/index.rst` is itself a normal
document (whitelisted via `api/**`) and would raise "document isn't included in
any toctree" (fatal under `-W`) unless entered into a toctree here.

```markdown
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
tutorials/30-minute-ecosystem
tutorials/fie-on-baltic-cod
:::

:::{toctree}
:maxdepth: 2
:caption: API Reference

api/index
:::
```

- [ ] **Step 3: Build WITHOUT `-W` to confirm the skeleton builds and stubs generate**

Run: `.venv/bin/sphinx-build -b html docs docs/_build/html`
Expected: exits 0 (warnings ARE expected at this stage — docstring RST quirks and the usage-guide link warnings; they are fixed in Tasks 6–7). The build must COMPLETE.

Run: `test -f docs/api/_autosummary/osmose.rst && echo STUBS_OK`
Expected: prints `STUBS_OK`. Recursive autosummary writes generated `.rst` STUBS
into the SOURCE dir `docs/api/_autosummary/`; the rendered `.html` lands only in
`docs/_build/html/api/_autosummary/` — so check the `.rst` stub, not an `.html`.

If `STUBS_OK` does NOT print, no stubs were generated — inspect the contents of
`docs/api/_autosummary/`. The likely cause is `api/index.rst` not being picked up
by the `api/**` whitelist glob; fallback: in `docs/conf.py`, add the explicit
entry `"api/index.rst"` to `include_patterns` (alongside `"api/**"`), then re-run.
(In Sphinx ≥7 `api/**` already matches `api/index.rst`, so this is only a safety net.)

- [ ] **Step 4: Commit**

```bash
git add docs/api/index.rst docs/index.md
git commit -m "docs: add API autosummary root and landing page with toctrees"
```

---

## Task 6: Rewrite `usage-guide.md` outbound links (eliminate `myst.xref_missing`)

`docs/usage-guide.md` is whitelisted, so MyST resolves its links. Six links point
either above the source root (`../README.md`, `../CHANGELOG.md` — never valid MyST
doc targets) or to in-`docs/` files excluded by the whitelist
(`baltic_example.md`, `baltic_ices_validation_2026-04-18.md`, `parity-roadmap.md`).
Each emits `myst.xref_missing`, fatal under `-W`. Rewrite all six to absolute
GitHub blob URLs (external links are not resolved as xrefs). Leave the two
`tutorials/30-minute-ecosystem.md` links relative — that page is whitelisted.

The same-document anchor `[§6](#6-choose-an-engine--reproduce-results)` in
`usage-guide.md:84` targets heading "## 6. Choose an engine & reproduce results",
whose actual Sphinx section id is `choose-an-engine-reproduce-results` (the
leading digit is dropped and "& " collapses to a single dash). With
`myst_heading_anchors = 3` (Task 3) the existing link resolves via MyST's
*tolerant* cross-reference matching (verified clean under `-W` in myst-parser
5.1.0), but Step 1 also rewrites it to the exact id so it does not depend on fuzzy
matching that a stricter future MyST could drop.

**Files:**
- Modify: `docs/usage-guide.md`

- [ ] **Step 1: Rewrite the six link targets**

Apply these exact substring replacements in `docs/usage-guide.md` (replace the
parenthesized link target only):

| Old link target | New link target |
|---|---|
| `(../README.md#quick-start)` | `(https://github.com/razinkele/osmopy/blob/master/README.md#quick-start)` |
| `(../README.md#api-sketch)` | `(https://github.com/razinkele/osmopy/blob/master/README.md#api-sketch)` |
| `(../CHANGELOG.md)` | `(https://github.com/razinkele/osmopy/blob/master/CHANGELOG.md)` |
| `(baltic_example.md)` | `(https://github.com/razinkele/osmopy/blob/master/docs/baltic_example.md)` |
| `(baltic_ices_validation_2026-04-18.md)` | `(https://github.com/razinkele/osmopy/blob/master/docs/baltic_ices_validation_2026-04-18.md)` |
| `(parity-roadmap.md)` | `(https://github.com/razinkele/osmopy/blob/master/docs/parity-roadmap.md)` |
| `(#6-choose-an-engine--reproduce-results)` | `(#choose-an-engine-reproduce-results)` |

- [ ] **Step 2: Verify no `myst.xref_missing` warnings remain**

Run: `.venv/bin/sphinx-build -b html docs docs/_build/html 2>&1 | grep -c "myst.xref_missing" || true`
Expected: prints `0` (zero MyST cross-reference warnings; docstring RST warnings may still appear — those are Task 7).

- [ ] **Step 3: Commit**

```bash
git add docs/usage-guide.md
git commit -m "docs: rewrite usage-guide outbound links to absolute URLs for -W build"
```

---

## Task 7: Make the `-W` build warning-free (iterative)

This is the open-ended cleanup. Warnings are discovered at build time. Fixes are
**docstring text edits** (and, if any surface, narrative-markdown/config edits) —
**never** changes to code behavior, signatures, or logic, and **never** a blanket
`suppress_warnings`.

(All `-W` builds in this plan use `--keep-going`, which intentionally supersedes
the spec's bare `sphinx-build -W`: the zero-warning gate is identical — both exit
non-zero on any warning — `--keep-going` only collects every warning per pass so
the cleanup loop converges faster.)

**Files:**
- Modify: `osmose/**/*.py` docstrings as needed (text only)

- [ ] **Step 1: Run the strict build and capture the warnings**

Run: `rm -rf docs/_build docs/api/_autosummary && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html 2>&1 | tee /tmp/sphinx_warnings.txt`
Expected initially: exits non-zero; `/tmp/sphinx_warnings.txt` lists each warning with `file:line: WARNING: ...`. (`--keep-going` collects ALL warnings in one pass instead of stopping at the first. The `rm -rf` forces a from-scratch read so Sphinx's incremental cache cannot silently skip an unchanged-but-still-broken module — without it a partial fix can show a false "build succeeded" locally that then fails on the fresh-checkout CI build.)

- [ ] **Step 2: Fix each warning at its source**

For each warning, open the cited `file:line` and fix the docstring. Common classes and fixes:
- **RST inline-markup errors** ("Inline strong/emphasis ... without end-string", "Inline literal start-string without end-string"): an unescaped `*`, `` ` ``, or `_` in prose — wrap the term in double backticks (` ``like_this`` `) or escape with a backslash.
- **"Unexpected indentation" / "Block quote ends without a blank line"**: a wrapped line or list item is mis-indented — align continuation lines and add the required blank line before/after blocks.
- **"Title underline too short"**: only in `.rst` docstrings — extend the underline to match the title length.
- **"duplicate object description"**: a symbol documented twice (e.g. re-exported). Prefer documenting it once at its definition; if it is an intentional re-export, the autosummary public-filter usually avoids this — verify it is not caused by a manual `automodule` elsewhere.
- **"Undefined substitution referenced"** (a docutils ERROR — fatal under `-W`): a bar-delimited token in prose is parsed as a `|substitution|` reference. The real repo emits this for `osmose/analysis.py` (`|% change|` at lines ~240/287; field refs `|pct_delta|` / `|abs_delta|` at ~244). Fix by escaping the pipes (`\|% change\|`) or wrapping the phrase in double backticks (` ``% change`` `). Note it can appear with `<autosummary>:N` provenance because the one-line summary is re-parsed standalone.

Edit docstring TEXT only. Do not alter signatures, defaults, or logic.

- [ ] **Step 3: Re-run until clean**

Run: `rm -rf docs/_build docs/api/_autosummary && .venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html`
Repeat Step 2 until this command exits 0 with no warnings. (The `rm -rf` each
pass is essential — see Step 1; an incremental rebuild can hide a still-broken
module and report a false "build succeeded".)
Expected (final): `build succeeded.` and exit code 0.

- [ ] **Step 4: Confirm the runtime test suite still passes (no behavior changed)**

Run: `.venv/bin/python -m pytest -q -k "docstring or import or schema" tests/`
Then run the full suite once: `.venv/bin/python -m pytest -n auto -q`
Expected: all green (docstring edits must not change behavior). Also run lint:
`.venv/bin/ruff check osmose/` and `.venv/bin/ruff format --check osmose/` → clean.

- [ ] **Step 5: Commit**

```bash
git add osmose/
git commit -m "docs: make docstrings clean under sphinx-build -W"
```

---

## Task 8: GitHub Pages build + deploy workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: Docs

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v5

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        # numba is installed so autodoc renders the JIT code path identically;
        # osmose modules import fine without it (import-guarded pure fallback).
        run: pip install -e ".[docs]" numba

      - name: Build docs (warnings are errors)
        run: sphinx-build -W --keep-going -b html docs docs/_build/html

      - name: Upload Pages artifact
        # PR runs upload too (validates the artifact packs cleanly); only the
        # deploy job below publishes, and only on master.
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_build/html

  deploy:
    needs: build
    if: github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    # Serialize deploys so two quick master merges cannot race; let an
    # in-flight publish finish rather than cancelling it.
    concurrency:
      group: "pages"
      cancel-in-progress: false
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Validate the workflow YAML parses**

Run: `.venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/docs.yml')); print('YAML OK')"`
Expected: prints `YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add Docs workflow (sphinx -W build on PR; deploy to Pages on master)"
```

**Note (one-time manual, repo owner — cannot be done from the tree):** in GitHub
repo Settings → Pages, set **Source = "GitHub Actions"**. Required before the
first `deploy` job can publish.

---

## Task 9: Make the published site discoverable

The GitHub Pages URL for this project repo is deterministic:
`https://razinkele.github.io/osmopy/`. It returns 404 until Pages is enabled and
the first deploy completes, after which these links resolve. Docs/metadata only —
no runtime change.

**Files:**
- Modify: `README.md` (the "Documentation index" table, around line 321–334)
- Modify: `pyproject.toml`

- [ ] **Step 1: Add a row to the README Documentation index**

In `README.md`, in the `## Documentation index` table, add this row directly under
the header row (above the 30-minute-tutorial row at ~line 327):

```markdown
| Rendered docs site (API reference + guides) | [razinkele.github.io/osmopy](https://razinkele.github.io/osmopy/) |
```

- [ ] **Step 2: Add `[project.urls]` to pyproject**

In `pyproject.toml`, add a `[project.urls]` table (place it immediately after the
`[project.optional-dependencies]` block ends):

```toml
[project.urls]
Documentation = "https://razinkele.github.io/osmopy/"
Repository = "https://github.com/razinkele/osmopy"
```

- [ ] **Step 3: Verify pyproject still parses**

Run: `.venv/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('TOML OK')"`
Expected: prints `TOML OK`.

- [ ] **Step 4: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: link the published Pages site from README and pyproject urls"
```

---

## Final verification (before finishing the branch)

- [ ] `.venv/bin/python -m pytest tests/test_docs_build.py -v` → both tests pass.
- [ ] `.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html` → exits 0, `build succeeded.`, zero warnings.
- [ ] `docs/api/_autosummary/osmose.rst` exists (and the rendered `docs/_build/html/api/_autosummary/osmose.html`); spot-check that a private-module stub `docs/api/_autosummary/osmose.engine._netcdf.rst` was NOT generated.
- [ ] `.venv/bin/python -m pytest -n auto -q` → full suite green (no behavior changed).
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/` and `.venv/bin/ruff format --check osmose/ ui/ tests/` → clean.
- [ ] `.venv/bin/pyright --pythonpath .venv/bin/python` (or per repo convention) → no new errors in changed files.

Then use **superpowers:finishing-a-development-branch** to open the PR.

---

## Spec coverage map

- Whitelist + unified site (API + narrative) → Tasks 3, 5.
- Recursive autosummary, public-filtered → Tasks 4, 5.
- `-W` build clean (narrative links, anchor, docstrings) → Tasks 3 (myst_heading_anchors), 6, 7.
- Furo / autodoc_typehints / intersphinx timeout / import-safe conf → Task 3.
- `docs.yml` (pinned actions, python 3.12, permissions, concurrency, no `.nojekyll`) → Task 8.
- Guard test (import-all + conf data globals, dep-superset caveat) → Task 2.
- One-time Pages enablement → Task 8 note.
- Discoverability (README + `[project.urls]`) → Task 9.
- Out of scope (ui/mcp/scripts, versioned docs, custom CSS, `__all__`) → not implemented, per spec.
