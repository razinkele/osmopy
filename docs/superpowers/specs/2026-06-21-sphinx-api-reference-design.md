# Sphinx API Reference — Design

**Date:** 2026-06-21
**Status:** Approved (design)
**Author:** brainstorming session

## Goal

Stand up an auto-generated documentation site for the `osmose/` library using
Sphinx + `myst-parser`, published to GitHub Pages. The site combines a
recursive, self-maintaining API reference with the existing narrative
documentation (usage guide + both tutorials). The docs build is gated in CI
with warnings-as-errors so docstrings stay honest as the code grows.

There is no documentation tooling in the repo today: no `conf.py`, no Sphinx in
dependencies, no ReadTheDocs/Pages config. `osmose/__init__.py` exports only
`__version__`, so the public API is implicitly the modules themselves.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Publishing target | **GitHub Pages**, auto-deployed from CI on push to `master` |
| Site scope | **Unified** — API reference + existing narrative markdown in one site |
| CI strictness | **Warnings-as-errors** (`sphinx-build -W`) |
| Tooling | Sphinx + `myst-parser` (per backlog) |
| API generation | `sphinx.ext.autosummary` with `:recursive:` over the `osmose` package |
| Theme | **Furo** |
| Type hints | Built-in `autodoc_typehints = "description"` (no `sphinx-autodoc-typehints`) |

## Approach (chosen: A)

**A — Sphinx rooted at `docs/`, recursive autosummary, Furo, Pages via Actions.**
A single `conf.py` rooted at the existing `docs/` tree so the narrative markdown
is reusable in place. `autosummary` with `:recursive:` auto-discovers every
`osmose` module, so the API reference regenerates itself as code grows (zero
per-module maintenance). A dedicated `docs.yml` workflow builds with `-W` (PRs
build only; `master` builds + deploys).

Rejected alternatives:
- **B — Hand-written `.rst` stubs per subpackage.** More layout control, but each
  new module must be added by hand → maintenance burden; defeats "auto-generated".
- **C — MkDocs Material + mkdocstrings.** Simpler config but off-brief (backlog
  specifies Sphinx + myst-parser); Sphinx autosummary/intersphinx is the
  scientific-Python standard.

## Architecture & file layout

Sphinx source is rooted at the existing `docs/` directory. To keep the `-W`
build clean, `conf.py` uses an **`include_patterns` whitelist**: only the landing
page, the narrative docs, and the generated API tree are treated as pages.
The loose diagnostic / plan / perf markdown already in `docs/` stays ignored,
which avoids "document is not included in any toctree" warnings.

```
docs/
  conf.py                          # Sphinx config
  index.md                         # landing page: intro + install + toctrees (myst)
  api/
    index.rst                      # autosummary :recursive: root over `osmose`
  _templates/autosummary/
    module.rst                     # recursive-autosummary template (handles modules AND packages)
  usage-guide.md                   # EXISTING — linked into a "Guides" toctree
  tutorials/
    30-minute-ecosystem.md         # EXISTING — linked into "Guides"
    fie-on-baltic-cod.md           # EXISTING — linked into "Guides"
  _build/                          # build output (gitignored)
  api/_autosummary/                # generated stubs (gitignored)
.github/workflows/docs.yml         # build (-W) on PR; build + deploy to Pages on master
pyproject.toml                     # new [project.optional-dependencies] docs extra
tests/test_docs_build.py           # autodoc import-guard + conf.py importability
.gitignore                         # add docs/_build/ and generated autosummary dir
```

Notes:
- `docs/index.md` MUST contain toctree(s) that list **every** whitelisted page:
  the narrative docs (`usage-guide.md`, `tutorials/30-minute-ecosystem.md`,
  `tutorials/fie-on-baltic-cod.md`) **and `api/index`** (the autosummary
  `:recursive:` root). The autosummary `:toctree:` option only links the
  *generated* `_autosummary/*` stubs; `api/index.rst` is itself a normal document
  and, because it is whitelisted via `api/**`, will raise "document isn't included
  in any toctree" (fatal under `-W`) unless `index.md` enters it into a toctree.
- The narrative markdown is reused in place but is **not** clean for a `-W` build
  as-is — see "Known `-W` warning classes" below. Its outbound cross-links must be
  rewritten (a required task, not "light cleanup").
- The generated autosummary stub directory is gitignored; it is recreated on
  every build by `autosummary_generate = True`.

## Sphinx configuration (`conf.py`)

- **Extensions**:
  - `sphinx.ext.autodoc`
  - `sphinx.ext.autosummary` (`autosummary_generate = True`)
  - `sphinx.ext.napoleon` (Google + NumPy docstring sections)
  - `sphinx.ext.intersphinx` — with an **explicit** `intersphinx_mapping`
    (python, numpy, pandas, xarray, scipy) and **`intersphinx_timeout = 10`**.
    Caveat: `nitpicky = False` suppresses *missing-reference* warnings, but it
    does **not** suppress the warning intersphinx emits when it cannot *download*
    a remote `objects.inv` — and `intersphinx_timeout` defaults to `None` (no
    timeout). Under `-W` a transient CI network failure to docs.python.org / etc.
    would non-deterministically red the gate. The timeout bounds the hang; if
    flakiness persists, intersphinx may be dropped for v1 (it only adds outbound
    cross-links, which `nitpicky = False` already makes non-load-bearing).
  - `sphinx.ext.viewcode` (source links)
  - `myst_parser` (markdown pages) with **`myst_heading_anchors = 3`** so
    same-document anchor links in the narrative docs (e.g. `usage-guide.md`'s
    `[§6](#…)`) resolve instead of raising `myst.xref_missing` under `-W`.
- **Type hints**: `autodoc_typehints = "description"` — renders the codebase's
  annotations into parameter docs. `sphinx-autodoc-typehints` is intentionally
  NOT added (fewer deps, smaller `-W` warning surface).
- **Theme**: `html_theme = "furo"`.
- **Whitelist**: `include_patterns` limited to `index.md`, `usage-guide.md`,
  `tutorials/30-minute-ecosystem.md`, `tutorials/fie-on-baltic-cod.md`, and
  `api/**`.
- `nitpicky = False`.
- `release` is sourced from `osmose.__version__` (`release =
  osmose.__version__.__version__`); `project` and `author` are hardcoded string
  literals in `conf.py` (e.g. `project = "OSMOSE Python"`,
  `author = "OSMOSE Python contributors"`) — `osmose.__version__` defines only
  `__version__`, no author.
- `add_module_names = False` (cleaner symbol names).
- **Import-safety contract**: `conf.py` MUST be importable/exec-able WITHOUT the
  `[docs]` extra installed (the fast `[dev]`-leg test exec's it). Keep all config
  as data (string-literal `extensions`, `html_theme = "furo"`); do not `import
  sphinx`/`import furo` at module top level — defer any extension/theme imports
  into a Sphinx `setup(app)` hook.

## Recursive autosummary

`docs/api/index.rst` contains a single autosummary block that points at the
top-level package with `:recursive:`:

```rst
.. autosummary::
   :toctree: _autosummary
   :recursive:

   osmose
```

A single `docs/_templates/autosummary/module.rst` follows the standard recursive
recipe (a Python package has objtype `module`, so the same template renders both
modules and packages; there is no `package` objtype/template). Each generated
page lists the module's classes, functions, and data, and recurses into
submodules/subpackages via the template's `modules` variable. This is what makes
the reference self-maintaining.

**Public-filter requirement:** the template MUST iterate the public-filtered
template variables (`modules`, `functions`, `classes`, `attributes`,
`exceptions`) and MUST NOT use `all_modules` / `all_functions` / etc. or pass
`:private-members:`. This keeps private modules out of the published reference —
the repo has two underscore-leaf modules (`osmose.engine._netcdf`,
`osmose.__version__`) that would otherwise leak. (The Sphinx-bundled default
template already does this; the requirement guards against a future hand-edit
silently defeating the "self-maintaining public API" goal.)

## Dependencies & CI

### pyproject extra

```toml
[project.optional-dependencies]
docs = ["sphinx>=7", "myst-parser>=2", "furo>=2024.1"]
```

### `.github/workflows/docs.yml`

Separate from `ci.yml` because Pages deploy needs special permissions and an
environment, and should only run on `master`.

**All actions are major-pinned** per the repo convention (`ci.yml` uses
`@v5`/`@v6`); `setup-python` pins `python-version: "3.12"`. Add a workflow-level
concurrency group so overlapping `master` deploys serialize instead of racing:

```yaml
concurrency:
  group: "pages"
  cancel-in-progress: false   # let an in-flight publish finish
```

- **`build` job** (runs on pull requests AND pushes to `master`):
  - `permissions: { contents: read }` (least-privilege; matches `visual.yml`)
  - `actions/checkout@v5`, `actions/setup-python@v6` (`python-version: "3.12"`)
  - `pip install .[docs] numba` — numba is installed so autodoc renders the JIT
    code path identically, **not** because modules need it to import (numba is
    `try/except ImportError`-guarded with a pure-Python fallback; all `osmose`
    modules import without it).
  - `sphinx-build -W -b html docs docs/_build/html`
  - `actions/upload-pages-artifact@v3` with `docs/_build/html`
    (the PR-leg upload intentionally validates the artifact packs cleanly;
    it is not deployed)
- **`deploy` job** (runs on `master` only, `needs: build`):
  - `permissions: { pages: write, id-token: write }`
  - environment `github-pages`
  - `actions/deploy-pages@v4`

PR runs build with `-W` (catch breakage) but do not deploy. `.nojekyll` is **not**
needed: `upload-pages-artifact`/`deploy-pages` serve the tarball without Jekyll,
so Furo's `_static`/`_sources` underscore dirs are preserved.

### One-time manual step (repo owner)

In GitHub repo Settings → Pages, set **Source = "GitHub Actions"**. This cannot
be done from the working tree; it is a prerequisite for the first successful
deploy. Documented here and in the plan.

### Post-first-deploy discoverability (gated on the live URL)

Once the first Pages deploy succeeds and the URL is known
(`https://razinkele.github.io/osmopy/`), wire it into the repo so the site is
discoverable — docs/metadata only, no runtime change:
- Add a row to `README.md`'s existing "Documentation index" table (and/or a docs
  badge near the top status line).
- Add a `[project.urls]` table to `pyproject.toml` with a `Documentation` entry.

This is gated on the live URL being known so the link is never dead.

## Testing

- **`tests/test_docs_build.py`** (fast; runs in the normal pytest CI leg):
  - Walk `osmose` with `pkgutil.walk_packages` and import every submodule. Fails
    if any module is unimportable — this surfaces an autodoc-breaking change in
    the regular (fast) CI leg, before the slower Sphinx job runs. (A smoke run of
    this during design found 0 import failures across all `osmose` modules.)
    Caveat: this leg runs under the `[dev]` extra, a **superset** of the docs
    job's deps (`[docs] + numba`). It therefore cannot catch a regression where a
    module gains a top-level import of a dev-only dep (e.g. `httpx`, `pillow`) —
    that class is caught only by the authoritative `-W` docs build. No such
    module-level imports exist today.
  - Exec `docs/conf.py` and assert only its **data** globals (`project`,
    `author`, `release`, `html_theme`, `extensions`) — this keeps the test
    runnable in the `[dev]` leg without Sphinx installed (see the import-safety
    contract above).
- **Authoritative build test**: the `-W` Sphinx build in `docs.yml`. A clean
  `-W` build is the merge gate for documentation correctness.

## Known `-W` warning classes & required fixes

The whitelist + `-W` design *guarantees* the first build fails on the existing
narrative markdown unless these are fixed first. This was empirically reproduced
with Sphinx 9.1.0 + myst-parser (6 `myst.xref_missing` errors + 1 heading-anchor
error). These are **required tasks**, not "light cleanup", and they are
markdown/config edits — distinct from the docstring class below:

1. **Out-of-srcdir links in `docs/usage-guide.md`** (lines 7, 249, 250):
   `../README.md#quick-start`, `../README.md#api-sketch`, `../CHANGELOG.md`. Files
   above the `docs/` source root can never be MyST doc targets → rewrite to
   absolute GitHub blob URLs
   (`https://github.com/razinkele/osmopy/blob/master/README.md#quick-start`, …).
2. **In-tree-but-whitelist-excluded links in `docs/usage-guide.md`** (lines
   246–248): `baltic_example.md`, `baltic_ices_validation_2026-04-18.md`,
   `parity-roadmap.md`. They exist in `docs/` but are excluded by the whitelist →
   rewrite to GitHub blob URLs (adding them to the whitelist instead would
   reintroduce the orphan warnings the whitelist exists to suppress).
3. **Same-document anchor in `docs/usage-guide.md`** (line 84):
   `[§6](#6-choose-an-engine--reproduce-results)` → resolved by setting
   `myst_heading_anchors = 3` in `conf.py` (preferred — also covers future
   anchors).

Do **not** use a blanket `suppress_warnings = ["myst.xref_missing"]` — it would
hide genuinely broken cross-references site-wide, defeating the "docs stay honest"
goal.

## Effort & risk

After the narrative fixes above, the dominant remaining work is making the
auto-generated API stubs `-W`-clean. Auto-generated stubs over ~25 top-level
modules plus 7 top-level subpackages (10 including the nested
`engine.economics` / `engine.genetics` / `engine.processes`) will surface:
- docstring RST quirks (stray indentation, unescaped `*`/`` ` ``/`:`, malformed
  lists) → RST warnings,
- possible "duplicate object description" warnings for any re-exported symbol.

The implementation plan includes an **iterative task**: build locally, fix each
warning, repeat until the build is warning-free. Fixes are **docstring text edits
and the narrative-markdown/config edits listed above** — no changes to code
behavior, signatures, or logic.

Out of scope (YAGNI):
- Documenting `ui/`, `mcp_servers/`, `scripts/`, or `tests/` (library reference only).
- Versioned / multi-version docs (single `latest` site).
- Custom CSS / branding beyond the Furo defaults.
- Curating an explicit `osmose.__all__` public API (autosummary documents modules as-is).

## Success criteria

1. `sphinx-build -W -b html docs docs/_build/html` completes with zero warnings.
2. The site has a landing page, a "Guides" section (usage guide + both tutorials:
   30-minute-ecosystem and fie-on-baltic-cod), and a recursive "API Reference"
   covering all `osmose` modules (private/underscore modules excluded).
3. `docs.yml` builds on PRs and deploys to GitHub Pages on `master`.
4. `tests/test_docs_build.py` passes in the normal CI leg.
5. No changes to runtime/library behavior — docstring edits, narrative-markdown
   link/anchor edits, and new docs/CI files only.
6. After the first deploy, the published URL is linked from `README.md` and
   `pyproject.toml` `[project.urls]`.
