# Sphinx API Reference — Design

**Date:** 2026-06-21
**Status:** Approved (design)
**Author:** brainstorming session

## Goal

Stand up an auto-generated documentation site for the `osmose/` library using
Sphinx + `myst-parser`, published to GitHub Pages. The site combines a
recursive, self-maintaining API reference with the existing narrative
documentation (usage guide, 30-minute tutorial). The docs build is gated in CI
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
page, the two narrative docs, and the generated API tree are treated as pages.
The loose diagnostic / plan / perf markdown already in `docs/` stays ignored,
which avoids "document is not included in any toctree" warnings.

```
docs/
  conf.py                          # Sphinx config
  index.md                         # landing page: intro + install + toctrees (myst)
  api/
    index.rst                      # autosummary :recursive: root over `osmose`
  _templates/autosummary/
    module.rst                     # recursive-autosummary template (module)
    package.rst                    # recursive-autosummary template (package)
  usage-guide.md                   # EXISTING — linked into a "Guides" toctree
  tutorials/
    30-minute-ecosystem.md         # EXISTING — linked into "Guides"
  _build/                          # build output (gitignored)
  api/_autosummary/                # generated stubs (gitignored)
.github/workflows/docs.yml         # build (-W) on PR; build + deploy to Pages on master
pyproject.toml                     # new [project.optional-dependencies] docs extra
tests/test_docs_build.py           # autodoc import-guard + conf.py importability
.gitignore                         # add docs/_build/ and generated autosummary dir
```

Notes:
- The narrative files (`usage-guide.md`, `tutorials/30-minute-ecosystem.md`) are
  referenced verbatim from `docs/index.md` toctrees. Light markdown cleanup is
  applied only if needed to render cleanly under `-W`.
- The generated autosummary stub directory is gitignored; it is recreated on
  every build by `autosummary_generate = True`.

## Sphinx configuration (`conf.py`)

- **Extensions**:
  - `sphinx.ext.autodoc`
  - `sphinx.ext.autosummary` (`autosummary_generate = True`)
  - `sphinx.ext.napoleon` (Google + NumPy docstring sections)
  - `sphinx.ext.intersphinx` (python, numpy, pandas, xarray, scipy) — left
    **non-nitpicky** so unresolved cross-references do not fail the `-W` build
  - `sphinx.ext.viewcode` (source links)
  - `myst_parser` (markdown pages)
- **Type hints**: `autodoc_typehints = "description"` — renders the codebase's
  annotations into parameter docs. `sphinx-autodoc-typehints` is intentionally
  NOT added (fewer deps, smaller `-W` warning surface).
- **Theme**: `html_theme = "furo"`.
- **Whitelist**: `include_patterns` limited to `index.md`, `usage-guide.md`,
  `tutorials/30-minute-ecosystem.md`, and `api/**`.
- `nitpicky = False`.
- `project`, `author`, `release` sourced from `osmose.__version__`.
- `add_module_names = False` (cleaner symbol names).

## Recursive autosummary

`docs/api/index.rst` contains a single autosummary block that points at the
top-level package with `:recursive:`:

```rst
.. autosummary::
   :toctree: _autosummary
   :recursive:

   osmose
```

`docs/_templates/autosummary/module.rst` and `package.rst` follow the standard
recursive recipe: each generated page lists the module's classes, functions, and
data, and recurses into submodules/subpackages. This is what makes the reference
self-maintaining.

## Dependencies & CI

### pyproject extra

```toml
[project.optional-dependencies]
docs = ["sphinx>=7", "myst-parser>=2", "furo>=2024.1"]
```

### `.github/workflows/docs.yml`

Separate from `ci.yml` because Pages deploy needs special permissions and an
environment, and should only run on `master`.

- **`build` job** (runs on pull requests AND pushes to `master`):
  - `pip install .[docs] numba` (numba so all `osmose` modules import for autodoc)
  - `sphinx-build -W -b html docs docs/_build/html`
  - `actions/upload-pages-artifact` with `docs/_build/html`
- **`deploy` job** (runs on `master` only, `needs: build`):
  - permissions `pages: write`, `id-token: write`
  - environment `github-pages`
  - `actions/deploy-pages`

PR runs build with `-W` (catch breakage) but do not deploy.

### One-time manual step (repo owner)

In GitHub repo Settings → Pages, set **Source = "GitHub Actions"**. This cannot
be done from the working tree; it is a prerequisite for the first successful
deploy. Documented here and in the plan.

## Testing

- **`tests/test_docs_build.py`** (fast; runs in the normal pytest CI leg):
  - Walk `osmose` with `pkgutil.walk_packages` and import every submodule. Fails
    if any module is unimportable — this surfaces an autodoc-breaking change in
    the regular (fast) CI leg, before the slower Sphinx job runs. (A smoke run of
    this during design found 0 import failures across all `osmose` modules.)
  - Assert `docs/conf.py` is importable / exec-able and defines `project` and
    `html_theme`.
- **Authoritative build test**: the `-W` Sphinx build in `docs.yml`. A clean
  `-W` build is the merge gate for documentation correctness.

## Effort & risk

The dominant work is making the initial `-W` build clean. Auto-generated stubs
over ~25 top-level modules plus 9 subpackages will surface:
- docstring RST quirks (stray indentation, unescaped `*`/`` ` ``/`:`, malformed
  lists) → RST warnings,
- possible "duplicate object description" warnings for any re-exported symbol.

The implementation plan includes an **iterative task**: build locally, fix each
warning, repeat until the build is warning-free. All fixes are **docstring text
edits only** — no changes to code behavior, signatures, or logic.

Out of scope (YAGNI):
- Documenting `ui/`, `mcp_servers/`, `scripts/`, or `tests/` (library reference only).
- Versioned / multi-version docs (single `latest` site).
- Custom CSS / branding beyond the Furo defaults.
- Curating an explicit `osmose.__all__` public API (autosummary documents modules as-is).

## Success criteria

1. `sphinx-build -W -b html docs docs/_build/html` completes with zero warnings.
2. The site has a landing page, a "Guides" section (usage guide + 30-min
   tutorial), and a recursive "API Reference" covering all `osmose` modules.
3. `docs.yml` builds on PRs and deploys to GitHub Pages on `master`.
4. `tests/test_docs_build.py` passes in the normal CI leg.
5. No changes to runtime/library behavior — docstring edits and new docs/CI
   files only.
