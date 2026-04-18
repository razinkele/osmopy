# Codebase Review Fixes Implementation Plan

> **STATUS (verified 2026-04-18): COMPLETE — shipped as merge `2c45323` → release `e2bedc2` (v0.8.1, 2026-04-17). DO NOT RE-EXECUTE.** Evidence: `_require_creds()` at `mcp_servers/copernicus/server.py:37` (called at 174, 598); `imax_trait: NDArray[np.float64] | None = None` declared at `osmose/engine/state.py:81`; `_require_preflight` at `ui/pages/calibration_handlers.py:24` (called at 411); `TimeSeries.get(step: int)` protocol at `osmose/engine/timeseries.py:64` with aligned concrete implementations.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every actionable finding in `docs/review-findings.md` (the 2026-04-17 deep-review re-verification) — rotate exposed credentials, burn the 22 Pyright errors down to zero, re-enable Java tests in pom/workflow, and document the local-Maven dependency chain.

**Architecture:** Phased, TDD-first. Phase 1 is a security-first credential cleanup that can ship standalone. Phase 2 is a cluster-by-cluster Pyright burn-down that leaves CI gate-ready. Phase 3 tightens the Java build. Phase 4 is documentation + lint-debt cleanup in non-CI-scoped helpers.

**Tech Stack:** Python 3.12 / Ruff / Pyright (basic mode) / pytest / GitHub Actions / Maven / Java 11.

---

## Re-verified current state (2026-04-17)

Every finding in `docs/review-findings.md` was re-checked from cwd `/home/razinka/osmose/osmose-python` before this plan was written:

| Finding | Status | Evidence |
|---|---|---|
| 2401 tests passing | Engine passes; suite may now be 2169/15 (per MEMORY.md) — verify in Task 0 | `.venv/bin/python -m pytest` |
| Ruff passes | **Partially true:** `ruff check osmose/ ui/ tests/` (CI scope) passes; `ruff check .` fails with 6 errors (4 fixable) in `mcp_servers/copernicus/server.py` and `scripts/calibrate_baltic.py` | `.venv/bin/ruff check .` |
| 22 Pyright errors | **Confirmed:** exact count = 22 across 5 files, detailed triage below | `.venv/bin/python -m pyright` |
| Committed Copernicus credentials | **Mitigated but not gone:** password `Razinka@2026` and email are in the *working tree* (`.mcp.json` unstaged diff + untracked `mcp_servers/copernicus/server.py`). Not yet in git history, but one `git add -A && commit` will leak them. | `git log -S CMEMS_PASSWORD` returns empty; `git status --short` shows ` M .mcp.json` and `?? mcp_servers/copernicus/server.py`; `grep CMEMS_PASSWORD .mcp.json mcp_servers/copernicus/server.py` shows the literal password |
| `pom.xml` skips tests by default | **Confirmed:** `<skipTests>true</skipTests>` at `osmose-master/pom.xml:59` | read file |
| Workflow packages with `-DskipTests=true` | **Confirmed:** `mvn -B package -DskipTests=true` at `osmose-master/.github/workflows/java-compile.yml:58` | read file |
| 18 Java test files exist but never run | Confirmed | `find osmose-master/java/src/test -name "*.java"` |

**Updated hotspots** (`osmose-python`, last 90 days): `engine/config.py` 66, `engine/simulate.py` 57, `ui/pages/grid.py` 44, `app.py` 37, `ui/pages/results.py` 37, `processes/mortality.py` 35, `ui/pages/run.py` 34, `ui/pages/calibration.py` 28. No split required — review recommended "safe change strategy" rather than restructuring, and memory notes I-3 `from_dict` split already shipped in v0.7.0.

## Pyright triage (22 errors → 8 fix clusters)

| Cluster | Error count | File:lines | Root cause | Fix strategy |
|---|---|---|---|---|
| A — Optional `ctx.prey_density_scale` / `access_matrix` / `sa_obj` | 6 | `processes/mortality.py:1743, 1747, 1748, 1751, 1752, 1753` | Nested Optional attribute access; Pyright cannot narrow `config.stage_accessibility` across the `if ctx is not None and ctx.prey_density_scale is not None and has_access:` guard | Assign `sa_obj = config.stage_accessibility; assert sa_obj is not None` locally; also widen `access_matrix` return type in its producer |
| B — `access_matrix` union → `NDArray[float64]` | 2 | `processes/mortality.py:1757, 1765` | `apply_prey_scale_to_matrix` param is strict `NDArray[float64]`, caller passes union that includes `None` | Narrow with `assert access_matrix is not None` before first call |
| C — `SchoolState.imax_trait` attribute | 2 | `processes/mortality.py:296, 304` | `hasattr(state, "imax_trait")` guard is not understood by Pyright because `SchoolState` lacks the attribute declaration | Declare `imax_trait: NDArray[np.float64] \| None = None` on the dataclass/class; drop `hasattr` in favor of `is not None` |
| D — Optional config arrays (`foraging_k1_for`, `foraging_k2_for`, `bioen_k_for`) | 3 | `processes/mortality.py:302, 303, 305` | Attribute-access narrowing is not retained across the `if genetic:` branch | Hoist to locals after narrowing: `k1 = config.foraging_k1_for; assert k1 is not None` etc. |
| E — `bioen_starvation(eta: float)` | 1 | `processes/mortality.py:108` | `config.bioen_eta[sp_i]` is `NDArray \| None` element (unions to `ndarray \| float`) | Wrap with `float(...)` at call site |
| F — `fishing_catches` Optional subscript | 1 | `processes/fishing.py:197` | Same pattern as Cluster D | Narrow to local with assertion |
| G — `TimeSeries` protocol mismatch | 2 | `engine/timeseries.py:422, 428` | `ByYearTimeSeries.get(year: int)` has a parameter name mismatch with protocol's `get(step: int)`, and `ByClassTimeSeries` has no `get()` at all. `load_timeseries` return type is too narrow. | Broaden factory return type to `SingleTimeSeries \| ByYearTimeSeries \| ByClassTimeSeries`. Rename `ByYearTimeSeries.get(year)` → `get(step)` keeping semantics of "year index". |
| H — `calibration.py: Never is not iterable` | 1 | `ui/pages/calibration.py:299` | `cal_X = reactive.value(None)` infers generic as `None`, so narrowing `if X is None` leaves `Never` on the else branch | Annotate: `cal_X: reactive.value[np.ndarray \| None] = reactive.value(None)` (same for `cal_F`) |
| I — `calibration_charts.py: columns=list[str]` | 1 | `ui/pages/calibration_charts.py:84` | `param_names` incoming type is ambiguous (Axes collision with pandas stub) | Annotate parameter as `param_names: list[str]`; change call to `pd.DataFrame(X, columns=list(param_names))` to coerce |
| J — `calibration_handlers.py: Path \| None → Path` | 3 | `ui/pages/calibration_handlers.py:408, 409, 410` | Nonlocal `_shared_*` initialized to literal `None`, so Pyright infers `None` type; problem constructor requires `Path` | Annotate the three nonlocal closures as `Path \| None`; add `assert` guards at the call site that raise `RuntimeError("preflight must run first")` if unset |

Total: **22 errors → 8 clusters → 8 TDD fix tasks.**

---

## Phased execution order

1. **Phase 1 — Credential hygiene (Tasks 1–3):** Highest severity. Can merge independently.
2. **Phase 2 — Pyright burn-down (Tasks 4–11):** Cluster-by-cluster, TDD (regression test first, then fix, ratchet CI at the end).
3. **Phase 3 — Java build discipline (Tasks 12–14):** Re-enable tests in pom and workflow.
4. **Phase 4 — Documentation & lint polish (Tasks 15–16):** Document local Maven, fix script ruff errors.

Each phase leaves a working, shippable state. Commit after every task.

---

### Task 0: Baseline verification

**Files:**
- None (read-only)

- [ ] **Step 1: Confirm baseline counts**

Run from `/home/razinka/osmose/osmose-python`:

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
.venv/bin/python -m pyright 2>&1 | tail -1
.venv/bin/ruff check osmose/ ui/ tests/ 2>&1 | tail -2
```

Expected:
- pytest: "N passed, 15 skipped" (record N in the PR description)
- pyright: "22 errors, 0 warnings, 0 informations"
- ruff (CI scope): "All checks passed!"

- [ ] **Step 2: Record baselines in the PR description**

If pytest count has changed since memory (2169), note the drift. Do **not** proceed if pytest is red — fix first, then resume this plan.

---

## Phase 1: Credential hygiene

### Task 1: Remove hardcoded credential fallbacks from `server.py`

**Files:**
- Modify: `osmose-python/mcp_servers/copernicus/server.py:30-31`
- Test: `osmose-python/tests/test_copernicus_mcp_env.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_copernicus_mcp_env.py
"""CMEMS credentials must come from env; no default fallback is allowed."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_servers" / "copernicus" / "server.py"


def test_server_py_has_no_hardcoded_credentials():
    src = SERVER_PATH.read_text()
    # No password literal, no quoted email default for CMEMS_USERNAME
    assert "Razinka@2026" not in src
    # The os.environ.get call must not have a string default for password
    pw_line = next(
        (line for line in src.splitlines() if "CMEMS_PASSWORD" in line and "os.environ" in line),
        None,
    )
    assert pw_line is not None, "CMEMS_PASSWORD env lookup not found"
    assert not re.search(r"os\.environ\.get\(\s*['\"]CMEMS_PASSWORD['\"]\s*,\s*['\"]", pw_line), (
        f"CMEMS_PASSWORD must not have a hardcoded string default: {pw_line!r}"
    )
    user_line = next(
        (line for line in src.splitlines() if "CMEMS_USERNAME" in line and "os.environ" in line),
        None,
    )
    assert user_line is not None
    assert not re.search(r"os\.environ\.get\(\s*['\"]CMEMS_USERNAME['\"]\s*,\s*['\"]", user_line)


def test_server_py_module_globals_reflect_env(monkeypatch):
    """Load server.py via spec_from_file_location (mcp_servers is not a package)."""
    monkeypatch.delenv("CMEMS_USERNAME", raising=False)
    monkeypatch.delenv("CMEMS_PASSWORD", raising=False)
    spec = importlib.util.spec_from_file_location("_copernicus_server_test", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert getattr(mod, "CMEMS_USER", "sentinel") is None
    assert getattr(mod, "CMEMS_PASS", "sentinel") is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_copernicus_mcp_env.py -v
```

Expected: `FAILED test_server_py_has_no_hardcoded_credentials` (literal `Razinka@2026` still present).

- [ ] **Step 3: Edit `mcp_servers/copernicus/server.py`**

Replace lines 30-31 with:

```python
CMEMS_USER: str | None = os.environ.get("CMEMS_USERNAME")
CMEMS_PASS: str | None = os.environ.get("CMEMS_PASSWORD")
```

Update the `_login` helper (currently around line 159) so missing credentials raise a clear error:

```python
def _login() -> None:
    """Ensure CMEMS credentials are configured."""
    if not CMEMS_USER or not CMEMS_PASS:
        raise RuntimeError(
            "CMEMS_USERNAME and CMEMS_PASSWORD environment variables must be set. "
            "See mcp_servers/copernicus/README.md."
        )
    cm.login(username=CMEMS_USER, password=CMEMS_PASS, force_overwrite=True)
```

Also fix the second call site (around line 586-587) to use the same variables and rely on `_login` pre-check.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_copernicus_mcp_env.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_copernicus_mcp_env.py mcp_servers/copernicus/server.py
git commit -m "security: require CMEMS credentials from env, remove hardcoded fallback"
```

---

### Task 2: Scrub credentials from `.mcp.json` and gate with `.env.example`

**Files:**
- Modify: `.mcp.json` (remove `env` block with real creds, keep server registration)
- Create: `.env.example` (template users copy to `.env`)
- Create: `mcp_servers/copernicus/README.md` (setup instructions)
- Modify: `.gitignore` (already ignores `.env`; no change needed — verify)

**Context:** `.mcp.json` is tracked. The current unstaged diff adds literal credentials under `"copernicus-marine": { "env": { "CMEMS_PASSWORD": "Razinka@2026" } }`. The correct pattern: commit the server registration without an `env` block, and require the user to export `CMEMS_USERNAME` / `CMEMS_PASSWORD` in their shell (or via a `.env` loaded by their shell init) before starting Claude Code.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_config_hygiene.py
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_json_has_no_cmems_password():
    cfg = json.loads((REPO_ROOT / ".mcp.json").read_text())
    text = json.dumps(cfg)
    assert "CMEMS_PASSWORD" not in text or '"CMEMS_PASSWORD": ""' in text, (
        ".mcp.json must not ship a CMEMS_PASSWORD value"
    )
    assert "Razinka@2026" not in text


def test_env_example_documents_cmems_vars():
    p = REPO_ROOT / ".env.example"
    assert p.exists(), ".env.example must document required env vars"
    body = p.read_text()
    assert "CMEMS_USERNAME" in body
    assert "CMEMS_PASSWORD" in body
```

- [ ] **Step 2: Run to verify both fail**

```bash
.venv/bin/python -m pytest tests/test_mcp_config_hygiene.py -v
```

Expected: FAIL (password is present, `.env.example` does not exist).

- [ ] **Step 3: Edit `.mcp.json`**

Change the `copernicus-marine` entry so the `env` block is absent (let the server inherit the parent shell env):

```json
"copernicus-marine": {
  "command": "/home/razinka/osmose/osmose-python/.venv/bin/python",
  "args": ["/home/razinka/osmose/osmose-python/mcp_servers/copernicus/server.py"]
}
```

- [ ] **Step 4: Create `.env.example`**

```
# Copernicus Marine Service (CMEMS) credentials
# Copy this file to .env and fill in the values, OR export these in your shell.
# Register at https://data.marine.copernicus.eu
CMEMS_USERNAME=
CMEMS_PASSWORD=
```

- [ ] **Step 5: Create `mcp_servers/copernicus/README.md`**

```markdown
# Copernicus Marine MCP Server

Downloads CMEMS datasets as OSMOSE-compatible NetCDF forcing files.

## Setup

1. Register at https://data.marine.copernicus.eu to obtain credentials.
2. Export credentials in your shell **before** launching Claude Code:

   ```bash
   export CMEMS_USERNAME="you@example.com"
   export CMEMS_PASSWORD="your-password"
   ```

   Or copy `.env.example` to `.env` and source it from your shell init.
3. Credentials are consumed by `server.py` via `os.environ.get`.
   There is **no hardcoded fallback** — missing env vars raise `RuntimeError`.

## Security

Never commit `CMEMS_PASSWORD` into `.mcp.json`, `server.py`, or any tracked file.
`tests/test_copernicus_mcp_env.py` and `tests/test_mcp_config_hygiene.py` enforce this.
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_copernicus_mcp_env.py tests/test_mcp_config_hygiene.py -v
```

Expected: 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add .mcp.json .env.example mcp_servers/copernicus/README.md tests/test_mcp_config_hygiene.py
git commit -m "security: remove committed CMEMS credentials from .mcp.json, document env setup"
```

---

### Task 3: Rotate the CMEMS password (user action)

**Files:** None (external action).

- [ ] **Step 1: Rotate the password at https://data.marine.copernicus.eu**

Log in to the Copernicus portal and change the password. The current value `Razinka@2026` must be treated as compromised — it was present in plaintext in the working tree of a folder that has been indexed by editors, shells, and possibly Claude transcripts.

- [ ] **Step 2: Update local shell env with the new password**

In the user's shell rc file or `.env`:

```bash
export CMEMS_USERNAME="arturas.razinkovas-baziukas@ku.lt"
export CMEMS_PASSWORD="<new password>"
```

- [ ] **Step 3: Verify the server still works**

Manually start the copernicus MCP server via Claude Code; confirm `_login` no longer raises.

- [ ] **Step 4: Record in memory**

Add a memory note that the CMEMS password was rotated on `2026-04-17` and that `.mcp.json` must never carry it.

*(No commit — this is a user-side action.)*

---

## Phase 2: Pyright burn-down (Clusters A–J)

**Ground rules for every task in this phase:**
- Add a unit test **only** when the Pyright fix also has a runtime-observable contract (e.g. "raises on missing preflight"). Not every type fix needs a runtime test — pyright itself is the check. For pure-annotation fixes, the "test" is `.venv/bin/python -m pyright` returning zero errors for that file.
- After each task: re-run `.venv/bin/python -m pyright` and confirm the targeted errors drop to zero; also re-run the full pytest suite to catch runtime regressions.

### Task 4: Cluster C — `SchoolState.imax_trait` attribute declaration

**Files:**
- Modify: `osmose/engine/schools.py` (or wherever `SchoolState` is defined — find via `grep -rn "class SchoolState"`)
- Modify: `osmose/engine/processes/mortality.py:292-305`

- [ ] **Step 1: Locate the `SchoolState` class**

```bash
grep -rn "class SchoolState" osmose/
```

Record the file path.

- [ ] **Step 2: Declare the optional attribute**

In the `SchoolState` class body, add (preserving existing field ordering conventions):

```python
imax_trait: NDArray[np.float64] | None = None
```

Ensure `NDArray` is imported at the top of the file:

```python
from numpy.typing import NDArray
```

- [ ] **Step 3: Simplify the guard in `mortality.py:292-297`**

Replace:

```python
genetic = (
    config.foraging_k1_for is not None
    and config.foraging_k2_for is not None
    and hasattr(state, "imax_trait")
    and state.imax_trait is not None
)
```

with:

```python
genetic = (
    config.foraging_k1_for is not None
    and config.foraging_k2_for is not None
    and state.imax_trait is not None
)
```

- [ ] **Step 4: Verify Pyright drops Cluster C errors**

```bash
.venv/bin/python -m pyright osmose/engine/processes/mortality.py 2>&1 | grep -c "imax_trait\|SchoolState"
```

Expected: `0` (the two `Cannot access attribute "imax_trait"` errors are gone).

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: same pass count as baseline.

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/schools.py osmose/engine/processes/mortality.py
git commit -m "types: declare SchoolState.imax_trait to fix Pyright hasattr narrow"
```

---

### Task 5: Cluster D + F — Hoist Optional config arrays before use

**Files:**
- Modify: `osmose/engine/processes/mortality.py:295-313`
- Modify: `osmose/engine/processes/fishing.py:193-204`

- [ ] **Step 1: Edit `mortality.py` genetic branch**

Hoist all three Optional arrays to locals inside `if genetic:`:

```python
if genetic:
    k1_arr = config.foraging_k1_for
    k2_arr = config.foraging_k2_for
    assert k1_arr is not None and k2_arr is not None  # narrowed by `genetic`
    rate = foraging_rate(
        k_for=None,
        ndt_per_year=config.n_dt_per_year,
        k1_for=np.array([k1_arr[sp_i]]),
        k2_for=np.array([k2_arr[sp_i]]),
        imax_trait=np.array([state.imax_trait[idx]]),
        I_max=np.array([config.foraging_I_max[sp_i]]),
    )
```

If `config.foraging_I_max` is also Optional (check its annotation first), hoist it the same way.

- [ ] **Step 2: Edit `fishing.py` around line 197**

Hoist `config.fishing_catches` at the top of the species loop:

```python
catches = config.fishing_catches
assert catches is not None, "fishing_catches must be loaded before catch allocation"
for sp_i in range(config.n_species):
    sp_mask = sp == sp_i
    annual_catch = catches[sp_i]
    ...
```

Move the hoist outside the loop if `fishing_catches` is loop-invariant.

- [ ] **Step 3: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep -E "mortality\.py:30[2-5]|fishing\.py:197"
```

Expected: no matches.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/mortality.py osmose/engine/processes/fishing.py
git commit -m "types: hoist Optional config arrays to locals for Pyright narrowing"
```

---

### Task 6: Cluster E — Cast `eta` to `float`

**Files:**
- Modify: `osmose/engine/processes/mortality.py:106-108`

- [ ] **Step 1: Edit the call site**

Replace:

```python
eta = config.bioen_eta[sp_i] if config.bioen_eta is not None else 1.0
n_dead_arr, _new_gonad = bioen_starvation(e_net, gonad_w, weight, eta, n_subdt)
```

with:

```python
eta = float(config.bioen_eta[sp_i]) if config.bioen_eta is not None else 1.0
n_dead_arr, _new_gonad = bioen_starvation(e_net, gonad_w, weight, eta, n_subdt)
```

- [ ] **Step 2: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep "mortality.py:108"
```

Expected: no match.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "types: cast bioen eta to float to match bioen_starvation signature"
```

---

### Task 7: Cluster A + B — Narrow `stage_accessibility` and `access_matrix`

**Files:**
- Modify: `osmose/engine/processes/mortality.py:1735-1770` (the prey-scale block)

- [ ] **Step 1: Inspect the producer of `access_matrix`**

```bash
grep -n "access_matrix\s*=" osmose/engine/processes/mortality.py | head -20
```

If its declared return type is `NDArray | None`, add an `assert access_matrix is not None` at the top of the `if ctx is not None and ctx.prey_density_scale is not None and has_access:` block. If its producer can truly return `None`, early-`return`/`continue` above instead of asserting.

- [ ] **Step 2: Hoist and assert `stage_accessibility`**

Inside `if use_stage_access:`, replace:

```python
sa_obj = config.stage_accessibility
for sp_idx, sp_name in enumerate(config.all_species_names[: config.n_species]):
    for _norm in (sp_name, sp_name.lower(), sp_name.replace(" ", "")):
        if _norm in sa_obj.prey_lookup:
            ...
```

with:

```python
sa_obj = config.stage_accessibility
assert sa_obj is not None, "stage_accessibility must be loaded when use_stage_access"
for sp_idx, sp_name in enumerate(config.all_species_names[: config.n_species]):
    ...
```

- [ ] **Step 3: Before each `apply_prey_scale_to_matrix` call (lines 1756 and 1764), narrow `access_matrix`**

```python
assert access_matrix is not None
access_matrix = apply_prey_scale_to_matrix(
    access_matrix,
    ctx.prey_density_scale,
    ...
)
```

(Place the assertion once above the `if use_stage_access:` switch so both branches benefit.)

- [ ] **Step 4: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep -E "mortality\.py:17(4[3-8]|5[1-3]|57|65)"
```

Expected: no matches (all 8 errors cleared).

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "types: narrow stage_accessibility and access_matrix for prey-scale path"
```

---

### Task 8: Cluster G — Fix `TimeSeries` protocol mismatch

**Files:**
- Modify: `osmose/engine/timeseries.py:61-64, 139-143, 401-428` (protocol, `ByYearTimeSeries.get`, factory return type)
- Test: `osmose/engine/tests/test_timeseries_load.py` (create if absent) or extend existing

**Context:** `ByYearTimeSeries.get(year: int)` breaks the protocol param name `step`. `ByClassTimeSeries` has no `get()` at all. Fix keeps the factory's return narrower on the caller side: broaden to a union, so callers must branch.

- [ ] **Step 1: Write the regression test**

```python
# Add to tests/test_timeseries.py or create if missing
from osmose.engine.timeseries import (
    ByClassTimeSeries,
    ByYearTimeSeries,
    SingleTimeSeries,
    load_timeseries,
)


def test_load_timeseries_scalar_returns_single():
    cfg = {"foo.sp0": "1.5"}
    ts = load_timeseries(cfg, "foo", 0, ndt_per_year=24, ndt_simu=240)
    assert isinstance(ts, SingleTimeSeries)
    assert ts.get(0) == 1.5


def test_byyear_timeseries_get_by_step_parameter_name():
    # Protocol requires parameter name "step"; ByYearTimeSeries must honor it
    import inspect

    sig = inspect.signature(ByYearTimeSeries.get)
    assert "step" in sig.parameters, (
        "ByYearTimeSeries.get must name its parameter `step` to match the TimeSeries protocol"
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_timeseries.py::test_byyear_timeseries_get_by_step_parameter_name -v
```

Expected: FAIL.

- [ ] **Step 3: Rename parameter in `ByYearTimeSeries.get`**

`osmose/engine/timeseries.py:139`:

```python
def get(self, step: int) -> float:
    # For by-year series, `step` is interpreted as the year index.
    # Callers that need step-to-year conversion use `get_for_step`.
    return float(self.values[step])
```

- [ ] **Step 4: Broaden the factory return type**

`osmose/engine/timeseries.py:401`:

```python
def load_timeseries(
    config: dict[str, str],
    key_prefix: str,
    species_idx: int,
    ndt_per_year: int,
    ndt_simu: int,
) -> SingleTimeSeries | ByYearTimeSeries | ByClassTimeSeries:
    ...
```

- [ ] **Step 5: Update callers of `load_timeseries`**

```bash
grep -rn "load_timeseries" osmose/ ui/ tests/
```

Audit each caller. If any relied on the narrow `TimeSeries` return, add an explicit `isinstance` branch. If the caller only needs `.get(step)`, no change needed (union still satisfies the protocol shape on `SingleTimeSeries` / renamed `ByYearTimeSeries`; callers using `ByClassTimeSeries` will now see a proper type error they must handle).

- [ ] **Step 6: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep "timeseries.py"
```

Expected: no match.

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/timeseries.py tests/test_timeseries.py
git commit -m "types: broaden load_timeseries return, align ByYearTimeSeries param name"
```

---

### Task 9: Cluster H — Annotate `reactive.value` generics in calibration.py

**Files:**
- Modify: `ui/pages/calibration.py:191-192`

- [ ] **Step 1: Edit the reactive value declarations**

```python
cal_F: reactive.value[np.ndarray | None] = reactive.value(None)
cal_X: reactive.value[np.ndarray | None] = reactive.value(None)
```

Ensure `np` and `reactive` imports are already present (they should be).

- [ ] **Step 2: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep "calibration.py:299"
```

Expected: no match.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q -k calibration
```

- [ ] **Step 4: Commit**

```bash
git add ui/pages/calibration.py
git commit -m "types: annotate reactive.value generics for cal_X/cal_F"
```

---

### Task 10: Cluster I — Fix pandas column typing in `calibration_charts.py`

**Files:**
- Modify: `ui/pages/calibration_charts.py:70-93` (function signature + call)

- [ ] **Step 1: Edit signature and call**

Locate the function header (around line 70) that takes `param_names`. Ensure the parameter is explicitly typed:

```python
def make_correlation_chart(
    X: np.ndarray,
    F: np.ndarray,
    param_names: list[str],
    *,
    tmpl: str = "osmose",
) -> go.Figure:
    ...
    df = pd.DataFrame(X, columns=list(param_names))  # explicit list to defeat stub overload ambiguity
```

- [ ] **Step 2: Verify Pyright**

```bash
.venv/bin/python -m pyright 2>&1 | grep "calibration_charts.py:84"
```

Expected: no match.

- [ ] **Step 3: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q -k calibration
```

- [ ] **Step 4: Commit**

```bash
git add ui/pages/calibration_charts.py
git commit -m "types: widen param_names annotation for pandas DataFrame columns"
```

---

### Task 11: Cluster J — Encode preflight invariant for calibration handlers

**Files:**
- Modify: `ui/pages/calibration_handlers.py:371-373, 405-412`
- Test: `tests/test_calibration_handlers_preflight.py` (create)

**Context:** `_shared_base_config`, `_shared_jar_path`, `_shared_work_dir` start as `None` and are populated by a preflight handler before the optimization handler runs. Annotate as `Path | None` and raise a clear error if the optimization handler is called without preflight.

- [ ] **Step 1: Write the regression test**

```python
# tests/test_calibration_handlers_preflight.py
"""If optimization runs without preflight, we must fail loudly (not silently pass None)."""
import pytest

# Not importing the whole Shiny module; instead probe the invariant via a helper.
# If no public helper exists, this test validates the runtime assertion shape.


def test_preflight_invariant_raises():
    """Placeholder — wired to actual handler in Task 11 implementation."""
    from ui.pages.calibration_handlers import _require_preflight  # will exist after fix

    with pytest.raises(RuntimeError, match="preflight"):
        _require_preflight(None, None, None)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_calibration_handlers_preflight.py -v
```

Expected: FAIL (`_require_preflight` doesn't exist).

- [ ] **Step 3: Add the helper and annotations**

In `ui/pages/calibration_handlers.py`, near the top of the module (outside any function):

```python
from pathlib import Path


def _require_preflight(
    base_config: Path | None,
    jar_path: Path | None,
    work_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Return non-Optional paths or raise if preflight never ran."""
    if base_config is None or jar_path is None or work_dir is None:
        raise RuntimeError(
            "Calibration preflight must run before optimization. "
            "Click 'Preflight' first in the Calibration tab."
        )
    return base_config, jar_path, work_dir
```

Update the three nonlocal declarations (line 371-373) with explicit annotations:

```python
_shared_base_config: Path | None = None
_shared_jar_path: Path | None = None
_shared_work_dir: Path | None = None
```

Replace the call-site block (lines 405-412) with:

```python
base_config, jar_path, work_dir = _require_preflight(
    _shared_base_config, _shared_jar_path, _shared_work_dir
)

problem = OsmoseCalibrationProblem(
    free_params=free_params,
    objective_fns=objective_fns,
    base_config_path=base_config,
    jar_path=jar_path,
    work_dir=work_dir,
    n_parallel=n_parallel,
)
```

Apply the same pattern to the second usage around line 616 (`nonlocal _shared_base_config`, the preflight handoff block).

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_calibration_handlers_preflight.py -v
.venv/bin/python -m pytest -q
```

Expected: new test PASS, full suite PASS.

- [ ] **Step 5: Verify Pyright clears Cluster J**

```bash
.venv/bin/python -m pyright 2>&1 | grep "calibration_handlers.py:40[89]\|calibration_handlers.py:410"
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add ui/pages/calibration_handlers.py tests/test_calibration_handlers_preflight.py
git commit -m "types: encode calibration preflight invariant via _require_preflight helper"
```

---

### Task 12: Ratchet CI to require zero Pyright errors

**Files:**
- Modify: `.github/workflows/ci.yml` (confirm `pyright` already runs and exit code is honored)

- [ ] **Step 1: Confirm total count is zero locally**

```bash
.venv/bin/python -m pyright 2>&1 | tail -1
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 2: Verify CI already fails on pyright errors**

```bash
grep -n "pyright" .github/workflows/ci.yml
```

If the line is just `- run: pyright`, no change needed — pyright exits non-zero when errors exist and GitHub Actions fails the step. If pyright is piped to `|| true` or similar, remove the bypass.

- [ ] **Step 3: No-op commit if no change**

If the workflow is already strict, skip the commit. Otherwise:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: fail build on Pyright errors (strict)"
```

---

## Phase 3: Java build discipline

### Task 13: Invert `skipTests` default in `pom.xml`

**Files:**
- Modify: `osmose-master/pom.xml:59`

- [ ] **Step 1: Edit the property**

Change:

```xml
<skipTests>true</skipTests>
```

to:

```xml
<skipTests>false</skipTests>
```

- [ ] **Step 2: Verify Maven is installed** (blocks this task on user if not)

```bash
which mvn
```

If missing, the user must install Maven (`sudo apt install maven` on Ubuntu) before running Step 3. Mark the task blocked and move on; resume after install.

- [ ] **Step 3: Run the tests locally**

```bash
cd /home/razinka/osmose/osmose-master && mvn test
```

Expected: a mix of pass/fail across the 18 test classes. **Do not silently make failing tests pass by skipping** — if some fail, record them and treat the failures as Phase 3 follow-up work in a new plan. The goal of this task is only to unskip; fixing legacy failures is out of scope.

- [ ] **Step 4: Commit**

```bash
git -C /home/razinka/osmose/osmose-master add pom.xml
git -C /home/razinka/osmose/osmose-master commit -m "build: run Java tests by default (invert skipTests)"
```

---

### Task 14: Drop `-DskipTests=true` from the Java workflow

**Files:**
- Modify: `osmose-master/.github/workflows/java-compile.yml:57-58`

- [ ] **Step 1: Edit the workflow step**

Replace:

```yaml
- name: Compile Java code
  run:  |
      mvn build-helper:remove-project-artifact
      mvn -B package -DskipTests=true
```

with:

```yaml
- name: Compile and test Java code
  run:  |
      mvn build-helper:remove-project-artifact
      mvn -B verify
```

(`verify` runs tests; `package -DskipTests=true` did not.)

- [ ] **Step 2: Commit**

```bash
git -C /home/razinka/osmose/osmose-master add .github/workflows/java-compile.yml
git -C /home/razinka/osmose/osmose-master commit -m "ci: run Java tests via mvn verify instead of skipping"
```

- [ ] **Step 3: Watch the next Actions run**

After pushing, open the Actions tab for `osmose-master`. If legacy tests fail, open a tracking issue listing each failure — do **not** revert the skip. Failing tests are signal.

---

### Task 15: Document the local Maven repository dependency chain

**Files:**
- Modify: `osmose-master/java/local/README.md` (expand from 30 lines to a complete "what's here, why, how to refresh")

- [ ] **Step 1: Inventory the folder**

```bash
find /home/razinka/osmose/osmose-master/java/local -type f -name "*.jar" -o -name "*.pom"
```

Record each artifact's `groupId / artifactId / version` (from folder path).

- [ ] **Step 2: Expand the README**

Append the inventory to the README under a new section:

```markdown
## Vendored artifacts (2026-04-17 audit)

| Artifact | GAV | Why vendored |
|---|---|---|
| (fill in from Step 1) | group:artifact:version | e.g. "not on Maven Central; patched build" |

## Refresh procedure

If Maven Central or UCAR publishes a newer version:

1. Download the new `.jar` + `.pom` into `java/local/<group>/<artifact>/<version>/`.
2. Update `<version>` in `osmose-master/pom.xml`.
3. Run `mvn clean verify` to confirm the build resolves.
4. Commit the new jars with a conventional `build:` message.
```

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/osmose/osmose-master add java/local/README.md
git -C /home/razinka/osmose/osmose-master commit -m "docs: inventory vendored Maven artifacts in java/local"
```

---

## Phase 4: Lint polish (optional but recommended)

### Task 16: Clean up `ruff check .` errors outside CI scope

**Files:**
- Modify: `mcp_servers/copernicus/server.py` (remove unused `json`, `tempfile`, unused `result`, unused `si`)
- Modify: `scripts/calibrate_baltic.py` (remove unused `shutil`, `sys`)

- [ ] **Step 1: Run ruff with `--fix`**

```bash
.venv/bin/ruff check . --fix
```

This auto-removes the 4 fixable F401 imports.

- [ ] **Step 2: Manually fix remaining F841**

For the two `Local variable ... assigned to but never used` errors (`result` at line 226, `si` at line 392 of `server.py`), either:
- Delete the assignment if the returned value is truly unused, OR
- Prefix with `_` (e.g. `_result = cm.subset(...)`) if the call has side effects and we want to document intent.

Pick per-site: `cm.subset(...)` has side effects (downloads data) — keep the call, rename to `_result`. `_get_var(ds, "si")` — if `si` is never referenced, delete it; if it was meant to be used (diagnostic), prefix `_`.

- [ ] **Step 3: Verify ruff is clean across the whole repo**

```bash
.venv/bin/ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest -q
```

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/copernicus/server.py scripts/calibrate_baltic.py
git commit -m "chore: clean up ruff F401/F841 in non-CI-scoped scripts"
```

---

## Post-plan validation

- [ ] **Step 1: Final green check**

```bash
.venv/bin/ruff check .                # All checks passed!
.venv/bin/python -m pyright 2>&1 | tail -1  # 0 errors
.venv/bin/python -m pytest -q           # Same pass count as baseline (plus new tests from Tasks 1, 2, 8, 11)
```

- [ ] **Step 2: Update MEMORY.md**

Append one line under "Current Status" noting that the 2026-04-17 deep-review fixes landed. Memory file: `/home/razinka/.claude/projects/-home-razinka-osmose/memory/MEMORY.md`.

- [ ] **Step 3: Create a PR**

If the user confirms, use `commit-commands:commit-push-pr` to open a PR titled "chore: 2026-04-17 deep-review fixes (credentials, types, Java tests)". Link `docs/review-findings.md` in the PR description.

---

## Self-review summary

**Spec coverage:**
- Copernicus creds: Tasks 1–3 ✓
- 22 Pyright errors: Tasks 4–11 cover all 8 clusters (22 errors) ✓
- pom.xml skipTests: Task 13 ✓
- workflow -DskipTests=true: Task 14 ✓
- Local Maven documentation: Task 15 ✓
- Bonus (not in review): ruff cleanup in scripts: Task 16 ✓

**Placeholder scan:** All tasks contain concrete commands, exact file paths, and (where implementation changes happen) full code blocks. No "TBD" / "implement later".

**Type consistency:** `_require_preflight` is introduced in Task 11 and not used in any other task — safe. `SchoolState.imax_trait` is declared in Task 4 and read in unchanged callers (which already use `state.imax_trait`). `load_timeseries` return widening in Task 8 is caller-audited in Step 5 of that task.
