# Percid Trophic Refactor — Implementation Plan

**Design:** `docs/superpowers/specs/2026-08-02-percid-trophic-refactor-design.md`
**Scope:** Tier 0 (Tasks 1–6). Tier 1 (Tasks 7–10) is **conditional on Task 6's outcome** and must not
be started before it. Tier 2 is out of scope.

## Conventions used in this plan

* Run everything with `.venv/bin/python`, never bare `python`.
* `data/baltic/predation-accessibility.csv` is `;`-separated — rows are **prey**, columns are
  **predators**. Never write it comma-separated; the reader auto-detects the separator per line and a
  comma-written file is read as one column and fails obscurely.
* Never edit `data/baltic/` in a test. Copy the tree to a workdir first (see
  `tests/_tutorial_config.py::build_baltic_workdir`).
* Each task states its own done-condition. Do not proceed on a failed task — escalate per the task's
  escalation clause.
* A 50-yr Baltic run is ~4 min; budget accordingly and run multi-seed work in the background.

---

## Task 1: Pin the current coefficients with a characterisation test

**Why first:** the edit in Task 3 must be provably the edit intended, and #146 showed this repo ships
config that silently means something other than it reads.

Create `tests/test_percid_accessibility.py`:

```python
"""Percid trophic links: no offshore pelagics, smelt time-averaged over its availability window.

Design: docs/superpowers/specs/2026-08-02-percid-trophic-refactor-design.md
"""

from pathlib import Path

import pandas as pd
import pytest

ACC = Path(__file__).resolve().parents[1] / "data" / "baltic" / "predation-accessibility.csv"
OFFSHORE_PELAGICS = ("herring", "sprat")
PERCIDS = ("perch", "pikeperch")


def _matrix():
    return pd.read_csv(ACC, sep=";", index_col=0)


def test_percids_do_not_forage_on_offshore_pelagics():
    """Baltic percids are bay-resident with sub-bay feeding ranges (spec 1.3)."""
    df = _matrix()
    for prey in OFFSHORE_PELAGICS:
        for pred in PERCIDS:
            assert float(df.loc[prey, pred]) == 0.0, (
                f"{pred} still has accessibility {df.loc[prey, pred]} to {prey}. "
                f"Percids do not forage on the offshore pelagic stocks."
            )
```

Add `test_smelt_link_is_time_averaged` asserting the Task 2 values once `W` is fixed.

**Done when:** the test **fails** on current master with the real values (perch 0.2, pikeperch 0.3).
Watch it fail — a characterisation test that passes immediately is testing nothing.

---

## Task 2: Fix `W`, the smelt availability window

`W` is the **single free parameter** in Tier 0 and the smelt coefficients scale linearly in it.

1. Determine the smelt spawning-migration window for the modelled area from local phenology.
2. Record the source and the value in the spec's §4.1 as an **assumption**, not a finding.
3. Compute `perch→smelt = 0.5 × W/12`, `pikeperch→smelt = 0.6 × W/12`.

**`W = 3` (→ 0.125 / 0.15) is a working default only.** Do not ship it as if measured.

**Escalation:** if `W` cannot be established, proceed with `W = 3`, mark it clearly, and note that
every downstream number inherits that assumption.

---

## Task 3: Apply the accessibility edit

Edit `data/baltic/predation-accessibility.csv` **by column name**, never by string offset — the
positional-replace bug in #129 silently missed a column when one was inserted.

```python
import pandas as pd

path = "data/baltic/predation-accessibility.csv"
df = pd.read_csv(path, sep=";", index_col=0)
for prey in ("herring", "sprat"):
    for pred in ("perch", "pikeperch"):
        df.loc[prey, pred] = 0.0
df.loc["smelt", "perch"] = 0.5 * W / 12
df.loc["smelt", "pikeperch"] = 0.6 * W / 12
df.to_csv(path, sep=";")
```

**Leave alone:** `pikeperch→cod_west/cod_east` (0.05), cannibalism (0.05), stickleback (0.3), and all
resource columns.

**Done when:** Task 1's tests pass, and `git diff` touches exactly six cells.

---

## Task 4: Verify the change reaches realised diet

**This is acceptance criterion 2 and the task most likely to be skipped.** A config edit that does not
change realised diet has not been demonstrated to work — it has been demonstrated to be inert.

Run 50 yr, seed 42, and read the corrected `dietMatrix` (#146; note the prey axis is schools-then-
resources, and columns are wide `predator_prey`):

```python
d = res.diet_matrix()
late = d[d["Time"] >= 40]
for pred in ("perch", "pikeperch"):
    pre = f"{pred}_"
    shares = {c[len(pre):]: float(late[c].mean()) for c in late.columns if c.startswith(pre)}
    total = sum(shares.values())
    print(pred, {k: round(100 * v / total, 1) for k, v in shares.items() if v > 0})
```

**Done when:** `herring` and `sprat` are **absent** from both percid diets, and the diets are dominated
by benthos, zooplankton and stickleback.

**Escalation:** if either still appears, stop. The edit is not reaching the kernel — do not proceed to
Task 5, and do not "fix" it by editing further coefficients.

---

## Task 5: Measure, 3 seeds

50 yr, seeds `(42, 123, 7)`, final-decade means. Report **per-seed values, not just the mean** — the
2-seed A/B earlier in this work hid its own spread, and the 3-year diet run pointed the opposite way
from equilibrium.

Record for all 9 focal species: final-decade mean, envelope verdict, and the seed spread.

**Done when:** the table exists with per-seed values retained.

---

## Task 6: Judge against the acceptance criteria — and do not soften them

Apply spec §5 exactly:

1. **≥ 7/9 in envelope** across seeds. Below 7/9 is a **fail**, including a herring breach.
2. Herring/sprat absent from percid diets (Task 4).
3. Pikeperch residual factor **stated honestly** — "10× over instead of 56× over" is a result to
   report, not a success to claim.
4. Smelt reaching envelope is **not** a criterion and must not be claimed as one.

**If herring breaches** (the expected risk — it sits at 2.60 Mt against a 3.00 Mt ceiling and the 0.03
test already pushed it to 3.05 Mt): **do not revert Task 3.** Record the finding — herring's mortality
budget was implicitly leaning on percid predation that should not exist — and open an issue for
herring's mortality budget. That is the next piece of work, not a reason to undo this one.

**Do not loosen any threshold to make this pass.** If the criteria are not met, that is the result.

---

## Task 7 (conditional): decide whether Tier 1 is needed

**Gate:** only proceed if Task 6 shows the time-averaged surrogate is insufficient — i.e. the annual
mean is right but the *timing* of the percid–smelt interaction demonstrably matters (e.g. a spurious
year-round smelt suppression, or percid recruitment mistimed against the smelt pulse).

If Tier 0 met its criteria, **stop here** and schedule Tier 1 separately on phenological-realism
grounds. Record the decision either way.

---

## Task 8 (Tier 1): config schema + loader

Add to `osmose/engine/config.py`, following the `fisheries.seasonality.fshN` 24-value vector idiom:

```
predation.accessibility.seasonality.enabled;true
predation.accessibility.seasonality.pair0;smelt,pikeperch
predation.accessibility.seasonality.values.pair0;<n_dt values>
```

* **Sparse:** only declared pairs deviate; all others are a constant 1.0. Existing configs unaffected.
* Resolve to a dense `(n_dt, n_prey, n_pred)` float array **built once at config time** — the
  predation kernel is Numba-compiled, so no dict lookup in the hot loop.
* Fail fast on: unknown species name, wrong value count (must equal `n_dt_per_year`), values outside
  `[0, 1]`.
* Add the new keys to `osmose/engine/config_validation.py`'s allowlist if the AST walker does not pick
  them up. `tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs`
  must stay warning-free.

**Done when:** loader unit tests pass, including each fail-fast case.

---

## Task 9 (Tier 1): kernel wiring

Index the multiplier by `step % n_dt_per_year` at the accessibility lookup in
`osmose/engine/processes/mortality.py`.

**Done when:** with all-ones seasonality, output is **bit-identical** to the pre-change engine on the
Baltic config. Assert this — it is the only way to prove the feature is inert when unused.

---

## Task 10 (Tier 1): re-run Tasks 4–6 with the real seasonal vector

Replace the time-averaged smelt coefficients with the phenological vector and re-apply the full
acceptance criteria. Report the difference from Tier 0 explicitly: if the annual answer is unchanged,
say so — that is the useful finding, not a disappointment.

---

## Out of scope

**Tier 2 — percid stocks as separate coastal units.** Also fixes the target itself: the ICES envelope
is a **per-stock** figure while the model carries one aggregated basin-wide pikeperch, so the
comparison is not like-for-like even after Tiers 0–1. Separate, explicitly scoped work; the cod E/W
disaggregation (could not be fitted, remains a flagged experiment) is the cautionary precedent.
