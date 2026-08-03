# Percid Trophic Refactor — Implementation Plan

**Design:** `docs/superpowers/specs/2026-08-02-percid-trophic-refactor-design.md`
**Revised:** 2026-08-03 after adversarial review — tier order changed, Task 3/5/9 defects fixed.
**Scope:** Tier A (Tasks 1–5) then Tier B (Tasks 6–9). Tier C is gated on Task 9. Separate coastal
stocks are out of scope.

> **Tier A runs before Tier B.** The predation-release hypothesis (spec §1.5) is the primary lever; if
> the trophic edits go first, they get credited with an effect predation would have produced.

## Conventions

* `.venv/bin/python`, never bare `python`.
* `data/baltic/predation-accessibility.csv` is `;`-separated, rows = **prey**, cols = **predators**.
  Never write it comma-separated — the reader auto-detects per line and a comma-written file is read as
  one column and fails obscurely.
* Never edit `data/baltic/` from a test; copy the tree (`tests/_tutorial_config.py::build_baltic_workdir`).
* **Measure with `scripts/baltic_stability_certify.py`, 5 seeds.** The 7/9 baseline came from that
  script at 5 seeds; an ad-hoc 3-seed run is not comparable to it and must not be used to judge
  criterion 1.
* A 50-yr run is ~4 min; a 5-seed certification is ~20 min. Run in background.
* Each task states its done-condition. Do not proceed past a failure — escalate as written.

---

## Task 1: Record the baseline with the certifier

Run `scripts/baltic_stability_certify.py --params current` (50 yr, 5 seeds) on unmodified master.
Record per-species final-decade mean, **per-seed spread**, and envelope verdict.

**Why:** criterion 2 is expressed in units of the baseline spread, so it cannot be evaluated without
it. The 2-seed A/B earlier in this work hid its own spread and the 3-year diet run pointed opposite to
equilibrium.

**Done when:** the table exists, including the spread column, and reproduces 7/9.

---

## Task 2: Characterisation test for the accessibility edits

Create `tests/test_percid_accessibility.py`. Assert the **current** values so the file is a live record
of what the edits change:

```python
from pathlib import Path
import pandas as pd

ACC = Path(__file__).resolve().parents[1] / "data" / "baltic" / "predation-accessibility.csv"


def _m():
    return pd.read_csv(ACC, sep=";", index_col=0)


def test_pikeperch_predation_pressure_matches_design():
    """Tier A raises predation ON pikeperch toward perch's level (spec 4 Tier A)."""
    df = _m()
    got = {p: float(df.loc["pikeperch", p]) for p in ("cod_west", "cod_east", "Cormorant")}
    assert got == {"cod_west": 0.15, "cod_east": 0.10, "Cormorant": 0.60}, got
```

Add `test_percid_forage_links_match_design` asserting the Tier B values from Task 6.

**Do not assert `== 0.0` for herring.** Spec §2 documents pikeperch predation on YOY herring inside
brackish bays (Jensen et al., 2011) at sizes this config models; freezing a zero into CI would make
that deletion a regression-tested invariant.

**Done when:** each test **fails on current master** for the stated reason. Watch it fail — a
characterisation test that passes immediately tests nothing.

---

## Task 3: Apply the Tier A edit

Edit by column **name**. Set in `data/baltic/predation-accessibility.csv`:
`pikeperch` row → `cod_west 0.1→0.15`, `cod_east 0.05→0.10`, `Cormorant 0.4→0.60`.

```python
import pandas as pd
path = "data/baltic/predation-accessibility.csv"
df = pd.read_csv(path, sep=";", index_col=0)
df.loc["pikeperch", ["cod_west", "cod_east", "Cormorant"]] = [0.15, 0.10, 0.60]
df.to_csv(path, sep=";")
```

**Done when:** Task 2's first test passes, and **`git diff` shows only the intended value changes**.

`df.to_csv` rewrites the whole file, so the diff will legitimately include reformatting — trailing
`.0`, float precision, quoting. **Verify semantically, not by counting changed cells:**

```python
a = pd.read_csv("<HEAD version>", sep=";", index_col=0)
b = pd.read_csv(path, sep=";", index_col=0)
diff = (a != b) & ~(a.isna() & b.isna())
print(diff.stack()[lambda s: s].index.tolist())   # must be exactly the 3 intended cells
```

If reformatting noise is unacceptable, edit the three fields in place with a line-oriented rewrite
instead — but keep the name-based lookup to locate the column index.

---

## Task 4: Measure Tier A (5 seeds, certifier)

Re-run Task 1's command. Record the same table.

**Done when:** the table exists with per-seed spread.

---

## Task 5: Judge Tier A

Apply spec §5 exactly:

1. **≥ 7/9** — a fail below, no exceptions. Watch **cod_east** first: it has **2.3% headroom** to its
   upper bound and this tier feeds cod. It is the tightest species in the system, not herring.
2. Pikeperch falls by **>2× the baseline 5-seed spread**.
3. **Mechanism:** pikeperch's realised **predation mortality share of total Z** must rise — read from
   the `mortalityRate` output, not inferred from the coefficient. A coefficient that does not move
   realised mortality is inert.
4. **Collapse guard:** pikeperch final-decade *minimum* stays above 4,000 t.

**If cod_east breaches**, Tier A fails. Record it and consider whether a cormorant-only variant (leaving
cod untouched) preserves the effect — do not proceed to Tier B on a failed Tier A without saying so.

---

## Task 6: Apply the Tier B edits

Only after Task 5 is judged and recorded.

```python
df.loc["sprat", ["perch", "pikeperch"]] = [0.0, 0.0]        # sprat is genuinely offshore
df.loc["herring", ["perch", "pikeperch"]] = [0.06, 0.05]    # scaled, NOT zeroed — spec 4 Tier B.2
df.loc["smelt", ["perch", "pikeperch"]] = [0.04, 0.05]      # W ~ 1 month, spring run
```

Herring values derive from the time-weighted spatial overlap (perch 31.6%, pikeperch 16.2%; spec §1.3).
Smelt values assume a ~1-month spring run (Sendek & Bogdanov, 2019) — **record W as an assumption, and
note it is a lower bound** because smelt is present year-round in some basins.

**Done when:** Task 2's second test passes and the semantic diff shows exactly the 6 intended cells.

---

## Task 7: Verify Tier B reaches realised diet

50 yr, seed 42, corrected `dietMatrix` (#146 — wide `predator_prey` columns, prey axis is
schools-then-resources):

```python
late = res.diet_matrix().query("Time >= 40")
for pred in ("perch", "pikeperch"):
    pre = f"{pred}_"
    sh = {c[len(pre):]: float(late[c].mean()) for c in late.columns if c.startswith(pre)}
    tot = sum(sh.values())
    print(pred, {k: round(100 * v / tot, 1) for k, v in sh.items() if v > 0})
```

**Done when:** `sprat` is absent; **`herring` is present but reduced**.

**Escalation:** if `herring` disappears entirely, stop. A scaled coefficient behaving as a hard gate is
an engine defect, not a pass — investigate before measuring anything.

---

## Task 8: Measure Tier B (5 seeds, certifier)

As Task 4.

---

## Task 9: Judge Tier B, and decide on Tier C

Spec §5 again, with the same collapse guard and the same no-escape-clause rule: **any species leaving
envelope is a fail**, whatever the explanation. A breach may be informative — it would suggest that
species' mortality budget was leaning on a link that should not exist — but it is a failed tier, not a
passed one with a footnote.

**Tier C gate:** proceed only if the annual-mean smelt surrogate is demonstrably insufficient — i.e.
timing, not level, is what is wrong (e.g. percid recruitment mistimed against the smelt pulse). If
Tier B met its criteria, stop and schedule Tier C separately on phenological-realism grounds.

---

## Task 10 (Tier C): seasonal accessibility — config + loader

```
predation.accessibility.seasonality.enabled;true
predation.accessibility.seasonality.pair0;smelt,pikeperch
predation.accessibility.seasonality.values.pair0;<n_dt values>
```

Sparse: undeclared pairs stay at a constant 1.0, so existing configs are untouched. Resolve to a dense
`(n_dt, n_prey, n_pred)` array **built once at config time**. Fail fast on unknown species, wrong value
count (must equal `n_dt_per_year`), or values outside `[0, 1]`. Add keys to
`osmose/engine/config_validation.py` if the AST walker misses them;
`tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs` must stay
warning-free.

---

## Task 11 (Tier C): kernel wiring

**Do not put the lookup where the first draft said.** The accessibility read inside the compiled
predation kernel does not have `step` in scope. Use the **existing per-step rescale seam** in
`osmose/engine/processes/mortality.py` — the point where per-step accessibility is already prepared for
the kernel, upstream of the compiled inner loop. Locate it by tracing where the accessibility array is
passed into `_apply_predation_numba`, and apply the `step % n_dt_per_year` multiplier there.

**Done when:** with an all-ones seasonality vector, output is **bit-identical** to the pre-change
engine on the Baltic config. Multiplying float64 by exactly 1.0 is exact, so bit-identity is the right
bar and proves the feature is inert when unused.

---

## Task 12 (Tier C): re-run Tasks 7–9 with the phenological vector

Set the Tier B smelt coefficients back to their **unscaled** values (perch 0.5, pikeperch 0.6) and let
the seasonal vector carry the timing — otherwise the window is applied twice.

Report the difference from Tier B explicitly. If the annual answer is unchanged, say so: that is the
useful finding, not a disappointment.

---

## Out of scope

**Percid stocks as separate coastal units.** Also fixes the target itself — the ICES envelope is a
**per-stock** figure while the model carries one aggregated basin-wide pikeperch, so the comparison is
not like-for-like even after Tiers A–C. Separate, explicitly scoped work; cod E/W is the cautionary
precedent.
