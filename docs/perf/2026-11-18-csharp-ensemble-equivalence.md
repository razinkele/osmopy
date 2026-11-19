# C# engine: ensemble equivalence with the Python engine (task 1.10)

Date: 2026-11-18. Harness: `scripts/cross_engine_parity_440.py --engines python,python-control,csharp`.
Rule: `osmose/validation/engine_equivalence.py` (spec 10.2). Raw replicates of every arm are kept
in the run's JSON, so any rule below can be re-evaluated without rerunning.

## Design (spec 10.2, fixed before any C# data were looked at)

| | EEC full | Bay of Biscay |
|---|---|---|
| Years (spin-up dropped) | 10 (2) | 5 (1) |
| Replicates per arm | 32 | 32 |
| Arms | Python A (seeds 1000+), Python B (5000+), C# (9000+) | same |
| Metrics | biomass, abundance, yield, mean weight | same |
| Margins | 1.25x (0.097 log10); mean weight 1.05x (0.021) | same |
| Outputs | `output.minimal.enabled`, all arms | same |

The control is A against B; the treatment is A against C#. Both reruns reproduced the first runs
exactly (fixed seed sets on both engines).

## Results

| | Bay of Biscay | EEC |
|---|---|---|
| Equivalent cells, C# / control | 32 / 32 | 24 / 23 |
| The same, re-judged on 2026-11-19 (see the note below the table) | 26 / 26 of 26 counted | 24 / 23 of 54 counted |
| Largest \|d\|, C# / control | 0.011 / 0.007 | 0.314 / 0.382 |
| Verdict under the rule as specified | **PASS** | **FAIL** (criterion 1, 5 divergent cells) |

**Re-judged on 2026-11-19 under the amended criterion 5.** A cell that is one same number in
every replicate of all three arms is now listed and left out of the counts; before, only an
all-zero yield was. On Bay of Biscay that removes six cells (the abundance, biomass and mean
weight of Mackerel and Sole, which take one value on every seed of both engines), so 26 cells
are counted and all 26 are equivalent on both arms. On EEC nothing changes: its two vacuous
cells were zero yields already. Both verdicts and both frozen rungs are unchanged.

Pairwise, outside the rule (non-vacuous cells; d in log10):

| Pair | BoB max \|d\| | EEC max \|d\| | EEC median | EEC cells beyond margin |
|---|---|---|---|---|
| A vs C# | 0.011 | 0.314 | 0.019 | 11 / 54 |
| A vs B (Python vs Python) | 0.007 | 0.382 | 0.025 | 9 / 54 |
| **B vs C#** | 0.004 | **0.094** | 0.017 | **3 / 54** |

**On EEC the C# engine is closer to Python arm B than Python arm A is.** The five "divergent"
cells (horse mackerel abundance; sardine biomass and yield; dragonet and red mullet mean weight)
carry the same difference, same sign and size, in the Python-vs-Python control: the control and
the treatment share arm A, and arm A is atypical for those noisy species. B against C# has no
divergent cell.

## What the calibration showed about the rule

The rule's false-fail rate for **identical** engines, from 2,000 re-splits of the pooled Python
replicates, at each rung of the loosening ladder (spec: "loosen the constants until it does not
exceed 5%"):

| Rung (slack, ratio, variance) | BoB | EEC |
|---|---|---|
| 1, 1.5, 2 (the spec's defaults) | 0.716 | 0.987 |
| 2, 2.0, 4 | 0.262 | 0.804 |
| 5, 4.0, 8 | 0.054 | 0.786 |
| 6, 5.0, 10 (frozen) | **0.020** | **0.785** |

- On BoB the tunable constants reach 2%: criterion 4 (variance ratio) was the binding one.
- On EEC no rung gets below 78.5%, and at the loosest rung **every** remaining false fail is
  criterion 1. It has no constant: for a cell noisier than its margin, "divergent" reduces to
  "the 90% interval excludes zero", which identical engines do 10% of the time per cell by
  construction. Two Python arms on EEC produce on average 2.6 divergent cells (median 2, 95th
  percentile 8).

Also fixed on the way: a species identical across Python seeds read a C# spread of 4.7e-17 in
log10 (C# sums outputs in a seed-dependent store order), and the variance ratio 0 / 1e-34 failed
criterion 4. A standard error below 1e-12 in log10 is now treated as no spread (`SPREAD_FLOOR`).

## Decision (owner, 2026-11-18): criterion 1 uses Bonferroni

Criterion 1 as specified could not be used on EEC: it rejected identical engines 79% of the time.
Measured on the saved replicates (2,000 identical-engine re-splits):

| Criterion 1 | Identical-engine false-fail, EEC | C# on EEC | C# on BoB |
|---|---|---|---|
| As specified: no divergent cell | 78.7% | FAIL (5) | PASS |
| Count: at most the 95th percentile of identical engines (8 on EEC) | 2.8% | PASS (5) | PASS |
| **Bonferroni: a cell's interval at 1 - 0.10/m (m = counted cells)** | **3.2%** | **PASS (0)** | **PASS** |

The owner chose Bonferroni: it keeps the specified meaning ("no divergent cell"), needs no
threshold estimated from the data, and one strongly divergent species still fails the run.

## Final verdicts (the amended rule, re-judged on the saved replicates)

| | Frozen rung (slack, ratio, variance) | Identical-engine false-fail at that rung | Verdict |
|---|---|---|---|
| EEC full | 5, 4.0, 8 | 4.8% (criterion 1: 3.1%) | **PASS**, all four criteria |
| Bay of Biscay | 6, 5.0, 10 | 2.0% | **PASS**, all four criteria |

Re-judge without simulating: `--rejudge <saved json>` (the raw replicates are in it).

## Reproduce

```bash
PYTHONPATH=. python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config data/eec_full/eec_all-parameters.csv --n 32 --years 10 --spinup-years 2 \
  --persist-results equiv_eec.json
PYTHONPATH=. python scripts/cross_engine_parity_440.py --engines python,python-control,csharp \
  --config data/examples/osm_all-parameters.csv --n 32 --years 5 --spinup-years 1 \
  --persist-results equiv_bob.json
```

About 6 minutes each on a 14-core machine (the C# arm is 26-61 s of it).
