# EEC: lesserSpottedDogfish / mackerel Python-vs-Java divergence

Follow-up to `docs/java_cross_check_2026-09-27.md`, which flagged these two species as
the only residuals in the cross-engine check that looked like signal rather than a
measurement limit. This localises them and records what was ruled out.

**Headline: they are two different phenomena, not one, and the write-up that
prompted this investigation overstated their similarity.** That report said "Python
higher in all six" as if one mechanism drove both. `lesserSpottedDogfish` is a real
level difference; `mackerel` is a smaller shift sitting on top of a *much noisier
Java arm*.

## 1. What the difference actually is

12 reps per engine, EEC config, 10 years, 2-year spinup, varied seeds
(`scripts/cross_engine_parity_440.py:ensemble`). Geometric means of the post-spinup
biomass:

| species | Python | Java | ratio | distribution shape |
|---|---|---|---|---|
| **lesserSpottedDogfish** | 6,556 | 2,220 | **2.95×** | **nearly disjoint** — 11 of 12 Java reps fall below all but one Python rep |
| **mackerel** | 1.50e5 | 7.70e4 | **1.95×** | heavily overlapping; Java sd(log10) **0.42** vs Python **0.088** |
| cod (control) | 7.71e4 | 8.08e4 | 0.95× | fully overlapping, sd ≈ 0.03 in both |
| sardine | — | — | 0.65× | sd ≈ 0.34 / 0.38 — volatile in **both** engines |

Two things follow that the gated report did not say:

- **For mackerel the striking asymmetry is variance, not level.** Java's spread is
  ~5× wider in log terms. An earlier reading of this as "Java sometimes collapses
  mackerel" was **wrong** and is retracted: at n=12 both distributions are smooth and
  unimodal, and they overlap heavily. Java's arm is simply broad.
- **sardine's `eq=n` was a power problem**, as the cross-check suspected — it is
  intrinsically volatile in both engines (max/min 14–34× within a single engine).

### A trap for anyone re-running this

EEC does **not** set `simulation.rng.fixed`, so the Python engine is *not*
reproducible even at a fixed seed: two runs at seed 0 gave mackerel 2.87e5 and
1.90e5. A single rep is worthless here — one seed-0 rep reported mackerel at **17×**,
which is one draw from a wide distribution, not a finding.

## 2. Localised to predation mortality, concentrated in older fish

The Python/Java ratio **grows monotonically with age** for both problem species, while
the control is flat (biomass by age, final-5yr mean, single rep):

| age bin | dogfish py/jv | mackerel py/jv | cod py/jv |
|---|---|---|---|
| 0–2 | 1.8–3.1 | 14–22 | 2.5–12 |
| 4–6 | 3.9–4.2 | 65–257 | 0.5–1.9 |
| 9 | 849 | 10,840 | 0.88 |

Java's oldest occupied class is essentially empty for these two and not for cod.

Mortality by cause (final-5yr mean; Python's names mapped onto Java's) puts the
difference in `Mpred`, with fishing and adult/juvenile `Madd` matching almost exactly:

| | Python | Java |
|---|---|---|
| mackerel `Mpred` Adult | 0.205 | **1.305** |
| mackerel `Mpred` Juvenil | 3.55 | **6.32** |
| dogfish `Mpred` Juvenil | 1.085 | **1.479** |
| cod `Mpred` Juvenil (control) | 11.67 | 10.69 |
| mackerel `F` Adult | 0.142 | 0.142 |
| dogfish `Madd` Adult | 0.0870 | 0.0870 |

**These are SUMS of per-step rates over the saving interval, not cohort integrals**
(CLAUDE.md) — `exp(-rate)` is not survival. They are used here only to compare like
with like between engines, never converted to a survival fraction.

## 3. Cause or effect? (answered in §5 — effect)

**The `Mpred` gap above does not by itself establish causation.** Mortality rates are
per-capita: if Java holds ~3× fewer mackerel and
absolute predation is similar, the *rate* mechanically reads ~3× higher. So the gap
may be a **consequence** of the abundance difference rather than its cause.

Separating them needs **absolute** predation pressure (biomass eaten), not rates.
Both engines write `*_predatorPressure_Simu0.csv`; see §5.

## 4. Ruled out, with the evidence

Each of these was a plausible candidate and each is eliminated on measurement, not
on reading:

| Candidate | Why it is not the cause |
|---|---|
| Accessibility `-1 → 1.0` trap (CLAUDE.md) | Per-**role** audit (`prey_lookup` / `pred_lookup` separately, as that note prescribes): all 14 focal species have **both** a prey row and a predator column. Only the 10 resource groups lack predator columns, and resources never predate. |
| `fisheries.rate.byperiod` (java-only key) | `period.number = 1` and multiplier `1` for all 14 fisheries → ignoring it is a no-op on this config. |
| `predation.accessibility.stage.structure` (java-only) | Python hardcodes the threshold as age in years; the config declares `age`. Coincides here. **Latent risk** for any config declaring `size`. |
| `fisheries.movement.fishery.mapN` (java-only) | All 14 maps point to the same file; Python reads `map0` as a shared fishing map. |
| Predator/prey size-ratio handling | See below — a real asymmetry that nonetheless cancels. |
| A results-reader bug | One simulation read both ways (in-memory `from_outputs` vs written CSVs) agrees **exactly**, ratio 1.000 on every species and metric. |

### The size-ratio asymmetry that cancels

Worth recording because it looks alarming and is not:

- EEC authors `sizeratio.min > sizeratio.max` (dogfish `min=50, max=3`; mackerel
  `min=100, max=2.5`) — the convention being that these are predator:prey ratios, so
  the larger divisor gives the *smaller* prey.
- **The engines swap on opposite conditions.** Java swaps when `max > min`
  (`PredationMortality.java:128`); Python swaps when `min > max`
  (`osmose/engine/config.py:684`). On EEC, Java does **not** swap and Python **does**,
  so the same two numbers land in oppositely-named slots.
- They still agree. Java takes prey in `[predLen/min, predLen/max]` =
  `[predLen/50, predLen/3]`. Python accepts `r_min ≤ predLen/preyLen < r_max` which,
  with its swapped `(3, 50)`, is `(predLen/50, predLen/3]` — the same window bar
  boundary inclusivity.

Do not "fix" either swap in isolation: they are load-bearing in opposite directions,
and aligning one without the other would break the agreement that currently holds.

## 5. RESOLVED: the `Mpred` gap is an effect, not the cause

Done by aggregating Java's size-staged `predatorPressure` to species level (summing
its 3 predator stages and 3 prey stages) and comparing **absolute biomass eaten**.
Both engines divide the accumulation by `output.recordfrequency.ndt` (= 24 here) —
Java at `PredatorPressureOutput.java:167`, Python at `osmose/engine/output.py:666` —
so both report mean tons eaten **per simulation step** and the comparison is
apples-to-apples.

The result **reverses** what the rates implied:

| prey | Python eats | Java eats | jv/py |
|---|---|---|---|
| **mackerel** | 9,441 t | 312 t | **0.033** |
| lesserSpottedDogfish | 48.1 t | 17.5 t | 0.36 |
| cod | 65.2 t | 25.5 t | 0.39 |
| the other 11 focal prey | — | — | 0.28 – 0.65 |

**Java removes ~30× LESS mackerel biomass in absolute terms while showing a 6.4×
HIGHER per-capita `Mpred` rate.** That is exactly the per-capita artifact §3 warned
about: Java holds far fewer mackerel, so a much smaller absolute removal still yields
a much higher rate. The rate gap is therefore an **effect** of the abundance
difference and carries no causal information. Any future reading of these mortality
series must not treat a rate difference as evidence of predation intensity.

### The systemic finding this exposes

The interesting number is not mackerel's outlier but the **near-uniformity of the
rest**: Java removes 0.28–0.65× of Python's absolute predation for 13 of 14 species.
**Python's food web moves roughly 3× more biomass through predation than Java's,
across essentially the whole community.**

That is self-consistent with the biomass difference rather than contradicting it: if
Python's fish ingest ~3× more, they grow faster and support more biomass, and the
species most dependent on that throughput diverge most. It reframes the original
question — the thing to explain is not "why are two species high" but "why is
Python's whole predation throughput ~3× Java's", with dogfish and mackerel simply the
most sensitive indicators.

A near-uniform multiplicative factor across unrelated species is the signature of a
normalisation difference, not of ecology. The specific suspect is the per-step
ingestion cap: Python computes
`max_eatable = biomass * ingestion_rate / (n_dt_per_year * n_subdt)`
(`osmose/engine/processes/mortality.py:1141`, and the per-cell oracle at `:470`),
and EEC sets `mortality.subdt = 10`
with `ndtperyear = 24`. **Whether Java's cap carries the same two divisors has NOT
been verified** — that is the next step, and until it is checked this is a suspect,
not a diagnosis.

Note the factor is ~3, not 10 or 24, so a whole missing divisor is *not* the shape of
it; do not go looking for one.

## 6. Two defects found on the way

Both are engine-comparability defects, independent of the divergence itself.

### 6.1 The Python engine ignores `output.file.prefix`

`write_outputs(..., prefix: str = "osm")` (`osmose/engine/output.py:27`) is a
hardcoded default and `PythonEngine.run` never passes the config's value. On EEC,
Java honoured `output.file.prefix = eec` and wrote `eec_*`; Python wrote `osm_*`.

Anyone reading Python outputs with the prefix the config declares gets **nothing**,
silently — which is exactly what happened here, and cost a full diagnostic cycle. The
parity harness never hit it because it uses `run_in_memory` and never touches disk.
`output.file.prefix` is in the java-only set, so ignoring it is *declared*; writing a
different prefix instead of honouring the declared one is the defect.

### 6.2 The two engines' diet matrices are structurally incomparable

`output.diet.stage.structure` and its 14 `output.diet.stage.threshold.spN` keys are
java-only. Consequently:

- **Java** writes a size-staged prey×predator matrix — 520×45, labels like
  `mackerel in [30.000000, inf[`.
- **Python** writes a flat 10×338 matrix with `<predator>_<prey>` columns and no
  staging.

`OsmoseResults.diet_matrix()` returns each engine's own shape without normalising, so
a naive column-name comparison silently reads Java as **all zeros**. That happened
during this investigation and produced the false reading "no Java predator eats
mackerel"; it was caught only because Java's `Mpred` is plainly non-zero. Any
cross-engine diet comparison must aggregate Java's stages first.

## 7. Reproducing

```
PYTHONPATH=. .venv/bin/python scripts/cross_engine_parity_440.py \
  --engines python,4.4.1 --n 12 --years 10 --spinup-years 2
```

Needs a 4.4.1 jar in `osmose-java/`; build instructions and provenance are in
`docs/java_cross_check_2026-09-27.md` §Reproducing. Read Python's outputs with
prefix `osm`, Java's with the config's `eec` (§6.1).
