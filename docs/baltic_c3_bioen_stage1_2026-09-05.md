# Baltic C3 — bioenergetics activation, Stage 1 results (2026-09-05)

**Verdict: CLOSE BY CHARACTERIZATION.** Four of the five assessed stocks — cod_west, cod_east,
herring, flounder — fail all three pre-registered criteria at the certifying 50-yr, 5-seed scale:
final-decade mean is **exactly 0.0 t** in the `bioen` arm, every seed (bioen/baseline = 0.000,
ē/ĝ = 0.000, bioen/certified = 0.000). Sprat is the sole survivor of the five, passing all three
criteria. This is a clean pre-registered negative — nobody chose this threshold after seeing these
numbers — arrived at with Gate A reading bit-identical to the committed master fixture throughout
and Gate B (Task 9) PASS against Java 4.3.3. **Bounded claim on what those two gates cover**
(restated per an independent review, task-13-14-review.md F4): Gate A certifies only that the
bioen path is inert when *switched off* — it says nothing about the bioen-on path itself. Gate B's
PASS is measured on `data/examples_bioen`, a parity vehicle, not on this Baltic overlay — and this
Baltic overlay is **not itself Java-loadable as authored**
(`data/baltic/scenarios/c3_bioen/README.md:31`: the port's bioen parameter arrays are
focal-species-length, so the background-predator keys Java's own bioen block would need cannot be
authored for GreySeal/Cormorant). So: no bioen-path defect was found by any gate that ran; Gate B's
evidence is transferred from a different config, and this overlay has not itself been cross-engine
verified. §8 discusses what the finding *is* about, given that bound. Full numbers in §4; mechanism
and independent corroboration (an isolated 8-yr stress test with a bioen-off control) in the box
below and in §0/§1. Every number in §4–§7's tables is pulled directly from
`docs/diagnostics/baltic_c3_bioen_report.json` — none typed by hand; §8's growth-deficit figures
are read from a separate, explicitly-labelled in-engine measurement (not in that JSON — see §8).

Spec: `docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md` (binding — decisions,
§0 table, §4 gates and decision rule).
Plan: `docs/superpowers/plans/2026-08-30-baltic-c3-bioen-stage1.md`.
Harness: `scripts/baltic_c3_bioen_ab.py` (Task 12). Branch: `c3-bioen-stage1`.

**A note on the decision rule:** Task 12's review found the harness's pre-registered decision rule
originally failed **open** on `NaN` for criteria (i) and (ii) — `nan < x` is `False` in Python, so
a species with an undefined `e_over_g` (criterion ii, when the fitted `g_hat` is exactly 0) would
silently read as *passing* rather than *undetermined*; criterion (iii) failed closed correctly.
**Fixed on this branch before this run was launched** (`b14c7eb`,
`evaluate_decision_rule` in `scripts/baltic_c3_bioen_ab.py`): every criterion is now three-way
(`pass`/`fail`/`undetermined`) per species, with `undetermined` reported separately from `failed`
rather than folded into either a pass or a failure — no threshold or algebra changed. The rule
independently re-derives to `f = m + (1-m)·ē/ĝ`, giving `f ≥ 0.72` at the spec's `ē/ĝ ≥ 0.6,
m = 0.3`, matching spec §4's thresholds literally, with no CLI knobs to move them after the fact.
This run was launched against the fixed harness, so the verdict below is final, not provisional.

**No criterion came back `undetermined` on this run** (`decision_rule_undetermined: []` in the
JSON) — every one of the four failing species failed on a hard zero (`ē = 0.0`, not `NaN`), so
the `b14c7eb` NaN-guard did not need to fire here. It still mattered: had the pre-fix rule been
used and any species landed on the `g_hat == 0` path instead, it would have printed `STAGE 2:
WARRANTED` over an unevaluated criterion rather than the negative below. The verdict on this run
was produced by a rule that cannot fail open, not one that happened not to.

## Run provenance

- 3 arms — `baseline` (production config, unmodified), `bioen` (production + the flat overlay
  `data/baltic/scenarios/c3_bioen/c3_bioen_arm.json`), `bioen_plus2C` (`bioen` +
  `temperature.offset=2.0`) — × 5 house seeds `[42, 123, 7, 999, 2024]` × 50 yr. Fixed by Ruling
  R7 — not reduced (seeds, horizon, or arms) for this run.
- **Runtime: ~26 minutes wall clock** (11:15:51 launch, 11:42:32 completion, 5 seeds × 50 yr × 3
  arms, Gate C recompute included). Quote this as a **margin, not a point estimate**: it sits
  inside the 27–57 min point estimate and far inside the "under three hours even at 3×" load-bearing
  claim, but it is a **lower bound in the same sense Task 13 flagged** — per-step cost tracks
  school count, four of nine species collapse to zero well before year 50 in the `bioen` arm, and
  a run whose stocks did not collapse would cost more than this one did. Before this run, the only
  *measured* 50-yr number on this branch was the bioen-**OFF** baseline (Task 3's Gate A, ~9–19 min
  for 5 seeds × 50 yr through the same Numba kernel); no 50-year bioen-**ON** measurement existed
  anywhere before this run. This measured number now supersedes the point estimate for the arms
  run here.
- Gate C's builder-recomputation leg was run (not skipped with `--no-recompute`) so the on-disk
  temperature forcing file is independently re-verified against the CMEMS cache for this record
  run — Task 12 measured that leg at 35 s, not the "minutes" the module docstring assumes.
- Raw report: `docs/diagnostics/baltic_c3_bioen_report.json` (committed verbatim from the run).
- All 6 gates PASS: `gate_a` (baseline arm bit-identical to the committed master fixture, all 5
  seeds), `gate_c`, `gate_d_frames_layers`, `gate_d_structure`, `gate_e` (all 9 species checked,
  `{"cell": [25, 21], "step": 12}`), `gate_f` (all 9 species, `phi_t(T_p)==1.0` exact, argmax
  exact to the grid resolution — see §3). The collapse below is not a wiring failure; every
  structural and parity gate that could catch one passed.

**Context from Task 13, read before the numbers in §4:** Task 13 (this plan, prior task) forced
every species out of its seeding bootstrap after year 1
(`population.seeding.year.max=1`, an artificial stress condition, not this run's setting) and
found production Baltic + this same bioen overlay collapses cod_west and cod_east to total
extinction by year 4, flounder by year 7, and crashes herring >99.99% by year 8 — while an
identical, same-seed control with the overlay removed (classic growth) sustains and grows all
five species under the identical cutoff. Task 13 also measured the mechanism directly: per-cause
mortality rates show predation, not starvation, driving the collapse (predation climbs to a
literal complete-cohort wipeout — `inf` in the rate output — while starvation stays small and
*declines* over the same span), consistent with a hypothesis (not proven, stated as such) that
bioen's slower growth keeps juveniles in a predation-vulnerable size window longer.

**This run does not repeat Task 13's stress condition — it uses the unmodified production
seeding regime — and that turns out to matter for how to read the final decade.** Production
Baltic sets no `population.seeding.year.max`, so the engine default applies
(`osmose/engine/config.py:538-544`): the seeding window closes at each species' own
`species.lifespan.sp{i}` in years (`data/baltic/baltic_param-species.csv`) — cod_west 20 (the
longest), cod_east 15, herring 12, sprat 8, flounder 15 — not indefinitely, and not at year 1.
Seeding is gated off outside that window unconditionally, regardless of whether a species has
ever achieved positive SSB. This run's final-decade metric (years 41–50) is therefore **21 to 46
years past every assessed stock's own seeding-window closure** — nothing in it can be a live
seeding-injection artifact. A species reading 0.0 there is not "masked" by an ongoing bootstrap;
it is a population that, having lost access to seeding at its own window's close, has had two
decades or more with no possible rescue. Task 13's 8-year stress test is corroborating evidence
for the mechanism and the identity of the survivor (sprat in both), not the reason this run's
numbers are interpretable — they are interpretable on their own terms, at the full pre-registered
50-yr horizon, under the same seeding regime every certified Baltic result in this repo already
uses.

## §0/§1 — The parity finding, and what this A/B does and does not measure

### The spec's §0 finding (why this stage exists)

The C3 spec's own review (`docs/superpowers/specs/2026-08-30-baltic-c3-bioen-stage1-design.md`
§0) found the bioen path as it existed before this branch was **not** Java-parity — verified
against the Java 4.3.3 sources at
`/home/razinka/osmose-reference/osmose-master/java/src/main/java/fr/ird/osmose/`.
Java runs the whole energy budget in **tonnes per school**, converting to per-fish grams only at
the growth increment; the pre-branch port mixed per-school tonnes with per-fish grams throughout,
and every process downstream of the budget (starvation ordering, reproduction, the ingestion cap)
carried its own independent defect. Restated from the spec table:

| Quantity | Java | Pre-branch Python | Verdict |
|---|---|---|---|
| E_gross | survivor-rescaled ingestion, every death (`School.java:372-402`) | raw `preyed_biomass`, never rescaled | ✗ survivor scaling |
| E_maint | `c_m·(w·1e6)^β·Arr(T)/ndt·N·1e-6` (t/school) | `c_m·w_g^β·Arr(T)/ndt` (g/fish) | ✗ missing `·N·1e-6` |
| dw, dg | `E_net·(1−ρ)/N`, `ρ·E_net/N` (t/fish) | `E_net·(1−ρ)·1e-6` | ✗ = Java × N/1e6 |
| `enet_faced` | per-fish, annualized, larval-divided, updated *before* ρ | cumulative mean of raw E_net, read *before* update | ✗ normalization/order |
| ρ | `r/(η·enet_faced)·w_g^(1−β)`, unguarded then clamped | non-positive `enet_faced` → 1.0 first | ✗ guard semantics |
| Max ingestion | replaces predation; instantaneous abundance; bkg included | standard rate loop, post-hoc cap | ✗ form/units/cap site |
| Starvation | interleaved loop, previous step's E_net, gonad repay | once, post-budget, current E_net; cause removed | ✗ timing/order/eligible |
| Reproduction | egg wt × N × sexRatio × season; unlocated schools | no ×N/sexRatio/season; whole gonad flushed | ✗ every term |
| Egg length | computed at creation, preyed at that length | recomputed from weight after first bioen step | ✗ (Baltic-relevant) |
| Numba dispatch | n/a | batched kernels bypass the bioen check | ✗ "already bypassed" false |
| `mobilized.Tp`/`.e.D` | read case-insensitively | case-sensitive vs lowercased file → default | ✗ silent |

Full column text (Java line refs, exact Python module paths) is in the spec's own §0 table;
this is a restatement, not a replacement.

Tasks 0–11 of this plan (commits through `8a574ab`) fixed the budget units and ordering, the
survivor-scaled ingestion, the starvation timing/repayment, the reproduction term set, the egg
length, the Numba dispatch gate, and the key-case defect; wired a real two-layer CMEMS temperature
forcing; and fit a 9-species Baltic bioen parameter set offline against each species' own
literature growth optimum. **Gate B result and the control** (Task 9,
`docs/diagnostics/c3_gate_b_cross_engine.md`): on `data/examples_bioen` against Java 4.3.3 —
**bioen arm: `GATE (absolute Python<->4.3.3 equivalence + within 1 OoM): PASS`**; **control
(bioen off, `data/examples`): `REVIEW: biomass:Hake, yield:Hake, mean_weight:Hake`**, traced to a
pre-existing, deliberately-uncorrected data defect in `data/examples` unrelated to this branch's
changes (not a port defect). The control matters here specifically as **the tripwire's proof that
it fires on bioen and not on noise**: the same harness, the same species set, only Hake — and only
Hake — shows a pre-existing issue with bioen off, while every other species (bioen on and off)
passes. `mean_size` was also checked against 4.4.1 as a reported (non-gating) comparison: the port
agrees with 4.3.3 and disagrees with 4.4.1 in the same direction for all 8 species (`eq=16/16`,
CI90 ±0.00 — a tight, deterministic offset, not a parity failure against the pinned reference).

**Two framing obligations, stated plainly:**

- `data/examples_bioen` (Gate B's config) is a **parity vehicle, not a calibrated ecosystem** — no
  species reaches maturity there under *classic* growth either. Gate B certifies cross-engine
  agreement only; no ecological claim may be read off it, and none is made here.
- The C3 overlay's RMS ≤ 15% pin (`data/baltic/scenarios/c3_bioen/README.md`) is **blind to
  `t_opt` and `Linf`** — a review sensitivity sweep found an 8 °C error in `t_opt` moves the RMS by
  <0.3 points, and a 30% error in `Linf` *improves* it — while it does constrain `K`: a −50% `K`
  perturbation moved cod_west's RMS from 8.3% to 13.6%. "All nine species within 15% RMS"
  therefore validates that the fitted `(Imax, r)` reproduce the config's own already-calibrated
  growth curve (`K`), not that the literature-anchored `t_opt`/`Linf` inputs are themselves
  correct — those two are cited, not fitted, and the RMS pin cannot catch an error in either.

### The OUT-schools `e_net` parity question (herring, sprat) — traced and shown numerically inert

Carried into this task as a live, unquantified gap. Production Baltic is the first config in this
branch's history where it is even possible to test: `data/baltic/baltic_param-out-mortality.csv`
sets non-zero `mortality.out.rate` for exactly two of the nine focal species —
`mortality.out.rate.sp1 = 0.05` (herring), `mortality.out.rate.sp2 = 0.08` (sprat), every other
species (and `data/examples_bioen`, Gate B's config, per Task 9's direct instrumentation) 0. Both
species use `movement.distribution.method = maps`, one of the two distributions that can actually
produce `isOut()` schools.

**The claim (from the carried-items note):** Java's `School.setNdead` (used for the OUT-mortality
pass, `MortalityProcess.java:413`, distinct from `incrementNdead`'s five in-step death sites)
rescales a school's `e_net` and `ingestion` by the survivor fraction on every out-of-domain death
(`School.java:372-385`: `this.ingestion *= factor; this.e_net *= factor;`). The port's
`out_mortality` (`osmose/engine/processes/natural.py:184-208`) rescales neither. This is now
**traced against the Java source, not merely inferred** — the earlier write-up
(`java-parity-open-question-out-schools.md`) correctly guessed the shape of the gap but had not
read `MortalityProcess.java`/`School.java` directly; this task did.

**What was unmeasured, and is now measured:**

1. `preyed_biomass` (`ingestion` in Java's rescale): confirmed a no-op regardless — `preyed_biomass`
   is reset to zero every step before predation repopulates it (`osmose/engine/simulate.py:217`),
   so a missing rescale here cannot accumulate across steps in either engine.
2. `state.e_net` (raw, per-school): this is the field the carried-items note worried an
   out-of-domain school could "carry a larger energy budget into the next step" through. Traced
   and unit-confirmed to be **provably inert** in the current engine, by two independent facts:
   - `_bioen_step` (`osmose/engine/simulate.py:496-503,603`) **excludes** `is_out` schools from
     the energy budget entirely (an already-documented, deliberate divergence — spec decision 18
     — because Java's own `EnergyBudget.run` iterates out-of-domain schools too and would
     dereference `matrix[-1][-1]`, an `ArrayIndexOutOfBounds` with no defined Java behaviour to
     match) and **unconditionally sets their `e_net` to exactly `0.0` every step** it remains out
     (`e_net_arr = np.zeros(...)`, only overwritten for in-domain schools via `sp_masks`). Whatever
     value `out_mortality` left an out-of-domain school's `e_net` at — rescaled or not — is
     overwritten before anything else in that same timestep can read it, because `out_mortality`
     (`mortality.py:2789`) runs strictly before the `_bioen_step` call
     (`simulate.py:2003`) that follows it, with nothing in between reading raw `state.e_net`.
   - Confirmed empirically, not just by code inspection, with two isolated checks (no engine run,
     synthetic `SchoolState`, milliseconds of compute — scripts under this session's scratchpad,
     `verify_out_enet.py`):
     - `out_mortality` alone, herring/sprat rates: kills 0.208%/0.333% of an out-of-domain
       school's abundance per step (`1 − exp(−rate/24)`, the omitted survivor factors are
       **0.9979 (herring) / 0.9967 (sprat)**) while leaving `e_net` and `preyed_biomass`
       bit-identical to their pre-call values — confirms the gap exists in this function alone.
     - `_bioen_step`, called on the real bioen-arm `EngineConfig` and real temperature field, on a
       synthetic batch of in-domain and `is_out` herring/sprat schools carrying a large nonzero
       `e_net` walking in (42.0/99.0): every `is_out` school's `e_net` comes out **exactly 0.0**;
       every in-domain school gets a freshly computed, nonzero value unrelated to what it walked
       in with.
3. The instrument the decision rule's criterion (ii) actually reads (`meanEnetFaced`, the
   harness's `e_bar_meanEnetFaced`) is **not** raw `state.e_net` at all — it is the
   abundance-weighted mean of `e_net_avg` (Java's `enet_faced`) over focal, feeding,
   **in-domain** schools (`osmose/engine/simulate.py:1292-1307`,
   `eligible = focal & (age_dt >= first_feeding) & ~is_out`).
   Two independent reasons this instrument cannot see the gap even if the above reasoning were
   wrong: it already excludes `is_out` schools by construction, and Java's `setNdead` never
   rescales `e_net_avg`/`enet_faced` in the first place (only `ingestion` and `e_net`) — so there is
   no gap on this field in the port to begin with.

**Conclusion:** the missing survivor rescale in `out_mortality` is a real, now-traced divergence
from Java, but it is numerically inert for every state variable and every reported output in the
current engine — not "below noise," but structurally unreachable, because the one field it could
corrupt (raw `state.e_net`) is unconditionally overwritten before any other code reads it, and the
decision rule's own instrument reads a different, unaffected field. **Falsifier, stated so this
conclusion can be checked later:** if any code path is found that reads raw `state.e_net` between
`out_mortality` (`mortality.py:2789`) and the following `_bioen_step` call (`simulate.py:2003`)
within the same timestep, this conclusion changes and the gap should be re-measured. None was found
in this task's reading of `simulate.py`'s per-step call order. Herring and sprat's A/B numbers
below (§4) can therefore be read on the same footing as the other seven species — they do not
carry a residual asterisk from this question.

## §2 — Gate A–G evidence

Every row below is read directly from `gates` in the committed report JSON. **All six PASS** —
the collapse reported in §4 is not a wiring or gate failure; every structural, parity and
thermal-instrument check that could have caught one passed. All `json key` values are under
`docs/diagnostics/baltic_c3_bioen_report.json:gates.*` unless noted.

| gate | scope | result | json key |
|---|---|---|---|
| A — bioen-off inertness, `array_equal` to the master fixture | 5 seeds, baseline arm | **PASS** | `gate_a` |
| B — cross-engine parity of bioen-on (Python vs Java 4.3.3) | Task 9, not re-run here | bioen: PASS; control: REVIEW (Hake, pre-existing) | `c3_gate_b_cross_engine.md`¹ |
| C — temperature load-through, 3-way, per layer + range | `bioen`, `bioen_plus2C` arms | **PASS** | `gate_c` |
| D — frames/layers (24, 2) + structural/parameter asserts | temp file + both bioen configs | **PASS** (both) | `gate_d_frames_layers`, `gate_d_structure` |
| E — zlayer wiring, engine-side | `bioen` arm, seed 42, step 12 | **PASS** — all 9 species checked, cell (25,21) | `gate_e` |
| F — thermal instrument (`phi_t(T_p)==1`, argmax ±0.1°C, φT∈(0,1], direction) | `bioen` arm, all 9 sp | **PASS** — all 9, `phi_t(T_p)` exact 1.0, argmax exact | `gate_f` |
| G — Task-0 unit tests, hand-computed Java-formula transcription | committed suite, not re-run | Committed (Tasks 0-5); not re-executed here | `tests/`² |

¹ `docs/diagnostics/c3_gate_b_cross_engine.md`, Task 9 — control REVIEW is Hake only,
traced to a pre-existing `data/examples` defect, not a port defect (§0/§1 above).
² bioen budget/starvation/reproduction unit tests, per-task reports.

Gates C–F were also fire/restore-verified against the real production files during Task 12's
development pass (`task-12-report.md`) — every gate raises on a real, deliberately introduced
violation and passes again once restored; that transcript is not repeated here.

## §3 — Parameter table

Every species' offline-fitted growth optimum, the engine parameters solved from it, and its label
(`data/baltic/scenarios/c3_bioen/README.md`, `scripts/fit_baltic_bioen_params.py --baltic`).
`m` (maintenance share of Imax at the 16 °C anchor, decision 7) = **0.3** for every species
(`BioenFixed().m_share`, juvenile herring trials, Bernreuther et al. 2012).

Label key (full text in §7 and the README): (a) size compromise, Bjornsson & Steinarsson 2002;
(b) provisional, no literature optimum found; (c) consumption proxy (gastric evacuation), not
growth; (d) secondary quotation, Kusakabe et al. 2016 via Fonds et al. 1992; (e) lagoon species
fit against open-coast field; (f) secondary, a preference not a growth optimum; (g) Lefebure
et al. 2011.

| species | sp | t_opt °C | T_p °C | T̄ °C | c_m | φT(T̄) | inflation | Imax | r | RMS % | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cod_west | 0 | 10 | 11.291 | 8.69 | 5.467e11 | 0.963 | 1.48 | 13.895 | 1.161 | 8.33 | a |
| herring | 1 | 15 | 17.512 | 8.14 | 7.649e11 | 0.673 | 2.12 | 17.408 | 2.202 | 7.23 | b |
| sprat | 2 | 18 | 21.106 | 8.53 | 7.238e11 | 0.533 | 2.68 | 18.409 | 2.666 | 2.15 | c |
| flounder | 3 | 19 | 22.261 | 6.08 | 8.596e11 | 0.392 | 3.64 | 23.107 | 1.278 | 10.38 | d |
| perch | 4 | 25 | 28.862 | 7.32 | 8.129e11 | 0.257 | 5.56 | 33.311 | 1.214 | 10.41 | e |
| pikeperch | 5 | 27 | 30.984 | 9.24 | 1.065e12 | 0.259 | 5.51 | 51.015 | 1.334 | 10.71 | e |
| smelt | 6 | 15 | 17.512 | 7.17 | 8.141e11 | 0.625 | 2.28 | 18.529 | 2.572 | 1.84 | f |
| stickleback | 7 | 21.7 | 25.290 | 8.09 | 6.856e11 | 0.368 | 3.88 | 21.989 | 3.086 | 3.27 | g |
| cod_east | 8 | 10 | 11.291 | 5.96 | 6.395e11 | 0.859 | 1.66 | 16.254 | 1.447 | 6.00 | a |

`inflation` = 1/(φT(T̄)·(1−m)); `Imax` in g·g⁻β·yr⁻¹; `RMS %` is the RMS length-at-age fit
residual (§0/§1's RMS-pin caveat applies).

`T_p` sits above `t_opt` for every species because maintenance (a bare Arrhenius term, no peak)
pulls the net-growth optimum below the mobilized-energy peak `T_p` marks. `T̄` is each species' own
habitat-mean temperature (its own depth layer, own movement-map footprint). Every species sits
1.7–5.3 °C below its own `t_opt` at `T̄` (README.md), so a +2 °C perturbation (§6) moves every
species toward its optimum, not past it.

## §4 — Final-decade means (the A/B table)

5-seed mean (t), all values from `final_decade_means` in the report JSON. `baseline` reproduces
the certified means to within 0.5% for every assessed stock (Gate A) — see the anchor check
below. **Bold** = one of the five assessed stocks (spec §4's decision rule).

| species | baseline mean | bioen mean | bioen/baseline | bioen/certified | in envelope (bioen) |
|---|---:|---:|---:|---:|---|
| **cod_west** | 12,810.8 | **0.0** | 0.000 | 0.000 | no |
| **cod_east** | 65,251.2 | **0.0** | 0.000 | 0.000 | no |
| **herring** | 2,539,645.2 | **0.0** | 0.000 | 0.000 | no |
| **sprat** | 1,024,324.0 | **309,994.2** | 0.303 | 0.303 | no |
| **flounder** | 33,063.4 | **0.0** | 0.000 | 0.000 | no |
| perch | 43,774.0 | 0.0 | 0.000 | 0.000 | no |
| pikeperch | 1,400,081.1 | 128,120.2 | 0.092 | 0.090 | no |
| smelt | 680,580.0 | 294,616.2 | 0.433 | 0.431 | no |
| stickleback | 80,282.3 | 71,543.1 | 0.891 | 0.883 | yes |

**Five** species have a final-decade mean of exactly 0.0 t: cod_west, cod_east, herring, flounder
among the assessed set, plus perch among the indicative-tier species (not part of the verdict, per
`ASSESSED_STOCKS`, but the same pattern). Every zeroed bioen mean is bit-identical to 0.0
across **all 5 seeds** for those five species (`std: 0.0`, `per_seed: [0.0, 0.0, 0.0, 0.0,
0.0]` in the JSON) — this is not a noisy near-collapse, it is deterministic extinction by year 50
on every draw. Under the wider "bioen/baseline < 0.10" reading of "affected", **six** of nine
species qualify — the five zeros above plus pikeperch, whose bioen/baseline is 0.092 (a 91%
reduction, not an exact zero: 128,120.2 t survives against a 1,400,081.1 t baseline). These are two
different counts for two different definitions ("exactly extinct" vs. "reduced past the criterion
(i) threshold") — elsewhere in this document, "collapsed"/"zeroed" always means the five-species,
exactly-0.0 reading (§5, §6); "affected" in this paragraph is the wider six-species reading and is
not used again.

**Criterion (iii) anchor sanity check:** the spec text is ambiguous between anchoring the
bounded-displacement criterion on the certified means or on this run's own baseline arm; the
harness implements the certified-means anchor (`CERTIFIED_MEANS`, a literal snapshot of
`docs/baltic_certification_2026-08-14.md`). Checked: the baseline arm's own `ratio_to_certified`
for the 5 assessed stocks is cod_west 0.995, cod_east 1.001, herring 0.997, sprat 1.000, flounder
1.004 — every value within 0.5% of 1.0, so **the anchor choice is immaterial to this verdict**:
either anchor gives the identical pass/fail pattern.

## §5 — Instruments

**Realized ration ē/ĝ and f (decision 7, `realized_ration` in the JSON), `bioen` arm, final
window:**

| species | ē (`meanEnetFaced`) | ĝ (fitted `g_net`) | ē/ĝ | f = m+(1-m)·ē/ĝ |
|---|---:|---:|---:|---:|
| cod_west | 0.0 | 7.507 | 0.000 | 0.300 |
| cod_east | 0.0 | 8.526 | 0.000 | 0.300 |
| herring | 0.0 | 6.067 | 0.000 | 0.300 |
| sprat | 5.333 | 5.190 | 1.027 | 1.019 |
| flounder | 0.0 | 4.798 | 0.000 | 0.300 |
| perch | 0.0 | 4.829 | 0.000 | 0.300 |
| pikeperch | 8.466 | 7.402 | 1.144 | 1.101 |
| smelt | 5.484 | 6.151 | 0.891 | 0.924 |
| stickleback | 5.084 | 4.432 | 1.147 | 1.103 |

**Five** species (§4) show `ē = 0.0` **exactly**, not `NaN` — `bioen_enet_faced` is an
abundance-weighted mean over focal, feeding, in-domain schools (`simulate.py:1292-1307`); with
zero such schools left (the population extinct), the denominator guard `np.where(denom > 0, ...,
0.0)` returns `0.0` rather than `NaN`. This is a legitimate zero, not a NaN the decision-rule fix
had to protect against (matches `decision_rule_undetermined: []`) — but it means `ē = 0` here
reads as "no feeding population survived to the final window," not as a live starvation
measurement on a surviving population. The four non-zero species (sprat, pikeperch, smelt,
stickleback) all show `ē/ĝ` at or above 0.89, comfortably clearing the ≥ 0.6 threshold.

**Length-at-age (paired RMS %, ages ≥ 1 yr, `bioen` vs `baseline`, `length_at_age` in the
JSON):** computed for **cod_west only** (71.9%, `n_seeds: 5`); **`NaN` for all other 8 species**,
including sprat, pikeperch, smelt and stickleback, which did not collapse in the bioen arm.
**This was root-caused after this run, by an independent review (task-13-14-review.md E2/E8),
and the instrument has since been fixed in the harness code (task-14-fix-report.md) — the numbers
below are from the run as originally executed and were NOT recomputed; see the end of this
subsection for exactly what changed and what didn't.**

Root cause: the in-memory multi-species output cache concatenates each species' wide by-age frame
at its own width (`n_bins = lifespan_dt`-derived, one width per species,
`osmose/engine/output.py:_build_distribution_dataframes`) with `pd.concat`
(`osmose/results.py:351`), which NaN-pads every species out to the widest species' bin count —
cod_west (21 bins, the longest-lived species) is the only one with no padding, which is exactly
why it is the only species that produced a number. `length_from_age_bins`'s old guard
(`abundance <= 0`) did not reject a padded bin, because `float('nan') <= 0` is `False` in Python;
the NaN flowed through into the RMS for the other 8 species. Reproduced deterministically on a
fresh 2-yr, no-overlay Baltic run (no engine run was needed to find the cause, only to confirm
it): **8 of 9 species show
all-NaN age-bin columns in the raw `abundance_by_age()`/`biomass_by_age()` frames, cod_west the
sole exception** — row-for-row the same pattern as the committed JSON's NaN/non-NaN split.
**Provably non-gating**, independent of the root cause: the decision rule reads only `biomass()`
(a single cross-species frame that is never concatenated across species,
`_CROSS_SPECIES_OUTPUT_TYPES`) and `meanEnetFaced` (homogeneous per-species columns, also never
raggedly concatenated) — neither can carry this padding, so nothing this defect touched could have
changed §4's verdict.

**The fix** (`length_from_age_bins`, `scripts/baltic_c3_bioen_ab.py`): an `np.isfinite` guard on
both `abundance` and the derived `weight_g`, so a padded bin is dropped exactly like a
non-positive one. On the same 2-yr reproduction run, after the fix, the padded bins no longer
appear in the function's output at all (verified: 0 of the previously-NaN bins now emit a
value — they are cleanly absent, not zero-filled). A second, independent defect in the same
instrument was fixed alongside it: `run_c3`'s call site passed the **whole run's** frame to
`length_from_age_bins`, against that function's own documented contract ("frames already
restricted to the time window of interest") — now restricted to the final-decade window, matching
the convention `_final_window_mean` already uses elsewhere in this harness.

**Neither fix was applied to the committed `docs/diagnostics/baltic_c3_bioen_report.json` — the
50-yr, 5-seed A/B was deliberately NOT re-run to regenerate it** (an engine-time decision, not a
correctness one). Two consequences worth being explicit about, so a reader does not assume more
was recomputed than was:

1. The cod_west 71.9% figure above, and the 8 NaNs, are the **original, pre-fix, whole-run**
   numbers — they were the actual output of the run that produced every other number in §4–§7 and
   remain the citable numbers for that run. The one caveat about this number is corrected next.
2. Had the fix been in place for that run, the final-window restriction would return **no** value
   for cod_west either: `length_from_age_bins` on the final-decade frame finds no shared bins for
   a population that has been extinct for the entire decade, so the fixed instrument yields
   `n_seeds: 0`, not a number. **A final-decade length-at-age comparison cannot see growth for a
   stock that is not there in the final decade — collapsed stocks need a pre-collapse window
   instead.** §8 reports exactly that: a separate, explicitly-labelled measurement taken before
   the collapsed stocks died out. The fix is not purely subtractive, though: the `NaN`-guard half
   would also let the four species that *did not* collapse (sprat, pikeperch, smelt, stickleback —
   all `NaN` today purely from the padding, not from extinction) return a real final-decade RMS on
   a future re-run — trading cod_west's one whole-run number (which the final-window restriction
   nulls out, per the point above) for four final-decade ones on the species the instrument can
   actually still measure.

**Correcting the previous reading of the 71.9% number itself** (task-13-14-review.md F2): the
prior draft of this document explained it as coming from "the single final year" and as "most
likely a residual egg/YOY fragment." **Both premises were checked against the code and are
false.** `run_c3`'s pre-fix call (`scripts/baltic_c3_bioen_ab.py:1130-1152` at the time) passed the
whole 50-year `abundance_by_age()`/`biomass_by_age()` frames, not a final-year slice; and bin 0 is
explicitly excluded before the RMS is computed (`shared_bins = ... - {0}`), so it cannot be an egg
fragment — every contributing bin is a real age ≥ 1 from a year cod_west was alive. **The number
is a real whole-run growth-deficit average, and it agrees with the independent −75% in-engine
length-at-age measurement in §8** — it is not an artifact to be explained away, and the prior
framing that told a reader to disregard it was itself the error.

**Realized annual ingestion (decision 17, `realized_ingestion` in the JSON), `bioen` arm:**
cod_west/cod_east/herring/flounder/perch = 0.0 (consistent with §4's extinction); sprat 18.42,
pikeperch 15.22, smelt 46.37, stickleback 16.20. `baseline` is `null` for every species — the
`output.bioen.ingest.enabled` output family does not exist on a bioen-off run
(`FileNotFoundError`, caught by the harness), so no bioen-vs-baseline ratio could be formed; that
gap is in the instrument, not a result. **Unit caveat:** this output is `bioen_ingestion` =
`_species_mean(state.e_gross, ...)` (`simulate.py:1268`, Java's own "ingestion" output name) — the
mean `E_gross` per school in the tonnes-per-school budget framework, not a per-fish rate in the
same g·g⁻β·yr⁻¹ units as the parameter table's `Imax`. The two are not directly comparable
number-for-number; smelt's 46.37 exceeding its own fitted `Imax` of 18.53 reflects this unit
difference, not ingestion exceeding its own cap. The parameter table's `inflation` column (§3)
remains the citable, unit-consistent figure for decision 17's Imax-inflation framing.

**Seeding diagnostics:** `results.ssb()` raises `FileNotFoundError` in in-memory mode (no SSB
output family built by `_build_dataframes_from_outputs`) — skipped per the brief's documented
fallback, confirmed reachable. See the "Context from Task 13" box above for the SSB evidence that
*does* exist (from Task 13's dedicated 8-yr run with `output.ssb.enabled` set): cod_west and
cod_east never reached a single spawner in 8 years under the stress condition.

**Reminder for whoever reads these numbers next:** `biomass()`/`abundance()` exclude young-of-year
via `output.cutoff.age` (0.5 yr for every Baltic species); `*_by_age`/`*_by_size` do not — the
length-at-age instrument above uses the latter family, the final-decade means (§4) use the former;
they are not interchangeable. `biomass/abundance` read as a mean weight is egg-dominated
(~0.0005 g/fish) regardless of growth; `abundance_by_size()` bin occupancy cannot resolve sub-10 cm
growth. None of this task's numbers were read through either trap.

## §6 — The +2 °C arm

`bioen_plus2C` minus `bioen`, final-decade mean delta (t) and Gate F's habitat-mean `g_net`
shift — reported only, not gated on the decision rule (spec §3.5):

| species | Δ final-decade mean (t) | g_net shift |
|---|---:|---:|
| cod_west | 0.0 | +0.067 |
| cod_east | 0.0 | +0.674 |
| herring | 0.0 | +0.860 |
| sprat | +59,860.2 (+19.3%) | +0.843 |
| flounder | 0.0 | +0.899 |
| perch | 0.0 | +0.861 |
| pikeperch | +16,725.6 (+13.1%) | +1.299 |
| smelt | +5,619.0 (+1.9%) | +0.944 |
| stickleback | +18,937.3 (+26.5%) | +0.779 |

Every species sits 1.7–5.3 °C below its own `t_opt` at `T̄` (§3), so +2 °C moves every species'
habitat-mean net growth rate up (`g_net shift` positive for all 9, matching Gate F's direction
check exactly) — this is a mechanical consequence of the fitted thermal curves, not a new
finding. **The +2 °C arm does not rescue any of the five collapsed species** — all five remain at
exactly 0.0 in every seed, unchanged from the `bioen` arm — while the four survivors all grow
further (+1.9% to +26.5%). A 2 °C perturbation this small cannot compensate for a mechanism
(predation outrunning growth, per Task 13) that a warmer thermal curve does not address; the
delta pattern is consistent with the collapse being a growth/predation-timing problem, not a
temperature-forcing problem this lever can fix.

## §7 — Labels (spec §4, restated verbatim from the harness)

These are carried verbatim from `REPORT_LABELS` in `scripts/baltic_c3_bioen_ab.py` (and the
committed JSON's `labels` field), not re-typed by hand:

1. Single optimum per species (cod's is size-dependent, Bjornsson & Steinarsson 2002).
2. Herring optimum (15 °C) is PROVISIONAL — no herring growth optimum was retrieved in three
   literature searches.
3. Secondary-source optima for flounder (19 °C, via Kusakabe et al. 2016 quoting Fonds et al.
   1992) and smelt (15 °C, via Krause 2008 quoting Vinni et al. 2004).
4. Maintenance share m anchored on juvenile herring trials at 16 °C (Bernreuther et al. 2012),
   transplanted to every species.
5. No upper thermal limit at e_D = 1.5 — phi_t(T) never turns back down at high T in this
   parameterisation.
6. Perch and pikeperch are lagoon species fitted against the open-coast surface field — phiT
   peaks at 0.7-0.8 in their actual lagoon habitat, inflating the fitted Imax.
7. Ingestion is capped at Imax*w^beta BEFORE phiT (Java form) — consumption inflation for
   cold-habitat species, decision 17.
8. Food-unlimited offline fit vs a food-limited engine — the in-engine A/B measures the emergent
   departure from the fitted curve, not a re-run of the fit.
9. Larval phase (age < 1 yr) is unfitted — decision 10, reported not fitted.
10. Two-layer temperature is a proxy (surface nan-mean of 5 CMEMS depth levels; bottom = CMEMS
    bottomT), a climatology (1993-2021 monthly means, not a hindcast), and fo2 is off in Stage 1
    (decision 19).
11. Reproduction under bioen keeps the certified Python-side stock-recruitment regulation
    (decision 5) — this A/B changes growth structure, not recruitment structure.

Plus the two framing obligations already stated in §0/§1 above: Gate B's config is a parity
vehicle, not a calibrated ecosystem; the RMS pin validates `K`, not `t_opt`/`Linf`.

**Label 8's departure, quantified:** label 8 names the gap between the offline fit and the
in-engine result but does not size it. §8 below measures it directly — an 8–77% length-at-age
deficit at ages 1–2, paired treatment vs. control — and reads the pre-registered negative in light
of that number.

## §8 — What Stage 2 would do (or why C3 closes)

**C3 closes by characterization.** The pre-registered decision rule (spec §4), applied to this
run without adjustment, does not warrant Stage 2: four of the five assessed stocks fail all
three criteria — (i) no-structural-collapse, (ii) `ē/ĝ ≥ 0.6`, (iii) bounded displacement — with
`bi_mean = 0`, `ē/ĝ = 0`, `bioen/certified = 0` for cod_west, cod_east, herring and flounder, plus
the aggregate criterion-(iii) count (`0/5` within a factor of 2, need `≥3`). No criterion came
back `undetermined`; every failure is a hard, five-seed-identical zero. Sprat is the only
assessed stock that passes cleanly. **None of what follows changes this verdict** — it narrows
*what the verdict is evidence of*, which the rule itself cannot determine.

### What the negative is — and is not — evidence of

The pre-registered rule answers one question: does the bioen parameter set, as fitted and wired,
sustain the Baltic community at this scale? The answer is no, cleanly and reproducibly. It cannot
by itself answer a second, different question: is that because bioenergetics as a mechanism does
not work for the Baltic, or because *this offline-fitted parameter set's growth did not transfer
into the coupled engine*? A rule built on final-decade population outcomes cannot distinguish those
two — both produce the same zeros. The following measurement, taken specifically to separate them,
points at the second.

§3 reports cod_west's offline fit (growth curve alone, food-unlimited, no predation or competition)
at **RMS 8.33%** against its literature-anchored target. That number describes only the fit's own
internal consistency; it says nothing about what happens once the same parameters run inside the
full coupled engine. **A paired in-engine measurement does**: a treatment/control run (seed 42,
`population.seeding.year.max=1`, the same stress condition Task 13 used, `bioen` overlay vs the
overlay removed) with `length_from_age_bins` read at each species' own age-1/age-2 length in
years 2 and 3 — before the species that go extinct have done so, which the final-decade window
(§5) cannot see. **Every species is shorter at age 1, in both years, than its own baseline
trajectory, by 8% to 77%; eight of nine are shorter at age 2 too** (independently reproduced for
this fix — task-14-fix-report.md — and matching an earlier, separately-run measurement in
task-13-14-review.md E5 to within rounding). The one exception is herring's single year-3 age-2
reading, where the bioen arm is marginally *longer* than baseline (16.56 → 17.10 cm, +3%) — a
pervasive, directional deficit with one disclosed exception, not a universal one. Cod_west's own
age-1 length is representative of the worst end of the 8–77% range: **37.5 cm in the
classic-growth control at year 2, 8.8 cm under the bioen overlay at the same age (−77%); 49.6 →
12.5 cm at age 1 and 61.6 → 15.2 cm at age 2 by year 3 (both −75%)** — one data point inside the
same whole-run average §5 reports as 71.9% RMS.

This lines up with which species collapse and which survive, closely enough to be the more
specific and actionable reading of the two:

| species | m0, length at maturity (cm) | bioen/baseline (§4) | outcome |
|---|---:|---:|---|
| stickleback | 4.5 | 0.891 | survives, least affected |
| sprat | 9.0 | 0.303 | survives, sole `ASSESSED_STOCKS` pass |
| smelt | 10.0 | 0.433 | survives |
| herring | 18.0 | 0.000 | **extinct** |
| perch | 18.0 | 0.000 | **extinct** |
| flounder | 22.0 | 0.000 | **extinct** |
| cod_east | 22.0 | 0.000 | **extinct** |
| cod_west | 38.0 | 0.000 | **extinct** |
| pikeperch | 40.0 | 0.092 | survives — the exception |

`m0` is `species.maturity.size.sp{i}`, verified identical between the `baseline` and `bioen`
arms for every species (`data/baltic/baltic_param-species.csv`; task-13-14-review.md E3a) — the
overlay changes growth, not the maturity threshold a fish has to reach. If growth is running
8–77% too slow across the board, a species that only has to reach 4.5–10 cm can still get there
inside its lifespan; one that has to reach 18–38 cm increasingly cannot. Eight of the nine species
sort cleanly on that line. **Pikeperch does not**: it carries the single highest `m0` of all nine
(40.0, above even cod_west's 38.0) yet survives — reduced 91% (0.092×), the most severely affected
survivor in the table, but not extinct. This is a genuine exception to a monotone reading, not a
detail to omit: something about pikeperch (its own fitted `Imax`/`r`, its 15-yr lifespan versus
cod_west's 20, its predation exposure, or some combination) lets it clear its own threshold where
flounder and cod_east — needing a smaller size, 22.0 cm — do not. That was not chased down in this
task and is not resolved here.

**What this does and does not establish.** It does not prove the transfer-failure reading — that
would need tracing the growth deficit to a specific step in the coupled budget (competition for a
shared, non-infinite prey field; the ingestion cap interacting with realized prey density; or
something else) and showing the offline fit would reproduce in-engine if that step were corrected,
none of which was attempted here. What it does establish is that the pre-registered rule's negative
and the offline fit's RMS 8.33% are not in tension the way reading them side by side might suggest:
the fit describes an idealized, food-unlimited scenario, and the coupled engine is neither. A large,
consistent, mechanistically-explicable gap between the two — not a small one, not a scattered
one — is the evidence available, and it is evidence for "this parameter set did not transfer,"
not proof of it, and it does not by itself indict bioenergetics as a mechanism for the Baltic more
broadly.

Spec §4 frames Stage 2 as "bounded recalibration of the bioen parameter set only" — a scalar
rescale of `r`/`Imax` guided by the failing species' own `ē/ĝ`. **That framing does not fit what
this run found, for a more specific reason than "there is nothing to act on."** A one-parameter
rescale presumes a population that is underperforming its own fitted ration curve by a bounded
amount (`ē/ĝ` somewhat below 1, as sprat/pikeperch/smelt/stickleback show at 0.89–1.15) — the four
survivors are exactly that case. It has nothing to act on for a population whose final-decade
`ē/ĝ` is `0`, but **that `0` is where the instrument is placed, not evidence there is no signal to
rescale against**: it is the final decade of a run in which those five stocks are already extinct,
not a measurement of a live, underperforming population. In the years those stocks *did* have fish
(above), the signal is large, consistent, and directional — a candidate rescale target, not an
absence of one. Task 13's independent mechanism finding (predation climbing to a complete-cohort
wipeout while starvation stays small and declines) is consistent with this: bioen's growth is slow
enough that juveniles spend longer in a predation-vulnerable size window, which is exactly what an
8–77% length-at-age deficit at ages 1–2 would produce. Whether a scalar `Imax`/`r` rescale,
sized against the pre-collapse deficit measured above, is sufficient to close that gap — or
whether the mismatch is structural enough (a competition or ingestion-cap interaction under
realistic prey density, not just a scale factor) to need changing how fast individuals cross the
vulnerable window relative to the certified predation-accessibility matrix — is the open question
Stage 2 would need to answer, and answering it would need its own spec, its own gates, and its own
pre-registered decision rule, informed by the deficit measured here rather than starting from
scratch.

**What this stage leaves behind, independent of the verdict** (spec §8's own success criterion):
a Java-parity bioen budget (Tasks 0–5, Gate B PASS), a working two-layer temperature loader
(Tasks 6–7, Gate C/D/E/F PASS), a 9-species offline-fitted parameter set with its own
documented blind spots (§0/§1, README.md), a realistic-config bioen regression test that caught
the collapse before this A/B ran it at scale (Task 13), and this pre-registered A/B harness
itself, reusable for whatever recalibration or redesign is scoped next. None of that existed
before this branch.

## §9 — Follow-ups (not blocking this task; recorded for whoever picks C3 up next)

- **The growth/predation-timing mechanism itself (the main follow-up).** Task 13's finding —
  predation, not starvation, drives cod_west/cod_east/flounder/herring to collapse, with a
  bioen-off control on the identical config sustaining all five — is the standing, verified
  starting point for whatever comes after this stage. `tests/test_baltic_c3_bioen_smoke.py` is
  committed `xfail(strict=True)` with the diagnosis inline; it will flip to a loud `XPASS` the
  moment someone's change (a recalibration, a predation-accessibility adjustment, or something
  else) fixes this, which is the signal to re-open the question. Whether the fix belongs in the
  bioen fit, the predation-accessibility matrix, or the growth-rate structure itself was
  explicitly out of scope for both Task 13 and this task to chase.
- **Length-at-age `NaN` for 8/9 species (§5) — root-caused and fixed in code, not re-run.**
  Cause: `pd.concat` over per-species by-age frames of different widths (`osmose/results.py:351`,
  widths from `osmose/engine/output.py:_build_distribution_dataframes`) NaN-pads every species
  except the widest (cod_west); the old `abundance <= 0` guard in `length_from_age_bins` did not
  reject `NaN` (`float('nan') <= 0` is `False` in Python). Fixed: an `np.isfinite` guard on
  `abundance` and `weight_g`, plus restricting `run_c3`'s call site to the final-decade window
  (its previous whole-run call was a second, independent defect against the function's own
  documented contract). Verified `REPORTED`, non-gating throughout (spec §4) — the decision rule
  reads only `biomass()` and `meanEnetFaced`, neither of which can carry this kind of padding, so
  §4's verdict was never at risk. **This run's committed JSON was not regenerated** (the 50-yr,
  5-seed A/B was deliberately not re-run for this fix) — the 71.9%/NaN numbers in §5 remain the
  original run's numbers; a future re-run with the fixed harness would report `n_seeds: 0` for
  the extinct species at this final-decade window by construction (§5), so a length-at-age
  comparison for the collapsed stocks will always need a pre-collapse window, not this one, to
  produce a number. Full before/after reproduction: task-14-fix-report.md.
- **Numba bioen kernel.** CORRECTED 2026-09-05 (final-branch-review.md F4) — this bullet
  previously said the batched Numba mortality kernels were bypassed entirely under bioen and
  that a compiled specialisation was future work. Both claims were the inverse of what this
  branch shipped: the bioen-Numba-kernel sub-plan ported `_apply_starvation_for_school`'s bioen
  branch and the interleaved survivor-rescaling loop into the batched kernels and flipped both
  dispatch gates (`2cc1f69`), so bioen now runs on the same compiled path as bioen-off. Measured
  ~149× faster than the pre-flip per-cell path at a 4-yr horizon (157.8× on an independent
  re-run; the ratio grows with horizon, ~82× at 1 yr). `_mortality_in_cell_numba` remains in the
  source as the per-cell equivalence oracle the kernel tests diff against — not a production
  path (see CLAUDE.md's bioen/Numba gotcha).
- **f_o2 spec.** Bioen's oxygen-limitation term is off in Stage 1 (decision 19, label 10 above).
  The bottom-oxygen → benthos-K coupling already live in production Baltic
  (`ltl.oxygen.benthos.enabled`, CLAUDE.md) is a different mechanism (resource carrying capacity,
  not per-fish assimilation) and is unaffected by this decision either way.
- **Maturity latch.** Java latches maturity once (`setIsMature`); the port recomputes it every
  step. Identical while `m1 = 0` (Stage 1's fitted parameter set uses `m1 = 0` for every species,
  per the spec §0 table) — deferred, not fixed, and would need its own gate if `m1` is ever fit
  nonzero.
- **B2 bottom-T swap.** The two-layer temperature forcing here is a climatology (1993-2021 monthly
  means), not a hindcast. B2's scenario machinery (`docs/baltic_b2_scenarios_2026-08-30.md` per
  MEMORY.md) already has a citable RCP×load table built on a different bottom-temperature series;
  swapping this overlay's climatology for a hindcast or an RCP series is a config-only change
  (swap the forcing NetCDF + refit `T_p`/`Imax` against the new `T̄`) once C3's own verdict is in.
