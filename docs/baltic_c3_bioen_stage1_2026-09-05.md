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

  - **LOCATED 2026-09-13 — it is the fit's OBJECTIVE, not any of `c_m`, `Imax` or `beta`.**
    Four measurements on the production Baltic + this overlay, 6–10 yr, seed 42, eliminating one
    candidate each; then the age curve, which names the cause.

    | candidate | measurement | verdict |
    |---|---|---|
    | prey supply / accessibility | realized ingestion ÷ `Imax` cap, per species, pre-collapse: cod_east 0.97, perch 0.95, cod_west 0.95, flounder 0.92, herring 0.77 | **ruled out** — the collapsing stocks eat 92–98 % of everything they are allowed. The relation is *inverted*: the two lowest ratios (pikeperch 0.36, smelt 0.38) are both survivors. |
    | `c_m` | `m_share = e_maint/e_gross` vs the fitted 0.30 target: cod_east 0.135, cod_west 0.178, flounder 0.290, perch 0.312, herring 0.320 | **ruled out** — collapsing stocks sit *at or below* target; cod has 82–86 % of gross energy free for growth. |
    | `beta` | `m_share` across weight terciles | **ruled out** — flat (cod_west 0.18/0.18/0.17). Both `e_gross` and `e_maint` scale as `w^beta`, so a mismatch would show as drift with size. |
    | temperature / the `1/(φT(T̄)(1−m))` inflation | measured `phi_T` vs §3's `φT(T̄)` | **ruled out** — matches within 8–19 %, and mostly *higher* than assumed (cod_west 0.887 vs 0.963; perch 0.304 vs 0.257). The inflation is doing its job. |

    Every input the fit reasons about is as designed, which leaves the objective that chose them.
    Running `simulate_growth` (`osmose/calibration/bioen_offline.py:100`) with the **committed**
    parameters against each species' own config vBGF:

    | age | cod_west fit/vBGF | perch | stickleback |
    |---:|---:|---:|---:|
    | 1 | **0.58** | **0.49** | 1.01 |
    | 2 | 1.00 | 0.81 | 0.93 |
    | 3 | 1.14 | 1.04 | 0.94 |
    | 4–8 | 1.16–1.17 | 0.88–0.99 | 0.95 |

    **The fit is 42–51 % short at age 1 and then matches or overshoots from age 2 on.** Its RMS
    runs over the whole ≥1 yr range on *absolute* lengths, so ages 2–8 — where cod_west is 14–17 %
    **over** target — dominate the residual. That is how §3's 8.33 % RMS coexists with a halved
    first year, and it is the RMS-pin caveat (§0/§1) biting in a way nobody had quantified.

    This reconciles the rest of the stage rather than competing with it: the budget is healthy
    (ingestion at the cap, `m_share` at target), so the fish are **not starving** — they start the
    race at half size and spend an extra year inside the predation window before any size refuge,
    which is exactly the predation-not-starvation signature Task 13 measured with its bioen-off
    control. It also explains the `m0` ordering in §8: stickleback (m0 4.5 cm, on target from age 1)
    survives, while cod (38), pikeperch (40), flounder/cod_east (22) and herring/perch (18) must
    cross a far larger gap starting from half size.

    **Target for a re-fit:** the juvenile regime, not the global parameters.
    `simulate_growth` gives larvae their own cap (`imax + (theta−1)·c_rate` while
    `age_dt < larvae_thres_dt`); weighting early ages in the residual — log-length, or an explicit
    age-1 constraint — would raise that boost without disturbing the adult curve the present fit
    already reproduces well. Note the pikeperch caveat from §8 still stands: at 0.47 fit/vBGF at
    age 1 it is among the worst early-growth cases yet survives, so early growth is not the whole
    story for every species.

  - **REFUTED BY INTERVENTION, same day.** The growth account above is **wrong as a cause of the
    collapse**, and it was overturned by doing the thing it implied rather than by further
    argument. Recorded in full because every step of it was individually sound and it still failed
    — which is the point.

    The engine already implements a juvenile cap boost that the committed overlay switches off
    (`theta = 1.0`, `c_rate = 0.0`; `per_fish_ingestion_cap` computes
    `i_eff = imax + (theta−1)·c_rate` while `age_dt < larvae_thres_dt`), and the fit's forward
    model does not implement it at all. Activating it as a **third fitted parameter** with a
    one-year window does exactly what the diagnosis predicted it would:

    | | committed | with juvenile boost |
    |---|---|---|
    | cod_west age-1 fit/vBGF | 0.53 | **1.05** |
    | cod_west RMS | 8.33 % | **0.86 %** |
    | all nine species, age 1 | 0.53–1.03 | **1.03–1.09** |
    | all nine species, RMS | 1.84–10.71 % | **0.80–1.65 %** |

    Strictly better on both axes — the whole curve on target, ages 2/3/5/8 at 0.98–1.03. Then the
    engine test, 8 yr, `population.seeding.year.max = 1`, seed 42, three arms:

    **the same five stocks collapse.** cod_west, cod_east, flounder extinct; perch and herring at
    zero biomass; herring marginally *worse* than committed. The bioen-off control sustains all
    nine, as before.

    Two checks confirm this is a real null and not a switched-off knob or a starved one: the
    mechanism engages (`larvae_thres_dt = 24`, larval `i_eff` ≈ 2× adult, read back from
    `EngineConfig.from_dict`), and the extra food is genuinely **eaten** — juvenile ingestion/cap
    on the boost arm is 0.94–0.98 for every collapsing stock, so raising the cap did not merely
    expose a prey ceiling.

    **Growth is therefore not the binding constraint.** Fix it completely and nothing changes.

  - **The surviving candidate is RECRUITMENT, and it correlates almost perfectly.** Total eggs
    produced over the same 8 yr stress run, bioen ÷ baseline:

    | survives | ratio | | collapses | ratio |
    |---|---:|---|---|---:|
    | stickleback | 1.13 | | cod_west | 0.48 |
    | smelt | 1.11 | | herring | 0.20 |
    | sprat | 0.79 | | cod_east | 0.17 |
    | pikeperch | 0.45 | | perch | 0.12 |
    | | | | flounder | 0.11 |

    Bioen produces **11–48 %** of baseline egg output for every collapsing stock — a 2–9×
    recruitment shortfall — while every survivor except pikeperch sits at 0.79–1.13. Under bioen
    eggs come from gonad energy (`rho`, `e_net`); under classic growth they come from a prescribed
    relationship. A population that is not replaced cannot persist once seeding stops, and
    predation then shows up as the *proximate* cause while recruitment is the *ultimate* one —
    which is exactly the predation-not-starvation signature Task 13 measured, now with a mechanism
    behind it.

    **Held to the same standard that just cost the growth account:** this is a strong correlation
    across nine species with one exception (pikeperch, 0.45 and surviving — the same species that
    breaks the `m0` ordering and has the lowest ingestion/cap). **It has not been tested by
    intervention.** The growth story had comparable correlational support and failed. Do not treat
    recruitment as established until someone raises bioen egg output and shows the stocks persist.

    Reproduce all of the above with `scripts/c3_growth_deficit_diagnosis.py`. The boost re-fit has
    since been committed as an opt-in path (`fit_species(..., juvenile_boost=True)`,
    `osmose/calibration/bioen_offline.py`, `374bc26`, default-off and bit-identical when off); the
    egg count was a scratch experiment at the time and has since been **reproduced 9/9** by the
    committed harness below.

  - **TESTED BY INTERVENTION 2026-09-13 — the hypothesis is ILL-POSED for four of the five stocks,
    not refuted.** `scripts/c3_recruitment_intervention.py`, committed with its reading
    pre-registered (`413bce8`) *before* the run. It wraps `regulate_recruitment` — the single
    shared choke point for both reproduction paths (classic `reproduction.py:344`, bioen
    `simulate.py:885`) — so it is instrument and intervention at once, with no engine edit. On the
    bioen arms it swaps the gonad-derived egg count for the baseline path's own
    `sex_ratio · relative_fecundity · SSB · season · 1e6 · K`, skipping seeded steps (which already
    use that formula). Growth, maintenance, starvation, the TPC, the Shepherd curve, the RV gate
    and egg-school creation are all untouched. Four arms at the growth refutation's exact stress
    (8 yr, `seeding.year.max = 1`, seed 42), so the two tests are directly comparable. **Two runs
    byte-identical** — deterministic.

    | species | baseline | bioen | bioen_rec (K=1) | bioen_rec10 (K=10) | R = rec10/bioen eggs |
    |---|---:|---:|---:|---:|---:|
    | cod_west | 1 454.5 | **0.0 ✗** | **0.0 ✗** | **0.0 ✗** | 1.00 |
    | cod_east | 108 826.0 | **0.0 ✗** | **0.0 ✗** | **0.0 ✗** | 1.00 |
    | flounder | 35 684.8 | **0.0 ✗** | **0.0 ✗** | **0.0 ✗** | 1.00 |
    | perch | 56 511.2 | **0.0 ✗** | 0.0 | **0.0 ✗** | 1.00 |
    | herring | 2 458 398.6 | 1.2 | 0.7 | **2 026.7** | 2.40 |
    | sprat | 922 205.4 | 239 248.8 | 424 233.3 | **1 611 006.0** | 11.52 |
    | pikeperch | 1 545 341.8 | 285 387.4 | 287 005.4 | 371 836.2 | 17.36 |
    | smelt | 668 519.6 | 416 821.5 | 433 905.9 | 861 009.8 | 9.51 |
    | stickleback | 91 392.1 | 57 556.1 | 57 392.3 | 153 847.2 | 7.63 |

    Final-year biomass (t); ✗ = zero abundance. The pre-registered rule needed `R ≥ 5` to read a
    verdict at all, and **no collapsing stock reached it**, so the registered outcome is
    `INCONCLUSIVE` across the board. That is recorded as-is. But *why* it is inconclusive is the
    finding, and it is decisive.

    **There is no recruitment to boost.** Decomposing run-total SSB against the year-1 seeding
    contribution (`24 × population.seeding.biomass`):

    | species | SSB total | seeded part | REAL spawning stock, 168 post-seeding steps |
    |---|---:|---:|---:|
    | cod_west | 1 200 000.0 | 1 200 000.0 | **exactly 0.0 t** |
    | cod_east | 2 400 000.0 | 2 400 000.0 | **exactly 0.0 t** |
    | flounder | 1 920 168.8 | 1 920 000.0 | 168.8 t (0.0088 %) |
    | perch | 720 004.4 | 720 000.0 | 4.4 t (0.0006 %) |

    cod_west and cod_east produce **not one gonad-derived egg** in years 2–8; flounder and perch
    produce a rounding error. The intervention multiplied zero by ten and got zero — `R = 1.00`
    exactly, on every arm. Confirmed independently by the arithmetic: cod_west's entire bioen egg
    output is `0.087 × 1.250e13`, and **0.0870 is precisely the Shepherd factor at SSB = 50 000**
    with `ssb_half = 15 000`, `β = 1.9520` — i.e. 100 % year-1 bootstrap passed through the curve.
    And by a third, independent field: the seeding-event count. Seeding fires only when SSB == 0,
    and on bioen **all nine species seed all 24 steps of year 1**, while on baseline herring/sprat/
    smelt/stickleback/cod_east seed only 16/16/15/18/21 — baseline stocks acquire a spawning stock
    partway through year 1; bioen stocks never do.

    **The knob is potent — it simply has nothing to turn.** Built-in positive control: where a
    spawning stock exists the intervention delivered 7.6–17.4× and moved biomass hard (sprat
    239 k → 1 611 k, *above* its own baseline; smelt 417 k → 861 k; stickleback 58 k → 154 k). So
    "the wrapper is broken" is excluded by the same run that returns `R = 1.00` elsewhere.

    **Herring is the one genuine partial engagement, and the script's binary mis-sorted it.**
    `collapsed = abundance ≤ 0` classified herring "not part of the test" because it holds a
    remnant, but §9 lists it among the collapsing stocks and it is the only one with a real
    spawning stock that is neither zero nor healthy: 1 355 233 t against baseline's 71 446 734 t,
    a **52.7× gap**. There `R = 2.40` — below the pre-registered bar, so still formally
    inconclusive — yet biomass moved **1.2 → 2 026.7 t**. Recruitment is causally potent for
    herring and still leaves it ~1 200× below baseline, so it is not sufficient on its own either.

    **What this establishes, stated positively.** The earlier 11–48 % egg ratios were **right** —
    all nine reproduce to two decimals (cod_west 0.478 vs 0.48, herring 0.198 vs 0.20, sprat 0.789
    vs 0.79, flounder 0.110 vs 0.11, perch 0.124 vs 0.12, pikeperch 0.455 vs 0.45, smelt 1.115 vs
    1.11, stickleback 1.129 vs 1.13, cod_east 0.171 vs 0.17), which also confirms the recording
    wrapper is inert. It was the *interpretation* that was wrong, exactly as with the growth
    account: a shortfall measured against baseline is not evidence of a causal pathway when the
    pathway carries **zero flux**. A hypothesis about gonad-derived egg production being too weak
    cannot be tested where that production is identically zero, and cannot have *caused* a collapse
    that had already removed every spawner. **Recruitment is downstream** of whatever eliminates
    the pre-maturity cohort. Do not write "refuted" — write ill-posed, with the causal direction
    now fixed.

    **The next test, named not run.** Growth is refuted by intervention; recruitment is downstream.
    What remains is Task 13's measured predation wipeout. The test: suppress predation on the
    juvenile stages of one collapsing stock and ask whether **SSB becomes non-zero** — SSB, not
    biomass, is the instrument, because SSB is the quantity shown here to be identically zero.

  - > ## 🚨 THE PREDATION VERDICT BELOW IS VOID — the `accALL` arm never removed predation
    >
    > Found 2026-09-13 by an adversarial trace, verified in source before this note was written.
    > **GreySeal (sp15) has no predator COLUMN in `predation-accessibility.csv`** — the header runs
    > `cod_west … Benthos, Cormorant` and stops. `AccessibilityMatrix.resolve_name("GreySeal")`
    > returns `None`, so `pred_access_idx == -1` for every seal school. And the production kernel
    > (`mortality.py:1172-1180`) reads:
    >
    > ```
    > access_coeff = 1.0
    > if has_access:
    >     if use_stage_access:
    >         p_acc = pred_access_idx[p_idx]; q_acc = prey_access_idx[q_idx]
    >         if p_acc >= 0 and q_acc >= 0:        # <-- -1 SKIPS THE WHOLE BLOCK
    >             ...
    >             if access_coeff <= 0: continue   # <-- including this test
    > ```
    >
    > **A `-1` does not mean "inaccessible". It means the default `access_coeff = 1.0` survives —
    > FULL accessibility.** So in every arm below, including `accALL`, GreySeal ate cod_west at
    > coefficient **1.0**, twenty times the 0.05 that every *listed* predator was capped at, and
    > zeroing cod_west's prey row could not touch it because there is no column to zero.
    >
    > GreySeal's prey window seals the case: ratio 3–12 on 110 cm and 170 cm bodies gives
    > **9.2–36.7 cm** and **14.2–56.7 cm** — covering the entire 10–20 cm band where cod_west
    > disappears, and on past the 38 cm maturity length.
    >
    > Consequences: (a) "predation does not prevent cod_west from maturing" is **unsupported** — the
    > intervention removed four predators at 0.05 and left the biggest one at 1.0; (b) the
    > "killer not yet enumerated" of the attrition correction is very probably **the seal**; and
    > (c) because every other cod_west predator IS a column, GreySeal is the only species with
    > `pred_access_idx == -1`, so **any cod_west predation death in the `accALL` arm is GreySeal by
    > construction** — one instrumented run settles it.
    >
    > This is also a **latent engine/config defect independent of C3**: a background predator
    > declared in the config but absent from the accessibility matrix is silently granted full
    > accessibility rather than none, and nothing warns. Everything below is retained as the record
    > of what was run and concluded; read it knowing the arms were not what they claimed.
    >
    > ### ✅ RESOLVED — SEALGATE, same day. The ceiling was the seal.
    >
    > `scripts/c3_sealgate_intervention.py`, pre-registered at `a21eeac` before running. The
    > intervention adds the missing GreySeal COLUMN and sets **exactly one cell**: `[cod_west prey,
    > GreySeal predator]`. Nothing else moves.
    >
    > | arm | GreySeal column | cod_west real SSB | **max occupied size bin** |
    > |---|---|---:|---:|
    > | baseline (bioen off) | absent | 129 265.5 | 110 cm |
    > | bioen | absent (⇒ coeff 1.0) | 0.0 | 15 cm |
    > | **sham** | **1.0 for every prey** | **0.0** | **15 cm** |
    > | **treat** | 1.0 except cod_west = **0.0** | 1.0 | **75 cm** |
    >
    > **The sham is the load-bearing control and it passed exactly.** Writing 1.0 into a column that
    > did not exist reproduced `bioen` to the digit on every species — sprat 239 248.8, pikeperch
    > 285 387.4, smelt 416 821.5, stickleback 57 556.1, herring 1.2 — which is only possible if the
    > `-1` path already yields 1.0. The defect is not inferred; it is demonstrated by a no-op that
    > is bit-identical to the production behaviour it replaces.
    >
    > **With the seal unable to eat cod_west, the size ceiling goes 15 cm → 75 cm** — straight past
    > the 38 cm maturity length it had never once reached in any previous arm. The "survival edge"
    > of the attrition correction, the "killer not yet enumerated", was GreySeal eating cod_west at
    > an accessibility of 1.0 that no one ever wrote down.
    >
    > **What this does NOT show, stated plainly.** cod_west is not restored. Real SSB reaches 1.0 t
    > against the control's 129 265.5 t, and final biomass is still 0.0 t (below the 0.5 yr output
    > cutoff) though abundance is now non-zero. **The verdict fired on the size criterion, not the
    > SSB criterion** — and that exposes a weakness in my own pre-registration: the floor was
    > disjunctive (`SSB > 1 %` **OR** `size ≥ 38 cm`), so an OR makes a floor far weaker than it
    > looks. SSB missed its floor by four orders of magnitude. Removing the seal is **necessary to
    > let cod_west reach maturity at all, and not sufficient to restore the stock.**
    >
    > **Scope, untested but strongly implied.** Only cod_west's cell was changed. flounder, perch
    > and cod_east remain extinct in `treat` — and they are exposed to the identical defect, since
    > the seal reaches every prey row at 1.0. A realistic GreySeal column (the Cormorant's 0.05
    > would be the obvious comparator) applied to all prey is the next test, and it is a config
    > change, not a code change.
    >
    > **The headline stands: a large part of the C3 "bioenergetics collapse" is a CONFIG/ENGINE
    > DEFECT** — one missing matrix column silently promoting a top predator to twenty times the
    > accessibility of every predator that was written down.
    >
    > ### COLUMN REPAIR — the seal caps SIZE, not NUMBERS (`6259abb`)
    >
    > SEALGATE changed one cell. `scripts/c3_seal_column_repair.py` repairs the whole column:
    > `realistic` copies **Cormorant's column verbatim** (cod_west 0.05, herring 0.15, sprat 0.15,
    > flounder 0.1, perch 0.6, pikeperch 0.4, smelt 0.25, stickleback 0.15, cod_east 0.05,
    > **resources 0**) as the only in-config example of what a background predator's accessibility
    > should look like; `sealfree` zeroes it entirely as an upper bound.
    >
    > **Max occupied size bin (cm) — the defect is large and real:**
    >
    > | species | baseline | bioen | realistic | sealfree |
    > |---|---:|---:|---:|---:|
    > | cod_west | 110 | **15** | **75** | 75 |
    > | cod_east | 110 | **15** | **75** | 75 |
    > | flounder | 40 | **20** | **40** | 40 |
    > | perch | 45 | **15** | 30 | 35 |
    > | pikeperch | 90 | 40 | 45 | 45 |
    >
    > Both cods go 15 → 75 cm, **flounder is fully restored to its baseline 40 cm**, perch 15 → 30.
    > `realistic` ≈ `sealfree` throughout, so the work is done by going from the unwritten **1.0**
    > down to a sane coefficient — not by removing the seal altogether.
    >
    > **But 0 of 4 collapsed stocks recover on the pre-registered criterion** (real SSB > 1 % of its
    > own baseline real SSB), on either arm. Real SSB does rise — cod_west 0.0 → 1.7 t, flounder
    > 168.8 → 1 654.9 t, perch 4.4 → 377.0 t, cod_east 0.0 → 123.5 t, factors of 10–100× — and still
    > sits **three to four orders of magnitude below the floor** (1 292.7 / 31 125.2 / 47 254.3 /
    > 76 129.7 t). Final biomass stays 0.0 t for all four.
    >
    > **So the missing column is a real, material defect that governs SIZE STRUCTURE, and it is NOT
    > the collapse mechanism.** The collapse is an **abundance** problem, not a size problem: with
    > the seal sane, these stocks grow to normal adult lengths and there are still almost none of
    > them. That splits the remaining question cleanly in two, and only the second is still open.
    >
    > **Caveat bounding the SSB half.** With `seeding.year.max = 1` exactly ONE cohort ever exists,
    > so recovery needs that cohort's offspring to mature inside the remaining window — cod_west
    > matures at ~2.6 yr, so its progeny are only ~1–2 generations deep by year 8. The **size**
    > result is within-cohort and robust to this; the **SSB** result may be window-limited and must
    > not be read as "the stock cannot recover". The clean follow-up is the repaired matrix at the
    > production seeding policy over a longer horizon — which is also what re-running the Stage-1
    > verdict would require.
    >
    > ### ⚖️ RESTAGE AT PRODUCTION SEEDING, 50 YR — **THE STAGE-1 VERDICT STANDS** (`12e52e1`)
    >
    > `scripts/c3_repaired_matrix_restage.py`, pre-registered before running. Removes the
    > single-cohort caveat entirely: **production seeding** (no `seeding.year.max` override — engine
    > default is per-species `lifespan`, cod_west 20 yr) at the **certifying 50-yr horizon**, so the
    > final decade sits 21–46 years past every assessed stock's seeding-window closure. Same metric
    > as Stage 1: final-decade mean biomass.
    >
    > | species | baseline | bioen | repaired | floor (1 %) |
    > |---|---:|---:|---:|---:|
    > | cod_west | 12 335.9 | **0.0** | **0.0** | 123.4 |
    > | herring | 2 523 427.1 | **0.0** | **0.0** | 25 234.3 |
    > | flounder | 32 401.2 | **0.0** | **0.0** | 324.0 |
    > | cod_east | 66 446.1 | **0.0** | **0.0** | 664.5 |
    > | perch | 42 554.4 | 0.0 | 0.0 | 425.5 |
    > | sprat | 1 035 926.7 | 318 726.4 | 313 900.6 | 10 359.3 |
    > | pikeperch | 1 375 582.1 | 130 829.5 | 125 734.9 | 13 755.8 |
    > | smelt | 672 269.5 | 320 559.8 | 285 180.0 | 6 722.7 |
    > | stickleback | 84 954.2 | 69 952.3 | 68 277.8 | 849.5 |
    >
    > **E1 passed — the `bioen` arm reproduces the published Stage-1 collapse exactly**, all four
    > assessed stocks at 0.0 final-decade mean. That is what makes these rows comparable to the
    > Stage-1 table rather than merely similar to it. E2 and E3 passed too.
    >
    > **0 of 4 recover. The missing GreySeal column is NOT what collapses these stocks.** Repairing
    > it moves nothing at this scale — `repaired` is if anything marginally *lower* than `bioen` for
    > the survivors (food-web rebalancing, all well inside the noise of a single seed).
    >
    > **So the defect and the collapse are two separate things, and both conclusions hold:**
    > - The missing column is a **real, material defect** that governs **size structure** — cod_west
    >   and cod_east 15 → 75 cm, flounder fully restored to its baseline 40 cm. Worth fixing on its
    >   own merits, and now recorded in CLAUDE.md.
    > - **C3's headline negative is a genuine bioenergetics result, not a config artifact.** It
    >   survives direct testing against the defect that looked most likely to explain it.
    >
    > **The open question is now precisely one thing: what caps ABUNDANCE.** Not growth (intake is
    >
    > ### 🎯 ANSWERED — **THE BIRTH SIDE CAPS ABUNDANCE** (`c3_abundance_balance.py`)
    >
    > Abundance is an accounting identity, `N(t+1) = N(t) + births − deaths`, so this measured BOTH
    > sides on the same run rather than measuring one and inferring a mechanism — the error that
    > failed three times earlier in this investigation. Deaths from `step_observer` (fires after
    > mortality, before `compact()`, so `state.n_dead` holds that step's deaths across all 8 causes
    > with zeroed schools still present) as **counts, never rates**; births from
    > `regulate_recruitment`. Both arms on the **repaired** matrix, 50 yr, production seeding.
    >
    > | stock | total per-capita mortality | **births** | fewer eggs |
    > |---|---:|---:|---:|
    > | cod_west | 1.00× | **0.050×** | **20×** |
    > | cod_east | 1.00× | **0.061×** | 16× |
    > | herring | 1.01× | **0.061×** | 16× |
    > | flounder | 1.00× | **0.042×** | **24×** |
    > | **sprat — SURVIVES** | 1.00× | **0.797×** | **1.3×** |
    >
    > **Mortality is identical.** Not similar — identical, 1.00–1.01× on every collapsed stock.
    > `ADDITIONAL` carries 99.4–100 % of all deaths in BOTH arms at the same per-capita rate: it is
    > the larval mortality the two configs share (`mortality.additional.larva.rate`, applied once per
    > egg cohort). `STARVATION` is 3.4–7.2× higher under bioen but carries 0.000–0.004 of deaths —
    > real, and immaterial. `PREDATION` 0.6–1.5×, also immaterial in share.
    >
    > **Births are 16–24× lower.** And the survivor contrast is the clean separator the whole
    > investigation has been missing: sprat, the one assessed stock that survives, is at **0.797×** —
    > barely down — while every collapsed stock sits at 4–6 %.
    >
    > **This VINDICATES the original §9 recruitment intuition and does NOT contradict the
    > ill-posed verdict** — the distinction is scale, and it matters:
    > - The earlier recruitment test ran at `seeding.year.max = 1`, where cod_west/cod_east real SSB
    >   is **exactly 0.0 t**. The pathway carried **zero flux**, so a 10× egg boost multiplied zero by
    >   ten. That test was ill-posed *because of its stress condition*, and remains correctly labelled.
    > - At **production seeding** spawners exist for `lifespan` years, the pathway carries real flux,
    >   and the deficit is measurable: 16–24×.
    >
    > So the original candidate was right, and the test first run against it was posed at a scale
    > where it could not be tested. Worth recording as its own lesson: **an intervention that returns
    > a null at one scale has not tested the hypothesis at another.**
    >
    > **Honest gap:** the docstring pre-registered an E4 (balance-closure check — that the abundance
    > change tracks births minus deaths on the baseline arm) which was **not implemented**; E1–E3 ran
    > and passed. The mortality result does not depend on it (the 1.00× ratio is a like-for-like
    > comparison computed identically on both arms), but the accounting identity itself is asserted
    > rather than verified here.
    >
    > **Next, and it is a decomposition rather than a search:** is the egg deficit fewer SPAWNERS or
    > fewer eggs PER spawner? Under bioen eggs come from gonad energy (`rho`, `e_net`); under classic
    > from a prescribed `fecundity × SSB`. Measure final-decade SSB and eggs/SSB per arm — one
    > instrumented run settles which half of the product is short, and `rho` is the parameter that
    > would follow.
    >
    > ### ➡️ DECOMPOSED — **fewer SPAWNERS, not fewer eggs per spawner** (`c3_gonad_flush_test.py`)
    >
    > A candidate mechanism was traced and then **refuted by measurement**. The bioen starvation
    > substep does compare a PER-SCHOOL deficit against a PER-FISH gonad
    > (`mortality.py:1366-1391`; self-documented at `bioen_starvation.py:58`), zeroing the gonad
    > whenever it fires — and a per-step flush predicts an egg-deficit ceiling of exactly
    > `n_dt/sum(season)` = **24.00×**, against measured deficits of 23.8 / 20.0 / 16.4 / 16.4×.
    > A near-perfect fit. **It is nonetheless not the mechanism**, on two independent counts:
    >
    > | species | frac(e_net < 0) | frac(gonad == 0) | **eggs per unit SSB** | SSB ratio |
    > |---|---:|---:|---:|---:|
    > | cod_west | **0.0011** | 0.248 | **1.244** | 1.143 |
    > | cod_east | **0.0004** | 0.329 | **1.072** | 0.095 |
    > | herring | 0.1157 | 0.169 | **1.025** | 0.188 |
    > | flounder | 0.0295 | 0.159 | **1.159** | 0.326 |
    > | sprat (survives) | 0.1138 | 0.068 | 0.534 | 0.290 |
    >
    > 1. **The trigger is far too rare.** The flush requires `e_net < 0`, which for cod_west and
    >    cod_east happens on **0.04–0.11 %** of mature-school steps — it cannot produce 25–33 %
    >    zero gonads. And herring (0.1157) and sprat (0.1138) trigger at an *identical* rate while
    >    one collapses and the other survives, so the trigger does not separate the groups at all.
    > 2. **Eggs per spawner are NORMAL — 1.02–1.24× baseline.** A gonad flush would appear here as a
    >    16–24× shortfall in precisely this quantity. It appears as a slight *surplus*.
    >
    > The 24× fit was a coincidence. Recorded rather than quietly dropped, because the fit was
    > compelling and the mechanism genuinely exists in the source — it simply is not the operative
    > one. **That is the fourth time in this investigation that a number matching a prediction turned
    > out not to be the cause.**
    >
    > **What the run establishes cleanly: the egg deficit is entirely a SPAWNER deficit.** Eggs track
    > SSB almost exactly — cod_east SSB 0.095× → eggs 0.101×; herring 0.188× → 0.192×; flounder
    > 0.326× → 0.377×. The reproductive machinery works per unit of spawning biomass; there is simply
    > far less spawning biomass.
    >
    > **My own instrument error, recorded.** E2 was mis-specified: I expected `frac(gonad == 0) ≈ 0`
    > on the baseline arm as a control, but classic growth never uses `gonad_weight` at all, so it is
    > **1.0000 by construction** on every species. The check was uninformative rather than failed —
    > it could not have discriminated anything. The `e_net` and eggs/SSB columns carried the result.
    >
    > **So "what caps abundance" resolves one step further: low SSB with normal per-spawner
    > fecundity.** SSB = Σ(mature abundance × weight), so the remaining question is whether bioen has
    > *fewer* mature fish or *smaller* ones — cod_west reaches 75 cm under the repaired matrix against
    > 110 cm on baseline, so the weight term is live, and maturity is length-based (38 cm), which
    > couples the two. Next is again a measurement, not a search: mature abundance and mean mature
    > weight, per arm, per species.
    >
    > ### ✅ RESOLVED — **A MATURATION BOTTLENECK** (`c3_ssb_decomposition.py`)
    >
    > `SSB = N_total × frac_mature × mean_weight_mature` is an identity, so the three bioen/baseline
    > ratios must multiply back to the SSB ratio. **They do, for every species (E1 passed)** — which
    > is why this step is not another inference.
    >
    > Measured over years 2–6, while all stocks still hold real numbers. (By years 15–19 the four
    > collapsers sit at 1e-11–1e-45 and their decomposition describes the collapse rather than its
    > cause — the window matters, and the saved per-year arrays let it be re-cut without re-running.)
    >
    > | species | m0 (cm) | SSB | **N_total** | **frac_mature** | mean_w | outcome |
    > |---|---:|---:|---:|---:|---:|---|
    > | cod_west | **38** | 0.0000 | **0.790** | **0.0000** | 0.324 | collapse |
    > | cod_east | 22 | 0.0000 | 0.109 | **0.0032** | 0.118 | collapse |
    > | flounder | 22 | 0.0011 | 0.177 | **0.0115** | 0.535 | collapse |
    > | herring | 18 | 0.0356 | 0.077 | 0.5644 | 0.821 | collapse |
    > | **sprat** | **9** | 0.2386 | **0.839** | **0.5725** | 0.497 | **survives** |
    >
    > **cod_west holds 79 % of baseline's fish and essentially ZERO of them mature.** That single row
    > is the answer: the fish are there, and they do not cross the maturity length. sprat likewise
    > carries 84 % of baseline's fish with 57 % of the maturation — and survives on it.
    >
    > **The chain, every link measured rather than inferred:**
    > 1. Mortality is identical between arms (1.00–1.01× per-capita).
    > 2. Births are 16–24× down.
    > 3. That is a **spawner** deficit, not a fecundity one — eggs per unit SSB are 1.02–1.24×.
    > 4. The SSB deficit is **`frac_mature`**, with `N_total` largely preserved.
    > 5. Maturity here is **length-based** (`species.maturity.size`; `species.maturity.age` absent),
    >    and bioen size-at-age is **0.32–0.82×** baseline — so far fewer fish ever cross `m0`.
    > 6. No spawners → no eggs → no recruitment → decay to extinction, mortality normal throughout.
    >
    > **This vindicates the `m0` ordering noticed at the very start of §9** — stickleback (4.5 cm)
    > survives, cod (38) and flounder/cod_east (22) collapse — but now as a *measured mechanism*
    > rather than a nine-point correlation. sprat's `m0 = 9.0 cm` is the smallest of the assessed
    > stocks and is the survivor. pikeperch (`m0 = 40`) remains the standing exception it has been
    > throughout, and is still unexplained.
    >
    > ### 🐟 THE PIKEPERCH EXCEPTION — it is a **beneficiary of the collapse**, not an exception to it
    >
    > pikeperch is **not** exempt from the maturation bottleneck. Its mature fraction is crushed as
    > hard as any collapser: ratio **0.0888**, absolute 0.0028 against a baseline 0.0313. Its SSB
    > ratio (0.0340) is *lower* than herring's (0.0356), and herring dies. So neither `m0`,
    > `frac_mature` nor the SSB ratio explains it.
    >
    > **What the year-by-year trajectory shows** (bioen arm, mature abundance):
    >
    > | yr | cod_west N | cod_east N | pikeperch N_total | pikeperch N_mature |
    > |---:|---:|---:|---:|---:|
    > | 0–2 | 3.5e8 → 3.1e8 | 1.8e9 → 1.4e9 | 1.4e12 → 9.4e11 | **0** |
    > | 3 | 6.7e7 | 1.7e7 | 3.6e11 | 5e-4 |
    > | 4 | 1.3e6 | 6.7e4 | 1.2e11 | **1.3e8** |
    > | 5 | **33** | **430** | **6.0e10** ← floor | 9.5e8 |
    > | 6 | 0.2 | 0.4 | 2.2e11 | **3.7e9** |
    > | 8 | ~0 | ~0 | 8.7e11 | 6.8e9 |
    > | 10 | ~0 | ~0 | **2.8e12** | 2.6e9 |
    >
    > **pikeperch holds ZERO mature fish until year 3, then takes off exactly as cod_west and
    > cod_east go extinct** — and its total abundance rebounds **22×** from a year-5 floor of 6.0e10.
    > Both cods are listed predators of pikeperch (`cod_west 0.1`, `cod_east 0.05`).
    >
    > **The decisive quantity is whether the JUVENILE POOL survives the crash window.** Floors during
    > years 3–5: pikeperch **6.0e10**; flounder 2.6e7 (then 5e-2 by yr 8); herring 4.0e10 → 9.1e8;
    > cod_east **430**; cod_west **33**. pikeperch is the only collapsing-profile stock whose pool
    > never drops below a recoverable level, and it banks those juveniles cheaply:
    > `mortality.additional.rate` = **0.0137 yr⁻¹**, **92× lower than cod_west's 1.2546**, so its pool
    > drains ~1.4 % a year while it waits for its predators to disappear.
    >
    > **No single factor explains it, and it would be wrong to claim one.** M alone does not separate
    > the groups — flounder has the *lowest* M (0.006) and still dies. Pool size alone does not —
    > herring starts largest (1.1e13) and still dies, draining at M = 2.2472, the highest of the nine.
    > What pikeperch uniquely combines is **a large juvenile pool × a very low drain rate × predators
    > that die first**.
    >
    > > **⚠️ TESTED 2026-09-14 — THE RELEASE STORY IS REFUTED.**
    > > `scripts/c3_pikeperch_release_test.py` zeroed the `cod_west` and `cod_east` cells of
    > > pikeperch's prey row (E1: exactly those two cells differ, nothing else). Over **years 0–3,
    > > while the cods are alive**, `nocod/ref` pikeperch `N_total` = **0.998**. Removing cod
    > > predation entirely changes nothing. Specificity check passed (years 8–19 ratio 0.897, cods
    > > extinct on both arms; E2 passed).
    > >
    > > **The cods could never have mattered, and it is obvious in hindsight:** pikeperch numbers
    > > 1.39e12 at year 0 against the two cod stocks' combined 2.18e9 — **630× fewer predators than
    > > prey**, at accessibility 0.10/0.05. They cannot make a dent. The year-4 takeoff is
    > > **growth-timed** (vBGF `t(m0)` = 2.97 yr) and the cod-extinction coincidence was exactly a
    > > coincidence. **Fifth time in this investigation that a compelling correlation was not causal
    > > — and the first where the named intervention was actually run instead of the correlation
    > > being left to stand.**
    > >
    > > **What actually explains pikeperch is arithmetic:** mature stock = pool × mature fraction.
    > >
    > > | species | pool yr0 | frac_mature | → mature | fate |
    > > |---|---:|---:|---:|---|
    > > | pikeperch | **1.39e12** | 0.0028 | **3.9e9** | survives |
    > > | sprat | 1.27e12 | 0.4486 | 5.7e11 | survives |
    > > | flounder | 2.37e9 | 0.0039 | 9.2e6 | dies |
    > > | cod_east | 1.83e9 | 0.0013 | 2.4e6 | dies |
    > > | cod_west | 3.46e8 | 0.0000 | **0** | dies |
    > >
    > > pikeperch suffers the *same* maturation bottleneck — but a tiny fraction of an enormous pool
    > > is still ~4e9 spawners, and it drains that pool at only 0.0137 yr⁻¹ (92× below cod_west).
    > > cod_west has a zero fraction of a pool three orders of magnitude smaller. **Same mechanism,
    > > different starting stock.** No predation release required, and none present.
    >
    > **Status (superseded above): the timing correlation is strong; the causal claim is NOT yet
    > tested by intervention.** Given how often that distinction has mattered here, it is labelled rather than
    > asserted. The test: suppress cod predation on pikeperch from year 0 and ask whether its mature
    > stock rises *earlier* than year 4; or hold the cods alive and ask whether pikeperch still
    > recovers.
    >
    > **Also worth recording — perch rebounds too.** Its pool bottoms at 7.1e-9 in year 9 and returns
    > to 1.3e9 by year 12, despite reading 0.0 in the 50-yr final decade. So the collapse is not
    > uniformly monotonic, and a final-decade mean can hide a mid-run recovery that later fails.
    >
    > **Correcting my own earlier reading.** I wrote that realized growth was "entirely healthy" on
    > the strength of `dw/w` at 12–33 % per step with intake at 90–100 % of cap. That was **too
    > strong**: per-step weight *gain* is vigorous, but size-*at-age* is 0.32–0.82× baseline, and for
    > a length-based maturity threshold it is size-at-age that decides. Both measurements are correct;
    > the first does not support the conclusion I drew from it.
    >
    > **What this means for C3.** The Stage-1 negative stands and is now *explained*: bioen as
    > parameterised cannot carry these stocks past their maturity lengths. The lever is the growth
    > trajectory against `m0` — either the bioen parameters that set size-at-age, or `m0` itself — and
    > that is a calibration question with a named target, not an open search.
    >
    > > **⚠️ THE m0 HALF OF THAT LEVER IS REFUTED — TESTED 2026-09-14.**
    > > `scripts/c3_m0_lever_test.py` lowered `species.maturity.m0` on the four collapsing stocks
    > > only, as a dose ladder (×1.00 / ×0.50 / ×0.25), leaving the four survivors untouched as an
    > > internal control. **E1** read the thresholds back per arm, **E2** confirmed the survivors
    > > unmoved (sprat 0.357/0.353/0.358; pikeperch 0.0020/0.0020/0.0019), **E3** confirmed the knob
    > > engaged hard — cod_east `frac_mature` **0.0009 → 0.1960 → 0.6700**, a 744× rise.
    > >
    > > **0 of 4 recover. Final-decade biomass is 0.0 t at every dose.** Crossing `m0` is
    > > **NECESSARY BUT NOT SUFFICIENT** — the last link of the chain is not causal on its own.
    > >
    > > **Worse, the response is non-monotone, and for the cods lowering m0 is actively HARMFUL:**
    > >
    > > | arm | cod_west yr3 | cod_west yr5 | flounder yr11 |
    > > |---|---:|---:|---:|
    > > | m0 ×1.00 | 6.7e7 | 3.4e1 | 3.8e-16 |
    > > | m0 ×0.50 | 7.1e5 | 5.7e-1 | 1.3e2 |
    > > | m0 ×0.25 | **1.8e3** | **3.0e-2** | **2.6e4** |
    > >
    > > flounder decays 20 orders of magnitude more slowly; **both cods collapse FASTER.**
    > >
    > > **The mechanism is in the source and is a genuine model trade-off**: `rho` is 0 for immature
    > > fish and positive once mature (`energy_budget.py:313`), and `dw = (1−rho)·E_net/N` against
    > > `dg = rho·E_net/N`. **Maturity taxes somatic growth.** Lowering `m0` makes fish mature
    > > smaller and then grow more slowly — which is exactly the wrong medicine for a stock whose
    > > problem is already size-at-age.
    > >
    > > **So the Stage-2 lever is size-at-age, NOT m0.** Growth raises maturation *and* weight
    > > together; lowering the threshold buys maturation by taxing the growth that was short in the
    > > first place. The "or `m0` itself" half of the sentence above is dead.
    > >
    > > **And one link still unexplained:** cod_east reaches **67 % maturation** at ×0.25 and still
    > > reads 0.0 t. With eggs tracking SSB at 1.02–1.24×, more spawners should mean more eggs. That
    > > they do not translate into biomass points at egg→recruit survival, which no test here has
    > > yet isolated.
    > 90–100 % of cap, `m_share` below target), not size (the seal explained that and fixing it
    > changes no biomass), not recruitment as originally framed (there were never spawners to begin
    > with), and not the listed predators. Something removes the numbers while leaving the energetics
    > and — once the seal is sane — the growth trajectory intact.
    >
    > **CERTIFIED AT 5 SEEDS** (`c3_repaired_matrix_restage.py`, Stage 1's own
    > `(42, 123, 7, 999, 2024)` — parsed from `baltic_c3_bioen_ab.py:121` and asserted to match, so a
    > drifted constant cannot quietly invalidate the comparison). 15 engine runs. **All four
    > engagement checks pass, E1 on every seed individually.**
    >
    > | species | baseline | bioen | repaired | floor (1 %) | seeds clearing floor |
    > |---|---:|---:|---:|---:|---:|
    > | cod_west | 12 810.8 | **0.0** | **0.0** | 128.1 | **0/5** |
    > | cod_east | 65 251.2 | **0.0** | **0.0** | 652.5 | **0/5** |
    > | herring | 2 539 645.2 | **0.0** | **0.0** | 25 396.5 | **0/5** |
    > | flounder | 33 063.4 | **0.0** | **0.0** | 330.6 | **0/5** |
    > | sprat | 1 024 324.0 | 309 994.2 | 309 636.5 | 10 243.2 | — |
    > | pikeperch | 1 400 081.1 | 128 120.2 | 128 024.1 | 14 000.8 | — |
    >
    > Across-seed means; every one of the four reads **exactly 0.0 on all five seeds** in both the
    > `bioen` and `repaired` arms — deterministic extinction, not noisy near-collapse, exactly the
    > character Stage 1 reported. The per-seed tables are in the script output and show no seed-level
    > variation for an average to hide.
    >
    > **0 of 4 recover, 0/5 seeds each. The single-seed caveat is now discharged:** this null is held
    > to the same 5-seed standard as the claim it tests, so it is no longer weaker evidence than a
    > positive would have been.

  - **PREDATION TESTED BY INTERVENTION 2026-09-13 — and the answer is a SIZE CEILING.**
    `scripts/c3_predation_intervention.py`, pre-registered at `cd7eea5` before the run. Config-only
    and surgical: `predation.accessibility.stage.structure = age` and the CSV takes `"name < T"`
    age labels (`accessibility.py:_parse_label`), with prey ROWS and predator COLUMNS parsed
    independently — so cod_west's PREY row is split and scaled while its predator COLUMN is left
    alone. This changes what eats cod_west, never what cod_west eats, and touches no growth,
    bioenergetics or reproduction parameter. cod_west has only four predators (itself, pikeperch,
    smelt, Cormorant), all at accessibility 0.05.

    | arm | juv accessibility | cod_west final t | real SSB | juv biomass 1–3 yr | **max size bin** |
    |---|---|---:|---:|---:|---:|
    | baseline (bioen off) | 0.05 | 1 454.5 | **129 265.5** | 4 555 | **110 cm** |
    | bioen | 0.05 | 0.0 ✗ | 0.0 | 11.77 | 15 cm |
    | acc50 | 0.025 | 0.0 ✗ | 0.0 | 16.44 | 15 cm |
    | acc10 | 0.005 | 0.0 ✗ | 0.0 | 29.24 | 20 cm |
    | acc00 | 0.0 (juveniles) | 0.0 ✗ | 0.0 | 36.09 | 15 cm |
    | **accALL** | **0.0 at EVERY age** | **0.0 ✗** | **0.0** | 36.09 | **15 cm** |

    Size bins are 5 cm wide (`output.distrib.bysize` 0–120 step 5), so "15 cm" means the largest
    occupied bin is [15, 20). Maturity needs **38 cm** (`species.maturity.m0.sp0`, length-only —
    `species.maturity.age.sp0` is absent).

    **Predation is real but is NOT the binding constraint.** The dose ladder is clean and monotone
    — juvenile biomass 11.77 → 16.44 → 29.24 → 36.09 as accessibility falls, a **3.07×** gain under
    full immunity — so predation genuinely does kill cod_west juveniles, and this knob engages in a
    way the recruitment knob never could. But **the size ceiling does not move**: at `accALL`, where
    cod_west is inedible to every predator at every age, it still tops out at 15–20 cm against the
    38 cm it needs. It cannot mature, so SSB is structurally zero, so there are no eggs. Full
    immunity also leaves juvenile biomass **126× below** the bioen-off control (36.09 vs 4 555).

    > ### ⚠️ CORRECTION, same day — the paragraph that stood here was WRONG
    >
    > It read: *"The binding constraint is realized in-engine growth: a 15–20 cm ceiling."* The
    > **observation** (max occupied bin 15–20 cm under bioen, 110 cm bioen-off) is correct and
    > stands. The **causal attribution to growth was refuted within the hour** by
    > `scripts/c3_growth_ceiling_diagnosis.py` (pre-registered `d18a8fb`), which measured the
    > energy budget by size class and found growth entirely healthy at the ceiling:
    >
    > | length cm | ing/cap | m_share | dw/w per step | mean w g |
    > |---|---:|---:|---:|---:|
    > | 0–5 | 0.898 | 0.155 | 0.325 | 0.15 |
    > | 5–10 | 0.903 | 0.201 | 0.200 | 3.81 |
    > | 10–15 | 0.924 | 0.184 | 0.158 | 17.27 |
    > | **15–20** | **0.956** | **0.135** | **0.138** | 38.73 |
    >
    > (predation-immune `accALL` arm; the plain `bioen` arm is within a percent of it). cod_west
    > eats **90–100 % of its allometric cap at every size**, `m_share` is **0.13–0.20 — *below* the
    > fit's 0.30 target and not rising with size**, and specific growth in the top occupied bin is
    > **13.8 % of body weight per step**. Nothing is stalling. Fish grow well and then disappear.
    >
    > **So the 15–20 cm ceiling is a SURVIVAL EDGE, not a growth ceiling** — case (2) of that
    > script's pre-registered reading, ATTRITION rather than STALL, returned identically on both
    > arms.
    >
    > **The error is the same class, for the third time in this investigation: inferring a
    > mechanism from a distribution without measuring the mechanism.** The size distribution showed
    > a ceiling; I attributed it to growth; growth was fine. Exactly as the 11–48 % egg ratios were
    > real while the recruitment story built on them was wrong. A distribution tells you *where*
    > things stop, never *why*.

    **What survives from that paragraph.** The distinction it drew is still worth keeping, because
    it now applies to the *observation* rather than to a cause: the 15–20 cm vs 110 cm gap is real,
    is a factor of ~7 in length, and is not the already-refuted offline-fit growth account — that
    one was about the fit's curve and died when fixing the curve changed nothing. But the gap is
    produced by mortality, not by slow growth.

    **What now reconciles the earlier results.** SSB is exactly 0 because nothing survives to
    38 cm — not because nothing grows to 38 cm. Recruitment was ill-posed because there were never
    any spawners. Fixing the offline fit changed nothing because the fit was never what was broken,
    and this run confirms why from the other side: realized intake is already at 90–100 % of cap, so
    the fit's curve was never the limiting input.

    **The open question is now sharp and small: what kills cod_west between 10 and 20 cm?** It is
    not predation (removed entirely in `accALL`, ceiling unmoved), not starvation (`e_net` is
    positive and `m_share` is 0.13–0.20), not the offline fit, and not fishing —
    `fisheries.rate.base.fsh0` is **0.039 yr⁻¹**, negligible, though note
    `fisheries.selectivity.type.fsh0 = 0` is **knife-edge by AGE at 2.0 yr**, so slow growth buys no
    protection from whatever fishing there is. Larval additional mortality is enormous
    (`mortality.additional.larva.rate.sp0` = 243.76 yr⁻¹) but `larva_mortality`
    (`natural.py:103`) applies it **only to `is_egg` schools, once per cohort**, so it cannot reach
    a 15 cm fish. That leaves `mortality.additional.rate.sp0` = 1.2546 yr⁻¹ ≈ 5.2 % per step, which
    is far too small to explain the observed drop between size classes. **Something not yet
    enumerated is removing them, and the next measurement is deaths BY CAUSE for cod_west resolved
    by size class — not rates, which are summed per-step and cannot be exponentiated (CLAUDE.md).**

    **Verdict discipline.** The pre-registered rule returns `INCONCLUSIVE (window/growth-limited)`
    for the predation question, and that is recorded as-is — the test could not reach the question
    it asked, because the cohort never survives to maturity under any predation regime. What *is*
    established positively is narrower and stronger than a null: predation does not prevent cod_west
    from maturing, because removing it entirely leaves the ceiling unchanged. (The branch label
    "window/growth-limited" is itself now known to be a misnomer — see the correction above: the
    ceiling is a survival edge, not a growth limit. The branch fired on the right *evidence*, max
    length below 0.9 × m0, and drew the wrong *inference* from it.)

    **A flaw in this pre-registration that did not bite, recorded anyway.** The rule `real SSB > 0`
    → PREDATION CONFIRMED carries **no magnitude floor**, so it would have fired on a biologically
    dead remnant — 0.001 t of spawners would have read as a confirmation. It happened to return
    exactly 0.0, so nothing turned on it, but the next version of this rule needs a floor expressed
    as a fraction of the bioen-off control's SSB.

    **Engagement checks all passed**, and they were worth the trouble — four separate instrument
    defects were caught and fixed before any verdict was read (`71ab3fd`, `4ab46a7`, `a91b7db`),
    each of which would have produced a confident wrong answer:
    - E1 the loaded matrix differs per arm, read back from a *constructed* `EngineConfig` rather
      than from the CSV written — `accALL` resolves to a single all-ages stage with every entry 0.
    - E2 the bioen-off control sustains all nine species.
    - E3 juvenile survivorship differs between arms (11.77 → 36.09), so the knob demonstrably bit.
      E3 was silently NaN on the first two runs because `abundanceByAge` is not produced in-memory
      and `biomass_by_age` returns LONG format, not wide.
    - E4 the size-ceiling check, added *because* an adversarial read flagged that an age-based
      accessibility split need not coincide with a length-based maturity threshold. Without it this
      run would have read as "predation refuted" instead of "the fish never grew".

    **Next, and this one is now well-posed:** find why realized growth stalls at 15–20 cm. Measure
    `e_gross`, `e_maint` and `e_net` for cod_west *by size class* over the run, against the offline
    fit's expectation at the same weights. The specific suspicion worth testing first is that
    maintenance overtakes intake at small size — `e_net → 0` — which would be a hard ceiling of
    exactly this shape rather than a slow-growth effect.

  - **The boost's fitted VALUES were checked against the literature** —
    `docs/validation/juvenile_ingestion_boost_literature_2026-09-13.md` (reproducer:
    `scripts/c3_juvenile_boost_literature_check.py`). A separate question from whether the boost
    cures the collapse, which the intervention above settled in the negative. Findings: the
    **direction is verified** — Morell et al. (2024, *Ecol. Lett.* 27(11),
    [10.1111/ele.70017](https://doi.org/10.1111/ele.70017)) documents higher mass-specific
    ingestion in early life stages as a deliberate Bioen-OSMOSE assumption with a stated rationale
    — but the **1.44–4.64× magnitude is unverified**: no retrievable source gives a larva:adult
    ingestion ratio for any of the nine species, and both Bioen-OSMOSE papers are abstract-only
    through scite, so the published `theta`/`c_rate`/`larvaeThresDt` values could not be read.
    The model's `beta = 0.8` against Kiørboe & Hirst's measured `w^0.75` is **not** what generates
    the fitted values — that predictor has ~1.2× of dynamic range against a 3.24× spread in `j`.
    Two independent anchors do bracket the median (2.21× Wuenschel & Werner 2004; 2.5× Kaufmann
    1990), so **if the boost is ever enabled, a single shared `j` ≈ 2.2–2.5 is defensible where
    nine free per-species values are not.** Nothing shipped is affected — `c3_bioen_arm.json`
    still carries `theta = 1.0`, `c_rate = 0.0`.
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
