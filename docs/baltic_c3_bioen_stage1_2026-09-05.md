# Baltic C3 — bioenergetics activation, Stage 1 results (2026-09-05)

**Verdict: PENDING — run not yet launched at time of writing this scaffold.** This document is
being written incrementally, per the task's own instruction, starting with the sections that do
not depend on the 50-yr run. All run-dependent sections below are marked `[PENDING RUN]` and will
be replaced with numbers pulled directly from `docs/diagnostics/baltic_c3_bioen_report.json` —
none typed by hand.

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

## Run provenance

- 3 arms — `baseline` (production config, unmodified), `bioen` (production + the flat overlay
  `data/baltic/scenarios/c3_bioen/c3_bioen_arm.json`), `bioen_plus2C` (`bioen` +
  `temperature.offset=2.0`) — × 5 house seeds `[42, 123, 7, 999, 2024]` × 50 yr. Fixed by Ruling
  R7 — not reduced (seeds, horizon, or arms) for this run.
- Runtime: `[PENDING RUN]`. Quote this as a **margin, not a point estimate**: the only *measured*
  50-yr number anywhere on this branch before this run is the bioen-**OFF** baseline (Task 3's
  Gate A, ~9–19 min for 5 seeds × 50 yr through the same Numba kernel). No 50-year bioen-**ON**
  measurement existed anywhere before this run — both bioen arms here are the first. The
  load-bearing pre-registered claim (Task 12) was that even at 3× the point estimate the whole
  3-arm run stays under three hours serial; this document reports what was actually measured,
  which now supersedes that estimate for the arms run here.
- Gate C's builder-recomputation leg was run (not skipped with `--no-recompute`) so the on-disk
  temperature forcing file is independently re-verified against the CMEMS cache for this record
  run — Task 12 measured that leg at 35 s, not the "minutes" the module docstring assumes.
- Raw report: `docs/diagnostics/baltic_c3_bioen_report.json` (committed verbatim from the run).

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

`[PENDING RUN]` — every row below is read directly from `gates` in the committed report JSON
once the run completes; nothing here is typed ahead of the run. All `json key` values are under
`docs/diagnostics/baltic_c3_bioen_report.json:gates.*` unless noted.

| gate | scope | result | json key |
|---|---|---|---|
| A — bioen-off inertness, `array_equal` to the master fixture | 5 seeds, baseline arm | `[PENDING RUN]` | `gate_a` |
| B — cross-engine parity of bioen-on (Python vs Java 4.3.3) | Task 9, not re-run here | bioen: PASS; control: REVIEW (Hake, pre-existing) | `c3_gate_b_cross_engine.md`¹ |
| C — temperature load-through, 3-way, per layer + range | `bioen`, `bioen_plus2C` arms | `[PENDING RUN]` | `gate_c` |
| D — frames/layers (24, 2) + structural/parameter asserts | temp file + both bioen configs | `[PENDING RUN]` | `gate_d_frames_layers`, `gate_d_structure` |
| E — zlayer wiring, engine-side | `bioen` arm, seed 42, step 12 | `[PENDING RUN]` | `gate_e` |
| F — thermal instrument (`phi_t(T_p)==1`, argmax ±0.1°C, φT∈(0,1], direction) | `bioen` arm, all 9 sp | `[PENDING RUN]` | `gate_f` |
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

`[PENDING RUN]`. Will be populated from `final_decade_means` in the report JSON: 5-seed mean and
std per species per arm, ratio to `CERTIFIED_MEANS` (`docs/baltic_certification_2026-08-14.md`)
and to `ENVELOPE` (`scripts/baltic_stability_certify.py`), `in_envelope` where defined.

**Criterion (iii) anchor sanity check (requested by the coordinator's review):** the spec text is
ambiguous between anchoring the bounded-displacement criterion on the certified means or on this
run's own baseline arm; the harness implements the certified-means anchor
(`CERTIFIED_MEANS`, a literal snapshot of `docs/baltic_certification_2026-08-14.md`). `[PENDING
RUN]` will report the baseline arm's own `ratio_to_certified` for the 5 assessed stocks — if every
value sits near 1.0, the anchor choice is immaterial to the verdict; if any diverges materially,
both anchors' criterion (iii) results will be reported.

## §5 — Instruments

`[PENDING RUN]`. Per species: realized ration `ē`/`ĝ` and `f` (decision 7, `realized_ration` in
the JSON) — **NaN values will be reported explicitly as undetermined, not silently treated as
passing**, per the decision-rule note above; length-at-age paired RMS % (ages ≥ 1 yr, `bioen` vs
`baseline`, `length_at_age` in the JSON); realized annual ingestion vs the Imax inflation factor
(decision 17, `realized_ingestion`); seeding diagnostics (`seeding_diagnostics_note` — the brief's
documented in-memory fallback: `OsmoseResults.ssb()` raises `FileNotFoundError` in in-memory mode,
so this diagnostic is a skip note, not a number, confirmed in Task 12's development pass).

**Reminder for whoever reads these numbers next:** `biomass()`/`abundance()` exclude young-of-year
via `output.cutoff.age` (0.5 yr for every Baltic species); `*_by_age`/`*_by_size` do not — the
length-at-age instrument above uses the latter family, the final-decade means (§4) use the former;
they are not interchangeable. `biomass/abundance` read as a mean weight is egg-dominated
(~0.0005 g/fish) regardless of growth; `abundance_by_size()` bin occupancy cannot resolve sub-10 cm
growth. None of this task's numbers were read through either trap.

## §6 — The +2 °C arm

`[PENDING RUN]`. `bioen_plus2C` minus `bioen` deltas per species (`bioen_plus2C_minus_bioen` in the
JSON), alongside Gate F's per-species habitat-mean `g_net` shift — reported only, not gated on the
decision rule (spec §3.5). Every species sits below its own `t_opt` at `T̄` (§3), so Gate F's
direction check (`g_plus2 > g_base` for every species) is the expected sign; the magnitude of the
shift is what this section reports.

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

## §8 — What Stage 2 would do (or why C3 closes)

`[PENDING RUN]`. Will state, from the fixed decision rule's verdict on this run's numbers: either
Stage 2 is warranted (bounded recalibration of the bioen parameter set only — spec §4), with the
specific per-species recalibration magnitude implied by `ē/ĝ` and the `r`/`Imax` rescales that
would restore `W∞` and the juvenile rate offline; or C3 is closed by characterization, naming
which criterion failed (and which was undetermined, if any, and why), by how much, for which
species — the same discipline C4 (salinity arms) used to close without adjusting its own
pre-registered rule after seeing the numbers.

## §9 — Follow-ups (not blocking this task; recorded for whoever picks C3 up next)

- **Numba bioen kernel.** The batched Numba mortality kernels are bypassed entirely under bioen
  (5–10× slower than the bioen-off path) — `_apply_starvation_for_school`'s bioen branch and the
  interleaved survivor-rescaling loop are Python-only. A compiled specialisation would matter for
  any future work at a horizon or ensemble size beyond this stage's 50 yr × 5 seeds × 3 arms.
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
