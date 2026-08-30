# Baltic C4 — salinity-gate sensitivity arms (2026-08-30)

**Verdict: ALL GATES PASS.** Builder zero-check: PASS. `movement.salinity.field.constant`
absence: PASS on every arm config. Frame-count assert (24 frames): PASS on every arm field.
Three-way load-through (engine-loaded == on-disk == builder-recomputed offset): PASS for
`zero`, `ds_m1`, `ds_m2`, `ds_m3`. Ramp ordering (negative ΔS ⇒ w′ ≤ w, per wet cell): PASS
for all four non-baseline arms. Zero-arm run bit-identity to baseline: PASS, **0 violations**
across 5 seeds. All-zero (map, frame) guard events: **0 at every arm**, including `ds_m3`
(−3 PSU) — the whole-frame un-gate hazard never fires in this run.

This is a **mechanism-characterization run, not a projection** — see "Not a projection" under
Labels below. Two headlines, in the order they matter:

1. **The July regime-shift chain does not reproduce.** Stickleback and percid responses sit
   at seed-noise level at every lever tested here, a completely different picture from July's
   OFF→ON flip (+94% / −33% / −35%).
2. **cod_west — the spec's own pre-registered "saturated null control" — turns out to be the
   largest responder.** It moves +5.9% / +11.0% / +27.9% (−1 / −2 / −3 PSU), monotone and, at
   −2/−3, far outside its own seed noise. The spec's saturation statement was correct about the
   *baseline* field and wrong as a predictor of behaviour under an offset — see Headline 2 below
   for why.

Spec: `docs/superpowers/specs/2026-08-30-baltic-c4-salinity-sensitivity-arms-design.md`
(binding — decisions 1–6, success criteria). Plan:
`docs/superpowers/plans/2026-08-30-baltic-c4-salinity-arms.md`.

## Run provenance

- 5 arms — `baseline`, `zero` (all machinery engaged, zero offset), `ds_m1` (ΔS=−1 PSU),
  `ds_m2` (ΔS=−2 PSU), `ds_m3` (ΔS=−3 PSU) — × 5 house seeds `[42, 123, 7, 999, 2024]`
  (`simulation.rng.fixed=true`) × 50 yr.
- Branch `c4-salinity-arms`, harness `scripts/baltic_c4_salinity_ab.py`.
- Runtime: ~87 min (25 runs, in line with the plan's "~1.6 h at B2 pace" estimate).
- **This is a local validation run, not a CI gate** — same status as the B2 and C1 A/B
  precedents. Production certification is untouched.
- Raw report: `docs/diagnostics/baltic_c4_salinity_report.json` (copied verbatim from the run
  at `/tmp/c4_salinity_report.json`).

## Delta spec — arms and rationale

`data/baltic/scenarios/c4_salinity_sensitivity.json`. ΔS values are **chosen levers, not
citations** — the schema deliberately separates a `rationale` field (levers) from
`context_citations` (real numbers, quoted below), so a fake citation is never attached to a
chosen delta.

| arm | ΔS (PSU) | rationale |
|---|---:|---|
| `ds_m1` | −1.0 | sub-ramp-width lever: characterizes redistribution transmission where exclusions cannot fire (expected TV≈0.03, exclusions ≈0 on cod_east) |
| `ds_m2` | −2.0 | two-thirds-ramp lever: expected TV≈0.10, exclusions ≤0.23% on cod_east |
| `ds_m3` | −3.0 | full-ramp exclusion-regime lever: baseline cells below 6 PSU reach w=0; approaches the July OFF→ON flip from below; all-zero (map, frame) events reported by the builder |

**Not a projection:** no ensemble generation supplies a citable mean freshening delta (Meier
et al. 2022, doi:10.5194/esd-13-159-2022, Table 8: BalticAPP −0.06, ECOSUPPORT −0.15, CLIMSEA
≈0 g/kg SSS; Sect. 3.2.4: "salinity changes are not robust; i.e. the ensemble spread is larger
than the signal"). Only first-generation, superseded modelling reached larger numbers (Meier
2006, cited in Meier et al. 2022: "decreases of as much as 45%") — a 2006-era extreme, not a
modern estimate. The −1/−2/−3 PSU levers used here are **chosen to span the production ramp
(3–6 PSU)**, not derived from either citation.

## Blocking gates (spec decision 4, all PASS)

| gate | scope | result |
|---|---|---|
| (a) builder zero-check (zero-arm field value-identical NaN-aware to production) | 1 field | **PASS** |
| (b) `movement.salinity.field.constant` absent from every arm config | 4 arm configs | **PASS**¹ |
| (c) three-way load-through (engine-loaded via `_load_salinity_gate` == on-disk == builder-recomputed offset) | `zero`, `ds_m1`, `ds_m2`, `ds_m3` | **PASS** — true for all four |
| (d) ramp ordering per wet cell (negative ΔS ⇒ w′ ≤ w) | `zero`, `ds_m1`, `ds_m2`, `ds_m3` | **PASS** — true for all four |
| (e) frame-count assert (24) on every arm field, harness-side | 4 arm fields | **PASS**² |
| (f) zero-arm run bit-identity to baseline, per seed | 5 seeds | **PASS** — 0 violations |

¹/² Gates (b) and (e) are not recorded keys in the committed `gates` JSON (which carries only
`zero_check`, `load_through`, `ramp_ordering`) — they pass by **non-raising harness assertion**
(`assert_no_salinity_constant` / `assert_arm_frame_count` in `scripts/baltic_c4_salinity_ab.py`,
which raise `ValueError` on failure), and the run completing end-to-end is the evidence they
held. Gate (b) additionally inspects the **same inherited `base_cfg` value** four times by
construction — `arm_overlays` never emits `.constant` for any arm — so this is not four
independent per-arm checks; it is the harness's own poisoned-cfg unit test
(`tests/test_baltic_c4_harness_helpers.py`) that proves the guard actually fires, per Task 3's
own caveat.

## Chain table — final-decade mean vs baseline, all nine species

| species | −1 PSU | −2 PSU | −3 PSU | baseline seed sd |
|---|---:|---:|---:|---:|
| **cod_west** | **+5.87%** | **+10.96%** | **+27.85%** | 5.2% |
| cod_east | +0.90% | +1.68% | −2.33% | 1.3% |
| stickleback | +2.72% | +0.14% | +0.64% | 3.2% |
| perch | −1.14% | −2.72% | +0.96% | 1.7% |
| pikeperch | +1.90% | +1.64% | +1.64% | 2.2% |
| smelt | −0.46% | +0.18% | +0.52% | 1.1% |
| herring | −1.01% | +0.30% | −0.93% | 1.6% |
| sprat | −0.96% | −1.69% | −2.64% | 0.8% |
| flounder | −2.13% | −0.59% | −1.80% | 2.5% |

cod_west is the sole outlier: at −2 PSU it is already ~2.1× its own baseline seed sd, and at
−3 PSU ~5.4× — a real signal, not noise. Every other species stays within roughly 1–2× its own
baseline seed sd at every lever — the largest non-cod_west swings are stickleback +2.72%
(−1 PSU, sd 3.2%) and perch −2.72% (−2 PSU, sd 1.7%, ~1.6× sd) — small in absolute terms but
not uniformly sub-noise; **the lever is cod-specific as wired**, and cod_west dominates the
whole table.

## Instruments — the two gated species' maps

TV distance, mean-Δw (wiring check only — see the framing sentence below), and
newly-excluded-cell fraction, per life stage, per arm. `saturated_fraction` (share of wet
cells at w=1.0 on the baseline map) is included since it is what makes the two species'
behaviour diverge.

**The gate conserves total occupancy — it redistributes and excludes, it never removes fish.
`mean_dw` is a wiring check only, never a stock-response metric on its own.**

| species | stage | baseline mean w | baseline saturated | ΔS | TV | mean Δw | newly-excluded |
|---|---|---:|---:|---|---:|---:|---:|
| cod_west | juvenile | 1.0000 | 100.0% | −1 | 0.0021 | −0.0021 | 0.000% |
| cod_west | juvenile |  |  | −2 | 0.0184 | −0.0209 | 0.000% |
| cod_west | juvenile |  |  | −3 | 0.0707 | −0.0827 | 0.000% |
| cod_west | adult/spawning | 1.0000 | 100.0% | −1 | 0.0017 | −0.0017 | 0.000% |
| cod_west | adult/spawning |  |  | −2 | 0.0171 | −0.0191 | 0.000% |
| cod_west | adult/spawning |  |  | −3 | 0.0641 | −0.0733 | 0.000% |
| cod_east | juvenile | 0.9923 | 96.2% | −1 | 0.0201 | −0.0224 | 0.060% |
| cod_east | juvenile |  |  | −2 | 0.0849 | −0.1191 | 0.060% |
| cod_east | juvenile |  |  | −3 | 0.1963 | −0.2762 | 3.353% |
| cod_east | adult | 0.9794 | 93.3% | −1 | 0.0278 | −0.0313 | 0.227% |
| cod_east | adult |  |  | −2 | 0.0988 | −0.1400 | 0.227% |
| cod_east | adult |  |  | −3 | 0.2185 | −0.3039 | 5.213% |
| cod_east | spawning | 0.9929 | 99.0% | −1 | 0.0141 | −0.0160 | 0.087% |
| cod_east | spawning |  |  | −2 | 0.0808 | −0.1148 | 0.087% |
| cod_east | spawning |  |  | −3 | 0.1924 | −0.2798 | 0.434% |

- **cod_west's gate is saturated on the real production maps** — mean w = 1.0000 on all three
  life-stage maps at baseline, exactly as the spec's pre-computed expectation stated — and its
  `excluded_fraction` stays **0.000% at every ΔS tested, including −3**. Its non-zero movers
  are TV and mean-Δw only, i.e. pure redistribution among already-saturated cells, never
  exclusion. This is the mechanism behind Headline 2 below.
- **cod_east's newly-excluded-cell fraction is identical between −1 and −2 PSU** (juvenile
  0.060%, adult 0.227%, spawning 0.087%, all arms) — the same set of cells crosses the w=0
  threshold at both offsets (reconciled in Task 2's review), consistent with the spec's stated
  ≤0.23% ceiling for the −1/−2 regime. Only at −3 PSU does the exclusion fraction move
  materially (juvenile 3.353%, adult 5.213%, spawning 0.434%) — the exclusion-regime lever the
  spec predicted would only fire there.
- TV at −1/−2 (cod_east adult: 0.028 / 0.099) matches the spec's pre-computed expectation
  (0.028 / 0.099) to three decimal places.

### Prey-overlap instrument — predicted change in normalized cod occupancy mass over each prey species' map cells

Adult and juvenile rows both reported (Task 2 review added the juvenile row after the initial
build was adult-only). **The gate conserves total occupancy — it redistributes and excludes,
it never removes fish; the table below is a redistribution metric, not a stock-response
metric on its own.**

| predator | stage | prey | −1 PSU | −2 PSU | −3 PSU |
|---|---|---|---:|---:|---:|
| cod_west | adult | stickleback | −0.00004 | +0.00020 | +0.00181 |
| cod_west | adult | perch | 0.00000 | 0.00000 | 0.00000 |
| cod_west | adult | pikeperch | −0.00074 | −0.00507 | −0.01934 |
| cod_west | adult | smelt | 0.00000 | 0.00000 | 0.00000 |
| cod_west | juvenile | stickleback | −0.00006 | +0.00018 | +0.00190 |
| cod_west | juvenile | perch | 0.00000 | 0.00000 | 0.00000 |
| cod_west | juvenile | pikeperch | −0.00086 | −0.00594 | −0.02257 |
| cod_west | juvenile | smelt | 0.00000 | 0.00000 | 0.00000 |
| cod_east | adult | stickleback | −0.00014 | −0.00071 | −0.00192 |
| cod_east | adult | perch | −0.01064 | −0.02173 | −0.03342 |
| cod_east | adult | pikeperch | −0.00339 | −0.01028 | −0.01795 |
| cod_east | adult | smelt | −0.01043 | −0.02224 | −0.03400 |
| cod_east | juvenile | stickleback | −0.00011 | −0.00065 | −0.00185 |
| cod_east | juvenile | perch | −0.00802 | −0.01685 | −0.02515 |
| cod_east | juvenile | pikeperch | −0.00414 | −0.01130 | −0.01889 |
| cod_east | juvenile | smelt | −0.00323 | −0.00245 | +0.00248 |

All entries are ≤3.4% in magnitude at the largest lever (−3 PSU) — an order of magnitude
smaller than the July flip's chain (cod moved to concentrate away from the coast by an
OFF→ON gate switch, not a graded few-percent occupancy shift). The predator-side occupancy
mass over prey cells barely moves at any lever, which is the direct instrument-level evidence
behind Headline 1.

## Headline 1 — the July chain does not reproduce

The July `docs/baltic_salinity_gate_percid_mechanism_2026-07-05.md` chain (gate OFF→ON, same
cod-only lever, pre-calibration 8-species config) found: cod excluded from low-salinity
coastal cells → stickleback released from cod predation, **+94.0%** → competition/egg
predation on percids → **perch −35.2%, pikeperch −33.1%**.

**None of that reproduces here.** Across all three graded levers (−1/−2/−3 PSU), stickleback
moves +2.72% / +0.14% / +0.64% (baseline seed sd 3.2%) and the percids move −2.72%…+0.96%
(perch, sd 1.7%) and +1.64%…+1.90% (pikeperch, sd 2.2%) — every one of these sits at or below
~1.6× its own baseline seed noise, a completely different signature from July's ±33–94% moves. This is a
genuine, reportable "chain does not fire" result, not a null result from an underpowered
instrument — the biomass measurement in this same run demonstrably resolves a large response
elsewhere (cod_west +27.85% at ~5.4× its own baseline seed sd, Headline 2 below), so flat
stickleback/percid biomass at every lever is not a resolution artifact of the harness; it is a
measured absence of transmission.

Three candidate dampers were identified at spec time; **this run cannot separate them, and it
would be dishonest to pick one**:

1. **Graded-vs-flip:** every arm here is a small additive offset on top of an already-gated
   production baseline (the gate has been live since July). July's comparison was OFF→ON — a
   ~10× larger perturbation in kind, not degree. The arms measure a transmission *gradient*
   from an already-partially-gated state; July measured the full step function.
2. **Pre-computed exclusion near-vacuity:** the newly-excluded-cell fraction — the July
   mechanism's actual lever (cod physically excluded from cells) — stays ≤0.23% at −1/−2 PSU
   and only reaches 5.2% (cod_east adult) at −3 PSU. Most of the graded run's signal is
   redistribution (TV), not exclusion; July's OFF state had cod fully present in every coastal
   cell, so its ON transition excluded a far larger fraction at once.
3. **The August herring–stickleback clamp:** `docs/baltic_certification_2026-08-14.md`
   (citing `docs/baltic_stickleback_mechanism_2026-08-12.md`, Run 8) established, independent
   of this experiment, that stickleback's biomass is governed by **herring predation on its
   eggs and young-of-year** (herring takes 55–63% of stickleback's early-stage deaths), not by
   cod predation pressure — the July chain's proposed release-from-cod-predation pathway may
   simply be a minor term next to the herring term that was characterized a month later. A
   ±2-timestep herring spawning shift moves stickleback ∓20% through that pathway alone,
   dwarfing anything seen here.

The run does not measure which of these three explains the gap, or in what proportion — saying
so honestly is the finding.

## Headline 2 — cod_west, the spec's own "saturated null control," is the largest responder

The spec's pre-registered expectation (decision 6, label 6) was explicit: "cod_west's gate is
a no-op in production (mean w=1.0000 on all three maps, every frame) and stays ~0 at
dS=−1/−2 — this is effectively a cod_east experiment." That statement about the *baseline
field* is confirmed exactly by this run (mean w = 1.0000 on all three cod_west maps, every
month, per the instruments table above). **The prediction built on top of it — that cod_west
would therefore be inert — is wrong**, and increasingly wrong as the lever grows:

| ΔS | cod_west final-decade Δ | vs baseline seed sd (5.2%) |
|---|---:|---:|
| −1 PSU | +5.87% | ≈1.1× sd — borderline |
| −2 PSU | +10.96% | ≈2.1× sd — clearly above noise |
| −3 PSU | +27.85% | ≈5.4× sd — dominant signal in the whole table |

Monotone, and by −2/−3 PSU far outside anything explainable as seed noise.

**Resolution (measured half):** "saturated" describes the *baseline* field — every wet cell in
cod_west's range sits at w=1.0 today, so an infinitesimal negative offset changes nothing (this
is why `excluded_fraction` stays exactly 0.000% at every lever — no cell is ever pushed to
w=0, per the framing sentence above: the gate redistributes and excludes, it never removes
fish). But saturation at the baseline field says nothing about what happens as the field itself
is shifted: once the underlying salinity is offset by −2 or −3 PSU, some of cod_west's map
cells that were comfortably inside the ramp's saturated top (≥6 PSU) drop toward or into the
3–6 PSU ramp band, and the occupancy *weights* on those cells fall below 1 even though no cell
is excluded outright — mean-Δw at −2 PSU is only **−0.019 to −0.021** across stages, small
numbers that nonetheless correspond to a real shift in the *pattern* of where cod_west's
occupancy mass sits (TV 0.017–0.018 at −2, 0.064–0.071 at −3). This much — that cod_west's
weights move nontrivially in pattern despite zero exclusion, and that its biomass moves a lot
— is measured directly.

**Resolution (inferred half — flagged as such):** this run has no spatial predation, diet, or
mortality-by-cause decomposition for cod_west, so *why* a small occupancy-pattern shift
produces a disproportionately large biomass gain is not directly measured here. The plausible
reading, consistent with the same concentration pathway July documented for cod under the
OFF→ON flip (cod +14.9% when the gate concentrated cod into productive saline basins), is that
cod_west's response works the same way at smaller amplitude — redistribution toward
already-favourable cells rather than exclusion from unfavourable ones. **This run measured the
ratio (occupancy-pattern shift small, biomass shift large), not the mechanism, so
"consistent with" is as far as this evidence goes.** Settling it would need a mortality-by-cause
or spatial-biomass breakdown for cod_west specifically, comparing predation/starvation/growth
terms across arms — out of scope here.

**What this teaches, stated plainly:** baseline-field saturation is not the same property as
saturation-under-perturbation. A gate that is a documented no-op against today's field can
still be the most sensitive lever in the whole system once the field itself moves — the
"null control" label described where cod_west sits today, not how it responds to a shift. The
spec's own framing (label 6, restated below) is corrected in this doc rather than papered over:
cod_west is **not** an inert control across this run's full ΔS range; it is inert only at the
zero-offset limit, and its sensitivity grows fastest of any species tested.

## Labels (spec decision 6, all restated)

1. **Not a projection:** no ensemble generation supplies a citable mean freshening delta
   (Meier et al. 2022 Table 8: BalticAPP −0.06, ECOSUPPORT −0.15, CLIMSEA ≈0 g/kg SSS; only
   2006-era extremes reached −45%). The ΔS levers are chosen, not cited (see "Delta spec"
   above).
2. **RV confound:** this is an occupancy-pathway-only instrument; cod_east recruitment is
   RV-prescribed (gate factor 0.32–0.87 across the scored decade), so its response here is
   conditioned on that prescription, not free-running. cod_east's chain-table and instrument
   numbers above should be read with that in mind.
3. **Single-source climatology:** the bottom-salinity field's provenance is CMEMS PHY,
   deepest-valid level (per the file's own attrs) — one source, not an ensemble.
4. **Fixed production ramp 3–6 PSU:** the ramp bounds are the live production values, not
   retuned by this experiment (non-goal).
5. **Uniform-offset spatial blindness:** ΔS is a spatially uniform additive offset — it does
   not represent any real spatial pattern of change.
6. **cod_west = saturated null control — NOW CORRECTED with the empirical outcome.** The gate
   is genuinely a no-op on cod_west's *baseline* field (mean w=1.0000, all three maps, every
   frame — confirmed above). It is **not** inert once the field is offset: cod_west is the
   largest responder in the whole chain table at −2/−3 PSU (+11.0%, +27.9%). See Headline 2.
7. **The all-zero/un-gate guard status is reported per arm:** the engine's all-zero guard
   silently reverts a species to UNGATED movement for any (map, frame) where map·w sums to
   zero — a wiring hazard the builder turns visible, not a harness-fixed bug. **This run: 0
   events at every arm** (`zero`, `ds_m1`, `ds_m2`, `ds_m3`), including `ds_m3` — the guard
   never fired.
8. **Java gap:** Java silently ignores `movement.salinity.*` — no Java cross-check exists for
   this experiment (joins the C1 thermal item, both waiting on the user-dirty `runner.py`).
9. **The gate conserves total occupancy** — it redistributes and excludes, it never removes
   fish. `mean_dw` is a wiring check only, never a stock-response metric on its own (applied
   throughout the instruments section above).

## Both loader gaps — Stage-2 items (not fixed here, per non-goals)

Neither gap was touched by this task; both are surfaced as findings for the Stage-2
time-policy work, exactly as the spec's non-goals section requires:

1. **Frame-count wrap:** the salinity loader (`osmose/engine/config.py::_load_salinity_gate`)
   has **no frame-count validation of its own** — a mismatched-length field would silently
   wrap via `step % <loaded frame count>` rather than raise, misaligning the month-to-step
   mapping partway through the year. This run's harness-side frame-count assert (gate (e)
   above) caught this class of error at the harness level, but the engine itself remains
   unguarded — the same class of gap the oxygen coupling closed with a `ValueError` (see
   CLAUDE.md's bottom-oxygen note); the salinity loader has not received the equivalent fix.
2. **All-zero un-gate:** the engine's all-zero (map, frame) guard silently reverts a species to
   UNGATED movement whenever `map · w` sums to zero for a given (map, frame) pair, rather than
   raising or logging. This run's builder made the guard's status **visible** (label 7 above)
   and confirmed it never fired at any arm tested — but the guard itself is still silent by
   design; a future arm or species combination could trip it without any indication in the
   biomass output alone.

## What this is not

No pass/fail envelope claim for any arm (spec decision 5 — arms are reported, not certified).
No ramp retuning, no percid-side gating, no reproduction-side salinity mechanism, no engine
changes (both loader gaps above are surfaced findings only), no recalibration. No claim that
the July chain is "refuted" in general — only that it does not reproduce **at these three
graded levers on top of today's already-gated production baseline**, for reasons this run
cannot disentangle (Headline 1).

## Deliverables

- Delta spec: `data/baltic/scenarios/c4_salinity_sensitivity.json` (commit `6ed46fd`).
- Builder: `scripts/build_baltic_c4_forcing.py` (commits `7506c87`, `594875f`).
- Harness: `scripts/baltic_c4_salinity_ab.py` + `tests/test_baltic_c4_harness_helpers.py`
  (commit `947533c`).
- This results doc + copied report: `docs/baltic_c4_salinity_2026-08-30.md`,
  `docs/diagnostics/baltic_c4_salinity_report.json` (this task).
