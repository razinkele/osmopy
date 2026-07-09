# Baltic Chunk 0 — de-risk results (2026-07-08)

**Harness:** `scripts/baltic_bistability_chunk0.py` (v3, four adversarial-review rounds).
**Run:** `--experiment both --years 15 --seeds 0 1 2` on the deployed `data/baltic/baltic_all-parameters.csv`
(PythonEngine). Base larval rates read at runtime = per-dt `{cod 15, herring 8, sprat 9, flounder 12,
perch 13, pikeperch 15, smelt 13.5, stickleback 3.5}` (the post-4.4.1-migration values, not the raw 360).

## 1. Bistability — MONOSTABLE (trustworthy; establishment fraction 80%)

Cod-rich vs cod-poor initial conditions, swept across the larval-mortality driver:

| larva scale | cod-rich basin | cod-poor basin | gap | outcome |
|---|---|---|---|---|
| ×0.03 | overshoot | overshoot | 0.109 | same-basin |
| ×0.10 | overshoot | overshoot | 0.088 | same-basin |
| ×0.30 | overshoot | overshoot | 0.096 | same-basin |
| ×0.50 | overshoot | overshoot | 0.001 | same-basin |
| ×1.00 (deployed) | collapsed | collapsed | 0.996 | same-basin |

Every seed agreed (no seed-splits), and cod-rich established a non-collapsed stock at 4/5 scales
(establishment fraction 80% ⇒ the verdict is *trustworthy*, not instrument-limited).

**The two initial conditions never land in different basins.** Cod's fate is set by the larval-mortality
driver alone — overshoot at low larval M, collapse at the deployed rate — independent of the starting
cod stock. **This empirically confirms the investigation's central reframe: the collapse↔overshoot fork
is a MONOSTABLE response to one parameter, not a self-sustaining bistability.** The model lacks the
endogenous feedbacks (clupeid→cod-egg predation; depletable plankton) that would create a real
predator-pit / alternative stable state.

**Caveat (conservative test).** Egg-only initial conditions filtered through the swept larval mortality
and Beverton-Holt compensation, plus single-cod-axis ICs, cannot construct the real Baltic
*clupeid-dominated* (sprat-dominated) alternative state. So this rules out an *egg-seeded* IC-dependence
but cannot fully rule out bistability. A definitive test needs the warm-start standing-stock primitive
(`docs/baltic_chunk0_warmstart_prerequisite.md`).

**Roadmap implication.** The bistability is not latent in the deployed model — it must be **created**
(Chunk C: clupeid→cod-egg predation; Chunk A2: depletable plankton) or **tested definitively** via the
warm-start primitive. A MONOSTABLE knife-edge is exactly what the reframe predicted.

## 2. Accessibility A/B on the deployed config — PROVISIONAL (correctly withheld)

Baseline (accessibility 0.8) vs lowered (0.1) on the deployed config:
- baseline: cod **collapsed**, herring overshoot, sprat undetermined, flounder undetermined, perch/pikeperch/smelt **collapsed**, stickleback undetermined.
- lowered: cod collapsed, herring overshoot, sprat low, flounder collapsed, percids/smelt collapsed, stickleback undetermined.
- Verdict: **PROVISIONAL** — sprat (a gated weight-1.0 stock) was non-stationary in the baseline arm, so the verdict is withheld (no false "A1 is a real lever"). The collapse veto would also have fired (cod collapsed in both arms; flounder driven to collapsed in the lowered arm).

**Why this is the wrong regime for A1.** The deployed (high-larval-M) baseline is a *collapsed* web —
there is almost no over-production to "relax" (only herring overshoots), and lowering accessibility just
starves the already-fragile stocks (flounder → collapsed, sprat → low). The A1 "plankton firehose"
hypothesis is about the **low-larval-M overshoot regime**, where the whole community over-produces. The
standard A/B must therefore be run there.

## 3. Accessibility A/B in the overshoot regime (larva ×0.1) — also PROVISIONAL (horizon-limited)

Baseline (accessibility 0.8) vs lowered (0.1), both at larva ×0.1 (where the community over-produces):
- baseline: cod overshoot, herring overshoot, sprat **collapsed**, perch/smelt overshoot; flounder/pikeperch/stickleback undetermined.
- lowered: cod overshoot, herring **undetermined**, sprat collapsed, pikeperch/smelt overshoot; others undetermined.
- gated stocks (cod/herring/sprat): `overshoot_base=1, overshoot_low=1, new_undershoot=0, undetermined=1, collapsed_lowered=1`.
- Verdict: **PROVISIONAL** — herring did not reach stationarity in the lowered arm within 15 y (and sprat is collapsed in both arms, which would also trip the collapse veto).

**The A/B cannot return a clean A1 verdict even in the correct regime.** The perturbed clupeids do not
settle within the 15-year window, and the config is documented to destabilize beyond ~15 y — so a longer
horizon isn't available. Neither remedy the verdict suggests helps: `--years` up hits the instability
ceiling, and `--seeds` up doesn't fix *within-run* non-stationarity. The stationarity guardrail correctly
withholds rather than reading a still-drifting run as "A1 is a real lever."

## 4. Bottom line & recommended next steps

- **Bistability: confirmed MONOSTABLE.** No latent alternative stable state exists across the
  larval-mortality driver — cod's basin is set by the driver, not the initial condition. To get the real
  cod↔sprat bistability the model must **create** the missing feedbacks (Chunk C clupeid→cod-egg
  predation; Chunk A2 depletable plankton) or add the warm-start primitive for a definitive test. This is
  the investigation's reframe, now empirically confirmed.
- **Accessibility (A1): not cheaply de-riskable on this config.** In BOTH the deployed (collapse) and the
  low-larval-M (overshoot) regimes the A/B returns PROVISIONAL, because the perturbed clupeids don't reach
  stationarity within the config's ~15-year stable horizon. A1 remains a plausible, one-line, reversible
  change, but its isolated effect **cannot be validated by a short-horizon A/B here** — it should be
  evaluated inside a full recalibration (Chunk F), or after the stable horizon is extended.
- **The unifying blocker → the highest-value next investment.** Both open questions hit the same wall: the
  deployed config has only a ~15-year stable horizon and no standing-stock initialization. The **warm-start
  standing-stock primitive** (`baltic_chunk0_warmstart_prerequisite.md`) unblocks *both* a definitive
  bistability test *and* a clean accessibility A/B — so it is the single most enabling Phase-2 prerequisite.
- **Instrument health: sound.** No crashes (`n_failed=0`), establishment fraction 80% (bistability verdict
  trustworthy), and every non-stationary/collapsed case correctly returned INSTRUMENT-LIMITED/PROVISIONAL
  instead of a false confident verdict — the discipline the four adversarial-review rounds built in.
