# Baltic Chunk A2 — depletable plankton results (2026-07-09)

**Feature:** opt-in per-resource logistic regrowth in `ResourceState.update()` (`ltl.depletable.enabled`,
`ltl.depletable.floor`, `species.regrowth.rate.sp{i}`); default off = byte-identical (990 engine tests
unchanged). Spec/plan: `docs/superpowers/{specs,plans}/2026-07-09-baltic-chunka2-depletable-plankton*`.
**Predecessors:** `docs/baltic_chunk0_warmstart_results_2026-07-09.md` (MONOSTABLE),
`docs/baltic_chunkc_results_2026-07-09.md` (egg predation alone insufficient — overshoot must be fixed
first). Reframe lever #2.

**Run config:** phytoplankton (sp8,9) regrowth 5.0 (≈chemostat, lightly grazed); zooplankton + benthos
(sp10–13) the depletable knob; floor 0.05; warm-start regime-shift sweeps at 25 y, seeds 0–2.

## 1. Depletion sanity — the feature works, with large teeth (larva ×0.3, off vs on, zoo rate 0.3)

| species | off | on | change |
|---|---|---|---|
| cod | 9.18 M | 2.14 M | −76.6% |
| herring | 4.97 M | 0.56 M | −88.7% |
| perch | 2.42 M | 0.29 M | −88.0% |
| pikeperch | 3.43 M | 0.79 M | −76.9% |
| flounder | 1.17 M | 0.11 M | −90.4% |
| sprat | 17 k | 22 k | +26.9% |

Finite, no blow-up, correct direction: the depletion feedback slashes the overshooting community 76–90%,
and sprat rises as the others fall (competition/predation release). Rate 0.3 is clearly *too strong*
(herring pushed under band), motivating the rate sweep.

## 2. Rate sweep — A2 is a powerful over-production brake (a calibration lever), 15 y

Zooplankton regrowth rate vs the ICES bands (in / over / under):

**Larva ×0.3 (overshoot regime):** off → cod 9.2M(over), herring 5.0M(over), perch 2.4M(over). Depletion
compresses the whole community; herring lands **in band** at rate 0.6–2.0 (e.g. rate 1.0: herring 1.64M
in), while cod stays over (2.5–4.5M — reduced 2–4× but not into band) and perch/flounder stay over.

**Larva ×1.0 (deployed):** off → herring 18.1M(over), sprat 4.4M(over), cod collapsed. Depletion brings
herring **into band** (rate 0.3–0.6: 1.4–2.2M) but cod/perch/flounder stay collapsed (larval-M-driven).

**Takeaway:** A2 dramatically relaxes the overshoot — herring 18M → in-band is a real improvement — but it
does **not** fully calibrate: cod remains over at low M / collapsed at ×1.0, and best in-band count is
~1/5. Larval mortality still sets cod's regime; A2 sets the *magnitude*. **A2 at rate ≈ 0.6 is a genuine
calibration lever worth considering on its own merits**, separate from the bistability question.

## 3. Bistability — A2 alone: MONOSTABLE (zoo rate 0.6, warm-start regime-shift, 25 y)

Cod never diverges between the cod-dominated and clupeid-dominated ICs at any scale (overshoot in both at
×0.03–0.5, collapsed in both at ×1.0); clupeid gap ≤ 0.052. Depletion lowered the biomasses (clupeids
0.6–2.9 Mt vs 3–23 Mt without it) but created **no second basin** (`regime_shift = False`).

**Why:** depletion is a *negative*, stabilizing feedback — it damps overshoot toward a **single**
equilibrium, which is if anything *anti*-bistability. Bistability requires a *positive* (Allee /
depensation) feedback. That is Chunk C — hence the combined test.

## 4. Bistability — A2 + Chunk C combined: MONOSTABLE (definitive, 4/5 determinate)

Depletion (zoo rate 0.6) **plus** clupeid→cod-egg predation (cod→herring/sprat accessibility 0.4), on the
now-relaxed community:

| larva scale | cod a / cod b | clupeid a / b | outcome |
|---|---|---|---|
| ×0.03 | overshoot / overshoot | 776k / 801k | same-basin |
| ×0.10 | overshoot / overshoot | 660k / 781k | same-basin |
| ×0.30 | overshoot / overshoot | 947k / 891k | same-basin |
| ×0.50 | overshoot / overshoot | 910k / 995k | provisional |
| ×1.00 | collapsed / collapsed | 2.72M / 2.88M | same-basin |

`regime_shift = False`, determinate fraction 0.8 — the **cleanest monostable result of the whole
investigation**. Even with the overshoot removed (so egg predation is no longer swamped) *and* the
cultivation-depensation positive feedback active, the two ICs converge: cod overshoots in both arms at low
M and collapses in both at ×1.0, and the clupeid gap stays ≤ 0.155.

## Conclusion — the deployed Baltic model is robustly MONOSTABLE

Across the full investigation — warm-start reciprocal invasion, Chunk C (egg predation), Chunk A2
(depletable plankton), and A2 + Chunk C combined — **cod's regime is set by the larval-mortality driver
alone** (overshoot at low M, collapse at the deployed rate), independent of the starting community and of
every food-web feedback tested. **There is no alternative stable state reachable by these levers.** The
collapse↔overshoot fork is a *monostable response to one parameter*, not a bistability; the reframe's
hypothesis that it is a latent two-regime bistability is **falsified for the model as configured**.

**Two actionable outcomes:**
1. **Bistability is not reachable via food-web feedbacks.** If a genuine Baltic cod↔sprat regime shift is
   wanted in the model, it is not egg-predation- or plankton-depletion-mediated at these strengths — it
   would need a different mechanism (e.g. a larval-mortality / recruitment Allee term, an explicit
   depensatory recruitment function, or fishing-driven hysteresis), or it must be accepted that the model
   expresses the transition as a *parameter* sweep (larval M / F), not a bistability.
2. **A2 is a real calibration lever (positive side-finding).** Depletable plankton at zoo rate ≈ 0.6 pulls
   the massively overshooting community toward the ICES bands (herring 18 Mt → in-band; whole community
   −76–90%). This is worth pursuing as a *calibration* improvement in its own right (independent of the
   bistability question), most naturally combined with a larval-mortality setting that keeps cod alive.

## Outputs

- `osmose/engine/resources.py` (feature), `tests/test_engine_resources.py`, `config_validation.py`.
- `docs/diagnostics/baltic_chunka2_rate_sweep.json`, `baltic_chunka2_regime-shift_zr0.6.json`,
  `baltic_chunka2_chunkc_regime-shift.json`, and the variant matrix `predation-accessibility-a2c-s0.4.csv`.
