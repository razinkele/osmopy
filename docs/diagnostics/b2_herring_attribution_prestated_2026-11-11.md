# B2 herring attribution A/B: pre-stated criterion, 2026-11-11

Written and committed **before** any run. Harness: `scripts/b2_herring_attribution.py`.

## Question

B2's `rcp85_ref` arm puts +2.9 °C through the C1 reproduction thermal gate and cut herring's
final-decade biomass by **62.27%** in August 2026. On v1.9.0 it cuts it by **26.47%**. Which change
halved the response?

Between v1.8.8 (`1d8bd214`) and v1.9.0 only data changed. That is #236. In `osmose/`, the only
diffs are the version string and the opt-in SP1b module. #236 changed four data files that matter
here; the movement file changed in comments only.

## Arms

Each arm reverts files to their v1.8.8 content at **file level**, in the `osmose_demo` copy,
before the config is read. The harness asserts each arm's resolved per-step larval rates against
the expected values, so a skipped or doubled ÷24 migration cannot pass silently.

| arm | reverted to v1.8.8 |
|---|---|
| A0 | nothing (v1.9.0 as shipped) |
| A1 | herring larval rate sp1: 28.89 → 98.56 /yr |
| A2 | herring reproduction seasonality: autumn share restored (25%) |
| A3 | LTL forcing file |
| A4 | all of #236: A1 + A2 + A3 + A5 |
| A5 | cod-only pair: cod_east spawning map (SD 25 → old) and cod_west larval rate sp0 |

Each arm runs B2's `baseline` and `rcp85_ref` configs, built the way `run_b2` builds them
(`arm_overlays` on the same knob series and O₂ artifact), over B2's 5 house seeds, for 50 years.
That makes 60 runs.

## Metric

The metric is B2's own. **D** = (mean over the 5 seeds of the `rcp85_ref` final-decade herring
mean) / (the same for `baseline`) − 1, using the arm's own baseline. Also recorded:
- per-seed paired declines, and their SE;
- each arm's baseline herring biomass.

## Gate

**Gate (validity):** A0 must reproduce B2's v1.9.0 `rcp85_ref` decline of −26.47% to within
0.5 pp. B2 ran serially with multithreaded Numba; this harness runs one thread per worker. If A0
misses, stop: no revert arm is interpreted until the instrument offset is resolved.

## Criterion

Fixed now. With D_Aug = −62.27% and D_A0 from this run:

- **Fraction restored:** f = (D_arm − D_A0) / (D_Aug − D_A0).
- **A change "moves the response"** if |D_arm − D_A0| > 2 × the combined SE,
  sqrt(SE_arm² + SE_A0²).
- **The pivot is A4.**
  - **f(A4) ≥ 0.8:** #236 holds the August gap, and A1, A2, A3 and A5 say which part of it.
  - **f(A4) < 0.8:** the rest of the gap sits in the August → v1.8.8 window. That window holds
    engine T3/T1, the movement fixes, the GreySeal column, sprat F, #230 and others. Tonight's
    single reverts are then *sensitivities on the current state*, not an attribution of the
    published halving. The next bisection is the B2 pair at the August merge commit in a
    worktree, then v1.8.7 data on current code. It is not run tonight.
- **Single reverts from a jointly fitted config need not sum to A4.** If f(A1) + f(A2) + f(A3) +
  f(A5) differs from f(A4) by more than the SEs allow, it is reported as an **interaction**, not as
  unexplained.

## Limits

- August's per-seed values are not in its report, so the noise on D_Aug is unknown. The gap is
  treated as exact.
- f says which change moves the response, not why. No mechanism is claimed without a measurement.
