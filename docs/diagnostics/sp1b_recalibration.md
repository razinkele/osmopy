# SP1b recalibration diagnostic

RECAL_RATE = 14.6551  (cod larval mortality, resolved per-cohort; d0=15.0)
mean cod: off=2054.3  on_recal=2067.8  rel_err=0.007  (target <= 0.02)

## Overshoot (max/mean over years 3-14) — measured, NOT gated
off=1.43  on_recal=1.88  ratio=1.31  (does not damp the boom/bust)

## Superseded by the cod disaggregation (2026-07-29)

`RECAL_RATE = 14.6551` was solved for the **aggregate** cod stock. After cod was split into
`cod_west` (sp0, ~6.4 kt in the 15-yr sim) + `cod_east` (sp8, ~137 kt), the frozen rate no longer
holds cod neutral: applied to the disaggregated config it drives the small western stock extinct
(cod_west 6432 → 1 t under SP1) while cod_east barely moves (4.6%), so total cod drifts ~8.9% — far
outside the 2% neutral-drift target. The larval mortality that neutralised a ~140 kt aggregate stock
simply annihilates the 6.4 kt western one.

Consequences:
- `mean_cod()` (`osmose/calibration/larva_recal.py`) now returns **total** cod = cod_west + cod_east
  (aggregate `cod` column fallback retained), so the SP1b scripts/tests no longer crash with
  `KeyError: 'cod'` on the disaggregated config.
- `test_sp1b_mean_neutral_drift_guard` **skips** on a disaggregated config (its neutral-drift premise
  is structurally false there — see the skip reason).
- Restoring the guard requires **re-solving RECAL_RATE per stock** (cod_west and cod_east have very
  different scales and cod_east is separately RV-gated) — a maintainer-host recalibration task, out of
  scope for the disaggregation itself.
