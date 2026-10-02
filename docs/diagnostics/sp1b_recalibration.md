# SP1b recalibration diagnostic

SP1 (spatial RV egg-survival clip) enabled on: cod_west, cod_east.
Each stock's larval rate (resolved per-cohort) is solved so its own SP1-on mean matches its SP1-off mean: 1-D solver tol 0.02 on the stock's own axis, joint band JOINT_TOL = 0.05 on the pair (see 'Coupling' below). Rates are frozen with the d0 they were solved against; `sp1_on_config` refuses a config whose d0 has moved.

| stock | d0 | rate | baseline (t) | on_recal (t) | rel_err | overshoot off | overshoot on |
|---|---|---|---|---|---|---|---|
| cod_west | 10.1569 | 9.0259 | 5990.8 | 6061.2 | 0.0117 | 1.56 | 1.64 |
| cod_east | 10.1569 | 9.0687 | 105120.4 | 102061.7 | 0.0291 | 1.36 | 1.22 |

total cod: off=111111.3  on_recal=108122.9  drift=-2.69% (measured, not gated)

## Overshoot (max/mean over years 3-14) — measured, NOT gated
cod_west: off=1.56  on_recal=1.64  ratio=1.05  (does not damp the boom/bust)
cod_east: off=1.36  on_recal=1.22  ratio=0.90  (damps the boom/bust)

## Coupling: why the joint band is wider than the 1-D tol
With its own rate fixed at the root, each stock's mean still moves with the OTHER stock's rate, and not smoothly: on the 2026-10-01 solve (47 evaluations, two Gauss-Seidel sweeps, docs/diagnostics/sp1b_solve.log) the half-range within +-0.1 of the roots was 2.0-2.1% for cod_east across cod_west's rate and 3.8% for cod_west across cod_east's rate. Sweep 1 ended cod_east -2.9% / cod_west +1.2%, sweep 2 +4.0% / +0.9%; neither sweep put both inside 2%, and the jitter is as wide as that band, so a joint 2% is below the coupling's resolution. The frozen state is the best sweep (1) and the joint band is set above the jitter. A third sweep could land inside 2% by chance, which would make the drift guard trip on trajectory sensitivity rather than drift.

## Caveat: stacked RV terms on cod_east
cod_east carries the temporal RV gate (`reproduction.rv.gate.*`, its dominant control) AND, under this overlay, the spatial RV clip. The two have not been A/B gated against each other; the per-stock neutrality above holds for the stacked pair as a whole, not for either term alone.

## History
The 2026-07-02 solve (`RECAL_RATE = 14.6551`, aggregate cod, d0=15.0) was retired on 2026-10-01: the Baltic baseline was recalibrated on 2026-07-24 (sp0 larva rate 360 -> 243.76 per year, resolved d0 15.0 -> 10.157), so the frozen constant had silently become a larval-mortality INCREASE; stacked on the spatial clip it drove cod_west extinct under SP1 (6432 -> 1 t). Issue #131 attributed this to the 2026-07-25 cod E/W split; the split only made it visible. Rates now carry their d0 so this cannot recur silently.
