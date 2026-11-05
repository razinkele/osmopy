# SP1b recalibration diagnostic

SP1 (spatial RV egg-survival clip) enabled on: cod_west, cod_east.
Each stock's larval rate (resolved per-cohort) is solved so its own SP1-on mean matches its SP1-off mean: 1-D solver tol 0.02 on the stock's own axis, joint band JOINT_TOL = 0.05 on the pair (see 'Coupling' below). Rates are frozen with the d0 they were solved against; `sp1_on_config` refuses a config whose d0 has moved.

| stock | d0 | rate | baseline (t) | on_recal (t) | rel_err | overshoot off | overshoot on |
|---|---|---|---|---|---|---|---|
| cod_west | 10.1569 | 9.0138 | 6235.7 | 6237.8 | 0.0003 | 1.66 | 1.71 |
| cod_east | 10.1569 | 9.0687 | 105990.1 | 107482.8 | 0.0141 | 1.33 | 1.22 |

total cod: off=112225.8  on_recal=113720.6  drift=+1.33% (measured, not gated)

## Overshoot (max/mean over years 3-14) — measured, NOT gated
cod_west: off=1.66  on_recal=1.71  ratio=1.03  (does not damp the boom/bust)
cod_east: off=1.33  on_recal=1.22  ratio=0.91  (damps the boom/bust)

## Coupling: why the joint band is wider than the 1-D tol
With its own rate fixed at the root, each stock's mean still moves with the OTHER stock's rate, and not smoothly: on the 2026-10-01 solve (47 evaluations, two Gauss-Seidel sweeps; its log is docs/diagnostics/sp1b_solve.log at 3a3b1eed) the half-range within +-0.1 of the roots was 2.0-2.1% for cod_east across cod_west's rate and 3.8% for cod_west across cod_east's rate. Sweep 1 ended cod_east -2.9% / cod_west +1.2%, sweep 2 +4.0% / +0.9%; neither sweep put both inside 2%, and the jitter is as wide as that band, so a joint 2% is below the coupling's resolution. The frozen state is the best sweep (1) and the joint band is set above the jitter. A third sweep could land inside 2% by chance, which would make the drift guard trip on trajectory sensitivity rather than drift. The 2026-11-05 re-solve did land both inside 2% in one sweep (cod_east +1.4%, cod_west +0.03%); the band stays at JOINT_TOL, so that landing is not what the drift guard holds it to.

## Caveat: stacked RV terms on cod_east
cod_east carries the temporal RV gate (`reproduction.rv.gate.*`, its dominant control) AND, under this overlay, the spatial RV clip. The two have not been A/B gated against each other; the per-stock neutrality above holds for the stacked pair as a whole, not for either term alone.

## History
The 2026-07-02 solve (`RECAL_RATE = 14.6551`, aggregate cod, d0=15.0) was retired on 2026-10-01: the Baltic baseline was recalibrated on 2026-07-24 (sp0 larva rate 360 -> 243.76 per year, resolved d0 15.0 -> 10.157), so the frozen constant had silently become a larval-mortality INCREASE; stacked on the spatial clip it drove cod_west extinct under SP1 (6432 -> 1 t). Issue #131 attributed this to the 2026-07-25 cod E/W split; the split only made it visible. Rates now carry their d0 so this cannot recur silently.

The 2026-10-01 rates were re-solved on 2026-11-05: the sprat F change of 2026-10-29 (3d74d572, 0.1637 -> 0.369) left d0 untouched, so the d0 check stayed quiet, but it moved both SP1-off baselines (cod_east 105120 -> 105990 t, cod_west 5991 -> 6236 t) and left cod_west 4.5% below neutral. The drift guard caught it, but it skips under CI, so it went unnoticed for a week. Only cod_west's rate moved (9.0259 -> 9.0138).
