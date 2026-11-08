# SP1b recalibration diagnostic

SP1 (spatial RV egg-survival clip) enabled on: cod_west, cod_east.
Each stock's larval rate (resolved per-cohort) is solved so its own SP1-on mean matches its SP1-off mean: 1-D solver tol 0.02 on the stock's own axis, joint band JOINT_TOL = 0.05 on the pair (see 'Coupling' below). Rates are frozen with the d0 they were solved against; `sp1_on_config` refuses a config whose d0 has moved.

| stock | d0 | rate | baseline (t) | on_recal (t) | rel_err | overshoot off | overshoot on |
|---|---|---|---|---|---|---|---|
| cod_west | 10.1569 | 9.2558 | 3204.5 | 3151.7 | 0.0165 | 1.59 | 2.08 |
| cod_east | 10.1569 | 9.0973 | 101816.3 | 101624.8 | 0.0019 | 1.32 | 1.16 |

total cod: off=105020.9  on_recal=104776.6  drift=-0.23% (measured, not gated)

## Overshoot (max/mean over years 3-14) — measured, NOT gated
cod_west: off=1.59  on_recal=2.08  ratio=1.31  (does not damp the boom/bust)
cod_east: off=1.32  on_recal=1.16  ratio=0.88  (damps the boom/bust)

## Coupling: why the joint band is wider than the 1-D tol
With its own rate fixed at the root, each stock's mean still moves with the OTHER stock's rate, and not smoothly: on the 2026-10-01 solve (47 evaluations, two Gauss-Seidel sweeps; its log is docs/diagnostics/sp1b_solve.log at 3a3b1eed) the half-range within +-0.1 of the roots was 2.0-2.1% for cod_east across cod_west's rate and 3.8% for cod_west across cod_east's rate. Sweep 1 ended cod_east -2.9% / cod_west +1.2%, sweep 2 +4.0% / +0.9%; neither sweep put both inside 2%, and the jitter is as wide as that band, so a joint 2% is below the coupling's resolution. That solve froze its best sweep (1) and the joint band is set above the jitter. A third sweep could land inside 2% by chance, which would make the drift guard trip on trajectory sensitivity rather than drift. The 2026-11-05 re-solve did land both inside 2% in one sweep (cod_east +1.4%, cod_west +0.03%), and the 2026-11-08 GreySeal re-solve in its second (sweep 1 left cod_east at -4.9%; sweep 2 ended +0.01% / -1.3%), and the C1a re-solve the same day in one (-0.19% / -1.6%); the band stays at JOINT_TOL, so those landings are not what the drift guard holds it to.

## Caveat: stacked RV terms on cod_east
cod_east carries the temporal RV gate (`reproduction.rv.gate.*`, its dominant control) AND, under this overlay, the spatial RV clip. The two have not been A/B gated against each other; the per-stock neutrality above holds for the stacked pair as a whole, not for either term alone.

## History
The 2026-07-02 solve (`RECAL_RATE = 14.6551`, aggregate cod, d0=15.0) was retired on 2026-10-01: the Baltic baseline was recalibrated on 2026-07-24 (sp0 larva rate 360 -> 243.76 per year, resolved d0 15.0 -> 10.157), so the frozen constant had silently become a larval-mortality INCREASE; stacked on the spatial clip it drove cod_west extinct under SP1 (6432 -> 1 t). Issue #131 attributed this to the 2026-07-25 cod E/W split; the split only made it visible. Rates now carry their d0 so this cannot recur silently.

The 2026-10-01 rates were re-solved on 2026-11-05: the sprat F change of 2026-10-29 (3d74d572, 0.1637 -> 0.369) left d0 untouched, so the d0 check stayed quiet, but it moved both SP1-off baselines (cod_east 105120 -> 105990 t, cod_west 5991 -> 6236 t) and left cod_west 4.5% below neutral. The drift guard caught it, but it skips under CI, so it went unnoticed for a week. Only cod_west's rate moved (9.0259 -> 9.0138).

The 2026-11-05 rates were re-solved on 2026-11-08 for the GreySeal accessibility column: the seal had no column, so the kernel's -1 path let it eat every prey at 1.0; it now carries the Java arm's values (BG_ACCESS). d0 is untouched, but both SP1-off baselines moved (cod_east 105990 -> 102491 t, cod_west 6236 -> 3048 t), and the drift guard caught it before merge this time. cod_west's halving is in this 15-yr window only: its 50-yr final-decade mean is unchanged (12.3-14.3k -> 12.8-13.7k t over 5 seeds). Both rates moved (cod_east 9.0687 -> 9.0973, cod_west 9.0138 -> 9.2763).

Re-solved again the same day for C1a, which stops packing every adult herring onto a 10-cell autumn map (docs/baltic_spawning_fixes_2026-11-08.md). The SP1-off baselines moved to cod_east 102491 -> 101816 t and cod_west 3048 -> 3205 t. The solve converged in one sweep (34 evaluations). cod_east's rate came out identical (9.0973) and only cod_west's moved (9.2763 -> 9.2558).
