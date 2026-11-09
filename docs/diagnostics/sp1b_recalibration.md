# SP1b recalibration diagnostic

SP1 (spatial RV egg-survival clip) enabled on: cod_west, cod_east.
Each stock's larval rate (resolved per-cohort) is solved so its own SP1-on mean matches its SP1-off mean: 1-D solver tol 0.02 on the stock's own axis, joint band JOINT_TOL = 0.05 on the pair (see 'Coupling' below). Rates are frozen with the d0 they were solved against; `sp1_on_config` refuses a config whose d0 has moved.

| stock | d0 | rate | baseline (t) | on_recal (t) | rel_err | overshoot off | overshoot on |
|---|---|---|---|---|---|---|---|
| cod_west | 10.4442 | 9.3162 | 3769.4 | 3678.2 | 0.0242 | 2.15 | 1.75 |
| cod_east | 10.1569 | 9.1407 | 105961.0 | 105774.6 | 0.0018 | 1.53 | 1.09 |

total cod: off=109730.4  on_recal=109452.7  drift=-0.25% (measured, not gated)

## Overshoot (max/mean over years 3-14) — measured, NOT gated
cod_west: off=2.15  on_recal=1.75  ratio=0.82  (damps the boom/bust)
cod_east: off=1.53  on_recal=1.09  ratio=0.71  (damps the boom/bust)

## Coupling: why the joint band is wider than the 1-D tol
With its own rate fixed at the root, each stock's mean still moves with the OTHER stock's rate, and not smoothly: on the 2026-10-01 solve (47 evaluations, two Gauss-Seidel sweeps; its log is docs/diagnostics/sp1b_solve.log at 3a3b1eed) the half-range within +-0.1 of the roots was 2.0-2.1% for cod_east across cod_west's rate and 3.8% for cod_west across cod_east's rate. Sweep 1 ended cod_east -2.9% / cod_west +1.2%, sweep 2 +4.0% / +0.9%; neither sweep put both inside 2%, and the jitter is as wide as that band, so a joint 2% is below the coupling's resolution. That solve froze its best sweep (1) and the joint band is set above the jitter. A third sweep could land inside 2% by chance, which would make the drift guard trip on trajectory sensitivity rather than drift. The 2026-11-05 re-solve did land both inside 2% in one sweep (cod_east +1.4%, cod_west +0.03%), and the 2026-11-08 GreySeal re-solve in its second (sweep 1 left cod_east at -4.9%; sweep 2 ended +0.01% / -1.3%), the C1a re-solve the same day in one (-0.19% / -1.6%), and the T3+T1 engine re-solve in its second (sweep 1 left cod_east at +3.8%; sweep 2 ended +1.6% / +1.1%). The corrected-LTL re-solve converged cod_east (-0.18%) but left cod_west at -2.4% after 20 bisection steps, because two evaluations at the same printed rates differed by 5% there. That is this jitter, and it is why the guard holds the pair to JOINT_TOL and not the 1-D tol.

## Caveat: stacked RV terms on cod_east
cod_east carries the temporal RV gate (`reproduction.rv.gate.*`, its dominant control) AND, under this overlay, the spatial RV clip. The two have not been A/B gated against each other; the per-stock neutrality above holds for the stacked pair as a whole, not for either term alone.

## History
The 2026-07-02 solve (`RECAL_RATE = 14.6551`, aggregate cod, d0=15.0) was retired on 2026-10-01: the Baltic baseline was recalibrated on 2026-07-24 (sp0 larva rate 360 -> 243.76 per year, resolved d0 15.0 -> 10.157), so the frozen constant had silently become a larval-mortality INCREASE; stacked on the spatial clip it drove cod_west extinct under SP1 (6432 -> 1 t). Issue #131 attributed this to the 2026-07-25 cod E/W split; the split only made it visible. Rates now carry their d0 so this cannot recur silently.

The 2026-10-01 rates were re-solved on 2026-11-05: the sprat F change of 2026-10-29 (3d74d572, 0.1637 -> 0.369) left d0 untouched, so the d0 check stayed quiet, but it moved both SP1-off baselines (cod_east 105120 -> 105990 t, cod_west 5991 -> 6236 t) and left cod_west 4.5% below neutral. The drift guard caught it, but it skips under CI, so it went unnoticed for a week. Only cod_west's rate moved (9.0259 -> 9.0138).

The 2026-11-05 rates were re-solved on 2026-11-08 for the GreySeal accessibility column: the seal had no column, so the kernel's -1 path let it eat every prey at 1.0; it now carries the Java arm's values (BG_ACCESS). d0 is untouched, but both SP1-off baselines moved (cod_east 105990 -> 102491 t, cod_west 6236 -> 3048 t), and the drift guard caught it before merge this time. cod_west's halving is in this 15-yr window only: its 50-yr final-decade mean is unchanged (12.3-14.3k -> 12.8-13.7k t over 5 seeds). Both rates moved (cod_east 9.0687 -> 9.0973, cod_west 9.0138 -> 9.2763).

Re-solved again the same day for C1a, which stops packing every adult herring onto a 10-cell autumn map (docs/baltic_spawning_fixes_2026-11-08.md). The SP1-off baselines moved to cod_east 102491 -> 101816 t and cod_west 3048 -> 3205 t. The solve converged in one sweep (34 evaluations). cod_east's rate came out identical (9.0973) and only cod_west's moved (9.2763 -> 9.2558).

Re-solved a third time that day for two engine fixes from the 2026-11-07 review. T3: a school under one fish is dead, as in Java. T1: no length cap at L-infinity unless species.lmax is set. Baselines moved to cod_east 101816 -> 103151 t and cod_west 3205 -> 4332 t. The solve converged in two sweeps (49 evaluations). Rates moved to cod_east 9.0973 -> 9.1403 and cod_west 9.2558 -> 9.1582.

Re-solved a fourth time that day for the corrected-LTL recalibration (docs/baltic_ltl_recalibration_2026-11-08.md): corrected LTL forcing, the C2 spawning map, no herring autumn eggs, and re-fitted herring and cod_west larval rates. cod_west's resolved d0 moved 10.157 -> 10.444, the case the d0 guard exists for. Baselines moved to cod_east 103151 -> 105961 t and cod_west 4332 -> 3769 t. Rates are cod_east 9.1403 -> 9.1407 and cod_west 9.1582 -> 9.3162. cod_west's 2.4% misses the 1-D tol, but it is inside JOINT_TOL (see Coupling).
