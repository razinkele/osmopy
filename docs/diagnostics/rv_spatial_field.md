# Spatial RV field diagnostic

RV_ref = 16.57 m
within-basin CV = 2.470  (GO/NO-GO: go if >= 0.20)  ->  GO
mean(s_cell) over RV>0 spawning cells = 0.577  (mean-anchor target [0.6, 1.0])
fraction of cod_spawning cells with RV > 0 = 0.273

## Basin contrast
spawn vs fresh northern gulf = inf  (mean_spawn=4.53, mean_fresh=0.00)
spawn vs ALL non-spawning ocean = 0.94  (mean_coast=4.83) -- CONFOUNDED, see finding

## Finding: the Danish-straits confound
The viable-thickness metric (salinity >= 11 PSU AND O2 >= 89.3 mmol/m3) makes the ultra-saline Danish straits/Kattegat the highest-RV cells (up to ~220 m), even though they are outside the cod spawning range and receive no eggs. A naive spawn-vs-all-ocean contrast therefore inverts. This does NOT affect the mechanism (eggs are placed only on the cod_spawning map), and the within-basin CV -- the real go/no-go -- is a strong GO.

## Gate on/off cod biomass (15-yr, years 3-14 mean)
off=2054  on=1728  delta=-16%  (SP1b larval-M recalibration restores the mean; not done here)
