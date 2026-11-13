# Pre-stated: sprat maturity age 1 yr, 2026-11-13

Committed before any run of the candidate config.

## The request and how it is read

The maintainer asked: "set a maturity age of 1 year for sprat, but be aware that only a part of the
cohort - around 15% to 40%". The sentence is read as: sprat should not spawn before age 1, and at
age 1 only part of the cohort, about 15–40%, should be mature. The maturity age is fixed by the
request, so it is not a free parameter here.

## The change

`species.maturity.age.sp2 = 1.0` in `data/baltic/baltic_param-species.csv`. Nothing else changes:
not the maturity size (9 cm), the larval rate, the SR curve, or any other species.

OSMOSE maturity is knife-edge: a school is mature when `length >= maturity size AND age >= maturity
age`. A probe on current master (seed 1000, last 5 of 15 yr) found that age-1 sprat are 10–13 cm at
spawning, so the 9 cm size never binds. Under the new rule, partial maturity at age 1 comes only from
birthdays falling inside the spawning season (steps 6–13): each fish becomes mature at its first
birthday. Measured on master's birth dates, that is 0% of the age-1 cohort at the first spawning step,
97% at the last, and 38.2% season-weighted. The same run gives a 0.69 SSB ratio (new rule / current).
These master figures motivate the change; they are not the test.

## C1: the age-1 mature share (the maintainer's band)

- **Metric:** the share by numbers of ICES-age-1 sprat (born in the previous calendar year) that is
  mature under the live rule. It is measured on the state the reproduction step sees, at each sprat
  spawning step, weighted by the spawning-season weight, over the final decade (years 40–49).
- **Measured on the candidate config,** not master, because the new rule moves egg output within the
  season and so next year's birth dates. Three seeds: 42, 123, 7. 50 years.
- **Pass:** the mean over seeds is in [0.15, 0.40].
- **Out of band is reported, not tuned.** The age is fixed by the request, and the size cannot reduce
  the share without also making part of age 2 immature. The note will name the lever (the age in
  steps) and leave it to the maintainer.
- **Also reported:** the plain mean over the eight steps, the per-step ramp, and the age-2+ mature
  share (expected 100%).

## C2: certification

- `scripts/baltic_stability_certify.py --params current`: 50 yr, seeds 42, 123, 7, 999, 2024.
- **Pass:** 5/5 assessed stocks persistent and in envelope, every seed. Cliff rule: no assessed
  stock's 5-seed range crosses its envelope.
- **Stop rule:** if any assessed stock leaves its envelope, stop and report. A larval re-fit, or any
  other parameter change to rescue it, is a recalibration that was not asked for.

## C3: SP1b drift guard

- Run `tests/test_larva_recal.py` explicitly (the d0 guard cannot see a shift in the cod baseline).
- If the guard fails because cod moved, re-solve `RECAL_RATES` as #250 did. Accept the re-solve only
  inside the 1-D tolerance (2%) and JOINT_TOL (5%).

## Reported findings, not gates

- Sprat SSB against its ICES SSB envelope (888–1,121 kt), from the validator, on a candidate 50-yr
  output. If it falls out while biomass still certifies, that is a finding, not something to re-fit.
- Sprat biomass, SSB and egg output before and after. The Shepherd curve (`ssbhalf` 213 kt,
  shape 1.72) is on its descending limb at current SSB, so lower SSB is expected to raise egg output.
  The size of that effect is not predicted here.
- Sprat's "Juvenil" stage now runs to age 1 (same maturity rule), so stage mortality rates are not
  comparable across the change.
- Which v1.10.0 results go stale (B2, C4, the Java cross-check, SMS M2). They are listed, not re-run.
- Java reads `species.maturity.age.sp#` only if verified. Unverified Java-side sprat numbers are not
  quoted.

## Instrument checks (before C1/C2)

- The candidate config loads with `maturity_age_dt[2] == 24` (1.0 yr × 24 steps).
- An explicit `species.maturity.age.sp2 = 0.0` is bit-identical to master (the default), so the key
  enters only through the value.
