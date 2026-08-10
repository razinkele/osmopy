# The computed RV does not reproduce the prescribed series — measured, not assumed

**Date:** 2026-08-10
**Bears on:** `docs/superpowers/specs/2026-08-10-baltic-computed-rv-design.md` (C2b), which is
**withdrawn as written** on the strength of this measurement.

## What happened

The C2(b) spec proposed downloading CMEMS reanalysis and building a depth-integrated reproductive
volume. An adversarial review of the spec established two things that made the whole design
premature:

1. **The machinery already exists.** `osmose/forcing/reproductive_volume.py:build_rv_field_interannual`
   implements exactly the specified integral (S ≥ 11 PSU, O₂ ≥ 89.3 mmol m⁻³, viable column
   thickness summed over depth); `scripts/download_baltic_rv_forcing.py` and
   `scripts/build_baltic_rv_field.py --interannual` drive it; 80 cached CMEMS files cover
   1993–2021; and the materialised field is committed at
   `data/baltic_rv/baltic_rv_field_interannual.nc` (696 steps = 29 yr × 24, `start_year=1993`).
   Only the field → annual-scalar aggregation was ever missing. The spec asked to build what was
   already built — a search failure on my part, not a data gap.
2. **The proposed integration domain was wrong.** The spec took the horizontal extent from
   `data/baltic/maps/cod_spawning.csv`. That mask contains **zero cells between 55.05°N and
   55.85°N** — the entire Bornholm Basin, which the literature identifies as the eastern stock's
   principal spawning ground and, since the late 1980s, effectively the only one producing viable
   eggs (MacKenzie, Hinrichsen & Plikshs 2000, *MEPS* 193:143–156, doi:10.3354/meps193143; Köster
   et al. 2005, *ICES JMS* 62:1408–1425, doi:10.1016/j.icesjms.2005.05.004). A related, separate
   config issue: the cod E/W split assigned the Bornholm Deep cell to **cod_west**, while the RV
   gate runs on **cod_east** only.

## The measurement

Annual series aggregated from the existing field (May–August, mean over the season, summed over the
domain), compared against `data/baltic/reference/baltic_cod_reproductive_volume.csv` over the
1993–2020 overlap:

| domain | Spearman ρ | p | CV | peak in 2002–05 | peak in 2014–17 |
|---|---|---|---|---|---|
| Bornholm only (54.9–56.0 °N, 14.5–17.0 °E) | **+0.042** | 0.83 | 0.144 | **2003** | 2016 |
| SD 25+26+28 (54.5–58.5 °N, 14.5–21.0 °E) | −0.159 | 0.42 | 0.490 | 2004 | 2017 |
| cod_east mask bbox | −0.175 | 0.37 | 0.736 | 2004 | 2017 |

Prescribed series CV over the same years: **0.595**.

## Reading

**No domain yields positive rank agreement.** Fixing the Bornholm omission — the review's critical
finding — moves ρ from −0.175 to +0.042, i.e. from anti-correlated to *uncorrelated*. It does not
produce agreement.

Two signals point in opposite directions and both are informative:

* **In favour of the computation:** the Bornholm-only series puts its 2002–05 maximum exactly on
  **2003**, the Major Baltic Inflow year. That is the most falsifiable physical check available and
  the computation passes it, unprompted.
* **Against:** the Bornholm-only series has CV 0.144 against the prescribed 0.595 — it is far too
  flat, and its 2014–17 maximum lands on 2016 rather than 2015 (the December-2014 MBI's first full
  spawning season). Widening the domain restores variance but destroys the inflow timing and
  turns the correlation negative.

So the computed and prescribed quantities are **not the same measurement**, and the difference is
not a units or scaling artefact — rank correlation is scale-free. Candidate explanations, none yet
tested: a different threshold pair (the literature also uses O₂ ≥ 1 ml/L); a different season or a
season that shifted over the period; monthly-mean reanalysis smoothing the sharp haloclines and
episodic inflows RV depends on; or the prescribed series itself being a stock-assessment-derived
quantity rather than a purely hydrographic one.

## Consequence: C2(b) is not implementable as specified

The spec's A2 (in-sample rank agreement) fails on the existing data, and A1 (variance floor) fails
on the only domain that passes A3 (inflow timing). Those criteria were written to be falsifiable
and they falsified the design — which is the criteria working, not a reason to relax them.

A second, independent blocker was confirmed by the same review and would apply even if agreement
had held: `rv_gate_factor` clamps past the series end (`idx = min(offset + year, n_years - 1)`), so
a **1993-start, 29-row series leaves the certification-scored final decade (model years 40–49)
entirely clamped to a single value** — 10 of 10 scored years constant, versus 3 of 10 for the
current 47-row series. The offline criteria would certify interannual structure that the scored
window never sees. Any future attempt must resolve the series-length/run-horizon mismatch first;
note that `reproduction.rv.gate.start.year` makes this *worse*, not better.

## What is worth doing instead

1. **Explain the divergence before replacing anything.** The prescribed series is load-bearing
   (gate-off puts cod_east 1.61× over its ceiling) and is currently the better-validated input.
   The useful question is not "can we compute an RV" — we can — but "why does the computed one
   disagree", which is a threshold/season/product-resolution investigation costing hours, not a
   config swap.
2. **Fix the two config defects this uncovered**, independent of C2(b): the cod_east spawning mask
   excludes the Bornholm Basin, and the E/W split assigned the Bornholm Deep to cod_west while the
   gate runs on cod_east. Both are plausibly consequential for the disaggregated configuration and
   neither depends on this work.
3. **Correct the CMEMS catalogue** in `mcp_servers/copernicus/server.py` (missing `o2b` on the BGC
   reanalysis, `sob` and others on the PHY reanalysis) — a small documentation defect surfaced
   here.

## Method note

Everything above was obtained from data already in the repository, in minutes, before any
implementation. The spec's offline-first acceptance architecture (A1–A3 evaluated cheaply, only a
passing series earning a 35-minute certification run) is what made a negative result affordable —
that part of the design worked exactly as intended.
