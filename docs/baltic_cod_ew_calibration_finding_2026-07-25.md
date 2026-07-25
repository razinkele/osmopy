# Cod E/W disaggregation — Task 7 calibration finding (2026-07-25)

Phase 1 (cod disaggregation) Tasks 1–6 are complete and committed: cod is now
cod_west (sp0) + cod_east (sp8) with distinct life-history, salinity-niched maps,
an expanded predation matrix, split ICES targets, and cod_east RV gate + own
fishery. The **structure works** — the 9-species config runs end-to-end and is
warn-mode clean.

Task 7 (re-calibration) **did not reach an acceptable fit**, and the failure is
scientifically informative rather than a mere optimizer nuisance.

## What happened

The 9-species phase-13 DE (bounded ~3.6 h, eff_popsize 90, maxiter 40) converged
to **objective 12.34 vs the pre-split baseline's 2.33** and certified **2/9**
in-envelope. Every species ran 10–80× over its ICES envelope at *both* the 40-yr
calibration horizon and the 50-yr certification:

| species | 40yr mean | envelope | over |
|---|---|---|---|
| cod_west | 711 kt | 4–25 kt | 28× |
| cod_east | 5.08 Mt | 60–85 kt | 60× |
| herring | 1.71 Mt | 0.8–3 Mt | OK |
| sprat | 3.09 Mt | 0.8–2.5 Mt | 1.2× |
| flounder | 2.30 Mt | 20–100 kt | 23× |
| perch | 413 kt | 8–50 kt | 8× |
| pikeperch | 2.04 Mt | 4–25 kt | 82× |
| smelt | 175 kt | 20–120 kt | 1.5× |
| stickleback | 62 kt | 50–500 kt | OK |

The RV-wrap window mismatch (calibration final decade in the low-RV trough, 50-yr
cert wrapping into high-RV years) is **not** the cause — cod_east is 60× over at
40 yr too. The DE's own optimum let cod_east **boom** (not collapse); it never
suppressed the eastern stock.

## Why — the apex-predator-release finding

The pre-split baseline's sp1-7 params were tuned with a **full cod apex predator**
(~150 kt) cropping the prey field. Disaggregating cod and suppressing the eastern
stock removes that top-down control, so the prey (pikeperch, flounder, perch,
sprat) are released far above their ICES envelopes. The objective then faces a
**tension**: hitting cod_east's low ~70 kt target releases the prey and *worsens*
the prey fits. The DE resolved that tension by NOT suppressing cod_east — booming
everything to a mediocre-but-balanced obj 12.34. A hand-built warm-start that
forces cod_east down (baseline sp1-7 + suppressed cod_east) scores **1817** —
confirming the two goals fight each other.

**This is a real result about the disaggregated food web: a collapsed eastern-cod
Baltic cannot hold the prey species in their ICES envelopes without the apex
predator — consistent with the prior finding that even the aggregated 8-species
model is only 2/8 stable (see the UQ Baltic validation memory).**

## Options

1. **Proper warm-started re-calibration** (~6–8 h): warm-start all 9 from the
   baseline, `fsh8` clamped to the moratorium (done — forces M+RV, not fishing),
   larger budget (popsize_mult ≥ 6). The DE left prey fishing low (flounder
   F=0.008) despite the boom, so it *may* hold the prey via higher prey F — but
   holding flounder/pikeperch at F≈3 is itself fidelity-questionable.
2. **Accept the structural finding**: keep the disaggregation structure, document
   that a suppressed eastern cod releases the prey field, and treat residual
   envelope misses as the known Baltic limitation rather than chasing 9/9.
3. **Reconsider the target semantics**: the prey booms are partly because the
   model matches total biomass to ICES SSB/biomass targets; a background-predator
   term (seals/cormorants already partially in) or wider prey-mortality bounds
   would give the prey a top-down control that isn't cod.

The pre-split 5/8 baseline (`phase13_equilibrium.json`) is intact and can be
restored to `data/baltic` at any time.
