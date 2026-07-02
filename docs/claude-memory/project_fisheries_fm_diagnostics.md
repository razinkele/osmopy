---
name: project_fisheries_fm_diagnostics
description: Fishing-vs-natural mortality (F/M) diagnostics SHIPPED to origin/master 2026-06-03; rescoped from full Kobe/B-Bmsy after in-loop review killed the marquee.
metadata: 
  node_type: memory
  type: project
  originSessionId: d8b85e1b-0860-44a3-8a89-b2d35b87a866
---

Per-species **F/M (fishing vs natural mortality) diagnostic** MERGED + PUSHED to origin/master 2026-06-03 (`9dad350..6d8b2f5`, branch deleted). Built via brainstorming→writing-plans→subagent-driven (FM1–FM7). Spec `docs/superpowers/specs/2026-06-03-fisheries-diagnostics-design.md`, plan `…/plans/2026-06-03-fisheries-diagnostics-plan.md`, feature doc `docs/baltic_example.md`.

Deliverables: `osmose/validation/fisheries.py` (`read_mortality`, `annual_rate`, `MortalityBalance`, `discover_species`, `compute_mortality_balance`, `format_mortality_report`), `osmose/plotting.py::make_fm_ratio_bars`, `scripts/compute_mortality_balance.py` CLI, + a fix to `osmose/results.py` (mortalityRate reader). Real-data result (baltic): sprat F/M=1.79 (overexploited ✓ — matches reality, sprat is the major Baltic fishery), herring 0.90, others 0.12–0.31. No engine changes, no calibration runs.

## RESCOPED after a 4-angle in-loop plan review (the key episode)
First draft aimed at full stock-status: F/M + **B/Bmsy + F/Fmsy + Kobe plot**. The review (verified against real code/data) killed the marquee and corrected 3 FALSE reconnaissance premises — **because the recon subagent INFERRED data formats from headers instead of EXECUTING the readers.** Lesson: reconnaissance must run the actual readers, not infer. The 3 BLOCKERs:
1. `results.mortality()` CRASHED with `pandas.errors.ParserError` on the real `mortalityRate-{sp}` CSV (1 preamble line + 2 header rows (cause,stage) + trailing comma = 25 vs 26 fields). Pre-existing bug. FIXED: route mortalityRate through `pd.read_csv(skiprows=1, header=[0,1])` + drop all-NaN trailing col → (cause,stage) MultiIndex, gated on a 2-row-header detector so non-mortality outputs are untouched. (Also fixed a latent circular import osmose.results↔engine.output by deferring an import.)
2. ICES reference points are JSON STRINGS (`"fmsy":"0.34"`) → arithmetic needs float() coercion (moot now — ref-point math was dropped).
3. B/Bmsy+F/Fmsy+Kobe cover only **sprat** on Baltic (flounder=index, herring/cod=mixed index+tonnes, coastal=none). Single-point Kobe. Plus MSY-Btrigger (≈Bpa < Bmsy) as Bmsy proxy OVERSTATES health; Recruits-stage F ≠ ICES age-ranged Fbar. → DEFERRED (documented follow-up): needs a config with broad ICES tonnes coverage + defensible Bmsy + Fbar-aligned F.

## CRITICAL F/M science finding (don't regress)
**F/M must be computed on the EXPLOITED life stage(s), not "Recruits" and not all-stages-summed.** Two wrong iterations caught by the real-run smoke:
- Recruits-only → F=0 for EVERY species (OSMOSE Baltic applies fishing to the **Pre-recruits** stage; F,Recruits genuinely 0). Useless.
- All-stages-summed → M swamped by the Recruits stage's huge starvation mortality (sprat Recruits Mstarv≈3.0/yr on UNfished adults) → F/M≈0.07 everywhere. Uninformative.
- CORRECT: per-stage windowed-annual F & M; fished_stages = stages with annual F>tol; F & M summed over fished stages only (unfished → M over Pre-recruits+Recruits, F=0). Gives sprat Pre-recruits F=0.237/M=0.133 → **F/M=1.79**. F/M is a model-internal pressure ratio on the exploited cohort, NOT an ICES Fbar (documented).

## Gotchas (carry forward)
- `steps_per_year` for annualizing per-step mortality = `ndtPerYear / output.recordfrequency.ndt` (=1 for shipped baltic/eec where both=24). **Never infer from row counts** (biomass & mortality save at the same cadence → equal lengths → ratio always 1, silently ~N×-wrong F if recordfreq<ndtPerYear). CLI `_resolve_steps_per_year` reads config (exact-key match to avoid `output.restart.recordfrequency.ndt` shadow; handles trailing-comma values), loud-warns + defaults 1 if underivable.
- mortalityRate CSV per-stage columns: causes `Mpred/Mstarv/Madd/F/Zout/Mfor/Mdis/Mage` × stages `Eggs/Pre-recruits/Recruits`; Time col label is `("Time","Unnamed: 0_level_1")`.
- A malformed mortality file raises `ParserError` (NOT a ValueError) — compute's try/except catches `(KeyError, ValueError, pd.errors.ParserError)`.

See [[project_predator_functional_response]], [[project_percid_overshoot_diagnostic]] (same session's prior Baltic work).
