---
name: project-python-engine-community-outputs
description: Python engine now writes community DistribBySize + realized 1D meanTL CSVs — SHIPPED 2026-06-17 PR
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c92cf51-9a04-490e-b26d-2e39302eb27f
---

Python-engine community outputs — SHIPPED 2026-06-17, master `a9a3017` (PR #75). **PROD-VERIFIED working 2026-06-18** (live Baltic run at `https://laguna.ku.lt/osmose/`): Sheldon NBSS panel + chart populate (slope ≈ −1.38, 10 bins, size diversity 0.59), MTL = 2.03 (biomass-weighted), ABC W = 0.002, 0 console errors — the exact diagnostics that degraded in the PR #73 prod check now show real data. Closes the gap exposed when verifying [[project-community-size-spectrum-extension]] on prod: the Python engine didn't persist the outputs the Size Spectrum / Sheldon spectrum / MTL-MTI diagnostics read. Pure OUTPUT-LAYER work — no predation hot-loop change.
**Prod-deploy gotcha (2026-06-18):** the first `deploy.sh` run did NOT take — verify it landed by checking the prod clone actually advanced: `git -C /srv/shiny-server/osmose-src log --oneline -1` == GitHub master, `.git/FETCH_HEAD` mtime fresh, and `systemctl show osmose-shiny -p ActiveEnterTimestamp` updated. Don't run the live verification against a stale clone.
**Expected on a short cold-start Baltic run:** MTI shows `n/a` (0/N species ≥ TL 3.25) because realized community TL (~2.0) hasn't climbed past the cutoff yet — NOT a bug; longer/calibrated runs accumulate higher-TL predators.

Two outputs added (`osmose/engine/output.py` builders+writers, wired into `write_outputs`):
- **`biomassDistribBySize`/`abundanceDistribBySize`** — Java community layout `Time, Size, <species>` reshaped from the per-size `StepOutput` dicts the engine already computes (gated by existing `output.{biomass,abundance}.bysize.enabled`). The engine ALSO still writes the per-species `biomassBySize_<sp>` files (different consumers; kept).
- **1D `meanTL`** — `StepOutput.mean_tl` + `_collect_mean_tl` in `simulate.py` aggregating the ALREADY-EMERGENT per-school `state.trophic_level` (maintained by predation in `mortality.py`: `TL = 1 + Σ(prey_TL·eaten)/preyed`). New `output.meantl.enabled` flag.

KEY DECISIONS (review-driven):
- **meanTL is BIOMASS-weighted** `Σ(bm·tl)/Σbm` over focal schools with TL>0 — matches Java `MeanTrophicLevel` (its output header literally says "weighted by fish biomass", confirmed at `data/eec_full/output/Trophic/eec_meanTL_Simu0.csv`). Locked by a DISTINGUISHING unit test (unequal per-school weights → 2.75, not the 3.5 abundance-weighting gives). In-loop review caught the original abundance-weighting default.
- **Config key MUST be lowercase** `output.meantl.enabled` in `config.py` `_enabled(...)` — `osmose/config/reader.py` lowercases ALL keys; mixed case → flag silently always-False AND a config-validation warning. (Blocker caught in review.)
- **Disk + in-memory parity:** both `write_outputs` AND `OsmoseResults.from_outputs`→`_build_dataframes_from_outputs` (`osmose/results.py`) call the same builders; the new keys added to `_CROSS_SPECIES_OUTPUT_TYPES`. The in-memory miss was caught by `test_from_outputs_populates_all_written_keys` (disk-vs-memory guard) during final gates.

Process: brainstorm→spec→plan→4-angle in-loop review (+verify round; 1 blocker + 2 highs fixed PRE-code)→subagent-driven TDD (6 tasks). Spec/plan `docs/superpowers/{specs,plans}/2026-06-17-python-engine-community-outputs*`. Baltic config enables it (`baltic_param-output.csv`). Full suite green except known [[feedback_ci_clean_venv_reproduction]]-adjacent xdist flakes (test_runner/test_study_fullmodel, pass in isolation).
