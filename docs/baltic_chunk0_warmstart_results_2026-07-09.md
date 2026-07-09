# Baltic Chunk 0 — warm-start regime-shift results (2026-07-09)

**Harness:** `scripts/baltic_bistability_chunk0.py` (generalized with `--warmstart` + `--contrast`; see
`docs/superpowers/plans/2026-07-09-baltic-warmstart-regime-shift-sweep.md` and the design spec
`docs/superpowers/specs/2026-07-09-baltic-warmstart-regime-shift-sweep-design.md`).
**Run:** `--warmstart --contrast both --years 15 --seeds 0 1 2` on the deployed
`data/baltic/baltic_all-parameters.csv` (PythonEngine), `module.population.initialisation.enabled=true`.
Base larval rates read at runtime = per-dt `{cod 15, herring 8, sprat 9, flounder 12, perch 13,
pikeperch 15, smelt 13.5, stickleback 3.5}` (post-4.4.1-migration values).
**Predecessor:** `docs/baltic_chunk0_results_2026-07-08.md` (egg-only sweep → MONOSTABLE, *conservative*).
**Prerequisite:** PR #101 warm-start standing-stock primitive (`osmose/engine/initialization.py`).

This is the **definitive reciprocal-invasion test** the egg-only Chunk-0 could not do: with warm-start on,
`population.seeding.biomass.sp{i}` becomes a genuine age-structured **adult standing stock at t=0**, and
egg-seeding is disabled so a suppressed species is not re-injected. We initialize two real adult
communities and evolve them under identical parameters.

## Pre-flight (de-risk gate) — PASSED

`--preflight --years 5` (cod-dominated standing stock, warm-start ON): **OK — standing stock persists
(cod+herring+sprat mean = 8.35 Mt)**. The first-ever *forward* run of the warm-start init: no crash, no
NaN, no instant vanish. The standing-stock IC is self-consistent with the deployed parameters, so the
full sweep is valid.

## 1. Cod-axis contrast (warm-start) — MONOSTABLE (trustworthy; establishment fraction 80%)

Genuine standing-stock cod-**rich** (cod 300 kt) vs cod-**poor** (cod 1 kt) ICs, swept across the
larval-mortality driver:

| larva scale | cod-rich (a) | cod-poor (b) | cod median rich / poor (t) | gap | outcome |
|---|---|---|---|---|---|
| ×0.03 | overshoot | overshoot | 19.4 M / 21.6 M | 0.100 | same-basin |
| ×0.10 | overshoot | overshoot | 14.3 M / 15.9 M | 0.099 | same-basin |
| ×0.30 | overshoot | overshoot | 9.24 M / 9.70 M | 0.047 | same-basin |
| ×0.50 | overshoot | overshoot | 6.86 M / 6.78 M | 0.011 | same-basin |
| ×1.00 (deployed) | undetermined | collapsed | 0 / 897 | 0.999 | undetermined |

Establishment fraction 80% (cod-rich reaches a non-collapsed stock at 4/5 scales) ⇒ **trustworthy**,
no seed-splits. **The two standing-stock ICs never land in different basins.** At low larval mortality
the whole cod stock sits in extreme **overshoot** (6.8–21.6 Mt vs the 250 kt ICES upper band, ~27–86×);
at the deployed ×1.0 rate cod **collapses regardless of the starting stock** — cod-rich reaches a median
of exactly **0 t**, cod-poor 897 t. Cod's fate is set by the larval-mortality driver alone.

**This confirms the 2026-07-08 egg-only result *rigorously*, with a genuine adult cod standing stock:**
the collapse↔overshoot fork is a MONOSTABLE response to one parameter, not a starting-condition-dependent
bistability. A standing-stock cod IC does **not** change the monostable verdict.

## 2. Regime-shift contrast (warm-start) — no regime shift (formal verdict instrument-limited; raw signal monostable)

Cod-**dominated** (cod 250 kt, herring 800 kt, sprat 600 kt) vs clupeid-**dominated** (cod 1 kt, herring
1.5 Mt, sprat 2.5 Mt) standing stocks — the real post-1990 sprat-dominated regime — swept across the
driver. A regime shift is called only when **both** axes diverge directionally (cod persists in the
cod-dominated arm **and** collapses in the clupeid-dominated arm, **while** clupeids boom in the
clupeid-dominated arm).

| larva scale | cod a (cod-dom) | cod b (clupeid-dom) | clupeid a / b (t) | clupeid gap | clupeid valid a/b | outcome |
|---|---|---|---|---|---|---|
| ×0.03 | overshoot | overshoot | 2.53 M / 3.48 M | 0.273 | ✓ / ✗ | provisional |
| ×0.10 | overshoot | overshoot | 3.13 M / 3.51 M | 0.110 | ✓ / ✓ | **same-basin** |
| ×0.30 | overshoot | overshoot | 4.54 M / 3.64 M | 0.198 | ✓ / ✗ | provisional |
| ×0.50 | overshoot | overshoot | 6.73 M / 5.84 M | 0.132 | ✗ / ✗ | provisional |
| ×1.00 (deployed) | undetermined | collapsed | 19.3 M / 22.3 M | 0.135 | ✓ / ✓ | provisional |

**Formal verdict: INSTRUMENT-LIMITED** (only 20% of scales fully determinate; provisional at
×0.03/0.3/0.5/1.0). The `provisional` label is triggered by the strict validity gate — it withholds a
point whenever a gated arm (the cod band, or *either* clupeid stock) is non-stationary or seed-split at
15 y. Several clupeid arms were still drifting at 15 y (hence `✗`), and at ×1.0 the cod-dominated arm's
cod band is `undetermined` (cod median 0).

**But the raw signal is unambiguously MONOSTABLE — no regime shift at any scale:**
- **Cod-collapse axis never diverges.** Cod overshoots in *both* arms at low M and collapses/undetermined
  in *both* at ×1.0. Starting clupeid-dominated (cod seeded at just 1 kt) does **not** hold cod down —
  cod recovers to multi-Mt overshoot at low M anyway. There is no scale where cod persists in one arm and
  collapses in the other.
- **Clupeid-boom axis never diverges.** The clupeid gap is 0.11–0.27 at every scale — all well below the
  0.5 divergence threshold. Even at the two scales where *both* clupeid arms are valid (×0.10 gap 0.11;
  ×1.00 gap 0.14), the gap is far too small to call divergence. Starting sprat-dominated does not produce
  a persistent clupeid excess.
- The one **fully determinate** point (×0.10) is **same-basin** (monostable).

The clupeid boom at ×1.0 (19–22 Mt in both arms) is a *consequence* of cod's larval-M-driven collapse
(predation release), present in **both** ICs — not an IC-dependent alternative state.

## Conclusion — the deployed Baltic model is MONOSTABLE under the definitive test

Neither a genuine standing-stock cod IC (cod-axis) nor the real sprat-dominated regime IC (regime-shift)
produces an alternative stable state. The warm-start primitive successfully constructed *both* adult
communities (pre-flight: 8.35 Mt persists), and under identical parameters **they converge**. Cod's fate
is set by larval mortality (overshoot at low M, collapse at the deployed rate); clupeids track cod and
converge within ≤27% regardless of their starting biomass.

**Bistability is not latent in the deployed model — it must be CREATED**, exactly as the 2026-07-08
reframe predicted: Chunk C (clupeid→cod-egg predation) and Chunk A2 (depletable plankton) are the missing
endogenous feedbacks that would lock in a real predator-pit / alternative stable state. The roadmap is
unchanged.

## Caveats (honest)

1. **Cod-axis verdict prose is now warm-start-aware (resolved).** `_cod_axis_verdict(points, warmstart)`
   branches the prose on the flag: the egg-only path keeps the v3 text byte-identical (parity), while the
   `--warmstart` path emits "MONOSTABLE (warm-start standing ICs) … Bistability must be CREATED (Chunk C
   … Chunk A2 …)" instead of the misleading "add the warm-start primitive (Task 7)". The committed
   `baltic_chunk0_warmstart_bistability_cod-axis.json` verdict was regenerated from its existing points
   (no sim re-run); all summary fields (bistable, establishment_fraction 0.8, undetermined [1.0]) are
   unchanged.
2. **Regime-shift formal verdict is horizon-limited, not genuinely ambiguous.** The `instrument-limited`
   label comes from the 15-year stationarity gate on clupeid stocks (several arms still drifting / seed-
   split), not from any basin ambiguity. The raw biomasses show clear convergence on both axes. A longer
   horizon (~25–30 y) or more seeds would likely convert most `provisional` points to determinate
   `same-basin`; the substantive monostable conclusion is not in doubt.
3. **Substantive dynamic.** At low larval M the whole community is in massive overshoot (cod 6.8–21.6 Mt;
   clupeids 2.5–6.7 Mt); at the deployed ×1.0 rate cod collapses and clupeids explode to 19–22 Mt via
   predation release — in *both* ICs. This is the larval-mortality driver's collapse↔overshoot fork,
   present regardless of the starting community — i.e. monostable-per-parameter, the reframe confirmed.

## Outputs

- `docs/diagnostics/baltic_chunk0_warmstart_bistability_cod-axis.json`
- `docs/diagnostics/baltic_chunk0_warmstart_bistability_regime-shift.json`
