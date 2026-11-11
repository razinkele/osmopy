# Phase-14 FR half-saturation identifiability: result (issue #232), 2026-11-11

**Verdict, by the pre-stated rule: INCONCLUSIVE.** The rule was committed before the runs, in
`phase14_fr_identifiability_prestated_2026-11-11.md`. The cod_west positive control failed its
gate, so no verdict is drawn on GreySeal or Cormorant, and phase 14's free set stays at four.

The runs show something more basic: **phase 14's frozen base is not a viable ecosystem on the
post-split config.** In all 24 runs, whatever the treatment, cod_west sits at about 1.7 Mt, sprat
at about 6.4 Mt, and cod_east and flounder at exactly 0 or a few tonnes. Every objective
difference measured here is a stock crossing that extinction edge. None of it is the knob's
information content in a working model.

Harness: `scripts/phase14_fr_identifiability.py` at `163da593`. Raw output:
`phase14_fr_identifiability_2026-11-11.json`. Runs: 24 × 40 yr, 12 single-threaded workers, 422 s.

## Primary instrument: the phase-14 objective

The criterion is identifiable ⇔ |mean paired Δ| > 2 × the pooled across-seed SD.

| predator | mean Δ (K 5.0 − 0.5) | pooled seed SD | ratio | criterion |
|---|---:|---:|---:|---|
| **cod_west (control)** | −425.4 | 263.3 | **1.62** | **fails → gate fails** |
| GreySeal | −976.9 | 0.97 | 1011 | passes |
| Cormorant | −308.2 | 3.36 | 92 | passes |

## What the runs are (final-decade mean biomass, t; 4 seeds per row)

| arm | objective | cod_west | cod_east | sprat | flounder |
|---|---|---|---|---|---|
| cod_west K 0.5 | 1025.8 ×3, 736.3 | 1.72–1.74e6 | 0 | 6.4–6.5e6 | 0 ×3, 0.006 |
| cod_west K 5.0 | 13.3, 699.4–699.7 ×3 | 1.11–1.15e6 | 47,600 (seed 42), 0 ×3 | 7.0–7.2e6 | 7–145 |
| GreySeal K 0.5 | 1025.8 ×4 | 1.68–1.70e6 | 0 | 6.4–6.5e6 | 0 |
| GreySeal K 5.0 | 47.0–50.2 | 1.68–1.70e6 | 5.8–8.5 | 6.4–6.6e6 | 41–52 |
| Cormorant K 0.5 | 1025.8 ×4 | 1.69–1.70e6 | 0 | 6.4–6.5e6 | 0 |
| Cormorant K 5.0 | 714.3–724.7 | 1.67–1.71e6 | 0 | 6.4–6.5e6 | 0.04–0.22 |

For scale, the certified production config holds cod_west at 15–18 kt (envelope 4–25 kt),
cod_east at 76–80 kt, sprat at 0.88–0.90 Mt and flounder at 48–50 kt. The phase-14 base is about
100× high on cod_west and has lost cod_east and flounder.

- **Why the control failed.** cod_west's K flips cod_east between 0 and 47.6 kt in one seed out of
  four (objective 13.3 against ~700 for the others). That bimodality inflates the seed SD.
  Per-stock, cod_west's K is plainly visible: its own biomass moves at 43× its seed SD, and herring
  and sprat at 10–13×. The pre-stated gate looks only at the objective, so it fails as written.
- **Why GreySeal and Cormorant score so high.** At K = 0.5 every arm loses flounder and cod_east,
  and the objective saturates at the extinction-class value 1025.8. At K = 5.0 GreySeal keeps
  41–52 t of flounder and 6–9 t of cod_east, and the objective drops to ~48. The 1000σ "effect" is
  a few tonnes either side of zero. It does not show how much K constrains a calibration in a
  model whose stocks exist.
- **Not measured: the cause of the non-viable base.** It is recorded as a fact. The base freezes
  parameters fitted before the cod split, before the GreySeal column, before the LTL correction and
  before the 2026-11-08 larval re-fit, which it overrides. Any of these could be responsible; none
  was tested.

## Deviation from the pre-stated protocol

The upper K ran at **4.999999999999999**, not 5.0. `10 ** log10(5.0)` is 5.000000000000001, which
the engine's [0.1, 5.0] half-saturation check rejects. The first attempt died on exactly that, and
none of its output was used. 4.999999999999999 is one ULP down and the largest value the engine
accepts. The same defect sat in `get_phase14_params` itself: a DE candidate on the upper bound
failed its simulation and scored the 1e6 sentinel. It is fixed alongside this note
(`_log10_inside`, test `test_phase14_bound_corners_pass_the_engine_halfsat_check`).

## What this does and does not say about #232

- **Untested here: #232's premise** that seals are food-saturated on the *production* config. The
  phase-14 frame is a different model: Shepherd recruitment, frozen phase-13 mortality and fishing,
  and FR on four predators.
- **Not established for any predator: identifiability in phase 14.** The landscape the objective
  would present to DE is dominated by an extinction cliff.
- **The free set is unchanged.** Both decision rows that could apply, gate-fail or "identifiable",
  say keep sp15 and sp16.
- **Phase 14 needs a rebuilt base before any run is meaningful.** When it has one, this harness
  re-runs as is: it now builds its base through the calibrator's own `apply_phase14_base`.
