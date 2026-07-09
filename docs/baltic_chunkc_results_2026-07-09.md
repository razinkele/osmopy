# Baltic Chunk C — clupeid→cod-egg predation results (2026-07-09)

**Harness:** `scripts/baltic_bistability_chunk0.py --chunk-c-strength` + `scripts/chunkc_accessibility.py`
(PR-in-progress; see the spec `docs/superpowers/specs/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation-design.md`
and plan `docs/superpowers/plans/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation.md`).
**Predecessor:** `docs/baltic_chunk0_warmstart_results_2026-07-09.md` (deployed Baltic → MONOSTABLE;
bistability must be CREATED).

**Lever.** Chunk C sets cod-as-prey accessible to herring/sprat (deployed matrix has cod→herring =
cod→sprat = 0). The herring/sprat size-ratio window (`[5,500]`; herring Linf 27 cm, sprat 16 cm; cod egg
0.15 cm) automatically restricts this predation to egg/larval cod (prey window ~0.03–4 cm) — adult cod is
size-inaccessible. This is the Baltic cod↔sprat **cultivation-depensation** predator-pit: a booming
clupeid stock eats the eggs of recovering cod. Applied config-only via a variant
`predation.accessibility.file`; the deployed matrix is never modified.

## 1. De-risk — the mechanism is real and has teeth (in the right regime)

Two A/B checks (Chunk C off vs on at strength X = 0.4, seed 0):

- **Deployed config, larva ×0.1 (cod-dominated):** cod 14.08 Mt → 13.93 Mt (**−1.1%**). Weak — but this is
  the *wrong* regime: cod dominates (14 Mt) and clupeids are the minority (herring 4 Mt, sprat 0.03 Mt), so
  egg predation is a small perturbation. Cod/herring/sprat all responded, confirming the mechanism is wired.
- **Clupeid-dominated warm-start IC (sprat seeded 2.5 Mt), 25 y:**
  - larva ×0.3: cod 10.81 Mt → 9.85 Mt (**−8.8%**);
  - larva ×0.5: cod 7.28 Mt → 4.88 Mt (**−33.0%**), and **sprat released from collapse: 0 → 1.15 Mt**,
    herring 7.6 → 10.2 Mt.

So where clupeids are relevant, Chunk C bites — cod down, clupeids up (the regime-shift direction), and
the effect strengthens toward the deployed larval mortality. The mechanism is not the question; its
*sufficiency* is.

## 2. Strength sweep — NO bistability created at any strength (regime-shift contrast, warm-start, 25 y)

Cod-dominated vs clupeid-dominated standing-stock ICs, swept across the larval-mortality driver, for
X ∈ {0.1, 0.2, 0.4}. Table shows the cod band in each arm (cod-dominated / clupeid-dominated) and the
point outcome:

| larva scale | X = 0.1 | X = 0.2 | X = 0.4 |
|---|---|---|---|
| ×0.03 | overshoot / overshoot · provisional | overshoot / overshoot · provisional | overshoot / overshoot · provisional |
| ×0.10 | overshoot / overshoot · same-basin | overshoot / overshoot · same-basin | overshoot / overshoot · provisional |
| ×0.30 | overshoot / overshoot · provisional | overshoot / overshoot · provisional | overshoot / overshoot · same-basin |
| ×0.50 | overshoot / overshoot · same-basin | overshoot / overshoot · same-basin | overshoot / overshoot · provisional |
| ×1.00 | undet. / collapsed · provisional | undet. / collapsed · provisional | **collapsed / collapsed** · provisional |
| **verdict** | **regime_shift = False** | **regime_shift = False** | **regime_shift = False** |

**The cod-collapse axis never diverges at any (strength, scale).** Cod recovers to overshoot in *both*
arms at low mortality and collapses in *both* at ×1.0 — its fate is set by larval mortality alone, exactly
as in the no-Chunk-C monostable control. The clupeid gap is tiny everywhere (0.008–0.149 vs the 0.5
divergence threshold) and frequently the *wrong* direction (more clupeids in the cod-dominated arm).
Stronger egg predation (X = 0.4) deepens cod's collapse **universally** at ×1.0 (both arms → collapsed)
rather than selectively in the clupeid-dominated arm — the opposite of an alternative stable state.

## 3. ICES calibration check — Chunk C does not help (deployed config, larva ×1.0, 25 y)

ICES bands: cod (60k / 120k / 250k), herring (800k / 1.5M / 3M), sprat (800k / 1.5M / 2.5M).

| | cod | herring | sprat |
|---|---|---|---|
| deployed OFF | 334 (collapsed) | 16.4 M (overshoot) | 6.6 M (overshoot) |
| X = 0.1 | 25 | 16.4 M | 6.7 M |
| X = 0.2 | 33 | 16.4 M | 6.6 M |
| X = 0.4 | 17 | 16.2 M | 6.6 M |

At the deployed rate cod is already collapsed; Chunk C only **deepens** the collapse (334 → 17–33 t) and
leaves the clupeid overshoot (herring ~5–20×, sprat ~2.6× over band) untouched. It does not move the
deployed config toward ICES.

## Conclusion — robust negative; the over-production must be fixed first

**Config-only clupeid→cod-egg predation, at accessibility strengths up to 0.4, does not create a
cod↔sprat regime-shift bistability, and does not improve the deployed calibration.** The lever is real
(the de-risk shows −33% cod + sprat release where clupeids dominate), but it is **insufficient** against
the scale of the problem: cod overshoot is **20–90× the ICES band** (5–22 Mt vs 250 kt), so even a −33%
egg-predation hit only *trims* the overshoot — it cannot flip cod into a collapsed basin. The bottom-up
plankton firehose (all six LTL groups at `accessibility2fish = 0.8`, non-depletable) drives cod
recruitment so hard that top-down egg predation cannot open a predator-pit.

**Next lever (recommended):** layer Chunk C on a community that is *not already in runaway overshoot* —
i.e. combine it with **lever #1 (lower plankton `accessibility2fish` 0.8 → ~0.1)** or **Chunk A2
(depletable plankton)**. Reduce the over-production first so the standing biomasses sit near the ICES
bands, then test whether clupeid→cod-egg predation creates the depensation the deployed (overshooting)
model is too saturated to express. A stronger single-lever test (X ≈ 0.8, near the accessibility ceiling
of 1.0) is possible but low-value: the mechanism-level reason (overshoot ≫ band) predicts it stays a
trim, not a flip.

## Outputs

- `docs/diagnostics/baltic_chunkc_regime-shift_s{0.1,0.2,0.4}.json` (strength sweeps).
- `docs/diagnostics/predation-accessibility-chunkc-s{0.1,0.2,0.4}.csv` (variant matrices; regenerable via
  `scripts/chunkc_accessibility.py`).
