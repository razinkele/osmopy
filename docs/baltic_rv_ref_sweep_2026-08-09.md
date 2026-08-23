# RV gate `ref` sweep on the adopted config — the admissible band, re-derived

**Date:** 2026-08-09 (harness output, commit `7642586`; title, band derivation and reading added
2026-08-23 — the table is unchanged).
**Config:** 9-species Baltic master **with** the bottom-O₂ → benthos K coupling (adopted
2026-08-09), 50 yr × 5 seeds, Python engine. Run through the reusable Phase 1 harness, hence the
header naming: **"off" is the production config** (`reproduction.rv.gate.ref=150`); the refNNN
arms override only `ref`.
**Replaces:** the admissible factor band 0.331–0.449 in
`docs/baltic_rv_gate_mechanism_ab_2026-08-02.md`, which was swept on the pre-adoption config and
is void for this one.

## Harness output (verbatim)

**Arms:** off, ref100, ref120, ref170, ref200 · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024]

| species | off mid (t) | ref100 mid (t) | ref120 mid (t) | ref170 mid (t) | ref200 mid (t) | Δ ref100 vs off | Δ ref120 vs off | Δ ref170 vs off | Δ ref200 vs off | gated |
|---|---|---|---|---|---|---|---|---|---|---|
| cod_west | 12,875 | 13,236 | 13,629 | 12,876 | 13,528 | +2.8% | +5.9% | +0.0% | +5.1% | yes |
| cod_east | 65,209 | 95,298 | 81,660 | 55,539 | 44,561 | +46.1% | +25.2% | -14.8% | -31.7% | yes |
| herring | 2,547,746 | 2,523,119 | 2,558,280 | 2,542,799 | 2,548,062 | -1.0% | +0.4% | -0.2% | +0.0% | yes |
| sprat | 1,024,567 | 943,931 | 988,980 | 1,052,044 | 1,089,913 | -7.9% | -3.5% | +2.7% | +6.4% | yes |
| flounder | 32,937 | 32,645 | 32,804 | 33,331 | 33,830 | -0.9% | -0.4% | +1.2% | +2.7% | yes |
| perch | 43,701 | 43,894 | 42,772 | 43,238 | 43,563 | +0.4% | -2.1% | -1.1% | -0.3% | yes |
| pikeperch | 1,417,535 | 1,367,113 | 1,381,452 | 1,417,122 | 1,379,491 | -3.6% | -2.5% | -0.0% | -2.7% | tracked only |
| smelt | 683,303 | 685,558 | 684,938 | 684,123 | 689,278 | +0.3% | +0.2% | +0.1% | +0.9% | tracked only |
| stickleback | 81,025 | 82,042 | 81,559 | 83,405 | 78,445 | +1.3% | +0.7% | +2.9% | -3.2% | yes |

**GATE [off]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [ref100]: FAIL (cod_east)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [ref120]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [ref170]: FAIL (cod_east)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [ref200]: FAIL (cod_east)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

## The admissible band on the adopted config (derived 2026-08-23)

cod_east biomass is monotone decreasing in `ref` across all five arms, so the envelope crossings
(60,000–85,000 t, `data/baltic/reference/biomass_targets.csv`) can be interpolated linearly
between adjacent arms' mid biomasses:

| edge | crossing | between arms | `ref` at crossing |
|---|---|---|---|
| ceiling (85 kt) | 95,298 → 81,660 t | ref 100 ↔ 120 | **≈115** |
| floor (60 kt) | 65,209 → 55,539 t | ref 150 ↔ 170 | **≈161** |

**Admissible `ref`: ≈115–161.** Production (`ref=150`) sits **7.2% below the floor-side edge** and
23% above the ceiling-side edge (in `ref` units). This is the band already quoted in the
2026-08-09 annotation of `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md`;
the derivation lives here.

In factor terms — the final-decade mean of `clip(rv/ref, 0, 1)` over model years 40–49 (series
years 2014–2020, tail clamped on the 2020 terminal value; `recruitment_gate.py`) — computed
offline from the prescribed series. The factor trajectory depends only on the series and `ref`,
not on the ecology; the computation reproduces the pre-adoption sweep's measured 0.3865 at
`ref=170` exactly:

| point | final-decade mean factor |
|---|---|
| ceiling edge (`ref≈115`) | 0.558 |
| **shipped (`ref=150`)** | **0.438** |
| floor edge (`ref≈161`) | 0.409 |

**Admissible final-decade mean factor: ≈0.409–0.558**, replacing the void pre-adoption band
0.331–0.449. The shipped 0.438 sits **6.7% above the floor-side edge** and 27% below the
ceiling-side edge.

### The asymmetry reversed

Pre-adoption, the shipped factor sat +2.4% below the *ceiling-side* breach and ~25% above the
floor side — the danger was a replacement series running *stronger* than the prescribed one.
Post-adoption the tight edge is the **floor**: a series running *weaker* (factor below ~0.41 in
the scored decade) breaches quickly, so in-sample validation of any replacement must be gated on
the low side first. Note both failure modes stay live: the one computed RV actually measured
(`docs/baltic_rv_divergence_explained_2026-08-10.md`) is flat near ~300 — at `ref=150` its factor
saturates at 1.0, far past the *ceiling* edge — which is one of the reasons that swap was
withdrawn.

### Caveats

* Band edges interpolate **mid** (across-seed) biomasses between adjacent arms; the harness table
  does not retain per-seed spread, so treat the edges as soft — read "≈115–161", not three
  significant figures. The pre-adoption sweep saw seed noise grow toward sweep extremes
  (0.3% at the shipped point, 1–2.3% at the ends); the same caution applies here.
* cod_west responds non-monotonically across arms (+0% to +5.9%) at the level of seed noise; only
  cod_east's response is the signal.
