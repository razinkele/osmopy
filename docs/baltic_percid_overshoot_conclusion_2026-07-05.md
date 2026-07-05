# Baltic percid overshoot — final conclusion

- **Date:** 2026-07-05. **Scope:** the definitive synthesis of the Baltic OSMOSE perch (`sp4`) /
  pikeperch (`sp5`) population overshoot investigation. Supersedes the per-lever verdicts by tying them
  together; the foundational diagnostic is `docs/baltic_percid_overshoot_diagnostic_2026-06-03.md`
  (master `9dad350`).
- **One-line verdict:** Baltic percid overshoot is **structural** (coarse-grid habitat carrying capacity),
  not fixable from the recruitment side *or* the mortality side without destabilizing the system and
  harming the high-weight stocks. **Eight levers are now ruled out. Accept the overshoot** — the weighted
  calibration objective correctly tolerates these two weight-0.2 species. Run Baltic at `nyear=15`.

---

## 1. The problem

The Baltic config exhibits population-level percid overshoot of **×38–96** vs the ICES/HELCOM reference
envelope (perch worse than pikeperch). The strict in-range count is capped at 1–2/8 species *because of*
the percids. The question pursued across many sessions: is there any mechanism that damps percid biomass
into range without breaking the rest of the food web?

## 2. Eight levers, all ruled out

| # | Lever | Date · ref | Outcome |
|---|---|---|---|
| 1 | **Parameter recalibration** (SP-A) | 07-01 · `0d4a655` | Params alone can't stabilize — 0/8 in-envelope; Python+Java agree. |
| 2 | **Grid refinement** (SP-B, 2× finer) | 07-02 · `3b3cde5` | 2× finer grid does **not** cure ×38–96; population-level, not a resolution artifact at 2×. |
| 3 | **Salinity-correct spawning areas** | 07-02 · `c602717` | Cod overshoot 63.6→63.7× — no effect. |
| 4 | **Spatial egg-survival** (SP1/SP1b) | 07-02/03 · PR #97 | Larval-M recal mean-neutral, but the spatial gate made overshoot 1.43→1.88 (**31% worse**). |
| 5 | **RV recruitment gate** | 07-02 · `756dc0b` (PR #96) | Config-gated egg multiplier by reproductive volume — does **not** damp overshoot in either mode. |
| 6 | **Recruitment ceiling** (McGregor 2019) | 07-04 · `66b374f` | Unfished-level cap — cod overshoot 1.714→1.733 (**slightly worse**). |
| 7 | **Thermal year-class gating** (Pekcan-Hekim/Olin) | 07-05 · `72a8356`+`5df2291` | Real CMEMS `thetao` A/B: perch −3.9%, pikeperch −15.9% — **not recruitment-limited** (see §3). |
| 8 | **Density-dependent cannibalism** | 07-05 · diagnostic | Already in the model; strengthening 10× → perch −1.4% (destabilizes), pikeperch −11% — **not mortality-fixable** (see §3). |

All ship inert / were reverted; the default Baltic config is unchanged.

## 3. The decisive evidence — two independent probes converge

Levers 7 and 8 were run **specifically to test the two opposite hypotheses** for what limits percid biomass,
and they converge on the same answer.

**Recruitment side (thermal gate, real CMEMS `thetao`).** The gate cut perch egg production to **~1–18% of
baseline every year** (perch's real Jun–Jul coastal index, 11.7–15.7 °C, sits well below the year-class
threshold). Yet perch biomass moved only **−3.9%**. Gutting recruitment barely touches the population →
**perch overshoot is not recruitment-limited.** (Pikeperch −15.9%, modest, far short of range.)

**Mortality side (cannibalism A/B).** Percid cannibalism is already modelled (self-accessibility 0.05, and
the predator/prey size windows — perch 3–50×, pikeperch 2.5–30× — let adults eat juvenile conspecifics), and
OSMOSE predation is inherently density-dependent. Strengthening it **10× (0.05→0.5)** moved perch only
**−1.4%** and *destabilized* it (boom/bust 76→2591). Adding density-dependent mortality barely touches perch
either → **perch overshoot is not mortality-fixable.**

Neither the recruitment tap nor the mortality tap moves perch. What's left is the **standing stock the
coarse-grid habitat will support** — i.e. carrying capacity. This matches the 2026-06-03 β-probe, where even
maximum recruitment suppression (Shepherd β=5.0) floored perch at **×107** (from ×166), unable to reach range.

## 4. Why it's structural, and why forcing it is harmful

- **Footprints are already confined** (2026-06-03): perch adult occupies 62/616 ocean cells (10%),
  pikeperch 27 (4.4%). Map-concentration is a dead end — they are not over-broad.
- **Coarse-grid carrying capacity.** In the aggregated grid, each coastal percid cell represents a large
  area whose modelled prey/space supports far more percid biomass than the real, patchy littoral does. The
  overshoot is the model's habitat capacity, not a rate error. SP-B showed 2× refinement is insufficient to
  dissolve it.
- **The freed-prey side effect.** Suppressing percids releases their prey and **worsens the high-weight
  stocks**: at β=5.0 the 2026-06-03 probe drove cod ×33→×37.5, herring ×2.3→×3.9, sprat ×5→×6.1; the
  cannibalism A/B showed the same signature milder (cod −7.4%). Any lever strong enough to dent percids
  perturbs the *sound* cod/herring/sprat fit.
- **Over-compensation / paradox of enrichment.** Hard capping blows up inter-annual CV (perch 0.03→0.42,
  pikeperch 0.07→0.77 at β=5.0; boom/bust 76→2591 under strong cannibalism). Perch is a chaotic,
  near-extinction-prone tiny population (~16 t) whose response to any strong lever is dominated by
  stochastic collapse, not a clean signal (its measured response swings −1.4% → −33% on small parameter
  changes).

**Pikeperch is a partial exception.** It is somewhat recruitment/mortality-addressable (thermal −16%,
cannibalism −11%, and DE left its Shepherd β stuck at 0.50 under-compensation), so a *moderate* pikeperch-β
increase inside a re-weighted calibration is an easy partial win — **but only guarded against the freed-prey
harm to high-weight species, and it still won't reach range.**

## 5. What is NOT worth building (with reasons)

- **A per-species spatial carrying-capacity cap** (density-dependent local mortality scaled to habitat
  capacity) — the "clean" fix for perch. Deferred 2026-06-03 and reaffirmed here: medium engine effort +
  a multi-hour recalibration, headline payoff is two weight-0.2 species, real risk to the high-weight fit,
  and perch would still floor above range.
- **Type-III (sigmoidal) cannibalism** via the already-merged predator-FR engine (PR-A, `e1f4173`). The
  cleanest density-dependent self-limiter in principle (bites only at high percid density), but it hits the
  same structural wall (perch CC-limited), needs the never-done PR-B calibration, and carries the same
  freed-prey risk. Poor value-per-effort.
- **A strict-in-range-count objective term** — would weight grid-under-resolved weight-0.2 pikeperch equal
  to weight-1.0 cod and force the destabilizing behaviour. The weighted objective correctly tolerates the
  percid overshoot.

## 6. Recommendation

**Accept the Baltic percid overshoot.** It is a structural consequence of coarse-grid coastal habitat
capacity, not a rate/recruitment/mortality error, and no lever removes it without harming the sound
high-weight (cod/herring/sprat) calibration. The weighted objective already treats it correctly. Run Baltic
at the short horizon (`nyear=15`).

The only avenue that even *targets* the true cause is a **substantially finer coastal grid** (well beyond the
2× SP-B tried), which is a large, separate undertaking with its own forcing/mapping cost — pursue only if
coastal percid realism becomes a first-class project goal in its own right, not as an overshoot patch.

---

### References (levers + foundational diagnostic)

- `docs/baltic_percid_overshoot_diagnostic_2026-06-03.md` (master `9dad350`) — the β-probe + footprint + freed-prey diagnostic this synthesis builds on.
- Thermal gate: `docs/superpowers/specs|plans/2026-07-05-baltic-percid-thermal-recruitment-gate*.md`; real-field A/B `5df2291`.
- Recruitment lit review: `docs/baltic_recruitment_literature_review_2026-07-03.md`.
- Percid low-salinity refuge review: `docs/baltic_percid_low_salinity_refuge_literature_review_2026-07-04.md`.
- Scientific grounding (thermal): Pekcan-Hekim et al. 2011, *Ambio* (doi:10.1007/s13280-011-0143-7); Olin et al. 2019, *Hydrobiologia* (doi:10.1007/s10750-019-04008-z).
