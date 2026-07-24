# Percids are a proven spatial-resolution limitation (2026-07-24)

The equilibrium-calibrated Baltic Shepherd baseline reaches **5/8 ICES envelopes**
(cod, herring, sprat, flounder, stickleback — all the well-assessed species). The
three residual over-predictions are perch (×5.5), **pikeperch (×217)**, and smelt
(×11). This note records why the percids — pikeperch especially — cannot be brought
into range by defensible tuning on the current 50×40 (~0.4°×0.3°) grid, so that
future work does not re-litigate it.

## Mechanism

Perch and pikeperch are apex coastal predators with (a) very broad, high prey
accessibility (herring, sprat, smelt, stickleback at 0.2–0.6, plus meso/macro-
zooplankton and benthos), (b) almost no predation on themselves (pikeperch
eaten-by-access = 0.15), and (c) implausibly low fishing that the calibration chose
only because their assessment weight is 0.2 (perch F 0.029, pikeperch F 0.0095).
Their equilibrium carrying capacity therefore sits far above the ICES targets, and
the Shepherd over-compensation exponent (pikeperch β = 1.72) cannot claw it back.

## Probes (fresh runs of the committed 5/8 config)

**Fishing** (raise the percid F, `scratchpad/percid_fishing_probe.py`):

| F (perch, pikeperch) | perch ×target | pikeperch ×target |
|----------------------|---------------|-------------------|
| 0.029, 0.0095 (calibrated) | ×5.6 | ×217 |
| 0.35, 0.35 | ×2.7 | ×151 |
| 0.70, 0.70 | **×1.7 (in range)** | ×130 |

→ **Perch IS fixable** by a realistic F (~0.5–0.7; its 0.029 is implausibly low for
a fished species). **Pikeperch is not** — 70× its current F only moves ×217→×130.

**Diet** (trim pikeperch's fish access — herring 0.3→0.1, sprat 0.3→0.05, smelt
0.6→0.25, cod/flounder→0; `scratchpad/pikeperch_diet_probe.py`):

| pikeperch F | full diet | reduced fish diet |
|-------------|-----------|-------------------|
| 0.15 | ×195 | ×157 |
| 0.35 | ×151 | ×134 |

→ **Insufficient.** With the fish diet cut, pikeperch is still ~×134 over, because
its carrying capacity is dominated by **zooplankton + benthos (LTL) access** in its
cells — biomass the coarse grid provides to open-water cells that a real estuary/
lagoon would not. Cutting its plankton/benthos access to force it down would be
fit-chasing an implausible diet, not a model fix.

## Conclusion

> **Correction (2026-07-24, reconciled with the SP-A branch):** an earlier version of
> this note called the pikeperch overshoot a *spatial-resolution* limitation fixable by
> a finer grid. That is **wrong** — the `fix+baltic-salinity-spawning` branch's SP-B
> experiment upsampled the grid 2× (80×100 cells) and the percid/cod overshoot was
> **unmoved** (×38–96), and its salinity-correct-spawning experiment left cod overshoot
> unchanged (×63.6→×63.7). The overshoot is a **population-level** quantity (recruitment
> magnitude × mortality balance): a spawning map or finer grid controls *where* eggs are
> placed, not *how many* recruits the stock-recruitment produces, so no spatial refinement
> can move it. See `docs/baltic_habitat_followup_2026-07-02.md`.

Neither fishing nor a defensible diet correction reaches pikeperch on this grid, and —
per the branch's negative experiments — **neither does a finer grid or a corrected
spawning map**, because the overshoot is recruitment-driven, not spatial. The
mechanistically-correct lever is a **dynamic reproductive-volume recruitment gate**
(cod recruitment gated on deep-basin salinity ≥11 PSU + oxygen ≥2 ml/l), which caps
recruits in low-volume years — the Phase-0 mechanism in
`docs/superpowers/specs/2026-07-24-baltic-stock-disaggregation-design.md`. Perch is
separately fixable via a realistic fishing correction but is a data-poor, weight-0.2
stock; that single-species gain (6/8) was deemed marginal.

**Decision:** bank 5/8 (the well-assessed-species ceiling on this grid) and treat
the percids as a documented, proven limitation. See
`docs/superpowers/2026-07-23-uq-real-data-validation.md` and
`docs/baltic_model_report_2026-07-24.docx`.
