# RV gate re-measured on the adopted config — still load-bearing, binding bound flipped

**Date:** 2026-08-09 (harness output, commit `7642586`; title, provenance and reading added
2026-08-23 — the table is unchanged, and the reading follows that commit's own message).
**Config:** 9-species Baltic master **with** the bottom-O₂ → benthos K coupling (adopted
2026-08-09), 50 yr × 5 seeds, Python engine.
**Harness note:** run through the reusable Phase 1 A/B harness
(`scripts/baltic_depletable_ab.py --skip-default-arms --extra-arm`), hence the arm naming:
**"off" means depletable-LTL off, i.e. the production config with the RV gate ON**
(`raw_cap`, `ref=150`, cod_east/sp8); "gateoff" is the same config with
`reproduction.rv.gate.enabled=false`.
**Supersedes:** the gate-off arithmetic in `docs/baltic_rv_gate_mechanism_ab_2026-08-02.md`,
which was measured pre-adoption.

## Harness output (verbatim)

**Arms:** off, gateoff · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024]

| species | off mid (t) | gateoff mid (t) | Δ gateoff vs off | gated |
|---|---|---|---|---|
| cod_west | 12,875 | 12,522 | -2.7% | yes |
| cod_east | 65,209 | 137,302 | +110.6% | yes |
| herring | 2,547,746 | 2,505,514 | -1.7% | yes |
| sprat | 1,024,567 | 842,458 | -17.8% | yes |
| flounder | 32,937 | 30,407 | -7.7% | yes |
| perch | 43,701 | 43,468 | -0.5% | yes |
| pikeperch | 1,417,535 | 1,395,118 | -1.6% | tracked only |
| smelt | 683,303 | 683,149 | -0.0% | tracked only |
| stickleback | 81,025 | 82,503 | +1.8% | yes |

**GATE [off]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [gateoff]: FAIL (cod_east)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

## Reading

* **The gate remains the dominant control on cod_east.** Gate-off lands at 137,302 t — 1.61× the
  85 kt envelope ceiling. Replacing the prescribed series remains a high-risk swap (and the
  computed-RV replacement was separately withdrawn on other grounds —
  `docs/baltic_rv_divergence_explained_2026-08-10.md`).
* **The O₂→benthos coupling absorbed part of the load.** Gate-off measured 167,377 t
  pre-adoption, 137,302 t post-adoption (−18%).
* **The binding bound flipped.** Pre-adoption the gated stock sat at 82,968 t, 2.4% **under the
  85 kt ceiling**; post-adoption it sits at 65,209 t, ≈8% **over the 60 kt floor** and 23% under
  the ceiling. Every tolerance statement derived from the pre-adoption operating point is void —
  the re-derived band is in `docs/baltic_rv_ref_sweep_2026-08-09.md`.
* **Trophic side-effect worth recording:** the gate-off arm drops sprat −17.8% (1.02 Mt →
  842 kt) — the released cod biomass eats it, so the gate is indirectly propping up sprat as well
  as capping cod.
