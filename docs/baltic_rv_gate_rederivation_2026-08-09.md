# Depletable LTL A/B (Phase 1, spec 2026-08-08)

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
