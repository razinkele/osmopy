# Depletable LTL A/B (Phase 1, spec 2026-08-08)

**Arms:** off, shift-2, shift00, shift+2 · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024]

| species | off mid (t) | shift-2 mid (t) | shift00 mid (t) | shift+2 mid (t) | Δ shift-2 vs off | Δ shift00 vs off | Δ shift+2 vs off | gated |
|---|---|---|---|---|---|---|---|---|
| cod_west | 12,875 | 12,447 | 12,875 | 13,580 | -3.3% | +0.0% | +5.5% | yes |
| cod_east | 65,209 | 64,301 | 65,209 | 65,364 | -1.4% | +0.0% | +0.2% | yes |
| herring | 2,547,746 | 2,690,044 | 2,547,746 | 2,292,245 | +5.6% | +0.0% | -10.0% | yes |
| sprat | 1,024,567 | 1,028,875 | 1,024,567 | 1,020,292 | +0.4% | +0.0% | -0.4% | yes |
| flounder | 32,937 | 32,841 | 32,937 | 32,249 | -0.3% | +0.0% | -2.1% | yes |
| perch | 43,701 | 42,210 | 43,701 | 44,117 | -3.4% | +0.0% | +1.0% | yes |
| pikeperch | 1,417,535 | 1,440,327 | 1,417,535 | 1,414,829 | +1.6% | +0.0% | -0.2% | tracked only |
| smelt | 683,303 | 681,169 | 683,303 | 679,487 | -0.3% | +0.0% | -0.6% | tracked only |
| stickleback | 81,025 | 64,357 | 81,025 | 97,820 | -20.6% | +0.0% | +20.7% | yes |

**GATE [off]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [shift-2]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [shift00]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [shift+2]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)
