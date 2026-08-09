# Depletable LTL A/B (Phase 1, spec 2026-08-08)

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
