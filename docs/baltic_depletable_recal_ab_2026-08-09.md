# Depletable LTL A/B (Phase 1, spec 2026-08-08)

**Arms:** off, fitted · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024]

| species | off mid (t) | fitted mid (t) | Δ fitted vs off | gated |
|---|---|---|---|---|
| cod_west | 14,343 | 10,100 | -29.6% | yes |
| cod_east | 83,000 | 24,844 | -70.1% | yes |
| herring | 2,591,007 | 513,719 | -80.2% | yes |
| sprat | 1,060,584 | 780,711 | -26.4% | yes |
| flounder | 40,502 | 19,967 | -50.7% | yes |
| perch | 45,089 | 12,405 | -72.5% | yes |
| pikeperch | 1,400,444 | 240,139 | -82.9% | tracked only |
| smelt | 682,441 | 659,265 | -3.4% | tracked only |
| stickleback | 77,578 | 77,580 | +0.0% | yes |

**GATE [off]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [fitted]: FAIL (cod_east, herring, sprat, flounder)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)
