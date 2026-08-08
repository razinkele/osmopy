# Depletable LTL A/B (Phase 1, spec 2026-08-08)

**Arms:** off, on, on-benthoslit · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024]

| species | off mid (t) | on mid (t) | on-benthoslit mid (t) | Δ on vs off | Δ on-benthoslit vs off | gated |
|---|---|---|---|---|---|---|
| cod_west | 14,343 | 11,585 | 8,758 | -19.2% | -38.9% | yes |
| cod_east | 83,000 | 37,003 | 12,474 | -55.4% | -85.0% | yes |
| herring | 2,591,007 | 526,213 | 294,969 | -79.7% | -88.6% | yes |
| sprat | 1,060,584 | 721,436 | 779,181 | -32.0% | -26.5% | yes |
| flounder | 40,502 | 25,841 | 20,467 | -36.2% | -49.5% | yes |
| perch | 45,089 | 18,293 | 11,558 | -59.4% | -74.4% | yes |
| pikeperch | 1,400,444 | 352,386 | 277,451 | -74.8% | -80.2% | tracked only |
| smelt | 682,441 | 663,520 | 680,861 | -2.8% | -0.2% | tracked only |
| stickleback | 77,578 | 60,542 | 73,821 | -22.0% | -4.8% | yes |

**GATE [off]: PASS** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [on]: FAIL (cod_east, herring, sprat)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)

**GATE [on-benthoslit]: FAIL (cod_east, herring, sprat, flounder)** (required: cod_west, cod_east, herring, sprat, flounder, perch, stickleback)
