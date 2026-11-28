# Baltic model vs ICES assessments and advice — 2026-11-28

**Model:** master = v1.22.0, `data/baltic` as shipped, Python engine, 50 yr × seeds 1, 2, 3;
model means over the last 5 simulated years. Verdicts identical across seeds (values within a few %).

**ICES side:** `data/baltic/reference/ices_snapshots` (created 2026-11-06, advice year 2026),
window 2020–2024 (cod.27.22-24 ends 2022 → 2020–2022). The live ICES MCP returns OLDER
assessments (e.g. sprat ends 2023 vs 2026 in the snapshot), so the snapshots are the latest.

**Tool:** `PYTHONPATH=. python scripts/validate_outputs_vs_ices.py --prefix baltic
--results-dir <run> --ices-window 2020-2024 --sms-m2` (note: the Baltic output prefix is
`baltic`, not the default `osm` — with the default every species is silently skipped).

## SSB vs ICES

| species | model SSB (t) | ICES SSB envelope (t) | verdict |
|---|---:|---:|---|
| sprat (spr.27.22-32) | ~798k | 635k – 1.12M | ✓ in range, 0.95× |
| cod_west (cod.27.22-24) | 15.8–17.1k | 5.3k – 9.1k | ✗ 2.3–2.5× high |
| herring | ~1.39M | 781k – 813k* | ✗ 1.73–1.79× (*partial envelope) |
| cod_east, flounder | 66–68k / 42k | index units | not comparable |
| perch, pikeperch, smelt, stickleback | 33k / 380k / 572k / 96k | no ICES assessment | — |

\* The herring envelope sums only the tonnes-unit stocks (SD 28, 30–31, 20–24); the central
Baltic stock her.27.25-2932 is index-only and excluded, so the real ICES total is larger and
1.73× overstates the gap.

## Catch and F (5-yr means)

| species | model catch | ICES catch 2020–24 | model Y/B | ICES F / Fmsy |
|---|---:|---:|---:|---|
| sprat | 292 kt | 273 kt | 0.29 | 0.38 / 0.34 — consistent |
| herring (all stocks) | ~440 kt | ~229 kt | 0.29 | — model catch ~1.9× ICES |
| cod_west | 0.45 kt | 1.3 kt | 0.02 | 0.90 / 0.26 |
| cod_east | 0.57 kt | 1.5 kt | 0.01 | no reference points |
| flounder | ~39 kt | ~9.5 kt (4 stocks) | 0.82 | ~4× ICES catch |

Y/B uses biomass ≥ cutoff age (0.5 yr), so it is only a rough proxy for F.

## WGSAM SMS cod-predation M2 (report-only)

| prey | SMS M2 | model cod M2 | model/SMS |
|---|---:|---:|---:|
| herring | 0.074 | 0.037 (cod_west 0.010, cod_east 0.027) | 0.51 |
| sprat | 0.091 | 0.149 (cod_west 0.016, cod_east 0.133) | 1.64 |

## Reading

1. **Sprat is the only stock consistent with ICES on SSB, catch and F.**
2. **cod_west: SSB 2.4× high while catch is a third of ICES** (Y/B ≈ 0.02 vs ICES F ≈ 0.9).
   One-sided gap → fishing pressure is likely too light; first lever to examine.
3. **Herring and flounder catches exceed ICES** (~2× and ~4×), consistent with the known
   flounder base-F gap (6.4× the ICES anchor) and the large herring biomass.
4. **Advice context:** western Baltic cod is below Blim (15 kt) in ICES with zero-catch advice;
   the model's 16–17 kt sits just above Blim, below Bpa (23.5 kt) — healthier than ICES.

Only 4 of 9 focal species have tonnes-unit ICES SSB, so "1/3 in envelope" is a narrow test.
