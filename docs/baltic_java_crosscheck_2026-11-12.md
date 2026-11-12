# Baltic Java 4.4.1 cross-check on v1.10.0 (herring recalibration): 2026-11-12

**Trigger:** v1.10.0 (#250) recalibrated herring:
- maturity size 18 → 14.5 cm;
- adult additional mortality 2.25 → 0.2 /yr;
- larval rate re-fitted, 28.89 → 192.0 /yr;
- SP1b re-solved.

Both engines read the same data, so the v1.9.0 cross-check (`docs/baltic_java_crosscheck_2026-11-11.md`)
is re-run here. The method, jar, pins and background-weight workaround are as in that note.

**Answer: the pattern is unchanged.**
- **Herring still agrees roughly** (Java/Python 0.88×; it was 1.01×).
- **Java still drives sprat extinct.**
- **cod_west's gap narrows from about 10× to 4×.**
- **The rest stay 4–39× above Python.**

**Setup:** config at `05b517f3` (v1.10.0), `--params current`, 50 yr. Python ran 5 seeds; Java ran
once. The run took 26 min.

## Final-decade mean biomass, tonnes

| species | Python (5 seeds) | Java | **Java / Python** | on v1.9.0 | Java now / Java v1.9.0 |
|---|---|---|---|---|---|
| cod_west | 18,713–19,309 | 77,745 | **4.1×** | 10× | 0.45 |
| cod_east | 73,673–75,636 | 742,351 | **9.9×** | 10× | 0.94 |
| herring | 1,509,028–1,586,230 | 1,358,576 | **0.88×** | 1.01× | 0.89 |
| sprat | 900,446–917,592 | **0** | extinct | extinct | — |
| flounder | 47,562–49,132 | 1,887,070 | **39×** | 36× | 1.08 |
| perch | 33,873–36,407 | 787,048 | **22×** | 17× | 0.77 |
| pikeperch | 514,937–528,762 | 2,689,179 | **5.2×** | 4.4× | 1.01 |
| smelt | 655,408–662,902 | 2,716,558 | **4.1×** | 3.7× | 1.11 |
| stickleback | 82,069–98,213 | 1,955,771 | **22×** | 22× | 0.92 |

Ratios are taken against the midpoint of Python's range.

- **Python: 5/5 assessed.** This is #250's certification reproduced exactly; the ranges are
  identical.
- **Java: 1/5 assessed** (herring only), as before.
- **Java's herring took the same recalibration and stayed in its envelope** (1.36 Mt). Herring is
  still the one stock where the engines agree.
- **cod_west fell 55% on the Java side** (172 → 78 kt). Python's cod_west rose slightly (to about
  19 kt). Together that narrows the cod_west gap from 10× to 4×. Why Java's cod_west fell is not
  measured: herring is cod food, and herring's structure changed in both engines.
- **The sprat extinction on Java survives this change too.** It now holds across the Cormorant fix,
  the background-weight workaround, the corrected food field, and the herring recalibration.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --java \
    --out docs/baltic_java_crosscheck_2026-11-12.md
```
