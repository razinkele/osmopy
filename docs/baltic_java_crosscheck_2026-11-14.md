# Baltic Java 4.4.1 cross-check on v1.10.1 (sprat matures by age): 2026-11-14

**Trigger:** v1.10.1 (#255) changed sprat's maturity:
- `species.maturity.age.sp2 = 1.0` added and the 9 cm maturity size removed, so sprat matures by age
  only;
- SP1b's cod larval rates re-solved.

Unlike a size + age form, the age-only form is read the same way by both engines. Java 4.4.1 treats
maturity size and age as alternatives (a size wins); with no size, it matures sprat by age, as the
Python engine does.

**Run:** `scripts/baltic_stability_certify.py --params current --java` on `8a578f9e`. Python: 5
seeds × 50 yr. Java 4.4.1: one run, staged the usual way (the benthos-oxygen coupling and the salinity
gate pinned off for the Java arm, background weights rescaled). Verdicts:
- **Python: 5/5 assessed**, identical to #255's certification;
- **Java: 1/5 assessed** (herring), as on v1.10.0.

## Final-decade biomass (t)

| species | Python (5 seeds) | Java | **Java / Python** | on v1.10.0 | Java now / Java v1.10.0 |
|---|---|---|---|---|---|
| cod_west | 18,947–19,398 | 84,263 | **4.4×** | 4.1× | 1.08 |
| cod_east | 72,461–74,220 | 755,610 | **10.3×** | 9.9× | 1.02 |
| herring | 1,520,219–1,547,826 | 1,402,526 | **0.91×** | 0.88× | 1.03 |
| sprat | 996,320–1,009,241 | **0** | extinct | extinct | — |
| flounder | 46,946–48,339 | 1,832,257 | **38×** | 39× | 0.97 |
| perch | 34,052–35,842 | 768,030 | **22×** | 22× | 0.98 |
| pikeperch | 515,161–531,839 | 2,587,668 | **4.9×** | 5.2× | 0.96 |
| smelt | 650,930–665,946 | 2,650,902 | **4.0×** | 4.1× | 0.98 |
| stickleback | 86,330–94,748 | 2,037,020 | **22×** | 22× | 1.04 |

Java / Python is the Java value over the midpoint of the Python range, the convention of the
v1.10.0 note (it reproduces that note's 4.1×, 9.9× and 0.88× exactly).

- **Java still drives sprat extinct,** as it has since v1.9.0. So sprat's maturity change has no
  equilibrium to act on in the Java arm.
- **Every other Java value moved by at most 8%,** and every Java / Python ratio by at most 0.4×.
  Why Java's values moved is not measured; one candidate is sprat's early years, before it dies out,
  when its maturity rule still applies.
- **Herring is still the one stock where the engines agree** (0.91×).
- **The large gaps are the pre-existing ones** (cod_east about 10×, flounder about 38×); nothing in
  #255 addresses them.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --java \
    --out /tmp/baltic_java_crosscheck_certify.md
```

The script overwrites `--out` with its generated certification tables, so write it to a scratch
path, not to this note; the comparison and narrative here were written from that output.

The v1.10.0 note now carries a superseded line.
