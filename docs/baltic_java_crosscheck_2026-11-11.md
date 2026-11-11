# Baltic Java 4.4.1 cross-check on the corrected food field: 2026-11-11

**Trigger:** v1.9.0 changed the inputs that both engines read:
- regenerated LTL forcing, about 2.5× less food overall and with no empty coastal cells (#234, #236);
- C2: cod_east spawns in SD 25;
- herring's autumn spawning folded into spring;
- larval mortality re-fitted (herring sp1 98.56 → 28.89 /yr, cod_west sp0 243.76 → 250.66 /yr);
- a GreySeal column in the accessibility matrix (#227).

Every number in `docs/baltic_java_crosscheck_2026-11-06.md` was computed on the old inputs, so this
re-runs it.

**Answer: the divergence is narrower, but its pattern is unchanged.**
- **Herring now agrees** (Java/Python 1.01×; it was 0.72×).
- Java still drives **sprat extinct**.
- Every other species still runs **4–36× above Python**, down from 6–96×.
- No conclusion of the 2026-11-06 note changes in kind.

**Setup:**
- Jar: `osmose-java/osmose-4.4.1-jar-with-dependencies.jar`.
- Config: production `data/baltic` at `95baf49b` (v1.9.0), `--params current`, 50 yr.
- Python ran 5 seeds. Java ran once, because 4.4.1 has no numeric seed.
- The Java arm pins off the two Python-only features it cannot run: `ltl.oxygen.benthos.enabled`
  and `movement.salinity.gate.enabled`.
- The staging carries the background-weight units workaround from the 2026-11-06 note, §3.

## Final-decade mean biomass, tonnes

| species | Python (5 seeds) | Java | **Java / Python** | on 2026-11-06 | Java now / Java 11-06 |
|---|---|---|---|---|---|
| cod_west | 14,884–18,316 | 172,316 | **10×** | 27× | 0.48 |
| cod_east | 76,106–79,977 | 789,575 | **10×** | 14× | 0.88 |
| herring | 1,482,700–1,555,568 | 1,534,479 | **1.01×** | 0.72× | 0.83 |
| sprat | 879,100–897,647 | **0** | extinct | extinct | — |
| flounder | 47,637–49,828 | 1,744,862 | **36×** | 96× | 0.55 |
| perch | 60,028–62,912 | 1,017,948 | **17×** | 54× | 0.45 |
| pikeperch | 597,628–619,079 | 2,657,462 | **4.4×** | 6.0× | 0.31 |
| smelt | 648,232–667,980 | 2,455,659 | **3.7×** | 6.1× | 0.59 |
| stickleback | 90,305–102,046 | 2,132,816 | **22×** | 55× | 0.47 |

Ratios are taken against the midpoint of Python's 5-seed range.

- **Python: 5/5 assessed stocks persistent and in envelope.** This is the v1.9.0 certification
  reproduced exactly: the same ranges as `docs/baltic_ltl_recalibration_2026-11-08.md`. Indicative
  tier 1/4: perch, pikeperch and smelt sit above their literature envelopes, and stickleback is
  inside its envelope.
- **Java: 1/5 assessed** (herring only), as on 2026-11-06. All species except sprat persist.
  The script's line "Java cross-check: 1/9 persistent" is the in-envelope count, not persistence.
  The 2026-11-06 note flagged the same wording.

## Reading it

- **Herring is the one stock where the engines now agree.** On the old food field Java's herring sat
  below Python's (0.72×). Python's herring has since been re-fitted down to its 1.5 Mt target, and
  Java's fell less, by 17%, so the two meet.
- **Java's boom shrank with the food.** Every Java equilibrium fell, by 12–69%, and the percids and
  stickleback roughly halved. Python's demersals rose instead: cod_west +25%, flounder +47% and
  perch +46%, from the recalibration. The gap therefore closed from both sides.
- **The sprat extinction is untouched.** It survived the Cormorant fix, the units workaround and now
  2.5× less food. Whatever kills sprat in Java does not depend on the absolute food level.
- **Attribution is not measured.** Six input changes landed together, listed under Trigger. They
  were not separated on the Java arm. One mechanism is plausible: Java's demersals and percids are
  food-limited, so cutting the food cut them. It is a hypothesis, not a result.

## Reproduce

```
PYTHONPATH=. .venv/bin/python scripts/baltic_stability_certify.py --params current --java \
    --out docs/baltic_java_crosscheck_2026-11-11.md
```

Start 19:50:13, end 20:14:06 (24 min, Python and Java together).

## Appendix: generated certification output

**Params:** current · **horizon:** 50 yr · **seeds:** [42, 123, 7, 999, 2024] · **seeding:** config default

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.43e+04 | [14884.304470914285, 18316.011739530844] |
| cod_east | ✓ | ✓ | 5.82e+04 | [76106.50020646442, 79976.78794456266] |
| herring | ✓ | ✓ | 1.30e+06 | [1482700.0836698213, 1555568.044812445] |
| sprat | ✓ | ✓ | 8.25e+05 | [879100.7370015759, 897647.2158901148] |
| flounder | ✓ | ✓ | 4.59e+04 | [47636.68919977022, 49827.814263464614] |
| perch | ✓ | ✗ | 5.73e+04 | [60027.52990427827, 62912.2343908112] |
| pikeperch | ✓ | ✗ | 5.13e+05 | [597627.8651308322, 619078.6310205504] |
| smelt | ✓ | ✗ | 6.07e+05 | [648231.93047098, 667979.6603383113] |
| stickleback | ✓ | ✓ | 6.75e+04 | [90305.44129901088, 102045.9672731523] |

Java 4.4.1 (single run): cod_west 172,316 · cod_east 789,575 · herring 1,534,479 (in envelope) ·
sprat 0 (collapse) · flounder 1,744,862 · perch 1,017,948 · pikeperch 2,657,462 · smelt 2,455,659 ·
stickleback 2,132,816.
