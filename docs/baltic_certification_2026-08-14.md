# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]  ·  **seeding:** config default

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod_west | ✓ | ✓ | 1.14e+04 | [11948.479170111133, 13800.534609436358] |
| cod_east | ✓ | ✓ | 4.73e+04 | [63972.24554234184, 66446.06525502432] |
| herring | ✓ | ✓ | 2.25e+06 | [2489031.5165172284, 2606459.760888887] |
| sprat | ✓ | ✓ | 9.61e+05 | [1013208.2987491975, 1035926.6918703874] |
| flounder | ✓ | ✓ | 2.86e+04 | [31839.751920479448, 34034.62534292196] |
| perch | ✓ | ✓ | 3.81e+04 | [42554.35412166352, 44848.57463315508] |
| pikeperch | ✓ | ✗ | 1.21e+06 | [1375582.0819530422, 1459487.6104924665] |
| smelt | ✓ | ✗ | 6.38e+05 | [672269.4534443791, 694336.4623709053] |
| stickleback | ✓ | ✓ | 6.42e+04 | [77096.56215668567, 84954.15255095414] |

**Python verdict: 5/5 ASSESSED species persistent & in-envelope.** All 5 pass — candidate is certifiable; verify value round-trip before writing data/baltic.

*Indicative tier: 2/4 (perch w=0.2, pikeperch w=0.2, smelt w=0.3, stickleback w=0.2).* These targets are **not ICES assessments** — ICES does not assess Baltic pikeperch, perch, smelt or stickleback. `biomass_targets.csv` sources them as literature estimates at weight ≤ 0.3, noting the coarse grid under-resolves species concentrated in estuaries and lagoons. They are reported for information and are **not** part of the verdict; do not tune against them. (Legacy all-species figure, for comparison with notes written before 2026-08-04: 7/9.)

---

## Stickleback margin, read against the herring-predation finding (2026-08-15)

This certification was run to check one thing the gate does not report: **how much headroom
stickleback has**, now that `docs/baltic_stickleback_mechanism_2026-08-12.md` (Run 8) has
established that its biomass is set by **herring predation on its eggs and young-of-year** —
herring takes 55–63% of stickleback's early-stage deaths, and a ±2-timestep shift in herring
spawning moves stickleback ∓20%.

Nothing was changed to produce this run. The finding is diagnostic; it alters no parameter.

| | value |
|---|---|
| envelope | 50,000 – 500,000 t (width 450,000) |
| certified final-decade mean | 77,097 – 84,954 t (midpoint 81,025) |
| position in envelope | **6.9%** of the way from floor to ceiling |
| headroom to floor | 31,025 t (−38.3%) |
| headroom to ceiling | 418,975 t (+517%) |
| the ±20.7% phenology swing | 64,334 – 97,798 t — **both ends stay in envelope** |
| swing as a fraction of envelope width | 7.5% |

**The sensitivity is absorbed.** A ±20% swing consumes 7.5% of the envelope and does not threaten
the verdict, which is consistent with all three shift arms certifying PASS in
`docs/baltic_herring_phenology_a0_2026-08-12.md`. Breaching the floor would take roughly **1.9×**
the ±2-step perturbation, if the response stays near-linear.

**But PASS here is a weak constraint, and should not be read as the model being well-constrained
on stickleback.** The envelope spans a **factor of ten** because the target is a weight-0.2
literature estimate, not an ICES assessment — the same caveat the note above already carries for the
indicative tier. A ±20% biomass swing being invisible to this gate is a property of a loose target,
not evidence of a robust prediction. The mechanism work found real 20% structure that this
certification cannot see.

**Headroom is strongly asymmetric** — 38% to the floor against 517% to the ceiling — so stickleback
is a floor-risk species, and the term now known to control it (herring predation on early stages)
is governed by *herring's* calibration rather than any stickleback parameter. Anything that raises
herring biomass pushes stickleback toward its floor through a pathway that is a rounding error
(0.038%) in herring's own diet.

**Recording a scope difference worth knowing:** the A/B harness gate
(`scripts/baltic_depletable_ab.py`) pins stickleback and perch as **required** species, while this
certification's headline verdict classes both as **indicative** and excludes them. That is
deliberate on both sides — the gate is identity-pinned for A/B comparability — but a "GATE PASS"
and a "5/5 ASSESSED" are therefore not statements about the same species set.
