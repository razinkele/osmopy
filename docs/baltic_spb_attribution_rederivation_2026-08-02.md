# SP-B "params alone cannot stabilise" — re-derived against the corrected `persists`

**Date:** 2026-08-02
**Closes:** handoff item 3 ("re-derive SP-B attributions, which were fed by the old persistence flag").

## Why these needed re-deriving

The SP-B gate reads: *"the failing species (not PASS above) are candidates params alone cannot
stabilise; record whether sweeping their params moved them (structural vs tunable)."* Its candidate
list is therefore **whatever the certification marked as failing** — and until `556ba3d` that list was
produced by a `persists` criterion measuring the **whole-run** minimum, which is dominated by the
Baltic seeding bootstrap. Species that dipped during initialisation and then settled comfortably
inside their envelope were being handed to SP-B as structural-instability candidates.

Reclassification rule, from `docs/baltic_certification_reread_2026-08-01.md`: a row reading
**`persists ✗` + `in-envelope ✓`** is the artifact signature — the final-decade mean is in range, so
the flagged minimum came from the bootstrap, not from a 50-year dip.

## `docs/baltic_percid_removals_certification_2026-07-28.md` — 2/8 → **6/8**

| species | as published | re-read | why |
|---|---|---|---|
| cod | ✗ / ✓ | **PASS** | artifact — mean 73.1–75.7 kt in envelope |
| herring | ✓ / ✓ | PASS | |
| sprat | ✗ / ✓ | **PASS** | artifact — mean 1.13–1.16 Mt |
| flounder | ✗ / ✓ | **PASS** | artifact — mean 42.5–44.5 kt |
| perch | ✗ / ✓ | **PASS** | artifact — mean 43.9–45.9 kt |
| pikeperch | ✓ / ✗ | **FAIL** | real — over envelope |
| smelt | ✓ / ✗ | **FAIL** | real — over envelope |
| stickleback | ✓ / ✓ | PASS | |

**SP-B candidate list shrinks from 6 species to 2** — `pikeperch` and `smelt`. Neither is a
persistence failure: both persist fine and fail `in_envelope` on the high side. "Params alone cannot
stabilise" was the wrong question for all six; for the surviving two the question is whether params
can bring an overshoot *down*, which is a different problem with a different answer.

## `docs/baltic_disagg_percid_certification_2026-07-28.md` — 2/9 → **6/9**

| species | as published | re-read | why |
|---|---|---|---|
| cod_west | ✗ / ✓ | **PASS** | artifact — mean 14.0–14.5 kt |
| cod_east | ✗ / ✗ | **FAIL (real)** | final-decade mean **5.2e-21 – 2.6e-20** — genuinely extinct |
| herring | ✓ / ✓ | PASS | |
| sprat | ✗ / ✓ | **PASS** | artifact — mean 1.23–1.25 Mt |
| flounder | ✗ / ✓ | **PASS** | artifact — mean 43.6–46.2 kt |
| perch | ✗ / ✓ | **PASS** | artifact — mean 42.7–44.6 kt |
| pikeperch | ✓ / ✗ | **FAIL** | real — over envelope |
| smelt | ✓ / ✗ | **FAIL** | real — over envelope |
| stickleback | ✓ / ✓ | PASS | |

**`cod_east` here is a real collapse and must not be swept up in the correction.** Its final-decade
mean is ~1e-20 — twenty orders of magnitude below its envelope floor. That is extinction at
equilibrium, not a bootstrap dip, and it is the one row in either table where the original
`persists ✗` verdict meant what it said. It is also the only genuine *persistence* failure across both
documents.

## What this changes for SP-B

* The candidate list across both certifications drops from **7 species to 3**: `cod_east`,
  `pikeperch`, `smelt`.
* Only **one** of those three (`cod_east`, in the disaggregation config) is a stability failure. The
  other two are `in_envelope` overshoots — the known percid problem, untouched by the criterion fix
  because it was never a persistence issue.
* So "params alone cannot stabilise the Baltic" was substantially an artifact of the criterion. What
  survives is narrower and already known: the disaggregated eastern-cod configuration collapses, and
  the percids overshoot.

## Limits

**Inferred from the committed tables under the audit's rule — not fresh runs.** Only `--params
current` has been re-certified live under the corrected criterion
(`docs/baltic_8species_recert_corrected_criterion_2026-08-01.md`, 5/8). The reclassification is
mechanical and the artifact signature is unambiguous, but the corrected counts here (6/8, 6/9) are a
re-read, not a re-measurement. Re-certifying either config would put them beyond dispute.

Note the two configs are not comparable to each other: the 8-species table is the percid-removals
config, the 9-species is the cod-disaggregation experiment.
