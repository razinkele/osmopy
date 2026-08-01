# Re-reading prior Baltic stability conclusions against the corrected `persists` criterion

**Date:** 2026-08-01 · **Trigger:** `556ba3d` rescoped `persists` from the whole-run minimum to the
final-decade minimum, after the seeding A/B showed the old criterion was measuring the initialisation
transient rather than the equilibrium.

## The defect, restated

`persists` was `min(whole run) > 0.1 × envelope-lower`. The Baltic seeding bootstrap drives stocks
through a deep transient dip before they settle, so the criterion was dominated by initialisation.
`in_envelope`, by contrast, has always used the **final-decade mean**. The two halves of every verdict
were describing different windows.

A species reported **`persists ✗` but `in-envelope ✓`** is the signature: it ends inside its ICES
envelope while having dipped during bootstrap. Under the corrected criterion those become PASS.

## Audit of committed certification notes

Rows flagged `✗ persists` / `✓ in-envelope` — i.e. transient artifacts, not collapses:

| note | affected species |
|---|---|
| `baltic_stability_certification_2026-07-01.md` | cod, sprat, flounder |
| `baltic_percid_baseline_2026-07-28.md` | cod, sprat, flounder |
| `baltic_percid_removals_certification_2026-07-28.md` | cod, sprat, flounder, perch |
| `baltic_cod_east_fix_certification_2026-07-28.md` | cod_west, sprat, flounder, perch |
| `baltic_disagg_percid_certification_2026-07-28.md` | cod_west, sprat, flounder, perch |
| `baltic_cod_east_M09_certification_2026-07-28.md` | cod_west, **cod_east**, sprat, flounder, perch |

**The same species recur in every note** — cod (both stocks after disaggregation), sprat, flounder,
perch. These are precisely the five that flipped COLLAPSE → PASS when the current config was
re-certified under the corrected criterion on 2026-08-01 (2/9 → 7/9).

**Genuine failures that survive the correction:** `perch` in the two earliest notes
(2026-07-01, percid baseline) is flagged `✗ persists` **and** `✗ in-envelope` — a real failure on both
counts, not an artifact. The percid overshoot (`pikeperch`, `smelt` over envelope) is likewise real and
unaffected: it is an `in_envelope` failure, which the correction does not touch.

## What this changes

* **"The Baltic baseline collapses" is not supported.** Re-certified today: **7/9**, with the two
  failures being `pikeperch` and `smelt` over envelope. **No species collapses at equilibrium.**
* **`cod_east` was never collapsing.** `baltic_cod_east_M09_certification_2026-07-28.md` reports it as
  `✗ persists` on a whole-run minimum of 17 t; its final-decade mean is ~83 kt, inside the 60–85 kt
  envelope, and its final-decade minimum is 58–60 kt. The M=0.9 "fix" was tuned against a verdict that
  was measuring the bootstrap.
* **Conclusions about which species "params alone cannot stabilise"** rest on the SP-B gate, which was
  fed by the same flag. Those attributions are worth re-deriving.

## What this does NOT change

* The **percid overshoot** (`pikeperch` ~56× over envelope, `smelt`) — an `in_envelope` failure,
  untouched by the correction, and still the outstanding calibration problem.
* Any conclusion drawn from **final-decade means**, which were never affected. The seeding A/B's means
  were byte-identical before and after the criterion change.
* The **transient itself** is real: stocks genuinely do dip during bootstrap. What changed is whether
  that dip is reported as a stability verdict.

## Recommended follow-up

Re-certify the configs behind the notes above under the corrected criterion before citing any of their
persistence verdicts. Only `--params current` has been re-run so far. The notes are left in place with
this audit as the correction record rather than being edited, so the original claims and their revision
both remain visible.
