# 8-species baseline re-certified under the corrected `persists` criterion

**Measured 2026-08-01**, settling the inference in `docs/baltic_certification_reread_2026-08-01.md`. Run at commit `646a36d` using that commit's own engine and `data/baltic`, with only the `persists` one-liner patched to the final-decade window (`556ba3d`) — the criterion is the single variable.

**Result: 5/8, matching the audit's reclassification exactly.** cod, herring, sprat, flounder, stickleback PASS; perch, pikeperch, smelt persist but sit over envelope. The 2/8 figure in the committed July note was the whole-run minimum reading the seeding bootstrap.

The corrected persist set is identical to the in-envelope set, so the "5/8 ICES vs 2/8 stable" gap that motivated Phase 0 is **measurably zero**, not merely inferred to be. Note this identity is the audit's structural argument, not an independent confirmation of it: once `persists` uses the final decade and the mean is in envelope, `in_envelope` is the only binding constraint. What this run adds is that the arithmetic was applied to the right rows.

Settles the caveat in [#145](https://github.com/razinkele/osmopy/issues/145).

---

# Baltic stability — SP-A certification

**Params:** current  ·  **horizon:** 50 yr  ·  **seeds:** [42, 123, 7, 999, 2024]

| species | persists | in-envelope | min biomass | final-decade mean range |
|---|---|---|---|---|
| cod | ✓ | ✓ | 4.19e+04 | [60931.06760986075, 68364.12014491015] |
| herring | ✓ | ✓ | 2.04e+06 | [2210504.0450091357, 2300439.2797759203] |
| sprat | ✓ | ✓ | 9.96e+05 | [1046624.7253830075, 1087010.9384440961] |
| flounder | ✓ | ✓ | 3.95e+04 | [43205.75589590842, 43858.63633645097] |
| perch | ✓ | ✗ | 1.03e+05 | [108471.61151806972, 112765.67292445777] |
| pikeperch | ✓ | ✗ | 1.82e+06 | [2093201.1828933074, 2380675.652803406] |
| smelt | ✓ | ✗ | 6.02e+05 | [636017.5617750797, 646342.7859896086] |
| stickleback | ✓ | ✓ | 6.30e+04 | [80313.21355746567, 83202.80909677768] |

**Python verdict: 5/8 persistent & in-envelope.** Not 8/8 — SP-B gate: the failing species (not PASS above) are candidates params alone cannot stabilise; record whether sweeping their params moved them (structural vs tunable).
