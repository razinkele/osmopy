---
name: Deep review v3 remediation — SHIPPED
description: V3 deep review (34 findings) fully remediated across Critical/Important/Minor tiers. Historical provenance and the few useful "why we did it this way" notes.
type: project
originSessionId: 077a9827-9ad8-4127-8ebe-28935f931333
---
**Status:** Deep review v3 fully resolved and shipped to master. All three tiers closed:

- **Critical (C-1..C-8)** shipped 2026-04-11/12 in commits `f177926`, `fa0c5f9`, `3d2d134`.
- **Important + Minor (I-1..I-10, M-2..M-14)** shipped per the 23-task plan at `docs/superpowers/plans/2026-04-12-deep-review-v3-remediation-plan.md`.
- **Deferred items (I-3, M-5, M-7, D-1, M-9)** all subsequently shipped — see [project_i3_from_dict_split.md](project_i3_from_dict_split.md) for the I-3 split; MovementMapSet strict coverage mode shipped for M-7; reactive isolate test for D-1; seeding docs + UI helper extractions for M-5 + M-9.

**Findings document:** `docs/superpowers/reviews/2026-04-11-fresh-deep-review-v3.md`.

**Load-bearing patterns worth remembering:**

- **Plan drift is real.** The v2 plan had 11 of 27 tasks already-fixed (absorbed into a prior v0.6.0 refactor); the v3 plan caught this via pre-flight grep before dispatching subagents. **Always spot-check task targets before dispatching implementer subagents.**
- **Follow-up pattern:** review-driven follow-ups on the same branch happened for ~30% of tasks — plan-specified minimal fix → reviewer flags gap → one-commit targeted follow-up. Pattern held across v2 and v3.
- **`_require_file` helper** (added in `fa0c5f9`) replaced 5 silent-fallback file-resolution sites. When adding new file-loading code in engine, prefer this helper over bare `Path(...).exists()` checks.
- **`simulate()`-based fixture pinning** beat building bespoke `ResourceState` fixtures from scratch — see `tests/test_engine_rng_consumers.py` and `tests/test_engine_diet.py` for the template.

**Why these patterns matter now:** Future deep reviews will generate similar plans. The grep-before-dispatch rule + the follow-up-commit budget are the two pieces of institutional knowledge that survived both v2 and v3.
