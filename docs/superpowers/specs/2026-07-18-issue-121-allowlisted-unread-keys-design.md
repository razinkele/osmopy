# Fix #121 — allowlisted-but-unread config keys — design

**Date:** 2026-07-18 · **Issue:** [#121](https://github.com/razinkele/osmopy/issues/121)
**Branch:** `fix/issue-121-allowlisted-unread-keys`
**Scope:** Layers A + B + C. Layer D (systemic known-but-unread warning) + `conversion2tons` aliasing are deferred to **[#123](https://github.com/razinkele/osmopy/issues/123)** (filed 2026-07-18).

> **Revised after adversarial workflow review (2026-07-18).** Three corrections the review surfaced, now reflected in the plan: (1) the config-fix inventory below missed `data/minimal` (it sets `output.frequency.ndtperyear`); the true live set is `data/examples` + `data/eec` + `data/minimal` — verified by repo-wide grep, NOT `baltic`/`eec_full`/`examples_433_orig`. (2) The Layer-D follow-up is now a real issue (#123), so the guide's systemic-case reference repoints there and this PR closes #121 honestly. (3) Layer-B testing must prove the output is **produced** (run the engine, assert non-empty CSV), not merely that the flag parses. Also banked: `output.tl.enabled` → mean-TL is semantically verified (gates Java's `WeightedSpeciesOutput(getTrophicLevel, getWeight)`), and all replacement keys are real Java 4.4.1 keys so eec parity holds.

## Method — this issue is about false "it's dead" claims, so every claim here was executed

The whole failure mode #121 documents is reasoning from an allowlist entry or a line number to a
runtime behavior. Applying that lesson to the fix itself already corrected the issue **twice**:

- The issue listed `output.diet.stage.threshold.sp{idx}` as dead. It is **not** — it is emitted
  for the Java 4.4.1 engine at `osmose/java_background_staging.py:182` (`"4.4.1 requires"` it).
- The issue listed `species.conversion2tons` / `ltl.conversion2tons` as dead-invented. They are
  **legacy 4.3.x forms with real lineage**: the 4.4.1 key is `plankton.conversion2tons.plk`
  (renamed to `resource.conversion2tons`, `osmose/demo.py:41`), and `data/examples_433_orig`
  uses the legacy forms *as a deliberately-preserved 4.3.3 config*.

**Every key removed or claimed dead in this spec was cleared on three fronts by execution:**
not read by the Python engine (grep incl. `startswith`), 0 hits in **both** vendored jars, and
not emitted by `java_background_staging.py`. The implementation plan must re-run that clearance
before removing any key — an allowlist entry is not evidence of dead-ness.

## Layer A — Canonical realignment (correctness; 2 keys)

Two keys are genuine user-facing bugs: osmopy invented its own name and **ignores the real
upstream name**, so an authentic R/Java config silently loses output.

| Real upstream name (osmopy ignores) | osmopy-invented name it reads today | Read site |
|---|---|---|
| `output.tl.enabled` (in 4.4.1 jar) | `output.meantl.enabled` (0 jar hits) | `config.py:923` |
| `module.bioeconomics.enabled` (in 4.4.1 jar, `Releases$15`) | `simulation.economic.enabled` (0 jar hits) | `config.py:2431` |

**Fix: read-site fallback, upstream name first.** Not an alias subsystem — YAGNI for two keys,
and it keeps the change to two lines with no load-order concerns.

- `config.py:923` → `_enabled(cfg, "output.tl.enabled") or _enabled(cfg, "output.meantl.enabled")`
- `config.py:2431` → `_enabled(cfg, "module.bioeconomics.enabled") or _enabled(cfg, "simulation.economic.enabled")`

**Why this direction (canonical, not minimal-alias):** osmopy is a *port*; it should honor real
OSMOSE key names, not enshrine invented ones. The economy case falls out cleanly — `economy.enabled`
already renames to `module.bioeconomics.enabled` via `RENAMES_440` (`aliases.py:177`, the faithful
Java port), so an authentic legacy config now reaches a key the engine actually reads.
Back-compat is preserved: the invented names remain the fallback, so **no existing config or test
breaks**. `aliases.py` / `RENAMES_440` is **not touched** — it is correct as-is.

**Side effect (good):** `output.tl.enabled` and `module.bioeconomics.enabled` become string
literals in `config.py`, so the validator's AST walk recognizes them as genuinely read. They can
be dropped from `_SUPPLEMENTARY_ALLOWLIST` (now redundant); the plan verifies validation stays
clean either way.

## Layer B — Hygiene (remove 5 verified-dead invented keys; fix the configs that set them)

**Removal set — 5 keys, each cleared on all three fronts** (py-unread, 0 hits in *both* jars,
not staged):
`output.byage.enabled`, `output.bysize.enabled`, `output.meansize.enabled`,
`output.trophiclevel.enabled`, `output.frequency.ndtperyear`.

These are osmopy-invented names that appear only in our own bundled configs; the working keys
(all verified read in `config.py`) are:

| Dead invented key (remove) | Working key(s) it should have been (verified read) |
|---|---|
| `output.byage.enabled` | `output.biomass.byage.enabled` **+** `output.abundance.byage.enabled` |
| `output.bysize.enabled` | `output.biomass.bysize.enabled` **+** `output.abundance.bysize.enabled` |
| `output.meansize.enabled` | `output.size.enabled` |
| `output.trophiclevel.enabled` | `output.tl.enabled` (canonical after Layer A; `output.meantl.enabled` also works) |
| `output.frequency.ndtperyear` | `output.recordfrequency.ndt` |

**Explicitly NOT removed** (verified to have real lineage — removing would flag a legitimate key):
`output.diet.stage.threshold.sp{idx}` (staged for Java 4.4.1), `output.diet.stage.structure`
(sits under the real `output.diet.stage` jar prefix; conservative keep),
`species.conversion2tons.sp{idx}` / `ltl.conversion2tons.rsc{idx}` (legacy 4.3.x, real
`plankton.conversion2tons` lineage, used by the preserved `examples_433_orig`). A future issue
could alias the legacy conversion forms to `resource.conversion2tons`, but that is alias work,
out of this scope.

**Two parts:**
1. **Remove the 5 keys from `_SUPPLEMENTARY_ALLOWLIST`** (`config_validation.py`) so strict
   validation flags them as unknown — which is correct, since nothing reads them.
2. **Fix the bundled configs that set them — replace, not delete** (user-approved). The configs
   are `data/examples/osm_param-output.csv`, `data/eec/osm_param-output.csv`,
   `data/examples_433_orig/osm_param-output.csv` (re-grep for the exact set per key). Replace each
   dead key with the working key(s) from the table above, so the example actually produces the
   output it asks for (`data/examples` is the new-user starting point and currently sets
   `output.byage.enabled ; true` while producing no by-age output).

   ⚠️ **Output-changing:** replacing adds real output CSVs (by-age, by-size, mean-size). The plan
   MUST find and intentionally re-baseline any snapshot/output test over these configs, and must
   NOT silently absorb a diff. If a config is a frozen fixture whose output is asserted elsewhere,
   the plan flags it rather than changing it.

## Layer C — Correct the false / misleading allowlist comments

`config_validation.py:99-100` claims the removed keys "control the Java engine's output; the
Python engine ... does not parse these." False — they have 0 jar hits, dead on both engines.
Rewrite to state the truth: osmopy-invented names, nothing reads them, use the working keys.

`config_validation.py:135` claims `conversion2tons` is "Read by the Java engine for biomass-unit
conversion." This is **not simply false** — the real 4.4.1 key is `plankton.conversion2tons` /
`resource.conversion2tons`; the allowlisted `species.`/`ltl.` forms are legacy 4.3.x. Rewrite to
say *that* accurately (do not delete — the keys stay allowlisted for the preserved 4.3.3 config).

Comment-only; no behavior change. Truth-checked against the jars.

## Testing

- **Layer A (both directions, both keys):** setting the upstream key flips the engine attribute
  (`output_meantl` / `economics_enabled`) from its default; setting the invented key still flips
  it too (back-compat). A config setting neither leaves it at default.
- **Layer B:** each of the 5 removed keys is now reported unknown by `validate(cfg, "warn")`;
  the replacement keys are recognized (not unknown). The fixed bundled configs, run through the
  engine, actually produce the by-age / by-size / mean-size / TL output they request (assert the
  output family is non-empty, not just that the flag parses).
- **Layer C:** comment-only, no runtime test; covered by Layer B's "flagged unknown" assertions.
- **Whole-suite guard:** the existing suite must stay green. Layer A **will break two
  assertions** in `tests/test_r_dialect_migration_claims.py` (PR #122), verified:
  - `:196` `_probe(..., {"output.tl.enabled": "true"}).output_meantl is False` → becomes `True`.
  - `:243` `_probe(..., {"module.bioeconomics.enabled": "true"}).economics_enabled is False` → `True`.

  These are the guide's own tripwires, *designed* to go red when the defect is fixed. Update them
  to assert the fixed behavior (upstream key now flips the attribute). Keep the fallback
  assertions (`:199`, `:246`) — the invented names still work.

### The documentation ripple — fixing #121 partly un-ships PR #122's guide (SCOPE DECISION)

Layer A fixes exactly the two keys the guide's §2 "two traps you can verify right now" subsection
is built around: `output.tl.enabled` (its **headline** trap) and `economy.enabled` /
`module.bioeconomics.enabled` (its **latent** trap). After this fix they are no longer traps.
The guide must not keep claiming live silent-ignore for behavior that now works. **This spec's
recommended handling** (confirm at user review):

- **Reframe, don't gut.** The guide's core value is teaching the *class* (config keys that load
  clean and silently no-op) and the *method* (run `check_config.py` + strict validation on YOUR
  config). That stands. Rewrite the "two traps" subsection + appendix rows to present them as
  **worked examples now fixed in #121**, illustrating the class — with the still-live members of
  the class (the spatial-inputs `.nc` trap, missing sub-configs, cross-file precedence) carrying
  the "here is what still bites you" weight.
- The spatial-inputs trap remains the guide's #1 example and is untouched by this fix.
- Re-point / close the "tracked in #121" references, and update the `TRAPS` fixture table
  (`:57-58`) accordingly.

This guide surgery is **in scope for this fix** — the fix and the doc it invalidates should land
together, or the guide ships false claims. If the user prefers to keep the fix minimal (tests
only) and defer the prose reframe, that is a valid alternative but leaves the guide temporarily
inaccurate; flag at review.

## Out of scope (explicit)

- **Layer D** — a systemic "known-but-unread" warning. Highest value, but hard to do without
  false-positiving on keys legitimately read via `startswith`/dynamic (the ~910-candidate
  problem). Its own issue.
- **`species.lw.*`** — fails *loudly* with a `KeyError` naming the correct key when its
  `species.length2weight.*` twin is absent; it is not a silent no-op and not a #121-class defect.
- **`aliases.py` / `RENAMES_440`** — correct as-is; not touched.
- **The `conversion2tons` legacy forms** — real lineage; leave allowlisted, correct the comment
  only. Aliasing them to `resource.conversion2tons` is possible future work.

## Success criteria

- An authentic config setting `output.tl.enabled` gets mean-TL output; one setting
  `module.bioeconomics.enabled` (or migrated from `economy.enabled`) gets economics — on the
  Python engine, verified by running.
- `data/examples` produces the by-age output it requests, verified by running the engine.
- The 5 removed keys are flagged unknown under strict validation; no key with real Java lineage
  is flagged.
- Every allowlist comment touched is true against the jars.
- No existing config or test silently breaks; the migration-guide tripwires are updated to the
  fixed behavior (not deleted, not left red).
- Every "dead" claim in the shipped diff was cleared on three fronts by execution.
