# BoB (Bay of Biscay) 4.4.1 migration + Phase 3 ICES consistency — design

**Date:** 2026-07-06
**Status:** APPROVED — in-loop review CONVERGED (5 rounds, 5 reviewer types; rounds 4-5 clean:
"READY" / "CONFIRMED ready for writing-plans"). Ready for writing-plans on user sign-off.
**Author:** cutover-completion session

**Correction (2026-07-06, during execution — supersedes ALL `species.tl`→`species.trophic.level`
rename text below):** do NOT rename `species.tl`→`species.trophic.level`. Implementation verified the
4.4.1 Java jar reads `species.tl` (`ResourceSpecies.java`), and native EEC keeps `species.tl` too, so
the migration KEEPS `species.tl` unchanged; the Python `species.type` path simply defaults resource
trophic level (diagnostic-only, traced to `mortality.py:551-556` — never affects biomass/abundance/
yield; EEC does the same and parity passed). Every "rename" mention in A2/A4.2/§7 below is stale — the
executed plan (`docs/superpowers/plans/2026-07-06-bob-440-migration-phase3-ices.md`) is authoritative.
**Supersedes/extends:** `docs/superpowers/plans/2026-06-19-jar-swap-440-validated-resume.md`
(Phase 1 Task 1.2 = BoB resource-forcing; Phase 3 = ICES consistency). Builds on the C1
native-4.4.1 conversion (`project-c1-native-440-cutover`, master `ac03be0`) and the EEC
Phase-2 cross-engine parity PASS (`scripts/cross_engine_parity_440.py`, master `a421681`).

**Revision note (round 1):** the first draft justified a "hybrid" config on the false premise
that BoB lacks `species.*` resource descriptors. It does carry them (verified,
`osm_all-parameters.csv:102-131`). This revision switches to **fully-native** (matching the EEC
pattern exactly), which also removes a dual-source-of-truth hazard, an unverified "Java ignores
`ltl.*`" assumption, and the need for a new `aliases.py` code path. Harness/gate sections were
corrected against the real code (`cross_engine_parity_440.py` is single-config and gates on the
relative test; the `simulation.rng.fixed` determinism key does not exist).

---

## 1. Context and scope

The 4.4.1 engine cutover is, in practice, already done and live in prod as of 2026-07-06:
default jar is 4.4.1 (`ui/state.py:42`), the EEC/minimal/Baltic/baltic_ev configs are native
4.4.1 on disk, write-target derives from the selected jar, and EEC cross-engine parity PASSED
(N=16). Two deferred pieces remain, covered here in order:

1. **Part A — Migrate `data/examples` (Bay of Biscay) to 4.4.1.** It is the *only* bundled
   config still on `osmose.version;4.3.3`, blocked by a 365-daily-step resource forcing the
   4.4.1 engine cannot consume at `ndtperyear=24`. Migrating it also unblocks the BoB half of
   Phase 3.
2. **Part B — Phase 3 ICES/empirical consistency validation** across EEC + BoB. The master plan
   frames this as a cross-engine *consistency* tripwire, **not** an empirical-realism gate
   (EEC/BoB are uncalibrated demo configs). We honour that framing.

**Non-goals:** flipping the *bare* write-default from 4.3.3; dropping the 4.3.3 jar (kept for
rollback); prod redeploy; new 4.4.x Java features; BoB ICES *calibration*.

---

## 2. Ground truth (verified 2026-07-06)

**BoB config `data/examples/`** (`osm_all-parameters.csv`):
- `osmose.version;4.3.3` (line 64, active), `simulation.time.ndtperyear ; 24` (line 3),
  `simulation.nyear ; 50`, `simulation.nspecies ; 8` (line 5), `simulation.nresource ; 6` (line 6).
- Native resource keys present: `species.type.sp8-13 ; resource` (lines 82-92), `species.name.sp8-13`
  = SmallPhyto, LargePhyto, SmallZoo, LargeZoo, SmallDetritus, LargeDetritus (lines 94-99) — **1:1
  with the NetCDF data vars** → `varname = species.name.spN` (same rule as EEC).
- **A full `species.*` resource-descriptor block is ALSO present** (lines 102-131), values equal to
  the legacy `ltl.*` copies: `species.size.min/max.sp8-13`, `species.tl.sp8-13`,
  `species.accessibility2fish.sp8-13`, `species.conversion2tons.sp8-13`.
- Legacy `ltl.*` scheme present too (`osm_param-ltl.csv`): `ltl.netcdf.file ;
  ltl/roms_n2p2z2d2_biscay.nc` (line 7) + per-resource `ltl.{name,tl,size.min,size.max,
  accessibility2fish,conversion2tons}.rsc0-5`. Deprecated `# ltl.nstep ; 365`.
- **Key-name mismatch (decisive for the migration):** the Python engine's `species.type` load
  path reads trophic level from `species.trophic.level.spN` (`resources.py:155`), but BoB carries
  `species.tl.spN`. So `species.trophic.level.sp8-13` is **absent** (verified: grep count 0) —
  forcing BoB onto the `species.type` path without a rename would default resource TL to 1.0
  (losing 2.0/2.5 for SmallZoo/LargeZoo). Resource TL is diagnostic-only (EEC already defaults it
  and parity passed), but we rename it anyway to stay strictly lossless.

**Forcing NetCDF `data/examples/ltl/roms_n2p2z2d2_biscay.nc`:** dims `time=365, lat=20, lon=30`;
6 data vars (SmallPhyto … LargeDetritus), each `(time, lat, lon)`.

**Python engine resource forcing (`osmose/engine/resources.py`):**
- Load-path dispatch (line 73): `has_ltl_keys = any(k.startswith("ltl.name.rsc") ...)` → BoB
  currently takes `_load_config_ltl()` (line 76, reads `ltl.netcdf.file`). The fully-native
  migration (§3 A2) removes the `ltl.name.rsc*` keys so BoB takes `_load_config_species_type()`
  (line 109, reads `species.file.spN` + `species.*` params), exactly like EEC.
- `_load_netcdf` (line 181) sets `self._n_forcing_steps` from the NetCDF first-var time dim (line 192).
- **Time indexing (lines 219-221):** `forcing_idx = int((step % ndt) * n_forcing_steps / ndt)`.
  With `n_forcing=365, ndt=24` this **subsamples** one day per step (days 0,15,30,…,349) — not a
  window average. With a 24-step file, `forcing_idx = step_in_year` (reads field i for step i).

**The core blocker:** 4.4.1's `ForcingFile.update()` requires the forcing's `nsteps.year` to
divide `ndt=24`; 365 does not, and the init guard is commented out upstream, so a wrong value
passes init then mis-maps in `update()`. The forcing must be resampled to a 24-step axis.

**Consequence that drives the gate design:** because Python *subsamples* today, a **bin-averaged**
24-step file changes what Python reads too (window-mean vs single-day). So BoB's migration is
**not dynamics-neutral** — a C1-style "bit-exact vs the old 365-step source" run gate is
unachievable by construction and is replaced (§3 A4).

**Harness/tooling reality (verified):**
- `scripts/cross_engine_parity_440.py` is **single-config**: it hardcodes `EEC =
  data/eec_full/eec_all-parameters.csv` (line 42), `prefix="eec"` (line 91), and runs all three
  engine arms from that one input via `write_temp_config(..., target_version=engine)` (line 104).
  There is no per-arm config-path mechanism. Its binary GATE is the **relative** test +
  1-OoM tripwire (`no_worse = abs(d1) <= abs(d3)+delta; if not no_worse or abs(d1) >= 1.0:
  overall_fail`, lines 196-199); the **absolute** TOST equivalence `eq1` is computed and printed
  but **never gated** (line 194 vs 199). Part A must parametrize the config path AND add the
  absolute gate (§3 A4.4).
- `scripts/migrate_bundled_to_440.py` (IN_SCOPE line 30) and `scripts/native_440_parity.py`
  (IN_SCOPE line 24) both **explicitly exclude BoB** (`raise SystemExit(... BoB/examples
  excluded)`, migrate line 118). Both must be extended to include it.
- `osmose/config/aliases.py::_emit_resource_biomass_forcing` emits `species.biomass.{mode,file,
  varname,nsteps.year}.spN` for each `species.type.spN==resource` **that has `species.file.spN`**,
  skipping any that lack it (line 137). It has **no** `ltl.*` handling. Once BoB is EEC-shaped
  (per-species `species.file.spN`), this existing path handles BoB — **no new aliases path is
  needed** (revised from draft).
- `osmose/validation/ices.py`: `IcesSnapshot` (45), `load_snapshot` (104), and
  `compare_outputs_to_ices(results, snapshot, *, window_years=5, ices_window=range(2018,2023))`
  (212) — returns per species `in_range: bool` + `magnitude_factor`. `index.json` fields
  `model_species_to_ices_stocks` / `units_by_stock` / `advice_year_by_stock`.

**Non-forcing config families — the "mirror EEC" assumption has ONE real gap (fishing).** BoB is
NOT structurally identical to EEC outside the resource forcing:
- **Fishing (the gap):** BoB uses the **legacy per-species scheme** —
  `simulation.fishing.mortality.enabled;true`, `simulation.nfisheries;3`, per-species
  `mortality.fishing.rate.spN` (`osm_param-fishing.csv`), and **no** fleet `fishery-catchability`
  file. EEC (the parity-passed "mirror") is **fleet-based**. The Python engine dispatches on
  `module.multispecies.fisheries.enabled` (`config.py:1976-1998`) and BoB's legacy path already
  works on Python and the 4.3.3 jar (`tests/test_engine_java_comparison.py` currently green). But
  **the legacy fishing scheme has NEVER been exercised on the 4.4.1 Java jar** (EEC only exercises
  the fleet path), and `reference_osmose_java_4_4_0.md` lists a fisheries-module rename +
  "fishing/discards tracked in numbers" as 4.4.0 breaking changes. So A4.1's smoke is the FIRST
  test of BoB legacy fishing on 4.4.1 — treat a smoke failure here as a **separate, unbudgeted
  fishing-migration task**, not part of the resource-forcing work (see §5 rescope trigger).
- **Movement:** BoB is random-walk (not map-based like EEC). `RENAMES_440` doesn't touch movement
  keys → low risk, but it is another "mirrors EEC" assumption asserted, not gated. A4.1 covers it.
- BoB has `nbackground=0` → Java-safe (unlike Baltic).

---

## 3. Part A — BoB 4.4.1 migration

### A0. Precondition gate — BoB runs on the Python engine (the true first step, HARD gate)

Every downstream gate assumes BoB runs clean on the Python engine, but BoB is excluded from all
current tooling and there is no in-tree evidence it does. **First**, confirm `data/examples` runs
end-to-end on the Python engine with the current 365-step forcing (and, after A1, the 24-step
forcing). Pin an explicit small `nyear` for this run (the on-disk config carries
`simulation.nyear;50` — do NOT inherit it). This is a **hard gate**: if BoB does not run on the
Python engine, the entire plan is moot and stops here.

### A0b. Snapshot the pre-migration config (required by A2 + A4.4 + rollback)

Because A2 rewrites `data/examples` **in place** and A4.4's "OLD (before)" arm needs the *original
365-step ltl config CSVs* (not just the raw NetCDF — `to_target_keys(4.3.3)` cannot reconstruct the
`ltl.*` scheme from a native config), copy the entire pre-migration config tree to a preserved
location **before A2 mutates it** — e.g. `data/examples_433_orig/` (committed) or a captured git
ref. A4.4's OLD arm points the harness at this snapshot. This snapshot is also the **config
rollback** artifact (revert = restore it / `git checkout data/examples`).

### A1. Forcing resampler — `scripts/resample_bob_forcing.py`

- Read `roms_n2p2z2d2_biscay.nc` (365, 20, 30).
- **Bin-average to 24 steps**: assign input day `d` (0..364) to output step `floor(d * 24 / 365)`
  (∈ 0..23), mean each bin over the time axis. Bins are contiguous, sizes 15-16 (365 = 24×15 + 5),
  all non-empty; each bin's **window mean is conserved** (no aliasing). Output step index `s` maps
  to the engine's read index exactly (`forcing_idx = step_in_year = s` for a 24-step file). (The
  old single-day subsample read ~the first day of each bin; the bin-mean differs from it by design
  — that difference is characterized in A4.3, not gated.)
- Write `data/examples/ltl/roms_n2p2z2d2_biscay_24step.nc` (24, 20, 30), preserving `lat/lon`
  coords, variable names, attributes. **Keep the original 365-step file** for provenance and the
  A4.3 characterization / A4.4 4.3.3 baseline arm.
- Deterministic, idempotent. Committed as a data artifact. Unit test asserts 24-step output,
  conserved bin means, preserved coords/attrs, idempotency.

### A2. Fully-native on-disk conversion of `data/examples` (matches the EEC pattern)

Rewrite `data/examples` in place to a **fully-native 4.4.1** config, exactly mirroring native EEC.
**This is a substantial BoB-specific code path in `scripts/migrate_bundled_to_440.py`, NOT a
one-line IN_SCOPE add** — the script today only *renames* keys in `RENAMES_440`, scales larval
rates, stamps version, and drops lines on rename-collision. The BoB conversion needs three
capabilities the script lacks and that must be built (its own task + TDD):
- **Add keys** — `species.file.sp8-13` = `ltl/roms_n2p2z2d2_biscay_24step.nc` (the per-species
  forcing path the EEC-shaped Python path + `_emit_resource_biomass_forcing` both key off).
- **A BoB-specific rename** — `species.tl.sp8-13` → `species.trophic.level.sp8-13`, so the Python
  `species.type` path reads TL correctly (`resources.py:155` reads `species.trophic.level`; BoB
  carries `species.tl`). **This is NOT a universal 4.4.0 rename** (native EEC uses unrenamed
  `species.tl` and defaults TL to 1.0, parity still passed → resource TL is diagnostic-only for the
  gated biomass/yield/abundance metrics). So it goes in a **BoB-specific fixup, NOT `RENAMES_440`,
  and needs NO inverse** in `to_target_keys` (the A4.4 OLD arm uses the preserved original config,
  never a reverse-mapped one). We do the rename purely to keep A4.2's load-path-equivalence gate
  bit-exact (preserve TL 2.0/2.5 for the zoo resources); it is not required for the biomass gates.
- **Drop a whole key family across an *included* file** — `ltl.netcdf.file` + all `ltl.*.rsc0-5`.
  These physically live in the included sub-file `data/examples/osm_param-ltl.csv` (reached via
  `osmose.configuration.ltl`), not the master, and that file is **100% `ltl.*` keys** (only comments
  otherwise). The conversion must edit the sub-files (traverse via the existing
  `_collect_param_files`), because **a single leftover `ltl.name.rscN` anywhere in the merged config
  silently re-routes BoB back onto `_load_config_ltl()`** (`resources.py:73`). **Decision:** drop the
  `ltl.*` key lines but **keep the (now comment-only) `osm_param-ltl.csv` and its
  `osmose.configuration.ltl` include line** — minimal churn, no edit to the master's include list,
  and a valid empty include. (A2's TDD asserts against this: file present, zero active `ltl.*` keys.)

This BoB conversion is a **name-gated branch** in `migrate_bundled_to_440.py` (`if name ==
"examples": ...`) — the add-key / rename / drop-family steps are BoB-specific and must not run for
the other IN_SCOPE configs.

Then: stamp `osmose.version;4.4.1`. The `species.{size.min,size.max,accessibility2fish,
conversion2tons}.sp8-13` block already on disk (lines 102-131) + the renamed
`species.trophic.level.*` fully replace the dropped `ltl.*`, so there is no data loss and no
dual-source-of-truth.

**Do NOT bake `species.biomass.{mode,file,varname,nsteps.year}.sp8-13` into the on-disk config.**
Per the C1 convention (`migrate_bundled_to_440.py:132-135`; native EEC on disk has `species.file.spN`
but NOT `species.biomass.*`), those are emitted at Java-stage time by `_emit_resource_biomass_forcing`
(verified: it derives `nsteps.year` from `simulation.time.ndtperyear`=24, `file`/`varname` from
`species.file`/`species.name` — correct for BoB with no `aliases.py` change). Baking them would make
the Python validator flag unknown keys — verify the unknown-key handling mode on the result.

(Optional hardening, low priority: have `to_target_keys`/`_emit_resource_biomass_forcing` raise
loudly if a resource forcing's step count does not divide `ndt`, so a future un-resampled config
fails fast instead of mis-mapping — guards the class of bug that caused the original blocker.)

### A3. (folded into A2 — no separate hybrid step)

### A4. Gates (BoB is not dynamics-neutral → refined from the C1 pattern)

1. **Load + run smoke** — run a fixed `nyear` (pin an explicit small value, e.g. 6, for the gate;
   do NOT inherit the source `nyear;50`) on the 4.4.1 jar against the migrated `data/examples`.
   Resource keys are read per-step in `update()`, not just at init, so the smoke must complete
   ≥1 full year to catch a bad time-mapping. This is the Phase-1 exit criterion and the gate that
   originally surfaced the 365-step blocker.
2. **Key-conversion losslessness = a load-path-equivalence gate (reframed).** The goal is to prove
   the *key conversion* (ltl → native) does not perturb Python outputs — separately from the
   *forcing resample* (365→24), which is a deliberate dynamics change (A4.3), not gated bit-exact.
   These two changes are bundled in A2, so a naive "before-conversion vs after-conversion" run
   (the C1 `native_440_parity` shape) compares 365-subsample vs 24-bin-average and **fails by
   construction** — it is the wrong gate for BoB. Instead, hold the forcing fixed at 24 steps on
   both sides and compare the two Python **load paths**:
   - Build a **test-only intermediate config** = the original ltl-keyed BoB (from the **A0b
     snapshot** — the in-tree config is native after A2) with `ltl.netcdf.file` repointed to the
     24-step file (so Python reads 24-step via `_load_config_ltl`). This artifact is constructed by
     the test, not shipped.
   - Compare it against the native 24-step config (Python reads 24-step via
     `_load_config_species_type`), on the standard **biomass / abundance / yield** output arrays.
   - Assert the Python outputs are **bit-exact** (`np.array_equal`). This isolates "do the ltl and
     species.type load paths agree on identical forcing" = key-conversion losslessness. (Note: the
     compared biomass/abundance/yield arrays do not read resource TL — `resources.py:216-232` — so
     the `species.tl`→`species.trophic.level` rename is **losslessness insurance**, not the thing
     that makes these arrays bit-exact; it only matters if a TL-derived diagnostic is added to the
     compared set.)
   - **Preconditions** (state explicitly; verified against BoB): larval rates are all 0.0
     (`osm_param-species.csv`, `mortality.additional.larva.rate.sp0-7`), so the ~1-ULP `×ndt/÷ndt`
     larval drift that forces the C1 1e-9 tolerance never occurs; no `species.lmax`/`species.beta`,
     so nothing the Python engine reads is stripped; and **no `species.multiplier`/`species.offset`/
     `species.accessibility2fish.file.spN`** (all absent → defaults → the `species.type` path reads
     nothing extra the `ltl` path lacks). If any precondition is violated (e.g. a future edit adds
     `species.multiplier.spN`), fall back to a tight relative tolerance and say so.
   - **Determinism mechanism (corrected):** determinism comes from the `seed=` argument to the
     in-memory run. Do NOT rely on `simulation.rng.fixed` — the engine never reads it (note:
     `native_440_parity.py:42` sets that dead key and leans on `seed=` alone; do not copy the dead
     key). If `seed=` alone proves insufficient for bit-exactness on BoB, add
     `movement.randomseed.fixed=true` + `stochastic.mortality.randomseed.fixed=true` (the real
     flags, `osmose/engine/config.py:2260-2262`, as used in the SP1/calibration work) — but
     **verify** whether BoB actually needs them rather than assuming.
3. **Forcing-change characterization (report, not a gate)** — quantify Python-365-subsample vs
   Python-24-bin-average divergence on per-species resource + fish biomass. Confirm it is small
   and seasonally faithful; document it as the *intended* improvement (both engines now read
   identical forcing). This is an accepted, deliberate change (bin-average was chosen over
   subsample), not a regression.
4. **Cross-engine parity (absolute-primary, relative tripwire).** Parametrize
   `scripts/cross_engine_parity_440.py` (own task): it is currently EEC-hardcoded (`EEC` const
   line 42, `prefix="eec"` line 91) **and** writes to a hardcoded personal tmp dir (line 159) —
   the parametrization must accept a config path + output prefix AND fix that tmp path (use the
   scratchpad / a `tempfile` dir). Run **two** parity computations:
   - **OLD (before) reference:** the **A0b snapshot** (`data/examples_433_orig/`, 365-step ltl
     config) on {Python-365-subsample, 4.3.3-Java-365} → per-species agreement `A_old`. The 4.3.3
     arm must use this original config, never the migrated one (`to_target_keys(4.3.3)` cannot
     reconstruct the `ltl.*` scheme from a native config — the reason the config-path parameter is
     required).
   - **NEW:** the migrated 24-step native BoB on {Python-24, 4.4.1-Java-24} → agreement `A_new`.
   - **GATE — what actually enters `overall_fail` (be precise, since the harness only gates the
     relative test today):**
     (a) **PRIMARY, add to `overall_fail`:** the absolute per-species equivalence `eq1` (TOST at
         Δ = log10(3)) on the NEW pair must PASS — `if not eq1: overall_fail.append(...)`. `eq1` is
         already computed (line 194) but currently discarded; wiring it in is the core gate change.
     (b) **TRIPWIRE, reported not gated:** the 1-OoM catastrophic-divergence check (existing) stays
         a hard fail; the relative `no_worse = |A_new| ≤ |A_old| + Δ` (existing) is **reported**
         but treated as advisory for BoB, because `A_old`/`A_new` use different forcing (365 vs 24)
         so the relative comparison is confounded (the forcing gap inflates `|A_old|` and biases it
         toward PASS). Keep it in the printed table; do not let it alone decide PASS.
     (c) **REPORTED ONLY (explicitly NOT gated):** KS / variance-ratio. The harness computes these
         only for the 4.4.1 arm today and discards the 4.3.3 arm's (`d3,_,_,_,_,_`, line 195);
         wiring a cross-arm KS/variance comparison is out of scope for this cycle — report `ks1/vr1`
         for the NEW pair and say the distributional check is informational, not gated. (If a later
         cycle wants it gated, that is its own task.)
   - The **load-bearing BoB claim is the absolute Python-24 ↔ 4.4.1-Java-24 equivalence** (a). If a
     Java arm fails to launch, the harness must **report** the dropped arm, not silently omit it
     (a dropped arm degrades the reference invisibly).

---

## 4. Part B — Phase 3 ICES / empirical consistency (EEC + BoB)

### B1. Build the EEC/BoB ICES snapshot (mechanical, value-limited)

Match `osmose/validation/ices.py`'s `IcesSnapshot`/`index.json` layout (mirror the existing
`data/baltic/reference/ices_snapshots/index.json`). Populate via the ICES MCP (`list_stocks`,
`get_stock_assessment`, `get_reference_points`) using **only cleanly mapping tonnes-unit stocks**:
- EEC: sole → `sol.27.7d`, plaice → `ple.27.7d`.
- BoB: anchovy → `ane.27.8`, sardine → `pil.27.8abd`, sole → `sol.27.8ab`, hake → `hke.27.8c9a`
  (BoB sp0/1/6/5; real Division-8 codes, species verified present).

Mark scale-mismatched/NEA-wide stocks `index`-unit or exclude; record no-assessment species as
uncovered. Commit with an honest coverage map + spatial/scale caveats in the manifest comments.

### B2. Cross-engine consistency gate — light reporting layer on A4.4

**Interface reality (verified):** `compare_outputs_to_ices(results, snapshot, ...)` takes a single
`OsmoseResults` object and computes its own trailing-window mean (`model_biomass_window_mean`,
line 241). The parity harness computes per-replicate final-year *means* and **discards** the
`OsmoseResults`. So the ensemble dicts cannot be fed to `compare_outputs_to_ices` — "reuse the
ensembles" as literally stated is not implementable against that function. **Decision: persist one
representative `OsmoseResults` per engine per config** from the A4.4 / Phase-2 runs (a small
addition to the parametrized harness — keep one results dir instead of discarding it), then call
`compare_outputs_to_ices` on each to get per-species `magnitude_factor`. (Rejected alternative: 6
dedicated fresh runs — more sim work and it is not "reuse"; only fall back to it if persisting a
results dir proves impractical.)

**Run-length constraint:** `compare_outputs_to_ices` takes a trailing-window mean over
`window_years` (default 5, `ices.py:204,212`). So the A4.4 run whose `OsmoseResults` B2 reuses must
be **at least `spinup + window_years` years** (or `window_years` lowered to match) — otherwise B2's
window is degenerate. Do NOT feed B2 a short smoke-length run; pin `A4.4 nyear ≥ spinup +
window_years` for the persisted-results run specifically (the A4.1 smoke can stay shorter).

**Gate:** require the three engines to **agree within Δ = log10(3)** on each mapped species'
`magnitude_factor`. Report the per-species {magnitude_factor per engine, agree?} table.

**Why not gate on `in_range`:** because EEC/BoB are uncalibrated, `compare_outputs_to_ices` reports
`in_range=False` for essentially every species on all three engines (model total biomass
structurally over-estimates ICES SSB). A "same in/out relation" test where nothing is near a
boundary is trivially satisfied — so the gate is the cross-engine `magnitude_factor` agreement, not
the in/out flag.

**Docs must state plainly:** EEC/BoB are uncalibrated demo configs; B2 is cross-engine
consistency, not empirical validation. The genuine empirical anchor is Baltic ICES/HOLAS-3 on the
Python engine (Java-blocked, swap-unaffected). Empirical confidence is transitive: Python is
empirically validated (Baltic) and 4.4.1-Java matches Python on EEC/BoB (A4.4).

---

## 5. Task ordering and testing strategy

**Ordering (critical path — the in-place rewrite forces sequence):**
`A0` (Python-BoB hard gate) → `A1` (resample + unit test) → `A0b` (snapshot original config **before**
mutation) → `A2` (in-place fully-native conversion — its own substantial code task) →
`A4.1` (4.4.1 smoke) → `A4.2` (load-path-equivalence, needs A1's 24-step file + the original ltl
config for the intermediate) → `A4.3` (characterization, needs A0's working Python-BoB) →
`A4.4` (parity — needs the A0b snapshot for the OLD arm + the parametrized harness) →
`B1` (ICES snapshot, independent — can run in parallel) → `B2` (needs A4.4/Phase-2 results per
engine). The harness parametrization (config-path + tmp-path fix + absolute-gate + persist-one-
`OsmoseResults`) is a prerequisite shared by A4.4 and B2 — sequence it before both.

**One implementation cycle is reasonable IF** these genuinely-new code paths are each their own task
with TDD: (1) the forcing resampler `scripts/resample_bob_forcing.py` (A1); (2) the BoB conversion
in `migrate_bundled_to_440.py` (add-key + BoB rename + drop-family-across-included-file, name-gated);
(3) the `cross_engine_parity_440.py` parametrization (config-path + tmp-path fix + absolute-gate +
persist-one-`OsmoseResults`); (4) the `native_440_parity.py` load-path-equivalence extension; (5)
**existing-test remediation** (below). Plus the smaller owned steps: A0b snapshot (a `cp -r`/git
ref), B1 ICES snapshot build (mechanical, ICES MCP). If the plan gets large, split at the A/B
boundary (A must land before B regardless).

**Existing-test remediation (task 5) — A2's in-place rewrite breaks committed BoB fixtures.** Before
/ alongside A2, audit `tests/` for direct `data/examples`/`EXAMPLES_CONFIG` readers (grep first) and
handle at least these two: (a) `tests/test_engine_parity.py` `TestBaselineParity` does an `atol=0`
exact match against `tests/baselines/parity_baseline_bob_1yr_seed42.npz` +
`statistical_baseline_bob_1yr_10seeds.npz` — these WILL fail post-migration by design (load path +
forcing both change), so regenerate them (`scripts/save_parity_baseline.py --config bob`) after A2
and document that the regeneration reflects the intended forcing change (A4.3); (b)
`tests/test_engine_java_comparison.py:39` hardcodes the 4.3.3 jar against `EXAMPLES_CONFIG` — bump it
to the 4.4.1 jar (adjust assertions) OR repoint it at the A0b snapshot for a 4.3.3-only check. (The
parent plan's Task 9, `2026-06-19-...:176`, already flagged this file — it did not carry forward
before; it does now.)

**Rescope trigger:** if A4.1's smoke fails for a reason **unrelated to resource forcing** (e.g. the
legacy fishing keys on 4.4.1, §2), STOP and rescope — a fishing-scheme migration is separate,
unbudgeted work, not part of this cycle.

- **TDD per code task**: resampler; the BoB conversion (assert version stamp, `species.trophic.level`
  rename, `species.file.sp8-13`→24-step added, `ltl.*` fully dropped incl. the included sub-file,
  no leftover `ltl.name.rscN`); harness parametrization + absolute-gate wiring. ruff/pyright clean.
- **Gate scripts** (A4.1 smoke, A4.2 equivalence, A4.4 parity, B2 consistency) are runnable checks;
  A4.2 should also be a fast deterministic regression test.
- CI note: real-engine ensemble gates (A4.4, B2) are numerically non-reproducible across runner
  core counts (`feedback-ci-fragile-emergent-tests`) → run locally / mark CI-skip, don't chase.

---

## 6. Success criteria

- BoB runs clean on the Python engine at a pinned `nyear` with the 365-step forcing (A0, hard gate).
- Pre-migration config snapshotted (`data/examples_433_orig/`) before A2 — serves the A4.4 OLD arm
  and config rollback (A0b).
- `data/examples` is fully-native 4.4.1 (`ltl.*` dropped incl. the included sub-file,
  `species.trophic.level` renamed, `species.file.sp8-13` → 24-step file) and runs the pinned
  `nyear` on the 4.4.1 jar with no load/parameter error (A2, A4.1).
- BoB key conversion proven lossless on the Python engine via the load-path-equivalence gate
  (A4.2, bit-exact under the stated preconditions).
- BoB **absolute** Python-24 ↔ 4.4.1-Java-24 equivalence PASS at Δ=log10(3), with the absolute
  test wired into the harness gate (A4.4).
- EEC + BoB ICES snapshot committed with honest coverage map (B1); cross-engine
  `magnitude_factor` consistency reported for all three engines within Δ (B2).
- Docs/CHANGELOG: BoB is the 5th fully-native 4.4.1 bundled config; Phase 3 result with the
  consistency-not-realism framing; 4.3.3 jar retained for rollback.

---

## 7. Honest caveats

- BoB migration **changes BoB dynamics slightly** (bin-average forcing vs the old single-day
  subsample) — deliberate and characterized (A4.3), not neutral. This is the one place BoB
  departs from the C1 "pure key rename" pattern.
- Resource **trophic level** is diagnostic-only in this model (EEC defaults it and parity passed);
  we rename `species.tl`→`species.trophic.level` for strict losslessness, not because dynamics
  depend on it.
- The A4.4 relative-vs-4.3.3 comparison spans a forcing change (365 vs 24 step) and is a tripwire
  only; the load-bearing BoB claim is the absolute Python-24 ↔ 4.4.1-Java-24 equivalence, which
  this design *adds* to the harness gate (it is not gated today).
- Phase 3 is a **consistency tripwire on uncalibrated demo configs**, not empirical realism: only
  ~2/14 EEC and ~4 BoB species map cleanly to tonnes-unit ICES stocks; the validator compares
  total model biomass vs ICES SSB (a structural over-estimate) and does not gate reference points;
  `in_range` will be uniformly False, so the gate is cross-engine `magnitude_factor` agreement.
