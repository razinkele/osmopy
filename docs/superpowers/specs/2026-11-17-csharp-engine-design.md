# C# simulation engine (third backend): design

> Date: 2026-11-17. Status: proposal for owner review. No code exists yet.
> Read against master `a1fa75e4`. Python citations are by **file and function name**; line
> numbers drift (CLAUDE.md's `simulate.py:836` for `_collect_biomass_abundance` is already at
> 1080), so re-find a function with `grep -n` before relying on a number.
> Companion: `2026-11-17-csharp-engine-implementation-plan.md`.

Every number is labelled **measured** (with its source), **read** (from code, at the commit
above) or **inferred**. Nothing was run for this document except a few seconds of
`EngineConfig.from_dict` introspection; no simulation, no benchmark, no .NET code.

---

## 0. Decisions at a glance

| # | Topic | Decision |
|---|---|---|
| D1 | Runtime | **.NET 10 LTS** (`net10.0`), C# 14. This deviates from the brief; see section 2. |
| D2 | School layout | **Split array-of-structs**: four parallel arrays of blittable 64-byte structs grouped by access pattern, plus one `double[]` length lane. Rebuilt in cell order once per step. |
| D3 | Step structure | Python's process order, unchanged. Inside mortality the loops are **inverted: cell outer, sub-step inner**. |
| D4 | Parallelism | One `Parallel.For` over worker slots per step; workers claim cells from a cost-sorted list. Movement is parallel over schools. |
| D5 | SIMD | Only for the length-window prefilter and for exactly-rounded element-wise passes. Never for `exp`/`pow`, never for a reduction whose order would change. |
| D6 | RNG | **Counter-based Philox4x64-10**. Every draw is a pure function of (seed, step, sub-step, cell or school id, purpose, index). |
| D7 | Bit-equality with Python or Java | **Out of scope.** No cheap route exists (section 5.5). Kernel-level exact tests use injected draws. |
| D8 | Contract | `OsmoseEngine.RunSimulation(SimulationConfig)`. Python resolves everything to dense typed arrays; C# never sees a config key, a file or a species name it must interpret. |
| D9 | Marshalling | Zero-copy both ways: NumPy buffer addresses in, Python-allocated NumPy result buffers filled in place. |
| D10 | Cancel, progress, timeout | A shared `int64` control block and a deadline in the bundle. No callback into Python in the MVP. |
| D11 | Unsupported features | **Fail closed.** A capability manifest plus `csharp_engine_block_reason`; the engine refuses, it never ignores. |
| D12 | MVP | `data/eec_full` and `data/examples` (Bay of Biscay), bioen off, run with `output.minimal.enabled` (its four arrays). Stock configs without that key are refused until Phase 2. |
| D13 | Go/no-go | Two spikes before any engine code: the pythonnet bridge, and a kernel benchmark against Numba. |

---

## 1. Context and goals

### 1.1 What is being added

A third engine beside `PythonEngine` (`osmose/engine/__init__.py`) and the Java runner
(`osmose/runner.py`). The owner's three requirements:

1. A C#-native design, not a transliteration of the Java or Python structures.
2. Struct arrays for the fish schools.
3. A pythonnet bridge exposing one entry point, `RunSimulation(config)`, that takes parameters
   Python has already parsed, runs the whole multi-year simulation, and returns biomass and
   catch arrays to Python.

The Shiny UI, the CMEMS downloads, config reading (`OsmoseConfigReader`,
`EngineConfig.from_dict`) and all xarray/NetCDF work stay in Python.

### 1.2 Goals

- **G1. Same model.** The C# engine implements the semantics of the Python engine's
  production path, listed in section 6. Where Python and Java differ, Python is the reference.
- **G2. Deterministic.** A given bundle and seed give bit-identical output at any thread count
  and under any scheduling, on a given machine image. Across machines the only source of
  difference is the platform's `exp` and `pow` (risk R7).
- **G3. Faster where it is structurally possible**, with targets that are stated as inferred
  and gated by measurement (section 10.4).
- **G4. A third backend, not a fork.** One selection mechanism for the UI, calibration and
  scripts; existing Python and Java paths untouched.

### 1.3 Non-goals

- Bit-equality with the Python or Java engines (D7).
- Replacing the Python engine. It stays the reference and the only engine for features the
  C# engine has not reached.
- Reading configuration, NetCDF or CSV in C#.
- Porting what Python parses but does not apply (section 6.10).
- A GPU path. The plan's appendix already rules it out for this algorithm
  (`docs/osmopy-performance-optimization-plan.md`, "Phase 6").

### 1.4 What the measurements say the design must serve

All measured, sources in `docs/osmopy-performance-optimization-plan.md` and `docs/perf/2026-11-*.md`.

- One-thread 10-yr Baltic run: mortality is about 78% of the time, movement 11%, everything
  else 11% (profile of 2026-11-14). At 14 threads: mortality 51%, **movement 25% and serial**,
  the rest 24%.
- Median Baltic cell: 26 schools (mean 29, max 390). 57.5% of scanned pairs sit in cells over
  50 schools. About 88% of scanned prey fail the size window.
- Baltic, EEC and Bay of Biscay all run `mortality.subdt = 10` (read from the three configs).
- The #279 prefilter saved 10.6% of a one-thread Baltic run by building per-predator candidate
  lists once per step. It is neutral at 14 threads because its build is serial.
- A contiguous per-cell gather was built in September, was bit-exact, and was **slightly
  slower** (5.342 vs 5.376 s; the plan's table does not name the config, the timings match EEC,
  inferred). EEC cells are small; Baltic was not measured.
- The June native-C leaf spike reported 10-17x. The plan itself says not to plan against that
  ratio (it measured call overhead). **This document does not use it.**

Two consequences shape everything below:

1. There is no evidence that a managed JIT beats Numba's LLVM output on the same scalar loop.
   The honest expectation is parity or slightly worse per instruction (inferred). Any gain
   must come from **structure**: locality, less orchestration, parallel movement, fewer random
   draws, no per-step array rebuilds.
2. The September negative on contiguous layout means the layout decision in section 3 is a
   hypothesis until the kernel spike measures it on Baltic-sized cells.

---

## 2. Runtime and language target

**Decision D1: target .NET 10 LTS (`net10.0`).**

**Why this deviates from the brief.** The brief asked for .NET 8 LTS, or .NET 9 if justified.
Microsoft ended support for **both .NET 8 and .NET 9 on 2026-11-10**, one week before this
document; .NET 10 is the current LTS, supported to November 2028 (Microsoft .NET blog,
"dotnet-8-9-end-of-support"; fetched 2026-11-17). Starting a new component on a runtime that no
longer receives security patches is not defensible, and the box's `~/.dotnet` already carries
.NET 10 first-use markers.

**Why .NET 10 costs nothing in design terms.** Nothing below needs a post-.NET 8 API. The
project multi-targets `net10.0;net8.0` at no cost, so a host that only has the .NET 8 runtime
can still load it. If the owner prefers to pin .NET 8 anyway, only the `TargetFramework`
changes (open question Q1).

**Alternatives rejected.**

- *.NET 8 only.* Out of support.
- *.NET 9.* Out of support, and never was an LTS.
- *NativeAOT shared library with a C ABI, loaded by `ctypes`/`cffi`.* Technically attractive:
  no CLR hosting, no pythonnet, startup in milliseconds. Rejected because the owner asked for
  pythonnet. The contract in section 8 is deliberately pointer-and-blittable, so this stays
  available as a fallback if the bridge spike fails (risk R1).
- *Mono.* pythonnet's Linux default. Slower JIT, no `Vector512`, no tiered PGO. Rejected.

**Runtime settings** (in `Osmose.Engine.runtimeconfig.json`):
`System.GC.Server=false`, `System.GC.Concurrent=false`, tiered PGO on (the default). Workstation
non-concurrent GC because the engine allocates nothing per step (section 4.5) and calibration
runs one process per worker.

**pythonnet.** Version 3.2.1 (released 2026-10-08) supports CPython 3.11 to 3.15 (PyPI page,
fetched 2026-11-17). The dev `.venv` is 3.12 and production's micromamba env is 3.13.14
(checked). Its documentation states that coreclr needs .NET Core 3.1 or later and that
**self-contained deployments are not supported**, so the engine is published
framework-dependent and a .NET runtime must be installed on every host. Whether pythonnet
3.2.1 and `clr_loader` load a .NET 10 runtime cleanly was **not verified** here; it is the
first check of the bridge spike.

---

## 3. Data layout for schools

### 3.1 What the code touches, per access pattern

Read from `osmose/engine/state.py::SchoolState` (27 per-school fields, today 27 separate
NumPy arrays) and the kernels in `osmose/engine/processes/mortality.py`.

| Pattern | Frequency | Reads | Writes |
|---|---|---|---|
| **Prey scan** (`_apply_predation_numba`, Phase 1a) | per predator visit x per school in the cell x `n_subdt` | prey `length`; for survivors of the window: `inst_abd`, `egg_retained`, `weight`, access row | none |
| **Prey kill** (Phase 2) | per eaten prey | prey `weight`, prey `trophic_level` | prey `n_dead[PREDATION]`, `inst_abd` |
| **Predator bookkeeping** (Phase 3) | per predator visit | own `length`, `weight`, `inst_abd`, stage, species, age | own `pred_success_rate`, `preyed_biomass`, `tl_weighted_sum` |
| **Single cause** (`_apply_single_cause`) | 3 per school per sub-step | own rate, `inst_abd` | own `n_dead[cause]`, `inst_abd` |
| **Movement** | per school per step | species, age, cell | cell, `is_out` |
| **Growth** | per school per step | age, species, length, success, `is_out`, abundance | length, weight, biomass |
| **Reproduction** | per school per step | length, age, abundance, weight, species | age, `is_egg`; **appends** egg schools |
| **Outputs** | per school per step | species, age, abundance, weight, all `n_dead` | none |
| **Compaction** | per school per step | everything | everything |

The first three rows are the 78%. The prey scan is the inner loop, and 88% of its iterations
need **only the length**.

### 3.2 The three candidates

**A. One array of one struct, `School[]`.** A full school is about 250 bytes without
bioenergetics: 8 death counters, about 17 other doubles, a handful of ints and flags. That is
four cache lines per school. A prey scan that needs 8 bytes would pull 250. Compaction and the
cell sort would move 250 bytes per school. Rejected: it is the layout the owner's phrase
suggests, and it is the worst one for the dominant loop.

**B. Pure struct-of-arrays** (today's Python layout). Dense for a single-field pass, and a
length lane is exactly what the window test wants. But a surviving prey then needs
`inst_abd`, `egg_retained`, `weight` and its access row from four different arrays, reached
through an index (`cell_indices[q_pos]`), so up to four cache misses per candidate. A kill
touches two more. Twenty-seven arrays also make the per-step reorder (section 3.4) twenty-seven
gathers. Rejected as the primary layout; kept for the one lane where it wins.

**C. Split array-of-structs, hot and cold, plus one scalar lane.** Chosen.

### 3.3 Decision D2: the layout

Four parallel arrays of blittable structs share one school index, and a fifth array holds a
copy of the lengths.

```csharp
[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolCore            // everything a prey scan, a kill or a cause visit touches
{
    public double Length;           // cm
    public double Weight;           // tonnes per fish
    public double Abundance;        // instantaneous abundance inside mortality; committed outside
    public double EggRetained;      // eggs still withheld from the prey pool
    public double DeadPredation;    // n_dead[PREDATION]: the only death counter written per eaten prey
    public double TrophicLevel;     // read per eaten prey (section 6.4, item 10); rewritten after the loop
    public int    SpeciesId;        // focal 0..n-1, background n..n+b-1
    public int    AgeDt;
    public short  PreyAccessRow;    // -1 = unresolved (the "full access" trap, section 6.4)
    public short  PredAccessCol;    // -1 = unresolved
    public byte   FeedingStage;
    public SchoolFlags Flags;       // IsEgg | IsOut | IsBackground | FromSeeding
    public short  FirstFeedingAgeDt;// 1 for focal schools, -1 for background
}

[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolStep            // rebuilt every step; the cause loop's working set
{
    public double RateStarvation;   // per sub-step, already divided (section 6.3)
    public double RateAdditional;
    public double RateFishing;
    public double DiscardFraction;
    public double EggRelease;       // post-larva abundance / n_subdt, eggs only
    public double PredSuccess;
    public double PreyedBiomass;
    public double TlWeightedSum;
}

[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolLedger          // death accounting; read by outputs and the commit
{
    public double DeadStarvation, DeadAdditional, DeadFishing, DeadOut;
    public double DeadForaging, DeadDiscards, DeadAging;
    public double AbundanceAtMortalityStart;
}

[StructLayout(LayoutKind.Sequential, Size = 32)]
public struct SchoolLife            // cold: touched once per step
{
    public ulong  Id;               // stable identity, keys the movement RNG (section 5)
    public double StarvationRate;   // lagged: computed this step, applied next step
    public double GonadWeight;      // bioen only; zero otherwise
    public int    CellId;           // y * nx + x, or -1 when unlocated or out of domain
    private int   _reserved;
}

public sealed class SchoolStore     // owns the buffers; capacity doubles, never shrinks in a run
{
    public Span<SchoolCore>   Core   { get; }
    public Span<SchoolStep>   Step   { get; }
    public Span<SchoolLedger> Ledger { get; }
    public Span<SchoolLife>   Life   { get; }
    public Span<double>       LengthLane { get; }   // == Core[i].Length, contiguous per cell
    public ReadOnlySpan<int>  CellStart { get; }    // n_cells + 1 offsets into the arrays
    public int Count { get; }
}
```

Bioenergetics adds a fifth struct array (`SchoolBioen`: `ENetAvg`, `EGross`, `EMaint`, `ENet`,
`Rho`, `CapPerFish`, `RateForaging`, `Eta`), allocated only when bioen is on.

**Why this split.**

- *Bytes per scanned prey.* The window prefilter reads the length lane only: 8 bytes per
  school, eight schools per cache line, contiguous, and a fit for `Vector256`/`Vector512`
  (section 4.6). The 12% that survive read exactly one `SchoolCore` line, which holds every
  other field Phase 1a needs.
- *Bytes per kill.* `DeadPredation`, `Abundance` and the prey's `TrophicLevel` (read for
  every eaten prey, because trophic tracking is always on in production) sit in the line the
  scan just loaded. Python touches `n_dead[q, 0]`, `inst_abd[q]` and `trophic_level[q]` in
  three different arrays.
- *The cell id is not in the hot line.* During mortality a school's cell is implied by
  `CellStart`; only movement and the regroup read `CellId`, so it lives in `SchoolLife`.
- *A cause visit* touches `SchoolStep` (rate), `SchoolCore` (abundance) and `SchoolLedger`
  (counter): three lines, against three arrays today. No loss.
- *`SchoolStep` is disposable.* It is rebuilt each step, so the reorder never moves it.
- *Cold data stays out of the hot lines.* Identity, the cell id, the lagged starvation rate
  and gonad weight are read once per step.

Total without bioen: 232 bytes per school. At 40,000 schools that is about 9 MB, twice (double
buffer, section 3.4).

**How "struct arrays" is honoured.** Schools are stored as arrays of blittable structs and
every hot loop runs over `Span<SchoolCore>`. What is declined is the literal reading, one
struct holding all 27 fields, because the measured access pattern punishes it. The length lane
is the single field kept in scalar form, and it is a redundant copy, not the source of truth.

**Dropped fields.** `length_start` is written in `_reset_step_variables` and read nowhere in
`osmose/` (checked with `grep`). `biomass` is always `abundance * weight` and is computed where
needed. `imax_trait` is documented as vestigial in `state.py`. None is carried.

### 3.4 Cell order: one regroup pass per step

Python sorts an index array by cell (`mortality()`, the `np.argsort(..., kind="mergesort")`
block) and gathers through it; it compacts dead schools in a separate pass at the end of the
step (`SchoolState.compact`), appends background schools with a concatenation and strips them
with a slice.

**Decision.** One pass, `SchoolStore.Regroup`, runs after movement and before mortality. It is
a counting sort on `CellId` from the front buffer into the back buffer that, in the same sweep:

- drops schools with `Abundance == 0` (the compaction);
- places each cell's schools contiguously and records `CellStart`;
- writes the step's background schools at the end of their cell's span;
- puts unlocated and out-of-domain schools (`CellId == -1`) in a tail segment;
- copies `Core.Length` into the length lane.

Then the buffers swap. Cost is one linear pass over about 9 MB per step (inferred: about a
millisecond, against roughly 110 ms per step at one thread on Baltic).

Within a cell, schools keep their previous relative order, so the order is deterministic. It
is **not** Python's order, and need not be: the guardrails about within-cell order, `n_local`
and the #279 sort-back (`docs/osmopy-performance-optimization-plan.md`, section 6, items 1-2)
protect Python's `atol=0` baselines against Python's own random stream. A different engine with
a different stream is bound only by determinism.

**Equivalence of the moved compaction.** Python compacts at the end of step `t`; C# drops dead
schools during step `t+1`'s regroup. Between those two points Python runs nothing on the
school table, and C# runs incoming flux, the reset, the resource update and movement. Rule:
those four skip `Abundance == 0`. Movement's draws are keyed by school id (section 5), so
skipping a dead school shifts no other school's draws.

**Appends.** Reproduction and incoming flux append at the tail, serially, and assign
`SchoolLife.Id` from a run-wide counter. Capacity doubles when exhausted; both buffers grow
together. No per-step allocation otherwise.

**Alternative rejected: keep schools in creation order and index by cell.** It is what Python
does and what the September experiment compared against. It keeps the indirection in the inner
loop and rules out the length lane. If the kernel spike shows the regrouped layout is not
faster on Baltic-sized cells, this is the fallback, and only `SchoolStore.Regroup` and the
kernel's addressing change.

---

## 4. Process pipeline and parallelism

### 4.1 The step

The order is `simulate()` in `osmose/engine/simulate.py`, unchanged. Optional features are in
brackets.

```
for step in 0 .. n_steps-1:
   0  cancel / deadline check                       (control block, section 8.5)
      [economics annual reset] [dynamic accessibility at year boundaries]
   1  incoming flux: append schools                 [optional]
   2  reset per-step variables
   3  resources: set this step's biomass per cell   (reset to K, or regrow when depletable)
   4  movement                                      parallel over schools
   5  regroup: compact + cell sort + inject background
   6  mortality
        6a larva pre-pass on eggs
        6b egg retention; per-school rates; feeding stages; access indices
        6c per cell, per sub-step: interleaved causes     parallel over cells
        6d commit abundance; out-of-domain mortality; next step's starvation rate; trophic level
   7  background outputs; from here on no pass sees a background school
   8  floor schools under one fish to zero
   9  growth                                        [or the bioen energy budget]
  10  aging mortality
  11  reproduction: SSB, seeding, eggs, age += 1, append egg schools
  12  collect outputs into the record window; flush a record every record_freq steps
```

Background schools are physically dropped by the next regroup, but logically they end at
stage 7, as `_strip_background` does in Python: stages 8 to 12 iterate focal schools only (a
flag test on `IsBackground`), so no background school is floored, grown, aged, counted in the
spawning stock or in the focal outputs.

Each stage is a method on an internal `StepPipeline`. There is no class per process and no
virtual dispatch in a loop: optional features are resolved once, at run start, into a list of
delegates for the per-step stages and into `static` generic specialisations for the per-cell
kernel (`MortalityKernel<TBioen>` with struct type parameters, so the bioen-off kernel carries
no bioen branch).

### 4.2 Decision D3: cell outer, sub-step inner

Python (and Java) loop sub-steps on the outside and cells on the inside: `mortality()` calls
the batch kernel once per sub-step.

**Decision.** The C# engine processes one cell through **all** its sub-steps before moving to
the next cell.

**Why it is the same model.** Nothing in the sub-step loop crosses a cell (read in
`mortality()` and the kernels):

- schools do not move during mortality;
- resource biomass is `rsc_biomass[r, cell_id]`, one column per cell;
- egg release is per school, from the step-start abundance;
- every write (`n_dead`, `inst_abd`, `pred_success_rate`, `preyed_biomass`,
  `tl_weighted_sum`, the diet row) is at an index in the cell's own slice, which is the
  disjointness argument `_mortality_all_cells_parallel`'s docstring already makes;
- Python draws a fresh seed per sub-step and derives a per-cell seed from it, so its stream is
  already per (sub-step, cell). The C# stream is keyed the same way.

So the two loop orders compute the same thing for the same draws.

**Why do it.** A cell's schools stay in cache across ten sub-steps instead of being evicted
ten times. There is one fork/join per step instead of ten. The length-window results are
reused across sub-steps (section 4.6). Expected gain is unmeasured.

### 4.3 The cell kernel

```csharp
internal static class MortalityKernel<TMode, TDraws>
    where TMode : struct, IKernelMode      // bioen on/off, diet tracking on/off
    where TDraws : struct, IDrawSource     // struct type parameter: no interface dispatch
{
    // One cell, all sub-steps. Writes only inside the cell's own spans.
    public static void RunCell(
        in StepContext ctx,                 // step, n_subdt, species tables, access matrix
        int cellId,
        Span<SchoolCore> core,              // the cell's schools, contiguous
        ReadOnlySpan<double> lengthLane,
        Span<SchoolStep> step,
        Span<SchoolLedger> ledger,
        Span<double> resourceBiomass,       // this cell's n_rsc values, contiguous
        ref CellScratch scratch,            // worker-owned buffers
        ref TDraws draws);                  // Philox in production, injected arrays in tests
}
```

Per cell: build the per-predator window masks once (section 4.6). Then for each sub-step:
release eggs; draw four permutations (five under bioen) and one cause-order code per school
slot; walk the slots applying each cause to its own permuted school, exactly as
`_mortality_all_cells_parallel` does.

Two storage changes follow from "one cell at a time":

- Resource biomass is stored **cell-major**, `(n_cells, n_rsc)`, so a cell's resources are one
  contiguous span. Python stores `(n_rsc, n_cells)`.
- The stage accessibility matrix is stored **predator-major**. Python reads
  `access_matrix[q_acc, p_acc]` (prey row, predator column); for one predator scanning many
  prey the predator is fixed, so its column becomes a contiguous row.

### 4.4 Decision D4: scheduling

**Cells are very unequal** (median 26 schools, max 390, and cost grows faster than linearly
with occupancy). `Parallel.For` over the cell range with its default range partitioner hands
each worker a contiguous block of cells; one block holding the dense cells finishes last while
the others idle.

**Decision.** Per step:

1. Build the list of occupied cells and an integer cost estimate per cell from its
   occupancy alone, `n_local^2 + c * n_local`, read off `CellStart`. No pre-pass: the window
   masks are built inside `RunCell` (section 4.3), where the cell's data is about to be used
   anyway. The estimate only has to order the cells, not predict their time.
2. Sort the list by cost, descending.
3. Run `Parallel.For(0, workers, slot => ...)`. Each slot loops
   `while ((i = Interlocked.Increment(ref next)) < count) RunCell(cells[i], ref scratch[slot])`.

Heaviest first with single-cell claims is the classic longest-processing-time heuristic: the
largest cells are spread across workers and the many small cells fill the gaps. The claim is
one interlocked increment per cell (tens of nanoseconds against microseconds per cell).

Results do not depend on which worker runs a cell or in what order, because cells are disjoint
and draws are counter-based.

**`workers == 1` takes a plain `for` loop on the calling thread**: no delegate, no thread-pool
hop. That is the calibration regime (`thread_policy.worker_thread_count`).

**Alternatives rejected.**

- *`Parallel.For` over cells, default partitioner.* Poor balance, as above.
- *`Partitioner.Create(0, n, 1)` or `EnumerablePartitionerOptions.NoBuffering`.* Gives dynamic
  single-cell claims but not the cost ordering, and adds enumerator overhead per claim.
- *A dedicated thread crew with a barrier.* Saves the fork/join. With one join per step
  (section 4.2) that is about 240 joins per simulated decade, so there is nothing to save.
- *Splitting one huge cell across threads.* Within a cell the causes mutate shared abundance in
  a permuted sequential order; that is the algorithm. Not splittable without changing the model.

**Other parallel passes.**

- *Movement.* Each school's move depends only on its own state, the maps and its own draws.
  `Parallel.For` over fixed chunks of 2,048 schools. This removes the largest serial share of a
  threaded Baltic run (25%, measured).
- *Per-school passes* (rates, growth, reset, commit, aging). Same fixed-chunk loop, used only
  above a size threshold.
- *Regroup, reproduction, appends, id assignment.* Serial. They are linear and order-defining.
- *Output reductions.* Per-species sums are computed per fixed chunk of 4,096 schools and the
  chunk partials are added in ascending chunk order. The chunk size is a constant, so the
  floating-point result is the same at any thread count.

### 4.5 Memory: nothing allocated per step

- **Run-scoped buffers** (school arrays, length lane, resource biomass, per-worker scratch,
  window masks, the record window) are rented from `ArrayPool<T>.Shared` at run start and
  returned in a `finally`. A calibration worker that calls `RunSimulation` thousands of times
  reuses the same memory and never touches the large object heap after the first run.
- **`CellScratch`** is one per worker slot: permutation buffers, the prey buffer, mask words.
  It is sized for the largest cell of the step, once per step.
- **`stackalloc`** for fixed small scratch only: the Philox output block (32 bytes), the cause
  table (5 bytes), the mask words of cells up to 512 schools (64 bytes). Larger cells use the
  worker scratch. No `stackalloc` whose size depends on unbounded data.
- **No LINQ, no closures, no boxing, no `params` in any per-step path.** The `Parallel.For`
  body is one cached delegate.
- **Bounds checks.** The kernel validates span lengths once per cell and then walks `ref`s
  (`Unsafe.Add`). A `CHECKED` build configuration keeps ordinary indexing and is what the test
  suite runs.

Acceptance test: at one thread, `GC.GetAllocatedBytesForCurrentThread()` does not change across
steps 2 to N of a run whose capacity was pre-sized.

### 4.6 Decision D5: SIMD, where it applies and where it does not

**Applies: the size-window prefilter.** For one predator the eligible prey lengths are the
interval `(L/r_max, L/r_min]`. The kernel compares the cell's length lane against that
interval, widened by a relative `1e-9` on both sides, with `Vector256<double>` (or
`Vector512<double>` if the kernel spike shows it pays on this CPU), and stores the result as a
bitmask, one bit per school. The 88% that fail cost two vector compares per four or eight schools and no
division.

The survivors then take the **exact scalar test**, verbatim from `_apply_predation_numba`:
`ratio = pred_len / prey_len; if ratio < r_min or ratio >= r_max: skip`. The widened interval is
a superset, so the eligible set is exactly Python's. This is the same argument as #279
(`_cell_prey_candidates`), without the sort, the binary searches or the sort-back, because
walking set bits already visits prey in cell order.

Edge cases follow `_cell_prey_candidates`, so the mask is a superset in every case:
`r_max <= 0` gives an empty mask (no positive ratio is below `r_max`); `r_min <= 0` removes the
upper length bound; a NaN ratio bound gives a full mask and leaves the decision to the exact
tests; an inverted window gives an empty mask. `EngineConfig` already swaps inverted ratios, so
that last case guards memory, not results.

The masks depend only on lengths, stages and species, which do not change during mortality
(`mortality()` builds its candidate lists once per step for the same reason). So they are built
once per cell and reused by all ten sub-steps.

**Applies: exactly-rounded element-wise passes.** The reset, egg release, the abundance commit
`max(0, a - dead)`, the length-lane copy. On finite inputs IEEE add, subtract, multiply,
divide, min, max and compare give the same bits in scalar and in any vector width, so results
do not depend on whether the host has AVX2 or AVX-512.

**Does not apply.**

| Where | Why not |
|---|---|
| `exp` in `1 - exp(-D)`, selectivity, growth; `pow` in the length-weight relation | A vector `exp` is a different polynomial from scalar `Math.Exp`. Mixing them, or letting the choice depend on vector width, breaks G2. CLAUDE.md's Numba-versus-NumPy `exp` note is this trap. All transcendentals are scalar `Math.Exp`/`Math.Pow`. |
| Phase 2, sharing the eaten biomass among prey | Sequential scatter into prey that later predators in the same cell read. |
| The interleaved cause loop | Each visit reads abundance the previous visit wrote. |
| Fisher-Yates, rejection sampling | Data-dependent control flow. |
| Per-species sums | Scatter-add; and a vector reduction would change the summation order. |
| Fused multiply-add | Never used, explicitly or implicitly: it changes rounding. RyuJIT does not contract `a*b+c` on its own; the code never calls `Math.FusedMultiplyAdd`. |

**Alternative rejected: port #279's sorted candidate lists.** They win in Numba because a
linear scan there is scalar. With a vector compare over a dense lane the scan costs less than
the list bookkeeping, and the build disappears from the serial part of the step. To be
confirmed by the kernel spike; if the masks lose on the largest cells, the lists are a drop-in
replacement behind the same interface.

---

## 5. Random numbers and reproducibility

### 5.1 What Python does today (read)

- `PythonEngine._prepare_run` builds `np.random.default_rng(seed)` (PCG64).
- Mortality: per sub-step, `mortality()` draws one 63-bit seed from that generator. In the
  parallel kernel each cell reseeds Numba's MT19937 with `rng_seed + cell * 7919`, then draws
  four permutations (five under bioen) and shuffles the cause list once per school slot.
- Movement: one seed per step, then a single serial MT19937 stream over all map-moved schools
  (`_map_move_batch_numba`). That serial stream is why movement cannot be parallelised in
  Python without changing fixed-seed output (plan item 2).
- Random-walk species, patch building, incoming flux and warm-start placement draw from the
  main generator directly.
- The per-species movement and mortality generators built by `build_rng` are not consumed on
  the production path (settled on 2026-11-17, per the brief).

### 5.2 Requirements

1. Same output for a given seed at any thread count and under any scheduling.
2. Replicate seeds, including consecutive integers, give independent streams.
3. Cheap: Baltic needs on the order of 10^8 draws per simulated decade (inferred: about 20,000
   schools x 5 draws x 2,400 sub-steps).

### 5.3 Decision D6: counter-based Philox4x64-10

Every random number is `Philox4x64_10(key, counter)`, a pure function. There is no generator
state anywhere in the engine.

```csharp
public readonly struct PhiloxKey { public readonly ulong K0, K1; }

public static class Philox
{
    // Random123 Philox4x64-10. Fills 4 x 64 random bits.
    public static void Block(in PhiloxKey key, ulong c0, ulong c1, ulong c2, ulong c3,
                             Span<ulong> out4);
}

public enum DrawPurpose : ulong
{
    PermPredation = 1, PermStarvation, PermFishing, PermAdditional, PermForaging, CauseOrder,
    MoveMapPlace, MoveMapWalk, MoveRandomPlace, MoveRandomWalk,
    FluxPlace, InitPlace, PatchStart, /* later: FleetChoice, Genetics */
}
```

- **Key.** `K0 = SplitMix64(seed)`, `K1 =` a fixed domain constant XOR the RNG scheme version.
  Distinct keys give independent streams by construction, which is the property counter-based
  generators are designed and tested for; the SplitMix step is belt and braces for consecutive
  seeds.
- **Counter.** `c3 = purpose`, `c2 = (step << 16) | substep`, `c1 = entity` (the cell id for
  mortality, the school `Id` for movement and placement, the species for patch starts),
  `c0 = block index` within that draw sequence.
- **Bounded integers.** Lemire's multiply-shift with rejection on 64-bit words: unbiased.
- **Uniform doubles.** `(x >> 11) * 2^-53`.
- **Permutations.** Fisher-Yates, `i` from `n-1` down to `1`, `j = bounded(i + 1)`.
- **Cause order.** One bounded draw in `[0, 24)` (or `[0, 120)` under bioen) decoded through a
  Lehmer table into an order of the four (five) causes. A uniform draw over all orderings is
  the same distribution as Python's per-slot `shuffle`, for one draw instead of three or four.
- **Scheme version.** `rng.scheme = 1` is recorded in the bundle and the run summary. Any
  change to keys, counters or the decoding bumps it, so "same seed, different numbers" is
  always attributable.

**Why Philox4x64-10 specifically.** It is the generator `numpy.random.Philox` implements, so a
Python test can produce the engine's raw words from NumPy and feed the same permutations to the
Numba oracle kernel (section 10.1). The plan's first RNG task is a known-answer test pinning
the C# function to Random123's published vectors and establishing the exact NumPy call that
reproduces a block; if NumPy's stream convention cannot be lined up, a 30-line pure-Python
Philox replaces it in the tests.

**Why it satisfies the requirements.** No draw depends on which thread asks or when. School ids
are assigned in serial code. So requirement 1 holds by construction, not by a disjointness
argument. Requirement 3: a block of four words costs ten rounds of two 64x64-bit multiplies
(inferred: a few nanoseconds), so well under one second per Baltic decade.

### 5.4 Alternatives rejected

| Alternative | Why not |
|---|---|
| One generator per thread | Output depends on scheduling. Fails requirement 1. |
| One persistent generator per cell | Deterministic, but a cell's stream then depends on its whole history; movement would need one per school; independence between cells and between replicates rests on seeding discipline. |
| Reseed a small stateful generator per (step, sub-step, cell) from a hash | This is Python's scheme. Workable, but the per-cell seeding cost is paid 10^6 times per decade and independence across cells depends on the hash. Python's `rng_seed + cell * 7919` is the weak form of it. |
| Philox4x32-10 or Threefry | Equally sound. Not chosen because NumPy does not expose them, which would cost the cross-check. |
| `System.Random` or xoshiro via `Random.Shared` | Stateful, and its algorithm is not a contract across .NET versions. |

### 5.5 Decision D7: bit-equality with Python or Java is out of scope

Reproducing Python's numbers would need, at minimum:

1. PCG64 and NumPy's bounded-integer routine, for the seeds and the direct draws.
2. Numba's MT19937 seeding and Numba's own `permutation` and `shuffle` algorithms, per cell.
3. Python's within-cell school order and `n_local`, including not-yet-compacted dead schools.
4. NumPy's `exp`, which is NumPy's own SIMD routine and not libm (CLAUDE.md, the Numba `exp`
   note), and NumPy's `power` (probably the same; not checked), for every pre-computed rate,
   the growth curve and the length-weight relation.
5. NumPy's summation order in every reduction.

Items 1 to 3 are tedious but finite, and they would forbid sections 3.4, 4.2 and 5.3. Item 4
cannot be done from .NET at all short of linking NumPy's routines. So there is no cheap route,
and taking the expensive one would turn the port into the transliteration the owner asked not
to have.

Java is further away still: `osmose/engine/rng_compat.py` documents Java's generator as
`XSRandom`, an xorshift (CLAUDE.md says MT19937; one of the two notes is stale and it does not
matter here), and Python itself is only statistically equivalent to Java.

**What is offered instead.** Three tiers, defined in section 6.1: bit-exact process arithmetic,
libm-tolerance process arithmetic, and statistical equivalence of whole runs. Plus an
injected-draws seam (`IDrawSource`) so the mortality kernel can be compared with
`_mortality_in_cell_numba` on identical permutations.

---

## 6. Semantics contract

The reference is the Python engine's **production path**: `simulate()` and, for mortality,
`mortality()` with `_precompute_effective_rates` and the batch kernels
(`_mortality_all_cells_parallel`, `_apply_predation_numba`, `_apply_single_cause`). The
standalone functions `fishing.fishing_mortality`, `natural.additional_mortality` and
`starvation.starvation_mortality` are **not** on that path and differ from it (for example
they scale deaths, not the rate, by the spatial factor). They are not a reference for anything.

### 6.1 Three tiers of agreement

| Tier | Meaning | How it is tested |
|---|---|---|
| **E, exact** | Only IEEE add, subtract, multiply, divide, min, max, compare and integer logic. Given identical inputs the C# result equals Python's bit for bit. | `==` on arrays |
| **L, libm** | Involves `exp` or `pow`, or a sum whose order differs. NumPy, Numba and .NET disagree in the last unit on a few percent of arguments, and `1 - exp(-D)` amplifies that by about `1/D` (CLAUDE.md, the Numba `exp` note: 1.7e-10 worst case at engine rates). | relative tolerance, `1e-9` for `1 - exp(-D)` terms and `1e-12` elsewhere, with the worst observed error recorded, not just gated |
| **S, statistical** | Depends on random draws. Whole runs are always in this tier. | ensembles, section 10.2 |

A process test with injected draws is tier E or L. Nothing involving the engine's own random
stream is ever compared at tier E or L against Python.

**Tiers do not separate inside a coupled computation.** In a whole-cell mortality run, a death
count that came from `exp` lowers the abundance a later predator reads, so every later number
inherits the libm error. A whole-cell or whole-step comparison is therefore **one** tier, L.
Tier-E coverage of the predation arithmetic comes from fixtures whose starvation, additional
and fishing rates are all zero, where no `exp` is evaluated at all.

**The mortality oracle is Numba, not NumPy.** Numba lowers `exp` to the C library; .NET's
`Math.Exp` calls the C library too. Whether the two agree bit for bit on this machine is a
Phase 0 check. If they do, whole-cell comparisons run at `atol=0` and tier L applies only to
what NumPy computes (the pre-computed rates, growth, the length-weight relation).

### 6.2 Process order (tier E as a sequence)

Exactly the list in section 4.1. Points that are easy to get wrong, each read in `simulate()`:

- Background schools are injected **after** movement and stripped **after** mortality. Their
  biomass and abundance are recorded before the strip.
- The sub-one-fish floor runs after the strip and **before** growth, so growth, the spawning
  stock and the outputs all see the floored values (`_floor_sub_one_fish`).
- Aging runs after growth and before reproduction, with the test `age_dt >= lifespan_dt - 1`,
  because reproduction is what increments the age (`aging_mortality`).
- Reproduction increments `age_dt` for **every existing school**, recomputes
  `is_egg = age_dt < first_feeding_age_dt`, and only then appends the new egg schools at age 0.
- Outputs are collected **after** reproduction.
- The starvation rate computed at the end of mortality is applied in the **next** step
  (`update_starvation_rate`).

### 6.3 Per-school rates, built once per step (`_precompute_effective_rates`)

`denom = n_dt_per_year * n_subdt`. "Exempt" means: background schools, schools with
`age_dt < first_feeding_age_dt`, and any negative rate, all set to zero (`_zero_exempt`).

| Rate | Rule | Tier |
|---|---|---|
| Starvation | `starvation_rate / denom`, exempt. Zero under bioen. | E |
| Additional | species rate, replaced by `by_dt[step % len]` when a series exists; **multiplied by the spatial factor** (0 where the cell is outside the map, NaN or non-positive); exempt; `/ denom`; non-finite becomes 0. | E |
| Fishing | only when fishing is enabled. Species rate, replaced by `by_year[year]` when `year < len`. Times selectivity, spatial factor and MPA factor. **With seasonality: `* season[sp, step % ndt] / n_subdt`. Without: `/ denom`.** Exempt; non-finite becomes 0. | E, or L for type 1 |
| Selectivity | type 0: 0 when `age_dt / ndt < a50`, else 1. Type 1: `1 / (1 + exp(-slope * (L - l50)))`. **Every other type**, including -1: 0 when `l50 > 0 and L < l50`, else 1. | E / L |
| Discards | `discard_rate[sp]` where the fishing rate is positive, else 0. | E |
| Deaths | `dead = abundance * (1 - exp(-rate))`. Fishing splits it: `FISHING += dead * (1 - d)`, `DISCARDS += dead * d`, one abundance decrement of `dead`. | L |

The seasonality branch divides by `n_subdt` only. That is not a typo to fix; the seasonal
table already carries the within-year split.

### 6.4 Predation, one predator visit (`_apply_predation_numba`)

1. The predator does nothing if `age_dt < first_feeding_age_dt`, if its instantaneous abundance
   is not positive, or if `max_eatable <= 0`, where
   `max_eatable = abundance * weight * ingestion_rate[sp] / (ndt * n_subdt)`. (E)
2. **School prey**, in cell order, skipping itself. Available abundance is
   `max(abundance - egg_retained, 0)`; skip when not positive. Skip when `prey_len <= 0`.
3. **Size window.** `ratio = pred_len / prey_len`. **Reject when `ratio < r_min` or
   `ratio >= r_max`**, with `r_min`, `r_max` taken from `size_ratio_min/max[species, stage]`
   of the predator. The ratio is predator over prey; the lower bound is inclusive and the
   upper bound exclusive. (E)
4. **Accessibility, stage mode.** The coefficient starts at **1.0**. Only when **both** the
   predator's column index and the prey's row index are non-negative is the matrix read, at
   `[prey_row, pred_col]`, and the prey skipped when the coefficient is not positive. **An
   unresolved index, -1, leaves the coefficient at 1.0: full access.** This is CLAUDE.md's
   "-1 means FULL access" trap and it must be reproduced, not repaired. The bundle carries the
   -1 through (section 8.2), and the engine reports unresolved species in its run summary as a
   warning. (E)
5. `eligible = available * weight * coefficient`; skip when the biomass is not positive.
6. **Resource prey**, in resource order. Skip when the cell's biomass is not positive or when
   `r_min <= 0 or r_max <= 0`. The eligible share is the overlap of the resource's size range
   with `[pred_len / r_max, pred_len / r_min]`, divided by the resource's range. Accessibility
   as in 4, with the resource's row. (E)
7. `total_available` is the **running sum in scan order**: school prey first, in cell order,
   then resources. Stop when it is not positive. (E for a given order)
8. **Functional response.** Type I: `eaten = min(total_available, max_eatable)`. Types II and
   III: `r = total / max_eatable`, `g = r / (r + K)` or `r^2 / (r^2 + K^2)`, clamped by
   `min(1, r)`, `eaten = max_eatable * g`. (E)
9. Each prey loses `eaten * eligible_k / total_available`. A school prey loses that biomass
   over its weight in fish, recorded as predation deaths, when its weight and the count are
   positive. A resource's biomass becomes `max(0, biomass - eaten_k)`. (E)
10. The predator's weighted trophic sum gains `prey_tl * eaten_k`, with a prey trophic level
    that is not positive counted as 1.0. (E)
11. `pred_success += min(eaten / max_eatable, 1) / n_subdt`; `preyed_biomass += eaten`. (E)

Feeding stages are computed once per step before the loop: the count of thresholds that the
metric meets or exceeds (`compute_feeding_stages`, `searchsorted(..., side="right")`), with
metric size, age in years, weight in grams, or trophic level.

Stage accessibility indices are also per step: the index of the first stage whose threshold
exceeds the school's age in years, else the last stage
(`AccessibilityMatrix.compute_school_indices`).

### 6.5 The interleaved loop (tier S for the order, E for each application)

Per cell, per sub-step, with `n_local` schools (`_mortality_all_cells_parallel`):

- One permutation of the cell's schools **per cause**: predation, starvation, additional,
  fishing, and foraging under bioen.
- For slot `i` from 0 to `n_local - 1`: a fresh uniformly random order of the causes; each
  cause in that order is applied to **its own** permuted school, `perm_cause[i]`. So in one
  slot the four causes generally hit four different schools.
- Every school is therefore visited exactly once per cause per sub-step, and a predator eats
  once per sub-step.
- The cause set is predation, starvation, additional, fishing, plus foraging only under bioen
  (`_get_mortality_causes`). Out-of-domain and aging mortality are outside the loop.

Combining causes into one hazard would change the model (plan guardrail 4).

### 6.6 Eggs

- **Larva pre-pass** (`larva_mortality`): for `is_egg` schools only, once, before the loop:
  `dead = abundance * (1 - exp(-rate))` with the **full** rate, recorded as ADDITIONAL. The
  reader has already divided the annual file value by `n_dt_per_year`
  (`osmose/config/reader.py`, the 4.4.0 larval-rate migration; CLAUDE.md), so the bundle
  carries the resolved per-step value and C# divides nothing. (L)
- **Retention**: after the pre-pass, `egg_retained = abundance` for eggs.
- **Release**: at the start of each sub-step, for eggs with something still retained,
  `egg_retained = max(0, egg_retained - A / n_subdt)`, where `A` is the abundance after the
  pre-pass, **not** the instantaneous abundance (`mortality()`, the `release` expression). (E)
- Eggs are prey for the released part only. They are not predators and take no starvation,
  additional or fishing mortality inside the loop (the exemption above).
- **New egg schools** (`create_egg_schools`): none when the egg count is not positive or the
  species has no schools; **one** school holding everything when the count is below
  `n_schools`; otherwise `n_schools` equal schools. They are created **unlocated**; movement
  places them at the start of the next step. `first_feeding_age_dt` is 1.

### 6.7 After the loop (`mortality()`, `out_mortality`, `update_starvation_rate`)

- `abundance = max(0, A - sum of the loop's deaths over causes)`, summed in cause-enum order;
  then the larva deaths are added to the ADDITIONAL counter. (E for a given order)
- **Unlocated and out-of-domain schools skip the cell loop** (`valid = cell_x >= 0 & cell_y >= 0`)
  but take the larva pre-pass, out mortality and the rate update.
- Out mortality: for `is_out` schools, `dead = abundance * (1 - exp(-out_rate[sp] / ndt))`,
  once per step. (L)
- Next starvation rate: `max_rate * (1 - sr / csr)` when `sr <= csr` (dividing by 1 when
  `csr` is 0), else 0; clamped at 0; zero for background. (E)
- Trophic level: where `preyed_biomass > 0` and the weighted sum is positive,
  `TL = 1 + weighted_sum / preyed_biomass`. (E)

### 6.8 Floor, growth, aging, reproduction

- **Floor** (`_floor_sub_one_fish`): a school with `0 < abundance < 1` is set to 0. (E)
- **Growth** (`growth`, `expected_length_vb`): `delta = L(age + 1) - L(age)` from the
  three-phase von Bertalanffy curve (egg size at age 0; linear to the threshold age; the VB
  formula after). Schools at `age_dt == 0` or out of domain get `delta`. Others get
  `factor * delta * (sr - csr) / (1 - csr)` when `sr >= csr`, else 0 (and the full
  `factor * delta` when `csr >= 1` and `sr >= 1`). `L = min(L + growth, lmax)`;
  `W = c * L^b * 1e-6`. (L)
- **Aging**: `age_dt >= lifespan_dt - 1` moves the whole abundance to AGING. (E)
- **Reproduction** (`reproduction`, `regulate_recruitment`, `apply_stock_recruitment`):
  - Mature means `length >= maturity_size` **and** `age_dt >= maturity_age_dt` **and**
    `abundance > 0`. Python requires both keys where Java treats them as alternatives; the
    port follows Python.
  - `SSB = sum(abundance * weight)` over mature schools.
  - Seeding: when `SSB == 0.0` exactly and `step < seeding_max_step`, SSB becomes the seeding
    biomass and the species is "seeded this step".
  - `eggs = sex_ratio * fecundity * SSB * season * 1e6`.
  - The stock-recruitment curve (none, Beverton-Holt, Ricker, hockey stick, Shepherd), then, in
    this order, the RV gate, the ceiling, the thermal gate and the depensation gate. Every gate
    is skipped for a seeded species. Under `seeding_mode == "linear"` a seeded species also
    skips the curve.
  - Tier L (a sum, and `exp`/`pow` in Ricker and Shepherd).

### 6.9 Outputs (`_collect_outputs`, `_average_step_outputs`)

- **Biomass and abundance**, per species including background: focal schools with
  `age_dt / ndt >= output_cutoff_age[sp]` (`_collect_biomass_abundance`). Background values are
  the post-mortality ones and are added **without** the cutoff.
- **Mortality by cause**: focal deaths summed per species over all eight causes, no cutoff.
- **Yield**: `dead_fishing * weight` per focal species, no cutoff, **with the weight after this
  step's growth** (`_collect_yield` runs after `_growth`). Discards are not yield.
- **Record window**: biomass and abundance are **means** over the window; mortality and yield
  are **sums**. A window of one step is passed through. A trailing partial window is flushed
  with its own length. With `output.step0.include`, a record at step -1 precedes the run.
- Later families keep their own rules, in particular: the by-age and by-size distributions do
  **not** apply the cutoff, and mortality-rate outputs are **sums of per-step rates** that read
  `inf` when a stage is emptied (CLAUDE.md). The engine output is reproduced as is; robustness
  is the reader's job.

### 6.10 Deliberately not reproduced

| Python or Java behaviour | C# engine | Why |
|---|---|---|
| Catch-based fishing, by-age/size fishing and additional rates: parsed, never applied, warned about (`EngineConfig._warn_unsupported_mortality_features`) | refuses the config | Reproducing "ignored" silently in a new engine would be a second copy of a known gap. |
| Selectivity types 2 and 3 run as a length knife-edge, with a warning | refuses the config | Same. |
| A NaN length passes both window comparisons and becomes eligible | asserts lengths are finite and positive in checked builds | An input defect, not model behaviour. |
| The bioen-off stream draws a fifth, unused permutation in the per-cell oracle | not applicable | It protects Python's baselines only. |
| `java_compat_rng`, the per-species generator lists, `length_start`, `imax_trait` | not carried | Unused on the production path. |
| Java's background-school weights in grams, `BG_ACCESS` staging, first-definition-wins keys | not applicable | They are Java-staging matters. Like Python, the C# engine keeps weights in tonnes and receives one resolved matrix. |
| `out_mortality` not rescaling `e_net` for out-of-domain schools | follows Python | CLAUDE.md documents it as inert. |

### 6.11 Movement, forcing and injected schools

`_reset_step_variables` clears `is_out` before movement, every step.

**Map movement** (`_map_move_batch_numba`, `_precompute_map_indices`), per school of a
"maps" species:

1. The map index is `index_maps[age_dt, step]`, or -1 outside the table. A negative index, a
   species without maps, or a null map sends the school **out of domain**: cell -1 and
   `is_out` set. (E)
2. **Same map** means: `age_dt > 0` and `step > 0` and the index equals
   `index_maps[age_dt - 1, step - 1]` (-1 when that is outside the table). (E)
3. **Placement**, when the map changed or the school is unlocated: up to 10,000 tries. Draw a
   cell uniformly over the whole `ny * nx` grid; accept it when the map value there is positive
   and not NaN, and either `max_p == 0` or `value >= U * max_p`. `max_p` is 0 for
   presence/absence maps (maximum at least 1) and the map's maximum otherwise
   (`MovementMapSet`). Placement does not consult the ocean mask; only the map. Ten thousand
   failures send the school out of domain. (S; the accept rule E)
4. **Walk**, when the map is the same and the school is located: candidates are the cells of
   the `(2r + 1)^2` square around it, clipped to the grid, that are ocean **and** have a
   positive, non-NaN map value. No candidate: the school stays. Otherwise one is drawn
   uniformly. (S; the candidate set E)
5. **Salinity gate**, for gated species when the maximum of `map * weight` is positive: the
   map value is replaced by `map * weight` in the accept rule (against that maximum) and in the
   walk, where candidates also need a positive product and are drawn **in proportion to it**.
   When the maximum is not positive the species moves ungated. Python recomputes that maximum
   over the whole grid for every school; it depends only on the map and the frame, so C#
   computes it once per (map, step). Same value, tier E.

**Random movement** (`movement`, `random_walk`, `build_random_patches`):

- An unlocated school is placed on a uniformly random cell of its species' domain: the whole
  ocean, or a connected patch of `ncell` cells when that key is set. Python builds the domains
  and the bundle carries them. **A school placed this step does not also walk.**
- A located school moves to a uniformly random cell among the in-grid ocean cells of the
  `(2r + 1)^2` square around it, its own cell included. Range 0 or no candidate: it stays.

Schools of any other method do not move.

**Resources** (`ResourceState.update`), tier E. The forcing frame for a step is
`min(int(step_in_year * n_frames / ndt), n_frames - 1)`. Carrying capacity is
`forcing * multiplier * accessibility(step)`, or
`multiplier * (uniform + offset) * accessibility` for a uniform resource. The named resource's
capacity is multiplied by the oxygen factor when that coupling is on. Biomass is **reset** to
capacity every step, or regrown towards it with `logistic_regrow` when resources are
depletable. Within a step, predation depletes it across sub-steps with no reset in between.

**Background schools** (`BackgroundState.get_schools`), tier E. Every step, one school per
background species, per size class, per ocean cell: biomass is the cell's biomass times the
class proportion; abundance is biomass over the class weight (zero when the weight is not
positive); length, age, trophic level come from the class; `first_feeding_age_dt` is -1, so
they always feed. The forcing frame is
`min(int(step_in_year * n_declared / ndt), n_frames - 1)`.

**Incoming flux** (`IncomingFluxState.get_incoming_schools`). The table row is
`step % n_rows`. A class with positive biomass and weight becomes `n_schools` equal schools when
its abundance reaches `n_schools`, else one school, each on a uniformly random ocean cell (S),
not eggs.

---

## 7. Feature scope and phases

### 7.1 Inventory

"Used by" was read from the three configs with `EngineConfig.from_dict` on 2026-11-17 (EEC =
`data/eec_full`, BoB = `data/examples`, Baltic = `data/baltic`).

| Feature | Python source | Used by | Phase |
|---|---|---|---|
| Step loop, record window, cancel | `simulate.py::simulate` | all | 1 |
| Von Bertalanffy growth, predation-success gate | `processes/growth.py` | all | 1 |
| Interleaved mortality, sub-steps, type I response | `processes/mortality.py` | all | 1 |
| Stage accessibility matrix, incl. the -1 rule | `accessibility.py` | all | 1 |
| Feeding stages by size | `processes/feeding_stage.py` | EEC (2 stages for one species) | 1 |
| LTL resources from forcing, per-step accessibility | `resources.py` | all | 1 |
| Larva mortality (constant), egg retention and release | `processes/natural.py`, `mortality.py` | all | 1 |
| Starvation (lagged) | `processes/starvation.py::update_starvation_rate` | all | 1 |
| Fishing by rate: age knife-edge or no selectivity, seasonality, spatial maps, discards | `mortality.py::_precompute_effective_rates` | EEC, Baltic (BoB: flat rate) | 1 |
| Reproduction: season table, seeding, egg weight override | `processes/reproduction.py` | all | 1 |
| Movement by maps, with same-map random walk | `processes/movement.py`, `movement_maps.py` | EEC, Baltic | 1 |
| Movement "random": domain placement and walk | `movement.py::random_walk`, `build_random_patches` | BoB | 1 |
| Output cutoff age | `simulate.py::_collect_biomass_abundance` | EEC, Baltic | 1 |
| Minimal outputs: biomass, abundance, mortality by cause, yield | `simulate.py::_collect_outputs` | all | 1 |
| Trophic level tracking | `mortality.py` | all (feeds outputs and the "tl" stage metric) | 1 |
| Background species (uniform or forcing, classes) | `background.py` | Baltic (2) | 2 |
| Out-of-domain mortality, null maps | `natural.py::out_mortality` | Baltic | 2 |
| `species.lmax` cap | `growth.py` | Baltic | 2 |
| Shepherd / Beverton-Holt / Ricker / hockey-stick recruitment | `reproduction.py` | Baltic (Shepherd) | 2 |
| RV recruitment gate (series) | `processes/recruitment_gate.py` | Baltic | 2 |
| Salinity movement gate | `processes/salinity_gate.py`, `movement.py` | Baltic | 2 |
| Oxygen to benthos carrying capacity | `resources.py::update`, `oxygen_function.py` | Baltic | 2 |
| Full default outputs: life-stage split and rates, by-age and by-size, SSB, mean TL, mean size, yield in numbers, diet and predator pressure | `simulate.py::_collect_*`, `output.py` | all, without `output.minimal.enabled` | 2 |
| Depletable LTL with logistic regrowth | `resources.py::logistic_regrow` | `baltic_a2` demo | 3 |
| Incoming flux | `incoming_flux.py` | none of the three | 3 |
| MPAs; by-year fishing rate; by-step additional and larva rates; additional-mortality maps | `mortality.py`, `natural.py` | none of the three | 3 |
| Logistic selectivity (type 1) | `mortality.py` | none of the three | 3 |
| Ceiling, thermal and depensation gates; spatial RV egg survival | `reproduction.py`, `thermal_gate.py`, `depensation_gate.py`, `natural.py` | demos only | 3 |
| Type II and III functional response | `mortality.py` | none of the three | 3 |
| Species-level (non-stage) accessibility matrix; dynamic accessibility | `mortality.py`, `dynamic_accessibility.py` | none of the three | 3 |
| Gompertz growth; feeding stages by age, weight, TL | `growth.py`, `feeding_stage.py` | none of the three | 3 |
| Warm-start initial population | `initialization.py` | opt-in | 3 (built in Python, passed as initial state) |
| Spatial outputs, carbon flux | `simulate.py::_collect_spatial_outputs`, `_collect_carbon` | opt-in | 3 |
| Bioenergetics: energy budget, bioen predation cap, bioen starvation, foraging, bioen reproduction, temperature and oxygen forcing | `simulate.py::_bioen_step`, `_bioen_reproduction`, `processes/bioen_*.py`, `energy_budget.py`, `temp_function.py` | `c3_bioen` scenario | 4 |
| Genetics (Ev-OSMOSE) | `osmose/engine/genetics/` | opt-in | not planned |
| Economics (fleet dynamics) | `osmose/engine/economics/` | opt-in | not planned |
| Live map (`step_observer`) | `simulate.py`, `ui/pages/run.py` | UI | stays Python-only |

"Not planned" is a recommendation (open question Q5): genetics and economics are research
features with no performance pressure, each a subpackage with its own state. A config that
enables them is refused by the C# engine and runs on Python.

### 7.2 Phases and acceptance

**Phase 0, two spikes. Go/no-go.**

- *Bridge spike.* A 50-line assembly and a pytest prove, on this machine: pythonnet 3.2.1
  loads .NET 10 through `clr_loader`; a NumPy buffer address passed as `Int64` is read and
  written in place; the GIL is released during the call (a Python thread makes progress);
  a control block cancels a running call; a .NET exception arrives as a Python exception with
  its message; a second concurrent call from another Python thread works.
- *Kernel spike.* The cell kernel of section 4.3 for bioen-off, stage accessibility, type I,
  on a Baltic mortality step captured from Python as a bundle file, benchmarked with
  BenchmarkDotNet against Numba's kernel on the same captured step, one thread, interleaved.
- **Accept:** every bridge check passes; the kernel agrees with `_mortality_in_cell_numba` on
  injected draws (bit for bit with zero non-predation rates, and as one whole-cell comparison
  with real rates; section 6.1); kernel throughput is at least 1.0x Numba's.
- **Stop and report** if the bridge cannot release the GIL or cannot share buffers, or if the
  kernel is below 0.8x after one tuning pass. Between 0.8x and 1.0x the owner decides, since
  parallel movement and the removed orchestration may still pay.

**Phase 1, MVP: EEC full and Bay of Biscay, bioen off.**

- **Accept:**
  1. Every process test of section 10.1 passes at its tier.
  2. Determinism: output bit-identical at 1, 2 and 14 threads and across repeated runs, on both
     configs.
  3. Death accounting holds for every school and step to 1e-9 relative.
  4. Ensemble equivalence with the Python engine on EEC (10 years) and Bay of Biscay (5 years),
     by the rule in section 10.2.
  5. `CSharpEngine.run_in_memory` returns an `OsmoseResults` whose `biomass()`, `abundance()`,
     `yield_biomass()` and `mortality()` frames have the Python engine's shape, columns and
     dtypes.
  6. Any config outside the phase is refused with a message naming the feature. **That
     includes the stock demo configs as shipped**: none of the three sets
     `output.minimal.enabled`, and all three enable the diet output (read with
     `EngineConfig.from_dict`). In Phase 1 the engine runs only with
     `output.minimal.enabled = true`, which switches every optional output off
     (`config.py`, the `minimal` block of the output-flag loader); without it the refusal names
     the full output families as a Phase 2 feature. The wrapper never sets the key on the
     caller's behalf. Phase 1 is therefore an engine for calibration, scripts and tests, which
     already run minimal; **the UI gets the third engine in Phase 2**, once the Results pages
     have something to show.
  7. The performance gates of section 10.4 for EEC.

**Phase 2, Baltic.**

- **Accept:** items 1 to 4 and 6 above on `data/baltic` (15 years, the config's own length,
  16 replicates, the first 5 years dropped); the full default output families agree with Python's
  frames in shape and, statistically, in value; `scripts/baltic_stability_certify.py` run with
  the C# engine gives the same per-species verdicts as with Python on the same seeds, or every
  difference is explained; the Baltic performance gates of section 10.4.

**Phase 3, optional features, one PR each.** Each is accepted on its own process test plus an
A/B: with the feature's flag off the output is bit-identical to before the PR; with it on, a
small ensemble matches Python. The demo variants in `osmose/demo.py` are the fixtures.

**Phase 4, bioenergetics.** Its own design note first. The Python bioen path took a
sub-plan and several corrections to reach Java parity (CLAUDE.md, the bioen notes); the port
should be planned against that record, not estimated here.

---

## 8. The Python to C# contract

### 8.1 Principle: Python resolves, C# computes

`EngineConfig` is not the whole input. `simulate()` also builds `ResourceState`,
`BackgroundState`, `IncomingFluxState`, one `MovementMapSet` per species, the oxygen, salinity
and RV fields, the stage-accessibility lookups and the feeding-stage threshold lists, and each
of those reads `raw_config`, opens a file, or resolves a name.

**Decision D8.** A Python **bundle builder** runs all of that and hands C# only dense, typed,
already-indexed arrays:

- every name is resolved to an index (with -1 preserved where Python yields -1);
- every file is loaded and regridded to the model grid;
- every time series is expanded to one value per simulation step wherever that is small
  (`n_steps x n_species`), so C# implements no cycling, by-year or by-season lookup;
- where a per-step table would be large (forcing frames), the bundle carries the frames plus a
  resolved `frame_by_step` index array, so the frame arithmetic stays in one place, Python;
- every quantity whose formula involves `pow` and is constant for the run (egg weight,
  background class weights) is computed in Python, so both engines use the same value.

C# therefore contains model arithmetic and nothing else. A change to a config key, an alias or
a file format never touches the C# code.

**Alternative rejected: pass the raw config dictionary and let C# parse.** It would duplicate
`osmose/engine/config.py` (2,874 lines), the reader's separator and lowercasing rules and the
config-validation allowlist in a second language. That is the largest drift risk available.

### 8.2 The bundle

A bundle is a set of named entries. An entry is a scalar (`int64` or `float64`) or an array
with a dtype (`f8`, `i4`, `i8`, `u1`), a shape of up to four dimensions and an address. Arrays
are C-contiguous and little-endian. Names are lowercase and dot-separated, as config keys are.

Groups and their main entries (`n_total = n_focal + n_background`):

| Group | Entries (shape) |
|---|---|
| `contract` | `version`; `features` (bitmask of what this bundle needs); `rng.scheme` |
| `run` | `seed`; `threads`; `deadline_seconds`; `checks` (debug invariants on or off) |
| `time` | `n_steps`; `n_dt_per_year`; `n_subdt` |
| `grid` | `ny`; `nx`; `ocean_mask` u1 (ny, nx) |
| `species` | `n_focal`; `n_background`; `n_schools` i4 (n_focal); `lifespan_dt` i4 (n_total) |
| `growth` | `class` u1; `linf`, `k`, `t0`, `egg_size`, `vb_threshold_age`, `condition_factor`, `allometric_power`, `lmax` (+inf when unset), `delta_lmax_factor`, `critical_success_rate`, each f8 (n_total) |
| `predation` | `ingestion_rate`, `fr_halfsat` f8 (n_total); `fr_shape` i4; `size_ratio_min`, `size_ratio_max` f8 (n_total, n_stage_max); `stage.metric` u1; `stage.offsets` i4 (n_total + 1); `stage.thresholds` f8, ascending |
| `access` | `mode` (none, species, stage); `matrix` f8 (n_prey_rows, n_pred_cols), as Python holds it; for each of `prey` and `pred`: `offsets` i4 (n_total + 1), `thresholds` f8 (years), `index` i4. **An empty range for a species means "unresolved" and yields -1.** `resource_rows` i4 (n_rsc), -1 when unresolved |
| `mortality` | `additional.rate_by_step` f8 (n_steps, n_total); `additional.map_index` i4 (n_total, -1 for none) and `additional.maps` f8 (n_maps, ny, nx); `larva.rate_by_step` f8 (n_steps, n_focal), the resolved per-step value; `out.rate` f8 (n_total); `starvation.rate_max` f8 (n_total) |
| `fishing` | `enabled`; `rate_by_step` f8 (n_steps, n_total), by-year overrides applied; `season_by_step` f8 (n_steps, n_total), **absent** when there is no seasonality (the two branches of section 6.3 differ); `selectivity.type` i4, `l50`, `a50`, `slope`; `map_index`, `maps`; `discard_rate` |
| `reproduction` | `sex_ratio`, `relative_fecundity`, `maturity_size`, `seeding_biomass`, `egg_length`, `egg_weight` f8 (n_focal); `maturity_age_dt`, `seeding_max_step` i4; `season_by_step` f8 (n_steps, n_focal); `sr.type` u1, `sr.ssb_half`, `sr.shepherd_beta`; `seeding_mode`; `gate.rv_by_step`, `gate.thermal_by_step` f8 (n_steps, n_focal), 1.0 where off; `gate.ceiling_by_step`, +inf where off; `gate.depensation.enabled`, `.s50`, `.theta` |
| `movement` | `method` u1 (n_total: none, random, maps); `walk_range` i4; `maps.stack` f8 (n_maps, ny, nx), NaN kept; `maps.max_proba`; `maps.is_null` u1; `maps.species_offset` i4; `maps.index` i4, the per-species `(lifespan_dt, n_steps)` tables concatenated, with `maps.index.offsets` i8 and `maps.index.shape` i4; `random.domain.offsets`, `random.domain.cells` i4; `salinity.weight` f8 (n_frames, ny, nx), `salinity.frame_by_step` i4, `salinity.species` u1 |
| `resources` | `n`; `size_min`, `size_max`, `trophic_level` f8; `forcing` f8 (n_rsc, n_frames, n_cells), regridded and multiplied by the resource multiplier; `frame_by_step` i4 (n_steps, n_rsc); `accessibility_by_step` f8 (n_steps, n_rsc); `k_factor` f8 (n_frames, n_cells) with `k_factor.resource` and `k_factor.frame_by_step` for the oxygen coupling, the Hill factor already evaluated; `depletable`, `depletable.floor`, `regrowth_rate` |
| `background` | `class.offsets` i4; `class.length`, `class.weight`, `class.trophic_level` f8; `class.age_dt` i4; `proportion_by_step` f8 (n_steps, n_classes); `ocean_cells` i4; `biomass` f8 (n_bkg, n_frames, n_ocean) with `frame_by_step` |
| `state` | optional initial schools (species, abundance, length, weight, age, cell), built by Python's `build_initial_population`; empty by default |
| `output` | `cutoff_age` f8 (n_total), 0 when unset; `record_frequency`; `step0_include`; `families` (bitmask) |
| `result` | Python-allocated buffers, section 8.4 |
| `control` | address of the shared control block, section 8.5 |

Pre-multiplying the resource forcing by its multiplier in Python is exact:
`ResourceState.update` computes `forcing * multiplier * access` left to right, so
`(forcing * multiplier) * access` is the same expression.

**Versioning.** `contract.version` is an integer. The C# engine accepts exactly the versions it
lists; anything else throws before any work. Adding an optional entry does not bump it; changing
the meaning or shape of an existing entry does. `contract.features` names what the bundle
needs; the engine compares it with what it implements and refuses on any missing bit (D11).

**Validation.** `SimulationConfig` is turned into an immutable, strongly typed
`EngineParameters` once, at the start of `RunSimulation`. That step checks presence, dtype,
shape consistency (every `(n_steps, ...)` table against `time.n_steps`, every per-species array
against `n_total`), index ranges and finiteness, and reports **all** problems in one exception.
After it, no code looks an entry up by name.

### 8.3 The entry point

```csharp
namespace Osmose.Engine;

public static class OsmoseEngine
{
    /// Runs the whole simulation. Blocking. Thread-safe and re-entrant:
    /// no static mutable state, one SimulationRun object per call.
    public static RunSummary RunSimulation(SimulationConfig config);
}

public sealed class SimulationConfig
{
    public SimulationConfig(int contractVersion);
    public void SetInt(string name, long value);
    public void SetFloat(string name, double value);
    // Borrowed memory: the caller keeps the buffer alive and unmoved for the whole call.
    public void SetArray(string name, DType dtype, long address,
                         long d0, long d1 = -1, long d2 = -1, long d3 = -1);
    public static SimulationConfig Load(string path);   // owned memory, from a bundle file
    public void Save(string path);
}

public sealed class RunSummary
{
    public RunStatus Status { get; }           // Completed, Cancelled, TimedOut
    public int StepsCompleted { get; }
    public int RecordsWritten { get; }
    public int PeakSchools { get; }
    public double ElapsedSeconds { get; }
    public double[] PhaseSeconds { get; }      // movement, regroup, mortality, growth, ...
    public string[] Warnings { get; }          // e.g. species with unresolved accessibility
    public int RngScheme { get; }
}
```

`RunSimulation` is the only method that does work. `SimulationConfig` is data. There is no
second entry point for capabilities: the supported contract versions and feature bits are
written at build time to `Osmose.Engine.capabilities.json` beside the assembly, and Python
reads that file.

**Addresses are `Int64`.** pythonnet converts a Python `int` to `System.Int64` through an
explicit case in its converter; whether it converts to `IntPtr` implicitly was not established
(a fetched summary of `Converter.cs` suggests not). `long` is safe either way and the engine
casts to `nint` itself.

**Bundle files.** `Save` and `Load` use one flat format: a header (magic, contract version,
entry count), a table of contents (name, dtype, shape, offset) and 64-byte-aligned raw
little-endian data. Python writes it with `ndarray.tofile`; C# memory-maps it and builds the
same `SimulationConfig` over the mapped pointers. This is what lets `dotnet test`, the
benchmarks and a bug report run with no Python process, and it is how a failing production run
is captured for replay.

**Alternative rejected: return .NET arrays and convert them in Python.** Turning a .NET
`double[]` into a NumPy array through pythonnet is slow without extra codec work (not measured
here), and the data would be copied at least once. Python-allocated buffers avoid both.

### 8.4 Results

Python allocates the result arrays with NumPy, registers their addresses in the bundle and
keeps them; C# writes into them. Nothing is converted or copied on the way back.

The record count is known before the run:
`n_records = (1 if step0_include else 0) + ceil(n_steps / record_frequency)`.

| Entry | dtype, shape | Content |
|---|---|---|
| `result.record_step` | i4 (n_records) | last step of the window; -1 for the initial record |
| `result.window_steps` | i4 (n_records) | steps folded into the record |
| `result.biomass` | f8 (n_records, n_total) | window mean, tonnes |
| `result.abundance` | f8 (n_records, n_total) | window mean |
| `result.mortality` | f8 (n_records, n_focal, 8) | window sum of deaths, causes in `MortalityCause` order |
| `result.yield` | f8 (n_records, n_focal) | window sum, tonnes |

These four value arrays are exactly what `output.minimal.enabled` keeps
(`_collect_outputs`, the `config.output_minimal` branch: biomass, abundance, mortality by cause
and yield). The Python wrapper builds one `StepOutput` per record from array rows (views, no
loop over elements) and hands the list to `OsmoseResults.from_outputs` or to `write_outputs`,
both unchanged.

A cancelled or timed-out run leaves the first `RecordsWritten` records valid.

**Richer outputs later.** Each output family is one more optional `result.*` buffer whose
shape Python can compute before the run, requested through `output.families`:

- life-stage split and rates: `(n_records, n_focal, 3)` and `(n_records, n_focal, 8, 3)`;
- by-age and by-size distributions: `(n_records, n_focal, n_bins_max)`, NaN-padded, with the
  bin edges computed by Python;
- SSB, mean size, mean TL, yield in numbers: `(n_records, n_focal)`;
- diet: `(n_records, n_predators, n_prey)`;
- spatial: `(n_records, n_focal, ny, nx)`, opt-in because of its size.

A family the engine does not implement is a missing feature bit and is refused. The Python
wrapper maps each buffer to the `StepOutput` field it fills; `StepOutput` does not change.

### 8.5 Cancellation, progress, deadline, errors

**Decision D10: a shared control block.** `control` is a NumPy `int64[8]` owned by Python:

| Slot | Writer | Meaning |
|---|---|---|
| 0 | Python | cancel requested (non-zero) |
| 1 | C# | steps completed |
| 2 | C# | records written |
| 3 | C# | live school count |
| 4-7 | - | reserved |

C# reads slot 0 with `Volatile.Read` once per step, the same granularity as
`simulate()`'s `cancel_token.is_set()` check, and writes the others once per step. The Python
wrapper bridges the UI's `threading.Event` to slot 0 from a small watcher thread and exposes
slot 1 for progress. No call crosses the boundary during the run, so no GIL is taken and no
delegate is marshalled.

**Signals.** Python runs a signal handler only between bytecodes on the main thread, so a
`KeyboardInterrupt` or the `SIGALRM` that `scripts/calibrate_baltic.py::_run_with_timeout`
raises as `_SimTimeout` would not be delivered while the main thread sits inside a C# call.
The wrapper therefore runs `RunSimulation` on a worker thread whenever it is called from the
main thread, and the main thread waits in a `join(timeout)` loop, where signals are delivered
normally. `_run_with_timeout` then works unchanged.

**Buffer lifetime: the invariant.** The bundle, and every NumPy buffer it points to, is owned
by the wrapper **until `RunSimulation` has returned**. If any `BaseException` reaches the
waiting thread (`KeyboardInterrupt`, `_SimTimeout`, anything), the wrapper sets the cancel
slot, **joins the worker thread**, and only then re-raises. It never lets the exception
propagate while C# may still be writing into borrowed memory. The engine checks the cancel slot
once per step, so the join is bounded by one step.

**Deadline.** `run.deadline_seconds` makes the engine stop with `TimedOut` on its own. It
serves callers that are not on the main thread, where `_run_with_timeout` is a no-op by its
own docstring.

**The GIL.** pythonnet's method binder releases the GIL around a .NET call by default
(`MethodBinder.allow_threads = DefaultAllowThreads = true`, with
`PythonEngine.BeginAllowThreads()` before the invoke; read from the pythonnet source through a
fetched summary). The UI depends on it: the engine runs on a background thread and the Shiny
event loop must keep polling. It is the second check of the bridge spike.

**Errors.**

| Situation | C# | Python wrapper |
|---|---|---|
| Wrong contract version, missing or malformed entries | `BundleValidationException`, all problems listed | `CSharpEngineError` |
| Feature bit not implemented | `UnsupportedFeatureException`, features named | `CSharpEngineError`; normally caught earlier by `csharp_engine_block_reason` |
| Failure inside a parallel loop | the first inner exception of the `AggregateException`, rethrown with step and cell | `CSharpEngineError` |
| Cancelled | `Status = Cancelled`, no exception | raises the existing `SimulationCancelled` |
| Timed out | `Status = TimedOut` | raises `CSharpEngineTimeout` |
| Debug invariant broken (`run.checks`) | `InvariantViolationException` with school id, step, cause | `CSharpEngineError` |

**Progress callbacks: deferred.** A .NET-to-Python delegate per step is feasible (pythonnet
wraps a callable as a delegate and takes the GIL when it is invoked), but it adds a
re-entrancy path for one integer that the control block already carries. The live map needs
the whole school table and stays on the Python engine (section 9.2).

---

## 9. Integration as a third backend

### 9.1 Where things live

```
csharp/
  Osmose.Engine.sln
  src/Osmose.Engine/            net10.0 (;net8.0), unsafe blocks on, nullable on, warnings as errors,
                                InvariantGlobalization=true
  tests/Osmose.Engine.Tests/    xUnit; reads bundle-file fixtures
  bench/Osmose.Engine.Bench/    BenchmarkDotNet
osmose/engine_cs/
  __init__.py       CSharpEngine, CSharpEngineError, csharp_engine_available()
  bundle.py         build_bundle(): pure Python and NumPy, no CLR
  capabilities.py   csharp_engine_block_reason(): reads capabilities.json, no CLR
  bridge.py         the only module that imports pythonnet/clr
  bin/              build output (gitignored): Osmose.Engine.dll, .runtimeconfig.json,
                    .deps.json, Osmose.Engine.capabilities.json
scripts/build_csharp_engine.sh  dotnet publish -c Release -f net10.0 -o osmose/engine_cs/bin
```

`csharp/` is a top-level directory. `pyproject.toml` already restricts package discovery to
`osmose*` (CLAUDE.md, the setuptools gotcha), so it does not disturb the Python build.

The publish is framework-dependent and portable: one IL assembly that runs on any OS and
architecture that has the runtime.

### 9.2 `CSharpEngine`

Same two methods and signatures as `PythonEngine`:

```python
class CSharpEngine:
    def run(self, config, output_dir, seed=0, *, cancel_token=None, step_observer=None) -> RunResult
    def run_in_memory(self, config, seed=0, *, cancel_token=None) -> OsmoseResults
```

One run:

1. `warn_unread_java_only_keys(config)`, `EngineConfig.from_dict(config)` and the grid
   resolution, exactly as `PythonEngine._prepare_run` does. `_resolve_grid` moves to a
   module-level function so both engines share it.
2. `csharp_engine_block_reason(config, engine_config)`; raise `CSharpEngineError` with the
   reason if there is one.
3. `build_bundle(engine_config, grid, seed, threads, deadline)`: resolve everything
   (section 8.1), allocate the result buffers and the control block.
4. `bridge.run(bundle)`: load the runtime if needed, fill a `SimulationConfig`, call
   `OsmoseEngine.RunSimulation`, with the cancel watcher running.
5. Map the summary: raise `SimulationCancelled` or `CSharpEngineTimeout`, log warnings through
   the `osmose` logger (so they reach the UI run console like the Python engine's).
6. Build the `StepOutput` list from the result buffers; return
   `OsmoseResults.from_outputs(outputs, engine_config, grid)`, or call `write_outputs` and
   return a `RunResult`.

`step_observer` is rejected with a clear error: the observer receives the Python
`SchoolState`, which does not exist here. **The live map stays a Python-engine feature.**

`CSharpEngine` holds no state; concurrent calls from several Python threads are allowed
(the entry point is re-entrant), which also makes the calibration **thread** backend truly
parallel for this engine, since the GIL is released. That is a possible later gain, not an MVP
claim.

### 9.3 Selection points

| Where | Today | Change |
|---|---|---|
| A registry | none; callers import `PythonEngine` | `osmose/engine/select.py::get_engine(name)` for the in-process engines, `"python"` and `"csharp"`. Java stays behind `osmose/runner.py`. |
| UI state | `AppState.engine_mode`, `"python"` by default (`ui/state.py`) | a third value, `"csharp"`, offered only when `csharp_engine_available()`. **Phase 2**, after the full output families (section 7.2) |
| UI run page | `ui/pages/run.py` tests `engine_mode != "python"` to mean "Java" (the two `panel_conditional`s, `engine_indicator`'s label, the Java block-reason branch of `handle_run`) or "no live view" (`live_movement_status`); `genetics.py`, `economic.py` and `diagnostics.py` test it to mean "feature unavailable" | **Every such test is rewritten** as an explicit predicate: `is_java(mode)`, `is_in_process(mode)`, `engine_supports(mode, feature)`. With three values, "not Python" no longer means Java. The C# branch reuses `_python_engine_thread`, generalised to take an engine, with the live pane disabled. |
| Block reason | `java_engine_block_reason` | `csharp_engine_block_reason`, same shape, shown the same way |
| Calibration | `OsmoseCalibrationProblem(use_java_engine: bool)` (`osmose/calibration/problem.py`) | `engine: Literal["python", "java", "csharp"] = "python"`; `use_java_engine=True` kept as a deprecated alias. `_run_python_engine` takes the engine from the registry. `_python_engine_errors` gains `CSharpEngineError`. **`_cache_key` gains the engine name**, so cached results of different engines are never mixed. |
| Scripts | `scripts/calibrate_baltic.py::run_simulation` builds a `PythonEngine` | an `engine` argument and a `--engine` flag there, in `scripts/cross_engine_parity_440.py` (`--engines python,csharp,4.4.1`), `scripts/benchmark_engine.py` and `scripts/profile_engine.py` |
| CLI | `osmose run` | `--engine` |

About sixty other scripts import `PythonEngine` directly; they stay as they are.

**There is no environment variable or global default that switches the engine.** Every number
in `docs/` was produced by a named engine; a silent switch would break that provenance. The
engine is always chosen explicitly and recorded in the run's log.

### 9.4 Loading the runtime

`bridge.load()` runs once per process, at the first C# run, never at import:

1. Find the assembly: `OSMOSE_CSHARP_ENGINE_DIR`, else `osmose/engine_cs/bin`.
2. Find the runtime: `DOTNET_ROOT`, else `~/.dotnet`, else `dotnet` on `PATH`.
3. `pythonnet.load("coreclr", runtime_config=<bin>/Osmose.Engine.runtimeconfig.json)`, then
   `clr.AddReference(<bin>/Osmose.Engine.dll)`.
4. Compare the assembly's contract versions with the bundle builder's; mismatch is an error
   that names both versions and the build command.

Rules:

- **Importing `osmose.engine_cs` must not load the CLR** (tested by checking `sys.modules`).
  The capability check and the bundle builder work without it.
- **Never load the CLR in a process that later forks.** CoreCLR's runtime threads do not
  survive `fork`. The calibration and UQ pools already use forkserver and spawn (plan
  guardrail 5), so each worker loads its own runtime. This is the same family of trap as the
  Numba-threads-in-a-forking-parent note.
- pythonnet allows one runtime per process. If something else already loaded Mono, `load`
  fails; the error says so.
- `csharp_engine_available()` returns `(False, reason)` instead of raising when pythonnet, the
  runtime or the assembly is missing, so the UI can hide the option and say why.

**Threads.** The engine uses exactly `run.threads` workers. Python resolves the number through
the existing policy: `thread_policy.resolve_engine_threads` for a single run (physical cores)
and `worker_thread_count(n_workers)` inside a pool. The .NET and Numba thread pools are
independent; running the C# engine does not touch Numba's settings.

### 9.5 Dependencies

- `pyproject.toml`: a new extra `csharp = ["pythonnet>=3.2.1"]`, also added to `dev` so CI and
  pyright see the package. pythonnet is imported only in `bridge.py`; .NET namespaces are
  reached through `importlib` and typed as `Any`, so pyright has nothing unresolved to report.
- No new Python dependency for users who do not select the engine.

### 9.6 Developer setup without sudo

1. Download Microsoft's `dotnet-install.sh` and run it with
   `--channel 10.0 --install-dir "$HOME/.dotnet"` (the SDK includes the runtime).
2. Export `DOTNET_ROOT="$HOME/.dotnet"` and add it to `PATH`.
3. `.venv/bin/pip install -e ".[dev]"`.
4. `bash scripts/build_csharp_engine.sh`.
5. `.venv/bin/python -m pytest tests/test_csharp_engine*.py`.

### 9.7 Deployment: three surfaces

| Surface | Change |
|---|---|
| **systemd** (`deploy.sh`, the prod clone, `/opt/micromamba/envs/shiny`, Python 3.13) | Install the .NET SDK into `/opt/dotnet` with `dotnet-install.sh` if absent; build the assembly **inside the prod clone** from the deployed ref; `pip install pythonnet` into the shiny env; add `Environment=DOTNET_ROOT=/opt/dotnet` to the unit; a post-deploy self-test that runs a one-step bundle with the service's interpreter and fails the deploy loudly if it does not complete. |
| **Docker** (`Dockerfile`, `python:3.12-slim`, already copies a JRE from a Java stage) | A build stage from the .NET SDK image publishes the assembly; the runtime is copied from the .NET runtime image the same way `JAVA_HOME` is today. `InvariantGlobalization` removes the need for ICU in the slim image. |
| **Apptainer** (`apptainer/osmose.def`) | Install the SDK in `%post`, build, keep the runtime; add a self-test line to the CI smoke step beside the existing `java -version` and Numba-cache checks. |

The C# engine is optional on every surface: without the runtime the app runs exactly as today
and the UI does not offer the third engine.

**Building on the host versus shipping an artifact.** Building in the clone guarantees the
assembly matches the deployed source, at the price of an SDK on the server. The alternative is
a CI-built assembly attached to the GitHub release, fetched by tag and checked by SHA-256,
needing only the runtime. Recommended: build on the host (open question Q6).

### 9.8 CI

A new job, `csharp-engine`, beside the existing ones in `.github/workflows/ci.yml`:

1. `actions/setup-python` (3.12) and `actions/setup-dotnet` (`10.0.x`).
2. `pip install -e ".[dev]"`.
3. `dotnet format --verify-no-changes`, `dotnet build -c Release` with warnings as errors.
4. Generate the process fixtures **from the Python engine** (section 10.1).
5. `dotnet test` in the `CHECKED` configuration.
6. `bash scripts/build_csharp_engine.sh`.
7. `OSMOSE_REQUIRE_CSHARP=1 pytest tests/test_csharp_engine*.py`.

`OSMOSE_REQUIRE_CSHARP=1` turns "engine unavailable" from a skip into a failure. Without it the
tests skip, with a reason that says when to opt in (the env-var pattern CLAUDE.md's
`@pytest.mark.slow` note prescribes). A skip is a green, so the one job that is supposed to run
these tests must not be allowed to skip them.

Coverage: `bundle.py` and `capabilities.py` are pure Python and are tested in the main `test`
job, inside the 90% floor. `bridge.py` is omitted from the main job's coverage and measured in
the `csharp-engine` job.

The ensemble comparison is too slow and too noisy for a per-PR gate. It is a scheduled and
manually triggered job with a reduced design, and a required step before each phase is
declared done.

---

## 10. Validation plan

### 10.1 Deterministic process tests

Fixtures are **generated by the Python engine**, written as bundle files, and consumed by
`dotnet test`. They are regenerated in CI on every run, so a change to Python's semantics turns
the C# tests red instead of drifting silently. Source configs: the tutorial config
(`tests/_tutorial_config.build_config`), EEC and Baltic. **Never `data/minimal`**, which has no
predation event at all (CLAUDE.md).

| C# unit | Python oracle | Tier |
|---|---|---|
| Philox block | Random123 known-answer vectors; `numpy.random.Philox` | E |
| Bounded integers, permutations, cause-order decoding | chi-square uniformity over all outcomes for small `n`; every cause order reachable | S |
| Per-school rates | `_precompute_effective_rates` on captured EEC and Baltic states | E (L for logistic selectivity) |
| Feeding stages; stage accessibility indices, including unresolved species | `compute_feeding_stages`; `AccessibilityMatrix.compute_school_indices` | E |
| One predator visit | `_apply_predation_numba` on captured cells | E |
| **Cell kernel, all sub-steps, injected draws, zero non-predation rates** | `_mortality_in_cell_numba` driven sub-step by sub-step with the same permutations and cause orders | E |
| **Cell kernel, all sub-steps, injected draws, real rates** | the same | L as a whole (`atol=0` if `Math.Exp` equals Numba's `exp`; section 6.1) |
| Larva pre-pass, egg retention and release | `larva_mortality`; the release expression in `mortality()` | L, E |
| Commit, out mortality, next starvation rate, trophic level | the tail of `mortality()`, `out_mortality`, `update_starvation_rate` | E, L |
| Floor, aging | `_floor_sub_one_fish`, `aging_mortality` | E |
| Growth | `growth` | L |
| Reproduction: SSB, seeding, curve, gates, egg schools | `reproduction` (it draws nothing) | L; school counts E |
| Resource update, oxygen factor, depletable regrowth | `ResourceState.update` | E |
| Background schools | `BackgroundState.get_schools` | E |
| Movement, the deterministic part: map index, same-map flag, null map to out-of-domain, the accessible set, the gated maximum | `_precompute_map_indices` and an enumeration in the fixture generator | E |
| Movement, the random part | chi-square of placement frequencies against the map over many draws | S |
| Output collectors and window averaging | `_collect_outputs`, `_average_step_outputs` on captured states | L (summation order) |

The injected-draw seam is internal (`InternalsVisibleTo` the test assembly, the `TDraws` type
parameter of section 4.3). It does not add a public entry point.

**Whole-engine invariants, tested in C# and through pytest:**

- *Death accounting.* For every school and step, abundance removed equals the sum of its death
  counters plus what the floor zeroed. This is the ledger the performance plan says does not
  exist yet (its section 3); here it is built in from the start and switchable with
  `run.checks`.
- *Thread invariance.* Result buffers are byte-identical at 1, 2 and 14 threads.
- *Repeatability.* Two runs with one seed are byte-identical; two seeds differ.
- *No allocation* in steady state (section 4.5).
- *Refusal.* Each unsupported feature, switched on in an otherwise valid config, is refused
  with a message naming it.

**Every test is mutation-checked once, by hand, before it is trusted.** At least these five
mutations must each turn a named test red: `>=` to `>` on the upper window bound; an
unresolved accessibility index treated as zero access; seasonal fishing divided by `denom`; egg
release computed from the instantaneous abundance; the cutoff applied to yield. A test that
survives its mutation is rewritten.

### 10.2 Ensemble equivalence against the Python engine

**Harness.** `scripts/cross_engine_parity_440.py` gains a `csharp` arm and keeps its `tost()`.

**Why not reuse the Java margins.** The Java cross-check uses 3x for biomass, yield and
abundance and 1.5x for mean weight and size (`docs/java_cross_check_2026-11-11.md`). Those
margins allow for two different implementations of the model. Here the algorithm is the same
and only the random stream differs, so the margins are tighter.

**Design, fixed in advance.**

| | EEC full | Bay of Biscay | Baltic |
|---|---|---|---|
| Years (spin-up dropped) | 10 (2) | 5 (1) | 15 (5) |
| Replicates per arm | 32 | 32 | 16 |
| Arms | Python seeds A, Python seeds B, C# seeds C | same | same |
| Metrics | biomass, abundance, yield, mean weight (biomass / abundance); mean size from Phase 2 | same | same |
| Margin | 1.25x (0.097 in log10) for biomass, abundance, yield; 1.05x (0.021) for mean weight and size | same | same |

Bay of Biscay is kept to five years because it is a dying ecosystem beyond year ten
(CLAUDE.md).

**The control arm is what makes the verdict readable.** In the Java check, sardine's 90%
interval was +/-0.42 in log10 at 16 replicates, wider than any reasonable margin: a per-replicate
spread of about 0.7 in log10 (inferred from that interval). No feasible replicate count
resolves such a species at 1.25x (it would take about 290 per arm). So Python is also compared
with **itself** on disjoint seeds, through the same harness. That arm shows what "identical
engines" look like on this config at this sample size.

At 32 replicates, a cell whose per-replicate spread is up to about 0.23 in log10 is resolvable
at 1.25x (inferred from the t interval). Noisier cells are judged against the control.

**Acceptance.** All of:

1. No **divergent** cell: none where the 90% interval excludes zero **and** `|d|` exceeds the
   margin.
2. The count of equivalent cells for Python-versus-C# is at least the count for
   Python-versus-Python minus one.
3. The largest `|d|` for Python-versus-C# is at most 1.5 times the largest for
   Python-versus-Python, or below the margin.
4. The variance ratio of each cell is within a factor of two, except where the control arm
   also fails it. Kolmogorov-Smirnov p-values are reported, not gated.
5. Cells with zero yield in both engines (unfished species) are listed as vacuous, as the Java
   note does, and not counted.

**The rule is calibrated before any C# data are looked at.** Criteria 1 and 3 can fail by
chance on a noisy cell. So the Python replicates of arms A and B are pooled and re-split at
random a few thousand times, and the rule is applied to each split. That gives the rule's
false-fail rate for identical engines on this config, with no new simulation. If it exceeds 5%,
the constants (the "minus one", the 1.5, the factor of two) are loosened until it does not, and
frozen. Only then is the C# arm compared.

A failure is investigated at the process level first: an ensemble shows **where** two engines
differ, not why.

**What this cannot show.** A bias smaller than the control arm's own noise. That is why the
process tests of section 10.1 carry the weight for correctness, and the ensemble is the check
that the processes were assembled in the right order.

**Java is not re-compared.** Agreement of C# with Python does not transfer Python's agreement
with Java automatically, but nothing in the C# engine is derived from Java. A three-way EEC
table can be produced from the same harness for the record.

### 10.3 Seeds

- Consecutive seeds give different keys and unrelated streams: the first million words for
  seeds `s` and `s + 1` share no value at the same counter, and their cross-correlation is
  within sampling error.
- Between-replicate variance of final biomass for seeds `0..15` is compared with sixteen seeds
  spaced far apart, on EEC. This is the method the performance plan proposed for the Python
  engine's adjacent-seed question.

### 10.4 Performance

**Protocol.** The performance plan's A/B protocol: control and treatment interleaved, warm,
medians of at least seven, every pair agreeing in sign, **one engine job at a time on an
otherwise idle machine**. Python is measured fresh on the day with a warm Numba cache; the
figures in `docs/perf/` are not used as the baseline. C# is measured after one warm-up run in
the same process, and the cold first call is reported separately. Both use
`output.minimal.enabled`.

**Targets.** Ratio = Python wall time / C# wall time. All inferred; Phase 0 replaces them
with measurements.

| Config | Threads | Gate (must) | Target | Where it would come from |
|---|---|---|---|---|
| Baltic, 10 yr | 1 | 1.15x | 1.4x | Mortality is 78% and the rest 22% (measured). A kernel that only **matches** Numba, with the rest cut to a third, gives 1 / (0.78 + 0.073) = 1.17x. The target needs the kernel 1.25x faster as well, from locality, the mask prefilter and fewer draws: 1 / (0.624 + 0.073) = 1.43x. |
| Baltic, 10 yr | 14 | 1.5x | 2.0x | Mortality 51%, movement 25% serial, the rest 24% serial (measured, under load, before #279). Movement parallel at about 8x and the rest cut to a third: 1 / (0.51 + 0.03 + 0.08) = 1.6x. The target also needs better mortality scaling from cost-ordered scheduling and one join per step. |
| EEC full, 10 yr | 1 and 14 | 1.0x | 1.2x | Small cells: little for the kernel to gain. The gate is "no regression". |
| Calibration, 14 workers x 1 thread, Baltic | - | 1.15x evaluations per hour | 1.4x | Follows the one-thread row, up to the memory-bandwidth ceiling the April data showed. |

**Budgets.**

- First call in a process (runtime load and JIT): at most 2 s.
- Bundle build plus marshalling: at most 5% of a 10-yr EEC run.
- Peak memory per run: no more than the Python engine's (about 400 MB for a 50-yr Baltic run,
  plan item 1).

**What the gates mean.** Phase 1 can only measure the EEC row; the Baltic rows are measured at
the end of Phase 2. If the EEC gate fails, or the Phase 0 kernel number is below 1.0x, the
engine has no performance case, and the work stops before Phase 2 unless the owner has another
reason to want it (open question Q2).

**Not claimed.** A 2-5x single-run speed-up. The performance plan found no support for that
figure in this repository's history, and nothing here changes that.

---

## 11. Risks and open questions

### 11.1 Risks

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | The pythonnet bridge fails on .NET 10, cannot share buffers, or holds the GIL | unknown, not testable here | Phase 0 bridge spike. Fallback: NativeAOT library with a C ABI through `ctypes`; the bundle contract is unchanged. |
| R2 | No speed-up: RyuJIT does not match Numba's LLVM code on the kernel | real | Phase 0 kernel spike with a stop rule. Targets rest on structure, not code generation. |
| R3 | **Two implementations of one model.** Every semantic change must land twice, and the C# side drifts | certain, permanent | Python resolves all inputs (one place for config semantics). Fixtures regenerated from Python in CI make drift loud. Fail-closed capability bits. A line in CLAUDE.md: an engine-semantics PR either updates the C# engine or raises its block reason. |
| R4 | Statistical tests miss a small bias in a noisy species | real | Exact process tests carry correctness; control arm; mutation checks. |
| R5 | A second runtime to operate on three surfaces, with its own end-of-life cadence (.NET 10 ends November 2028) | certain, moderate | Optional everywhere; post-deploy self-test; multi-target build. |
| R6 | In-process CLR hazards: signals not delivered during a call, `fork`, no unloading, one runtime per process | known | Deadline and control block; worker-thread wrapper for Ctrl-C; lazy load; forkserver and spawn only. |
| R7 | `Math.Exp` and `Math.Pow` call the platform C library, so results can differ in the last bit between operating systems or libc versions | real across machines, none on one machine | Goal G2 is "same machine image". The run summary carries a fingerprint (a hash of `exp` and `pow` over fixed arguments) so a cross-machine difference is attributable. Option: bundled managed `exp`/`pow` (Q7). |
| R8 | UI code that reads "not Python" as "Java" misroutes the third engine | certain if not audited | Section 9.3: explicit predicates, with a test per page. |
| R9 | The regrouped layout is not faster (September's negative on EEC) | possible | Measured in the kernel spike; the index-based layout is a contained fallback. |

Goal G2 in section 1.2 should be read with R7: bit-identical for a given bundle, seed and
machine image, at any thread count.

### 11.2 Open questions for the owner

| # | Question | Recommendation |
|---|---|---|
| Q1 | .NET 8 and .NET 9 both went out of support on 2026-11-10. Target .NET 10 LTS instead? | **Yes**, `net10.0`, with `net8.0` as a second target only if a host needs it. |
| Q2 | Is the aim performance, or is a C# code base wanted for another reason (integration with other .NET software, team skills)? | Treat the performance gates as go/no-go unless there is another reason. It decides whether a failed kernel spike ends the project. |
| Q3 | The C# engine's numbers will never equal the Python engine's for a given seed; fixed-seed baselines and calibration caches are per engine. Acceptable? | **Yes.** It is what permits a C#-native design at all (sections 3.4, 4.2, 5). |
| Q4 | For features Python parses but ignores with a warning (catch-based fishing, by-class rates, selectivity types 2 and 3): refuse, or imitate? | **Refuse.** |
| Q5 | Where does the scope end? | Phases 1-3 as planned. Decide on bioenergetics after the Phase 2 measurements. Leave genetics and economics on the Python engine. |
| Q6 | Production: build the assembly on the server (needs the SDK) or ship a CI artifact (needs only the runtime)? | Build on the server, from the deployed ref. |
| Q7 | Is cross-machine bit-reproducibility required? | No for the MVP: platform libm plus the fingerprint. If yes, add bundled `exp`/`pow` and measure the cost. |
| Q8 | Should calibration switch to the C# engine by default once Phase 2 passes? | Opt-in until one full Baltic certification reproduces on the C# engine. |
| Q9 | "One entry point": are `SimulationConfig.Load` and `Save` (bundle files, no simulation) acceptable beside `RunSimulation`? | **Yes.** Without them the C# tests and benchmarks need a Python process. |
| Q10 | Is the live map needed on the C# engine? | Not in scope. Later through an opt-in snapshot buffer. |
| Q11 | Ensemble margins of 1.25x and 1.05x with a Python-versus-Python control arm, 32 replicates: acceptable as the equivalence criterion? | **Yes**, fixed before the first run. |

**Answered by the owner, 2026-11-17:**
- **Q1:** .NET 10 LTS.
- **Q2:** performance; the gates are go/no-go.
- **Q6:** build on the server.
- **Q11:** the margins are accepted.
- **D12's sequencing:** calibration and scripts in Phase 1, the UI in Phase 2.

Q4 takes its recommendation (refuse), which follows from D11 (fail closed). The rest stand as recommended until the owner says otherwise.

---

## Appendix A. What was read for this document

At `a1fa75e4`, in full or in the cited functions:

- `osmose/engine/simulate.py`: `simulate`, `StepOutput`, `SimulationContext`,
  `_reset_step_variables`, `_collect_background_outputs`, `_strip_background`,
  `_floor_sub_one_fish`, `_collect_biomass_abundance`, `_collect_mortality`,
  `mortality_rates_from_counts`, `_collect_by_life_stage`, `_collect_yield`, `_collect_outputs`,
  `_average_step_outputs`.
- `osmose/engine/processes/mortality.py`: `_get_mortality_causes`, `_consume`,
  `_pre_generate_cell_rng`, `_precompute_resource_arrays`, `_zero_exempt`,
  `_precompute_effective_rates`, `_precompute_foraging_rates`, `_apply_predation_numba`,
  `_apply_single_cause`, `_mortality_in_cell_numba`, `_cell_prey_candidates`,
  `_mortality_all_cells_numba`, `_mortality_all_cells_parallel`, `mortality`.
- `osmose/engine/processes/`: `natural.py`, `starvation.py`, `growth.py`, `feeding_stage.py`,
  `reproduction.py`, and `movement.py` (`_map_move_school`, `build_random_patches`,
  `random_walk`, `movement`, `_map_move_batch_numba`, `_flatten_all_map_sets`,
  `_precompute_map_indices`).
- `osmose/engine/`: `state.py`, `rng.py`, `rng_compat.py` (header), `__init__.py`,
  `resources.py` (`logistic_regrow`, `ResourceState.__init__`, `update`),
  `background.py` (`BackgroundState`), `incoming_flux.py` (`get_incoming_schools`),
  `initialization.py`, `accessibility.py`, `movement_maps.py` (class docstring and lookups),
  `config.py` (the `EngineConfig` field list, `_warn_unsupported_mortality_features`).
- `osmose/runner.py::java_engine_block_reason`; `osmose/results.py`
  (`_build_dataframes_from_outputs`, `from_outputs`); `osmose/calibration/problem.py`
  (`_run_python_engine`); `scripts/calibrate_baltic.py` (`run_simulation`,
  `_run_with_timeout`); `scripts/cross_engine_parity_440.py` (`tost`); `ui/pages/run.py`
  (`_python_engine_thread`, the engine panels); `.github/workflows/ci.yml`; `deploy.sh`
  (header); `Dockerfile` and `apptainer/osmose.def` (base images).
- `docs/osmopy-performance-optimization-plan.md` (all);
  `docs/perf/2026-06-24-native-predation-kernel-spike.md`;
  `docs/perf/2026-11-17-phase1a-prefilter.md` (head); `docs/java_cross_check_2026-11-11.md`.

**Not read**, so the spec makes no detailed claim about them: `_bioen_step` and the bioen
process modules, the genetics and economics subpackages, `output.py`'s writers, the by-age and
by-size collectors, `timeseries.py` beyond its class list.

**Introspection run** (no simulation): `EngineConfig.from_dict` on the three configs, printing
which optional fields are set. That is the source of every "used by" entry in section 7.1.

## Appendix B. External sources (fetched 2026-11-17)

- Microsoft .NET blog, ".NET 8 and .NET 9 end of support":
  `https://devblogs.microsoft.com/dotnet/dotnet-8-9-end-of-support/`.
- pythonnet documentation, runtime loading and supported runtimes:
  `https://pythonnet.github.io/pythonnet/python.html`.
- pythonnet on PyPI (version 3.2.1, Python 3.11-3.15): `https://pypi.org/project/pythonnet/`.
- pythonnet source, `src/runtime/MethodBinder.cs` and `Converter.cs`, read through a fetched
  summary, not line by line. The two claims taken from them (the GIL is released around a
  call; `Int64` has an explicit conversion) are checks in the bridge spike, not assumptions.

Not verified anywhere: that pythonnet 3.2.1 loads .NET 10; the cost of a Philox block in
RyuJIT; every performance figure marked inferred.


