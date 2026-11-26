using System.Diagnostics;
using System.Runtime.ExceptionServices;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// One simulation (spec 4.1). Each stage is a method so a probe can drive part of a step with
/// injected draws. No static state: one object per <see cref="OsmoseEngine.RunSimulation"/> call.
/// With <c>run.threads</c> above 1, mortality runs cells heaviest first with single-cell claims
/// and movement runs fixed chunks of schools (spec D4); everything else is serial. Results do not
/// depend on the thread count: cells are disjoint, a school's move touches only that school,
/// draws are counter-based, and every reduction is serial.
/// </summary>
public sealed class SimulationRun
{
    private readonly EngineParameters _p;
    /// <summary>Schools per movement chunk (spec D4).</summary>
    public const int MoveChunk = 2048;

    private readonly KernelTables _tables;
    private readonly int _workers;
    private readonly ParallelOptions _options;
    private readonly CellScratch[] _cellScratch;
    private readonly int[][] _moveScratch;
    private readonly int[] _cellOrder;
    private readonly long[] _cellCost;
    private readonly double[] _rsc;
    private readonly double[] _ssb, _eggs;
    private readonly bool[] _seeded;
    private readonly PhiloxKey _key;
    private readonly ControlBlock _control;
    private readonly ResultBuffers? _buffers;
    private double[] _larva = new double[64];
    private readonly long[] _phaseTicks = new long[5];
    private readonly double[] _bkgByOrdinal, _bkgBiomass, _bkgAbundance;

    public SimulationRun(EngineParameters p, SimulationConfig config)
    {
        _p = p;
        _tables = KernelTables.From(p);
        int nCells = p.Ny * p.Nx, nFocal = p.Species.NFocal, nTotal = p.Species.NTotal;
        Store = new SchoolStore(nCells, initialCapacity: 1024);
        _rsc = new double[Math.Max(1, p.Resources.N * nCells)];
        _workers = Math.Max(1, p.Threads);
        _options = new ParallelOptions { MaxDegreeOfParallelism = _workers };
        _cellScratch = new CellScratch[_workers];
        _moveScratch = new int[_workers][];
        for (int w = 0; w < _workers; w++)
        {
            _cellScratch[w] = new CellScratch();
            _moveScratch[w] = new int[Math.Max(1, Movement.ScratchLength(p.Movement))];
        }

        _cellOrder = new int[nCells];
        _cellCost = new long[nCells];
        bool hasBackground = p.Background.NSpecies > 0;
        _bkgByOrdinal = new double[hasBackground ? 2 * Background.SchoolsPerStep(p.Background) : 0];
        _bkgBiomass = new double[hasBackground ? nTotal : 0];
        _bkgAbundance = new double[hasBackground ? nTotal : 0];
        _ssb = new double[nFocal];
        _eggs = new double[nFocal];
        _seeded = new bool[nFocal];
        _key = PhiloxKey.ForRun(p.Seed, p.RngScheme);
        int nRecords = p.Output.Records(p.NSteps, p.Restart.StartStep);
        OutputLayout layout = p.Output.Layout(nTotal, nFocal);
        if ((layout.Families & OutputFamilies.Diet) != 0)
        {
            Store.EnableDiet(layout.DietColumns);
        }

        if ((layout.Families & OutputFamilies.DietStaged) != 0)
        {
            Store.EnableStagedDiet(layout.DietStages.Width);
        }

        LoadInitialSchools(p);
        Results = new RunResults(nRecords, layout);
        Collector = new OutputCollector(layout, p.Output.RecordFrequency);
        Collector.SetLifespans(p.Species.LifespanDt);
        _control = ControlBlock.From(config);
        _buffers = ResultBuffers.From(config, nRecords, layout);
    }

    public SchoolStore Store { get; }

    /// <summary>The run's resource biomass, cell-major (<c>[cell * nRsc + r]</c>): each step's
    /// update writes it, the kernel eats from it, and with depletable resources the next
    /// update reads what is left. For the probes.</summary>
    internal Span<double> ResourceBuffer => _rsc;

    public RunResults Results { get; }

    public OutputCollector Collector { get; }

    /// <summary>The starting school table of a restart or initialisation file (Java
    /// NetcdfPopulator): each row a new school, as the probes load states.</summary>
    private void LoadInitialSchools(EngineParameters p)
    {
        RestartParameters r = p.Restart;
        for (int i = 0; i < r.Species.Length; i++)
        {
            int cell = r.CellX[i] >= 0 && r.CellY[i] >= 0 ? (r.CellY[i] * p.Nx) + r.CellX[i] : -1;
            Store.Append(
                new SchoolCore
                {
                    SpeciesId = r.Species[i],
                    AgeDt = r.AgeDt[i],
                    Length = r.Length[i],
                    Weight = r.Weight[i],
                    Abundance = r.Abundance[i],
                    TrophicLevel = r.TrophicLevel[i],
                    Flags = r.IsEgg[i] != 0 ? SchoolFlags.IsEgg : SchoolFlags.None,
                    FirstFeedingAgeDt = checked((short)r.FirstFeedingAgeDt[i]),
                },
                new SchoolLife { CellId = cell });
        }
    }

    /// <summary>Runs every step, or until cancelled or past the deadline.</summary>
    public RunSummary Run()
    {
        var clock = Stopwatch.StartNew();
        int records = 0, steps = 0, peak = 0;
        if (_p.Output.Step0Include && _p.Restart.StartStep == 0) // a resumed run has none (Java)
        {
            Collector.SnapshotStages(_p, Store); // no step produced this state: split it as it is
            Store.ClearDiet(); // the rows must exist, zeroed, before the snapshot reads them...
            Collector.SnapshotDiet(Store); // ...and nothing was eaten
            Collector.Collect(_p, Store); // the initial state (empty without a warm start)
            Flush(records++, -1);
        }

        RunStatus status = RunStatus.Completed;
        int freq = _p.Output.RecordFrequency;
        RestartParameters rs = _p.Restart;
        for (int step = rs.StartStep; step < _p.NSteps; step++)
        {
            if (_control.CancelRequested)
            {
                status = RunStatus.Cancelled;
                break;
            }

            if (_p.DeadlineSeconds > 0 && clock.Elapsed.TotalSeconds >= _p.DeadlineSeconds)
            {
                status = RunStatus.TimedOut;
                break;
            }

            RunStep(step);
            if (rs.Directory is { } dir && rs.Write && (rs.Due(step) || step == _p.NSteps - 1))
            {
                SnapshotWriter.Write(Store, _p.Species.NFocal, step, System.IO.Path.Combine(dir, $"snap_{step}.bin"));
            }

            if ((step + 1) % freq == 0)
            {
                Flush(records++, step);
            }

            steps = step + 1;
            peak = Math.Max(peak, Store.Count);
            _control.Report(steps, records, Store.Count);
        }

        if (status == RunStatus.Completed && Collector.Pending > 0)
        {
            Flush(records++, _p.NSteps - 1); // a trailing partial window, with its own length
            _control.Report(steps, records, Store.Count);
        }

        return new RunSummary(status, steps, records, peak, clock.Elapsed.TotalSeconds, _p.RngScheme, PhaseSeconds(), PhaseNames);
    }

    /// <summary>Stages 2 to 12 of one step, the record flush excepted.</summary>
    public void RunStep(int step)
    {
        long t0 = Stopwatch.GetTimestamp();
        BeginStep();
        Move(step);
        long t1 = Stopwatch.GetTimestamp();

        // Background schools join after movement and before the regroup that sorts them into
        // their cells, and leave right after mortality (simulate()'s order).
        bool background = _p.Background.NSpecies > 0;
        ulong firstBackground = background ? Background.Inject(_p, Store, step) : 0;
        Regroup();
        long t2 = Stopwatch.GetTimestamp();
        var draws = new PhiloxCellDraws(_key, step);
        Mortality<PhiloxCellDraws, PhiloxDrawSource>(step, ref draws);
        Collector.SnapshotDiet(Store); // before the background schools leave: they have rows
        Collector.SnapshotStagedDiet(_p, Store); // likewise: background predators are staged now
        if (background)
        {
            RemoveBackground(firstBackground);
        }

        Collector.SnapshotFisheries(_p, Store, step); // the lengths the rates saw, before growth
        long t3 = Stopwatch.GetTimestamp();
        AfterMortality(step);
        long t4 = Stopwatch.GetTimestamp();
        CollectStep();
        long t5 = Stopwatch.GetTimestamp();
        _phaseTicks[0] += t1 - t0;
        _phaseTicks[1] += t2 - t1;
        _phaseTicks[2] += t3 - t2;
        _phaseTicks[3] += t4 - t3;
        _phaseTicks[4] += t5 - t4;
    }

    /// <summary>Names of <see cref="RunSummary.PhaseSeconds"/>, in order.</summary>
    public static readonly string[] PhaseNames = ["movement", "regroup", "mortality", "life_cycle", "outputs"];

    /// <summary>Seconds spent in each phase so far (<see cref="PhaseNames"/>).</summary>
    public double[] PhaseSeconds() => [.. _phaseTicks.Select(t => t / (double)Stopwatch.Frequency)];

    /// <summary>Stage 2, the per-step reset (<c>_reset_step_variables</c>): no school is out of
    /// the domain until movement says so. Deaths are reset by the mortality step, the step
    /// scratch after the regroup.</summary>
    public void BeginStep() => ClearOutOfDomain(Store.Core);

    /// <summary>Clears <see cref="SchoolFlags.IsOut"/> on every school.</summary>
    public static void ClearOutOfDomain(Span<SchoolCore> core)
    {
        for (int i = 0; i < core.Length; i++)
        {
            core[i].Flags &= ~SchoolFlags.IsOut;
        }
    }

    /// <summary>Stage 4: fixed chunks of schools across the workers.</summary>
    public void Move(int step)
    {
        int n = Store.Count;
        if (_workers == 1 || n <= MoveChunk)
        {
            Movement.Run(_p, Store, step, _key, _moveScratch[0], 0, n);
            return;
        }

        MoveInChunks(step, n);
    }

    /// <summary>A separate method: the lambda's captured locals are hoisted into a closure
    /// object allocated on ENTRY to the method that declares them, so the one-thread path above
    /// must not declare them (measured: 40 bytes per step otherwise).</summary>
    private void MoveInChunks(int step, int n)
    {
        int chunks = (n + MoveChunk - 1) / MoveChunk, next = -1;
        Parallel(slot =>
        {
            int c;
            while ((c = Interlocked.Increment(ref next)) < chunks)
            {
                Movement.Run(_p, Store, step, _key, _moveScratch[slot], c * MoveChunk, Math.Min(n, (c + 1) * MoveChunk));
            }
        });
    }

    /// <summary>Stage 5: drop dead schools, make cells contiguous, and clear the per-step scratch
    /// (predation success and preyed biomass are only ever added to).</summary>
    public void Regroup()
    {
        Store.Regroup();
        Store.Step.Clear();
    }

    /// <summary>Stage 6, with the draws of the caller's choosing.</summary>
    public void Mortality<TFactory, TDraws>(int step, ref TFactory draws)
        where TFactory : struct, ICellDraws<TDraws>
        where TDraws : struct, IDrawSource
    {
        if (_larva.Length < Store.Count)
        {
            _larva = new double[Math.Max(Store.Count, _larva.Length * 2)];
        }

        Collector.SnapshotBiomassDietStage(_p, Store); // Java initStep: before the larva pre-pass
        MortalityStep.Prepare(_p, Store, step, _rsc, _larva);
        Collector.SnapshotResourceBiomass(_p, _rsc); // the step's resources, now set
        if (_workers == 1)
        {
            for (int cell = 0; cell < Store.CellCount; cell++)
            {
                MortalityStep.RunCell<TFactory, TDraws>(_tables, Store, cell, _rsc, _cellScratch[0], ref draws);
            }
        }
        else
        {
            RunCellsHeaviestFirst<TFactory, TDraws>(draws);
        }

        MortalityStep.Commit(_p, Store, step, _larva);
    }

    /// <summary>Spec D4: the occupied cells sorted by an occupancy cost, heaviest first, each
    /// claimed by one interlocked increment (longest-processing-time first).</summary>
    private void RunCellsHeaviestFirst<TFactory, TDraws>(TFactory draws)
        where TFactory : struct, ICellDraws<TDraws>
        where TDraws : struct, IDrawSource
    {
        ReadOnlySpan<int> start = Store.CellStart;
        int count = 0;
        for (int cell = 0; cell < Store.CellCount; cell++)
        {
            long n = start[cell + 1] - start[cell];
            if (n > 0)
            {
                _cellOrder[count] = cell;
                _cellCost[count] = -((n * n) + (8 * n)); // ascending sort = heaviest first
                count++;
            }
        }

        Array.Sort(_cellCost, _cellOrder, 0, count);
        int next = -1;
        Parallel(slot =>
        {
            TFactory local = draws; // ForCell does not mutate the factory: a copy per worker
            CellScratch scratch = _cellScratch[slot];
            int i;
            while ((i = Interlocked.Increment(ref next)) < count)
            {
                MortalityStep.RunCell<TFactory, TDraws>(_tables, Store, _cellOrder[i], _rsc, scratch, ref local);
            }
        });
    }

    /// <summary>Runs <paramref name="body"/> once per worker slot; a failure inside surfaces as
    /// its own exception, not an <see cref="AggregateException"/>.</summary>
    private void Parallel(Action<int> body)
    {
        try
        {
            System.Threading.Tasks.Parallel.For(0, _workers, _options, body);
        }
        catch (AggregateException e) when (e.InnerExceptions.Count > 0)
        {
            ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }
    }

    /// <summary>Stages 8 to 11: floor, growth, aging, reproduction.</summary>
    public void AfterMortality(int step)
    {
        LifeCycle.Floor(Store.Core);
        LifeCycle.Growth(_p, Store.Core, Store.Step);
        LifeCycle.Aging(_p, Store.Core, Store.Ledger);

        // The life-stage split is taken here, as simulate() takes it: reproduction increments
        // ages and clears the egg flag, after which this step's egg deaths would be charged to
        // juveniles. Nothing to do for a minimal run.
        Collector.SnapshotStages(_p, Store);
        LifeCycle.Reproduction(_p, Store, step, _ssb, _eggs, _seeded);
    }

    /// <summary>The background schools' totals, then their removal (the stage between mortality
    /// and the life cycle). <paramref name="firstBackground"/> is the id of the first of them.</summary>
    public void RemoveBackground(ulong firstBackground) =>
        Background.CollectAndRemove(_p, Store, firstBackground, _bkgByOrdinal, _bkgBiomass, _bkgAbundance);

    /// <summary>The step's collection, with the background totals <see cref="RemoveBackground"/> left.</summary>
    public void CollectStep() => Collector.Collect(_p, Store, _bkgBiomass, _bkgAbundance);

    private void Flush(int record, int recordStep)
    {
        Collector.Flush(Results, record, recordStep);
        _buffers?.Write(Results, record);
    }
}

/// <summary>The shared control block (spec 8.5): slot 0 cancel (Python), 1 steps completed,
/// 2 records written, 3 live schools. Absent, or file-backed, it is inert.</summary>
internal sealed unsafe class ControlBlock
{
    private readonly long* _slots;

    private ControlBlock(long* slots) => _slots = slots;

    public static ControlBlock From(SimulationConfig config)
    {
        if (config.IsFileBacked || !config.TryGet("control.block", out BundleEntry e))
        {
            return new ControlBlock(null);
        }

        if (e.Kind != EntryKind.Array || e.DType != DType.I8 || e.Length < 4)
        {
            throw new BundleValidationException(["control.block: expected an int64 array of at least 4 slots"]);
        }

        return new ControlBlock((long*)e.Address);
    }

    public bool CancelRequested => _slots != null && Volatile.Read(ref _slots[0]) != 0;

    public void Report(int steps, int records, int schools)
    {
        if (_slots == null)
        {
            return;
        }

        Volatile.Write(ref _slots[3], schools);
        Volatile.Write(ref _slots[2], records);
        Volatile.Write(ref _slots[1], steps); // last: a reader that sees the step sees its records
    }
}

/// <summary>The Python-owned result buffers (spec 8.4), written record by record so that a
/// cancelled run leaves its finished records valid. Absent, or file-backed, results stay in
/// <see cref="RunResults"/> only.</summary>
internal sealed unsafe class ResultBuffers
{
    private readonly int* _recordStep, _windowSteps;
    private readonly double* _biomass, _abundance, _mortality, _yield;
    private readonly (string Name, int Width, nint Address)[] _extra;

    private ResultBuffers(int* recordStep, int* windowSteps, double* biomass, double* abundance, double* mortality, double* yield, (string Name, int Width, nint Address)[] extra)
    {
        _recordStep = recordStep;
        _windowSteps = windowSteps;
        _biomass = biomass;
        _abundance = abundance;
        _mortality = mortality;
        _yield = yield;
        _extra = extra;
    }

    public static ResultBuffers? From(SimulationConfig config, int nRecords, int nTotal, int nFocal, OutputFamilies families = OutputFamilies.None) =>
        From(config, nRecords, new OutputLayout(nTotal, nFocal, families));

    /// <summary>The buffers of the families the layout has, each required with its exact shape
    /// (<c>[records, ...the family's record shape]</c>); a buffer of a family the run will not
    /// fill is a problem too.</summary>
    public static ResultBuffers? From(SimulationConfig config, int nRecords, OutputLayout layout)
    {
        if (config.IsFileBacked || !config.TryGet("result.biomass", out _))
        {
            return null;
        }

        int nTotal = layout.NTotal, nFocal = layout.NFocal;
        var problems = new List<string>();
        nint Get(string name, DType dtype, params long[] shape)
        {
            if (!config.TryGet(name, out BundleEntry e) || e.Kind != EntryKind.Array || e.DType != dtype || !e.GetShape().SequenceEqual(shape))
            {
                problems.Add($"{name}: expected {dtype} [{string.Join(", ", shape)}]");
                return 0;
            }

            return e.Address;
        }

        nint rs = Get("result.record_step", DType.I4, nRecords), ws = Get("result.window_steps", DType.I4, nRecords);
        nint b = Get("result.biomass", DType.F8, nRecords, nTotal), a = Get("result.abundance", DType.F8, nRecords, nTotal);
        nint m = Get("result.mortality", DType.F8, nRecords, nFocal, RunResults.MortalityCauses);
        nint y = Get("result.yield", DType.F8, nRecords, nFocal);
        IReadOnlyList<FamilySpec> specs = layout.Specs();
        var extra = new (string Name, int Width, nint Address)[specs.Count];
        for (int i = 0; i < specs.Count; i++)
        {
            FamilySpec spec = specs[i];
            long[] shape = [nRecords, .. spec.Shape.Select(v => (long)v)];
            extra[i] = (spec.Name, spec.Width, Get($"result.{spec.Name}", DType.F8, shape));
        }

        foreach (string name in OutputLayout.AllNames)
        {
            if (!specs.Any(spec => spec.Name == name) && config.TryGet($"result.{name}", out _))
            {
                problems.Add($"result.{name}: the bundle's output.families does not include it");
            }
        }

        if (problems.Count > 0)
        {
            throw new BundleValidationException(problems);
        }

        return new ResultBuffers((int*)rs, (int*)ws, (double*)b, (double*)a, (double*)m, (double*)y, extra);
    }

    public void Write(RunResults r, int record)
    {
        _recordStep[record] = r.RecordStep[record];
        _windowSteps[record] = r.WindowSteps[record];
        Copy(r.Biomass, _biomass, record, r.NTotal);
        Copy(r.Abundance, _abundance, record, r.NTotal);
        Copy(r.Mortality, _mortality, record, r.NFocal * RunResults.MortalityCauses);
        Copy(r.Yield, _yield, record, r.NFocal);
        foreach ((string name, int width, nint address) in _extra)
        {
            Copy(r.Extra(name), (double*)address, record, width);
        }
    }

    private static void Copy(double[] from, double* to, int record, int width) =>
        from.AsSpan(record * width, width).CopyTo(new Span<double>(to + ((long)record * width), width));
}
