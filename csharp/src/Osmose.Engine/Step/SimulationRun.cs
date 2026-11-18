using System.Diagnostics;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// One simulation (spec 4.1), single-threaded (plan task 1.8; parallelism is 1.9). Each stage is
/// a method so a probe can drive part of a step with injected draws. No static state: one object
/// per <see cref="OsmoseEngine.RunSimulation"/> call.
/// </summary>
public sealed class SimulationRun
{
    private readonly EngineParameters _p;
    private readonly KernelTables _tables;
    private readonly CellScratch _cellScratch = new();
    private readonly double[] _rsc;
    private readonly int[] _moveScratch;
    private readonly double[] _ssb, _eggs;
    private readonly bool[] _seeded;
    private readonly PhiloxKey _key;
    private readonly ControlBlock _control;
    private readonly ResultBuffers? _buffers;
    private double[] _larva = new double[64];

    public SimulationRun(EngineParameters p, SimulationConfig config)
    {
        _p = p;
        _tables = KernelTables.From(p);
        int nCells = p.Ny * p.Nx, nFocal = p.Species.NFocal, nTotal = p.Species.NTotal;
        Store = new SchoolStore(nCells, initialCapacity: 1024);
        _rsc = new double[Math.Max(1, p.Resources.N * nCells)];
        _moveScratch = new int[Math.Max(1, Movement.ScratchLength(p.Movement))];
        _ssb = new double[nFocal];
        _eggs = new double[nFocal];
        _seeded = new bool[nFocal];
        _key = PhiloxKey.ForRun(p.Seed, p.RngScheme);
        int nRecords = p.Output.Records(p.NSteps);
        Results = new RunResults(nRecords, nTotal, nFocal);
        Collector = new OutputCollector(nTotal, nFocal, p.Output.RecordFrequency);
        _control = ControlBlock.From(config);
        _buffers = ResultBuffers.From(config, nRecords, nTotal, nFocal);
    }

    public SchoolStore Store { get; }

    public RunResults Results { get; }

    public OutputCollector Collector { get; }

    /// <summary>Runs every step, or until cancelled or past the deadline.</summary>
    public RunSummary Run()
    {
        var clock = Stopwatch.StartNew();
        int records = 0, steps = 0, peak = 0;
        if (_p.Output.Step0Include)
        {
            Collector.Collect(_p, Store); // the initial state (empty without a warm start)
            Flush(records++, -1);
        }

        RunStatus status = RunStatus.Completed;
        int freq = _p.Output.RecordFrequency;
        for (int step = 0; step < _p.NSteps; step++)
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

            BeginStep();
            Move(step);
            Regroup();
            var draws = new PhiloxCellDraws(_key, step);
            Mortality<PhiloxCellDraws, PhiloxDrawSource>(step, ref draws);
            AfterMortality(step);
            Collector.Collect(_p, Store);
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

        return new RunSummary(status, steps, records, peak, clock.Elapsed.TotalSeconds, _p.RngScheme);
    }

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

    /// <summary>Stage 4.</summary>
    public void Move(int step) => Movement.Run(_p, Store, step, _key, _moveScratch);

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

        MortalityStep.Run<TFactory, TDraws>(_p, _tables, Store, step, _rsc, _cellScratch, _larva, ref draws);
    }

    /// <summary>Stages 8 to 11: floor, growth, aging, reproduction.</summary>
    public void AfterMortality(int step)
    {
        LifeCycle.Floor(Store.Core);
        LifeCycle.Growth(_p, Store.Core, Store.Step);
        LifeCycle.Aging(_p, Store.Core, Store.Ledger);
        LifeCycle.Reproduction(_p, Store, step, _ssb, _eggs, _seeded);
    }

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

    private ResultBuffers(int* recordStep, int* windowSteps, double* biomass, double* abundance, double* mortality, double* yield)
    {
        _recordStep = recordStep;
        _windowSteps = windowSteps;
        _biomass = biomass;
        _abundance = abundance;
        _mortality = mortality;
        _yield = yield;
    }

    public static ResultBuffers? From(SimulationConfig config, int nRecords, int nTotal, int nFocal)
    {
        if (config.IsFileBacked || !config.TryGet("result.biomass", out _))
        {
            return null;
        }

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
        if (problems.Count > 0)
        {
            throw new BundleValidationException(problems);
        }

        return new ResultBuffers((int*)rs, (int*)ws, (double*)b, (double*)a, (double*)m, (double*)y);
    }

    public void Write(RunResults r, int record)
    {
        _recordStep[record] = r.RecordStep[record];
        _windowSteps[record] = r.WindowSteps[record];
        Copy(r.Biomass, _biomass, record, r.NTotal);
        Copy(r.Abundance, _abundance, record, r.NTotal);
        Copy(r.Mortality, _mortality, record, r.NFocal * RunResults.MortalityCauses);
        Copy(r.Yield, _yield, record, r.NFocal);
    }

    private static void Copy(double[] from, double* to, int record, int width) =>
        from.AsSpan(record * width, width).CopyTo(new Span<double>(to + ((long)record * width), width));
}
