using System.Runtime.InteropServices;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Diagnostics;

/// <summary>
/// Test harnesses for plan task 1.8, callable through pythonnet.
/// <see cref="Step"/> replays one step of <see cref="SimulationRun"/> from mortality entry
/// (regroup, mortality with recorded draws, floor, growth, aging, reproduction, output collection)
/// on a captured state, writing schools by id (the input's order, then new egg schools) and the
/// step's outputs. <see cref="Windows"/> collects several states and folds them into records.
/// </summary>
public static class PipelineProbe
{
    public static void Step(string modelPath, string statePath, string drawsPath, int step, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig state = SimulationConfig.Load(statePath);
        using SimulationConfig drawsBundle = SimulationConfig.Load(drawsPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var run = new SimulationRun(p, model);
        int n = Load(state, p, run.Store);

        // A regroup reorders schools but not the per-step scratch, so a real step inherits the
        // previous step's values at each position. Poison them: the pipeline must clear them.
        var poison = new SchoolStep { PredSuccess = double.NaN, PreyedBiomass = double.NaN, TlWeightedSum = double.NaN, EggRelease = double.NaN };
        run.Store.Step.Fill(poison);
        run.Regroup();
        var draws = new InjectedCellDraws(InjectedDraws.FromBundle(drawsBundle, run.Store.CellStart, p.Ny * p.Nx, p.NSubdt));
        run.Mortality<InjectedCellDraws, InjectedDrawSource>(step, ref draws);
        run.AfterMortality(step);
        run.Collector.Collect(p, run.Store);
        run.Collector.Flush(run.Results, 0, step);

        SchoolStore s = run.Store;
        int m = s.Count;
        if ((long)m - n != CountNew(s, n))
        {
            throw new InvalidOperationException("school ids are not dense");
        }

        var o = new Columns(m);
        for (int i = 0; i < m; i++)
        {
            o.Set((int)s.Life[i].Id, s.Core[i], s.Ledger[i], s.Life[i], p.Nx);
        }

        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            o.Write(output, handles, m);
            WriteResults(output, handles, run.Results);
            output.Save(outPath);
        }
        finally
        {
            Free(handles);
        }
    }

    /// <summary>Collects K states (concatenated, <c>state.offsets</c> K + 1) as K consecutive
    /// steps of a run with the model's record frequency and step0 setting, and writes the records.
    /// The step0 record is the first state when <c>output.step0_include</c> is set.</summary>
    public static void Windows(string modelPath, string statesPath, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        using SimulationConfig states = SimulationConfig.Load(statesPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var r = new BundleReader(states, 1);
        long[] offsets = r.Array("state.offsets", DType.I8, BundleReader.Any)!.ToArray<long>();
        r.ThrowIfProblems();
        int k = offsets.Length - 1, freq = p.Output.RecordFrequency, first = p.Output.Step0Include ? 1 : 0;
        int nSteps = k - first;
        int nRecords = first + ((nSteps + freq - 1) / freq);
        OutputLayout layout = p.Output.Layout(p.Species.NTotal, p.Species.NFocal);
        var results = new RunResults(nRecords, layout);
        var collector = new OutputCollector(layout, freq);
        int record = 0;
        for (int t = 0; t < k; t++)
        {
            var store = new SchoolStore(p.Ny * p.Nx, 16);
            LoadSlice(states, p, store, offsets[t], offsets[t + 1]);
            collector.SnapshotStages(p, store); // the state as it is, as _collect_outputs falls back to
            collector.Collect(p, store);
            int step = t - first;
            if (step < 0)
            {
                collector.Flush(results, record++, -1);
            }
            else if ((step + 1) % freq == 0)
            {
                collector.Flush(results, record++, step);
            }
        }

        if (collector.Pending > 0)
        {
            collector.Flush(results, record++, nSteps - 1);
        }

        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            WriteResults(output, handles, results);
            output.Save(outPath);
        }
        finally
        {
            Free(handles);
        }
    }

    /// <summary>A whole run of the model bundle (file-backed: the records stay in the run's own
    /// results), written with the summary's counts.</summary>
    public static void Run(string modelPath, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var run = new SimulationRun(p, model);
        RunSummary summary = run.Run();
        long[] counts = [(long)summary.Status, summary.StepsCompleted, summary.RecordsWritten, summary.PeakSchools];
        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            WriteResults(output, handles, run.Results);
            output.SetArray("out.summary", DType.I8, Pin(handles, counts), counts.Length);
            output.Save(outPath);
        }
        finally
        {
            Free(handles);
        }
    }

    /// <summary>Steady-state allocation (plan 1.9): run <paramref name="warmSteps"/> steps, let
    /// tiered compilation settle, then count this thread's managed allocations over the next
    /// <paramref name="measureSteps"/>. Writes [allocated bytes, store capacity before, after,
    /// schools at the end]; a capacity change means the population grew, not a per-step leak.</summary>
    public static void Allocations(string modelPath, int warmSteps, int measureSteps, string outPath)
    {
        using SimulationConfig model = SimulationConfig.Load(modelPath);
        EngineParameters p = EngineParameters.From(model, checkFeatures: false);
        var run = new SimulationRun(p, model);
        int freq = p.Output.RecordFrequency;
        void Step(int step)
        {
            run.RunStep(step);
            if (run.Collector.Pending == freq)
            {
                run.Collector.Flush(run.Results, 0, step); // record 0, overwritten: nothing kept
            }
        }

        for (int step = 0; step < warmSteps; step++)
        {
            Step(step);
        }

        Thread.Sleep(500);
        for (int step = warmSteps; step < warmSteps + 3; step++)
        {
            Step(step); // re-touch every path once more after the sleep
        }

        long capacityBefore = run.Store.Capacity;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int step = warmSteps + 3; step < warmSteps + 3 + measureSteps; step++)
        {
            Step(step);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        long[] counts = [allocated, capacityBefore, run.Store.Capacity, run.Store.Count];
        var handles = new List<GCHandle>();
        try
        {
            var output = new SimulationConfig(1);
            output.SetArray("out.allocations", DType.I8, Pin(handles, counts), counts.Length);
            output.Save(outPath);
        }
        finally
        {
            Free(handles);
        }
    }

    private static long CountNew(SchoolStore s, int n)
    {
        long c = 0;
        for (int i = 0; i < s.Count; i++)
        {
            c += s.Life[i].Id >= (ulong)n ? 1 : 0;
        }

        return c;
    }

    private static int Load(SimulationConfig state, EngineParameters p, SchoolStore store)
    {
        var r = new BundleReader(state, 1);
        long n = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        r.ThrowIfProblems();
        LoadSlice(state, p, store, 0, n);
        return (int)n;
    }

    private static void LoadSlice(SimulationConfig state, EngineParameters p, SchoolStore store, long from, long to)
    {
        var r = new BundleReader(state, 1);
        long total = r.Array("state.species_id", DType.I4, BundleReader.Any)!.Length;
        ReadOnlySpan<int> sp = r.Array("state.species_id", DType.I4, total)!.AsSpan<int>();
        ReadOnlySpan<int> age = r.Array("state.age_dt", DType.I4, total)!.AsSpan<int>();
        ReadOnlySpan<double> len = r.Array("state.length", DType.F8, total)!.AsSpan<double>();
        ReadOnlySpan<double> w = r.Array("state.weight", DType.F8, total)!.AsSpan<double>();
        ReadOnlySpan<double> a = r.Array("state.abundance", DType.F8, total)!.AsSpan<double>();
        ReadOnlySpan<double> tl = r.Array("state.trophic_level", DType.F8, total)!.AsSpan<double>();
        ReadOnlySpan<int> cx = r.Array("state.cell_x", DType.I4, total)!.AsSpan<int>();
        ReadOnlySpan<int> cy = r.Array("state.cell_y", DType.I4, total)!.AsSpan<int>();
        ReadOnlySpan<byte> bkg = r.Array("state.is_background", DType.U1, total)!.AsSpan<byte>();
        ReadOnlySpan<byte> egg = r.Array("state.is_egg", DType.U1, total)!.AsSpan<byte>();
        ReadOnlySpan<byte> isOut = r.Array("state.is_out", DType.U1, total)!.AsSpan<byte>();
        ReadOnlySpan<byte> seed = r.Array("state.from_seeding", DType.U1, total)!.AsSpan<byte>();
        ReadOnlySpan<int> ffa = r.Array("state.first_feeding_age_dt", DType.I4, total)!.AsSpan<int>();
        ReadOnlySpan<double> starv = r.Array("state.starvation_rate", DType.F8, total)!.AsSpan<double>();
        ReadOnlySpan<double> dead = r.Array("state.n_dead", DType.F8, total, RunResults.MortalityCauses)!.AsSpan<double>();
        r.ThrowIfProblems();
        for (long i = from; i < to; i++)
        {
            SchoolFlags flags = (bkg[(int)i] != 0 ? SchoolFlags.IsBackground : 0) | (egg[(int)i] != 0 ? SchoolFlags.IsEgg : 0)
                | (isOut[(int)i] != 0 ? SchoolFlags.IsOut : 0) | (seed[(int)i] != 0 ? SchoolFlags.FromSeeding : 0);
            ReadOnlySpan<double> d = dead.Slice((int)i * RunResults.MortalityCauses, RunResults.MortalityCauses);
            int at = store.Append(
                new SchoolCore
                {
                    SpeciesId = sp[(int)i],
                    AgeDt = age[(int)i],
                    Length = len[(int)i],
                    Weight = w[(int)i],
                    Abundance = a[(int)i],
                    TrophicLevel = tl[(int)i],
                    Flags = flags,
                    FirstFeedingAgeDt = checked((short)ffa[(int)i]),
                    DeadPredation = d[0],
                },
                new SchoolLife { StarvationRate = starv[(int)i], CellId = cx[(int)i] >= 0 && cy[(int)i] >= 0 ? (cy[(int)i] * p.Nx) + cx[(int)i] : -1 });
            store.Ledger[at] = new SchoolLedger
            {
                DeadStarvation = d[1],
                DeadAdditional = d[2],
                DeadFishing = d[3],
                DeadOut = d[4],
                DeadForaging = d[5],
                DeadDiscards = d[6],
                DeadAging = d[7],
            };
        }
    }

    private static void WriteResults(SimulationConfig output, List<GCHandle> handles, RunResults r)
    {
        int n = r.NRecords;
        output.SetArray("out.record_step", DType.I4, Pin(handles, r.RecordStep), n);
        output.SetArray("out.window_steps", DType.I4, Pin(handles, r.WindowSteps), n);
        output.SetArray("out.biomass", DType.F8, Pin(handles, r.Biomass), n, r.NTotal);
        output.SetArray("out.abundance", DType.F8, Pin(handles, r.Abundance), n, r.NTotal);
        output.SetArray("out.mortality", DType.F8, Pin(handles, r.Mortality), n, r.NFocal, RunResults.MortalityCauses);
        output.SetArray("out.yield", DType.F8, Pin(handles, r.Yield), n, r.NFocal);
        foreach (FamilySpec spec in r.Specs)
        {
            long[] d = [.. spec.Shape.Select(v => (long)v), -1, -1];
            output.SetArray($"out.{spec.Name}", DType.F8, Pin(handles, r.Extra(spec.Name)), n, d[0], d[1], d[2]);
        }
    }

    private static long Pin(List<GCHandle> handles, Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        handles.Add(h);
        return h.AddrOfPinnedObject();
    }

    private static void Free(List<GCHandle> handles)
    {
        foreach (GCHandle h in handles)
        {
            h.Free();
        }
    }

    /// <summary>Per-school output columns, indexed by school id.</summary>
    private sealed class Columns(int m)
    {
        private readonly int[] _sp = new int[m], _age = new int[m], _x = new int[m], _y = new int[m];
        private readonly double[] _a = new double[m], _len = new double[m], _w = new double[m], _tl = new double[m], _starv = new double[m];
        private readonly double[] _dead = new double[m * RunResults.MortalityCauses];
        private readonly byte[] _egg = new byte[m], _out = new byte[m];

        public void Set(int id, in SchoolCore c, in SchoolLedger l, in SchoolLife life, int nx)
        {
            _sp[id] = c.SpeciesId;
            _age[id] = c.AgeDt;
            _x[id] = life.CellId >= 0 ? life.CellId % nx : -1;
            _y[id] = life.CellId >= 0 ? life.CellId / nx : -1;
            _a[id] = c.Abundance;
            _len[id] = c.Length;
            _w[id] = c.Weight;
            _tl[id] = c.TrophicLevel;
            _starv[id] = life.StarvationRate;
            _egg[id] = (byte)((c.Flags & SchoolFlags.IsEgg) != 0 ? 1 : 0);
            _out[id] = (byte)((c.Flags & SchoolFlags.IsOut) != 0 ? 1 : 0);
            double[] row = [c.DeadPredation, l.DeadStarvation, l.DeadAdditional, l.DeadFishing, l.DeadOut, l.DeadForaging, l.DeadDiscards, l.DeadAging];
            row.CopyTo(_dead, id * RunResults.MortalityCauses);
        }

        public void Write(SimulationConfig output, List<GCHandle> handles, int m)
        {
            output.SetArray("s.species_id", DType.I4, Pin(handles, _sp), m);
            output.SetArray("s.age_dt", DType.I4, Pin(handles, _age), m);
            output.SetArray("s.cell_x", DType.I4, Pin(handles, _x), m);
            output.SetArray("s.cell_y", DType.I4, Pin(handles, _y), m);
            output.SetArray("s.abundance", DType.F8, Pin(handles, _a), m);
            output.SetArray("s.length", DType.F8, Pin(handles, _len), m);
            output.SetArray("s.weight", DType.F8, Pin(handles, _w), m);
            output.SetArray("s.trophic_level", DType.F8, Pin(handles, _tl), m);
            output.SetArray("s.starvation_rate", DType.F8, Pin(handles, _starv), m);
            output.SetArray("s.is_egg", DType.U1, Pin(handles, _egg), m);
            output.SetArray("s.is_out", DType.U1, Pin(handles, _out), m);
            output.SetArray("s.n_dead", DType.F8, Pin(handles, _dead), m, RunResults.MortalityCauses);
        }
    }
}
