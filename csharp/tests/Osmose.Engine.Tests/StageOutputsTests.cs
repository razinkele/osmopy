using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.6a: the life-stage split and the per-step mortality rates
/// (<c>_collect_by_life_stage</c>, <c>mortality_rates_from_counts</c>). The checks against the
/// Python engine on real states are on the Python side (tests/test_csharp_outputs_stage.py);
/// these pin the binning, the rate formula's branches and the window rule on cases small enough
/// to read. Expected rates are NumPy's, with the C library's log (2026-11-18).</summary>
public sealed class StageOutputsTests : IDisposable
{
    private const int NFocal = 2;
    private const int NTotal = 3;
    private const int Causes = RunResults.MortalityCauses;
    private const int Stages = OutputCollector.Stages;

    private static readonly double[] MaturitySize = [20.0, 12.0];
    private static readonly int[] MaturityAgeDt = [48, 0];
    private static readonly double[] Cutoff = [0.0, 0.0, 0.0];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static double Bits(ulong b) => BitConverter.UInt64BitsToDouble(b);

    private static int At(int sp, int cause, int stage) => (sp * Causes * Stages) + (cause * Stages) + stage;

    private static void Add(SchoolStore store, int species, double abundance, double length, int ageDt, SchoolFlags flags, double predation = 0.0, double fishing = 0.0, double aging = 0.0)
    {
        store.Append(
            new SchoolCore { SpeciesId = species, Abundance = abundance, Length = length, AgeDt = ageDt, Flags = flags, DeadPredation = predation, Weight = 0.002 },
            new SchoolLife { CellId = 0 });
        ref SchoolLedger l = ref store.Ledger[store.Count - 1];
        l.DeadFishing = fishing;
        l.DeadAging = aging;
    }

    private static OutputCollector Collector(int frequency = 1) => new(NTotal, NFocal, frequency, OutputFamilies.Stage);

    [Fact]
    public void ASchoolIsAnEggOrMatureByLengthAndAgeOrAJuvenile()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 100.0, 0.1, 0, SchoolFlags.IsEgg, predation: 5.0); // egg
        Add(store, 0, 30.0, 25.0, 47, 0, fishing: 2.0); // long enough, too young: juvenile
        Add(store, 0, 40.0, 19.9, 96, 0, predation: 1.0); // old enough, too short: juvenile
        Add(store, 0, 7.0, 20.0, 48, 0, fishing: 3.0, aging: 0.5); // both, at the bounds: adult
        Add(store, 1, 11.0, 12.0, 0, 0); // maturity age 0: adult by length alone
        Add(store, 1, 0.0, 30.0, 96, SchoolFlags.IsEgg, aging: 9.0); // the egg flag wins over size
        Add(store, 2, 500.0, 110.0, 48, SchoolFlags.IsBackground, predation: 4.0); // not focal

        OutputCollector c = Collector();
        c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        c.Collect(Cutoff, 24.0, store, [], []);
        var r = new RunResults(1, NTotal, NFocal, OutputFamilies.Stage);
        c.Flush(r, 0, 0);

        Assert.Equal(new[] { 100.0, 70.0, 7.0, 0.0, 0.0, 11.0 }, r.AbundanceByStage);
        var deaths = new double[NFocal * Causes * Stages];
        deaths[At(0, 0, 0)] = 5.0;
        deaths[At(0, 0, 1)] = 1.0;
        deaths[At(0, 3, 1)] = 2.0;
        deaths[At(0, 3, 2)] = 3.0;
        deaths[At(0, 7, 2)] = 0.5;
        deaths[At(1, 7, 0)] = 9.0;
        Assert.Equal(deaths, r.DeathsByStage);
    }

    [Fact]
    public void RatesAreNumPysForAnOrdinaryAnEmptiedAndAnUntouchedStage()
    {
        // deaths[cause][stage]; n_end = [1000, 0, 50]. Stage 0's eight causes sum to 10.851 added
        // in cause order and to 10.850999999999999 pairwise: NumPy adds them in order here.
        double[] abundance = [1000.0, 0.0, 50.0];
        var deaths = new double[Causes * Stages];
        deaths[At(0, 0, 0)] = 3.25;
        deaths[At(0, 0, 1)] = 40.0;
        deaths[At(0, 2, 0)] = 7.5;
        deaths[At(0, 2, 1)] = 0.125;
        deaths[At(0, 3, 0)] = 0.1;
        deaths[At(0, 7, 0)] = 1e-3;
        deaths[At(0, 7, 1)] = 9.0;
        var rates = new double[Causes * Stages];
        OutputCollector.StageRates(abundance, deaths, rates, 1);

        var want = new double[Causes * Stages];
        want[At(0, 0, 0)] = Bits(0x3F6A7B07D0A50235); // 0.00323249365206404
        want[At(0, 2, 0)] = Bits(0x3F7E8DF553348C64); // 0.007459600735532399
        want[At(0, 3, 0)] = Bits(0x3F1A12BCDD304130); // 9.946134314043199e-05
        want[At(0, 7, 0)] = Bits(0x3EB0AFD50870CD90); // 9.9461343140432e-07
        want[At(0, 0, 1)] = double.PositiveInfinity; // the stage was emptied: every cause that
        want[At(0, 2, 1)] = double.PositiveInfinity; // killed reads inf, the others 0
        want[At(0, 7, 1)] = double.PositiveInfinity;
        Assert.Equal(want, rates); // stage 2 had no deaths: exactly 0
    }

    [Fact]
    public void AStageWithNoSchoolsReadsZeroAndDeathsTooFewToMoveTheCountReadMinusZero()
    {
        double[] abundance = [0.0, 1e6, 2.5e15];
        var deaths = new double[Causes * Stages];
        deaths[At(0, 0, 1)] = 1e-13; // 1e6 + 1e-13 == 1e6: survival is exactly 1
        deaths[At(0, 0, 2)] = 1e16;
        deaths[At(0, 4, 2)] = 1e16;
        deaths[At(0, 2, 2)] = 3.0;
        foreach (int cause in new[] { 1, 3, 5, 6, 7 })
        {
            deaths[At(0, cause, 2)] = 1.0;
        }

        var rates = new double[Causes * Stages];
        OutputCollector.StageRates(abundance, deaths, rates, 1);

        Assert.All(Enumerable.Range(0, Causes), cause => Assert.Equal(0.0, rates[At(0, cause, 0)]));
        Assert.Equal(Bits(0x8000000000000000), rates[At(0, 0, 1)]); // share * -log(1) = -0.0
        Assert.True(double.IsNegative(rates[At(0, 0, 1)]));
        Assert.Equal(Bits(0x3FF193EA7AAD030A), rates[At(0, 0, 2)]); // 1.0986122886681096
        Assert.Equal(Bits(0x3FF193EA7AAD030A), rates[At(0, 4, 2)]);
        Assert.Equal(Bits(0x3CB7BFBEF4747D76), rates[At(0, 2, 2)]); // 3.2958368660043287e-16
        Assert.Equal(Bits(0x3C9FAA53F09B51F3), rates[At(0, 1, 2)]); // 1.0986122886681096e-16
    }

    [Fact]
    public void AWindowAveragesAbundanceAndSumsDeathsAndRates()
    {
        OutputCollector c = Collector(frequency: 3);
        var r = new RunResults(1, NTotal, NFocal, OutputFamilies.Stage);
        double[] juveniles = [80.0, 0.0, 60.0]; // the second step empties the juvenile stage
        double[] fished = [20.0, 80.0, 15.0];
        var perStep = new double[3];
        for (int t = 0; t < 3; t++)
        {
            var store = new SchoolStore(4);
            Add(store, 0, juveniles[t], 5.0, 10, 0, fishing: fished[t]);
            c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
            c.Collect(Cutoff, 24.0, store, [], []);
            perStep[t] = -Math.Log(juveniles[t] / (juveniles[t] + fished[t]));
        }

        c.Flush(r, 0, 2);
        Assert.Equal((80.0 + 0.0 + 60.0) / 3, r.AbundanceByStage[1]);
        Assert.Equal(20.0 + 80.0 + 15.0, r.DeathsByStage[At(0, 3, 1)]);
        Assert.Equal(double.PositiveInfinity, perStep[1]);
        Assert.Equal(double.PositiveInfinity, r.RatesByStage[At(0, 3, 1)]); // a sum of per-step rates
        Assert.Equal(0.0, r.RatesByStage[At(0, 0, 1)]);

        // Without the emptied step the record is the plain sum of the two steps' rates.
        for (int t = 0; t < 3; t += 2)
        {
            var store = new SchoolStore(4);
            Add(store, 0, juveniles[t], 5.0, 10, 0, fishing: fished[t]);
            c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
            c.Collect(Cutoff, 24.0, store, [], []);
        }

        c.Flush(r, 0, 1);
        Assert.Equal(perStep[0] + perStep[2], r.RatesByStage[At(0, 3, 1)]);
    }

    [Fact]
    public void TheSnapshotIsTakenOncePerStepAndIsNothingWhenTheFamilyIsOff()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 10.0, 5.0, 10, 0);
        OutputCollector on = Collector();
        InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => on.Collect(Cutoff, 24.0, store, [], []));
        Assert.Contains("life-stage snapshot", missing.Message, StringComparison.Ordinal);
        on.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        Assert.Throws<InvalidOperationException>(() => on.SnapshotStages(MaturitySize, MaturityAgeDt, store));

        var off = new OutputCollector(NTotal, NFocal, 1);
        off.SnapshotStages(MaturitySize, MaturityAgeDt, store); // no family: nothing to do
        off.Collect(Cutoff, 24.0, store, [], []);
        var r = new RunResults(1, NTotal, NFocal);
        off.Flush(r, 0, 0);
        Assert.Empty(r.AbundanceByStage);
        Assert.Empty(r.RatesByStage);
        Assert.Equal(10.0, r.Abundance[0]);
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Buffers(int n, bool stage, double[] abundance, double[] deaths, double[] rates, int stagesDeclared = Stages)
    {
        var c = new SimulationConfig(1);
        c.SetArray("result.record_step", DType.I4, Pin(new int[n]), n);
        c.SetArray("result.window_steps", DType.I4, Pin(new int[n]), n);
        c.SetArray("result.biomass", DType.F8, Pin(new double[n * NTotal]), n, NTotal);
        c.SetArray("result.abundance", DType.F8, Pin(new double[n * NTotal]), n, NTotal);
        c.SetArray("result.mortality", DType.F8, Pin(new double[n * NFocal * Causes]), n, NFocal, Causes);
        c.SetArray("result.yield", DType.F8, Pin(new double[n * NFocal]), n, NFocal);
        if (stage)
        {
            c.SetArray("result.abundance_by_stage", DType.F8, Pin(abundance), n, NFocal, stagesDeclared);
            c.SetArray("result.mortality_by_cause_stage", DType.F8, Pin(deaths), n, NFocal, Causes, Stages);
            c.SetArray("result.mortality_rate_by_cause_stage", DType.F8, Pin(rates), n, NFocal, Causes, Stages);
        }

        return c;
    }

    [Fact]
    public void TheStageFamiliesAreWrittenIntoTheCallersBuffersRecordByRecord()
    {
        const int n = 2;
        double[] abundance = new double[n * NFocal * Stages], deaths = new double[n * NFocal * Causes * Stages], rates = new double[n * NFocal * Causes * Stages];
        var r = new RunResults(n, NTotal, NFocal, OutputFamilies.Stage);
        for (int i = 0; i < r.AbundanceByStage.Length; i++)
        {
            r.AbundanceByStage[i] = 1.0 + i;
        }

        for (int i = 0; i < r.DeathsByStage.Length; i++)
        {
            r.DeathsByStage[i] = 100.0 + i;
            r.RatesByStage[i] = 1000.0 + i;
        }

        ResultBuffers b = ResultBuffers.From(Buffers(n, stage: true, abundance, deaths, rates), n, NTotal, NFocal, OutputFamilies.Stage)!;
        b.Write(r, 1); // only the second record
        int w = NFocal * Stages, wd = NFocal * Causes * Stages;
        Assert.All(abundance[..w], v => Assert.Equal(0.0, v));
        Assert.Equal(r.AbundanceByStage[w..], abundance[w..]);
        Assert.All(deaths[..wd], v => Assert.Equal(0.0, v));
        Assert.Equal(r.DeathsByStage[wd..], deaths[wd..]);
        Assert.Equal(r.RatesByStage[wd..], rates[wd..]);
    }

    [Fact]
    public void BuffersMustMatchTheFamiliesTheBundleDeclares()
    {
        const int n = 2;
        double[] a = new double[n * NFocal * Stages], d = new double[n * NFocal * Causes * Stages], q = new double[n * NFocal * Causes * Stages];
        BundleValidationException missing = Assert.Throws<BundleValidationException>(
            () => ResultBuffers.From(Buffers(n, stage: false, a, d, q), n, NTotal, NFocal, OutputFamilies.Stage));
        Assert.Contains(missing.Problems, m => m.Contains("result.abundance_by_stage", StringComparison.Ordinal));
        Assert.Contains(missing.Problems, m => m.Contains("result.mortality_rate_by_cause_stage", StringComparison.Ordinal));

        BundleValidationException shape = Assert.Throws<BundleValidationException>(
            () => ResultBuffers.From(Buffers(n, stage: true, new double[n * NFocal * 2], d, q, stagesDeclared: 2), n, NTotal, NFocal, OutputFamilies.Stage));
        Assert.Contains(shape.Problems, m => m.Contains("result.abundance_by_stage: expected F8 [2, 2, 3]", StringComparison.Ordinal));

        // A minimal run ignores nothing silently: stage buffers it will never fill are refused.
        BundleValidationException extra = Assert.Throws<BundleValidationException>(
            () => ResultBuffers.From(Buffers(n, stage: true, a, d, q), n, NTotal, NFocal, OutputFamilies.None));
        Assert.Contains(extra.Problems, m => m.Contains("result.abundance_by_stage: the bundle's output.families does not include it", StringComparison.Ordinal));
    }

    private SimulationConfig OutputGroup(long families)
    {
        var c = new SimulationConfig(1);
        c.SetArray("output.cutoff_age", DType.F8, Pin(new double[NTotal]), NTotal);
        c.SetInt("output.record_frequency", 24);
        c.SetInt("output.step0_include", 0);
        c.SetInt("output.families", families);
        return c;
    }

    private static (OutputParameters Output, IReadOnlyList<string> Problems) Read(SimulationConfig c, bool feature)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Output(r, NTotal, feature), r.Problems);
    }

    [Fact]
    public void TheFamiliesFollowTheDeclaredFeature()
    {
        (OutputParameters full, IReadOnlyList<string> none) = Read(OutputGroup(1), feature: true);
        Assert.Empty(none);
        Assert.Equal(OutputFamilies.Stage, full.Families);

        (OutputParameters minimal, IReadOnlyList<string> clean) = Read(OutputGroup(0), feature: false);
        Assert.Empty(clean);
        Assert.Equal(OutputFamilies.None, minimal.Families);

        (_, IReadOnlyList<string> undeclared) = Read(OutputGroup(1), feature: false);
        Assert.Contains(undeclared, m => m.Contains("output.families: 1, but the bundle does not declare feature full_outputs", StringComparison.Ordinal));

        (_, IReadOnlyList<string> empty) = Read(OutputGroup(0), feature: true);
        Assert.Contains(empty, m => m.Contains("full_outputs is declared, but output.families is 0", StringComparison.Ordinal));

        (_, IReadOnlyList<string> unknown) = Read(OutputGroup(1 | (1L << 40)), feature: true);
        Assert.Contains(unknown, m => m.Contains("output.families: unknown bits", StringComparison.Ordinal));

        (_, IReadOnlyList<string> noStage) = Read(OutputGroup(1L << 40), feature: true);
        Assert.Contains(noStage, m => m.Contains("output.families", StringComparison.Ordinal));
    }
}
