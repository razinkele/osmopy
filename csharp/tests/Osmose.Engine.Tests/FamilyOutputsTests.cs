using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.6b: the distributions by age and by size, SSB, yield in numbers and the
/// two weighted means (<c>_collect_distributions</c>, <c>_collect_ssb</c>, <c>_collect_yield_n</c>,
/// <c>_collect_mean_tl</c>, <c>_collect_mean_size</c>). The checks against the Python engine on
/// real states are on the Python side (tests/test_csharp_outputs_families.py); these pin the
/// binning, the masks and the window rules on cases small enough to read.</summary>
public sealed class FamilyOutputsTests : IDisposable
{
    private const int NFocal = 2;
    private const int NTotal = 3;
    private const int Ndt = 24;

    private static readonly double[] MaturitySize = [20.0, 12.0];
    private static readonly int[] MaturityAgeDt = [48, 0];
    private static readonly double[] Cutoff = [0.5, 0.0, 0.0]; // species 0: half a year
    private static readonly int[] AgeOffsets = [0, 4, 6]; // 4 age classes, then 2
    private static readonly double[] Edges = [2.0, 6.0, 10.0];

    // the diet and the Java-rule distributions have their own tests
    private const OutputFamilies All = OutputFamilies.Known & ~OutputFamilies.Diet & ~OutputFamilies.JavaDistributions & ~OutputFamilies.CatchFamilies;

    private static readonly string[] JavaNames =
    [
        "biomass_by_tl", "meansize_byage_num", "meansize_byage_den", "meantl_bysize_num", "meantl_bysize_den",
        "yield_by_size", "size_catch_num", "size_catch_den", "tl_catch_num", "tl_catch_den", "mortality_rate_by_cause_size",
        "yield_by_fishery_landed", "yield_by_fishery_discarded",
    ];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static OutputLayout Layout(OutputFamilies families = All) =>
        new(NTotal, NFocal, families) { AgeBinOffsets = AgeOffsets, SizeEdges = Edges };

    private static void Add(SchoolStore store, int species, double abundance, double length, int ageDt, double weight = 0.5, double tl = 3.0, SchoolFlags flags = 0, double fishing = 0.0)
    {
        store.Append(
            new SchoolCore { SpeciesId = species, Abundance = abundance, Length = length, AgeDt = ageDt, Weight = weight, TrophicLevel = tl, Flags = flags },
            new SchoolLife { CellId = 0 });
        store.Ledger[store.Count - 1].DeadFishing = fishing;
    }

    private static RunResults CollectOne(SchoolStore store, OutputFamilies families = All)
    {
        var c = new OutputCollector(Layout(families), 1);
        c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        c.Collect(Cutoff, Ndt, MaturitySize, MaturityAgeDt, store, [], []);
        var r = new RunResults(1, Layout(families));
        c.Flush(r, 0, 0);
        return r;
    }

    [Theory]
    [InlineData(1.0, 0)] // below the first edge: clipped into the first class
    [InlineData(2.0, 0)]
    [InlineData(5.999999, 0)]
    [InlineData(6.0, 1)] // on an edge: the class that edge opens
    [InlineData(9.5, 1)]
    [InlineData(10.0, 2)]
    [InlineData(1e9, 2)] // beyond the last edge: the last class
    public void ALengthFallsInTheClassOfTheLastEdgeAtOrBelowIt(double length, int cls) =>
        Assert.Equal(cls, OutputCollector.SizeClass(Edges, length));

    [Fact]
    public void ASingleEdgeHoldsEveryLength()
    {
        Assert.Equal(0, OutputCollector.SizeClass([5.0], 1.0));
        Assert.Equal(0, OutputCollector.SizeClass([5.0], 50.0));
    }

    [Fact]
    public void DistributionsCountEveryFocalSchoolEggsAndEmptyOnesIncluded()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 100.0, 0.1, 0, weight: 0.001, flags: SchoolFlags.IsEgg); // age class 0, size class 0
        Add(store, 0, 10.0, 7.0, 30); // age class 1 (30 // 24), size class 1
        Add(store, 0, 4.0, 25.0, 24 * 9); // age class 9: clipped to the species' last, 3
        Add(store, 0, 0.0, 6.0, 47); // an empty school still lands in its classes
        Add(store, 1, 8.0, 10.0, 23); // species 1: age class 0, size class 2
        Add(store, 1, 2.0, 3.0, 48); // age class 2: clipped to 1
        Add(store, 2, 500.0, 110.0, 48, flags: SchoolFlags.IsBackground); // not focal
        RunResults r = CollectOne(store);

        Assert.Equal(new[] { 100.0, 10.0, 0.0, 4.0, 8.0, 2.0 }, r.Extra("abundance_by_age"));
        Assert.Equal(new[] { 100.0 * 0.001, 10.0 * 0.5, 0.0, 4.0 * 0.5, 8.0 * 0.5, 2.0 * 0.5 }, r.Extra("biomass_by_age"));
        Assert.Equal(new[] { 100.0, 10.0, 4.0, 2.0, 0.0, 8.0 }, r.Extra("abundance_by_size"));
        Assert.Equal(new[] { 100.0 * 0.001, 10.0 * 0.5, 4.0 * 0.5, 2.0 * 0.5, 0.0, 8.0 * 0.5 }, r.Extra("biomass_by_size"));

        // The distributions have no cutoff: species 0's egg school is in them, and not in the
        // abundance output (cutoff 0.5 yr).
        Assert.Equal(114.0, r.Extra("abundance_by_age")[..4].Sum());
        Assert.Equal(10.0 + 4.0 + 0.0, r.Abundance[0]);
    }

    [Fact]
    public void SsbNeedsLengthAgeAndSurvivorsAndYieldInNumbersIsTheFishingDeaths()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 7.0, 20.0, 48, weight: 2.0, fishing: 3.0); // mature, at both bounds
        Add(store, 0, 9.0, 30.0, 47, fishing: 1.5); // too young
        Add(store, 0, 9.0, 19.9, 96); // too short
        Add(store, 0, 0.0, 40.0, 96, weight: 5.0, fishing: 0.25); // mature, no survivors
        Add(store, 1, 11.0, 12.0, 0, weight: 0.1); // maturity age 0: by length alone
        RunResults r = CollectOne(store);
        Assert.Equal(new[] { 7.0 * 2.0, 11.0 * 0.1 }, r.Extra("ssb"));
        Assert.Equal(new[] { 3.0 + 1.5 + 0.25, 0.0 }, r.Extra("yield_n"));
    }

    [Fact]
    public void TheWeightedMeansSkipUnfedEmptyAndYoungSchoolsAndASpeciesWithNoneIsAbsent()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 10.0, 8.0, 11, tl: 3.0); // species 0, below its cutoff age (11 / 24 < 0.5): in neither mean
        Add(store, 0, 10.0, 20.0, 12, weight: 0.5, tl: 3.5); // exactly at the cutoff (12 / 24 >= 0.5): counted
        Add(store, 0, 30.0, 40.0, 48, weight: 2.0, tl: 4.5);
        Add(store, 0, 5.0, 50.0, 48, weight: 1.0, tl: 0.0); // unfed: counts for the size, not the TL
        Add(store, 0, 0.0, 60.0, 48, tl: 5.0); // empty: in neither
        Add(store, 0, 1000.0, 0.0, 48, tl: 0.0); // no length (and unfed): in neither
        Add(store, 1, 0.0, 9.0, 30, tl: 2.5); // species 1 has no survivor at all
        RunResults r = CollectOne(store);

        double b1 = 10.0 * 0.5, b2 = 30.0 * 2.0;
        Assert.Equal(((b1 * 3.5) + (b2 * 4.5)) / (b1 + b2), r.Extra("mean_tl")[0]);
        Assert.Equal(((10.0 * 20.0) + (30.0 * 40.0) + (5.0 * 50.0)) / (10.0 + 30.0 + 5.0), r.Extra("mean_size")[0]);
        Assert.True(double.IsNaN(r.Extra("mean_tl")[1])); // absent, not 0
        Assert.True(double.IsNaN(r.Extra("mean_size")[1]));
    }

    [Fact]
    public void OnlyTheFamiliesThatAreOnAreCollected()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 7.0, 20.0, 48, weight: 2.0, fishing: 3.0);
        RunResults r = CollectOne(store, OutputFamilies.Stage | OutputFamilies.AbundanceByAge | OutputFamilies.Ssb);
        Assert.Equal(["abundance_by_stage", "mortality_by_cause_stage", "mortality_rate_by_cause_stage", "abundance_by_age", "ssb"], r.Specs.Select(s => s.Name));
        Assert.Equal(new[] { 0.0, 0.0, 7.0, 0.0, 0.0, 0.0 }, r.Extra("abundance_by_age"));
        Assert.Equal(new[] { 14.0, 0.0 }, r.Extra("ssb"));
        Assert.Empty(r.Extra("biomass_by_age"));
        Assert.Empty(r.Extra("mean_tl"));
        Assert.Empty(r.Extra("yield_n"));
    }

    [Fact]
    public void EveryFamilyHasItsShapeAndWindowRuleInRecordOrder()
    {
        IReadOnlyList<FamilySpec> specs = Layout().Specs();
        Assert.Equal(OutputLayout.AllNames.Where(n => n != "diet_by_species" && !JavaNames.Contains(n)), specs.Select(s => s.Name));
        Assert.Equal(new[] { 6 }, specs.Single(s => s.Name == "biomass_by_age").Shape);
        Assert.Equal(AgeOffsets, specs.Single(s => s.Name == "abundance_by_age").Blocks); // one block per species
        Assert.Equal(new[] { NFocal, 3 }, specs.Single(s => s.Name == "biomass_by_size").Shape);
        Assert.Equal(new[] { 0, 3, 6 }, specs.Single(s => s.Name == "abundance_by_size").Blocks);
        Assert.Equal(new[] { 0, NFocal }, specs.Single(s => s.Name == "ssb").Blocks); // the record is one array
        Assert.Equal(WindowFold.Mean, specs.Single(s => s.Name == "ssb").Fold);
        Assert.Equal(WindowFold.Sum, specs.Single(s => s.Name == "yield_n").Fold);
        Assert.Equal(WindowFold.MeanOfPresent, specs.Single(s => s.Name == "mean_tl").Fold);
        Assert.Equal(WindowFold.MeanOfPresent, specs.Single(s => s.Name == "mean_size").Fold);
        Assert.Equal(WindowFold.Mean, specs.Single(s => s.Name == "biomass_by_age").Fold);

        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.BiomassByAge).Specs());
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.AbundanceBySize).Specs());
    }

    private static double[] Vector(int k)
    {
        var v = new double[k];
        for (int i = 0; i < k; i++)
        {
            v[i] = (((i % 7) - 3) * Math.Pow(10.0, (i % 5) * 4)) + (1.0 / (i + 1));
        }

        return v;
    }

    [Fact]
    public void ABlockOfOneValueIsSummedPairwiseAndAWiderOneInOrder()
    {
        // 17 rows of [x, v, y, z]: the block [1, 2) is one value per row, like a species with
        // a single class; NumPy sums such an array pairwise (-2.999900019999e+16 for this
        // vector, against -2.9999000199989996e+16 left to right: OutputsTests).
        double[] v = Vector(17);
        var rows = new double[17 * 4];
        for (int t = 0; t < 17; t++)
        {
            rows[(t * 4) + 0] = 1.0;
            rows[(t * 4) + 1] = v[t];
            rows[(t * 4) + 2] = v[t];
            rows[(t * 4) + 3] = 2.0;
        }

        double[] into = new double[4], scratch = new double[17];
        OutputCollector.FoldBlock(rows, 17, 4, 1, 1, mean: false, scratch, into);
        Assert.Equal(-2.999900019999e+16, into[1]);
        OutputCollector.FoldBlock(rows, 17, 4, 2, 2, mean: false, scratch, into);
        Assert.Equal(-2.9999000199989996e+16, into[2]); // two values per row: added in order
        Assert.Equal(34.0, into[3]);
        Assert.Equal(0.0, into[0]); // outside both blocks: untouched

        OutputCollector.FoldBlock(rows, 17, 4, 1, 1, mean: true, scratch, into);
        Assert.Equal(-2.999900019999e+16 / 17, into[1]);
        OutputCollector.FoldBlock(rows, 1, 4, 1, 1, mean: true, scratch, into);
        Assert.Equal(v[0], into[1]); // one step: passes through
    }

    [Fact]
    public void AMeanOfPresentAveragesTheStepsThatHaveTheValue()
    {
        // Three values per row over five steps: always present; present in steps 1 and 3; never.
        double nan = double.NaN;
        double[] rows =
        [
            1.0, nan, nan,
            2.0, 10.0, nan,
            4.0, nan, nan,
            8.0, 30.0, nan,
            16.0, nan, nan,
        ];
        double[] into = new double[3], scratch = new double[5];
        OutputCollector.FoldPresent(rows, 5, 3, scratch, into);
        Assert.Equal((1.0 + 2.0 + 4.0 + 8.0 + 16.0) / 5, into[0]);
        Assert.Equal((10.0 + 30.0) / 2, into[1]); // over the two steps that have it, not five
        Assert.True(double.IsNaN(into[2]));

        OutputCollector.FoldPresent(rows, 1, 3, scratch, into); // one step: passes through
        Assert.Equal(1.0, into[0]);
        Assert.True(double.IsNaN(into[1]));

        // np.mean of a 1-D array is its PAIRWISE sum over the count: 17 present values among 20
        // steps (NumPy: -2.999900019999e+16 for this vector; left to right ends ...89996e+16).
        double[] v = Vector(17);
        var column = new double[20];
        Array.Fill(column, nan);
        for (int i = 0; i < 17; i++)
        {
            column[i + (i >= 5 ? 3 : 0)] = v[i]; // absent in steps 5, 6 and 7
        }

        double[] one = new double[1], wide = new double[20];
        OutputCollector.FoldPresent(column, 20, 1, wide, one);
        Assert.Equal(-2.999900019999e+16 / 17, one[0]);
    }

    [Fact]
    public void AWindowFoldsEachFamilyByItsRule()
    {
        var c = new OutputCollector(Layout(), 2);
        var r = new RunResults(1, Layout());
        double[] abundance = [10.0, 30.0];
        double[] tl = [3.0, 0.0]; // the second step has no fed school: mean TL is absent there
        for (int t = 0; t < 2; t++)
        {
            var store = new SchoolStore(4);
            Add(store, 1, abundance[t], 12.0, 30, weight: 0.5, tl: tl[t], fishing: 1.0 + t);
            c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
            c.Collect(Cutoff, Ndt, MaturitySize, MaturityAgeDt, store, [], []);
        }

        c.Flush(r, 0, 1);
        Assert.Equal((10.0 + 30.0) / 2, r.Extra("abundance_by_age")[4 + 1]); // a mean
        Assert.Equal(((10.0 * 0.5) + (30.0 * 0.5)) / 2, r.Extra("ssb")[1]); // a mean
        Assert.Equal(1.0 + 2.0, r.Extra("yield_n")[1]); // a sum
        Assert.Equal(3.0, r.Extra("mean_tl")[1]); // the one step that has it
        Assert.Equal(12.0, r.Extra("mean_size")[1]);
        Assert.True(double.IsNaN(r.Extra("mean_tl")[0])); // species 0 never present
    }

    [Fact]
    public void SsbWithoutTheMaturityThresholdsIsRefused()
    {
        var store = new SchoolStore(4);
        Add(store, 0, 7.0, 20.0, 48);
        var c = new OutputCollector(Layout(OutputFamilies.Stage | OutputFamilies.Ssb), 1);
        c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        Assert.Throws<InvalidOperationException>(() => c.Collect(Cutoff, Ndt, store, [], []));
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig OutputGroup(long families, int[]? offsets, double[]? edges)
    {
        var c = new SimulationConfig(1);
        c.SetArray("output.cutoff_age", DType.F8, Pin(new double[NTotal]), NTotal);
        c.SetInt("output.record_frequency", 24);
        c.SetInt("output.step0_include", 0);
        c.SetInt("output.families", families);
        if (offsets is not null)
        {
            c.SetArray("output.age_bin_offsets", DType.I4, Pin(offsets), offsets.Length);
        }

        if (edges is not null)
        {
            c.SetArray("output.size_edges", DType.F8, Pin(edges), edges.Length);
        }

        return c;
    }

    private static (OutputParameters Output, IReadOnlyList<string> Problems) Read(SimulationConfig c)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Output(r, NTotal, NFocal, feature: true, nResources: 0), r.Problems);
    }

    [Fact]
    public void TheBinningTablesAreReadWithTheirFamilies()
    {
        (OutputParameters o, IReadOnlyList<string> problems) = Read(OutputGroup((long)All, AgeOffsets, Edges));
        Assert.Empty(problems);
        Assert.Equal(All, o.Families);
        Assert.Equal(AgeOffsets, o.AgeBinOffsets);
        Assert.Equal(Edges, o.SizeEdges);
        Assert.Equal(OutputLayout.AllNames.Where(n => n != "diet_by_species" && !JavaNames.Contains(n)), o.Layout(NTotal, NFocal).Specs().Select(s => s.Name));
    }

    [Fact]
    public void ATableWithoutItsFamilyOrAFamilyWithoutItsTableIsAProblem()
    {
        long stage = (long)OutputFamilies.Stage;
        (_, IReadOnlyList<string> extra) = Read(OutputGroup(stage, AgeOffsets, Edges));
        Assert.Contains(extra, m => m.Contains("output.age_bin_offsets: present, but output.families has no by-age family", StringComparison.Ordinal));
        Assert.Contains(extra, m => m.Contains("output.size_edges: present, but output.families has no by-size family", StringComparison.Ordinal));

        (OutputParameters o, IReadOnlyList<string> missing) = Read(OutputGroup((long)All, null, null));
        Assert.Contains(missing, m => m.Contains("output.age_bin_offsets: missing", StringComparison.Ordinal));
        Assert.Contains(missing, m => m.Contains("output.size_edges: missing", StringComparison.Ordinal));
        Assert.Equal(OutputFamilies.None, o.Families & (OutputFamilies.ByAge | OutputFamilies.BySize)); // a layout that can still be built
        Assert.NotEmpty(o.Layout(NTotal, NFocal).Specs());
    }

    [Fact]
    public void BrokenTablesAreNamed()
    {
        (_, IReadOnlyList<string> offsets) = Read(OutputGroup((long)All, [0, 4, 4], Edges));
        Assert.Contains(offsets, m => m.Contains("output.age_bin_offsets: must start at 0 and increase", StringComparison.Ordinal));

        (_, IReadOnlyList<string> from1) = Read(OutputGroup((long)All, [1, 4, 6], Edges));
        Assert.Contains(from1, m => m.Contains("output.age_bin_offsets: must start at 0 and increase", StringComparison.Ordinal));

        (_, IReadOnlyList<string> edges) = Read(OutputGroup((long)All, AgeOffsets, [2.0, 6.0, 6.0]));
        Assert.Contains(edges, m => m.Contains("output.size_edges: needs at least one edge, strictly increasing", StringComparison.Ordinal));
    }

    [Fact]
    public void EveryFamilysBufferIsBoundByNameAndShapeAndWrittenRecordByRecord()
    {
        const int n = 2;
        OutputLayout layout = Layout();
        var c = new SimulationConfig(1);
        c.SetArray("result.record_step", DType.I4, Pin(new int[n]), n);
        c.SetArray("result.window_steps", DType.I4, Pin(new int[n]), n);
        c.SetArray("result.biomass", DType.F8, Pin(new double[n * NTotal]), n, NTotal);
        c.SetArray("result.abundance", DType.F8, Pin(new double[n * NTotal]), n, NTotal);
        c.SetArray("result.mortality", DType.F8, Pin(new double[n * NFocal * RunResults.MortalityCauses]), n, NFocal, RunResults.MortalityCauses);
        c.SetArray("result.yield", DType.F8, Pin(new double[n * NFocal]), n, NFocal);
        var buffers = new Dictionary<string, double[]>();
        var r = new RunResults(n, layout);
        int seed = 0;
        foreach (FamilySpec spec in layout.Specs())
        {
            buffers[spec.Name] = new double[n * spec.Width];
            long[] d = [.. spec.Shape.Select(v => (long)v), -1, -1];
            c.SetArray($"result.{spec.Name}", DType.F8, Pin(buffers[spec.Name]), n, d[0], d[1], d[2]);
            double[] values = r.Extra(spec.Name);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (1000.0 * ++seed) + i; // every family its own numbers
            }
        }

        ResultBuffers b = ResultBuffers.From(c, n, layout)!;
        b.Write(r, 1); // only the second record
        foreach (FamilySpec spec in layout.Specs())
        {
            Assert.All(buffers[spec.Name][..spec.Width], v => Assert.Equal(0.0, v));
            Assert.Equal(r.Extra(spec.Name)[spec.Width..], buffers[spec.Name][spec.Width..]);
        }

        // The same buffers against a layout without the by-age families: two too many.
        OutputLayout fewer = Layout(All & ~OutputFamilies.ByAge);
        BundleValidationException extra = Assert.Throws<BundleValidationException>(() => ResultBuffers.From(c, n, fewer));
        Assert.Equal(2, extra.Problems.Count);
        Assert.Contains(extra.Problems, m => m.Contains("result.biomass_by_age: the bundle's output.families does not include it", StringComparison.Ordinal));
    }

    [Fact]
    public void JavaClassCountsTheThresholdsAtOrBelowAndTheJavaFamiliesNeedTheirTables()
    {
        double[] t = [1.0, 2.0, 3.0];
        Assert.Equal(0, OutputCollector.JavaClass(t, 0.5));
        Assert.Equal(1, OutputCollector.JavaClass(t, 1.0)); // on a threshold: the class it opens
        Assert.Equal(1, OutputCollector.JavaClass(t, 1.5));
        Assert.Equal(3, OutputCollector.JavaClass(t, 9.0)); // beyond the last: the last class
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.BiomassByTl).Specs());
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.MeanSizeByAge).Specs());
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.MeanTlBySize).Specs());
        FamilySpec tl = new OutputLayout(NTotal, NFocal, OutputFamilies.BiomassByTl) { TlThresholds = t }.Specs().Single();
        Assert.Equal(new[] { NFocal, 4 }, tl.Shape); // thresholds + 1 classes per focal species
        Assert.Equal(WindowFold.Mean, tl.Fold);
    }

    [Fact]
    public void ClassRatesSplitZByCauseShareAndTheSizeFamiliesShareOneThresholdTable()
    {
        // one species, two classes; class 0: 80 survivors, 10 predation + 10 fishing; class 1 untouched
        int n = RunResults.MortalityCauses;
        double[] abundance = [80.0, 50.0];
        double[] deaths = new double[n * 2];
        deaths[0 * 2] = 10.0;
        deaths[3 * 2] = 10.0;
        double[] rates = new double[n * 2];
        OutputCollector.ClassRates(abundance, deaths, rates, 1, 2);
        double z = -Math.Log(80.0 / 100.0);
        Assert.Equal(z / 2, rates[0 * 2], 15);
        Assert.Equal(z / 2, rates[3 * 2], 15);
        Assert.All(Enumerable.Range(0, n), c => Assert.Equal(0.0, rates[(c * 2) + 1]));
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.YieldBySize).Specs());
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.MortalityBySize) { JavaSizeThresholds = [10.0] }.Specs());
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.YieldByFishery).Specs());
        var specs = new OutputLayout(NTotal, NFocal, OutputFamilies.Stage | OutputFamilies.CatchFamilies) { JavaSizeThresholds = [10.0, 20.0], FisheryCount = 3 }.Specs();
        Assert.Equal(new[] { NFocal, 3 }, specs.Single(sp => sp.Name == "yield_by_fishery_landed").Shape);
        Assert.All(specs.Where(sp => JavaNames.Contains(sp.Name)), sp => Assert.Equal(WindowFold.Sum, sp.Fold));
        Assert.Equal(new[] { NFocal, n, 3 }, specs.Single(sp => sp.Name == "mortality_rate_by_cause_size").Shape);
    }
}
