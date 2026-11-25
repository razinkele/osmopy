using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.6c: the diet family (<c>aggregate_diet_by_species</c> and its window).
/// The kernel's per-school rows are checked against the Python per-cell oracle on real states
/// (tests/test_csharp_outputs_diet.py); these pin the store's rows, the aggregation, the window
/// rule and the bundle contract on cases small enough to read.</summary>
public sealed class DietOutputsTests : IDisposable
{
    // Two focal species and one background species; prey columns: the two focal species, the
    // background species and one resource (the reference's layout: species, then resources).
    private const int NFocal = 2;
    private const int NTotal = 3;
    private const int Cols = 4;
    private const OutputFamilies Families = OutputFamilies.Stage | OutputFamilies.Diet;

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

    private static OutputLayout Layout() => new(NTotal, NFocal, Families) { DietColumns = Cols };

    private static SchoolStore Store(params int[] species)
    {
        var store = new SchoolStore(4);
        store.EnableDiet(Cols);
        foreach (int sp in species)
        {
            store.Append(
                new SchoolCore { SpeciesId = sp, Abundance = 10.0, Length = 5.0, Weight = 0.5, Flags = sp >= NFocal ? SchoolFlags.IsBackground : 0 },
                new SchoolLife { CellId = 0 });
        }

        store.ClearDiet();
        return store;
    }

    private static void Eat(SchoolStore store, int school, params double[] row) =>
        row.CopyTo(store.DietArray.AsSpan(school * Cols, Cols));

    [Fact]
    public void TheStoresDietRowsFollowItsSchoolsAndAreClearedPerStep()
    {
        var off = new SchoolStore(4);
        off.Append(new SchoolCore { SpeciesId = 0 }, new SchoolLife { CellId = 0 });
        off.ClearDiet(); // not tracked: nothing to do, nothing allocated
        Assert.Equal(0, off.DietCols);
        Assert.True(off.Diet.IsEmpty);

        var store = new SchoolStore(4, initialCapacity: 2);
        store.EnableDiet(Cols);
        Assert.True(store.Diet.IsEmpty); // no school yet
        for (int i = 0; i < 5; i++)
        {
            store.Append(new SchoolCore { SpeciesId = 0 }, new SchoolLife { CellId = 0 }); // grows past the capacity
        }

        store.ClearDiet();
        Assert.Equal(5 * Cols, store.Diet.Length);
        Eat(store, 4, 1.0, 2.0, 3.0, 4.0);
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, store.Diet.Slice(4 * Cols, Cols).ToArray());
        store.ClearDiet();
        Assert.All(store.Diet.ToArray(), v => Assert.Equal(0.0, v));
        Assert.Throws<ArgumentOutOfRangeException>(() => store.EnableDiet(0));
    }

    [Fact]
    public void TheDietIsAddedPerPredatorSpeciesInStoreOrderBackgroundPredatorsIncluded()
    {
        // Schools of species 0, 2 (background), 0, 1, 0. Species 0's three rows are chosen so
        // that the order they are added in shows in the last bit.
        SchoolStore store = Store(0, 2, 0, 1, 0);
        Eat(store, 0, 1e16, 0.0, 1.0, 0.0);
        Eat(store, 1, 0.0, 7.0, 0.0, 0.5);
        Eat(store, 2, 1.0, 0.0, 1e-3, 0.0);
        Eat(store, 3, 0.0, 0.0, 0.0, 2.0);
        Eat(store, 4, 1.0, 0.0, 0.25, 0.0);

        var c = new OutputCollector(Layout(), 1);
        c.SnapshotDiet(store);
        c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        c.Collect(Cutoff, 24, store, [], []);
        var r = new RunResults(1, Layout());
        c.Flush(r, 0, 0);

        double[] diet = r.Extra("diet_by_species");
        Assert.Equal(NTotal * Cols, diet.Length);
        Assert.Equal((1e16 + 1.0) + 1.0, diet[0]); // store order: 1e16, then 1, then 1
        Assert.NotEqual(1e16 + (1.0 + 1.0), diet[0]);
        Assert.Equal(new[] { 0.0, (1.0 + 1e-3) + 0.25, 0.0 }, diet[1..4]);
        Assert.Equal(new[] { 0.0, 0.0, 0.0, 2.0 }, diet[4..8]); // species 1
        Assert.Equal(new[] { 0.0, 7.0, 0.0, 0.5 }, diet[8..12]); // the background predator's row
    }

    [Fact]
    public void AWindowSumsItsStepsDiets()
    {
        var c = new OutputCollector(Layout(), 3);
        var r = new RunResults(1, Layout());
        double[] eaten = [1e16, 1.0, 1.0];
        foreach (double e in eaten)
        {
            SchoolStore store = Store(1);
            Eat(store, 0, 0.0, e, 0.0, 2.0);
            c.SnapshotDiet(store);
            c.SnapshotStages(MaturitySize, MaturityAgeDt, store);
            c.Collect(Cutoff, 24, store, [], []);
        }

        c.Flush(r, 0, 2);
        double[] diet = r.Extra("diet_by_species");
        Assert.Equal((1e16 + 1.0) + 1.0, diet[Cols + 1]); // a sum, step by step
        Assert.Equal(6.0, diet[Cols + 3]);
        Assert.Equal(0.0, diet[0]);
    }

    [Fact]
    public void TheDietIsTakenOncePerStepBeforeTheCollectionAndIsNothingWhenOff()
    {
        SchoolStore store = Store(0);
        var on = new OutputCollector(Layout(), 1);
        on.SnapshotStages(MaturitySize, MaturityAgeDt, store);
        InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => on.Collect(Cutoff, 24, store, [], []));
        Assert.Contains("the diet was not taken", missing.Message, StringComparison.Ordinal);
        on.SnapshotDiet(store);
        Assert.Throws<InvalidOperationException>(() => on.SnapshotDiet(store));

        var untracked = new SchoolStore(4);
        untracked.Append(new SchoolCore { SpeciesId = 0 }, new SchoolLife { CellId = 0 });
        var other = new OutputCollector(Layout(), 1);
        InvalidOperationException columns = Assert.Throws<InvalidOperationException>(() => other.SnapshotDiet(untracked));
        Assert.Contains("0 diet columns, the layout has 4", columns.Message, StringComparison.Ordinal);

        var off = new OutputCollector(NTotal, NFocal, 1);
        off.SnapshotDiet(untracked); // no family: nothing to do
        off.Collect(Cutoff, 24, untracked, [], []);
        var r = new RunResults(1, NTotal, NFocal);
        off.Flush(r, 0, 0);
        Assert.Empty(r.Extra("diet_by_species"));
    }

    [Fact]
    public void TheDietIsTheLastFamilyWithARowPerSpeciesAndASum()
    {
        OutputLayout layout = new(NTotal, NFocal, OutputFamilies.Known)
        {
            AgeBinOffsets = [0, 2, 4],
            SizeEdges = [0.0, 5.0],
            DietColumns = Cols,
            TlThresholds = [2.0, 3.0],
            AgeThresholds = [1.0],
            JavaSizeThresholds = [10.0, 20.0],
            FisheryCount = 2,
        };
        IReadOnlyList<FamilySpec> specs = layout.Specs();
        Assert.Equal(OutputLayout.AllNames, specs.Select(s => s.Name));
        FamilySpec diet = specs[^1];
        Assert.Equal("diet_by_species", diet.Name);
        Assert.Equal(new[] { NTotal, Cols }, diet.Shape); // background predators have rows
        Assert.Equal(WindowFold.Sum, diet.Fold);
        Assert.Throws<ArgumentException>(() => new OutputLayout(NTotal, NFocal, OutputFamilies.Diet).Specs());
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private (OutputParameters Output, IReadOnlyList<string> Problems) Read(long families, long? columns)
    {
        var c = new SimulationConfig(1);
        c.SetArray("output.cutoff_age", DType.F8, Pin(new double[NTotal]), NTotal);
        c.SetInt("output.record_frequency", 24);
        c.SetInt("output.step0_include", 0);
        c.SetInt("output.families", families);
        if (columns is long n)
        {
            c.SetInt("output.diet_columns", n);
        }

        var r = new BundleReader(c, 1);
        return (ModelReader.Output(r, NTotal, NFocal, feature: true, nResources: Cols - NTotal), r.Problems);
    }

    [Fact]
    public void TheColumnCountTravelsWithTheDietFamily()
    {
        (OutputParameters on, IReadOnlyList<string> none) = Read((long)Families, Cols);
        Assert.Empty(none);
        Assert.Equal(Families, on.Families);
        Assert.Equal(Cols, on.DietColumns);
        Assert.Equal(Cols, on.Layout(NTotal, NFocal).DietColumns);

        (OutputParameters dropped, IReadOnlyList<string> missing) = Read((long)Families, null);
        Assert.Contains(missing, m => m.Contains("output.diet_columns: missing", StringComparison.Ordinal));
        Assert.Equal(OutputFamilies.Stage, dropped.Families); // a layout that can still be built

        // Any count but species + resources is refused: with fewer columns the kernel would book
        // a resource under a species' label, with more it would leave the labels short.
        foreach (int wrong in new[] { NTotal - 1, Cols - 1, Cols + 1 })
        {
            (OutputParameters refused, IReadOnlyList<string> why) = Read((long)Families, wrong);
            Assert.Contains(why, m => m.Contains($"output.diet_columns: {wrong}, but the prey columns are the 3 species (focal, then background) and the 1 resources", StringComparison.Ordinal));
            Assert.Equal(OutputFamilies.Stage, refused.Families);
        }

        (_, IReadOnlyList<string> extra) = Read((long)OutputFamilies.Stage, Cols);
        Assert.Contains(extra, m => m.Contains("output.diet_columns: present, but output.families has no diet family", StringComparison.Ordinal));
    }
}
