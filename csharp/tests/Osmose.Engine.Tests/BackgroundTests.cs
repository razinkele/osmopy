using System.Runtime.InteropServices;
using Osmose.Engine.Schools;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.1: background species (feature <c>background_species</c>). The checks
/// against <c>BackgroundState.get_schools</c> and <c>_collect_background_outputs</c> on Baltic are
/// on the Python side (tests/test_csharp_background.py); these pin the order, the arithmetic,
/// the removal and the bundle contract on a case small enough to read.</summary>
public sealed class BackgroundTests : IDisposable
{
    // Two focal species; two background species: the first with two classes, the second with one.
    // Three ocean cells of a 2 x 3 grid, listed out of cell order on purpose.
    private const int NFocal = 2;
    private const int NCells = 6;
    private const int NSteps = 2;

    private static readonly int[] Ocean = [4, 0, 2];
    private static readonly double[] BaseFrames =
    [
        0.3, 0.7, 1.1, // frame 0
        0.9, 0.0, 0.5, // frame 1: no biomass on the second ocean cell
        2.0, 2.0, 2.0, // frame 2: the second species, uniform
    ];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static BackgroundParameters Parameters() => new()
    {
        NSpecies = 2,
        NClasses = 3,
        ClassOffsets = [0, 2, 3],
        ClassLength = [110.0, 170.0, 80.0],
        ClassWeight = [0.053, 0.196, 0.0],
        ClassTrophicLevel = [4.5, 4.8, 4.2],
        ClassAgeDt = [48, 144, 24],
        ProportionByStep = [0.35, 0.65, 1.0, 0.1, 0.9, 1.0], // [step * 3 + class]
        OceanCells = Ocean,
        Base = BaseFrames,
        FrameByStep = [0, 2, 1, 2], // [step * 2 + species]
    };

    [Fact]
    public void InjectAppendsSpeciesThenClassThenOceanCell()
    {
        BackgroundParameters b = Parameters();
        var store = new SchoolStore(NCells);
        store.Append(new SchoolCore { SpeciesId = 0, Abundance = 5.0 }, new SchoolLife { CellId = 1 });
        ulong first = Background.Inject(b, NFocal, store, step: 0);

        Assert.Equal(1UL, first);
        Assert.Equal(1 + 9, store.Count);
        Assert.Equal(9, Background.SchoolsPerStep(b));
        (int Species, int Class, int Ocean)[] order =
        [
            (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (3, 2, 0), (3, 2, 1), (3, 2, 2),
        ];
        for (int j = 0; j < order.Length; j++)
        {
            SchoolCore c = store.Core[1 + j];
            (int species, int cls, int ocean) = order[j];
            Assert.Equal(species, c.SpeciesId);
            Assert.Equal(b.ClassAgeDt[cls], c.AgeDt);
            Assert.Equal(b.ClassLength[cls], c.Length);
            Assert.Equal(b.ClassWeight[cls], c.Weight);
            Assert.Equal(b.ClassTrophicLevel[cls], c.TrophicLevel);
            Assert.Equal(-1, c.FirstFeedingAgeDt); // eligible to feed at any age
            Assert.Equal(SchoolFlags.IsBackground, c.Flags);
            Assert.Equal(Ocean[ocean], store.Life[1 + j].CellId);
            Assert.Equal(first + (ulong)j, store.Life[1 + j].Id);
        }
    }

    [Fact]
    public void AbundanceIsBaseTimesProportionOverWeightAndZeroForAWeightlessClass()
    {
        BackgroundParameters b = Parameters();
        var store = new SchoolStore(NCells);
        Background.Inject(b, NFocal, store, step: 0);
        Assert.Equal(0.3 * 0.35 / 0.053, store.Core[0].Abundance); // (base * proportion) / weight
        Assert.NotEqual(0.3 * (0.35 / 0.053), store.Core[0].Abundance); // the association is part of it
        Assert.Equal(1.1 * 0.65 / 0.196, store.Core[5].Abundance);
        Assert.Equal(0.0, store.Core[6].Abundance); // weight 0: no individuals

        var later = new SchoolStore(NCells);
        Background.Inject(b, NFocal, later, step: 1); // frame 1 and step 1's proportions
        Assert.Equal(0.9 * 0.1 / 0.053, later.Core[0].Abundance);
        Assert.Equal(0.0, later.Core[1].Abundance); // no biomass on that cell
        Assert.Equal(0.5 * 0.9 / 0.196, later.Core[5].Abundance);
    }

    [Fact]
    public void TotalsAddInInjectionOrderAfterTheRegroupAndTheSchoolsAreRemoved()
    {
        BackgroundParameters b = Parameters() with { ClassWeight = [0.053, 0.196, 0.004] };
        var store = new SchoolStore(NCells);
        store.Append(new SchoolCore { SpeciesId = 0, Abundance = 5.0 }, new SchoolLife { CellId = 4 });
        store.Append(new SchoolCore { SpeciesId = 1, Abundance = 7.0 }, new SchoolLife { CellId = 0 });
        ulong first = Background.Inject(b, NFocal, store, step: 0);
        double[] injected = [.. Enumerable.Range(2, 9).Select(i => store.Core[i].Abundance)];
        double[] weights = [0.053, 0.053, 0.053, 0.196, 0.196, 0.196, 0.004, 0.004, 0.004];

        store.Regroup(); // by cell: 0, 2, 4, so no longer in injection order
        Assert.NotEqual(first, store.Life[1].Id);

        double[] scratch = new double[18], biomass = new double[4], abundance = new double[4];
        biomass[0] = abundance[0] = 123.0; // focal entries are left alone
        Background.CollectAndRemove(b, NFocal, store, first, scratch, biomass, abundance);

        double b2 = 0.0, a2 = 0.0, b3 = 0.0, a3 = 0.0;
        for (int j = 0; j < 6; j++)
        {
            b2 += injected[j] * weights[j];
            a2 += injected[j];
        }

        for (int j = 6; j < 9; j++)
        {
            b3 += injected[j] * weights[j];
            a3 += injected[j];
        }

        Assert.Equal(new[] { 123.0, 0.0, b2, b3 }, biomass);
        Assert.Equal(new[] { 123.0, 0.0, a2, a3 }, abundance);

        Assert.Equal(2, store.Count); // the two focal schools, in cell order
        Assert.Equal(new[] { 1, 0 }, new[] { store.Core[0].SpeciesId, store.Core[1].SpeciesId });
        Assert.All(store.Core.ToArray(), c => Assert.Equal(SchoolFlags.None, c.Flags & SchoolFlags.IsBackground));
    }

    [Fact]
    public void RemovalKeepsEachRemainingSchoolsOwnData()
    {
        var store = new SchoolStore(NCells);
        for (int i = 0; i < 8; i++)
        {
            bool background = i % 3 == 1;
            store.Append(
                new SchoolCore { SpeciesId = i, Abundance = 10.0 + i, Length = 100.0 + i, Flags = background ? SchoolFlags.IsBackground : 0 },
                new SchoolLife { CellId = i % NCells });
            store.Step[i].PredSuccess = 1000.0 + i;
            store.Ledger[i].DeadFishing = 2000.0 + i;
        }

        Assert.Equal(3, store.RemoveBackground()); // schools 1, 4 and 7
        Assert.Equal(5, store.Count);
        int[] kept = [0, 2, 3, 5, 6];
        for (int j = 0; j < kept.Length; j++)
        {
            int i = kept[j];
            Assert.Equal(i, store.Core[j].SpeciesId);
            Assert.Equal((ulong)i, store.Life[j].Id);
            Assert.Equal(1000.0 + i, store.Step[j].PredSuccess);
            Assert.Equal(2000.0 + i, store.Ledger[j].DeadFishing);
            Assert.Equal(100.0 + i, store.LengthLane[j]);
        }

        Assert.Equal(0, store.RemoveBackground()); // nothing left to remove
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Bundle(bool entries, int[]? frames = null, int[]? offsets = null)
    {
        var c = new SimulationConfig(1);
        if (!entries)
        {
            return c;
        }

        BackgroundParameters b = Parameters();
        c.SetArray("background.class.offsets", DType.I4, Pin(offsets ?? b.ClassOffsets), 3);
        c.SetArray("background.class.length", DType.F8, Pin(b.ClassLength), 3);
        c.SetArray("background.class.weight", DType.F8, Pin(b.ClassWeight), 3);
        c.SetArray("background.class.trophic_level", DType.F8, Pin(b.ClassTrophicLevel), 3);
        c.SetArray("background.class.age_dt", DType.I4, Pin(b.ClassAgeDt), 3);
        c.SetArray("background.proportion_by_step", DType.F8, Pin(b.ProportionByStep), NSteps, 3);
        c.SetArray("background.ocean_cells", DType.I4, Pin(Ocean), 3);
        c.SetArray("background.base", DType.F8, Pin(BaseFrames), 3, 3);
        c.SetArray("background.frame_by_step", DType.I4, Pin(frames ?? b.FrameByStep), NSteps, 2);
        return c;
    }

    private static (BackgroundParameters Parameters, IReadOnlyList<string> Problems) Read(SimulationConfig c, int nBackground, bool feature)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Background(r, NSteps, nBackground, NCells, feature), r.Problems);
    }

    [Fact]
    public void TheGroupIsReadWhenTheFeatureIsDeclared()
    {
        (BackgroundParameters p, IReadOnlyList<string> problems) = Read(Bundle(entries: true), 2, feature: true);
        Assert.Empty(problems);
        Assert.Equal(2, p.NSpecies);
        Assert.Equal(3, p.NClasses);
        Assert.Equal(Ocean, p.OceanCells);
        Assert.Equal(BaseFrames, p.Base);
        Assert.Equal(Parameters().FrameByStep, p.FrameByStep);
    }

    [Fact]
    public void WithoutTheFeatureBackgroundSpeciesOrTheirTablesAreRefused()
    {
        (BackgroundParameters p, IReadOnlyList<string> problems) = Read(Bundle(entries: true), 2, feature: false);
        Assert.Equal(10, problems.Count); // the species count, and nine entries
        Assert.Contains(problems, m => m.Contains("species.n_background: 2, but the bundle does not declare feature background_species", StringComparison.Ordinal));
        Assert.Equal(0, p.NSpecies); // and none is injected

        (_, IReadOnlyList<string> none) = Read(Bundle(entries: false), 0, feature: false);
        Assert.Empty(none); // a plain bundle
    }

    [Fact]
    public void ADeclaredFeatureNeedsItsSpeciesAndItsTables()
    {
        (_, IReadOnlyList<string> missing) = Read(Bundle(entries: false), 2, feature: true);
        Assert.Contains(missing, m => m.Contains("background.class.offsets: missing", StringComparison.Ordinal));
        Assert.Contains(missing, m => m.Contains("background.base: missing", StringComparison.Ordinal));

        (_, IReadOnlyList<string> none) = Read(Bundle(entries: true), 0, feature: true);
        Assert.Contains(none, m => m.Contains("background_species is declared, but species.n_background is 0", StringComparison.Ordinal));
    }

    [Fact]
    public void BrokenTablesAreNamed()
    {
        (_, IReadOnlyList<string> frames) = Read(Bundle(entries: true, frames: [0, 2, 3, 2]), 2, feature: true);
        Assert.Contains(frames, m => m.Contains("background.frame_by_step: values must be in [0, 3)", StringComparison.Ordinal));

        (_, IReadOnlyList<string> offsets) = Read(Bundle(entries: true, offsets: [0, 2, 2]), 2, feature: true);
        Assert.Contains(offsets, m => m.Contains("background.class.offsets: must start at 0 and increase", StringComparison.Ordinal));
    }
}
