using System.Runtime.InteropServices;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.5: bottom oxygen scaling a resource's carrying capacity (feature
/// <c>oxygen_benthos</c>). The bit-exact check against <c>ResourceState.update</c> on real
/// configs is on the Python side (tests/test_csharp_prep.py); these pin the arithmetic and the
/// bundle contract.</summary>
public sealed class ResourceOxygenTests : IDisposable
{
    private const int NCells = 3;
    private const int NSteps = 3;

    // Values whose triple product depends on the association order, so the order is pinned.
    private static readonly double[] Forcing =
    [
        0.1, 0.2, 0.3, 1.1, 1.2, 1.3, // resource 0: frames 0 and 1
        0.7, 0.9, 1.7, 2.3, 3.1, 4.7, // resource 1: frames 0 and 1
    ];

    private static readonly int[] FrameByStep = [0, 0, 1, 1, 0, 1]; // [step * 2 + resource]
    private static readonly double[] AccessByStep = [0.8, 0.3, 0.8, 0.7, 0.6, 0.1];
    private static readonly double[] Factor = [0.11, 0.37, 0.93, 0.53, 0.19, 0.71]; // 2 frames x 3 cells
    private static readonly int[] OxygenFrameByStep = [0, 1, 1];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static ResourceParameters Parameters(bool oxygen) => new()
    {
        N = 2,
        NFrames = 2,
        Forcing = Forcing,
        FrameByStep = FrameByStep,
        AccessByStep = AccessByStep,
        OxygenCoupled = oxygen ? [0, 1] : [],
        OxygenFactor = oxygen ? Factor : [],
        OxygenFrameByStep = oxygen ? OxygenFrameByStep : [],
    };

    private static double Base(int r, int step, int c) =>
        Forcing[((((r * 2) + FrameByStep[(step * 2) + r]) * NCells) + c)] * AccessByStep[(step * 2) + r];

    [Fact]
    public void TheCoupledResourceIsScaledByItsStepsFactorAndTheOtherIsNot()
    {
        double[] k = new double[2 * NCells];
        bool orderMatters = false;
        for (int step = 0; step < NSteps; step++)
        {
            StepPreparation.UpdateResources(Parameters(oxygen: true), NCells, step, k);
            for (int c = 0; c < NCells; c++)
            {
                double f = Factor[(OxygenFrameByStep[step] * NCells) + c];
                Assert.Equal(Base(0, step, c), k[(c * 2) + 0]); // not coupled: exactly the base
                Assert.Equal(Base(1, step, c) * f, k[(c * 2) + 1]); // (forcing * access) * factor
                double forcing = Forcing[((((1 * 2) + FrameByStep[(step * 2) + 1]) * NCells) + c)];
                orderMatters |= forcing * (AccessByStep[(step * 2) + 1] * f) != k[(c * 2) + 1];
            }
        }

        Assert.True(orderMatters, "the fixture must distinguish (a*b)*c from a*(b*c)");
    }

    [Fact]
    public void WithoutTheFeatureEveryResourceIsTheBase()
    {
        double[] k = new double[2 * NCells];
        for (int step = 0; step < NSteps; step++)
        {
            StepPreparation.UpdateResources(Parameters(oxygen: false), NCells, step, k);
            for (int c = 0; c < NCells; c++)
            {
                Assert.Equal(Base(0, step, c), k[(c * 2) + 0]);
                Assert.Equal(Base(1, step, c), k[(c * 2) + 1]);
            }
        }
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Bundle(bool withOxygen, int[]? oxygenFrames = null)
    {
        var c = new SimulationConfig(1);
        c.SetInt("resources.n", 2);
        c.SetArray("resources.size_min", DType.F8, Pin(new[] { 0.1, 0.2 }), 2);
        c.SetArray("resources.size_max", DType.F8, Pin(new[] { 1.0, 2.0 }), 2);
        c.SetArray("resources.trophic_level", DType.F8, Pin(new[] { 1.0, 2.0 }), 2);
        c.SetArray("resources.access_rows", DType.I4, Pin(new[] { 0, 1 }), 2);
        c.SetArray("resources.forcing", DType.F8, Pin(Forcing), 2, 2, NCells);
        c.SetArray("resources.frame_by_step", DType.I4, Pin(FrameByStep), NSteps, 2);
        c.SetArray("resources.accessibility_by_step", DType.F8, Pin(AccessByStep), NSteps, 2);
        if (withOxygen)
        {
            c.SetArray("resources.oxygen.coupled", DType.U1, Pin(new byte[] { 0, 1 }), 2);
            c.SetArray("resources.oxygen.factor", DType.F8, Pin(Factor), 2, NCells);
            c.SetArray("resources.oxygen.frame_by_step", DType.I4, Pin(oxygenFrames ?? OxygenFrameByStep), NSteps);
        }

        return c;
    }

    private static (ResourceParameters Parameters, IReadOnlyList<string> Problems) Read(SimulationConfig c, bool feature)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Resources(r, NSteps, NCells, feature), r.Problems);
    }

    [Fact]
    public void TheTableIsReadWhenTheFeatureIsDeclared()
    {
        (ResourceParameters p, IReadOnlyList<string> problems) = Read(Bundle(withOxygen: true), feature: true);
        Assert.Empty(problems);
        Assert.Equal(new byte[] { 0, 1 }, p.OxygenCoupled);
        Assert.Equal(Factor, p.OxygenFactor);
        Assert.Equal(OxygenFrameByStep, p.OxygenFrameByStep);
    }

    [Fact]
    public void ADeclaredFeatureWithoutItsTableIsThreeProblems()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(withOxygen: false), feature: true);
        Assert.Equal(3, problems.Count);
        Assert.All(problems, m => Assert.Contains("resources.oxygen.", m, StringComparison.Ordinal));
        Assert.All(problems, m => Assert.Contains("missing", m, StringComparison.Ordinal));
    }

    [Fact]
    public void ATableWithoutTheFeatureIsRefusedNotIgnored()
    {
        (ResourceParameters p, IReadOnlyList<string> problems) = Read(Bundle(withOxygen: true), feature: false);
        Assert.Equal(3, problems.Count);
        Assert.All(problems, m => Assert.Contains("does not declare feature oxygen_benthos", m, StringComparison.Ordinal));
        Assert.Empty(p.OxygenCoupled); // and it is not applied
    }

    [Fact]
    public void AnOxygenFrameOutsideTheTableIsAProblem()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(withOxygen: true, oxygenFrames: [0, 2, 1]), feature: true);
        string m = Assert.Single(problems);
        Assert.Contains("resources.oxygen.frame_by_step: values must be in [0, 2)", m, StringComparison.Ordinal);
    }

    [Fact]
    public void TheCatalogueMarksTheFeatureImplemented()
    {
        Assert.True(FeatureCatalog.Named("oxygen_benthos").Implemented);
        Assert.Throws<InvalidOperationException>(() => FeatureCatalog.Named("no_such_feature"));
    }
}
