using System.Runtime.InteropServices;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 3.1: depletable resources (feature <c>depletable_ltl</c>). The bit-exact
/// checks against <c>ResourceState.update</c> on real runs are on the Python side
/// (tests/test_csharp_depletable.py); these pin the arithmetic, that the update reads the buffer
/// it is given, and the bundle contract.</summary>
public sealed class ResourceDepletableTests : IDisposable
{
    private const int NCells = 3;
    private const int NSteps = 2;
    private const double Floor = 0.05;

    private static readonly double[] Forcing = [100.0, 40.0, 0.0, 8.0, 8.0, 8.0]; // 2 resources x 1 frame x 3 cells
    private static readonly int[] FrameByStep = [0, 0, 0, 0];
    private static readonly double[] AccessByStep = [1.0, 0.5, 1.0, 0.5];
    private static readonly double[] Rates = [1.0, 2.5];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static ResourceParameters Parameters(bool depletable) => new()
    {
        N = 2,
        NFrames = 1,
        Forcing = Forcing,
        FrameByStep = FrameByStep,
        AccessByStep = AccessByStep,
        Depletable = depletable,
        DepletableFloor = depletable ? Floor : 0.0,
        RegrowthRate = depletable ? Rates : [],
    };

    [Fact]
    public void RegrowIsOneLogisticStepFromTheFloorAtTheLeastCappedAtTheCapacity()
    {
        Assert.Equal(5.0 + (1.0 * 5.0 * (1.0 - (5.0 / 100.0))), StepPreparation.Regrow(0.0, 100.0, 1.0, Floor)); // from the floor
        Assert.Equal(5.0 + (1.0 * 5.0 * (1.0 - (5.0 / 100.0))), StepPreparation.Regrow(2.0, 100.0, 1.0, Floor)); // under the floor
        Assert.Equal(75.0, StepPreparation.Regrow(50.0, 100.0, 1.0, Floor));
        Assert.Equal(100.0, StepPreparation.Regrow(100.0, 100.0, 1.0, Floor)); // at the capacity: stays
        Assert.Equal(100.0, StepPreparation.Regrow(90.0, 100.0, 5.0, Floor)); // 135 before the cap
        Assert.Equal(0.0, StepPreparation.Regrow(3.0, 0.0, 1.0, Floor)); // no capacity: nothing
        Assert.Equal(0.0, StepPreparation.Regrow(3.0, double.NaN, 1.0, Floor));
    }

    [Theory]
    [InlineData(0.3)]
    [InlineData(1.0)]
    [InlineData(2.5)]
    [InlineData(5.0)]
    [InlineData(50.0)]
    public void AboveTheCapacityTheResultIsTheCapacity(double rate)
    {
        // Issue #337. A resource is carried above its capacity whenever the capacity fell since
        // the last step. The logistic step from above has no lower bound (it went negative with
        // a steep rate), so the step starts from the capacity at most, and the result is the
        // capacity exactly. The Python function does the same.
        Assert.Equal(100.0, StepPreparation.Regrow(170.0, 100.0, rate, Floor));
        Assert.Equal(100.0, StepPreparation.Regrow(101.0, 100.0, rate, Floor));
        Assert.Equal(100.0, StepPreparation.Regrow(1e6, 100.0, rate, Floor));
        Assert.Equal(37.5, StepPreparation.Regrow(37.6, 37.5, rate, Floor));
    }

    [Fact]
    public void TheResultIsWithinZeroAndTheCapacityWhateverIsCarriedIn()
    {
        foreach (double rate in new[] { 0.1, 1.0, 2.0, 3.0, 5.0, 20.0 })
        {
            for (double carried = 0.0; carried <= 300.0; carried += 0.5)
            {
                double grown = StepPreparation.Regrow(carried, 100.0, rate, Floor);
                Assert.InRange(grown, 0.0, 100.0);
            }
        }
    }

    [Fact]
    public void TheUpdateRegrowsWhatIsInTheBufferWithEachResourcesOwnRate()
    {
        // Cell-major: [cell * 2 + resource]. Capacities: resource 0 = 100, 40, 0; resource 1 = 4.
        double[] buffer = [50.0, 1.0, 10.0, 4.0, 7.0, 0.0];
        StepPreparation.UpdateResources(Parameters(depletable: true), NCells, 0, buffer);
        Assert.Equal(StepPreparation.Regrow(50.0, 100.0, 1.0, Floor), buffer[0]);
        Assert.Equal(StepPreparation.Regrow(1.0, 4.0, 2.5, Floor), buffer[1]);
        Assert.Equal(StepPreparation.Regrow(10.0, 40.0, 1.0, Floor), buffer[2]);
        Assert.Equal(4.0, buffer[3]); // at its capacity
        Assert.Equal(0.0, buffer[4]); // no capacity in that cell, whatever it held
        Assert.Equal(StepPreparation.Regrow(0.0, 4.0, 2.5, Floor), buffer[5]); // from the floor
        Assert.NotEqual(StepPreparation.Regrow(1.0, 4.0, 1.0, Floor), buffer[1]); // the rates differ

        // A second update goes on from the first: the buffer is the state.
        double first = buffer[0];
        StepPreparation.UpdateResources(Parameters(depletable: true), NCells, 1, buffer);
        Assert.Equal(StepPreparation.Regrow(first, 100.0, 1.0, Floor), buffer[0]);

        // Without the feature the buffer's content is ignored: the capacity, every step.
        double[] reset = [50.0, 1.0, 10.0, 4.0, 7.0, 0.0];
        StepPreparation.UpdateResources(Parameters(depletable: false), NCells, 0, reset);
        Assert.Equal(new[] { 100.0, 4.0, 40.0, 4.0, 0.0, 4.0 }, reset);
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Bundle(bool entries, double floor = Floor, double[]? rates = null)
    {
        var c = new SimulationConfig(1);
        c.SetInt("resources.n", 2);
        c.SetArray("resources.size_min", DType.F8, Pin(new[] { 0.1, 0.2 }), 2);
        c.SetArray("resources.size_max", DType.F8, Pin(new[] { 1.0, 2.0 }), 2);
        c.SetArray("resources.trophic_level", DType.F8, Pin(new[] { 1.0, 2.0 }), 2);
        c.SetArray("resources.access_rows", DType.I4, Pin(new[] { 0, 1 }), 2);
        c.SetArray("resources.forcing", DType.F8, Pin(Forcing), 2, 1, NCells);
        c.SetArray("resources.frame_by_step", DType.I4, Pin(FrameByStep), NSteps, 2);
        c.SetArray("resources.accessibility_by_step", DType.F8, Pin(AccessByStep), NSteps, 2);
        if (entries)
        {
            c.SetFloat("resources.depletable.floor", floor);
            c.SetArray("resources.depletable.regrowth_rate", DType.F8, Pin(rates ?? Rates), 2);
        }

        return c;
    }

    private static (ResourceParameters Parameters, IReadOnlyList<string> Problems) Read(SimulationConfig c, bool feature)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Resources(r, NSteps, NCells, oxygen: false, depletable: feature), r.Problems);
    }

    [Fact]
    public void TheEntriesAreReadWithTheFeatureAndRefusedWithoutIt()
    {
        (ResourceParameters p, IReadOnlyList<string> none) = Read(Bundle(entries: true), feature: true);
        Assert.Empty(none);
        Assert.True(p.Depletable);
        Assert.Equal(Floor, p.DepletableFloor);
        Assert.Equal(Rates, p.RegrowthRate);

        (ResourceParameters plain, IReadOnlyList<string> clean) = Read(Bundle(entries: false), feature: false);
        Assert.Empty(clean);
        Assert.False(plain.Depletable);

        (_, IReadOnlyList<string> undeclared) = Read(Bundle(entries: true), feature: false);
        Assert.Equal(2, undeclared.Count);
        Assert.All(undeclared, m => Assert.Contains("present, but the bundle does not declare feature depletable_ltl", m, StringComparison.Ordinal));

        (_, IReadOnlyList<string> missing) = Read(Bundle(entries: false), feature: true);
        Assert.Equal(2, missing.Count); // a declared feature with neither of its two entries
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.5)]
    [InlineData(double.NaN)]
    public void AFloorOutsideZeroToOneIsRefused(double floor)
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(entries: true, floor: floor), feature: true);
        string only = Assert.Single(problems);
        Assert.Contains("resources.depletable.floor", only, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ARateThatIsNegativeOrNotFiniteIsRefused(double rate)
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(entries: true, rates: [1.0, rate]), feature: true);
        string only = Assert.Single(problems);
        Assert.Contains("resources.depletable.regrowth_rate: values must be finite and >= 0", only, StringComparison.Ordinal);
    }
}
