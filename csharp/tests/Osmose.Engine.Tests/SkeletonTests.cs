using Osmose.Engine.Spike;

namespace Osmose.Engine.Tests;

public class SkeletonTests
{
    [Fact]
    public void RunSimulationIsNotImplementedYet()
    {
        var ex = Assert.Throws<NotImplementedException>(
            () => OsmoseEngine.RunSimulation(new SimulationConfig(contractVersion: 1)));
        Assert.Contains("Phase 1", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RunSimulationRejectsNull()
    {
        Assert.Throws<ArgumentNullException>(() => OsmoseEngine.RunSimulation(null!));
    }

    [Fact]
    public unsafe void ScaleInPlaceWritesThroughTheAddress()
    {
        double[] values = [1.0, 2.0, 3.0];
        fixed (double* p = values)
        {
            BridgeSpike.ScaleInPlace((long)p, values.Length, 2.5);
        }

        Assert.Equal([2.5, 5.0, 7.5], values);
    }

    [Fact]
    public unsafe void LoopStopsWhenTheCancelSlotIsSet()
    {
        long[] control = new long[8];
        control[0] = 1;
        fixed (long* p = control)
        {
            Assert.Equal(0, BridgeSpike.LoopUntilCancelled((long)p, maxIterations: 10, sleepMilliseconds: 0));
        }

        Assert.Equal(0, control[1]);
    }
}
