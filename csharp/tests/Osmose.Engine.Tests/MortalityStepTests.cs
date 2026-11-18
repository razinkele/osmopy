using Osmose.Engine.Schools;
using Osmose.Engine.Step;
using Xunit;

namespace Osmose.Engine.Tests;

public class MortalityStepTests
{
    [Fact]
    public void Pairwise8_AddsInNumPysOrder()
    {
        // NumPy: np.array([[0.1, 0.7, 1e-17, 3.3, 0.2, 1e16, -1e16, 0.3]]).sum(axis=1) == 4.1;
        // left to right the same values give 4.3, so the order is observable.
        double[] v = [0.1, 0.7, 1e-17, 3.3, 0.2, 1e16, -1e16, 0.3];
        double sequential = 0.0;
        foreach (double x in v)
        {
            sequential += x;
        }

        Assert.Equal(4.1, MortalityStep.Pairwise8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]));
        Assert.NotEqual(4.1, sequential);
    }

    private static (SchoolCore[] Core, SchoolLedger[] Ledger, SchoolLife[] Life, double[] Larva) School(
        double start, double larva, double predation, double fishing, double final)
    {
        var core = new[] { new SchoolCore { Abundance = final, DeadPredation = predation } };
        var ledger = new[] { new SchoolLedger { AbundanceAtMortalityStart = start - larva, DeadAdditional = larva, DeadFishing = fishing } };
        var life = new[] { new SchoolLife { Id = 42 } };
        return (core, ledger, life, [larva]);
    }

    [Fact]
    public void DeathAccounting_PassesWhenRemovedEqualsCounted()
    {
        var (core, led, life, larva) = School(start: 100.0, larva: 10.0, predation: 20.0, fishing: 5.0, final: 65.0);
        MortalityStep.CheckDeathAccounting(core, led, life, larva, step: 7);
    }

    [Fact]
    public void DeathAccounting_ToleratesRoundingWithinTheBound()
    {
        var (core, led, life, larva) = School(100.0, 10.0, 20.0, 5.0, final: 65.0 + 1e-8);
        MortalityStep.CheckDeathAccounting(core, led, life, larva, step: 7);
    }

    [Fact]
    public void DeathAccounting_CatchesOverkillTheClampWouldHide()
    {
        // Counters sum to 130 of 100 fish; the commit's max(0, ...) leaves 0, so only the check sees it.
        var (core, led, life, larva) = School(100.0, 10.0, predation: 90.0, fishing: 30.0, final: 0.0);
        var e = Assert.Throws<InvariantViolationException>(() => MortalityStep.CheckDeathAccounting(core, led, life, larva, step: 7));
        Assert.Equal(42UL, e.SchoolId);
        Assert.Equal(7, e.Step);
        Assert.Contains("death accounting", e.Cause);
    }

    [Fact]
    public void DeathAccounting_CatchesAMissingCounterAndNaN()
    {
        var (core, led, life, larva) = School(100.0, 10.0, 20.0, 5.0, final: 60.0); // 5 fish uncounted
        Assert.Throws<InvariantViolationException>(() => MortalityStep.CheckDeathAccounting(core, led, life, larva, 3));

        (core, led, life, larva) = School(100.0, 10.0, double.NaN, 5.0, final: 65.0);
        Assert.Throws<InvariantViolationException>(() => MortalityStep.CheckDeathAccounting(core, led, life, larva, 3));
    }
}
