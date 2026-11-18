using Osmose.Engine.Schools;
using Osmose.Engine.Step;
using Xunit;

namespace Osmose.Engine.Tests;

public class OutputsTests
{
    private static double[] Vector(int k)
    {
        var v = new double[k];
        for (int i = 0; i < k; i++)
        {
            v[i] = (((i % 7) - 3) * Math.Pow(10.0, (i % 5) * 4)) + (1.0 / (i + 1));
        }

        return v;
    }

    [Theory]
    // NumPy: np.sum(v[:, None], axis=0)[0], the record window of a one-value row (2026-11-18).
    [InlineData(8, 9999999600010002.0)]
    [InlineData(9, 9997999600010002.0)]
    [InlineData(17, -2.999900019999e+16)] // left to right: -2.9999000199989996e+16
    [InlineData(130, -9999000400019992.0)] // left to right: -9999000400019972.0
    [InlineData(300, -9998000199989964.0)] // left to right: -9998000199989936.0
    public void PairwiseSum_IsNumPysWindowSum(int k, double numpy)
    {
        double[] v = Vector(k);
        Assert.Equal(numpy, OutputCollector.PairwiseSum(v));
        if (k >= 17)
        {
            double sequential = 0.0;
            foreach (double x in v)
            {
                sequential += x;
            }

            Assert.NotEqual(numpy, sequential); // the order is observable here
        }
    }

    [Fact]
    public void FoldWindow_OneValueRowsUseThePairwiseSumAndWiderRowsAddInOrder()
    {
        double[] v = Vector(17);
        var into = new double[1];
        OutputCollector.FoldWindow(v, 17, 1, mean: false, into);
        Assert.Equal(-2.999900019999e+16, into[0]); // NumPy, as in PairwiseSum_IsNumPysWindowSum
        OutputCollector.FoldWindow(v, 17, 1, mean: true, into);
        Assert.Equal(-2.999900019999e+16 / 17, into[0]);

        // Two columns: the same values in column 0 now add in order (NumPy's axis-0 reduction).
        var rows = new double[34];
        for (int t = 0; t < 17; t++)
        {
            rows[2 * t] = v[t];
            rows[(2 * t) + 1] = 1.0;
        }

        var two = new double[2];
        OutputCollector.FoldWindow(rows, 17, 2, mean: false, two);
        Assert.Equal(-2.9999000199989996e+16, two[0]);
        Assert.Equal(17.0, two[1]);
        OutputCollector.FoldWindow(rows, 1, 2, mean: true, two);
        Assert.Equal(rows[0], two[0]); // one step passes through, not divided
    }

    [Fact]
    public void BeginStep_ClearsEveryOutOfDomainFlag()
    {
        // The reset that _reset_step_variables does before movement. Map movement rewrites the
        // flag of every map school, so on Phase 1 configs nothing else can show it is done.
        var store = new SchoolStore(4, 8);
        for (int i = 0; i < 5; i++)
        {
            store.Append(new SchoolCore { Abundance = 1.0, Flags = SchoolFlags.IsOut | SchoolFlags.IsEgg }, new SchoolLife { CellId = i % 4 });
        }

        SimulationRun.ClearOutOfDomain(store.Core);
        foreach (SchoolCore c in store.Core)
        {
            Assert.Equal(SchoolFlags.IsEgg, c.Flags);
        }
    }
}
