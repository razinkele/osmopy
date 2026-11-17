using Osmose.Engine.Rng;

namespace Osmose.Engine.Tests;

/// <summary>
/// Plan task 0.5. Every literal below was produced by NumPy's Philox (which implements
/// Philox4x64-10) through the call in <c>osmose/engine_cs/philox.py</c>, and
/// <c>tests/test_csharp_philox_numpy.py</c> asserts the same literals, so C# and NumPy are pinned
/// to each other. The first three are also Random123's published known-answer vectors.
/// </summary>
public class PhiloxTests
{
    private const ulong Max = ulong.MaxValue;

    public static TheoryData<ulong[], ulong[], ulong[]> Random123Vectors => new()
    {
        { [0, 0], [0, 0, 0, 0], [0x16554d9eca36314c, 0xdb20fe9d672d0fdc, 0xd7e772cee186176b, 0x7e68b68aec7ba23b] },
        { [Max, Max], [Max, Max, Max, Max], [0x87b092c3013fe90b, 0x438c3c67be8d0224, 0x9cc7d7c69cd777b6, 0xa09caebf594f0ba0] },
        {
            [0x452821e638d01377, 0xbe5466cf34e90c6c],
            [0x243f6a8885a308d3, 0x13198a2e03707344, 0xa4093822299f31d0, 0x082efa98ec4e6c89],
            [0xa528f45403e61d95, 0x38c72dbd566e9788, 0xa5a1610e72fd18b5, 0x57bd43b5e52b7fe6]
        },
    };

    [Theory]
    [MemberData(nameof(Random123Vectors))]
    public void BlockMatchesRandom123KnownAnswers(ulong[] key, ulong[] ctr, ulong[] expected)
    {
        PhiloxBlock b = Philox.Block(new PhiloxKey(key[0], key[1]), ctr[0], ctr[1], ctr[2], ctr[3]);
        Assert.Equal(expected, new[] { b.W0, b.W1, b.W2, b.W3 });

        Span<ulong> out4 = stackalloc ulong[4];
        Philox.Block(new PhiloxKey(key[0], key[1]), ctr[0], ctr[1], ctr[2], ctr[3], out4);
        Assert.Equal(expected, out4.ToArray());
    }

    [Fact]
    public void SplitMix64MatchesReference()
    {
        Assert.Equal(0xe220a8397b1dcdafUL, Philox.SplitMix64(0));
        Assert.Equal(0xbdd732262feb6e95UL, Philox.SplitMix64(42));
    }

    [Fact]
    public void RunKeyIsSplitMixOfSeedAndDomainXorScheme()
    {
        Assert.Equal(new PhiloxKey(0xbdd732262feb6e95, 0x4f534d4f53450001), PhiloxKey.ForRun(42, scheme: 1));
        Assert.NotEqual(PhiloxKey.ForRun(42, 1).K0, PhiloxKey.ForRun(43, 1).K0);
        Assert.NotEqual(PhiloxKey.ForRun(42, 1), PhiloxKey.ForRun(42, 2));
    }

    public static TheoryData<ulong, ulong, int, int, ulong, ulong[]> EngineLayoutVectors => new()
    {
        // block, entity, step, substep, purpose -> NumPy's words for key ForRun(42, 1)
        { 0, 17, 5, 3, 1, [0xbecd341680fb139c, 0xa6a81d01e0bbf1ec, 0x4ff9b5fd10e6288e, 0xbe64881c5c0aea12] },
        { 1, 17, 5, 3, 1, [0x3cecbbfd3e374f5a, 0x1973133903269769, 0x6d4ca27a9a167d95, 0x59e3ef8da1c25e93] },
        { 0, 17, 5, 3, 6, [0x9b42371466055372, 0xc42faed7844d80dc, 0x3984ee50b3d807d0, 0x0c474d9eb9ba349c] },
        { 0, (1UL << 40) + 3, 1199, 9, 13, [0xc3fc78d7b29ccb47, 0x215fac61f9197312, 0x0a4233f1c4c8b53f, 0x586eec303ec9687d] },
    };

    [Theory]
    [MemberData(nameof(EngineLayoutVectors))]
    public void StreamUsesTheSpecCounterLayout(ulong block, ulong entity, int step, int substep, ulong purpose, ulong[] expected)
    {
        var stream = new PhiloxStream(PhiloxKey.ForRun(42, 1), (DrawPurpose)purpose, step, substep, entity);
        for (ulong skip = 0; skip < block * 4; skip++)
        {
            stream.NextUInt64();
        }

        Assert.Equal(expected, new[] { stream.NextUInt64(), stream.NextUInt64(), stream.NextUInt64(), stream.NextUInt64() });
    }

    [Fact]
    public void StepWordPacksStepAndSubstep()
    {
        Assert.Equal((5UL << 16) | 3, PhiloxStream.StepWord(5, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => PhiloxStream.StepWord(0, 65536));
        Assert.Throws<ArgumentOutOfRangeException>(() => PhiloxStream.StepWord(-1, 0));
    }

    [Fact]
    public void BoundedStaysInRangeAndRangeOneIsZero()
    {
        var s = new PhiloxStream(PhiloxKey.ForRun(1, 1), DrawPurpose.PatchStart, 0, 0, 0);
        for (int i = 0; i < 10_000; i++)
        {
            Assert.Equal(0UL, s.NextBounded(1));
            Assert.InRange(s.NextBounded(7), 0UL, 6UL);
            Assert.InRange(s.NextBounded((1UL << 63) + 1), 0UL, 1UL << 63);
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => s.NextBounded(0));
    }

    [Fact]
    public void LemireRejectsTheBiasedSliver()
    {
        // range 3: 2^64 mod 3 = 1, so exactly the words with x * 3 mod 2^64 == 0 (x = 0) are biased.
        Assert.False(PhiloxStream.TryBounded(0, 3, out _));
        Assert.True(PhiloxStream.TryBounded(1, 3, out ulong v1));
        Assert.Equal(0UL, v1);
        Assert.True(PhiloxStream.TryBounded(ulong.MaxValue, 3, out ulong vMax));
        Assert.Equal(2UL, vMax);
        // range r = 2^63 + 1: threshold (2^64 - r) mod r = 2^63 - 1, nearly half of all words.
        ulong r = (1UL << 63) + 1;
        Assert.True(PhiloxStream.TryBounded(1, r, out ulong vr)); // low = r >= threshold: kept
        Assert.Equal(0UL, vr);
        Assert.False(PhiloxStream.TryBounded(2, r, out _)); // low = 2r mod 2^64 = 2 < threshold
    }

    [Fact]
    public void DoublesAreInTheUnitIntervalWith53Bits()
    {
        var s = new PhiloxStream(PhiloxKey.ForRun(2, 1), DrawPurpose.InitPlace, 0, 0, 0);
        double sum = 0;
        const int n = 200_000;
        for (int i = 0; i < n; i++)
        {
            double u = s.NextDouble();
            Assert.InRange(u, 0.0, Math.BitDecrement(1.0));
            Assert.Equal(u, Math.Round(u * (1UL << 53)) / (1UL << 53));
            sum += u;
        }

        Assert.InRange(sum / n, 0.497, 0.503); // sd of the mean ~0.00065
    }

    [Fact]
    public void LehmerTablesHoldEveryOrderExactlyOnce()
    {
        foreach (int n in new[] { 4, 5 })
        {
            var seen = new HashSet<string>();
            for (int code = 0; code < CauseOrders.Count(n); code++)
            {
                byte[] order = CauseOrders.Decode(n, code).ToArray();
                Assert.Equal(Enumerable.Range(0, n).Select(i => (byte)i), order.Order());
                Assert.True(seen.Add(string.Join(",", order)));
            }

            Assert.Equal(CauseOrders.Count(n), seen.Count);
            Assert.Equal(Enumerable.Range(0, n).Select(i => (byte)i), CauseOrders.Decode(n, 0).ToArray());
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => CauseOrders.Decode(4, 24));
        Assert.Throws<ArgumentOutOfRangeException>(() => CauseOrders.Count(3));
    }

    // Chi-square, fixed counters: deterministic. Critical values at p = 0.001 (scipy chi2.ppf):
    // 49.728 for 23 degrees of freedom, 172.418 for 119.
    [Fact]
    public void PermutationsOfFourAreUniform()
    {
        var counts = new Dictionary<string, int>();
        Span<int> perm = stackalloc int[4];
        const int n = 240_000;
        for (int cell = 0; cell < n; cell++)
        {
            var src = new PhiloxDrawSource(PhiloxKey.ForRun(7, 1), step: 3, cellId: cell);
            src.BeginSubstep(0);
            src.Permutation(DrawPurpose.PermPredation, perm);
            string k = string.Join(",", perm.ToArray());
            counts[k] = counts.GetValueOrDefault(k) + 1;
        }

        Assert.Equal(24, counts.Count);
        Assert.InRange(ChiSquare(counts.Values, n / 24.0), 0.0, 49.728);
    }

    [Theory]
    [InlineData(4, 240_000, 49.728)]
    [InlineData(5, 600_000, 172.418)]
    public void CauseOrdersAreUniform(int nCauses, int draws, double critical)
    {
        int orders = CauseOrders.Count(nCauses);
        var counts = new Dictionary<string, int>();
        Span<byte> order = stackalloc byte[nCauses];
        var src = new PhiloxDrawSource(PhiloxKey.ForRun(11, 1), step: 9, cellId: 4);
        for (int sub = 0; sub < draws / 1000; sub++)
        {
            src.BeginSubstep(sub);
            for (int slot = 0; slot < 1000; slot++)
            {
                src.NextCauseOrder(order);
                string k = string.Join(",", order.ToArray());
                counts[k] = counts.GetValueOrDefault(k) + 1;
            }
        }

        Assert.Equal(orders, counts.Count);
        Assert.InRange(ChiSquare(counts.Values, (double)draws / orders), 0.0, critical);
    }

    [Fact]
    public void CellDrawsDoNotDependOnTheOrderCellsAreVisited()
    {
        var key = PhiloxKey.ForRun(5, 1);
        int[][] forward = Enumerable.Range(0, 50).Select(c => Draw(key, c)).ToArray();
        int[][] backward = Enumerable.Range(0, 50).Reverse().Select(c => Draw(key, c)).Reverse().ToArray();
        Assert.Equal(forward, backward);
        Assert.Contains(forward, p => !p.SequenceEqual(forward[0])); // cells differ

        static int[] Draw(PhiloxKey key, int cell)
        {
            var src = new PhiloxDrawSource(key, step: 12, cellId: cell);
            src.BeginSubstep(2);
            int[] p = new int[30];
            src.Permutation(DrawPurpose.PermFishing, p);
            return p;
        }
    }

    [Fact]
    public void PurposesSubstepsAndStepsGiveDifferentStreams()
    {
        var key = PhiloxKey.ForRun(5, 1);
        int[] a = Perm(key, 0, 0, DrawPurpose.PermPredation);
        Assert.NotEqual(a, Perm(key, 0, 0, DrawPurpose.PermStarvation));
        Assert.NotEqual(a, Perm(key, 0, 1, DrawPurpose.PermPredation));
        Assert.NotEqual(a, Perm(key, 1, 0, DrawPurpose.PermPredation));
        Assert.Equal(a, Perm(key, 0, 0, DrawPurpose.PermPredation));

        static int[] Perm(PhiloxKey key, int step, int substep, DrawPurpose purpose)
        {
            var src = new PhiloxDrawSource(key, step, cellId: 0);
            src.BeginSubstep(substep);
            int[] p = new int[40];
            src.Permutation(purpose, p);
            return p;
        }
    }

    [Fact]
    public void DrawingBeforeBeginSubstepThrows()
    {
        var src = new PhiloxDrawSource(PhiloxKey.ForRun(1, 1), 0, 0);
        Assert.Throws<InvalidOperationException>(() => src.Permutation(DrawPurpose.PermFishing, new int[3]));
        Assert.Throws<InvalidOperationException>(() => src.NextCauseOrder(new byte[4]));
    }

    [Fact]
    public void InjectedSourceReplaysExactlyAndRefusesToInvent()
    {
        int[][][] perms = [[[2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 1, 0], [0, 2, 1]]];
        byte[][] orders = [[3, 2, 1, 0, 0, 1, 2, 3]];
        var src = new InjectedDrawSource(perms, orders);
        src.BeginSubstep(0);

        int[] p = new int[3];
        src.Permutation(DrawPurpose.PermAdditional, p);
        Assert.Equal([2, 1, 0], p);

        byte[] o = new byte[4];
        src.NextCauseOrder(o);
        Assert.Equal([3, 2, 1, 0], o);
        src.NextCauseOrder(o);
        Assert.Equal([0, 1, 2, 3], o);
        Assert.Throws<InvalidOperationException>(() => src.NextCauseOrder(o));
        Assert.Throws<InvalidOperationException>(() => src.Permutation(DrawPurpose.PermPredation, new int[4]));
        Assert.Throws<ArgumentOutOfRangeException>(() => src.BeginSubstep(1));
    }

    private static double ChiSquare(IEnumerable<int> observed, double expected) =>
        observed.Sum(o => (o - expected) * (o - expected) / expected);
}
