using Osmose.Engine.Schools;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 1.3: the school store and its regroup.</summary>
public class SchoolStoreTests
{
    private static SchoolStore RandomStore(int nCells, int n, int seed, out Dictionary<ulong, (int Cell, double Abundance, double Length)> truth)
    {
        var rng = new Random(seed);
        var store = new SchoolStore(nCells, initialCapacity: 4);
        truth = [];
        for (int i = 0; i < n; i++)
        {
            int cell = rng.Next(-1, nCells); // -1: unlocated
            double abundance = rng.NextDouble() < 0.25 ? 0.0 : rng.NextDouble() * 1e6; // a quarter dead
            double length = 1 + (rng.NextDouble() * 80);
            int at = store.Append(new SchoolCore { Abundance = abundance, Length = length, SpeciesId = rng.Next(5) }, new SchoolLife { CellId = cell });
            truth[store.Life[at].Id] = (cell, abundance, length);
        }

        return store;
    }

    [Fact]
    public void AppendAssignsIncreasingIdsAndGrowsCapacity()
    {
        var store = new SchoolStore(nCells: 3, initialCapacity: 2);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(i, store.Append(new SchoolCore { Abundance = 1, Length = i }, new SchoolLife { CellId = 0 }));
        }

        Assert.Equal(10, store.Count);
        Assert.True(store.Capacity >= 10);
        Assert.Equal(Enumerable.Range(0, 10).Select(i => (ulong)i), store.Life.ToArray().Select(l => l.Id));
        Assert.Equal(10UL, store.NextId);
    }

    [Fact]
    public void RegroupKeepsExactlyTheLiveSchoolsInContiguousCellsAndTheTail()
    {
        const int nCells = 17;
        SchoolStore store = RandomStore(nCells, 5000, seed: 1, out var truth);
        store.Regroup();

        var live = truth.Where(kv => kv.Value.Abundance != 0.0).ToDictionary(kv => kv.Key, kv => kv.Value);
        Assert.Equal(live.Count, store.Count);
        Assert.Equal(live.Keys.Order(), store.Life.ToArray().Select(l => l.Id).Order()); // multiset of ids
        ReadOnlySpan<int> start = store.CellStart;
        Assert.Equal(0, start[0]);
        for (int c = 0; c < nCells; c++)
        {
            Assert.True(start[c] <= start[c + 1]);
            for (int i = start[c]; i < start[c + 1]; i++)
            {
                Assert.Equal(c, store.Life[i].CellId);
            }
        }

        for (int i = start[nCells]; i < start[nCells + 1]; i++)
        {
            Assert.Equal(-1, store.Life[i].CellId); // the unlocated tail
        }

        Assert.Equal(store.Count, start[nCells + 1]);
        for (int i = 0; i < store.Count; i++)
        {
            (int cell, double abundance, double length) = live[store.Life[i].Id];
            Assert.Equal(abundance, store.Core[i].Abundance);
            Assert.Equal(length, store.Core[i].Length);
            Assert.Equal(store.Core[i].Length, store.LengthLane[i]); // the lane mirrors Core.Length
        }
    }

    [Fact]
    public void RegroupIsStableWithinACell()
    {
        SchoolStore store = RandomStore(5, 2000, seed: 2, out _);
        store.Regroup();
        ReadOnlySpan<int> start = store.CellStart;
        for (int c = 0; c <= 5; c++) // every cell and the tail
        {
            for (int i = start[c] + 1; i < start[c + 1]; i++)
            {
                Assert.True(store.Life[i - 1].Id < store.Life[i].Id, "ids were appended in order, so a stable sort keeps them ascending");
            }
        }
    }

    [Fact]
    public void ARegroupOfARegroupedStoreChangesNothing()
    {
        SchoolStore store = RandomStore(9, 1000, seed: 3, out _);
        store.Regroup();
        ulong[] before = store.Life.ToArray().Select(l => l.Id).ToArray();
        int[] cells = store.CellStart.ToArray();
        store.Regroup();
        Assert.Equal(before, store.Life.ToArray().Select(l => l.Id));
        Assert.Equal(cells, store.CellStart.ToArray());
    }

    [Fact]
    public void TenThousandAppendRegroupCyclesAllocateNothingAfterWarmUp()
    {
        var store = new SchoolStore(nCells: 50, initialCapacity: 8192);
        var rng = new Random(4);
        void Cycle()
        {
            for (int k = 0; k < 3; k++)
            {
                store.Append(new SchoolCore { Abundance = rng.NextDouble(), Length = 10 }, new SchoolLife { CellId = rng.Next(-1, 50) });
            }

            // Kill a few so the store stays bounded.
            Span<SchoolCore> core = store.Core;
            for (int i = 0; i < core.Length; i += 2)
            {
                core[i].Abundance = rng.NextDouble() < 0.5 ? 0.0 : core[i].Abundance;
            }

            store.Regroup();
        }

        // Warm-up: the store reaches its steady size, and tiered compilation promotes the hot
        // methods (call counting, then a background rejit after a delay). A rejit landing inside
        // the measured window costs a fixed few KB once, which is not a per-cycle allocation.
        for (int round = 0; round < 3; round++)
        {
            for (int i = 0; i < 10_000; i++)
            {
                Cycle();
            }

            Thread.Sleep(250);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
        {
            Cycle();
        }

        long mid = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
        {
            Cycle();
        }

        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(0, mid - before);
        Assert.Equal(0, after - mid);
    }
}
