using System.Runtime.InteropServices;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 0.3: the bundle container, its file format and its validation.</summary>
public sealed class BundleTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("osm-bundle-").FullName;
    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }

        Directory.Delete(_dir, recursive: true);
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Sample()
    {
        var c = new SimulationConfig(contractVersion: 1);
        c.SetInt("time.n_steps", 240);
        c.SetFloat("run.deadline_seconds", 12.5);
        c.SetArray("growth.linf", DType.F8, Pin(new[] { 10.0, 20.5, 33.25 }), 3);
        c.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[] { 1, 0, 1, 1, 0, 1 }), 2, 3);
        c.SetArray("species.n_schools", DType.I4, Pin(new[] { 20, 30 }), 2);
        c.SetArray("movement.maps.index.offsets", DType.I8, Pin(new long[] { 0, 5, 9 }), 3);
        c.SetArray("state.empty", DType.F8, 0, 0);
        return c;
    }

    [Fact]
    public void SaveThenLoadRoundTripsEveryEntry()
    {
        string path = Path.Combine(_dir, "a.bin");
        Sample().Save(path);
        using SimulationConfig loaded = SimulationConfig.Load(path);

        Assert.Equal(1, loaded.ContractVersion);
        Assert.Equal(7, loaded.Count);
        Assert.True(loaded.TryGet("time.n_steps", out BundleEntry steps));
        Assert.Equal(240, steps.IntValue);
        Assert.True(loaded.TryGet("run.deadline_seconds", out BundleEntry deadline));
        Assert.Equal(12.5, deadline.FloatValue);
        Assert.True(loaded.TryGet("growth.linf", out BundleEntry linf));
        Assert.Equal([10.0, 20.5, 33.25], linf.AsSpan<double>().ToArray());
        Assert.True(loaded.TryGet("grid.ocean_mask", out BundleEntry mask));
        Assert.Equal([2L, 3L], mask.Shape.ToArray());
        Assert.Equal(new byte[] { 1, 0, 1, 1, 0, 1 }, mask.AsSpan<byte>().ToArray());
        Assert.True(loaded.TryGet("species.n_schools", out BundleEntry ns));
        Assert.Equal([20, 30], ns.AsSpan<int>().ToArray());
        Assert.True(loaded.TryGet("movement.maps.index.offsets", out BundleEntry off));
        Assert.Equal([0L, 5L, 9L], off.AsSpan<long>().ToArray());
        Assert.True(loaded.TryGet("state.empty", out BundleEntry empty));
        Assert.Equal(0, empty.Length);
    }

    [Fact]
    public void FileIsCanonicalWhateverTheInsertionOrder()
    {
        string a = Path.Combine(_dir, "a.bin");
        string b = Path.Combine(_dir, "b.bin");
        Sample().Save(a);

        var reversed = new SimulationConfig(1);
        reversed.SetArray("state.empty", DType.F8, 0, 0);
        reversed.SetArray("movement.maps.index.offsets", DType.I8, Pin(new long[] { 0, 5, 9 }), 3);
        reversed.SetArray("species.n_schools", DType.I4, Pin(new[] { 20, 30 }), 2);
        reversed.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[] { 1, 0, 1, 1, 0, 1 }), 2, 3);
        reversed.SetArray("growth.linf", DType.F8, Pin(new[] { 10.0, 20.5, 33.25 }), 3);
        reversed.SetFloat("run.deadline_seconds", 12.5);
        reversed.SetInt("time.n_steps", 240);
        reversed.Save(b);

        Assert.Equal(File.ReadAllBytes(a), File.ReadAllBytes(b));
    }

    [Fact]
    public void LoadedBundleSavesBackByteIdentical()
    {
        string a = Path.Combine(_dir, "a.bin");
        string b = Path.Combine(_dir, "b.bin");
        Sample().Save(a);
        using (SimulationConfig loaded = SimulationConfig.Load(a))
        {
            loaded.Save(b);
        }

        Assert.Equal(File.ReadAllBytes(a), File.ReadAllBytes(b));
    }

    [Fact]
    public void ArraysAreAlignedTo64Bytes()
    {
        string path = Path.Combine(_dir, "a.bin");
        Sample().Save(path);
        using SimulationConfig loaded = SimulationConfig.Load(path);
        foreach (BundleEntry e in loaded.Entries)
        {
            if (e.Kind == EntryKind.Array && e.Length > 0)
            {
                Assert.Equal(0, e.Address % 64);
            }
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("Growth.linf")]
    [InlineData("growth..linf")]
    [InlineData(".growth")]
    [InlineData("growth.linf.")]
    [InlineData("growth-linf")]
    public void InvalidNamesAreRejected(string name)
    {
        var c = new SimulationConfig(1);
        Assert.Throws<ArgumentException>(() => c.SetInt(name, 1));
    }

    [Fact]
    public void NamesLongerThan63BytesAreRejected()
    {
        var c = new SimulationConfig(1);
        c.SetInt(new string('a', 63), 1);
        Assert.Throws<ArgumentException>(() => c.SetInt(new string('a', 64), 1));
    }

    [Fact]
    public void DuplicatesBadShapesAndNullAddressesAreRejected()
    {
        var c = new SimulationConfig(1);
        c.SetInt("a.b", 1);
        Assert.Throws<ArgumentException>(() => c.SetFloat("a.b", 2.0));
        Assert.Throws<ArgumentException>(() => c.SetArray("x", DType.F8, 8, -1));       // no dimension
        Assert.Throws<ArgumentException>(() => c.SetArray("x", DType.F8, 8, 2, -1, 3)); // gap
        Assert.Throws<ArgumentException>(() => c.SetArray("x", DType.F8, 0, 4));        // null, non-empty
        Assert.Throws<ArgumentOutOfRangeException>(() => c.SetArray("x", (DType)9, 8, 1));
    }

    [Fact]
    public void LoadRejectsANonBundleFile()
    {
        string path = Path.Combine(_dir, "junk.bin");
        File.WriteAllBytes(path, new byte[100]);
        var ex = Assert.Throws<BundleFormatException>(() => SimulationConfig.Load(path));
        Assert.Contains("bad magic", ex.Message, StringComparison.Ordinal);

        File.WriteAllBytes(path, new byte[10]);
        Assert.Throws<BundleFormatException>(() => SimulationConfig.Load(path));
    }

    [Fact]
    public void LoadListsEveryCorruptRecord()
    {
        string path = Path.Combine(_dir, "a.bin");
        Sample().Save(path);
        byte[] bytes = File.ReadAllBytes(path);
        // Entries sorted: grid.ocean_mask(0), growth.linf(1), movement...(2), run...(3), species...(4)
        int rec0 = SimulationConfig.HeaderSize;
        bytes[rec0 + 65] = 9;                                   // grid.ocean_mask: unknown dtype
        int rec1 = rec0 + SimulationConfig.RecordSize;
        bytes[rec1 + 112] = 99;                                 // growth.linf: byte length disagrees
        int rec4 = rec0 + (4 * SimulationConfig.RecordSize);
        bytes[rec4 + 104] += 4;                                 // species.n_schools: misaligned data
        File.WriteAllBytes(path, bytes);

        var ex = Assert.Throws<BundleFormatException>(() => SimulationConfig.Load(path));
        Assert.Equal(3, ex.Problems.Count);
        Assert.Contains(ex.Problems, p => p.Contains("grid.ocean_mask", StringComparison.Ordinal));
        Assert.Contains(ex.Problems, p => p.Contains("growth.linf", StringComparison.Ordinal));
        Assert.Contains(ex.Problems, p => p.Contains("species.n_schools", StringComparison.Ordinal));
    }

    [Fact]
    public void ReaderListsEveryProblemInOneException()
    {
        var c = new SimulationConfig(contractVersion: 7);
        c.SetInt("time.n_steps", 240);
        c.SetArray("growth.linf", DType.I4, Pin(new[] { 1, 2, 3 }), 3);         // wrong dtype
        c.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[6]), 3, 2);       // wrong shape
        c.SetFloat("growth.k", 0.3);                                            // scalar, not array

        var r = new BundleReader(c, 1, 2);
        Assert.Equal(240, r.Int("time.n_steps"));
        Assert.Null(r.Array("growth.linf", DType.F8, 3));
        Assert.Null(r.Array("grid.ocean_mask", DType.U1, 2, 3));
        Assert.Null(r.Array("growth.k", DType.F8, 3));
        Assert.Null(r.Array("growth.t0", DType.F8, 3));                          // missing
        Assert.Equal(0, r.Int("time.n_dt_per_year"));                            // missing
        Assert.True(double.IsNaN(r.Float("time.n_steps")));                     // wrong kind

        var ex = Assert.Throws<BundleValidationException>(r.ThrowIfProblems);
        Assert.Equal(7, ex.Problems.Count);
        string all = ex.Message;
        foreach (string needle in new[] { "contract version 7", "growth.linf: dtype I4", "grid.ocean_mask: shape (3, 2)", "growth.k", "growth.t0: missing", "time.n_dt_per_year: missing", "time.n_steps: expected Float" })
        {
            Assert.Contains(needle, all, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ReaderAcceptsWildcardDimensions()
    {
        var c = new SimulationConfig(1);
        c.SetArray("resources.forcing", DType.F8, Pin(new double[24]), 2, 3, 4);
        var r = new BundleReader(c, 1);
        BundleEntry? e = r.Array("resources.forcing", DType.F8, 2, BundleReader.Any, 4);
        Assert.NotNull(e);
        Assert.Equal(3, e.Shape[1]);
        r.ThrowIfProblems();
    }

    [Fact]
    public void AsSpanRefusesTheWrongElementType()
    {
        var c = new SimulationConfig(1);
        c.SetArray("growth.linf", DType.F8, Pin(new[] { 1.0 }), 1);
        Assert.True(c.TryGet("growth.linf", out BundleEntry e));
        Assert.Throws<InvalidOperationException>(() => e.AsSpan<long>().ToArray());
        Assert.Equal(1.0, e.AsSpan<double>()[0]);
    }
}
