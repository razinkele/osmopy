using System.Runtime.InteropServices;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.4: the salinity movement gate (feature <c>salinity_gate</c>). The checks
/// against the Python kernel on Baltic's real gate are on the Python side
/// (tests/test_csharp_salinity_gate.py); these pin the rules on a grid small enough to read, and
/// the bundle contract.</summary>
public sealed class SalinityGateTests : IDisposable
{
    // A 3 x 4 grid, all ocean but cell 5. One map (presence, with a NaN and a zero), two frames.
    private const int Ny = 3;
    private const int Nx = 4;
    private const int NCells = Ny * Nx;
    private const int NSteps = 3;

    private static readonly byte[] Ocean = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1];
    private static readonly double[] Map = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, double.NaN, 1.0, 1.0, 1.0, 1.0];

    private static readonly double[] Weight =
    [
        0.1, 0.7, 0.9, 0.0, 0.3, 0.6, 0.2, 0.5, double.NaN, 0.4, 0.8, 1.0, // frame 0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // frame 1: excludes every cell
    ];

    private static readonly int[] FrameByStep = [0, 1, 0];
    private static readonly double[] Wmax = [1.0, 0.0]; // one map x two frames

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static MovementParameters Parameters() => new()
    {
        NMaps = 1,
        MapStack = Map,
        Ocean = Ocean,
        SalinitySpecies = [1, 0],
        SalinityWeight = Weight,
        SalinityFrameByStep = FrameByStep,
        SalinityWmax = Wmax,
        SalinityFrames = 2,
    };

    [Fact]
    public void TheGateAppliesToAGatedSpeciesOnAMapWithAPositiveProduct()
    {
        MovementParameters m = Parameters();
        Assert.True(Movement.Gate(m, 0, 0, step: 0, NCells, out ReadOnlySpan<double> w, out double wmax));
        Assert.Equal(1.0, wmax);
        Assert.Equal(0.7, w[1]);
        Assert.False(Movement.Gate(m, 1, 0, step: 0, NCells, out _, out _)); // species not gated
        Assert.False(Movement.Gate(m, 2, 0, step: 0, NCells, out _, out _)); // beyond the mask
        Assert.False(Movement.Gate(m, 0, 0, step: 1, NCells, out _, out _)); // wmax 0: the all-zero guard
        Assert.False(Movement.Gate(new MovementParameters(), 0, 0, step: 0, NCells, out _, out _)); // no feature
    }

    [Fact]
    public void GatedPlacementAcceptsAPlaceableProductOfAtLeastUTimesWmax()
    {
        Assert.True(Movement.AcceptsGated(0.5, 1.0, 0.5)); // equality accepts
        Assert.True(Movement.AcceptsGated(0.5, 1.0, 0.49));
        Assert.False(Movement.AcceptsGated(0.5, 1.0, 0.51));
        Assert.False(Movement.AcceptsGated(0.0, 1.0, 0.0)); // a zero product is never placeable
        Assert.False(Movement.AcceptsGated(double.NaN, 1.0, 0.0));
        Assert.True(Movement.AcceptsGated(0.2, 0.4, 0.5));
    }

    [Fact]
    public void GatedCandidatesNeedOceanAPlaceableMapAndAPlaceableProduct()
    {
        MovementParameters m = Parameters();
        Span<int> into = stackalloc int[9];
        ReadOnlySpan<double> w = Weight.AsSpan(0, NCells);

        // Around cell 5 (row 1, column 1), range 1: cells 0..2, 4..6, 8..10.
        int n = Movement.GatedWalkCandidates(m, Ny, Nx, 0, cell: 5, range: 1, w, into, out double total);

        // Out: 2 (map 0), 5 (land), 8 (weight NaN). In, row-major: 0, 1, 4, 6, 9, 10.
        Assert.Equal(new[] { 0, 1, 4, 6, 9, 10 }, into[..n].ToArray());
        Assert.Equal(0.1 + 0.7 + 0.3 + 0.2 + 0.4 + 0.8, total); // added in that order
        Assert.NotEqual(0.8 + 0.4 + 0.2 + 0.3 + 0.7 + 0.1, total); // the order is part of the value

        // Around cell 3 (row 0, column 3), range 1: cells 2, 3, 6, 7. 3 has weight 0, 7 a NaN map.
        n = Movement.GatedWalkCandidates(m, Ny, Nx, 0, cell: 3, range: 1, w, into, out total);
        Assert.Equal(new[] { 6 }, into[..n].ToArray());
        Assert.Equal(0.2, total);
    }

    [Fact]
    public void TheWeightedChoiceIsTheFirstCandidateWhoseRunningSumReachesR()
    {
        ReadOnlySpan<double> w = Weight.AsSpan(0, NCells);
        int[] cand = [0, 1, 4, 6, 9, 10]; // running sums 0.1, 0.8, 1.1, 1.3, 1.7, 2.5
        Assert.Equal(0, Movement.WeightedChoice(Map, w, cand, 0.0));
        Assert.Equal(0, Movement.WeightedChoice(Map, w, cand, 0.1)); // reaching counts
        Assert.Equal(1, Movement.WeightedChoice(Map, w, cand, 0.5));
        Assert.Equal(4, Movement.WeightedChoice(Map, w, cand, 0.9));
        Assert.Equal(9, Movement.WeightedChoice(Map, w, cand, 1.5));
        Assert.Equal(10, Movement.WeightedChoice(Map, w, cand, 2.4));
        Assert.Equal(10, Movement.WeightedChoice(Map, w, cand, 99.0)); // never reached: the last one
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Bundle(bool gateEntries, int[]? frames = null)
    {
        var c = new SimulationConfig(1);
        c.SetArray("movement.method", DType.U1, Pin(new byte[] { 2, 2 }), 2);
        c.SetArray("movement.walk_range", DType.I4, Pin(new[] { 1, 1 }), 2);
        c.SetInt("movement.maps.n_species", 1);
        c.SetArray("movement.maps.stack", DType.F8, Pin(Map), 1, Ny, Nx);
        c.SetArray("movement.maps.max_proba", DType.F8, Pin(new[] { 0.0 }), 1);
        c.SetArray("movement.maps.is_null", DType.U1, Pin(new byte[] { 0 }), 1);
        c.SetArray("movement.maps.species_offset", DType.I4, Pin(new[] { 0, -1 }), 2);
        c.SetArray("movement.maps.index.offsets", DType.I8, Pin(new long[] { 0, 1, 1 }), 3);
        c.SetArray("movement.maps.index.shape", DType.I4, Pin(new[] { 1, 1, 0, 0 }), 2, 2);
        c.SetArray("movement.maps.index", DType.I4, Pin(new[] { 0 }), 1);
        c.SetArray("movement.random.domain.offsets", DType.I4, Pin(new[] { 0, 0, 0 }), 3);
        c.SetArray("movement.random.domain.cells", DType.I4, Pin(new int[2]), 0, 2);
        if (gateEntries)
        {
            c.SetArray("movement.salinity.species", DType.U1, Pin(new byte[] { 1, 0 }), 2);
            c.SetArray("movement.salinity.weight", DType.F8, Pin(Weight), 2, NCells);
            c.SetArray("movement.salinity.frame_by_step", DType.I4, Pin(frames ?? FrameByStep), NSteps);
            c.SetArray("movement.salinity.wmax", DType.F8, Pin(Wmax), 1, 2);
        }

        return c;
    }

    private static (MovementParameters Parameters, IReadOnlyList<string> Problems) Read(SimulationConfig c, bool gate)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Movement(r, NSteps, nFocal: 2, nTotal: 2, Ny, Nx, Ocean, gate), r.Problems);
    }

    [Fact]
    public void TheGateTablesAreReadWhenTheFeatureIsDeclared()
    {
        (MovementParameters p, IReadOnlyList<string> problems) = Read(Bundle(gateEntries: true), gate: true);
        Assert.Empty(problems);
        Assert.Equal(new byte[] { 1, 0 }, p.SalinitySpecies);
        Assert.Equal(2, p.SalinityFrames);
        Assert.Equal(Weight, p.SalinityWeight);
        Assert.Equal(FrameByStep, p.SalinityFrameByStep);
        Assert.Equal(Wmax, p.SalinityWmax);
    }

    [Fact]
    public void ADeclaredGateWithoutItsTablesIsAProblemPerEntry()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(gateEntries: false), gate: true);
        Assert.Equal(4, problems.Count);
        Assert.All(problems, m => Assert.Contains("movement.salinity.", m, StringComparison.Ordinal));
    }

    [Fact]
    public void GateTablesWithoutTheFeatureAreRefusedNotIgnored()
    {
        (MovementParameters p, IReadOnlyList<string> problems) = Read(Bundle(gateEntries: true), gate: false);
        Assert.Equal(4, problems.Count);
        Assert.All(problems, m => Assert.Contains("does not declare feature salinity_gate", m, StringComparison.Ordinal));
        Assert.Empty(p.SalinitySpecies); // and the gate is not applied
    }

    [Fact]
    public void ASalinityFrameOutsideTheTableIsAProblem()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(gateEntries: true, frames: [0, 2, 1]), gate: true);
        string m = Assert.Single(problems);
        Assert.Contains("movement.salinity.frame_by_step: values must be in [0, 2)", m, StringComparison.Ordinal);
    }
}
