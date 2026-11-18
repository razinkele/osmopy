using System.Runtime.InteropServices;
using Osmose.Engine.Rng;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 1.1: the feature catalogue and the header of <see cref="EngineParameters"/>.</summary>
public sealed class ParametersTests : IDisposable
{
    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Header(int[]? features = null, Action<SimulationConfig>? tweak = null, int version = 1)
    {
        var c = new SimulationConfig(version);
        int[] f = features ?? [0];
        c.SetArray("contract.features", DType.I4, Pin(f), f.Length);
        c.SetInt("contract.rng.scheme", Philox.Scheme);
        c.SetInt("run.seed", 42);
        c.SetInt("run.threads", 1);
        c.SetFloat("run.deadline_seconds", 0.0);
        c.SetInt("run.checks", 0);
        c.SetInt("time.n_steps", 48);
        c.SetInt("time.n_dt_per_year", 24);
        c.SetInt("time.n_subdt", 10);
        c.SetInt("grid.ny", 2);
        c.SetInt("grid.nx", 3);
        c.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[] { 1, 0, 1, 1, 1, 0 }), 2, 3);
        tweak?.Invoke(c);
        return c;
    }

    [Fact]
    public void CatalogueLoadsWithDenseIdsAndCoreFirst()
    {
        Assert.NotEmpty(FeatureCatalog.All);
        Assert.Equal("core", FeatureCatalog.Core.Name);
        for (int i = 0; i < FeatureCatalog.All.Count; i++)
        {
            Assert.Equal(i, FeatureCatalog.All[i].Id);
        }

        Assert.Contains(1, FeatureCatalog.ContractVersions);
        Assert.All(FeatureCatalog.All, f => Assert.Contains(f.Status, new[] { "planned", "refused", "not_planned" }));
    }

    [Fact]
    public void ACoreOnlyHeaderPassesTheFeatureCheck()
    {
        // core is implemented (task 1.8): the header and feature checks pass, and the refusal
        // that remains is the model stage, for the tables this header-only bundle lacks.
        var ex = Assert.Throws<BundleValidationException>(() => EngineParameters.From(Header()));
        Assert.DoesNotContain("core", ex.Message, StringComparison.Ordinal);
        Assert.Contains("species.", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryUnsupportedFeatureIsNamedWithItsReason()
    {
        int refused = FeatureCatalog.All.First(f => f.Status == "refused").Id;
        int notPlanned = FeatureCatalog.All.First(f => f.Status == "not_planned").Id;
        int phase2 = FeatureCatalog.All.First(f => f.Status == "planned" && f.Phase == 2).Id;
        var ex = Assert.Throws<UnsupportedFeatureException>(
            () => EngineParameters.From(Header([0, refused, notPlanned, phase2])));
        Assert.Equal(3, ex.Features.Count); // core is implemented: not among them
        Assert.DoesNotContain(ex.Features, f => f.Name == "core");
        Assert.Contains("refused: spec Q4", ex.Message, StringComparison.Ordinal);
        Assert.Contains("use the Python engine", ex.Message, StringComparison.Ordinal);
        Assert.Contains("phase 2", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryHeaderProblemIsListedInOneException()
    {
        var c = new SimulationConfig(contractVersion: 9);
        c.SetArray("contract.features", DType.I4, Pin(new[] { 1, 1, 999 }), 3); // no core, duplicate, bad id
        c.SetInt("contract.rng.scheme", Philox.Scheme + 1);
        c.SetInt("run.threads", 0);
        c.SetFloat("run.deadline_seconds", -1.0);
        c.SetInt("time.n_steps", 0);
        c.SetInt("time.n_dt_per_year", 24);
        c.SetInt("grid.ny", 2);
        c.SetInt("grid.nx", 2);
        c.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[] { 1, 2, 1, 1 }), 2, 2);

        var ex = Assert.Throws<BundleValidationException>(() => EngineParameters.From(c));
        string m = ex.Message;
        foreach (string needle in new[]
        {
            "contract version 9", "contract.rng.scheme", "run.threads: 0", "run.deadline_seconds",
            "time.n_steps: 0", "run.seed: missing", "time.n_subdt: missing", "run.checks: missing",
            "grid.ocean_mask: values", "listed twice", "999 is not a catalogue id", "core is missing",
        })
        {
            Assert.Contains(needle, m, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void OceanMaskShapeMustMatchTheGrid()
    {
        var ex = Assert.Throws<BundleValidationException>(() => EngineParameters.From(Transposed()));
        Assert.Contains("grid.ocean_mask: shape (2, 3), expected (3, 2)", ex.Message, StringComparison.Ordinal);
    }

    private SimulationConfig Transposed()
    {
        // Same mask, grid declared transposed.
        var c = new SimulationConfig(1);
        c.SetArray("contract.features", DType.I4, Pin(new[] { 0 }), 1);
        c.SetInt("contract.rng.scheme", Philox.Scheme);
        c.SetInt("run.seed", 1);
        c.SetInt("run.threads", 1);
        c.SetFloat("run.deadline_seconds", 0.0);
        c.SetInt("run.checks", 0);
        c.SetInt("time.n_steps", 1);
        c.SetInt("time.n_dt_per_year", 24);
        c.SetInt("time.n_subdt", 10);
        c.SetInt("grid.ny", 3);
        c.SetInt("grid.nx", 2);
        c.SetArray("grid.ocean_mask", DType.U1, Pin(new byte[6]), 2, 3);
        return c;
    }
}
