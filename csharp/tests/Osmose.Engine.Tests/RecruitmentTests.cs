using System.Runtime.InteropServices;
using Osmose.Engine.Step;

namespace Osmose.Engine.Tests;

/// <summary>Plan task 2.3: recruitment regulation (features <c>stock_recruitment</c>,
/// <c>seeding_linear</c>, <c>rv_gate</c>). The bit-exact checks against the Python engine's
/// <c>reproduction</c> on real states are on the Python side (tests/test_csharp_life_cycle.py);
/// these pin each formula, the order of the three stages, and the bundle contract.</summary>
public sealed class RecruitmentTests : IDisposable
{
    private const int NFocal = 5;
    private const int NSteps = 30; // two model years at 24 steps a year
    private const int Ndt = 24;
    private const double Linear = 7.3e9;
    private const double Ssb = 1234.5;

    private static readonly double[] Half = [900.0, 2100.0, 900.0, 2100.0, 900.0];
    private static readonly double[] Beta = [1.63, 1.63, 1.63, 1.63, 1.63];
    private static readonly double[] GateByYear = [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25];
    private static readonly byte[] Allee = [0, 0, 1, 0, 1];
    private static readonly double[] S50 = [0.0, 0.0, 700.0, 0.0, 3000.0]; // below and above the stock
    private static readonly double[] Theta = [1.0, 1.0, 2.5, 1.0, 4.0];

    private readonly List<GCHandle> _pins = [];

    public void Dispose()
    {
        foreach (GCHandle h in _pins)
        {
            h.Free();
        }
    }

    private static ReproductionParameters Parameters(byte type, int seedingMode = 0, bool gate = false, bool allee = false) => new()
    {
        SrType = [type, type, type, type, type],
        SrSsbHalf = Half,
        ShepherdBeta = Beta,
        SeedingMode = seedingMode,
        RvEnabled = gate ? [0, 0, 1, 1, 0] : [],
        RvFactorByYear = gate ? GateByYear : [],
        DepensationEnabled = allee ? Allee : [],
        DepensationS50 = allee ? S50 : [],
        DepensationTheta = allee ? Theta : [],
    };

    private static double AlleeTerm(int sp, double ssb = Ssb) =>
        Math.Pow(ssb, Theta[sp]) / (Math.Pow(S50[sp], Theta[sp]) + Math.Pow(ssb, Theta[sp]));

    private static double Eggs(ReproductionParameters r, int sp, int year = 0, bool seeded = false, double ssb = Ssb) =>
        LifeCycle.RegulateRecruitment(r, sp, NFocal, year, Linear, ssb, seeded);

    [Fact]
    public void WithoutACurveTheLinearCountPassesThrough()
    {
        Assert.Equal(Linear, Eggs(Parameters(0), 0));
    }

    [Fact]
    public void TheDepensationGateIsTheAlleeTermOfTheStock()
    {
        ReproductionParameters r = Parameters(0, allee: true);
        Assert.Equal(Linear * AlleeTerm(2), Eggs(r, 2)); // stock above its S50: more than half
        Assert.Equal(Linear * AlleeTerm(4), Eggs(r, 4)); // stock below its S50: less than half
        Assert.True(AlleeTerm(2) > 0.5 && AlleeTerm(4) < 0.5);
        Assert.Equal(0.5, LifeCycle.DepensationFactor(700.0, 700.0, 2.5)); // one half at S50
        Assert.Equal(0.0, LifeCycle.DepensationFactor(0.0, 700.0, 2.5)); // nothing from an empty stock
        Assert.Equal(Linear, Eggs(r, 1)); // a species the gate is not on
        Assert.Equal(Linear, Eggs(r, 2, seeded: true)); // a seeded species is spared
    }

    [Fact]
    public void TheDepensationGateComesAfterTheReproductiveVolumeGate()
    {
        // Species 2 has both. The order of the two multiplications is production's, and it is
        // what makes the result equal to the last bit.
        // A volume factor that is no power of two: multiplying by 0.5 or 0.25 is exact, and with
        // such a factor no order of the two multiplications could ever show.
        const double Volume = 0.3;
        ReproductionParameters r = Parameters(0, gate: true, allee: true) with
        {
            RvFactorByYear = [.. Enumerable.Repeat(Volume, 2 * NFocal)],
        };
        int otherOrderDiffers = 0;
        for (int i = 0; i < 200; i++)
        {
            double ssb = 400.0 + (37.0 * i);
            double allee = AlleeTerm(2, ssb);
            Assert.Equal((Linear * Volume) * allee, Eggs(r, 2, year: 1, ssb: ssb));
            if ((Linear * Volume) * allee != Linear * (Volume * allee))
            {
                otherOrderDiffers++;
            }
        }

        // Or the loop above could not tell the orders apart, and would pin nothing.
        Assert.True(otherOrderDiffers > 0);
        Assert.Equal(Linear * Volume, Eggs(r, 3, year: 0)); // the volume gate alone
        Assert.Equal(Linear * AlleeTerm(4), Eggs(r, 4, year: 1)); // the depensation gate alone
    }

    [Fact]
    public void EachCurveIsItsFormula()
    {
        for (int sp = 0; sp < 2; sp++) // stock above (sp 0) and below (sp 1) the characteristic SSB
        {
            double x = Ssb / Half[sp];
            Assert.Equal(Linear / (1.0 + x), Eggs(Parameters(1), sp));
            Assert.Equal(Linear * Math.Exp(-Ssb / Half[sp]), Eggs(Parameters(2), sp));
            Assert.Equal(Linear / (1.0 + Math.Pow(x, 1.63)), Eggs(Parameters(4), sp));
        }

        Assert.Equal(Linear * (Half[0] / Ssb), Eggs(Parameters(3), 0)); // above the breakpoint: capped
        Assert.Equal(Linear, Eggs(Parameters(3), 1)); // below it: no correction
        Assert.Equal(Linear, Eggs(Parameters(3), 0, ssb: Half[0])); // at it: no correction either
    }

    [Fact]
    public void ACurveIsSkippedAtAnEmptyStock()
    {
        foreach (byte type in new byte[] { 1, 2, 3, 4 })
        {
            Assert.Equal(Linear, Eggs(Parameters(type), 0, ssb: 0.0));
        }
    }

    [Fact]
    public void ASeededSpeciesKeepsTheCurveByDefaultAndSkipsItInLinearMode()
    {
        double curved = Linear / (1.0 + (Ssb / Half[0]));
        Assert.Equal(curved, Eggs(Parameters(1, seedingMode: 0), 0, seeded: true));
        Assert.Equal(Linear, Eggs(Parameters(1, seedingMode: 1), 0, seeded: true));
        Assert.Equal(curved, Eggs(Parameters(1, seedingMode: 1), 0, seeded: false)); // real stock: the curve
    }

    [Fact]
    public void TheGateScalesEnabledUnseededSpeciesByTheirYearsFactorAfterTheCurve()
    {
        ReproductionParameters r = Parameters(1, gate: true);
        double curved2 = Linear / (1.0 + (Ssb / Half[2]));
        Assert.Equal(curved2 * 0.5, Eggs(r, 2, year: 0));
        Assert.Equal(curved2 * 0.25, Eggs(r, 2, year: 1));
        Assert.Equal(curved2, Eggs(r, 2, year: 1, seeded: true)); // a seeded step is never gated
        Assert.Equal(Linear / (1.0 + (Ssb / Half[0])), Eggs(r, 0, year: 1)); // not enabled
    }

    private long Pin(Array a)
    {
        GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
        _pins.Add(h);
        return h.AddrOfPinnedObject();
    }

    private SimulationConfig Bundle(byte type = 0, bool curveEntries = false, bool gateEntries = false, int seedingMode = 0, bool alleeEntries = false, double[]? s50 = null)
    {
        var c = new SimulationConfig(1);
        double[] ones = [1.0, 1.0, 1.0, 1.0, 1.0];
        foreach (string name in new[] { "sex_ratio", "relative_fecundity", "maturity_size", "seeding_biomass", "egg_length", "egg_weight" })
        {
            c.SetArray($"reproduction.{name}", DType.F8, Pin(ones), NFocal);
        }

        c.SetArray("reproduction.maturity_age_dt", DType.I4, Pin(new int[NFocal]), NFocal);
        c.SetArray("reproduction.seeding_max_step", DType.I4, Pin(new int[NFocal]), NFocal);
        c.SetArray("reproduction.season_by_step", DType.F8, Pin(new double[NSteps * NFocal]), NSteps, NFocal);
        c.SetArray("reproduction.sr.type", DType.U1, Pin(new[] { type, type, type, type, type }), NFocal);
        c.SetInt("reproduction.seeding_mode", seedingMode);
        if (curveEntries)
        {
            c.SetArray("reproduction.sr.ssb_half", DType.F8, Pin(Half), NFocal);
            c.SetArray("reproduction.sr.shepherd_beta", DType.F8, Pin(Beta), NFocal);
        }

        if (gateEntries)
        {
            c.SetArray("reproduction.rv.enabled", DType.U1, Pin(new byte[] { 0, 0, 1, 1, 0 }), NFocal);
            c.SetArray("reproduction.rv.factor_by_year", DType.F8, Pin(GateByYear), 2, NFocal);
        }

        if (alleeEntries)
        {
            c.SetArray("reproduction.depensation.enabled", DType.U1, Pin((byte[])Allee.Clone()), NFocal);
            c.SetArray("reproduction.depensation.s50", DType.F8, Pin(s50 ?? S50), NFocal);
            c.SetArray("reproduction.depensation.theta", DType.F8, Pin(Theta), NFocal);
        }

        return c;
    }

    private static (ReproductionParameters Parameters, IReadOnlyList<string> Problems) Read(
        SimulationConfig c, bool curves = false, bool rvGate = false, bool seedingLinear = false, bool depensationGate = false)
    {
        var r = new BundleReader(c, 1);
        return (ModelReader.Reproduction(r, NSteps, NFocal, Ndt, curves, rvGate, seedingLinear, depensationGate), r.Problems);
    }

    [Fact]
    public void TheDepensationGateReadsItsEntriesAndRefusesThemUndeclared()
    {
        (ReproductionParameters p, IReadOnlyList<string> none) = Read(Bundle(alleeEntries: true), depensationGate: true);
        Assert.Empty(none);
        Assert.Equal(Allee, p.DepensationEnabled);
        Assert.Equal(S50, p.DepensationS50);
        Assert.Equal(Theta, p.DepensationTheta);

        (_, IReadOnlyList<string> undeclared) = Read(Bundle(alleeEntries: true));
        Assert.Equal(3, undeclared.Count);
        Assert.All(undeclared, m => Assert.Contains("present, but the bundle does not declare feature depensation_gate", m, StringComparison.Ordinal));

        (_, IReadOnlyList<string> missing) = Read(Bundle(), depensationGate: true);
        Assert.Equal(3, missing.Count); // a declared feature with none of its three entries

        // A gated species with no S50 would divide by a zero power: refused, as the Python
        // loader refuses it. An ungated species may hold anything.
        (_, IReadOnlyList<string> bad) = Read(Bundle(alleeEntries: true, s50: [0.0, 0.0, 0.0, 0.0, 3000.0]), depensationGate: true);
        string only = Assert.Single(bad);
        Assert.Contains("reproduction.depensation: species 2 has s50 0", only, StringComparison.Ordinal);
    }

    [Fact]
    public void DeclaredFeaturesReadTheirEntries()
    {
        (ReproductionParameters p, IReadOnlyList<string> problems) = Read(
            Bundle(type: 4, curveEntries: true, gateEntries: true, seedingMode: 1),
            curves: true, rvGate: true, seedingLinear: true);
        Assert.Empty(problems);
        Assert.Equal(Half, p.SrSsbHalf);
        Assert.Equal(Beta, p.ShepherdBeta);
        Assert.Equal(1, p.SeedingMode);
        Assert.Equal(new byte[] { 0, 0, 1, 1, 0 }, p.RvEnabled);
        Assert.Equal(GateByYear, p.RvFactorByYear); // two years: ceil(30 / 24)
    }

    [Fact]
    public void APlainBundleReadsWithoutProblemsAndWithoutRegulation()
    {
        (ReproductionParameters p, IReadOnlyList<string> problems) = Read(Bundle());
        Assert.Empty(problems);
        Assert.Empty(p.SrSsbHalf);
        Assert.Empty(p.RvEnabled);
    }

    [Fact]
    public void RegulationWithoutItsFeatureIsRefusedNotIgnored()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(type: 2, curveEntries: true, gateEntries: true, seedingMode: 1));
        string all = string.Join("\n", problems);
        Assert.Contains("reproduction.sr.type: a curve is set, but the bundle does not declare feature stock_recruitment", all, StringComparison.Ordinal);
        Assert.Contains("reproduction.sr.ssb_half: present, but the bundle does not declare feature stock_recruitment", all, StringComparison.Ordinal);
        Assert.Contains("reproduction.rv.enabled: present, but the bundle does not declare feature rv_gate", all, StringComparison.Ordinal);
        Assert.Contains("reproduction.seeding_mode: linear, but the bundle does not declare feature seeding_linear", all, StringComparison.Ordinal);
        Assert.Equal(6, problems.Count); // sr.type, two sr entries, two rv entries, the seeding mode
    }

    [Fact]
    public void ADeclaredFeatureWithoutItsEntriesIsAProblemPerEntry()
    {
        (_, IReadOnlyList<string> problems) = Read(Bundle(type: 1), curves: true, rvGate: true);
        Assert.Equal(4, problems.Count);
        Assert.All(problems, m => Assert.Contains("missing", m, StringComparison.Ordinal));
    }

    [Fact]
    public void AnUnknownCurveOrSeedingModeIsAProblem()
    {
        (_, IReadOnlyList<string> p1) = Read(Bundle(type: 5, curveEntries: true), curves: true);
        Assert.Contains(p1, m => m.Contains("reproduction.sr.type: values must be 0 (none) to 4 (shepherd)", StringComparison.Ordinal));
        (_, IReadOnlyList<string> p2) = Read(Bundle(seedingMode: 2), seedingLinear: true);
        Assert.Contains(p2, m => m.Contains("reproduction.seeding_mode: 2 must be 0", StringComparison.Ordinal));
    }
}
