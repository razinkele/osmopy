using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// The life cycle after mortality (plan task 1.7, spec 6.8), each in the Python engine's order and
/// arithmetic: the sub-one-fish floor, von Bertalanffy growth, aging, and reproduction (spawning
/// stock, seeding, egg count, age increment, new egg schools). Phase 1: von Bertalanffy only, no
/// stock-recruitment curve and no recruitment gates; those are later-phase features the engine
/// refuses.
/// </summary>
public static class LifeCycle
{
    private const double TonnesToGrams = 1_000_000.0;

    /// <summary><c>_floor_sub_one_fish</c>: a school holding fewer than one fish (but more than
    /// none) is dead. Returns the abundance zeroed, for the death-accounting invariant.</summary>
    public static double Floor(Span<SchoolCore> core)
    {
        double zeroed = 0.0;
        for (int i = 0; i < core.Length; i++)
        {
            double a = core[i].Abundance;
            if (a > 0 && a < 1.0)
            {
                zeroed += a;
                core[i].Abundance = 0.0;
            }
        }

        return zeroed;
    }

    /// <summary><c>expected_length_vb</c>: the egg size at age 0, linear to the threshold age, the
    /// von Bertalanffy curve after. Tier L (<c>exp</c>).</summary>
    public static double ExpectedLengthVb(GrowthParameters g, int species, int ageDt, int nDtPerYear)
    {
        double ageYears = ageDt / (double)nDtPerYear;
        double linf = g.Linf[species], k = g.K[species], t0 = g.T0[species];
        double egg = g.EggSize[species], threshold = g.VbThresholdAge[species];
        if (ageDt == 0)
        {
            return egg;
        }

        if (ageYears < threshold)
        {
            double atThreshold = linf * (1 - Math.Exp(-k * (threshold - t0)));
            double frac = threshold > 0 ? ageYears / threshold : 1.0;
            return egg + ((atThreshold - egg) * frac);
        }

        return linf * (1 - Math.Exp(-k * (ageYears - t0)));
    }

    /// <summary>
    /// <c>growth</c>: the length increment <c>L(age + 1) - L(age)</c>, gated by the step's
    /// predation success against the critical rate; age-0 and out-of-domain schools take the
    /// full increment. Then <c>L = min(L + growth, lmax)</c> and <c>W = c * L^b * 1e-6</c>.
    /// </summary>
    public static void Growth(EngineParameters p, Span<SchoolCore> core, ReadOnlySpan<SchoolStep> step)
    {
        GrowthParameters g = p.Growth;
        int ndt = p.NDtPerYear;
        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            int sp = c.SpeciesId;
            if (g.Class[sp] != 0)
            {
                throw new NotSupportedException("Gompertz growth is the gompertz_growth feature (phase 3)");
            }

            double delta = ExpectedLengthVb(g, sp, c.AgeDt + 1, ndt) - ExpectedLengthVb(g, sp, c.AgeDt, ndt);
            double csr = g.CriticalSuccessRate[sp], sr = step[i].PredSuccess;
            double maxDelta = g.DeltaLmaxFactor[sp] * delta;
            double denom = csr < 1.0 ? 1.0 - csr : 1.0;
            double factor = sr >= csr ? maxDelta * (sr - csr) / denom : 0.0;
            if (csr >= 1.0 && sr >= 1.0)
            {
                factor = maxDelta;
            }

            if (c.AgeDt == 0 || (c.Flags & SchoolFlags.IsOut) != 0)
            {
                factor = delta;
            }

            c.Length = Math.Min(c.Length + factor, g.Lmax[sp]);
            c.Weight = g.ConditionFactor[sp] * Math.Pow(c.Length, g.AllometricPower[sp]) * 1e-6;
        }
    }

    /// <summary><c>aging_mortality</c>: a school at <c>age_dt >= lifespan_dt - 1</c> dies of old
    /// age (aging runs before reproduction increments the age). Tier E.</summary>
    public static void Aging(EngineParameters p, Span<SchoolCore> core, Span<SchoolLedger> ledger)
    {
        int[] lifespan = p.Species.LifespanDt;
        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            if (c.AgeDt >= lifespan[c.SpeciesId] - 1)
            {
                ledger[i].DeadAging += c.Abundance;
                c.Abundance = 0.0;
            }
        }
    }

    /// <summary>
    /// <c>regulate_recruitment</c> for one species, in its order: the stock-recruitment curve
    /// (<c>apply_stock_recruitment</c>; skipped at <c>ssb &lt;= 0</c>), the linear seeding mode
    /// (a seeded species keeps the linear count), then the reproductive-volume gate and the
    /// depensation gate, in that order (both skipped on a seeded step, so that the bootstrap from
    /// an empty stock is neither scaled nor trapped). <paramref name="ssb"/> is the stock AFTER
    /// seeding. Ricker's <c>exp</c> and the two <c>pow</c> are the C library's (tier L against
    /// NumPy's <c>exp</c>).
    /// </summary>
    public static double RegulateRecruitment(
        ReproductionParameters r, int sp, int nFocal, int year, double linear, double ssb, bool seeded)
    {
        double eggs = linear;
        byte type = r.SrType[sp];
        if (type != 0 && ssb > 0.0)
        {
            double half = r.SrSsbHalf[sp];
            eggs = type switch
            {
                1 => linear / (1.0 + (ssb / half)),
                2 => linear * Math.Exp(-ssb / half),
                3 => ssb > half ? linear * (half / ssb) : linear,
                4 => linear / (1.0 + Math.Pow(ssb / half, r.ShepherdBeta[sp])),
                _ => throw new InvalidOperationException($"unknown stock-recruitment type {type}"),
            };
        }

        if (r.SeedingMode == 1 && seeded)
        {
            eggs = linear;
        }

        if (!seeded && r.RvEnabled.Length > 0 && r.RvEnabled[sp] != 0)
        {
            eggs *= r.RvFactorByYear[(year * nFocal) + sp];
        }

        if (!seeded && r.DepensationEnabled.Length > 0 && r.DepensationEnabled[sp] != 0)
        {
            eggs *= DepensationFactor(ssb, r.DepensationS50[sp], r.DepensationTheta[sp]);
        }

        return eggs;
    }

    /// <summary>
    /// <c>depensation_factor</c> for one species: the Allee term
    /// <c>S^theta / (S50^theta + S^theta)</c>, which is one half at <c>S = S50</c>, goes to 1
    /// above it and to 0 below it; 0 at an empty stock.
    /// </summary>
    public static double DepensationFactor(double ssb, double s50, double theta)
    {
        if (ssb <= 0.0)
        {
            return 0.0;
        }

        double s = Math.Pow(ssb, theta);
        return s / (Math.Pow(s50, theta) + s);
    }

    /// <summary>
    /// <c>reproduction</c>: the spawning stock of each focal species (mature: length and age at
    /// maturity, abundance positive), the seeding biomass when that stock is exactly 0 within the
    /// seeding period, <c>eggs = sex_ratio * fecundity * SSB * season * 1e6</c> regulated by
    /// <see cref="RegulateRecruitment"/>; then every existing school ages one step (and stops being
    /// an egg once fed), and the new egg schools are appended, unlocated, species by species.
    /// <paramref name="ssb"/> and <paramref name="eggs"/> are focal-length scratch and hold the
    /// step's values afterwards.
    /// </summary>
    public static void Reproduction(EngineParameters p, SchoolStore store, int step, Span<double> ssb, Span<double> eggs, Span<bool> seeded)
    {
        ReproductionParameters r = p.Reproduction;
        int nFocal = p.Species.NFocal;
        Span<SchoolCore> core = store.Core;
        ssb[..nFocal].Clear();
        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            int sp = c.SpeciesId;
            if (sp < nFocal && c.Length >= r.MaturitySize[sp] && c.AgeDt >= r.MaturityAgeDt[sp] && c.Abundance > 0)
            {
                ssb[sp] += c.Abundance * c.Weight;
            }
        }

        int year = step / p.NDtPerYear;
        for (int sp = 0; sp < nFocal; sp++)
        {
            seeded[sp] = ssb[sp] == 0.0 && step < r.SeedingMaxStep[sp];
            if (seeded[sp])
            {
                ssb[sp] = r.SeedingBiomass[sp];
            }

            double linear = r.SexRatio[sp] * r.RelativeFecundity[sp] * ssb[sp] * r.SeasonByStep[(step * nFocal) + sp] * TonnesToGrams;
            eggs[sp] = RegulateRecruitment(r, sp, nFocal, year, linear, ssb[sp], seeded[sp]);
        }

        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            c.AgeDt += 1;
            c.Flags = c.AgeDt < c.FirstFeedingAgeDt ? c.Flags | SchoolFlags.IsEgg : c.Flags & ~SchoolFlags.IsEgg;
        }

        int[] nSchools = p.Species.NSchools;
        for (int sp = 0; sp < nFocal; sp++)
        {
            int n = nSchools[sp];
            if (eggs[sp] <= 0 || n <= 0)
            {
                continue;
            }

            if (eggs[sp] < n)
            {
                n = 1; // fewer eggs than schools: one school holds them all
            }

            var egg = new SchoolCore
            {
                SpeciesId = sp,
                AgeDt = 0,
                Abundance = eggs[sp] / n,
                Length = r.EggLength[sp],
                Weight = r.EggWeight[sp],
                TrophicLevel = 0.0,
                FirstFeedingAgeDt = 1,
                Flags = SchoolFlags.IsEgg | (seeded[sp] ? SchoolFlags.FromSeeding : 0),
            };
            for (int j = 0; j < n; j++)
            {
                store.Append(egg, new SchoolLife { CellId = -1 });
            }
        }
    }
}
