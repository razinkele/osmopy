using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// Per-step preparation before mortality (plan task 1.4, spec 6.3): resource carrying capacity,
/// feeding stages, stage-accessibility indices and per-school rates. Every operation is exact
/// IEEE arithmetic in the Python engine's order (tier E), so the results equal
/// <c>_precompute_effective_rates</c>, <c>compute_feeding_stages</c>,
/// <c>AccessibilityMatrix.compute_school_indices</c> and <c>ResourceState.update</c> bit for bit.
/// </summary>
public static class StepPreparation
{
    /// <summary>The resources of every cell for <paramref name="step"/>, cell-major
    /// (<c>[cell * nRsc + r]</c>, spec 4.3). The carrying capacity is
    /// <c>forcing[r, frame] * access</c> (times the oxygen factor for a coupled resource). By
    /// default the buffer is reset to it. With feature <c>depletable_ltl</c> the buffer's own
    /// content, what the last step's predation left, is regrown toward it
    /// (<see cref="Regrow"/>), so the buffer must be the run's and must start at zero.</summary>
    public static void UpdateResources(EngineParameters p, int step, Span<double> cellMajor) =>
        UpdateResources(p.Resources, p.Ny * p.Nx, step, cellMajor);

    /// <summary>As <see cref="UpdateResources(EngineParameters, int, Span{double})"/>, on the
    /// resource group alone.</summary>
    public static void UpdateResources(ResourceParameters rp, int nCells, int step, Span<double> cellMajor)
    {
        int nRsc = rp.N;
        if (nRsc == 0)
        {
            return;
        }

        if (cellMajor.Length < nRsc * nCells)
        {
            throw new ArgumentException($"needs {nRsc * nCells} values", nameof(cellMajor));
        }

        for (int r = 0; r < nRsc; r++)
        {
            int frame = rp.FrameByStep[(step * nRsc) + r];
            double access = rp.AccessByStep[(step * nRsc) + r];
            int baseIdx = ((r * rp.NFrames) + frame) * nCells;
            double rate = rp.Depletable ? rp.RegrowthRate[r] : 0.0;
            if (rp.OxygenCoupled.Length > 0 && rp.OxygenCoupled[r] != 0)
            {
                // ResourceState.update: k_row = (forcing * multiplier * access) * f_o2_hill(...),
                // the same left-to-right product, on the builder's pre-evaluated factor. The
                // regrowth, if any, is toward the capacity so scaled.
                int oxygenBase = rp.OxygenFrameByStep[step] * nCells;
                for (int c = 0; c < nCells; c++)
                {
                    double k = rp.Forcing[baseIdx + c] * access * rp.OxygenFactor[oxygenBase + c];
                    int i = (c * nRsc) + r;
                    cellMajor[i] = rp.Depletable ? Regrow(cellMajor[i], k, rate, rp.DepletableFloor) : k;
                }

                continue;
            }

            for (int c = 0; c < nCells; c++)
            {
                double k = rp.Forcing[baseIdx + c] * access;
                int i = (c * nRsc) + r;
                cellMajor[i] = rp.Depletable ? Regrow(cellMajor[i], k, rate, rp.DepletableFloor) : k;
            }
        }
    }

    /// <summary>
    /// <c>logistic_regrow</c> for one cell: from the larger of the carried biomass and
    /// <c>floor * k</c>, and from <paramref name="k"/> at most, one logistic step toward
    /// <paramref name="k"/>, capped at it; 0 where there is no capacity. The arithmetic is the
    /// Python function's, in its order. Starting from the capacity at most (issue #337) keeps
    /// the result within [0, k]: the step from above the capacity has no lower bound.
    /// </summary>
    public static double Regrow(double carried, double k, double rate, double floor)
    {
        if (!(k > 0.0))
        {
            return 0.0;
        }

        double b = Math.Min(Math.Max(carried, floor * k), k);
        double grown = b + (rate * b * (1.0 - (b / k)));
        return Math.Min(grown, k);
    }

    /// <summary>Feeding stage and accessibility indices of every school (they depend only on
    /// species, age, size, weight and trophic level, so once per step).</summary>
    public static void StagesAndAccess(EngineParameters p, Span<SchoolCore> core)
    {
        PredationParameters pr = p.Predation;
        AccessParameters ac = p.Access;
        double ndt = p.NDtPerYear;
        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            int sp = c.SpeciesId;
            double value = pr.StageMetric[sp] switch
            {
                0 => c.Length,
                1 => c.AgeDt / ndt,
                2 => c.Weight * 1e6,
                _ => c.TrophicLevel,
            };
            int lo = pr.StageOffsets[sp], hi = pr.StageOffsets[sp + 1];
            c.FeedingStage = (byte)CountAtMost(pr.StageThresholds.AsSpan(lo, hi - lo), value);
            if (ac.Mode == 2)
            {
                double ageYears = c.AgeDt / ndt;
                c.PreyAccessRow = (short)StageIndex(ac.PreyOffsets, ac.PreyThresholds, ac.PreyIndex, sp, ageYears);
                c.PredAccessCol = (short)StageIndex(ac.PredOffsets, ac.PredThresholds, ac.PredIndex, sp, ageYears);
            }
        }
    }

    /// <summary>
    /// Per-school sub-step rates for <paramref name="step"/>, written to <see cref="SchoolStep"/>:
    /// starvation <c>rate / (ndt * n_subdt)</c>; additional <c>rate / (ndt * n_subdt)</c>; fishing
    /// <c>rate * selectivity * spatial * season / n_subdt</c> with seasonality, else
    /// <c>rate * selectivity * spatial / (ndt * n_subdt)</c>. Background and pre-feeding schools and
    /// negative values are zeroed; NaN and infinities become zero; the discard fraction applies
    /// where the fishing rate is positive.
    /// </summary>
    public static void Rates(EngineParameters p, Span<SchoolCore> core, ReadOnlySpan<SchoolLife> life, Span<SchoolStep> step, int t)
    {
        int nTotal = p.Species.NTotal, nFocal = p.Species.NFocal, nsub = p.NSubdt, ndt = p.NDtPerYear;
        int denom = ndt * nsub;
        int nx = p.Nx, ny = p.Ny;
        FishingParameters f = p.Fishing;
        double[] add = p.Mortality.AdditionalRateByStep;
        double[] termsBuf = f.Fishery is { } fmb ? new double[fmb.Count] : [];
        for (int i = 0; i < core.Length; i++)
        {
            ref SchoolCore c = ref core[i];
            ref SchoolStep s = ref step[i];
            int sp = c.SpeciesId;
            bool exempt = (c.Flags & SchoolFlags.IsBackground) != 0 || c.AgeDt < c.FirstFeedingAgeDt;

            s.RateStarvation = Exempt(life[i].StarvationRate / denom, exempt);
            s.RateAdditional = Finite(Exempt(add[(t * nTotal) + sp], exempt) / denom);

            double fishing = 0.0, discard = 0.0;
            if (f.Enabled && f.Fishery is { } fm)
            {
                // The per-fishery path: osmose.engine.fisheries.per_fishery_rates, term for term and
                // in the same floating-point order, then _per_fishery_effective's exempt/finite.
                double total = 0.0, disc = 0.0;
                if (sp < nFocal)
                {
                    Span<double> terms = termsBuf.AsSpan(0, fm.Count);
                    FisheryTerms(p, fm, c, life[i].CellId, t, terms);
                    for (int k = 0; k < fm.Count; k++)
                    {
                        total += terms[k];
                        disc += terms[k] * fm.Discard[(sp * fm.Count) + k];
                    }
                }

                double frac = total > 0 ? disc / total : 0.0;
                fishing = Finite(Exempt(total, exempt));
                discard = fishing > 0 ? frac : 0.0;
            }
            else if (f.Enabled)
            {
                double rate = f.RateByStep[(t * nTotal) + sp];
                double sel = 1.0, spatial = 1.0;
                if (sp < nFocal)
                {
                    sel = Selectivity(f, sp, c, ndt);
                    int map = f.MapIndex[sp];
                    if (map >= 0)
                    {
                        int cell = life[i].CellId;
                        int cy = cell >= 0 ? cell / nx : -1, cx = cell >= 0 ? cell % nx : -1;
                        double v = (cy >= 0 && cy < ny && cx >= 0 && cx < nx) ? f.Maps[(((map * ny) + cy) * nx) + cx] : 0.0;
                        spatial = double.IsNaN(v) || v <= 0 ? 0.0 : v;
                    }
                }

                fishing = f.SeasonByStep is { } season
                    ? rate * sel * spatial * season[(t * nTotal) + sp] / nsub
                    : rate * sel * spatial / denom;
                fishing = Finite(Exempt(fishing, exempt));
                if (f.DiscardRate is { } d && fishing > 0)
                {
                    discard = d[sp];
                }
            }

            s.RateFishing = fishing;
            s.DiscardFraction = discard;
        }
    }

    /// <summary>Each fishery's rate term for one focal school at step <paramref name="t"/>
    /// (<c>osmose.engine.fisheries.per_fishery_terms</c>), 0 where the fishery does not fish it.
    /// <see cref="Rates"/> sums them in fishery order; the yield-by-fishery output splits a
    /// school's catch by them.</summary>
    public static void FisheryTerms(EngineParameters p, FisheryParameters fm, in SchoolCore c, int cell, int t, Span<double> terms)
    {
        int nsub = p.NSubdt, ndt = p.NDtPerYear, nx = p.Nx, ny = p.Ny;
        int denom = ndt * nsub;
        int sp = c.SpeciesId;
        int cy = cell >= 0 ? cell / nx : -1, cx = cell >= 0 ? cell % nx : -1;
        int ts = Math.Min(t, fm.NSteps - 1);
        terms.Clear();
        for (int k = 0; k < fm.Count; k++)
        {
            double q = fm.Catchability[(sp * fm.Count) + k];
            if (!(q > 0) || fm.Rate[k] <= 0)
            {
                continue;
            }

            int mi = fm.MapIndex[(k * fm.NSteps) + ts];
            if (mi == FisheryParameters.Off)
            {
                continue;
            }

            double sel = FisherySelectivity(fm, k, c, ndt);
            double spatial = 1.0;
            if (mi != FisheryParameters.NoMap)
            {
                double v = (cy >= 0 && cy < ny && cx >= 0 && cx < nx) ? fm.Maps[(((mi * ny) + cy) * nx) + cx] : 0.0;
                spatial = double.IsNaN(v) || v <= 0 ? 0.0 : v;
            }

            double b = q * fm.Rate[k] * sel * spatial;
            terms[k] = fm.Season is { } se ? b * se[(k * ndt) + (t % ndt)] / nsub : b / denom;
        }
    }

    private static double FisherySelectivity(FisheryParameters fm, int k, in SchoolCore c, int ndt)
    {
        int type = fm.SelectivityType[k];
        if (type == 0)
        {
            return (c.AgeDt / (double)ndt) < fm.SelectivityA50[k] ? 0.0 : 1.0;
        }

        if (type == 1)
        {
            throw new NotSupportedException("logistic selectivity is the selectivity_logistic feature (phase 3)");
        }

        double l50 = fm.SelectivityL50[k];
        return l50 > 0 && c.Length < l50 ? 0.0 : 1.0;
    }

    private static double Selectivity(FishingParameters f, int sp, in SchoolCore c, int ndt)
    {
        int type = f.SelectivityType[sp];
        if (type == 0)
        {
            return (c.AgeDt / (double)ndt) < f.SelectivityA50[sp] ? 0.0 : 1.0;
        }

        if (type == 1)
        {
            throw new NotSupportedException("logistic selectivity is the selectivity_logistic feature (phase 3)");
        }

        double l50 = f.SelectivityL50[sp];
        return l50 > 0 && c.Length < l50 ? 0.0 : 1.0;
    }

    /// <summary>The Python engine's <c>_zero_exempt</c>: exempt schools and negative values become 0.</summary>
    private static double Exempt(double value, bool exempt) => exempt || value < 0 ? 0.0 : value;

    /// <summary><c>np.nan_to_num(nan=0, posinf=0, neginf=0)</c>.</summary>
    private static double Finite(double value) => double.IsFinite(value) ? value : 0.0;

    /// <summary><c>np.searchsorted(sorted, v, side="right")</c>: how many thresholds are at most v.
    /// NumPy sorts NaN last, so a NaN value counts every threshold.</summary>
    private static int CountAtMost(ReadOnlySpan<double> sorted, double v)
    {
        if (double.IsNaN(v))
        {
            return sorted.Length;
        }

        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= v)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }

        return lo;
    }

    /// <summary><c>compute_school_indices</c>: the stage of the first threshold above the age,
    /// clamped to the last stage; an empty range (species absent from the CSV) gives -1.</summary>
    private static int StageIndex(int[] offsets, double[] thresholds, int[] index, int sp, double ageYears)
    {
        int lo = offsets[sp], hi = offsets[sp + 1];
        if (hi == lo)
        {
            return -1;
        }

        int k = CountAtMost(thresholds.AsSpan(lo, hi - lo), ageYears);
        return index[lo + Math.Min(k, hi - lo - 1)];
    }
}
