using System.Globalization;
using System.Text;
using Osmose.Engine.Mortality;
using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Spike;

/// <summary>
/// Kernel spike harness (plan task 0.6, gate G0), callable from Python through pythonnet. Loads a
/// fixture written by <c>scripts/csharp/make_fixtures.py</c> into two layouts:
/// <list type="bullet">
/// <item><b>regrouped</b>: schools contiguous per cell, the spec's layout;</item>
/// <item><b>indexed</b>: schools at their Python state index, a cell being an index list, which is
/// Python's memory layout.</item>
/// </list>
/// It runs one whole mortality step (every cell, every sub-step) with a chosen prefilter and draw
/// source, compares against the Numba oracle's outputs, and checks <c>Math.Exp</c> against Numba's
/// <c>exp</c>. Throwaway: Phase 1 replaces it with the real step loop.
/// </summary>
public sealed class KernelSpike : IDisposable
{
    private readonly SimulationConfig _fixture;
    private readonly KernelTables _tables;
    private readonly int _n;
    private readonly int _nCells;
    private readonly int _nState;
    private readonly int _nRsc;
    private readonly int _subdt;
    private readonly int _step;
    private readonly long[] _cellStart;
    private readonly int[] _pythonIndex;
    private readonly int[] _sortedIndices; // python index per cell-order position (indexed layout)

    private readonly SchoolCore[] _core0;
    private readonly SchoolStep[] _step0;
    private readonly double[] _rsc0; // cell-major
    private readonly SchoolCore[] _core;
    private readonly SchoolStep[] _stepArr;
    private readonly SchoolLedger[] _ledger;
    private readonly double[] _lane;
    private readonly SchoolCore[] _icore;
    private readonly SchoolStep[] _istep;
    private readonly SchoolLedger[] _iledger;
    private readonly double[] _rsc;
    private readonly CellScratch _scratch = new();
    private readonly InjectedDrawSource[] _injected;
    private readonly int[] _cellOfState; // cell of each state index, -1 for unlocated
    private readonly int[] _cursor;
    private string _lastLayout = "regrouped";

    private KernelSpike(SimulationConfig fixture)
    {
        _fixture = fixture;
        var r = new BundleReader(fixture, 1);
        _step = (int)r.Int("fixture.step");
        _subdt = (int)r.Int("fixture.n_subdt");
        int ndt = (int)r.Int("fixture.n_dt_per_year");
        int nSpecies = (int)r.Int("fixture.n_species");
        _nCells = (int)r.Int("fixture.n_cells");
        _nRsc = (int)r.Int("fixture.n_resources");
        _nState = (int)r.Int("fixture.n_state");
        bool hasAccess = r.Int("fixture.has_access") != 0;
        bool stage = r.Int("fixture.use_stage_access") != 0;
        if (r.Int("fixture.bioen") != 0)
        {
            throw new NotSupportedException("the kernel spike covers bioen-off fixtures only");
        }

        BundleEntry? lenE = r.Array("school.length", DType.F8, BundleReader.Any);
        r.ThrowIfProblems();
        _n = (int)lenE!.Length;
        long n = _n;
        BundleEntry ratioMin = r.Array("species.size_ratio_min", DType.F8, BundleReader.Any, BundleReader.Any)!;
        BundleEntry? access = r.Array("access.matrix", DType.F8, BundleReader.Any, BundleReader.Any);
        BundleEntry? rscBio = r.Array("resources.biomass", DType.F8, _nRsc, _nCells);
        r.ThrowIfProblems();
        int nStage = (int)ratioMin.Shape[1];
        (double[] byPred, int preyCount, int predCount) = KernelTables.PredatorMajor(
            access!.AsSpan<double>(), (int)access.Shape[0], (int)access.Shape[1], stage);
        _tables = new KernelTables
        {
            NSpeciesFocal = nSpecies,
            NStage = nStage,
            NDtPerYear = ndt,
            NSubdt = _subdt,
            NResources = _nRsc,
            SizeRatioMin = ratioMin.ToArray<double>(),
            SizeRatioMax = r.Array("species.size_ratio_max", DType.F8, ratioMin.Shape[0], nStage)!.ToArray<double>(),
            IngestionRate = r.Array("species.ingestion_rate", DType.F8, BundleReader.Any)!.ToArray<double>(),
            FrShape = r.Array("species.fr_shape", DType.I4, BundleReader.Any)!.ToArray<int>(),
            FrHalfSat = r.Array("species.fr_halfsat", DType.F8, BundleReader.Any)!.ToArray<double>(),
            HasAccess = hasAccess,
            UseStageAccess = stage,
            AccessByPred = byPred,
            AccessPreyCount = preyCount,
            AccessPredCount = predCount,
            ResourceAccessRows = r.Array("resources.access_rows", DType.I4, _nRsc)!.ToArray<int>(),
            RscSizeMin = r.Array("resources.size_min", DType.F8, _nRsc)!.ToArray<double>(),
            RscSizeMax = r.Array("resources.size_max", DType.F8, _nRsc)!.ToArray<double>(),
            RscTrophicLevel = r.Array("resources.trophic_level", DType.F8, _nRsc)!.ToArray<double>(),
        };

        _cellStart = r.Array("cell.start", DType.I8, _nCells + 1)!.ToArray<long>();
        _pythonIndex = r.Array("school.python_index", DType.I4, n)!.ToArray<int>();
        ReadOnlySpan<double> length = lenE.AsSpan<double>();
        ReadOnlySpan<double> weight = F8(r, "school.weight", n);
        ReadOnlySpan<double> abundance = F8(r, "school.abundance", n);
        ReadOnlySpan<double> eggRetained = F8(r, "school.egg_retained", n);
        ReadOnlySpan<double> tl = F8(r, "school.trophic_level", n);
        ReadOnlySpan<int> species = I4(r, "school.species_id", n);
        ReadOnlySpan<int> age = I4(r, "school.age_dt", n);
        ReadOnlySpan<int> ffa = I4(r, "school.first_feeding_age_dt", n);
        ReadOnlySpan<int> stageArr = I4(r, "school.feeding_stage", n);
        ReadOnlySpan<int> predIdx = I4(r, "school.pred_access_idx", n);
        ReadOnlySpan<int> preyIdx = I4(r, "school.prey_access_idx", n);
        ReadOnlySpan<byte> isEgg = r.Array("school.is_egg", DType.U1, n)!.AsSpan<byte>();
        ReadOnlySpan<byte> isBkg = r.Array("school.is_background", DType.U1, n)!.AsSpan<byte>();
        ReadOnlySpan<double> rs = F8(r, "school.rate_starvation", n);
        ReadOnlySpan<double> ra = F8(r, "school.rate_additional", n);
        ReadOnlySpan<double> rf = F8(r, "school.rate_fishing", n);
        ReadOnlySpan<double> disc = F8(r, "school.discard_fraction", n);
        ReadOnlySpan<double> psr = F8(r, "school.pred_success_rate", n);
        ReadOnlySpan<double> preyed = F8(r, "school.preyed_biomass", n);
        r.ThrowIfProblems();

        _core0 = new SchoolCore[_n];
        _step0 = new SchoolStep[_n];
        for (int i = 0; i < _n; i++)
        {
            _core0[i] = new SchoolCore
            {
                Length = length[i],
                Weight = weight[i],
                Abundance = abundance[i],
                EggRetained = eggRetained[i],
                TrophicLevel = tl[i],
                SpeciesId = species[i],
                AgeDt = age[i],
                PreyAccessRow = checked((short)preyIdx[i]),
                PredAccessCol = checked((short)predIdx[i]),
                FeedingStage = checked((byte)stageArr[i]),
                Flags = (isEgg[i] != 0 ? SchoolFlags.IsEgg : 0) | (isBkg[i] != 0 ? SchoolFlags.IsBackground : 0),
                FirstFeedingAgeDt = checked((short)ffa[i]),
            };
            _step0[i] = new SchoolStep
            {
                RateStarvation = rs[i],
                RateAdditional = ra[i],
                RateFishing = rf[i],
                DiscardFraction = disc[i],
                EggRelease = abundance[i] / _subdt,
                PredSuccess = psr[i],
                PreyedBiomass = preyed[i],
            };
        }

        ReadOnlySpan<double> rscPy = rscBio!.AsSpan<double>();
        _rsc0 = new double[_nRsc * _nCells];
        for (int rr = 0; rr < _nRsc; rr++)
        {
            for (int c = 0; c < _nCells; c++)
            {
                _rsc0[(c * _nRsc) + rr] = rscPy[(rr * _nCells) + c];
            }
        }

        _core = new SchoolCore[_n];
        _stepArr = new SchoolStep[_n];
        _ledger = new SchoolLedger[_n];
        _lane = new double[_n];
        _icore = new SchoolCore[_nState];
        _istep = new SchoolStep[_nState];
        _iledger = new SchoolLedger[_nState];
        _rsc = new double[_rsc0.Length];
        _sortedIndices = _pythonIndex;
        _injected = BuildInjected(r);
        _cellOfState = new int[_nState];
        Array.Fill(_cellOfState, -1);
        for (int c = 0; c < _nCells; c++)
        {
            for (long i = _cellStart[c]; i < _cellStart[c + 1]; i++)
            {
                _cellOfState[_pythonIndex[i]] = c;
            }
        }

        _cursor = new int[_nCells + 1];
        Reset();
    }

    public int SchoolCount => _n;

    public int CellCount => _nCells;

    public int Substeps => _subdt;

    /// <summary>Opens a fixture bundle.</summary>
    public static KernelSpike Load(string path) => new(SimulationConfig.Load(path));

    /// <summary>Restores the step's starting state in both layouts.</summary>
    public void Reset()
    {
        Array.Copy(_core0, _core, _n);
        Array.Copy(_step0, _stepArr, _n);
        Array.Clear(_ledger);
        Array.Clear(_icore);
        Array.Clear(_istep);
        Array.Clear(_iledger);
        for (int i = 0; i < _n; i++)
        {
            _lane[i] = _core0[i].Length;
            _icore[_pythonIndex[i]] = _core0[i];
            _istep[_pythonIndex[i]] = _step0[i];
        }

        Array.Copy(_rsc0, _rsc, _rsc0.Length);
    }

    /// <summary>
    /// The spec's regroup (section 3.4) for the spike: a stable counting sort of the indexed
    /// (Python-order) schools into contiguous cell order, plus the length-lane copy. Part of the
    /// regrouped variants' timed work, as Python's argsort is part of its step.
    /// </summary>
    public void Regroup()
    {
        Array.Clear(_cursor);
        for (int i = 0; i < _nState; i++)
        {
            int c = _cellOfState[i];
            if (c >= 0)
            {
                _cursor[c + 1]++;
            }
        }

        for (int c = 0; c < _nCells; c++)
        {
            _cursor[c + 1] += _cursor[c];
        }

        for (int i = 0; i < _nState; i++)
        {
            int c = _cellOfState[i];
            if (c >= 0)
            {
                int at = _cursor[c]++;
                _core[at] = _icore[i];
                _stepArr[at] = _istep[i];
                _ledger[at] = _iledger[i];
                _lane[at] = _icore[i].Length;
            }
        }
    }

    /// <summary>
    /// Runs the whole step, one thread. <paramref name="layout"/>: "regrouped" or "indexed".
    /// <paramref name="filter"/>: "full", "mask" or "lists". <paramref name="draws"/>:
    /// "injected" (the oracle's draws) or "philox" (production, keyed by <paramref name="seed"/>).
    /// </summary>
    public void RunStep(string layout, string filter, string draws, long seed)
    {
        _lastLayout = layout;
        var key = PhiloxKey.ForRun(seed, Philox.Scheme);
        for (int c = 0; c < _nCells; c++)
        {
            int start = (int)_cellStart[c];
            int count = (int)_cellStart[c + 1] - start;
            if (count == 0)
            {
                continue;
            }

            Span<double> rsc = _rsc.AsSpan(c * _nRsc, _nRsc);
            if (layout == "regrouped")
            {
                var cell = new ContiguousCell(_core, _stepArr, _ledger, _lane, start, count);
                Dispatch(ref cell, filter, draws, key, c, rsc);
            }
            else if (layout == "indexed")
            {
                _scratch.Ensure(count, _nRsc);
                var cell = new IndexedCell(_icore, _istep, _iledger, _sortedIndices, start, count, _scratch.Gather);
                Dispatch(ref cell, filter, draws, key, c, rsc);
            }
            else
            {
                throw new ArgumentException($"unknown layout {layout}", nameof(layout));
            }
        }
    }

    private void Dispatch<TCell>(ref TCell cell, string filter, string draws, PhiloxKey key, int c, Span<double> rsc)
        where TCell : struct, ICellSchools
    {
        if (draws == "injected")
        {
            InjectedDrawSource d = _injected[c];
            RunFilter(ref cell, filter, ref d, rsc);
        }
        else if (draws == "philox")
        {
            var d = new PhiloxDrawSource(key, _step, c);
            RunFilter(ref cell, filter, ref d, rsc);
        }
        else
        {
            throw new ArgumentException($"unknown draw source {draws}", nameof(draws));
        }
    }

    private void RunFilter<TCell, TDraws>(ref TCell cell, string filter, ref TDraws d, Span<double> rsc)
        where TCell : struct, ICellSchools
        where TDraws : struct, IDrawSource
    {
        switch (filter)
        {
            case "full":
                MortalityKernel.RunCell<TCell, FullScan, TDraws>(_tables, ref cell, rsc, _scratch, ref d);
                break;
            case "mask":
                MortalityKernel.RunCell<TCell, MaskPrefilter, TDraws>(_tables, ref cell, rsc, _scratch, ref d);
                break;
            case "lists":
                MortalityKernel.RunCell<TCell, CandidateLists, TDraws>(_tables, ref cell, rsc, _scratch, ref d);
                break;
            default:
                throw new ArgumentException($"unknown filter {filter}", nameof(filter));
        }
    }

    /// <summary>
    /// Compares the last run's outputs with the oracle's. One line per quantity:
    /// <c>name n_bit_diff max_abs max_rel</c>. Bit differences are counted with NaN == NaN.
    /// </summary>
    public string Compare()
    {
        var sb = new StringBuilder();
        double[] got = new double[_n];
        void Row(string name, string expectedName, Func<int, double> pick, int cols = 1, int col = 0)
        {
            ReadOnlySpan<double> exp = _fixture.TryGet(expectedName, out BundleEntry e)
                ? e.AsSpan<double>()
                : throw new InvalidOperationException($"fixture lacks {expectedName}");
            int diff = 0;
            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < _n; i++)
            {
                double a = pick(i), b = exp[(i * cols) + col];
                if (BitConverter.DoubleToInt64Bits(a) != BitConverter.DoubleToInt64Bits(b) && !(double.IsNaN(a) && double.IsNaN(b)))
                {
                    diff++;
                    double abs = Math.Abs(a - b);
                    maxAbs = Math.Max(maxAbs, abs);
                    maxRel = Math.Max(maxRel, abs / Math.Max(Math.Abs(b), double.Epsilon));
                }
            }

            sb.Append(CultureInfo.InvariantCulture, $"{name} {diff} {maxAbs:R} {maxRel:R}\n");
        }

        bool idx = _lastLayout == "indexed";
        ref SchoolCore C(int i) => ref idx ? ref _icore[_pythonIndex[i]] : ref _core[i];
        ref SchoolStep S(int i) => ref idx ? ref _istep[_pythonIndex[i]] : ref _stepArr[i];
        ref SchoolLedger L(int i) => ref idx ? ref _iledger[_pythonIndex[i]] : ref _ledger[i];
        Row("abundance", "expected.abundance", i => C(i).Abundance);
        Row("egg_retained", "expected.egg_retained", i => C(i).EggRetained);
        Row("pred_success_rate", "expected.pred_success_rate", i => S(i).PredSuccess);
        Row("preyed_biomass", "expected.preyed_biomass", i => S(i).PreyedBiomass);
        Row("tl_weighted_sum", "expected.tl_weighted_sum", i => S(i).TlWeightedSum);
        Row("dead_predation", "expected.n_dead", i => C(i).DeadPredation, 8, 0);
        Row("dead_starvation", "expected.n_dead", i => L(i).DeadStarvation, 8, 1);
        Row("dead_additional", "expected.n_dead", i => L(i).DeadAdditional, 8, 2);
        Row("dead_fishing", "expected.n_dead", i => L(i).DeadFishing, 8, 3);
        Row("dead_discards", "expected.n_dead", i => L(i).DeadDiscards, 8, 6);

        _fixture.TryGet("expected.resources_biomass", out BundleEntry re);
        ReadOnlySpan<double> rexp = re.AsSpan<double>();
        int rdiff = 0;
        double rmax = 0;
        for (int rr = 0; rr < _nRsc; rr++)
        {
            for (int c = 0; c < _nCells; c++)
            {
                double a = _rsc[(c * _nRsc) + rr], b = rexp[(rr * _nCells) + c];
                if (BitConverter.DoubleToInt64Bits(a) != BitConverter.DoubleToInt64Bits(b))
                {
                    rdiff++;
                    rmax = Math.Max(rmax, Math.Abs(a - b) / Math.Max(Math.Abs(b), double.Epsilon));
                }
            }
        }

        sb.Append(CultureInfo.InvariantCulture, $"resources_biomass {rdiff} - {rmax:R}\n");
        return sb.ToString();
    }

    /// <summary>SHA-256 over the bits of every output of the last run, in state-index order, so two
    /// layouts or prefilters can be compared with each other exactly (check 4).</summary>
    public string ResultDigest()
    {
        bool idx = _lastLayout == "indexed";
        var order = new int[_n];
        for (int i = 0; i < _n; i++)
        {
            order[i] = i;
        }

        Array.Sort((int[])_pythonIndex.Clone(), order);
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        byte[] buf = new byte[8];
        void Put(double v)
        {
            BitConverter.TryWriteBytes(buf, v);
            hash.AppendData(buf);
        }

        foreach (int i in order)
        {
            SchoolCore c = idx ? _icore[_pythonIndex[i]] : _core[i];
            SchoolStep st = idx ? _istep[_pythonIndex[i]] : _stepArr[i];
            SchoolLedger l = idx ? _iledger[_pythonIndex[i]] : _ledger[i];
            Put(c.Abundance);
            Put(c.EggRetained);
            Put(c.DeadPredation);
            Put(st.PredSuccess);
            Put(st.PreyedBiomass);
            Put(st.TlWeightedSum);
            Put(l.DeadStarvation);
            Put(l.DeadAdditional);
            Put(l.DeadFishing);
            Put(l.DeadDiscards);
        }

        foreach (double v in _rsc)
        {
            Put(v);
        }

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary><c>Math.Exp(-x)</c> against Numba's <c>exp(-x)</c> on the fixture's arguments:
    /// <c>n_args n_bit_diff max_ulp</c>.</summary>
    public string LibmCheck()
    {
        _fixture.TryGet("libm.args", out BundleEntry a);
        _fixture.TryGet("libm.exp_neg", out BundleEntry e);
        ReadOnlySpan<double> args = a.AsSpan<double>(), numba = e.AsSpan<double>();
        int diff = 0;
        long maxUlp = 0;
        for (int i = 0; i < args.Length; i++)
        {
            long x = BitConverter.DoubleToInt64Bits(Math.Exp(-args[i]));
            long y = BitConverter.DoubleToInt64Bits(numba[i]);
            if (x != y)
            {
                diff++;
                maxUlp = Math.Max(maxUlp, Math.Abs(x - y));
            }
        }

        return string.Create(CultureInfo.InvariantCulture, $"{args.Length} {diff} {maxUlp}");
    }

    public void Dispose() => _fixture.Dispose();

    private InjectedDrawSource[] BuildInjected(BundleReader r)
    {
        ReadOnlySpan<int> pp = r.Array("draws.perm_predation", DType.I4, _subdt, _n)!.AsSpan<int>();
        ReadOnlySpan<int> ps = r.Array("draws.perm_starvation", DType.I4, _subdt, _n)!.AsSpan<int>();
        ReadOnlySpan<int> pf = r.Array("draws.perm_fishing", DType.I4, _subdt, _n)!.AsSpan<int>();
        ReadOnlySpan<int> pa = r.Array("draws.perm_additional", DType.I4, _subdt, _n)!.AsSpan<int>();
        ReadOnlySpan<byte> co = r.Array("draws.cause_orders", DType.U1, _subdt, _n, 4)!.AsSpan<byte>();
        var result = new InjectedDrawSource[_nCells];
        for (int c = 0; c < _nCells; c++)
        {
            int s = (int)_cellStart[c], m = (int)_cellStart[c + 1] - s;
            var perms = new int[_subdt][][];
            var orders = new byte[_subdt][];
            for (int sub = 0; sub < _subdt; sub++)
            {
                int o = (sub * _n) + s;
                // IDrawSource order: purpose - 1 = predation, starvation, fishing, additional.
                perms[sub] = [pp.Slice(o, m).ToArray(), ps.Slice(o, m).ToArray(), pf.Slice(o, m).ToArray(), pa.Slice(o, m).ToArray()];
                orders[sub] = co.Slice(((sub * _n) + s) * 4, m * 4).ToArray();
            }

            result[c] = new InjectedDrawSource(perms, orders);
        }

        return result;
    }

    private static ReadOnlySpan<double> F8(BundleReader r, string name, long n) =>
        r.Array(name, DType.F8, n) is { } e ? e.AsSpan<double>() : default;

    private static ReadOnlySpan<int> I4(BundleReader r, string name, long n) =>
        r.Array(name, DType.I4, n) is { } e ? e.AsSpan<int>() : default;
}
