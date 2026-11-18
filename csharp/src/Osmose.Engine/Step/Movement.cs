using Osmose.Engine.Rng;
using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// Movement (plan task 1.6, spec 6.11): map-based species are placed by rejection or walk on
/// their current map, random-method species are placed on their domain or walk on ocean, others
/// stay. Every draw comes from a Philox stream keyed by the school's id, the step and the
/// purpose, so a school's move does not depend on the store's order or on other schools.
/// </summary>
/// <remarks>
/// The candidate sets, the map index, the same-map test and the accept rule are tier E against
/// the Python engine; where a school lands is tier S.
/// <para>
/// Salinity gate (feature salinity_gate, plan task 2.4): for a gated species, occupancy follows
/// <c>map * weight</c> instead of the map. Placement accepts a drawn cell when that product is
/// placeable and at least <c>u * wmax</c>; a walk chooses among the cells where both the map and
/// the product are placeable, with probability proportional to the product. When no cell of the
/// map has a positive product this step (<c>wmax</c> 0) the school moves exactly as ungated.
/// </para>
/// <para>
/// <see cref="SchoolFlags.IsOut"/> is written for map-based schools only, as Python's map kernel
/// does: set when the school leaves the domain, cleared when it is placed, walks or stays. The
/// per-step reset of the flag for every school belongs to the step pipeline
/// (<c>_reset_step_variables</c>), not here.
/// </para>
/// </remarks>
public static class Movement
{
    /// <summary>Rejection-sampling attempts before a map-based school is sent out of the domain.</summary>
    public const int MaxPlacementTries = 10_000;

    /// <summary>Scratch length <see cref="Run"/> needs: the largest <c>(2r + 1)^2</c> square.</summary>
    public static int ScratchLength(MovementParameters m) => ((2 * m.MaxWalkRange) + 1) * ((2 * m.MaxWalkRange) + 1);

    /// <summary>A map value a school can be placed on or walk to: positive and not NaN.</summary>
    public static bool IsPlaceable(double value) => value > 0 && !double.IsNaN(value);

    /// <summary>The placement accept rule for a drawn cell: placeable, and either a presence map
    /// (<paramref name="maxP"/> 0) or <c>value >= u * maxP</c>.</summary>
    public static bool Accepts(double value, double maxP, double u) => IsPlaceable(value) && (maxP == 0.0 || value >= u * maxP);

    /// <summary>The gated placement accept rule: the product <c>map * weight</c> of the drawn cell
    /// is placeable and at least <c>u * wmax</c> (a uniform is always drawn for a placeable cell).</summary>
    public static bool AcceptsGated(double product, double wmax, double u) => IsPlaceable(product) && product >= u * wmax;

    /// <summary>Whether the salinity gate applies to a school of <paramref name="species"/> on map
    /// <paramref name="globalMap"/> at <paramref name="step"/>, with the step's weight row and the
    /// map's <paramref name="wmax"/>. False (ungated) for a species outside the gate and when
    /// the map has no positive product this step (the all-zero guard).</summary>
    public static bool Gate(MovementParameters m, int species, int globalMap, int step, int nCells, out ReadOnlySpan<double> weight, out double wmax)
    {
        weight = default;
        wmax = 0.0;
        if (species >= m.SalinitySpecies.Length || m.SalinitySpecies[species] == 0)
        {
            return false;
        }

        int frame = m.SalinityFrameByStep[step];
        wmax = m.SalinityWmax[(globalMap * m.SalinityFrames) + frame];
        if (!(wmax > 0.0))
        {
            return false;
        }

        weight = m.SalinityWeight.AsSpan(frame * nCells, nCells);
        return true;
    }

    /// <summary>Gated walk candidates: the cells of <see cref="MapWalkCandidates"/> whose product
    /// <c>map * weight</c> is placeable too, row-major. <paramref name="total"/> is the sum of
    /// their products, added in that order (the Python kernel's <c>W</c>).</summary>
    public static int GatedWalkCandidates(
        MovementParameters m, int ny, int nx, int globalMap, int cell, int range, ReadOnlySpan<double> weight, Span<int> into, out double total)
    {
        ReadOnlySpan<double> map = m.MapStack.AsSpan(globalMap * ny * nx, ny * nx);
        int cy = cell / nx, cx = cell % nx, n = 0;
        int yHi = Math.Min(ny, cy + range + 1), xLo = Math.Max(0, cx - range), xHi = Math.Min(nx, cx + range + 1);
        total = 0.0;
        for (int y = Math.Max(0, cy - range); y < yHi; y++)
        {
            for (int x = xLo; x < xHi; x++)
            {
                int k = (y * nx) + x;
                if (m.Ocean[k] != 0 && IsPlaceable(map[k]))
                {
                    double product = map[k] * weight[k];
                    if (IsPlaceable(product))
                    {
                        into[n++] = k;
                        total += product;
                    }
                }
            }
        }

        return n;
    }

    /// <summary>The weighted choice of the gated walk: the first candidate at which the running
    /// sum of products reaches <paramref name="r"/> (<c>u * total</c>), else the last one.
    /// The sum is taken in candidate order, as <see cref="GatedWalkCandidates"/> took the total.</summary>
    public static int WeightedChoice(ReadOnlySpan<double> map, ReadOnlySpan<double> weight, ReadOnlySpan<int> candidates, double r)
    {
        double acc = 0.0;
        for (int i = 0; i < candidates.Length; i++)
        {
            int k = candidates[i];
            acc += map[k] * weight[k];
            if (acc >= r)
            {
                return k;
            }
        }

        return candidates[^1];
    }

    /// <summary><c>index_maps[age_dt, step]</c> of a focal species, or -1 when either lies outside
    /// the table or the species has no maps.</summary>
    public static int MapIndex(MovementParameters m, int species, int ageDt, int step)
    {
        if (species >= m.MapSpeciesOffset.Length || m.MapIndexShape.Length < (2 * species) + 2)
        {
            return -1;
        }

        int nAges = m.MapIndexShape[2 * species], nSteps = m.MapIndexShape[(2 * species) + 1];
        if (ageDt < 0 || ageDt >= nAges || step < 0 || step >= nSteps)
        {
            return -1;
        }

        return m.MapIndex[m.MapIndexOffsets[species] + ((long)ageDt * nSteps) + step];
    }

    /// <summary>The school's map is the one it had last step: <c>age_dt > 0</c>, <c>step > 0</c> and
    /// the index equals <c>index_maps[age_dt - 1, step - 1]</c> (-1 outside the table).</summary>
    public static bool SameMap(MovementParameters m, int species, int ageDt, int step, int current) =>
        ageDt > 0 && step > 0 && current == MapIndex(m, species, ageDt - 1, step - 1);

    /// <summary>Walk candidates on a map: the cells of the <c>(2r + 1)^2</c> square around
    /// <paramref name="cell"/>, clipped to the grid, that are ocean and placeable on the map.
    /// Row-major order. Returns how many were written to <paramref name="into"/>.</summary>
    public static int MapWalkCandidates(MovementParameters m, int ny, int nx, int globalMap, int cell, int range, Span<int> into)
    {
        ReadOnlySpan<double> map = m.MapStack.AsSpan(globalMap * ny * nx, ny * nx);
        int cy = cell / nx, cx = cell % nx, n = 0;
        int yHi = Math.Min(ny, cy + range + 1), xLo = Math.Max(0, cx - range), xHi = Math.Min(nx, cx + range + 1);
        for (int y = Math.Max(0, cy - range); y < yHi; y++)
        {
            for (int x = xLo; x < xHi; x++)
            {
                int k = (y * nx) + x;
                if (m.Ocean[k] != 0 && IsPlaceable(map[k]))
                {
                    into[n++] = k;
                }
            }
        }

        return n;
    }

    /// <summary>Random-walk candidates: the in-grid ocean cells of the <c>(2r + 1)^2</c> square,
    /// the school's own cell included. Row-major order.</summary>
    public static int RandomWalkCandidates(MovementParameters m, int ny, int nx, int cell, int range, Span<int> into)
    {
        int cy = cell / nx, cx = cell % nx, n = 0;
        int yHi = Math.Min(ny, cy + range + 1), xLo = Math.Max(0, cx - range), xHi = Math.Min(nx, cx + range + 1);
        for (int y = Math.Max(0, cy - range); y < yHi; y++)
        {
            for (int x = xLo; x < xHi; x++)
            {
                int k = (y * nx) + x;
                if (m.Ocean[k] != 0)
                {
                    into[n++] = k;
                }
            }
        }

        return n;
    }

    /// <summary>Moves every school of the store for <paramref name="step"/>.
    /// <paramref name="scratch"/> holds at least <see cref="ScratchLength"/> values.</summary>
    public static void Run(EngineParameters p, SchoolStore store, int step, in PhiloxKey key, Span<int> scratch) =>
        Run(p, store, step, key, scratch, 0, store.Count);

    /// <summary>Moves schools <paramref name="from"/> (inclusive) to <paramref name="to"/>
    /// (exclusive). A school's move touches only that school, so ranges may run in parallel,
    /// each with its own <paramref name="scratch"/>.</summary>
    public static void Run(EngineParameters p, SchoolStore store, int step, in PhiloxKey key, Span<int> scratch, int from, int to)
    {
        MovementParameters m = p.Movement;
        Span<SchoolCore> core = store.Core;
        Span<SchoolLife> life = store.Life;
        for (int i = from; i < to; i++)
        {
            int sp = core[i].SpeciesId;
            byte method = sp < m.Method.Length ? m.Method[sp] : MovementParameters.MethodNone;
            if (method == MovementParameters.MethodMaps)
            {
                MoveOnMap(p, ref core[i], ref life[i], step, key, scratch);
            }
            else if (method == MovementParameters.MethodRandom)
            {
                MoveRandomly(p, sp, ref life[i], step, key, scratch);
            }
        }
    }

    private static void MoveOnMap(EngineParameters p, ref SchoolCore c, ref SchoolLife l, int step, in PhiloxKey key, Span<int> scratch)
    {
        MovementParameters m = p.Movement;
        int sp = c.SpeciesId, nCells = p.Ny * p.Nx;
        int index = MapIndex(m, sp, c.AgeDt, step);
        int offset = sp < m.MapSpeciesOffset.Length ? m.MapSpeciesOffset[sp] : -1;
        if (index < 0 || offset < 0 || offset + index >= m.NMaps || m.MapIsNull[offset + index] != 0)
        {
            SendOut(ref c, ref l);
            return;
        }

        int g = offset + index;
        bool gated = Gate(m, sp, g, step, nCells, out ReadOnlySpan<double> weight, out double wmax);
        ReadOnlySpan<double> map = m.MapStack.AsSpan(g * nCells, nCells);
        if (!SameMap(m, sp, c.AgeDt, step, index) || l.CellId < 0)
        {
            double maxP = m.MapMaxProba[g];
            var draws = new PhiloxStream(key, DrawPurpose.MoveMapPlace, step, 0, l.Id);
            for (int t = 0; t < MaxPlacementTries; t++)
            {
                int cell = (int)draws.NextBounded((ulong)nCells);
                double v = map[cell];
                bool accepted;
                if (gated)
                {
                    double product = v * weight[cell];
                    accepted = IsPlaceable(product) && AcceptsGated(product, wmax, draws.NextDouble());
                }
                else
                {
                    // The uniform is drawn only for a placeable cell of a proportional map, as Python does.
                    accepted = IsPlaceable(v) && (maxP == 0.0 || Accepts(v, maxP, draws.NextDouble()));
                }

                if (accepted)
                {
                    l.CellId = cell;
                    c.Flags &= ~SchoolFlags.IsOut;
                    return;
                }
            }

            SendOut(ref c, ref l);
            return;
        }

        if (gated)
        {
            int count = GatedWalkCandidates(m, p.Ny, p.Nx, g, l.CellId, m.WalkRange[sp], weight, scratch, out double total);
            if (count > 0)
            {
                var draws = new PhiloxStream(key, DrawPurpose.MoveMapWalk, step, 0, l.Id);
                l.CellId = WeightedChoice(map, weight, scratch[..count], draws.NextDouble() * total);
            }
        }
        else
        {
            int n = MapWalkCandidates(m, p.Ny, p.Nx, g, l.CellId, m.WalkRange[sp], scratch);
            if (n > 0)
            {
                var draws = new PhiloxStream(key, DrawPurpose.MoveMapWalk, step, 0, l.Id);
                l.CellId = scratch[(int)draws.NextBounded((ulong)n)];
            }
        }

        c.Flags &= ~SchoolFlags.IsOut; // walked, or stranded where it is
    }

    private static void MoveRandomly(EngineParameters p, int sp, ref SchoolLife l, int step, in PhiloxKey key, Span<int> scratch)
    {
        MovementParameters m = p.Movement;
        if (l.CellId < 0)
        {
            int lo = m.DomainOffsets[sp], hi = m.DomainOffsets[sp + 1];
            if (hi > lo)
            {
                var place = new PhiloxStream(key, DrawPurpose.MoveRandomPlace, step, 0, l.Id);
                int k = lo + (int)place.NextBounded((ulong)(hi - lo));
                l.CellId = (m.DomainCells[(2 * k) + 1] * p.Nx) + m.DomainCells[2 * k];
            }

            return; // placed this step: it does not also walk
        }

        int range = m.WalkRange[sp];
        if (range <= 0)
        {
            return;
        }

        int n = RandomWalkCandidates(m, p.Ny, p.Nx, l.CellId, range, scratch);
        if (n > 0)
        {
            var walk = new PhiloxStream(key, DrawPurpose.MoveRandomWalk, step, 0, l.Id);
            l.CellId = scratch[(int)walk.NextBounded((ulong)n)];
        }
    }

    private static void SendOut(ref SchoolCore c, ref SchoolLife l)
    {
        l.CellId = -1;
        c.Flags |= SchoolFlags.IsOut;
    }
}
