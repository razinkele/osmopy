namespace Osmose.Engine.Rng;

/// <summary>
/// The mortality kernel's source of randomness for one cell (spec section 4.3). Used as a generic
/// struct type parameter (<c>where TDraws : struct, IDrawSource</c>), so calls are devirtualised:
/// Philox in production, injected arrays when a test compares the kernel with the Python oracle.
/// </summary>
public interface IDrawSource
{
    /// <summary>Starts sub-step <paramref name="substep"/> of the cell. Call before its draws.</summary>
    void BeginSubstep(int substep);

    /// <summary>Fills <paramref name="perm"/> with a permutation of <c>0..perm.Length-1</c> for
    /// <paramref name="purpose"/> in the current sub-step.</summary>
    void Permutation(DrawPurpose purpose, Span<int> perm);

    /// <summary>Writes the next school slot's cause order (<c>order.Length</c> causes, 4 or 5).
    /// Slots are drawn in walk order.</summary>
    void NextCauseOrder(Span<byte> order);
}

/// <summary>Production draws for one (run, step, cell): every value a pure function of the key
/// and counter, whatever thread runs the cell and whenever.</summary>
public struct PhiloxDrawSource : IDrawSource
{
    private readonly PhiloxKey _key;
    private readonly int _step;
    private readonly ulong _cell;
    private int _substep;
    private PhiloxStream _causes;
    private bool _started;

    /// <summary>Draws for cell <paramref name="cellId"/> during <paramref name="step"/>.</summary>
    public PhiloxDrawSource(in PhiloxKey key, int step, int cellId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(cellId);
        _key = key;
        _step = step;
        _cell = (ulong)cellId;
        _substep = 0;
        _causes = default;
        _started = false;
    }

    /// <inheritdoc/>
    public void BeginSubstep(int substep)
    {
        _substep = substep;
        _causes = new PhiloxStream(_key, DrawPurpose.CauseOrder, _step, substep, _cell);
        _started = true;
    }

    /// <inheritdoc/>
    public readonly void Permutation(DrawPurpose purpose, Span<int> perm)
    {
        EnsureStarted();
        var stream = new PhiloxStream(_key, purpose, _step, _substep, _cell);
        stream.FillPermutation(perm);
    }

    /// <inheritdoc/>
    public void NextCauseOrder(Span<byte> order)
    {
        EnsureStarted();
        int code = (int)_causes.NextBounded((ulong)CauseOrders.Count(order.Length));
        CauseOrders.Decode(order.Length, code).CopyTo(order);
    }

    private readonly void EnsureStarted()
    {
        if (!_started)
        {
            throw new InvalidOperationException("BeginSubstep must be called before drawing");
        }
    }
}

/// <summary>
/// Test draws: the permutations and cause orders the Python oracle consumed, replayed exactly
/// (plan task 0.6 compares the kernel with <c>_mortality_in_cell_numba</c> this way). Throws on any
/// mismatch in length or count rather than inventing numbers.
/// </summary>
public struct InjectedDrawSource : IDrawSource
{
    private readonly int[][][] _permutations; // [substep][purpose - 1]
    private readonly byte[][] _causeOrders;   // [substep] -> slot orders, concatenated
    private int _substep;
    private int _cursor;

    /// <summary>Wraps per-sub-step permutations (indexed by <c>purpose - 1</c>) and cause orders.</summary>
    public InjectedDrawSource(int[][][] permutations, byte[][] causeOrders)
    {
        ArgumentNullException.ThrowIfNull(permutations);
        ArgumentNullException.ThrowIfNull(causeOrders);
        if (permutations.Length != causeOrders.Length)
        {
            throw new ArgumentException("one permutation set and one cause-order list per sub-step");
        }

        _permutations = permutations;
        _causeOrders = causeOrders;
        _substep = -1;
        _cursor = 0;
    }

    /// <inheritdoc/>
    public void BeginSubstep(int substep)
    {
        if ((uint)substep >= (uint)_permutations.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(substep), substep, "no injected draws");
        }

        _substep = substep;
        _cursor = 0;
    }

    /// <inheritdoc/>
    public readonly void Permutation(DrawPurpose purpose, Span<int> perm)
    {
        int[][] set = _permutations[CheckedSubstep()];
        int index = (int)purpose - 1;
        if ((uint)index >= (uint)set.Length || set[index].Length != perm.Length)
        {
            throw new InvalidOperationException(
                $"no injected {purpose} permutation of length {perm.Length} in sub-step {_substep}");
        }

        set[index].CopyTo(perm);
    }

    /// <inheritdoc/>
    public void NextCauseOrder(Span<byte> order)
    {
        byte[] orders = _causeOrders[CheckedSubstep()];
        if (_cursor + order.Length > orders.Length)
        {
            throw new InvalidOperationException($"injected cause orders exhausted in sub-step {_substep}");
        }

        orders.AsSpan(_cursor, order.Length).CopyTo(order);
        _cursor += order.Length;
    }

    private readonly int CheckedSubstep() =>
        _substep >= 0 ? _substep : throw new InvalidOperationException("BeginSubstep must be called first");
}
