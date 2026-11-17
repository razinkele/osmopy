using System.Runtime.CompilerServices;

namespace Osmose.Engine.Rng;

/// <summary>What a draw is for: the counter's top word (spec section 5.3). Never renumber an
/// existing member; that would change every stream (bump <see cref="Philox.Scheme"/> instead).</summary>
public enum DrawPurpose : ulong
{
    PermPredation = 1,
    PermStarvation = 2,
    PermFishing = 3,
    PermAdditional = 4,
    PermForaging = 5,
    CauseOrder = 6,
    MoveMapPlace = 7,
    MoveMapWalk = 8,
    MoveRandomPlace = 9,
    MoveRandomWalk = 10,
    FluxPlace = 11,
    InitPlace = 12,
    PatchStart = 13,
}

/// <summary>
/// One draw sequence: counter <c>(block, entity, step|substep, purpose)</c> with the block index
/// counting up from zero. A mutable struct meant to live on the stack for one sequence; it holds
/// no shared state, so two threads can never perturb each other's numbers.
/// </summary>
public struct PhiloxStream
{
    private readonly PhiloxKey _key;
    private readonly ulong _entity;
    private readonly ulong _stepWord;
    private readonly ulong _purpose;
    private ulong _nextBlock;
    private PhiloxBlock _buffer;
    private int _position; // 4 = buffer empty

    /// <summary>Starts the sequence for one (purpose, step, sub-step, entity).</summary>
    public PhiloxStream(in PhiloxKey key, DrawPurpose purpose, int step, int substep, ulong entity)
    {
        _key = key;
        _purpose = (ulong)purpose;
        _stepWord = StepWord(step, substep);
        _entity = entity;
        _nextBlock = 0;
        _buffer = default;
        _position = 4;
    }

    /// <summary>The counter's third word: <c>(step &lt;&lt; 16) | substep</c>.</summary>
    public static ulong StepWord(int step, int substep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        if ((uint)substep > ushort.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(substep), substep, "must fit in 16 bits");
        }

        return ((ulong)step << 16) | (uint)substep;
    }

    /// <summary>The next 64 random bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong NextUInt64()
    {
        if (_position == 4)
        {
            _buffer = Philox.Block(_key, _nextBlock++, _entity, _stepWord, _purpose);
            _position = 0;
        }

        return _buffer[_position++];
    }

    /// <summary>Uniform in <c>[0, range)</c>, unbiased: Lemire's multiply-shift with rejection.</summary>
    public ulong NextBounded(ulong range)
    {
        if (range == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(range), "range must be positive");
        }

        ulong value;
        while (!TryBounded(NextUInt64(), range, out value))
        {
        }

        return value;
    }

    /// <summary>
    /// One step of Lemire's method: maps word <paramref name="x"/> to <c>[0, range)</c>, or returns
    /// false when the word falls in the biased sliver and must be redrawn. Separate so the rejection,
    /// which fires with probability about <c>range / 2^64</c>, can be tested with a crafted word.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryBounded(ulong x, ulong range, out ulong value)
    {
        value = Math.BigMul(x, range, out ulong low);
        return low >= range || low >= unchecked(0UL - range) % range;
    }

    /// <summary>Uniform in <c>[0, 1)</c> with 53 random bits: <c>(x &gt;&gt; 11) * 2^-53</c>.</summary>
    public double NextDouble() => (NextUInt64() >> 11) * (1.0 / (1UL << 53));

    /// <summary>Fills <paramref name="perm"/> with a uniform permutation of <c>0..n-1</c>
    /// (Fisher-Yates, i from n-1 down to 1, j uniform in [0, i]).</summary>
    public void FillPermutation(Span<int> perm)
    {
        for (int i = 0; i < perm.Length; i++)
        {
            perm[i] = i;
        }

        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = (int)NextBounded((ulong)i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }
    }
}
