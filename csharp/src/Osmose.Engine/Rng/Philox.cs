using System.Runtime.CompilerServices;

namespace Osmose.Engine.Rng;

/// <summary>A Philox4x64 key: two 64-bit words (spec section 5.3).</summary>
public readonly record struct PhiloxKey(ulong K0, ulong K1)
{
    /// <summary>
    /// "OSMOSE\0\0": the domain constant XOR-ed with the scheme version into <see cref="K1"/>, so
    /// another scheme (or another program) seeded with the same integer gets a different stream.
    /// </summary>
    public const ulong DomainConstant = 0x4F534D4F53450000UL;

    /// <summary>The engine's key for a run: <c>(SplitMix64(seed), DomainConstant ^ scheme)</c>.</summary>
    public static PhiloxKey ForRun(long seed, int scheme) =>
        new(Philox.SplitMix64(unchecked((ulong)seed)), DomainConstant ^ (ulong)(uint)scheme);
}

/// <summary>One Philox4x64-10 output block: four 64-bit words.</summary>
public readonly record struct PhiloxBlock(ulong W0, ulong W1, ulong W2, ulong W3)
{
    /// <summary>Word <paramref name="i"/> (0 to 3).</summary>
    public ulong this[int i] => i switch
    {
        0 => W0,
        1 => W1,
        2 => W2,
        3 => W3,
        _ => throw new ArgumentOutOfRangeException(nameof(i)),
    };
}

/// <summary>
/// Random123's Philox4x64-10 (Salmon et al., SC'11), the generator <c>numpy.random.Philox</c>
/// implements. Every draw in the engine is a pure function of key and counter, so results do not
/// depend on threads or scheduling (spec decision D6).
/// </summary>
public static class Philox
{
    /// <summary>The RNG scheme this code implements. Bump on any change to keys, counters or
    /// decoding, so that "same seed, different numbers" is always attributable.</summary>
    public const int Scheme = 1;

    private const ulong M0 = 0xD2E7470EE14C6C93UL;
    private const ulong M1 = 0xCA5A826395121157UL;
    private const ulong W0 = 0x9E3779B97F4A7C15UL; // golden ratio
    private const ulong W1 = 0xBB67AE8584CAA73BUL; // sqrt(3) - 1

    /// <summary>The block for counter <c>(c0, c1, c2, c3)</c>, c0 the least significant word.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static PhiloxBlock Block(in PhiloxKey key, ulong c0, ulong c1, ulong c2, ulong c3)
    {
        ulong k0 = key.K0, k1 = key.K1;
        for (int round = 0; round < 10; round++)
        {
            if (round > 0)
            {
                k0 += W0;
                k1 += W1;
            }

            ulong hi0 = Math.BigMul(M0, c0, out ulong lo0);
            ulong hi1 = Math.BigMul(M1, c2, out ulong lo1);
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
        }

        return new PhiloxBlock(c0, c1, c2, c3);
    }

    /// <summary>The spec's signature (section 5.3): writes the block into <paramref name="out4"/>.</summary>
    public static void Block(in PhiloxKey key, ulong c0, ulong c1, ulong c2, ulong c3, Span<ulong> out4)
    {
        if (out4.Length < 4)
        {
            throw new ArgumentException("needs room for 4 words", nameof(out4));
        }

        PhiloxBlock b = Block(key, c0, c1, c2, c3);
        out4[0] = b.W0;
        out4[1] = b.W1;
        out4[2] = b.W2;
        out4[3] = b.W3;
    }

    /// <summary>Vigna's SplitMix64 finaliser: decorrelates consecutive seeds before keying.</summary>
    public static ulong SplitMix64(ulong x)
    {
        ulong z = unchecked(x + 0x9E3779B97F4A7C15UL);
        z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
        z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
        return z ^ (z >> 31);
    }
}
