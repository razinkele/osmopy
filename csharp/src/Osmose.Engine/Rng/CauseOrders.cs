namespace Osmose.Engine.Rng;

/// <summary>
/// Lehmer tables decoding one bounded draw into an order of the mortality causes (spec section
/// 5.3): 4 causes without bioenergetics (24 orders), 5 with it (120). A uniform code gives a
/// uniform order, the same distribution as Python's per-slot shuffle, for one draw instead of
/// three or four.
/// </summary>
public static class CauseOrders
{
    private static readonly byte[] Table4 = Build(4);
    private static readonly byte[] Table5 = Build(5);

    /// <summary>Number of orders of <paramref name="nCauses"/> causes (4 or 5).</summary>
    public static int Count(int nCauses) => nCauses switch
    {
        4 => 24,
        5 => 120,
        _ => throw new ArgumentOutOfRangeException(nameof(nCauses), nCauses, "4 or 5 causes"),
    };

    /// <summary>The order with Lehmer code <paramref name="code"/>, in <c>[0, Count(nCauses))</c>.</summary>
    public static ReadOnlySpan<byte> Decode(int nCauses, int code)
    {
        if ((uint)code >= (uint)Count(nCauses))
        {
            throw new ArgumentOutOfRangeException(nameof(code), code, "outside [0, n!)");
        }

        byte[] table = nCauses == 4 ? Table4 : Table5;
        return table.AsSpan(code * nCauses, nCauses);
    }

    private static byte[] Build(int n)
    {
        int count = n == 4 ? 24 : 120;
        var table = new byte[count * n];
        Span<byte> remaining = stackalloc byte[n];
        for (int code = 0; code < count; code++)
        {
            for (int i = 0; i < n; i++)
            {
                remaining[i] = (byte)i;
            }

            int rest = code;
            int left = n;
            for (int position = 0; position < n; position++)
            {
                int radix = Factorial(left - 1);
                int digit = rest / radix;
                rest %= radix;
                table[(code * n) + position] = remaining[digit];
                remaining[(digit + 1)..left].CopyTo(remaining[digit..]);
                left--;
            }
        }

        return table;
    }

    private static int Factorial(int k)
    {
        int f = 1;
        for (int i = 2; i <= k; i++)
        {
            f *= i;
        }

        return f;
    }
}
