using System.Runtime.CompilerServices;

namespace Osmose.Engine;

/// <summary>Element type of a bundle array (spec section 8.2). Values are the on-disk codes.</summary>
public enum DType : byte
{
    /// <summary>float64.</summary>
    F8 = 1,

    /// <summary>int32.</summary>
    I4 = 2,

    /// <summary>int64.</summary>
    I8 = 3,

    /// <summary>uint8 (flags, masks, small enums).</summary>
    U1 = 4,
}

/// <summary>What an entry holds. Values are the on-disk codes.</summary>
public enum EntryKind : byte
{
    /// <summary>A 64-bit integer scalar.</summary>
    Int = 1,

    /// <summary>A 64-bit float scalar.</summary>
    Float = 2,

    /// <summary>A C-contiguous little-endian array of up to four dimensions.</summary>
    Array = 3,
}

/// <summary>One named bundle entry. Arrays point at memory the bundle does not copy: the caller's
/// NumPy buffer (borrowed) or the memory-mapped bundle file (owned by the config).</summary>
public sealed class BundleEntry
{
    private readonly long[] _shape;

    internal BundleEntry(string name, EntryKind kind, DType dtype, long[] shape, long bits, nint address)
    {
        Name = name;
        Kind = kind;
        DType = dtype;
        _shape = shape;
        Bits = bits;
        Address = address;
    }

    /// <summary>Lowercase dot-separated name, e.g. <c>growth.linf</c>.</summary>
    public string Name { get; }

    /// <summary>Scalar or array.</summary>
    public EntryKind Kind { get; }

    /// <summary>Element type of an array; <c>default</c> for scalars.</summary>
    public DType DType { get; }

    /// <summary>Array shape (one to four dimensions); empty for scalars.</summary>
    public ReadOnlySpan<long> Shape => _shape;

    /// <summary>Number of dimensions; 0 for scalars.</summary>
    public int Rank => _shape.Length;

    /// <summary>Raw 64 bits of a scalar (the integer, or the double's bit pattern).</summary>
    internal long Bits { get; }

    /// <summary>Address of an array's first element.</summary>
    internal nint Address { get; }

    /// <summary>The integer value of an <see cref="EntryKind.Int"/> entry.</summary>
    public long IntValue => Kind == EntryKind.Int ? Bits : throw WrongKind(EntryKind.Int);

    /// <summary>The value of a <see cref="EntryKind.Float"/> entry.</summary>
    public double FloatValue => Kind == EntryKind.Float ? BitConverter.Int64BitsToDouble(Bits) : throw WrongKind(EntryKind.Float);

    /// <summary>Number of array elements.</summary>
    public long Length
    {
        get
        {
            long n = 1;
            foreach (long d in _shape)
            {
                n *= d;
            }

            return Kind == EntryKind.Array ? n : 0;
        }
    }

    /// <summary>Array size in bytes.</summary>
    public long ByteLength => Kind == EntryKind.Array ? Length * ItemSize(DType) : 0;

    /// <summary>Bytes per element of <paramref name="dtype"/>.</summary>
    public static int ItemSize(DType dtype) => dtype switch
    {
        DType.F8 or DType.I8 => 8,
        DType.I4 => 4,
        DType.U1 => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "unknown dtype"),
    };

    /// <summary>The array's elements, typed. <typeparamref name="T"/> must match the dtype
    /// (double/F8, int/I4, long/I8, byte/U1).</summary>
    public unsafe ReadOnlySpan<T> AsSpan<T>()
        where T : unmanaged
    {
        if (Kind != EntryKind.Array)
        {
            throw WrongKind(EntryKind.Array);
        }

        if (!Matches<T>(DType))
        {
            throw new InvalidOperationException($"{Name} is {DType}, not {typeof(T).Name}");
        }

        return new ReadOnlySpan<T>((void*)Address, checked((int)Length));
    }

    /// <summary>A copy of the array's elements. For callers that cannot hold a span (pythonnet
    /// cannot marshal one): <c>entry.ToArray[System.Double]()</c>.</summary>
    public T[] ToArray<T>()
        where T : unmanaged => AsSpan<T>().ToArray();

    /// <summary>A copy of <see cref="Shape"/>, for callers that cannot hold a span.</summary>
    public long[] GetShape() => (long[])_shape.Clone();

    internal static bool Matches<T>(DType dtype)
        where T : unmanaged =>
        dtype switch
        {
            DType.F8 => typeof(T) == typeof(double),
            DType.I4 => typeof(T) == typeof(int),
            DType.I8 => typeof(T) == typeof(long),
            DType.U1 => typeof(T) == typeof(byte),
            _ => false,
        } && Unsafe.SizeOf<T>() == ItemSize(dtype);

    private InvalidOperationException WrongKind(EntryKind wanted) =>
        new($"{Name} is {Kind}, not {wanted}");
}
