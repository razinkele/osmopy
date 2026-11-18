using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.RegularExpressions;

namespace Osmose.Engine;

/// <summary>
/// The parameter bundle Python hands to <see cref="OsmoseEngine.RunSimulation"/> (spec sections
/// 8.2 and 8.3): named scalars and arrays, nothing interpreted. Arrays set from Python are
/// borrowed (the caller keeps the NumPy buffer alive and unmoved for the whole call); a bundle
/// loaded from a file owns its memory map until <see cref="Dispose"/>.
/// </summary>
/// <remarks>
/// File format version 1, all integers little-endian:
/// <list type="bullet">
/// <item>Header, 64 bytes: magic <c>OSMBNDL1</c>; u32 format version; i32 contract version;
/// u32 entry count; u32 zero; u64 table offset (64); u64 data offset; zero padding.</item>
/// <item>Table, one 128-byte record per entry, sorted by name (ordinal): name, 64 bytes ASCII
/// NUL-padded; u8 kind; u8 dtype (0 for scalars); u8 rank; 5 zero bytes; i64 dims[4] (0 when
/// unused); i64 scalar bits or array file offset; i64 array byte length (0 for scalars);
/// 8 zero bytes.</item>
/// <item>Data: each array at a 64-byte-aligned file offset, in table order, zero-padded.</item>
/// </list>
/// Sorting makes the file canonical, so two writers agree byte for byte.
/// </remarks>
public sealed partial class SimulationConfig : IDisposable
{
    /// <summary>Version of the file layout above.</summary>
    public const int FileFormatVersion = 1;

    /// <summary>Longest entry name, in bytes (one table slot keeps a NUL).</summary>
    public const int MaxNameLength = 63;

    internal const int HeaderSize = 64;
    internal const int RecordSize = 128;
    internal const int Alignment = 64;
    internal static readonly byte[] Magic = "OSMBNDL1"u8.ToArray();

    private readonly SortedDictionary<string, BundleEntry> _entries = new(StringComparer.Ordinal);
    private MemoryMappedFile? _map;

    /// <summary>Loaded from a bundle file: its arrays are a READ-ONLY memory map, so the engine
    /// never writes results or control slots into it (a write would kill the process).</summary>
    internal bool IsFileBacked => _map is not null;
    private MemoryMappedViewAccessor? _view;

    /// <summary>Creates an empty bundle for one contract version.</summary>
    public SimulationConfig(int contractVersion)
    {
        ContractVersion = contractVersion;
    }

    /// <summary>The contract version the bundle builder wrote.</summary>
    public int ContractVersion { get; }

    /// <summary>Number of entries.</summary>
    public int Count => _entries.Count;

    /// <summary>All entries, sorted by name.</summary>
    public IEnumerable<BundleEntry> Entries => _entries.Values;

    /// <summary>Adds a 64-bit integer scalar.</summary>
    public void SetInt(string name, long value) =>
        Add(new BundleEntry(CheckName(name), EntryKind.Int, default, [], value, 0));

    /// <summary>Adds a 64-bit float scalar.</summary>
    public void SetFloat(string name, double value) =>
        Add(new BundleEntry(CheckName(name), EntryKind.Float, default, [], BitConverter.DoubleToInt64Bits(value), 0));

    /// <summary>
    /// Adds a borrowed C-contiguous little-endian array of shape <c>(d0[, d1[, d2[, d3]]])</c>:
    /// trailing dimensions are -1 when absent. The caller keeps the buffer alive and unmoved
    /// until the bundle is no longer used. Addresses are <see cref="long"/> because pythonnet
    /// converts a Python int to Int64 explicitly (spec section 8.3).
    /// </summary>
    public void SetArray(string name, DType dtype, long address, long d0, long d1 = -1, long d2 = -1, long d3 = -1)
    {
        CheckName(name);
        long[] shape = CheckShape(name, d0, d1, d2, d3);
        _ = BundleEntry.ItemSize(dtype); // rejects unknown codes
        long length = shape.Aggregate(1L, (a, b) => checked(a * b));
        if (length > 0 && address == 0)
        {
            throw new ArgumentException($"{name}: null address for a non-empty array", nameof(address));
        }

        Add(new BundleEntry(name, EntryKind.Array, dtype, shape, 0, (nint)address));
    }

    /// <summary>The entry called <paramref name="name"/>, if present.</summary>
    public bool TryGet(string name, out BundleEntry entry) => _entries.TryGetValue(name, out entry!);

    /// <summary>Writes the bundle in the canonical file format (class remarks).</summary>
    public unsafe void Save(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        BundleEntry[] entries = [.. _entries.Values];
        long dataOffset = Align(HeaderSize + ((long)entries.Length * RecordSize));
        var offsets = new long[entries.Length];
        long cursor = dataOffset;
        for (int i = 0; i < entries.Length; i++)
        {
            if (entries[i].Kind == EntryKind.Array)
            {
                offsets[i] = cursor;
                cursor = Align(cursor + entries[i].ByteLength);
            }
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[HeaderSize];
        header.Clear();
        Magic.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header[8..], FileFormatVersion);
        BinaryPrimitives.WriteInt32LittleEndian(header[12..], ContractVersion);
        BinaryPrimitives.WriteUInt32LittleEndian(header[16..], (uint)entries.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(header[24..], HeaderSize);
        BinaryPrimitives.WriteUInt64LittleEndian(header[32..], (ulong)dataOffset);
        stream.Write(header);

        Span<byte> record = stackalloc byte[RecordSize];
        for (int i = 0; i < entries.Length; i++)
        {
            BundleEntry e = entries[i];
            record.Clear();
            Encoding.ASCII.GetBytes(e.Name, record);
            record[64] = (byte)e.Kind;
            record[65] = (byte)e.DType;
            record[66] = (byte)e.Rank;
            for (int d = 0; d < e.Rank; d++)
            {
                BinaryPrimitives.WriteInt64LittleEndian(record[(72 + (8 * d))..], e.Shape[d]);
            }

            BinaryPrimitives.WriteInt64LittleEndian(record[104..], e.Kind == EntryKind.Array ? offsets[i] : e.Bits);
            BinaryPrimitives.WriteInt64LittleEndian(record[112..], e.ByteLength);
            stream.Write(record);
        }

        Pad(stream, dataOffset);
        for (int i = 0; i < entries.Length; i++)
        {
            BundleEntry e = entries[i];
            if (e.Kind != EntryKind.Array)
            {
                continue;
            }

            Pad(stream, offsets[i]);
            byte* p = (byte*)e.Address;
            for (long done = 0; done < e.ByteLength;)
            {
                int chunk = (int)Math.Min(e.ByteLength - done, 1 << 30);
                stream.Write(new ReadOnlySpan<byte>(p + done, chunk));
                done += chunk;
            }
        }

        Pad(stream, cursor);
    }

    /// <summary>Opens a bundle file. Its arrays point into a memory map held until <see cref="Dispose"/>.
    /// Every structural problem is reported at once (<see cref="BundleFormatException"/>).</summary>
    public static unsafe SimulationConfig Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        long fileLength = new FileInfo(path).Length;
        if (fileLength < HeaderSize)
        {
            throw new BundleFormatException(path, [$"file is {fileLength} bytes, shorter than the {HeaderSize}-byte header"]);
        }

        var map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        MemoryMappedViewAccessor view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* basePtr = null;
        try
        {
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            basePtr += view.PointerOffset;
            var file = new ReadOnlySpan<byte>(basePtr, checked((int)Math.Min(fileLength, int.MaxValue)));
            var problems = new List<string>();
            if (!file[..8].SequenceEqual(Magic))
            {
                throw new BundleFormatException(path, ["not a bundle file (bad magic)"]);
            }

            uint format = BinaryPrimitives.ReadUInt32LittleEndian(file[8..]);
            if (format != FileFormatVersion)
            {
                throw new BundleFormatException(path, [$"file format {format}, this engine reads {FileFormatVersion}"]);
            }

            int contract = BinaryPrimitives.ReadInt32LittleEndian(file[12..]);
            uint count = BinaryPrimitives.ReadUInt32LittleEndian(file[16..]);
            ulong tableOffset = BinaryPrimitives.ReadUInt64LittleEndian(file[24..]);
            ulong dataOffset = BinaryPrimitives.ReadUInt64LittleEndian(file[32..]);
            ulong tableEnd = tableOffset + ((ulong)count * RecordSize);
            if (tableOffset != HeaderSize || tableEnd > (ulong)fileLength || dataOffset < tableEnd)
            {
                throw new BundleFormatException(path, [$"table of {count} entries does not fit (offset {tableOffset}, data at {dataOffset}, file {fileLength})"]);
            }

            var config = new SimulationConfig(contract);
            string? previous = null;
            for (uint i = 0; i < count; i++)
            {
                ReadOnlySpan<byte> rec = file.Slice((int)(tableOffset + (i * RecordSize)), RecordSize);
                int nul = rec[..64].IndexOf((byte)0);
                string name = Encoding.ASCII.GetString(rec[..(nul < 0 ? 64 : nul)]);
                string where = $"entry {i} ({name})";
                if (!IsValidName(name))
                {
                    problems.Add($"{where}: invalid name");
                    continue;
                }

                if (previous is not null && string.CompareOrdinal(previous, name) >= 0)
                {
                    problems.Add($"{where}: table not sorted or duplicated after {previous}");
                }

                previous = name;
                var kind = (EntryKind)rec[64];
                var dtype = (DType)rec[65];
                int rank = rec[66];
                long bits = BinaryPrimitives.ReadInt64LittleEndian(rec[104..]);
                long byteLength = BinaryPrimitives.ReadInt64LittleEndian(rec[112..]);
                if (kind is EntryKind.Int or EntryKind.Float)
                {
                    if (rec[65] != 0 || rank != 0 || byteLength != 0)
                    {
                        problems.Add($"{where}: a scalar with dtype/rank/length set");
                        continue;
                    }

                    config.Add(new BundleEntry(name, kind, default, [], bits, 0));
                    continue;
                }

                if (kind != EntryKind.Array || !Enum.IsDefined(dtype) || rank is < 1 or > 4)
                {
                    problems.Add($"{where}: bad kind {rec[64]}, dtype {rec[65]} or rank {rank}");
                    continue;
                }

                var shape = new long[rank];
                long length = 1;
                bool shapeOk = true;
                for (int d = 0; d < rank; d++)
                {
                    shape[d] = BinaryPrimitives.ReadInt64LittleEndian(rec[(72 + (8 * d))..]);
                    shapeOk &= shape[d] >= 0;
                    length = shapeOk ? checked(length * shape[d]) : 0;
                }

                long expected = length * BundleEntry.ItemSize(dtype);
                if (!shapeOk || byteLength != expected)
                {
                    problems.Add($"{where}: shape ({string.Join(", ", shape)}) needs {expected} bytes, table says {byteLength}");
                    continue;
                }

                if (bits % Alignment != 0 || (ulong)bits < dataOffset || bits + byteLength > fileLength)
                {
                    problems.Add($"{where}: data at offset {bits} (+{byteLength}) is misaligned or outside the file");
                    continue;
                }

                config.Add(new BundleEntry(name, kind, dtype, shape, 0, (nint)(basePtr + bits)));
            }

            if (problems.Count > 0)
            {
                throw new BundleFormatException(path, problems);
            }

            config._map = map;
            config._view = view;
            return config;
        }
        catch
        {
            if (basePtr != null)
            {
                view.SafeMemoryMappedViewHandle.ReleasePointer();
            }

            view.Dispose();
            map.Dispose();
            throw;
        }
    }

    /// <summary>Releases a loaded bundle's memory map. Its array entries are invalid afterwards.</summary>
    public void Dispose()
    {
        if (_view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
            _view = null;
        }

        _map?.Dispose();
        _map = null;
    }

    /// <summary>Whether <paramref name="name"/> is a valid entry name: lowercase dot-separated
    /// words of <c>[a-z0-9_]</c>, at most <see cref="MaxNameLength"/> bytes.</summary>
    public static bool IsValidName(string? name) =>
        name is { Length: > 0 and <= MaxNameLength } && NamePattern().IsMatch(name);

    [GeneratedRegex(@"^[a-z0-9_]+(\.[a-z0-9_]+)*$", RegexOptions.CultureInvariant)]
    private static partial Regex NamePattern();

    private static string CheckName(string name) =>
        IsValidName(name)
            ? name
            : throw new ArgumentException(
                $"invalid entry name '{name}': lowercase dot-separated [a-z0-9_] words, at most {MaxNameLength} bytes",
                nameof(name));

    private static long[] CheckShape(string name, long d0, long d1, long d2, long d3)
    {
        long[] dims = [d0, d1, d2, d3];
        int rank = Array.FindLastIndex(dims, d => d != -1) + 1;
        for (int i = 0; i < rank; i++)
        {
            if (dims[i] < 0)
            {
                throw new ArgumentException($"{name}: dimension {i} is {dims[i]}; only trailing dimensions may be -1 (absent)");
            }
        }

        return rank == 0
            ? throw new ArgumentException($"{name}: an array needs at least one dimension")
            : dims[..rank];
    }

    private void Add(BundleEntry entry)
    {
        if (!_entries.TryAdd(entry.Name, entry))
        {
            throw new ArgumentException($"entry '{entry.Name}' is already set", nameof(entry));
        }
    }

    private static long Align(long offset) => (offset + Alignment - 1) / Alignment * Alignment;

    private static void Pad(FileStream stream, long target)
    {
        Span<byte> zeros = stackalloc byte[Alignment];
        zeros.Clear();
        while (stream.Position < target)
        {
            stream.Write(zeros[..(int)Math.Min(Alignment, target - stream.Position)]);
        }
    }
}
