using System.Globalization;

namespace Osmose.Engine;

/// <summary>A bundle file is structurally broken; every problem found is listed.</summary>
public sealed class BundleFormatException : Exception
{
    /// <summary>Wraps the problems found in <paramref name="path"/>.</summary>
    public BundleFormatException(string path, IReadOnlyList<string> problems)
        : base($"{path}: {problems.Count} problem(s) in the bundle file:\n  - " + string.Join("\n  - ", problems))
    {
        Problems = problems;
    }

    /// <summary>One line per problem.</summary>
    public IReadOnlyList<string> Problems { get; }
}

/// <summary>A bundle does not match what the engine needs (wrong version, missing entries, wrong
/// dtype or shape). Every problem is listed at once (spec section 8.2).</summary>
public sealed class BundleValidationException : Exception
{
    /// <summary>Wraps the collected problems.</summary>
    public BundleValidationException(IReadOnlyList<string> problems)
        : base($"{problems.Count} problem(s) in the parameter bundle:\n  - " + string.Join("\n  - ", problems))
    {
        Problems = problems;
    }

    /// <summary>One line per problem.</summary>
    public IReadOnlyList<string> Problems { get; }
}

/// <summary>
/// Reads a bundle while collecting problems instead of stopping at the first. The engine's typed
/// parameters (Phase 1) are built through one reader and then <see cref="ThrowIfProblems"/>, so a
/// broken bundle reports every missing entry, wrong dtype and wrong shape in one exception.
/// </summary>
public sealed class BundleReader
{
    /// <summary>A shape dimension that matches any size.</summary>
    public const long Any = -1;

    private readonly SimulationConfig _config;
    private readonly List<string> _problems = [];

    /// <summary>Starts reading <paramref name="config"/>, checking its contract version first.</summary>
    public BundleReader(SimulationConfig config, params int[] supportedVersions)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        if (System.Array.IndexOf(supportedVersions, config.ContractVersion) < 0)
        {
            _problems.Add($"contract version {config.ContractVersion} is not supported (this engine reads {string.Join(", ", supportedVersions)})");
        }
    }

    /// <summary>Problems found so far.</summary>
    public IReadOnlyList<string> Problems => _problems;

    /// <summary>Whether the bundle carries an entry of this name (no problem recorded either way).</summary>
    public bool Has(string name) => _config.TryGet(name, out _);

    /// <summary>Records a problem found by the caller.</summary>
    public void Problem(string message) => _problems.Add(message);

    /// <summary>An integer scalar; records a problem and returns 0 when absent or not an integer.</summary>
    public long Int(string name) => Scalar(name, EntryKind.Int)?.IntValue ?? 0;

    /// <summary>A float scalar; records a problem and returns NaN when absent or not a float.</summary>
    public double Float(string name) => Scalar(name, EntryKind.Float)?.FloatValue ?? double.NaN;

    /// <summary>An array of <paramref name="dtype"/> and <paramref name="shape"/> (<see cref="Any"/>
    /// matches any size); records a problem and returns null on any mismatch.</summary>
    public BundleEntry? Array(string name, DType dtype, params long[] shape)
    {
        if (!_config.TryGet(name, out BundleEntry entry))
        {
            _problems.Add($"{name}: missing (expected {dtype} array {Describe(shape)})");
            return null;
        }

        if (entry.Kind != EntryKind.Array)
        {
            _problems.Add($"{name}: expected {dtype} array {Describe(shape)}, found {entry.Kind} scalar");
            return null;
        }

        bool ok = true;
        if (entry.DType != dtype)
        {
            _problems.Add($"{name}: dtype {entry.DType}, expected {dtype}");
            ok = false;
        }

        bool shapeOk = entry.Rank == shape.Length;
        for (int d = 0; shapeOk && d < shape.Length; d++)
        {
            shapeOk = shape[d] == Any || shape[d] == entry.Shape[d];
        }

        if (!shapeOk)
        {
            _problems.Add($"{name}: shape {Describe(entry.Shape)}, expected {Describe(shape)}");
            ok = false;
        }

        return ok ? entry : null;
    }

    /// <summary>Throws <see cref="BundleValidationException"/> listing every problem, if any.</summary>
    public void ThrowIfProblems()
    {
        if (_problems.Count > 0)
        {
            throw new BundleValidationException([.. _problems]);
        }
    }

    private BundleEntry? Scalar(string name, EntryKind kind)
    {
        if (!_config.TryGet(name, out BundleEntry entry))
        {
            _problems.Add($"{name}: missing (expected {kind} scalar)");
            return null;
        }

        if (entry.Kind != kind)
        {
            _problems.Add($"{name}: expected {kind} scalar, found {entry.Kind}");
            return null;
        }

        return entry;
    }

    private static string Describe(ReadOnlySpan<long> shape)
    {
        var parts = new string[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            parts[i] = shape[i] == Any ? "*" : shape[i].ToString(CultureInfo.InvariantCulture);
        }

        return "(" + string.Join(", ", parts) + ")";
    }
}
