using System.Text.Json;

namespace Osmose.Engine;

/// <summary>One catalogue entry (spec D11). <see cref="Status"/>: planned, refused, not_planned.</summary>
public sealed record FeatureInfo(int Id, string Name, int Phase, string Status, bool Implemented, string Description)
{
    /// <summary>A one-line reason a bundle needing this feature is refused.</summary>
    public string RefusalReason => Status switch
    {
        "planned" => $"{Name} (not implemented yet; phase {Phase})",
        "refused" => $"{Name} (refused: spec Q4)",
        _ => $"{Name} (not planned for the C# engine; use the Python engine)",
    };
}

/// <summary>
/// The feature catalogue, embedded from <c>osmose/engine_cs/features.json</c> at build time: the
/// same file the Python capability check reads, so the two cannot drift. An entry's index in
/// <see cref="All"/> is its id.
/// </summary>
public static class FeatureCatalog
{
    private const string ResourceName = "Osmose.Engine.features.json";

    static FeatureCatalog()
    {
        using Stream stream = typeof(FeatureCatalog).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"embedded resource {ResourceName} is missing");
        using JsonDocument doc = JsonDocument.Parse(stream);
        JsonElement root = doc.RootElement;
        ContractVersions = [.. root.GetProperty("contract_versions").EnumerateArray().Select(v => v.GetInt32())];
        var features = new List<FeatureInfo>();
        foreach (JsonElement f in root.GetProperty("features").EnumerateArray())
        {
            features.Add(new FeatureInfo(
                f.GetProperty("id").GetInt32(),
                f.GetProperty("name").GetString()!,
                f.GetProperty("phase").GetInt32(),
                f.GetProperty("status").GetString()!,
                f.GetProperty("implemented").GetBoolean(),
                f.GetProperty("description").GetString()!));
        }

        for (int i = 0; i < features.Count; i++)
        {
            if (features[i].Id != i)
            {
                throw new InvalidOperationException($"features.json: entry {i} has id {features[i].Id}; ids must be 0..n-1 in order");
            }
        }

        All = features;
    }

    /// <summary>Every feature, indexed by id.</summary>
    public static IReadOnlyList<FeatureInfo> All { get; }

    /// <summary>The bundle contract versions this engine reads.</summary>
    public static IReadOnlyList<int> ContractVersions { get; }

    /// <summary>The <c>core</c> feature (id 0): the Phase 1 engine itself, required by every bundle.</summary>
    public static FeatureInfo Core => All[0];
}
