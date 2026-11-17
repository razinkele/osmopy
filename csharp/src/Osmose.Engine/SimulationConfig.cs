namespace Osmose.Engine;

/// <summary>
/// The parameter bundle Python hands to <see cref="OsmoseEngine.RunSimulation"/>
/// (spec section 8.2). A placeholder until task 0.3 adds the named, typed entries and the
/// bundle-file format.
/// </summary>
public sealed class SimulationConfig
{
    /// <summary>Creates an empty bundle for one contract version.</summary>
    public SimulationConfig(int contractVersion)
    {
        ContractVersion = contractVersion;
    }

    /// <summary>The contract version the Python bundle builder wrote.</summary>
    public int ContractVersion { get; }
}
