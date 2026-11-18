namespace Osmose.Engine;

/// <summary>
/// The engine's single entry point (spec section 8.3). Python resolves every key, file and
/// name into a <see cref="SimulationConfig"/>; this method runs the whole simulation and
/// writes its results into buffers the caller owns.
/// </summary>
public static class OsmoseEngine
{
    /// <summary>
    /// Runs the whole simulation. Blocking, thread-safe and re-entrant: no static mutable
    /// state, one run object per call.
    /// </summary>
    /// <exception cref="BundleValidationException">The bundle is malformed (every problem listed).</exception>
    /// <exception cref="UnsupportedFeatureException">
    /// The bundle needs a feature this engine does not implement. Until task 1.8 that includes
    /// <c>core</c>, so every valid bundle is refused here, by name (fail closed, spec D11).
    /// </exception>
    public static RunSummary RunSimulation(SimulationConfig config)
    {
        EngineParameters parameters = EngineParameters.From(config);
        throw new NotImplementedException(
            $"unreachable until core is implemented (task 1.8); bundle declared {parameters.Features.Count} features");
    }
}
