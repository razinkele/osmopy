namespace Osmose.Engine.Mortality;

/// <summary>
/// Per-step constants the mortality kernel reads (spec section 4.3's <c>StepContext</c>, for the
/// kernel spike). The accessibility matrix is stored PREDATOR-major whichever mode it came from,
/// so one predator scanning many prey reads one contiguous row.
/// </summary>
public sealed class KernelTables
{
    /// <summary>Number of focal species (<c>config.n_species</c>); resources follow it in
    /// species-mode accessibility columns.</summary>
    public required int NSpeciesFocal { get; init; }

    /// <summary>Width of the size-ratio tables (feeding stages).</summary>
    public required int NStage { get; init; }

    public required int NDtPerYear { get; init; }

    public required int NSubdt { get; init; }

    public required int NResources { get; init; }

    /// <summary><c>size_ratio_min[species, stage]</c>, flattened row-major.</summary>
    public required double[] SizeRatioMin { get; init; }

    /// <summary><c>size_ratio_max[species, stage]</c>, flattened row-major.</summary>
    public required double[] SizeRatioMax { get; init; }

    public required double[] IngestionRate { get; init; }

    /// <summary>Holling form per species: 1, 2 or 3.</summary>
    public required int[] FrShape { get; init; }

    public required double[] FrHalfSat { get; init; }

    public required bool HasAccess { get; init; }

    public required bool UseStageAccess { get; init; }

    /// <summary><c>AccessByPred[pred * AccessPreyCount + prey]</c>.</summary>
    public required double[] AccessByPred { get; init; }

    public required int AccessPreyCount { get; init; }

    public required int AccessPredCount { get; init; }

    /// <summary>Stage-mode accessibility row of each resource; -1 when unresolved.</summary>
    public required int[] ResourceAccessRows { get; init; }

    public required double[] RscSizeMin { get; init; }

    public required double[] RscSizeMax { get; init; }

    public required double[] RscTrophicLevel { get; init; }

    /// <summary>
    /// Builds the predator-major matrix from Python's layout: stage mode holds
    /// <c>[prey_row, pred_col]</c>, species mode <c>[pred_species, prey_species]</c>.
    /// </summary>
    public static (double[] ByPred, int PreyCount, int PredCount) PredatorMajor(
        ReadOnlySpan<double> pythonMatrix, int rows, int cols, bool stageMode)
    {
        if (!stageMode)
        {
            return (pythonMatrix.ToArray(), cols, rows);
        }

        var byPred = new double[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                byPred[(c * rows) + r] = pythonMatrix[(r * cols) + c];
            }
        }

        return (byPred, rows, cols);
    }
}
