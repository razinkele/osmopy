using System.Runtime.InteropServices;

namespace Osmose.Engine.Schools;

/// <summary>Per-school boolean state, packed (spec section 3.3).</summary>
[Flags]
public enum SchoolFlags : byte
{
    None = 0,
    IsEgg = 1,
    IsOut = 2,
    IsBackground = 4,
    FromSeeding = 8,
}

/// <summary>
/// Everything a prey scan, a kill or a cause visit touches: one 64-byte cache line (spec D2).
/// </summary>
[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolCore
{
    /// <summary>cm.</summary>
    public double Length;

    /// <summary>Tonnes per fish.</summary>
    public double Weight;

    /// <summary>Instantaneous abundance inside mortality; committed outside it.</summary>
    public double Abundance;

    /// <summary>Eggs still withheld from the prey pool.</summary>
    public double EggRetained;

    /// <summary><c>n_dead[PREDATION]</c>: the only death counter written per eaten prey.</summary>
    public double DeadPredation;

    /// <summary>Read per eaten prey; rewritten after the loop.</summary>
    public double TrophicLevel;

    /// <summary>Focal 0..n-1, background n..n+b-1.</summary>
    public int SpeciesId;

    /// <summary>Age in time steps.</summary>
    public int AgeDt;

    /// <summary>Stage-accessibility prey row; -1 = unresolved, which means FULL access.</summary>
    public short PreyAccessRow;

    /// <summary>Stage-accessibility predator column; -1 = unresolved, which means FULL access.</summary>
    public short PredAccessCol;

    /// <summary>Feeding stage (index into the size-ratio tables).</summary>
    public byte FeedingStage;

    /// <summary>Egg, out of domain, background, seeded.</summary>
    public SchoolFlags Flags;

    /// <summary>Age (time steps) before which the school does not feed.</summary>
    public short FirstFeedingAgeDt;
}

/// <summary>Rebuilt every step: the cause loop's working set (spec D2).</summary>
[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolStep
{
    /// <summary>Per sub-step, already divided.</summary>
    public double RateStarvation;

    /// <summary>Per sub-step, already divided.</summary>
    public double RateAdditional;

    /// <summary>Per sub-step, already divided.</summary>
    public double RateFishing;

    /// <summary>Fraction of the fishing deaths that are discards.</summary>
    public double DiscardFraction;

    /// <summary>Post-larva abundance / n_subdt, released from retention each sub-step (eggs).</summary>
    public double EggRelease;

    /// <summary><c>pred_success_rate</c>.</summary>
    public double PredSuccess;

    /// <summary><c>preyed_biomass</c>.</summary>
    public double PreyedBiomass;

    /// <summary><c>tl_weighted_sum</c>.</summary>
    public double TlWeightedSum;
}

/// <summary>Death accounting other than predation (spec D2).</summary>
[StructLayout(LayoutKind.Sequential, Size = 64)]
public struct SchoolLedger
{
    public double DeadStarvation;
    public double DeadAdditional;
    public double DeadFishing;
    public double DeadOut;
    public double DeadForaging;
    public double DeadDiscards;
    public double DeadAging;
    public double AbundanceAtMortalityStart;
}
