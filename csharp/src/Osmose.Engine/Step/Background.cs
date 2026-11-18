using Osmose.Engine.Schools;

namespace Osmose.Engine.Step;

/// <summary>
/// Background (forced) species (plan task 2.1, feature <c>background_species</c>): each step, one
/// school per species, class and ocean cell is appended after movement, takes part in mortality,
/// has its remaining biomass and abundance added to the outputs, and is removed again.
/// </summary>
/// <remarks>
/// Tier E against the Python engine: <see cref="Inject"/> builds the schools
/// <c>BackgroundState.get_schools</c> builds, in its order (species, class, ocean cell), with its
/// arithmetic (<c>biomass = base * proportion</c>, <c>abundance = biomass / weight</c>), and
/// <see cref="CollectAndRemove"/> adds them up in that same order, as
/// <c>_collect_background_outputs</c>'s <c>np.add.at</c> does, although the store has been
/// regrouped by cell in between.
/// </remarks>
public static class Background
{
    /// <summary>Schools a step injects: every class of every species on every ocean cell.</summary>
    public static int SchoolsPerStep(BackgroundParameters b) => b.NClasses * b.OceanCells.Length;

    /// <summary>
    /// Appends the background schools of <paramref name="step"/> and returns the id of the first
    /// one (ids are consecutive, in injection order). A school is eligible to feed at any age
    /// (first feeding age -1); one on a cell with no biomass is appended at abundance 0 and
    /// dropped by the next regroup.
    /// </summary>
    public static ulong Inject(EngineParameters p, SchoolStore store, int step) =>
        Inject(p.Background, p.Species.NFocal, store, step);

    /// <summary>As <see cref="Inject(EngineParameters, SchoolStore, int)"/>, on the background
    /// group alone.</summary>
    public static ulong Inject(BackgroundParameters b, int nFocal, SchoolStore store, int step)
    {
        int nOcean = b.OceanCells.Length;
        ulong first = store.NextId;
        for (int k = 0; k < b.NSpecies; k++)
        {
            ReadOnlySpan<double> biomass = b.Base.AsSpan(b.FrameByStep[(step * b.NSpecies) + k] * nOcean, nOcean);
            for (int c = b.ClassOffsets[k]; c < b.ClassOffsets[k + 1]; c++)
            {
                double proportion = b.ProportionByStep[(step * b.NClasses) + c];
                double weight = b.ClassWeight[c];
                var school = new SchoolCore
                {
                    SpeciesId = nFocal + k,
                    AgeDt = b.ClassAgeDt[c],
                    Length = b.ClassLength[c],
                    Weight = weight,
                    TrophicLevel = b.ClassTrophicLevel[c],
                    FirstFeedingAgeDt = -1,
                    Flags = SchoolFlags.IsBackground,
                };
                for (int o = 0; o < nOcean; o++)
                {
                    school.Abundance = weight > 0.0 ? biomass[o] * proportion / weight : 0.0;
                    store.Append(school, new SchoolLife { CellId = b.OceanCells[o] });
                }
            }
        }

        return first;
    }

    /// <summary>
    /// After mortality: adds each background school's biomass (<c>abundance * weight</c>) and
    /// abundance to its species' totals, in injection order, then removes the background schools
    /// from the store. <paramref name="firstId"/> is what <see cref="Inject"/> returned;
    /// <paramref name="byOrdinal"/> is scratch of at least twice <see cref="SchoolsPerStep"/>.
    /// The totals are written for the background species (indices <c>nFocal..</c>) only.
    /// </summary>
    public static void CollectAndRemove(
        EngineParameters p, SchoolStore store, ulong firstId, Span<double> byOrdinal, Span<double> biomass, Span<double> abundance) =>
        CollectAndRemove(p.Background, p.Species.NFocal, store, firstId, byOrdinal, biomass, abundance);

    /// <summary>As the overload on <see cref="EngineParameters"/>, on the background group alone.</summary>
    public static void CollectAndRemove(
        BackgroundParameters b, int nFocal, SchoolStore store, ulong firstId, Span<double> byOrdinal, Span<double> biomass, Span<double> abundance)
    {
        int nOcean = b.OceanCells.Length, n = SchoolsPerStep(b);
        Span<double> schoolBiomass = byOrdinal[..n], schoolAbundance = byOrdinal.Slice(n, n);
        schoolBiomass.Clear();
        schoolAbundance.Clear();
        ReadOnlySpan<SchoolCore> core = store.Core;
        ReadOnlySpan<SchoolLife> life = store.Life;
        for (int i = 0; i < core.Length; i++)
        {
            if ((core[i].Flags & SchoolFlags.IsBackground) != 0)
            {
                int ordinal = checked((int)(life[i].Id - firstId));
                schoolBiomass[ordinal] = core[i].Abundance * core[i].Weight;
                schoolAbundance[ordinal] = core[i].Abundance;
            }
        }

        for (int k = 0; k < b.NSpecies; k++)
        {
            double sumBiomass = 0.0, sumAbundance = 0.0;
            int from = b.ClassOffsets[k] * nOcean, to = b.ClassOffsets[k + 1] * nOcean;
            for (int j = from; j < to; j++)
            {
                sumBiomass += schoolBiomass[j];
                sumAbundance += schoolAbundance[j];
            }

            biomass[nFocal + k] = sumBiomass;
            abundance[nFocal + k] = sumAbundance;
        }

        store.RemoveBackground();
    }
}
