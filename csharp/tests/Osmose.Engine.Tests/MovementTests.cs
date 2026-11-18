using Osmose.Engine.Step;
using Xunit;

namespace Osmose.Engine.Tests;

public class MovementTests
{
    [Theory]
    [InlineData(1.0, 0.0, 0.999, true)] // presence map: any placeable value, whatever the draw
    [InlineData(0.3, 0.0, 0.999, true)]
    [InlineData(0.0, 0.0, 0.0, false)] // not placeable
    [InlineData(-99.0, 0.0, 0.0, false)] // the land marker of the shipped maps
    [InlineData(double.NaN, 0.0, 0.0, false)]
    [InlineData(double.NaN, 1.0, 0.0, false)]
    [InlineData(0.5, 1.0, 0.5, true)] // value == u * maxP: accepted (>=)
    [InlineData(0.5, 1.0, 0.5000000001, false)]
    [InlineData(0.25, 0.5, 0.0, true)] // proportional map, a zero draw always accepts
    [InlineData(-0.25, 0.5, 0.0, false)]
    public void Accepts_IsPythonsRule(double value, double maxP, double u, bool expected)
    {
        Assert.Equal(expected, Movement.Accepts(value, maxP, u));
    }

    [Fact]
    public void IsPlaceable_RejectsZeroNegativeAndNaN()
    {
        Assert.True(Movement.IsPlaceable(double.Epsilon));
        Assert.False(Movement.IsPlaceable(0.0));
        Assert.False(Movement.IsPlaceable(-0.0));
        Assert.False(Movement.IsPlaceable(double.NaN));
        Assert.True(Movement.IsPlaceable(double.PositiveInfinity));
    }

    [Fact]
    public void MapIndex_GuardsTheTableAndSameMapNeedsAgeAndStep()
    {
        // One species, a 3-age x 4-step table: index = age * 10 + step, so every cell is distinct
        // except where set equal to the previous diagonal.
        int[] table = new int[12];
        for (int a = 0; a < 3; a++)
        {
            for (int s = 0; s < 4; s++)
            {
                table[(a * 4) + s] = (a * 10) + s;
            }
        }

        table[(2 * 4) + 3] = table[(1 * 4) + 2]; // age 2, step 3 keeps age 1, step 2's map
        var m = new MovementParameters
        {
            MapSpeciesOffset = [0],
            MapIndex = table,
            MapIndexOffsets = [0, 12],
            MapIndexShape = [3, 4],
        };
        Assert.Equal(12, Movement.MapIndex(m, 0, 1, 2));
        Assert.Equal(-1, Movement.MapIndex(m, 0, 3, 0)); // age past the table
        Assert.Equal(-1, Movement.MapIndex(m, 0, 0, 4)); // step past the table
        Assert.Equal(-1, Movement.MapIndex(m, 0, -1, 0));
        Assert.Equal(-1, Movement.MapIndex(m, 1, 0, 0)); // a species without maps

        Assert.True(Movement.SameMap(m, 0, 2, 3, Movement.MapIndex(m, 0, 2, 3)));
        Assert.False(Movement.SameMap(m, 0, 1, 2, Movement.MapIndex(m, 0, 1, 2)));
        Assert.False(Movement.SameMap(m, 0, 0, 3, Movement.MapIndex(m, 0, 0, 3))); // age 0
        Assert.False(Movement.SameMap(m, 0, 2, 0, Movement.MapIndex(m, 0, 2, 0))); // step 0

        // Both outside the table: -1 == -1 is "same", as in Python (the school then goes out).
        Assert.True(Movement.SameMap(m, 0, 5, 3, Movement.MapIndex(m, 0, 5, 3)));
    }
}
