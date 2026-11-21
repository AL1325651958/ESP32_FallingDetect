namespace VL53L8.Services;

/// <summary>Coordinates are in the original row-major 8 by 8 sensor grid.</summary>
public static class HeatmapInterpolation
{
    public static double? Sample(ushort[] distances, double x, double y)
    {
        if (distances.Length != 64) throw new ArgumentException("Expected 64 distances.", nameof(distances));
        x = Math.Clamp(x, 0, 7); y = Math.Clamp(y, 0, 7);
        // Keep invalid source zones visibly empty instead of inventing a measurement.
        if (distances[(int)Math.Round(y) * 8 + (int)Math.Round(x)] == ushort.MaxValue) return null;
        int x0 = (int)x, y0 = (int)y, x1 = Math.Min(x0 + 1, 7), y1 = Math.Min(y0 + 1, 7);
        double tx = x - x0, ty = y - y0, total = 0, weight = 0;
        Add(x0, y0, (1 - tx) * (1 - ty)); Add(x1, y0, tx * (1 - ty));
        Add(x0, y1, (1 - tx) * ty); Add(x1, y1, tx * ty);
        return weight > 0 ? total / weight : null;
        void Add(int column, int row, double w)
        {
            ushort value = distances[row * 8 + column];
            if (value == ushort.MaxValue) return;
            total += value * w; weight += w;
        }
    }
}
