using VL53L8.Services;

namespace VL53L8.Views;

public sealed class HeatmapDrawable : IDrawable
{
    public const int Resolution = 96;
    private Color[]? _pixels;
    private static readonly Color Empty = Color.FromArgb("#24445E");
    private static readonly Color[] Palette = [Color.FromArgb("#EF4444"), Color.FromArgb("#F97316"),
        Color.FromArgb("#FACC15"), Color.FromArgb("#22C55E"), Color.FromArgb("#06B6D4"), Color.FromArgb("#2563EB")];

    public void Update(ushort[] distances)
    {
        var pixels = new Color[Resolution * Resolution];
        for (int y = 0; y < Resolution; y++)
        for (int x = 0; x < Resolution; x++)
        {
            var value = HeatmapInterpolation.Sample(distances, x * 7.0 / (Resolution - 1), y * 7.0 / (Resolution - 1));
            pixels[y * Resolution + x] = value.HasValue ? DistanceColor(value.Value) : Empty;
        }
        _pixels = pixels;
    }

    public static Color DistanceColor(double distance)
    {
        double position = Math.Clamp(distance / 3000.0, 0, 1) * (Palette.Length - 1);
        int index = Math.Min((int)position, Palette.Length - 2);
        float t = (float)(position - index);
        Color a = Palette[index], b = Palette[index + 1];
        return new Color(a.Red + (b.Red - a.Red) * t, a.Green + (b.Green - a.Green) * t, a.Blue + (b.Blue - a.Blue) * t);
    }

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        canvas.FillColor = Empty; canvas.FillRectangle(dirtyRect);
        if (_pixels is null) return;
        canvas.Antialias = false;
        float side = Math.Min(dirtyRect.Width, dirtyRect.Height);
        float left = dirtyRect.X + (dirtyRect.Width - side) / 2;
        float top = dirtyRect.Y + (dirtyRect.Height - side) / 2;
        float step = side / Resolution;
        for (int y = 0; y < Resolution; y++)
        for (int x = 0; x < Resolution; x++)
        {
            canvas.FillColor = _pixels[y * Resolution + x];
            canvas.FillRectangle(left + x * step, top + y * step, step + 0.2f, step + 0.2f);
        }
    }
}
