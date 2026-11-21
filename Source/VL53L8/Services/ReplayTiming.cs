namespace VL53L8.Services;

public static class ReplayTiming
{
    public static TimeSpan Offset(DateTimeOffset origin, DateTimeOffset receivedAt, TimeSpan previousOffset) =>
        receivedAt - origin > previousOffset ? receivedAt - origin : previousOffset;
}
