using System.Buffers.Binary;

namespace VL53L8.Services;

/// <summary>A7/v1, one 20-byte BLE notification; probabilities are model scores.</summary>
public sealed class Vl53L8GestureEventArgs : EventArgs
{
    public DateTimeOffset ReceivedAt { get; }
    public ushort FrameNumber { get; }
    public int RawIndex { get; }
    public int StableIndex { get; }
    public byte ValidZones { get; }
    public byte NearZones { get; }
    public float RockProbability { get; }
    public float PaperProbability { get; }
    public float ScissorsProbability { get; }

    private Vl53L8GestureEventArgs(ReadOnlySpan<byte> packet, DateTimeOffset receivedAt)
    {
        ReceivedAt = receivedAt;
        FrameNumber = BinaryPrimitives.ReadUInt16LittleEndian(packet[2..]);
        RawIndex = unchecked((sbyte)packet[4]);
        StableIndex = unchecked((sbyte)packet[5]);
        ValidZones = packet[6];
        NearZones = packet[7];
        RockProbability = BinaryPrimitives.ReadSingleLittleEndian(packet[8..]);
        PaperProbability = BinaryPrimitives.ReadSingleLittleEndian(packet[12..]);
        ScissorsProbability = BinaryPrimitives.ReadSingleLittleEndian(packet[16..]);
    }

    public static bool TryParse(ReadOnlySpan<byte> packet, out Vl53L8GestureEventArgs? gesture)
    {
        gesture = null;
        if (packet.Length != 20 || packet[0] != 0xA7 || packet[1] != 1) return false;
        var parsed = new Vl53L8GestureEventArgs(packet, DateTimeOffset.UtcNow);
        if (parsed.RawIndex is < 0 or > 2 || parsed.StableIndex is < -1 or > 2 ||
            parsed.ValidZones > 64 || parsed.NearZones > parsed.ValidZones) return false;
        var probabilities = new[] { parsed.RockProbability, parsed.PaperProbability, parsed.ScissorsProbability };
        if (probabilities.Any(p => !float.IsFinite(p) || p < 0 || p > 1) ||
            Math.Abs(probabilities.Sum() - 1f) > 0.001f) return false;
        gesture = parsed;
        return true;
    }
}
