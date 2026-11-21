namespace VL53L8.Services;

/// <summary>Reassembles the eight 20-byte BLE notifications emitted by the ESP32.</summary>
public sealed class Vl53L8FrameAssembler
{
    private readonly ushort[] _distances = new ushort[64];
    private readonly bool[] _received = new bool[8];
    private ushort _frame;
    private bool _hasFrame;

    public bool TryAddPacket(ReadOnlySpan<byte> packet, out ushort frameNumber, out ushort[] distances)
    {
        frameNumber = 0;
        distances = Array.Empty<ushort>();
        if (packet.Length < 20 || packet[0] != 0xA5)
            return false;

        var frame = (ushort)(packet[1] | (packet[2] << 8));
        var packetIndex = packet[3];
        if (packetIndex > 7)
            return false;

        if (!_hasFrame || frame != _frame)
        {
            _frame = frame;
            _hasFrame = true;
            Array.Clear(_received);
        }

        for (var i = 0; i < 8; i++)
            _distances[packetIndex * 8 + i] = (ushort)(packet[4 + i * 2] | (packet[5 + i * 2] << 8));
        _received[packetIndex] = true;

        if (_received.Any(received => !received))
            return false;

        frameNumber = _frame;
        distances = (ushort[])_distances.Clone();
        Array.Clear(_received);
        return true;
    }
}
