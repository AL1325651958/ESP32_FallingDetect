using System.Buffers.Binary;

namespace VL53L8.Services;

/// <summary>A8/v2 shared by BLE and UDP; legacy v1 remains readable.</summary>
public sealed class Vl53L8FallEventArgs : EventArgs
{
    public const byte WindowFrames = 24;
    public byte ProtocolVersion { get; }
    public DateTimeOffset ReceivedAt { get; }
    public ushort FrameNumber { get; }
    public int RawIndex { get; }
    public int StableIndex { get; }
    public byte ValidZones { get; }
    public byte BufferedFrames { get; }
    public float FallProbability { get; }
    public float MotionProbability { get; }
    public float PersonStillProbability { get; }
    public uint InferenceMicroseconds { get; }
    public bool IsReady => RawIndex is >= 0 and <= 2;
    public bool ObservationMissing => RawIndex == 3 || ValidZones < 4;
    public bool HasAlert => StableIndex is 1 or 4;
    public string StateText => StableIndex switch
    {
        1 => ObservationMissing ? "检测到跌倒 · 目标丢失" : "检测到跌倒",
        4 => ObservationMissing ? "疑似跌倒 · 目标丢失" : "疑似跌倒",
        _ when ObservationMissing => "暂无法判断",
        _ when !IsReady => "正在识别",
        0 => "有人运动",
        2 => "有人静止",
        _ => "结果待稳定"
    };

    private Vl53L8FallEventArgs(ReadOnlySpan<byte> p)
    {
        ReceivedAt = DateTimeOffset.UtcNow;
        ProtocolVersion = p[1];
        FrameNumber = BinaryPrimitives.ReadUInt16LittleEndian(p[2..]);
        RawIndex = unchecked((sbyte)p[4]); StableIndex = unchecked((sbyte)p[5]);
        ValidZones = p[6]; BufferedFrames = p[7];
        MotionProbability = BinaryPrimitives.ReadSingleLittleEndian(p[8..]);
        FallProbability = BinaryPrimitives.ReadSingleLittleEndian(p[12..]);
        PersonStillProbability = RawIndex is >= 0 and <= 2
            ? Math.Max(0, 1 - FallProbability - MotionProbability) : 0;
        InferenceMicroseconds = BinaryPrimitives.ReadUInt32LittleEndian(p[16..]);
    }

    public static bool TryParse(ReadOnlySpan<byte> packet, out Vl53L8FallEventArgs? fall)
    {
        fall = null;
        if (packet.Length != 20 || packet[0] != 0xA8 || packet[1] is not (1 or 2)) return false;
        var p = new Vl53L8FallEventArgs(packet);
        if (p.RawIndex is < -1 or > 3 || p.StableIndex is < -1 or > 4 ||
            p.ValidZones > 64 || p.BufferedFrames > WindowFrames ||
            !float.IsFinite(p.FallProbability) || !float.IsFinite(p.MotionProbability) ||
            p.FallProbability is < 0 or > 1 || p.MotionProbability is < 0 or > 1) return false;
        if (p.RawIndex == 3)
        {
            if ((p.ProtocolVersion == 1 ? p.StableIndex != 3 || p.ValidZones != 0
                    : p.StableIndex is not (1 or 3 or 4) || p.ValidZones >= 4) || p.BufferedFrames != 0 ||
                p.FallProbability != 0 || p.MotionProbability != 0) return false;
        }
        else if (p.IsReady)
        {
            if (p.BufferedFrames != WindowFrames || p.ValidZones < 4 || p.StableIndex == 3 ||
                (p.ProtocolVersion == 1 && p.StableIndex == 4) ||
                p.FallProbability + p.MotionProbability > 1.00001f ||
                (p.RawIndex == 0 && p.MotionProbability + 1e-6f < Math.Max(p.FallProbability, p.PersonStillProbability)) ||
                (p.RawIndex == 1 && p.FallProbability + 1e-6f < Math.Max(p.MotionProbability, p.PersonStillProbability)) ||
                (p.RawIndex == 2 && p.PersonStillProbability + 1e-6f < Math.Max(p.MotionProbability, p.FallProbability))) return false;
        }
        else if ((p.ProtocolVersion == 1 ? p.StableIndex != -1 : p.StableIndex is not (-1 or 1 or 4)) ||
                 (p.ProtocolVersion == 2 && p.ValidZones < 4) ||
                 p.BufferedFrames == WindowFrames || p.FallProbability != 0 || p.MotionProbability != 0)
            return false;
        fall = p;
        return true;
    }
}

/// <summary>Unknown input or a transport interruption cannot dismiss an alert.</summary>
public sealed class FallStatusDisplay
{
    private int _alert = -1;
    public bool HasAlert => _alert >= 0;
    public string Update(Vl53L8FallEventArgs value)
    {
        if (value.HasAlert) _alert = _alert == 1 ? 1 : value.StableIndex;
        else if (value.IsReady && value.StableIndex is 0 or 2) _alert = -1;
        return _alert switch
        {
            1 => value.ObservationMissing ? "检测到跌倒 · 目标丢失" : "检测到跌倒",
            4 => value.ObservationMissing ? "疑似跌倒 · 目标丢失" : "疑似跌倒",
            _ => value.StateText
        };
    }
    public string TimeoutText => _alert switch
    {
        1 => "检测到跌倒 · 连接中断", 4 => "疑似跌倒 · 连接中断", _ => "连接中断"
    };
}
