using System.Buffers.Binary;
using VL53L8.Services;

void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
var vectors = File.ReadAllLines(args[0]).Select(Convert.FromHexString).ToArray();
var frames = new ushort[] {0,255,65535};
for (var k = 0; k < 3; ++k)
{
    Check(Vl53L8GestureEventArgs.TryParse(vectors[k], out var result), "C packet rejected");
    Check(result!.FrameNumber == frames[k] && result.RawIndex == k && result.StableIndex == (k == 0 ? -1 : k), "Frame/index mismatch");
    Check(result.ValidZones == 64 && result.NearZones == 12, "Zone counts mismatch");
    var expected = new[] {0.8f,0.85f,0.95f}[k];
    var score = new[] {result.RockProbability,result.PaperProbability,result.ScissorsProbability}[k];
    Check(Math.Abs(score-expected)<1e-7, "Float32 mismatch");
}
var valid = vectors[0];
void Reject(byte[] packet, string name) => Check(!Vl53L8GestureEventArgs.TryParse(packet,out _), name+" accepted");
Reject(valid[..19], "truncated");
Reject(valid.Concat(new byte[1]).ToArray(), "oversized");
foreach (var (offset,value) in new[] {(0,0xA5),(1,2),(4,255),(4,3),(5,254),(5,3),(6,65),(7,65)})
{
    var packet=(byte[])valid.Clone();packet[offset]=(byte)value;Reject(packet,"header/range");
}
foreach (var value in new[] {float.NaN,float.PositiveInfinity,-0.1f,1.1f,0.3f})
{
    var packet=(byte[])valid.Clone();BinaryPrimitives.WriteSingleLittleEndian(packet.AsSpan(8),value);Reject(packet,"probability");
}
Console.WriteLine("PASS: C encoder to C# parser; all classes, unknown, frame boundaries, float32, malformed packets");

if (args.Length > 1)
{
    var packets = File.ReadAllLines(args[1]);
    foreach (var hex in packets)
        Check(Vl53L8GestureEventArgs.TryParse(Convert.FromHexString(hex), out _), "Live BLE packet rejected by APP parser");
    Console.WriteLine($"PASS: APP parser accepted {packets.Length} captured BLE packets");
}
