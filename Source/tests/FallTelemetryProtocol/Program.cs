using System.Buffers.Binary;
using VL53L8.Services;
void Check(bool ok,string message) { if (!ok) throw new Exception(message); }
var vectors=File.ReadAllLines(args[0]).Where(l=>!string.IsNullOrWhiteSpace(l)).Select(Convert.FromHexString).ToArray();
foreach(var p in vectors) Check(Vl53L8FallEventArgs.TryParse(p,out _),"C/device packet rejected");
byte[] Packet(int raw,int stable,byte buffered,float motion,float fall,byte valid=64,ushort frame=65535,byte version=2) {
    var b=new byte[20]; b[0]=0xA8;b[1]=version;BinaryPrimitives.WriteUInt16LittleEndian(b.AsSpan(2),frame);
    b[4]=unchecked((byte)(sbyte)raw); b[5]=unchecked((byte)(sbyte)stable);b[6]=valid;b[7]=buffered;
    BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(8),motion);
    BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(12),fall);
    BinaryPrimitives.WriteUInt32LittleEndian(b.AsSpan(16),12345);return b;
}
foreach(byte version in new byte[]{1,2})
foreach(ushort frame in new ushort[]{0,255,65535})
foreach(int raw in new[]{0,1,2})
foreach(int stable in new[]{-1,0,1,2}) {
    float motion=raw==0?.9f:.05f, fall=raw==1?.9f:.05f;
    if(raw==2) { motion=.05f;fall=.05f; }
    Check(Vl53L8FallEventArgs.TryParse(Packet(raw,stable,Vl53L8FallEventArgs.WindowFrames,motion,fall,64,frame,version),out var p),"ready packet rejected");
    Check(p!.FrameNumber==frame&&p.RawIndex==raw&&p.StableIndex==stable&&p.InferenceMicroseconds==12345,"field mismatch");
    Check(Math.Abs(p.MotionProbability-motion)<1e-7 && Math.Abs(p.FallProbability-fall)<1e-7,"probability field order mismatch");
}
for(byte n=0;n<Vl53L8FallEventArgs.WindowFrames;++n) Check(Vl53L8FallEventArgs.TryParse(Packet(-1,-1,n,0,0),out var p)&&!p!.IsReady,"warmup rejected");
foreach(byte version in new byte[]{1,2}) {
    Check(Vl53L8FallEventArgs.TryParse(Packet(3,3,0,0,0,0,version:version),out var empty)&&!empty!.IsReady&&empty.PersonStillProbability==0,"missing observation rejected");
    Check(empty!.StateText=="暂无法判断"&&!empty.HasAlert,"missing measurement displayed as vacancy or fall");
}
foreach(int alert in new[]{1,4}) {
    foreach(byte validZones in new byte[]{0,1,3})
        Check(Vl53L8FallEventArgs.TryParse(Packet(3,alert,0,0,0,validZones),out var lost)&&lost!.HasAlert&&lost.StateText.Contains("目标丢失"),"alert lost on disappearance");
    for(byte n=0;n<24;++n)
        Check(Vl53L8FallEventArgs.TryParse(Packet(-1,alert,n,0,0),out var recovery)&&recovery!.HasAlert,"alert lost during reacquisition");
}
Check(Vl53L8FallEventArgs.TryParse(Packet(1,-1,24,.05f,.9f),out var unconfirmed)&&unconfirmed!.StateText=="结果待稳定"&&!unconfirmed.HasAlert,"unguarded model score displayed as confirmed fall");
Check(Vl53L8FallEventArgs.TryParse(Packet(0,4,24,.9f,.05f),out var suspected)&&suspected!.StateText=="疑似跌倒","suspected state rejected");
void Reject(byte[] p,string name)=>Check(!Vl53L8FallEventArgs.TryParse(p,out _),name+" accepted");
var valid=Packet(1,1,Vl53L8FallEventArgs.WindowFrames,.05f,.9f);
Reject(valid[..19],"truncated");Reject(valid.Concat(new byte[1]).ToArray(),"oversized");
foreach(var (offset,value) in new[]{(0,0xA7),(1,3),(4,2),(4,4),(4,254),(5,3),(5,5),(5,254),(6,65),(6,3),(7,33),(7,31)}) {
    var p=(byte[])valid.Clone();p[offset]=(byte)value;Reject(p,"range/header");
}
foreach(float value in new[]{float.NaN,float.PositiveInfinity,-.1f,1.1f,.2f}) {
    var p=(byte[])valid.Clone();BinaryPrimitives.WriteSingleLittleEndian(p.AsSpan(8),value);Reject(p,"probability");
}
Reject(Packet(-1,1,5,0,0,version:1),"legacy alert during warmup");Reject(Packet(-1,-1,Vl53L8FallEventArgs.WindowFrames,0,0),"unknown full buffer");
Reject(Packet(-1,3,5,0,0),"loss state during valid warmup");Reject(Packet(-1,-1,5,.2f,0),"nonzero warmup probability");
Reject(Packet(3,3,1,0,0,0),"missing observation with buffered frames");Reject(Packet(3,0,0,0,0,0),"missing observation with normal decision");
Reject(Packet(3,3,0,0,0,4),"missing observation with usable frame");Reject(Packet(3,1,0,.1f,0,0),"missing observation with invented scores");
Reject(Packet(3,4,0,0,0,0,version:1),"new loss state under old version");
Reject(Packet(-1,-1,0,0,0,0),"v2 warmup without usable ranges");
var display = new FallStatusDisplay();
Vl53L8FallEventArgs Parse(byte[] packet) { Check(Vl53L8FallEventArgs.TryParse(packet,out var value),"display vector rejected");return value!; }
display.Update(Parse(Packet(1,1,24,.05f,.9f)));
Check(display.Update(Parse(Packet(3,3,0,0,0,0,version:1))).Contains("检测到跌倒"),"legacy missing data erased known fall");
Check(display.Update(Parse(Packet(-1,-1,2,0,0))).Contains("检测到跌倒"),"restarted warmup erased known fall");
Check(display.TimeoutText.Contains("检测到跌倒")&&display.HasAlert,"transport timeout erased known fall");
Check(display.Update(Parse(Packet(3,4,0,0,0,0))).Contains("检测到跌倒"),"suspected packet downgraded confirmed fall");
display.Update(Parse(Packet(0,-1,24,.9f,.05f)));
Check(display.HasAlert,"unstable recovery dismissed alert");
Check(display.Update(Parse(Packet(0,0,24,.9f,.05f)))=="有人运动"&&!display.HasAlert,"stable recovery failed to clear alert");
Console.WriteLine($"PASS: APP parser accepted {vectors.Length} C/device packets; v1 compatibility, v2 loss/suspected states, retained alerts, display labels, warmup, frame wrap and malformed inputs verified.");
