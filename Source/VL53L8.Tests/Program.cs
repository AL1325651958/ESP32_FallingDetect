using System.IO.Compression;
using System.Text.Json;
using VL53L8.Services;

var testRoot = Path.Combine(Path.GetTempPath(), "VL53L8Tests_" + Guid.NewGuid().ToString("N"));
FileSystem.AppDataDirectory = testRoot;
FileSystem.CacheDirectory = Path.Combine(testRoot, "cache");
try
{
    var samples = Enumerable.Range(0, 64).Select(i => (ushort)(100 + i % 8 * 10 + i / 8 * 20)).ToArray();
    Check(HeatmapInterpolation.Sample(samples, 2.25, 3.5) == 192.5, "bilinear affine field");
    Check(HeatmapInterpolation.Sample(samples, 7, 7) == 310, "bottom-right boundary");
    Check(HeatmapInterpolation.Sample(samples, -1, -1) == 100, "clamped boundary");
    var invalid = Enumerable.Repeat(ushort.MaxValue, 64).ToArray();
    Check(HeatmapInterpolation.Sample(invalid, 3.2, 4.8) is null, "all invalid stays empty");
    samples[0] = ushort.MaxValue;
    Check(HeatmapInterpolation.Sample(samples, 0, 0) is null, "invalid source zone stays empty");
    Check(HeatmapInterpolation.Sample(samples, 0.75, 0) == 110, "invalid neighbor excluded");
    var timestamp = DateTimeOffset.Parse("2026-10-26T20:30:40.123+08:00");
    var frame = new RecordedFrame(65535, timestamp, samples);
    var photo = new CapturedPhoto([0xFF, 0xD8, 0xFF, 0xD9], timestamp.AddMilliseconds(40), timestamp.AddMilliseconds(100), frame);
    string single = await RecordingService.SaveSingleAsync(frame, photo);
    var photoIndex = await File.ReadAllLinesAsync(Path.Combine(single, "photos.csv"));
    var photoRow = photoIndex[1].Split(',');
    Check(photoIndex.Length == 2 && photoRow[3] == "65535" && photoRow[5] == "40.000", "photo timestamp and ToF association");
    Check((await File.ReadAllBytesAsync(Path.Combine(single, photoRow[0]))).SequenceEqual(photo.Jpeg), "photo bytes preserved");
    using (var singleMeta = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(single, "metadata.json"))))
        Check(singleMeta.RootElement.GetProperty("photoCount").GetInt64() == 1, "single photo count");
    var lines = await File.ReadAllLinesAsync(Path.Combine(single, "frames.csv"));
    Check(lines.Length == 2 && lines[0].Split(',').Length == 68, "single CSV header and row count");
    var cells = lines[1].Split(',');
    Check(cells.Length == 68 && cells[3] == "65535" && cells[4] == "-1", "raw frame and invalid sentinel");
    Check(DateTimeOffset.Parse(cells[0]) == timestamp && long.Parse(cells[2]) == timestamp.ToUnixTimeMilliseconds(), "exact receipt timestamp");
    // Deleting before any export must work even when the cache root does not exist.
    string neverExported = await RecordingService.SaveSingleAsync(frame, photo);
    Check(!Directory.Exists(FileSystem.CacheDirectory), "cache directory initially absent");
    var neverExportedItem = new RecordingItem(neverExported, "never exported", "");
    await RecordingService.DeleteAsync(neverExportedItem);
    Check(!Directory.Exists(neverExported), "never-exported record deleted without cache directory");
    await RecordingService.DeleteAsync(neverExportedItem);
    Check(!Directory.Exists(neverExported), "repeated deletion with missing cache succeeds");
    var session = new RecordingService();
    await session.StartAsync();
    for (int i = 0; i < 200; i++) Check(session.TryRecord(frame with { FrameNumber = (ushort)i, ReceivedAt = timestamp.AddMilliseconds(i * 73) }), "queue accepts frame");
    await session.SavePhotoAsync(photo);
    await session.SavePhotoAsync(photo with { RequestedAt = timestamp.AddSeconds(1) });
    string continuous = (await session.StopAsync())!;
    Check(!session.IsRecording && session.FrameCount == 200, "stop drains queue");
    Check((await File.ReadAllLinesAsync(Path.Combine(continuous, "frames.csv"))).Length == 201, "all queued rows persisted");
    using var metadata = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(continuous, "metadata.json")));
    Check(metadata.RootElement.GetProperty("frameCount").GetInt64() == 200, "metadata count");
    var items = await RecordingService.ListAsync();
    Check(items.Count == 2, "record browser list");
    var singleReplay = await RecordingService.OpenReplayAsync(new RecordingItem(single, "single", ""));
    Check(singleReplay.FrameCount == 1, "single replay count");
    var singleFrame = await singleReplay.ReadFrameAsync(0);
    Check(singleFrame.FrameNumber == 65535 && singleFrame.ReceivedAt == timestamp && singleFrame.Distances.SequenceEqual(samples), "single replay raw values and timestamp");
    var continuousReplay = await RecordingService.OpenReplayAsync(new RecordingItem(continuous, "continuous", ""));
    Check(continuousReplay.FrameCount == 200, "continuous replay count");
    foreach (int index in new[] { 199, 0, 100, 1, 198 })
    {
        var actual = await continuousReplay.ReadFrameAsync(index);
        Check(actual.FrameNumber == index && actual.ReceivedAt == timestamp.AddMilliseconds(index * 73) && actual.Distances.SequenceEqual(samples), "random seek preserves original frames");
    }
    Check(ReplayTiming.Offset(timestamp, timestamp.AddMilliseconds(73), TimeSpan.Zero) == TimeSpan.FromMilliseconds(73), "original timestamp interval");
    Check(ReplayTiming.Offset(timestamp, timestamp.AddSeconds(30), TimeSpan.FromMilliseconds(73)) == TimeSpan.FromSeconds(30), "long original gap retained");
    Check(ReplayTiming.Offset(timestamp, timestamp.AddSeconds(-1), TimeSpan.FromSeconds(30)) == TimeSpan.FromSeconds(30), "clock regression never reverses playback");
    try { await continuousReplay.ReadFrameAsync(200); throw new Exception("FAIL: out of range replay accepted"); }
    catch (ArgumentOutOfRangeException) { }
    using (var cancelled = new CancellationTokenSource())
    {
        cancelled.Cancel();
        try { await continuousReplay.ReadFrameAsync(0, cancelled.Token); throw new Exception("FAIL: cancellation ignored"); }
        catch (OperationCanceledException) { }
    }
    string originalCsv = await File.ReadAllTextAsync(Path.Combine(single, "frames.csv"));
    foreach (string newline in new[] { "\n", "\r\n" })
    {
        var content = originalCsv.Replace("\r\n", "\n").Replace("\n", newline);
        await File.WriteAllTextAsync(Path.Combine(single, "frames.csv"), content + "incomplete tail", new System.Text.UTF8Encoding(true));
        var recovered = await RecordingService.OpenReplayAsync(new RecordingItem(single, "single", ""));
        Check(recovered.FrameCount == 1 && (await recovered.ReadFrameAsync(0)).Distances.SequenceEqual(samples), "BOM newline and incomplete tail recovery");
    }
    await File.WriteAllTextAsync(Path.Combine(single, "frames.csv"), originalCsv, new System.Text.UTF8Encoding(true));
    string emptyDirectory = Path.Combine(RecordingService.RootDirectory, "empty_replay_00000000");
    Directory.CreateDirectory(emptyDirectory);
    await File.WriteAllTextAsync(Path.Combine(emptyDirectory, "frames.csv"), originalCsv.Split('\n')[0] + "\n");
    var emptyReplay = await RecordingService.OpenReplayAsync(new RecordingItem(emptyDirectory, "empty", ""));
    Check(emptyReplay.FrameCount == 0, "empty replay has no frames");
    Directory.Delete(emptyDirectory, true);
    try { await RecordingService.OpenReplayAsync(new RecordingItem(testRoot, "outside", "")); throw new Exception("FAIL: outside replay accepted"); }
    catch (InvalidOperationException) { }
    string allZip = await RecordingService.ExportAllAsync();
    using (var allArchive = ZipFile.OpenRead(allZip))
    {
        foreach (var item in items)
        {
            string prefix = Path.GetFileName(item.DirectoryPath) + "/";
            Check(allArchive.GetEntry(prefix + "frames.csv") is not null, "all-export retains each recording CSV");
            Check(allArchive.GetEntry(prefix + "metadata.json") is not null, "all-export retains metadata");
            Check(allArchive.GetEntry(prefix + "photos.csv") is not null, "all-export retains photo index");
            Check(allArchive.Entries.Any(e => e.FullName.StartsWith(prefix + "photos/") && e.FullName.EndsWith(".jpg")), "all-export retains nested photos");
            using var reader = new StreamReader(allArchive.GetEntry(prefix + "frames.csv")!.Open());
            Check(await reader.ReadToEndAsync() == await File.ReadAllTextAsync(Path.Combine(item.DirectoryPath, "frames.csv")), "all-export CSV content preserved");
        }
        Check(allArchive.Entries.Select(e => e.FullName.Split('/')[0]).Distinct().Count() == items.Count, "all-export keeps recording folders distinct");
    }
    string zip = await RecordingService.ExportAsync(items[0]);
    zip = await RecordingService.ExportAsync(items[0]);
    using var archive = ZipFile.OpenRead(zip);
    Check(archive.GetEntry("frames.csv") is not null && archive.GetEntry("metadata.json") is not null && archive.GetEntry("photos.csv") is not null, "repeat ZIP export contents");
    Check(archive.Entries.Count(e => e.FullName.StartsWith("photos/") && e.FullName.EndsWith(".jpg")) >= 1, "nested photos included in ZIP");
    Check(session.PhotoCount == 2, "continuous photo count");
    try { await session.SavePhotoAsync(photo); throw new Exception("FAIL: save after stop"); }
    catch (InvalidOperationException) { }
    archive.Dispose();
    string survivor = items[1].DirectoryPath;
    Directory.Delete(Path.Combine(FileSystem.CacheDirectory, "exports"), true);
    // Simulate Android clearing exported ZIPs and the entire exports directory.

    await RecordingService.DeleteAsync(items[0]);
    Check(!Directory.Exists(items[0].DirectoryPath) && !File.Exists(zip), "recording and export deleted");
    Check(Directory.Exists(survivor) && (await RecordingService.ListAsync()).Count == 1, "other recording preserved");
    await RecordingService.DeleteAsync(items[0]);
    try
    {
        await RecordingService.DeleteAsync(new RecordingItem(testRoot, "outside", ""));
        throw new Exception("FAIL: outside directory deletion accepted");
    }
    catch (InvalidOperationException) { }
    Check(Directory.Exists(survivor), "outside deletion rejected");
    string unrelated = Path.Combine(testRoot, "preferences.json");
    await File.WriteAllTextAsync(unrelated, "keep");
    string extraRecord = await RecordingService.SaveSingleAsync(frame, photo);
    Check((await RecordingService.ListAsync()).Count == 2, "delete all starts with multiple recordings");
    string remainingZip = await RecordingService.ExportAllAsync();
    string remainingSingleZip = await RecordingService.ExportAsync(items[1]);
    await RecordingService.DeleteAllAsync();
    Check((await RecordingService.ListAsync()).Count == 0 && !Directory.Exists(survivor) && !Directory.Exists(extraRecord), "delete all clears all recordings");
    Check(!File.Exists(remainingZip) && !File.Exists(remainingSingleZip), "delete all clears batch and single exports");
    Check(await File.ReadAllTextAsync(unrelated) == "keep", "delete all preserves unrelated app files");
    await RecordingService.DeleteAllAsync();
    Directory.Delete(FileSystem.CacheDirectory, true);
    await RecordingService.DeleteAllAsync();
    try { await RecordingService.ExportAllAsync(); throw new Exception("FAIL: empty export accepted"); }
    catch (InvalidOperationException) { }
    Console.WriteLine("PASS: replay random seek, raw timestamps, original playback timing, cancellation, empty replay, partial tail recovery, delete all with missing cache, all-record ZIP content, separate folders, empty export handling, interpolation, invalid zones, timestamp fidelity, single save, 200-frame recording, metadata, photo association, JPEG persistence, nested photo ZIP export, scoped recording deletion, absent cache directory, OS-cleared exports and repeated deletion.");
}
finally { Directory.Delete(testRoot, true); }

static void Check(bool condition, string name)
{
    if (!condition) throw new Exception("FAIL: " + name);
}

namespace VL53L8.Services
{
    public static class FileSystem
    {
        public static string AppDataDirectory { get; set; } = "";
        public static string CacheDirectory { get; set; } = "";
    }
    public sealed class Vl53L8FrameEventArgs : EventArgs
    {
        public ushort FrameNumber { get; init; }
        public DateTimeOffset ReceivedAt { get; init; }
        public ushort[] Distances { get; init; } = [];
    }
}
