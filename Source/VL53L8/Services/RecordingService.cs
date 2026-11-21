using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace VL53L8.Services;

public sealed record RecordedFrame(ushort FrameNumber, DateTimeOffset ReceivedAt, ushort[] Distances)
{
    public static RecordedFrame From(Vl53L8FrameEventArgs frame) =>
        new(frame.FrameNumber, frame.ReceivedAt, (ushort[])frame.Distances.Clone());
}

public sealed record CapturedPhoto(byte[] Jpeg, DateTimeOffset RequestedAt, DateTimeOffset CompletedAt, RecordedFrame? AssociatedFrame);

public sealed record RecordingItem(string DirectoryPath, string Name, string Summary);

/// <summary>Bounded queue; disk IO never blocks the UDP or UI callbacks.</summary>
public sealed class RecordingService
{
    public static string RootDirectory => Path.Combine(FileSystem.AppDataDirectory, "Recordings");
    private Channel<RecordedFrame>? _queue;
    private Task? _writerTask;
    private string? _directory;
    private DateTimeOffset _started;
    private string _kind = "continuous";
    private long _count;
    private long _photoCount;
    public long PhotoCount => Interlocked.Read(ref _photoCount);
    private Exception? _error;
    public bool IsRecording => _queue is not null;
    public long FrameCount => Interlocked.Read(ref _count);
    public Exception? Error => _error;

    public async Task StartAsync(string kind = "continuous")
    {
        if (IsRecording) throw new InvalidOperationException("已有记录正在进行。");
        _started = DateTimeOffset.UtcNow; _kind = kind; _count = 0; _photoCount = 0; _error = null;
        _directory = Path.Combine(RootDirectory, $"{_started:yyyyMMdd_HHmmss_fff}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_directory);
        await File.WriteAllTextAsync(Path.Combine(_directory, "metadata.json"), JsonSerializer.Serialize(new
        {
            schemaVersion = 1, kind, startedUtc = _started, state = "recording", rows = 8, columns = 8,
            units = "mm", invalidValue = -1, timestampSource = "phone UDP receipt; not sensor capture time"
        }, new JsonSerializerOptions { WriteIndented = true }));
        var writer = new StreamWriter(Path.Combine(_directory, "frames.csv"), false, new UTF8Encoding(true));
        try
        {
            await writer.WriteLineAsync("received_utc,received_local,unix_ms,frame_number," +
                string.Join(",", Enumerable.Range(0, 64).Select(i => $"zone_{i:D2}_mm")));
            await writer.FlushAsync();
        }
        catch { await writer.DisposeAsync(); throw; }
        var queue = Channel.CreateBounded<RecordedFrame>(new BoundedChannelOptions(256)
            { SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait });
        _queue = queue;
        _writerTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var frame in queue.Reader.ReadAllAsync())
                {
                    string row = frame.ReceivedAt.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture) + "," +
                        frame.ReceivedAt.ToLocalTime().ToString("O", CultureInfo.InvariantCulture) + "," +
                        frame.ReceivedAt.ToUnixTimeMilliseconds().ToString(CultureInfo.InvariantCulture) + "," +
                        frame.FrameNumber.ToString(CultureInfo.InvariantCulture) + "," +
                        string.Join(",", frame.Distances.Select(v => v == ushort.MaxValue ? "-1" : v.ToString(CultureInfo.InvariantCulture)));
                    await writer.WriteLineAsync(row);
                    Interlocked.Increment(ref _count);
                    // Flush every frame so a crash does not lose the whole session.
                    await writer.FlushAsync();
                }
            }
            catch (Exception ex) { _error = ex; queue.Writer.TryComplete(ex); }
            finally { await writer.DisposeAsync(); }
        });
    }

    public bool TryRecord(RecordedFrame frame)
    {
        if (_queue is null) return false;
        if (_error is not null) return false;
        if (_queue.Writer.TryWrite(frame)) return true;
        _error = new IOException("保存队列已满，记录已停止，请检查存储空间或降低采样频率。");
        return false;
    }

    public async Task<string?> StopAsync()
    {
        var queue = _queue;
        if (queue is null) return null;
        _queue = null; queue.Writer.TryComplete();
        if (_writerTask is not null) await _writerTask;
        string directory = _directory!;
        await File.WriteAllTextAsync(Path.Combine(directory, "metadata.json"), JsonSerializer.Serialize(new
        {
            schemaVersion = 1, kind = _kind, startedUtc = _started, endedUtc = DateTimeOffset.UtcNow,
            state = _error is null ? "completed" : "failed", frameCount = FrameCount, photoCount = PhotoCount, error = _error?.Message,
            cameraTimestampSource = "phone capture request and image completion; not hardware synchronized",
            rows = 8, columns = 8, order = "row-major; zone_00 to zone_63", units = "mm", invalidValue = -1,
            timestampSource = "phone UDP receipt; not sensor capture time",
            displayInterpolation = "bilinear 96x96; exported distances are original 8x8 samples"
        }, new JsonSerializerOptions { WriteIndented = true }));
        if (_error is not null) throw new IOException("记录保存异常：" + _error.Message, _error);
        return directory;
    }

    // Caller serializes camera captures and awaits them before stopping the recording.
    public async Task SavePhotoAsync(CapturedPhoto photo)
    {
        if (!IsRecording || _directory is null) throw new InvalidOperationException("记录已结束。");
        if (photo.Jpeg.Length < 2 || photo.Jpeg[0] != 0xFF || photo.Jpeg[1] != 0xD8)
            throw new InvalidDataException("相机返回的图像不是 JPEG。");
        string folder = Path.Combine(_directory, "photos");
        Directory.CreateDirectory(folder);
        string filename = $"photos/{photo.RequestedAt:yyyyMMdd_HHmmss_fff}_{Guid.NewGuid():N}.jpg";
        await File.WriteAllBytesAsync(Path.Combine(_directory, filename), photo.Jpeg);
        string index = Path.Combine(_directory, "photos.csv");
        if (!File.Exists(index))
            await File.WriteAllTextAsync(index, "file,capture_requested_utc,capture_completed_utc,tof_frame_number,tof_received_utc,request_minus_tof_ms\n", new UTF8Encoding(true));
        var frame = photo.AssociatedFrame;
        string row = filename + "," + photo.RequestedAt.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture) + "," +
            photo.CompletedAt.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture) + "," +
            frame?.FrameNumber.ToString(CultureInfo.InvariantCulture) + "," +
            frame?.ReceivedAt.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture) + "," +
            (frame is null ? "" : (photo.RequestedAt - frame.ReceivedAt).TotalMilliseconds.ToString("F3", CultureInfo.InvariantCulture)) + "\n";
        await File.AppendAllTextAsync(index, row);
        Interlocked.Increment(ref _photoCount);
    }

    public static async Task<string> SaveSingleAsync(RecordedFrame frame, CapturedPhoto? photo = null)
    {
        var recording = new RecordingService();
        await recording.StartAsync("single");
        try
        {
            recording.TryRecord(frame);
            if (photo is not null) await recording.SavePhotoAsync(photo);
        }
        catch (Exception ex)
        {
            recording._error = ex;
            await recording.StopAsync();
            throw;
        }
        return (await recording.StopAsync())!;
    }

    public static Task<List<RecordingItem>> ListAsync() => Task.Run(() =>
    {
        Directory.CreateDirectory(RootDirectory);
        return Directory.GetDirectories(RootDirectory).OrderByDescending(Path.GetFileName).Select(directory =>
        {
            string summary = "未正常结束的记录（可导出已写入数据）";
            try
            {
                using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "metadata.json")));
                var item = json.RootElement;
                if (item.TryGetProperty("frameCount", out var count))
                    summary = $"{(item.GetProperty("kind").GetString() == "single" ? "单次保存" : "持续记录")} · {count.GetInt64()} 帧 · {item.GetProperty("state").GetString()}";
                if (item.TryGetProperty("photoCount", out var photos)) summary += $" · {photos.GetInt64()} 张照片";
                if (item.TryGetProperty("startedUtc", out var time))
                    summary = time.GetDateTimeOffset().ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss") + " · " + summary;
            }
            catch (Exception ex) when (ex is IOException or JsonException or InvalidOperationException) { }
            return new RecordingItem(directory, Path.GetFileName(directory)[..19], summary);
        }).ToList();
    });

    public static Task DeleteAsync(RecordingItem item) => Task.Run(() =>
    {
        string directory = Path.TrimEndingDirectorySeparator(Path.GetFullPath(item.DirectoryPath));
        string recordingRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(RootDirectory));
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!string.Equals(Path.GetDirectoryName(directory), recordingRoot, comparison))
            throw new InvalidOperationException("只能删除数据记录目录中的单条记录。");
        // A record may never have been exported, or the OS may have cleared its cache.
        // Missing files/directories mean there is nothing left to delete, not a failure.
        string export = Path.Combine(FileSystem.CacheDirectory, "exports", Path.GetFileName(directory) + ".zip");
        try { File.Delete(export); }
        catch (FileNotFoundException) { }
        catch (DirectoryNotFoundException) { }
        try
        {
            if ((File.GetAttributes(directory) & FileAttributes.ReparsePoint) != 0)
                throw new InvalidOperationException("不能删除链接目录。");
            Directory.Delete(directory, recursive: true);
        }
        catch (FileNotFoundException) when (!Directory.Exists(directory)) { }
        catch (DirectoryNotFoundException) when (!Directory.Exists(directory)) { }
    });

    public static async Task DeleteAllAsync()
    {
        var records = await ListAsync();
        foreach (var record in records)
            await DeleteAsync(record);

        string exports = Path.Combine(FileSystem.CacheDirectory, "exports");
        if (Directory.Exists(exports))
        {
            foreach (var file in Directory.EnumerateFiles(exports))
            {
                try { File.Delete(file); }
                catch (FileNotFoundException) { }
                catch (DirectoryNotFoundException) { }
            }
        }
    }

    public static async Task<RecordingReplay> OpenReplayAsync(RecordingItem item)
    {
        string directory = Path.TrimEndingDirectorySeparator(Path.GetFullPath(item.DirectoryPath));
        string recordingRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(RootDirectory));
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!string.Equals(Path.GetDirectoryName(directory), recordingRoot, comparison))
            throw new InvalidOperationException("只能回放数据记录目录中的记录。");
        if ((File.GetAttributes(directory) & FileAttributes.ReparsePoint) != 0)
            throw new InvalidOperationException("不能回放链接目录。");

        string framesPath = Path.Combine(directory, "frames.csv");
        var offsets = new List<long>();
        await using var stream = new FileStream(framesPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 65536, useAsync: true);
        byte[] buffer = new byte[65536];
        long absolute = 0;
        long lineStart = 0;
        bool header = true;
        int read;
        while ((read = await stream.ReadAsync(buffer.AsMemory(), CancellationToken.None)) > 0)
        {
            for (int i = 0; i < read; i++)
            {
                if (buffer[i] != (byte)'\n') continue;
                if (!header && absolute + i > lineStart)
                    offsets.Add(lineStart);
                header = false;
                lineStart = absolute + i + 1;
            }
            absolute += read;
        }
        return new RecordingReplay(framesPath, offsets.ToArray());
    }

    public static async Task<string> ExportAllAsync()
    {
        var records = await ListAsync();
        if (records.Count == 0) throw new InvalidOperationException("暂无记录可导出，请先保存数据。");
        string exports = Path.Combine(FileSystem.CacheDirectory, "exports");
        Directory.CreateDirectory(exports);
        string zip = Path.Combine(exports, $"VL53L8_All_{DateTimeOffset.UtcNow:yyyyMMdd_HHmmss_fff}_{Guid.NewGuid():N}.zip");
        string temporary = zip + ".tmp";
        try
        {
            await Task.Run(() =>
            {
                using var archive = ZipFile.Open(temporary, ZipArchiveMode.Create);
                foreach (var record in records)
                {
                    // Use the complete directory ID to keep recordings with similar times distinct.
                    string prefix = Path.GetFileName(record.DirectoryPath);
                    foreach (var file in Directory.EnumerateFiles(record.DirectoryPath, "*", SearchOption.AllDirectories))
                    {
                        string relative = Path.GetRelativePath(record.DirectoryPath, file).Replace('\\', '/');
                        archive.CreateEntryFromFile(file, prefix + "/" + relative, CompressionLevel.Fastest);
                    }
                }
            });
            File.Move(temporary, zip);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
        return zip;
    }

    public static async Task<string> ExportAsync(RecordingItem item)
    {
        string exports = Path.Combine(FileSystem.CacheDirectory, "exports");
        Directory.CreateDirectory(exports);
        string zip = Path.Combine(exports, Path.GetFileName(item.DirectoryPath) + ".zip");
        string temporary = zip + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await Task.Run(() =>
            {
                using var archive = ZipFile.Open(temporary, ZipArchiveMode.Create);
                foreach (var file in Directory.EnumerateFiles(item.DirectoryPath, "*", SearchOption.AllDirectories))
                    archive.CreateEntryFromFile(file, Path.GetRelativePath(item.DirectoryPath, file).Replace('\\', '/'), CompressionLevel.Fastest);
            });
            File.Move(temporary, zip, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
        return zip;
    }
}

public sealed class RecordingReplay(string framesPath, long[] frameOffsets)
{
    public int FrameCount => frameOffsets.Length;

    public async Task<RecordedFrame> ReadFrameAsync(int index, CancellationToken cancellationToken = default)
    {
        if ((uint)index >= (uint)frameOffsets.Length) throw new ArgumentOutOfRangeException(nameof(index));
        await using var stream = new FileStream(framesPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1024, useAsync: true);
        stream.Position = frameOffsets[index];
        using var line = new MemoryStream(256);
        byte[] buffer = new byte[512];
        bool complete = false;
        while (!complete)
        {
            int count = await stream.ReadAsync(buffer.AsMemory(), cancellationToken);
            if (count == 0) break;
            int newline = Array.IndexOf(buffer, (byte)'\n', 0, count);
            int length = newline >= 0 ? newline : count;
            await line.WriteAsync(buffer.AsMemory(0, length), cancellationToken);
            complete = newline >= 0;
        }
        if (!complete) throw new InvalidDataException("记录末尾的数据行不完整。");

        string text = Encoding.UTF8.GetString(line.GetBuffer(), 0, checked((int)line.Length)).TrimEnd('\r');
        string[] fields = text.Split(',');
        if (fields.Length != 68) throw new InvalidDataException("记录帧格式错误。");
        var receivedAt = DateTimeOffset.Parse(fields[0], CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind);
        var frameNumber = ushort.Parse(fields[3], NumberStyles.None, CultureInfo.InvariantCulture);
        var distances = new ushort[64];
        for (int i = 0; i < distances.Length; i++)
        {
            int value = int.Parse(fields[4 + i], NumberStyles.Integer, CultureInfo.InvariantCulture);
            if (value == -1) distances[i] = ushort.MaxValue;
            else if (value is >= 0 and <= ushort.MaxValue) distances[i] = (ushort)value;
            else throw new InvalidDataException("记录帧中存在无效距离值。");
        }
        return new RecordedFrame(frameNumber, receivedAt, distances);
    }
}
