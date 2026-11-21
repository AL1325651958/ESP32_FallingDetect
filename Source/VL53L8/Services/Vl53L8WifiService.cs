using System.Net;
using System.Net.Sockets;

namespace VL53L8.Services;

public sealed class Vl53L8WifiService : IAsyncDisposable
{
    public const int Port = 3333;
    private UdpClient? _udp;
    private CancellationTokenSource? _cancellation;
    public event EventHandler<Vl53L8FrameEventArgs>? FrameReceived;
    public event EventHandler<Vl53L8FallEventArgs>? FallReceived;
    public event EventHandler<string>? StatusChanged;

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        await StopAsync();
        _udp = new UdpClient(Port) { EnableBroadcast = true };
        _cancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        StatusChanged?.Invoke(this, $"Wi-Fi 数据监听中（UDP {Port}）");
        _ = ReceiveLoopAsync(_cancellation.Token);
    }

    private async Task ReceiveLoopAsync(CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested && _udp is not null)
            {
                var result = await _udp.ReceiveAsync(cancellationToken);
                var receivedAt = DateTimeOffset.UtcNow;
                var data = result.Buffer;
                if (Vl53L8FallEventArgs.TryParse(data, out var fall))
                {
                    MainThread.BeginInvokeOnMainThread(() => FallReceived?.Invoke(this, fall!));
                    continue;
                }
                if (data.Length < 132 || data[0] != 0xA6 || data[3] != 0x40) continue;
                var distances = new ushort[64];
                for (var i = 0; i < 64; i++) distances[i] = (ushort)(data[4 + i * 2] | (data[5 + i * 2] << 8));
                var frame = (ushort)(data[1] | (data[2] << 8));
                MainThread.BeginInvokeOnMainThread(() => FrameReceived?.Invoke(this, new Vl53L8FrameEventArgs(frame, distances, receivedAt)));
            }
        }
        catch (OperationCanceledException) { }
        catch (ObjectDisposedException) when (cancellationToken.IsCancellationRequested) { }
        catch (SocketException ex) { StatusChanged?.Invoke(this, $"Wi-Fi 接收失败：{ex.Message}"); }
    }

    public async Task StopAsync()
    {
        _cancellation?.Cancel();
        _udp?.Dispose();
        _udp = null;
        _cancellation?.Dispose();
        _cancellation = null;
        await Task.CompletedTask;
    }

    public async ValueTask DisposeAsync() => await StopAsync();
}
