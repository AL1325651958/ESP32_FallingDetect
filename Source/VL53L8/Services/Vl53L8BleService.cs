using Plugin.BLE;
using Plugin.BLE.Abstractions.Contracts;
using Plugin.BLE.Abstractions.EventArgs;

namespace VL53L8.Services;

public sealed class Vl53L8FrameEventArgs(ushort frameNumber, ushort[] distances, DateTimeOffset? receivedAt = null) : EventArgs
{
    public DateTimeOffset ReceivedAt { get; } = receivedAt ?? DateTimeOffset.UtcNow;
    public ushort FrameNumber { get; } = frameNumber;
    public ushort[] Distances { get; } = distances;
}

public sealed class Vl53L8BleService : IAsyncDisposable
{
    public static readonly Guid ServiceUuid = Guid.Parse("12345678-1234-5678-1234-56789ABCDEF0");
    // Existing ESP32 firmware exposes the Wi-Fi provisioning characteristic as ...DEF3.
    public static readonly Guid CharacteristicUuid = Guid.Parse("12345678-1234-5678-1234-56789ABCDEF3");

    private readonly IAdapter _adapter = CrossBluetoothLE.Current.Adapter;
    private IDevice? _device;
    public static readonly Guid TelemetryUuid = Guid.Parse("12345678-1234-5678-1234-56789ABCDEF1");
    private ICharacteristic? _provisioningCharacteristic;
    private ICharacteristic? _telemetryCharacteristic;
    public event EventHandler<string>? StatusChanged;
    public event EventHandler<Vl53L8GestureEventArgs>? GestureReceived;
    public event EventHandler<Vl53L8FallEventArgs>? FallReceived;

    public Vl53L8BleService()
    {
        _adapter.DeviceDisconnected += OnDeviceDisconnected;
        _adapter.DeviceConnectionLost += OnDeviceConnectionLost;
    }
    public bool IsConnected => _device is not null;
    public string? DeviceName => _device?.Name;

    public async Task<IReadOnlyList<IDevice>> ScanAsync(CancellationToken cancellationToken = default)
    {
        var devices = new Dictionary<Guid, IDevice>();
        void OnDiscovered(object? _, DeviceEventArgs args)
        {
            if (args.Device.Name?.Equals("FallingDetect", StringComparison.OrdinalIgnoreCase) == true)
                devices[args.Device.Id] = args.Device;
        }

        _adapter.DeviceDiscovered += OnDiscovered;
        try
        {
            _adapter.ScanTimeout = 8000;
            StatusChanged?.Invoke(this, "正在扫描 FallingDetect…");
            // Firmware advertises its device name and adds the GATT service after
            // connection, so scan by name instead of filtering advertisements by UUID.
            await _adapter.StartScanningForDevicesAsync(new Plugin.BLE.Abstractions.ScanFilterOptions(), cancellationToken: cancellationToken);
        }
        finally
        {
            _adapter.DeviceDiscovered -= OnDiscovered;
        }
        return devices.Values.ToArray();
    }

    public async Task ConnectAsync(IDevice device, CancellationToken cancellationToken = default)
    {
        await DisconnectAsync();
        StatusChanged?.Invoke(this, "正在连接…");
        await _adapter.ConnectToDeviceAsync(device, cancellationToken: cancellationToken);
        _device = device;
        try
        {
            var service = await device.GetServiceAsync(ServiceUuid, cancellationToken)
                ?? throw new InvalidOperationException("设备未提供 FallingDetect 服务");
            _provisioningCharacteristic = await service.GetCharacteristicAsync(CharacteristicUuid);
            _telemetryCharacteristic = await service.GetCharacteristicAsync(TelemetryUuid)
                ?? throw new InvalidOperationException("设备未提供识别结果通知特征");
            _telemetryCharacteristic.ValueUpdated += OnTelemetryUpdated;
            await _telemetryCharacteristic.StartUpdatesAsync();
            StatusChanged?.Invoke(this, $"已连接 {device.Name ?? "FallingDetect"} · 识别结果接收中");
        }
        catch
        {
            await DisconnectAsync();
            throw;
        }
    }

    public async Task WriteWifiCredentialsAsync(string ssid, string password, CancellationToken cancellationToken = default)
    {
        if (_provisioningCharacteristic is null)
            throw new InvalidOperationException("请先连接设备");
        var ssidBytes = System.Text.Encoding.UTF8.GetBytes(ssid);
        var passwordBytes = System.Text.Encoding.UTF8.GetBytes(password);
        if (ssidBytes.Length is < 1 or > 32 || passwordBytes.Length > 64)
            throw new ArgumentException("SSID 最多 32 字节，密码最多 64 字节");
        var payload = new byte[3 + ssidBytes.Length + passwordBytes.Length];
        payload[0] = 0x01;
        payload[1] = (byte)ssidBytes.Length;
        payload[2] = (byte)passwordBytes.Length;
        ssidBytes.CopyTo(payload, 3);
        passwordBytes.CopyTo(payload, 3 + ssidBytes.Length);
        await _provisioningCharacteristic.WriteAsync(payload, cancellationToken);
        StatusChanged?.Invoke(this, "Wi-Fi 配置已下发");
    }

    private void OnTelemetryUpdated(object? sender, CharacteristicUpdatedEventArgs args)
    {
        var source = args.Characteristic;
        Vl53L8FallEventArgs.TryParse(source.Value, out var fall);
        Vl53L8GestureEventArgs.TryParse(source.Value, out var gesture);
        if (fall is null && gesture is null) return;
        MainThread.BeginInvokeOnMainThread(() =>
        {
            // Ignore a queued notification from an earlier connection.
            if (ReferenceEquals(source, _telemetryCharacteristic) && IsConnected)
            {
                if (fall is not null) FallReceived?.Invoke(this, fall);
                if (gesture is not null) GestureReceived?.Invoke(this, gesture);
            }
        });
    }

    private void ClearCharacteristics()
    {
        if (_telemetryCharacteristic is not null)
            _telemetryCharacteristic.ValueUpdated -= OnTelemetryUpdated;
        _telemetryCharacteristic = null;
        _provisioningCharacteristic = null;
    }

    private void OnDeviceDisconnected(object? sender, DeviceEventArgs args)
    {
        if (_device?.Id != args.Device.Id) return;
        ClearCharacteristics();
        _device = null;
        StatusChanged?.Invoke(this, "未连接");
    }

    private void OnDeviceConnectionLost(object? sender, DeviceErrorEventArgs args)
    {
        if (_device?.Id != args.Device.Id) return;
        ClearCharacteristics();
        _device = null;
        StatusChanged?.Invoke(this, "连接已断开");
    }

    public async Task DisconnectAsync()
    {
        var device = _device;
        var telemetry = _telemetryCharacteristic;
        ClearCharacteristics();
        _device = null;
        if (telemetry is not null)
        {
            try { await telemetry.StopUpdatesAsync(); } catch { }
        }
        if (device is not null)
        {
            try { await _adapter.DisconnectDeviceAsync(device); } catch { }
        }
        StatusChanged?.Invoke(this, "未连接");
    }

    public async ValueTask DisposeAsync()
    {
        await DisconnectAsync();
        _adapter.DeviceDisconnected -= OnDeviceDisconnected;
        _adapter.DeviceConnectionLost -= OnDeviceConnectionLost;
    }
}
