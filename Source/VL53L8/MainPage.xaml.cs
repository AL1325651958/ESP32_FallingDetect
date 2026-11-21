using CommunityToolkit.Maui.Views;
using CommunityToolkit.Maui.Core;
using Plugin.BLE.Abstractions.Contracts;
using VL53L8.Services;
using VL53L8.Views;

namespace VL53L8;

public partial class MainPage : ContentPage
{
    private readonly Vl53L8BleService _ble = new();
    private readonly Vl53L8WifiService _wifi = new();
    private readonly HeatmapDrawable _heatmap = new();
    private readonly RecordingService _recording = new();
    private CameraView? _camera;
    private CancellationTokenSource? _cameraOpening;
    private bool _cameraOpen;
    private bool _cameraChanging;
    private readonly SemaphoreSlim _captureGate = new(1, 1);
    private CancellationTokenSource? _photoCancellation;
    private Task? _photoTask;
    private RecordedFrame? _latest;
    private bool _changingRecording;
    private bool _showingData;
    private CancellationTokenSource? _scanCancellation;
    private IDispatcherTimer? _gestureTimer;
    private DateTimeOffset? _lastGestureAt;
    private DateTimeOffset? _lastFallAt;
    private readonly FallStatusDisplay _fallDisplay = new();
    private int _noticeVersion;

    public MainPage()
    {
        InitializeComponent();
        HeatmapView.Drawable = _heatmap;
        _wifi.FrameReceived += OnFrameReceived;
        _ble.GestureReceived += OnGestureReceived;
        _ble.FallReceived += OnFallReceived;
        _wifi.FallReceived += OnFallReceived;
        _ble.StatusChanged += (_, status) => MainThread.BeginInvokeOnMainThread(() =>
        {
            SetStatus(status);
            ScanButton.Text = _ble.IsConnected ? "断开" : "扫描设备";
            if (!_ble.IsConnected) ClearGesture("蓝牙未连接");
        });
    }

    private async void OnScanClicked(object? sender, EventArgs e)
    {
        if (_ble.IsConnected)
        {
            await _ble.DisconnectAsync();
            return;
        }

        ScanButton.IsEnabled = false;
        _scanCancellation?.Cancel();
        _scanCancellation = new CancellationTokenSource();
        try
        {
            var devices = await _ble.ScanAsync(_scanCancellation.Token);
            DeviceList.ItemsSource = devices;
            DeviceList.IsVisible = devices.Count > 0;
            SetStatus(devices.Count == 0 ? "未发现设备" : $"发现 {devices.Count} 台设备");
        }
        catch (Exception ex)
        {
            SetStatus("扫描失败");
            ShowError("扫描失败，请检查蓝牙权限", ex);
        }
        finally { ScanButton.IsEnabled = true; }
    }

    private async void OnDeviceSelected(object? sender, SelectionChangedEventArgs e)
    {
        if (e.CurrentSelection.FirstOrDefault() is not IDevice device) return;
        DeviceList.SelectedItem = null;
        ScanButton.Text = "断开";
        try
        {
            await _ble.ConnectAsync(device);
            await _wifi.StartAsync();
            DeviceList.IsVisible = false;
        }
        catch (Exception ex)
        {
            ScanButton.Text = "扫描设备";
            SetStatus("连接失败");
            ShowError("连接失败，请重试", ex);
        }
    }

    private void OnWifiSettingsClicked(object? sender, EventArgs e)
    {
        WifiSettingsPanel.IsVisible = !WifiSettingsPanel.IsVisible;
        WifiSettingsButton.Text = WifiSettingsPanel.IsVisible ? "收起 Wi-Fi 设置" : "Wi-Fi 设置";
    }

    private async void OnProvisionClicked(object? sender, EventArgs e)
    {
        if (!_ble.IsConnected) { ShowNotice("请先连接设备", true); return; }
        if (string.IsNullOrWhiteSpace(WifiSsidEntry.Text)) { ShowNotice("请输入 Wi-Fi 名称", true); return; }
        try
        {
            await _ble.WriteWifiCredentialsAsync(WifiSsidEntry.Text?.Trim() ?? "", WifiPasswordEntry.Text ?? "");
            WifiSettingsPanel.IsVisible = false;
            WifiSettingsButton.Text = "Wi-Fi 设置";
            ShowNotice("配置已发送");
        }
        catch (Exception ex) { ShowError("配网失败，请检查网络信息", ex); }
    }

    private static string GestureName(int index) => index switch
    {
        0 => "石头", 1 => "布", 2 => "剪刀", _ => "unknown"
    };

    private static string GestureIcon(int index) => index switch
    {
        0 => "✊",
        1 => "✋",
        2 => "✌",
        _ => "?"
    };

    private void OnGestureReceived(object? sender, Vl53L8GestureEventArgs e)
    {
        GestureCard.IsVisible = true;
        _lastGestureAt = e.ReceivedAt;
        GestureIconLabel.Text = GestureIcon(e.StableIndex);
        GestureLabel.Text = GestureName(e.StableIndex);
    }

    private void ClearGesture(string status)
    {
        _lastGestureAt = null;
        GestureIconLabel.Text = "?";
        GestureLabel.Text = status;
    }

    private void OnFallReceived(object? sender, Vl53L8FallEventArgs e)
    {
        _lastFallAt = e.ReceivedAt;
        FallLabel.Text = _fallDisplay.Update(e);
        FallLabel.TextColor = Color.FromArgb(_fallDisplay.HasAlert ? "#C62828" : e.IsReady && e.StableIndex >= 0 ? "#287D52" : "#627D98");
    }

    private void CheckFallTimeout()
    {
        if (_lastFallAt is null || DateTimeOffset.UtcNow - _lastFallAt.Value <= TimeSpan.FromSeconds(3)) return;
        _lastFallAt = null;
        FallLabel.Text = _fallDisplay.TimeoutText;
        FallLabel.TextColor = Color.FromArgb(_fallDisplay.HasAlert ? "#C62828" : "#627D98");
    }

    private void OnFrameReceived(object? sender, Vl53L8FrameEventArgs e)
    {
        var valid = 0;
        var min = ushort.MaxValue;
        var max = 0;
        for (var i = 0; i < e.Distances.Length; i++)
        {
            var value = e.Distances[i];
            if (value == ushort.MaxValue) continue;
            valid++; min = Math.Min(min, value); max = Math.Max(max, value);

        }
        _latest = RecordedFrame.From(e);
        _heatmap.Update(e.Distances);
        HeatmapView.Invalidate();
        if (_recording.IsRecording)
        {
            if (!_recording.TryRecord(_latest)) _ = FinishRecordingAsync();
            else RecordingLabel.Text = $"录制中 · {_recording.FrameCount} 帧";
        }
        RangeLabel.Text = valid == 0 ? "无有效数据" : $"{min}–{max} mm";
    }

    private async void OnSaveOnceClicked(object? sender, EventArgs e)
    {
        if (_latest is null) { ShowNotice("暂无数据", true); return; }
        if (DateTimeOffset.UtcNow - _latest.ReceivedAt > TimeSpan.FromSeconds(3))
        { ShowNotice("数据已中断", true); return; }
        SaveOnceButton.IsEnabled = false;
        RecordButton.IsEnabled = false;
        try
        {
            var frame = _latest;
            CapturedPhoto? photo = _cameraOpen ? await CapturePhotoAsync(CancellationToken.None) : null;
            if (photo?.AssociatedFrame is { } associated) frame = associated;
            await RecordingService.SaveSingleAsync(frame, photo);
            ShowNotice("已保存");
        }
        catch (Exception ex) { ShowError("保存失败，请重试", ex); }
        finally { SaveOnceButton.IsEnabled = true; RecordButton.IsEnabled = true; }
    }

    private async void OnRecordClicked(object? sender, EventArgs e)
    {
        if (_changingRecording) return;
        if (_recording.IsRecording) { await FinishRecordingAsync(); return; }
        if (_latest is null || DateTimeOffset.UtcNow - _latest.ReceivedAt > TimeSpan.FromSeconds(3))
        { ShowNotice("暂无实时数据", true); return; }
        _changingRecording = true; RecordButton.IsEnabled = false;
        try
        {
            await _recording.StartAsync();
            StartPhotoLoop();
            RecordButton.Text = "停止录制";
            RecordingLabel.Text = "录制中";
            RecordingLabel.IsVisible = true;
            DeviceDisplay.Current.KeepScreenOn = true;
        }
        catch (Exception ex) { ShowError("无法开始录制", ex); }
        finally { _changingRecording = false; RecordButton.IsEnabled = true; }
    }

    private async Task FinishRecordingAsync()
    {
        if (_changingRecording) return;
        _changingRecording = true; RecordButton.IsEnabled = false;
        try
        {
            await StopPhotoLoopAsync();
            await _recording.StopAsync();
            ShowNotice($"已保存 {_recording.FrameCount} 帧");
        }
        catch (Exception ex) { ShowError("录制保存失败", ex); }
        finally
        {
            RecordButton.Text = "开始录制";
            RecordingLabel.IsVisible = false;
            DeviceDisplay.Current.KeepScreenOn = false;
            RecordButton.IsEnabled = true; _changingRecording = false;
        }
    }

    private async void OnDataClicked(object? sender, EventArgs e)
    {
        if (_changingRecording || _showingData || !SaveOnceButton.IsEnabled || _cameraChanging) return;
        if (_recording.IsRecording) await FinishRecordingAsync();
        await CloseCameraAsync();
        _showingData = true;
        try { await Navigation.PushAsync(new DataPage()); }
        catch (Exception ex) { _showingData = false; ShowError("无法打开记录", ex); }
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        _showingData = false;
        _gestureTimer ??= Dispatcher.CreateTimer();
        _gestureTimer.Interval = TimeSpan.FromSeconds(1);
        _gestureTimer.Tick -= OnGestureTimer;
        _gestureTimer.Tick += OnGestureTimer;
        _gestureTimer.Start();
        CheckFallTimeout();
        if (_lastGestureAt is not null && DateTimeOffset.UtcNow - _lastGestureAt.Value > TimeSpan.FromSeconds(3))
            ClearGesture("手势数据已超时");
        if (Application.Current?.Windows.FirstOrDefault() is { } window)
        {
            window.Stopped -= OnWindowStopped;
            window.Stopped += OnWindowStopped;
        }
        // UDP reception is independent of the provisioning BLE connection.
        try { await _wifi.StartAsync(); }
        catch (Exception ex) { ShowError("数据接收不可用，请重试", ex); }
    }

    private void OnGestureTimer(object? sender, EventArgs e)
    {
        CheckFallTimeout();
        if (_lastGestureAt is not null && DateTimeOffset.UtcNow - _lastGestureAt.Value > TimeSpan.FromSeconds(3))
            ClearGesture("手势数据已超时");
    }

    private async void OnWindowStopped(object? sender, EventArgs e)
    {
        if (_recording.IsRecording) await FinishRecordingAsync();
        await CloseCameraAsync();
    }

    private async void OnCameraClicked(object? sender, EventArgs e)
    {
        if (_cameraChanging || _changingRecording || !SaveOnceButton.IsEnabled) return;
        _cameraChanging = true; CameraButton.IsEnabled = false;
        try
        {
            if (_cameraOpen) { await CloseCameraAsync(); return; }
            var permission = await Permissions.RequestAsync<Permissions.Camera>();
            if (permission != PermissionStatus.Granted)
            { ShowNotice("请允许相机权限", true); return; }
            _cameraOpening = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            var openingToken = _cameraOpening.Token;
            _camera ??= new CameraView();
            CameraHost.Content = _camera; CameraHost.IsVisible = true;
            if (_camera.Handler is null)
            {
                var loaded = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                void OnLoaded(object? s, EventArgs args) => loaded.TrySetResult();
                _camera.Loaded += OnLoaded;
                try { await loaded.Task.WaitAsync(TimeSpan.FromSeconds(10), openingToken); }
                finally { _camera.Loaded -= OnLoaded; }
            }
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(openingToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(15));
            var cameras = await _camera.GetAvailableCameras(timeout.Token);
            _camera.SelectedCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Rear) ?? cameras.FirstOrDefault();
            if (_camera.SelectedCamera is null) throw new InvalidOperationException("未找到可用相机。");
            await _camera.StartCameraPreview(timeout.Token);
            openingToken.ThrowIfCancellationRequested();
            _cameraOpen = true; CameraButton.Text = "关闭相机";
            if (_recording.IsRecording) StartPhotoLoop();
        }
        catch (Exception ex)
        {
            await CloseCameraAsync();
            ShowError("无法打开相机", ex);
        }
        finally { _cameraChanging = false; CameraButton.IsEnabled = true; }
    }

    private async Task<CapturedPhoto> CapturePhotoAsync(CancellationToken cancellationToken)
    {
        await _captureGate.WaitAsync(cancellationToken);
        try
        {
            if (!_cameraOpen || _camera is null) throw new InvalidOperationException("相机未打开。");
            var requested = DateTimeOffset.UtcNow;
            var associated = _latest;
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(10));
            using var media = await _camera.CaptureImage(timeout.Token);
            var completed = DateTimeOffset.UtcNow;
            using var memory = new MemoryStream();
            await media.CopyToAsync(memory, timeout.Token);
            return new CapturedPhoto(memory.ToArray(), requested, completed, associated);
        }
        finally { _captureGate.Release(); }
    }

    private void StartPhotoLoop()
    {
        if (!_cameraOpen || !_recording.IsRecording || _photoTask is not null) return;
        _photoCancellation = new CancellationTokenSource();
        _photoTask = RecordPhotosAsync(_photoCancellation.Token);
    }

    private async Task RecordPhotosAsync(CancellationToken token)
    {
        try
        {
            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(1));
            do
            {
                var photo = await CapturePhotoAsync(token);
                await _recording.SavePhotoAsync(photo);
            } while (await timer.WaitForNextTickAsync(token));
        }
        catch (OperationCanceledException) when (token.IsCancellationRequested) { }
        catch (Exception ex)
        {
            ShowError("拍照已停止，测距仍在录制", ex);
        }
    }

    private async Task StopPhotoLoopAsync()
    {
        _photoCancellation?.Cancel();
        if (_photoTask is not null) await _photoTask;
        _photoTask = null;
        _photoCancellation?.Dispose(); _photoCancellation = null;
    }

    private async Task CloseCameraAsync()
    {
        _cameraOpening?.Cancel();
        _cameraOpening?.Dispose(); _cameraOpening = null;
        await StopPhotoLoopAsync();
        await _captureGate.WaitAsync();
        try
        {
            _cameraOpen = false;
            _camera?.StopCameraPreview();
            CameraHost.IsVisible = false; CameraButton.Text = "打开相机";
        }
        catch (Exception ex) { ShowError("相机关闭失败", ex); }
        finally { _captureGate.Release(); }
    }

    private void SetStatus(string text)
    {
        var connected = _ble.IsConnected;
        StatusLabel.Text = text switch
        {
            var s when s.StartsWith("正在扫描", StringComparison.Ordinal) => "扫描中",
            var s when s.StartsWith("正在连接", StringComparison.Ordinal) => "连接中",
            var s when s.StartsWith("已连接", StringComparison.Ordinal) => "已连接",
            var s when s.StartsWith("Wi-Fi 配置", StringComparison.Ordinal) => connected ? "已连接" : "未连接",
            var s when s.StartsWith("发现 ", StringComparison.Ordinal) => "请选择设备",
            _ => text
        };
        StatusDot.Color = connected ? Color.FromArgb("#45B97C") : Color.FromArgb("#F0A34A");
        ScanButton.Text = connected ? "断开" : "扫描设备";
    }

    private void ShowError(string message, Exception error)
    {
        System.Diagnostics.Debug.WriteLine(error);
        ShowNotice(message, true);
    }

    private async void ShowNotice(string message, bool isError = false)
    {
        int version = ++_noticeVersion;
        FeedbackLabel.Text = message;
        FeedbackLabel.TextColor = Color.FromArgb(isError ? "#B42318" : "#287D52");
        FeedbackLabel.IsVisible = true;
        await Task.Delay(TimeSpan.FromSeconds(isError ? 5 : 3));
        if (version == _noticeVersion) FeedbackLabel.IsVisible = false;
    }

    protected override async void OnDisappearing()
    {
        ++_noticeVersion;
        FeedbackLabel.IsVisible = false;
        _gestureTimer?.Stop();
        base.OnDisappearing();
        if (_showingData) return;
        if (_recording.IsRecording) await FinishRecordingAsync();
        await CloseCameraAsync();
        _scanCancellation?.Cancel();
        await _wifi.StopAsync();
        await _ble.DisconnectAsync();
    }
}
