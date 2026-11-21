using System.Diagnostics;
using VL53L8.Services;
using VL53L8.Views;

namespace VL53L8;

public sealed class ReplayPage : ContentPage
{
    private readonly RecordingItem _item;
    private readonly HeatmapDrawable _drawable = new();
    private readonly GraphicsView _heatmap;
    private readonly Label _status = new() { Text = "读取中…", FontSize = 16, TextColor = Color.FromArgb("#627D98") };
    private readonly Label _details = new() { FontSize = 16, TextColor = Color.FromArgb("#102A43") };
    private readonly Label _position = new() { Text = "0 / 0 帧", FontSize = 16 };
    private readonly Slider _progress = new() { Minimum = 0, Maximum = 1, IsEnabled = false };
    private readonly Button _play = new() { Text = "播放", IsEnabled = false };
    private readonly Button _previous = new() { Text = "上一帧", IsEnabled = false };
    private readonly Button _next = new() { Text = "下一帧", IsEnabled = false };
    private RecordingReplay? _replay;
    private RecordedFrame? _frame;
    private int _index;
    private bool _visible, _loading, _updatingSlider, _dragging, _playing;
    private int _version;
    private CancellationTokenSource? _operation;

    public ReplayPage(RecordingItem item)
    {
        _item = item;
        Title = "数据回放";
        _heatmap = new GraphicsView { Drawable = _drawable, HeightRequest = 320, HorizontalOptions = LayoutOptions.Fill };
        _play.Clicked += async (_, _) => { if (_playing) Stop(); else await PlayAsync(); };
        _previous.Clicked += async (_, _) => await SeekAsync(_index - 1);
        _next.Clicked += async (_, _) => await SeekAsync(_index + 1);
        _progress.DragStarted += (_, _) => { _dragging = true; Stop(); };
        _progress.DragCompleted += async (_, _) => { _dragging = false; await SeekAsync((int)Math.Round(_progress.Value)); };
        _progress.ValueChanged += async (_, e) =>
        {
            if (!_updatingSlider && !_dragging) await SeekAsync((int)Math.Round(e.NewValue));
        };
        var controls = new Grid { ColumnDefinitions = { new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star) }, ColumnSpacing = 8 };
        controls.Add(_previous, 0); controls.Add(_play, 1); controls.Add(_next, 2);
        Content = new ScrollView { Content = new VerticalStackLayout
        {
            Padding = 20, Spacing = 14, Children =
            {
                new Label { Text = item.Name, FontSize = 22, TextColor = Color.FromArgb("#102A43") },
                _heatmap,
                _position, _progress, controls, _details, _status
            }
        }};
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();
        _visible = true;
        if (_loading) return;
        if (_replay is not null) { await SeekAsync(_index); return; }
        _loading = true;
        try
        {
            _replay = await RecordingService.OpenReplayAsync(_item);
            if (!_visible) return;
            _progress.Maximum = Math.Max(1, _replay.FrameCount - 1);
            _progress.IsEnabled = _replay.FrameCount > 1;
            if (_replay.FrameCount == 0) { _status.Text = "暂无可回放数据"; return; }
            await SeekAsync(0);
        }
        catch (Exception ex) { if (_visible) ShowError("无法打开回放", ex); }
        finally { _loading = false; }
    }

    protected override void OnDisappearing()
    {
        _visible = false;
        Stop();
        base.OnDisappearing();
    }

    private void Stop()
    {
        ++_version;
        _operation?.Cancel();
        _operation?.Dispose();
        _operation = null;
        _playing = false;
        _play.Text = "播放";
    }

    private async Task SeekAsync(int index)
    {
        if (!_visible || _replay is null || _replay.FrameCount == 0) return;
        Stop();
        int version = _version;
        using var cancellation = new CancellationTokenSource();
        _operation = cancellation;
        try
        {
            int target = Math.Clamp(index, 0, _replay.FrameCount - 1);
            var frame = await _replay.ReadFrameAsync(target, cancellation.Token);
            if (!_visible || version != _version) return;
            ShowFrame(target, frame);
            _status.IsVisible = false;
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { if (_visible && version == _version) ShowError("读取失败，请重试", ex); }
        finally { if (ReferenceEquals(_operation, cancellation)) _operation = null; }
    }

    private async Task PlayAsync()
    {
        if (!_visible || _replay is null || _replay.FrameCount < 2) return;
        if (_frame is null || _index >= _replay.FrameCount - 1) await SeekAsync(0);
        if (!_visible || _frame is null) return;
        Stop();
        int version = _version;
        using var cancellation = new CancellationTokenSource();
        var token = cancellation.Token;
        _operation = cancellation;
        _playing = true;
        _status.IsVisible = false;
        _play.Text = "暂停";
        var origin = _frame.ReceivedAt;
        var elapsed = Stopwatch.StartNew();
        TimeSpan scheduled = TimeSpan.Zero;
        try
        {
            while (_index < _replay.FrameCount - 1)
            {
                int target = _index + 1;
                var frame = await _replay.ReadFrameAsync(target, token);
                // Preserve receipt intervals; a phone clock adjustment must not reverse playback.
                scheduled = ReplayTiming.Offset(origin, frame.ReceivedAt, scheduled);
                while (scheduled > elapsed.Elapsed)
                {
                    var remaining = scheduled - elapsed.Elapsed;
                    if (remaining <= TimeSpan.Zero) break;
                    await Task.Delay(remaining > TimeSpan.FromSeconds(1) ? TimeSpan.FromSeconds(1) : remaining, token);
                }
                token.ThrowIfCancellationRequested();
                if (!_visible || version != _version) return;
                ShowFrame(target, frame);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { if (_visible && version == _version) ShowError("回放失败，请重试", ex); }
        finally
        {
            if (version == _version) { _playing = false; _play.Text = "播放"; }
            if (ReferenceEquals(_operation, cancellation)) _operation = null;
        }
    }

    private void ShowFrame(int index, RecordedFrame frame)
    {
        _index = index;
        _frame = frame;
        _drawable.Update(frame.Distances);
        _heatmap.Invalidate();
        _position.Text = $"{index + 1} / {_replay!.FrameCount} 帧";
        _updatingSlider = true;
        try { _progress.Value = index; }
        finally { _updatingSlider = false; }
        _previous.IsEnabled = index > 0;
        _next.IsEnabled = index < _replay.FrameCount - 1;
        _play.IsEnabled = _replay.FrameCount > 1;
        var valid = frame.Distances.Where(d => d != ushort.MaxValue).ToArray();
        string range = valid.Length == 0 ? "无有效距离" : $"{valid.Min()}–{valid.Max()} mm";
        _details.Text = range;
    }

    private void ShowError(string message, Exception error)
    {
        Debug.WriteLine(error);
        _status.Text = message;
        _status.IsVisible = true;
    }
}
