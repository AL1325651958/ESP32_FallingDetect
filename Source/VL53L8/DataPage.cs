using VL53L8.Services;

namespace VL53L8;

public sealed class DataPage : ContentPage
{
    private readonly VerticalStackLayout _list = new() { Spacing = 12 };
    private readonly Label _status = new() { TextColor = Color.FromArgb("#627D98"), FontSize = 16 };
    private readonly Button _exportAll = new() { Text = "导出全部", BackgroundColor = Color.FromArgb("#287D52"), IsEnabled = false };
    private readonly Button _refresh = new() { Text = "刷新", BackgroundColor = Color.FromArgb("#102A43") };
    private readonly Button _deleteAll = new() { Text = "删除全部", BackgroundColor = Color.FromArgb("#B42318"), IsEnabled = false };
    private bool _hasRecords;
    private bool _busy;

    public DataPage()
    {
        Title = "数据记录";
        _refresh.Clicked += async (_, _) => { if (!_busy) await LoadAsync(); };
        _exportAll.Clicked += async (_, _) => await ExportAllAsync();
        _deleteAll.Clicked += async (_, _) => await DeleteAllAsync();
        Content = new ScrollView
        {
            Content = new VerticalStackLayout
            {
                Padding = 20, Spacing = 14,
                Children =
                {
                    _exportAll, _deleteAll, _refresh, _status, _list
                }
            }
        };
    }

    protected override async void OnAppearing() { base.OnAppearing(); await LoadAsync(); }

    private async Task LoadAsync()
    {
        try
        {
            var records = await RecordingService.ListAsync();
            _hasRecords = records.Count > 0;
            SetBusy(_busy);
            _list.Children.Clear();
            _status.Text = records.Count == 0 ? "暂无记录" : $"{records.Count} 条记录";
            foreach (var item in records)
            {
                var share = new Button { Text = "导出", BackgroundColor = Color.FromArgb("#287D52") };
                share.Clicked += async (_, _) => await ExportAsync(item, share);
                var delete = new Button { Text = "删除", BackgroundColor = Color.FromArgb("#B42318") };
                delete.Clicked += async (_, _) => await DeleteAsync(item, delete);
                var preview = new VerticalStackLayout { Spacing = 8, Padding = new Thickness(0, 0, 0, 8), Children =
                {
                    new Label { Text = item.Name, FontSize = 18, TextColor = Color.FromArgb("#102A43") },
                    new Label { Text = "回放", FontSize = 16, TextColor = Color.FromArgb("#3366CC") }
                }};
                var tap = new TapGestureRecognizer();
                tap.Tapped += async (_, _) => await OpenReplayAsync(item);
                preview.GestureRecognizers.Add(tap);
                var card = new Border
                {
                    Padding = 14, Stroke = Color.FromArgb("#D9E2EC"),
                    Content = new VerticalStackLayout { Spacing = 8, Children = { preview, share, delete } }
                };
                _list.Add(card);
            }
        }
        catch (Exception ex) { ShowError("读取失败，请刷新", ex); }
    }

    private void SetBusy(bool busy)
    {
        _busy = busy;
        _refresh.IsEnabled = !busy;
        _exportAll.IsEnabled = !busy && _hasRecords;
        _deleteAll.IsEnabled = !busy && _hasRecords;
        _list.IsEnabled = !busy;
    }

    private async Task OpenReplayAsync(RecordingItem item)
    {
        if (_busy) return;
        SetBusy(true);
        try { await Navigation.PushAsync(new ReplayPage(item)); }
        catch (Exception ex) { ShowError("无法打开回放", ex); }
        finally { SetBusy(false); }
    }

    private async Task DeleteAllAsync()
    {
        if (_busy || !_hasRecords) return;
        SetBusy(true);
        try
        {
            var records = await RecordingService.ListAsync();
            if (records.Count == 0) { await LoadAsync(); return; }
            bool confirmed = await DisplayAlertAsync("删除全部记录？",
                $"删除 {records.Count} 条记录及照片？删除后无法恢复。",
                "全部删除", "取消");
            if (!confirmed) return;
            _status.Text = "删除中…";
            await RecordingService.DeleteAllAsync();
            await LoadAsync();
            _status.Text = "暂无记录";
        }
        catch (Exception ex) { ShowError("删除失败，请重试", ex); }
        finally { SetBusy(false); }
    }

    private async Task ExportAllAsync()
    {
        if (_busy) return;
        SetBusy(true);
        try
        {
            _status.Text = "导出中…";
            string zip = await RecordingService.ExportAllAsync();
            await Share.Default.RequestAsync(new ShareFileRequest
            {
                Title = "分享 VL53L8 全部记录", File = new ShareFile(zip, "application/zip"),
                PresentationSourceBounds = new Rect(0, 0, Width, Math.Max(1, Height / 2))
            });
            _status.Text = "已导出";
        }
        catch (Exception ex) { ShowError("导出失败，请重试", ex); }
        finally { SetBusy(false); }
    }

    private async Task DeleteAsync(RecordingItem item, Button button)
    {
        if (_busy) return;
        SetBusy(true); button.IsEnabled = false;
        try
        {
            bool confirmed = await DisplayAlertAsync("删除记录？",
                item.Name + "\n删除记录及照片，无法恢复。",
                "删除", "取消");
            if (!confirmed) return;
            await RecordingService.DeleteAsync(item);
            await LoadAsync();
        }
        catch (Exception ex) { ShowError("删除失败，请重试", ex); }
        finally { SetBusy(false); button.IsEnabled = true; }
    }

    private async Task ExportAsync(RecordingItem item, Button button)
    {
        if (_busy) return;
        SetBusy(true); button.IsEnabled = false;
        try
        {
            _status.Text = "导出中…";
            string zip = await RecordingService.ExportAsync(item);
            await Share.Default.RequestAsync(new ShareFileRequest
            {
                Title = "分享 VL53L8 测距记录", File = new ShareFile(zip, "application/zip"),
                PresentationSourceBounds = new Rect(0, 0, Width, Math.Max(1, Height / 2))
            });
            _status.Text = "已导出";
        }
        catch (Exception ex) { ShowError("导出失败，请重试", ex); }
        finally { SetBusy(false); button.IsEnabled = true; }
    }

    private void ShowError(string message, Exception error)
    {
        System.Diagnostics.Debug.WriteLine(error);
        _status.Text = message;
    }
}
