"""Local dataset review: python NN/dataset_review/server.py --no-browser."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import secrets
import shutil
import threading
import time
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
LABELS = ('运动', '跌倒', '有人静止', '无人静止')
MAX_IMPORT_BYTES = 50 * 1024 * 1024
MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def parse_record(body):
    reader = csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
    fields = reader.fieldnames or []
    required = ['unix_ms'] + [f'zone_{i:02d}_mm' for i in range(64)]
    if any(k not in fields for k in required):
        raise ValueError('CSV 缺少 unix_ms 或 64 个 zone_xx_mm 字段')
    rows = list(reader)
    if not rows:
        raise ValueError('CSV 没有数据帧')
    raw = np.asarray([[int(r[k]) for k in required[1:]] for r in rows], dtype=np.int64)
    if np.any((raw < -1) | (raw > 65535)):
        raise ValueError('距离超出 -1～65535 mm 范围')
    raw[raw == 65535] = -1
    times = np.asarray([int(r['unix_ms']) for r in rows], dtype=np.int64)
    valid = raw >= 0
    counts = valid.sum(1)
    dt = np.diff(times)
    common = valid[1:] & valid[:-1]
    energy = np.r_[0., (np.abs(raw[1:] - raw[:-1]) * common).sum(1) / np.maximum(common.sum(1), 1)]
    weights = np.where(valid, (1 - np.clip(raw, 0, 4000) / 4000.) ** 2, 0)
    mass = weights.sum(1)
    cy = (weights * np.repeat(np.arange(8), 8)).sum(1) / np.maximum(mass, 1e-9)
    cy = [float(v) if m > 0 else None for v, m in zip(cy, mass)]
    longest = run = 0
    for i in range(len(rows)):
        if counts[i] < 4:
            run = 0
        else:
            run = run + 1 if i and 40 <= dt[i - 1] <= 175 else 1
            longest = max(longest, run)
    flags = []
    if longest < 24:
        flags.append('不足24连续帧')
    if np.any((dt < 40) | (dt > 175)):
        flags.append('时间间隔异常')
    if np.any(counts < 4):
        flags.append('有效区域不足')
    return dict(fields=fields, rows=rows, raw=raw.tolist(), times=times.tolist(),
                counts=counts.tolist(), energy=energy.round(3).tolist(), cy=cy,
                fingerprint=hashlib.sha256(raw.astype('<i4').tobytes()).hexdigest(),
                frames=len(rows), duration_s=round(max(0, int(times[-1] - times[0])) / 1000, 3),
                median_interval_ms=float(np.median(dt)) if len(dt) else None,
                mean_motion_mm=round(float(energy[1:].mean()), 2) if len(rows) > 1 else 0,
                valid_percent=round(float(valid.mean() * 100), 1), longest=longest, flags=flags)


class ReviewStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / 'annotations.json'
        self.document = json.loads(self.path.read_text('utf-8')) if self.path.exists() else {'version': 1, 'records': {}}
        self.records = {}
        self.errors = []
        self.root = None
        self.lock = threading.RLock()

    def scan(self, root):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('数据目录不存在')
        # Scan fully before replacing the current view; a bad path never loses edits.
        records, errors = {}, []
        for path in sorted(root.rglob('*')):
            if not path.is_file() or self.workspace == path or self.workspace in path.parents:
                continue
            if path.suffix.lower() != '.zip' and path.name.lower() != 'frames.csv':
                continue
            try:
                if path.suffix.lower() == '.zip':
                    with zipfile.ZipFile(path) as archive:
                        entries = [(name, archive.read(name)) for name in archive.namelist() if name.endswith('frames.csv')]
                else:
                    entries = [('', path.read_bytes())]
                for member, body in entries:
                    try:
                        data = parse_record(body)
                        source = str(path) + ('::' + member if member else '')
                        key = hashlib.sha256(source.encode('utf-8')).hexdigest()[:24]
                        inferred = next((part for part in reversed(path.parts[:-1]) if part in LABELS), None)
                        name = (member.rsplit('/frames.csv', 1)[0] if '/' in member else path.stem) if member else path.parent.name
                        records[key] = dict(id=key, source=source, name=name,
                                            archive=path.name, original_label=inferred,
                                            sha256=hashlib.sha256(body).hexdigest(), data=data)
                    except (ValueError, TypeError, OverflowError, KeyError, UnicodeError) as ex:
                        errors.append(f'{path.name} / {member}: {ex}')
            except (OSError, zipfile.BadZipFile, RuntimeError) as ex:
                errors.append(f'{path.name}: {ex}')
        groups = {}
        for r in records.values():
            groups.setdefault(r['data']['fingerprint'], []).append(r['id'])
        for r in records.values():
            r['duplicates'] = [k for k in groups[r['data']['fingerprint']] if k != r['id']]
        self.records, self.errors, self.root = records, errors, root
        return self.listing()

    def annotation(self, record):
        saved = self.document['records'].get(record['id'])
        if saved and saved['sha256'] == record['sha256']:
            return saved
        return None

    def import_file(self, filename, content):
        if (not isinstance(filename, str) or not filename or len(filename) > 180
                or any(c in filename for c in '/\\:') or any(ord(c) < 32 for c in filename)
                or filename in ('.', '..') or filename.endswith((' ', '.'))):
            raise ValueError('文件名无效，请使用不含路径的 ZIP 或 CSV 文件名')
        suffix = Path(filename).suffix.lower()
        if suffix not in ('.csv', '.zip') or not content:
            raise ValueError('请选择非空 ZIP 或 CSV 文件')
        if len(content) > MAX_IMPORT_BYTES:
            raise ValueError('单个文件不能超过 50 MB')
        if self.root is None:
            raise ValueError('请先扫描数据目录')
        try:
            if suffix == '.csv':
                if len(content) > MAX_CSV_BYTES:
                    raise ValueError('单个 CSV 不能超过 20 MB')
                parse_record(content)
                count = 1
            else:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    infos = archive.infolist()
                    if len(infos) > 2000 or sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
                        raise ValueError('ZIP 内容过大（展开后最多 100 MB、2000 个条目）')
                    for info in infos:
                        name = PurePosixPath(info.filename)
                        if name.is_absolute() or '..' in name.parts or '\\' in info.filename or ':' in info.filename:
                            raise ValueError('ZIP 中含有不安全的文件路径')
                    recordings = [info for info in infos if not info.is_dir() and
                                  (info.filename == 'frames.csv' or info.filename.endswith('/frames.csv'))]
                    if not 1 <= len(recordings) <= 100:
                        raise ValueError('ZIP 必须包含 1～100 个 frames.csv 录制')
                    if len({info.filename for info in recordings}) != len(recordings):
                        raise ValueError('ZIP 中的录制路径重复')
                    if sum(info.file_size for info in recordings) > MAX_CSV_BYTES:
                        raise ValueError('ZIP 中 CSV 总量不能超过 20 MB')
                    for info in recordings:
                        if info.file_size > MAX_CSV_BYTES:
                            raise ValueError('ZIP 中单个 CSV 不能超过 20 MB')
                        parse_record(archive.read(info))
                    count = len(recordings)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as ex:
            raise ValueError('无法读取 ZIP：' + str(ex)) from ex
        except (OverflowError, csv.Error) as ex:
            raise ValueError('CSV 内容或数值超出支持范围：' + str(ex)) from ex
        imports = self.root / 'imports'
        if imports.is_symlink() or (imports.exists() and imports.resolve().parent != self.root):
            raise ValueError('导入目录不能是外部链接')
        destination = imports / hashlib.sha256(content).hexdigest()
        if destination.exists():
            return dict(imported=0, skipped=count, listing=self.listing())
        imports_existed = imports.exists()
        imports.mkdir(exist_ok=True)
        destination.mkdir()
        previous = self.records, self.errors, self.root
        try:
            if suffix == '.zip':
                path = destination / filename
            else:
                path = destination / Path(filename).stem / 'frames.csv'
                path.parent.mkdir()
            path.write_bytes(content)
            listing = self.scan(self.root)
            return dict(imported=count, skipped=0, listing=listing)
        except Exception:
            self.records, self.errors, self.root = previous
            shutil.rmtree(destination)
            if not imports_existed:
                imports.rmdir()
            raise

    def listing(self):
        result = []
        for r in self.records.values():
            data = r['data']
            a = self.annotation(r)
            stale = r['id'] in self.document['records'] and a is None
            flags = list(data['flags'])
            if r['duplicates']:
                flags.append('重复录制')
            if stale:
                flags.append('原始内容已变更')
            result.append({**{k: r[k] for k in ('id', 'source', 'name', 'archive', 'original_label', 'duplicates')},
                           **{k: data[k] for k in ('frames', 'duration_s', 'longest', 'valid_percent', 'mean_motion_mm', 'median_interval_ms')},
                           'flags': flags, 'annotation': a})
        return dict(root=str(self.root or ''), workspace=str(self.workspace), records=result, errors=self.errors)

    def detail(self, key):
        r = self.records[key]
        d = r['data']
        return dict(id=key, sha256=r['sha256'], raw=d['raw'], times=d['times'], counts=d['counts'], energy=d['energy'], cy=d['cy'], annotation=self.annotation(r))

    def save(self, body):
        r = self.records[body['id']]
        if body.get('sha256') != r['sha256']:
            raise ValueError('数据内容已变更，请重新打开后标注')
        label = body.get('label') or None
        if label is not None and label not in LABELS:
            raise ValueError('未知分类')
        ranges = body.get('ranges', [])
        if not isinstance(ranges, list):
            raise ValueError('片段格式错误')
        clean = []
        for item in ranges:
            if not isinstance(item, dict):
                raise ValueError('片段格式错误')
            start, end = item.get('start'), item.get('end')
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < r['data']['frames'] or item.get('label') not in LABELS:
                raise ValueError('片段范围或分类错误')
            clean.append(dict(start=start, end=end, label=item['label']))
        clean.sort(key=lambda x: x['start'])
        if any(b['start'] <= a['end'] for a, b in zip(clean, clean[1:])):
            raise ValueError('片段不能重叠，请先删除重叠片段')
        previous = self.document['records'].get(r['id'])
        value = dict(sha256=r['sha256'], source=r['source'], label=label,
                     excluded=bool(body.get('excluded')), note=str(body.get('note', ''))[:4000], ranges=clean,
                     updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        self.document['records'][r['id']] = value
        try:
            self.persist()
        except Exception:
            if previous is None:
                self.document['records'].pop(r['id'], None)
            else:
                self.document['records'][r['id']] = previous
            raise
        return value

    def persist(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.document, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    def export(self):
        target = self.workspace / 'exports' / ('reviewed_' + time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
        selected = []
        for r in self.records.values():
            a = self.annotation(r)
            if not a or a['excluded']:
                continue
            # Range labels override the whole-record label; preserve unselected spans
            # only when a whole-record label was explicitly saved.
            cursor = 0
            for segment in a['ranges']:
                if a['label'] and cursor < segment['start']:
                    selected.append((r, a['label'], cursor, segment['start'] - 1))
                selected.append((r, segment['label'], segment['start'], segment['end']))
                cursor = segment['end'] + 1
            if a['label'] and cursor < r['data']['frames']:
                selected.append((r, a['label'], cursor, r['data']['frames'] - 1))
        if not selected:
            raise ValueError('没有已标注且未排除的数据，请先保存整段分类或片段标注')
        target.mkdir(parents=True)
        manifest = []
        for r, label, start, end in selected:
            name = f'{r["id"]}_{start + 1}-{end + 1}'
            path = target / label / (name + '.zip')
            path.parent.mkdir(exist_ok=True)
            entry = dict(source=r['source'], sha256=r['sha256'], label=label, start_frame=start + 1, end_frame=end + 1,
                         export=str(path.relative_to(target)))
            out = io.StringIO(newline='')
            writer = csv.DictWriter(out, fieldnames=r['data']['fields'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(r['data']['rows'][start:end + 1])
            with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('frames.csv', out.getvalue().encode('utf-8-sig'))
                archive.writestr('metadata.json', json.dumps({'schemaVersion': 1, 'review': entry, 'timestampSource': 'original CSV unix_ms'}, ensure_ascii=False))
            manifest.append(entry)
        (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return dict(path=str(target), clips=len(manifest), frames=sum(x['end_frame'] - x['start_frame'] + 1 for x in manifest))


def make_handler(store, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, payload, mime='application/json; charset=utf-8'):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                page = Path(__file__).with_name('index.html').read_text('utf-8').replace('__SESSION_TOKEN__', token)
                return self.respond(200, page.encode('utf-8'), 'text/html; charset=utf-8')
            if self.headers.get('X-Review-Token') != token:
                return self.respond(403, {'error': '请在本工具页面中操作'})
            try:
                with store.lock:
                    if self.path == '/api/records':
                        return self.respond(200, store.listing())
                    if self.path.startswith('/api/record/'):
                        return self.respond(200, store.detail(self.path.rsplit('/', 1)[-1]))
                self.respond(404, {'error': '未找到'})
            except (KeyError, ValueError, TypeError):
                self.respond(404, {'error': '记录已变化，请重新扫描'})

        def do_POST(self):
            if self.headers.get('X-Review-Token') != token:
                return self.respond(403, {'error': '请在本工具页面中操作'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                route = urlsplit(self.path)
                if route.path == '/api/import':
                    if size > MAX_IMPORT_BYTES:
                        return self.respond(413, {'error': '单个文件不能超过 50 MB'})
                    if size <= 0:
                        raise ValueError('文件为空')
                    filename = parse_qs(route.query).get('filename', [''])[0]
                    content = self.rfile.read(size)
                    if len(content) != size:
                        raise ValueError('文件上传不完整，请重试')
                    with store.lock:
                        result = store.import_file(filename, content)
                    return self.respond(200, result)
                if not 0 < size <= 1024 * 1024:
                    raise ValueError('请求过大或为空')
                body = json.loads(self.rfile.read(size))
                with store.lock:
                    if self.path == '/api/scan':
                        result = store.scan(body['root'])
                    elif self.path == '/api/save':
                        result = store.save(body)
                    elif self.path == '/api/export':
                        result = store.export()
                    else:
                        return self.respond(404, {'error': '未找到'})
                self.respond(200, result)
            except (ValueError, KeyError, OSError, TypeError) as ex:
                self.respond(400, {'error': str(ex)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description='VL53L8CX 本地数据集分类筛查')
    parser.add_argument('--data', type=Path, default=PROJECT / 'NN' / 'Data')
    parser.add_argument('--workspace', type=Path, default=PROJECT / 'NN' / 'review')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    store = ReviewStore(args.workspace)
    store.scan(args.data)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store, secrets.token_urlsafe(32)))
    print(f'Dataset review: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Annotations: {store.path}', flush=True)
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{server.server_port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
