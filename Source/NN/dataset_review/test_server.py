"""Regression checks for review labels and lossless frame selection."""
import csv
import hashlib
import http.client
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import quote
from unittest.mock import patch

from server import ReviewStore, make_handler, parse_record


def fixture(frames=30, offset=0, gap=False):
    out = io.StringIO(newline='')
    fields = ['unix_ms', 'frame_number', 'note'] + [f'zone_{i:02d}_mm' for i in range(64)]
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(fields)
    for i in range(frames):
        values = [65535 if j == 0 else 500 + i + j for j in range(64)]
        writer.writerow([1700000000000 + offset + i * 100 + (1000 if gap and i >= 10 else 0),
                         i + 1, '原始,备注', *values])
    return out.getvalue().encode('utf-8-sig')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'data'
        self.root.mkdir()
        self.workspace = Path(self.temp.name) / 'review'
        self.path = self.root / '运动' / 'sample.zip'
        self.path.parent.mkdir()
        self.body = fixture()
        self.write_zip(self.path, self.body)
        self.store = ReviewStore(self.workspace)
        self.store.scan(self.root)
        self.record = next(iter(self.store.records.values()))

    def write_zip(self, path, body):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('录制/frames.csv', body)

    def save(self, **kwargs):
        return self.store.save(dict(id=self.record['id'], sha256=self.record['sha256'], **kwargs))

    def test_parse_quality_and_invalid_distances(self):
        data = parse_record(self.body)
        self.assertEqual(data['longest'], 30)
        self.assertEqual(data['flags'], [])
        self.assertEqual(data['raw'][0][0], -1)
        self.assertEqual(data['rows'][0]['zone_00_mm'], '65535')
        gap = parse_record(fixture(gap=True))
        self.assertEqual(gap['longest'], 20)
        self.assertIn('时间间隔异常', gap['flags'])
        self.assertIn('不足24连续帧', gap['flags'])

    def test_scan_duplicates_errors_and_unpacked(self):
        other = self.root / '有人静止' / 'record'
        other.mkdir(parents=True)
        (other / 'frames.csv').write_bytes(fixture(offset=70000))
        (self.root / 'bad.zip').write_bytes(b'not a zip')
        broken = self.root / 'bad'
        broken.mkdir()
        (broken / 'frames.csv').write_bytes(b'no columns\n')
        result = self.store.scan(self.root)
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(len(result['errors']), 2)
        self.assertTrue(all(len(r['duplicates']) == 1 for r in result['records']))
        self.assertEqual({r['original_label'] for r in result['records']}, {'运动', '有人静止'})

    def test_save_reload_and_stale_annotation(self):
        self.save(label='跌倒', note='审核备注')
        reloaded = ReviewStore(self.workspace)
        result = reloaded.scan(self.root)
        self.assertEqual(result['records'][0]['annotation']['note'], '审核备注')
        self.write_zip(self.path, fixture(31))
        result = reloaded.scan(self.root)
        self.assertIsNone(result['records'][0]['annotation'])
        self.assertIn('原始内容已变更', result['records'][0]['flags'])
        with self.assertRaises(ValueError):
            reloaded.save(dict(id=self.record['id'], sha256=self.record['sha256'], label='运动'))

    def test_reject_invalid_ranges_and_labels(self):
        cases = [dict(label='其他'), dict(ranges=[dict(start=0, end=30, label='运动')]),
                 dict(ranges=[dict(start=True, end=5, label='运动')]),
                 dict(ranges=[dict(start=0, end=9, label='运动'), dict(start=9, end=12, label='跌倒')])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.save(**case)
        self.assertFalse(self.store.path.exists())

    def test_failed_save_rolls_back(self):
        old = self.save(label='运动')
        with patch.object(self.store, 'persist', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.save(label='跌倒')
        self.assertEqual(self.store.annotation(self.record), old)

    def test_export_override_preserves_all_original_csv_values(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.save(label='运动', ranges=[dict(start=8, end=15, label='有人静止')])
        result = self.store.export()
        self.assertEqual((result['clips'], result['frames']), (3, 30))
        manifest = json.loads((Path(result['path']) / 'manifest.json').read_text('utf-8'))
        self.assertEqual([x['label'] for x in manifest], ['运动', '有人静止', '运动'])
        self.assertEqual([(x['start_frame'], x['end_frame']) for x in manifest], [(1, 8), (9, 16), (17, 30)])
        actual = []
        for item in manifest:
            self.assertEqual(item['sha256'], self.record['sha256'])
            with zipfile.ZipFile(Path(result['path']) / item['export']) as archive:
                actual.extend(list(csv.DictReader(io.StringIO(archive.read('frames.csv').decode('utf-8-sig')))))
        expected = list(csv.DictReader(io.StringIO(self.body.decode('utf-8-sig'))))
        self.assertEqual(actual, expected)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_export_ranges_only_and_exclusion(self):
        self.save(ranges=[dict(start=2, end=4, label='跌倒')])
        result = self.store.export()
        self.assertEqual((result['clips'], result['frames']), (1, 3))
        self.save(label='运动', excluded=True)
        with self.assertRaises(ValueError):
            self.store.export()

    def test_workspace_exports_not_rescanned(self):
        store = ReviewStore(self.root / 'review')
        store.scan(self.root)
        store.save(dict(id=self.record['id'], sha256=self.record['sha256'], label='运动'))
        store.export()
        self.assertEqual(len(store.scan(self.root)['records']), 1)

    def upload(self, filename, content, token='test-token', length=None):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.store, 'test-token'))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                headers = {'Content-Type': 'application/octet-stream', 'X-Review-Token': token}
                if length is not None:
                    connection.putrequest('POST', '/api/import?filename=' + quote(filename, safe=''))
                    for name, value in headers.items():
                        connection.putheader(name, value)
                    connection.putheader('Content-Length', str(length))
                    connection.endheaders(content)
                else:
                    connection.request('POST', '/api/import?filename=' + quote(filename, safe=''), content, headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_import_csv_preserves_existing_annotations_and_original_bytes(self):
        self.save(label='跌倒', note='保留原标注')
        before = self.path.read_bytes()
        saved = self.store.path.read_bytes()
        content = fixture(31)
        status, result = self.upload('新录制.csv', content)
        self.assertEqual(status, 200, result)
        self.assertEqual(result['imported'], 1)
        self.assertEqual(len(result['listing']['records']), 2)
        new_record = next(r for r in self.store.records.values() if r['id'] != self.record['id'])
        uploaded = Path(new_record['source'])
        self.assertEqual(uploaded.name, 'frames.csv')
        self.assertTrue(uploaded.is_relative_to(self.root / 'imports'))
        self.assertEqual(uploaded.read_bytes(), content)
        self.assertIsNone(self.store.annotation(new_record))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.path.read_bytes(), saved)
        reloaded = ReviewStore(self.workspace)
        self.assertEqual(len(reloaded.scan(self.root)['records']), 2)

    def test_import_zip_keeps_multiple_recordings_and_skips_repeated_content(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('一/frames.csv', fixture(31))
            archive.writestr('二/frames.csv', fixture(32))
            archive.writestr('metadata.json', '{}')
        status, result = self.upload('手机导出.zip', content.getvalue())
        self.assertEqual(status, 200, result)
        self.assertEqual(result['imported'], 2)
        new_record = next(r for r in self.store.records.values() if r['id'] != self.record['id'])
        self.assertEqual(Path(new_record['source'].split('::', 1)[0]).read_bytes(), content.getvalue())
        files_before = sorted(self.root.rglob('*'))
        status, result = self.upload('改名后.zip', content.getvalue())
        self.assertEqual(status, 200, result)
        self.assertEqual(result['imported'], 0)
        self.assertEqual(result['skipped'], 2)
        self.assertEqual(len(result['listing']['records']), 3)
        self.assertEqual(sorted(self.root.rglob('*')), files_before)

    def test_import_same_filename_different_content_does_not_overwrite(self):
        first_status, first = self.upload('frames.csv', fixture(31))
        self.assertEqual(first_status, 200, first)
        status, result = self.upload('frames.csv', fixture(32))
        self.assertEqual(status, 200, result)
        self.assertEqual(result['imported'], 1)
        self.assertEqual(len(result['listing']['records']), 3)
        self.assertEqual(sorted(p.read_bytes() for p in (self.root / 'imports').rglob('frames.csv')),
                         sorted([fixture(31), fixture(32)]))

    def test_import_rejects_invalid_files_without_changing_view_or_files(self):
        invalid_zip = io.BytesIO()
        with zipfile.ZipFile(invalid_zip, 'w') as archive:
            archive.writestr('good/frames.csv', fixture(31))
            archive.writestr('bad/frames.csv', b'missing columns\n')
        empty_zip = io.BytesIO()
        with zipfile.ZipFile(empty_zip, 'w') as archive:
            archive.writestr('notes.txt', 'no recordings')
        traversal_zip = io.BytesIO()
        with zipfile.ZipFile(traversal_zip, 'w') as archive:
            archive.writestr('../frames.csv', fixture(31))
        cases = [('bad.zip', b'not a zip'), ('bad.csv', b'no columns\n'),
                 ('overflow.csv', self.body.replace(b',501,', b',99999999999999999999999999999,', 1)),
                 ('partial.zip', invalid_zip.getvalue()), ('empty.zip', empty_zip.getvalue()),
                 ('traversal.zip', traversal_zip.getvalue()), ('script.py', self.body),
                 ('../escape.csv', fixture(31)), ('..\\escape.csv', fixture(31)), ('empty.csv', b'')]
        files_before = sorted(self.root.rglob('*'))
        listing_before = self.store.listing()
        for filename, content in cases:
            with self.subTest(filename=filename):
                status, result = self.upload(filename, content)
                self.assertEqual(status, 400, result)
                self.assertIn('error', result)
                self.assertEqual(self.store.listing(), listing_before)
                self.assertEqual(sorted(self.root.rglob('*')), files_before)

    def test_import_rejects_missing_token_and_oversized_body(self):
        status, _ = self.upload('frames.csv', fixture(31), token='wrong')
        self.assertEqual(status, 403)
        status, _ = self.upload('frames.csv', b'x', length=50 * 1024 * 1024 + 1)
        self.assertEqual(status, 413)
        self.assertEqual(len(self.store.records), 1)

    def test_import_limits_total_csv_size_before_writing(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('first/frames.csv', fixture(31))
            archive.writestr('second/frames.csv', fixture(32))
        with patch('server.MAX_CSV_BYTES', len(fixture(31)) * 2 - 1):
            status, result = self.upload('large.zip', archive_bytes.getvalue())
        self.assertEqual(status, 400, result)
        self.assertFalse((self.root / 'imports').exists())

    def test_import_rolls_back_when_rescan_fails(self):
        self.save(label='运动')
        original_listing = self.store.listing()
        saved = self.store.path.read_bytes()
        with patch.object(self.store, 'scan', side_effect=OSError('disk unavailable')):
            status, result = self.upload('new.csv', fixture(31))
        self.assertEqual(status, 400, result)
        self.assertEqual(self.store.listing(), original_listing)
        self.assertEqual(self.store.path.read_bytes(), saved)
        self.assertFalse((self.root / 'imports').exists())


if __name__ == '__main__':
    unittest.main()
