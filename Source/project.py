"""Project commands and dataset path preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def prepare_data(root):
    root = Path(root).resolve()
    path = root / 'NN' / 'review' / 'annotations.json'
    if not path.exists():
        return 0
    document = json.loads(path.read_text(encoding='utf-8'))
    records = {}
    data = (root / 'NN' / 'Data').resolve()
    for record in document['records'].values():
        source, separator, member = record['source'].partition('::')
        normalized = source.replace('\\', '/')
        marker = normalized.casefold().find('nn/data/')
        if marker < 0:
            raise ValueError('Annotation source is outside NN/Data')
        relative = PurePosixPath(normalized[marker + len('nn/data/'):])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe annotation source')
        target = data.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(data) or not target.is_file():
            raise ValueError('Annotation source does not exist inside NN/Data')
        source = str(target) + (separator + member if separator else '')
        key = hashlib.sha256(source.encode('utf-8')).hexdigest()[:24]
        if key in records:
            raise ValueError('Duplicate annotation source')
        records[key] = {**record, 'source': source}
    updated = {**document, 'records': records}
    if updated != document:
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description='FallingDetect project commands')
    parser.add_argument('command', choices=['prepare', 'train', 'verify', 'quantize', 'review', 'build', 'test'])
    args, remaining = parser.parse_known_args()
    prepare_data(ROOT)
    scripts = {'train': 'NN/train_physics_fall.py', 'verify': 'NN/verify_physics_fall.py',
               'quantize': 'NN/quantize_physics_fall.py', 'review': 'NN/dataset_review/server.py',
               'build': 'NN/run_idf.py'}
    if args.command == 'prepare':
        return 0
    if args.command == 'test':
        for directory, pattern in [('NN', 'test_*.py'), ('NN/dataset_review', 'test_server.py')]:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', directory,
                            '-p', pattern, '-v'], cwd=ROOT, check=True)
        return 0
    forwarded = (['build'] if args.command == 'build' else []) + remaining
    return subprocess.call([sys.executable, str(ROOT / scripts[args.command]), *forwarded], cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
