"""Flash the ESP32-S3 release using esptool."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Flash the ESP32-S3 firmware')
    parser.add_argument('--port', required=True, help='Serial port, for example COM3 or /dev/ttyACM0')
    parser.add_argument('--baud', type=int, default=460800)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'flash.json').read_text(encoding='utf-8'))
    segments = []
    for segment in config['segments']:
        path = root / segment['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != segment['sha256']:
            parser.error('Firmware checksum mismatch: ' + segment['file'])
        segments.extend([segment['offset'], str(path)])
    command = [sys.executable, '-m', 'esptool', '--chip', config['chip'],
               '--port', args.port, '--baud', str(args.baud),
               '--before', 'default-reset', '--after', 'hard-reset',
               'write-flash', '--flash-mode', config['flash_mode'],
               '--flash-size', config['flash_size'], '--flash-freq', config['flash_freq'], *segments]
    return subprocess.call(command, cwd=root)


if __name__ == '__main__':
    sys.exit(main())
