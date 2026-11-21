"""Run ESP-IDF in the active ESP-IDF environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
idf_path = os.environ.get('IDF_PATH')
entry = str(Path(idf_path) / 'tools' / 'idf.py') if idf_path else shutil.which('idf.py')
if not entry or not Path(entry).is_file():
    sys.exit('Activate the ESP-IDF 6.1 environment before building.')
sys.exit(subprocess.call([sys.executable, entry, *sys.argv[1:]], cwd=root))
