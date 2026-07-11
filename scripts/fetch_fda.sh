#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/fda/manifest.json"

if [ ! -f "$MANIFEST" ]; then
  echo "Error: manifest.json not found at $MANIFEST"
  exit 1
fi

python3 -c "
import json
import os
import subprocess
import sys

with open('$MANIFEST') as f:
  docs = json.load(f)

for doc in docs:
  filename = doc['filename']
  url = doc['url']
  filepath = os.path.join('$REPO_ROOT/fda', filename)

  # Create parent directory if needed
  os.makedirs(os.path.dirname(filepath), exist_ok=True)

  # Skip if file exists
  if os.path.exists(filepath):
    print(f'✓ {filename}')
    continue

  print(f'⬇ {filename}', file=sys.stderr)
  result = subprocess.run(['curl', '-sS', '-L', url, '-o', filepath],
                         capture_output=True)

  if result.returncode != 0:
    print(f'✗ Download failed for {filename}', file=sys.stderr)
    sys.exit(1)

  # Verify PDF header
  with open(filepath, 'rb') as f:
    header = f.read(4)
    if header != b'%PDF':
      print(f'✗ Invalid PDF header for {filename}', file=sys.stderr)
      os.remove(filepath)
      sys.exit(1)

  print(f'✓ {filename}')
" || exit 1
