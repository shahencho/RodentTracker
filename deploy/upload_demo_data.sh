#!/usr/bin/env bash
# Copies the demo data listed in deploy/demo_files.txt to the droplet (videos, model, cached results; not in git).
# Run from the laptop, repo root, in Git Bash:  bash deploy/upload_demo_data.sh
# Adds or replaces those files only; anything else on the droplet (uploads, results made there) stays.
set -euo pipefail
cd "$(dirname "$0")/.."
files=$(grep -vE '^(#|$)' deploy/demo_files.txt)
for f in $files; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
du -ch $files | tail -1
tar -cf - $files | ssh do-deploy "tar -xf - -C RodentTracker && echo 'uploaded to ~/RodentTracker'"
