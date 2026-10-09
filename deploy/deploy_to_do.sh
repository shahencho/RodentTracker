#!/usr/bin/env bash
# Deploys the RodentTracker demo. Runs ON the shared droplet (do-deploy), inside ~/RodentTracker, as pm2 app
# "rodent-demo". Copied from Shoplifting faces/deploy/deploy_to_do.sh. Steady state, from the laptop:
#   git push origin master:main
#   ssh do-deploy "bash ~/RodentTracker/deploy/deploy_to_do.sh"
# First-time machine setup: deploy/setup_droplet.sh (see deploy/README.md).
#
# Editing this script is safe: the whole file is parsed before main runs, and after the pull the
# script re-runs its new copy, so there's no bootstrap step.
set -euo pipefail

BRANCH="${BRANCH:-main}"
PY=.venv/bin/python
HEALTH_URL="http://127.0.0.1:8013/login"

main() {
  cd "$(dirname "$0")/.."    # repo root

  if [ "${1:-}" != "--pulled" ]; then
    echo "==> Fetching origin/$BRANCH"
    git fetch origin "$BRANCH"
    # Tracked files only. Gitignored files (.env, data/, models/*.pt, outputs/, .venv/) are untouched.
    # Never `git clean` here: it would delete the demo videos, the model and the cached results.
    git checkout -q -f -B "$BRANCH" "origin/$BRANCH"
    echo "    at $(git log -1 --format='%h %s')"
    exec bash deploy/deploy_to_do.sh --pulled
  fi

  echo "==> Checking .env"
  if ! grep -Eq "^DEMO_PASSWORD=.+" .env 2>/dev/null; then
    echo "FATAL: DEMO_PASSWORD is missing in ~/RodentTracker/.env (required here: the demo is on the internet)."
    exit 1
  fi

  echo "==> Python venv"
  [ -x "$PY" ] || python3 -m venv .venv
  $PY -m pip install -q --upgrade pip
  # CPU builds of torch / torchvision first (the default Linux wheels bring ~3 GB of CUDA libraries)
  grep -E '^torch(vision)?==' requirements.txt > /tmp/rodent-demo-torch.txt
  $PY -m pip install -q -r /tmp/rodent-demo-torch.txt --index-url https://download.pytorch.org/whl/cpu
  $PY -m pip install -q -r requirements.txt

  echo "==> Demo data (not in git: deploy/README.md step 4)"
  $PY - <<'EOF'
import os, sys
from app.server import CFG, list_videos, raw_path
weights = CFG["detector"]["weights"]
problems = [f"missing {p}" for p in (weights, "outputs/validation/mouse_det_v2/per_video.csv",
                                      "outputs/validation/mouse_det_v2/engine_vs_reference.csv")
            if not os.path.exists(p)]
vids = list_videos()
cached = [v for v in vids if os.path.exists(raw_path(v))]
print(f"    {len(vids)} videos, {len(cached)} with cached results")
if problems or not vids:
    sys.exit("FATAL: " + "; ".join(problems or ["no videos in data/"]))
EOF

  echo "==> Memory before restart (shared box)"
  free -m | head -2

  echo "==> Restarting rodent-demo (pm2)"
  pm2 startOrRestart deploy/ecosystem.config.js --update-env
  pm2 save > /dev/null        # comes back after a reboot, like the other apps

  echo "==> Verifying health"
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
      echo "Deploy OK: the demo answered (attempt $attempt)."
      exit 0
    fi
    sleep 1
  done
  echo "Deploy FAILED: no answer on $HEALTH_URL within 30 s. Check: pm2 logs rodent-demo --lines 100"
  exit 1
}

main "$@"
exit
