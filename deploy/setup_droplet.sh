#!/usr/bin/env bash
# One-time setup of the RodentTracker demo on the SHARED droplet (do-deploy, next to fire, theft and faces demos).
# Safe to re-run. Run from the laptop (repo root):
#   ssh do-deploy "RODENT_DOMAIN=rodent.alarmius.duckdns.org bash -s" < deploy/setup_droplet.sh
# (DuckDNS resolves every name under alarmius.duckdns.org to the droplet: nothing to add at duckdns.org.)
#
# Additive only: its own clone ~/RodentTracker, one nginx site + its certbot certificate and a daily clean-up of
# uploaded videos. Does not create .env or start the app: see deploy/README.md.
# Copied from Shoplifting faces/deploy/setup_droplet.sh.
set -euo pipefail

main() {
  REPO_URL="${REPO_URL:-https://github.com/shahencho/RodentTracker.git}"
  BRANCH="${BRANCH:-main}"
  APP_DIR="$HOME/RodentTracker"
  RODENT_DOMAIN="${RODENT_DOMAIN:?set RODENT_DOMAIN, e.g. RODENT_DOMAIN=rodent.alarmius.duckdns.org}"
  SITE=/etc/nginx/sites-available/rodent-demo

  echo "==> Packages (only what is missing)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -yq --no-upgrade python3-venv libglib2.0-0 libgl1     # libgl1, libglib: OpenCV

  echo "==> Repo $APP_DIR ($BRANCH)"
  if [ ! -d "$APP_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  fi

  echo "==> nginx site $RODENT_DOMAIN -> 127.0.0.1:8013"
  if [ ! -f "$SITE" ]; then      # certbot edits the site later; never overwrite it on a re-run
    sed "s/RODENT_DOMAIN/$RODENT_DOMAIN/" "$APP_DIR/deploy/nginx-rodent-demo.conf" > "$SITE"
    ln -sf "$SITE" /etc/nginx/sites-enabled/rodent-demo
  fi
  nginx -t
  systemctl reload nginx

  echo "==> HTTPS certificate (certbot, like the other sites)"
  if certbot certificates 2>/dev/null | grep -q "Domains: $RODENT_DOMAIN"; then
    echo "    already have one"
  else
    my_ip=$(curl -fsS --max-time 5 https://api.ipify.org)
    dns_ip=$(getent hosts "$RODENT_DOMAIN" | awk '{print $1}' | head -1)
    if [ "$dns_ip" != "$my_ip" ]; then
      echo "!! $RODENT_DOMAIN resolves to '${dns_ip:-nothing}', not $my_ip. Fix it at duckdns.org, then re-run."
      exit 2
    fi
    certbot --nginx -d "$RODENT_DOMAIN" --non-interactive --redirect
  fi

  echo "==> Daily clean-up: uploaded videos older than 7 days"
  cat > /etc/cron.daily/rodent-demo-cleanup <<EOF
#!/bin/sh
find $APP_DIR/data/uploads -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {} + 2>/dev/null
exit 0
EOF
  chmod 755 /etc/cron.daily/rodent-demo-cleanup

  echo
  echo "Setup done. Demo (after the first deploy): https://$RODENT_DOMAIN"
  echo "Next: create $APP_DIR/.env, copy the demo data up, then: bash $APP_DIR/deploy/deploy_to_do.sh"
}

main "$@"
exit
