# RodentTracker demo on the DigitalOcean droplet

Runs on the shared droplet (`ssh do-deploy`) next to the fire, theft and faces demos, the same way: root, pm2,
nginx + certbot. Demo: **https://rodent.alarmius.duckdns.org**

```
laptop ──git push──► GitHub (shahencho/RodentTracker, main) ◄──git fetch── droplet ~/RodentTracker
                                                                           ├─ pm2: rodent-demo (uvicorn app.server, 127.0.0.1:8013)
browser ──HTTPS──► nginx :443 (certbot) ───────────────────────────────────┘   uploads up to 200 MB
laptop ──tar over ssh──► demo videos, model, cached results (not in git; deploy/demo_files.txt)
```

- Same app as `run_ui.bat`, plus a login (`DEMO_USER` / `DEMO_PASSWORD` in `~/RodentTracker/.env`; without a
  password, as on the laptop, there is no login).
- The demo videos open with results: their setups and detections are copied up from the laptop. The detector
  (YOLO11n, CPU) runs online only for an uploaded video, or when the arena outline is moved so far that the cached
  detections no longer fit. One run at a time. It is slow on CPU: ~6-7 frames/s on the laptop, likely less on the
  droplet's 2 shared CPUs, so a 5-minute video (7,500 frames) takes roughly 20-40 minutes and slows the other demos.
- **Shared box: check memory first.** Idle ~150 MB (the detector is loaded only for a run); ~700 MB during a run
  (measured on the laptop; pm2 cap 1000 MB). Run `ssh do-deploy "free -m; pm2 ls"` before the first deploy.

| File | What |
|---|---|
| `setup_droplet.sh` | once: packages, clone, nginx site + certificate, daily clean-up of uploads. Additive only |
| `deploy_to_do.sh` | every deploy: pull, venv (CPU torch), demo data check, pm2 restart, health check |
| `ecosystem.config.js` | pm2 app `rodent-demo` (port 8013, 1000 MB cap) |
| `nginx-rodent-demo.conf` | nginx site template (domain filled in by setup; 200 MB uploads) |
| `demo_files.txt`, `upload_demo_data.sh` | the demo data (~200 MB) and the laptop script that copies it up |

## 1. One-time setup

1. Push to GitHub (the laptop's `master` is the droplet's `main`):
   ```bash
   git remote add origin https://github.com/shahencho/RodentTracker.git
   git push -u origin master:main
   ```
2. Setup (from the laptop, repo root):
   ```bash
   ssh do-deploy "RODENT_DOMAIN=rodent.alarmius.duckdns.org bash -s" < deploy/setup_droplet.sh
   ```
3. Login: `~/RodentTracker/.env` (chmod 600) with `DEMO_USER` and `DEMO_PASSWORD` (strong: it's on the internet).
   After changing it: `pm2 restart rodent-demo --update-env`.
   ```bash
   ssh do-deploy "cd RodentTracker && printf 'DEMO_USER=demo\nDEMO_PASSWORD=...\n' > .env && chmod 600 .env"
   ```
4. Demo data (Git Bash, repo root): `bash deploy/upload_demo_data.sh`
5. First deploy: `ssh do-deploy "bash ~/RodentTracker/deploy/deploy_to_do.sh"` (the first one installs torch: a few
   minutes).
6. Open the demo → log in → Setup: the videos marked ✓ open with results; Validation: the 14-video comparison.

## Every deploy after that

```bash
git push origin master:main
ssh do-deploy "bash ~/RodentTracker/deploy/deploy_to_do.sh"
```

Not touched by deploys: `.env`, `data/` (videos, uploads), `models/*.pt`, `outputs/` (setups, cached detections).
To change the demo videos: edit `demo_files.txt` (a video plus its `outputs/ui/<video>_<hash>` folder) and run
`upload_demo_data.sh` again.

| Task | Command |
|---|---|
| Live log | `ssh do-deploy "pm2 logs rodent-demo"` |
| Restart / stop | `ssh do-deploy "pm2 restart rodent-demo"` / `pm2 stop rodent-demo` |
| Memory | `ssh do-deploy "free -m; pm2 ls"` |
