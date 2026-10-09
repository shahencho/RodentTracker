// pm2 process for the RodentTracker demo on the shared droplet (next to fire-demo, theft-demo, faces-demo & co).
// Started / restarted by deploy/deploy_to_do.sh. Logs: pm2 logs rodent-demo
const path = require("path");

module.exports = {
  apps: [{
    name: "rodent-demo",
    cwd: path.resolve(__dirname, ".."),
    script: ".venv/bin/python",
    args: "-m uvicorn app.server:app --host 127.0.0.1 --port 8013",
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1",
      MPLBACKEND: "Agg",
    },
    // idle ~150 MB (the detector is loaded only for a run); ~700 MB during a detector run, one run at a time.
    // Shared 4 GB box: check `free -m` and `pm2 ls` first (deploy/README.md)
    max_memory_restart: "1000M",
    restart_delay: 5000,
  }],
};
