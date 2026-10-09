@echo off
rem Start the RodentTracker demo UI on this PC and open it in the browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8765/
.venv\Scripts\python -m uvicorn app.server:app --host 127.0.0.1 --port 8765
