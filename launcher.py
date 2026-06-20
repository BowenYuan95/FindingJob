#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# launcher.py - wrap jobfinder into a desktop app window (no browser tab).
#   1. start Streamlit silently in background (headless, no browser)
#   2. start backfill in background thread (60s delay)
#   3. open a native desktop window pointing to the dashboard
# deps:  py -m pip install pywebview streamlit pandas requests

import os
import sys
import time
import socket
import logging
import threading
import subprocess

import requests
import webview

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8501
URL  = f"http://localhost:{PORT}"


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_streamlit() -> subprocess.Popen | None:
    if _port_open(PORT):
        return None
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=HERE, creationflags=flags)


def start_backfill() -> None:
    def _run() -> None:
        time.sleep(60)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            subprocess.Popen([sys.executable, "backfill_scores.py", "--watch"],
                             cwd=HERE, creationflags=flags)
        except Exception as e:
            logger.warning(f"[launcher] backfill failed: {e}")
    threading.Thread(target=_run, daemon=True).start()


def wait_until_ready(timeout: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    st_proc = start_streamlit()

    if not wait_until_ready():
        import webbrowser
        webbrowser.open(URL)
        return

    webview.create_window("JobFinder", URL, width=1200, height=850)
    webview.start()

    if st_proc:
        try:
            st_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()