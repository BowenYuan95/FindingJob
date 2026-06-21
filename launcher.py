#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# launcher.py - wrap jobfinder into a desktop app window (no browser tab).
#   1. start Streamlit silently in background (headless, no browser)
#   2. start backfill in background thread (60s delay)
#   3. open a native desktop window pointing to the dashboard
#   4. on window close: terminate backfill (if still running) + unload LLM models
# deps:  py -m pip install pywebview streamlit pandas requests

import os
import sys
import time
import socket
import logging
import threading
import subprocess
import webbrowser

import requests
import webview


class _Api:
    """Python API exposed to JavaScript inside the webview."""
    def open_url(self, url: str) -> None:
        webbrowser.open(url)

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8501
URL  = f"http://localhost:{PORT}"

_backfill_proc: list[subprocess.Popen] = []   # mutable holder; set by background thread


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
            proc = subprocess.Popen(
                [sys.executable, "-m", "pipeline.backfill_scores"],  # no --watch: exits + unloads when done
                cwd=HERE, creationflags=flags)
            _backfill_proc.append(proc)
        except Exception as e:
            logger.warning(f"[launcher] backfill failed: {e}")
    threading.Thread(target=_run, daemon=True).start()


def unload_models() -> None:
    """Unload LLM models from LM Studio to free VRAM/RAM."""
    for m in ["qwen/qwen3.5-9b", "text-embedding-nomic-embed-text-v1.5"]:
        try:
            subprocess.run(
                ["lms", "unload", m],
                capture_output=True, timeout=30,
                shell=(os.name == "nt"),
            )
            logger.info(f"[launcher] 已卸载 {m}")
        except Exception as e:
            logger.warning(f"[launcher] 卸载 {m} 失败: {e}")


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
        webbrowser.open(URL)
        return

    start_backfill()

    _JS = """(function(){
        document.addEventListener('click', function(e){
            var a = e.target.closest('a');
            if (a && a.href && !a.href.includes('localhost')) {
                e.preventDefault(); e.stopPropagation();
                if (window.pywebview && window.pywebview.api)
                    window.pywebview.api.open_url(a.href);
            }
        }, true);
    })();"""

    window = webview.create_window("JobFinder", URL, width=1200, height=850, js_api=_Api())
    window.events.loaded += lambda: window.evaluate_js(_JS)
    webview.start()

    # Window closed — clean up regardless of whether backfill finished
    if _backfill_proc:
        try:
            _backfill_proc[0].terminate()
            logger.info("[launcher] backfill 进程已终止")
        except Exception:
            pass
    unload_models()

    if st_proc:
        try:
            st_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
