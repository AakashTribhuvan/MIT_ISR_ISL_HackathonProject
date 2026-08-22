"""Local control server for the Mudra dashboard."""

import json
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
PORT = 8000

ACTIONS = {
    "extract": ["build_dataset.py"],
    "dataset": ["build_dataset.py"],
    "train": ["train_model.py"],
    "train-static": ["train_static_model.py"],
    "capture": ["capture_pose.py"],
    "recognize": ["live_recognize.py"],
}

process_lock = threading.Lock()
active_process = None
process_info = {
    "action": None,
    "status": "idle",
    "pid": None,
    "started_at": None,
    "last_output": [],
    "returncode": None,
}


def read_output(process, action):
    global active_process
    for line in iter(process.stdout.readline, ""):
        text = line.rstrip()
        if text:
            with process_lock:
                process_info["last_output"].append(text)
                process_info["last_output"] = process_info["last_output"][-80:]
    returncode = process.wait()
    with process_lock:
        process_info["status"] = "done" if returncode == 0 else "error"
        process_info["returncode"] = returncode
        process_info["pid"] = None
        active_process = None


def start_action(action, extra_args=None):
    global active_process, process_info
    if action not in ACTIONS:
        return False, f"Unknown action: {action}"

    with process_lock:
        if active_process is not None and active_process.poll() is None:
            return False, f"{process_info['action']} is already running"

        command = [sys.executable, *ACTIONS[action]]
        if extra_args:
            command.extend(extra_args)
        active_process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        process_info = {
            "action": action,
            "status": "running",
            "pid": active_process.pid,
            "started_at": __import__("time").time(),
            "last_output": ["$ " + " ".join(command)],
            "returncode": None,
        }
        threading.Thread(target=read_output, args=(active_process, action), daemon=True).start()
    return True, process_info


def stop_action():
    with process_lock:
        if active_process is None or active_process.poll() is not None:
            return False, "No action is running"
        active_process.terminate()
        return True, "Stop requested"


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            with process_lock:
                payload = dict(process_info)
                payload["last_output"] = list(process_info["last_output"])
            extraction_status = DASHBOARD_DIR / "status.json"
            if extraction_status.exists():
                try:
                    payload.update(json.loads(extraction_status.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    pass
            self.send_json(payload)
            return
        if parsed.path == "/api/files":
            files_path = DASHBOARD_DIR / "files.json"
            if files_path.exists():
                self.send_json(json.loads(files_path.read_text(encoding="utf-8")))
            else:
                self.send_json({"videos": [], "extracted": []})
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_json({"error": "Not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode("utf-8"))
        action = payload.get("action")
        extra_args = payload.get("args", [])
        if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
            self.send_json({"error": "Arguments must be strings"}, 400)
            return
        if action == "stop":
            ok, result = stop_action()
        else:
            ok, result = start_action(action, extra_args)
        self.send_json({"ok": ok, "result": result}, 200 if ok else 409)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), DashboardHandler)
    print(f"Mudra dashboard: http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
