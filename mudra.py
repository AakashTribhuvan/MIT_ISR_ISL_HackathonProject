"""Single entry point for Mudra's local tools."""

import subprocess
import sys
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

COMMANDS = {
    "dashboard": ["dashboard_server.py"],
    "extract": ["build_dataset.py"],
    "dataset": ["build_dataset.py"],
    "capture": ["capture_pose.py"],
    "recognize": ["live_recognize.py"],
    "dynamic": ["train_model.py"],
    "static": ["train_static_model.py"],
}


def run(command, args=None):
    if command not in COMMANDS:
        raise ValueError(f"Unknown Mudra command: {command}")
    return subprocess.run([PYTHON, *COMMANDS[command], *(args or [])], cwd=BASE_DIR).returncode


def prepare():
    steps = [
        ("Building dataset", "dataset"),
        ("Training dynamic model", "dynamic"),
        ("Training static model", "static"),
    ]
    for label, command in steps:
        print(f"\n=== {label} ===")
        if run(command) != 0:
            print(f"{label} failed.")
            return 1
    print("\nReady. Use `run.bat recognize` or the dashboard to test live.")
    return 0


def menu():
    options = [
        ("Open dashboard", "dashboard"),
        ("Extract one video", "extract"),
        ("Build dataset", "dataset"),
        ("Prepare and train both models", "prepare"),
        ("Capture static poses", "capture"),
        ("Run live recognition", "recognize"),
    ]
    print("Mudra tools\n-----------")
    for index, (label, _) in enumerate(options, 1):
        print(f"{index}. {label}")
    print("0. Exit")
    choice = input("\nSelect an action: ").strip()
    if choice == "0":
        return 0
    if not choice.isdigit() or not 1 <= int(choice) <= len(options):
        print("Please choose one of the listed numbers.")
        return 1
    command = options[int(choice) - 1][1]
    if command == "prepare":
        return prepare()
    if command == "dashboard":
        subprocess.Popen([PYTHON, *COMMANDS[command]], cwd=BASE_DIR)
        webbrowser.open("http://127.0.0.1:8000")
        return 0
    return run(command, sys.argv[2:])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "menu"
    if command == "menu":
        raise SystemExit(menu())
    if command == "prepare":
        raise SystemExit(prepare())
    raise SystemExit(run(command, sys.argv[2:]))
