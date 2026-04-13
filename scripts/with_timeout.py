#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("with_timeout.py: missing command", file=sys.stderr)
        return 2
    if args.seconds <= 0:
        return subprocess.call(command)

    process = subprocess.Popen(command, start_new_session=True)
    deadline = time.monotonic() + args.seconds
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            return int(exit_code)
        if time.monotonic() >= deadline:
            print(
                f"nanocoop: timeout after {args.seconds:g}s; stopping run",
                file=sys.stderr,
            )
            _terminate_process_group(process)
            _write_timeout_status(args.seconds, command)
            return 124
        time.sleep(0.2)


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    timeout = time.monotonic() + 5.0
    while time.monotonic() < timeout:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _write_timeout_status(seconds: float, command: list[str]) -> None:
    record_dir = os.getenv("NANOCOOP_TIMEOUT_RECORD_DIR", "").strip()
    if not record_dir:
        return
    out = Path(record_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "timed_out": True,
        "timeout_seconds": int(seconds) if seconds.is_integer() else seconds,
        "timeout_mode": "dev_guard",
        "command": command,
    }
    (out / "timeout_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
