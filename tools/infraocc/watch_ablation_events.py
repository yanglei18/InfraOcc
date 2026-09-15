#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


RUN_RE = re.compile(r"^\[run\]\s+(.+\.py)\s*$")
EXIT_RE = re.compile(r"EXIT_CODE:(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit ablation queue milestone events.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--queue-log",
        type=Path,
        default=Path("logs/infraocc_ablation/core_chain_primary_queue.log"),
    )
    parser.add_argument(
        "--status-root",
        type=Path,
        default=Path("work_dirs/infraocc_ablation/status"),
    )
    parser.add_argument(
        "--events-root",
        type=Path,
        default=Path("logs/infraocc_ablation/events"),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--process-pattern",
        default="projects/InfraOcc/configs/ablation/run_primary_ablations.sh",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def queue_running(process_pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", process_pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1] if len(parts) > 1 else ""
        if pid == current_pid or "watch_ablation_events.py" in cmdline:
            continue
        if process_pattern in cmdline:
            return True
    return False


def latest_run(queue_log: Path) -> Optional[str]:
    if not queue_log.exists():
        return None
    last = None
    with queue_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = RUN_RE.match(line.strip())
            if match:
                last = match.group(1)
    return last


def latest_exit_code(queue_log: Path) -> Optional[int]:
    if not queue_log.exists():
        return None
    last = None
    with queue_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = EXIT_RE.search(line)
            if match:
                last = int(match.group(1))
    return last


def status_json_for_run(status_root: Path, queue_log: Path) -> Optional[Path]:
    current_run = latest_run(queue_log)
    if not current_run:
        return None
    candidate = status_root / f"{Path(current_run).stem}.json"
    return candidate if candidate.exists() else None


def append_event(events_root: Path, event: dict) -> None:
    events_root.mkdir(parents=True, exist_ok=True)
    with (events_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    with (events_root / "latest_event.json").open("w", encoding="utf-8") as handle:
        json.dump(event, handle, indent=2, ensure_ascii=False)
    with (events_root / "latest_event.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{event['time']} {event['type']}\n")
        for key, value in event.items():
            if key in {"time", "type"}:
                continue
            handle.write(f"{key}: {value}\n")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    queue_log = (repo_root / args.queue_log).resolve()
    status_root = (repo_root / args.status_root).resolve()
    events_root = (repo_root / args.events_root).resolve()

    state_path = events_root / ".watch_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        state = {
            "last_run": None,
            "last_status_json": None,
            "queue_running": None,
            "last_exit_code": None,
        }

    while True:
        current_run = latest_run(queue_log)
        current_status = status_json_for_run(status_root, queue_log)
        current_status_str = str(current_status) if current_status else None
        current_running = queue_running(args.process_pattern)
        current_exit_code = latest_exit_code(queue_log)

        if current_run and current_run != state["last_run"]:
            append_event(events_root, {
                "time": now_iso(),
                "type": "next_config_started",
                "config_rel": current_run,
            })
            state["last_run"] = current_run

        if current_status_str and current_status_str != state["last_status_json"]:
            with open(current_status_str, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            append_event(events_root, {
                "time": now_iso(),
                "type": "config_finished_and_backfilled",
                "status_json": current_status_str,
                "metrics": {
                    "gIoU": metrics.get("gIoU"),
                    "mIoU": metrics.get("mIoU"),
                    "mean_dynamic": metrics.get("mean_dynamic"),
                    "mean_static": metrics.get("mean_static"),
                },
            })
            state["last_status_json"] = current_status_str

        if state["queue_running"] is True and not current_running:
            append_event(events_root, {
                "time": now_iso(),
                "type": "queue_stopped",
                "exit_code": current_exit_code,
            })
        state["queue_running"] = current_running
        state["last_exit_code"] = current_exit_code

        events_root.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
