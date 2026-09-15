#!/usr/bin/env python3

import argparse
import fcntl
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
ETA_RE = re.compile(r"eta:\s*(\d+):(\d+):(\d+)")
RUN_RE = re.compile(r"^\[run\]\s+(.+\.py)\s*$")
VAL_RE = re.compile(r"Epoch\(val\)|Starting Evaluation|^\s*\d+%\|", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-frequency ablation queue monitor.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--queue-script",
        type=Path,
        default=Path("projects/InfraOcc/configs/ablation/run_primary_ablations.sh"),
    )
    parser.add_argument(
        "--queue-log",
        type=Path,
        default=Path("logs/infraocc_ablation/core_chain_primary_queue.log"),
    )
    parser.add_argument("--log-root", type=Path, default=Path("logs/infraocc_ablation"))
    parser.add_argument("--min-sleep", type=int, default=600, help="Minimum poll interval in seconds.")
    parser.add_argument("--max-sleep", type=int, default=1800, help="Maximum poll interval in seconds.")
    parser.add_argument("--eta-buffer", type=int, default=300, help="Extra seconds beyond ETA.")
    parser.add_argument(
        "--fallback-sleep",
        type=int,
        default=900,
        help="Sleep interval when ETA cannot be parsed.",
    )
    parser.add_argument(
        "--process-pattern",
        default="projects/InfraOcc/configs/ablation/run_primary_ablations.sh",
    )
    return parser.parse_args()


def now_str() -> str:
    return datetime.now().strftime("%F %T")


def read_last_run(queue_log: Path) -> Optional[str]:
    if not queue_log.exists():
        return None
    last_run = None
    with queue_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = RUN_RE.match(line.strip())
            if match:
                last_run = match.group(1)
    return last_run


def current_experiment_log(log_root: Path, queue_log: Path) -> Optional[Path]:
    last_run = read_last_run(queue_log)
    if not last_run:
        return None
    run_name = Path(last_run).stem
    log_path = log_root / f"{run_name}.log"
    return log_path if log_path.exists() else None


def parse_log_state(log_path: Path) -> Tuple[Optional[Tuple[datetime, int]], bool]:
    latest = None
    in_validation = False
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts_match = TIMESTAMP_RE.match(line)
            eta_match = ETA_RE.search(line)
            if not ts_match or not eta_match:
                if VAL_RE.search(line):
                    in_validation = True
                continue
            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            h, m, s = (int(eta_match.group(i)) for i in range(1, 4))
            latest = (ts, h * 3600 + m * 60 + s)
            in_validation = False
    return latest, in_validation


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
        if pid == current_pid or "watch_ablation_queue.py" in cmdline:
            continue
        if process_pattern in cmdline:
            return True
    return False


def launch_queue(args: argparse.Namespace) -> None:
    queue_script = (args.repo_root / args.queue_script).resolve()
    queue_log = (args.repo_root / args.queue_log).resolve()
    queue_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    cmd = (
        f"cd {shlex.quote(str(args.repo_root))} && "
        f"env CUDA_VISIBLE_DEVICES={shlex.quote(env.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))} "
        f"GPUS={shlex.quote(env.get('GPUS', '4'))} "
        f"TORCHRUN_BIN={shlex.quote(env.get('TORCHRUN_BIN', '/home/bxk/.conda/envs/4DR360/bin/torchrun'))} "
        f"CONFIG_FILTER_REGEX={shlex.quote(env.get('CONFIG_FILTER_REGEX', ''))} "
        f"bash -x {shlex.quote(str(queue_script))} >> {shlex.quote(str(queue_log))} 2>&1"
    )
    subprocess.Popen(
        ["setsid", "bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def choose_sleep_seconds(
    args: argparse.Namespace, eta_seconds: Optional[int], in_validation: bool
) -> int:
    if in_validation:
        return min(180, args.min_sleep)
    if eta_seconds is None:
        return args.fallback_sleep
    if eta_seconds + args.eta_buffer <= args.max_sleep:
        return max(args.min_sleep, eta_seconds + args.eta_buffer)
    return args.max_sleep


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[{now_str()}] monitor already running: {lock_path}", flush=True)
        sys.exit(0)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    lock_file = acquire_lock(args.repo_root / "logs/infraocc_ablation/.queue_monitor.lock")

    queue_log = (args.repo_root / args.queue_log).resolve()
    log_root = (args.repo_root / args.log_root).resolve()

    while True:
        running = queue_running(args.process_pattern)
        if not running:
            print(f"[{now_str()}] queue not running, relaunching", flush=True)
            launch_queue(args)
            time.sleep(args.min_sleep)
            continue

        exp_log = current_experiment_log(log_root, queue_log)
        eta_seconds = None
        in_validation = False
        if exp_log and exp_log.exists():
            latest, in_validation = parse_log_state(exp_log)
            if latest is not None:
                _, eta_seconds = latest

        sleep_seconds = choose_sleep_seconds(args, eta_seconds, in_validation)
        eta_text = "unknown" if eta_seconds is None else f"{eta_seconds}s"
        phase = "validation" if in_validation else "train"
        target = datetime.now().timestamp() + sleep_seconds
        target_text = datetime.fromtimestamp(target).strftime("%F %T")
        print(
            f"[{now_str()}] queue healthy, phase={phase}, eta={eta_text}, next check at {target_text}",
            flush=True,
        )
        time.sleep(sleep_seconds)

    lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
