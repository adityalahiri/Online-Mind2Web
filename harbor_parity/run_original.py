#!/usr/bin/env python3
"""Original-side parity runner for the Harbor Online-Mind2Web adapter.

Runs Browser Use directly on the host (no Harbor, no container) for each task, writing
upstream v2 submissions to <output-dir>/<task_id>/. Score them afterwards with the
upstream evaluator:

    python src/run.py --mode WebJudge_Online_Mind2Web_eval --model gpt-6-luna \
        --trajectories_dir <output-dir> --api_key "$OPENAI_API_KEY" \
        --output_path <output-dir>_result --score_threshold 3 --num_worker 1

The agent runner (`run_browser_use.py`) and task selection are imported from the Harbor
adapter, so prompts, Browser Use settings and the trajectory format are identical on
both sides. Only the execution environment differs (host Chrome vs. container Chromium).

Usage:
    uv run --python 3.12 --with browser-use==0.13.14 --with huggingface_hub \
        python harbor_parity/run_original.py \
        --adapter-src ~/harbor-adapters/src/online-mind2web/src \
        --dataset-path Online_Mind2Web.json --split parity --output-dir runs/original_run1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-src", required=True, type=Path,
                    help="Path to the adapter's src/ dir (contains online_mind2web/)")
    ap.add_argument("--dataset-path", type=Path, default=None)
    ap.add_argument("--split", choices=["full", "parity"], default="parity")
    ap.add_argument("--parity-size", type=int, default=100)
    ap.add_argument("--task-ids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--n-concurrent", type=int, default=3)
    ap.add_argument("--timeout-sec", type=int, default=2400, help="per task, matches Harbor agent timeout")
    ap.add_argument("--chrome-path", default=os.environ.get(
        "CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.adapter_src.resolve()))
    from online_mind2web.adapter import OnlineMind2WebAdapter  # noqa: E402

    runner = args.adapter_src.resolve() / "online_mind2web" / "run_browser_use.py"
    adapter = OnlineMind2WebAdapter(
        output_dir=Path(tempfile.mkdtemp()), dataset_path=args.dataset_path, split=args.split,
        parity_size=args.parity_size, task_ids=args.task_ids, limit=args.limit,
    )
    tasks = adapter.select_tasks()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir.parent / f"{args.output_dir.name}_logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "ANONYMIZED_TELEMETRY": "false", "BROWSER_USE_CLOUD_SYNC": "false"}

    def run_one(task: dict) -> tuple[str, int | str]:
        if (args.output_dir / task["task_id"] / "result.json").exists():
            return task["task_id"], "skipped (exists)"
        task_json = logs / f"{task['task_id']}.task.json"
        task_json.write_text(json.dumps(task))
        cmd = [sys.executable, str(runner), "--task-json", str(task_json),
               "--output-dir", str(args.output_dir), "--model", args.model,
               "--chrome-path", args.chrome_path]
        with open(logs / f"{task['task_id']}.log", "w") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    timeout=args.timeout_sec).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        return task["task_id"], rc

    with ThreadPoolExecutor(args.n_concurrent) as pool:
        for task_id, rc in pool.map(run_one, tasks):
            print(f"{task_id}: {rc}", flush=True)


if __name__ == "__main__":
    main()
