#!/usr/bin/env python3
"""Success rate for one parity run, on a fixed task list (missing trajectories count as 0).

Upstream src/run.py divides by the number of trajectory folders present, so a task whose
agent crashed or timed out without writing a trajectory silently leaves the denominator.
Harbor scores such tasks 0. This script makes the original side use the same denominator.

    python harbor_parity/score.py --results <output-dir>_result/WebJudge_Online_Mind2Web_eval_gpt-6-luna_score_threshold_3_auto_eval_results.json \
        --task-list-dir <output-dir>_logs   # *.task.json written by run_original.py
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path, help="upstream auto-eval JSONL")
    ap.add_argument("--task-list-dir", required=True, type=Path)
    args = ap.parse_args()

    task_ids = sorted(json.loads(p.read_text())["task_id"] for p in args.task_list_dir.glob("*.task.json"))
    labels = {}
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["task_id"]] = int(row["predicted_label"])
    per_task = {t: labels.get(t, 0) for t in task_ids}
    missing = [t for t in task_ids if t not in labels]
    rate = 100.0 * sum(per_task.values()) / len(task_ids) if task_ids else 0.0
    print(json.dumps({"n_tasks": len(task_ids), "n_success": sum(per_task.values()),
                      "success_rate": round(rate, 2), "missing_counted_as_0": missing,
                      "per_task": per_task}, indent=2))


if __name__ == "__main__":
    main()
