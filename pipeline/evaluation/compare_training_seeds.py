#!/usr/bin/env python3
"""Create aligned training-curve and action-distribution tables for seeds 1-5."""

from __future__ import annotations

import csv
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
ACTIONS = [
    "Explain/ExplainNewFact", "Explain/RepeatFact", "Explain/ClarifyFact",
    "AskQuestion/AskOpinion", "AskQuestion/AskMemory", "AskQuestion/AskClarification",
    "OfferTransition/SummarizeAndSuggest", "Conclude/WrapUp",
]


def load_metrics(seed: int, runs: Path) -> Tuple[Path, Dict[str, Any]]:
    run = runs / f"seed_{seed}"
    preferred = run / "checkpoints" / "checkpoint_ep500_metrics.json"
    candidates = [preferred] if preferred.is_file() else []
    candidates.extend(sorted((run / "logs").glob("metrics_tracker_*.json")))
    best = None
    best_n = -1
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = len(data.get("episode_returns", []))
        except Exception:
            continue
        if n > best_n or (n == best_n and best and path.stat().st_mtime > best[0].stat().st_mtime):
            best = (path, data)
            best_n = n
    if best is None:
        raise FileNotFoundError(f"No metrics found for seed {seed} under {run}")
    return best


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else math.nan


def safe_sd(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def convergence_diagnostics(returns: Sequence[float]) -> Tuple[List[Dict[str, Any]], int | None]:
    """Conservative stable-tail diagnostic used for all five runs.

    Search starts at episode 375 (75% of training), uses all remaining returns,
    and requires at least 50 remaining episodes. This reproduces the adopted
    seed-1 boundary while allowing a later boundary when another seed is less stable.
    """
    values = np.asarray(returns, dtype=float)
    final_100_mean = float(values[-100:].mean())
    rows = []
    detected = None
    for episode in range(375, max(375, len(values) - 49) + 1):
        tail = values[episode - 1:]
        if len(tail) < 50:
            continue
        slope = float(np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0])
        tail_mean = float(tail.mean())
        cv = float(tail.std() / abs(tail_mean)) if tail_mean else math.inf
        c1 = abs(slope) < 0.02
        c2 = tail_mean >= 0.95 * final_100_mean
        c3 = cv < 0.20
        passes = c1 and c2 and c3
        rows.append({
            "candidate_episode": episode,
            "remaining_episodes": len(tail),
            "tail_slope_reward_per_episode": slope,
            "tail_mean": tail_mean,
            "final_100_mean": final_100_mean,
            "tail_to_final100_ratio": tail_mean / final_100_mean if final_100_mean else math.nan,
            "tail_cv": cv,
            "slope_pass": c1,
            "performance_pass": c2,
            "cv_pass": c3,
            "all_three_pass": passes,
        })
        if passes and detected is None:
            detected = episode
    return rows, detected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="cluster_runs")
    parser.add_argument("--output-dir", default="cluster_analysis")
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    out = Path(args.output_dir)
    if not runs.is_absolute():
        runs = HERE / runs
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    curve_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    convergence_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for seed in (1, 2, 3, 4, 5):
        try:
            source, metrics = load_metrics(seed, runs)
        except FileNotFoundError as exc:
            print(f"Skipping seed {seed}: {exc}")
            continue
        returns = [float(value) for value in metrics["episode_returns"]]
        lengths = [float(value) for value in metrics["episode_lengths"]]
        coverage = [float(value) for value in metrics["episode_coverage"]]
        usage = metrics["episode_option_usage"]
        diagnostics, convergence_episode = convergence_diagnostics(returns)
        for row in diagnostics:
            convergence_rows.append({"learner_seed": seed, **row})

        for index, reward in enumerate(returns, start=1):
            start = max(0, index - 50)
            curve_rows.append({
                "learner_seed": seed,
                "episode": index,
                "return": reward,
                "moving_average_50": safe_mean(returns[start:index]),
                "episode_length": lengths[index - 1],
                "coverage": coverage[index - 1],
            })

        for window_name, start in (("all_episodes", 0), ("final_100", max(0, len(usage) - 100)), ("episodes_375_500", 374)):
            selected = usage[start:]
            counts: Counter[str] = Counter()
            for episode_usage in selected:
                counts.update({key: int(value) for key, value in episode_usage.items()})
            total = sum(counts.values())
            for action in ACTIONS:
                action_rows.append({
                    "learner_seed": seed,
                    "window": window_name,
                    "action": action,
                    "count": counts[action],
                    "proportion": counts[action] / total if total else 0.0,
                    "total_turns": total,
                })

        summaries.append({
            "learner_seed": seed,
            "simulator_seed": 42,
            "metrics_source": str(source),
            "episodes": len(returns),
            "total_turns": int(sum(lengths)),
            "overall_return_mean": safe_mean(returns),
            "overall_return_sd": safe_sd(returns),
            "final_100_return_mean": safe_mean(returns[-100:]),
            "final_100_return_sd": safe_sd(returns[-100:]),
            "overall_length_mean": safe_mean(lengths),
            "final_100_length_mean": safe_mean(lengths[-100:]),
            "overall_coverage_mean": safe_mean(coverage),
            "final_100_coverage_mean": safe_mean(coverage[-100:]),
            "three_condition_episode": convergence_episode if convergence_episode else "",
        })

    write_csv(
        out / "learning_curves.csv", curve_rows,
        ("learner_seed", "episode", "return", "moving_average_50", "episode_length", "coverage"),
    )
    write_csv(
        out / "action_distributions.csv", action_rows,
        ("learner_seed", "window", "action", "count", "proportion", "total_turns"),
    )
    write_csv(
        out / "convergence_diagnostics.csv", convergence_rows,
        ("learner_seed", "candidate_episode", "remaining_episodes", "tail_slope_reward_per_episode",
         "tail_mean", "final_100_mean", "tail_to_final100_ratio", "tail_cv", "slope_pass",
         "performance_pass", "cv_pass", "all_three_pass"),
    )
    write_csv(
        out / "training_seed_summary.csv", summaries,
        ("learner_seed", "simulator_seed", "metrics_source", "episodes", "total_turns",
         "overall_return_mean", "overall_return_sd", "final_100_return_mean", "final_100_return_sd",
         "overall_length_mean", "final_100_length_mean", "overall_coverage_mean",
         "final_100_coverage_mean", "three_condition_episode"),
    )

    complete = [row for row in summaries if row["episodes"] == 500]
    aggregate = {
        "available_seeds": [row["learner_seed"] for row in summaries],
        "completed_seeds": [row["learner_seed"] for row in complete],
        "simulator_seed": 42,
        "final_100_return_across_seed_mean": safe_mean([row["final_100_return_mean"] for row in complete]),
        "final_100_return_across_seed_sd": safe_sd([row["final_100_return_mean"] for row in complete]),
        "final_100_length_across_seed_mean": safe_mean([row["final_100_length_mean"] for row in complete]),
        "final_100_coverage_across_seed_mean": safe_mean([row["final_100_coverage_mean"] for row in complete]),
        "convergence_episodes": {
            str(row["learner_seed"]): row["three_condition_episode"] for row in complete
        },
    }
    (out / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
