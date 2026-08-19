"""
Export training metrics required for thesis analysis.

Usage:
    python src/utils/export_training_metrics.py --experiment-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def action_name_from_turn(turn: Dict[str, Any]) -> str:
    action = turn.get("action", {})
    if action.get("flat_action_name"):
        return action["flat_action_name"]

    option_name = action.get("option_name") or turn.get("context", {}).get("option")
    subaction_name = action.get("subaction_name") or turn.get("context", {}).get("subaction")
    if option_name and subaction_name:
        return f"{option_name}/{subaction_name}"
    if option_name:
        return str(option_name)
    if subaction_name:
        return str(subaction_name)
    return "Unknown"


def load_episode_logs(detailed_logs_dir: Path) -> List[Tuple[int, Dict[str, Any]]]:
    episodes: List[Tuple[int, Dict[str, Any]]] = []
    for ep_dir in sorted(detailed_logs_dir.glob("episode_*")):
        log_file = ep_dir / "episode_log.json"
        if not log_file.exists():
            continue
        data = read_json(log_file)
        episode_id = int(data.get("episode_number", ep_dir.name.split("_")[-1]))
        episodes.append((episode_id, data))
    return episodes


def load_periodic_snapshots(analytics_dir: Path) -> Dict[int, Dict[str, Any]]:
    aggregate_file = analytics_dir / "periodic_snapshots.json"
    if not aggregate_file.exists():
        return {}
    try:
        snapshots = read_json(aggregate_file)
    except Exception:
        return {}
    return {int(item.get("episode")): item for item in snapshots if "episode" in item}


def safe_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_var(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.var(arr))


def compute_discounted_returns(turns: List[Dict[str, Any]], gamma: float) -> List[float]:
    returns = [0.0] * len(turns)
    running = 0.0
    for idx in range(len(turns) - 1, -1, -1):
        reward_total = float(turns[idx].get("reward", {}).get("total", 0.0))
        running = reward_total + gamma * running
        returns[idx] = running
    return returns


def export_episode_metrics(episodes: List[Tuple[int, Dict[str, Any]]], output_dir: Path) -> None:
    rows = []
    for episode_id, episode in episodes:
        turns = episode.get("turns", [])
        reward_engagement = sum(float(t.get("reward", {}).get("engagement", 0.0)) for t in turns)
        reward_novelty = sum(float(t.get("reward", {}).get("novelty", 0.0)) for t in turns)
        reward_responsiveness = sum(float(t.get("reward", {}).get("responsiveness", 0.0)) for t in turns)
        reward_transition = sum(
            float(t.get("reward", {}).get("transition_insufficiency", 0.0))
            + float(t.get("reward", {}).get("transition_exploration", 0.0))
            for t in turns
        )
        reward_conclude = sum(float(t.get("reward", {}).get("conclude", 0.0)) for t in turns)

        rows.append({
            "episode_id": episode_id,
            "reward_total": float(episode.get("episode_reward", 0.0)),
            "reward_engagement": reward_engagement,
            "reward_novelty": reward_novelty,
            "reward_responsiveness": reward_responsiveness,
            "reward_transition": reward_transition,
            "reward_conclude": reward_conclude,
            "turns": int(episode.get("total_turns", len(turns))),
        })

    csv_path = output_dir / "episode_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "episode_id", "reward_total", "reward_engagement", "reward_novelty",
            "reward_responsiveness", "reward_transition", "reward_conclude", "turns"
        ])
        writer.writeheader()
        writer.writerows(rows)


def export_every_10_episode_snapshots(
    episodes: List[Tuple[int, Dict[str, Any]]],
    periodic_snapshots: Dict[int, Dict[str, Any]],
    output_dir: Path,
    inferred_policy_temperature: Optional[float],
) -> None:
    all_actions = sorted({
        action_name_from_turn(turn)
        for _, episode in episodes
        for turn in episode.get("turns", [])
    })

    rows = []
    for end_idx in range(10, len(episodes) + 1, 10):
        end_episode_id = episodes[end_idx - 1][0]
        window = episodes[end_idx - 10:end_idx]
        counts = defaultdict(int)
        for _, episode in window:
            for turn in episode.get("turns", []):
                counts[action_name_from_turn(turn)] += 1

        snapshot = periodic_snapshots.get(end_episode_id, {})
        row = {
            "episode_id": end_episode_id,
            "window_start_episode": window[0][0],
            "window_end_episode": window[-1][0],
            "epsilon": snapshot.get("epsilon"),
            "policy_temperature": snapshot.get("policy_temperature", inferred_policy_temperature),
            "source": "recorded" if snapshot else "derived",
        }
        for action_name in all_actions:
            row[action_name] = counts.get(action_name, 0)
        rows.append(row)

    csv_path = output_dir / "every_10_episode_snapshots.csv"
    fieldnames = ["episode_id", "window_start_episode", "window_end_episode", "epsilon", "policy_temperature", "source"] + all_actions
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_checkpoint_manifest(experiment_dir: Path, output_dir: Path) -> None:
    rows = []

    for file_path in sorted((experiment_dir / "checkpoints").glob("*.pt")):
        rows.append({
            "kind": "checkpoint",
            "file": str(file_path),
            "episode": parse_episode_from_name(file_path.name),
        })

    for file_path in sorted((experiment_dir / "models").glob("*.pt")):
        rows.append({
            "kind": "model",
            "file": str(file_path),
            "episode": parse_episode_from_name(file_path.name),
        })

    csv_path = output_dir / "checkpoint_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "file", "episode"])
        writer.writeheader()
        writer.writerows(rows)


def parse_episode_from_name(name: str) -> Optional[int]:
    for part in name.replace(".", "_").split("_"):
        if part.startswith("ep") and part[2:].isdigit():
            return int(part[2:])
    return None


def export_completion_action_distribution(episodes: List[Tuple[int, Dict[str, Any]]], output_dir: Path) -> None:
    if not episodes:
        return

    scopes = {
        "all_episodes": episodes,
        "final_episode": [episodes[-1]],
    }

    rows = []
    for scope_name, scope_episodes in scopes.items():
        buckets = {
            "completion_lt_1": defaultdict(int),
            "completion_eq_1": defaultdict(int),
        }
        for _, episode in scope_episodes:
            for turn in episode.get("turns", []):
                completion = float(turn.get("context", {}).get("current_exhibit_completion", 0.0))
                action_name = action_name_from_turn(turn)
                if math.isclose(completion, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    buckets["completion_eq_1"][action_name] += 1
                elif completion < 1.0:
                    buckets["completion_lt_1"][action_name] += 1

        for bucket_name, counts in buckets.items():
            total = sum(counts.values())
            for action_name, count in sorted(counts.items()):
                rows.append({
                    "scope": scope_name,
                    "bucket": bucket_name,
                    "action": action_name,
                    "count": count,
                    "frequency": (count / total) if total else 0.0,
                })

    csv_path = output_dir / "completion_action_distribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "bucket", "action", "count", "frequency"])
        writer.writeheader()
        writer.writerows(rows)


def export_action_value_estimates(
    episodes: List[Tuple[int, Dict[str, Any]]],
    gamma: float,
    output_dir: Path,
) -> Dict[str, Any]:
    empirical_returns = defaultdict(list)
    q_proxy_when_selected = defaultdict(list)
    q_proxy_all_states = defaultdict(list)
    has_policy_proxy = False

    for _, episode in episodes:
        turns = episode.get("turns", [])
        discounted_returns = compute_discounted_returns(turns, gamma)
        for turn, discounted_return in zip(turns, discounted_returns):
            action_name = action_name_from_turn(turn)
            empirical_returns[action_name].append(discounted_return)

            policy = turn.get("policy") or {}
            logits = policy.get("action_logits")
            state_value = policy.get("state_value")
            if logits is not None and state_value is not None:
                has_policy_proxy = True
                logits_arr = np.array(logits, dtype=float)
                action_space = infer_action_space_from_turn(turn, len(logits_arr))
                mean_logit = float(np.mean(logits_arr))
                for idx, action_label in enumerate(action_space):
                    q_proxy_all_states[action_label].append(float(state_value) + float(logits_arr[idx] - mean_logit))
                selected_idx = action_space.index(action_name) if action_name in action_space else None
                if selected_idx is not None:
                    q_proxy_when_selected[action_name].append(float(state_value) + float(logits_arr[selected_idx] - mean_logit))

    action_names = sorted(set(empirical_returns.keys()) | set(q_proxy_all_states.keys()))
    rows = []
    for action_name in action_names:
        rows.append({
            "action": action_name,
            "mean_discounted_return_to_go": safe_mean(empirical_returns[action_name]),
            "var_discounted_return_to_go": safe_var(empirical_returns[action_name]),
            "count_selected": len(empirical_returns[action_name]),
            "mean_q_proxy_when_selected": safe_mean(q_proxy_when_selected[action_name]),
            "mean_q_proxy_all_states": safe_mean(q_proxy_all_states[action_name]),
        })

    csv_path = output_dir / "action_value_estimates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "action",
                "mean_discounted_return_to_go",
                "var_discounted_return_to_go",
                "count_selected",
                "mean_q_proxy_when_selected",
                "mean_q_proxy_all_states",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"has_policy_proxy": has_policy_proxy}


def infer_action_space_from_turn(turn: Dict[str, Any], action_dim: int) -> List[str]:
    action_name = action_name_from_turn(turn)
    policy = turn.get("policy") or {}
    probabilities = policy.get("action_probabilities") or []
    if action_name and action_dim == 1:
        return [action_name]
    if action_name and probabilities:
        labels = [f"action_{idx}" for idx in range(len(probabilities))]
        if action_name not in labels:
            labels[0] = action_name
        return labels
    return [f"action_{idx}" for idx in range(action_dim)]


def export_state_statistics(episodes: List[Tuple[int, Dict[str, Any]]], output_dir: Path) -> None:
    states = []
    for _, episode in episodes:
        for turn in episode.get("turns", []):
            state = turn.get("state")
            if state:
                states.append(state)

    if not states:
        return

    state_array = np.array(states, dtype=float)
    rows = []
    for idx in range(state_array.shape[1]):
        rows.append({
            "state_dim": idx,
            "mean": float(np.mean(state_array[:, idx])),
            "variance": float(np.var(state_array[:, idx])),
            "std": float(np.std(state_array[:, idx])),
        })

    csv_path = output_dir / "state_mean_variance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state_dim", "mean", "variance", "std"])
        writer.writeheader()
        writer.writerows(rows)


def export_exhibit_dwell(episodes: List[Tuple[int, Dict[str, Any]]], output_dir: Path) -> None:
    dwell_by_exhibit = defaultdict(list)
    for _, episode in episodes:
        for turn in episode.get("turns", []):
            exhibit = turn.get("context", {}).get("current_exhibit", "Unknown")
            dwell = float(turn.get("context", {}).get("dwell", 0.0))
            dwell_by_exhibit[exhibit].append(dwell)

    rows = []
    for exhibit, dwells in sorted(dwell_by_exhibit.items()):
        rows.append({
            "exhibit": exhibit,
            "mean_dwell": safe_mean(dwells),
            "variance_dwell": safe_var(dwells),
            "count": len(dwells),
        })

    csv_path = output_dir / "exhibit_mean_dwell.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exhibit", "mean_dwell", "variance_dwell", "count"])
        writer.writeheader()
        writer.writerows(rows)


def export_action_transition_matrix(episodes: List[Tuple[int, Dict[str, Any]]], output_dir: Path) -> None:
    actions = sorted({
        action_name_from_turn(turn)
        for _, episode in episodes
        for turn in episode.get("turns", [])
    })
    counts = {src: {dst: 0 for dst in actions} for src in actions}

    for _, episode in episodes:
        turns = episode.get("turns", [])
        for prev_turn, next_turn in zip(turns, turns[1:]):
            src = action_name_from_turn(prev_turn)
            dst = action_name_from_turn(next_turn)
            counts[src][dst] += 1

    count_csv = output_dir / "action_transition_matrix_counts.csv"
    with open(count_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from_action", *actions])
        for src in actions:
            writer.writerow([src, *[counts[src][dst] for dst in actions]])

    prob_csv = output_dir / "action_transition_matrix_probabilities.csv"
    with open(prob_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from_action", *actions])
        for src in actions:
            row_total = sum(counts[src].values())
            probs = [(counts[src][dst] / row_total) if row_total else 0.0 for dst in actions]
            writer.writerow([src, *probs])


def build_recording_audit(
    periodic_snapshots: Dict[int, Dict[str, Any]],
    action_value_meta: Dict[str, Any],
    inferred_policy_temperature: Optional[float],
) -> Dict[str, Any]:
    return {
        "episode_metrics": "recorded_from_detailed_logs",
        "every_10_episode_action_counts": "recorded" if periodic_snapshots else "derived_from_episode_logs",
        "epsilon": "recorded_if_available_else_null",
        "policy_temperature": "recorded" if periodic_snapshots else ("inferred_default" if inferred_policy_temperature is not None else "missing"),
        "action_value_estimates": "empirical_discounted_return_to_go",
        "policy_q_proxy": "recorded_from_turn_policy" if action_value_meta.get("has_policy_proxy") else "missing_for_this_run",
        "state_mean_variance": "recorded_from_turn_states",
        "exhibit_mean_dwell": "recorded_from_turn_context",
        "action_transition_matrix": "recorded_from_turn_sequence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis training metrics.")
    parser.add_argument("--experiment-dir", required=True, help="Path to training_logs/experiments/... directory")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to <experiment-dir>/exported_metrics")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_dir / "exported_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(experiment_dir / "metadata.json") if (experiment_dir / "metadata.json").exists() else {}
    gamma = float(metadata.get("gamma", 0.99))
    detailed_logs_dir = experiment_dir / "detailed_logs"
    analytics_dir = experiment_dir / "analytics"

    episodes = load_episode_logs(detailed_logs_dir)
    periodic_snapshots = load_periodic_snapshots(analytics_dir) if analytics_dir.exists() else {}

    inferred_policy_temperature = None
    if metadata.get("experiment_type") == "flat_rl" or "flat" in str(experiment_dir.name).lower():
        inferred_policy_temperature = 1.0

    export_episode_metrics(episodes, output_dir)
    export_every_10_episode_snapshots(episodes, periodic_snapshots, output_dir, inferred_policy_temperature)
    export_checkpoint_manifest(experiment_dir, output_dir)
    export_completion_action_distribution(episodes, output_dir)
    action_value_meta = export_action_value_estimates(episodes, gamma, output_dir)
    export_state_statistics(episodes, output_dir)
    export_exhibit_dwell(episodes, output_dir)
    export_action_transition_matrix(episodes, output_dir)

    audit = build_recording_audit(periodic_snapshots, action_value_meta, inferred_policy_temperature)
    write_json(output_dir / "recording_audit.json", audit)

    summary = {
        "experiment_dir": str(experiment_dir),
        "episodes_exported": len(episodes),
        "gamma": gamma,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "export_summary.json", summary)
    print(f"[OK] Exported metrics to {output_dir}")


if __name__ == "__main__":
    main()
