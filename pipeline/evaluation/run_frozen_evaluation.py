#!/usr/bin/env python3
"""Frozen evaluation of the 151-d flat actor--critic in HybridSimulator.

This script never creates an optimizer and never updates model parameters.  It
reconstructs the state, masks, reward configuration and simulator used by the
training run, while using the deterministic action rule used by the deployed
museum agent by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "evaluation_config.json"
REWARD_INFO_KEYS = (
    "reward_engagement",
    "reward_novelty",
    "reward_bnov_new",
    "reward_bnov_rep",
    "reward_bnov_clar",
    "reward_bnov_ask",
    "reward_bnov_stale",
    "reward_bnov_transition",
    "reward_responsiveness",
    "reward_conclude",
    "reward_transition_insufficiency",
    "reward_transition_exploration",
    "reward_question_asking",
    "reward_response_type",
    "reward_completion",
    "reward_action_repeat",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def source_tree_provenance(root: Path) -> Dict[str, Any]:
    included_suffixes = {".py", ".json", ".sh", ".sbatch"}
    excluded_parts = {
        ".cache", ".git", ".venv", "__pycache__", "outputs", "slurm_logs",
        "training_logs",
    }
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in included_suffixes:
            continue
        relative = path.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        files.append({
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    digest = hashlib.sha256()
    for entry in files:
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return {"root": str(root), "sha256": digest.hexdigest(), "file_count": len(files)}


def runtime_provenance_from_env(required: bool) -> Dict[str, Any] | None:
    value = os.environ.get("IUI27_RUNTIME_PROVENANCE")
    if not value:
        if required:
            raise RuntimeError(
                "Slurm local-Llama evaluation is missing IUI27_RUNTIME_PROVENANCE; "
                "use frozen_policy_evaluation/slurm/evaluate_frozen_local_llama.sbatch"
            )
        return None
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"Runtime provenance file does not exist: {path}")
    return read_json(path)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(to_builtin(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def shared_utterance_type(text: str) -> str:
    """Surface-only label that can be applied identically to simulator/humans."""
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return "silence"
    confusion_markers = (
        "i don't understand", "i do not understand", "not sure what you mean",
        "what do you mean", "confused", "could you clarify",
    )
    if any(marker in cleaned for marker in confusion_markers):
        return "confusion"
    if "?" in cleaned or re.match(
        r"^(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|would|will|have|has)\b",
        cleaned,
    ):
        return "question"
    acknowledgement_markers = (
        "okay", "ok", "yes", "yeah", "right", "i see", "interesting",
        "thanks", "thank you", "got it", "sure", "wow",
    )
    word_count = len(re.findall(r"\b\w+\b", cleaned))
    if word_count <= 8 and any(cleaned.startswith(marker) for marker in acknowledgement_markers):
        return "acknowledgment"
    return "statement"


def completion(coverage: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    valid = [item for item in coverage.values() if int(item.get("total", 0)) > 0]
    total = sum(int(item["total"]) for item in valid)
    mentioned = sum(int(item["mentioned"]) for item in valid)
    mean_exhibit = sum(float(item["coverage"]) for item in valid) / len(valid) if valid else 0.0
    return {
        "fact_weighted": mentioned / total if total else 0.0,
        "mean_exhibit": mean_exhibit,
    }


def configure_environment(
    training_metadata: Mapping[str, Any], config: Mapping[str, Any], template_mode: bool
) -> Dict[str, str]:
    reward = training_metadata["reward_parameters"]
    settings = {
        "HRL_BERT_MODE": str(training_metadata.get("bert_mode", "standard")),
        "HRL_REWARD_MODE": str(training_metadata["reward_mode"]),
        "HRL_W_ENGAGEMENT": str(reward["w_engagement"]),
        "HRL_CENTRED_ENGAGEMENT": "1" if reward["centred_engagement"] else "0",
        "HRL_ENGAGEMENT_GATED_NOVELTY": "1" if reward["engagement_gated_novelty"] else "0",
        "HRL_DWELL_EMA_ALPHA": str(reward["dwell_ema_alpha"]),
        "HRL_NOVELTY_PER_FACT": str(reward["novelty_per_fact"]),
        "HRL_BROADENED_NOVELTY": "1" if reward["broadened_novelty"] else "0",
        "HRL_ALPHA_NEW": str(reward["alpha_new"]),
        "HRL_ALPHA_REP": str(reward["alpha_rep"]),
        "HRL_ALPHA_CLAR": str(reward["alpha_clar"]),
        "HRL_ALPHA_ASK": str(reward["alpha_ask"]),
        "HRL_ALPHA_STALE": str(reward["alpha_stale"]),
        "HRL_ALPHA_TRANSITION": str(reward["alpha_transition"]),
        "HRL_ACTION_REPEAT_PENALTY": str(reward["action_repeat_penalty"]),
        "HRL_ACTION_REPEAT_THRESHOLD": str(reward["action_repeat_threshold"]),
        "HRL_W_RESPONSIVENESS": str(reward["w_responsiveness"]),
        "HRL_W_CONCLUDE": str(reward["w_conclude"]),
        "HRL_RESPONSE_TYPE_FEATURE": "1" if reward["response_type_feature"] else "0",
        "HRL_RESPONSE_TYPE_REWARD": "1" if reward["response_type_reward"] else "0",
        "HRL_W_RESPONSE_TYPE": str(reward["w_response_type"]),
        "HRL_EXHAUSTION_PENALTY": str(reward["exhaustion_penalty"]),
        "HRL_TRANSITION_BONUS": str(reward["transition_bonus"]),
        "HRL_ENF_DECAY_RATE": str(reward["enf_decay_rate"]),
        "HRL_ENF_DECAY_FLOOR": str(reward["enf_decay_floor"]),
        "HRL_ZERO_ENGAGEMENT_EXHAUSTED": "0",
        "HRL_TEMPLATE_MODE": "1" if template_mode else "0",
        "HRL_VERBOSE": "0",
        "HRL_LLM_BACKEND": str(config["llm_backend"]),
        "HRL_LLM_MODEL": str(config["llm_model"]),
        "HRL_AGENT_TEMPERATURE": str(config.get("agent_llm_temperature", 0.3)),
        "HRL_AGENT_MAX_TOKENS": str(config.get("agent_llm_max_tokens", 300)),
        "HRL_SIMULATOR_TEMPERATURE": str(config.get("simulator_llm_temperature", 0.6)),
        "HRL_SIMULATOR_MAX_TOKENS": str(config.get("simulator_llm_max_tokens", 150)),
        "HF_HOME": os.environ.get(
            "HF_HOME", str(Path(config["project_root"]) / ".hf_cache")
        ),
        "TRANSFORMERS_CACHE": os.environ.get(
            "TRANSFORMERS_CACHE", str(Path(config["project_root"]) / ".hf_cache")
        ),
    }
    settings.update({
        "LOCAL_LLM_BASE_URL": os.environ.get(
            "LOCAL_LLM_BASE_URL", str(config["local_llm_base_url"])
        ),
        "LOCAL_LLM_API_KEY": os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
        "LOCAL_LLM_SEED": str(config.get("local_llm_seed", 42)),
        "LOCAL_LLM_TOP_P": str(config.get("local_llm_top_p", 0.9)),
    })
    if template_mode:
        # Smoke tests must remain offline and fast. Formal runs intentionally do
        # not set these flags because they would change the 128 embedding inputs.
        settings["HRL_FAST_MODE"] = "1"
        settings["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.update(settings)
    return settings


def checkpoint_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_weights(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("agent_state_dict", "agent_state", "network", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload  # raw state_dict
    raise ValueError("Unsupported checkpoint format: no agent weights found")


def validate_checkpoint(config: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint = Path(config["checkpoint"])
    payload = checkpoint_payload(checkpoint)
    weights = extract_weights(payload)
    input_weight = weights.get("encoder.weight_ih_l0")
    policy_weight = weights.get("policy_head.2.weight")
    if input_weight is None or policy_weight is None:
        raise ValueError("Checkpoint is not the expected LSTM flat actor--critic")
    state_dim = int(input_weight.shape[1])
    action_dim = int(policy_weight.shape[0])
    if state_dim != int(config["expected_state_dim"]):
        raise ValueError(f"Expected {config['expected_state_dim']}-d input, found {state_dim}")
    if action_dim != int(config["expected_action_dim"]):
        raise ValueError(f"Expected {config['expected_action_dim']} actions, found {action_dim}")
    return {
        "checkpoint": str(checkpoint),
        "sha256": sha256(checkpoint),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "checkpoint_keys": sorted(payload.keys()),
        "options": payload.get("options"),
        "subactions": payload.get("subactions"),
    }


def session_schedule(profiles: Sequence[str], count: int, seed_start: int) -> List[Dict[str, Any]]:
    schedule = []
    total = len(profiles) * count
    profile_numbers = Counter()
    for offset in range(total):
        profile = profiles[offset % len(profiles)]
        profile_numbers[profile] += 1
        schedule.append({
            "session_index": offset + 1,
            "session_id": f"{profile.lower()}_{profile_numbers[profile]:03d}",
            "profile": profile,
            "evaluation_seed": seed_start + offset,
        })
    return schedule


def sync_focus(env: Any, simulator: Any) -> None:
    exhibit = simulator.get_current_aoi()
    focus = env.exhibit_keys.index(exhibit) + 1 if exhibit in env.exhibit_keys else 0
    env.update_user_state(focus=focus)


def update_simulator_after_action(env: Any, simulator: Any, info: Mapping[str, Any]) -> Dict[str, Any]:
    option = info.get("option")
    subaction = info.get("subaction")
    effective_option = "OfferTransition" if option == "Transition" or (
        subaction == "SuggestMove" and option != "OfferTransition"
    ) else option
    response = simulator.generate_user_response(
        info.get("agent_utterance", ""),
        agent_option=effective_option,
        agent_subaction=subaction,
        target_exhibit=info.get("target_exhibit"),
        current_exhibit_completion=info.get("current_exhibit_completion", 0.0),
        exhibit_exhausted=info.get("exhibit_exhausted", False),
        target_exhibit_completion=info.get("target_exhibit_completion", 0.0),
        target_exhibit_exhausted=info.get("target_exhibit_exhausted", False),
    )
    env.update_user_state(
        utterance=response.get("utterance", ""),
        response_type=response.get("response_type", "statement"),
        visitor_state=response.get("visitor_state"),
    )
    gaze = response.get("gaze_features") or []
    if gaze:
        env.update_user_state(dwell=float(gaze[0]))
    sync_focus(env, simulator)
    if (info.get("option") == "OfferTransition" or info.get("subaction") == "SuggestMove") \
            and response.get("transition_success", False):
        env.record_successful_transition()
    return response


def termination_reason(done: bool, info: Mapping[str, Any], turn: int, max_turns: int) -> str | None:
    if not done:
        return None
    if info.get("auto_concluded"):
        return "all_exhibits_complete"
    if info.get("option") == "Conclude" or info.get("subaction") == "WrapUp":
        return "policy_wrap_up"
    if turn >= max_turns:
        return "max_turns"
    return "environment_done"


def run_session(
    *, env: Any, simulator: Any, agent: Any, spec: Mapping[str, Any],
    max_turns: int, deterministic: bool, turns_path: Path,
) -> Dict[str, Any]:
    from pipeline.utils.state_timing import states_equal, validate_state

    seed = int(spec["evaluation_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    simulator.initialize_session(persona="Agreeable", persona_profile=spec["profile"])
    intro = simulator.inject_introduction(simulator.get_current_aoi())
    env.set_initial_dialogue(intro["agent_greeting"], intro["user_response"])
    sync_focus(env, simulator)
    obs = env._get_obs()
    agent.reset()

    total_reward = 0.0
    action_sequence: List[str] = []
    native_types: Counter[str] = Counter()
    shared_types: Counter[str] = Counter()
    final_reason = "max_turns"
    final_completion = {"fact_weighted": 0.0, "mean_exhibit": 0.0}
    turn_rows = 0
    timing_counts = Counter({
        "state_exhibit_mismatch_count": 0,
        "state_coverage_mismatch_count": 0,
        "temporal_state_mismatch_count": 0,
        "action_mask_state_mismatch_count": 0,
        "reward_response_mismatch_count": 0,
        "zero_fact_offer_transition_count": 0,
        "consecutive_offer_transition_count": 0,
    })
    previous_was_transition = False

    for turn in range(1, max_turns + 1):
        sync_focus(env, simulator)
        state_for_selection = env._get_obs()
        temporal_state_match = states_equal(obs, state_for_selection)
        obs = state_for_selection
        coverage_before = to_builtin(env._get_museum_exhibit_coverage())
        completion_before = completion(coverage_before)
        exhibit_before = simulator.get_current_aoi()
        simulator_before = simulator.get_current_state()
        state_before_validation = validate_state(env, obs, exhibit_before, coverage_before)
        mask = env.get_flat_action_mask()
        action_mask_state_match = states_equal(obs, env._get_obs())
        available_options = env._get_available_options()
        available_subactions = {
            option: env._get_available_subactions(option) for option in available_options
        }
        action = agent.select_action(
            state=obs,
            available_options=available_options,
            available_subactions_dict=available_subactions,
            deterministic=deterministic,
        )

        next_obs, reward, done, truncated, info = env.step(
            int(action["flat_action"]),
            post_action_callback=lambda callback_info: update_simulator_after_action(
                env, simulator, callback_info
            ),
        )
        response = info.get("simulator_response", {})
        simulator_after = simulator.get_current_state()
        coverage_after = to_builtin(env._get_museum_exhibit_coverage())
        completion_after = completion(coverage_after)
        exhibit_after = simulator.get_current_aoi()
        state_next_validation = validate_state(env, next_obs, exhibit_after, coverage_after)
        gaze = response.get("gaze_features") or []
        response_dwell = float(gaze[0]) if gaze else None
        utterance = response.get("utterance", "")
        native_type = response.get("response_type", "statement")
        surface_type = shared_utterance_type(utterance)
        selected_name = action["flat_action_name"]
        is_transition = (
            action["option_name"] == "OfferTransition"
            or action["subaction_name"] == "SuggestMove"
        )
        zero_fact_transition = bool(
            is_transition
            and int(coverage_before.get(exhibit_before, {}).get("mentioned", 0)) == 0
        )
        consecutive_transition = bool(is_transition and previous_was_transition)
        previous_was_transition = is_transition
        reward_response_match = (
            (response_dwell is None or abs(float(info.get("dwell", 0.0)) - response_dwell) <= 1e-6)
            and info.get("reward_response_utterance", "") == utterance
            and info.get("reward_response_type") == native_type
        )
        state_exhibit_match = (
            state_before_validation["exhibit_match"]
            and state_next_validation["exhibit_match"]
        )
        state_coverage_match = (
            state_before_validation["coverage_match"]
            and state_next_validation["coverage_match"]
        )
        timing_counts["state_exhibit_mismatch_count"] += int(not state_exhibit_match)
        timing_counts["state_coverage_mismatch_count"] += int(not state_coverage_match)
        timing_counts["temporal_state_mismatch_count"] += int(not temporal_state_match)
        timing_counts["action_mask_state_mismatch_count"] += int(not action_mask_state_match)
        timing_counts["reward_response_mismatch_count"] += int(not reward_response_match)
        timing_counts["zero_fact_offer_transition_count"] += int(zero_fact_transition)
        timing_counts["consecutive_offer_transition_count"] += int(consecutive_transition)
        action_sequence.append(selected_name)
        native_types[native_type] += 1
        shared_types[surface_type] += 1
        total_reward += float(reward)

        row = {
            "schema_version": "1.0",
            "session_index": spec["session_index"],
            "session_id": spec["session_id"],
            "profile": spec["profile"],
            "evaluation_seed": seed,
            "turn_index": turn,
            "policy_mode": "greedy" if deterministic else "sample",
            "state_dim": len(obs),
            "state_before": to_builtin(obs),
            "state_next_before_simulator": info["state_next_before_simulator"],
            "state_next_after_simulator": info["state_next_after_simulator"],
            "state_next_actual": to_builtin(next_obs),
            "action_mask": mask,
            "available_options": available_options,
            "available_subactions": available_subactions,
            "selected_action_index": action["flat_action"],
            "selected_action": selected_name,
            "selected_option": action["option_name"],
            "selected_subaction": action["subaction_name"],
            "selected_action_probability": action["selected_action_probability"],
            "action_logits_unmasked": action["action_logits"],
            "action_probabilities_masked": action["action_probabilities"],
            "state_value": action["state_value"],
            "executed_option": info.get("option"),
            "executed_subaction": info.get("subaction"),
            "auto_concluded": bool(info.get("auto_concluded", False)),
            "guide_utterance": info.get("agent_utterance", ""),
            "visitor_utterance": utterance,
            "visitor_word_count": len(re.findall(r"\b\w+\b", utterance)),
            "visitor_question_mark": "?" in utterance,
            "visitor_response_type_native": native_type,
            "visitor_utterance_type_shared": surface_type,
            "visitor_state_before": simulator_before.get("visitor_state"),
            "visitor_state_after": response.get("visitor_state"),
            "simulator_internal_engagement_before": simulator_before.get("engagement_level"),
            "simulator_internal_engagement_after": simulator_after.get("engagement_level"),
            "simulator_response_dwell": response_dwell,
            "simulator_gaze_features": gaze,
            "reward_input_dwell": info.get("dwell"),
            "reward_response_utterance": info.get("reward_response_utterance"),
            "reward_response_type": info.get("reward_response_type"),
            "reward_response_match": reward_response_match,
            "reward_total": float(reward),
            "reward_components": {key: info.get(key, 0.0) for key in REWARD_INFO_KEYS},
            "current_exhibit_before": exhibit_before,
            "current_exhibit_after": exhibit_after,
            "focus_encoded_in_state": state_before_validation["focus_encoded_in_state"],
            "focus_encoded_in_next_state": state_next_validation["focus_encoded_in_state"],
            "coverage_encoded_in_state": state_before_validation["coverage_encoded_in_state"],
            "coverage_encoded_in_next_state": state_next_validation["coverage_encoded_in_state"],
            "target_exhibit": info.get("target_exhibit"),
            "transition_success": bool(response.get("transition_success", False)),
            "current_exhibit_completion_for_transition": info.get("current_exhibit_completion"),
            "exhibit_exhausted": info.get("exhibit_exhausted"),
            "facts_new_count": info.get("facts_shared", 0),
            "facts_new_ids": info.get("facts_mentioned_in_utterance", []),
            "facts_hallucinated_ids": info.get("hallucinated_facts", []),
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "state_exhibit_match": state_exhibit_match,
            "state_coverage_match": state_coverage_match,
            "temporal_state_match": temporal_state_match,
            "action_mask_state_match": action_mask_state_match,
            "zero_fact_offer_transition": zero_fact_transition,
            "consecutive_offer_transition": consecutive_transition,
            "completion_fact_weighted_before": completion_before["fact_weighted"],
            "completion_fact_weighted_after": completion_after["fact_weighted"],
            "completion_mean_exhibit_before": completion_before["mean_exhibit"],
            "completion_mean_exhibit_after": completion_after["mean_exhibit"],
            "done": bool(done),
            "truncated": bool(truncated),
            "termination_reason": termination_reason(bool(done), info, turn, max_turns),
            "agent_llm_seconds": info.get("agent_llm_time", 0.0),
            "simulator_llm_seconds": response.get("simulator_llm_time", 0.0),
        }
        append_jsonl(turns_path, to_builtin(row))
        turn_rows += 1
        obs = next_obs
        final_completion = completion_after
        if done or truncated:
            final_reason = row["termination_reason"] or "truncated"
            break

    return {
        "schema_version": "1.0",
        **dict(spec),
        "turn_count": turn_rows,
        "episode_return": total_reward,
        "final_completion_fact_weighted": final_completion["fact_weighted"],
        "final_completion_mean_exhibit": final_completion["mean_exhibit"],
        "termination_reason": final_reason,
        "action_sequence": action_sequence,
        "action_counts": dict(Counter(action_sequence)),
        "visitor_response_type_native_counts": dict(native_types),
        "visitor_utterance_type_shared_counts": dict(shared_types),
        "state_timing_counts": dict(timing_counts),
        "completed_at": utc_now(),
    }


def write_session_csv(sessions_jsonl: Path, csv_path: Path) -> None:
    rows = list(read_jsonl(sessions_jsonl))
    fields = [
        "session_index", "session_id", "profile", "evaluation_seed", "turn_count",
        "episode_return", "final_completion_fact_weighted",
        "final_completion_mean_exhibit", "termination_reason", "action_sequence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat["action_sequence"] = " > ".join(row.get("action_sequence", []))
            writer.writerow(flat)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default="eval_151_hybrid_300_state_synced")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--sessions-per-profile", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--policy-mode", choices=("greedy", "sample"))
    parser.add_argument("--template-mode", action="store_true",
                        help="Offline smoke testing only; do not use for the reported evaluation.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = read_json(args.config.resolve())
    for key, value in (
        ("sessions_per_profile", args.sessions_per_profile),
        ("seed_start", args.seed_start),
        ("max_turns", args.max_turns),
        ("policy_mode", args.policy_mode),
    ):
        if value is not None:
            config[key] = value

    if config.get("llm_backend") != "local_openai":
        raise ValueError(
            "Experiment A requires llm_backend=local_openai; remote LLM backends are disabled."
        )

    validation = validate_checkpoint(config)
    if args.validate_only:
        print(json.dumps({"status": "valid", **validation}, indent=2))
        return 0

    project_root = Path(config["project_root"]).resolve()
    training_metadata = read_json(Path(config["training_metadata"]).resolve())
    env_settings = configure_environment(training_metadata, config, args.template_mode)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from pipeline.flat_rl.agent import FlatActorCriticAgent
    from pipeline.flat_rl.env import FlatDialogueEnv
    from simulator import get_simulator

    env = FlatDialogueEnv(
        knowledge_graph_path=str(Path(config["knowledge_graph"]).resolve()),
        max_turns=int(config["max_turns"]),
    )
    if int(env.observation_space.shape[0]) != int(config["expected_state_dim"]):
        raise RuntimeError(f"Reconstructed environment is {env.observation_space.shape[0]}-d, expected 151-d")

    payload = checkpoint_payload(Path(config["checkpoint"]).resolve())
    agent = FlatActorCriticAgent(
        state_dim=int(config["expected_state_dim"]),
        options=env.options,
        subactions=env.subactions,
        hidden_dim=256,
        lstm_hidden_dim=128,
        use_lstm=True,
        device="cpu",
        sampling_temperature=float(config["policy_temperature"]),
    )
    agent.network.load_state_dict(extract_weights(payload), strict=True)
    agent.network.eval()
    for parameter in agent.network.parameters():
        parameter.requires_grad_(False)

    run_dir = (args.output_root / args.run_name).resolve()
    turns_path = run_dir / "turns.jsonl"
    sessions_path = run_dir / "sessions.jsonl"
    failures_path = run_dir / "failures.jsonl"
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{run_dir} is not empty; use --resume or choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    deterministic = config["policy_mode"] == "greedy"
    schedule = session_schedule(
        config["profiles"], int(config["sessions_per_profile"]), int(config["seed_start"])
    )
    completed = {row["session_id"] for row in read_jsonl(sessions_path)} if args.resume else set()
    runtime_provenance = runtime_provenance_from_env(
        required=bool(
            os.environ.get("SLURM_JOB_ID") and config["llm_backend"] == "local_openai"
        )
    )
    manifest = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": utc_now(),
        "purpose": "Frozen post-training evaluation of the reconstructed 151-d policy",
        "claim_boundary": "This is not a re-creation of the original human user-study checkpoint.",
        "config": config,
        "checkpoint_validation": validation,
        "training_metadata_snapshot": training_metadata,
        "environment_variables": env_settings,
        "policy_frozen": True,
        "optimizer_created": False,
        "action_selection": {
            "mode": config["policy_mode"],
            "deterministic": deterministic,
            "temperature": config["policy_temperature"],
            "reason": "greedy matches museum_agent_export deployment" if deterministic else "training-style masked categorical sampling",
        },
        "seed_schedule": schedule,
        "project_git_commit": git_commit(project_root),
        "code_provenance": source_tree_provenance(project_root),
        "evaluation_code_provenance": source_tree_provenance(HERE),
        "runtime_provenance": runtime_provenance,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "command": sys.argv,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for spec in schedule:
        if spec["session_id"] in completed:
            continue
        simulator = get_simulator(
            simulator_type="hybrid",
            knowledge_graph=env.knowledge_graph,
            seed=int(spec["evaluation_seed"]),
            stochasticity=float(config["simulator_stochasticity"]),
        )
        try:
            summary = run_session(
                env=env, simulator=simulator, agent=agent, spec=spec,
                max_turns=int(config["max_turns"]), deterministic=deterministic,
                turns_path=turns_path,
            )
            append_jsonl(sessions_path, to_builtin(summary))
            print(
                f"[{spec['session_index']:03d}/{len(schedule)}] {spec['session_id']} "
                f"turns={summary['turn_count']} return={summary['episode_return']:.3f}"
            )
        except Exception as exc:
            append_jsonl(failures_path, {
                **spec, "error": repr(exc), "traceback": traceback.format_exc(), "at": utc_now(),
            })
            if not args.continue_on_error:
                raise

    write_session_csv(sessions_path, run_dir / "sessions.csv")
    completed_rows = list(read_jsonl(sessions_path))
    count_keys = (
        "state_exhibit_mismatch_count",
        "state_coverage_mismatch_count",
        "temporal_state_mismatch_count",
        "action_mask_state_mismatch_count",
        "reward_response_mismatch_count",
        "zero_fact_offer_transition_count",
        "consecutive_offer_transition_count",
    )
    consistency = {
        key: sum(int(row.get("state_timing_counts", {}).get(key, 0)) for row in completed_rows)
        for key in count_keys
    }
    mismatch_keys = count_keys[:5]
    consistency["all_state_mismatches_zero"] = all(consistency[key] == 0 for key in mismatch_keys)
    consistency["completed_sessions"] = len(completed_rows)
    consistency["expected_sessions"] = len(schedule)
    (run_dir / "state_consistency_summary.json").write_text(
        json.dumps(consistency, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    protocol_complete = len(completed_rows) == len(schedule)
    manifest["status"] = (
        "complete" if protocol_complete and consistency["all_state_mismatches_zero"]
        else "validation_failed" if protocol_complete
        else "incomplete"
    )
    manifest["completed_at"] = utc_now()
    manifest["completed_sessions"] = len(completed_rows)
    manifest["expected_sessions"] = len(schedule)
    manifest["state_consistency"] = consistency
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation data to {run_dir}")
    print(json.dumps({"state_consistency": consistency}, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
