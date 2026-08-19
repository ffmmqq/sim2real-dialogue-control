#!/usr/bin/env python3
"""Create analysis-ready tables from a frozen evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def completion_bin(value: float) -> str:
    if value >= 1.0 - 1e-9:
        return "1.00"
    if value >= 0.75:
        return "0.75-<1.00"
    if value >= 0.50:
        return "0.50-<0.75"
    if value >= 0.25:
        return "0.25-<0.50"
    return "0.00-<0.25"


def descriptive(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": "", "sd": "", "min": "", "max": ""}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "sd": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def action_distribution(turns: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    denominators: Counter[Tuple[str, str]] = Counter()
    for turn in turns:
        profile = turn["profile"]
        band = completion_bin(float(turn["completion_fact_weighted_before"]))
        action = turn["selected_action"]
        for profile_key in (profile, "ALL"):
            for band_key in (band, "ALL"):
                counts[(profile_key, band_key, action)] += 1
                denominators[(profile_key, band_key)] += 1
    rows = []
    for (profile, band, action), count in sorted(counts.items()):
        rows.append({
            "profile": profile,
            "completion_bin_before_action": band,
            "action": action,
            "count": count,
            "proportion": count / denominators[(profile, band)],
            "denominator_turns": denominators[(profile, band)],
        })
    return rows


def transitions(sessions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    source_totals: Counter[Tuple[str, str]] = Counter()
    for session in sessions:
        sequence = session.get("action_sequence", [])
        for first, second in zip(sequence, sequence[1:]):
            for profile in (session["profile"], "ALL"):
                counts[(profile, first, second)] += 1
                source_totals[(profile, first)] += 1
    return [
        {
            "profile": profile,
            "from_action": first,
            "to_action": second,
            "count": count,
            "conditional_probability": count / source_totals[(profile, first)],
            "from_action_total": source_totals[(profile, first)],
        }
        for (profile, first, second), count in sorted(counts.items())
    ]


def dwell_by_type(turns: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: defaultdict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for turn in turns:
        dwell = turn.get("simulator_response_dwell")
        if dwell is None:
            continue
        native = turn.get("visitor_response_type_native", "unknown")
        shared = turn.get("visitor_utterance_type_shared", "unknown")
        for profile in (turn["profile"], "ALL"):
            groups[(profile, native, shared)].append(float(dwell))
    rows = []
    for (profile, native, shared), values in sorted(groups.items()):
        rows.append({
            "source": "simulator",
            "profile": profile,
            "native_response_type": native,
            "shared_surface_type": shared,
            **descriptive(values),
        })
    return rows


def dwell_by_action_and_type(turns: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: defaultdict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for turn in turns:
        dwell = turn.get("simulator_response_dwell")
        if dwell is not None:
            groups[(turn["profile"], turn["selected_action"], turn["visitor_utterance_type_shared"])].append(float(dwell))
    rows = []
    for (profile, action, utterance_type), values in sorted(groups.items()):
        rows.append({
            "profile": profile,
            "action": action,
            "shared_surface_type": utterance_type,
            **descriptive(values),
        })
    return rows


def sequence_rows(sessions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for session in sessions:
        sequence = session.get("action_sequence", [])
        first_transition = next(
            (index + 1 for index, action in enumerate(sequence) if action.startswith("OfferTransition/")),
            None,
        )
        rows.append({
            "session_id": session["session_id"],
            "profile": session["profile"],
            "evaluation_seed": session["evaluation_seed"],
            "turn_count": session["turn_count"],
            "first_transition_turn": first_transition if first_transition is not None else "",
            "action_sequence": " > ".join(sequence),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    turns = read_jsonl(run_dir / "turns.jsonl")
    sessions = read_jsonl(run_dir / "sessions.jsonl")
    if not turns or not sessions:
        raise RuntimeError("The run has no completed turn/session data")

    action_rows = action_distribution(turns)
    transition_rows = transitions(sessions)
    dwell_rows = dwell_by_type(turns)
    dwell_action_rows = dwell_by_action_and_type(turns)
    sequences = sequence_rows(sessions)
    write_csv(
        run_dir / "action_distribution.csv", action_rows,
        ("profile", "completion_bin_before_action", "action", "count", "proportion", "denominator_turns"),
    )
    write_csv(
        run_dir / "action_transitions.csv", transition_rows,
        ("profile", "from_action", "to_action", "count", "conditional_probability", "from_action_total"),
    )
    write_csv(
        run_dir / "simulator_dwell_by_utterance_type.csv", dwell_rows,
        ("source", "profile", "native_response_type", "shared_surface_type", "n", "mean", "sd", "min", "max"),
    )
    write_csv(
        run_dir / "simulator_dwell_by_action_and_type.csv", dwell_action_rows,
        ("profile", "action", "shared_surface_type", "n", "mean", "sd", "min", "max"),
    )
    write_csv(
        run_dir / "action_sequences.csv", sequences,
        ("session_id", "profile", "evaluation_seed", "turn_count", "first_transition_turn", "action_sequence"),
    )

    profile_sessions = Counter(session["profile"] for session in sessions)
    profile_turns = Counter(turn["profile"] for turn in turns)
    summary = {
        "sessions": len(sessions),
        "turns": len(turns),
        "sessions_by_profile": dict(profile_sessions),
        "turns_by_profile": dict(profile_turns),
        "mean_session_length": mean(float(session["turn_count"]) for session in sessions),
        "mean_final_completion_fact_weighted": mean(
            float(session["final_completion_fact_weighted"]) for session in sessions
        ),
        "termination_reasons": dict(Counter(session["termination_reason"] for session in sessions)),
        "generated_tables": [
            "action_distribution.csv",
            "action_transitions.csv",
            "action_sequences.csv",
            "simulator_dwell_by_utterance_type.csv",
            "simulator_dwell_by_action_and_type.csv",
        ],
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
