#!/usr/bin/env python3
"""Export only the approved anonymous derived-measurement columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


GAZE = {
    "FC": "gaze_fixation_count",
    "TFT": "TFT_ms",
    "MDT": "MDT_per_painting_ms",
    "FRA": "FRA_per_painting",
    "NSL": "NSL_per_painting",
    "GTE": "GTE_per_painting",
    "MSD": "MSD",
    "K": "K",
    "AR": "AR_per_painting",
    "SGV": "SGV",
}
CONVERSATION = {
    "NAT": "agent_window_count",
    "NUT": "NUT",
    "MARL": "MARL_chars",
    "MURL": "MURL_chars",
    "TCD": "TCD",
    "CDP": "CDP",
    "PC": "PC",
}
ALIGNMENT = {"MCFR-A": "MCFR_A", "MCSL-A": "MCSL_A"}


def export(rows, output: Path, mapping: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *mapping])
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            public = {"sample_id": f"sample_{index:03d}"}
            public.update({public_name: row.get(source_name, "") for public_name, source_name in mapping.items()})
            writer.writerow(public)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Private source table; never copied to the release.")
    parser.add_argument("--output", type=Path, default=Path("data/measurements"))
    args = parser.parse_args()
    with args.source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    export(rows, args.output / "gaze.csv", GAZE)
    export(rows, args.output / "conversation.csv", CONVERSATION)
    export(rows, args.output / "multimodal_alignment.csv", ALIGNMENT)
    print(f"Exported {len(rows)} anonymous rows to {args.output}")


if __name__ == "__main__":
    main()

