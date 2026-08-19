#!/usr/bin/env python3
"""Fail closed on likely secrets, prohibited raw data, VR assets, or large files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SKIP_DIRS = {".git", "__pycache__", ".cache", ".venv"}
PROHIBITED_PARTS = {
    "raw_data", "participants", "participant_data", "raw_gaze", "transcripts",
    "questionnaires", "audio", "video", "assets", "packages", "projectsettings",
}
PROHIBITED_SUFFIXES = {".wav", ".mp3", ".mp4", ".mov", ".avi", ".unity", ".prefab", ".asset"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".csv", ".sh", ".example"}
PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "machine-specific home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\]+\\\\"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-mb", type=float, default=10.0)
    args = parser.parse_args()
    failures = []
    for path in sorted(args.root.resolve().rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(args.root.resolve())
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & PROHIBITED_PARTS or path.suffix.lower() in PROHIBITED_SUFFIXES:
            failures.append(f"prohibited release path: {relative}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > args.max_mb:
            failures.append(f"large file ({size_mb:.1f} MiB): {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".env.example"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
    if failures:
        print("Release check failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release check passed: no prohibited paths, obvious secrets, machine paths, or oversized files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

