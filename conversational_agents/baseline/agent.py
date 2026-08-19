"""Portable baseline chain: knowledge graph -> prompt -> LLM response."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def load_knowledge_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_exhibit(graph: dict[str, Any], exhibit: str) -> tuple[str, dict[str, Any]]:
    exhibits = graph.get("exhibits", {})
    if exhibit in exhibits:
        return exhibit, exhibits[exhibit]
    for key, value in exhibits.items():
        if str(value.get("object_name", "")).lower() == exhibit.lower():
            return key, value
    raise ValueError(f"Unknown exhibit '{exhibit}'. Available: {', '.join(exhibits)}")


def build_prompt(message: str, exhibit_key: str, exhibit: dict[str, Any]) -> str:
    facts = [
        exhibit.get("painting_name"),
        exhibit.get("description"),
        exhibit.get("more_info"),
        exhibit.get("artist"),
        exhibit.get("year"),
        exhibit.get("style"),
        exhibit.get("location"),
    ]
    context = "\n".join(f"- {fact}" for fact in facts if fact)
    return f"""You are a concise, conversational museum guide.
Answer only from the supplied knowledge-graph facts. If the answer is absent,
say that the available museum information does not contain it. Do not invent facts.

Exhibit key: {exhibit_key}
Knowledge-graph facts:
{context}

Visitor: {message}
Guide:"""


def generate(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--exhibit", default="King_Caspar")
    parser.add_argument("--knowledge-graph", type=Path, default=HERE / "knowledge_graph.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--dry-run", action="store_true", help="Print the grounded prompt without calling an LLM.")
    args = parser.parse_args()

    graph = load_knowledge_graph(args.knowledge_graph)
    exhibit_key, exhibit = resolve_exhibit(graph, args.exhibit)
    prompt = build_prompt(args.message, exhibit_key, exhibit)
    print(prompt if args.dry_run else generate(prompt, args.model))


if __name__ == "__main__":
    main()
