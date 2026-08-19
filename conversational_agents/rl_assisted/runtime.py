"""Deployment chain: trained policy -> grounded prompt -> LLM response.

Training code is intentionally imported from :mod:`pipeline`; it is not copied
into this deployment folder. The default policy is the final seed-1 checkpoint.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from pipeline.utils.knowledge_graph import SimpleKnowledgeGraph

from .model_loader import create_agent_from_checkpoint, load_model_checkpoint
from .response_type_provider import predict_response_type
from .state_builder import build_state, get_projection_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "seed_1" / "checkpoint.pt"


def load_policy(checkpoint_path: Path, device: str = "cpu"):
    checkpoint = load_model_checkpoint(str(checkpoint_path), device=device)
    agent, model_type, metadata = create_agent_from_checkpoint(checkpoint, device=device)
    return checkpoint, agent, model_type, metadata


def available_actions(knowledge_graph, exhibit, facts_mentioned, options, subactions):
    available_options = list(options)
    available_subactions = {option: list(subactions[option]) for option in options}
    facts = knowledge_graph.get_exhibit_facts(exhibit)
    mentioned = facts_mentioned.get(exhibit, set())
    remaining = [fact for fact in facts if knowledge_graph.extract_fact_id(fact) not in mentioned]
    if not remaining and "ExplainNewFact" in available_subactions.get("Explain", []):
        available_subactions["Explain"].remove("ExplainNewFact")
    if not mentioned and "RepeatFact" in available_subactions.get("Explain", []):
        available_subactions["Explain"].remove("RepeatFact")
    available_options = [option for option in available_options if available_subactions.get(option)]
    return available_options, available_subactions


def build_grounded_prompt(action: str, message: str, exhibit: str, knowledge_graph) -> str:
    facts = knowledge_graph.get_exhibit_facts(exhibit)
    context = "\n".join(f"- {knowledge_graph.strip_fact_id(fact)}" for fact in facts)
    return f"""You are a concise museum guide. Follow the dialogue action exactly.
Use only the supplied knowledge-graph facts; never invent missing information.

Selected dialogue action: {action}
Exhibit: {exhibit}
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
    parser.add_argument("--message")
    parser.add_argument("--exhibit", default="King_Caspar")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--knowledge-graph", type=Path, default=HERE / "knowledge_graph.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Only load and validate the checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Select an action and print the prompt without calling an LLM.")
    args = parser.parse_args()

    checkpoint, agent, model_type, metadata = load_policy(args.checkpoint, args.device)
    if args.smoke:
        print(f"checkpoint_ok model={model_type} state_dim={metadata['state_dim']} episodes={checkpoint.get('total_episodes')}")
        return
    if not args.message:
        parser.error("--message is required unless --smoke is used")

    knowledge_graph = SimpleKnowledgeGraph(str(args.knowledge_graph))
    facts_mentioned = defaultdict(set)
    action_counts = defaultdict(int)
    action_labels = [item for option in metadata["options"] for item in metadata["subactions"][option]]
    state = build_state(
        user_message=args.message,
        exhibit=args.exhibit,
        dialogue_history=[],
        knowledge_graph=knowledge_graph,
        action_labels=action_labels,
        facts_mentioned=facts_mentioned,
        action_counts=action_counts,
        turn_number=0,
        projection_matrix=get_projection_matrix(),
        include_availability=True,
        include_response_type=False,
        response_type=predict_response_type(args.message),
    )
    if len(state) != metadata["state_dim"]:
        raise RuntimeError(f"State dimension mismatch: built {len(state)}, checkpoint expects {metadata['state_dim']}")
    options, subactions = available_actions(
        knowledge_graph, args.exhibit, facts_mentioned, metadata["options"], metadata["subactions"]
    )
    selected = agent.select_action(state, options, subactions, deterministic=True)
    action = selected["flat_action_name"]
    prompt = build_grounded_prompt(action, args.message, args.exhibit, knowledge_graph)
    print(prompt if args.dry_run else generate(prompt, args.model))


if __name__ == "__main__":
    main()
