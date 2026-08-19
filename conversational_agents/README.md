# Conversational-agent study conditions

Both conditions use the same local museum knowledge graph and an LLM to realize a
grounded response.

- `baseline/`: KG facts are placed directly into the baseline guide prompt.
- `rl_assisted/`: the final seed-1 policy first selects a dialogue action; the LLM
  realizes that action using KG facts. It loads `checkpoints/seed_1/checkpoint.pt`
  and imports shared inference classes from `pipeline/`; training is not duplicated.

Prompt-only checks do not make network calls:

```bash
python -m conversational_agents.baseline.agent --message "What am I looking at?" --dry-run
python -m conversational_agents.rl_assisted.runtime --smoke
```

For full generation, copy `.env.example` to `.env`, set an API key for an
OpenAI-compatible endpoint, and omit `--dry-run`.

