# sim2real-dialogue-control


The repository exposes the research chain:

```text
Hybrid visitor simulator -> five-seed RL training -> trained policy checkpoint
                                                     |
                                                     v
visitor -> policy dialogue action -> KG-grounded LLM -> RL-assisted response
visitor ---------------------------> KG-grounded LLM -> baseline response
```

## Repository map

- `simulator/` — the final HybridSimulator and supporting visitor models used in training.
- `pipeline/` — final training, evaluation, analysis, and reproducibility code.
- `checkpoints/` — exactly five final inference checkpoints (learner seeds 1–5).
- `results/seed_1/` — rollout, metrics, and summary from the final seed-1 checkpoint.
- `conversational_agents/baseline/` — runnable baseline LLM + KG condition.
- `conversational_agents/rl_assisted/` — runnable LLM + KG + policy-inference condition.
- `data/measurements/` — anonymous public derived measurements only.
- `configs/` and `scripts/` — frozen configurations and convenience commands.

No VR application codeare include d. No raw human data,
participant identifiers, raw gaze streams, audio/video, raw transcripts, or raw
questionnaires are included.

## Quick start

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Validate a released policy without an API call or model download.
python -m conversational_agents.rl_assisted.runtime --smoke

# Inspect the baseline grounded prompt without an API call.
python -m conversational_agents.baseline.agent \
  --message "What is special about this painting?" --dry-run

# Validate release boundaries and large files.
python scripts/check_release.py
```

The DialogueBERT encoder is downloaded by Transformers on first state-building
inference unless `HRL_BERT_MODEL` points to a local compatible model.

## Reproduce training

The paper configuration uses a fixed HybridSimulator seed of 42, learner seeds
1–5, 500 episodes, 50 maximum turns, a 151-dimensional state, eight flat actions,
and a local OpenAI-compatible Llama 3.1 8B endpoint.

```bash
cp .env.example .env
# export the values from .env in your preferred way, then start the local LLM.
./scripts/train_all_seeds.sh --dry-run
./scripts/train_all_seeds.sh
```

`--dry-run` prints and audits all five commands without starting training. New
outputs go to ignored `runs/`; committed checkpoints are never overwritten.

## Reproduce seed-1 evaluation

An offline template-mode smoke run:

```bash
./scripts/evaluate_seed1.sh \
  --template-mode --sessions-per-profile 1 --max-turns 10
```

The full evaluation uses the local Llama endpoint and the values frozen in
`configs/evaluation_seed1.json`:

```bash
./scripts/evaluate_seed1.sh
```

## User-study conditions

The baseline and RL-assisted conditions share the same knowledge source and LLM
response realization. The experimental difference is dialogue control: baseline
prompting has no learned policy, while the RL-assisted condition loads an already
trained policy checkpoint and uses its selected dialogue action to condition the
grounded LLM response. The user-study deployment does not train or update the policy.

## Data boundary

Only approved derived columns are released: gaze FC, TFT, MDT, FRA, NSL, GTE,
MSD, K, AR, SGV; conversation NAT, NUT, MARL, MURL, TCD, CDP, PC; and multimodal
alignment MCFR-A, MCSL-A. The anonymous export can be rebuilt inside the controlled
private environment with `scripts/build_public_measurements.py`; its private input
must never be copied into this repository.

## License

See `LICENSE`. Third-party model and service terms remain applicable to downloaded
models and external LLM endpoints.

