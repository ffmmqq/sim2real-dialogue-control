# sim2real-dialogue-control

Code, trained policies, evaluation results, and reproducibility materials for **From Simulated to Real Visitors: RL-Assisted Dialogue Control in a VR Museum**.

The repository implements the following research pipeline:

```text
Hybrid visitor simulator -> five-seed RL training -> trained policy checkpoint
                                                     |
                                                     v
visitor -> policy dialogue action -> KG-grounded LLM -> RL-assisted response
visitor ---------------------------> KG-grounded LLM -> baseline response
```

## Repository structure

- `simulator/` — the final HybridSimulator and supporting visitor models used for RL training and evaluation.
- `pipeline/` — training, evaluation, analysis, and reproducibility code.
- `checkpoints/` — final trained policies for learner seeds 1–5.
- `results/seed_1/` — simulator rollouts, metrics, and summaries generated with the final seed-1 policy.
- `conversational_agents/baseline/` — the baseline conversational agent using an LLM and knowledge graph.
- `conversational_agents/rl_assisted/` — the RL-assisted conversational agent using an LLM, knowledge graph, and trained policy.
- `data/measurements/` — anonymized derived gaze, conversation, and multimodal-alignment measurements.
- `configs/` — configurations for training and frozen-policy evaluation.
- `scripts/` — commands for reproducing training, evaluation, and measurement preparation.

## Quick start

Python 3.10–3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the released seed-1 policy:

```bash
python -m conversational_agents.rl_assisted.runtime --smoke
```

Inspect a knowledge-graph-grounded baseline prompt:

```bash
python -m conversational_agents.baseline.agent \
  --message "What is special about this painting?" \
  --dry-run
```

The DialogueBERT encoder is downloaded by Transformers during the first state-building inference. A compatible local model can be specified with `HRL_BERT_MODEL`.

## Reproduce training

The training configuration uses:

- HybridSimulator with simulator seed 42
- Learner seeds 1–5
- 500 episodes per seed
- 50 maximum turns per episode
- A 151-dimensional state representation
- Eight flat dialogue actions
- A local OpenAI-compatible Llama 3.1 8B endpoint

Review the five training commands:

```bash
./scripts/train_all_seeds.sh --dry-run
```

Run training for all five learner seeds:

```bash
cp .env.example .env
./scripts/train_all_seeds.sh
```

Training outputs are written to `runs/`.

## Reproduce seed-1 evaluation

Run a small offline template-mode evaluation:

```bash
./scripts/evaluate_seed1.sh \
  --template-mode \
  --sessions-per-profile 1 \
  --max-turns 10
```

Run the full frozen-policy evaluation using the configuration in `configs/evaluation_seed1.json`:

```bash
./scripts/evaluate_seed1.sh
```

The evaluation covers the Explorer, Focused, and Impatient simulator profiles and records turn-level rollouts, action distributions, transitions, dwell metrics, completion statistics, and state-consistency checks.

## Conversational-agent conditions

### Baseline

The baseline condition uses an LLM and the museum knowledge graph to generate grounded responses to visitor messages.

```text
Visitor message -> Knowledge graph -> LLM -> Response
```

### RL-assisted

The RL-assisted condition loads a trained policy checkpoint and uses it to select a dialogue action before generating the final knowledge-graph-grounded response.

```text
Visitor message -> State representation -> Trained policy
                -> Dialogue action -> Knowledge graph -> LLM -> Response
```

The trained policy remains fixed during conversational-agent deployment.

## Measurements

The public measurement tables are organized into three groups.

### Gaze measurements

- FC
- TFT
- MDT
- FRA
- NSL
- GTE
- MSD
- K
- AR
- SGV

### Conversation measurements

- NAT
- NUT
- MARL
- MURL
- TCD
- CDP
- PC

### Multimodal-alignment measurements

- MCFR-A
- MCSL-A

The tables use anonymous release-specific sample identifiers. Measurement definitions and units follow the accompanying paper.

## Checkpoints

The repository provides one final inference checkpoint for each learner seed:

```text
checkpoints/
├── seed_1/checkpoint.pt
├── seed_2/checkpoint.pt
├── seed_3/checkpoint.pt
├── seed_4/checkpoint.pt
└── seed_5/checkpoint.pt
```

Each checkpoint was produced after 500 training episodes. Corresponding metadata and SHA256 checksums are provided for verification.

## License

See `LICENSE` for the repository license. Third-party model and service terms apply to downloaded models and external LLM endpoints.
