# Final checkpoints

This directory contains exactly the five final learner-seed policies used for the
reproducibility study. Each `checkpoint.pt` is the inference checkpoint saved after
500 episodes; the adjacent `metadata.json` records the seed and training settings.

The HybridSimulator seed was fixed at 42 across runs while learner seeds were
1, 2, 3, 4, and 5. Checkpoint loading can be verified without an LLM:

```bash
python -m conversational_agents.rl_assisted.runtime --smoke
```

