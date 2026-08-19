"""
README Templates for Major Results

Provides README templates for each model variation in major_results/.
Each template includes model description, key differences, configuration, and metrics.

Aligned with paper.tex hypotheses:
- Baseline: SMDP + baseline rewards (engagement + novelty)
- H1: Flat MDP + baseline rewards
- H2: Augmented rewards (2x2 design)
- H3: Simulator signal quality (State Machine vs Sim8)
- H4: State representation efficiency (compact 23-d or hybrid BERT)
- H5: Termination strategies (Fixed-3 vs Threshold)
"""

from pathlib import Path


def get_readme_template(model_name: str, metadata: dict = None) -> str:
    """
    Get README template for a model variation.
    
    Args:
        model_name: Model name (normalized)
        metadata: Optional metadata dictionary from training
        
    Returns:
        README content as string
    """
    templates = {
        'baseline': get_baseline_readme(metadata),
        'h1_flat_mdp': get_h1_readme(metadata),
        'h2_augmented_rewards': get_h2_readme(metadata),
        'h3_simulator_signal': get_h3_readme(metadata),
        'h4_state_efficiency': get_h4_readme(metadata),
        'h5_termination': get_h5_readme(metadata),
    }
    
    return templates.get(model_name, get_default_readme(model_name, metadata))


def get_baseline_readme(metadata: dict = None) -> str:
    """Baseline model README template."""
    return f"""# Baseline: SMDP (Hierarchical) with Baseline Rewards

## Model Description

This is the baseline hierarchical reinforcement learning (HRL) system using the Option-Critic 
architecture (Bacon et al., 2017) built on Actor-Critic with TD(0) learning.

### Architecture
- **Algorithm**: Option-Critic (hierarchical Actor-Critic)
- **State Representation**: 143-dimensional vector
  - `f_t`: Focus vector (6-d for exhibits)
  - `h_t`: Dialogue history (9-d: exhibit completion + option usage)
  - `i_t`: Intent embedding (64-d, projected from 768-d DialogueBERT)
  - `c_t`: Dialogue context (64-d, projected from 768-d DialogueBERT)
- **Options**: Explain, AskQuestion, OfferTransition, Conclude
- **Subactions**: 3 per option for Explain/AskQuestion, 1 for others
- **Termination**: Learned termination functions (Option-Critic style)

### Reward Function (Baseline)
**R_t = r^eng + r^nov** (Engagement + Novelty ONLY)

- Engagement: dwell_t × w_engagement
- Novelty: novelty_per_fact × |new_facts|

NO auxiliary components (responsiveness, transition, conclude, question-asking).

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `metrics/training_metrics.json` for:
- Episode returns and coverage
- Option usage and durations
- Policy and value losses
- Convergence analysis

## Comparison

This baseline is compared against:
- **H1**: Flat MDP (no hierarchical structure)
- **H2**: Augmented rewards
- **H3**: Different simulator (State Machine vs Sim8)
- **H4**: Compact state representation
- **H5**: Different termination strategies
"""


def get_h1_readme(metadata: dict = None) -> str:
    """H1: Flat MDP README template."""
    return f"""# H1: Flat MDP (No Hierarchical Structure)

## Model Description

This variant tests whether hierarchical option structure improves long-horizon behavior. 
It uses a flat Actor-Critic policy over all primitive actions without hierarchical structure.

### Architecture
- **Algorithm**: Standard Actor-Critic with TD(0) learning
- **State Representation**: Same as baseline (143-d)
- **Action Space**: Flat discrete space (8 actions = all subactions)
- **No Options**: Direct policy over primitive actions
- **No Termination Learning**: Actions selected directly each turn

### Reward Function
**Same as baseline**: R_t = r^eng + r^nov (Engagement + Novelty ONLY)

### Key Differences from Baseline
- **No hierarchical structure**: Single policy head instead of option-level + intra-option
- **No termination functions**: Actions selected directly, no option duration learning
- **Same state representation**: Uses same 143-d state
- **Same rewards**: Baseline reward function

### Hypothesis (H1)
An SMDP (hierarchical) policy will outperform an MDP (flat) policy on long-horizon 
objectives when both are trained under baseline reward conditions.

Expected outcomes:
- Higher episodic return with lower variance
- Longer coherent stretches under a chosen strategy
- Fewer needless switches between strategies
- Smoother training dynamics

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h1_metrics.json` for:
- Episode returns (compared to baseline)
- Coherent span lengths (consecutive turns under same strategy)
- Switch rate (switches per 100 turns)
- Policy entropy
- Training stability metrics
"""


def get_h2_readme(metadata: dict = None) -> str:
    """H2: Augmented Rewards README template."""
    return f"""# H2: Augmented Reward Function

## Model Description

This variant tests whether auxiliary reward components improve policy learning 
beyond the baseline reward function. Uses a 2x2 design testing both SMDP and MDP.

### Architecture
- **Same as Baseline**: Can use either SMDP (hierarchical) or MDP (flat)
- **State Representation**: Same as baseline (143-d)
- **Key Difference**: Augmented reward function

### Reward Function (Augmented)
**R_t = r^eng + r^nov + r^resp + r^conclude + r^trans + r^ask**

Baseline components:
- Engagement: dwell_t × w_engagement
- Novelty: novelty_per_fact × |new_facts|

Augmented components:
- **Responsiveness**: +w_resp for answering questions, -0.6*w_resp for deflecting
- **Transition**: Penalties for premature/delayed transitions, bonuses for appropriate timing
- **Conclude**: w_conclude × |exhibits_covered| for session-ending
- **Question-asking**: Hybrid reward for engagement probing

### Key Differences from Baseline
- **Responsiveness reward**: Incentivizes answering visitor questions
- **Transition shaping**: Guides transition timing based on coverage
- **Conclude bonus**: Rewards appropriate session conclusion
- **Question-asking incentive**: Encourages engagement probing

### Hypothesis (H2)
Adding auxiliary reward components improves policy learning beyond baseline for 
both SMDP and MDP architectures.

Expected outcomes:
- H2.1: Both architectures will show improved responsiveness metrics
- H2.2: SMDP will show reduced option collapse (less ExplainNewFact dominance)
- H2.3: AskQuestion usage will increase toward 8-12%
- H2.4: SMDP will show larger improvement magnitude than MDP

### Training Configuration
{_format_metadata(metadata)}

## 2x2 Experimental Design

| | Baseline Rewards | Augmented Rewards |
|---|---|---|
| **SMDP** | baseline/ | h2_augmented_rewards/ (smdp) |
| **MDP** | h1_flat_mdp/ | h2_augmented_rewards/ (mdp) |

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h2_metrics.json` for:
- Option usage distribution
- AskQuestion frequency
- Responsiveness rate
- Transition timing metrics
- Coverage breadth
"""


def get_h3_readme(metadata: dict = None) -> str:
    """H3: Simulator Signal Quality README template."""
    return f"""# H3: Simulator Signal Quality (State Machine vs Sim8)

## Model Description

This variant tests whether clearer simulator signals produce better policies. 
Compares State Machine (pedagogical, clear signals) vs Sim8 (realistic, ambiguous).

### Architecture
- **Same as Baseline**: SMDP (hierarchical)
- **State Representation**: Same as baseline (143-d)
- **Reward Function**: Baseline (engagement + novelty)
- **Key Difference**: Simulator used during training

### Simulator Comparison

| Aspect | State Machine | Sim8 |
|--------|---------------|------|
| **Dwell Ranges** | Non-overlapping | Overlapping |
| **State Labels** | Explicit (9 states) | Implicit |
| **Thresholds** | Stricter (3+) | Looser (4+) |
| **Anti-spam** | Recovery fatigue | None |
| **Signal Clarity** | High (pedagogical) | Low (realistic) |

### Key Differences from Baseline
- **State Machine**: Uses StateMachineSimulator with clear state labels
- **Sim8**: Uses Sim8Simulator with persona-conditioned, ambiguous responses
- **Same architecture and rewards**: Only simulator differs

### Hypothesis (H3)
A pedagogically-designed simulator with clearer reward signals (State Machine) 
will produce policies with greater strategy diversity and more effective recovery actions.

Expected outcomes:
- Higher recovery action rate when visitor is overloaded/fatigued
- Higher clarification rate when visitor is confused
- Greater strategy diversity (higher entropy of option distribution)
- More effective recovery success rates

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h3_metrics.json` for:
- Strategy diversity (entropy of option distribution)
- Recovery action rate
- Clarification rate
- Recovery success rate
- Cumulative dwell comparison
"""


def get_h4_readme(metadata: dict = None) -> str:
    """H4: State Representation Efficiency README template."""
    return f"""# H4: State Representation Efficiency

## Model Description

This variant tests whether compact state representations can maintain performance.
Two variants: (i) Dialogue-act classification (23-d), (ii) Hybrid BERT (143-d).

### Architecture
- **Same as Baseline**: SMDP (hierarchical)
- **Reward Function**: Baseline (engagement + novelty)
- **Key Difference**: State representation

### Variant A: Compact Dialogue-Act (23-d)
- `f_t`: Focus vector (6-d)
- `h_t`: Dialogue history (9-d)
- `a_t`: Dialogue act probabilities (8-d) - **replaces BERT embeddings**
- **Compression**: 84% reduction from 143-d

### Variant B: Hybrid BERT (143-d)
- `i_t`: Standard BERT for intent (no turn/role embeddings)
- `c_t`: DialogueBERT for context (with turn/role embeddings)
- **Rationale**: Multi-turn context benefits from turn-awareness

### Hypothesis (H4)
State representation can be made more efficient without sacrificing performance.

Expected outcomes:
- Compact (23-d): Comparable return and coverage with faster training
- Hybrid BERT: Improved dialogue coherence and context-dependent responsiveness

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h4_metrics.json` for:
- State dimension and compression ratio
- Episode returns (compared to baseline)
- Training time comparison
- Coverage metrics
- Responsiveness rate
"""


def get_h5_readme(metadata: dict = None) -> str:
    """H5: Termination Strategies README template."""
    return f"""# H5: Termination Strategies

## Model Description

This variant tests whether learned termination (baseline) outperforms simpler 
termination strategies for engagement-adaptive pacing.

### Architecture
- **Same as Baseline**: SMDP (hierarchical)
- **State Representation**: Same as baseline (143-d)
- **Reward Function**: Baseline (engagement + novelty)
- **Key Difference**: Termination mechanism

### Termination Strategies

| Strategy | Description |
|----------|-------------|
| **Learned** (baseline) | Option-Critic β(s) trained end-to-end |
| **Fixed-3** | Options always terminate after exactly 3 turns |
| **Threshold** | Options terminate when dwell drops below 0.5 |

### Key Differences from Baseline
- **Fixed-3**: Deterministic 3-turn option duration
- **Threshold**: Reactive termination based on engagement signal
- **Learned**: Adaptive termination conditioned on full state

### Hypothesis (H5)
Learned option termination (via Option-Critic β(s)) outperforms simpler 
termination strategies in engagement-adaptive pacing.

Expected outcomes:
- Learned termination shows positive correlation between option duration and dwell
- Earlier termination after detected intent changes
- Higher overall engagement compared to fixed/threshold strategies

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h5_metrics.json` for:
- Correlation between Explain duration and dwell time
- Time-to-termination after intent changes
- Termination rate statistics
- Overall engagement metrics
"""


def get_default_readme(model_name: str, metadata: dict = None) -> str:
    """Default README template for unknown model types."""
    return f"""# {model_name}

## Model Description

Custom model variant.

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`
"""


def _format_metadata(metadata: dict = None) -> str:
    """Format metadata as markdown table."""
    if not metadata:
        return "*(Configuration details will be added after training)*"
    
    lines = []
    for key, value in metadata.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) if lines else "*(No metadata available)*"
