"""
Option Configurations for H6: Option Granularity

This module defines different option granularity configurations:
- MEDIUM_OPTIONS (4 options): Function-based design - Explain, AskQuestion, OfferTransition, Conclude
- COARSE_OPTIONS (2 options): Reward-aligned - Explain, Engage
- COARSE_3OPT_OPTIONS (3 options): Reward-aligned - Explain, Engage, Transition (DEPRECATED: no Conclude)
- COARSE_4OPT_OPTIONS (4 options): Reward-aligned with Conclude - Explain, Engage, Transition, Conclude

The Coarse-4opt configuration is recommended for H6 experiments:
- Explain (novelty-focused): ExplainNewFact only
- Engage (engagement-focused): Clarification and questions
- Transition (coverage-focused): SuggestMove only
- Conclude (termination): WrapUp

This allows fair comparison with the 4-option function-based baseline (MEDIUM).

Usage:
    from pipeline.agent.option_configs import get_option_config
    
    options, subactions = get_option_config("medium")      # Function-based baseline
    options, subactions = get_option_config("coarse_4opt") # Reward-aligned (recommended)
    options, subactions = get_option_config("coarse_3opt") # Deprecated (no Conclude)
    options, subactions = get_option_config("coarse")      # 2-option extreme
"""

from typing import Dict, List, Tuple


# ============================================================
# MEDIUM (4 OPTIONS) - Current Design
# ============================================================
MEDIUM_OPTIONS = ["Explain", "AskQuestion", "OfferTransition", "Conclude"]

MEDIUM_SUBACTIONS = {
    "Explain": ["ExplainNewFact", "RepeatFact", "ClarifyFact"],
    "AskQuestion": ["AskOpinion", "AskMemory", "AskClarification"],
    "OfferTransition": ["SuggestMove"],
    "Conclude": ["WrapUp"]
}


# ============================================================
# COARSE (2 OPTIONS) - Novelty vs Engagement Split
# ============================================================
COARSE_OPTIONS = ["Explain", "Engage"]

COARSE_SUBACTIONS = {
    # Novelty-focused: Only ExplainNewFact (introduces new KB content)
    "Explain": ["ExplainNewFact"],
    
    # Engagement-focused: Everything else (maintains/recovers engagement)
    "Engage": [
        "RepeatFact",      # From Explain - reinforcement
        "ClarifyFact",     # From Explain - clarification
        "AskOpinion",      # From AskQuestion
        "AskMemory",       # From AskQuestion
        "AskClarification",# From AskQuestion
        "SuggestMove",     # From OfferTransition
        "WrapUp"           # From Conclude
    ]
}

# Mapping from coarse subactions to original option context
# (useful for prompt construction that references original option semantics)
COARSE_SUBACTION_ORIGIN = {
    "ExplainNewFact": "Explain",
    "RepeatFact": "Explain",
    "ClarifyFact": "Explain",
    "AskOpinion": "AskQuestion",
    "AskMemory": "AskQuestion",
    "AskClarification": "AskQuestion",
    "SuggestMove": "OfferTransition",
    "WrapUp": "Conclude"
}


# ============================================================
# COARSE_3OPT (3 OPTIONS) - Transition as First-Class Option
# ============================================================
# This configuration separates Transition from Engage to ensure
# the agent can learn to transition between exhibits. In the 2-option
# coarse config, SuggestMove is buried as 1/7 subactions under Engage,
# resulting in only 0.7% transition probability per turn when Engage
# is selected ~5% of time. With 3 options, Transition gets equal
# exploration probability (~33%) at the option level.
#
# NOTE: This config lacks a Conclude option, causing episodes to run
# to max length (~50 turns). Use coarse_4opt for fair comparison with baseline.

COARSE_3OPT_OPTIONS = ["Explain", "Engage", "Transition"]

COARSE_3OPT_SUBACTIONS = {
    # Novelty-focused: Only ExplainNewFact (introduces new KB content)
    "Explain": ["ExplainNewFact"],
    
    # Engagement-focused: Clarification, questions, and wrap-up
    "Engage": [
        "RepeatFact",       # From Explain - reinforcement
        "ClarifyFact",      # From Explain - clarification
        "AskOpinion",       # From AskQuestion
        "AskMemory",        # From AskQuestion
        "AskClarification", # From AskQuestion
        "WrapUp"            # From Conclude
    ],
    
    # Coverage-focused: Transition to new exhibits
    "Transition": ["SuggestMove"]
}

# Mapping for coarse_3opt configuration
COARSE_3OPT_SUBACTION_ORIGIN = {
    "ExplainNewFact": "Explain",
    "RepeatFact": "Explain",
    "ClarifyFact": "Explain",
    "AskOpinion": "AskQuestion",
    "AskMemory": "AskQuestion",
    "AskClarification": "AskQuestion",
    "SuggestMove": "OfferTransition",
    "WrapUp": "Conclude"
}


# ============================================================
# COARSE_4OPT (4 OPTIONS) - Reward-Aligned with Conclude
# ============================================================
# This configuration is reward-aligned like coarse_3opt but includes
# a dedicated Conclude option for fair comparison with the 4-option
# function-based baseline (MEDIUM). Each option maps to a reward component:
# - Explain -> novelty reward
# - Engage -> engagement reward  
# - Transition -> coverage reward
# - Conclude -> episode termination (conclude bonus)

COARSE_4OPT_OPTIONS = ["Explain", "Engage", "Transition", "Conclude"]

COARSE_4OPT_SUBACTIONS = {
    # Novelty-focused: Only ExplainNewFact (introduces new KB content)
    "Explain": ["ExplainNewFact"],
    
    # Engagement-focused: Clarification and questions (maintains/recovers engagement)
    "Engage": [
        "RepeatFact",       # From Explain - reinforcement
        "ClarifyFact",      # From Explain - clarification
        "AskOpinion",       # From AskQuestion
        "AskMemory",        # From AskQuestion
        "AskClarification", # From AskQuestion
    ],
    
    # Coverage-focused: Transition to new exhibits
    "Transition": ["SuggestMove"],
    
    # Termination: Proper episode ending
    "Conclude": ["WrapUp"]
}

# Mapping for coarse_4opt configuration
COARSE_4OPT_SUBACTION_ORIGIN = {
    "ExplainNewFact": "Explain",
    "RepeatFact": "Explain",
    "ClarifyFact": "Explain",
    "AskOpinion": "AskQuestion",
    "AskMemory": "AskQuestion",
    "AskClarification": "AskQuestion",
    "SuggestMove": "OfferTransition",
    "WrapUp": "Conclude"
}


def get_option_config(granularity: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Get option configuration by granularity name.
    
    Args:
        granularity: One of:
            - "medium" or "4": Function-based 4 options (baseline)
            - "coarse" or "2": Reward-aligned 2 options
            - "coarse_3opt" or "3": Reward-aligned 3 options (no Conclude - deprecated)
            - "coarse_4opt": Reward-aligned 4 options with Conclude (recommended)
        
    Returns:
        Tuple of (options list, subactions dict)
    """
    granularity = granularity.lower()
    
    if granularity == "medium" or granularity == "4":
        return MEDIUM_OPTIONS.copy(), {k: v.copy() for k, v in MEDIUM_SUBACTIONS.items()}
    elif granularity == "coarse" or granularity == "2":
        return COARSE_OPTIONS.copy(), {k: v.copy() for k, v in COARSE_SUBACTIONS.items()}
    elif granularity == "coarse_3opt" or granularity == "3":
        return COARSE_3OPT_OPTIONS.copy(), {k: v.copy() for k, v in COARSE_3OPT_SUBACTIONS.items()}
    elif granularity == "coarse_4opt":
        return COARSE_4OPT_OPTIONS.copy(), {k: v.copy() for k, v in COARSE_4OPT_SUBACTIONS.items()}
    else:
        raise ValueError(f"Unknown granularity: {granularity}. Use 'medium', 'coarse', 'coarse_3opt', or 'coarse_4opt'.")


def get_subaction_origin(subaction: str, granularity: str = "coarse") -> str:
    """
    Get the original option that a subaction belongs to (useful for coarse configs).
    
    This is used to maintain prompt semantics when using coarse options.
    For example, when "RepeatFact" is selected under "Engage", we still want
    the prompt to reference the Explain context.
    
    Args:
        subaction: Subaction name
        granularity: Configuration name ("coarse", "coarse_3opt", or "coarse_4opt")
        
    Returns:
        Original option name
    """
    granularity_lower = granularity.lower()
    if granularity_lower == "coarse_3opt":
        return COARSE_3OPT_SUBACTION_ORIGIN.get(subaction, "Unknown")
    elif granularity_lower == "coarse_4opt":
        return COARSE_4OPT_SUBACTION_ORIGIN.get(subaction, "Unknown")
    return COARSE_SUBACTION_ORIGIN.get(subaction, "Unknown")


def is_novelty_action(subaction: str) -> bool:
    """Check if a subaction is novelty-focused (introduces new content)."""
    return subaction == "ExplainNewFact"


def is_engagement_action(subaction: str) -> bool:
    """Check if a subaction is engagement-focused (maintains/recovers engagement)."""
    return subaction in COARSE_SUBACTIONS.get("Engage", [])

