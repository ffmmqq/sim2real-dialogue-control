"""
Recovery Metrics for H3: Simulator Signal Quality + Recovery Behavior

This module tracks recovery behavior metrics:
- Recovery action rate: Proportion of correct recovery actions after negative states
- Clarification rate: ClarifyFact usage after CONFUSED states
- Recovery success rate: Fraction of recovery attempts that succeed

These metrics test whether explicit negative state modeling (State Machine)
teaches context-appropriate repair strategies that Sim8's ambiguous signals cannot.

Usage:
    tracker = RecoveryMetricsTracker()
    
    # During episode
    tracker.record_state_action(visitor_state="OVERLOADED", action="AskQuestion")
    tracker.record_recovery_outcome(success=True)
    
    # After episode
    metrics = tracker.compute_metrics()
    print(f"Recovery action rate: {metrics['recovery_action_rate']:.2%}")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RecoveryEvent:
    """A single recovery attempt."""
    visitor_state: str
    action_taken: str
    subaction_taken: str
    was_recovery_action: bool  # Was the action appropriate for the state?
    recovery_succeeded: bool   # Did the visitor return to ENGAGED?
    dwell_before: float
    dwell_after: float


class RecoveryMetricsTracker:
    """
    Tracks recovery behavior metrics across episodes.
    
    Recovery actions per state:
    - OVERLOADED: AskQuestion or OfferTransition
    - FATIGUED: AskQuestion or OfferTransition  
    - CONFUSED: ClarifyFact
    - CURIOUS: Explain (with fact)
    - BORED_OF_TOPIC: Explain (different topic) or OfferTransition
    - READY_TO_MOVE: OfferTransition
    - DISENGAGED: OfferTransition
    """
    
    # Define correct recovery actions for each negative state
    RECOVERY_ACTIONS = {
        "OVERLOADED": {"AskQuestion", "OfferTransition"},
        "FATIGUED": {"AskQuestion", "OfferTransition"},
        "CONFUSED": {"ClarifyFact"},
        "CURIOUS": {"Explain"},
        "BORED_OF_TOPIC": {"Explain", "OfferTransition"},
        "READY_TO_MOVE": {"OfferTransition"},
        "DISENGAGED": {"OfferTransition"},
    }
    
    # States that require recovery
    NEGATIVE_STATES = {
        "OVERLOADED", "FATIGUED", "CONFUSED", "CURIOUS", 
        "BORED_OF_TOPIC", "READY_TO_MOVE", "DISENGAGED"
    }
    
    def __init__(self):
        self.events: List[RecoveryEvent] = []
        self.current_state: Optional[str] = None
        self.current_dwell: float = 0.0
        
        # Per-state counters
        self.state_occurrences: Dict[str, int] = defaultdict(int)
        self.correct_recovery_attempts: Dict[str, int] = defaultdict(int)
        self.successful_recoveries: Dict[str, int] = defaultdict(int)
    
    def reset(self):
        """Reset for new episode."""
        self.events.clear()
        self.current_state = None
        self.current_dwell = 0.0
        self.state_occurrences.clear()
        self.correct_recovery_attempts.clear()
        self.successful_recoveries.clear()
    
    def update_state(self, visitor_state: str, dwell: float):
        """Update current visitor state (call before action)."""
        self.current_state = visitor_state.upper() if visitor_state else None
        self.current_dwell = dwell
        
        if self.current_state in self.NEGATIVE_STATES:
            self.state_occurrences[self.current_state] += 1
    
    def record_action(
        self,
        option: str,
        subaction: str,
        new_visitor_state: str,
        new_dwell: float
    ):
        """
        Record an action taken and its outcome.
        
        Args:
            option: Option name (Explain, AskQuestion, etc.)
            subaction: Subaction name (ExplainNewFact, ClarifyFact, etc.)
            new_visitor_state: Visitor state after the action
            new_dwell: Dwell time after the action
        """
        if self.current_state not in self.NEGATIVE_STATES:
            return  # Only track recovery from negative states
        
        # Check if action was appropriate for the state
        correct_actions = self.RECOVERY_ACTIONS.get(self.current_state, set())
        
        # For subaction-specific checks (e.g., ClarifyFact specifically)
        was_recovery_action = (
            option in correct_actions or
            subaction in correct_actions
        )
        
        # Check if recovery succeeded (returned to ENGAGED or better)
        new_state_upper = new_visitor_state.upper() if new_visitor_state else ""
        recovery_succeeded = new_state_upper in {"ENGAGED", "HIGHLY_ENGAGED"}
        
        # Record event
        event = RecoveryEvent(
            visitor_state=self.current_state,
            action_taken=option,
            subaction_taken=subaction,
            was_recovery_action=was_recovery_action,
            recovery_succeeded=recovery_succeeded,
            dwell_before=self.current_dwell,
            dwell_after=new_dwell
        )
        self.events.append(event)
        
        # Update counters
        if was_recovery_action:
            self.correct_recovery_attempts[self.current_state] += 1
            if recovery_succeeded:
                self.successful_recoveries[self.current_state] += 1
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute recovery metrics for the episode.
        
        Returns:
            Dictionary with recovery metrics
        """
        total_negative_states = sum(self.state_occurrences.values())
        total_correct_attempts = sum(self.correct_recovery_attempts.values())
        total_successful = sum(self.successful_recoveries.values())
        
        # Overall metrics
        recovery_action_rate = (
            total_correct_attempts / total_negative_states 
            if total_negative_states > 0 else 0.0
        )
        recovery_success_rate = (
            total_successful / total_correct_attempts
            if total_correct_attempts > 0 else 0.0
        )
        
        # ClarifyFact-specific (for CONFUSED state)
        confused_occurrences = self.state_occurrences.get("CONFUSED", 0)
        clarify_attempts = self.correct_recovery_attempts.get("CONFUSED", 0)
        clarification_rate = (
            clarify_attempts / confused_occurrences
            if confused_occurrences > 0 else 0.0
        )
        
        # Per-state breakdown
        per_state_metrics = {}
        for state in self.NEGATIVE_STATES:
            occurrences = self.state_occurrences.get(state, 0)
            attempts = self.correct_recovery_attempts.get(state, 0)
            successes = self.successful_recoveries.get(state, 0)
            
            per_state_metrics[state] = {
                "occurrences": occurrences,
                "correct_attempts": attempts,
                "successes": successes,
                "attempt_rate": attempts / occurrences if occurrences > 0 else 0.0,
                "success_rate": successes / attempts if attempts > 0 else 0.0
            }
        
        return {
            "recovery_action_rate": recovery_action_rate,
            "recovery_success_rate": recovery_success_rate,
            "clarification_rate": clarification_rate,
            "total_negative_states": total_negative_states,
            "total_correct_attempts": total_correct_attempts,
            "total_successful_recoveries": total_successful,
            "per_state": per_state_metrics,
            "events": len(self.events)
        }
    
    def get_summary_string(self) -> str:
        """Get human-readable summary."""
        metrics = self.compute_metrics()
        return (
            f"Recovery Metrics:\n"
            f"  Action Rate: {metrics['recovery_action_rate']:.1%} "
            f"({metrics['total_correct_attempts']}/{metrics['total_negative_states']})\n"
            f"  Success Rate: {metrics['recovery_success_rate']:.1%} "
            f"({metrics['total_successful_recoveries']}/{metrics['total_correct_attempts']})\n"
            f"  Clarification Rate: {metrics['clarification_rate']:.1%}"
        )

