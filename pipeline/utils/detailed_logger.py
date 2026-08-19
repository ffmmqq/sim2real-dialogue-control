"""
Detailed Episode Logger for HRL Training

This module provides comprehensive logging of all episode details including:
- Prompts (agent and simulator)
- Dialogues (full conversation history)
- States (observations, actions, rewards)
- Parameterization metrics (reward breakdowns, action distributions)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np


class DetailedEpisodeLogger:
    """Logs detailed episode information for analysis and debugging"""
    
    def __init__(self, experiment_dir: Path, reward_mode: str = "baseline"):
        """
        Initialize detailed logger.
        
        Args:
            experiment_dir: Path to experiment directory (e.g., training_logs/experiments/exp_001)
            reward_mode: "baseline" (Eng+Nov only) or "augmented" (all components)
        """
        self.experiment_dir = Path(experiment_dir)
        self.detailed_logs_dir = self.experiment_dir / "detailed_logs"
        self.detailed_logs_dir.mkdir(parents=True, exist_ok=True)
        self.reward_mode = reward_mode
        
        # Current episode data
        self.current_episode = None
        self.episode_data = None
        
    def start_episode(
        self,
        episode_num: int,
        profile: Optional[str] = None,
        persona: Optional[str] = None,
        starting_exhibit: Optional[str] = None,
    ):
        """Initialize logging for a new episode"""
        self.current_episode = episode_num
        self.episode_data = {
            "episode_number": episode_num,
            "timestamp": datetime.now().isoformat(),
            "profile": profile,
            "persona": persona,
            "starting_exhibit": starting_exhibit,
            "turns": []
        }
        
    def log_turn(self, 
                 turn_num: int,
                 state: np.ndarray,
                 action: Dict[str, Any],
                 reward: float,
                 info: Dict[str, Any],
                 next_state: Optional[np.ndarray] = None,
                 agent_prompt: Optional[str] = None,
                 agent_system_prompt: Optional[str] = None,
                 simulator_prompt: Optional[str] = None,
                 simulator_system_prompt: Optional[str] = None,
                 user_response: Optional[Dict[str, Any]] = None):
        """
        Log detailed information for a single turn.
        
        Args:
            turn_num: Turn number in episode
            state: Current observation state
            action: Action taken (option, subaction)
            reward: Reward received
            info: Info dict from environment
            next_state: Next observation state
            agent_prompt: Full prompt sent to agent LLM
            agent_system_prompt: System prompt for agent LLM
            simulator_prompt: Full prompt sent to simulator LLM
            simulator_system_prompt: System prompt for simulator LLM
            user_response: User response dict from simulator
        """
        if self.episode_data is None:
            return  # Episode not started
        
        turn_data = {
            "turn_number": turn_num,
            "state": state.tolist() if isinstance(state, np.ndarray) else state,
            "action": {
                "option": action.get("option"),
                "option_name": action.get("option_name"),
                "subaction": action.get("subaction"),
                "subaction_name": action.get("subaction_name"),
                "option_index": action.get("option", 0),
                "subaction_index": action.get("subaction", 0),
                "flat_action": action.get("flat_action"),
                "flat_action_name": action.get("flat_action_name"),
            },
            "next_state": next_state.tolist() if isinstance(next_state, np.ndarray) and next_state is not None else next_state,
            "reward": {
                "total": reward,
                # Baseline components (always present)
                "engagement": info.get("reward_engagement", 0.0),
                "novelty": info.get("reward_novelty", 0.0),
                # Augmented components (only in H2 mode)
                "responsiveness": info.get("reward_responsiveness", 0.0),
                "conclude": info.get("reward_conclude", 0.0),
                "transition_insufficiency": info.get("reward_transition_insufficiency", 0.0),
                "transition_exploration": info.get("reward_transition_exploration", 0.0),
                "question_asking": info.get("reward_question_asking", 0.0),
            },
            "visitor_state": user_response.get("visitor_state", "unknown") if user_response else "unknown",
            "dialogue": {
                "agent_utterance": info.get("agent_utterance", ""),
                "user_utterance": user_response.get("utterance", "") if user_response else "",
                "response_type": user_response.get("response_type", "unknown") if user_response else "unknown"
            },
            "prompts": {
                "agent_prompt": agent_prompt,
                "agent_system_prompt": agent_system_prompt,
                "simulator_prompt": simulator_prompt,
                "simulator_system_prompt": simulator_system_prompt
            },
            "facts": {
                "facts_shared_this_turn": info.get("facts_shared", 0),
                "new_fact_ids": info.get("facts_mentioned_in_utterance", []),
                "hallucinated_fact_ids": info.get("hallucinated_facts", [])
            },
            "context": {
                "current_exhibit": info.get("current_exhibit", "Unknown"),
                "target_exhibit": info.get("target_exhibit"),
                "current_exhibit_completion": info.get("current_exhibit_completion", 0.0),
                "dwell": info.get("dwell", 0.0),
                "option": info.get("option", "Unknown"),
                "subaction": info.get("subaction", "Unknown"),
                "turns_in_option": info.get("turns_in_option", 0),
                "current_option": info.get("current_option"),
                "available_options": info.get("available_options", []),
                "available_subactions": info.get("available_subactions", [])
            },
            "dialoguebert": info.get("dialoguebert_insights", {}),
            "policy": {
                "sampling_temperature": action.get("sampling_temperature"),
                "selected_action_probability": action.get("selected_action_probability"),
                "state_value": action.get("state_value"),
                "action_logits": action.get("action_logits"),
                "action_probabilities": action.get("action_probabilities"),
            },
            "performance": {
                "agent_llm_time": info.get("agent_llm_time", 0.0),
                "simulator_llm_time": user_response.get("simulator_llm_time", 0.0) if user_response else 0.0
            },
            "timing_validation": info.get("timing_validation", {}),
        }
        
        self.episode_data["turns"].append(turn_data)
        
    def end_episode(self, episode_reward: float, episode_stats: Optional[Dict[str, Any]] = None):
        """
        Finalize and save episode log.
        
        Args:
            episode_reward: Total reward for episode
            episode_stats: Additional episode statistics
        """
        if self.episode_data is None:
            return
        
        self.episode_data["episode_reward"] = episode_reward
        self.episode_data["total_turns"] = len(self.episode_data["turns"])
        if episode_stats:
            self.episode_data["stats"] = episode_stats
        
        # Save to file
        episode_dir = self.detailed_logs_dir / f"episode_{self.current_episode:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main episode log
        episode_file = episode_dir / "episode_log.json"
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(self.episode_data, f, indent=2, ensure_ascii=False)
        
        # Save readable dialogue summary
        self._save_dialogue_summary(episode_dir)
        
        # Reset for next episode
        self.episode_data = None
        
    def _save_dialogue_summary(self, episode_dir: Path):
        """Save a human-readable dialogue summary"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"EPISODE {self.current_episode} DIALOGUE SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Total Turns: {len(self.episode_data['turns'])}")
        summary_lines.append(f"Episode Reward: {self.episode_data.get('episode_reward', 0.0):.3f}")
        summary_lines.append(f"Profile: {self.episode_data.get('profile', 'Unknown')}")
        summary_lines.append("")
        
        for turn in self.episode_data["turns"]:
            summary_lines.append(f"\n--- TURN {turn['turn_number']} ---")
            
            # Show option and subaction names instead of numeric indices
            option_name = turn['context'].get('option', 'Unknown')
            subaction_name = turn['context'].get('subaction', 'Unknown')
            summary_lines.append(f"Option: {option_name}")
            summary_lines.append(f"Subaction: {subaction_name}")
            
            summary_lines.append(f"Exhibit: {turn['context']['current_exhibit']}")
            
            # Show reward components based on reward_mode
            reward = turn['reward']
            reward_str = f"Reward: {reward['total']:.3f} (Eng: {reward['engagement']:.3f}, Nov: {reward['novelty']:.3f}"
            # Only show augmented components in augmented mode
            if self.reward_mode == "augmented":
                if reward.get('responsiveness', 0.0) != 0.0:
                    reward_str += f", Resp: {reward['responsiveness']:.3f}"
                if reward.get('conclude', 0.0) != 0.0:
                    reward_str += f", Concl: {reward['conclude']:.3f}"
                if reward.get('transition_insufficiency', 0.0) != 0.0:
                    reward_str += f", Trans: {reward['transition_insufficiency']:.3f}"
                if reward.get('transition_exploration', 0.0) != 0.0:
                    reward_str += f", Expl: {reward['transition_exploration']:.3f}"
                if reward.get('question_asking', 0.0) != 0.0:
                    reward_str += f", Ask: {reward['question_asking']:.3f}"
            reward_str += ")"
            summary_lines.append(reward_str)
            
            # Show visitor state for debugging
            visitor_state = turn.get('visitor_state', 'unknown')
            if visitor_state and visitor_state != 'unknown':
                summary_lines.append(f"Visitor State: {visitor_state}")
            
            # Add agent prompt for debugging
            agent_prompt = turn.get('prompts', {}).get('agent_prompt')
            if agent_prompt:
                summary_lines.append(f"\n[AGENT PROMPT]")
                summary_lines.append(agent_prompt)
                summary_lines.append("[END PROMPT]\n")
            
            if turn['dialogue']['agent_utterance']:
                summary_lines.append(f"Agent: {turn['dialogue']['agent_utterance']}")
            if turn['dialogue']['user_utterance']:
                summary_lines.append(f"User: {turn['dialogue']['user_utterance']}")
            
            if turn['facts']['new_fact_ids']:
                summary_lines.append(f"New Facts: {turn['facts']['new_fact_ids']}")
        
        summary_file = episode_dir / "dialogue_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
