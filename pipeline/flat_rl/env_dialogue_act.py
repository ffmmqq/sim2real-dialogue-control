"""
Flat Dialogue Environment with Dialogue Act State Representation (H4)

Wraps H5StateAblationEnv to provide flat action space for MDP training
while maintaining dialogue act classification state representation.
"""

from typing import List, Tuple
from gymnasium import spaces
import copy

from pipeline.environment.h4_env import H5StateAblationEnv


class FlatDialogueActEnv:
    """
    Flat action space wrapper for H5StateAblationEnv (dialogue act state representation).
    
    Wraps H5StateAblationEnv to expose a flat action space consisting solely of
    the primitive subactions, while maintaining the dialogue act state representation.
    """

    def __init__(self, base_env: H5StateAblationEnv):
        """
        Initialize flat dialogue act environment from base H5StateAblationEnv.
        
        Args:
            base_env: Pre-initialized H5StateAblationEnv instance
        """
        # Store reference to base environment
        self._base_env = base_env
        
        # Delegate all attribute access to base environment
        self.observation_space = base_env.observation_space
        self.options = base_env.options
        self.subactions = base_env.subactions
        self.exhibit_keys = base_env.exhibit_keys
        self.simulator = base_env.simulator
        self.n_exhibits = base_env.n_exhibits
        
        # Delegate methods that might be called
        self._get_available_options = base_env._get_available_options
        self._get_available_subactions = base_env._get_available_subactions
        
        # Build deterministic flat action ordering
        self.flat_actions: List[Tuple[str, str]] = []
        for option in self.options:
            for subaction in self.subactions[option]:
                self.flat_actions.append((option, subaction))

        # Replace hierarchical action space with a single discrete one
        self.action_space = spaces.Discrete(len(self.flat_actions))

        # Track usage for diagnostics
        self.flat_action_counts = {self._format_flat_name(idx): 0 for idx in range(len(self.flat_actions))}

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def get_flat_action_names(self) -> List[str]:
        """Return human-readable names for each flat action."""
        return [self._format_flat_name(idx) for idx in range(len(self.flat_actions))]

    def get_flat_action_mask(self) -> List[int]:
        """
        Build a mask (1 = available, 0 = masked) for each flat action based on
        the environment's option/subaction availability logic.
        """
        mask = []
        available_options = set(self._base_env._get_available_options())

        for option, subaction in self.flat_actions:
            if option not in available_options:
                mask.append(0)
                continue

            available_subactions = self._base_env._get_available_subactions(option)
            mask.append(1 if subaction in available_subactions else 0)

        return mask

    def step(self, action_index: int, post_action_callback=None):
        """
        Execute a flat action by mapping it back to hierarchical indices and
        delegating to the parent environment's `step`.
        """
        option_name, subaction_name = self._decode_flat_action(action_index)

        available_options = self._base_env._get_available_options()
        if not available_options:
            # Fallback to default parent behaviour
            return self._base_env.step(
                {"option": 0, "subaction": 0, "terminate_option": False},
                post_action_callback=post_action_callback,
            )

        if option_name not in available_options:
            option_name = available_options[0]

        available_subactions = self._base_env._get_available_subactions(option_name)
        if not available_subactions:
            # Should not happen, but fallback to the first option's first subaction
            fallback_option = available_options[0]
            option_name = fallback_option
            available_subactions = self._base_env._get_available_subactions(fallback_option)

        if subaction_name not in available_subactions:
            subaction_name = available_subactions[0]

        option_idx = available_options.index(option_name)
        subaction_idx = available_subactions.index(subaction_name)

        action_dict = {
            "option": option_idx,
            "subaction": subaction_idx,
            "terminate_option": False,
        }

        # Track usage
        flat_name = self._format_flat_name_from_parts(option_name, subaction_name)
        self.flat_action_counts[flat_name] = self.flat_action_counts.get(flat_name, 0) + 1

        obs, reward, done, truncated, info = self._base_env.step(
            action_dict, post_action_callback=post_action_callback
        )
        info = copy.copy(info)
        info["flat_action_index"] = action_index
        info["flat_action_name"] = flat_name
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self._base_env.reset(seed=seed, options=options)
        # Reset action counts if they exist (may not be initialized on first reset)
        if hasattr(self, 'flat_action_counts'):
            for flat_name in self.flat_action_counts:
                self.flat_action_counts[flat_name] = 0
        return obs, info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _decode_flat_action(self, action_index: int) -> Tuple[str, str]:
        if action_index < 0 or action_index >= len(self.flat_actions):
            return self.flat_actions[0]
        return self.flat_actions[action_index]

    def _format_flat_name(self, idx: int) -> str:
        option, subaction = self.flat_actions[idx]
        return self._format_flat_name_from_parts(option, subaction)

    @staticmethod
    def _format_flat_name_from_parts(option: str, subaction: str) -> str:
        return f"{option}/{subaction}"
