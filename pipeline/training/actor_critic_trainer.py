"""
Actor-Critic Training Algorithm for Option-Critic

This implements the full Option-Critic algorithm (Bacon et al., 2017) with:
- TD updates for all three value functions: V(s), Q_Ω(s,ω), Q_U(s,ω,a)
- Correct advantage calculations: A_O(s,o) = Q_O(s,o) - V(s) and A_U(s,o,a) = Q_U(s,o,a) - Q_O(s,o)
- Policy gradients using the correct advantages

Algorithm:
1. Collect experience (s, ω, a, r, s')
2. Update all value functions via TD(0): V(s), Q_Ω(s,ω), Q_U(s,ω,a)
3. Compute advantages: A_O = Q_O - V, A_U = Q_U - Q_O
4. Update option policy using A_O
5. Update intra-option policies using A_U
6. Update termination functions using A_O

References:
- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Bacon et al. (2017): The Option-Critic Architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import os


class ActorCriticTrainer:
    """
    Actor-Critic trainer for Option-Critic agent.
    
    Implements the full Option-Critic algorithm (Bacon et al., 2017) with:
    - TD(0) updates for V(s), Q_Ω(s,ω), and Q_U(s,ω,a)
    - Option policy gradients using A_O(s,o) = Q_O(s,o) - V(s)
    - Intra-option policy gradients using A_U(s,o,a) = Q_U(s,o,a) - Q_O(s,o)
    - Termination function updates using A_O(s,o)
    """
    
    def __init__(
        self,
        agent,
        learning_rate: float = 1e-4,  # Restored to 1e-4 for better learning speed with stability measures
        lr_intra_option: float = None,  # H1 Advanced: Separate LR for intra-option policies (default: same as learning_rate)
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.15,  # Increased from 0.08 to 0.15 for better exploration (will decay)
        entropy_coef_option: float = None,  # H1 Advanced: Separate entropy for option policy (default: same as entropy_coef)
        entropy_coef_intra: float = None,  # H1 Advanced: Separate entropy for intra-option policies (default: same as entropy_coef)
        entropy_decay_start: int = 100,  # Episode to start entropy decay (after initial exploration)
        entropy_decay_end: int = 1000,  # Episode to end entropy decay (slower decay from 500)
        entropy_final: float = 0.01,  # Final entropy coefficient after decay (lower for exploitation)
        termination_reg: float = 0.01,
        max_grad_norm: float = 0.5,  # Reduced from 1.0 for stricter clipping
        value_clip: float = 10.0,  # Clip value predictions to prevent extreme values
        device: str = 'cpu',
        max_episodes: int = 2000,  # For learning rate scheduling (increased from 500)
        use_target_network: bool = True,  # Use target network for value stability
        target_update_interval: int = 20,  # Episodes between target network updates
        normalize_advantages: bool = True,  # Normalize advantages per batch
        intra_option_threshold: float = 0.1,  # Threshold for intra-option advantage termination signal
        intra_option_weight: float = 0.5,  # Weight for intra-option termination signal
        beta_supervision_weight: float = 0.0,  # H1 Advanced: Weight for heuristic termination supervision (0.0 = pure Option-Critic)
        adaptive_entropy: bool = False,  # H1 Phase 2: Enable OCI-aware adaptive entropy
        adaptive_entropy_threshold: float = 2.5,  # H1 Phase 2: OCI threshold for entropy boost
        adaptive_entropy_multiplier: float = 1.5,  # H1 Phase 2: Multiplier for entropy boost
        last_oci: float = 0.0  # H1 Phase 2: Last observed OCI (will be updated from training loop)
    ):
        """
        Initialize Actor-Critic trainer.
        
        Args:
            agent: ActorCriticAgent instance
            learning_rate: Initial learning rate (will decay with exponential schedule)
            gamma: Discount factor
            value_loss_coef: Value loss coefficient
            entropy_coef: Initial entropy regularization (will decay from this value)
            entropy_decay_start: Episode to start entropy decay
            entropy_decay_end: Episode to end entropy decay
            entropy_final: Final entropy coefficient after decay
            termination_reg: Termination regularization
            max_grad_norm: Gradient clipping threshold
            value_clip: Clip value predictions to prevent extreme values
            device: 'cpu' or 'cuda'
            max_episodes: Maximum episodes for learning rate scheduling
            use_target_network: Whether to use target network for value stability
            target_update_interval: Episodes between target network updates
            normalize_advantages: Whether to normalize advantages per batch
        """
        self.agent = agent
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        
        # H1 Advanced: Separate entropy coefficients
        self.entropy_coef_option = entropy_coef_option if entropy_coef_option is not None else entropy_coef
        self.entropy_coef_intra = entropy_coef_intra if entropy_coef_intra is not None else entropy_coef
        self.initial_entropy_coef_option = self.entropy_coef_option
        self.initial_entropy_coef_intra = self.entropy_coef_intra
        self.entropy_coef = entropy_coef  # Keep for backward compatibility
        self.initial_entropy_coef = entropy_coef
        self.entropy_decay_start = entropy_decay_start
        self.entropy_decay_end = entropy_decay_end
        self.entropy_final = entropy_final
        self.termination_reg = termination_reg
        self.max_grad_norm = max_grad_norm
        self.value_clip = value_clip
        self.device = device
        self.initial_lr = learning_rate
        self.max_episodes = max_episodes
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval
        self.normalize_advantages = normalize_advantages
        self.intra_option_threshold = intra_option_threshold
        self.intra_option_weight = intra_option_weight
        self.beta_supervision_weight = beta_supervision_weight  # H1 Advanced: heuristic termination supervision
        self.current_episode = 0
        
        # H1 Phase 2: Adaptive entropy control parameters
        self.adaptive_entropy = adaptive_entropy
        self.adaptive_entropy_threshold = adaptive_entropy_threshold
        self.adaptive_entropy_multiplier = adaptive_entropy_multiplier
        self.last_oci = last_oci
        
        # Subaction name mapping for intra-option advantage calculation
        # Map option names to their subaction name lists
        self.subaction_names = agent.subactions  # Dict mapping option names to subaction lists
        self.option_names = agent.options  # List of option names
        
        # Value normalization for stability
        self.value_running_mean = 0.0
        self.value_running_std = 1.0
        self.value_norm_momentum = 0.99
        
        # H1 Advanced: Separate learning rates for option vs intra-option policies
        # If lr_intra_option is specified, create separate parameter groups
        self.lr_intra_option = lr_intra_option if lr_intra_option is not None else learning_rate
        self.initial_lr_intra = self.lr_intra_option
        
        if lr_intra_option is not None and lr_intra_option != learning_rate:
            # Separate parameter groups for different learning rates
            # Option policy, termination functions, and value functions use learning_rate
            # Intra-option policies use lr_intra_option
            option_params = []
            intra_params = []
            value_params = []
            
            for name, param in self.agent.network.named_parameters():
                if 'intra_option_policies' in name:
                    intra_params.append(param)
                elif 'option_policy' in name or 'termination_functions' in name:
                    option_params.append(param)
                else:
                    # Value functions and encoder
                    value_params.append(param)
            
            # Create optimizer with separate parameter groups
            # Group 0: option params, Group 1: intra params, Group 2: value params
            self.optimizer = optim.Adam([
                {'params': option_params, 'lr': learning_rate},
                {'params': intra_params, 'lr': self.lr_intra_option},
                {'params': value_params, 'lr': learning_rate}
            ])
            self.use_separate_lrs = True
            self.intra_param_group_idx = 1  # Store index of intra-option params group
        else:
            # Single optimizer for all parameters (baseline)
            self.optimizer = optim.Adam(
                self.agent.network.parameters(),
                lr=learning_rate
            )
            self.use_separate_lrs = False
            self.intra_param_group_idx = None
        
        # ===== TARGET NETWORK FOR VALUE FUNCTION STABILITY =====
        # Create target network by cloning the main network
        if self.use_target_network:
            import copy
            self.target_network = copy.deepcopy(self.agent.network)
            # Freeze target network parameters (no gradients)
            for param in self.target_network.parameters():
                param.requires_grad = False
            self.target_network.eval()
        else:
            self.target_network = None
        
        # Training statistics
        self.stats = defaultdict(list)
        
    def update(
        self,
        states: List[np.ndarray],
        options: List[int],
        subactions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        available_subactions: List[List[str]],  # List of available subaction names for each experience
        episode: int = None,  # Episode number for decay and scheduling
        beta_targets: List[float] = None  # H1 Advanced: heuristic termination targets (0.0 or 1.0)
    ) -> Dict[str, float]:
        """
        Update agent using Actor-Critic algorithm.
        
        Args:
            states: List of states
            options: List of selected options
            subactions: List of selected subactions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            episode: Episode number for decay and scheduling
            
        Returns:
            Training statistics
        """
        # Update episode counter and apply scheduling
        if episode is not None:
            self.current_episode = episode
            self._update_schedules(episode)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        options_t = torch.LongTensor(options).to(self.device)
        subactions_t = torch.LongTensor(subactions).to(self.device)
        
        # CRITICAL FIX: Process states with proper LSTM hidden state handling
        # For TD(0) learning, we compute V(s) and V(s') independently.
        # However, if using LSTM, we want to maintain temporal context within each batch.
        # Since states and next_states are from the same episode in order, we can
        # process them as a sequence, but we need to be careful about hidden state.
        
        # Reset hidden state at the start of the batch
        self.agent.network.hidden_state = None
        
        # Forward pass for current states
        # Process batch as sequence - LSTM will maintain context across batch
        outputs = self.agent.network.forward(states_t, reset_hidden=True)
        
        # Forward pass for next states (for bootstrapping)
        # IMPORTANT: For TD(0), we compute V(s') independently, so we reset hidden state
        # This ensures each state is evaluated independently, which is correct for TD(0)
        # If we wanted to maintain temporal context, we would NOT reset here, but that
        # would require processing states and next_states interleaved, which is complex.
        
        # Use target network if enabled (more stable value estimates)
        if self.use_target_network and self.target_network is not None:
            self.target_network.hidden_state = None
            with torch.no_grad():
                next_outputs = self.target_network.forward(next_states_t, reset_hidden=True)
                next_values = next_outputs['state_value']
                next_option_values = next_outputs['option_values']  # Q_Ω(s',ω) for TD targets
                next_action_values = next_outputs['action_values']  # Q_U(s',ω,a) for TD targets
        else:
            self.agent.network.hidden_state = None
            with torch.no_grad():
                next_outputs = self.agent.network.forward(next_states_t, reset_hidden=True)
                next_values = next_outputs['state_value']
                next_option_values = next_outputs['option_values']
                next_action_values = next_outputs['action_values']
        
        # ===== CRITIC LOSS (TD Error) =====
        # Update all three value functions: V(s), Q_Ω(s,ω), Q_U(s,ω,a)
        
        # State value V(s) update
        current_values = outputs['state_value']
        state_targets = rewards_t + self.gamma * next_values * (1.0 - dones_t)
        
        # Update value normalization statistics for stability
        with torch.no_grad():
            batch_mean = state_targets.mean().item()
            batch_std = state_targets.std().item() + 1e-8
            self.value_running_mean = (self.value_norm_momentum * self.value_running_mean + 
                                      (1 - self.value_norm_momentum) * batch_mean)
            self.value_running_std = (self.value_norm_momentum * self.value_running_std + 
                                     (1 - self.value_norm_momentum) * batch_std)
        
        # Clip value predictions to prevent extreme values
        current_values_clipped = torch.clamp(current_values, -self.value_clip, self.value_clip)
        state_targets_clipped = torch.clamp(state_targets.detach(), -self.value_clip, self.value_clip)
        value_loss = F.mse_loss(current_values_clipped, state_targets_clipped)
        
        # Option value Q_Ω(s,ω) update
        # TD target: r + γ * V(s') for one-step updates (simplified from full SMDP)
        option_values = outputs['option_values']  # Q_Ω(s,ω) shape: (batch_size, num_options)
        selected_option_values = option_values.gather(1, options_t.unsqueeze(1)).squeeze(1)  # Q_Ω(s,ω) for selected option
        option_targets = rewards_t + self.gamma * next_values * (1.0 - dones_t)  # Same target as V(s)
        
        selected_option_values_clipped = torch.clamp(selected_option_values, -self.value_clip, self.value_clip)
        option_targets_clipped = torch.clamp(option_targets.detach(), -self.value_clip, self.value_clip)
        option_value_loss = F.mse_loss(selected_option_values_clipped, option_targets_clipped)
        
        # Action value Q_U(s,ω,a) update
        # TD target: r + γ * V(s') for one-step updates
        action_values = outputs['action_values']  # List of Q_U(s,ω,a) tensors, one per option
        batch_size = len(states)
        action_value_losses = []
        
        for i in range(batch_size):
            opt_idx = options[i]
            sub_idx = subactions[i]
            q_u = action_values[opt_idx][i, sub_idx]  # Q_U(s,ω,a) for selected action
            q_u_target = rewards_t[i] + self.gamma * next_values[i] * (1.0 - dones_t[i])
            
            q_u_clipped = torch.clamp(q_u, -self.value_clip, self.value_clip)
            q_u_target_clipped = torch.clamp(q_u_target.detach(), -self.value_clip, self.value_clip)
            action_value_losses.append(F.mse_loss(q_u_clipped.unsqueeze(0), q_u_target_clipped.unsqueeze(0)))
        
        action_value_loss = torch.stack(action_value_losses).mean()
        
        # Total value loss (all three Q-functions)
        total_value_loss = value_loss + option_value_loss + action_value_loss
        
        # ===== ADVANTAGE COMPUTATION (Option-Critic) =====
        # According to Bacon et al. (2017):
        # A_O(s,o) = Q_O(s,o) - V(s) for option policy
        # A_U(s,o,a) = Q_U(s,o,a) - Q_O(s,o) for intra-option policy
        
        # Option-level advantage: A_O(s,o) = Q_O(s,o) - V(s)
        state_value = outputs['state_value'].unsqueeze(1)  # V(s) shape: (batch_size, 1)
        option_advantages = option_values - state_value  # A_O(s,o) shape: (batch_size, num_options)
        selected_option_advantages = option_advantages.gather(1, options_t.unsqueeze(1)).squeeze(1)
        
        # Action-level advantage: A_U(s,o,a) = Q_U(s,o,a) - Q_O(s,o)
        action_advantages_list = []
        for i in range(batch_size):
            opt_idx = options[i]
            sub_idx = subactions[i]
            q_u = action_values[opt_idx][i, sub_idx]  # Q_U(s,ω,a)
            q_omega = option_values[i, opt_idx]  # Q_O(s,ω)
            a_u = q_u - q_omega  # A_U(s,ω,a) = Q_U(s,ω,a) - Q_O(s,ω)
            action_advantages_list.append(a_u)
        action_advantages = torch.stack(action_advantages_list)
        
        # Normalize advantages if enabled
        if self.normalize_advantages:
            if len(selected_option_advantages) > 1:
                opt_adv_mean = selected_option_advantages.mean()
                opt_adv_std = selected_option_advantages.std() + 1e-8
                selected_option_advantages = (selected_option_advantages - opt_adv_mean) / opt_adv_std
            
            if len(action_advantages) > 1:
                act_adv_mean = action_advantages.mean()
                act_adv_std = action_advantages.std() + 1e-8
                action_advantages = (action_advantages - act_adv_mean) / act_adv_std
        else:
            # Clip advantages to prevent extreme values
            selected_option_advantages = torch.clamp(selected_option_advantages, -self.value_clip, self.value_clip)
            action_advantages = torch.clamp(action_advantages, -self.value_clip, self.value_clip)
        
        # ===== ACTOR LOSS (Policy Gradient) =====
        
        # Option policy loss: uses A_O(s,o) = Q_O(s,o) - V(s)
        option_logits = outputs['option_logits']
        option_log_probs = F.log_softmax(option_logits, dim=-1)
        selected_option_log_probs = option_log_probs.gather(1, options_t.unsqueeze(1)).squeeze(1)
        option_policy_loss = -(selected_option_log_probs * selected_option_advantages.detach()).mean()
        
        # Option entropy
        option_probs = F.softmax(option_logits, dim=-1)
        option_entropy = -(option_probs * option_log_probs).sum(dim=-1).mean()
        
        # Intra-option policy loss: uses A_U(s,o,a) = Q_U(s,o,a) - Q_O(s,o)
        subaction_log_probs_list = []
        subaction_entropy_list = []
        
        for i in range(batch_size):
            opt_idx = options[i]
            sub_idx = subactions[i]
            
            # Get subaction logits for this option (already batched)
            sub_logits = outputs['intra_option_logits'][opt_idx][i:i+1]  # (1, num_subactions)
            sub_log_probs = F.log_softmax(sub_logits, dim=-1)
            sub_log_prob = sub_log_probs[0, sub_idx]
            subaction_log_probs_list.append(sub_log_prob)
            
            # Entropy for this subaction
            sub_probs = F.softmax(sub_logits, dim=-1)
            sub_entropy = -(sub_probs * sub_log_probs).sum()
            subaction_entropy_list.append(sub_entropy)
        
        # Stack and compute mean
        subaction_log_probs_tensor = torch.stack(subaction_log_probs_list)
        subaction_policy_loss = -(subaction_log_probs_tensor * action_advantages.detach()).mean()
        
        subaction_entropy_tensor = torch.stack(subaction_entropy_list)
        subaction_entropy = subaction_entropy_tensor.mean()
        
        # ===== TERMINATION LOSS =====
        # Option-Critic termination gradient: ∇_β J ≈ -E[∇_β β_o(s) * A_O(s,o)]
        # where A_O(s,o) = Q_O(s,o) - V(s) is the option-level advantage
        # See Bacon et al. (2017) Option-Critic, equation 9.
        
        termination_probs = outputs['termination_probs']
        selected_term_probs = termination_probs.gather(1, options_t.unsqueeze(1)).squeeze(1)
        
        # ===== INTRA-OPTION ADVANTAGE FOR TERMINATION =====
        # Compute best available subaction value for each experience
        # This provides additional termination signal when best available subaction has low value
        best_available_values = []
        for i in range(batch_size):
            opt_idx = options[i]
            available_subs = available_subactions[i]  # List of available subaction names
            
            # Get option name from index
            option_name = self.option_names[opt_idx]
            
            # Get Q_U values for this option
            q_u_values = action_values[opt_idx][i]  # Shape: (num_subactions,)
            
            # Get subaction names for this option
            option_subactions = self.subaction_names.get(option_name, [])
            
            # Map available subaction names to indices
            available_indices = []
            for sub_name in available_subs:
                if sub_name in option_subactions:
                    sub_idx = option_subactions.index(sub_name)
                    available_indices.append(sub_idx)
            
            # Get best available subaction value
            if available_indices:
                best_available_value = max([q_u_values[idx].item() for idx in available_indices])
            else:
                # Fallback: use first subaction if no mapping found
                best_available_value = q_u_values[0].item()
            
            best_available_values.append(best_available_value)
        
        best_available_tensor = torch.tensor(best_available_values, device=self.device, dtype=torch.float32)
        q_omega_selected = option_values.gather(1, options_t.unsqueeze(1)).squeeze(1)
        
        # Intra-option advantage: A_U_best = Q_U_best - Q_O
        # If best available subaction is worse than option value, this signals termination
        intra_option_advantage = best_available_tensor - q_omega_selected
        
        # Use option-level advantages (already computed above)
        # Create mask: True where option advantage < 0 (encourage termination)
        negative_option_advantage_mask = selected_option_advantages < 0
        
        # Compute loss: if advantage < 0, use log(term_prob), else log(1 - term_prob)
        term_loss_negative = -torch.log(selected_term_probs + 1e-10)
        term_loss_positive = -torch.log(1.0 - selected_term_probs + 1e-10)
        
        # Select based on option advantage sign
        termination_loss = torch.where(
            negative_option_advantage_mask,
            term_loss_negative,
            term_loss_positive
        ).mean()
        
        # Add intra-option advantage termination signal
        # If best available subaction is significantly worse than option value, encourage termination
        intra_option_termination_signal = torch.where(
            intra_option_advantage < -self.intra_option_threshold,
            -torch.log(selected_term_probs + 1e-10),  # Encourage termination
            torch.tensor(0.0, device=self.device)
        )
        termination_loss = termination_loss + self.intra_option_weight * intra_option_termination_signal.mean()
        
        # ===== BETA SUPERVISION LOSS (H1 Advanced) =====
        # Heuristic termination supervision: encourages termination when beta_target=1.0
        # beta_target = 1.0 when exhibit completion >= 70% or exhibit exhausted
        # beta_target = 0.0 otherwise
        # Weight = 0.0 means pure Option-Critic (no supervision), higher = more heuristic guidance
        beta_supervision_loss = torch.tensor(0.0, device=self.device)
        if self.beta_supervision_weight > 0.0 and beta_targets is not None:
            beta_targets_t = torch.tensor(beta_targets, device=self.device, dtype=torch.float32)
            # BCE loss: encourage term_prob to match beta_target
            # When beta_target = 1.0: -log(term_prob) - encourage termination
            # When beta_target = 0.0: -log(1 - term_prob) - discourage termination
            beta_supervision_loss = F.binary_cross_entropy(
                selected_term_probs, 
                beta_targets_t,
                reduction='mean'
            )
            termination_loss = termination_loss + self.beta_supervision_weight * beta_supervision_loss
        
        # ===== TOTAL LOSS =====
        # H1 Advanced: Use separate entropy coefficients for option vs intra-option policies
        total_loss = (
            self.value_loss_coef * total_value_loss +
            option_policy_loss -
            self.entropy_coef_option * option_entropy +
            subaction_policy_loss -
            self.entropy_coef_intra * subaction_entropy +
            self.termination_reg * termination_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Calculate gradient norm BEFORE clipping (for tracking)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
        
        # Store parameter values before update (for update norm calculation)
        param_values_before = {}
        for name, param in self.agent.network.named_parameters():
            if param.requires_grad:
                param_values_before[name] = param.data.clone()
        
        self.optimizer.step()
        
        # Calculate parameter update norm (L2 norm of parameter changes)
        update_norm = 0.0
        for name, param in self.agent.network.named_parameters():
            if param.requires_grad and name in param_values_before:
                param_update = param.data - param_values_before[name]
                update_norm += param_update.norm(2).item() ** 2
        update_norm = update_norm ** 0.5
        
        # Statistics with enhanced debugging info
        stats = {
            'value_loss': value_loss.item(),
            'option_value_loss': option_value_loss.item(),
            'action_value_loss': action_value_loss.item(),
            'total_value_loss': total_value_loss.item(),
            'policy_loss': (option_policy_loss + subaction_policy_loss).item(),
            'entropy': (option_entropy + subaction_entropy).item(),
            'termination_loss': termination_loss.item(),
            'beta_supervision_loss': beta_supervision_loss.item() if self.beta_supervision_weight > 0.0 else 0.0,
            'mean_option_advantage': selected_option_advantages.mean().item(),
            'mean_action_advantage': action_advantages.mean().item(),
            'mean_value': current_values.mean().item(),
            'mean_option_value': selected_option_values.mean().item(),
            # Enhanced RL metrics
            'gradient_norm': grad_norm.item(),
            'update_norm': update_norm,
            # Value function debugging
            'mean_target': state_targets.mean().item(),
            'max_value': current_values.max().item(),
            'min_value': current_values.min().item(),
            'max_target': state_targets.max().item(),
            'min_target': state_targets.min().item(),
            'value_std': current_values.std().item(),
            'target_std': state_targets.std().item(),
            # Add scheduling info
            'current_entropy_coef': self.entropy_coef,
            'current_entropy_coef_option': self.entropy_coef_option,
            'current_entropy_coef_intra': self.entropy_coef_intra,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'current_lr_intra': self.lr_intra_option if self.use_separate_lrs else self.optimizer.param_groups[0]['lr'],
            'option_advantage_mean': selected_option_advantages.mean().item(),
            'option_advantage_std': selected_option_advantages.std().item(),
            'action_advantage_mean': action_advantages.mean().item(),
            'action_advantage_std': action_advantages.std().item(),
            # Detailed HRL metrics for per-episode analysis (Bacon et al. 2017)
            'option_advantages_all': selected_option_advantages.cpu().detach().numpy().tolist(),
            'action_advantages_all': action_advantages.cpu().detach().numpy().tolist(),
            'termination_probs': selected_term_probs.cpu().detach().numpy().tolist(),
            'option_qvalues': selected_option_values.cpu().detach().numpy().tolist(),
            # Per-option advantages (all options, not just selected) for collapse analysis
            'all_option_advantages': option_advantages.cpu().detach().numpy().tolist(),  # Shape: (batch_size, num_options)
            'all_option_qvalues': option_values.cpu().detach().numpy().tolist(),  # Shape: (batch_size, num_options)
            'state_values': state_value.squeeze(1).cpu().detach().numpy().tolist()  # V(s) for all states
        }
        
        for k, v in stats.items():
            self.stats[k].append(v)
        
        return stats
    
    def _update_schedules(self, episode: int):
        """Update learning rate, entropy coefficient, and beta supervision based on episode number."""
        # ===== LEARNING RATE SCHEDULING (Exponential Decay) =====
        # Decay: lr = initial_lr * 0.995^episode (decays ~25% over 500 episodes, ~60% over 2000)
        decay_rate = 0.995
        new_lr = self.initial_lr * (decay_rate ** episode)
        
        if self.use_separate_lrs:
            # Update learning rates for each parameter group separately
            # Group 0: option params, Group 1: intra params, Group 2: value params
            new_lr_intra = self.initial_lr_intra * (decay_rate ** episode)
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i == self.intra_param_group_idx:  # Intra-option params group
                    param_group['lr'] = new_lr_intra
                else:
                    param_group['lr'] = new_lr
        else:
            # Single learning rate for all parameters
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # ===== ENTROPY DECAY (Linear with Optional Adaptive Boost) =====
        # H1 Advanced: Decay option and intra-option entropies separately
        # Linear decay from initial_entropy_coef to entropy_final
        # Start decay at entropy_decay_start, finish at entropy_decay_end
        if self.adaptive_entropy and self.last_oci > self.adaptive_entropy_threshold:
            # COLLAPSE DETECTED - boost entropy to maintain exploration
            target_entropy_option = min(self.initial_entropy_coef_option, self.entropy_coef_option * self.adaptive_entropy_multiplier)
            target_entropy_intra = min(self.initial_entropy_coef_intra, self.entropy_coef_intra * self.adaptive_entropy_multiplier)
            self.entropy_coef_option = target_entropy_option
            self.entropy_coef_intra = target_entropy_intra
            self.entropy_coef = (target_entropy_option + target_entropy_intra) / 2  # Average for backward compatibility
            # Log this event
            if episode % 50 == 0:  # Don't spam logs
                print(f"[ADAPTIVE ENTROPY] Episode {episode}: OCI={self.last_oci:.2f} > {self.adaptive_entropy_threshold:.2f}, boosted entropies to option={self.entropy_coef_option:.4f}, intra={self.entropy_coef_intra:.4f}")
        else:
            # Normal decay schedule for both entropies
            if episode >= self.entropy_decay_start and episode <= self.entropy_decay_end:
                # Linear interpolation
                progress = (episode - self.entropy_decay_start) / (self.entropy_decay_end - self.entropy_decay_start)
                self.entropy_coef_option = self.initial_entropy_coef_option * (1 - progress) + self.entropy_final * progress
                self.entropy_coef_intra = self.initial_entropy_coef_intra * (1 - progress) + self.entropy_final * progress
                self.entropy_coef = (self.entropy_coef_option + self.entropy_coef_intra) / 2  # Average for backward compatibility
            elif episode > self.entropy_decay_end:
                self.entropy_coef_option = self.entropy_final
                self.entropy_coef_intra = self.entropy_final
                self.entropy_coef = self.entropy_final
            else:
                self.entropy_coef_option = self.initial_entropy_coef_option
                self.entropy_coef_intra = self.initial_entropy_coef_intra
                self.entropy_coef = self.initial_entropy_coef
    
    def update_target_network(self):
        """Update target network by copying weights from main network."""
        if self.use_target_network and self.target_network is not None:
            self.target_network.load_state_dict(self.agent.network.state_dict())
            self.target_network.eval()  # Ensure it's in eval mode
    
    def update_oci(self, oci: float):
        """Update the last observed OCI for adaptive entropy control."""
        self.last_oci = oci
    
    def save_checkpoint(self, path: str, episode: int):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'network': self.agent.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': dict(self.stats),
            'entropy_coef': self.entropy_coef,  # Save current entropy coefficient
            'current_episode': self.current_episode
        }
        
        # Save target network if it exists
        if self.use_target_network and self.target_network is not None:
            checkpoint['target_network'] = self.target_network.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = defaultdict(list, checkpoint.get('stats', {}))
        
        # Restore entropy coefficient and episode if available
        if 'entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['entropy_coef']
        if 'current_episode' in checkpoint:
            self.current_episode = checkpoint['current_episode']
        
        # Load target network if it exists
        if self.use_target_network and self.target_network is not None:
            if 'target_network' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network'])
            else:
                # If target network not in checkpoint, copy from main network
                self.target_network.load_state_dict(self.agent.network.state_dict())
            self.target_network.eval()
        
        return checkpoint['episode']

