"""
Clean live progress tracker for training
Shows episode-level stats without verbose turn-by-turn details
"""

import sys
from collections import defaultdict
from datetime import datetime, timedelta

class LiveProgressTracker:
    """Clean, real-time training progress display"""
    
    def __init__(self, max_episodes, is_flat_rl=False, experiment_name=None):
        self.max_episodes = max_episodes
        self.is_flat_rl = is_flat_rl  # Flag to distinguish Flat RL from HRL
        self.experiment_name = experiment_name or "Training"
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.option_counts = defaultdict(int)
        self.subaction_counts = defaultdict(int)
        self.current_episode = 0
        self.current_ep_reward = 0.0
        self.current_ep_turns = 0
        self.start_time = datetime.now()
        
    def start_episode(self, episode_num):
        """Start tracking a new episode"""
        self.current_episode = episode_num
        self.current_ep_reward = 0.0
        self.current_ep_turns = 0
    
    def update_turn(self, reward, option, subaction=None):
        """Update with turn-level info (no printing)"""
        self.current_ep_reward += reward
        self.current_ep_turns += 1
        self.option_counts[option] += 1
        if subaction:
            # Track as "Option/Subaction" for clarity
            combined_key = f"{option}/{subaction}"
            self.subaction_counts[combined_key] += 1
    
    def end_episode(self, total_reward, length, episode_time=None, coverage_dict=None, timing=None):
        """End episode and show summary
        
        Args:
            total_reward: Total reward for the episode
            length: Number of turns in the episode
            episode_time: Time taken for this episode (seconds)
            coverage_dict: Dict mapping exhibit names to coverage rates (0.0-1.0)
            timing: Dict with component timing breakdown (bert_time, agent_llm_time, etc.)
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        if episode_time is not None:
            # Store episode time if provided
            if not hasattr(self, 'episode_times'):
                self.episode_times = []
            self.episode_times.append(episode_time)
        self.current_episode_coverage = coverage_dict  # Store for printing
        self.current_episode_timing = timing  # Store timing for printing
        self._print_episode_summary()
    
    def _print_episode_summary(self):
        """Print clean episode summary"""
        # Calculate stats
        recent_window = min(10, len(self.episode_rewards))
        recent_avg = sum(self.episode_rewards[-recent_window:]) / recent_window if recent_window > 0 else 0.0
        
        # Elapsed time
        elapsed = (datetime.now() - self.start_time).total_seconds()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Episode progress
        progress_pct = (self.current_episode / self.max_episodes) * 100
        
        # Calculate ETA and finish time using recent episode times (more accurate)
        if len(self.episode_times) >= 5:
            # Use average of last 20 episodes for more stable and recent ETA
            recent_time_window = min(20, len(self.episode_times))
            avg_time_per_ep = sum(self.episode_times[-recent_time_window:]) / recent_time_window
            remaining_eps = self.max_episodes - self.current_episode
            eta_sec = remaining_eps * avg_time_per_ep
            eta_hours = int(eta_sec // 3600)
            eta_mins = int((eta_sec % 3600) // 60)
            eta_secs = int(eta_sec % 60)
            
            # Calculate finish time
            finish_time = datetime.now() + timedelta(seconds=eta_sec)
            finish_str = finish_time.strftime("%H:%M:%S")
            eta_str = f"{eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
        elif len(self.episode_times) > 0:
            # Estimate using available episodes (less accurate but still useful)
            avg_time_per_ep = sum(self.episode_times) / len(self.episode_times)
            remaining_eps = self.max_episodes - self.current_episode
            eta_sec = remaining_eps * avg_time_per_ep
            eta_hours = int(eta_sec // 3600)
            eta_mins = int((eta_sec % 3600) // 60)
            eta_secs = int(eta_sec % 60)
            
            # Calculate finish time
            finish_time = datetime.now() + timedelta(seconds=eta_sec)
            finish_str = finish_time.strftime("%H:%M:%S")
            eta_str = f"{eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
        else:
            # No episode times available yet, use elapsed time as estimate
            if elapsed > 0 and self.current_episode > 0:
                avg_time_per_ep = elapsed / self.current_episode
                remaining_eps = self.max_episodes - self.current_episode
                eta_sec = remaining_eps * avg_time_per_ep
                eta_hours = int(eta_sec // 3600)
                eta_mins = int((eta_sec % 3600) // 60)
                eta_secs = int(eta_sec % 60)
                
                # Calculate finish time
                finish_time = datetime.now() + timedelta(seconds=eta_sec)
                finish_str = finish_time.strftime("%H:%M:%S")
                eta_str = f"{eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
            else:
                eta_str = "??:??:??"
                finish_str = "??:??:??"
        
        # Option distribution (top 3 cumulative)
        total_turns = sum(self.option_counts.values())
        option_dist = {k: (v/total_turns*100) for k, v in self.option_counts.items()} if total_turns > 0 else {}
        top_options = sorted(option_dist.items(), key=lambda x: -x[1])[:3]
        options_str = ', '.join([f'{k}({v:.0f}%)' for k, v in top_options]) if top_options else "N/A"
        
        # Subaction distribution (top 3 cumulative)
        total_subactions = sum(self.subaction_counts.values())
        subaction_dist = {k: (v/total_subactions*100) for k, v in self.subaction_counts.items()} if total_subactions > 0 else {}
        top_subactions = sorted(subaction_dist.items(), key=lambda x: -x[1])[:3]
        subactions_str = ', '.join([f'{k}({v:.0f}%)' for k, v in top_subactions]) if top_subactions else "N/A"
        
        # Current episode turns (not cumulative)
        ep_turns = self.episode_lengths[-1] if self.episode_lengths else 0
        
        # Episode time (time for this specific episode)
        # Should always be available after episode ends, but handle edge case
        if len(self.episode_times) > 0:
            ep_time_sec = self.episode_times[-1]
            ep_time_mins = int(ep_time_sec // 60)
            ep_time_secs = int(ep_time_sec % 60)
            ep_time_str = f"{ep_time_mins:02d}:{ep_time_secs:02d}"
        elif elapsed > 0 and self.current_episode > 0:
            # Fallback: estimate from elapsed time per episode
            estimated_ep_time = elapsed / self.current_episode
            ep_time_mins = int(estimated_ep_time // 60)
            ep_time_secs = int(estimated_ep_time % 60)
            ep_time_str = f"{ep_time_mins:02d}:{ep_time_secs:02d}"
        else:
            ep_time_str = "00:00"
        
        # Timing information
        timing_str = ""
        if hasattr(self, 'current_episode_timing') and self.current_episode_timing:
            t = self.current_episode_timing
            # Show timing breakdown (only non-zero components)
            timing_parts = []
            if t.get('bert_time', 0) > 0.01:
                timing_parts.append(f"BERT:{t['bert_time']:.1f}s")
            if t.get('agent_llm_time', 0) > 0.01:
                timing_parts.append(f"AgentLLM:{t['agent_llm_time']:.1f}s")
            if t.get('agent_template_time', 0) > 0.001:
                timing_parts.append(f"AgentTpl:{t['agent_template_time']*1000:.0f}ms")
            if t.get('simulator_llm_time', 0) > 0.01:
                timing_parts.append(f"SimLLM:{t['simulator_llm_time']:.1f}s")
            if timing_parts:
                timing_str = f" | {', '.join(timing_parts)}"
        
        # Coverage information
        coverage_str = ""
        if hasattr(self, 'current_episode_coverage') and self.current_episode_coverage:
            try:
                # Format individual exhibit coverage
                exhibit_coverage_parts = []
                for exhibit_name, coverage_data in sorted(self.current_episode_coverage.items()):
                    if isinstance(coverage_data, dict):
                        coverage_pct = coverage_data.get('coverage', 0.0) * 100
                    else:
                        coverage_pct = float(coverage_data) * 100
                    # Shorten exhibit names for display
                    short_name = exhibit_name.replace('_', '')[:8]  # Max 8 chars
                    exhibit_coverage_parts.append(f"{short_name}({coverage_pct:.0f}%)")
                
                # Calculate total coverage (average across all exhibits)
                coverage_values = list(self.current_episode_coverage.values())
                if coverage_values and isinstance(coverage_values[0], dict):
                    total_coverage = sum(v.get('coverage', 0.0) for v in coverage_values) / len(coverage_values) * 100
                else:
                    total_coverage = sum(float(v) for v in coverage_values) / len(coverage_values) * 100
                
                coverage_str = f" | Coverage: {', '.join(exhibit_coverage_parts)} | Total: {total_coverage:.0f}%"
            except (ValueError, TypeError, ZeroDivisionError):
                # Handle any errors in coverage calculation gracefully
                coverage_str = ""
        
        # Print on new line (no overwriting)
        # For Flat RL, show "Actions" instead of "Opts/Subs"
        if self.is_flat_rl:
            action_label = "Actions"
            action_str = options_str  # In Flat RL, option_counts tracks flat actions
            print(f"[{self.experiment_name}] Ep {self.current_episode}/{self.max_episodes} ({progress_pct:.1f}%) | "
                  f"R: {self.episode_rewards[-1]:.2f} | "
                  f"Avg: {recent_avg:.2f} | "
                  f"Turns: {ep_turns} | "
                  f"{action_label}: {action_str} | "
                  f"EpTime: {ep_time_str}{timing_str} | "
                  f"Time {hours:02d}:{minutes:02d}:{seconds:02d} | "
                  f"ETA: {eta_str} → {finish_str}{coverage_str}", flush=True)
        else:
            # HRL: show both options and subactions
            print(f"[{self.experiment_name}] Ep {self.current_episode}/{self.max_episodes} ({progress_pct:.1f}%) | "
                  f"R: {self.episode_rewards[-1]:.2f} | "
                  f"Avg: {recent_avg:.2f} | "
                  f"Turns: {ep_turns} | "
                  f"Opts: {options_str} | "
                  f"Subs: {subactions_str} | "
                  f"EpTime: {ep_time_str}{timing_str} | "
                  f"Time {hours:02d}:{minutes:02d}:{seconds:02d} | "
                  f"ETA: {eta_str} → {finish_str}{coverage_str}", flush=True)
        
        # Every 50 episodes, print detailed progress
        if self.current_episode % 50 == 0:
            print()
            self._print_detailed_progress()
    
    def _print_detailed_progress(self):
        """Print detailed stats every 50 episodes"""
        print("\n" + "=" * 80)
        print(f"CHECKPOINT @ Episode {self.current_episode}/{self.max_episodes}")
        print("=" * 80)
        
        # Reward stats
        recent_100 = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        print(f"Rewards:")
        print(f"  Last episode: {self.episode_rewards[-1]:.3f}")
        print(f"  Recent avg:   {sum(recent_100)/len(recent_100):.3f} (last {len(recent_100)} eps)")
        print(f"  Overall avg:  {sum(self.episode_rewards)/len(self.episode_rewards):.3f}")
        
        # Episode length
        recent_len = self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths
        print(f"\nEpisode Length:")
        print(f"  Last: {self.episode_lengths[-1]} turns")
        print(f"  Avg:  {sum(recent_len)/len(recent_len):.1f} turns")
        
        # Option/Action distribution
        total_turns = sum(self.option_counts.values())
        dist_label = "Action Distribution" if self.is_flat_rl else "Option Distribution"
        print(f"\n{dist_label} (all episodes):")
        for option, count in sorted(self.option_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_turns * 100) if total_turns > 0 else 0
            bar_len = int(pct / 2)  # Scale to 50 chars max
            bar = "#" * bar_len
            # Adjust label width for flat actions (longer names)
            label_width = 25 if self.is_flat_rl else 15
            print(f"  {option:{label_width}s}: {bar} {pct:5.1f}% ({count})")
        
        # Time estimate using recent episode times
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if len(self.episode_times) >= 5:
            # Use average of last 20 episodes for more stable and recent ETA
            recent_time_window = min(20, len(self.episode_times))
            avg_time_per_ep = sum(self.episode_times[-recent_time_window:]) / recent_time_window
            remaining_eps = self.max_episodes - self.current_episode
            eta_sec = remaining_eps * avg_time_per_ep
            eta_hours = int(eta_sec // 3600)
            eta_mins = int((eta_sec % 3600) // 60)
            eps_per_sec = 1.0 / avg_time_per_ep if avg_time_per_ep > 0 else 0
        else:
            eps_per_sec = self.current_episode / elapsed if elapsed > 0 else 0
            remaining_eps = self.max_episodes - self.current_episode
            eta_sec = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
            eta_hours = int(eta_sec // 3600)
            eta_mins = int((eta_sec % 3600) // 60)
        
        print(f"\nTime:")
        print(f"  Elapsed: {int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}")
        print(f"  ETA:     {eta_hours:02d}:{eta_mins:02d}:00 ({eps_per_sec:.2f} eps/sec)")
        
        print("=" * 80 + "\n")

