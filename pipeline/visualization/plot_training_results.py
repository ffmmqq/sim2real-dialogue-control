"""
Generate publication-quality plots from training metrics

Usage:
    python src/visualization/plot_training_results.py training_logs/metrics_tracker_TIMESTAMP.json

Generates:
- Learning curve with confidence bands
- Reward decomposition over episodes
- Option usage bar chart
- Dwell distribution histogram
- Episode length trends
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import argparse
import sys

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_learning_curve(data: dict, save_path: str = "plots/learning_curve.png"):
    """Plot smoothed learning curve with confidence bands"""
    returns = data.get("episode_returns", [])
    
    if not returns:
        print("⚠️  No episode returns found in data")
        return
    
    # Compute moving average
    window = min(50, len(returns) // 4)  # Adaptive window
    if window < 2:
        window = 2
    
    smoothed = []
    stds = []
    
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_data = returns[start:i+1]
        smoothed.append(np.mean(window_data))
        stds.append(np.std(window_data))
    
    episodes = np.arange(len(returns))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw returns (faint)
    ax.plot(episodes, returns, alpha=0.2, color='gray', label='Raw Returns', linewidth=0.8)
    
    # Plot smoothed curve
    ax.plot(episodes, smoothed, linewidth=2.5, color='#2E86AB', label=f'{window}-Episode MA')
    
    # Confidence band
    smoothed = np.array(smoothed)
    stds = np.array(stds)
    ax.fill_between(episodes, smoothed - stds, smoothed + stds, alpha=0.25, color='#2E86AB')
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=13, fontweight='bold')
    # Try to detect hypothesis from data path if provided
    hypothesis = 'H1'  # Default
    if 'save_path' in locals() or 'save_path' in globals():
        path_str = str(save_path) if 'save_path' in locals() else ''
        for h in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            if h in path_str:
                hypothesis = h
                break
    
    ax.set_title(f'{hypothesis}: Learning Curve', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add final performance annotation
    if len(returns) > 10:
        recent_avg = np.mean(returns[-10:])
        ax.axhline(y=recent_avg, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(len(returns) * 0.98, recent_avg, f'Final: {recent_avg:.2f}', 
                ha='right', va='bottom', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curve to {save_path}")


def plot_option_usage(data: dict, save_path: str = "plots/option_usage.png"):
    """Bar chart of option usage proportions"""
    option_counts = data.get("option_counts", {})
    
    if not option_counts:
        print("⚠️  No option counts found in data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    options = list(option_counts.keys())
    counts = list(option_counts.values())
    
    colors = ['#06AED5', '#F4D35E', '#DD6E42', '#78BC61'][:len(options)]
    bars = ax.bar(options, counts, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_xlabel('Option', fontsize=13, fontweight='bold')
    ax.set_title('Option Usage During Training', fontsize=15, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(counts) * 0.01),
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = (count / total * 100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{pct:.1f}%',
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved option usage to {save_path}")


def plot_episode_lengths(data: dict, save_path: str = "plots/episode_lengths.png"):
    """Plot episode length trends"""
    lengths = data.get("episode_lengths", [])
    
    if not lengths:
        print("⚠️  No episode lengths found in data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(lengths))
    
    # Plot raw lengths
    ax.plot(episodes, lengths, alpha=0.4, color='gray', marker='o', 
            markersize=3, linewidth=0.8, label='Episode Length')
    
    # Add moving average
    window = min(20, len(lengths) // 4)
    if window >= 2:
        smoothed = []
        for i in range(len(lengths)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(lengths[start:i+1]))
        ax.plot(episodes, smoothed, linewidth=2.5, color='#DD6E42', label=f'{window}-Episode MA')
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Episode Length (turns)', fontsize=13, fontweight='bold')
    ax.set_title('Episode Length Trends', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add mean line
    mean_length = np.mean(lengths)
    ax.axhline(y=mean_length, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(len(lengths) * 0.02, mean_length, f'Mean: {mean_length:.1f}', 
            ha='left', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved episode lengths to {save_path}")


def plot_coverage_trends(data: dict, save_path: str = "plots/coverage_trends.png"):
    """Plot exhibit coverage over episodes"""
    coverage = data.get("episode_coverage", [])
    
    if not coverage:
        print("⚠️  No coverage data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(coverage))
    coverage_pct = [c * 100 for c in coverage]  # Convert to percentage
    
    # Plot raw coverage
    ax.plot(episodes, coverage_pct, alpha=0.4, color='gray', marker='o', 
            markersize=3, linewidth=0.8, label='Coverage')
    
    # Add moving average
    window = min(20, len(coverage) // 4)
    if window >= 2:
        smoothed = []
        for i in range(len(coverage_pct)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(coverage_pct[start:i+1]))
        ax.plot(episodes, smoothed, linewidth=2.5, color='#78BC61', label=f'{window}-Episode MA')
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Exhibit Coverage Trends', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # Add target line at 80%
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
    ax.text(len(episodes) * 0.98, 80, 'Target: 80%', 
            ha='right', va='bottom', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved coverage trends to {save_path}")


def plot_option_entropy_evolution(data: dict, save_path: str = "plots/option_entropy_evolution.png"):
    """
    Plot option entropy over episodes.
    
    H1 Termination Tuning: Measures balance of option usage.
    Target: > 1.0 bits for 4 options (max is 2.0 bits).
    """
    entropy_per_episode = data.get("option_entropy_per_episode", [])
    
    if not entropy_per_episode:
        print("⚠️  No option entropy data found")
        return
    
    # Compute moving average
    window = min(50, len(entropy_per_episode) // 4)
    if window < 2:
        window = 2
    
    smoothed = []
    for i in range(len(entropy_per_episode)):
        start = max(0, i - window + 1)
        window_data = entropy_per_episode[start:i+1]
        smoothed.append(np.mean(window_data))
    
    episodes = np.arange(len(entropy_per_episode))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw entropy (faint)
    ax.plot(episodes, entropy_per_episode, alpha=0.2, color='gray', 
            label='Raw Entropy', linewidth=0.8)
    
    # Plot smoothed curve
    ax.plot(episodes, smoothed, linewidth=2.5, color='#6B4C9A', 
            label=f'{window}-Episode MA')
    
    # Target line at 1.0 bits (balanced 4 options)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(len(episodes) * 0.98, 1.0, 'Target: 1.0 bits', 
            ha='right', va='bottom', fontsize=10, color='green')
    
    # Max entropy line (for 4 options: 2.0 bits)
    ax.axhline(y=2.0, color='blue', linestyle=':', alpha=0.3, linewidth=1.5)
    ax.text(len(episodes) * 0.98, 2.0, 'Max: 2.0 bits', 
            ha='right', va='bottom', fontsize=9, color='blue', alpha=0.6)
    
    ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax.set_ylabel('Option Entropy (bits)', fontsize=13, fontweight='bold')
    ax.set_title('H1 Termination Tuning: Option Entropy Evolution', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 2.2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved option entropy evolution to {save_path}")
    plt.close()


def plot_option_duration_distribution(data: dict, save_path: str = "plots/option_duration_distribution.png"):
    """
    Plot distribution of option durations (turns per option instance).
    
    H1 Termination Tuning: Measures how long the agent stays in each option.
    Target: 2-4 turns mean duration (indicates active termination).
    """
    option_durations = data.get("option_durations", {})
    
    if not option_durations:
        print("⚠️  No option duration data found")
        return
    
    # Flatten all durations
    all_durations = []
    for opt, durations in option_durations.items():
        all_durations.extend(durations)
    
    if not all_durations:
        print("⚠️  No option duration data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Overall distribution
    ax1 = axes[0]
    ax1.hist(all_durations, bins=range(1, max(all_durations) + 2), 
             alpha=0.7, color='#2E86AB', edgecolor='black')
    ax1.axvline(x=np.mean(all_durations), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_durations):.1f}')
    ax1.axvline(x=3.0, color='green', linestyle=':', 
                linewidth=2, label='Target: 3.0')
    ax1.set_xlabel('Duration (turns)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Option Duration Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Per-option distribution (box plot)
    ax2 = axes[1]
    option_names = list(option_durations.keys())
    option_data = [option_durations[opt] for opt in option_names]
    
    bp = ax2.boxplot(option_data, labels=option_names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(option_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=3.0, color='green', linestyle=':', linewidth=2, label='Target: 3.0')
    ax2.set_xlabel('Option', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Duration (turns)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Option Duration Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('H1 Termination Tuning: Option Duration Analysis', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved option duration distribution to {save_path}")
    plt.close()


def generate_summary_report(data: dict, save_path: str = "plots/summary_report.txt"):
    """Generate text summary of training"""
    summary = data.get("summary", {})
    
    if not summary:
        print("⚠️  No summary data found")
        return
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Episodes: {summary.get('total_episodes', 0)}\n")
        f.write(f"Total Turns: {summary.get('total_turns', 0)}\n\n")
        
        f.write("RETURNS:\n")
        f.write(f"  Mean Return: {summary.get('mean_return', 0.0):.3f}\n")
        f.write(f"  Std Return: {summary.get('std_return', 0.0):.3f}\n")
        f.write(f"  Recent Mean (last 100): {summary.get('recent_mean_return', 0.0):.3f}\n\n")
        
        f.write("EPISODES:\n")
        f.write(f"  Mean Length: {summary.get('mean_length', 0.0):.1f} turns\n")
        f.write(f"  Recent Mean Length: {summary.get('recent_mean_length', 0.0):.1f} turns\n\n")
        
        f.write("COVERAGE:\n")
        f.write(f"  Mean Coverage: {summary.get('mean_coverage', 0.0):.1%}\n")
        f.write(f"  Recent Mean Coverage: {summary.get('recent_mean_coverage', 0.0):.1%}\n")
        f.write(f"  Mean Facts/Episode: {summary.get('mean_facts_per_episode', 0.0):.1f}\n\n")
        
        f.write("ENGAGEMENT:\n")
        f.write(f"  Mean Dwell: {summary.get('mean_dwell', 0.0):.3f}\n")
        f.write(f"  Median Dwell: {summary.get('median_dwell', 0.0):.3f}\n\n")
        
        f.write("SUCCESS RATES:\n")
        f.write(f"  Transition Success: {summary.get('transition_success_rate', 0.0):.1%}\n")
        f.write(f"  Question Answer Rate: {summary.get('question_answer_rate', 0.0):.1%}\n\n")
        
        option_usage = summary.get('option_usage', {})
        if option_usage:
            f.write("OPTION USAGE:\n")
            for opt, prop in sorted(option_usage.items(), key=lambda x: -x[1]):
                f.write(f"  {opt}: {prop:.1%}\n")
            f.write("\n")
        
        option_durations = summary.get('option_mean_durations', {})
        if option_durations:
            f.write("OPTION DURATIONS (mean turns):\n")
            for opt, dur in sorted(option_durations.items()):
                f.write(f"  {opt}: {dur:.1f}\n")
            f.write("\n")
        
        # H1 Termination Tuning metrics
        f.write("H1 TERMINATION TUNING METRICS:\n")
        f.write(f"  Mean Option Entropy: {summary.get('mean_option_entropy', 0.0):.3f} bits\n")
        f.write(f"  Recent Option Entropy: {summary.get('recent_option_entropy', 0.0):.3f} bits\n")
        f.write(f"  Option Collapse Rate: {summary.get('option_collapse_rate', 0.0):.1%}\n")
        f.write(f"  Max Option Proportion: {summary.get('max_option_proportion', 0.0):.1%}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Saved summary report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from metrics file')
    parser.add_argument('metrics_file', type=str, 
                       help='Path to metrics JSON file (e.g., training_logs/metrics_tracker_*.json)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"❌ Error: File not found: {metrics_path}")
        sys.exit(1)
    
    print(f"📊 Loading metrics from {metrics_path}...")
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Saving plots to {output_dir}/")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    plot_learning_curve(data, save_path=output_dir / "learning_curve.png")
    plot_option_usage(data, save_path=output_dir / "option_usage.png")
    plot_episode_lengths(data, save_path=output_dir / "episode_lengths.png")
    plot_coverage_trends(data, save_path=output_dir / "coverage_trends.png")
    
    # H1 Termination Tuning: Generate entropy and duration plots
    plot_option_entropy_evolution(data, save_path=output_dir / "option_entropy_evolution.png")
    plot_option_duration_distribution(data, save_path=output_dir / "option_duration_distribution.png")
    
    # Generate summary report
    generate_summary_report(data, save_path=output_dir / "summary_report.txt")
    
    print(f"\n✅ All plots generated successfully in {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  • learning_curve.png")
    print(f"  • option_usage.png")
    print(f"  • episode_lengths.png")
    print(f"  • coverage_trends.png")
    print(f"  • summary_report.txt")


if __name__ == "__main__":
    main()

