"""Attempt Distribution Analysis

This script analyzes the policy's multi-attempt behavior by running full episodes
and examining how many attempts are needed to succeed.

Key Metrics:
  - n_attempts distribution (how many tries per episode)
  - Success rate by attempt number (1st try vs 2nd try vs 3rd try)
  - Improvement rate (whether 2nd attempt is better than 1st)
  - Typical failure modes

This helps us understand:
  1. Is the policy learning to adapt across attempts?
  2. Does performance improve on subsequent attempts?
  3. What percentage succeed on first try (already good) vs need multiple tries?

Usage:
  python -m agl.eval.analyze_attempts --ckpt runs/recipe_v3/ckpt_latest.pt --n 100 --out results/
"""
import argparse
import copy
import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..config import load_config
from ..models.policy import Policy
from ..sim import scene
from ..sim.env import GapEnv, OUTCOME


OUTCOME_NAMES = {value: key for key, value in OUTCOME.items()}


@torch.no_grad()
def run_episodes(model: Policy, cfg, n_episodes: int, device: str) -> Dict:
    """Run full episodes and record detailed attempt-level information.

    Returns:
        results: Dict with episode records and attempt-level data
    """
    # Setup environment
    if n_episodes <= 0:
        raise ValueError('n must be positive')
    torch.manual_seed(8888)
    ecfg = copy.deepcopy(cfg)
    ecfg.sim.n_envs = n_episodes
    ecfg.curriculum.enabled = False
    env = GapEnv(ecfg, device, difficulty=1.0)

    # Generate test tasks
    gen = torch.Generator(device=device)
    gen.manual_seed(8888)  # Fixed seed for reproducibility
    tasks = scene.sample_tasks(n_episodes, ecfg, 1.0, device, gen)
    env._reset_envs(torch.arange(n_episodes, device=device), tasks=tasks)

    # Initialize tracking
    episode_records = []
    max_steps = ecfg.sim.ep_len

    h = model.init_hidden(n_episodes, device)
    obs = env.observe()
    priv = torch.zeros(n_episodes, 21, device=device)
    finished = torch.zeros(n_episodes, dtype=torch.bool, device=device)

    # Per-episode attempt tracking
    current_attempts = [[] for _ in range(n_episodes)]
    current_attempt_data = [{'steps': 0, 'clearances': []} for _ in range(n_episodes)]

    print(f"Running {n_episodes} episodes (max {max_steps} steps each)...")

    for step in range(max_steps):
        # Policy step
        mean, _, _, h_new = model.step(obs['img'], obs['vec'], priv, h)
        act = mean.clamp(-1, 1)

        # Environment step
        obs, rew, done, info = env.step(act)
        priv = info['priv']

        # Record per-step data
        for i in range(n_episodes):
            if not finished[i]:
                current_attempt_data[i]['steps'] += 1
                current_attempt_data[i]['clearances'].append(float(info['clearance'][i]))

        # Check for attempt endings
        end_event = info['end_event']
        if end_event.any():
            for i in range(n_episodes):
                if end_event[i] and not finished[i]:
                    # Record this attempt
                    attempt_info = {
                        'attempt_num': len(current_attempts[i]) + 1,
                        'outcome': OUTCOME_NAMES[int(info['end_outcome'][i])],
                        'outcome_code': int(info['end_outcome'][i]),
                        'steps': current_attempt_data[i]['steps'],
                        'min_clearance': min(current_attempt_data[i]['clearances']),
                        'final_clearance': float(info['clearance'][i]),
                        'success': bool(info['success'][i]),
                        'collision': bool(info['collision'][i])
                    }
                    current_attempts[i].append(attempt_info)

                    # Reset attempt tracking for next attempt
                    current_attempt_data[i] = {'steps': 0, 'clearances': []}

        # Update hidden state
        wipe = done.clone()
        if cfg.model.reset_between_attempts:
            wipe |= info['end_event']
        h = h_new * (~wipe).unsqueeze(-1)

        # Check for episode completions
        newly_done = done & ~finished
        if newly_done.any():
            for i in range(n_episodes):
                if newly_done[i]:
                    # Record episode
                    episode_records.append({
                        'episode_id': i,
                        'n_attempts': len(current_attempts[i]),
                        'attempts': current_attempts[i],
                        'final_success': bool(info['success'][i]),
                        'total_steps': step + 1,
                        'gap_width': float(tasks['gap_w'][i]),
                        'gap_height': float(tasks['gap_h'][i])
                    })

        finished |= done
        if finished.all():
            break

        # Progress indicator
        if (step + 1) % 100 == 0:
            n_done = finished.sum().item()
            print(f"  Step {step + 1}/{max_steps}, {n_done}/{n_episodes} episodes finished")

    print(f"Completed {len(episode_records)} episodes")

    return {'episodes': episode_records}


def analyze_attempts(results: Dict) -> Dict:
    """Analyze attempt distributions and compute statistics."""
    episodes = results['episodes']
    if not episodes:
        raise ValueError('No completed episodes; cannot compute outcome rates')

    # Overall statistics
    n_attempts_list = [ep['n_attempts'] for ep in episodes]
    final_success_list = [ep['final_success'] for ep in episodes]

    stats = {
        'n_episodes': len(episodes),
        'mean_attempts': float(np.mean(n_attempts_list)),
        'median_attempts': float(np.median(n_attempts_list)),
        'std_attempts': float(np.std(n_attempts_list)),
        'max_attempts': int(np.max(n_attempts_list)),
        'min_attempts': int(np.min(n_attempts_list)),
        'final_success_rate': float(np.mean(final_success_list) * 100)
    }

    # Success rate by attempt number
    success_by_attempt = defaultdict(lambda: {'total': 0, 'success': 0})

    for ep in episodes:
        for attempt in ep['attempts']:
            attempt_num = attempt['attempt_num']
            success_by_attempt[attempt_num]['total'] += 1
            if attempt['success']:
                success_by_attempt[attempt_num]['success'] += 1

    # Compute success rates
    attempt_stats = {}
    for attempt_num in sorted(success_by_attempt.keys()):
        data = success_by_attempt[attempt_num]
        rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
        attempt_stats[f'attempt_{attempt_num}'] = {
            'total': data['total'],
            'success': data['success'],
            'success_rate': rate
        }

    stats['by_attempt'] = attempt_stats

    # Attempt distribution
    attempt_distribution = {}
    for n in range(1, stats['max_attempts'] + 1):
        count = sum(1 for ep in episodes if ep['n_attempts'] == n)
        pct = count / len(episodes) * 100
        attempt_distribution[n] = {'count': count, 'percentage': pct}

    stats['attempt_distribution'] = attempt_distribution
    stats['cumulative_success_by_attempt'] = {
        str(k): sum(any(a['success'] and a['attempt_num'] <= k for a in ep['attempts'])
                    for ep in episodes) / len(episodes) * 100
        for k in sorted(success_by_attempt)
    }

    # Improvement analysis: compare attempt 1 vs attempt 2
    improvement_data = []
    for ep in episodes:
        if len(ep['attempts']) >= 2:
            attempt1 = ep['attempts'][0]
            attempt2 = ep['attempts'][1]

            # Did second attempt succeed where first failed?
            improved = (not attempt1['success']) and attempt2['success']

            # Clearance improvement
            clearance_delta = attempt2['min_clearance'] - attempt1['min_clearance']

            improvement_data.append({
                'improved': improved,
                'clearance_delta': clearance_delta,
                'attempt1_outcome': attempt1['outcome'],
                'attempt2_outcome': attempt2['outcome']
            })

    if improvement_data:
        improvement_rate = sum(1 for x in improvement_data if x['improved']) / len(improvement_data) * 100
        avg_clearance_delta = np.mean([x['clearance_delta'] for x in improvement_data])
        stats['improvement_rate'] = float(improvement_rate)
        stats['avg_clearance_delta'] = float(avg_clearance_delta)
        stats['n_multi_attempt_episodes'] = len(improvement_data)
    else:
        stats['improvement_rate'] = None
        stats['avg_clearance_delta'] = None
        stats['n_multi_attempt_episodes'] = 0

    stats['interpretation'] = 'Conditional retry outcomes, not causal adaptation evidence.'

    return stats


def plot_results(results: Dict, stats: Dict, out_dir: str):
    """Generate visualization plots."""
    os.makedirs(out_dir, exist_ok=True)

    episodes = results['episodes']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: n_attempts distribution
    n_attempts_list = [ep['n_attempts'] for ep in episodes]
    bins = np.arange(1, stats['max_attempts'] + 2) - 0.5
    axes[0, 0].hist(n_attempts_list, bins=bins, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(stats['mean_attempts'], color='red', linestyle='--',
                       label=f"Mean: {stats['mean_attempts']:.2f}")
    axes[0, 0].set_xlabel('Number of Attempts')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Attempts per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Success rate by attempt number
    attempt_nums = sorted([int(k.split('_')[1]) for k in stats['by_attempt'].keys()])
    success_rates = [stats['by_attempt'][f'attempt_{n}']['success_rate'] for n in attempt_nums]
    totals = [stats['by_attempt'][f'attempt_{n}']['total'] for n in attempt_nums]

    axes[0, 1].bar(attempt_nums, success_rates, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Attempt Number')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Success Rate by Attempt Number')
    axes[0, 1].set_xticks(attempt_nums)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add counts as text
    for i, (num, rate, total) in enumerate(zip(attempt_nums, success_rates, totals)):
        axes[0, 1].text(num, rate + 2, f'n={total}', ha='center', fontsize=9)

    # Plot 3: Cumulative success rate
    cumulative_success = []
    cumulative_total = 0
    cumulative_success_count = 0

    for n in attempt_nums:
        cumulative_total = stats['n_episodes']
        cumulative_success_count += stats['by_attempt'][f'attempt_{n}']['success']
        rate = stats['cumulative_success_by_attempt'][str(n)]
        cumulative_success.append(rate)

    axes[1, 0].plot(attempt_nums, cumulative_success, marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Maximum Attempts Allowed')
    axes[1, 0].set_ylabel('Overall Success Rate (%)')
    axes[1, 0].set_title('Cumulative Success Rate')
    axes[1, 0].set_xticks(attempt_nums)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])

    # Plot 4: Summary statistics
    summary_text = f"""
    Attempt Distribution Analysis

    Episodes analyzed: {stats['n_episodes']}

    Mean attempts: {stats['mean_attempts']:.2f}
    Median attempts: {stats['median_attempts']:.0f}
    Std deviation: {stats['std_attempts']:.2f}

    Final success rate: {stats['final_success_rate']:.1f}%

    Success by attempt:
      1st try: {stats['by_attempt'].get('attempt_1', {}).get('success_rate', 0):.1f}%
      2nd try: {stats['by_attempt'].get('attempt_2', {}).get('success_rate', 0):.1f}%
      3rd try: {stats['by_attempt'].get('attempt_3', {}).get('success_rate', 0):.1f}%

    Improvement rate (2nd > 1st):
      {stats['improvement_rate']}% (None = insufficient data)
      (out of {stats['n_multi_attempt_episodes']} multi-attempt episodes)

    Avg clearance improvement:
      {stats['avg_clearance_delta']}m
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'attempt_distribution_analysis.png'), dpi=150)
    print(f"Saved plot: {out_dir}/attempt_distribution_analysis.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze attempt distribution')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of episodes to run')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = load_config(overrides=ck['cfg'])
    model = Policy(cfg).to(args.device)
    model.load_state_dict(ck['model'])
    model.eval()

    print(f"Model config:")
    print(f"  use_memory: {cfg.model.use_memory}")
    print(f"  reset_between_attempts: {cfg.model.reset_between_attempts}")

    # Run episodes
    print(f"\nRunning {args.n} episodes...")
    results = run_episodes(model, cfg, args.n, args.device)

    # Analyze
    print("\nAnalyzing attempt patterns...")
    stats = analyze_attempts(results)

    # Print results
    print("\n" + "="*60)
    print("ATTEMPT DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Episodes analyzed:       {stats['n_episodes']}")
    print(f"Mean attempts:           {stats['mean_attempts']:.2f}")
    print(f"Median attempts:         {stats['median_attempts']:.0f}")
    print(f"Final success rate:      {stats['final_success_rate']:.1f}%")
    print(f"\nSuccess rate by attempt:")
    for attempt_num in sorted([int(k.split('_')[1]) for k in stats['by_attempt'].keys()]):
        data = stats['by_attempt'][f'attempt_{attempt_num}']
        print(f"  Attempt {attempt_num}: {data['success_rate']:5.1f}% (n={data['total']})")
    print(f"\nImprovement analysis:")
    print(f"  Conditional retry success: {stats['improvement_rate']}% (None = no data)")
    print(f"  Multi-attempt episodes: {stats['n_multi_attempt_episodes']}")
    print(f"  Avg clearance change: {stats['avg_clearance_delta']}m")
    print("="*60)

    # Save results
    os.makedirs(args.out, exist_ok=True)

    # Save detailed episode data
    with open(os.path.join(args.out, 'attempt_analysis_episodes.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save statistics
    with open(os.path.join(args.out, 'attempt_analysis_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Generate plots
    plot_results(results, stats, args.out)

    print(f"\nResults saved to: {args.out}")


if __name__ == '__main__':
    main()
