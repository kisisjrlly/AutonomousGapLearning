"""Visualize the abstract safe-probe reference and its safety sweep."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_reference(data, out):
    rows = [r for r in data['tasks'] if r.get('eligible')]
    if not rows:
        raise ValueError('reference contains no eligible tasks')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax = axes[0]
    task = rows[0]['task']
    ax.set_title('Reference probe and return (one task)')
    ax.set_xlabel('forward x (m)')
    ax.set_ylabel('lateral y (m)')
    ax.set_xlim(-.2, 3.3); ax.set_ylim(-1.2, 1.2)
    # front wall sides and rear wall sides, schematic only
    for x0, x1, y0, y1 in [(2, 2.15, -5, -.7), (2, 2.15, .7, 5),
                           (2.15, 2.4, -5, task['center']-task['width']/2),
                           (2.15, 2.4, task['center']+task['width']/2, 5)]:
        ax.add_patch(Rectangle((x0, max(-1.2, y0)), x1-x0,
                               max(0, min(1.2, y1)-max(-1.2, y0)),
                               color='0.72', alpha=.8))
    probe = np.asarray(rows[0]['probe']['trace'])
    retreat = np.asarray(rows[0]['retreat_trace'])
    ax.plot(probe[:, 0], probe[:, 1], 'o-', lw=2, color='#2674c8', label='safe probe')
    ax.plot(retreat[:, 0], retreat[:, 1], 'o--', lw=1.5, color='#e57928', label='safe return')
    ax.scatter([0], [0], color='black', zorder=4, label='rest state')
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .02, 'schematic oracle geometry; not flight evidence',
            transform=ax.transAxes, fontsize=8, color='0.35')

    ax = axes[1]
    ax.set_title('Information ablation across eligible tasks')
    names = ['real\ninformation', 'removed\ninformation', 'swapped\ninformation']
    vals = [data['summary'][k]['successes'] for k in ('real', 'removed', 'swapped')]
    abstain = [data['summary'][k]['abstentions'] for k in ('real', 'removed', 'swapped')]
    x = np.arange(3)
    ax.bar(x, vals, color='#2ca25f', label='accepted and completed')
    ax.bar(x, abstain, bottom=vals, color='#9ecae1', label='safe abstention')
    rejected = [data['summary'][k]['interventions'] for k in ('real', 'removed', 'swapped')]
    ax.bar(x, rejected, bottom=np.asarray(vals)+np.asarray(abstain),
           color='#e57928', label='safety intervention')
    ax.set_xticks(x, names); ax.set_ylabel('tasks (same task bank)')
    ax.set_ylim(0, data['n_eligible'] + 1)
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .02, 'hand-coded reference; not learned adaptation',
            transform=ax.transAxes, fontsize=8, color='0.35')
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_sweep(data, out):
    rows = data['rows']
    speeds = sorted({r['speed'] for r in rows})
    errors = sorted({r['position_error'] for r in rows})
    latency = max(r['latency'] for r in rows)
    subset = [r for r in rows if r['latency'] == latency]
    mat = np.full((len(speeds), len(errors)), np.nan)
    for r in subset:
        mat[speeds.index(r['speed']), errors.index(r['position_error'])] = r['real_successes']
    fig, ax = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    im = ax.imshow(mat, origin='lower', aspect='auto', cmap='viridis',
                   vmin=0, vmax=data['n_tasks'])
    ax.set_xticks(range(len(errors)), [f'{v:.3f}' for v in errors])
    ax.set_yticks(range(len(speeds)), [f'{v:.1f}' for v in speeds])
    ax.set_xlabel('position error (m)')
    ax.set_ylabel('assumed speed (m/s)')
    ax.set_title(f'Reference accepted tasks at latency={latency:.2f}s')
    fig.colorbar(im, ax=ax, label='accepted-and-completed tasks')
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path)
    p.add_argument('--sweep', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if bool(args.reference) == bool(args.sweep):
        p.error('provide exactly one of --reference or --sweep')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.reference:
        with args.reference.open() as f:
            plot_reference(json.load(f), args.out)
    else:
        with args.sweep.open() as f:
            plot_sweep(json.load(f), args.out)
    print(args.out)


if __name__ == '__main__':
    main()
