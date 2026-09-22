"""Animate recorded oracle-reference paths, not rigid-body or learned flight."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def frames(row, branch, spacing=.08):
    segments = [('probe', row['probe']['trace']),
                ('return', row['retreat_trace']),
                ('retry', row['branches'][branch]['trace'])]
    result = []
    for phase, path in segments:
        for a, b in zip(path, path[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(np.array(b)-a)/spacing))+1)
            result.extend((phase, p) for p in np.linspace(a, b, count))
    if not result:
        raise ValueError('No recorded motion')
    result.extend([('end', result[-1][1])] * 15)
    return result


def animate(data, out, task_id=1, branch='real'):
    row = next(r for r in data['tasks'] if r['task_id'] == task_id)
    if not row['eligible']:
        raise ValueError('Task has no completed reference probe')
    sequence = frames(row, branch)
    task = row['task']
    radius = data['safety']['radius']
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-.3, 3.4), ylim=(-1, 1), zlim=(0, 2.2),
           xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((3.7, 2, 2.2))
    ax.view_init(elev=28, azim=-58)
    # Extrusion of the actual 2D reference obstacles, NOT a new gap model.
    for x, width, center in [(2., 1.4, 0.), (2.15, task['width'], task['center'])]:
        depth = .15 if x == 2. else .25
        for lo, hi in [(-1, center-width/2), (center+width/2, 1)]:
            ax.bar3d(x, lo, 0, depth, hi-lo, 2.2, color='#8998a6', alpha=.3)
    route, = ax.plot([], [], [], color='#2878b5', lw=2)
    body, = ax.plot([], [], [], color='#e58b24', lw=2)
    arms, = ax.plot([], [], [], color='#252b35', lw=3)
    text = ax.text2D(.02, .94, '', transform=ax.transAxes)
    fig.suptitle('Recorded geometric reference - NOT learned / rigid-body flight')
    angle = np.linspace(0, 2*np.pi, 40)
    outcome = row['branches'][branch]

    def update(i):
        phase, p = sequence[i]
        past = np.array([s[1] for s in sequence[:i+1]])
        route.set_data_3d(past[:, 0], past[:, 1], np.full(len(past), 1.1))
        body.set_data_3d(p[0]+radius*np.cos(angle), p[1]+radius*np.sin(angle),
                         np.full(len(angle), 1.1))
        arms.set_data_3d(p[0]+radius*np.array([-1, 1, 0, 0, 0]),
                         p[1]+radius*np.array([0, 0, 0, -1, 1]), np.full(5, 1.1))
        label = phase
        if phase == 'end':
            label = 'completed' if outcome['success'] else (
                'request rejected' if outcome['intervention'] else 'abstained')
        text.set_text(f'Task {task_id} | {branch} | {label}\n'
                      'Interpolated waypoints; playback time is not physical time')
        return route, body, arms, text

    anim = FuncAnimation(fig, update, frames=len(sequence), interval=80)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', type=int, default=1)
    p.add_argument('--branch', choices=['real', 'removed', 'swapped'], default='real')
    a = p.parse_args()
    animate(json.loads(a.input.read_text()), a.out, a.task, a.branch)


if __name__ == '__main__':
    main()
