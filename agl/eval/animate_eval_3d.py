"""Animate a recorded GapEnv evaluation trajectory in 3-D.

Unlike the reference demo, this consumes actual ``evaluate.py`` NPZ records.
It is still an offline playback: the animation cannot establish safety or
replace a live simulator.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from ..viz.episode import load_eval_episode, quat_rotate_wxyz


def _block(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    return [[v[i] for i in f] for f in ((0,1,2,3),(4,5,6,7),(0,1,5,4),
                                        (2,3,7,6),(1,2,6,5),(0,3,7,4))]


def _wall_faces(task):
    wx, th = task['wall_x'], task['thick']
    cy, cz, hw, hh = task['gap_cy'], task['gap_cz'], task['gap_w']/2, task['gap_h']/2
    x0, x1 = wx, wx + th
    # The renderer's gap is rolled in the y-z plane. The slab remains
    # axis-aligned; the rotated opening outline is drawn separately below.
    return _block(x0, x1, -5, 5, 0, 4.5)


def _rotate_gap_outline(task):
    cy, cz = task['gap_cy'], task['gap_cz']
    hw, hh, roll = task['gap_w']/2, task['gap_h']/2, task['gap_roll']
    corners = np.array([[-hw,-hh], [hw,-hh], [hw,hh], [-hw,hh], [-hw,-hh]])
    r = np.array([[np.cos(roll), -np.sin(roll)],
                  [np.sin(roll), np.cos(roll)]])
    yz = corners @ r.T + np.array([cy, cz])
    x = np.full(len(yz), task['wall_x'] - .002)
    return np.c_[x, yz]


def load_record(path, task_id):
    d = np.load(path)
    for key in ('rec_p','rec_v','rec_clear','rec_attempt_id','rec_collision'):
        if key not in d:
            raise KeyError(f'missing {key}')
    n = d['rec_p'].shape[1]
    if not 0 <= task_id < n:
        raise ValueError(f'task must be in [0,{n})')
    steps = int(d['steps'][task_id]) if 'steps' in d else d['rec_p'].shape[0]
    steps = min(steps, d['rec_p'].shape[0])
    task = {k: float(d[f'task_{k}'][task_id]) for k in
            ('wall_x','thick','gap_cy','gap_cz','gap_w','gap_h','gap_roll')}
    return {k: d[f'rec_{k}'][:steps, task_id] for k in
            ('p','v','clear','attempt_id','collision','end_event') if f'rec_{k}' in d}, task


def animate(path, out, task_id=0, fps=20):
    rec, task = load_record(path, task_id)
    ep = load_eval_episode(path, task_id)
    p = rec['p']; v = rec['v']; clear = rec['clear']; attempt = rec['attempt_id']; collision = rec['collision']
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(float(p[:,0].min())-.4, max(task['wall_x']+task['thick']+.2,float(p[:,0].max())+.4)),
           ylim=(min(float(p[:,1].min())-.4,task['gap_cy']-task['gap_w']/2-.3),
                 max(float(p[:,1].max())+.4,task['gap_cy']+task['gap_w']/2+.3)),
           zlim=(min(float(p[:,2].min())-.4,task['gap_cz']-task['gap_h']/2-.3),
                 max(float(p[:,2].max())+.4,task['gap_cz']+task['gap_h']/2+.3)), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((3.8,4,3)); ax.view_init(elev=24, azim=-62)
    # Wireframe opening avoids drawing a solid slab across the actual hole.
    outline = _rotate_gap_outline(task)
    ax.plot(outline[:,0], outline[:,1], outline[:,2], color='#b23a48', lw=2)
    ax.text(task['wall_x'], task['gap_cy'], task['gap_cz']+task['gap_h']/2+.15, 'recorded gap', color='#b23a48')
    path_line, = ax.plot([], [], [], color='#2878b5', lw=2); drone, = ax.plot([], [], [], 'o', color='#e58b24', ms=8)
    arms, = ax.plot([], [], [], color='#e58b24', lw=3)
    fig.suptitle(ep.meta.get('scope', 'Recorded GapEnv evaluation'), fontsize=10)
    if ep.meta.get('info_gate_enabled'):
        px = task['wall_x'] - ep.meta['info_probe_distance']
        ax.plot([px,px,px,px,px], [-2,2,2,-2,-2], [.3,.3,3.2,3.2,.3], '--', color='#477a54')
    dt = ep.meta.get('dt_ctrl', .025)
    # Preserve physical duration; fps changes sampling, not simulation time.
    indices = np.unique(np.r_[np.minimum((np.arange(0, len(p)*dt, 1/fps)/dt).astype(int), len(p)-1), len(p)-1])
    status = ax.text2D(.02,.94,'', transform=ax.transAxes); info = ax.text2D(.02,.88,'', transform=ax.transAxes, fontsize=9)
    def update(i):
        q=p[:i+1]; path_line.set_data_3d(q[:,0],q[:,1],q[:,2]); drone.set_data_3d([p[i,0]],[p[i,1]],[p[i,2]])
        c=float(clear[i]); col=bool(collision[i]); phase='CONTACT' if col else ('attempt '+str(int(attempt[i])) if attempt[i]>0 else 'approach')
        phase = ep.phase(i)
        body = quat_rotate_wxyz(ep.rec['q'][i], np.array([[-.16,0,0],[.16,0,0],[0,0,0],[0,-.16,0],[0,.16,0]])) + p[i]
        arms.set_data_3d(body[:,0],body[:,1],body[:,2])
        status.set_text(f't={i*dt:.3f}s | task {task_id} | {phase} | collision={int(col)}'); info.set_text(f'x={p[i,0]:.2f}  speed={np.linalg.norm(v[i]):.2f} m/s  interval clearance={c:.3f} m')
        drone.set_color('#d62728' if col else '#e58b24'); return path_line,drone,status,info
    ani=FuncAnimation(fig,update,frames=indices,interval=1000/fps,blit=False); out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True); ani.save(out,writer=PillowWriter(fps=fps)); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--task',type=int,default=0); p.add_argument('--fps',type=int,default=20); a=p.parse_args(); animate(a.input,a.out,a.task,a.fps)

if __name__ == '__main__': main()
