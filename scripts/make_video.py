import json
import os
import os.path as osp
from BurgersEnv.Burgers import Burgers
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import utils.ptu as ptu
import time
import argparse

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def show(env, save_path=None, save_name=None, title=None, weno_solution='rk4', dx=None):
    fig = plt.figure(figsize = (15, 10))
    ax = fig.add_subplot(2,1,1)
    ax.set_xlim((env.x_low ,env.x_high))
    ymin, ymax = np.min(env.precise_weno_solutions) - 0.1, np.max(env.precise_weno_solutions) + 0.1
    # ymin, ymax = -20, 20
    ax.set_ylim((ymin, ymax))
    lineweno, = ax.plot(env.x_grid, [0 for _ in range(env.num_x)], lw=2, marker='+', label = 'reference')
    linerl, = ax.plot(env.x_grid, [0 for _ in range(env.num_x)],lw=2, marker='o', label = 'RL-WENO')
    lineweno_coarse, = ax.plot(env.x_grid, [0 for _ in range(env.num_x)], marker='*', lw = 2, label = 'WENO')
    # linezero, = ax.plot(env.x_grid, [0 for x in range(env.num_x)], lw=2, linestyle='dashed')

    weno_coarse_grid_euler = env.weno_coarse_grid_euler.copy()
    weno_coarse_grid_rk4 = env.weno_coarse_grid_rk4.copy()

    draw_data = np.zeros((env.num_t, 3*env.num_x))
    draw_data[:,env.num_x:2*env.num_x] = env.RLgrid
    if weno_solution == 'euler':
        draw_data[:, env.num_x*2:env.num_x*3] = weno_coarse_grid_euler[:env.num_t, :]
    elif weno_solution == 'rk4':
        draw_data[:, env.num_x*2:env.num_x*3] = weno_coarse_grid_rk4[:env.num_t, :]

    for t in range(env.num_t):
        draw_data[t, :env.num_x] = env.get_precise_value(t * env.dt) # when doing showing, use the grid values

    error_ax = fig.add_subplot(2,1,2)
    coarse_error = np.zeros(env.num_t)
    RL_error = np.zeros(env.num_t)
    for i in range(env.num_t):
        coarse_error[i] = env.relative_error(draw_data[i, :env.num_x], draw_data[i, 2 * env.num_x:3*env.num_x])
        RL_error[i] = env.relative_error(draw_data[i, :env.num_x], draw_data[i, env.num_x:2*env.num_x])
    RL_error_line, = error_ax.plot(range(env.num_t), RL_error,  'r', lw= 2, label = 'RLWENO_relative_error')
    weno_coarse_error_line, = error_ax.plot(range(env.num_t), coarse_error,  'b', lw = 2, label = 'weno_relative_error')
    RL_error_point, = error_ax.plot([], [], 'ro', markersize = 5)
    weno_coarse_error_point, = error_ax.plot([], [], 'bo', markersize = 5)

    def init():    
        linerl.set_data([], [])
        lineweno.set_data([],[])
        lineweno_coarse.set_data([], [])
        # linezero.set_data([], [])
        RL_error_point.set_data([],[])
        weno_coarse_error_point.set_data([],[])
        linerl.set_label('RL-WENO')
        lineweno.set_label('Reference')
        lineweno_coarse.set_label('WENO')
        return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point#, linezero

    def func(i):
        # print('make animations, step: ', i)
        x = np.linspace(env.x_low, env.x_high, env.num_x)
        yweno = draw_data[i,:env.num_x]
        yrl = draw_data[i, env.num_x:2 * env.num_x]
        yweno_coarse = draw_data[i, 2 * env.num_x: 3 * env.num_x]
        linerl.set_data(x,yrl)
        lineweno.set_data(x, yweno)
        lineweno_coarse.set_data(x, yweno_coarse)
        # linezero.set_data(x, [0 for _ in x])
        RL_error_point.set_data(i, RL_error[i])
        weno_coarse_error_point.set_data(i, coarse_error[i])
        error_ax.set_title("evolving time step: {}".format(i))
        return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point#, linezero


    anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=env.num_t, interval=50)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        anim.save(osp.join(save_path, save_name), writer=writer)
    else:
        plt.show()

    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, default=0)
parser.add_argument('--forcing', type=int, default=0)
parser.add_argument('--dx', type=float, default=0.02)
parser.add_argument('--flux', type=str, default='u2')
args = parser.parse_args()

path = 'data/seuss/9-20-many-64-multiple-dx-normalize-eta-0-all-break/9-20-many-64-multiple-dx-normalize-eta-0-all-break_2020_09_20_04_04_24_0006/'
epoch = '1200'

path = 'data/seuss/9-23-many-64-multiple-dx-normalize-eta-0.01/9-23-many-64-multiple-dx-normalize-eta-0.01_2020_09_25_03_49_37_0005/'
epoch = '12800'

vv = json.load(open(osp.join(path, 'variant.json')))
if vv.get('batch_norm') is None:
    vv['batch_norm'] = False
if vv.get('dx') is None:
    vv['dx'] = args.dx
if vv.get('eta') is None:
    vv['eta'] = args.eta

vv['eta'] = args.eta
if args.flux == 'u2':
    ran = range(25, 50)
    if args.eta == 0:
        # pass
        print("here")
        vv['solution_data_path'] = 'data/local/solutions/8-14-50'
    else:
        vv['solution_data_path'] = 'data/local/solutions/9-17-50-eta-{}'.format(args.eta)

    if args.forcing:
        vv['eta'] = 0.01
        vv['solution_data_path'] = 'data/local/solutions/9-24-50-eta-0.01-forcing-1'
elif args.flux == 'u4':
    vv['flux'] = 'u4'
    vv['solution_data_path'] = 'data/local/solutions/9-24-50-u4-eta-0-forcing-0'
    ran = range(0, 25)


# vv['policy_hidden_layers'] = [64, 64, 64, 64, 64, 64]
# vv['state_mode'] = 'normalize'
ddpg = DDPG(vv, GaussNoise(initial_sig=vv['noise_beg'], final_sig=vv['noise_end']))
agent = ddpg
agent.load(osp.join(path, epoch), actor_only=True)
# agent.load(osp.join('data/local', '6150'), actor_only=True)
env = Burgers(vv, agent=agent)

# ptu.set_gpu_mode(True)

dx = args.dx if not args.forcing else args.dx * np.pi
beg = time.time()
num_t = int(0.9 * 10 / dx) if not args.forcing else int(0.9 * np.pi * 10 / dx)
for solution_idx in ran:
    print("solution_idx: ", solution_idx)

    pre_state = env.reset(solution_idx=solution_idx, num_t=num_t, dx=dx)
    # pre_state = env.reset(solution_idx=solution_idx, num_t=200)
    horizon = env.num_t
    for t in range(1, horizon):
        # print(t)
        action = agent.action(pre_state, deterministic=True) # action: (state_dim -2, 1) batch
        next_state, reward, done, _ = env.step(action, Tscheme='rk4')
        pre_state = next_state

    error, relative_error = env.error('rk4')
    print("relative error: ", relative_error)
    save_path = osp.join(path, 'videos', "{}_{}_{}_{}".format(epoch, vv['eta'], dx, args.flux))
    if not osp.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_name = "{}.mp4".format(solution_idx)
    # show(env, title='{}-{}-{}'.format(solution_idx, epoch, dx), dx=dx)
    show(env, save_path=save_path, save_name=save_name, title='{}-{}'.format(solution_idx, epoch))
print("cost time: ", time.time() - beg)