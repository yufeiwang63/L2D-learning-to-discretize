"""
created 02/03/2020, yufei
visualize RL-weno weights and weno weights at singularities.
"""

from get_args import get_args
from init_util import init_env
from DDPG.DDPG_new import DDPG
from DDPG.train_util import DDPG_train, DDPG_test
from DDPG.util import GaussNoise
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

font = {
        'size'   : 15}

matplotlib.rc('font', **font)

def process_singularity(env):
    grid = copy.deepcopy(env.RLgrid)
    new_grid = np.zeros((grid.shape[0], grid.shape[1] + 2))
    new_grid[:, 1:-1] = grid
    new_grid[:, 0] = grid[:, -1]
    new_grid[:, -1] = grid[:, 0]
    dx = (new_grid[:, 1:] - new_grid[:, :-1]) / env.dx
    group1_idx = dx <= -40
    group2_idx = dx <= -5

    group1 = []
    group2 = []
    all_statess = []
    for idx in range(grid.shape[0]):
        print("dealing with row {}".format(idx))
        all_states = env.get_state(grid[idx])
        all_states = np.asarray(all_states)
        all_statess.append(all_states)
        for j in range(grid.shape[1]):
            if group1_idx[idx][j]:
                group1.append(all_states[j])
            if group2_idx[idx][j]:
                group2.append(all_states[j])

    return np.asarray(group1).reshape((-1, 7)), np.asarray(group2).reshape((-1, 7)), np.asarray(all_statess).reshape((-1, 7))

def plot(rl_actions, weno_actions, data):
    rl_actions = rl_actions
    weno_actions = weno_actions
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    actions = [rl_actions, weno_actions]
    labels = ["RL-WENO", "WENO"]
    fmts = ['ro', 'b*']
    for idx in range(len(actions)):
        action = actions[idx][:, :4]
        label = labels[idx]
        mean_a = np.mean(action, axis=0)
        std_a = np.std(action, axis=0)        
        ax1.plot(range(len(mean_a)), mean_a, fmts[idx], label=label, markersize=10)
        ax1.plot(range(len(mean_a)), mean_a + std_a, linewidth=0.001)
        ax1.plot(range(len(mean_a)), mean_a - std_a, linewidth=0.001)
        ax1.fill_between(range(len(mean_a)), mean_a - std_a, mean_a + std_a,
            alpha=0.2, facecolor=fmts[idx][0],
            linewidth=1, antialiased=True)
        # markers, caps, bars = ax1.errorbar(range(len(mean_a)), mean_a, std_a, fmt = fmts[idx], label=label + '_left')
        # # [marker.set_alpha(0.1) for marker in markers]
        # [cap.set_alpha(0.2) for cap in caps]
        # [bar.set_alpha(0.2) for bar in bars]

    for idx in range(len(actions)):
        action = actions[idx][:, 4:]
        label = labels[idx]
        mean_a = np.mean(action, axis=0)
        std_a = np.std(action, axis=0)        
        ax2.plot(range(len(mean_a)), mean_a, fmts[idx], label=label, markersize=10)
        ax2.plot(range(len(mean_a)), mean_a + std_a, linewidth=0.001)
        ax2.plot(range(len(mean_a)), mean_a - std_a, linewidth=0.001)
        ax2.fill_between(range(len(mean_a)), mean_a - std_a, mean_a + std_a,
            alpha=0.2, facecolor=fmts[idx][0],
            linewidth=1, antialiased=True)
        # markers, caps, bars = ax2.errorbar(range(len(mean_a)), mean_a, std_a,  fmt = fmts[idx], label=label + '_right')
        # # [marker.set_alpha(0.1) for marker in markers]
        # [cap.set_alpha(0.2) for cap in caps]
        # [bar.set_alpha(0.2) for bar in bars]

    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels([r"$w_{3-1/2}^{-2}$", r"$w_{3-1/2}^{-1}$", r"$w_{3-1/2}^0$", r"$w_{3-1/2}^1$"])
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([r"$w_{3+1/2}^{-2}$", r"$w_{3+1/2}^{-1}$", r"$w_{3+1/2}^0$", r"$w_{3+1/2}^1$"]) 
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()

    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    ax3.plot(range(len(mean_data)), mean_data, "-o", markersize=8, label="state")
    ax3.plot([3], mean_data[3], 'b+', markersize=25)
    # ax3.plot(range(len(mean_data)), mean_data + std_data, linestyle='dashed')
    # ax3.plot(range(len(mean_data)), mean_data - std_data, linestyle='dashed')
    ax3.fill_between(range(len(mean_data)), mean_data - std_data, mean_data + std_data,
        alpha=0.5, #edgecolor='purple', facecolor='cyan',
        linewidth=1, antialiased=True)
    # ax3.errorbar(range(len(mean_data)), mean_data, std_data, label='data')
    ax3.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax3.set_xticklabels([r"$u_{0}$", r"$u_{1}$", r"$u_{2}$", r"$u_{3}$", r"$u_{4}$", r"$u_{5}$", r"$u_{6}$", r"$u_{7}$"])
    ax3.legend()
    # [bar.set_alpha(0.5) for bar in bars]    
    # [cap.set_alpha(0.5) for cap in caps]

args, tb_writter = get_args()
agent = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg, final_sig=args.noise_end), tb_logger=tb_writter)
_, _, train_env, test_env = init_env(args, agent)

if args.load_path is not None:
    agent.load(args.load_path)

print('=' * 50, 'load models!', '=' * 50)
DDPG_test(agent, test_env, args)

for env in test_env:
    singularities_a, singularities_b, all_states = process_singularity(env)
    print(singularities_a.shape)
    print(singularities_b.shape)
    print(all_states.shape)
    RL_weno_actions = agent.action(singularities_a, deterministic=True)
    weno_actions = agent.action(singularities_a, mode='weno')
    plot(RL_weno_actions, weno_actions, singularities_a)

    plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig('./figs/weight_visual_{}.png'.format(env.args.init))
    plt.clf()
    plt.cla()



