import numpy as np
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise
from DDPG.train_util import DDPG_test
from DDPG.util import GaussNoise
from get_args import get_args
import torch
import copy
from init.init_util_eval import init_env

args = get_args()
### Important: fix numpy and torch seed! ####
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

### Initialize RL agents 
ddpg = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg, final_sig=args.noise_end))
agent = ddpg

assert args.load_path is not None
agent.load(args.load_path)
print('load models done!')

# quick test
dt_set = [t * 0.001 for t in range(2, 4)]
dx_set = [0.04, 0.05]

# full test
# dt_set = [t * 0.001 for t in range(2, 9)]
# dx_set = [0.02, 0.04, 0.05]

mean_RL_euler = np.zeros((len(dt_set), len(dx_set)))
mean_RL_rk4 = np.zeros((len(dt_set), len(dx_set)))
std_RL_euler = np.zeros((len(dt_set), len(dx_set)))
std_RL_rk4 = np.zeros((len(dt_set), len(dx_set)))

mean_weno_euler = np.zeros((len(dt_set), len(dx_set)))
mean_weno_rk4 = np.zeros((len(dt_set), len(dx_set)))
std_weno_euler = np.zeros((len(dt_set), len(dx_set)))
std_weno_rk4 = np.zeros((len(dt_set), len(dx_set)))

_, _, _, test_envs = init_env(args, agent)
print("true solutions load done!")

for idx_t, dt in enumerate(dt_set):
    for idx_x, dx in enumerate(dx_set):

        for env in test_envs:
            env.args.dx = dx
            env.args.dt = dt
            env.args.cfl = dt / dx
            env.reset()
            env.get_weno_corase()

        print("dt {} dx {} environment prepared ready!".format(dt, dx))
        RL_euler_errors, _ = DDPG_test(agent, test_envs, Tscheme='euler')
        RL_rk4_errors, _ = DDPG_test(agent, test_envs, Tscheme='rk4')
        print("DDPG test done!")

        weno_euler_errors = []
        weno_rk4_errors = []
        for env in test_envs:
            error_euler, error_rk4 = env.get_weno_error(recompute=True)
            weno_euler_errors.append(error_euler)
            weno_rk4_errors.append(error_rk4)
        
        RL_euler_mean, RL_euler_std = np.mean(RL_euler_errors), np.std(RL_euler_errors)
        RL_rk4_mean, RL_rk4_std = np.mean(RL_rk4_errors), np.std(RL_rk4_errors)

        weno_euler_mean, weno_euler_std = np.mean(weno_euler_errors), np.std(weno_euler_errors)
        weno_rk4_mean, weno_rk4_std = np.mean(weno_rk4_errors), np.std(weno_rk4_errors)

        mean_RL_euler[idx_t][idx_x] = RL_euler_mean
        std_RL_euler[idx_t][idx_x] = RL_euler_std
        mean_RL_rk4[idx_t][idx_x] = RL_rk4_mean
        std_RL_rk4[idx_t][idx_x] = RL_rk4_std

        mean_weno_euler[idx_t][idx_x] = weno_euler_mean
        std_weno_euler[idx_t][idx_x] = weno_euler_std
        mean_weno_rk4[idx_t][idx_x] = weno_rk4_mean
        std_weno_rk4[idx_t][idx_x] = weno_rk4_std


np.save('RL_euler_mean', mean_RL_euler)
np.save('RL_euler_std', std_RL_euler)
np.save('RL_rk4_mean', mean_RL_rk4)
np.save('RL_rk4_std', std_RL_rk4)
np.save('weno_euler_mean', mean_weno_euler)
np.save('weno_euler_std', std_weno_euler)
np.save('weno_rk4_mean', mean_weno_rk4)
np.save('weno_rk4_std', std_weno_rk4)

schemes = ['euler', 'rk4']
mean_RLs = [mean_RL_euler, mean_RL_rk4]
std_RLs = [std_RL_euler, std_RL_rk4]
mean_wenos = [mean_weno_euler, mean_weno_rk4]
std_wenos = [std_weno_euler, std_weno_rk4]
for scheme_idx in range(2):
    print("Tscheme: ", schemes[scheme_idx])
    mean_RL = mean_RLs[scheme_idx]
    std_RL = std_RLs[scheme_idx]
    mean_weno = mean_wenos[scheme_idx]
    std_weno = std_wenos[scheme_idx]

    for idx_t in range(len(dt_set)):
        print(dt_set[idx_t], end = ' ')
        for idx_x in range(len(dx_set)):
            print(r'& {:.2f} ({:.2f}) & {:.2f} ({:.2f})'.format(
                mean_RL[idx_t][idx_x] * 100, 
                std_RL[idx_t][idx_x] * 100,
                mean_weno[idx_t][idx_x] * 100,
                std_weno[idx_t][idx_x] * 100
                ), end = '')
        print(r'\\ \hline')

    print()

