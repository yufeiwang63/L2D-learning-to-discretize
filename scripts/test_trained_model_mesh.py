'''
This file tests RL-WENO under non-viscous burgers, without forcing, with different initial conditions.
'''

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
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--flux', type=str, default='u2')
args = parser.parse_args()

path = 'data/seuss/9-20-many-64-multiple-dx-normalize-eta-0-all-break/9-20-many-64-multiple-dx-normalize-eta-0-all-break_2020_09_20_04_04_24_0006/'
epoch = '1200'

vv = json.load(open(osp.join(path, 'variant.json')))
if vv.get('batch_norm') is None:
    vv['batch_norm'] = False
if vv.get('dx') is None:
    vv['dx'] = 0.04
if vv.get('eta') is None:
    vv['eta'] = 0

vv['flux'] = args.flux
if args.flux == 'u2':
    vv['solution_data_path'] = 'data/local/solutions/8-14-50'
elif args.flux == 'u4':
    vv['solution_data_path'] = 'data/local/solutions/9-24-50-u4-eta-0-forcing-0'
else:
    raise NotImplementedError

print(vv)


ddpg = DDPG(vv, GaussNoise(initial_sig=vv['noise_beg'], final_sig=vv['noise_end']))
agent = ddpg
agent.load(osp.join(path, epoch), actor_only=True)
print("agent loaded!")

env = Burgers(vv, agent=agent)

dt_list = [0.002, 0.003, 0.004, 0.005, 0.006]
dx_list = [0.02, 0.04, 0.05]

len_dt_list = len(dt_list)
RL_errors = np.zeros((len(dx_list), len_dt_list, 2))
weno_errors = np.zeros((len(dx_list), len_dt_list, 2))
RL_all_errors = []
weno_all_errors = []

for x_idx, dx in enumerate(dx_list):
    for t_idx, dt in enumerate(dt_list):
        print("test on dt {} dx {}".format(dt, dx))
        num_t = int(0.9 / dt)
        RL_error = []
        weno_error = []
        for solution_idx in range(25):
            pre_state = env.reset(solution_idx=solution_idx, num_t=num_t, dx=dx, dt=dt, weno_regenerate=True)
            horizon = env.num_t
            for t in range(1, horizon):
                action = agent.action(pre_state, deterministic=True) # action: (state_dim -2, 1) batch
                next_state, reward, done, _ = env.step(action, Tscheme='rk4')
                pre_state = next_state
            error, relative_error = env.error('rk4')
            weno_error_rk4 = env.weno_error_rk4
            RL_error.append(error)
            weno_error.append(weno_error_rk4)
            print("solution_idx {} relative error {}".format(solution_idx, relative_error))
        
        RL_all_errors.append(RL_error)
        weno_all_errors.append(weno_error)
        RL_errors[x_idx][t_idx][0] = np.mean(RL_error)
        weno_errors[x_idx][t_idx][0] = np.mean(weno_error)
        RL_errors[x_idx][t_idx][1] = np.std(RL_error)
        weno_errors[x_idx][t_idx][1] = np.std(weno_error)

# data_dict = {
#     'dt_list': dt_list,
#     'dx_list': dx_list,
#     'error_RL': RL_errors,
#     'error_weno': weno_errors,
#     'all_errors_RL': RL_all_errors,
#     'all_errors_weno': weno_all_errors
# }

# torch.save(data_dict, 'data/ComPhy/RL_and_weno_error_mesh_and_flux_{}.pkl'.format(args.flux))

for t_idx, dt in enumerate(dt_list):
    print(dt, end = ' ')
    for x_idx, dx in enumerate(dx_list):
        RL_error_mean = round(RL_errors[x_idx][t_idx][0] * 100, 2)
        weno_error_mean = round(weno_errors[x_idx][t_idx][0] * 100, 2)
        RL_error_std = round(RL_errors[x_idx][t_idx][1] * 100, 2)
        weno_error_std = round(weno_errors[x_idx][t_idx][1] * 100, 2)
        print("\& {} ({}) \& {} ({})".format(RL_error_mean, RL_error_std, 
            weno_error_mean, weno_error_std), end=' ')
    print('\\\hline')


# for x_idx, dx in enumerate(dx_list):
#     print("({}, {})".format(dx, dx * 0.1), end='')
#     RL_error = round(RL_errors[x_idx] * 100, 2)
#     weno_error = round(weno_errors[x_idx] * 100, 2)
#     print("\& {} \& {}".format(RL_error, weno_error), end=' ')
#     print('\\\hline')