'''
this file tests RL-WENO under the setting of L3D.
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
parser.add_argument('--eta', type=float, default=0)
args = parser.parse_args()

# non-viscous model
# path = 'data/seuss/9-20-many-64-multiple-dx-normalize-eta-0-all-break/9-20-many-64-multiple-dx-normalize-eta-0-all-break_2020_09_20_04_04_24_0006/'
# epoch = '1200'

# forcing modle 
# path = 'data/seuss/9-25-many-64-multiple-dx-forcing-eta-0.01/9-25-many-64-multiple-dx-forcing-eta-0.01_2020_09_25_05_35_10_0006/'
# epoch = '4900'

# viscous model
path = 'data/seuss/9-23-many-64-multiple-dx-normalize-eta-0.01/9-23-many-64-multiple-dx-normalize-eta-0.01_2020_09_25_03_49_37_0005/'
epoch = '12800'


vv = json.load(open(osp.join(path, 'variant.json')))
if vv.get('batch_norm') is None:
    vv['batch_norm'] = False
if vv.get('dx') is None:
    vv['dx'] = 0.04
if vv.get('eta') is None:
    vv['eta'] = 0


eta_list = [0.01, 0.02, 0.04]
dx_list = [0.02, 0.04, 0.05]
dx_list = [dx * np.pi for dx in dx_list]

RL_errors = np.zeros((len(eta_list), len(dx_list), 2))
weno_errors = np.zeros((len(eta_list), len(dx_list), 2))

for eta_idx, eta in enumerate(eta_list): 
    vv['eta'] = eta
    vv['solution_data_path'] = 'data/local/solutions/9-24-50-eta-{}-forcing-1-regenerate'.format(eta)

    print(vv)

    ddpg = DDPG(vv, GaussNoise(initial_sig=vv['noise_beg'], final_sig=vv['noise_end']))
    agent = ddpg
    agent.load(osp.join(path, epoch), actor_only=True)
    print("agent loaded!")

    env = Burgers(vv, agent=agent)

    len_dt_list = 1 
   
    RL_all_errors = []
    weno_all_errors = []

    for x_idx, dx in enumerate(dx_list):
        dt_list = [dx * 0.1]

        for t_idx, dt in enumerate(dt_list):
            print("test on eta {} dx {} dt {}".format(eta, dx, dt))
            num_t = int(0.9 * np.pi / dt)
            RL_error = []
            weno_error = []
            for solution_idx in range(25, 50):
                pre_state = env.reset(solution_idx=solution_idx, num_t=num_t, dx=dx, dt=dt)
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
            RL_errors[eta_idx][x_idx][0] = np.mean(RL_error)
            weno_errors[eta_idx][x_idx][0] = np.mean(weno_error)
            RL_errors[eta_idx][x_idx][1] = np.std(RL_error)
            weno_errors[eta_idx][x_idx][1] = np.std(weno_error)

data_dict = {
    'eta_list': eta_list,
    'dx_list': dx_list,
    'error_RL': RL_errors,
    'error_weno': weno_errors,
    'all_errors_RL': RL_all_errors,
    'all_errors_weno': weno_all_errors
}

torch.save(data_dict, 'data/ComPhy/RL_and_weno_error_forcing_eta_{}.pkl'.format(epoch))

for x_idx, dx in enumerate(dx_list):
    for eta_idx, eta in enumerate(eta_list):
        RL_error_mean = round(RL_errors[eta_idx][x_idx][0] * 100, 2)
        weno_error_mean = round(weno_errors[eta_idx][x_idx][0] * 100, 2)
        RL_error_std = round(RL_errors[eta_idx][x_idx][1] * 100, 2)
        weno_error_std = round(weno_errors[eta_idx][x_idx][1] * 100, 2)
        print("& {} ({}) & {} ({})".format(RL_error_mean, RL_error_std, 
            weno_error_mean, weno_error_std), end=' ')
    print('\\\\\\hline')

