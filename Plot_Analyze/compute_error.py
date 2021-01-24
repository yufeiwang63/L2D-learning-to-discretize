'''
from ziju shen. ask him for doc.
'''

from DDPG.DDPG_new import DDPG
from BurgersEnv.Burgers import Burgers
from Weno.weno3_2 import Weno3
import copy
import torch
import numpy as np
import copy
import os
import time
import math
import random
import init_util_2 as init_util

from get_args import get_args
from PPO.train_util import PPO_train, ppo_test, PPO_FCONV_train
from DDPG.train_util import DDPG_train, DDPG_test
from PPO.PPO_new import PPO
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise

args, tb_writter = get_args()


### Important: fix numpy and torch seed! ####
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

print('State dimension: {0}'.format(args.state_dim))
print('Action dimension: {0}'.format(args.action_dim))

### RL agents 
agent = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg), tb_logger=tb_writter)
#agent_dic = {'ppo': ppo, 'ddpg': ddpg}
#agent = agent_dic[args.agent]

### init training and testing encironments
if not args.large_scale_train:
    from init_util import init_env
else:  ### if we want train on large scales
    from init_util_loadprecomputed import init_env
#argss, train_env, test_env = init_env(args, agent)

#args.num_train = len(train_env)
#args.num_test = len(test_env)
#num_train, num_test = len(train_env), len(test_env)

dx_set=[0.01,0.02,0.04]
dt_max=0.02*args.cfl
dt_set=[dt_max,dt_max/2,dt_max/4]
Tscheme_set=['euler','rk4']
flux_set=['u2','u4']
if not args.large_scale_train:
    init_funcs = [init_util.init_condition_a, init_util.init_condition_b, init_util.init_condition_c, init_util.init_condition_d, 
    init_util.init_condition_e, init_util.init_condition_e1, init_util.init_condition_e2, init_util.init_condition_e3, init_util.init_condition_e4, init_util.init_condition_e5, 
    init_util.init_condition_k, init_util.init_condition_l]
    init_name=['1_cos4','-1_cos4','-1_p2sin4','-1.5_p2sin4','1.5_m1.5sin4',
                '1_cos2','-1_cos2','-1_p2sin2','-1.5_p2sin2','1.5_m1.5sin2','-1.5_p2cos2','1.5_m1.5cos2']
else:
    pass







#############################################################################
#################### Initialization Over ###################################
#############################################################################
#### training ####
if args.load_path is not None:
    print('load models!')
    agent.load(args.load_path)


###### different dx experiement :rk4
print("generate different dx")
for i in range(0):
    for dx in dx_set:
        exp_args=copy.copy(args)
        exp_args.T=0.5
        exp_args.dx=dx
        exp_args.init=init_name[i]
        exp_args.dt=dx*exp_args.cfl
        exp_args.save_RL_weno_animation_path+='dx/'
        exp_args.Tscheme='rk4'
        exp_args.flux='u2'
        print(exp_args.save_RL_weno_animation_path)
        env= Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)

for i in range(0):
    for dx in dx_set:
        exp_args=copy.copy(args)
        exp_args.T=0.5
        exp_args.dx=dx
        exp_args.init=init_name[i]
        exp_args.dt=dx*exp_args.cfl
        exp_args.save_RL_weno_animation_path+='dx/'
        exp_args.flux='u2'
        exp_args.Tscheme='euler'
        print(exp_args.save_RL_weno_animation_path)
        env= Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)

###### different dt experiement: rk4
print("generate different dt")
for i in range(0):
    for dt in dt_set:
        exp_args=copy.copy(args)
        exp_args.T=1
        exp_args.dx=0.02
        exp_args.dt=dt
        exp_args.init=init_name[i]
        exp_args.save_RL_weno_animation_path+='dt/'
        exp_args.Tscheme='rk4'
        exp_args.flux='u2'
        print(exp_args.save_RL_weno_animation_path)
        env=Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)

print("generate different dt")
for i in range(0):
    for dt in dt_set:
        exp_args=copy.copy(args)
        exp_args.T=1
        exp_args.dx=0.02
        exp_args.dt=dt
        exp_args.init=init_name[i]
        exp_args.save_RL_weno_animation_path+='dt/'
        exp_args.flux='u2'
        exp_args.Tscheme='euler'
        print(exp_args.save_RL_weno_animation_path)
        env=Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)

####### different flux
for i in range(10):
    for tscheme in Tscheme_set:
        exp_args=copy.copy(args)
        exp_args.T=1
        exp_args.dx=0.02
        exp_args.dt=0.02*args.cfl
        exp_args.init=init_name[i]
        exp_args.Tscheme=tscheme
        exp_args.flux='u2'
        exp_args.cfl=0.2
        exp_args.save_RL_weno_animation_path+='flux/'
        print(exp_args.save_RL_weno_animation_path)
        env=Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)


for i in range(10):
    for tscheme in Tscheme_set:
        exp_args=copy.copy(args)
        exp_args.T=1
        exp_args.dx=0.02
        exp_args.dt=0.02*0.1
        exp_args.init=init_name[i]
        exp_args.Tscheme=tscheme
        exp_args.flux='u4'
        exp_args.cfl=0.1
        exp_args.save_RL_weno_animation_path+='flux/'
        print(exp_args.save_RL_weno_animation_path)
        env=Burgers(exp_args,init_funcs[i],agent=agent)
        env.get_weno_precise()
        env.get_weno_corase()
        errors=DDPG_test(agent,[env],exp_args)
    


        



