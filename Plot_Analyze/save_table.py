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
import init_util
import matplotlib
from matplotlib import pyplot as plt

from get_args import get_args
from PPO.train_util import PPO_train, ppo_test, PPO_FCONV_train
from DDPG.train_util import DDPG_train, DDPG_test
from PPO.PPO_new import PPO
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise

import pickle

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

dx_set=[0.02,0.04,0.05,0.08]
dt_set=[0.002,0.003,0.004,0.005,0.006,0.007,0.008]
if not args.large_scale_train:
    #init_funcs = [init_util.init_condition_a, init_util.init_condition_b, init_util.init_condition_c, init_util.init_condition_d, 
    #init_util.init_condition_e, init_util.init_condition_e1, init_util.init_condition_e2, init_util.init_condition_e3, init_util.init_condition_e4, init_util.init_condition_e5, 
    #init_util.init_condition_k, init_util.init_condition_l]
    #init_name=['1_cos4','-1_cos4','-1_p2sin4','-1.5_p2sin4','1.5_m1.5sin4',
    #            '1_cos2','-1_cos2','-1_p2sin2','-1.5_p2sin2','1.5_m1.5sin2','-1.5_p2cos2','1.5_m1.5cos2']
    #init_funcs = [init_util.init_condition_a, init_util.init_condition_b, init_util.init_condition_c,#, init_util.init_condition_d,
    #init_util.init_condition_e, init_util.init_condition_e1, init_util.init_condition_e2, init_util.init_condition_e3, init_util.init_condition_e4]#, init_util.init_condition_e5, 
    #init_util.init_condition_k, init_util.init_condition_l]
    init_name=['1_cos4','-1_cos4','-1_p2sin4','1.5_m1.5sin4',
                '1_cos2','-1_cos2','-1_p2sin2','-1.5_p2sin2','1.5_m1.5sin2','-1.5_p2cos2','1.5_m1.5cos2']
    #train init
    init_funcs = [init_util.init_condition_t1, init_util.init_condition_t2, init_util.init_condition_t3,
                  init_util.init_condition_t5]
else:
    pass

init_num=4

if args.load_path is not None:
    print('load models!')
    agent.load(args.load_path)


answerfile=open(args.save_table_path+args.Tscheme+'_'+args.flux+'.txt','w')

for dx in dx_set:
    for dt in dt_set:
        RL_error_set=[]
        coarse_error_set=[]
        #if dt/dx>args.cfl:
        #    print("{} {} no".format(dx,dt))
        #    answerfile.write("RL {} {} no no coarse {} {} no no\n".format(dx,dt,dx,dt))
        #    continue

        for i in range(init_num):
            exp_args=copy.copy(args)
            exp_args.T=args.T
            exp_args.dx=dx
            exp_args.dt=dt
            exp_args.init=init_name[i]
            exp_args.Tscheme=args.Tscheme
            env= Burgers(exp_args,init_funcs[i],agent=agent)
            env.get_weno_precise()
            env.get_weno_corase()
            errors=DDPG_test(agent,[env],exp_args)
            coarse_error = np.zeros(env.num_t)
            RL_error = np.zeros(env.num_t)
            for i in range(env.num_t):
                coarse_error[i] = env.relative_error(env.get_precise_value(i * env.dt),env.weno_coarse_grid[i])
                RL_error[i] = env.relative_error(env.get_precise_value(i * env.dt),env.RLgrid[i])
            RL_mean_error=np.mean(RL_error)
            coarse_mean_error=np.mean(coarse_error)
            RL_error_set.append(RL_mean_error)
            coarse_error_set.append(coarse_mean_error)
        RL_error_set=np.array(RL_error_set)
        coarse_error_set=np.array(coarse_error_set)
        RL_mean=np.mean(RL_error_set)
        coarse_mean=np.mean(coarse_error_set)
        RL_std=np.std(RL_error_set)
        coarse_std=np.std(coarse_error_set)
        if np.isnan(RL_mean):
            RL_mean='out'
            RL_std='out'
        if np.isnan(coarse_mean):
            coarse_mean='out'
            coarse_std='out'
        print("dx:{},dt:{},RL error:{},RL std:{},coarse error:{},coarse std:{}".format(dx,dt,RL_mean,RL_std,coarse_mean,coarse_std))
        answerfile.write("RL {} {} {} {} coarse {} {} {} {}\n".format(dx,dt,RL_mean,RL_std,dx,dt,coarse_mean,coarse_std))

answerfile.close()
