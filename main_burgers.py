# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:43:06 2018

@author: Wangyf
"""
'''
File Description:
The main driver file for training / evaluating the agents.
1) parses the command line arguments
2) inits the Burgers Training/Testing Environments, the RL agents
3) perform training/evaluating as specified by the command line arugments.
'''

import torch
import numpy as np
import copy
import os
import time
import math
import random

from get_args import get_args
from PPO.train_util import PPO_train, ppo_test, PPO_FCONV_train
from DDPG.train_util import DDPG_train, DDPG_test
from PPO.PPO_new import PPO
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise

args, tb_writter = get_args()

### Important: fix numpy and torch seed! ####
np.random.seed(args.np_rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.torch_rng_seed)

print('State dimension: {0}'.format(args.state_dim))
print('Action dimension: {0}'.format(args.action_dim))

### Initialize RL agents 
ppo = PPO(copy.copy(args), tb_writter)
ddpg = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg, final_sig=args.noise_end), tb_logger=tb_writter)
agent_dic = {'ppo': ppo, 'ddpg': ddpg}
agent = agent_dic[args.agent]

### Initialize training and testing encironments
if not args.large_scale_train:
    from init_util import init_env
else:  ### if we want train on large scales
    from init_util_loadprecomputed import init_env
argss, train_env, test_env = init_env(args, agent)
random.shuffle(train_env)

args.num_train = len(train_env)
args.num_test = len(test_env)
num_train, num_test = len(train_env), len(test_env)

### write record logs
if not args.test:
    log_file = open(args.save_path + 'log.txt', 'w')
    train_log_file = open(args.save_path + 'train_log.txt', 'w')
    log_file.write(str(args))
    log_file.write('\n')
    for i in range(num_train):
        tmparg = train_env[i].args
        dx, dt, T, init_t, init_condition = tmparg.dx, tmparg.dt, tmparg.T, tmparg.initial_t, tmparg.init

        log_file.write('----------  Initial Condition idx {0}, initial condition {1} ------------\n'.format(i, init_condition))
        log_file.write('dx {0}\t dt {1}\t T {2} initial_t{3} \n'.format(dx, dt, T, init_t))

    log_file.flush()

#############################################################################
##################### Initialization Over ###################################
#############################################################################

#### Start training ####
### load pretrained models if given
if args.load_path is not None:
    print('load models!')
    agent.load(args.load_path)

### evaluate models and exit
if args.test or args.animation or args.compute_tvd:
    # for env in test_env:
    #     env.show()
    # exit()
    if args.agent == 'ppo':
        ppo_test(agent, test_env, args)
        exit()
    elif args.agent == 'ddpg':
        DDPG_test(agent, test_env, args)
        exit()

### train models
print('begining training!')
if args.agent == 'ppo':
    args.ppo_agent = min(args.ppo_agent, len(train_env))
    agents = [PPO(copy.copy(args)) for i in range(args.ppo_agent)]
    trained_agents = PPO_train(args, train_env, agent, agents, test_env, log_file, train_log_file, tb_writter)
elif args.agent == 'ddpg':
    agents = [DDPG(copy.copy(args), GaussNoise(args.noise_beg)) for i in range(args.num_process)]
    trained_agents = DDPG_train(args, train_env, agent, agents, test_env, log_file, train_log_file, tb_writter)



    





        
            
        
    
    
        
                                          
                                          
        
        
