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
import os, sys
import time
import math
import random

from get_args import get_args
from DDPG.train_util import DDPG_train, DDPG_test
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise
from viskit import logger

args = get_args()

### Important: fix numpy and torch seed! ####
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

### Initialize RL agents 
ddpg = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg, final_sig=args.noise_end))
agent = ddpg

### Initialize training and testing encironments
from init.init_util_eval import init_env
_, _, _, test_env = init_env(args, agent)
print("environment prepared ready!")

### load pretrained models if given
assert args.load_path is not None
agent.load(args.load_path)
print('load models!')

for env in test_env:
    env.get_weno_corase()

### evaluate models and exit
DDPG_test(agent, test_env, args)


    





        
            
        
    
    
        
                                          
                                          
        
        
