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


def run_task(vv, log_dir, exp_name):
    import torch
    import numpy as np
    import copy
    import os, sys
    import time
    import math
    import random
    import json

    from get_args import get_args
    from DDPG.train_util import DDPG_train, DDPG_test
    from DDPG.DDPG_new import DDPG
    from DDPG.util import GaussNoise
    from chester import logger
    from BurgersEnv.Burgers import Burgers
    import utils.ptu as ptu

    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)

    ### dump vv
    logger.configure(dir=log_dir, exp_name=exp_name)
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    ### load vv
    ddpg_load_epoch = None
    if vv['load_path'] is not None:
        solution_data_path = vv['solution_data_path']
        dx = vv['dx']
        test_interval = vv['test_interval']
        load_path = os.path.join('data/local', vv['load_path'])
        ddpg_load_epoch = str(vv['load_epoch'])
        with open(os.path.join(load_path, 'variant.json'), 'r') as f:
            vv = json.load(f)
        vv['noise_beg'] = 0.1
        vv['solution_data_path'] = solution_data_path
        vv['test_interval'] = test_interval
        if vv.get('dx') is None:
            vv['dx'] = dx

    ### Important: fix numpy and torch seed!
    seed = vv['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    ### Initialize RL agents 
    ddpg = DDPG(vv, GaussNoise(initial_sig=vv['noise_beg'], final_sig=vv['noise_end']))
    agent = ddpg
    if ddpg_load_epoch is not None:
        print("load ddpg models from {}".format(os.path.join(load_path, ddpg_load_epoch)))
        agent.load(os.path.join(load_path, ddpg_load_epoch))

    ### Initialize training and testing encironments
    env = Burgers(vv, agent=agent)

    ### train models
    print('begining training!')
    DDPG_train(vv, env, agent)



    





        
            
        
    
    
        
                                          
                                          
        
        
