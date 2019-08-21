# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:43:06 2018

@author: Wangyf
"""
import torch
import numpy as np
import copy
import os
import time
import math

from helper_functions import *
from get_args import get_args
from train_util import PPO_train, ppo_test, PPO_FCONV_train

### input and parse all arguments
args, tb_writter = get_args()
if args.mode == 'constrained_flux':
    from PPO_constrained_flux import PPO
else:
    from PPO_new import PPO

### Important: fix numpy and torch seed! ####
np.random.seed(args.np_rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.torch_rng_seed)
random.seed(args.py_rng_seed)

### state: /*f(u_{i-width}), ... , f(u_i), ... , f(u_{i+ width}),*/ u_{i-width}, ..., u_i, u_{i + width}, a(u_i) 
print('State dimension: {0}'.format(args.state_dim))
### action dim: number of filters
print('Action dimension: {0}'.format(args.action_dim))

### RL agents 
ppo = PPO(copy.copy(args), tb_writter)
agent = ppo

### init training and testing encironments
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


### print the stencil the policy network chooses
def show_filter(action):
    filter_dic = {0: 'zxcf', 1: '1ryf', 2: '1lyf', 3: '2ryf', 4: '2lyf'}
    if args.mode == 'eno':
        filters = [filter_dic[a] for a in action]
    elif args.mode == 'weno':
        filters = [[(filter_dic[j], a[j]) for j in range(len(a))] for a in action[:10]]
    return filters
    

### test the performance of the RL policy network
def test(agent, test_plot_errors, log_file, e = 0):
    """
    test the trained agent. the metric is the MSE error at the terminal time point.
    """
    sum_error = 0
    test_plot_epoch_idxes.append(e)
    log_file.write('--------------- TestEpoch {0} ----------------\n'.format(e))
    for test_count in range(args.num_test): # test each environment
        # test_env[test_count].set_args(argss[test_count]) # set the args strictly following those specified by users
        pre_state = test_env[test_count].reset()
        t_range = test_env[test_count].num_t

        for t in range(1 , t_range):

            action = agent.action(pre_state, False)

            next_state, _, _, _ = test_env[test_count].step(action, t)
            pre_state = next_state

        error = test_env[test_count].error()
        test_plot_errors[test_count].append(error)
        # filter = show_filter(action)

        log_file.write('---TestIdx {0} TestInit {1}  RLError {2}'.format(test_count, 
                test_env[test_count].args.init, test_plot_errors[test_count][-5:]))
        log_file.write('\n\n')
        log_file.flush()
        train_log_file.flush()

        print('---TestIdx {0} TestInit {1}  RLError {2}'.format(test_count, 
                test_env[test_count].args.init, test_plot_errors[test_count][-5:]))
        sum_error += error
    
    if args.error_explore:
        agent.set_explore(sum_error)

    np.save(args.save_path + 'test_plot_errors', np.array(test_plot_errors))
    np.save(args.save_path + 'test_plot_idxes', np.array(test_plot_epoch_idxes))
  
### main training procedure
def train(agent, epoch, log_file, test_plot_errors):
    '''
    The main training procedure
    '''
    for i in range(epoch):
        train_idx = i//args.env_change_every % num_train
        error, last_filter = _train(agent, train_idx)
        if i % args.record_every == 0:
            train_log_file.write('Epoch {0} Train idx {1} Train Init {2} Initial T {3} \
                Final Error {4}\n'.format(i, train_idx, train_env[train_idx].args.init, 
                    train_env[train_idx].args.initial_t, error ))
            print('Epoch {0} Train idx {1} Train Init {2} Initial T {3} \
                Final Error {4}'.format(i, train_idx, train_env[train_idx].args.init, 
                    train_env[train_idx].args.initial_t, error ))
            train_plot_errors[train_idx].append(error)
            train_plot_epoch_idxes[train_idx].append(i)

        if i % args.test_every == 0:
            test(agent, test_plot_errors, log_file, e = i)

        # # when using weno, it's ddpg or naf, need to decrease exploration rate
        # if args.mode == 'weno' and i % args.explore_decrease_every == 0 and i > 0:
        #     agent.explore.decrease(args.explore_decrease)

        if i % args.save_every == 0:
            agent.save(args.save_path + str(i))
            
    return agent

### the subroutine training procedure
def _train(agent, train_count):
    # pre_state: batch (state_dim - 2, 6)
    # space_range = int((argss[train_count].x_high - argss[train_count].x_low) / argss[train_count].dx + 1)
    train_env[train_count].set_args()
    pre_state = train_env[train_count].reset()
    t_range = train_env[train_count].num_t

    for t in range(1, t_range):
        # print(t)
        action = agent.action(pre_state) # action: (state_dim -2, 1) batch
        # next_state, reward, done, all batches
        next_state, reward, done, _ = train_env[train_count].step(action, t)
        
        # reward_idx = train_env[train_count].reward_index
        # p = np.random.rand()
        # if p < args.train_prob or t in reward_idx:
        # if p < args.train_prob:
        if args.formulation == 'MLP':
            if args.agent == 'dqn':
                pre_state_ = random.sample(pre_state, args.dqn_train_num)
                action_ = random.sample(list(action), args.dqn_train_num)
                reward_ = random.sample(reward, args.dqn_train_num)
                next_state_ = random.sample(next_state, args.dqn_train_num)
                done_ = random.sample(done, args.dqn_train_num)
                [agent.train(pre_state_[i], action_[i], reward_[i], next_state_[i], done_[i]) for i in range(len(pre_state_))]
            else:
                [agent.train(pre_state[i], action[i], reward[i], next_state[i], done[i]) for i in range(len(pre_state))]
        elif args.formulation == 'FCONV':
            agent.train(pre_state, action, reward, next_state, done)
        pre_state = next_state

    # print(action)
    error = train_env[train_count].error()
    return error, None

#### training ####
if args.load_path is not None:
    print('load models!')
    agent.load(args.load_path)
if args.test or args.animation or args.compute_tvd:
    if args.agent == 'ppo':
        ppo_test(agent, test_env, args)
    else:
        test(agent, test_plot_errors, log_file)
    exit()

print('begining training!')
if args.agent == 'ppo':
    args.ppo_agent = min(args.ppo_agent, len(train_env))
    agents = [PPO(copy.copy(args)) for i in range(args.ppo_agent)]
    if args.formulation == 'MLP':
        trained_agents = PPO_train(args, train_env, agent, agents, test_env, log_file, train_log_file, tb_writter)
    elif args.formulation == 'FCONV':
        trained_agents = PPO_FCONV_train(args, train_env, agent, agents, test_env, log_file, train_log_file)
else:
    trained_agent = train(agent, args.train_epoch, log_file, test_plot_errors)

### save data for ploting figures
np.save(args.save_path + 'train_plot_errors', np.array(train_plot_errors))
np.save(args.save_path + 'train_plot_idxes', np.array(train_plot_epoch_idxes))
np.save(args.save_path + 'test_plot_errors', np.array(test_plot_errors))
np.save(args.save_path + 'test_plot_idxes', np.array(test_plot_epoch_idxes))


    





        
            
        
    
    
        
                                          
                                          
        
        
