'''
File Description:
This file is the driver for training the DDPG agent.
'''


import torch
import numpy as np
from threading import Thread
import random, copy
from matplotlib import pyplot as plt
from chester import logger
import os
import os.path as osp

def DDPG_train(vv, env, agent):
    '''
    Given the training args, training/testing environments, log files, and the DDPG agents, this function 
    uses the python threading module to interact the agent with the training environment in multiple threads, 
    collect interacting samples, perform training of the agent, and record necessary logistics in log files.
    '''

    a_lr = vv['actor_lr']
    c_lr = vv['critic_lr']
    ### training begins
    for train_iter in range(vv['train_epoch']): 
        dx = np.random.choice(vv['dx'])
        if dx <= 0.02:
            weno_freq = vv['weno_freq']
        else:
            weno_freq = 0

        pre_state = env.reset(dx=dx)
        env_batch_size = env.num_x
        horizon = env.horizon

        # decay learning rate
        if train_iter > 0 and vv['lr_decay_interval'] > 0 and train_iter % vv['lr_decay_interval'] == 0:
            a_lr = max(vv['final_actor_lr'], a_lr / 2)
            c_lr = max(vv['final_critic_lr'], c_lr / 2)
            c_optimizers = [agent.critic_optimizer, agent.critic_optimizer2]
            for optim in c_optimizers:
                for param_group in optim.param_groups:
                    param_group['lr'] = c_lr
            
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = a_lr

        ret = 0
        for t in range(1, horizon):
            p = np.random.rand()
            if p < weno_freq:
                action = agent.action(pre_state, mode='weno')
            else:
                action = agent.action(pre_state) 
            
            ### next_state, reward, done, all batches
            next_state, reward, done, _ = env.step(action)
            ret += np.mean(reward)
            
            # TODO: change this to support store a batch
            for i in range(env_batch_size):
                agent.store(pre_state[i], action[i], reward[i], next_state[i], done[i])

            agent.train()
            pre_state = next_state

        error, relative_error = env.error('euler')
        solution_idx = env.solution_idx
        logger.record_tabular('Train/{}-error'.format(solution_idx), error)
        logger.record_tabular('Train/{}-ret'.format(solution_idx), ret)
        logger.record_tabular('Train/{}-relative-error'.format(solution_idx), relative_error)
        
        ### decrease exploration ratio of the threading agents
        if train_iter > 0 and train_iter % vv['noise_dec_every'] == 0:
            agent.action_noise.decrease(vv['noise_dec'])

        # TODO: implement possible learning rate decay
        # if train_iter > 0 and train_iter % vv['decay_learning_rate'] == 0:
        #     # update_linear_schedule(self.critic_optimizer, self.global_step, self.vv[train_epoch, self.args.c_lr, self.args.final_c_lr)
        #     # update_linear_schedule(self.critic_optimizer2, self.global_step, self.args.train_epoch, self.args.c_lr, self.args.final_c_lr)
        #     # update_linear_schedule(self.actor_optimizer, self.global_step, self.args.train_epoch, self.args.a_lr, self.args.final_a_lr)
        #     pass

        ### test central_agent in test_envs, both euler error and rk4 error.
        if  train_iter % vv['test_interval'] == 0:
            env.train_flag = False
            agent.train_mode(False)

            for dx in vv['dx']:
                print("test begin")
                errors, relative_errors = [], []
                for solution_idx in range(len(env.solution_path_list) // 2, len(env.solution_path_list)):
                    pre_state = env.reset(solution_idx=solution_idx, num_t=200, dx=dx)
                    horizon = env.num_t
                    for t in range(1, horizon):
                        action = agent.action(pre_state, deterministic=True) # action: (state_dim -2, 1) batch
                        next_state, reward, done, _ = env.step(action, Tscheme='rk4')
                        pre_state = next_state
                    error, relative_error = env.error('rk4')
                    errors.append(error)
                    relative_errors.append(relative_error)

                names = ['error', 'relative_error']
                all_errors = [errors, relative_errors]
                for i in range(len(names)):
                    name = names[i]
                    errors = all_errors[i]
                    logger.record_tabular(f'Test/{dx}_{name}_mean', np.mean(errors))
                    logger.record_tabular(f'Test/{dx}_{name}_max', np.max(errors))
                    logger.record_tabular(f'Test/{dx}_{name}_min', np.min(errors))
                    logger.record_tabular(f'Test/{dx}_{name}_median', np.median(errors))
                    logger.record_tabular(f'Test/{dx}_{name}_std', np.std(errors))
                print("test end")
            
            logger.dump_tabular()
            env.train_flag = True
            agent.train_mode(True)

        if  train_iter % vv['save_interval'] == 0 and train_iter > 0:
            agent.save(osp.join(logger.get_dir(), str(train_iter)))

    return agent

def DDPG_test(agent, test_envs, args = None, Tscheme = None):
    '''
    This function interacts the agent with the environment without exploration noise to test its
    deterministic performance.

    ### Arguments:
    agent (class DDPG object):
        the trained agent to be tested.
    test_envs (list of class Burgers objects):
        a list of test environments with different initial contions/dx/dt, etc.
    args (optional, python namespace):
        A namespace variable that stores all necessary parameters for the whole training procedure.
    Tscheme (optional, string, 'rk4' or 'euler'):
        When given, specifies the temporal discretization scheme, either Euler scheme or the Runge-kuta 4-th order scheme.
    
    ### Return
    errors (A list of floats): 
        A list of the relative errors at the terminal time step in each test environment.
    '''
    errors = []
    rets = []
    
    for env in test_envs:
        env.set_args()
        pre_state = env.reset()
        t_range = env.num_t

        ret = 0
        for t in range(1, t_range):
            action = agent.action(pre_state, True) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            ### when doing test, use rk4 as default, and mode == test ensures all actions are generated using the agent
            next_state, reward, done, _ = env.step(action, t, Tscheme = Tscheme, mode = 'test')
            ret += np.mean(reward)
            pre_state = copy.copy(next_state)
            
        rets.append(ret)
        error, _ = env.error()
        errors.append(error) 

    return errors, rets
