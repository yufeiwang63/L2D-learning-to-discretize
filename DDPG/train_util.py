'''
File Description:
This file is the driver for training the DDPG agent.
'''


import torch
import numpy as np
from threading import Thread
import random, copy
from matplotlib import pyplot as plt

def copy_(target, source):
    '''
    Copy all the parameters in the source model to the target model.
    Arg target (pytorch nn.Module):
        the target model, whose parameters will be copied from the source model.
    Arg source (pytorch nn.Module):
        the source model, whose parameters will be copied to the target model.
    Return: None.
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def DDPG_train(args, train_envs, central_agent, threading_agents, test_envs, log_file, train_log_file, tb_writter):
    '''
    Given the training args, training/testing environments, log files/tb_writters, and the DDPG agents, this function 
    uses the python threading module to interact the agent with the training environment in multiple threads, 
    collect interacting samples, perform training of the agent, and record necessary logistics in log files/tb_writters.

    ### Arguments:
    args (python namespace):
        A namespace variable that stores all necessary parameters for the whole training procedure.
    train_envs (list of (class) Burgers objects):
        The list of the training environments, each is a Burgers Env object with different init conditions/dx/dt, etc.
    central_agent ((class) DDPG object):
        The agent to be trained. Central means it will be used to synchronize different agents in different threads, where
        at each iteration its parameters will be copied to the threading agents.
    threading_agents (list of (class) Burgers object):
        A list of DDPG agents, each will be used in a thread to interact with the training envs to collect samples 
        for efficient exploration/fast training. At each iteration, the collected samples will be used to train the central_agent,
        and then the updated parameters will be copied back to these threading agents.
    test_envs (list of (class) Burgers object):
        A list of Test Burgers Envs with different initial conditions.
    log_file (python file):
        A file that records general logistics during the training procedure.
    train_log_file (python file):
        A file that records detailed training logistics.
    tb_wriiter (tensorboardX writter):
        A tensorboardX writter that records and plots general logistics during the training procedure.


    ### Returns: 
    central_agent:
        the well-trained central_agent.
    '''

    num_env = args.num_process
    env_batch_size = train_envs[0].num_x
    horizen = train_envs[0].num_t - 1
    state_size = args.state_dim
    action_size = args.action_dim

    if args.mem_type == 'multistep_return':
        '''
        In this setting, when training the Q-network, the Bellman Equation is expanded for multiple steps, and
        the tuples are randomly sampled from the replay memory.
        '''
        prev_state_dataset = np.zeros((num_env, env_batch_size, horizen, state_size))
        action_dataset = np.zeros((num_env, env_batch_size, horizen, action_size))
        reward_dataset = np.zeros((num_env, env_batch_size, horizen))
        next_state_dataset = np.zeros((num_env, env_batch_size, horizen, state_size))
        done_dataset = np.zeros((num_env, env_batch_size, horizen))
        datasets = [prev_state_dataset, action_dataset, reward_dataset, next_state_dataset, done_dataset,  'multistep_return']

    ### training begins
    for _ in range(args.train_epoch): 

        threads = []
        params = [[] for i in range(args.num_process)]
        
        if args.mem_type == 'batch_mem':
            '''
            In this setting, when training the Q-network, the Bellman Equation is always only expanded for one step.
            The tuples are sampled in a whole row randomly. This might help the agent to train on more samples that 
            are near the discontinuity, since a whole row for sure contains samples with discontinuity.
            '''
            datasets = [None for i in range(args.num_process)]
            datasets.append('batch_mem')

        threading_train_envs = []
        threading_train_envs = random.sample(train_envs, args.num_process)
        
        if args.mem_type == 'batch_mem':
            for i in range(args.num_process):
                datasets[i] = [[] for j in range(threading_train_envs[i].num_x)]
        
        errors = np.zeros(args.num_process)
        for idx in range(len(params)):
            params[idx].append(threading_agents[idx])
            params[idx].append(threading_train_envs[idx])
            params[idx].append(datasets)
            params[idx].append(errors)
            params[idx].append(idx)

        for i in range(args.num_process):
            process = Thread(target=_DDPG_train, args=params[i])
            process.start()
            threads.append(process)
        for process in threads:
            process.join()

        print('Training Epoch: ', _, 'errors: ', errors) ### print errors of all training environments
        for i in range(len(threading_train_envs)): ### print detailed info of each training environment
            init = threading_train_envs[i].args.init
            print('init: ', init, 'init_t: {0}  error: {1}  weno_self_error {2}'.format(threading_train_envs[i].args.initial_t, 
                errors[i],  threading_train_envs[i].get_weno_error()))
            tb_writter.add_scalar('Train error' + init, errors[i], _)

        if _ % args.record_every == 0:
            train_log_file.write('Training Epoch:' + str(_) + ' ' + str(errors) + '\n')
        
        ### remove the last element, which is the mem type indicator: 'multistep_return' or 'batch_mem'.
        central_agent.perceive(datasets[:-1]) 
        central_agent.train()
        
        for idx in range(args.num_process):
            copy_(threading_agents[idx].actor_policy_net, central_agent.actor_policy_net)
            copy_(threading_agents[idx].critic_policy_net, central_agent.critic_policy_net)

        ### decrease exploration ratio of the threading agents
        if _ > 0 and _ % args.noise_dec_every == 0:
            for agent in threading_agents:
                agent.action_noise.decrease(args.noise_dec)
                if agent.upwindnoise is not None: ### version4 remove marker, no upwindnoise
                    agent.upwindnoise.decrease(args.upwindnoisedec) 

        ### test central_agent in test_envs, both euler error and rk4 error.
        if  _ % args.test_every == 0:
            euler_train_errors = DDPG_test(central_agent, train_envs, Tscheme='euler')
            euler_test_errors = DDPG_test(central_agent, test_envs, Tscheme='euler')
            
            rk4_train_errors = DDPG_test(central_agent, train_envs, Tscheme='rk4')
            rk4_test_errors = DDPG_test(central_agent, test_envs, Tscheme='rk4')
    
            print('test epoch: {0}'.format(_))
            log_file.write('\ntest epoch: {0}\n'.format(_))
            for idx in range(len(train_envs)):
                init = train_envs[idx].args.init
                print('Train Env {0} init_t {1} euler final error {2} weno-self-error {3}'.format(init, 
                    train_envs[idx].args.initial_t, euler_train_errors[idx], train_envs[idx].get_weno_error()))
                log_file.write('Train Env {0} init_t {1} euler final error {2} weno-self-error {3}\n'.format(init, 
                    train_envs[idx].args.initial_t, euler_train_errors[idx], train_envs[idx].get_weno_error()))
                tb_writter.add_scalar('Test in Train Env Euler' + init, euler_train_errors[idx], _)

                print('Train Env {0} init_t {1} rk4 final error {2} weno-self-error {3}'.format(init, 
                    train_envs[idx].args.initial_t, rk4_train_errors[idx], train_envs[idx].get_weno_error()))
                log_file.write('Train Env {0} init_t {1} rk4 final error {2} weno-self-error {3}\n'.format(init, 
                    train_envs[idx].args.initial_t, rk4_train_errors[idx], train_envs[idx].get_weno_error()))
                tb_writter.add_scalar('Test in Train Env RK4' + init, rk4_train_errors[idx], _)
        
            for idx in range(len(test_envs)):
                init = test_envs[idx].args.init
                print('Test Env {0} euler final error {1} weno-self-error {2}'.format(init, euler_test_errors[idx], test_envs[idx].get_weno_error()))
                log_file.write('Test Env {0} euler final error {1} weno-self-error {2} \n'.format(init, euler_test_errors[idx], test_envs[idx].get_weno_error()))
                tb_writter.add_scalar('Test in Test Env Euler' + init, euler_test_errors[idx], _)

                print('Test Env {0} rk4 final error {1} weno-self-error {2}'.format(init, rk4_test_errors[idx], test_envs[idx].get_weno_error()))
                log_file.write('Test Env {0} rk4 final error {1} weno-self-error {2} \n'.format(init, rk4_test_errors[idx], test_envs[idx].get_weno_error()))
                tb_writter.add_scalar('Test in Test Env RK4' + init, rk4_test_errors[idx], _)
        
            central_agent.save(args.save_path + str(_))
            log_file.write('\n')
            log_file.flush()
    
    return central_agent

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
    tv_break = [] ### tmp things
    for env in test_envs:
        env.set_args()
        pre_state = env.reset()
        t_range = env.num_t

        for t in range(1, t_range):
            action = agent.action(pre_state, True) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            ### when doing test, use rk4 as default, and mode == test ensures all actions are generated using the agent
            next_state, reward, done, _ = env.step(action, t, Tscheme = Tscheme, mode = 'test')
            pre_state = copy.copy(next_state)
            
        error = env.error()
        errors.append(error) 
        if args is not None and args.compute_tvd:
            tv_break.append(env.tvd_break) 

    if args is not None and args.compute_tvd:
        print('plotting and saving tvd data!')
        plt.plot(range(len(tv_break)), tv_break) 
        plt.xlabel('test envs')
        plt.ylabel('sum of positive tv difference')
        plt.show()
        save_path = args.load_path[:-4] 
        plt.savefig(save_path + 'tvd.png')
        plt.close()
        np.save(save_path + 'tvd.npy', arr = np.array(tv_break))

    return errors


def _DDPG_train(agent, env, datasets, errors, idx):
    t_range = env.num_t
    x_range = env.num_x
    env.set_args()
    pre_state = env.reset()

    mem_type = datasets[-1]
    if mem_type == 'multistep_return':
        prev_state_dataset, action_dataset, reward_dataset, next_state_dataset, done_dataset, _ = datasets
        for t in range(1, t_range):
            action = agent.action(pre_state) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            ### need to specify the mode to be train, as might be needed by rk4.
            ### in rk4, if the mode is set to be train, then the first action is generated by the agent, and the rest is by weno.
            next_state, reward, done, _ = env.step(action, t, mode = 'train')
            for i in range(x_range):
                prev_state_dataset[idx][i][t - 1] = pre_state[i]
                action_dataset[idx][i][t - 1] = action[i]
                reward_dataset[idx][i][t - 1] = reward[i]
                next_state_dataset[idx][i][t - 1] = next_state[i]
                done_dataset[idx][i][t - 1] = int(done[i])
            pre_state = copy.copy(next_state)

    elif mem_type == 'batch_mem':
        for t in range(1, t_range):
            action = agent.action(pre_state) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            next_state, reward, done, _ = env.step(action, t, mode = 'train')
            for i in range(x_range):
                datasets[idx][i].append([pre_state[i], action[i], reward[i], next_state[i], done[i]])
            pre_state = copy.copy(next_state)


    error = env.error()
    errors[idx] = error
