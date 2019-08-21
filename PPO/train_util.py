import torch
import numpy as np
from threading import Thread
import random, copy
from matplotlib import pyplot as plt

def copy_(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def PPO_train(args, train_envs, central_agent, threading_agents, test_envs, log_file, train_log_file, tb_writter):
    '''
    doc
    '''
    # num_x = train_envs[0].num_x
    p_errors = np.ones(len(train_envs)) / len(train_envs)
    for _ in range(args.train_epoch):
        if _ > 0 and _ % args.prolong_every_epoch == 0:
            for env in train_envs:
                env.prolong_evolve()
                env.get_weno_error(recompute = True)

        threads = []
        params = [[] for i in range(args.ppo_agent)]
        datasets = [None for i in range(args.ppo_agent)]
        threading_train_envs = []
        threading_train_envs = random.sample(train_envs, args.ppo_agent)
        for i in range(args.ppo_agent):
            datasets[i] = [[] for j in range(threading_train_envs[i].num_x)]
        errors = np.zeros(args.ppo_agent)
        for idx in range(len(params)):
            params[idx].append(threading_agents[idx])
            params[idx].append(threading_train_envs[idx])
            params[idx].append(datasets)
            params[idx].append(errors)
            params[idx].append(idx)

        for i in range(args.ppo_agent):
            process = Thread(target=_PPO_train, args=params[i])
            process.start()
            threads.append(process)
        for process in threads:
            process.join()

        print('Training Epoch: ', _, 'errors: ', errors)
        for i in range(len(threading_train_envs)):
            init = threading_train_envs[i].args.init
            print('init: ', init, 'init_t: {0}  error: {1}  weno_self_error {2}'.format(threading_train_envs[i].args.initial_t, 
                errors[i],  threading_train_envs[i].get_weno_error()))
            tb_writter.add_scalar('Train error' + init, errors[i], _)

        if _ % args.record_every == 0:
            train_log_file.write('Training Epoch:' + str(_) + ' ' + str(errors) + '\n')
        
        central_agent.train(datasets)
        
        for idx in range(args.ppo_agent):
            # if args.mode == 'constrained_flux':
            #     threading_agents[idx].w1 = central_agent.w1.copy()
            #     threading_agents[idx].w2 = central_agent.w2.copy()
            # else:
            copy_(threading_agents[idx].policy_net, central_agent.policy_net)
            copy_(threading_agents[idx].value_net, central_agent.value_net)

        if  _ % args.test_every == 0:
            train_errors = ppo_test(central_agent, train_envs)
            test_errors = ppo_test(central_agent, test_envs)
            p_errors = np.array(train_errors)
            p_errors = p_errors / np.sum(p_errors)
           
            print('test epoch: {0}'.format(_))
            log_file.write('\ntest epoch: {0}\n'.format(_))
            for idx in range(len(train_envs)):
                init = train_envs[idx].args.init
                print('Train Env {0} init_t {1} final error {2} weno-self-error {3}'.format(init, 
                    train_envs[idx].args.initial_t, train_errors[idx], train_envs[idx].get_weno_error()))
                log_file.write('Train Env {0} init_t {1} final error {2} weno-self-error {3}\n'.format(init, 
                    train_envs[idx].args.initial_t, train_errors[idx], train_envs[idx].get_weno_error()))
                tb_writter.add_scalar('Test in Train Env' + init, train_errors[idx], _)
        
            for idx in range(len(test_envs)):
                init = test_envs[idx].args.init
                # print('Test Env {0} final error {1} self-weno-error {2}'.format(init, test_errors[idx], self_weno_errors[init]))
                print('Test Env {0} final error {1}'.format(init, test_errors[idx]))
                log_file.write('Test Env {0} final error {1}  \n'.format(init, test_errors[idx]))
                tb_writter.add_scalar('Test in Test Env' + init, test_errors[idx], _)
        
            central_agent.save(args.save_path + str(_))
            log_file.write('\n')
            log_file.flush()
    
    return central_agent

def ppo_test(agent, test_envs, args = None):
    errors = []
    tv_break = [] ### tmp things
    for env in test_envs:
        env.set_args()
        pre_state = env.reset()
        t_range = env.num_t

        for t in range(1, t_range):
            action = agent.action(pre_state, True) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            next_state, reward, done, _ = env.step(action, t)
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


def _PPO_train(agent, env, datasets, errors, idx):
    # print('idx is: ', idx)
    t_range = env.num_t
    x_range = env.num_x
    # print('t_range: ', t_range)
    # print('dataset[idx] length: ', len(datasets[idx]))
    env.set_args()
    pre_state = env.reset()
    [datasets[idx][i].append(pre_state[i]) for i in range(x_range)]

    for t in range(1, t_range):
        action, action_prob = agent.action(pre_state) # action: (state_dim -2, 1) batch
        ### next_state, reward, done, all batches
        next_state, reward, done, _ = env.step(action, t)
        # dataset[idx] += [[action[i], action_prob[i], reward[i], next_state[i]] for i in range(x_range)]
        for i in range(x_range):
            datasets[idx][i].append(action[i])
            datasets[idx][i].append(action_prob[i])
            datasets[idx][i].append(reward[i])
            datasets[idx][i].append(next_state[i])
        pre_state = copy.copy(next_state)

    error = env.error()
    errors[idx] = error


def PPO_FCONV_train(args, train_envs, central_agent, threading_agents, test_envs, log_file, train_log_file):
    '''
    doc
    '''
    # num_x = train_envs[0].num_x
    for _ in range(args.train_epoch):
        # datasets = [[[] for i in range(num_x)] for i in range(args.ppo_agent)]

        threads = []
        params = [[] for i in range(args.ppo_agent)]
        threading_train_envs = random.sample(train_envs, args.ppo_agent) 
        datasets = [None for i in range(args.ppo_agent)]
        for i in range(args.ppo_agent):
            datasets[i] = [[] for j in range(threading_train_envs[i].num_x)]
        errors = np.zeros(args.ppo_agent)
        for idx in range(len(params)):
            params[idx].append(threading_agents[idx])
            params[idx].append(threading_train_envs[idx])
            params[idx].append(datasets)
            params[idx].append(errors)
            params[idx].append(idx)

        for i in range(args.ppo_agent):
            process = Thread(target=_PPO_FCONV_train, args=params[i])
            process.start()
            threads.append(process)
        for process in threads:
            process.join()

        print('Training Epoch: ', _, 'errors: ', errors)
        for i in range(len(threading_train_envs)):
            print('init: ', threading_train_envs[i].args.init, 'init_t: {0}   \
                error: {1}'.format(threading_train_envs[i].args.initial_t, errors[i]))
        if _ > 0 and _ % args.record_every == 0:
            train_log_file.write('Training Epoch:' + str(_) + ' ' + str(errors) + '\n')
        central_agent.train(datasets)
        for idx in range(args.ppo_agent):
            copy_(threading_agents[idx].policy_net, central_agent.policy_net)
            copy_(threading_agents[idx].value_net, central_agent.value_net)

        if _ % args.test_every == 0:
            train_errors = ppo_FCONV_test(central_agent, train_envs)
            test_errors = ppo_FCONV_test(central_agent, test_envs)
            print('test epoch: {0}'.format(_))
            log_file.write('\ntest epoch: {0}\n'.format(_))
            for idx in range(len(train_envs)):
                print('Train Env {0} init_t {1} final error {2}'.format(train_envs[idx].args.init, 
                    train_envs[idx].args.initial_t, train_errors[idx]))
                log_file.write('Train Env {0} init_t {1} final error {2}\n'.format(train_envs[idx].args.init, 
                    train_envs[idx].args.initial_t, train_errors[idx]))
            for idx in range(len(test_envs)):
                print('Test Env {0} final error {1}'.format(test_envs[idx].args.init, test_errors[idx]))
                log_file.write('Test Env {0} final error {1}\n'.format(test_envs[idx].args.init, test_errors[idx]))
            central_agent.save(args.save_path + str(_))
            log_file.write('\n')
            log_file.flush()
    
    return central_agent

def ppo_FCONV_test(agent, test_envs):
    errors = []
    for env in test_envs:
        env.set_args()
        pre_state = env.reset()
        t_range = env.num_t

        for t in range(1, t_range):
            action = agent.action(pre_state, True) # action: (state_dim -2, 1) batch
            ### next_state, reward, done, all batches
            next_state, reward, done, _ = env.step(action, t)
            pre_state = copy.copy(next_state)

        error = env.error()
        errors.append(error)

    return errors


def _PPO_FCONV_train(agent, env, datasets, errors, idx):
    # print('idx is: ', idx)
    t_range = env.num_t
    x_range = env.num_x
    # print('x_range: ', x_range)
    # print('dataset[idx] length: ', len(datasets[idx]))
    env.set_args()
    pre_state = env.reset()
    [datasets[idx][i].append(pre_state) for i in range(x_range)]

    for t in range(1, t_range):
        action, action_prob = agent.action(pre_state) # action: (state_dim -2, 1) batch
        ### next_state, reward, done, all batches
        next_state, reward, done, _ = env.step(action, t)
        # dataset[idx] += [[action[i], action_prob[i], reward[i], next_state[i]] for i in range(x_range)]
        for i in range(x_range):
            datasets[idx][i].append(action[i])
            datasets[idx][i].append(action_prob[i])
            datasets[idx][i].append(reward)
            datasets[idx][i].append(next_state)
        pre_state = copy.copy(next_state)

    error = env.error()
    errors[idx] = error
    
#########################################################################
################## below is old not used functions ######################
#########################################################################
'''
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
    # The main training procedure
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
'''

