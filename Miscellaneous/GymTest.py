import gym
import torch
import numpy as np
import argparse
import sys
import os
import datetime

from helper_functions import *
from DQN_torch import DQN 
from NAF_torch import NAF
from DDPG_torch import DDPG
from AC_torch import AC
from CAC_torch import CAC
# from PPO_torch import PPO
import matplotlib.pyplot as plt
import time



# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('LunarLander-v2')
# env = gym.make('Pendulum-v0')

argparser = argparse.ArgumentParser(sys.argv[0])
## parameters
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--c_lr', type=float, default=1e-4)
argparser.add_argument('--a_lr', type=float, default=1e-4)
argparser.add_argument('--debug', type=bool, default=False)
argparser.add_argument('--env_name', type=str, default='CartPole-v0')
argparser.add_argument('--replay_size', type=int, default=5000)
argparser.add_argument('--batch_size', type=float, default=64)
argparser.add_argument('--gamma', type=float, default=0.99)
argparser.add_argument('--state_dim', type=int, default=2)
argparser.add_argument('--action_dim', type=int, default=2)
argparser.add_argument('--tau', type=float, default=0.01)
argparser.add_argument('--noise_type', type = str, default = 'gauss')
argparser.add_argument('--agent', type = str, default = 'dqn')
argparser.add_argument('--record_every', type = int, default = 20)
argparser.add_argument('--test_every', type = int, default = 100)
argparser.add_argument('--explore_initial', type = float, default = 1.)
argparser.add_argument('--explore_final', type = float, default = 0.05)
argparser.add_argument('--explore_decrease', type = float, default = 0.1)
argparser.add_argument('--explore_decrease_every', type = int, default = 50)
argparser.add_argument('--ac_train_epoch', type = int, default = 2)
argparser.add_argument('--update_every', type = int, default = 50)

argparser.add_argument('--save_path', type = str, default = None)
##
args = argparser.parse_args()

env = gym.make(args.env_name)

if not args.debug:
    if args.save_path is None:
        import time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args.save_path = './' + str(current_time) + '/'

    if not os.path.exists(args.save_path):
       os.makedirs(args.save_path) 


Replay_mem_size = args.replay_size
Train_batch_size = args.batch_size
Actor_Learning_rate = args.lr
Critic_Learning_rate = args.lr
Gamma = args.gamma
tau = args.tau

State_dim = env.observation_space.shape[0]
args.state_dim = State_dim
print('state dimension is {0}'.format(State_dim))

# Action_dim = env.action_space.shape[0]
Action_dim = env.action_space.n
args.action_dim = Action_dim
print('action dimension is {0}'.format(Action_dim))

Action_low = -1
Action_high = 1

if args.env_name != 'CartPole-v0':
    print('----action range---')
    print(env.action_space.high)
    print(env.action_space.low)
    Action_low = env.action_space.low[0].astype(float)
    Action_high = env.action_space.high[0].astype(float)

ounoise = OUNoise(Action_dim, 8, 3, 0.9995)
gsnoise = GaussNoise(2, 0.5, 0.9995)
Noise = gsnoise if args.noise_type == 'gauss' else ounoise

# for plotting learning curves
train_plot_rewards = []
train_plot_epoch_idxes = []
test_plot_rewards = []
test_plot_epoch_idxes = []

#################################################################################
####################### Initialization Over #####################################
#################################################################################

def play(agent, num_epoch, Epoch_step, show = False):
   
    acc_reward = 0
    for _ in range(num_epoch):
        pre_state = [env.reset()]
        for step in range(Epoch_step):
            if show:
                env.render()
            
            # action = agent.action(state_featurize.transfer(pre_state), False)
            action = agent.action(pre_state, False)[0]
            next_state, reward, done, _ = env.step(action)
            acc_reward += reward

            if done or step == Epoch_step - 1:
                break
            pre_state = [next_state]
    return acc_reward / num_epoch


def train(agent, Train_epoch, Epoch_step):        
    for epoch in range(Train_epoch):
        pre_state = env.reset()
        # if args.debug:
        #     print('Gym state is: ', pre_state)
        pre_state = [pre_state]
        acc_reward = 0
        transitions = []

        for step in range(Epoch_step):

            action = agent.action(pre_state)[0]

            # if args.debug:
            #     print('action is: ', action)

            # if action[0] != action[0]:
            #     raise('nan error!')

            next_state, reward, done, _ = env.step(action)
            acc_reward += reward

            transitions.append([pre_state, action, reward, next_state, done])
            
            # agent.train(state_featurize.transfer(pre_state), action, reward, state_featurize.transfer(next_state), done)
            if args.agent == 'dqn':
                agent.train(pre_state, action, reward, next_state, done)


            if done or step == Epoch_step - 1:
                # print(args.record_every)
                #print('episoid: ', epoch + 1, 'step: ', step + 1, ' reward is', acc_reward, )
                if args.agent == 'ac':
                    agent.train(transitions)
                
                if epoch % args.record_every == 0:
                    print('episoid: ', epoch + 1, 'step: ', step + 1, ' reward: ', acc_reward)
                    train_plot_epoch_idxes.append(epoch + 1)
                    train_plot_rewards.append(acc_reward)
                break
            
            pre_state = [next_state]
        
        if epoch % args.test_every == 0 and epoch > 0:
            avg_reward = play(agent, 1, 300, not True)
            print('--------------episode ', epoch,  'avg_reward: ', avg_reward, '---------------')
            test_plot_epoch_idxes.append(epoch)
            test_plot_rewards.append(avg_reward)
            if np.mean(test_plot_rewards[-5:]) > 190:
                print('----- using ', epoch, '  epochs')
                #agent.save_model()
                break
         
    return agent

naf = NAF(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
          Gamma, Critic_Learning_rate, Action_low, Action_high, tau, Noise, False, False)  
dqn = DQN(args)  
ddpg = DDPG(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, Action_low, Action_high, tau, Noise, False) 

ac = AC(args)

cac = CAC(State_dim, Action_dim, Replay_mem_size, Train_batch_size,
             Gamma, Actor_Learning_rate, Critic_Learning_rate, tau, Action_low, Action_high, 3, False)

# cppo = PPO(State_dim, Action_dim, Action_low, Action_high, Train_batch_size, 
#                 Gamma, Actor_Learning_rate, Critic_Learning_rate,  update_epoach=50, trajectory_number=100)

# agent = train(naf, 10000,300)
# agentnaf = train(naf, 3000, 300, r'./record/naf_lunar.txt')
# agentppo = train(cppo, 3000, 300, r'./record/ppo_lunar.txt')
# agentnaf_addloss = train(naf_addloss, 1500, 300, r'./record/naf_addloss_lunar.txt')
# agentddpg = train(ddpg, 3000, 300, r'./record/ddpg_lunar_PER.txt')
# agentnaf_addloss = train(naf_addloss, 1500, 300, r'D:\study\rl by david silver\Trainrecord\NAF_addloss.txt')
# agentnaf_ddpg = train(ddpg, 1500, 300, r'D:\study\rl by david silver\Trainrecord\ddpg_lunar.txt')
# agentac = train(ac, 3000, 300, r'./record/ac_lunar_land_continues.txt')
# agentcac = train(cac, 3000, 300, r'./record/cac_lunar_land_continues-PER.txt')
# agentdqn = train(dqn, 3000, 300, r'./record/dqn_lunar_dueling_PER_1e-3_0.3.txt')


if args.agent == 'ddpg':
    agent = train(ddpg, 10000, 200)
elif args.agent == 'naf':
    agent = train(naf, 10000, 200)
elif args.agent == 'cac':
    agent = train(cac, 10000, 200)
elif args.agent == 'ac':
    agent = train(ac, 10000, 200)
elif args.agent == 'dqn':
    agent = train(dqn, 2000, 200)

if not args.debug:
    np.save(args.save_path + 'train_plot_rewards', np.array(train_plot_rewards))
    np.save(args.save_path + 'train_plot_idxes', np.array(train_plot_epoch_idxes))
    np.save(args.save_path + 'test_plot_rewards', np.array(test_plot_rewards))
    np.save(args.save_path + 'test_plot_idxes', np.array(test_plot_epoch_idxes))
    param_log = open(args.save_path + 'Train_Parameters.txt', 'w')
    params = 'a_lr: {0} \t c_lr: {1} \t replay_size: {2} \n tau: {3} \t gamma: {4} \t'.format(args.a_lr, 
        args.c_lr, args.replay_size, args.tau, args.gamma)
    param_log.write(params)
    param_log.close()



#print('after train')

#print(play(agentnaf,300, False))
#print(play(agentnaf_addloss,300, False))

