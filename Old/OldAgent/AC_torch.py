#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:58:01 2018

@author: yufei
"""


import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from collections import deque
from torch_networks import AC_a_fc_network, AC_v_fc_network, CAC_a_fc_network, AC_fc_network
from helper_functions import SlidingMemory, PERMemory
import warnings

# warnings.simplefilter("ignore", RuntimeWarning)

        

class AC():  
    """
    DOCstring for actor-critic
    """  
    def __init__(self, args, if_PER = False, share_network = True):
        self.args = args
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma = args.gamma
        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr
        self.global_step = 0
        self.tau = args.tau
        self.if_PER = if_PER
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.replay_mem = PERMemory(self.mem_size) if if_PER else SlidingMemory(self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        self.cret = nn.MSELoss()
        
        self.share_network = share_network
        if not share_network:
            self.actor_policy_net = AC_a_fc_network(self.state_dim, self.action_dim).to(self.device)
            self.critic_policy_net = AC_v_fc_network(self.state_dim).to(self.device)
            self.critic_target_net = AC_v_fc_network(self.state_dim).to(self.device)
            self.critic_policy_net.apply(self._weight_init)
            self.actor_policy_net.apply(self._weight_init)
            self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)

        else:
            self.vloss_coef = 1
            self.net = AC_fc_network(self.state_dim, self.action_dim).to(self.device)
            self.net.apply(self._weight_init)
            self.optimizer = optim.Adam(self.net.parameters(), self.args.lr)

        self.save_path = './record/'
        self.train_epoch = args.ac_train_epoch

    def _weight_init(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)
    
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    ### new training process, using TD(args.forward_look_step)
    def train(self, transitions):
        states = [x[0] for x in transitions]
        actions = [x[1] for x in transitions]
        rewards = [x[2] for x in transitions]

        states = torch.tensor(states, dtype = torch.float, device = self.device).squeeze()
        actions = torch.tensor(actions, dtype = torch.long, device = self.device).unsqueeze(1)
        # print('states shape is: ', states.shape)
        # print('actions shape is: ', actions.shape)
        action_prob, pred_state_values = self.net(states)
        # print('action_prob shape is: ', action_prob.shape)
        action_prob = action_prob.gather(1, actions)

        advantages = []
        Returns = []
        R = 0
        for i  in range(len(transitions) - 1, -1, -1):
            R = rewards[i] + self.gamma * R
            advantages.append(R - pred_state_values[i].detach().cpu().numpy())
            Returns.append(R)

        eps = np.finfo(np.float32).eps.item()
        advantages = torch.tensor(advantages, dtype = torch.float, device = self.device).unsqueeze(1)
        Returns = torch.tensor(Returns, dtype = torch.float, device = self.device).unsqueeze(1)
        closs = (Returns - pred_state_values) ** 2
        closs = closs.sum()
        aloss = -torch.log(action_prob) * advantages
        aloss = aloss.sum()
        self.optimizer.zero_grad()
        loss = aloss + closs
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(),1)
        self.optimizer.step()


    ### old training process, using TD(0)                          
    def oldtrain(self, pre_state, action, reward, next_state, if_end):
        
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)
        
        if self.replay_mem.num() == self.mem_size - 1:
            print('replay memory is filled,  begin training!')  

        if self.replay_mem.num() < self.mem_size:
            return      

        for _ in range(self.train_epoch):
            
            train_batch = self.replay_mem.mem

            # adjust dtype to suit the gym default dtype
            pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device).squeeze()
            action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.long, device = self.device).unsqueeze(1)
            # view to make later computation happy
            reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).unsqueeze(1)
            next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device).squeeze()
            if_end = [x[4] for x in train_batch]
            if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).unsqueeze(1)
            
            
            # use the target_Q_network to get the target_Q_value
            with torch.no_grad():
                # v_next_state = self.critic_target_net(next_state_batch).detach()
                if not self.share_network:
                    v_next_state = self.critic_policy_net(next_state_batch).detach()
                else:
                    _, v_next_state = self.net(next_state_batch)
                    v_next_state = v_next_state.detach()
                v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch

            if not self.share_network:
                v_pred = self.critic_policy_net(pre_state_batch)
            else:
                _, v_pred = self.net(pre_state_batch)
            
            if not self.share_network:
                self.critic_optimizer.zero_grad()
                closs = (v_pred - v_target) ** 2 
                closs = closs.mean()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(),1)
                self.critic_optimizer.step()
                
                
                self.actor_optimizer.zero_grad()
                action_prob = self.actor_policy_net(pre_state_batch)
                # if self.args.debug:
                #     print('Prestate Batch Shape Is: ', pre_state_batch.shape)
                #     print('Torch Action Prob Shape Is: ', action_prob.shape)
                #     print('Action Batch Shape Is: ', action_batch.shape)
                    # print('Action Batch Unsqueeze Shape Is: ', action_batch.unsqu(1).shape)
                action_prob = action_prob.gather(1, action_batch)
                log_action_prob = torch.log(action_prob.clamp(min = 1e-10))

                with torch.no_grad(): 
                    v_next_state = self.critic_policy_net(next_state_batch).detach()
                    v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch
                    TD_error = v_target - self.critic_policy_net(pre_state_batch).detach()
                
                aloss = - log_action_prob * TD_error
                aloss = aloss.mean()

                aloss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(),1)
                self.actor_optimizer.step()
            
            else:
                eps = np.finfo(np.float32).eps.item()
                closs = (v_pred - v_target) ** 2 
                closs = closs.mean()

                action_prob, _ = self.net(pre_state_batch)
                action_prob = action_prob.gather(1, action_batch)
                log_action_prob = torch.log(action_prob.clamp(min = eps))
                TD_error = v_target - v_pred
                
                aloss = - log_action_prob * TD_error
                aloss = aloss.mean()

                # entropy_loss = 

                loss = aloss + self.vloss_coef * closs 
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1)
                self.optimizer.step()


        

        self.replay_mem.clear()  ### clear the replay buffer
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, test = True): # use flag to suit other models' action interface
        s = torch.tensor(s, dtype=torch.float, device = self.device)
        with torch.no_grad():
            if not self.share_network:
                action_prob = self.actor_policy_net(s).cpu().numpy()
            else:
                action_prob, _ = self.net(s)
                action_prob = action_prob.cpu().numpy()
            num = action_prob.shape[0]
            # if self.args.debug:
            #     print('action prob is: ', action_prob)

        if test == False:
            return [(np.random.choice(self.action_dim, p = action_prob[i])) for i in range(num)]
        else:
            # print(action_prob)
            return np.argmax(action_prob, axis = 1)

    def save(self, save_path = None):
        path = save_path if save_path is not None else self.save_path
        torch.save(self.actor_policy_net.state_dict(), path + '_actor_AC.txt' )
        torch.save(self.critic_policy_net.state_dict(), path + '_critic_AC.txt')

    def load(self, load_path):
        self.critic_policy_net.load_state_dict(torch.load(load_path + '_critic_AC.txt'))
        self.actor_policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt'))
        # self.critic_policy_net.load_state_dict(torch.load(load_path + '_critic.txt'))
        # self.actor_policy_net.load_state_dict(torch.load(load_path + '_actor.txt'))

    
        
    
        