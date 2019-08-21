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
from torch_networks import NAF_network
from helper_functions import SlidingMemory, PERMemory

class NAF():    
    '''
    doc for naf
    '''
    def __init__(self, args, noise, flag = False, if_PER = False):
        self.args = args
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma, self.lr = args.gamma, args.lr
        self.global_step = 0
        self.tau, self.explore = args.tau, noise
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.action_high, self.action_low = args.action_high, args.action_low
        self.if_PER = if_PER
        self.replay_mem = PERMemory(self.mem_size) if if_PER else SlidingMemory(self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = NAF_network(self.state_dim, self.action_dim, self.action_low, self.action_high, self.device).to(self.device)
        self.target_net = NAF_network(self.state_dim, self.action_dim, self.action_low, self.action_high, self.device).to(self.device)
        self.policy_net.apply(self._weight_init)
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        elif self.args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.lr)
        else:
            print('Invalied Optimizer!')
            exit()
        self.hard_update(self.target_net, self.policy_net)
        
        self.flag = flag
    
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
    
    ###  training process                          
    def train(self, pre_state, action, reward, next_state, if_end):
        
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)

        if self.replay_mem.num() == self.mem_size - 1:
            print('Replay Memory Filled, Now Start Training!')
        
        if self.replay_mem.num() < self.mem_size:
            return
                
        ### sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float).unsqueeze(1)
        
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device) 
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.float, device = self.device) 
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).unsqueeze(1)#.view(self.train_batch_size,1)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device)
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).unsqueeze(1)#.view(self.train_batch_size,1)
        
        ### use the target_Q_network to get the target_Q_value
        with torch.no_grad():
            q_target_, _ = self.target_net(next_state_batch)
            q_target = self.gamma * q_target_ * (1 - if_end) + reward_batch

        q_pred = self.policy_net(pre_state_batch, action_batch)
        
        if self.if_PER:
            TD_error_batch = np.abs(q_target.cpu().numpy() - q_pred.cpu().detach().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
        self.optimizer.zero_grad()
        loss = (q_pred - q_target) ** 2 
        if self.if_PER:
            loss *= weight_batch
            
        loss = torch.mean(loss)
        if self.flag:
            loss -= q_pred.mean() # to test one of my ideas
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
        ### update target network
        self.soft_update(self.target_net, self.policy_net, self.tau)
        # self.hard_update(self.target_net, self.policy_net)
        self.global_step += 1

        ### decrease explore ratio
        if self.global_step % self.args.explore_decrease_every == 0:
            self.explore.decrease(self.args.explore_decrease)
    
    ### store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
            
            
    ### give a state and action, return the action value
    def get_value(self, s, a):
        s = torch.tensor(s,dtype=torch.float, device = self.device)
        with torch.no_grad():
            val = self.policy_net(s.unsqueeze(0)).gather(1, torch.tensor(a,dtype = torch.long).unsqueeze(1)).view(1,1)
            
        return np.clip(val.item() + np.random.rand(1, self.explore_rate), self.action_low, self.action_high)
        
    
    ### use the policy net to choose the action with the highest Q value
    def action(self, s, add_noise = True):
        s = torch.tensor(s, dtype=torch.float, device = self.device)
        with torch.no_grad():
            _, action = self.policy_net(s) 
        
        action = action.cpu().numpy()
        if add_noise:
            noise = [[self.explore.noise() for j in range(action.shape[1])] for i in range(action.shape[0])]
        else:
            noise = [[0 for j in range(action.shape[1])] for i in range(action.shape[0])]
            
        noise = np.array(noise)
        action += noise
        action = np.exp(action) / np.sum(np.exp(action), axis = 1).reshape(-1,1)
        return action

    def save(self, save_path = None):
        torch.save(self.policy_net.state_dict(), save_path + 'nafnet.txt')
    
    def load(self, load_path = None):
        self.policy_net.load_state_dict(torch.load(load_path + 'nafnet.txt'))
        self.hard_update(self.target_net, self.policy_net)
    
    def set_explore(self, error):
        explore_rate = 0.5 * np.log(error) / np.log(10)
        if self.replay_mem.num() >= self.mem_size:
            print('reset explore rate as: ', explore_rate)
            self.explore.setnoise(explore_rate)