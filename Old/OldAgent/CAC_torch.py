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
from torch_networks import AC_a_fc_network, AC_v_fc_network, CAC_a_fc_network, CAC_a_sigma_fc_network
from helper_functions import SlidingMemory, PERMemory
import warnings

warnings.simplefilter("error", RuntimeWarning)

        

class CAC():    
    ''' doc for cac

    parameters:
    --------

    methods:
    --------
    
    '''
    def __init__(self, state_dim, action_dim, mem_size = 10000, train_batch_size = 64, \
                 gamma = 0.99, actor_lr = 1e-4, critic_lr = 1e-4, \
                 action_low = -1.0, action_high = 1.0, tau = 0.1, \
                 sigma = 2, if_PER = True, save_path = '/record/cac'):
        
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.actor_lr, self.critic_lr = gamma, actor_lr, critic_lr
        self.global_step = 0
        self.tau, self.if_PER= tau, if_PER
        self.state_dim, self.action_dim = state_dim, action_dim
        self.replay_mem = PERMemory(mem_size) if if_PER else SlidingMemory(mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        self.action_low, self.action_high = action_low, action_high
        self.actor_policy_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high, sigma, self.device).to(self.device)
        self.actor_target_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high, sigma, self.device).to(self.device)
        self.actor_policy_net = CAC_a_sigma_fc_network(state_dim, action_dim, action_low, action_high, sigma).to(self.device)
        self.actor_target_net = CAC_a_sigma_fc_network(state_dim, action_dim, action_low, action_high, sigma).to(self.device)
        self.critic_policy_net = AC_v_fc_network(state_dim).to(self.device)
        self.critic_target_net = AC_v_fc_network(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
    
    
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    #  training process                          
    def train(self, pre_state, action, reward, next_state, if_end):
        
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)
        
        if self.replay_mem.num() < self.mem_size:
            return
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float).unsqueeze(1)
        
        # adjust dtype to suit the gym default dtype
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device) 
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.float, device = self.device) 
        # view to make later computation happy
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).view(self.train_batch_size,1)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device)
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).view(self.train_batch_size,1)
        
        
        # use the target_Q_network to get the target_Q_value
        with torch.no_grad():
            v_next_state = self.critic_target_net(next_state_batch).detach()
            v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch

        v_pred = self.critic_policy_net(pre_state_batch)
        
        if self.if_PER:
            TD_error_batch = np.abs(v_target.cpu().numpy() - v_pred.cpu().detach().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
        self.critic_optimizer.zero_grad()
        closs = (v_pred - v_target) ** 2 
        if self.if_PER:
            closs *= weight_batch
        closs = closs.mean()
        closs.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(),2)
        self.critic_optimizer.step()
        
        
        self.actor_optimizer.zero_grad()
        
        dist = self.actor_policy_net(pre_state_batch)
        log_action_prob = dist.log_prob(action_batch)
        log_action_prob = torch.sum(log_action_prob, dim = 1)
        entropy = torch.mean(dist.entropy()) * 0.05
        
        with torch.no_grad(): 
            v_next_state = self.critic_policy_net(next_state_batch).detach()
            v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch
            TD_error = v_target - self.critic_policy_net(pre_state_batch).detach()
            
        aloss = -log_action_prob * TD_error
        aloss = aloss.mean() - entropy
        aloss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(),2)
        self.actor_optimizer.step()
    
        # update target network
        self.soft_update(self.actor_target_net, self.actor_policy_net, self.tau)
        self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)
        self.global_step += 1
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, sample = True): # use flag to suit other models' action interface
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        # print(s)
        with torch.no_grad():
            m = self.actor_policy_net(s)
            # print(m)
            a = np.clip(m.sample(), self.action_low, self.action_high) if sample else m.mean
            return a.cpu().numpy()[0]

    def save(self, save_path = None):
        path = save_path if save_path is not None else self.save_path
        torch.save(self.actor_policy_net.state_dict(), path + '_actor.txt' )
        torch.save(self.critic_policy_net.state_dict(), path + '_critic.txt')
    
        
    
        