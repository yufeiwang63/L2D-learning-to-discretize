# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:43:56 2018
# 
@author: Wangyf
"""

import numpy as np
import random
from collections import deque


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, initial_scale = 1, final_scale = 0.2, decay = 0.9995, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = initial_scale
        self.final_scale = final_scale
        self.decay = decay
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def decaynoise(self):
        self.scale *= self.decay
        self.scale = max(self.scale, self.final_scale)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        res = self.state * self.scale
        return res[0]
    
    def noisescale(self):
        return self.scale

    def decrease(self, x):
        self.scale = max(self.final_scale, self.scale - x)

    def setnoise(self, x):
        self.scale = x
    
class GaussNoise():
    '''
    This class implements the Gaussian Noise for exploration.  
    Arg initial_sig (float): the initial value of the std sigma.  
    Arg final_sig (float): the final value of the std sigma.  
    '''
    def __init__(self, initial_sig = 10, final_sig = 0):
       
        self.sig = initial_sig
        self.final_sig = final_sig
        
    def noise(self, size = 1, sig = None):
        return np.random.normal(0, self.sig, size) if sig is None else np.random.normal(0, sig, size) 
    
    def noisescale(self):
        return self.sig

    def decrease(self, x):
        self.sig = max(self.final_sig, self.sig - x)

    def setnoise(self, x):
        self.sig = x


class SlidingMemory():
    
    def __init__(self, mem_size):
        self.mem = deque()
        self.mem_size = mem_size
        
    def add(self, pair):
        self.mem.append(pair)
        if len(self.mem) > self.mem_size:
            self.mem.popleft()
            
    def num(self):
        return len(self.mem)
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def clear(self):
        self.mem.clear()

class SlidingMemory_new():
    
    def __init__(self, state_size, action_size, mem_size):
        self.mem_size = mem_size
        self.prev_states = np.zeros((self.mem_size, state_size))
        self.actions = np.zeros((self.mem_size, action_size))
        self.targets = np.zeros((self.mem_size, 1))
        self.pointer = 0
        self.full = False

    def add(self, prev_states, actions, targets):
        add_num = len(prev_states)
        if self.pointer + add_num <= self.mem_size:
            self.prev_states[self.pointer: self.pointer + add_num] = prev_states.copy()
            self.actions[self.pointer: self.pointer + add_num] = actions.copy()
            self.targets[self.pointer: self.pointer + add_num] = targets.copy()
            self.pointer = self.pointer + add_num
        else:
            self.full = True
            overflow_num = self.pointer + add_num - self.mem_size
            left_num = self.mem_size - self.pointer
            self.prev_states[self.pointer: ] = prev_states[:left_num].copy()
            self.actions[self.pointer: ] = actions[:left_num].copy()
            self.targets[self.pointer: ] = targets[:left_num].copy()
            self.prev_states[:overflow_num] = prev_states[left_num:].copy()
            self.actions[:overflow_num] = actions[left_num:].copy()
            self.targets[:overflow_num] = targets[left_num:].copy()
            self.pointer = overflow_num

    def num(self):
        return self.pointer
    
    def sample(self, batch_size):
        if self.full:
            indexes = random.sample(range(self.mem_size), batch_size)
        else:
            indexes = random.sample(range(self.pointer), batch_size)
        prev_state_batch = self.prev_states[indexes]
        action_batch = self.actions[indexes]
        target_batch = self.targets[indexes]

        return prev_state_batch, action_batch, target_batch
    
    def clear(self):
        self.prev_states.fill(0)
        self.actions.fill(0)
        self.targets.fill(0)

class SlidingMemory_batch():
    def __init__(self, mem_size):
        self.mem = deque()
        self.mem_size = mem_size
        
    def add(self, batch):
        self.mem.append(batch)
        if len(self.mem) > self.mem_size:
            self.mem.popleft()
            
    def num(self):
        return len(self.mem)
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def clear(self):
        self.mem.clear()


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.number = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        
        if idx >= self.capacity - 1:
            return idx
        
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        self.number = min(self.number + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def num(self):
        return self.number

    def clear(self):
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    

class PERMemory():
    
    def __init__(self, mem_size, alpha = 0.8, beta = 0.8, eps = 1e-2):
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.mem_size = mem_size
        self.mem = SumTree(mem_size)
        
    def add(self, state, action, reward, next_state, if_end):
        # here use reward for initial p, instead of maximum for initial p
        p = 1000
        self.mem.add([state, action, reward, next_state, if_end], p)
        
    def update(self, batch_idx, batch_td_error):
        for idx, error in zip(batch_idx, batch_td_error):
            p = (error + self.eps)  ** self.alpha 
            self.mem.update(idx, p)
        
    def num(self):
        return self.mem.num()
    
    def sample(self, batch_size):
        
        data_batch = []
        idx_batch = []
        p_batch = []
        
        segment = self.mem.total() / batch_size
        #print(self.mem.total())
        #print(segment * batch_size)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print(s < self.mem.total())
            idx, p, data = self.mem.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)
        
        p_batch = (1.0/ np.array(p_batch) /self.mem_size) ** self.beta
        p_batch /= max(p_batch)
        
        self.beta = min(self.beta * 1.00005, 1)
    
        return (data_batch, idx_batch, p_batch)
        
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, final_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - ((initial_lr - final_lr) * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    ounoise = OUNoise(args.action_dim, args.explore_initial, args.explore_final)
    gsnoise = GaussNoise(args.explore_initial, args.explore_final)
    noise = gsnoise if args.noise == 'gauss' else ounoise