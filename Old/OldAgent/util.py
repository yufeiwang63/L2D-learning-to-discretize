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
    def __init__(self, initial_var = 10, final_var = 0, decay = 0.995):
        self.var = initial_var
        self.final_var = final_var
        self.decay = decay
        
    def decaynoise(self):
        self.var *= self.decay
        self.var = max(self.final_var, self.var)
        
    def noise(self, var = None):
        return np.random.normal(0, self.var) if var is None else np.random.normal(0, var) 
    
    def noisescale(self):
        return self.var

    def decrease(self, x):
        self.var = max(self.final_var, self.var - x)

    def setnoise(self, x):
        self.var = x


class SlidingMemory():
    
    def __init__(self, mem_size):
        self.mem = deque()
        self.mem_size = mem_size
        
    def add(self, state, action, reward, next_state, if_end):
        self.mem.append([state, action, reward, next_state, if_end])
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
        

if __name__ == '__main__':
    ounoise = OUNoise(args.action_dim, args.explore_initial, args.explore_final)
    gsnoise = GaussNoise(args.explore_initial, args.explore_final)
    noise = gsnoise if args.noise == 'gauss' else ounoise