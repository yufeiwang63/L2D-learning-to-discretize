import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal, Normal
from torch.nn import init
import numpy as np

class constrained_flux_DDPG_policy_net(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim = 2, hidden_layers = [64]):
        super(constrained_flux_DDPG_policy_net, self).__init__()
        self.state_dim = state_dim
        self.w_in = nn.Parameter((torch.rand(self.state_dim, hidden_layers[0])))
        self.hidden_ws = nn.ParameterList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_ws.append(nn.Parameter((torch.rand(hidden_layers[i], hidden_layers[i+1]))))
        self.w_out = nn.Parameter((torch.rand(hidden_layers[-1], 1)))
        
    def forward(self, s):
        sleft = s[:, :-1]
        sright = s[:, 1:]
        
        w_in = self.w_in / torch.sum(self.w_in, dim =0)
        sleft = 0.5 * torch.mm(sleft, w_in) ** 2
        sright = 0.5 * torch.mm(sright, w_in) ** 2

        for hw in self.hidden_ws:
            hw_ = hw / torch.sum(hw, dim =0)
            sleft = F.relu(torch.mm(sleft, hw_))
            sright = F.relu(torch.mm(sright, hw_))

        w_out = self.w_out / torch.sum(self.w_out, dim =0)
        actionleft = torch.mm(sleft, w_out)
        actionright = torch.mm(sright, w_out)
        action = torch.cat((actionleft, actionright), dim = 1)

        return action

class weno_coef_DDPG_policy_net_fs(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, hidden_layers = [64]):
        super(weno_coef_DDPG_policy_net_fs, self).__init__()
        self.fc_in = nn.Linear((state_dim - 1) * 2, hidden_layers[0]) ### use f1,...f6, s1,...s6 as state
        self.hidden_fcs = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_fcs.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], action_dim // 2)
        
    def forward(self, s):
        '''
        here, left means left to the ith point, i.e., the flux f_{i - 1/2}, and right means to the right of the point,
        i.e., f_{i + 1/2}. 
        Left and right do not mean upwind direction.
        '''
        s_left = s[:, :-1]
        s_right = s[:, 1:]

        fs_left = s_left ** 2  / 2.
        fs_right = s_right ** 2 / 2.

        ### use f and u as state
        fs_left = torch.cat([s_left, fs_left], dim = 1)
        fs_right = torch.cat([s_right, fs_right], dim = 1)
 
        a_left = F.relu(self.fc_in(fs_left))
        a_right = F.relu(self.fc_in(fs_right))
        for layer in self.hidden_fcs:
            a_left = F.relu(layer(a_left))
            a_right = F.relu(layer(a_right))

        a_left = F.softmax(self.fc_out(a_left), dim = 1)
        a_right = F.softmax(self.fc_out(a_right), dim = 1)

        action = torch.cat((a_left, a_right), dim = 1)

        return action

class weno_coef_DDPG_policy_net(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, hidden_layers = [64], flux = 'u2'):
        super(weno_coef_DDPG_policy_net, self).__init__()
        self.fc_in = nn.Linear(state_dim, hidden_layers[0]) ### use f1,...f6, roe_speed as state
        self.hidden_fcs = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_fcs.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], action_dim // 2)
        self.flux = flux
        
    def forward(self, s):
        '''
        here, left means left to the ith point, i.e., the flux f_{i - 1/2}, and right means to the right of the point,
        i.e., f_{i + 1/2}. 
        Left and right do not mean upwind direction.
        '''
        s_left = s[:, :-1]
        s_right = s[:, 1:]

        if self.flux == 'u2':
            fs_left = s_left ** 2  / 2.
            fs_right = s_right ** 2 / 2.
        elif self.flux == 'u4':
            fs_left = s_left ** 4  / 16.
            fs_right = s_right ** 4 / 16.
        elif self.flux == 'u3':
            fs_left = s_left ** 3  / 9.
            fs_right = s_right ** 3 / 9.

        ### norm, coupled with roe speed
        max_left, _ = torch.max(torch.abs(fs_left), dim = 1, keepdim = True)
        max_right, _ = torch.max(torch.abs(fs_right), dim = 1, keepdim = True)
        fs_left /= max_left
        fs_right /= max_right
        if self.flux == 'u2' or self.flux == 'u4':
            roe_left = torch.sign(s_left[:, 3] + s_left[:, 2]).unsqueeze(1)
        elif self.flux == 'u3':
            roe_left = torch.sign(s_left[:, 3] ** 2 + s_left[:, 2] ** 2 + s_left[:, 2] * s_left[:, 3]).unsqueeze(1)
        fs_left = torch.cat((fs_left, roe_left), dim = 1)
        if self.flux == 'u2' or self.flux == 'u4':
            roe_right = torch.sign(s_right[:, 3] + s_right[:, 2]).unsqueeze(1)
        elif self.flux == 'u3':
            roe_right = torch.sign(s_right[:, 3] ** 2 + s_right[:, 2] ** 2 + s_right[:, 2] * s_right[:, 3]).unsqueeze(1)
        fs_right = torch.cat((fs_right, roe_right), dim =1)
 
        a_left = F.relu(self.fc_in(fs_left))
        a_right = F.relu(self.fc_in(fs_right))
        for layer in self.hidden_fcs:
            a_left = F.relu(layer(a_left))
            a_right = F.relu(layer(a_right))

        a_left = F.softmax(self.fc_out(a_left), dim = 1)
        a_right = F.softmax(self.fc_out(a_right), dim = 1)

        action = torch.cat((a_left, a_right), dim = 1)

        return action


class nonlinear_weno_coef_DDPG_policy_net(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, hidden_layers = [64]):
        super(nonlinear_weno_coef_DDPG_policy_net, self).__init__()
        self.fc_in = nn.Linear(state_dim, hidden_layers[0])
        self.hidden_fcs = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_fcs.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], action_dim // 2)
        
    def forward(self, s):
        '''
        here, left means left to the ith point, i.e., the flux f_{i - 1/2}, and right means to the right of the point,
        i.e., f_{i + 1/2}. 
        Left and right do not mean upwind direction.
        '''
        s_left = s[:, :-1]
        s_right = s[:, 1:]

        fs_left = s_left ** 2  / 2.
        fs_right = s_right ** 2 / 2.

        max_left, _ = torch.max(torch.abs(fs_left), dim = 1, keepdim = True)
        max_right, _ = torch.max(torch.abs(fs_right), dim = 1, keepdim = True)

        fs_left /= max_left
        fs_right /= max_right
        
        # roe_left = torch.sign((fs_left[:, 3]- fs_left[:, 2]) / (s_left[:, 3] - s_left[:, 2])).unsqueeze(1)
        roe_left = torch.sign(s_left[:, 3] + s_left[:, 2]).unsqueeze(1)
        fs_left = torch.cat((fs_left, roe_left), dim = 1)

        # roe_right = torch.sign((fs_right[:, 3] - fs_right[:, 2]) / (s_right[:, 3] - s_right[:, 2])).unsqueeze(1)
        roe_right = torch.sign(s_right[:, 3] + s_right[:, 2]).unsqueeze(1)
        fs_right = torch.cat((fs_right, roe_right), dim =1)
 
        a_left = F.relu(self.fc_in(fs_left))
        a_right = F.relu(self.fc_in(fs_right))

        for idx, layer in enumerate(self.hidden_fcs):
            a_left = F.relu(layer(a_left))
            a_right = F.relu(layer(a_right))

     
        a_left = self.fc_out(a_left)
        a_right = self.fc_out(a_right)
        a_left = a_left.clamp(-1, 1)
        a_right = a_right.clamp(-1, 1)

        action = torch.cat((a_left, a_right), dim = 1)

        return action

class DDPG_critic_network(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DDPG_critic_network, self).__init__()
        self.fc_in = nn.Linear(state_dim + action_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, s, a):
        feature = torch.cat((s,a), dim = 1)
        h = F.relu(self.fc_in(feature))
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        q = self.fc_out(h)        

        return q
    
class DDPG_actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        
        super(DDPG_actor_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):
        
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a = self.fc3(s)
        # print('In DDPG network, before softmax a.shape is: ', a.shape)
        # a = a.clamp(self.action_low, self.action_high)
        a = F.softmax(a, dim = 1)
        # print('In DDPG network, after softmax a is: ', a.shape)
        
        return a

        
