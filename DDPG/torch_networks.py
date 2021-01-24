import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal, Normal
from torch.nn import init
import numpy as np

def process_state(s, flux, eps, mode='p'):
    s_left = s[:, :-1]
    s_right = s[:, 1:]

    if flux == 'u2':
        fs_left = s_left ** 2  / 2.
        fs_right = s_right ** 2 / 2.
    elif flux == 'u4':
        fs_left = s_left ** 4  / 16.
        fs_right = s_right ** 4 / 16.
    elif flux == 'u3':
        fs_left = s_left ** 3  / 9.
        fs_right = s_right ** 3 / 9.
    elif flux == 'BL':
        fs_left = fs_left ** 2 / (fs_left ** 2 + 0.5 * (1-fs_left) ** 2)
        fs_right = fs_right ** 2 / (fs_right ** 2 + 0.5 * (1-fs_right) ** 2)

    max_left, _ = torch.max(torch.abs(fs_left), dim = 1, keepdim = True) 
    max_right, _ = torch.max(torch.abs(fs_right), dim = 1, keepdim = True)
    max_left += eps
    max_right += eps
    fs_left /= max_left
    fs_right /= max_right
    
    if mode == 'p':
        return fs_left, fs_right
    elif mode == 'q':
        return torch.cat([fs_left, fs_right], dim=-1)

def process_state_roe(s, mode, flux, eps):
    s_left = s[:, :-1]
    s_right = s[:, 1:]

    if flux == 'u2':
        fs_left = s_left ** 2  / 2.
        fs_right = s_right ** 2 / 2.
    elif flux == 'u4':
        fs_left = s_left ** 4  / 16.
        fs_right = s_right ** 4 / 16.
    elif flux == 'u3':
        fs_left = s_left ** 3  / 9.
        fs_right = s_right ** 3 / 9.
    elif flux == 'BL':
        fs_left = s_left ** 2 / (s_left ** 2 + 0.5 * (1-s_left) ** 2)
        fs_right = s_right ** 2 / (s_right ** 2 + 0.5 * (1-s_right) ** 2)
    elif flux.startswith('linear'):
        a = float(flux[len('linear'):])
        fs_left = s_left * a
        fs_right = s_right * a
    elif flux == 'identity':
        fs_left = s_left
        fs_right = s_right

    ### norm, coupled with roe speed
    if mode == 'normalize':
        max_left, _ = torch.max(torch.abs(fs_left), dim = 1, keepdim = True) 
        max_right, _ = torch.max(torch.abs(fs_right), dim = 1, keepdim = True)
        max_left += eps
        max_right += eps
        fs_left /= max_left
        fs_right /= max_right
    elif mode == 'mix':
        max_left, _ = torch.max(torch.abs(fs_left), dim = 1, keepdim = True)
        max_right, _ = torch.max(torch.abs(fs_right), dim = 1, keepdim = True)
        max_left += eps
        max_right += eps
        n_fs_left = fs_left / max_left
        n_fs_right = fs_right / max_right
        fs_left = torch.cat([fs_left, n_fs_left], dim = 1)
        fs_right = torch.cat([fs_right, n_fs_right], dim = 1)
    
    if flux == 'u2' or flux == 'u4':
        roe_left = (((s_left[:, 3] + s_left[:, 2]) >= 0).float() * 2 - 1).unsqueeze(1)
    elif flux == 'u3':
        roe_left = (((s_left[:, 3] ** 2 + s_left[:, 2] ** 2 + s_left[:, 2] * s_left[:, 3]) >= 0).float() * 2 - 1).unsqueeze(1)
    elif flux == 'BL':
        roe_left = 0.5 * (s_left[:, 3] + s_left[:, 2]) - s_left[:, 3] * s_left[:, 2]
        roe_left = (((roe_left) >= 0).float() * 2 - 1).unsqueeze(1)
    elif flux.startswith('linear'):
        a = float(flux[len('linear'):])
        sign = 1 if a >=0 else -1
        roe_left = torch.FloatTensor([sign]).unsqueeze(1).to(device)
        roe_left = roe_left.repeat(len(s_left), 1)
    elif flux == 'identity':
        roe_left = torch.ones_like((((s_left[:, 3] - s_left[:, 2]) >= 0).float() * 2 - 1).unsqueeze(1))


    fs_left = torch.cat((fs_left, roe_left), dim = 1)

    if flux == 'u2' or flux == 'u4':
        # roe_right = torch.sign(s_right[:, 3] + s_right[:, 2]).unsqueeze(1)
        roe_right = (((s_right[:, 3] + s_right[:, 2]) >=0).float() * 2 - 1).unsqueeze(1)
    elif flux == 'u3':
        # roe_right = torch.sign(s_right[:, 3] ** 2 + s_right[:, 2] ** 2 + s_right[:, 2] * s_right[:, 3]).unsqueeze(1)
        roe_right = (((s_right[:, 3] ** 2 + s_right[:, 2] ** 2 + s_right[:, 2] * s_right[:, 3]) >= 0).float() * 2 - 1).unsqueeze(1)
    elif flux == 'BL':
        roe_right = 0.5 * (s_right[:, 3] + s_right[:, 2]) - s_right[:, 3] * s_right[:, 2]
        roe_right = (((roe_right) >= 0).float() * 2 - 1).unsqueeze(1)
    elif flux.startswith('linear'):
        a = float(flux[len('linear'):])
        sign = 1 if a >=0 else -1
        roe_right = torch.FloatTensor([sign]).unsqueeze(1).to(device)
        roe_right = roe_right.repeat(len(s_right), 1)
    elif flux == 'identity':
        roe_right = torch.ones_like((((s_right[:, 3] + s_right[:, 2]) >=0).float() * 2 - 1).unsqueeze(1))

    fs_right = torch.cat((fs_right, roe_right), dim =1)

    return fs_left, fs_right
 
    


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
    def __init__(self, state_dim, action_dim, hidden_layers=[64], flux='u2', mode='normalize', batch_norm=False):
        super(weno_coef_DDPG_policy_net, self).__init__()
        self.mode = mode
        self.flux = flux
        self.eps = torch.tensor([1e-10], dtype = torch.float)
        self.batch_norm = batch_norm
        print("in weno_coef_DDPG_policy_net, batch_norm is: ", self.batch_norm)

        self.fc_in = nn.Linear(state_dim, hidden_layers[0]) ### use f1,...f6 (normal/origin), roe_speed as state
        if batch_norm:
            self.norm_in = nn.BatchNorm1d(hidden_layers[0])

        self.hidden_fcs = nn.ModuleList()
        if batch_norm:
            self.fc_norm_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_fcs.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if batch_norm:
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))

        self.fc_out = nn.Linear(hidden_layers[-1], action_dim)
        
    def forward(self, s, flux=None):
        '''
        here, left means left to the ith point, i.e., the flux f_{i - 1/2}, and right means to the right of the point,
        i.e., f_{i + 1/2}. 
        Left and right do not mean upwind direction.
        '''
        if flux is None:
            flux = self.flux

        fs_left, fs_right = process_state_roe(s, self.mode, flux, self.eps)

        a_left = F.relu(self.fc_in(fs_left))
        a_right = F.relu(self.fc_in(fs_right))
        if self.batch_norm:
            a_left = self.norm_in(a_left)
            a_right = self.norm_in(a_right)

        for idx in range(len(self.hidden_fcs)):
            layer = self.hidden_fcs[idx]
            a_left = F.relu(layer(a_left))
            a_right = F.relu(layer(a_right))
            if self.batch_norm:
                norm_layers = self.fc_norm_layers
                a_left = norm_layer(a_left)
                a_right = norm_layer(a_right)

        a_left = F.softmax(self.fc_out(a_left), dim = 1)
        a_right = F.softmax(self.fc_out(a_right), dim = 1)

        action = torch.cat((a_left, a_right), dim = 1)
        return action

    def to(self, device):
        self.eps = self.eps.to(device)
        return super(weno_coef_DDPG_policy_net, self).to(device)


# class DDPG_critic_network(nn.Module):
#     '''
#     Nips models.
#     '''
#     def __init__(self, state_dim, action_dim, hidden_layers):
#         super(DDPG_critic_network, self).__init__()
#         self.fc_in = nn.Linear(state_dim + action_dim, hidden_layers[0])
#         self.hidden_layers = nn.ModuleList()
#         for i in range(len(hidden_layers) - 1):
#             self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
#         self.fc_out = nn.Linear(hidden_layers[-1], 1)
        
#     def forward(self, s, a):
#         feature = torch.cat((s,a), dim = 1)
#         h = F.relu(self.fc_in(feature))
#         for layer in self.hidden_layers:
#             h = F.relu(layer(h))
#         q = self.fc_out(h)        

#         return q

class DDPG_critic_network(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, hidden_layers, mode, flux, batch_norm=False):
        super(DDPG_critic_network, self).__init__()
        self.mode = mode
        self.flux = flux
        self.eps = torch.tensor([1e-10], dtype = torch.float)
        self.batch_norm = batch_norm
        print("in DDPG_critic_network, batch_norm is: ", self.batch_norm)

        self.fc_in = nn.Linear(state_dim + action_dim, hidden_layers[0])
        if batch_norm:
            self.norm_layer_in = nn.BatchNorm1d(hidden_layers[0])

        self.hidden_layers = nn.ModuleList()
        if batch_norm:
            self.fc_norm_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if batch_norm:
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_layers[i+1]))

        self.fc_out = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, s, a):
        fs_left, fs_right = process_state_roe(s, self.mode, self.flux, self.eps)
        feature = torch.cat((fs_left, fs_right, a), dim = 1)
        
        h = F.relu(self.fc_in(feature))
        if self.batch_norm:
            h = self.norm_layer_in(h)

        for idx in range(len(self.hidden_layers)):
            layer = self.hidden_layers[idx]
            h = F.relu(layer(h))
            if self.batch_norm:
                norm_layer = self.fc_norm_layers[idx]
                h = norm_layer(h)
        
        q = self.fc_out(h)        

        return q.squeeze()
    
    def to(self, device):
        self.eps = self.eps.to(device)
        return super(DDPG_critic_network, self).to(device)


class bottleblock(nn.Module):
    def __init__(self, hidden, out):
        super(bottleblock, self).__init__()
        self.fc1 = nn.Linear(out, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        x0 = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += x0
        return F.relu(out)


class weno_coef_DDPG_policy_residual_net(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, residual_num = 3, flux = 'u2'):
        super(weno_coef_DDPG_policy_residual_net, self).__init__()
        self.resi0_0 = nn.Linear(state_dim, 64) ### use f1,...f6 + weno_weight as state. size 6 + 8
        self.resi0_1 = nn.Linear(64, 4)

        self.resi_blocks = nn.ModuleList()
        for i in range(residual_num - 1):
            self.resi_blocks.append(bottleblock(64, 4))
        
        self.fc_out = nn.Linear(4, action_dim // 2)
        self.flux = flux
        self.eps = torch.tensor([1e-100], dtype = torch.float)
        
    def forward(self, s):
        '''
        here, left means left to the ith point, i.e., the flux f_{i - 1/2}, and right means to the right of the point,
        i.e., f_{i + 1/2}. 
        Left and right do not mean upwind direction.
        '''
        weno_left_weights, weno_right_weights = s[:, -8:-4], s[:, -4:]
        fs_left, fs_right = process_state(s[:, :-8],  self.flux, self.eps)

        fs_left = torch.cat([weno_left_weights, fs_left], dim=1)
        fs_right = torch.cat([weno_right_weights, fs_right], dim=1)
        ss = [fs_left, fs_right]
        w0 = [weno_left_weights, weno_right_weights]
        out = [None, None]
        for idx in range(2):
            s = ss[idx]
            s = F.relu(self.resi0_0(s))
            s = self.resi0_1(s)
            s += w0[idx]
            s = F.relu(s)

            for idx2 in range(len(self.resi_blocks)):
                s = self.resi_blocks[idx2](s)

            s = self.fc_out(s)
            out[idx] = s

        a_left = F.softmax(out[0], dim = 1)
        a_right = F.softmax(out[1], dim = 1)
        action = torch.cat((a_left, a_right), dim = 1)
        return action

    def to(self, device):
        self.eps = self.eps.to(device)
        return super(weno_coef_DDPG_policy_residual_net, self).to(device)


class DDPG_critic_network_residual(nn.Module):
    '''
    doc
    '''
    def __init__(self, state_dim, action_dim, residual_num = 3, flux = 'u2'):
        super(DDPG_critic_network_residual, self).__init__()
        self.flux = flux

        self.resi0_0 = nn.Linear(state_dim + action_dim, 64) # state should be 6 + 6 + 8 + 8.
        self.resi0_1 = nn.Linear(64, state_dim + action_dim)

        self.resi_blocks = nn.ModuleList()
        for i in range(residual_num - 1):
            self.resi_blocks.append(bottleblock(64, state_dim + action_dim))
        
        self.fc_out = nn.Linear(state_dim + action_dim, 1)
        self.flux = flux
        self.eps = torch.tensor([1e-100], dtype = torch.float)
        
    def forward(self, s, a):
        fs = process_state(s[:, :-8], self.flux, self.eps, mode='q') # N x 12
        x = torch.cat([fs, s[:, -8:], a], dim = 1)  # N x (12 + 8 + 8)

        identity = x
        out = F.relu(self.resi0_0(x))
        out = self.resi0_1(out)
        out += identity
        out = F.relu(out)
        
        for block in self.resi_blocks:
            out = block(out)

        q = self.fc_out(out)
        return q
    
    def to(self, device):
        self.eps = self.eps.to(device)
        return super(DDPG_critic_network_residual, self).to(device)
