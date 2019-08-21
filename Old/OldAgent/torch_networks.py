import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal, Normal
from torch.nn import init
import numpy as np


class NAF_network(nn.Module):
        def __init__(self, state_dim, action_dim, action_low, action_high, device = 'cpu'):
            super(NAF_network, self).__init__()
            
            self.device = device
            self.sharefc1 = nn.Linear(state_dim, 64)
            self.sharefc2 = nn.Linear(64, 64)
            self.v_fc1 = nn.Linear(64, 1)
            self.miu_fc1 = nn.Linear(64, action_dim)
            self.L_fc1 = nn.Linear(64, action_dim ** 2)
            
            self.action_dim = action_dim
            self.action_low, self.action_high = action_low, action_high

            
        def forward(self, s, a = None):
            
            s = F.relu(self.sharefc1(s))
            s = F.relu(self.sharefc2(s))
            v = self.v_fc1(s)
            miu = self.miu_fc1(s)
            
            # currently could only clip according to the same one single value.
            # but different dimensions may mave different high and low bounds
            # modify to clip along different action dimension
            miu = torch.clamp(miu, self.action_low, self.action_high)
            miu = F.softmax(miu, dim =1)
            
            if a is None:
                return v, miu
        
            L = self.L_fc1(s)
            L = L.view(-1, self.action_dim, self.action_dim)

            # print('L shape is: ', L.shape)
            
            tril_mask = torch.tril(torch.ones(
             self.action_dim, self.action_dim, device = self.device), diagonal=-1).unsqueeze(0)
            diag_mask = torch.diag(torch.diag(
             torch.ones(self.action_dim, self.action_dim, device = self.device))).unsqueeze(0)

            # L = L * tril_mask.expand_as(L) +  torch.exp(L * diag_mask.expand_as(L))    
            L = L * tril_mask.expand_as(L) + L ** 2 * diag_mask.expand_as(L)
            
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (a - miu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)#[:, :, 0]

            # print('before squeeze, A.shape is: ', A.shape)
            ### before squeeze, A: 64 x 1 x 1, squeeze to be 64 x 1
            A = A.squeeze(2) 
            # print('after squeeze, A.shape is: ', A.shape)
            # print('v.shape is: ', v.shape)
            
            q = A + v
            
            return q

class FCONV_PPO_policy_net(nn.Module):
    def __init__(self, width, action_dim):
        super(FCONV_PPO_policy_net, self).__init__()
        kernel_size = 1 + 2 * width
        self.conv0 = nn.Conv1d(2, 16, kernel_size, padding = 0)
        self.conv1 = nn.Conv1d(16, 16, kernel_size, padding=width)
        self.conv2 = nn.Conv1d(16, action_dim, kernel_size, padding=width)

    def forward(self, s):
        s = F.relu(self.conv0(s))
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        action_prob = F.softmax(s, dim = 1)
        return action_prob

    def get_action_prob(self, s, a):
        action_probs = self.forward(s)
        chosen_action_prob = action_probs.gather(1, a)
        log_chosen_action_prob = torch.log(chosen_action_prob).squeeze()
        log_chosen_action_prob = torch.sum(log_chosen_action_prob, dim = 1, keepdim = True)
        chosen_action_prob = torch.exp(log_chosen_action_prob)    
        return chosen_action_prob

class FCONV_PPO_value_net(nn.Module):
    def __init__(self, width):
        super(FCONV_PPO_value_net, self).__init__()
        kernel_size = 2 * width + 1
        self.conv0 = nn.Conv1d(2, 16, kernel_size, padding=0)
        self.conv1 = nn.Conv1d(16, 8, kernel_size, padding=width)
        self.conv2 = nn.Conv1d(8, 1, kernel_size, padding=width) #### or could try with linear layer
       
    def forward(self, s):
        s = F.relu(self.conv0(s))
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))

        v = torch.sum(s, dim = 2)
        return v


class Constrained_flux_PPO_policy_net(nn.Module):
    '''
    ppo policy net for the constrained flux mode.
    '''
    def __init__(self, state_dim, hidden_layers, bias = 1):
        super(Constrained_flux_PPO_policy_net,self).__init__()
        self.state_dim = state_dim
        self.bias = bias
        
        if bias:
            self.w_in = nn.Parameter((torch.rand(self.state_dim, hidden_layers[0])))
            self.hidden_ws = nn.ParameterList()
            for i in range(len(hidden_layers) - 1):
                self.hidden_ws.append(nn.Parameter((torch.rand(hidden_layers[i], hidden_layers[i+1]))))
            self.w_out = nn.Parameter((torch.rand(hidden_layers[-1], 1)))
        else:
            self.fc_in = nn.Linear(self.state_dim, hidden_layers[0])
            self.hidden_layers = nn.ModuleList()
            for i in range(len(hidden_layers) - 1):
                self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.fc_out = nn.Linear(hidden_layers[-1], 1)


        # self.w1 = nn.Parameter(torch.rand(state_dim, hidden_neuron_num))
        # self.w2 = nn.Parameter(torch.rand(hidden_neuron_num, 1))
        self.sigma = nn.Parameter(torch.rand(1,1) * 30)

    def forward(self, s):
        sleft = s[:, :-1]
        sright = s[:, 1:]
        
        if self.bias:
            w_in = self.w_in / torch.sum(self.w_in, dim =0)
            sleft = 0.5 * torch.mm(sleft, w_in) ** 2
            sright = 0.5 * torch.mm(sright, w_in) ** 2

            for hw in self.hidden_ws:
                hw_ = hw / torch.sum(hw, dim =0)
                sleft = F.relu(torch.mm(sleft, hw_))
                sright = F.relu(torch.mm(sright, hw_))

            w_out = self.w_out / torch.sum(self.w_out, dim =0)
            meanleft = torch.mm(sleft, w_out)
            meanright = torch.mm(sright, w_out)
        else:
            sleft = F.relu(self.fc_in(sleft))
            sright = F.relu(self.fc_in(sright))
            for hidden_l  in self.hidden_layers:
                sleft = F.relu(hidden_l(sleft))
                sright = F.relu(hidden_l(sright))

            meanleft = self.fc_out(sleft)
            meanright = self.fc_out(sright)
        
        dist_left = Normal(meanleft, self.sigma)
        dist_right = Normal(meanright, self.sigma)

        return dist_left, dist_right
    
    def check_constraint(self):
        values = np.random.rand(10,1)
        input_values = np.repeat(values, self.state_dim, axis = 1)
        input_values = torch.tensor(input_values, dtype = torch.float)

        with torch.no_grad():
            w_in = self.w_in / torch.sum(self.w_in, dim =0)
            w_in = w_in.cpu()
            out = 0.5 * (torch.mm(input_values, w_in)) ** 2

            for hw in self.hidden_ws:
                hw_ = hw / torch.sum(hw, dim =0)
                hw_ = hw_.cpu()
                out = F.relu(torch.mm(out, hw_))

            w_out = self.w_out / torch.sum(self.w_out, dim =0)
            w_out = w_out.cpu()
            out = torch.mm(out, w_out).numpy()

            idealout = values ** 2 / 2.
            diff = np.abs(np.mean(idealout -out))
        
        # assert diff < 1e-5
        return diff

    def get_action_log_prob(self, state, action):
        left_dist, right_dist = self.forward(state)
        left_action = action[:,0].unsqueeze(1)
        right_action = action[:,1].unsqueeze(1)
        left_log_prob = left_dist.log_prob(left_action).sum(dim = -1, keepdim = True)
        right_log_prob = right_dist.log_prob(right_action).sum(dim = -1, keepdim = True)
        action_log_prob = left_log_prob + right_log_prob
        # left_entropy = left_dist.entropy()
        # right_entropy = right_dist.entropy()
        return action_log_prob # , (left_entropy + right_entropy) / 2

    def act(self, state, deterministic):
        with torch.no_grad():
            left_dist, right_dist = self.forward(state)
            if deterministic:
                left_actions, right_actions = left_dist.mean, right_dist.mean
                actions = torch.cat((left_actions, right_actions), dim = -1)
                actions = actions.cpu().numpy()
                return actions
            else:
                left_actions, right_actions = left_dist.sample(), right_dist.sample()
                actions = torch.cat((left_actions, right_actions), dim = -1)
                actions = actions.cpu().numpy()
                left_log_prob = left_dist.log_prob(left_actions).sum(dim = -1).cpu().numpy()
                right_log_prob = right_dist.log_prob(right_actions).sum(dim = -1).cpu().numpy()
                return actions, left_log_prob + right_log_prob

class PPO_continuous_policy_net(nn.Module):
    '''
    Continuous ppo policy net. Action is modeled as independent high-dimensional (diagmal) Gaussion distribution.
    Currently the var of the gaussian is fixed at 1.
    Could set the hidderlayers via hidden_layers params.
    Has func act to simplfy PPO class act func code.
    Implement get_action_log_prob instead of get_action_prob.
    Coupled with PPO_new.py.
    '''
    def __init__(self, state_dim, action_dim, device, hidden_layers = [64, 64]):
        super(PPO_continuous_policy_net, self).__init__()
        self.action_dim, self.state_dim = action_dim, state_dim
        self.fc_in = nn.Linear(self.state_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], self.action_dim)
        self.fc_sigma = nn.Linear(hidden_layers[-1], self.action_dim)
        # self.sigma = torch.ones((1,self.action_dim)) * 1
        self.sigma = nn.Parameter(torch.rand(1, self.action_dim) * 5)
        # self.sigma = self.sigma.to(device)
    def forward(self, s):
        x = F.relu(self.fc_in(s))
        for l in self.hidden_layers:
            x = F.relu(l(x))
        normal_mean = self.fc_out(x)
        # sigma = self.fc_sigma(x) ** 2
        # sigma = sigma.clamp(0, 10) 
        dist = Normal(normal_mean, self.sigma)
        return dist
    
    def act(self, states, deterministic = False):
        with torch.no_grad():
            dist = self.forward(states)
            if deterministic:
                actions = dist.mean
                actions = actions.cpu().numpy()
                return actions
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim = -1)
                log_probs = log_probs.cpu().numpy()
                actions = actions.cpu().numpy()
                return actions, log_probs
        
    def get_action_log_prob(self, states, actions):
        dist = self.forward(states)
        log_action_probs = dist.log_prob(actions).sum(dim=-1, keepdim = True)
        return log_action_probs

class New_PPO_value_net(nn.Module):
    '''
    New ppo value net. Could set hidden layers via hidden_layers param.
    Coupled with PPO_new.py.
    '''
    def __init__(self, state_dim, hidden_layers = [64, 64]):
        super(New_PPO_value_net, self).__init__()
        self.state_dim = state_dim
        self.fc_in = nn.Linear(state_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], 1)

    def forward(self, s):
        x = F.relu(self.fc_in(s))
        for l in self.hidden_layers:
            x = F.relu(l(x))
        x = self.fc_out(x)

        return x


class New_PPO_policy_net(nn.Module):
    '''
    New ppo policy net.
    Could set the hidderlayers via hidden_layers params.
    Has func act to simplfy PPO class act func code.
    Implement get_action_log_prob instead of get_action_prob.
    Coupled with PPO_new.py.
    '''
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64]):
        super(New_PPO_policy_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_in = nn.Linear(self.state_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], self.action_dim)
        

    def forward(self, s):
        x = F.relu(self.fc_in(s))
        for l in self.hidden_layers:
            x = F.relu(l(x))
        logits = self.fc_out(x)

        probs = F.softmax(logits, dim = 1)
        log_probs = F.log_softmax(logits, dim = 1)
        return probs, log_probs

    def act(self, s, deterministic = False):
        with torch.no_grad():
            probs, log_probs = self.forward(s)
            probs = probs.cpu().numpy()
            log_probs = log_probs.cpu().numpy()
        if deterministic:
            actions = np.argmax(probs, axis = 1)
            return actions
        else:
            num = s.shape[0]
            actions = [np.random.choice(self.action_dim, p = probs[i]) for i in range(num)]
            log_probs = log_probs[np.arange(len(log_probs)), actions]
            return actions, log_probs

    def get_action_log_prob(self, s, a):
        probs, log_probs = self.forward(s)
        probs, log_probs = probs.gather(1, a), log_probs.gather(1, a) ###### Note: shape must match

        return log_probs
        
class PPO_policy_net(nn.Module):
    '''
    old ppo policy net. 
    fixed 3 layers.
    no func act(self, s), but implemented in the ppo act function.
    has func get_action_prob instead of get_action_log_prob 
    couple with ppo.py and old ppo models
    '''
    def __init__(self, state_dim, action_dim):
        super(PPO_policy_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)
        probs = F.softmax(s, dim = 1)
        log_probs = F.log_softmax(s, dim = 1)
        return probs, log_probs
    
    def get_action_prob(self, s, a):
        probs, log_probs = self.forward(s)
        probs, log_probs = probs.gather(1, a), log_probs.gather(1, a) ###### Note: shape must match

        return probs

class PPO_value_net(nn.Module):
    '''
    old ppo value net. 
    fixed 3 layers. 
    couple with ppo.py and old ppo models
    '''
    def __init__(self, state_dim):
        super(PPO_value_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.norm1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)

        return s 

class deep_PPO_policy_net(nn.Module):
    '''
    old deep ppo policy net. 
    fixed 6 layers.
    no func act(self, s), but implemented in the ppo act function.
    has func get_action_prob instead of get_action_log_prob 
    couple with ppo.py and old ppo models
    '''
    def __init__(self, state_dim, action_dim):
        super(deep_PPO_policy_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, action_dim)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        s = F.relu(self.fc4(s))
        s = F.relu(self.fc5(s))
        s = self.fc6(s)
        probs = F.softmax(s, dim = 1)
        log_probs = F.log_softmax(s, dim = 1)
        return probs, log_probs
    
    def get_action_prob(self, s, a):
        probs, log_probs = self.forward(s)
        probs, log_probs = probs.gather(1, a), log_probs.gather(1, a) ###### Note: shape must match

        return probs

class deep_PPO_value_net(nn.Module):
    '''
    old deep ppo value net. 
    fixed 6 layers. 
    couple with ppo.py and old ppo models
    '''
    def __init__(self, state_dim):
        super(deep_PPO_value_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        s = F.relu(self.fc4(s))
        s = F.relu(self.fc5(s))
        s = self.fc6(s)

        return s 

class DDPG_critic_network(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        
        super(DDPG_critic_network, self).__init__()
        
        self.sfc1 = nn.Linear(state_dim, 64)
        # self.sfc2 = nn.Linear(64,32)
        
        self.afc1 = nn.Linear(action_dim, 64)
        # self.afc2 = nn.Linear(64,32)
        
        self.sharefc1 = nn.Linear(128, 64)
        self.sharefc2 = nn.Linear(64,1)
        
    def forward(self, s, a):
        s = F.relu(self.sfc1(s))
        # s = F.relu(self.sfc2(s))
        
        a = F.relu(self.afc1(a))
        # a = F.relu(self.afc2(a))
        
        qsa = torch.cat((s,a), 1)
        qsa = F.relu(self.sharefc1(qsa))
        qsa = self.sharefc2(qsa)

        return qsa
    
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

class DDPG_weno_actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        
        super(DDPG_weno_actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):
        
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        a = self.fc3(s)
        a = F.softmax(a)
        
        return a

class PDE_DDPG_actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, device, mat_constraint = True):
        
        super(PDE_DDPG_actor, self).__init__()
        
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        if mat_constraint:
            self.fc3 = nn.Linear(64, 1)
        else: # directly output the filter
            self.fc3 = nn.Linear(64, 3)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.mat_constraint = mat_constraint
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):

        batch_num = len(s)

        # print('batch num is {0}'.format(batch_num))

        lst = []

        s = s.cpu().numpy()

        aux = s[0][-3:]
        s = s[:,:-3]
        
        # [[lst.append(torch.cat((s_[i:i+3], aux))) for i in range(len(s_) - 5)] for s_ i in s]
        # [[lst.append(s_[i:i+3]) for i in range(self.state_dim - 5)] for s_ in s]
        [lst.append(s[:,i:i+3]) for i in range(self.state_dim - 5)]

        s_ = np.concatenate(lst, axis = 0)

        # s_ = s_.view(-1, 6)
        s_ = s_.reshape(-1, 3)
        aux = np.broadcast_to(aux,(s_.shape[0],3))
        s_ = np.concatenate((s_, aux), axis = 1)
        s_ = torch.tensor(s_, dtype = torch.float, device = self.device)

        s_ = F.relu(self.fc1(s_))
        a = self.fc3(s_)

        # print(a.size())

        a = a.view(batch_num, -1)
        a = a.clamp(self.action_low, self.action_high)
        
        return a


class PDE2_DDPG_actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, device):
        
        super(PDE2_DDPG_actor, self).__init__()
        
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
       
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):

        batch_num = len(s)

        # print('batch num is {0}'.format(batch_num))

        lst = []

        s = s.cpu().numpy()

        aux = s[0][-3:]
        s = s[:,:-3]
        
        # [[lst.append(torch.cat((s_[i:i+3], aux))) for i in range(len(s_) - 5)] for s_ i in s]
        # [[lst.append(s_[i:i+3]) for i in range(self.state_dim - 5)] for s_ in s]
        [lst.append(s[:,i:i+3]) for i in range(self.state_dim - 5)]

        s_ = np.concatenate(lst, axis = 0)

        # s_ = s_.view(-1, 6)
        s_ = s_.reshape(-1, 3)
        aux = np.broadcast_to(aux,(s_.shape[0],3))
        s_ = np.concatenate((s_, aux), axis = 1)
        s_ = torch.tensor(s_, dtype = torch.float, device = self.device)

        s_ = F.relu(self.fc1(s_))
        a = self.fc3(s_)

        # print(a)
        a = F.softmax(a)
        a = torch.max(a, 1)[1].float()
        # print(a)
        a = a.view(batch_num, -1)
        
        # a = torch.tensor(a, dtype = torch.float, device = self.device)

        return a
    
class AC_v_fc_network(nn.Module):
    
    def __init__(self, state_dim):
        super(AC_v_fc_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,1)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        v = self.fc3(s)
        
        return v
    
class AC_a_fc_network(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(AC_a_fc_network, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return F.softmax(x, dim = 1)

class AC_fc_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AC_fc_network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc0 = nn.Linear(state_dim, 64)
        self.fca = nn.Linear(64, action_dim)
        self.fcv = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        v = self.fcv(x)
        a = self.fca(x)
        a_prob = F.softmax(a, dim = 1)
        return a_prob, v
        
class CAC_a_fc_network(nn.Module):
    def __init__(self, input_dim, output_dim, action_low = -1.0, action_high = 1.0, sigma = 1, device = 'cpu'):
        super(CAC_a_fc_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.sigma = torch.ones((output_dim), device = device) * sigma
        self.action_low, self.action_high = action_low, action_high
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        mu = self.fc3(s)
        mu = torch.clamp(mu, self.action_low, self.action_high)
        
        # m = multivariate_normal.MultivariateNormal(loc = mu, covariance_matrix= self.sigma)
        m = normal.Normal(loc = mu, scale= self.sigma)

        return m


class CAC_a_sigma_fc_network(nn.Module):
    def __init__(self, input_dim, output_dim, action_low = -1.0, action_high = 1.0, sigma = 1):
        super(CAC_a_sigma_fc_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fcmu = nn.Linear(64, output_dim)
        self.fcsigma = nn.Linear(64, output_dim)
        
        self.action_low, self.action_high = action_low, action_high
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        mu = self.fcmu(s)
        mu = torch.clamp(mu, self.action_low, self.action_high)
        sigma = self.fcsigma(s)
        sigma = F.softplus(sigma)
        
        m = normal.Normal(loc = mu, scale=sigma)

        return m
        

        
