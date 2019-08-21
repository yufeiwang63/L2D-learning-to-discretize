import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal, Normal
from torch.nn import init
import numpy as np



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
       

class weno_coef_PPO_policy_net(nn.Module):
    '''
    doc
    '''
    def __init__(self, hidden_layers = [64]):
        super(weno_coef_PPO_policy_net, self).__init__()
        self.fc_in = nn.Linear(7, hidden_layers[0])
        self.hidden_fcs = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_fcs.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.fc_out = nn.Linear(hidden_layers[-1], 4)
        self.sigma = nn.Parameter(torch.rand(1, 4) * 2)
        
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
        
        roe_left = torch.sign((fs_left[:, 3]- fs_left[:, 2]) / (s_left[:, 3] - s_left[:, 2])).unsqueeze(1)
        fs_left = torch.cat((fs_left, roe_left), dim = 1)

        roe_right = torch.sign((fs_right[:, 3] - fs_right[:, 2]) / (s_right[:, 3] - s_right[:, 2])).unsqueeze(1)
        fs_right = torch.cat((fs_right, roe_right), dim =1)
 
        a_left = F.relu(self.fc_in(fs_left))
        a_right = F.relu(self.fc_in(fs_right))
        for layer in self.hidden_fcs:
            a_left = F.relu(layer(a_left))
            a_right = F.relu(layer(a_right))

        mean_left = F.tanh(self.fc_out(a_left))
        mean_right = F.tanh(self.fc_out(a_right))
        
        dist_left = Normal(mean_left, self.sigma) 
        dist_right = Normal(mean_right, self.sigma) 
        
        return dist_left, dist_right

    def act(self, states, deterministic = False):
        with torch.no_grad():
            dist_left, dist_right = self.forward(states)
            if deterministic:
                actions_left = dist_left.mean
                actions_right = dist_right.mean
                actions_left = actions_left.cpu().numpy()
                actions_right = actions_right.cpu().numpy()
                actions = np.concatenate((actions_left, actions_right), axis = 1)
                return actions
            else:
                left_actions, right_actions = dist_left.sample(), dist_right.sample()
                actions = torch.cat((left_actions, right_actions), dim = -1)
                actions = actions.cpu().numpy()
                left_log_prob = dist_left.log_prob(left_actions).sum(dim = -1).cpu().numpy()
                right_log_prob = dist_right.log_prob(right_actions).sum(dim = -1).cpu().numpy()
                return actions, left_log_prob + right_log_prob
        
    def get_action_log_prob(self, states, actions):
        left_dist, right_dist = self.forward(states)
        left_action = actions[:,:4].unsqueeze(1)
        right_action = actions[:,4:].unsqueeze(1)
        left_log_prob = left_dist.log_prob(left_action).sum(dim = -1, keepdim = True)
        right_log_prob = right_dist.log_prob(right_action).sum(dim = -1, keepdim = True)
        action_log_prob = left_log_prob + right_log_prob
        return action_log_prob # , (left_entropy + right_entropy) / 2
    
