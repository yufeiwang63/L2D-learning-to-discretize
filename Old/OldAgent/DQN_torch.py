import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from helper_functions import SlidingMemory, PERMemory


class DQN_fc_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 0):
        super(DQN_fc_network, self).__init__()
        
        self.fc_in = nn.Linear(input_dim, 64)
        # self.fc_hiddens = [nn.Linear(64,64) for i in range(hidden_layers)]
        self.hidden0 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc_in(x))
        # for layer in self.fc_hiddens:
        #     x = F.relu(layer(x))
        x = F.relu(self.hidden0(x))
        x = self.fc_out(x)
        return x
        
class DQN_dueling_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers = 0):
        super(DQN_dueling_network, self).__init__()
        self.fc_in = nn.Linear(input_dim, 64)
        # self.fc_hiddens = [nn.Linear(64,64) for i in range(hidden_layers)]
        self.hidden0 = nn.Linear(64, 64)
        self.hidden1 = nn.Linear(64, 64)
        
        self.fc_a_separate = nn.Linear(64, 32)
        self.fc_v_separate = nn.Linear(64, 32)
        self.fc_a_output = nn.Linear(32, output_dim)
        self.fc_v_output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc_in(x))
        
        # for layer in self.fc_hiddens:
        #     x = F.relu(layer(x))
        
        x = F.relu(self.hidden0(x))
        x = F.relu(self.hidden1(x))

        a = F.relu(self.fc_a_separate(x))
        a = self.fc_a_output(a)
        # print('advantage shape is: ', a.shape)
        a -= torch.mean(a, dim = 1, keepdim = True)
        v = F.relu(self.fc_v_separate(x))
        v = self.fc_v_output(v)
        q = a + v
        return q 


class DQN_FCONV_network(nn.Module):
    def __init__(self, padding_width, action_dim):
        super(DQN_FCONV_network, self).__init__()
        filter_size = 2 * padding_width + 1
        self.conv0 = nn.Conv1d(2, 16, kernel_size = filter_size, padding=0)
        self.conv1 = nn.Conv1d(16, 16, kernel_size = filter_size, padding=padding_width)
        self.conv2 = nn.Conv1d(16, 16, kernel_size = filter_size, padding=padding_width)
        self.conv3 = nn.Conv1d(16, 16, kernel_size = filter_size, padding=padding_width)
        self.conv4 = nn.Conv1d(16, 16, kernel_size = filter_size, padding=padding_width)
        self.conv5 = nn.Conv1d(16, action_dim, kernel_size = filter_size, padding=padding_width)

    def forward(self, s):
        s = F.relu(self.conv0(s))
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = F.relu(self.conv4(s))
        action_value = self.conv5(s)
        return action_value  


class DQN():    
    '''
    Doc for DQN
    '''
    def __init__(self, args, if_dueling = True, if_PER = False):
        self.args = args
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma, self.lr = args.gamma, args.lr
        self.global_step = 0
        self.tau = args.tau
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.if_PER = if_PER
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_mem = PERMemory(self.mem_size) if if_PER else SlidingMemory(self.mem_size)
        self.policy_net = DQN_fc_network(self.state_dim, self.action_dim, hidden_layers=1).to(self.device)
        self.target_net = DQN_fc_network(self.state_dim, self.action_dim, hidden_layers=1).to(self.device)
        self.epsilon = 1.0
        
        if if_dueling:
            self.policy_net = DQN_dueling_network(self.state_dim, self.action_dim, hidden_layers= 1).to(self.device)
            self.target_net = DQN_dueling_network(self.state_dim, self.action_dim, hidden_layers= 1).to(self.device)
        
        if args.formulation == 'FCONV':
            self.policy_net = DQN_FCONV_network(self.args.window_size, self.action_dim).to(self.device)
            self.target_net = DQN_FCONV_network(self.args.window_size, self.action_dim).to(self.device)

        self.policy_net.apply(self._weight_init)
        self.hard_update(self.target_net, self.policy_net)
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.lr)
        else:
            print('Error: Invalid Optimizer')
            exit()
        self.hard_update(self.target_net, self.policy_net)

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

    #  training process                          
    def train(self, pre_state, action, reward, next_state, if_end):
        
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)
        
        if self.replay_mem.num() == self.train_batch_size - 1:
            print('Replay Memory is Filled, now begin training!')

        if self.replay_mem.num() < self.train_batch_size:
            return
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float).unsqueeze(1)
            
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device).squeeze()
        # if self.args.debug:
        #     print('before squeeze, pre_state_batch shape is: ', pre_state_batch.shape)
        #     print('after squeeze, pre_state_batch_shape is: ', pre_state_batch.squeeze().shape)
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.long, device = self.device).unsqueeze(1) # dtype = long for gater
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).unsqueeze(1)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device).squeeze()
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float), dtype=torch.float, device = self.device)
        
        # use the target_Q_network to get the target_Q_value
        # torch.max[0] gives the max value, torch.max[1] gives the argmax index
        
        # vanilla dqn
        #q_target_ = self.target_net(next_state_batch).max(1)[0].detach() # detach to not bother the gradient
        #q_target_ = q_target_.view(self.train_batch_size,1)
        
        ### double dqn
        with torch.no_grad():
            next_best_action = self.policy_net(next_state_batch).max(1)[1].detach().unsqueeze(1)
            # if self.args.debug:
            #     print('Next State Batch Shape Is: ', next_state_batch.shape)
            #     print('Next Best Action Shape Is: ', next_best_action.shape)
            q_target_ = self.target_net(next_state_batch).gather(1, next_best_action)
            # if self.args.debug:
            #     print('q_target_ Shape Is: ', q_target_.shape)
            
        if_end = if_end.view_as(q_target_)
        q_target = self.gamma * q_target_ * ( 1 - if_end) + reward_batch
        q_pred = self.policy_net(pre_state_batch).gather(1, action_batch) 
        
        if self.if_PER:
            TD_error_batch = np.abs(q_target.numpy() - q_pred.detach().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
        self.optimizer.zero_grad()
        
        loss = (q_pred - q_target) ** 2 
        if self.if_PER:
            loss *= weight_batch
            
        loss = torch.mean(loss)    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
        ### soft update target network
        # self.soft_update(self.target_net, self.policy_net, self.tau)
        
        ### decrease exploration rate
        self.global_step += 1
        self.epsilon *= self.args.explore_decay
        # if self.global_step % self.args.explore_decrease_every == 0:
            # self.epsilon = max(self.args.explore_final, self.epsilon - self.args.explore_decrease)
            

        ### hard update target network
        if self.global_step % self.args.update_every == 0:
            self.hard_update(self.target_net, self.policy_net)

    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
            
    # give a state and action, return the action value
    def get_value(self, s, a):
        s = torch.tensor(s,dtype=torch.float)
        a = torch.tensor(a,dtype = torch.long).unsqueeze(0)
        with torch.no_grad():
            val = self.policy_net(s).gather(1, a).cpu().numpy()
            
        return val
    
    def save(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path + '_DQN.txt')
        
        
    # use the policy net to choose the action with the highest Q value
    def action(self, s, epsilon_greedy = True):
        s = torch.tensor(s, dtype=torch.float, device = self.device) 
        p = random.random() 
        if epsilon_greedy and p <= self.epsilon:
            if self.args.formulation == 'MLP':
                return [random.randint(0, self.action_dim - 1) for i in range(s.shape[0])]
            else:
                action = [random.randint(0, self.action_dim - 1) for i in range(s.shape[2] - self.args.window_size * 2)]
                # if self.args.debug:
                #     print('In Action, random action length is: ', len(action))
                return action
        else:
            with torch.no_grad():
            # torch.max gives max value, torch.max[1] gives argmax index
                _, action = self.policy_net(s).max(dim=1)
                action = action.cpu().numpy()
            
            if self.args.formulation == 'FCONV':
                action = action[0]

            # if self.args.debug:
            #     print('In Action, action.shape is: ', action.shape)
            return action 
    
    # choose an action according to the epsilon-greedy method
    # def e_action(self, s):
    #     p = random.random()
    #     if p <= self.epsilon:
    #         return random.randint(0, self.action_dim - 1)
    #     else:
    #         return self.action(s)

    def load(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path + '_DQN.txt'))
        self.hard_update(self.target_net, self.policy_net)

    def set_explore(self, error):
        explore_rate = 0.1 * np.log(error) / np.log(10)
        if self.replay_mem.num() >= self.mem_size:
            print('reset explore rate as: ', explore_rate)
            self.epsilon = explore_rate
        