import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from helper_functions import SlidingMemory, PERMemory
from torch_networks import DDPG_actor_network, DDPG_critic_network, NAF_network

        

class DDPG():    
    '''
    doc for ddpg
    '''
    def __init__(self, args, noise, if_PER = False):
        self.args = args
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma, self.actor_lr, self.critic_lr = args.gamma, args.a_lr, args.c_lr
        self.global_step = 0
        self.tau, self.explore = args.tau, noise
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.action_high, self.action_low = args.action_high, args.action_low
        self.replay_mem = PERMemory(self.mem_size) if if_PER else SlidingMemory(self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.if_PER = if_PER
        self.actor_policy_net = DDPG_actor_network(self.state_dim, self.action_dim, self.action_low, self.action_high).to(self.device)
        self.actor_target_net = DDPG_actor_network(self.state_dim, self.action_dim, self.action_low, self.action_high).to(self.device)
        self.critic_policy_net = DDPG_critic_network(self.state_dim, self.action_dim).to(self.device)
        self.critic_target_net = DDPG_critic_network(self.state_dim, self.action_dim).to(self.device)
        # self.critic_policy_net = NAF_network(state_dim, action_dim, action_low, action_high, self.device).to(self.device)
        # self.critic_target_net = NAF_network(state_dim, action_dim, action_low, action_high, self.device).to(self.device)
        self.critic_policy_net.apply(self._weight_init)
        self.actor_policy_net.apply(self._weight_init)
        if self.args.optimizer == 'adam':
            self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        elif self.args.optimizer == 'rmsprop':
            self.actor_optimizer = optim.RMSprop(self.actor_policy_net.parameters(), self.actor_lr)
            self.critic_optimizer = optim.RMSprop(self.critic_policy_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
    
    def _weight_init(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
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
            print('Replay Memory Filled! Now begin training!')

        if self.replay_mem.num() < self.mem_size:
            return
        
        ### sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float, device = self.device).unsqueeze(1)
        
        ### adjust dtype to suit the gym default dtype
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device) 
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.float, device = self.device) 
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).unsqueeze(1)#.view(self.train_batch_size,1)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device)
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).unsqueeze(1)#.view(self.train_batch_size,1)
        if self.args.debug:
            print('pre_state_batch shape: ', pre_state_batch.shape)
            print('action_batch shape: ', action_batch.shape)
            print('reward_batch shape: ', reward_batch.shape)
            print('next_state_batch shape: ', next_state_batch.shape)
            print('if_end_batch shape: ', if_end.shape)
            # exit()
        
        ### use the target_Q_network to get the target_Q_value
        with torch.no_grad():
            target_next_action_batch = self.actor_target_net(next_state_batch)
            q_target_ = self.critic_target_net(next_state_batch, target_next_action_batch)
            if self.args.debug:
                print('q_target_ shape is: ', q_target_.shape)
            q_target = self.gamma * q_target_ * (1 - if_end) + reward_batch

        q_pred = self.critic_policy_net(pre_state_batch, action_batch)
        
        if self.if_PER:
            TD_error_batch = np.abs(q_target.cpu().numpy() - q_pred.detach().cpu().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
        self.critic_optimizer.zero_grad()
        closs = (q_pred - q_target) ** 2 
        if self.if_PER:
            closs *= weight_batch
            
        closs = torch.mean(closs)
        closs.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(), 2)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        aloss = -self.critic_policy_net(pre_state_batch, self.actor_policy_net(pre_state_batch))
        
        aloss = aloss.mean()
        aloss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(), 2)
        self.actor_optimizer.step()
    

        ### update target network
        self.soft_update(self.actor_target_net, self.actor_policy_net, self.tau)
        self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)
        self.global_step += 1

        ### decrease explore ratio
        if self.global_step % self.args.explore_decrease_every == 0:
            self.explore.decrease(self.args.explore_decrease)
        
        
    ### store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    ### use the action_policy_net to compute the action
    def action(self, s, add_noise = True):
        ###### Note: do I need to squeeze or unsqueeze? 
        s = torch.tensor(s, dtype=torch.float, device = self.device)
        # if self.args.debug:
        #     print('In DDPG action func, s shape is: ', s.shape)
        with torch.no_grad():
            action = self.actor_policy_net(s).detach() 
        
        action = action.cpu().numpy()
        if add_noise: ### each action, each dimension add noise
            noise = [[self.explore.noise() for j in range(action.shape[1])] for i in range(action.shape[0])]
        else:
            noise = [[0 for j in range(action.shape[1])] for i in range(action.shape[0])]

        noise = np.array(noise)
        action += noise
        # if self.args.debug:
            # print('After noise adding, action (numpy) shape is: ', action.shape)
        action =  np.exp(action) / np.sum(np.exp(action), axis = 1).reshape(-1,1)
        return action
    

    def save(self, save_path = None):
        path = save_path if save_path is not None else self.args.save_path
        torch.save(self.actor_policy_net.state_dict(), path + 'ddpgactor.txt' )
        torch.save(self.critic_policy_net.state_dict(), path + 'ddpgcritic.txt')

    def load(self, load_path):
        self.critic_policy_net.load_state_dict(torch.load(load_path + 'ddpgcritic.txt'))
        self.actor_policy_net.load_state_dict(torch.load(load_path + 'ddpgactor.txt'))

    def set_explore(self, error):
        explore_rate = 0.5 * np.log(error) / np.log(10)
        if self.replay_mem.num() >= self.mem_size:
            print('reset explore rate as: ', explore_rate)
            self.explore.setnoise(explore_rate)
        

    