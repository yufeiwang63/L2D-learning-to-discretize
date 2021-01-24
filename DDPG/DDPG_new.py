import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from DDPG.util import GaussNoise
from DDPG.torch_networks import DDPG_critic_network, weno_coef_DDPG_policy_net
import utils.ptu as ptu
import copy

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = [self.obs_buf[idxs],
                self.act_buf[idxs],
                self.rew_buf[idxs],
                self.done_buf[idxs],
                self.obs2_buf[idxs],]
        return [torch.as_tensor(v, dtype=torch.float32).to(ptu.device) for v in batch]

class DDPG():    
    '''
    doc for ddpg
    '''
    def __init__(self, vv, noise):
        self.vv = copy.deepcopy(vv)
        self.action_noise = noise
        self.replay_mem = ReplayBuffer(vv['state_dim'], vv['action_dim'] * 2, vv['replay_buffer_size'])
    
        policy_hidden_layers = vv['policy_hidden_layers']
        critic_hidden_layers = vv['critic_hidden_layers']
      
        self.actor = weno_coef_DDPG_policy_net(vv['state_dim'], vv['action_dim'], 
            policy_hidden_layers, vv['flux'], vv['state_mode'], vv['batch_norm']).to(ptu.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = DDPG_critic_network(vv['state_dim'] * 2, vv['action_dim'] * 2, critic_hidden_layers, 
            vv['state_mode'], vv['flux'], vv['batch_norm']).to(ptu.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic2 = DDPG_critic_network(vv['state_dim'] * 2, vv['action_dim'] * 2, critic_hidden_layers,
            vv['state_mode'], vv['flux'], vv['batch_norm']).to(ptu.device) ### twin
        self.critic_target2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), vv['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), vv['critic_lr'])
        self.critic_optimizer2 = optim.RMSprop(self.critic2.parameters(), vv['critic_lr']) ### twin
        self.global_step = 0


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    ###  training process                          
    def train(self):
        if self.replay_mem.size < self.vv['batch_size']:
            return None, None, None
        
        ### sample $self.batch_size$ samples from the replay memory, and use them to train
        value_loss = 0
        value_loss2 = 0
        policy_loss = 0

        ### use the target_Q_network to get the target_Q_value
        for _ in range(self.vv['ddpg_value_train_iter']):
            prev_state, action, reward, done, next_state = self.replay_mem.sample_batch(self.vv['batch_size'])
            # print(prev_state)
            q_pred = self.critic(prev_state, action)
            
            with torch.no_grad():
                pi_targ = self.actor_target(next_state)

                # Target Q-values
                q1_pi_targ = self.critic_target(next_state, pi_targ)
                q2_pi_targ = self.critic_target2(next_state, pi_targ)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = reward + self.vv['gamma'] * (1 - done) * q_pi_targ

            closs = (q_pred - backup) ** 2             
            closs = torch.mean(closs)
            value_loss += closs.cpu().item()
            self.critic_optimizer.zero_grad()
            closs.backward()
            if self.vv['clip_gradient']:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.vv['max_grad_norm'])
            self.critic_optimizer.step()

            ### twin
            q_pred2 = self.critic2(prev_state, action)
            closs2 = (q_pred2 - backup) ** 2             
            closs2 = torch.mean(closs2)
            value_loss2 += closs2.cpu().item()
            self.critic_optimizer2.zero_grad()
            closs2.backward()
            if self.vv['clip_gradient']:
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.vv['max_grad_norm'])
            self.critic_optimizer2.step()

        value_loss /= self.vv['ddpg_value_train_iter']
        value_loss2 /= self.vv['ddpg_value_train_iter']

        aloss = - self.critic(prev_state, self.actor(prev_state))
        aloss = aloss.mean()            
        self.actor_optimizer.zero_grad()
        aloss.backward()
        policy_loss += aloss.cpu().item()
        if self.vv['clip_gradient']:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.vv['max_grad_norm'])
        self.actor_optimizer.step()
    
        self.global_step += 1
        self.soft_update(self.actor_target, self.actor, self.vv['tau'])
        self.soft_update(self.critic_target, self.critic, self.vv['tau'])
        self.soft_update(self.critic_target2, self.critic2, self.vv['tau']) ### twin

        return policy_loss, value_loss, value_loss2

    def store(self, state, action, reward, next_state, done):
        self.replay_mem.store(state, action, reward, next_state, done)

    ### use the action_policy_net to compute the action
    def action(self, s, deterministic=False, mode='agent', flux=None):
        assert mode in ['weno', 'agent'], 'only supports `weno` or `agent`'
        if mode == 'weno':
            weno_coef = self.weno_coef(np.array(s))
            return weno_coef

        s = torch.FloatTensor(s).to(ptu.device)
        with torch.no_grad():
            action = self.actor(s, flux=flux)

        action = action.cpu().numpy()
        action_num = action.shape[1] // 2
        if not deterministic: ### each action, each dimension add noise
            a_left, a_right = action[:, :action_num], action[:,action_num:]
            if not self.vv['batch_norm']:
                assert np.sum(np.abs(a_left[1:] - a_right[:-1])) < 1e-5
            size = a_left.shape
            noise_left = self.action_noise.noise(size=size)
            noise_right = self.action_noise.noise(size=size) 
            a_left += noise_left
            a_right += noise_right
            a_left = np.clip(a=a_left, a_min=1e-20, a_max=None)
            a_right = np.clip(a=a_right, a_min=1e-20, a_max=None)
            a_left = a_left / np.sum(a_left, axis=1).reshape(-1,1)
            a_right = a_right / np.sum(a_right, axis=1).reshape(-1,1)
            action = np.concatenate((a_left, a_right), axis=1)

        return action

    def weno_coef(self, s):
        '''
        This function directly construct num_x + 1 fluxes, which is 
        f_{-1/2}, f_{1/2}, ... , f_{num_x - 1 / 2}, f_{num_x + 1 / 2}

        param s: still assume each s has 7 elements, and in total there are num_x points (grid size). \
            we add three ghost points at the boundary. so a s is, e.g., {u-3, u-2, u-1, u0, u1, u2, u3} 
        '''
        num = len(s) 
        ### when computing betas, finite difference weno reconstrunction use the flux values as the cell average.
        f = s ** 2 / 2. 

        dleft2, dleft1, dleft0 = 0.1, 0.6, 0.3 ### ideal weight for reconstruction of the left index boundary. (or the minus one in the book.)
        dright2, dright1, dright0 = 0.3, 0.6, 0.1
        
        fl = np.zeros((num + 1, 5))
        fr = np.zeros((num + 1, 5))
        fl[:-1] = f[:, :5]
        fl[-1] = f[-1, 1:6] ### need to add for the flux f_{num_x + 1 / 2}
        fr[:-1] = f[:, 1:6]
        fr[-1] = f[-1, 2:7] ### need to add for the flux f_{num_x + 1 / 2}

        ### in the following coef related vars (beta, alpha), the number indicated 'r', i.e., the shift of the leftmost points of the stencil.
        betal0 = 13 / 12 * (fl[:,2] - 2 * fl[:,3] + fl[:,4]) ** 2 + 1 / 4 * (3 * fl[:,2] - 4 * fl[:,3] + fl[:,4]) ** 2
        betal1 = 13 / 12 * (fl[:,1] - 2 * fl[:,2] + fl[:,3]) ** 2 + 1 / 4 * (fl[:,1] - fl[:,3]) ** 2
        betal2 = 13 / 12 * (fl[:,0] - 2 * fl[:,1] + fl[:,2]) ** 2 + 1 / 4 * (fl[:,0] - 4 * fl[:,1] + 3 * fl[:,2]) ** 2

        betar0 = 13 / 12 * (fr[:,2] - 2 * fr[:,3] + fr[:,4]) ** 2 + 1 / 4 * (3 * fr[:,2] - 4 * fr[:,3] + fr[:,4]) ** 2
        betar1 = 13 / 12 * (fr[:,1] - 2 * fr[:,2] + fr[:,3]) ** 2 + 1 / 4 * (fr[:,1] - fr[:,3]) ** 2
        betar2 = 13 / 12 * (fr[:,0] - 2 * fr[:,1] + fr[:,2]) ** 2 + 1 / 4 * (fr[:,0] - 4 * fr[:,1] + 3 * fr[:,2]) ** 2
        
        eps = 1e-6
        
        alphal0 = dleft0 / (betal0 + eps) ** 2
        alphal1 = dleft1 / (betal1 + eps) ** 2
        alphal2 = dleft2 / (betal2 + eps) ** 2
        wl0 = alphal0 / (alphal0 + alphal1 + alphal2)
        wl1 = alphal1 / (alphal0 + alphal1 + alphal2)
        wl2 = alphal2 / (alphal0 + alphal1 + alphal2)

        alphar0 = dright0 / (betar0 + eps) ** 2
        alphar1 = dright1 / (betar1 + eps) ** 2
        alphar2 = dright2 / (betar2 + eps) ** 2
        wr0 = alphar0 / (alphar0 + alphar1 + alphar2)
        wr1 = alphar1 / (alphar0 + alphar1 + alphar2)
        wr2 = alphar2 / (alphar0 + alphar1 + alphar2)

        ### compute the roe speed, and for flux f_{i + 1/2} it is (f_{i+1} - f_{i}) / (u_{i+1} - u_{i})
        roe = np.zeros(num + 1)
        # roe[:-1] = (f[:,3] - f[:, 2]) / (s[:, 3] - s[:,2])
        # roe[-1] = (f[-1,4] - f[-1, 3]) / (s[-1, 4] - s[-1,3]) ### need to add one more roe speed for flux f_{num_x + 1 / 2}.
        roe[:-1] = (s[:, 3] + s[:,2])
        roe[-1] = (s[-1, 4] + s[-1,3]) ### need to add one more roe speed for flux f_{num_x + 1 / 2}.
        

        ### we put all four possible stencils all together for computation, while in weno only 3 can have positive weight at the same time.
        coef = np.zeros((num + 1, 4))
        # coef = np.zeros((num + 1,3)) 
        for i in range(num + 1):
            # judge = ori_s[i][2] if i < num else ori_s[-1][3]
            judge = roe[i]
            if judge >= 0: ### if roe speed > 0, use the minus (left) flux. 
                coef[i][0] = wl2[i]
                coef[i][1] = wl1[i]
                coef[i][2] = wl0[i]
            else: ### if roe speed < 0, use the plus (right) flux.
                coef[i][1] = wr2[i]
                coef[i][2] = wr1[i]
                coef[i][3] = wr0[i]

        action_left = coef[:-1]
        action_right = coef[1:]
        action = np.concatenate((action_left, action_right), axis = 1)
        return action

    def train_mode(self, flag):
        if flag:
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic2.train()
            self.critic_target.train()
            self.critic_target2.train()
        else:
            self.actor.eval()
            self.actor_target.eval()
            self.critic.eval()
            self.critic2.eval()
            self.critic_target.eval()
            self.critic_target2.eval()

    def save(self, save_path=None):
        path = save_path
        torch.save(self.actor.state_dict(), path + 'ddpgactor.txt' )
        torch.save(self.critic.state_dict(), path + 'ddpgcritic.txt')

    def load(self, load_path, actor_only=False):
        if not actor_only:
            self.critic.load_state_dict(torch.load(load_path + 'ddpgcritic.txt', map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(load_path + 'ddpgactor.txt', map_location=lambda storage, loc: storage))


    