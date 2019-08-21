import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from DDPG.util import SlidingMemory, update_linear_schedule
from DDPG.torch_networks import DDPG_actor_network, DDPG_critic_network, constrained_flux_DDPG_policy_net, weno_coef_DDPG_policy_net
        

class DDPG():    
    '''
    doc for ddpg
    '''
    def __init__(self, args, noise, tb_logger = None):
        self.args, self.tb_logger = args, tb_logger
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma, self.actor_lr, self.critic_lr = args.gamma, args.a_lr, args.c_lr
        self.global_step = 0
        self.tau = args.tau
        self.action_noise = noise
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.action_high, self.action_low = args.action_high, args.action_low
        self.replay_mem = SlidingMemory(self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        hidden_layers = [64 for _ in range(self.args.hidden_layer_num)]
        self.actor_policy_net = constrained_flux_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers).to(self.device)
        self.actor_target_net = constrained_flux_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers).to(self.device)
        self.critic_policy_net = DDPG_critic_network(self.state_dim + 1, 2, hidden_layers).to(self.device)
        self.critic_target_net = DDPG_critic_network(self.state_dim + 1, 2, hidden_layers).to(self.device)
        if args.mode == 'weno_coef':
            self.actor_policy_net = weno_coef_DDPG_policy_net().to(self.device)
            self.actor_target_net = weno_coef_DDPG_policy_net().to(self.device)
            self.critic_policy_net = DDPG_critic_network(7, 8, hidden_layers).to(self.device)
            self.critic_target_net = DDPG_critic_network(7, 8, hidden_layers).to(self.device)
            self.critic_policy_net_2 = DDPG_critic_network(7, 8, hidden_layers).to(self.device) ### twin
            self.critic_target_net_2 = DDPG_critic_network(7, 8, hidden_layers).to(self.device) ### twin
        
        self.critic_policy_net.apply(self._weight_init)
        self.critic_policy_net_2.apply(self._weight_init)
        self.actor_policy_net.apply(self._weight_init)
        self.actor_optimizer = optim.RMSprop(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.RMSprop(self.critic_policy_net.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.RMSprop(self.critic_policy_net_2.parameters(), self.critic_lr) ### twin
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
        self.hard_update(self.critic_target_net_2, self.critic_policy_net_2) ### twin
    
    def _weight_init(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    ###  training process                          
    def train(self):
        if self.args.mode == 'check_weno_coef':
            return
        
        if self.replay_mem.num() < self.train_batch_size:
            return
        
        ### sample $self.train_batch_size$ samples from the replay memory, and use them to train
        tb_value_loss = 0
        tb_policy_loss = 0
        update_linear_schedule(self.critic_optimizer, self.global_step, self.args.train_epoch, self.args.c_lr, self.args.final_c_lr)
        update_linear_schedule(self.critic_optimizer_2, self.global_step, self.args.train_epoch, self.args.c_lr, self.args.final_c_lr)
        update_linear_schedule(self.actor_optimizer, self.global_step, self.args.train_epoch, self.args.a_lr, self.args.final_a_lr)
        for _ in range(self.args.ddpg_train_iter):
            train_batch = self.replay_mem.sample(self.train_batch_size)
            # train_batch = self.replay_mem.mem
            
            ### adjust dtype to suit the gym default dtype
            pre_state_batch = np.array([x[0] for x in train_batch])
            action_batch = np.array([x[1] for x in train_batch])
            reward_batch = np.array([x[2] for x in train_batch])
            next_state_batch = np.array([x[3] for x in train_batch])
            if_end = np.array([x[4] for x in train_batch]).astype(float)

            pre_state_batch = torch.tensor(pre_state_batch, dtype=torch.float, device = self.device) 
            action_batch = torch.tensor(action_batch, dtype = torch.float, device = self.device) 
            reward_batch = torch.tensor(reward_batch, dtype=torch.float, device = self.device).unsqueeze(1)#.view(self.train_batch_size,1)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float, device = self.device)
            if_end = torch.tensor(if_end,device = self.device, dtype=torch.float).unsqueeze(1)#.view(self.train_batch_size,1)
            
            ### use the target_Q_network to get the target_Q_value
            for i in range(self.args.ddpg_value_train_iter):
                with torch.no_grad():
                    target_next_action_batch = self.actor_target_net(next_state_batch)
                    q_target_ = self.critic_target_net(next_state_batch, target_next_action_batch)
                    q_target_2 = self.critic_target_net_2(next_state_batch, target_next_action_batch) ### twin
                    q_target_ = torch.min(q_target_, q_target_2) ### twin
                    q_target = self.gamma * q_target_ * (1 - if_end) + reward_batch

                q_pred = self.critic_policy_net(pre_state_batch, action_batch)
                closs = (q_pred - q_target) ** 2             
                closs = torch.mean(closs)
                tb_value_loss += closs.cpu().item()
                self.critic_optimizer.zero_grad()
                closs.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(), self.args.grad_norm)
                self.critic_optimizer.step()

                ### twin
                q_pred = self.critic_policy_net_2(pre_state_batch, action_batch)
                closs = (q_pred - q_target) ** 2             
                closs = torch.mean(closs)
                self.critic_optimizer_2.zero_grad()
                closs.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(), self.args.grad_norm)
                self.critic_optimizer_2.step()
            
            aloss = -self.critic_policy_net(pre_state_batch, self.actor_policy_net(pre_state_batch))
            aloss = aloss.mean()
            self.actor_optimizer.zero_grad()
            aloss.backward()
            tb_policy_loss += aloss.cpu().item()
            # torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(), self.args.grad_norm)
            self.actor_optimizer.step()
        
        self.global_step += 1
        ### update target network
        if self.args.update_mode == 'soft':
            self.soft_update(self.actor_target_net, self.actor_policy_net, self.tau)
            self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)
            self.soft_update(self.critic_target_net_2, self.critic_policy_net_2, self.tau) ### twin
        else:
            if self.global_step % self.args.update_every == 0:
                self.hard_update(self.actor_target_net, self.actor_policy_net)
                self.hard_update(self.critic_target_net, self.critic_policy_net)
                self.hard_update(self.critic_target_net_2, self.critic_policy_net_2) ### twin

        if self.tb_logger is not None:
            self.tb_logger.add_scalar('value loss', tb_value_loss / self.args.ddpg_train_iter, self.global_step)
            self.tb_logger.add_scalar('policy loss', tb_policy_loss / self.args.ddpg_train_iter, self.global_step)
        # self.replay_mem.clear()

    ### store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, dataset):
        for env_dataset in dataset:
            for batch_dataset in env_dataset:
                for pairs in batch_dataset:
                    self.replay_mem.add(pairs)
        
    
    ### use the action_policy_net to compute the action
    def action(self, s, deterministic = False):
        if self.args.mode != 'check_weno_coef':
            s = torch.tensor(s, dtype=torch.float, device = self.device)
            with torch.no_grad():
                action = self.actor_policy_net(s)

            action = action.cpu().numpy()
            if not deterministic: ### each action, each dimension add noise
                a_left, a_right = action[:, :4], action[:,4:]
                assert np.sum(np.abs(a_left[1:] - a_right[:-1])) < 1e-5
                # print('before noise, a_left[0] is: ', a_left[0])
                size = a_left.shape
                noise_left = self.action_noise.noise(size = size)
                noise_right = self.action_noise.noise(size = size) 
                a_left += noise_left
                a_right += noise_right
                a_left -= np.max(a_left)
                a_right -= np.max(a_right)
                a_left = np.exp(a_left) / np.sum(np.exp(a_left), axis = 1).reshape(-1,1)
                a_right = np.exp(a_right) / np.sum(np.exp(a_right), axis = 1).reshape(-1,1)
                # print('after noise, a_left[0] is: ', a_left[0])
                action = np.concatenate((a_left, a_right), axis = 1)
        else:
            action = self.weno_ceof(np.array(s))

        return action

    def weno_ceof(self, s):
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
        roe[:-1] = (s[:, 3] + s[:,2])
        # roe[-1] = (f[-1,4] - f[-1, 3]) / (s[-1, 4] - s[-1,3]) ### need to add one more roe speed for flux f_{num_x + 1 / 2}.
        roe[-1] = (s[-1, 4] + s[-1,3]) ### need to add one more roe speed for flux f_{num_x + 1 / 2}.

        ### we put all four possible stencils all together for computation, while in weno only 3 can have positive weight at the same time.
        coef = np.zeros((num + 1, 4)) 
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

        return coef

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
            self.action_noise.setnoise(explore_rate)
        

    