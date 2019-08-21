import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from DDPG.util import SlidingMemory_new, update_linear_schedule, SlidingMemory_batch, GaussNoise
from DDPG.torch_networks import DDPG_actor_network, DDPG_critic_network, constrained_flux_DDPG_policy_net, weno_coef_DDPG_policy_net
from DDPG.torch_networks import nonlinear_weno_coef_DDPG_policy_net, weno_coef_DDPG_policy_net_fs

class DDPG():    
    '''
    doc for ddpg
    '''
    def __init__(self, args, noise, tb_logger = None):

        np.random.seed(args.np_rng_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.torch_rng_seed)
        random.seed(args.py_rng_seed)


        self.args, self.tb_logger = args, tb_logger
        self.mem_size, self.train_batch_size = args.replay_size, args.batch_size
        self.gamma, self.actor_lr, self.critic_lr = args.gamma, args.a_lr, args.c_lr
        self.global_step = 0
        self.tau = args.tau
        self.action_noise = noise
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.action_high, self.action_low = args.action_high, args.action_low
        if args.mem_type == 'multistep_return':
            self.replay_mem = SlidingMemory_new(args.state_dim, args.action_dim, self.mem_size)
        elif args.mem_type == 'batch_mem':
            self.replay_mem = SlidingMemory_batch(self.mem_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        hidden_layers = [64 for _ in range(self.args.hidden_layer_num)]
        critic_hidden_layers = [64 for _ in range(self.args.hidden_layer_num + 1)]

        if args.mode == 'weno_coef' or args.mode == 'weno_coef_four':
            if args.ddpg_net == 'roe':
                self.actor_policy_net = weno_coef_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers, self.args.flux).to(self.device)
                self.actor_target_net = weno_coef_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers, self.args.flux).to(self.device)
            elif args.ddpg_net == 'fu':
                self.actor_policy_net = weno_coef_DDPG_policy_net_fs(self.state_dim, self.action_dim, hidden_layers).to(self.device)
                self.actor_target_net = weno_coef_DDPG_policy_net_fs(self.state_dim, self.action_dim, hidden_layers).to(self.device)
        elif args.mode == 'nonlinear_weno_coef':
            print('ddpg enters nonlinear weno coef')
            self.actor_policy_net = nonlinear_weno_coef_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers).to(self.device)
            self.actor_target_net = nonlinear_weno_coef_DDPG_policy_net(self.state_dim, self.action_dim, hidden_layers).to(self.device)
        self.critic_policy_net = DDPG_critic_network(self.state_dim, self.action_dim, critic_hidden_layers).to(self.device)
        self.critic_target_net = DDPG_critic_network(self.state_dim, self.action_dim, critic_hidden_layers).to(self.device)
        self.critic_policy_net_2 = DDPG_critic_network(self.state_dim, self.action_dim, critic_hidden_layers).to(self.device) ### twin
        self.critic_target_net_2 = DDPG_critic_network(self.state_dim, self.action_dim, critic_hidden_layers).to(self.device) ### twin
    
        self.critic_policy_net.apply(self._weight_init)
        self.critic_policy_net_2.apply(self._weight_init)
        self.actor_policy_net.apply(self._weight_init)

        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_policy_net_2.parameters(), self.critic_lr) ### twin
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
        self.hard_update(self.critic_target_net_2, self.critic_policy_net_2) ### twin

        self.upwindnoise = None
        if not args.handjudge_upwind:
            self.upwindnoise = GaussNoise(self.args.upwindnoisebeg, self.args.upwindnoiseend, args.upwindnoisedec)
    
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
            ### this is coupled with using multiple-step return
            if self.args.mem_type == 'multistep_return':
                prev_state_batch, action_batch, target_batch = self.replay_mem.sample(self.train_batch_size)
                
                pre_state_batch = torch.tensor(prev_state_batch, dtype=torch.float, device = self.device) 
                action_batch = torch.tensor(action_batch, dtype = torch.float, device = self.device) 
                target_batch = torch.tensor(target_batch, dtype=torch.float, device = self.device)

            ### this is coupled with storing and sample a whole row as batch
            elif self.args.mem_type == 'batch_mem':
                batch = self.replay_mem.sample(self.train_batch_size)
                pre_state_batch = []
                action_batch = []
                reward_batch = []
                next_state_batch = []
                done_batch = []
                for row in batch:
                    for pair in row:
                        pre_state_batch.append(pair[0])
                        action_batch.append(pair[1])
                        reward_batch.append(pair[2])
                        next_state_batch.append(pair[3])
                        done_batch.append(pair[4])

                pre_state_batch = np.array(pre_state_batch)
                action_batch = np.array(action_batch)
                reward_batch = np.array(reward_batch)
                next_state_batch = np.array(next_state_batch)
                done_batch = np.array(done_batch).astype(float)

                pre_state_batch = torch.tensor(pre_state_batch, device = self.device, dtype = torch.float)
                action_batch = torch.tensor(action_batch, device = self.device, dtype = torch.float)
                reward_batch = torch.tensor(reward_batch, device = self.device, dtype = torch.float).unsqueeze(1)
                next_state_batch = torch.tensor(next_state_batch, device = self.device, dtype = torch.float)
                done_batch = torch.tensor(done_batch, device = self.device, dtype = torch.float).unsqueeze(1)

                with torch.no_grad():
                    next_state_action_batch = self.actor_target_net(next_state_batch)
                    target_Q_1 = self.critic_target_net(next_state_batch, next_state_action_batch)
                    target_Q_2 = self.critic_target_net_2(next_state_batch, next_state_action_batch)
                    target_Q = torch.min(target_Q_1, target_Q_2)
                    target_batch = self.gamma * target_Q * (1 - done_batch) + reward_batch

            ### use the target_Q_network to get the target_Q_value
            for i in range(self.args.ddpg_value_train_iter):
                q_pred = self.critic_policy_net(pre_state_batch, action_batch)
                closs = (q_pred - target_batch) ** 2             
                closs = torch.mean(closs)
                tb_value_loss += closs.cpu().item()
                self.critic_optimizer.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                ### twin
                q_pred = self.critic_policy_net_2(pre_state_batch, action_batch)
                closs = (q_pred - target_batch) ** 2             
                closs = torch.mean(closs)
                self.critic_optimizer_2.zero_grad()
                closs.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_policy_net_2.parameters(), self.args.max_grad_norm)
                self.critic_optimizer_2.step()

            ### mimic weno actions
            if self.args.supervise_weno:
                weno_actions = np.zeros((len(prev_state_batch), 8))
                for idx, s in enumerate(prev_state_batch):
                    weno_actions[idx] = self.weno_ceof(s[None, :])    
                weno_actions_tensor = torch.tensor(weno_actions, dtype = torch.float, device = self.device)
                ddpg_actions = self.actor_policy_net(pre_state_batch)
                supervise_weno_loss = (weno_actions_tensor - ddpg_actions) ** 2
                supervise_weno_loss = supervise_weno_loss.mean()

            aloss = -self.critic_policy_net(pre_state_batch, self.actor_policy_net(pre_state_batch))
            aloss = aloss.mean()
            if self.args.supervise_weno:
                aloss += supervise_weno_loss
            self.actor_optimizer.zero_grad()
            aloss.backward()
            tb_policy_loss += aloss.cpu().item()
            torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(), self.args.max_grad_norm)
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

    def perceive(self, dataset):
        if self.args.mem_type == 'batch_mem':
            self.perceive_batch(dataset)
        elif self.args.mem_type == 'multistep_return':
            self.perceive_multistep(dataset)

    def perceive_batch(self, dataset):
        for env_dataset in dataset:
            for t in range(len(env_dataset[0])): ### horizen
                self.replay_mem.add([env_dataset[_][t] for _ in range(len(env_dataset))])

    ### store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive_multistep(self, dataset):
        ### new replay mem, store (s, a, q-target)
        prev_state_dataset, action_dataset, reward_dataset, next_state_dataset, done_dataset = dataset
        horizen = prev_state_dataset.shape[-2]
        state_size = prev_state_dataset.shape[-1]
        action_size = action_dataset.shape[-1]
        prev_state_dataset = prev_state_dataset.reshape((-1, horizen, state_size))
        action_dataset = action_dataset.reshape((-1, horizen, action_size))
        reward_dataset = reward_dataset.reshape((-1, horizen))
        next_state_dataset = next_state_dataset.reshape((-1, horizen, state_size))
        done_dataset = done_dataset.reshape((-1, horizen))

        next_state_tensor = torch.tensor(next_state_dataset.reshape((-1, state_size)), device = self.device, dtype = torch.float)
        with torch.no_grad():
            target_next_action = self.actor_target_net(next_state_tensor)
            q_target_ = self.critic_target_net(next_state_tensor, target_next_action)
            q_target_2 = self.critic_target_net_2(next_state_tensor, target_next_action) ### twin
            q_target_ = torch.min(q_target_, q_target_2) ### twin
            # q_target_old = q_target_.cpu().numpy()
            q_target = q_target_.cpu().numpy()

        q_target = q_target.reshape((-1, horizen))
        q_target = q_target * (1 - done_dataset)

        traj_num = len(reward_dataset)
        target = np.zeros((traj_num, horizen))
        multistep = self.args.multistep_return

        for idx in range(horizen):
            R = q_target[:, min(idx + multistep - 1, horizen - 1)].copy()
            for idx2 in reversed(range(idx, min(idx + multistep, horizen))):
                R *= self.gamma
                R += reward_dataset[:, idx2]
            target[:, idx] = R

        prev_state_batch = prev_state_dataset.reshape((-1, state_size))
        action_batch = action_dataset.reshape((-1, action_size))
        target_batch = target.reshape((-1,1))

        self.replay_mem.add(prev_state_batch, action_batch, target_batch)
        
    
    ### use the action_policy_net to compute the action
    def action(self, s, deterministic = False, mode = 'agent'):
        if mode == 'agent':
            s = torch.tensor(s, dtype=torch.float, device = self.device)
            with torch.no_grad():
                action = self.actor_policy_net(s)

            action = action.cpu().numpy()
            action_num = self.action_dim // 2
            if not deterministic: ### each action, each dimension add noise
                a_left, a_right = action[:, :action_num], action[:,action_num:]
                if self.args.mode == 'weno_coef' and self.args.handjudge_upwind or self.args.mode == 'weno_coef_four':
                    assert np.sum(np.abs(a_left[1:] - a_right[:-1])) < 1e-5
                    # print('before noise, a_left[0] is: ', a_left[0])
                    size = a_left.shape
                    noise_left = self.action_noise.noise(size = size)
                    noise_right = self.action_noise.noise(size = size) 
                    a_left += noise_left
                    a_right += noise_right
                    # a_left -= np.max(a_left)
                    # a_right -= np.max(a_right)
                    # a_left = np.exp(a_left) / np.sum(np.exp(a_left), axis = 1).reshape(-1,1)
                    # a_right = np.exp(a_right) / np.sum(np.exp(a_right), axis = 1).reshape(-1,1)
                    # print('after noise, a_left[0] is: ', a_left[0])
                    a_left = np.clip(a = a_left, a_min = 1e-20, a_max = None)
                    a_right = np.clip(a = a_right, a_min = 1e-20, a_max = None)
                    a_left = a_left / np.sum(a_left, axis = 1).reshape(-1,1)
                    a_right = a_right / np.sum(a_right, axis = 1).reshape(-1,1)
                    action = np.concatenate((a_left, a_right), axis = 1)
                elif self.args.mode == 'weno_coef' and not self.args.handjudge_upwind:
                    num = a_left.shape[0]
                    noise_left = self.action_noise.noise(size = (num, 3))
                    noise_right = self.action_noise.noise(size = (num, 3)) 
                    dir_noise_left = self.upwindnoise.noise(size = (num, 1))
                    dir_noise_right = self.upwindnoise.noise(size = (num, 1))

                    weight_left = a_left[:, :3]
                    weight_right = a_right[:, :3]
                    dir_left = a_left[:, 3].reshape(-1,1)
                    dir_right = a_right[:, 3].reshape(-1,1)
                    
                    weight_left += noise_left
                    weight_right += noise_right
                    weight_left = np.clip(a = weight_left, a_min = 1e-20, a_max = None)
                    weight_right = np.clip(a = weight_right, a_min = 1e-20, a_max = None)
                    weight_left = weight_left / np.sum(weight_left, axis = 1).reshape(-1,1)
                    weight_right = weight_right / np.sum(weight_right, axis = 1).reshape(-1,1)
                    
                    dir_left += dir_noise_left
                    dir_right += dir_noise_right

                    action = np.concatenate((weight_left, dir_left, weight_right, dir_right), axis = 1)
                elif self.args.mode == 'nonlinear_weno_coef':
                    assert np.sum(np.abs(a_left[1:] - a_right[:-1])) < 1e-5
                    size = a_left.shape
                    noise_left = self.action_noise.noise(size = size)
                    noise_right = self.action_noise.noise(size = size) 
                    a_left += noise_left
                    a_right += noise_right
                    action = np.concatenate((a_left, a_right), axis = 1)

        elif mode == 'weno':
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
                # coef[i][0] = wr2[i]
                # coef[i][1] = wr1[i]
                # coef[i][2] = wr0[i]

        action_left = coef[:-1]
        action_right = coef[1:]
        action = np.concatenate((action_left, action_right), axis = 1)
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
            self.action_noise.setnoise(explore_rate)
        

    