import torch
from torch.optim import Adam, RMSprop
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random, time
from Agent.torch_networks import PPO_policy_net, PPO_value_net, deep_PPO_policy_net, deep_PPO_value_net, FCONV_PPO_policy_net, FCONV_PPO_value_net
from Agent.torch_networks import PPO_continuous_policy_net, New_PPO_value_net, New_PPO_policy_net
from Agent.torch_networks import Constrained_flux_PPO_policy_net


class PPO():
    '''
    doc
    '''
    def __init__(self, args, tb_writter = None):

        np.random.seed(args.np_rng_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.torch_rng_seed)
        random.seed(args.py_rng_seed)

        self.args, self.tb_writter = args, tb_writter
        self.action_dim, self.state_dim  = args.action_dim, args.state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hidden_layers = [64, 64, 64, 64] if args.large_scale_train else [64, 64, 64]
        hidden_layers = [64 for _ in range(args.hidden_layer_num)]
        if args.mode != 'constrained_flux':
            self.value_net = New_PPO_value_net(self.state_dim, hidden_layers).to(self.device)
        else:
            self.value_net = New_PPO_value_net(self.state_dim + 1, hidden_layers).to(self.device)

        if args.mode == 'eno':
            self.policy_net = New_PPO_policy_net(self.state_dim, self.action_dim, hidden_layers).to(self.device)
        elif args.mode == 'weno':
            self.policy_net = PPO_continuous_policy_net(self.state_dim, self.action_dim, self.device, hidden_layers).to(self.device)
        elif args.mode == 'compute_flux': ### action is two numerical flux
            self.policy_net = PPO_continuous_policy_net(self.state_dim, 2, self.device, hidden_layers).to(self.device)
        elif args.mode == 'continuous_filter': ### action is the 2,3,4 order of the filter
            self.policy_net = PPO_continuous_policy_net(self.state_dim, 3, self.device, hidden_layers).to(self.device)
        elif args.mode == 'constrained_flux':
            self.policy_net = Constrained_flux_PPO_policy_net(self.state_dim, hidden_layers, self.args.constrain).to(self.device)


        if args.formulation == 'FCONV':
            self.policy_net = FCONV_PPO_policy_net(args.window_size, self.action_dim).to(self.device)
            self.value_net = FCONV_PPO_value_net(args.window_size).to(self.device)
        
        self.policy_net.apply(self._weight_init)
        self.value_net.apply(self._weight_init)
        self.gamma = self.args.gamma
        self.train_batch_size = args.batch_size
        self.eps = args.ppo_eps
        self.entropy_coef = args.entropy_coef

        optimizer = RMSprop if self.args.optimizer == 'rmsprop' else Adam
        self.a_optimizer = optimizer(self.policy_net.parameters(), self.args.a_lr)
        self.a_scheduler = optim.lr_scheduler.LambdaLR(self.a_optimizer, lr_lambda=lambda epoch: 0.999 ** epoch)
        self.c_optimizer = optimizer(self.value_net.parameters(), self.args.c_lr)
        self.c_scheduler = optim.lr_scheduler.LambdaLR(self.c_optimizer, lr_lambda=lambda epoch: 0.999 ** epoch)
        self.global_step = 0

    def data_process_MLP(self, dataset):
        obj_data = []
        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            # print('num_x: ', num_x)
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                states = np.array([tmp_dataset[j * 4] for j in range(length // 4)])
                tensor_states = torch.tensor(states, dtype = torch.float, device = self.device)
                with torch.no_grad():
                    state_values = self.value_net(tensor_states).detach()
                state_values = state_values.cpu().numpy()
                rewards = np.array([tmp_dataset[j * 4 + 3] for j in range(length // 4 - 1)])[:, np.newaxis]
                td_errors = state_values[1:] * self.gamma + rewards - state_values[:-1] 

                advantages = np.zeros(len(td_errors))
                sum = 0
                for t in range(len(td_errors)-1, -1, -1):
                    sum = sum * self.gamma * self.args.ppo_lambda
                    sum += td_errors[t]
                    advantages[t] = sum

                returns = np.zeros(length // 4 - 1)
                sum = 0
                for t in range(len(returns) - 1, -1, -1):
                    sum = sum * self.gamma
                    sum += rewards[t]
                    returns[t] = sum

                tmpobjdata = []
                for t in range(length // 4 - 1):
                    tmpobjdata.append([states[t], tmp_dataset[t * 4 + 1], tmp_dataset[t * 4 + 2], advantages[t], returns[t]])

                ### obj_data shape: n * [[s,a,r,s'] * evolving step]
                obj_data.append(tmpobjdata)

        return obj_data

    def data_process_FCONV(self, dataset):
        obj_data = []
        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                states = np.array([tmp_dataset[j * 4] for j in range(length // 4)])
                tensor_states = torch.tensor(states, dtype = torch.float, device = self.device).squeeze()
                with torch.no_grad():
                    state_values = self.value_net(tensor_states).detach()
                state_values = state_values.cpu().numpy()
                rewards = np.array([tmp_dataset[j * 4 + 3] for j in range(length // 4 - 1)])[:, np.newaxis]
                td_errors = state_values[1:] * self.gamma + rewards - state_values[:-1] 

                advantages = np.zeros(len(td_errors))
                sum = 0
                for t in range(len(td_errors)-1, -1, -1):
                    sum = sum * self.gamma * self.args.ppo_lambda
                    sum += td_errors[t]
                    advantages[t] = sum

                returns = np.zeros(length // 4 - 1)
                sum = 0
                for t in range(len(returns) - 1, -1, -1):
                    sum = sum * self.gamma
                    sum += rewards[t]
                    returns[t] = sum

                for t in range(length // 4 - 1):
                    obj_data.append([states[t], tmp_dataset[t * 4 + 1], tmp_dataset[t * 4 + 2], advantages[t], returns[t]])

        return obj_data


    def train(self, dataset):
        self.global_step += 1

        ### compute advantage and construct obj data
        if self.args.formulation == 'MLP':
            obj_data = self.data_process_MLP(dataset)
        elif self.args.formulation == 'FCONV':
            obj_data = self.data_process_FCONV(dataset) 

        tb_value_loss = 0
        tb_policy_loss = 0
        for _ in range(self.args.ppo_train_epoch):
            if not self.args.sample_all:
                train_batch_ = random.sample(obj_data, 30)
            else:
                train_batch_ = obj_data
            train_batch = []
            for x in train_batch_:
                for y in x:
                    train_batch.append(y)
          
            beg = time.time()
            state_batch = np.array([x[0] for x in train_batch])
            action_batch = np.array([x[1] for x in train_batch])
            old_log_prob = np.array([x[2] for x in train_batch])
            advantages = np.array([x[3] for x in train_batch])
            Return = np.array([x[4] for x in train_batch])
            state_batch = torch.tensor(state_batch, dtype=torch.float, device = self.device)
            action_dtype = torch.long if self.args.mode == 'eno' else torch.float
            action_batch = torch.tensor(action_batch, dtype = action_dtype, device = self.device)
            if self.args.mode == 'eno':
                action_batch = action_batch.unsqueeze(1)
            new_log_prob = self.policy_net.get_action_log_prob(state_batch, action_batch) ###### note shape
            old_log_prob = torch.tensor(old_log_prob, dtype = torch.float, device = self.device).unsqueeze(1) ###### note shape
            advantages = torch.tensor(advantages, dtype = torch.float, device = self.device).unsqueeze(1) ###### note shape
            Return = torch.tensor(Return, dtype = torch.float, device = self.device).unsqueeze(1)
            # print('state/action/old_log_prob to tensor cost time: ', time.time() - beg)

            ### PPO 
            beg = time.time()
            if self.args.ppo_obj == 'ppo':
                r = torch.exp(new_log_prob - old_log_prob)  
                obj1 = r * advantages
                obj2 = r.clamp(1 - self.eps, 1 + self.eps) * advantages
                obj = torch.min(obj1, obj2) ###### the usage of min
                obj = obj.mean()
                loss = -obj
                tb_policy_loss += loss.detach().cpu().item()

                # print('loss is: ', loss)
                # entropy = -new_log_prob * new_prob
                # entropy = entropy.sum(dim = 1)
                # entropy = entropy.mean()
                # print('entropy is: ', entropy)
                # loss -= self.entropy_coef * entropy
                # print('policy update cost time: ', time.time() - beg)
            
            elif self.args.ppo_obj == 'pg':
            ## Policy gradient objective
                obj = new_log_prob * advantages
                loss = -obj.mean()

            # self.a_scheduler.step()
            self.a_optimizer.zero_grad()
            loss.backward()
            self.a_optimizer.step()

            # if self.args.ppo_obj == 'ppo':
            # self.c_scheduler.step()
            beg = time.time()
            for i in range(self.args.ppo_value_train_iter):
                state_values = self.value_net(state_batch)
                closs = (Return - state_values) ** 2
                closs = closs.mean()
                tb_value_loss += closs.detach().cpu().item()
                self.c_optimizer.zero_grad()
                closs.backward()
                self.c_optimizer.step()
            # print('value update cost time: ', time.time() - beg)

        tb_value_loss /= self.args.ppo_train_epoch * self.args.ppo_value_train_iter
        tb_policy_loss /= self.args.ppo_train_epoch
        if self.tb_writter is not None:
            self.tb_writter.add_scalar('value_loss', tb_value_loss, self.global_step)
            self.tb_writter.add_scalar('policy_loss', tb_policy_loss, self.global_step)

        if self.args.mode == 'constrained_flux' and self.args.constrain:
            violence = self.policy_net.check_constraint()
            if self.tb_writter is not None:
                self.tb_writter.add_scalar('constrain violence', violence.item(), self.global_step)

        if self.args.mode == 'constrained_flux':
            sigma = self.policy_net.sigma.detach().cpu().item()
            if self.tb_writter is not None:
                self.tb_writter.add_scalar('sigma', sigma, self.global_step)

        elif self.args.mode == 'continuous_filter':
            sigma = self.policy_net.sigma.detach().cpu().numpy()
            sigma = sigma[0]
            if self.tb_writter is not None:
                self.tb_writter.add_scalars('sigmas', {
                    'first  sigma': sigma[0].item(), 
                    'second  sigma': sigma[1].item(), 
                    'third sigma': sigma[2].item(),  
                }, self.global_step)

        elif self.args.mode == 'compute_flux':
            sigma = self.policy_net.sigma.detach().cpu().numpy()
            sigma = sigma[0]
            if self.tb_writter is not None:
                self.tb_writter.add_scalars('sigmas', {
                    'first  sigma': sigma[0].item(), 
                    'second  sigma': sigma[1].item(), 
                }, self.global_step)

    
    def action(self, s, test = False):
        beg = time.time()
        s = torch.tensor(s, dtype=torch.float, device = self.device) 
        if self.args.formulation == 'MLP':
            res =  self.policy_net.act(s, test)
            # print('act cost time: ', time.time() - beg)
            return res
        elif self.args.formulation == 'FCONV': ### action: 1 x |A| x s
            if test:
                action_probs = action_probs[0].T
                actions = np.argmax(action_probs, axis =1 )
                return actions
            else:
                action_probs = action_probs[0].T
                actions = [np.random.choice(self.action_dim, p = action_prob) for action_prob in action_probs]
                probs = action_probs[np.arange(len(action_probs)), actions]
                return actions, probs

            

    def save(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path + 'ppo_policy.txt')
        torch.save(self.value_net.state_dict(), save_path + 'ppo_value.txt')

    def load(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path + 'ppo_policy.txt'))
        self.value_net.load_state_dict(torch.load(load_path + 'ppo_value.txt'))

    def adjust_learning_rate(self, optimizer, decay_rate=.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def _weight_init(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0.01)

