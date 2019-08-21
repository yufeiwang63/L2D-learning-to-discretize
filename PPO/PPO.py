import torch
from torch.optim import Adam, RMSprop
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PPO.torch_networks import PPO_policy_net, PPO_value_net, deep_PPO_policy_net, deep_PPO_value_net, FCONV_PPO_value_net, FCONV_PPO_policy_net
import random


'''
Old PPO model. 
To load ppo models in the old directories such as in 'Burgers-2019-03-25-23-52-42', need to use this ppo model.
'''

class PPO():
    '''
    doc
    '''
    def __init__(self, args):

        np.random.seed(args.np_rng_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.torch_rng_seed)
        random.seed(args.py_rng_seed)

        self.args = args
        self.action_dim, self.state_dim  = args.action_dim, args.state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PPO_policy_net(self.state_dim, self.action_dim).to(self.device)
        self.value_net = PPO_value_net(self.state_dim).to(self.device)
        if args.formulation == 'FCONV':
            self.policy_net = FCONV_PPO_policy_net(args.window_size, self.action_dim).to(self.device)
            self.value_net = FCONV_PPO_value_net(args.window_size).to(self.device)
        if args.large_scale_train:
            self.policy_net = deep_PPO_policy_net(self.state_dim, self.action_dim).to(self.device)
            self.value_net = deep_PPO_value_net(self.state_dim).to(self.device)
        self.policy_net.apply(self._weight_init)
        self.value_net.apply(self._weight_init)
        self.gamma = self.args.gamma
        self.train_batch_size = args.batch_size
        self.eps = args.ppo_eps
        self.entropy_coef = args.entropy_coef

        optimizer = RMSprop if self.args.optimizer == 'rmsprop' else Adam
        self.a_optimizer = optimizer(self.policy_net.parameters(), self.args.a_lr)
        self.c_optimizer = optimizer(self.value_net.parameters(), self.args.c_lr)
        self.global_step = 0

    def data_process_MLP(self, dataset):
        obj_data = []
        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            # print('num_x: ', num_x)
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                # print('length: ', length)
                # exit()
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


        for _ in range(self.args.ppo_train_epoch):
            if not self.args.sample_all:
                train_batch_ = random.sample(obj_data, 30)
            else:
                train_batch_ = obj_data
            train_batch = []
            for x in train_batch_:
                for y in x:
                    train_batch.append(y)
            # if self.args.debug:
            #     print('len of train_batch: ', len(train_batch))
            #     exit()
            state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device)
            action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.long, device = self.device).unsqueeze(1)
            new_prob, new_log_prob = self.policy_net.get_action_prob(state_batch, action_batch) ###### note shape
            old_prob = torch.tensor([x[2] for x in train_batch], dtype = torch.float, device = self.device).unsqueeze(1) ###### note shape
            advantages = torch.tensor([x[3] for x in train_batch], dtype = torch.float, device = self.device).unsqueeze(1) ###### note shape
            Return = torch.tensor([x[4] for x in train_batch], dtype = torch.float, device = self.device).unsqueeze(1)

            ### PPO objective
            if self.args.ppo_obj == 'ppo':
                r = new_prob / old_prob  
                obj1 = r * advantages
                obj2 = r.clamp(1 - self.eps, 1 + self.eps) * advantages
                obj = torch.min(obj1, obj2) ###### the usage of min
                obj = obj.mean()
                loss = -obj

                # print('loss is: ', loss)
                # entropy = -new_log_prob * new_prob
                # entropy = entropy.sum(dim = 1)
                # entropy = entropy.mean()
                # print('entropy is: ', entropy)
                # loss -= self.entropy_coef * entropy

            elif self.args.ppo_obj == 'pg':
            ## Policy gradient objective
                obj = new_log_prob * advantages
                loss = -obj.mean()

            self.a_optimizer.zero_grad()
            loss.backward()
            self.a_optimizer.step()

            # if self.args.ppo_obj == 'ppo':
            for i in range(self.args.ppo_value_train_iter):
                state_values = self.value_net(state_batch)
                closs = (Return - state_values) ** 2
                closs = closs.mean()
                self.c_optimizer.zero_grad()
                closs.backward()
                self.c_optimizer.step()

    
    def action(self, s, test = False):
        s = torch.tensor(s, dtype=torch.float, device = self.device) 
        with torch.no_grad():
            action_probs, _ = self.policy_net(s)
            action_probs = action_probs.detach().cpu().numpy()
        if self.args.formulation == 'MLP':
            num = action_probs.shape[0]
            if test:
                actions = np.argmax(action_probs, axis = 1)
                return actions
            else:
                actions = [np.random.choice(self.action_dim, p = action_probs[i]) for i in range(num)]
                probs = action_probs[np.arange(len(action_probs)), actions]
                return actions, probs
        elif self.args.formulation == 'FCONV': ### action: 1 x |A| x s
            if test:
                action_probs = action_probs[0].T
                actions = np.argmax(action_probs, axis =1 )
                return actions
            else:
                action_probs = action_probs[0].T
                # if self.args.debug:
                #     print('In PPO action func, action_probs shape is: ', action_probs.shape)
                actions = [np.random.choice(self.action_dim, p = action_prob) for action_prob in action_probs]
                probs = action_probs[np.arange(len(action_probs)), actions]
                # prob = np.exp(np.sum(np.log(probs)))
                # if self.args.debug:
                #     print('In PPO action func, final actions are: ', actions)
                #     print('In PPO action func, final action prob is: ', prob)
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

