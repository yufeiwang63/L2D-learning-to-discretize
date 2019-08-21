import numpy as np
import scipy.optimize as soptim
import time
import torch
import torch.optim as optim
from PPO.torch_networks import New_PPO_value_net

class PPO():
    '''
    PPO with linear constraint on the neural network weight.
    Optimized using scipy.
    '''
    def __init__(self, args, tb_writter = None):
        
        np.random.seed(args.np_rng_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.torch_rng_seed)
        
        self.args, self.tb_writter  = args, tb_writter
        self.action_dim, self.state_dim  = args.action_dim, args.state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = self.args.gamma
        self.eps = args.ppo_eps
        self.w1 = np.random.rand(self.state_dim, args.flux_net_hidden_size)
        self.w2 = np.random.rand(args.flux_net_hidden_size, 1)
        self.w1 = np.exp(self.w1) / np.sum(np.exp(self.w1), axis = 0).reshape(1, args.flux_net_hidden_size)
        self.w2 = np.exp(self.w2) / np.sum(np.exp(self.w2))

        n, d = self.state_dim, args.flux_net_hidden_size
        self.linear_constraint_matrix = np.zeros((d + 1, n * d + d))
        for i in range(d):
            self.linear_constraint_matrix[i][i * n: (i+1) * n] = 1
        self.linear_constraint_matrix[-1][-d:] = 1
        righthand_b = np.ones(d + 1)
        self.linear_constraint = soptim.LinearConstraint(self.linear_constraint_matrix, righthand_b, righthand_b)
        self.w1_num = n * d
        self.w2_num = d
        self.w1_shape = (n, d)
        self.w2_shape = (d, 1)
        self.sigma = self.args.sigma
    
        hidden_layers = [64 for _ in range(args.hidden_layer_num)]
        self.value_net = New_PPO_value_net(2 * (self.state_dim + 1), hidden_layers).to(self.device)

        self.c_optimizer = optim.RMSprop(self.value_net.parameters(), self.args.c_lr)
        self.c_scheduler = optim.lr_scheduler.LambdaLR(self.c_optimizer, lr_lambda=lambda epoch: 0.999 ** epoch)
        self.train_times = 0
        
    def expand_value_state(self, state_batch):
        flux_state_batch = state_batch ** 2 / 2.
        value_state_batch = np.concatenate((state_batch, flux_state_batch), axis = -1)
        return value_state_batch

    def data_process_MLP(self, dataset):
        action_dataset = []
        state_dataset = []
        old_log_prob_dataset = []
        ret_dataset = []
        adv_dataset = []

        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            # print('num_x: ', num_x)
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                states = np.array([tmp_dataset[j * 4] for j in range(length // 4)])
                rewards = np.array([tmp_dataset[j * 4 + 3] for j in range(length // 4 - 1)])
                actions = np.array([tmp_dataset[j * 4 + 1] for j in range(length // 4 - 1)])
                old_log_probs = np.array([tmp_dataset[j * 4 + 2] for j in range(length // 4 - 1)])
                
                state_dataset.append(states[:-1])
                action_dataset.append(actions)
                old_log_prob_dataset.append(old_log_probs)
                
                returns = np.zeros(length // 4 - 1)
                sum = 0
                for t in range(len(returns) - 1, -1, -1):
                    sum = sum * self.gamma
                    sum += rewards[t]
                    returns[t] = sum
                ret_dataset.append(returns)

        tb_value_loss = 0
        state_batch = np.concatenate(state_dataset)
        return_batch = np.concatenate(ret_dataset)
        value_state_batch = self.expand_value_state(state_batch)
        state_batch_tensor = torch.tensor(value_state_batch, dtype = torch.float, device = self.device)
        Return = torch.tensor(return_batch, dtype = torch.float, device = self.device).unsqueeze(1)
        for i in range(self.args.constrained_ppo_value_train_iter):
            state_values = self.value_net(state_batch_tensor)
            closs = (Return - state_values) ** 2
            closs = closs.mean()
            tb_value_loss += closs.detach().cpu().item()
            self.c_optimizer.zero_grad()
            closs.backward()
            self.c_optimizer.step()

        tb_value_loss /= self.args.constrained_ppo_value_train_iter
        if self.tb_writter is not None:
            self.tb_writter.add_scalar('value_loss', tb_value_loss, self.train_times)

        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            # print('num_x: ', num_x)
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                states = np.array([tmp_dataset[j * 4] for j in range(length // 4)])
                rewards = np.array([tmp_dataset[j * 4 + 3] for j in range(length // 4 - 1)])

                value_states = self.expand_value_state(states)
                tensor_states = torch.tensor(value_states, device = self.device, dtype = torch.float)
                with torch.no_grad():
                    state_values = self.value_net(tensor_states).detach()
                state_values = state_values.cpu().squeeze().numpy()
                state_values[-1] = 0
                td_errors = state_values[1:] * self.gamma + rewards - state_values[:-1] 
                advantages = np.zeros((len(td_errors), 1))
                sum = 0
                for t in range(len(td_errors)-1, -1, -1):
                    sum = sum * self.gamma * self.args.ppo_lambda
                    sum += td_errors[t]
                    advantages[t] = sum

                adv_dataset.append(advantages)

        action_batch = np.concatenate(action_dataset)
        old_log_prob_batch = np.concatenate(old_log_prob_dataset)
        adv_batch = np.concatenate(adv_dataset)
                
        return state_batch, action_batch, old_log_prob_batch, adv_batch, return_batch

    def data_process_MLP_old(self, dataset):
        obj_data = []
        for idx in range(self.args.ppo_agent):
            num_x = len(dataset[idx])
            # print('num_x: ', num_x)
            for idx2 in range(num_x):
                tmp_dataset = dataset[idx][idx2]
                length = len(tmp_dataset)
                states = np.array([tmp_dataset[j * 4] for j in range(length // 4)])
                value_states = self.expand_value_state(states)
                tensor_states = torch.tensor(value_states, dtype = torch.float, device = self.device)
                with torch.no_grad():
                    state_values = self.value_net(tensor_states).detach()
                state_values = state_values.cpu().squeeze().numpy()
                print('in data process old, state_values shape is:', state_values.shape)
                rewards = np.array([tmp_dataset[j * 4 + 3] for j in range(length // 4 - 1)])
                print('in data process old, rewards shape is: ', rewards.shape)
                td_errors = state_values[1:] * self.gamma + rewards - state_values[:-1] 
                print('in data process old, td_errors shape is: ', td_errors.shape)


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

    def train(self, dataset):
        '''
        doc
        '''
        self.train_times += 1
        state_batch, action_batch, old_log_prob, advantages, ret_batch  = self.data_process_MLP(dataset)
       
        # obj_data = self.data_process_MLP_old(dataset)
        # train_batch_ = obj_data
        # train_batch = []
        # for x in train_batch_:
        #     for y in x:
        #         train_batch.append(y)
        
        # state_batch_old = np.array([x[0] for x in train_batch])
        # action_batch_old = np.array([x[1] for x in train_batch])
        # old_log_prob_old = np.array([x[2] for x in train_batch])
        # advantages_old = np.array([x[3] for x in train_batch]) ###### note shape
        # ret_old = np.array([x[4] for x in train_batch])
        # # print('ret_old shape is: ', ret_old.shape)
        # Return = torch.tensor([x[4] for x in train_batch], dtype = torch.float, device = self.device).unsqueeze(1)
        
        # print('state diff: ', np.sum(np.abs(state_batch - state_batch_old)))
        # print('action diff: ', np.sum(np.abs(action_batch - action_batch_old)))
        # print('old log prob diff: ', np.sum(np.abs(old_log_prob - old_log_prob_old)))
        # print('return diff: ', np.sum(np.abs(ret_batch - ret_old)))
        # exit()

        def ppo_obj(w, w1_num, state_dim, sigma, w1_shape, w2_shape, s_batch, a_batch, prev_log_prob_batch, adv_batch, eps, device):
            w1 = w[:w1_num]
            w2 = w[w1_num:]

            w1 = w1.reshape(w1_shape, order = 'F')
            w2 = w2.reshape(w2_shape, order = 'F')

            left_s_batch = s_batch[:, :state_dim]
            right_s_batch = s_batch[:, 1:]
            left_a_batch = a_batch[:, 0].reshape(-1,1)
            right_a_batch = a_batch[:, 1].reshape(-1,1)

            left_mean_batch = np.dot(0.5 * (np.dot(left_s_batch, w1)) ** 2, w2)
            right_mean_batch  = np.dot(0.5 * (np.dot(right_s_batch, w1)) ** 2, w2)
            left_new_log_prob_batch = -0.5 * np.log(2 * np.pi) - 0.5 * (left_a_batch - left_mean_batch) ** 2 / sigma ** 2 - np.log(sigma)
            right_new_log_prob_batch = -0.5 * np.log(2 * np.pi) - 0.5 * (right_a_batch - right_mean_batch) ** 2 / sigma ** 2 - np.log(sigma)
            new_log_prob_batch = left_new_log_prob_batch + right_new_log_prob_batch

            ratio = np.exp(new_log_prob_batch - prev_log_prob_batch)
            obj1 = ratio * adv_batch
            obj2 = np.clip(ratio, 1-eps, 1+eps) * adv_batch
            ppo_obj = -np.mean(np.minimum(obj1, obj2))

            return ppo_obj

        def ppo_obj_grad(w, w1_num, state_dim, sigma, w1_shape, w2_shape, s_batch, a_batch, prev_log_prob_batch, adv_batch, eps, device):
            ori_len = len(w)
            assert w1_num == w1_shape[0] * w1_shape[1]
            assert ori_len == w1_num + w2_shape[0] * w2_shape[1]
            w1 = w[:w1_num]
            w2 = w[w1_num:]

            w1 = w1.reshape(w1_shape, order = 'F')
            w2 = w2.reshape(w2_shape, order = 'F')
            w1 = torch.tensor(w1, dtype = torch.float, requires_grad = True, device = device)
            w2 = torch.tensor(w2, dtype = torch.float, requires_grad = True, device = device)

            left_s_batch = s_batch[:, :state_dim]
            right_s_batch = s_batch[:, 1:]
            left_a_batch = a_batch[:, 0].reshape(-1,1)
            right_a_batch = a_batch[:, 1].reshape(-1,1)

            left_s_batch = torch.tensor(left_s_batch, dtype = torch.float, device = device)
            right_s_batch = torch.tensor(right_s_batch, dtype = torch.float, device = device)
            left_a_batch = torch.tensor(left_a_batch, dtype = torch.float, device = device)
            right_a_batch = torch.tensor(right_a_batch, dtype = torch.float, device = device)
            prev_log_prob_batch = torch.tensor(prev_log_prob_batch, dtype = torch.float, device = device)
            adv_batch = torch.tensor(adv_batch, dtype = torch.float, device = device)

            sigma_tensor = torch.tensor(sigma, dtype = torch.float, device = device)
            pi_tensor = torch.tensor(-0.5 * np.log(2 * np.pi), dtype = torch.float, device = device)
            left_mean_batch = torch.mm(0.5 * (torch.mm(left_s_batch, w1)) ** 2, w2)
            right_mean_batch = torch.mm(0.5 * (torch.mm(right_s_batch, w1)) ** 2, w2)
            left_new_log_prob_batch = pi_tensor - 0.5 * (left_a_batch - left_mean_batch) ** 2 / (sigma_tensor ** 2) - torch.log(sigma_tensor)
            right_new_log_prob_batch = pi_tensor - 0.5 * (right_a_batch - right_mean_batch) ** 2 / (sigma_tensor ** 2) - torch.log(sigma_tensor)
            new_log_prob_batch = left_new_log_prob_batch + right_new_log_prob_batch

            ratio = torch.exp(new_log_prob_batch - prev_log_prob_batch)
            obj1 = ratio * adv_batch
            obj2 = torch.clamp(ratio, 1-eps, 1+eps) * adv_batch
            ppo_obj = -torch.min(obj1, obj2).mean()

            ppo_obj.backward()
            with torch.no_grad():
                w1_grad = w1.grad.cpu().numpy()
                w2_grad = w2.grad.cpu().numpy()
            w1_grad = w1_grad.reshape((-1,), order = 'F')
            w2_grad = w2_grad.reshape((-1,), order = 'F')
            grad = np.concatenate((w1_grad, w2_grad))
            assert len(grad) == ori_len
            return grad

        if self.train_times > self.args.value_pre_train_time:
            w1_ = self.w1.reshape((-1,), order = 'F')
            w2_ = self.w2.reshape((-1,), order = 'F')
            w0 = np.concatenate((w1_, w2_))
            # w, w1_num, w1_shape, w2_shape, s_batch, a_batch, prev_log_prob_batch, adv_batch, eps
            #  w, w1_num, state_dim, sigma, w1_shape, w2_shape, s_batch, a_batch, prev_log_prob_batch, adv_batch, eps, device):
            res = soptim.minimize(ppo_obj, w0, method='trust-constr',  jac=ppo_obj_grad, hess=soptim.BFGS(), constraints=[self.linear_constraint],
                args = (self.w1_num, self.state_dim, self.sigma, self.w1_shape, self.w2_shape, state_batch, action_batch, old_log_prob, advantages, self.eps, self.device), 
                options={'verbose': self.args.scipy_verbose})

            new_w = res.x
            self.w1 = new_w[:self.w1_num]
            self.w2 = new_w[self.w1_num:]
            self.w1 = self.w1.reshape(self.w1_shape, order = 'F')
            self.w2 = self.w2.reshape(self.w2_shape, order = 'F')

        # tb_value_loss = 0
        # value_states_old = self.expand_value_state(state_batch_old)
        # state_batch_tensor = torch.tensor(value_states_old, dtype = torch.float, device = self.device)
        # for i in range(self.args.constrained_ppo_value_train_iter):
        #     state_values = self.value_net(state_batch_tensor)
        #     closs = (Return - state_values) ** 2
        #     closs = closs.mean()
        #     tb_value_loss += closs.detach().cpu().item()
        #     self.c_optimizer.zero_grad()
        #     closs.backward()
        #     self.c_optimizer.step()

        # tb_value_loss /= self.args.constrained_ppo_value_train_iter
        # if self.tb_writter is not None:
        #     self.tb_writter.add_scalar('value_loss', tb_value_loss, self.train_times)

        constraint_violence = self.check_constraint()
        if self.tb_writter is not None:
            self.tb_writter.add_scalar('constraint_violence', constraint_violence, self.train_times)        

    def check_constraint(self):
        u = np.random.rand()
        tmp = u * np.ones((1, self.state_dim))
        mean = np.dot(0.5 * np.dot(tmp, self.w1) ** 2, self.w2)
        error = np.abs(0.5 * u ** 2 - mean)
        return error

    def action(self, s, test = False):
        num = s.shape[0]
        left_s = s[:, :self.state_dim]
        right_s = s[:, 1:] 
        left_mean = np.dot(0.5 * np.dot(left_s, self.w1) ** 2, self.w2)
        right_mean = np.dot(0.5 * np.dot(right_s, self.w1) ** 2, self.w2)
        if test:
            action = np.concatenate((left_mean, right_mean), axis = 1)
            return action
        else:
            left_sample = np.random.normal(size = (num, 1))
            right_sample = np.random.normal(size = (num, 1))
            left_flux = left_sample * self.sigma + left_mean
            right_flux = right_sample * self.sigma + right_mean
            left_log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * left_sample ** 2 - np.log(self.sigma)
            right_log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * right_sample ** 2 - np.log(self.sigma)
            action = np.concatenate((left_flux, right_flux), axis = 1)
            return action, left_log_prob + right_log_prob
        
    def save(self, save_path):
        torch.save(self.value_net.state_dict(), save_path + 'ppo_flux_value.txt')
        np.save(save_path + 'w1.npy', self.w1)
        np.save(save_path + 'w2.npy', self.w2)

    def load(self, load_path):
        self.value_net.load_state_dict(torch.load(load_path + 'ppo_flux_value.txt'))
        self.w1 = np.load(load_path + 'w1.npy')
        self.w2 = np.load(load_path + 'w2.npy')


