import numpy as np
import random
import math
import copy
import matplotlib
from matplotlib import pyplot as plt
from Weno.weno3_2 import Weno3
from Weno.finite_difference_weno import weno3_fd
from os import path as osp
import os
import torch
import gym
from gym.spaces import Box
from scipy import interpolate
from scripts.debug import debug
import time

class Burgers(gym.Env):
    """
    u_t + (1/2u^2))_x = 0
    u_t + u * u_x = 0
    PDE class for RL learning, having the same interface as Gym
    """

    def __init__(
        self,
        vv,
        agent=None
    ):
        '''
        parameters:
            args: a dictionary storing all kinds of arguments
            agent: only required for RK4 time scheme. 
        '''
        self.vv = copy.deepcopy(vv)
        self.reward_width = vv['reward_width']
        self.solution_data_path = vv['solution_data_path']
        solution_path_list = os.listdir(self.solution_data_path)
        self.solution_path_list = []
        for x in solution_path_list:
            if '.pkl' in x:
                self.solution_path_list.append(x)
        self.train_flag = True

        # agent for rk4
        self.agent = agent
        if self.vv['Tscheme'] == 'rk4':
            assert self.agent is not None, 'rk4 scheme is only used at test time. so you must provide '\
                'a trained RL agent!'

        # initial and boundary conditions
        self.boundary_condition = vv['boundary_condition']

        self.weno_w = np.array([[1/3, 0, 0, 0],
                                [-7/6, -1/6, 0, 0],
                                [11/6, 5/6, 1/3, 0],
                                [0, 1/3, 5/6, 11/6],
                                [0, 0, -1/6, -7/6],
                                [0, 0, 0, 1/3]])

        self.observation_space = Box(low=np.array([-np.inf] * 7), high=np.array([np.inf] * 7))
        self.action_space = Box(low=np.array([-1] * 4), high=np.array([1] * 4))

        self.eta = self.vv['eta'] # viscous term coefficient

    def compute_weno_error(self):
        error_euler = np.zeros(self.num_t - 1)
        error_rk4 = np.zeros(self.num_t - 1)
        for i in range(1, self.num_t):
            true_values = self.get_precise_value(i * self.dt)
            error_euler[i-1] = self.relative_error(true_values, self.weno_coarse_grid_euler[i])
            error_rk4[i-1] = self.relative_error(true_values, self.weno_coarse_grid_rk4[i])
        
        self.weno_error_euler = np.mean(error_euler)
        self.weno_error_rk4 = np.mean(error_rk4)
        self.weno_error_all_rk4 = error_rk4
        self.weno_error_all_euler = error_euler

    ### the burgers flux function. Can be changed to any other functions if needed for future useage.
    def flux(self, val):
        if self.vv['flux'] == 'u2':
            return val ** 2 / 2.
        elif self.vv['flux'] == 'u4': 
            return val ** 4 / 16.
        elif self.vv['flux'] == 'u3':
            return val ** 3 / 9.
        elif self.vv['flux'] == 'BL':
            return val ** 2 / (val ** 2 + 0.5 * (1-val) ** 2)
        elif self.vv['flux'].startswith("linear"):
            a = float(self.vv['flux'][len('linear'):])
            return a * val


    def reset(self, solution_idx=None, num_t=None, dx=None, dt=None, weno_regenerate=False):
        """
        return the initial state and reset some parameters
        """

        solution_num = len(self.solution_path_list)
        if solution_idx is None:
            if self.train_flag:
                self.solution_idx = np.random.randint(0, solution_num // 2)
            else:
                self.solution_idx = np.random.randint(solution_num // 2, solution_num)
        else:
            self.solution_idx = solution_idx

        solution_path = self.solution_path_list[self.solution_idx]
        solution_path = osp.join(self.solution_data_path, solution_path)
        solution_data = torch.load(solution_path)

        args = solution_data['args']
        
        if dx is None:
            self.dx = np.random.choice(self.vv['dx'])
        else:
            self.dx = dx

        if dt is None:
            self.dt = self.dx * args.cfl
        else:
            self.dt = dt

        self.precise_dx = args.precise_dx
        self.precise_dt = self.precise_dx * args.cfl
        self.x_low = args.x_low
        self.x_high = args.x_high
       

        if num_t is None:
            total_num_t = int(args.T /self.dt) + 1
            if not self.vv['same_time']:
                self.num_t = np.random.randint(total_num_t // 4, total_num_t)
            else:
                self.num_t = total_num_t // 4 * 3
        else:
            self.num_t = num_t

        self.weno_coarse_grid_euler = solution_data['coarse_solution_euler'][str(round(self.dx, 2))][:self.num_t]
        if isinstance(solution_data['coarse_solution_rk4'], tuple):
            self.weno_coarse_grid_rk4 = solution_data['coarse_solution_rk4'][0][str(round(self.dx, 2))][:self.num_t]
        else:
            self.weno_coarse_grid_rk4 = solution_data['coarse_solution_rk4'][str(round(self.dx, 2))][:self.num_t]

        self.forcing = None
        if solution_data.get('forcing') is not None:
            print("get forcing!")
            self.forcing = self.get_forcing_func(solution_data['forcing'])

        self.num_x = len(self.weno_coarse_grid_rk4[0])
        self.x_grid = np.linspace(self.x_low, self.x_high, self.num_x) 

        if weno_regenerate:
            self.weno_coarse_grid_rk4, self.weno_coarse_grid_euler = self.regenerate_weno_coarse_solution(solution_data)
            # debug(weno_coarse_grid_rk4, self.weno_coarse_grid_rk4)

        precise_num_t = int((self.num_t * self.dt) // self.precise_dt + 10)
        self.precise_weno_solutions = solution_data['precise_solution'][:precise_num_t]
        self.precise_weno_grid = np.linspace(args.x_low, args.x_high, len(self.precise_weno_solutions[0]))
    
        # first dim: x; second dim: t
        self.RLgrid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
        self.initial_value = self.weno_coarse_grid_euler[0]
        self.RLgrid[0, :] = self.initial_value
        self.t_idx = 1
        self.horizon = self.num_t - 1

        self.compute_weno_error()

        state = self.get_state(self.RLgrid[0])
        return state

    def regenerate_weno_coarse_solution(self, solution_data):
        args = copy.deepcopy(solution_data['args'])
        if args.__dict__.get('eta') is None:
            args.__dict__['eta'] = 0

        a, b, c, d, e = solution_data['a'], solution_data['b'], solution_data['c'], solution_data['d'], solution_data['e']
        init_value = a + b * np.sin(c * np.pi * self.x_grid) + d * np.cos(e * np.pi * self.x_grid)

        args.Tscheme = 'rk4'
        beg = time.time()
        coarse_solver = weno3_fd(args, 
            init_value=init_value, forcing=self.forcing, num_x=self.num_x, num_t=self.num_t, dt=self.dt, dx=self.dx)
        weno_coarse_grid_rk4 = coarse_solver.solve()
        self.weno_regenrate_time = time.time() - beg

        args.Tscheme = 'euler'
        coarse_solver = weno3_fd(args, 
            init_value=init_value, forcing=self.forcing, num_x=self.num_x, num_t=self.num_t, dt=self.dt, dx=self.dx)
        weno_coarse_grid_euler = coarse_solver.solve()
        return weno_coarse_grid_rk4, weno_coarse_grid_euler

    ### subsample the fine grid to get the coarse grid value.
    def get_precise_value(self, t):
        precise_t_idx = int(t / self.precise_dt)
        precise_val = self.precise_weno_solutions[precise_t_idx]
        f = interpolate.interp1d(self.precise_weno_grid, precise_val)
        ret = f(self.x_grid)
        
        return ret

    def get_forcing_func(self, forcing_params):
        a, omega, k, phi = forcing_params
        def func(x, t, period):
            spatial_phase = (2 * np.pi * k * x / period)
            signals = np.sin(omega * t + spatial_phase + phi)
            reference_forcing = np.sum(a * signals, axis=0)
            return reference_forcing
        return func

    ### the env moves on a step
    def step(self, action, Tscheme=None, eval=False):
        """ 
        parameters
        ----------
        action: a batch of size (self.num_x, 1)

        Return
        ------
        next_state: the state after taking the action  
        reward: the reward of taking the action at the current state  
        done: whether all points have reached the terminal time point  
        None: to fit the Gym interfaces
        """
        
        if Tscheme is None:
            Tscheme = self.vv['Tscheme']
        t_iter = self.t_idx

        if Tscheme == 'euler':
            self.RLgrid[t_iter] = self.RLgrid[t_iter - 1] + self.get_u_increment(self.RLgrid[t_iter - 1], action=action)
        elif Tscheme == 'rk4':
            self.RLgrid[t_iter] = self.rk_evolve(self.RLgrid[t_iter - 1], action)
        else:
            raise('invalid Time scheme!')
            
        # clip to avoid numeric explosion
        self.RLgrid[t_iter] = np.clip(self.RLgrid[t_iter], a_min = -50, a_max = 50)

        done = [0 for i in range(self.num_x)]
        reward = [0 for i in range(self.num_x)]

        # give reward. 
        if not eval:
            precise_val = self.get_precise_value(t_iter * self.dt)

            # error on u
            width = self.reward_width
            RL_neighbor = [self.RLgrid[t_iter][max(i-width, 0):min(i+width+1, self.num_x)] for i in range(self.num_x)]
            precise_neighbor = [precise_val[max(0, i-width):min(i+width+1, self.num_x)] for i in range(self.num_x)]
            errors = [np.max(np.abs(RL_neighbor[i] - precise_neighbor[i])) + 1e-300
                for i in range(self.num_x)] 
            
            # error on u_x
            width = 1
            RL_one_neighbor = self.expand_boundary(self.RLgrid[t_iter], width)
            precise_one_neighbor = self.expand_boundary(precise_val, width)

            RL_first_deri = [(RL_one_neighbor[i + 1] - RL_one_neighbor[i]) / self.dx for i in range(self.num_x)]
            precise_first_deri = [(precise_one_neighbor[i + 1] - precise_one_neighbor[i]) / self.dx for i in range(self.num_x)]
            first_deri_error_left = np.abs(np.array(RL_first_deri) - np.array(precise_first_deri))

            RL_first_deri = [(RL_one_neighbor[i + 2] - RL_one_neighbor[i + 1]) / self.dx for i in range(self.num_x)]
            precise_first_deri = [(precise_one_neighbor[i + 2] - precise_one_neighbor[i + 1]) / self.dx for i in range(self.num_x)]
            first_deri_error_right = np.abs(np.array(RL_first_deri) - np.array(precise_first_deri))

            reward = -np.log(errors) - \
                self.vv['reward_first_deriv_error_weight'] * np.log(first_deri_error_left) - \
                self.vv['reward_first_deriv_error_weight'] * np.log(first_deri_error_right)
            
            if not self.vv['no_done']:
                if t_iter == self.num_t - 1:
                    done = [1 for i in range(self.num_x)]

        next_state = self.get_state(self.RLgrid[t_iter])
        self.t_idx += 1
        return next_state, reward, done, None

    ### expand the boundary 
    def expand_boundary(self, val, left_width, right_width = None):
        '''
        expand the boundary points.
        '''
        if right_width is None:
            right_width = left_width
        if self.boundary_condition == 'periodic':
            tmp = list(val[-left_width - 1:-1]) + list(val) + list(val[1:right_width + 1])
        elif self.boundary_condition == 'outflow':
            tmp = list(val[:left_width]) + list(val) + list(val[-right_width:])
        else:
            raise('Invalide Boundary Condition!')
        return tmp

    ### generate the state
    def get_state(self, u):
        u_ = np.array(u)
        u_expand = self.expand_boundary(u_, 3)
        next_state = [u_expand[i:i + 6 + 1] for i in range(self.num_x)]
        return next_state
            
    ### the rk4 time scheme
    def rk_evolve(self, u_start, action):
        k1 = np.array(self.get_u_increment(u_start, action=action))
        k2 = np.array(self.get_u_increment(u_start + 0.5 * k1))
        k3 = np.array(self.get_u_increment(u_start + 0.5 * k2))
        k4 = np.array(self.get_u_increment(u_start + k3))
        return u_start + (k1 + 2 * (k2 + k3) + k4) / 6

    ### compute the change of the u-values 
    def get_u_increment(self, u, action = None): 
        if action is None: # rk4 step
            states = self.get_state(u)
            action = self.agent.action(states, True)

        flux_derivative = self.get_derivative(u, action)
       
        if self.eta > 0:
            u_xx = np.zeros_like(u)
            u_xx[1:-1] = (u[2:] + u[:-2] - 2 * u[1:-1]) / (self.dx ** 2)
            u_xx[0] = (u[1] + u[-2] - 2 * u[0]) / (self.dx ** 2)
            u_xx[-1] = (u[1] + u[-2] - 2 * u[-1]) / (self.dx ** 2)

        forcing_term = 0
        if self.forcing is not None:
            period = self.x_high - self.x_low
            forcing_term = self.forcing(x=self.x_grid, t=(self.t_idx-1) * self.dt, period=period)

        if self.eta > 0:
            increment = self.dt * (- flux_derivative + self.eta * u_xx  + forcing_term)
        else:
            increment = self.dt * (-flux_derivative + forcing_term)

        return increment

    def get_derivative(self, u, action):
        u_expand = self.expand_boundary(u, 3)
        u_expand = np.array(u_expand)
        fu_expand = self.flux(u_expand)
    
        action_num = action.shape[1] // 2
        left_flux_coef = action[:,:action_num] ### here left means i-1/2
        right_flux_coef = action[:, action_num:] ### here right means 1 + 1/2

        left_flux_points = np.array([fu_expand[i:i+6] for i in range(self.num_x)])
        right_flux_points = np.array([fu_expand[i+1:i+7] for i in range(self.num_x)])
        
        left_four_fluxes = left_flux_points.dot(self.weno_w)
        right_four_fluxes = right_flux_points.dot(self.weno_w)

        ## let RL judge upwind direction       
        left_flux = [left_four_fluxes[i].dot(left_flux_coef[i]) for i in range(self.num_x)]
        right_flux = [right_four_fluxes[i].dot(right_flux_coef[i]) for i in range(self.num_x)]
        left_flux = np.array(left_flux)
        right_flux = np.array(right_flux)
        return (right_flux - left_flux) / self.dx 

    ### Inf norm relative error
    def relative_error(self, precise, coarse, norm=2):
        return np.linalg.norm(precise - coarse, norm) / np.linalg.norm(precise, norm)

    ### compute the mean squared error compared with the fine-grid solution
    def error(self, baseline='rk4'):
        """
        return averaged relative l2 error.
        """

        error = np.zeros(self.num_t - 1)
        for i in range(1, self.num_t):
            rl_values = self.RLgrid[i]
            true_values = self.get_precise_value(self.dt * i)
            error[i - 1] = self.relative_error(true_values, rl_values)

        error = np.mean(error)
        rel_error = error / self.weno_error_euler if baseline == 'euler' else error / self.weno_error_rk4
        return error, rel_error