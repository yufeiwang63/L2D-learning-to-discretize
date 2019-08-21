import numpy as np
import random
import math
import copy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation
from Weno.weno3_2 import Weno3
from Weno.finite_difference_weno import weno3_fd
from Weno.Test_weno_error import get_weno_error

# Set up formatting for the animation files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

class Burgers(object):
    """
    u_t + (1/2u^2))_x = 0
    u_t + u * u_x = 0
    PDE class for RL learning, having the same interface as Gym
    """

    def __init__(self, args, init_func = None, true_solution_grids = None, agent = None, coarse_solutions = None):
        '''
        parameters:
            args: a dictionary storing all kinds of arguments
            init_func: a callable function giving the initial conditions. Interface should be f(x)
            true_solution_grids: the true solution grids
            agent: only required for RK4 time scheme. 
        '''
        self.T = args.T
        self.x_low, self.x_high = args.x_low, args.x_high
        self.dx = args.dx
        self.num_x = int((self.x_high - self.x_low) / self.dx + 1)
        self.num_t = int(args.T/args.dt + 1) # why + 1?
        self.x_grid = np.linspace(self.x_low, self.x_high, self.num_x, dtype = np.float64)
        
        # first dim: x; second dim: t
        self.RLgrid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
        # self.weno_grid = np.zeros((self.num_t, self.num_x))

        # width: determine how many points in the previous time iteration will be used
        self.state_width = args.state_window_size ### mainly for filter-based methods 
        self.action_width = args.action_window_size ### mainly for filter-based methods
        self.args = copy.copy(args)
        self.init_condition = init_func
       
        # agent for rk4
        self.agent = agent
        if self.args.Tscheme == 'rk4':
            assert self.agent is not None

        # record actions at each time step for animation
        self.actions = np.zeros((self.num_t, self.num_x))

        # currently trying fix dt
        self.dt = args.dt
        self.precise_dx =  self.args.precise_dx #self.args.dx #0.001
        self.precise_dt = self.precise_dx * args.cfl

        # initial and boundary conditions
        self.boundary_condition = args.boundary_condition

        # filters
        self.filters = np.array([[0,-0.5,0,0.5,0], [0,0,-1,1,0], [0,-1,1,0,0], [0,0,-1.5,2,-0.5], [0.5, -2, 1.5, 0,0]  ])
        if self.args.mode == 'eno' or self.args.mode == 'weno':
            assert len(self.filters) == self.args.action_dim
            assert len(self.filters[0]) == 1 + 2 * self.action_width

        self.initial_value = self.init_condition(self.x_grid, self.args.initial_t)
        self.precise_weno_solutions = None
        self.weno_coarse_grid = None
        
        # if true_solution_grids is None: ### True solution use fine grid finite-volume weno
        #     self.precise_weno_solutions = self.get_weno_grid(self.precise_dx, self.precise_dt, self.args.T + 0.1)
        # else:
        #     self.precise_weno_solutions = true_solution_grids

        # if coarse_solutions is None: ### coarse solution use coarse grid finite-difference weno
        #     corase_weno_solver = weno3_fd(args, init_value = self.initial_value)
        #     self.weno_coarse_grid = corase_weno_solver.solve()
        # else:
        #     self.weno_coarse_grid = coarse_solutions
        self.weno_error = None

        self.filter_inverse_matrix = np.linalg.inv(np.array([[1,1,1,1,1],[-2,-1,0,1,2],[4,1,0,1,4],[-8,-1,0,1,8],[16,1,0,1,16]]))
        self.weno_w = np.array([[1/3, 0, 0, 0],
                                [-7/6, -1/6, 0, 0],
                                [11/6, 5/6, 1/3, 0],
                                [0, 1/3, 5/6, 11/6],
                                [0, 0, -1/6, -7/6],
                                [0, 0, 0, 1/3]])

        if self.args.mode == 'nonlinear_weno_coef':
            self.corase_weno_solver = weno3_fd(self.args, init_value = self.initial_value)

    def get_weno_precise(self):
        self.precise_weno_solutions = self.get_weno_grid(self.precise_dx, self.precise_dt, self.args.T + 0.1)

    def get_weno_corase(self):
        corase_weno_solver = weno3_fd(self.args, init_value = self.initial_value)
        self.weno_coarse_grid = corase_weno_solver.solve()

    def save_weno_precise(self):
        np.save('../weno_solutions/{}-precise-{}-{}'.format(self.args.init, self.args.flux, self.args.cfl), self.precise_weno_solutions)
        
    def save_weno_coarse(self):
        np.save('../weno_solutions/{}-coarse-{}-{}-{}-{}'.format(self.args.init, self.args.Tscheme, self.args.dx, 
            self.args.flux, self.args.cfl), 
            self.weno_coarse_grid)

    def get_weno_error(self, recompute = False):
        if self.weno_error is not None:
            return self.weno_error
        else:
            assert self.weno_coarse_grid is not None
            true_values = self.get_precise_value(self.T)
            self.weno_error = np.mean((self.weno_coarse_grid[self.num_t - 1] - true_values) ** 2)
            return self.weno_error

    ### compute the fine grid weno solutions
    def get_weno_grid(self, dx, dt,  T):
        left_boundary = self.x_low - dx * 0.5
        right_boundary = self.x_high + dx * 0.5
        ncells = int((self.x_high - self.x_low) / dx + 1.) # need to be very small
        num_t = int(T/dt + 1.)
        w = Weno3(left_boundary, right_boundary, ncells, self.flux, self.flux_deriv, self.max_flux_deriv, 
            dx = dx, dt = dt, num_t = num_t + 100)
        x_center = w.get_x_center()
        u0 = self.init_condition(x_center, self.args.initial_t)
        w.integrate(u0, T)
        solutions = w.u_grid[:num_t,:]
        return solutions

    ### the burgers flux function. Can be changed to any other functions if needed for future useage.
    def flux(self, val):
        # val = np.array(val)
        if self.args.flux == 'u2':
            return val ** 2 / 2.
        elif self.args.flux == 'u4': 
            return val ** 4 / 16.
        elif self.args.flux == 'u3':
            return val ** 3 / 9.

    def flux_deriv(self, val):
        if self.args.flux == 'u2':
            return val
        elif self.args.flux == 'u4':
            return val ** 3 / 4.
        elif self.args.flux == 'u3':
            return val ** 2 / 3

    def max_flux_deriv(self, a, b):
        if self.args.flux == 'u2':
            return np.maximum(np.abs(a), np.abs(b))
        elif self.args.flux == 'u4':
            return np.maximum(np.abs(a ** 3 / 4.), np.abs(b ** 3 / 4.))
        elif self.args.flux == 'u3':
            return np.maximum(np.abs(a ** 2 / 3.), np.abs(b ** 2 / 3.))

    ### set dx, dt, T
    def set_args(self, args = None):
        if args is None:
            pass
        else:
            self.dx = args.dx
            self.dt = args.dt
            self.T = args.T

    ### gradually increasing the evolving time.
    def prolong_evolve(self):
        self.T = min(1.0, self.T + 0.1)
        self.num_t = int(self.T/self.dt + 1)

    ### reset the internal variables and return the initial states
    def reset(self):
        """
        return the initial state and reset some parameters
        """
        self.num_x = int((self.x_high - self.x_low) / self.dx + 1)
        self.num_t = int(self.T/self.dt + 1) # why + 1?
        self.x_grid = np.linspace(self.x_low, self.x_high, self.num_x, dtype = np.float64)
        self.initial_value = self.init_condition(self.x_grid, self.args.initial_t)
        self.upwindgrid = np.zeros((self.num_t, self.num_x))
        if self.args.plot_upwind:
            self.upwind_scheme()
 
        
        # first dim: x; second dim: t
        self.RLgrid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
        self.RLgrid[0,:] = self.initial_value

        reward_cnt = self.args.reward_time  # number of times to give reward
        reward_give_period = (self.num_t - 1) // reward_cnt
        self.reward_index = [t * reward_give_period for t in range(1, reward_cnt)]
        if self.num_t - 1 not in self.reward_index:
            self.reward_index.append(self.num_t - 1)

        state = self.get_state(self.RLgrid[0])
        return state

    ### subsample the fine grid to get the coarse grid value.
    def get_precise_value(self, t):
        current_t = t
        precise_t_idx = int(current_t / self.precise_dt)
        precise_val = self.precise_weno_solutions[precise_t_idx]
        factor = int(self.dx / self.precise_dx)
        ret = []
        for i in range(self.num_x):
            ret.append(precise_val[i * factor]) 
        return np.array(ret)

    ### the env moves on a step
    def step(self, action, t_iter, Tscheme = None, mode = 'test'):
        """ 
        parameters
        ----------
        action: a batch of size (self.num_x, 1)
        t_iter: the idx of the iterations of the time dimension  

        Return
        ------
        next_state: the state after taking the action  
        reward: the reward of taking the action at the current state  
        done: whether all points have reached the terminal time point  
        None: to fit the Gym interfaces
        """
        # record for visualization
        if self.args.test:
            if self.args.mode == 'eno':
                self.actions[t_iter] = action
        
        if self.args.formulation == 'MLP':
            done = [False for i in range(self.num_x)]
            reward = [0 for i in range(self.num_x)]
        else:
            done = False
            reward = 0  

        if self.args.mode == 'continuous_filter':
            num = len(action)
            action = np.array(action)
            # print('original action shape is: ', action.shape)
            zero_first_order = np.zeros((num, 2))
            zero_first_order[:,0] = 0
            zero_first_order[:,1] = 1
            action = np.concatenate((zero_first_order, action), axis = 1)
            # print('after concat action shape is: ', action.shape)
            action = self.filter_inverse_matrix.dot(action.transpose()).transpose()
            # print('after matrix constraint transformation, shape is: ', action.shape)


        if Tscheme is None:
            Tscheme = self.args.Tscheme

        if Tscheme == 'euler':
            self.RLgrid[t_iter] = self.RLgrid[t_iter - 1] + self.get_u_increment(self.RLgrid[t_iter - 1], action = action)
        elif Tscheme == 'rk4':
            self.RLgrid[t_iter] = self.rk_evolve(self.RLgrid[t_iter - 1], action, mode = mode)
        else:
            raise('invalid Time scheme!')
            
        # clip to avoid numeric explosion
        self.RLgrid[t_iter] = np.clip(self.RLgrid[t_iter], a_min = -50, a_max = 50)

        # give reward. 
        if t_iter in self.reward_index or self.args.reward_every_step:
            precise_val = self.get_precise_value(t_iter * self.dt)
            assert precise_val is not None

            if self.args.formulation == 'MLP':
                if self.args.reward_type == 'difference':
                    ### now error
                    now_precise=self.get_precise_value((t_iter-1)*self.dt)
                    now_error=np.abs(self.RLgrid[t_iter-1]-now_precise)
                    ### next state error
                    next_precise=self.get_precise_value(t_iter*self.dt)
                    next_error=np.abs(self.RLgrid[t_iter]-next_precise)
                    reward = now_error - next_error
                elif self.args.reward_type == 'single' or self.args.reward_type == 'neighbor':
                    if self.args.reward_type == 'single':
                        width = 1
                    elif self.args.reward_type == 'neighbor':
                        width = self.action_width 
                        if self.args.mode == 'weno_coef_four' or self.args.mode == 'nonlinear_weno_coef':
                            width = 3
                    RL_neighbor = [self.RLgrid[t_iter][max(i-width, 0) : min(i+width+1, self.num_x)] for i in range(self.num_x)]
                    precise_neighbor = [precise_val[max(0, i - width): min(self.num_x, i + width + 1)] for i in range(self.num_x)]
                    if self.args.log_reward:
                        if self.args.log_reward_type == 'max':
                            # errors = [10 * np.max(np.abs(RL_neighbor[i] - precise_neighbor[i])) + self.args.log_reward_clip
                            #     for i in range(self.num_x)] # 3.5 以前的reward_scale 是在这里的 np.max 前面乘上了10
                            errors = [np.max(np.abs(RL_neighbor[i] - precise_neighbor[i])) + 1e-300
                                for i in range(self.num_x)] 
                           
                        if self.args.log_reward_type == 'sum':
                            errors = [np.sum(np.abs(RL_neighbor[i] - precise_neighbor[i])) + 1e-300
                                for i in range(self.num_x)]
                        reward = -np.log(errors)
                    else:
                        reward = [-np.max(np.abs(RL_neighbor[i] - precise_neighbor[i])) for i in range(self.num_x)]
                elif self.args.reward_type == 'all':
                    error = np.abs(precise_val - self.RLgrid[t_iter])
                    if self.args.log_reward_type == 'sum':
                        error = np.log(np.sum(error) + 1e-100)
                    elif self.args.log_reward_type == 'max':
                        error = np.log(np.max(error) + 1e-100)
                    reward = [-error for i in range(self.num_x)]

                tv_diff = self.get_tv(self.RLgrid[t_iter]) - self.get_tv(self.RLgrid[t_iter - 1])
                if tv_diff > 0:
                    reward = [x - self.args.tv_reward_coef * tv_diff for x in reward]
                    
                reward = [x * self.args.reward_scale for x in reward]
                if t_iter == self.num_t - 1:
                    done = [True for i in range(self.num_x)]
            else:
                reward = np.max(np.abs(precise_val - self.RLgrid[t_iter]))
                if t_iter == self.num_t - 1:
                    done = True

        next_state = self.get_state(self.RLgrid[t_iter])
        return next_state, reward, done, None

    ### compute the total tv violation
    def get_tv(self, arr):
        res = 0
        for idx in range(len(arr) - 1):
            res += np.abs(arr[idx + 1] - arr[idx])

        return res

    ### expand the boundary 
    def expand_boundary(self, val, left_width, right_width = None):
        '''
        expand the boundary points.
        '''
        if right_width is None:
            right_width = left_width
        if self.boundary_condition == 'periodic':
            tmp = list(val[-left_width - 1:-1]) + list(val) + list(val[1:right_width + 1])
            # print(val[-left_width:])
            # print(val[:right_width])
            # assert (val[-left_width:] == val[:right_width]).all()
        elif self.boundary_condition == 'outflow':
            tmp = list(val[:left_width]) + list(val) + list(val[-right_width:])
        else:
            raise('Invalide Boundary Condition!')
        return tmp

    ### generate the state
    def get_state(self, u):
        u_ = np.array(u)
        if self.args.input_normalize:
            u_ = (u_ - np.mean(u_)) / (np.std(u_) + 1e-5)
        
        # next_state = [np.array(list(tmp[i:i+1 +2 * self.width]) + [self.dx, self.dt, au_values[i]]) for i in range(self.num_x)]
        if self.args.formulation == 'MLP':
            if self.args.mode == 'eno' or self.args.mode == 'weno':
                u_expand = self.expand_boundary(u_, self.state_width)
                fu_expand = self.flux(np.array(u_expand))
                au_values = u_
                next_state = [np.array(list(fu_expand[i:i + 1 +2 * self.state_width]) + list(u_expand[i:i + 1 + 2 * self.state_width]) +  [au_values[i]])
                        for i in range(self.num_x)]
            elif self.args.mode == 'compute_flux' or self.args.mode == 'continuous_filter' :
                u_expand = self.expand_boundary(u_, self.state_width)
                fu_expand = self.flux(np.array(u_expand))
                au_values = u_
                next_state = [np.array(list(fu_expand[i:i + 1 +2 * self.state_width]) + list(u_expand[i:i + 1 + 2 * self.state_width]))
                        for i in range(self.num_x)]
            elif self.args.mode == 'constrained_flux':
                u_expand = self.expand_boundary(u_, self.args.p + 1, self.args.q)
                len_ = self.args.p + self.args.q + 1
                next_state = np.zeros((self.num_x, len_ + 1))
                for i in range(self.num_x):
                    next_state[i][:len_] = u_expand[i:i+len_]
                    next_state[i][1:] = u_expand[i+1:i+1+len_]
            elif self.args.mode == 'weno_coef' or self.args.mode == 'weno_coef_four' or self.args.mode == 'nonlinear_weno_coef':
                # print('get state enter nonliner weno coef')
                u_expand = self.expand_boundary(u_, 3)
                next_state = [u_expand[i:i + 6 + 1] for i in range(self.num_x)]
             
        elif self.args.formulation == 'FCONV':
            next_state = np.zeros((1, 2, len(u_expand)))
            next_state[0, 0, :] = u_expand
            next_state[0, 1, :] = fu_expand
        
        return next_state

    ### the rk4 time scheme
    def rk_evolve(self, u_start, action, mode = 'test'):
        '''
        if mode == test: then all actions is generated by agent
        if mode == train: then first action is generated by agent, and rest of the actions are generated by weno
        '''
        k1 = np.array(self.get_u_increment(u_start, action = action))
        k2 = np.array(self.get_u_increment(u_start + 0.5 * k1, mode = mode))
        k3 = np.array(self.get_u_increment(u_start + 0.5 * k2, mode = mode))
        k4 = np.array(self.get_u_increment(u_start + k3, mode = mode))
        return u_start + (k1 + 2 * (k2 + k3) + k4) / 6

    ### compute the change of the u-values 
    def get_u_increment(self, u, action = None, upwind = False, mode = 'test'): 
        if action is None: # rk4 step
            states = self.get_state(u)
            if mode == 'test':
                action = self.agent.action(states, True)
            elif mode == 'train':
                action = self.agent.action(states, mode = 'weno')
        if self.args.mode == 'eno' or upwind:
            u = self.expand_boundary(u, self.action_width)
            f = self.flux(np.array(u))
            batch = [np.array(f[i:i+ 1 + 2 * self.action_width]) for i in range(self.num_x)]
            increment = [self.dt * -batch[i].dot(self.filters[action[i]]) / self.dx for i in range(self.num_x)]
        elif self.args.mode == 'compute_flux':
            increment = [self.dt * -(action[i][1] - action[i][0]) / self.dx for i in range(self.num_x)]
        elif self.args.mode == 'continuous_filter':
            u = self.expand_boundary(u, self.action_width)
            f = self.flux(np.array(u))
            batch = [np.array(f[i:i+ 1 + 2 * self.action_width]) for i in range(self.num_x)]
            increment = [self.dt * -batch[i].dot(action[i]) / self.dx for i in range(self.num_x)]
        elif self.args.mode == 'constrained_flux':
            increment = [self.dt * -(action[i][1] - action[i][0]) / self.dx for i in range(self.num_x)]
        elif self.args.mode == 'weno_coef_four':
            u_expand = self.expand_boundary(u, 3)
            u_expand = np.array(u_expand)
            fu_expand = self.flux(u_expand)
        
            action_num = self.args.action_dim // 2
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

            increment = self.dt * -(right_flux - left_flux) / self.dx 

        elif self.args.mode == 'weno_coef':
            u_expand = self.expand_boundary(u, 3)
            u_expand = np.array(u_expand)
            fu_expand = self.flux(u_expand)
        
            action_num = self.args.action_dim // 2
            left_flux_coef = action[:,:action_num] ### here left means i-1/2
            right_flux_coef = action[:, action_num:] ### here right means 1 + 1/2

            left_flux_points = np.array([fu_expand[i:i+6] for i in range(self.num_x)])
            right_flux_points = np.array([fu_expand[i+1:i+7] for i in range(self.num_x)])
            
            left_four_fluxes = left_flux_points.dot(self.weno_w)
            right_four_fluxes = right_flux_points.dot(self.weno_w)

            ### hand judge upwind direction
            if self.args.handjudge_upwind:
                left_flux = np.zeros(self.num_x)
                right_flux = np.zeros(self.num_x)
                for i in range(self.num_x):
                    # left_roe = (fu_expand[i+3] - fu_expand[i+2]) / (u_expand[i+3] - u_expand[i+2])
                    # right_roe = (fu_expand[i+4] - fu_expand[i+3]) / (u_expand[i+4] - u_expand[i+3])
                    left_roe = (u_expand[i+3] + u_expand[i+2])
                    right_roe = (u_expand[i+4] + u_expand[i+3]) ### last two lines reduce to these two lines with f = 1/2 u ** 2
                    if left_roe >= 0:
                        left_flux[i] = left_four_fluxes[i,:-1].dot(left_flux_coef[i])
                    else:
                        left_flux[i] = left_four_fluxes[i,1:].dot(left_flux_coef[i])
                    if right_roe >= 0:
                        right_flux[i] = right_four_fluxes[i,:-1].dot(right_flux_coef[i])
                    else:
                        right_flux[i] = right_four_fluxes[i, 1:].dot(right_flux_coef[i])

            ### the fourth dim determines the upwind direction
            else:
                left_flux = np.zeros(self.num_x)
                right_flux = np.zeros(self.num_x)
                for i in range(self.num_x):
                    if left_flux_coef[i][3] >= 0.5:
                        left_flux[i] = left_four_fluxes[i,:-1].dot(left_flux_coef[i, :3])
                    else:
                        left_flux[i] = left_four_fluxes[i,1:].dot(left_flux_coef[i, :3])
                    if right_flux_coef[i][3] >= 0.5:
                        right_flux[i] = right_four_fluxes[i,:-1].dot(right_flux_coef[i, :3])
                    else:
                        right_flux[i] = right_four_fluxes[i, 1:].dot(right_flux_coef[i, :3])

            increment = self.dt * -(right_flux - left_flux) / self.dx 
        
        elif self.args.mode == 'nonlinear_weno_coef':
            # print('get_u_increment enters nonlinear weno coef')
            u_expand = self.expand_boundary(u, 3)
            u_expand = np.array(u_expand)
            fu_expand = self.flux(u_expand)

            weno_left_fluxes, weno_right_fluxes = self.corase_weno_solver.get_flux(u)
        
            action_num = self.args.action_dim // 2
            left_flux_coef = action[:,:action_num] ### here left means i-1/2
            right_flux_coef = action[:, action_num:] ### here right means 1 + 1/2

            left_flux_points = np.array([fu_expand[i:i+6] for i in range(self.num_x)])
            right_flux_points = np.array([fu_expand[i+1:i+7] for i in range(self.num_x)])
            
            left_four_fluxes = left_flux_points.dot(self.weno_w)
            right_four_fluxes = right_flux_points.dot(self.weno_w)

            ### hand judge upwind direction
            left_flux = np.zeros(self.num_x)
            right_flux = np.zeros(self.num_x)
            for i in range(self.num_x):
                f1, f2, f3, f4 = left_four_fluxes[i][0], left_four_fluxes[i][1], left_four_fluxes[i][2], left_four_fluxes[i][3]
                # left_flux[i] = (f1 + f2 + f3 + f4) / 4 + left_flux_coef[i][0] * \
                #     max(0, left_flux_coef[i][1] * (f1 - f2) + left_flux_coef[i][2] * (f1 - f3) + left_flux_coef[i][3] * (f1 - f4))
                left_flux[i] = weno_left_fluxes[i] + left_flux_coef[i][0] * max(0, left_flux_coef[i][1] * \
                    max(0, left_flux_coef[i][2] * (f1 - f2) + left_flux_coef[i][3] * (f1 - f3) + left_flux_coef[i][4] * (f1 - f4)))
                
                f1, f2, f3, f4 = right_four_fluxes[i][0], right_four_fluxes[i][1], right_four_fluxes[i][2], right_four_fluxes[i][3]
                # right_flux[i] = (f1 + f2 + f3 + f4) / 4 + right_flux_coef[i][0] * \
                #     max(0, right_flux_coef[i][1] * (f1 - f2) + right_flux_coef[i][2] * (f1 - f3) + right_flux_coef[i][3] * (f1 - f4))
                right_flux[i] = weno_right_fluxes[i] + right_flux_coef[i][0] * max(0, right_flux_coef[i][1] * \
                    max(0, right_flux_coef[i][2] * (f1 - f2) + right_flux_coef[i][3] * (f1 - f3) + right_flux_coef[i][4] * (f1 - f4)))
            

            increment = self.dt * -(right_flux - left_flux) / self.dx
            
        elif self.args.mode == 'weno':
            action = np.array(action)
            action -= np.max(action, axis = 1).reshape(-1,1)
            action = np.exp(action) / np.sum(np.exp(action), axis = 1).reshape(-1,1)
            assert np.abs(np.sum(action[0]) - 1) < 1e-5
            assert self.args.action_dim == len(action[0])
            assert len(action) == self.num_x
            combine_filters = [np.sum(np.array([self.filters[j] * action[i][j] for j in range(self.args.action_dim)]), axis = 0) for i in range(self.num_x)]
            # assert np.abs(np.sum(combine_filters)) < 1e-5
            increment = [self.dt * -batch[i].dot(combine_filters[i])
                / self.dx for i in range(self.num_x)]

        return increment

    ### Inf norm relative error
    def relative_error(self, precise, coarse):
        # return np.max(np.abs(precise - coarse)) / np.max(np.abs(precise))
        return np.linalg.norm(precise - coarse, 2) / np.linalg.norm(precise, 2)

    ### make evolving animations of the trained network.
    def show(self):
        fig = plt.figure(figsize = (15, 10))
        # ax = plt.axes(xlim=(self.x_low ,self.x_high),ylim=(self.args.plot_y_low, self.args.plot_y_high))
        ax = fig.add_subplot(2,1,1)
        ax.set_xlim((self.x_low ,self.x_high))
        ax.set_ylim((self.args.plot_y_low, self.args.plot_y_high))
        lineweno, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)] ,lw=2, label = 'weno3')
        linerl, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)],lw=2, label = 'rl')
        lineweno_coarse, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)], lw = 2, label = 'weno_coarse')
        # line_upwind, = ax.plot([], [], lw = 2, label = 'first order upwind')
        # line_exact, = ax.plot([], [], lw = 2, label = 'exact solution')

        if self.weno_coarse_grid is None:
            self.weno_coarse_grid = np.zeros((self.num_t, self.num_x))

        draw_data = np.zeros((self.num_t, 4 * self.num_x))
        draw_data[:,self.num_x: 2 * self.num_x] = self.RLgrid
        draw_data[:, self.num_x * 2: self.num_x * 3] = self.weno_coarse_grid[:self.num_t, :]
        # draw_data[:, self.num_x * 3: self.num_x * 4] = self.upwindgrid
        for t in range(self.num_t):
            draw_data[t, : self.num_x] = self.get_precise_value(t * self.dt) # when doing showing, use the grid values

        if self.args.mode == 'eno':
            annotaions = [] # each x point get an annotation label
            for i in range(self.num_x):
                annotaions.append(ax.annotate(
                    'heihei', xy=(self.x_grid[i], self.RLgrid[0][i]), xytext=(self.x_grid[i], self.RLgrid[0][i] + 0.3),
                    arrowprops = {'arrowstyle': "->"}
                ))

            filter_dic = {0: 'cd', 1: '1r', 2: '1l', 3: '2r', 4: '2l'}
            filters = [[] for i in range(self.num_t)]
            for i in range(self.num_t):
                filters[i] = [filter_dic[a] for a in self.actions[i]]

        error_ax = fig.add_subplot(2,1,2)
        coarse_error = np.zeros(self.num_t)
        RL_error = np.zeros(self.num_t)
        for i in range(self.num_t):
            coarse_error[i] = self.relative_error(draw_data[i, :self.num_x], draw_data[i, 2 * self.num_x:3*self.num_x])
            RL_error[i] = self.relative_error(draw_data[i, :self.num_x], draw_data[i, self.num_x:2*self.num_x])
        RL_error_line, = error_ax.plot(range(self.num_t), RL_error,  'r', lw= 2, label = 'RL_relative_error')
        weno_coarse_error_line, = error_ax.plot(range(self.num_t), coarse_error,  'b', lw = 2, label = 'weno_coarse_relative_error')
        RL_error_point, = error_ax.plot([], [], 'ro', markersize = 5)
        weno_coarse_error_point, = error_ax.plot([], [], 'bo', markersize = 5)

        def init():    
            linerl.set_data([], [])
            lineweno.set_data([],[])
            lineweno_coarse.set_data([], [])
            RL_error_point.set_data([],[])
            weno_coarse_error_point.set_data([],[])
            # line_upwind.set_data([], [])
            linerl.set_label('RL solution')
            lineweno.set_label('weno solution')
            lineweno_coarse.set_label('weno coarse solution')
            # line_upwind.set_label('first order upwind solution')
            if self.args.mode == 'eno':
                # return linerl, lineweno, lineweno_coarse, line_upwind,  annotaions
                return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point,  annotaions
            else:
                # return linerl, lineweno, lineweno_coarse, line_upwind
                return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point

        def func(i):
            print('make animations, step: ', i)
            x = np.linspace(self.x_low, self.x_high, self.num_x)
            yweno = draw_data[i,:self.num_x]
            yrl = draw_data[i, self.num_x:2 * self.num_x]
            yweno_coarse = draw_data[i, 2 * self.num_x: 3 * self.num_x]
            # yupwind = draw_data[i, 3 * self.num_x: 4 * self.num_x]
            linerl.set_data(x,yrl)
            lineweno.set_data(x, yweno)
            lineweno_coarse.set_data(x, yweno_coarse)
            # linerl.set_label('RL solution')
            # lineweno.set_label('weno solution')
            # lineweno_coarse.set_label('weno coarse solution')
            RL_error_point.set_data(i, RL_error[i])
            weno_coarse_error_point.set_data(i, coarse_error[i])
            # line_upwind.set_data(x, yupwind)
            if self.args.mode == 'eno':
                for idx, an in enumerate(annotaions):
                    an.set_text(filters[i][idx])
                    an.set_position((self.x_grid[idx], self.RLgrid[i][idx] + 0.3))
                    an.xy = (self.x_grid[idx], self.RLgrid[i][idx])
                # return linerl, lineweno, lineweno_coarse, line_upwind,  annotaions
                return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point, annotaions
            else:
                # return linerl, lineweno, lineweno_coarse, line_upwind
                return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point


        anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=self.num_t, interval=50)
        plt.legend()
        plt.title('init {0}\n dx {1} dt {2} T {3} precise_dx {4} cfl {5}'.format(self.args.init, 
            self.dx, self.dt, self.T, self.precise_dx, self.args.cfl))
        plt.tight_layout()
        if self.args.save_RL_weno_animation:
            anim.save(self.args.save_RL_weno_animation_path + '{0}_dx{1}_dt{2}_T{3}_precise_dx{4}_cfl{5}_tscheme{6}_flux{7}.mp4'.format(
                self.args.init, self.dx, self.dt, self.T, self.precise_dx, self.args.cfl, self.args.Tscheme, self.args.flux
                ), writer=writer)
        elif self.args.show_RL_weno_animation:
            plt.show()
      
        plt.close()

    ### save figures at specified time point
    def save_figure(self, t_iter):
        rl_values = self.RLgrid[t_iter]
        true_values = self.get_precise_value(t_iter * self.dt)
        T = self.dt * t_iter
        plt.figure()
        plt.plot(self.x_grid, rl_values, 'o', markersize = 4, label = 'RL value')
        plt.plot(self.x_grid, true_values, '+', markersize = 4, label = 'weno value')
        plt.legend(fontsize = 13)
        plt.title('dx: {0}  dt: {1}  T {2} \n {3}'.format(self.dx, self.dt, T, self.args.init), fontsize = 13)
        plt.tight_layout()
        plt.savefig('./report/tmp/figure_' + self.args.init + str(T) + str(self.dx) + str(self.dt) +  '.png')
        plt.close()

    ### 
    def compute_tvd(self):
        sum = 0
        for i in range(self.num_t - 1):
            tv_diff = self.get_tv(self.RLgrid[i+1]) - self.get_tv(self.RLgrid[i])
            if tv_diff > 0:
                sum += tv_diff

        return sum

    ### compute the 1-order upwind scheme
    def upwind_scheme(self):
        self.upwindgrid[0,:] = self.initial_value
        for t in range(1, self.num_t):
            action = []
            for idx in range(self.num_x):
                # tmp = np.sign(
                #     (self.flux(self.upwindgrid[t-1][(idx + 1) % self.num_x]) - self.flux(self.upwindgrid[t-1][(idx - 1) % self.num_x])) \
                #         / (self.upwindgrid[t-1][(idx + 1) % self.num_x] - self.upwindgrid[t-1][(idx - 1) % self.num_x] + 1e-50)
                #         )
                tmp = self.upwindgrid[t - 1][idx]
                if tmp > 0:
                    action.append(2)
                else:
                    action.append(1)
            self.upwindgrid[t] = self.upwindgrid[t-1] + self.get_u_increment(self.upwindgrid[t-1], action, upwind = True)
            # print(t)
            # for idx in range(self.num_x):
            #     if self.upwindgrid[t-1][idx] < 0:
            #         self.upwindgrid[t][idx] = self.upwindgrid[t- 1][idx] - self.dt * \
            #             ((self.upwindgrid[t- 1][idx]) ** 2 / 2. - (self.upwindgrid[t- 1][(idx-1) % self.num_x]) ** 2 / 2.) / self.dx      
            #     else:
            #         self.upwindgrid[t][idx] = self.upwindgrid[t- 1][idx] - self.dt * \
            #             ((self.upwindgrid[t- 1][(idx + 1) % self.num_x]) ** 2 / 2. - (self.upwindgrid[t- 1][idx]) ** 2 / 2.) / self.dx 
        print('upwind compute done')       
                    
    ### compute the mean squared error compared with the fine-grid solution
    def error(self):
        """
        return the terminal time point MSE error
        """

        rl_values = self.RLgrid[-1]
        true_values = self.get_precise_value(self.T)
        error = np.mean((rl_values - true_values) ** 2)
        if self.args.test:
            print('init condition: ', self.args.init, ' Mean of L2 Error is: ', error)
   
        assert true_values is not None
        if self.args.animation:
            self.show()
            # self.compute_speed()

        if self.args.compute_tvd:
            self.tvd_break = self.compute_tvd()
        
        return error






