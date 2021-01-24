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
import torch

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
        # self.RLgrid = torch.zeros((self.num_t, self.num_x), device = self.device, dtype = torch.float)
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
        self.weno_error = None

        self.filter_inverse_matrix = np.linalg.inv(np.array([[1,1,1,1,1],[-2,-1,0,1,2],[4,1,0,1,4],[-8,-1,0,1,8],[16,1,0,1,16]]))
        self.weno_w = np.array([[1/3, 0, 0, 0],
                                [-7/6, -1/6, 0, 0],
                                [11/6, 5/6, 1/3, 0],
                                [0, 1/3, 5/6, 11/6],
                                [0, 0, -1/6, -7/6],
                                [0, 0, 0, 1/3]])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weno_w = torch.tensor(self.weno_w, device = self.device, dtype = torch.float)

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
            # self.weno_error = np.mean((self.weno_coarse_grid[self.num_t - 1] - true_values) ** 2)
            self.weno_error = np.linalg.norm(self.weno_coarse_grid[self.num_t - 1] - true_values, 2) / np.linalg.norm(true_values, 2)
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

    ### reset the internal variables and return the initial states
    def reset(self):
        """
        return the initial state and reset some parameters
        tensor done.
        """
        self.num_x = int((self.x_high - self.x_low) / self.dx + 1)
        self.num_t = int(self.T/self.dt + 1) # why + 1?
        self.x_grid = np.linspace(self.x_low, self.x_high, self.num_x, dtype = np.float64)
        self.initial_value = self.init_condition(self.x_grid, self.args.initial_t)
        self.upwindgrid = np.zeros((self.num_t, self.num_x))
        
        # first dim: x; second dim: t
        self.RLgrid = torch.zeros((self.num_t, self.num_x), device = self.device, dtype = torch.float) # record the value at each (x,t) point
        self.RLgrid[0,:] = torch.tensor(self.initial_value, device = self.device, dtype = torch.float)

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
    def step(self, action, t_iter, Tscheme = None):
        """ 
        parameters
        ----------
        action: torch.tensor of shape (N x 8)
        t_iter: the idx of the iterations of the time dimension  

        Return
        ------
        next_state: the state after taking the action  
        reward: the reward of taking the action at the current state  
        done: whether all points have reached the terminal time point  
        None: to fit the Gym interfaces

        Tensor Done.
        """
        assert type(action) == torch.Tensor

        if Tscheme is None:
            Tscheme = self.args.Tscheme

        self.RLgrid = self.RLgrid.detach()
        # self.RLgrid[t_iter - 1].requires_grad = False
        if Tscheme == 'euler':
            increment = self.get_u_increment(self.RLgrid[t_iter - 1].detach(), action = action)
            #print("in Burgers_SL step(), RLgrid[t_iter-1] shape is: ", self.RLgrid[t_iter - 1].shape)
            tmp = self.RLgrid[t_iter - 1] + increment
            #print("in Burgers_SL step(), after adding increment to RLgrid[t_iter-1], we get shape: ", tmp.shape)
        elif Tscheme == 'rk4':
            # only test should use rk4. 
            tmp = self.rk_evolve(self.RLgrid[t_iter - 1], action)
        else:
            raise('invalid Time scheme!')
            
        # clip to avoid numeric explosion
        self.RLgrid[t_iter] = torch.clamp(tmp, min = -50, max = 50)
        # if t_iter == 2:
        #     self.RLgrid[t_iter].mean().backward()
        #     #print("after clamp, self.RLgrid[t_iter] can backward")

        # give loss. 
        precise_val = self.get_precise_value(t_iter * self.dt)
        assert precise_val is not None
        precise_val = torch.tensor(precise_val, device = self.device, dtype = torch.float)

        width = 3
        RL_neighbor = [self.RLgrid[t_iter][max(i-width, 0) : min(i+width+1, self.num_x)] for i in range(self.num_x)]
        precise_neighbor = [precise_val[max(0, i - width): min(self.num_x, i + width + 1)] for i in range(self.num_x)]
        if self.args.reward_type == 'max':
            errors = [torch.max(torch.abs(RL_neighbor[i] - precise_neighbor[i])).unsqueeze(0) + 1e-300
                for i in range(self.num_x)]
            errors = torch.cat(errors)
                
            loss = torch.log(errors).mean()
        elif self.args.reward_type == 'l2':
            errors = [torch.mean((RL_neighbor[i] - precise_neighbor[i])**2).unsqueeze(0)
                for i in range(self.num_x)]
            errors = torch.cat(errors)
                
            loss = errors.mean()

        assert loss.requires_grad == True
        # if t_iter == 2:
        #     loss.mean().backward()
        #     #print("before get_state, loss can backward")

        next_state = self.get_state(self.RLgrid[t_iter])
        return next_state, loss

    ### compute the total tv violation
    def get_tv(self, arr):
        res = 0
        for idx in range(len(arr) - 1):
            res += np.abs(arr[idx + 1] - arr[idx])

        return res

    ### expand the boundary 
    def expand_boundary(self, val, left_width, right_width = None):
        '''
        expand the boundary points. Tensor Done.
        '''
        assert type(val) == torch.Tensor
        if right_width is None:
            right_width = left_width
        if self.boundary_condition == 'periodic':
            # tmp = list(val[-left_width - 1:-1]) + list(val) + list(val[1:right_width + 1])
            tmp = torch.cat([val[-left_width - 1:-1], val, val[1:right_width + 1]])
            #print("in Burgers_SL expand_boundary, tmp.shape is: ", tmp.shape)
        elif self.boundary_condition == 'outflow':
            tmp = list(val[:left_width]) + list(val) + list(val[-right_width:])
        else:
            raise('Invalide Boundary Condition!')
        return tmp

    ### generate the state
    def get_state(self, u):
        '''
        Tensor done.
        '''
        #print(type(u))
        assert type(u) == torch.Tensor
        u_expand = self.expand_boundary(u, 3)
        next_state = torch.cat([u_expand[i:i + 6 + 1] for i in range(self.num_x)])
        #print("in Burgers_SL get_state, next_state shape is: ", next_state.shape)
        next_state = next_state.view((-1, self.args.state_dim))
        return next_state

    ### the rk4 time scheme
    def rk_evolve(self, u_start, action):
        k1 = self.get_u_increment(u_start, action = action)
        k2 = self.get_u_increment(u_start + 0.5 * k1)
        k3 = self.get_u_increment(u_start + 0.5 * k2)
        k4 = self.get_u_increment(u_start + k3)

        return u_start + (k1 + 2 * (k2 + k3) + k4) / 6

    ### compute the change of the u-values 
    def get_u_increment(self, u, action = None, ): 
        '''
        Tensor Done.
        '''
        assert type(u) == torch.Tensor
        # assert type(action) == torch.Tensor
        if action is None: # rk4 step
            states = self.get_state(u)
            action = self.agent.action(states)
        u_expand = self.expand_boundary(u, 3)
        fu_expand = self.flux(u_expand)

        action_num = self.args.action_dim // 2
        left_flux_coef = action[:,:action_num] ### here left means i-1/2
        right_flux_coef = action[:, action_num:] ### here right means 1 + 1/2

        left_flux_points = torch.cat([fu_expand[i:i+6] for i in range(self.num_x)], dim = 0)
        right_flux_points = torch.cat([fu_expand[i+1:i+7] for i in range(self.num_x)], dim = 0)
        #print("in Burgers_SL get_u_increment, left_flux_points shape is: ", left_flux_points.shape)
        left_flux_points = left_flux_points.view((-1,6))
        right_flux_points = right_flux_points.view((-1,6))
        
        # Nx6 * 6x4 -> Nx4
        left_four_fluxes = torch.matmul(left_flux_points, self.weno_w)
        right_four_fluxes = torch.matmul(right_flux_points, self.weno_w)

        left_flux = torch.cat([torch.dot(left_four_fluxes[i], left_flux_coef[i]).unsqueeze(0) for i in range(self.num_x)])
        right_flux = torch.cat([torch.dot(right_four_fluxes[i], right_flux_coef[i]).unsqueeze(0) for i in range(self.num_x)])
        #print("in Burgers_SL get_u_increment, left_flux shape is: ", left_flux.shape)
        # left_flux.mean().backward()
        # #print("in Burgers_SL get_u_increment(), left_flux can backward")

        increment = self.dt * -(right_flux - left_flux) / self.dx 
        assert increment.requires_grad == True
        return increment

    ### Inf norm relative error
    def relative_error(self, precise, coarse):
        # return np.max(np.abs(precise - coarse)) / np.max(np.abs(precise))
        return np.linalg.norm(precise - coarse, 2) / np.linalg.norm(precise, 2)

    ### make evolving animations of the trained network.
    def show(self):
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(2,1,1)
        ax.set_xlim((self.x_low ,self.x_high))
        ax.set_ylim((self.args.plot_y_low, self.args.plot_y_high))
        lineweno, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)] ,lw=2, label = 'weno3')
        linerl, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)],lw=2, label = 'rl')
        lineweno_coarse, = ax.plot(self.x_grid, [0 for _ in range(self.num_x)], lw = 2, label = 'weno_coarse')

        if self.weno_coarse_grid is None:
            self.weno_coarse_grid = np.zeros((self.num_t, self.num_x))

        RLgrid = self.RLgrid.detach().cpu().numpy()
        draw_data = np.zeros((self.num_t, 4 * self.num_x))
        draw_data[:,self.num_x: 2 * self.num_x] = RLgrid
        draw_data[:, self.num_x * 2: self.num_x * 3] = self.weno_coarse_grid[:self.num_t, :]
        for t in range(self.num_t):
            draw_data[t, : self.num_x] = self.get_precise_value(t * self.dt) # when doing showing, use the grid values

        error_ax = fig.add_subplot(2,1,2)
        coarse_error = np.zeros(self.num_t)
        RL_error = np.zeros(self.num_t)
        for i in range(self.num_t):
            coarse_error[i] = self.relative_error(draw_data[i, :self.num_x], draw_data[i, 2 * self.num_x:3*self.num_x])
            RL_error[i] = self.relative_error(draw_data[i, :self.num_x], draw_data[i, self.num_x:2*self.num_x])
        RL_error_line, = error_ax.plot(range(self.num_t), RL_error,  'r', lw= 2, label = 'RL-weno relative error')
        weno_coarse_error_line, = error_ax.plot(range(self.num_t), coarse_error,  'b', lw = 2, label = 'weno relative error')
        RL_error_point, = error_ax.plot([], [], 'ro', markersize = 5)
        weno_coarse_error_point, = error_ax.plot([], [], 'bo', markersize = 5)

        def init():    
            linerl.set_data([], [])
            lineweno.set_data([],[])
            lineweno_coarse.set_data([], [])
            RL_error_point.set_data([],[])
            weno_coarse_error_point.set_data([],[])
            linerl.set_label('RL-weno solution')
            lineweno.set_label('weno solution')
            lineweno_coarse.set_label('reference solution')
            return linerl, lineweno, lineweno_coarse, RL_error_point, weno_coarse_error_point

        def func(i):
            print('make animations, step: ', i)
            x = np.linspace(self.x_low, self.x_high, self.num_x)
            yweno = draw_data[i,:self.num_x]
            yrl = draw_data[i, self.num_x:2 * self.num_x]
            yweno_coarse = draw_data[i, 2 * self.num_x: 3 * self.num_x]
            linerl.set_data(x,yrl)
            lineweno.set_data(x, yweno)
            lineweno_coarse.set_data(x, yweno_coarse)
            RL_error_point.set_data(i, RL_error[i])
            weno_coarse_error_point.set_data(i, coarse_error[i])
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
     
    ### compute the mean squared error compared with the fine-grid solution
    def error(self):
        """
        return the terminal time point MSE error
        """

        rl_values = self.RLgrid[-1].detach().cpu().numpy()
        true_values = self.get_precise_value(self.T)
        error = np.linalg.norm(rl_values - true_values, 2) / np.linalg.norm(true_values, 2)
        if self.args.test:
            print('init condition: ', self.args.init, ' Relative L2 Error is: ', error)
   
        assert true_values is not None
        if self.args.animation:
            self.show()

        return error






