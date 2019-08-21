import numpy as np
from Burgers import Burgers
import time
import math
import copy
from weno3_2 import Weno3

'''
### NO Longer Used. 2019.3.19 Marked ###
this initial file is for keep training the 2300_DQN obtained at from Burgers-2018-12-02-22-52-50.
2300_DQN is trained on the 0.04 grids, and has already gained pretty good performances on a set of initial conditions.
But there are still some weakenesses:
1. when the grid is more precise, e.g., 0.02 or 0.01, sometimes it will explode. That is, the numerical statbility is not good enough.
2. On some initial conditions, it still does not capture the movement of the wave. e.g.  u0 = -0.5 + 2 * np.cos(2 * np.pi * x_center)
This retraining aims at solving these two problems.
To do so:
1. train on more precise grids, e.g., 0.02
2. train on the initial conditions it did not perform well enough. e.g. include both  
    u0 = -0.5 + 2 * np.cos(2 * np.pi * x_center) and
    u0 = 0.5 + 2 * np.sin(2 * np.pi * x_center)
'''


### the burgers flux function. Can be changed to any other functions if needed for future useage.
def flux(val):
    # val = np.array(val)
    return val ** 2 / 2.

def flux_deriv( val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

### evolve 
def evolve(x_center, t, u0):
    dx = x_center[1] - x_center[0]
    cfl = 0.25
    left_boundary = x_center[0] - dx * 0.5
    right_boundary = x_center[-1] + dx * 0.5
    ncells = len(x_center)
    # print('in init_util evolve, dx is: ', dx)
    w = Weno3(left_boundary, right_boundary, ncells, flux, flux_deriv, max_flux_deriv, 
            dx = dx, dt = dx * cfl, char_speed = 1.0, cfl_number = cfl, record = False)
    return w.integrate(u0, t)
    # num_t = int(t / (dx * cfl))
    # # print('in init.py func evolve, num_t is: ', num_t)
    # return w.u_grid[num_t]

### intial conditions
def init_condition_a(x_center, t): # 0
    u0 = -0.5 + np.cos(2 * np.pi * x_center)
    return evolve(x_center, t, u0)  

def init_condition_b(x_center, t = 0): # 1
    u0 = 1 + np.sin(2 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_c(x_center, t = 0): # 2
    u0 = np.zeros(len(x_center))
    u0[x_center < -0.25] = 2
    u0[x_center > 0.25] = 2
    return evolve(x_center, t, u0)
    
def init_condition_d(x_center, t = 0): # 3
    u0 = 2 + np.cos(4 * np.pi * x_center)
    return evolve(x_center, t, u0)
    # u0 = 1.0 + 2 * np.exp(-60.0*(x_center - 0.5)**2)
    # return evolve(x_center, t, u0)
    
def init_condition_e(x_center, t = 0): # 4
    u0 = -1.5 + 2 * np.sin(4 * np.pi * x_center)
    # u0 = -0.5 + 2 * np.cos(2 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_f(x_center, t = 0): # 5
    u0 = np.zeros(len(x_center))
    u0[x_center <= 0.25] = x_center[x_center <= 0.25]
    u0[x_center > 0.25] = x_center[x_center > 0.25] - 1.
    return evolve(x_center, t, u0)

def init_condition_g(x_center, t = 0): # 6
    u0 = 2 + np.cos(4 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_h(x_center): # 7
    return 1.5 - 3 * np.cos(np.pi * x_center)

def init_condition_i(x_center): # 8
    return 1 + 2 * np.sin(np.pi * x_center)

def init_condition_j(x_center): # 9
    u0 = np.ones(len(x_center))
    u0[x_center > 0] = 2.5
    return u0

init_funcs = [init_condition_a, init_condition_b, init_condition_c, init_condition_d, 
    init_condition_e, init_condition_f, init_condition_g, init_condition_h, init_condition_i, init_condition_j]

def true_solution_c(x, t):
    return 1 if x < t /2.0 else 0


def init_env(args, agent = None):
    args.num_train, args.num_test = 6, 6
    args.reward_time = 5  ### give more rewards, to make sure the filter selection is better
    args.explore_initial = 0.2
    args.explore_final = 0.01
    # args.explore_decrease = 0.02
    # args.explore_decrease_every = 
    ### make very small explorations, as the network itself is already pretty good

    print('Here for more advanced training')
    argss = [copy.copy(args) for i in range(10)]

    argss[0].T, argss[1].T, argss[2].T, argss[3].T, argss[4].T, argss[5].T, argss[6].T = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 
    argss[0].dx, argss[1].dx, argss[2].dx, argss[3].dx, argss[4].dx, argss[5].dx, argss[6].dx = 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
    # argss[3].x_low, argss[3].x_high = -2, 0

    argss[0].init = '-0.5_cos2'
    argss[1].init = '1_sin2'
    argss[2].init = 'twobreak'
    argss[3].init = 'gaussian'
    argss[4].init = '-1.5_2sin4'
    argss[5].init = 'upright_linear'
    argss[6].init = '2_sin4'

    argss[7].init = '1.5add-3cos'
    argss[8].init = '1add2sin'
    argss[9].init = 'rarefraction'

    train_env = []
    test_env = []
    
    ### for each initial_condition, set different initial time 
    initial_ts = [0,    0.2,  0.4,  0.6, 0.8]
    for i in range(args.num_train):
        for init_t in initial_ts:
            argss[i].initial_t = init_t
            # print('In init.py i is {0}, init_t is {1}'.format(i, init_t))
            train_env.append(Burgers(args = argss[i], init_func=init_funcs[i], agent = agent))

    for i in range(args.num_test):
        argss[i].initial_t = 0.
        argss[i].T = 1.0
        if args.test:
            argss[i].dx = args.dx
            argss[i].dt = args.dt
        test_env.append(Burgers(args = argss[i], init_func=init_funcs[i], agent = agent))

    args.num_train = len(train_env)
    return argss, train_env, test_env