'''
File Description:
This file initializes the training and testing Burgers Environments, set their initial conditions, dx, dt, and etc. 
'''


import numpy as np
import time, os
import math
import copy

from BurgersEnv.Burgers import Burgers
from Weno.weno3_2 import Weno3

### version 4 TODO: if the flux funtion changes and the training initial time is not 0, needs to add
### different flux functions here.

### the burgers flux function. Can be changed to any other functions if needed for future useage.
def flux(val):
    return val ** 2 / 2.

### the deraviative of the flux function.
def flux_deriv( val):
    return val

### the maximum abs value of the deraviative of the flux function.
def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

### This function returns the terminal values on the grids x_center after the inital values u0 has evolved time t.
def evolve(x_center, t, u0):
    dx = x_center[1] - x_center[0]
    cfl = 0.2
    left_boundary = x_center[0] - dx * 0.5
    right_boundary = x_center[-1] + dx * 0.5
    ncells = len(x_center)
    w = Weno3(left_boundary, right_boundary, ncells, flux, flux_deriv, max_flux_deriv, 
            dx = dx, dt = dx * cfl, cfl_number = cfl, record = False)
    return w.integrate(u0, t)
    # num_t = int(t / (dx * cfl))
    # # print('in init.py func evolve, num_t is: ', num_t)
    # return w.u_grid[num_t]

######### Set the Training Environment ######### 
def init_condition_a(x_center, t): # 0
    # u0 = 1 + np.cos(2 * np.pi * x_center)
    u0 = 1 + np.cos(6 * np.pi * x_center)
    return evolve(x_center, t, u0)  

def init_condition_b(x_center, t = 0): # 1
    # u0 = -1 + np.cos(2 * np.pi * x_center)
    u0 = -1 + np.cos(6 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_c(x_center, t = 0): # 2
    # u0 = -1.5 + 2 * np.sin(2 * np.pi * x_center)
    u0 = -1.5 + 2 * np.sin(6 * np.pi * x_center)
    return evolve(x_center, t, u0)
    
def init_condition_d(x_center, t = 0): # 3
    # u0 = 1.5 - 1.5 * np.sin(2 * np.pi * x_center)
    u0 = 1.5 - 1.5 * np.sin(6 * np.pi * x_center)
    return evolve(x_center, t, u0)
    
def init_condition_e(x_center, t = 0): # 4
    # u0 = -1.5 + 2 * np.cos(2 * np.pi * x_center)
    u0 = -1.5 + 2 * np.cos(6 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_f(x_center, t = 0): # 5
    # u0 = 1.5 - 1.5 * np.cos(2 * np.pi * x_center)
    u0 = 1.5 - 1.5 * np.cos(6 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_g(x_center, t = 0): # 6
    u0 = np.zeros(len(x_center))
    u0[x_center < -0.5] = 2
    u0[x_center > 0.5] = 2
    return evolve(x_center, t, u0)

######### Set the Test Environment ################

def init_condition_h(x_center, t = 0): # 7
    u0 = 0.5 +  np.sin(2 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_i(x_center, t = 0): # 8
    u0 = -1 + np.sin(2 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_j(x_center, t = 0): # 9
    u0 = -1 + 2.5 * np.cos(4 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_k(x_center, t = 0): # 10
    u0 = 0.2 - 2 * np.sin(4 * np.pi * x_center)
    return evolve(x_center, t, u0)

def init_condition_l(x_center, t = 0): # 9
    u0 = np.ones(len(x_center))
    u0[x_center > 0] = 2.5
    return evolve(x_center, t, u0)

init_funcs = [init_condition_a, init_condition_b, init_condition_c, init_condition_d, 
    init_condition_e, init_condition_f, init_condition_g, init_condition_h, init_condition_i, init_condition_j, 
    init_condition_k, init_condition_l]

### version 4 delete
def true_solution_c(x, t):
    return 1 if x < t /2.0 else 0

def init_env(args, agent = None):
    '''
    This function initializes and returns the training and test Burgers environments.
    1) hand-set the initial function name, the solution plot y-axis limit, the training env grid size dx, evolving time T,
        and choose the training env idxes.
    2) For each enviroment, load pre-stored solutions or compute and store the solutions.  The solutions include the precise 
        solutions (computed using weno with dense grids), and the weno solutions computed under the same grid size as the RL  
        agent.

    ### Arguments:
    args (python namespace variable):
        A namespace variable that stores all necessary parameters for the whole training procedure.
    agent (RL agent object, optional):
        A RL agent. It will be passed to the Burgers Env object, and will be used when the temporal scheme is RK4.

    ### Return:
    argss (list of python namespace variables):
        A list that contains the args of each training/test environment. argss[i] is almost the same as the input args, except 
        these domains are modified: .init, .dx, .dt, .T, for training purposes.
    train_env (list of class Burgers objects):
        A list of the well-initialized Burgers Training environments. Each has different initial conditions, dx, dt, T.
    test_env (list of class Burgers objects):
        A list of the well-initialized Burgers Test environments. Each has different initial conditions, but the same dx, dt, T
        as the command line argument.
    '''
    argss = [copy.copy(args) for i in range(15)]

    ### set the name of the initial conditions of each training/test environment.
    argss[0].init = '1;1;cos;6'#'2_2cos2'##'0.5_m2cos4'
    argss[1].init = '-1;1;cos;6'#'m1_m3sin2'##'-1_p2sin4'
    argss[2].init = '-1.5;2;sin;6'
    argss[3].init = '1.5;-1.5;sin;6'
    argss[4].init = '-1.5;2;cos;6'
    argss[5].init = '1.5;-1.5;cos;6'
    argss[6].init = 'twobreak'

    argss[7].init = '0.5_sin2'
    argss[8].init = '-1_sin2'
    argss[9].init = '-1_2.5cos4'
    argss[10].init = '0.2_m2sin4'
    argss[11].init = 'rarefraction'

    ### set the y-axis limit when plotting the solution.
    argss[0].plot_y_low, argss[1].plot_y_low, argss[2].plot_y_low, argss[3].plot_y_low = -2, -4.5, -4, -0.5 
    argss[4].plot_y_low, argss[5].plot_y_low, argss[6].plot_y_low, argss[7].plot_y_low = -5, -1.5, -1, -6
    argss[8].plot_y_low, argss[9].plot_y_low, argss[10].plot_y_low = -1.5, 0, -2.5

    argss[0].plot_y_high, argss[1].plot_y_high, argss[2].plot_y_high, argss[3].plot_y_high = 3, 1, 1, 3.5 
    argss[4].plot_y_high, argss[5].plot_y_high, argss[6].plot_y_high, argss[7].plot_y_high = 1, 5.5, 3, 1
    argss[8].plot_y_high, argss[9].plot_y_high, argss[10].plot_y_high = 3.5, 4, 0.5
    
    train_env = []
    test_env = []
    
    if not os.path.exists('../weno_solutions/'):
        os.makedirs('../weno_solutions/')

    ### set the test environments
    ### version 4 TODO: add the test environment idxes, similar to the train_idxes.
    for i in range(args.num_test):
        argss[i].initial_t = args.initial_t
        argss[i].T = args.T
        if args.test: ### at testing, set grid size following the command line args
            argss[i].dx = args.dx
            argss[i].dt = args.dx * args.cfl
        test_env.append(Burgers(args = argss[i], init_func=init_funcs[i], agent = agent))

        ### first try to load the pre-stored solutions if there exist, otherwise compute and store the solutions.
        # precise solutions
        try:
            dense_solutions = np.load('../weno_solutions/{}-precise-{}-{}.npy'.format(argss[i].init, args.flux, args.cfl))
            precise_num_t = int(argss[i].T / (argss[i].precise_dx * args.cfl)) + 1
            test_env[-1].precise_weno_solutions = dense_solutions[:precise_num_t]
        except FileNotFoundError:
            print('{} build precise weno solutions, flux {} cfl {}'.format(argss[i].init, args.flux, args.cfl))
            test_env[-1].get_weno_precise() 
            # test_env[-1].save_weno_precise() ### version 4 TODO, move the save parts in Burgers Env here.
            np.save('../weno_solutions/{}-precise-{}-{}'.format(argss[i].init, args.flux, args.cfl), 
                test_env[-1].precise_weno_solutions)

        # weno solutions with the same grid size
        try:
            coarse_solutions = np.load('../weno_solutions/{}-coarse-{}-{}-{}-{}.npy'.format(argss[i].init, args.Tscheme, 
                args.dx, args.flux, args.cfl))
            coarse_num_t = int(argss[i].T / (argss[i].dx * args.cfl)) + 1
            test_env[-1].weno_coarse_grid = coarse_solutions[:coarse_num_t]
        except FileNotFoundError:
            print('{} build coarse weno solutions with time scheme {} dx {} flux {} cfl {}'.format(argss[i].init, args.Tscheme, 
                args.dx, args.flux, args.cfl))
            test_env[-1].get_weno_corase()
            # test_env[-1].save_weno_coarse() ### version 4 TODO, move the save parts in Burgers Env here.
            np.save('../weno_solutions/{}-coarse-{}-{}-{}-{}'.format(argss[i].init, args.Tscheme, args.dx, 
                args.flux, args.cfl), test_env[-1].weno_coarse_grid)

    ### For training environment, individually set the grid size for each initial condition
    argss[0].T, argss[1].T, argss[2].T, argss[3].T, argss[4].T, argss[5].T, argss[6].T = 0.8, \
         0.8, 0.8, 0.8, 0.8, 0.8, 0.8
    argss[7].T, argss[8].T, argss[9].T, argss[10].T = 0.4, 0.4, 0.4, 0.4   
    argss[0].dx, argss[1].dx, argss[2].dx, argss[3].dx, argss[4].dx, argss[5].dx, argss[6].dx = 0.02, \
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02
    argss[7].dx, argss[8].dx, argss[9].dx, argss[10].dx = 0.02, 0.02, 0.02, 0.02
    for i in range(len(init_funcs)):
        argss[i].dt = argss[i].dx * args.cfl

    ### set the training environment
    if not args.test:
        train_idxes = [0, 1, 2, 3, 4 ,5] # hand-choose the training environment idxes.
        for i in train_idxes:
            train_env.append(Burgers(args = argss[i], init_func=init_funcs[i], agent = agent))

            ### load the solutions if they are pre-computed and stored, otherwise compute and store them.
            try:
                dense_solutions = np.load('../weno_solutions/{}-precise-{}-{}.npy'.format(argss[i].init, args.flux, args.cfl))
                precise_num_t = int(argss[i].T / (argss[i].precise_dx * args.cfl)) + 1
                train_env[-1].precise_weno_solutions = dense_solutions[:precise_num_t]
            except FileNotFoundError:
                train_env[-1].get_weno_precise()

            try:
                coarse_solutions = np.load('../weno_solutions/{}-coarse-{}-{}-{}-{}'.format(argss[i].init, args.Tscheme, 
                    args.dx, args.flux, args.cfl))
                coarse_num_t = int(argss[i].T / (argss[i].dx * args.cfl)) + 1
                train_env[-1].weno_coarse_grid = coarse_solutions[:coarse_num_t]
            except FileNotFoundError:
                train_env[-1].get_weno_corase()
            
    args.num_train = len(train_env)
    return argss, train_env, test_env
