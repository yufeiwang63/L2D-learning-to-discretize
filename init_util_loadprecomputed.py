'''
File Description:
Load all pre-computed solutions in dir ../weno_solutions/
and generate the corresponding Burgers training envs. 
'''


import numpy as np
import time
import math
import copy, os
from BurgersEnv.Burgers import Burgers
from Weno.weno3_2 import Weno3

def construct_init_condition(a, b, func, c):
    # l = filename.split(';')
    # a = float(l[0])
    # b = float(l[1])
    # func = np.sin if l[2] == 'sin' else np.cos
    # c = int(l[3][:-4])
    def init(x, t):
        return a + b * func(c * np.pi * x)
    return init    

def exact_init_func(a, b, func, c):
    def init(x):
        return a + b * func(c * np.pi * x)
    return init

def exact_init_func_prime(a, b, func, c):
    func_prime = np.cos if func == np.sin else -np.sin
    def init_prime(x):
        return b * func_prime(c * np.pi * x) * c * np.pi
    return init_prime

def exact_init_func_prime_prime(a, b, func, c):
    func_prime_prime = np.sin if func == np.sin else np.cos
    def init_prime_prime(x):
        return b * -func_prime_prime(c * np.pi * x) * c**2 * np.pi ** 2
    return init_prime_prime


def init_env(args, agent = None):
    '''
    This function loads the precomputed solutions in '../weno_solutions' and build the corresponding training/testing environments.
    Arg args(python namespace): storing all necessary arguments for the whole training procedure.
    Arg agent(class DDPG/DQN/etc object): needed for RK4 temporal scheme
    version 4 TODO: read different fluxes.
    '''

    solution_dir = '../weno_solutions/'
    solution_files = os.listdir(solution_dir)
    assert(len(solution_files) % 2 == 0)
    train_num = len(solution_files)
    print('train_num: ', train_num)

    argss = [copy.copy(args) for i in range(train_num)]
    train_env = []
    test_env = []

    ### traverse all the solution files and build corresponding training/testing environment
    i = 0
    while(i < train_num):
        print('load file No. ', i // 2)
        print('precise solution file: ', solution_files[i+1])
        print('weno coarse solution file: ', solution_files[i])

        precise_solution_file = solution_files[i+1]
        weno_solution_file = solution_files[i]

        precise_init_end = precise_solution_file.find('-precise')
        init = precise_solution_file[:precise_init_end]
        l = init.split(';')
        a = float(l[0])
        b = float(l[1])
        func = np.sin if l[2] == 'sin' else np.cos
        c = int(l[3])
        argss[i].init = '{0} + {1}{2}({3} * \\pi * x)'.format(round(a,3), round(b,3), l[2], c)
        print(argss[i].init)

        ### different mode has different level of training difficulty, thus use different T.
        if args.mode == 'eno':
            argss[i].T = 0.5 
        elif args.mode == 'continuous_filter':
            argss[i].T = 0.2
        elif args.mode == 'compute_flux':
            argss[i].T = 0.3
        elif args.mode == 'weno_coef':
            argss[i].T = 0.3
        elif args.mode == 'weno_coef_four':
            argss[i].T = 0.8

        ### training envs evolving steps
        argss[i].dt = argss[i].dx * args.cfl
        precise_num_t = int(argss[i].T / (args.precise_dx * args.cfl)) + 10
        num_t = int(argss[i].T / argss[i].dt) + 10
        
        ### build the training envs
        init_condition = construct_init_condition(a, b, func, c)
        precise_solution = np.load(solution_dir + precise_solution_file)
        weno_solution = np.load(solution_dir + weno_solution_file)
        train_env.append(Burgers(args = argss[i], init_func=init_condition, agent = agent))
        train_env[-1].precise_weno_solutions = precise_solution[:precise_num_t]
        train_env[-1].weno_coarse_grid = weno_solution[:num_t]

        ### build the test envs
        precise_num_t = int(args.T / (args.precise_dx * args.cfl)) + 10
        num_t = int(args.T / argss[i].dt) + 10
        test_env.append(Burgers(args = argss[i], init_func=init_condition, agent = agent))
        test_env[-1].precise_weno_solutions = precise_solution[:precise_num_t]
        test_env[-1].weno_coarse_grid = weno_solution[:num_t]

        ### incresement 2, 1 for precise solution, 1 for weno solution under coarse grid.
        i += 2

    ### .mp4 stroing needs to remove special characters in the file name
    if args.animation:
        for x in test_env:
            init = x.args.init
            new_init = ''
            for s in init:
                if s == ' ' or s == '\\' or s == '(' or s == ')':
                    pass
                elif s == '+':
                    new_init += 'p'
                elif s == '-':
                    new_init += 'm'
                elif s == '*':
                    new_init += '_'
                else:
                    new_init += s
            print(new_init)
            x.args.init = new_init

    args.num_train = len(train_env)
    args.num_test = len(test_env)
    return argss, train_env, test_env

if __name__ == '__main__':
    from get_args import get_args
    args, _ = get_args()
    init_env(args)
    