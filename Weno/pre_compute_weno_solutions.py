'''
File Description:
randomly generate 40 initial conditions in the form 
''a + b sin or cos(c * \\pi * x), c = {6}''
pre-compute and store the solutions in the dir ../weno_solutions/.
Name conventions for precise solutions:
    '../weno_solutions/{}-precise-{}-{}.npy'.format(init_condition_name, flux_name, cfl_value)
Name conventions for weno solutions under coarse grid:
    '../weno_solutions/{}-coarse-{}-{}-{}-{}'.format(init_condition_name, Tscheme, dx, 
                flux_name, cfl_value)
'''

import numpy as np
import time
import math
import copy
from Weno.weno3_2 import Weno3, weno3_fv
from Weno.finite_difference_weno import weno3_fd

def random_init_condition():
    '''
    randomly generate initial conditions in the following form
    a + b * func(c * \\pi * x).   
    c = 6; to guarantee enough samples with discontinuty.
    func = {sin, cos}
    |a| + |b| <= 4
    -2 <= a <= 2
    -2 <= b <= 2
    '''
    a = 4 * np.random.random_sample() + -2
    b = min(4-a, 4 + a) if np.random.rand() < 0.5 else max(-4-a, -4 + a) ### make |a| + |b| = 4
    c = 6
    func = np.random.choice([np.sin, np.cos])

    return [a, b, func, c]

if __name__ == '__main__':

    import argparse, sys
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument('--x_low', type = float, default = -1)
    args.add_argument('--x_high', type = float, default = 1)
    args.add_argument('--dx', type = float, default = 0.02)
    args.add_argument('--cfl', type = float, default = 0.2)
    args.add_argument('--T', type = float, default = 1.10)
    args.add_argument('--precise_dx', type = float, default = 0.001)
    args.add_argument('--initial_t', type = float, default = 0.)
    args.add_argument('--num', type = int, default = 5)
    args.add_argument('--Tscheme', type = str, default = 'rk4')
    args.add_argument('--flux', type = str, default = 'u3')
    args = args.parse_args()

    np.random.seed(233)

    for i in range(args.num):
        print('iter: {0}'.format(i))
        a, b, func, c = random_init_condition()

        ### compute and store precise solutions
        fine_num_x = int((args.x_high - args.x_low) / args.precise_dx + 1)
        fine_x_grid = np.linspace(args.x_low, args.x_high, fine_num_x, dtype = np.float64) ### precise_dx = 0.001        
        u0 = a + b*func(c * np.pi * fine_x_grid)
        fine_solver = weno3_fv(args.flux)
        fine_solution = fine_solver.solve(fine_x_grid, args.T, u0, args.cfl)

        ### compute and store weno solutions under coarse grids
        coarse_num_x = int((args.x_high - args.x_low) / args.dx + 1)
        coarse_x_grid = np.linspace(args.x_low, args.x_high, coarse_num_x, dtype = np.float64) ### precise_dx = 0.001
        init_value = a + b*func(c * np.pi * coarse_x_grid)
        coarse_solver = weno3_fd(args, init_value=init_value)
        corase_solution = coarse_solver.solve()

        ### store the computed solutions
        func_name = 'sin' if func == np.sin else 'cos'
        init_name = '{0};{1};{2};{3}'.format(a, b, func_name, c)
        print('{0} + {1}{2}({3}\\pi * x)'.format(a, b, func_name, c))
        np.save(file = '../weno_solutions/{}-precise-{}-{}.npy'.format(init_name, args.flux, args.cfl), arr = fine_solution)
        np.save(file = '../weno_solutions/{}-coarse-{}-{}-{}-{}'.format(init_name, args.Tscheme, args.dx, 
                args.flux, args.cfl), arr = corase_solution)