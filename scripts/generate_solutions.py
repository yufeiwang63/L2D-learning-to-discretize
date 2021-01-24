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
import os 
import os.path as osp
import torch
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy import interpolate
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def random_init_condition():
    '''
    randomly generate initial conditions in the following form
    a + b * np.sin(c * pi * x) + d * np.cos(e * pi * x).
    c, e sampled from [2, 4, 6]   
    |a| + |b| + |d| = 4
    '''
    a = np.random.uniform(-1.2, 1.2)
    b = np.random.uniform(-3 + np.abs(a), 3 - np.abs(a))
    abs_d = 4 - np.abs(a) - np.abs(b)
    if np.random.rand() > 0.5:
        d = abs_d
    else:
        d = -abs_d
    
    # print(a, b, d)
    c = np.random.choice([4, 6])
    e = np.random.choice([4, 6])

    return a, b, c, d, e

def get_precise_value(reference_solution, precise_dt, T, precise_x_grid, corase_x_grid):
    t_idx = int(T / precise_dt)
    precise_val = reference_solution[t_idx]
    f = interpolate.interp1d(precise_x_grid, precise_val)
    ret = f(corase_x_grid)
    return ret

def relative_error(ref, sol, norm=2):
    return np.linalg.norm(ref - sol, norm) / np.linalg.norm(ref, norm)


def show(args, weno_coarse_rk4, reference_solution, dx=None, save_path=None, save_name=None, title=None):
    fig = plt.figure(figsize = (15, 10))
    ax = fig.add_subplot(2,1,1)
    ax.set_xlim((args.x_low ,args.x_high))
    # ymin, ymax = np.min(weno_coarse_rk4[0]) - 0.1, np.max(weno_coarse_rk4[0]) + 0.1
    ymin, ymax = np.min(reference_solution) - 0.1, np.max(reference_solution) + 0.1
    ax.set_ylim((ymin, ymax))

    num_x = len(weno_coarse_rk4[0])
    x_grid = np.linspace(args.x_low, args.x_high, num_x, dtype = np.float64)
    lineweno, = ax.plot(x_grid, [0 for _ in range(num_x)] ,lw=2, label = 'reference')
    lineweno_coarse, = ax.plot(x_grid, [0 for _ in range(num_x)], lw = 2, label = 'WENO')

    dt = dx * args.cfl
    num_t = int(args.T / dt) + 1 - 10
    draw_data = np.zeros((num_t, 2*num_x))
    draw_data[:, num_x:num_x*2] = weno_coarse_rk4[:num_t, :]

    # factor = int(dx / args.precise_dx)
    # print(f"dt {dt} factor {factor} precise_dx {args.precise_dx} cfl {args.cfl}")
    precise_x_grid = np.linspace(args.x_low, args.x_high, len(reference_solution[0]))
    for t in range(num_t):
        draw_data[t, :num_x] = get_precise_value(reference_solution, args.precise_dx * args.cfl, t * dt, precise_x_grid, x_grid) # when doing showing, use the grid values

    error_ax = fig.add_subplot(2,1,2)
    coarse_error = np.zeros(num_t)
    for i in range(num_t):
        coarse_error[i] = relative_error(draw_data[i, :num_x], draw_data[i, num_x:2*num_x])
    weno_coarse_error_line, = error_ax.plot(range(num_t), coarse_error,  'b', lw = 2, label = 'weno_relative_error')
    weno_coarse_error_point, = error_ax.plot([], [], 'bo', markersize = 5)

    def init():    
        lineweno.set_data([],[])
        lineweno_coarse.set_data([], [])
        weno_coarse_error_point.set_data([],[])
        lineweno.set_label('Reference')
        lineweno_coarse.set_label('WENO')
        return lineweno, lineweno_coarse, weno_coarse_error_point

    def func(i):
        # print('make animations, step: ', i)
        x = np.linspace(args.x_low, args.x_high, num_x)
        yweno = draw_data[i,:num_x]
        yweno_coarse = draw_data[i, num_x:2*num_x]
        lineweno.set_data(x, yweno)
        lineweno_coarse.set_data(x, yweno_coarse)
        weno_coarse_error_point.set_data(i, coarse_error[i])
        return lineweno, lineweno_coarse, weno_coarse_error_point

    ax.grid()
    anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=num_t, interval=50)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        anim.save(osp.join(save_path, save_name), writer=writer)
    else:
        plt.show()

    plt.close()


def get_weno_grid(init_condition, dx = 0.02, dt = 0.004,  T = 0.8, x_low = -1, x_high = 1,
            boundary='periodic', eta=0, forcing=None, ncells=None, num_t=None, flux_name='u2'):
        """
        dx: grid size. e.g. 0.02
        dt: time step size. e.g. 0.004
        T: evolve time. e.g. 1
        x_low: if the grid is in the range [-1,1], then x_low = -1.
        x_high: if the grid is in the range [-1,1], then x_low = 1.
        init_condition: a function that computes the initial values of u. e.g. the init func above.

        return: a grid of size num_t x num_x
            where   num_t = T/dt + 1
                    num_x = (x_high - x_low) / dx + 1
        """
        left_boundary = x_low - dx * 0.5
        right_boundary = x_high + dx * 0.5
        if ncells is None:
            ncells = int((x_high - x_low) / dx + 1.) # need to be very small
        if num_t is None:
            num_t = int(T/dt + 1.)
        w = Weno3(left_boundary, right_boundary, ncells, 
            lambda x: flux(val=x, flux=flux_name), 
            lambda x: flux_deriv(val=x, flux=flux_name), 
            lambda a, b: max_flux_deriv(a=a, b=b, flux=flux_name), 
            dx = dx, dt = dt, num_t = num_t + 100, boundary=boundary, 
            eta=eta, forcing=forcing)
        x_center = w.get_x_center()
        u0 = init_condition(x_center)
        w.integrate(u0, T, num_t=num_t)
        solutions = w.u_grid[:num_t,:]
        return solutions

### the burgers flux function. Can be changed to any other functions if needed for future useage.
def flux(val, flux='u2'):
    if flux == 'u2':
        return val ** 2 / 2.
    elif flux == 'u4': 
        return val ** 4 / 16.
    elif flux == 'u3':
        return val ** 3 / 9.
    elif flux == 'BL':
        return val ** 2 / (val ** 2 + 0.5 * (1-val) ** 2)
    elif flux.startswith("linear"):
        a = float(flux[len('linear'):])
        return a * val

def flux_deriv(val, flux='u2'):
    if flux == 'u2':
        return val
    elif flux == 'u4':
        return val ** 3 / 4.
    elif flux == 'u3':
        return val ** 2 / 3
    elif flux == 'BL':
        return (val - 3 * val**2) / (val ** 2 + 0.5 * (1 - val)**2) ** 2
    elif flux.startswith("linear"):
        a = float(flux[len('linear'):])
        return a

def max_flux_deriv(a, b, flux='u2'):
    if flux == 'u2':
        return np.maximum(np.abs(a), np.abs(b))
    elif flux == 'u4':
        return np.maximum(np.abs(a ** 3 / 4.), np.abs(b ** 3 / 4.))
    elif flux == 'u3':
        return np.maximum(np.abs(a ** 2 / 3.), np.abs(b ** 2 / 3.))
    elif flux == 'BL':
        a = (a - 3 * a**2) / (a ** 2 + 0.5 * (1 - a)**2) ** 2
        b = (a - 3 * b**2) / (b ** 2 + 0.5 * (1 - b)**2) ** 2
        return np.maximum(np.abs(a), np.abs(b))
    elif flux.startswith("linear"):
        a = float(flux[len('linear'):])
        return np.abs(a)

def get_init_func(a, b, c, d, e):
    def func(x, t=0):
        return a + b * np.sin(c * np.pi * x) + d * np.cos(np.pi * e * x)
    return func

def get_forcing():
    nparams = 20
    rs = np.random
    k_min = 3
    k_max = 6
    a = 0.5 * rs.uniform(-1, 1, size=(20, 1))
    omega = rs.uniform(-0.4, 0.4, size=(nparams, 1))
    k_values = np.arange(k_min, k_max + 1)
    k = rs.choice(np.concatenate([-k_values, k_values]), size=(nparams, 1))
    phi = rs.uniform(0, 2 * np.pi, size=(nparams, 1))

    def func(x, t, period):
        spatial_phase = (2 * np.pi * k * x / period)
        signals = np.sin(omega * t + spatial_phase + phi)
        reference_forcing = np.sum(a * signals, axis=0)
        return reference_forcing

    forcing_params = [a, omega, k, phi]

    return func, forcing_params


if __name__ == '__main__':

    import argparse, sys
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument('--prefix', type = str, default = '9-15-eta-0')
    args.add_argument('--x_low', type = float, default = -1)
    args.add_argument('--x_high', type = float, default = 1)
    args.add_argument('--dx', type = float, default = 0.02)
    args.add_argument('--cfl', type = float, default = 0.1)
    args.add_argument('--T', type = float, default = 1.0)
    args.add_argument('--precise_dx', type = float, default = 0.002)
    args.add_argument('--initial_t', type = float, default = 0.)
    args.add_argument('--num', type = int, default = 20)
    args.add_argument('--flux', type = str, default = 'u2')
    args.add_argument('--Tscheme', type = str, default = 'None')
    args.add_argument('--eta', type = float, default = 0)
    args.add_argument('--forcing', type = int, default = 0)
    args = args.parse_args()

    if args.forcing:
        args.x_low = 0
        args.x_high = 2 * np.pi
        args.precise_dx = args.x_high / 1000.
        args.T = np.pi

    precise_num_t = 5000

    if args.flux == 'u2':
        np.random.seed(666)
    else:
        np.random.seed(555)

    for i in range(53, args.num + 53):
        print('init condition {}'.format(i))
        a, b, c, d, e = random_init_condition()
        print(a, b, c, d, e)

        forcing = None
        if args.forcing:
            forcing, forcing_params = get_forcing()
            a, b, d = 0, 0, 0 # make the initial condition always being 0.

        init_func = get_init_func(a, b, c, d, e)

        ### compute and store precise solutions
        fine_num_x = 1001
        fine_x_grid = np.linspace(args.x_low, args.x_high, fine_num_x, dtype = np.float64) ### precise_dx = 0.001        
        fine_solution = get_weno_grid(init_func, dx=args.precise_dx, dt=args.precise_dx * args.cfl, 
            T=args.T, x_low=args.x_low, x_high=args.x_high, eta=args.eta, forcing=forcing, 
            ncells=fine_num_x, num_t=precise_num_t, flux_name=args.flux)

        ### compute and store weno solutions under coarse grids
        dx_list = [0.05, 0.04, 0.02]
        corase_num_x_list = [41, 51, 101]
        corase_num_t_list = [200, 250, 500]
        weno_coarse_solutions = {'rk4': {}, 'euler': {}}
        for dx_idx, dx in enumerate(dx_list):
            if args.forcing:
                dx *= np.pi

            print('dx: ', dx)
            args.dx = dx
            for tscheme in ['rk4', 'euler']:
                args.Tscheme = tscheme
                coarse_num_x = corase_num_x_list[dx_idx]
                corase_num_t = corase_num_t_list[dx_idx]
                coarse_x_grid = np.linspace(args.x_low, args.x_high, coarse_num_x, dtype = np.float64) ### precise_dx = 0.001
                
                init_value = a + b * np.sin(c * np.pi * coarse_x_grid) + d * np.cos(e * np.pi * coarse_x_grid)
                coarse_solver = weno3_fd(args, init_value=init_value, forcing=forcing, num_x=coarse_num_x, num_t=corase_num_t)
                corase_solution = coarse_solver.solve()
                
                weno_coarse_solutions[tscheme][str(round(dx, 2))] = corase_solution

                # num_t = int(args.T / (dx * args.cfl)) - 10
                # factor = int(dx / args.precise_dx)
                # for idx_t in range(num_t):
                #     fig = plt.figure(figsize=(15, 7))
                #     plt.grid()
                #     plt.plot(coarse_x_grid, weno_coarse_solutions[1][idx_t], 'bo-')
                #     plt.plot(fine_x_grid, fine_solution[idx_t * int(dx / args.precise_dx)], 'r')
                #     plt.show()

       
        ### store the computed solutions
        prefix = args.prefix
        if not osp.exists('./data/local/solutions/{}'.format(prefix)):
            os.makedirs('data/local/solutions/{}'.format(prefix), exist_ok=True)

        # for dx in dx_list:
        #     if args.forcing:
        #         dx = dx * np.pi
        #     show(args, weno_coarse_solutions['rk4'][str(round(dx, 2))], fine_solution, dx=dx, title=str(i), 
        #         save_path='data/local/solutions/{}'.format(prefix), save_name="{}-{}.mp4".format(i, dx))

        data = {
            'precise_solution': fine_solution,
            'coarse_solution_rk4': weno_coarse_solutions['rk4'],
            'coarse_solution_euler': weno_coarse_solutions['euler'],
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'e': e,
            'args': args        
        }

        if args.forcing:
            data['forcing'] = forcing_params

        torch.save(data, 'data/local/solutions/{}/{}.pkl'.format(prefix, i))