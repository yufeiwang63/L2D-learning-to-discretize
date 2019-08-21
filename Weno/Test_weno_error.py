'''
To compute the error between 0.02 weno grid and 0.001 weno grid, as the baseline.
'''


import numpy as np
import time
import math
import copy
import argparse, sys
from matplotlib import pyplot as plt
from matplotlib import animation
from Weno.weno3_2 import Weno3
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)



### the burgers flux function. Can be changed to any other functions if needed for future useage.
def flux(val):
    # val = np.array(val)
    return val ** 2 / 2.

def flux_deriv( val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

def evolve(x_center, u0, t, animation = 0):
    dx = x_center[1] - x_center[0]
    cfl = 0.2
    left_boundary = x_center[0] - dx * 0.5
    right_boundary = x_center[-1] + dx * 0.5
    ncells = len(x_center)
    dt = dx * cfl
    evolve_num = int(t / dt)
    w = Weno3(left_boundary, right_boundary, ncells, flux, flux_deriv, max_flux_deriv, 
            dx = dx, dt = dx * cfl, cfl_number = cfl)
    w.integrate(u0, t)
    solutions = w.u_grid[:evolve_num + 1,:]
    if animation:
        return solutions
    else:
        return solutions[-1]


def init_condition(x_center): # 0
    u0 = 0.5 + np.cos(4 * np.pi * x_center)
    # u0 = -1 + np.sin(2 * np.pi * x_center)
    return u0 

def make_grid(dx, x_low = -1, x_high = 1):
    num_x = int((x_high - x_low) / dx + 1)
    x_grid = np.linspace(x_low, x_high, num_x, dtype = np.float64)
    return x_grid

def get_precise_value(res_fine, factor, num_x):
    precise_val = res_fine
    ret = []
    for i in range(num_x):
        ret.append(precise_val[i * factor]) 
    return np.array(ret)

def show(corase_grid, fine_grid, args):
    fig = plt.figure(figsize = (15, 5))
    ax = plt.axes(xlim=(-1 , 1),ylim=(-1, 3))
    line_c, = ax.plot([],[],lw=2, label = 'weno {0}'.format(args.c_dx))
    line_f, = ax.plot([],[],lw=2, label = 'weno {0}'.format(args.f_dx))

    cfl = 0.2
    dt = args.c_dx * cfl
    num_t = len(corase_grid)
    num_x = len(corase_grid[0])

    draw_data = np.zeros((num_t, 2 * num_x))
    draw_data[:,num_x:] = corase_grid
    for t in range(num_t):
        # print(t)
        precise_t_idx = int(t *  args.c_dx / args.f_dx)
        factor = int(args.c_dx / args.f_dx)
        draw_data[t, : num_x] = get_precise_value(fine_grid[precise_t_idx], factor, num_x) # when doing showing, use the grid values

    def init():    
        line_c.set_data([], [])
        line_f.set_data([],[])
        line_c.set_label('corase weno solution')
        line_f.set_label('fine-grid weno solution')
        return line_c, line_f


    def func(i):
        print(i)
        x = np.linspace(-1, 1, num_x)
        y_c = draw_data[i,num_x:]
        y_f = draw_data[i, :num_x]
        line_c.set_data(x, y_c)
        line_f.set_data(x, y_f)
        return line_c, line_f

    anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=num_t, interval=50)
    plt.legend()
    plt.title('init {0}\n coarse_dx {1} fine_grid dx {2} T {3} cfl {4}'.format('0.5 + cos2', 
        args.c_dx, args.f_dx, args.T, 0.2))
    plt.tight_layout()
    if args.save_animation:
        anim.save('../WenoAnimations/' + '{0}_corase_dx{1}_finegrid_dx{2}_T{3}_cfl{4}.mp4'.format('0.5 + cos2', 
            args.c_dx, args.f_dx, args.T,  0.2), writer=writer)
    elif args.show_animation:
        plt.show()
    
    plt.close()

def get_weno_error(c_dx, f_dx, T, init_c):
    corase_grid = make_grid(c_dx)
    fine_grid = make_grid(f_dx)
    res_corase = evolve(corase_grid, init_c(corase_grid), T)
    print('corase weno evolve over')
    res_fine = evolve(fine_grid, init_c(fine_grid), T)
    print('fine-grid weno evolve over')
    res_fine_subsample = get_precise_value(res_fine, int(c_dx / f_dx), len(corase_grid))

    return np.mean((res_corase - res_fine_subsample)**2)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument('--T', type = float, default = 1)
    args.add_argument('--c_dx', type = float, default = 0.02)
    args.add_argument('--f_dx', type = float, default = 0.001)
    args.add_argument('--animation', type = int, default = 0)
    args.add_argument('--save_animation', type = int, default = 0)
    args.add_argument('--show_animation', type = int, default = 0)
    args = args.parse_args()
    corase_grid = make_grid(args.c_dx)
    fine_grid = make_grid(args.f_dx)   
    
    if not args.animation:
        res_corase = evolve(corase_grid, init_condition(corase_grid), args.T)
        res_fine = evolve(fine_grid, init_condition(fine_grid), args.T)
        res_fine_subsample = get_precise_value(res_fine, int(args.c_dx / args.f_dx), len(corase_grid))

        error = np.mean((res_corase - res_fine_subsample)**2)
        print('Weno corase and fine error is: ', error)
    else:
        res_corase = evolve(corase_grid, init_condition(corase_grid), args.T, 1)
        res_fine = evolve(fine_grid, init_condition(fine_grid), args.T, 1)
        res_fine_subsample = get_precise_value(res_fine[-1], int(args.c_dx / args.f_dx), len(corase_grid))

        error = np.mean((res_corase[-1] - res_fine_subsample)**2)
        print('Weno corase and fine error is: ', error)

        show(res_corase, res_fine, args)
