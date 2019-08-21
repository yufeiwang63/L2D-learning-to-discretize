import numpy as np
import sys, argparse
from Weno.finite_difference_weno import weno3_fd
from Weno.weno3_2 import Weno3
from matplotlib import pyplot as plt

def flux(val):
    return val ** 2 / 2.


def flux_deriv(val):
    return val

def max_flux_deriv(a, b):
    return np.maximum(np.abs(a), np.abs(b))

def init_condition(x):
    return 0.5 + 2 * np.sin(2 * np.pi * x)

def get_weno_grid(x_low, x_high, dx, dt,  T):
    left_boundary = x_low - dx * 0.5
    right_boundary = x_high + dx * 0.5
    ncells = int((x_high - x_low) / dx + 1.) # need to be very small
    num_t = int(T/dt + 1.)
    w = Weno3(left_boundary, right_boundary, ncells, flux, flux_deriv, max_flux_deriv, 
        dx = dx, dt = dt, num_t = num_t + 100)
    x_center = w.get_x_center()
    u0 = init_condition(x_center)
    w.integrate(u0, T)
    solutions = w.u_grid[:num_t,:]
    return solutions

def subsample_precise_value(precise_val, factor, num_x):
    ret = []
    for i in range(num_x):
        ret.append(precise_val[i * factor]) 
    return np.array(ret)

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument('--dx', default = 0.02, type = float)
args.add_argument('--cfl', default = 0.4, type = float)
args.add_argument('--T', default = 1.2, type = float)
args.add_argument('--flux', default = 'u2', type = str)
args.add_argument('--Tscheme', default = 'euler', type = str)
args.add_argument('--x_low', default = -1, type = float)
args.add_argument('--x_high', default = 1, type = float)
args = args.parse_args()


precise_dx = 0.001
precise_dt = 0.0002
num_x = int((args.x_high - args.x_low) / args.dx + 1)
reference_solution = get_weno_grid(args.x_low, args.x_high, precise_dx, precise_dt, args.T)[-1]
reference_solution = subsample_precise_value(reference_solution, int(args.dx / precise_dx), num_x)

init_x = np.linspace(args.x_low, args.x_high, num_x)
init_u = init_condition(init_x)
for dt in [0.01, 0.008, 0.006, 0.005, 0.004, 0.002, 0.001]:
    args.cfl = dt / args.dx
    solution = weno3_fd(args, init_u).solve()[-1]
    # solution1 = get_weno_grid(args.x_low, args.x_high, args.dx, dt, args.T)[-1]
    # plt.plot(solution1, 'ro', label = 'my weno')
    # plt.plot(solution2, 'b*', label = 'fe weno')
    # plt.plot(reference_solution, 'y+', label = 'reference')
    # plt.legend()
    # plt.show()
    error = np.linalg.norm(reference_solution - solution, 2) / np.linalg.norm(reference_solution, 2)
    print("dt: {}, error {}: ".format(dt, error))






