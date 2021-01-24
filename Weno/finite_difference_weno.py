'''
File Description:
This file contains the class weno3_fd, which implements the 5-th order finite difference WENO.
See the bottom for sample usage.
'''

import numpy as np 
import copy
from matplotlib import pyplot as plt

class weno3_fd():
    '''
    This class implements the finite difference 5-th order WENO scheme.
    Arg args (python namespace):
        should contain the following domain: x_high, x_low, dx, cfl, T, flux, Tscheme
    Arg init_value (np array):
        specifies the initial value.
    '''
    def __init__(self,  args, init_value=None, forcing=None, num_x=None, num_t=None, dt=None, dx=None):
     
        self.args = copy.copy(args)
        self.x_high, self.x_low = args.x_high, args.x_low
        self.dx, self.dt, self.T = args.dx, args.dx * args.cfl, args.T
        if dx is not None:
            self.dx = dx
        if dt is not None:
            self.dt = dt
        if num_x is None:
            self.num_x = int((self.x_high - self.x_low) / self.dx + 1)
        else:
            self.num_x = num_x
        if num_t is None:
            self.num_t = int(self.T/self.dt + 1) 
        else:
            self.num_t = num_t
        self.grid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
            
        self.grid[0,:] = init_value
        self.eta = args.eta # coefficient of viscous term
        self.forcing = forcing # the forcing term
        self.x_grid = np.linspace(self.x_low, self.x_high, self.num_x)

    def flux(self, u):
        if self.args.flux == 'u2':
            return u ** 2 / 2.
        elif self.args.flux == 'u4':
            return u ** 4 / 16.
        elif self.args.flux == 'u3':
            return u ** 3 / 9.
        elif self.args.flux == 'BL':
            return u ** 2 / (u ** 2 + 0.5 * (1-u) ** 2)
        elif self.args.flux.startswith("linear"):
            a = float(self.args.flux[len('linear'):])
            return a * u

    def expand_boundary(self, val, left_width, right_width = None, mode = 'periodic'):
        '''
        expand the boundary points.
        '''
        if right_width is None:
            right_width = left_width
        if mode == 'periodic':
            tmp = list(val[-left_width - 1:-1]) + list(val) + list(val[1:right_width + 1])
        elif mode == 'outflow':
            tmp = list(val[:left_width]) + list(val) + list(val[-right_width:])
        else:
            raise('Invalide Boundary Condition!')
        return tmp

    def get_flux(self, u):
        u_expand = self.expand_boundary(u, 3)
        flux_left = np.zeros(self.num_x + 1)
        flux_right = np.zeros(self.num_x + 1)
        flux = np.zeros(self.num_x + 1)
        
        dleft2, dleft1, dleft0 = 0.1, 0.6, 0.3 ### ideal weight for reconstruction of the minus index (spatial right) boundary. (or the minus one in the book.)
        dright2, dright1, dright0 = 0.3, 0.6, 0.1

        for i in range(self.num_x + 1):
            left_used = u_expand[i:i+5]
            right_used = u_expand[i+1:i+6]

            fl = self.flux(np.array(left_used))
            fr = self.flux(np.array(right_used))

            betal0 = 13 / 12 * (fl[2] - 2 * fl[3] + fl[4]) ** 2 + 1 / 4 * (3 * fl[2] - 4 * fl[3] + fl[4]) ** 2
            betal1 = 13 / 12 * (fl[1] - 2 * fl[2] + fl[3]) ** 2 + 1 / 4 * (fl[1] - fl[3]) ** 2
            betal2 = 13 / 12 * (fl[0] - 2 * fl[1] + fl[2]) ** 2 + 1 / 4 * (fl[0] - 4 * fl[1] + 3 * fl[2]) ** 2

            betar0 = 13 / 12 * (fr[2] - 2 * fr[3] + fr[4]) ** 2 + 1 / 4 * (3 * fr[2] - 4 * fr[3] + fr[4]) ** 2
            betar1 = 13 / 12 * (fr[1] - 2 * fr[2] + fr[3]) ** 2 + 1 / 4 * (fr[1] - fr[3]) ** 2
            betar2 = 13 / 12 * (fr[0] - 2 * fr[1] + fr[2]) ** 2 + 1 / 4 * (fr[0] - 4 * fr[1] + 3 * fr[2]) ** 2

            eps = 1e-6

            alphal0 = dleft0 / (betal0 + eps) ** 2
            alphal1 = dleft1 / (betal1 + eps) ** 2
            alphal2 = dleft2 / (betal2 + eps) ** 2
            wl0 = alphal0 / (alphal0 + alphal1 + alphal2)
            wl1 = alphal1 / (alphal0 + alphal1 + alphal2)
            wl2 = alphal2 / (alphal0 + alphal1 + alphal2)

            alphar0 = dright0 / (betar0 + eps) ** 2
            alphar1 = dright1 / (betar1 + eps) ** 2
            alphar2 = dright2 / (betar2 + eps) ** 2
            wr0 = alphar0 / (alphar0 + alphar1 + alphar2)
            wr1 = alphar1 / (alphar0 + alphar1 + alphar2)
            wr2 = alphar2 / (alphar0 + alphar1 + alphar2)

            fl2 = fl[0] * 1 / 3 + fl[1] * (- 7 / 6) + fl[2] * 11 / 6
            fl1 = fl[1] * (-1 / 6) + fl[2] * (5 / 6) + fl[3] * 1 / 3
            fl0 = fl[2] * 1 / 3 + fl[3] * (5 / 6) + fl[4] * -1 / 6

            fr2 = fr[0] * -1 / 6 + fr[1] * (5 / 6) + fr[2] * 1 / 3
            fr1 = fr[1] * (1 / 3) + fr[2] * (5 / 6) + fr[3] * -1 / 6
            fr0 = fr[2] * 11 / 6 + fr[3] * (-7 / 6) + fr[4] * 1 / 3

            flux_left[i] = wl0 * fl0 + wl1 * fl1 + wl2 * fl2
            flux_right[i] = wr0 * fr0 + wr1 * fr1 + wr2 * fr2

            if self.args.flux == 'u2' or self.args.flux == 'u4':
                roe = (u_expand[i + 3] + u_expand[i + 2]) ### last line reduces to this line with f = 1/2 u ** 2
            elif self.args.flux == 'u3':
                roe = (u_expand[i + 3] ** 2 + u_expand[i + 2] ** 2 + u_expand[i + 2] * u_expand[i + 3])
            
            judge = roe
            if judge >= 0:
                flux[i] = flux_left[i]
            else:
                flux[i] = flux_right[i]

        return flux[:-1], flux[1:]   

    def obtain_flux(self, u, identity=False):
        u_expand = self.expand_boundary(u, 3)
        flux_left = np.zeros(self.num_x + 1)
        flux_right = np.zeros(self.num_x + 1)
        flux = np.zeros(self.num_x + 1)
        
        dleft2, dleft1, dleft0 = 0.1, 0.6, 0.3 ### ideal weight for reconstruction of the minus index (spatial right) boundary. (or the minus one in the book.)
        dright2, dright1, dright0 = 0.3, 0.6, 0.1

        for i in range(self.num_x + 1):
            left_used = u_expand[i:i+5]
            right_used = u_expand[i+1:i+6]

            if not identity:
                fl = self.flux(np.array(left_used))
                fr = self.flux(np.array(right_used))
            else:
                fl = np.array(left_used)
                fr = np.array(right_used)

            betal0 = 13 / 12 * (fl[2] - 2 * fl[3] + fl[4]) ** 2 + 1 / 4 * (3 * fl[2] - 4 * fl[3] + fl[4]) ** 2
            betal1 = 13 / 12 * (fl[1] - 2 * fl[2] + fl[3]) ** 2 + 1 / 4 * (fl[1] - fl[3]) ** 2
            betal2 = 13 / 12 * (fl[0] - 2 * fl[1] + fl[2]) ** 2 + 1 / 4 * (fl[0] - 4 * fl[1] + 3 * fl[2]) ** 2

            betar0 = 13 / 12 * (fr[2] - 2 * fr[3] + fr[4]) ** 2 + 1 / 4 * (3 * fr[2] - 4 * fr[3] + fr[4]) ** 2
            betar1 = 13 / 12 * (fr[1] - 2 * fr[2] + fr[3]) ** 2 + 1 / 4 * (fr[1] - fr[3]) ** 2
            betar2 = 13 / 12 * (fr[0] - 2 * fr[1] + fr[2]) ** 2 + 1 / 4 * (fr[0] - 4 * fr[1] + 3 * fr[2]) ** 2

            eps = 1e-6

            alphal0 = dleft0 / (betal0 + eps) ** 2
            alphal1 = dleft1 / (betal1 + eps) ** 2
            alphal2 = dleft2 / (betal2 + eps) ** 2
            wl0 = alphal0 / (alphal0 + alphal1 + alphal2)
            wl1 = alphal1 / (alphal0 + alphal1 + alphal2)
            wl2 = alphal2 / (alphal0 + alphal1 + alphal2)

            alphar0 = dright0 / (betar0 + eps) ** 2
            alphar1 = dright1 / (betar1 + eps) ** 2
            alphar2 = dright2 / (betar2 + eps) ** 2
            wr0 = alphar0 / (alphar0 + alphar1 + alphar2)
            wr1 = alphar1 / (alphar0 + alphar1 + alphar2)
            wr2 = alphar2 / (alphar0 + alphar1 + alphar2)

            fl2 = fl[0] * 1 / 3 + fl[1] * (- 7 / 6) + fl[2] * 11 / 6
            fl1 = fl[1] * (-1 / 6) + fl[2] * (5 / 6) + fl[3] * 1 / 3
            fl0 = fl[2] * 1 / 3 + fl[3] * (5 / 6) + fl[4] * -1 / 6

            fr2 = fr[0] * -1 / 6 + fr[1] * (5 / 6) + fr[2] * 1 / 3
            fr1 = fr[1] * (1 / 3) + fr[2] * (5 / 6) + fr[3] * -1 / 6
            fr0 = fr[2] * 11 / 6 + fr[3] * (-7 / 6) + fr[4] * 1 / 3

            flux_left[i] = wl0 * fl0 + wl1 * fl1 + wl2 * fl2
            flux_right[i] = wr0 * fr0 + wr1 * fr1 + wr2 * fr2

            # roe = (self.flux(u_expand[i + 3]) - self.flux(u_expand[i + 2])) / (u_expand[i + 3] - u_expand[i + 2])
            if self.args.flux == 'u2' or self.args.flux == 'u4':
                roe = (u_expand[i + 3] + u_expand[i + 2]) ### last line reduces to this line with f = 1/2 u ** 2
            elif self.args.flux == 'u3':
                # print('enter here')
                roe = (u_expand[i + 3] ** 2 + u_expand[i + 2] ** 2 + u_expand[i + 2] * u_expand[i + 3])
            elif self.args.flux == 'BL':
                roe = 0.5 * (u_expand[i + 3] + u_expand[i + 2]) - u_expand[i + 2] * u_expand[i + 3]
            elif self.args.flux.startswith("linear"):
                roe = float(self.args.flux[len('linear'):])

            if identity:
                roe = 1
            
            judge = roe
            if judge >= 0:
                flux[i] = flux_left[i]
            else:
                flux[i] = flux_right[i]
        
        return flux
           
    def evolve(self, u, i):
        ### below implements weno
        ### here, left and right means the minus and plus (i.e., upwind direction) when computing the flux at the same location.
        def rhs(u):
            flux = self.obtain_flux(u)
            rhs = -(flux[1:] - flux[:-1]) / self.dx
            if self.eta > 0:
                # identity_flux = self.obtain_flux(u, identity=True)
                # u_x = (identity_flux[1:] - identity_flux[:-1]) / self.dx
                # identity_flux = self.obtain_flux(u_x, identity=True)
                # u_xx = (identity_flux[1:] - identity_flux[:-1]) / self.dx
                # u_x = np.zeros_like(u)
                # u_x[1:-1] = (u[2:] - u[:-2]) / (2 * self.dx)
                # u_x[0] = (u[1] - u[-2]) / (2 * self.dx)
                # u_x[-1] = (u[1] - u[-2]) / (2 * self.dx)

                # u_xx = np.zeros_like(u_x)
                # u_xx[1:-1] = (u_x[2:] - u_x[:-2]) / (2 * self.dx)
                # u_xx[0] = (u_x[1] - u_x[-2]) / (2 * self.dx)
                # u_xx[-1] = (u_x[1] - u_x[-2]) / (2 * self.dx)

                u_xx = np.zeros_like(u)
                u_xx[1:-1] = (u[2:] + u[:-2] - 2 * u[1:-1]) / (self.dx ** 2)
                u_xx[0] = (u[1] + u[-2] - 2 * u[0]) / (self.dx ** 2)
                u_xx[-1] = (u[1] + u[-2] - 2 * u[-1]) / (self.dx ** 2)

                rhs += self.eta * u_xx

            if self.forcing is not None:
                period = self.x_high - self.x_low
                rhs += self.forcing(x=self.x_grid, t=i*self.dt, period=period)

            return rhs

        ###  this runs rk4
        u_next_1 = u + self.dt * rhs(u)
        u_next_2 = (3 * u + u_next_1 + self.dt * rhs(u_next_1)) / 4
        u_next = (u + 2 * u_next_2 + 2 * self.dt * rhs(u_next_2)) / 3
        if self.args.Tscheme == 'rk4':
            return u_next
        elif self.args.Tscheme == 'euler':
            return u_next_1


    def solve(self):
        for i in range(1, self.num_t):
            self.grid[i] = self.evolve(self.grid[i-1], i-1)

        return self.grid


if __name__ == '__main__':
    import argparse, sys
    args = argparse.ArgumentParser(sys.argv[0])
    args.add_argument('--x_low', type = float, default = 0)
    args.add_argument('--x_high', type = float, default = 1)
    args.add_argument('--dx', type = float, default = 0.01)
    args.add_argument('--cfl', type = float, default = 0.1)
    args.add_argument('--T', type = float, default = 0.07)
    args.add_argument('--Tscheme', type = str, default = 'euler')
    args.add_argument('--flux', type = str, default = 'u2')
    args.add_argument('--eta', type = float, default = 0)
    args.add_argument('--save_path', type = str, default =None)

    args = args.parse_args()

    def init_simple(x, t=0):
        return 0.5 + np.sin(2 * np.pi * x)

    def show(solutions, x_gird, save_path=None):
        """
        plot a animation of the evolving process.
        solutions: solution array returned by get_weno_grid.
        x_grid: e.g. [-1, -0.96, ..., 0.96, 1]
        """

        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib import animation

        ### you need to install ffmpeg if you want to store the animations
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        x_low = x_grid[0]
        x_high = x_grid[-1]
        num_x = len(x_grid)
        num_t = len(solutions)

        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim((x_low, x_high))
        ymin = np.min(solutions[0]) - 0.1
        ymax = np.max(solutions[0]) + 0.1
        ax.set_ylim((ymin, ymax)) ## you might want to change these params
        line, = ax.plot(x_grid, [0 for _ in range(num_x)], lw = 2, label = 'solution')

        def init():    
            line.set_data([], [])
            return line

        def func(i):
            # print('make animations, step: ', i)
            x = np.linspace(x_low, x_high, num_x)
            y = solutions[i]
            line.set_data(x, y)
            return line

        anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=num_t, interval=200)
        plt.legend()
        plt.title('Solutions')
        plt.tight_layout()
        if save_path is not None:
            save_name = save_path
            anim.save("./data/video/" + save_name + ".mp4", writer=writer)
        # you need to install "ffmpeg" for storing animations.
        plt.show()
        plt.close()

    init = init_simple

    num_x = int((args.x_high - args.x_low) / args.dx + 1)
    x_grid = np.linspace(args.x_low, args.x_high, num_x)
    init_value = init(x_grid)
    # plt.plot(range(len(x_grid)), x_grid)
    # plt.plot(x_grid, init_value)
    # plt.show()

    # import time

    # test_time = 20
    # fd_weno_solver = weno3_fd(args, init_value = init_value)
    # timecosts = []
    # for i in range(test_time):
    #     start = time.time()
    #     solutions = fd_weno_solver.solve()
    #     tmp = time.time() - start
    #     print("test {}, use time {}".format(i, tmp))
    #     timecosts.append(tmp)

    # print("poor weno average time: ", np.mean(timecosts))

    fd_weno_solver = weno3_fd(args, init_value = init_value)
    solutions = fd_weno_solver.solve()
    show(solutions, x_grid, args.save_path)