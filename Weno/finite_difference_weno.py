'''
File Description:
This file contains the class weno3_fd, which implements the 5-th order finite difference WENO.
'''

import numpy as np 
import copy

class weno3_fd():
    '''
    This class implements the finite difference 5-th order WENO scheme.
    Arg args (python namespace):
        should contain the following domain: x_high, x_low, dx, cfl, T, flux, Tscheme
    Arg init_value (np array):
        specifies the initial value.
    '''
    def __init__(self,  args, init_value = None):
     
        self.args = copy.copy(args)
        self.x_high, self.x_low = args.x_high, args.x_low
        self.dx, self.dt, self.T = args.dx, args.dx * args.cfl, args.T
        self.num_x = int((self.x_high - self.x_low) / self.dx + 1)
        self.num_t = int(self.T/self.dt + 1) # why + 1?
        self.grid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
            
        self.grid[0,:] = init_value

    def flux(self, u):
        if self.args.flux == 'u2':
            return u ** 2 / 2.
        elif self.args.flux == 'u4':
            return u ** 4 / 16.
        elif self.args.flux == 'u3':
            return u ** 3 / 9.

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

            # roe = (self.flux(u_expand[i + 3]) - self.flux(u_expand[i + 2])) / (u_expand[i + 3] - u_expand[i + 2])
            if self.args.flux == 'u2' or self.args.flux == 'u4':
                roe = (u_expand[i + 3] + u_expand[i + 2]) ### last line reduces to this line with f = 1/2 u ** 2
            elif self.args.flux == 'u3':
                # print('enter here')
                roe = (u_expand[i + 3] ** 2 + u_expand[i + 2] ** 2 + u_expand[i + 2] * u_expand[i + 3])
            
            judge = roe
            if judge >= 0:
                flux[i] = flux_left[i]
            else:
                flux[i] = flux_right[i]

        return flux[:-1], flux[1:]   
           
    def evolve(self, u):
        ### below implements weno
        ### here, left and right means the minus and plus (i.e., upwind direction) when computing the flux at the same location.
        def innerfunc(u):
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

                # roe = (self.flux(u_expand[i + 3]) - self.flux(u_expand[i + 2])) / (u_expand[i + 3] - u_expand[i + 2])
                if self.args.flux == 'u2' or self.args.flux == 'u4':
                    roe = (u_expand[i + 3] + u_expand[i + 2]) ### last line reduces to this line with f = 1/2 u ** 2
                elif self.args.flux == 'u3':
                    # print('enter here')
                    roe = (u_expand[i + 3] ** 2 + u_expand[i + 2] ** 2 + u_expand[i + 2] * u_expand[i + 3])
                
                judge = roe
                if judge >= 0:
                    flux[i] = flux_left[i]
                else:
                    flux[i] = flux_right[i]
            
            return -(flux[1:] - flux[:-1]) / self.dx

        ###  this runs rk4
        u_next_1 = u + self.dt * innerfunc(u)
        u_next_2 = (3 * u + u_next_1 + self.dt * innerfunc(u_next_1)) / 4
        u_next = (u + 2 * u_next_2 + 2 * self.dt * innerfunc(u_next_2)) / 3
        if self.args.Tscheme == 'rk4':
            return u_next
        elif self.args.Tscheme == 'euler':
            return u_next_1


    def solve(self):
        for i in range(1, self.num_t):
            self.grid[i] = self.evolve(self.grid[i-1])

        return self.grid
