import numpy as np
import scipy.optimize as sco
from matplotlib import pyplot as plt

"""
This file implements 5-order finite volume WENO on Burgers Equation. 
See bottom for usage examples.
"""

class Weno3:
    def __init__(self, left_boundary, right_boundary, ncells, flux_callback, flux_deriv_callback, 
            max_flux_deriv_callback,  dx, dt, boundary = 'periodic', 
            char_speed = 1, cfl_number=0.8, eps=1.0e-6, record = True, num_t = 6000, 
            eta = 0, forcing=None):
        """

        :rtype : Weno3
        """
        a = left_boundary
        b = right_boundary
        self.N = ncells
        # self.dx = (b - a) / (self.N + 0.0)
        self.dx = dx
        self.dt = dt
        self.CFL_NUMBER = cfl_number
        self.CHAR_SPEED = char_speed
        self.t = 0.0
        ORDER_OF_SCHEME = 3
        self.EPS = eps
        # Ideal weights for the right boundary.
        self.iw_right = np.array([[3.0 / 10.0], [6.0 / 10.0], [1.0 / 10.0]])
        self.iw_left = np.array([[1.0 / 10.0], [6.0 / 10.0], [3.0 / 10.0]])
        self.flux = flux_callback
        self.flux_deriv = flux_deriv_callback
        self.max_flux_deriv = max_flux_deriv_callback
        self.x_boundary = np.linspace(a, b, self.N + 1)
        self.x_center = np.zeros(self.N)
        self.eta = eta # viscous term coefficient
        self.forcing = forcing # force term

        for i in range(0, self.N):
            self.x_center[i] = (self.x_boundary[i] + self.x_boundary[i + 1]) / 2.0

        self.u_right_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_left_boundary_approx = np.zeros((ORDER_OF_SCHEME, self.N))
        self.u_right_boundary = np.zeros(self.N)
        self.u_left_boundary = np.zeros(self.N)
        self.beta = np.zeros((ORDER_OF_SCHEME, self.N))
        self.alpha_right = np.zeros((ORDER_OF_SCHEME, self.N))
        self.alpha_left = np.zeros((ORDER_OF_SCHEME, self.N))
        self.sum_alpha_right = np.zeros(self.N)
        self.sum_alpha_left = np.zeros(self.N)
        self.omega_right = np.zeros((ORDER_OF_SCHEME, self.N))
        self.omega_left = np.zeros((ORDER_OF_SCHEME, self.N))
        self.fFlux = np.zeros(self.N + 1)
        self.rhsValues = np.zeros(self.N)
        self.u_multistage = np.zeros((3, self.N))
        if record:
            self.u_grid = np.zeros((num_t, self.N))
        self.record = record
        self.boundary = boundary
        # self.u_grid = []


    def integrate(self, u0, time_final, num_t=None):
        # self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        self.u_multistage[0] = u0
        if self.record:
            self.u_grid[0] = u0
        # self.u_grid.append(u0)
        index = 1

        if num_t is None:
            evolve_time = int(time_final / self.dt)
        else:
            evolve_time = num_t
        print('T is {0}, dt is {1}, evolve time is {2}'.format(time_final, self.dt, evolve_time))

        # while self.t < self.T:
        #     if self.t + self.dt > self.T:
        #         self.dt = self.T - self.t
        for _ in range(evolve_time):

            self.u_multistage[1] = self.u_multistage[0] + self.dt * self.rhs(self.u_multistage[0])
            self.u_multistage[2] = (3 * self.u_multistage[0] + self.u_multistage[1] + self.dt * self.rhs(self.u_multistage[1])) / 4.0
            self.u_multistage[0] = (self.u_multistage[0] + 2.0 * self.u_multistage[2] + 2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0
            if self.record:
                self.u_grid[index] = self.u_multistage[0]
            # self.u_grid.append(self.u_multistage[0])
            index += 1
            self.t += self.dt

        return self.u_multistage[0]


    def rhs(self, u):
        self._rhs(u)

        # Numerical flux calculation.
        self.fFlux[1:-1] = self.numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
        self.fFlux[0] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
        self.fFlux[self.N] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])

        # Right hand side calculation, for the u^2 flux part.
        rhsValues = self.fFlux[1:] - self.fFlux[0:-1]
        rhsValues = -rhsValues / self.dx

        # compute the viscous term part
        if self.eta > 0:    
            # self.fFlux[1:-1] = self.identity_numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
            # self.fFlux[0] = self.identity_numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
            # self.fFlux[self.N] = self.identity_numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
            # u_x = self.fFlux[1:] - self.fFlux[0:-1]
            # u_x = u_x / self.dx

            u_x = np.zeros_like(u)
            u_x[1:-1] = (u[2:] - u[:-2]) / (2 * self.dx)
            u_x[0] = (u[1] - u[-2]) / (2 * self.dx)
            u_x[-1] = (u[1] - u[-2]) / (2 * self.dx)

            # self._rhs(u_x) # recompute the left/right boundary using u_x as the grid value
            # self.fFlux[1:-1] = self.identity_numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
            # self.fFlux[0] = self.identity_numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
            # self.fFlux[self.N] = self.identity_numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
            # u_xx = self.fFlux[1:] - self.fFlux[0:-1]
            # u_xx = u_xx / self.dx

            u_xx = np.zeros_like(u_x)
            u_xx[1:-1] = (u_x[2:] - u_x[:-2]) / (2 * self.dx)
            u_xx[0] = (u_x[1] - u_x[-2]) / (2 * self.dx)
            u_xx[-1] = (u_x[1] - u_x[-2]) / (2 * self.dx)

            rhsValues += self.eta * u_xx

        if self.forcing is not None:
            period = self.x_center[-1] - self.x_center[0]
            rhsValues += self.forcing(x=self.x_center, t=self.t, period=period)

        return rhsValues


    def _rhs(self, u):
        # WENO Reconstruction
        # Approximations for inner cells 0 < i < N-1.
        self.u_right_boundary_approx[0][2:-2] = 1.0 / 3.0 * u[2:-2] + 5.0 / 6.0 * u[3:-1] - 1.0 / 6.0 * u[4:]
        self.u_right_boundary_approx[1][2:-2] = -1.0 / 6.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] + 1.0 / 3.0 * u[3:-1]
        self.u_right_boundary_approx[2][2:-2] = 1.0 / 3.0 * u[0:-4] - 7.0 / 6.0 * u[1:-3] + 11.0 / 6.0 * u[2:-2]
        self.u_left_boundary_approx[0][2:-2] = 11.0 / 6.0 * u[2:-2] - 7.0 / 6.0 * u[3:-1] + 1.0 / 3.0 * u[4:]
        self.u_left_boundary_approx[1][2:-2] = 1.0 / 3.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] - 1.0 / 6.0 * u[3:-1]
        self.u_left_boundary_approx[2][2:-2] = -1.0 / 6.0 * u[0:-4] + 5.0 / 6.0 * u[1:-3] + 1.0 / 3.0 * u[2:-2]

        if self.boundary == 'periodic':
            # Approximations for cell i = 0 (the leftmost cell).
            self.u_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
            self.u_right_boundary_approx[1][0] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
            self.u_right_boundary_approx[2][0] = 1.0 / 3.0 * u[-3] - 7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[0]
            self.u_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - 7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
            self.u_left_boundary_approx[1][0] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
            self.u_left_boundary_approx[2][0] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[0]

            # Approximations for cell i = 1.
            self.u_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + 5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
            self.u_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
            self.u_right_boundary_approx[2][1] = 1.0 / 3.0 * u[-2] - 7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
            self.u_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - 7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
            self.u_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
            self.u_left_boundary_approx[2][1] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]

            # Approximations for cell i = N-2.
            self.u_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[1]
            self.u_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
            self.u_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - 7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
            self.u_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - 7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[1]
            self.u_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + 5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
            self.u_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + 5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]

            # Approximations for cell i = N-1 (the rightmost cell).
            self.u_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
            self.u_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[1]
            self.u_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - 7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
            self.u_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - 7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
            self.u_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[1]
            self.u_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]

        elif self.boundary == 'outflow':
            print('outflow')
            # Approximations for cell i = 0 (the leftmost cell).
            self.u_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
            self.u_right_boundary_approx[1][0] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
            self.u_right_boundary_approx[2][0] = 1.0 / 3.0 * u[0] - 7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[0]
            self.u_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - 7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
            self.u_left_boundary_approx[1][0] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
            self.u_left_boundary_approx[2][0] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[0]

            # Approximations for cell i = 1.
            self.u_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + 5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
            self.u_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
            self.u_right_boundary_approx[2][1] = 1.0 / 3.0 * u[0] - 7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
            self.u_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - 7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
            self.u_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
            self.u_left_boundary_approx[2][1] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]

            # Approximations for cell i = N-2.
            self.u_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[-1]
            self.u_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
            self.u_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - 7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
            self.u_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - 7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[-1]
            self.u_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + 5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
            self.u_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + 5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]

            # Approximations for cell i = N-1 (the rightmost cell).
            self.u_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[-1]
            self.u_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[-1]
            self.u_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - 7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
            self.u_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - 7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[-1]
            self.u_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[-1]
            self.u_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]

        self.beta[0][2:-2] = 13.0 / 12.0 * (u[2:-2] - 2 * u[3:-1] + u[4:]) ** 2 + \
                             1.0 / 4.0 * (3*u[2:-2] - 4.0 * u[3:-1] + u[4:]) ** 2
        self.beta[1][2:-2] = 13.0 / 12.0 * (u[1:-3] - 2 * u[2:-2] + u[3:-1]) ** 2 + \
                             1.0 / 4.0 * (u[1:-3] - u[3:-1]) ** 2
        self.beta[2][2:-2] = 13.0 / 12.0 * (u[0:-4] - 2 * u[1:-3] + u[2:-2]) ** 2 + \
                           1.0 / 4.0 * (u[0:-4] - 4.0 * u[1:-3] + 3 * u[2:-2]) ** 2

        if self.boundary == 'periodic':
            self.beta[0][0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                            1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
            self.beta[1][0] = 13.0 / 12.0 * (u[-2] - 2 * u[0] + u[1]) ** 2 + \
                            1.0 / 4.0 * (u[-2] - u[1]) ** 2
            self.beta[2][0] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[0]) ** 2 + \
                            1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[0]) ** 2

            self.beta[0][1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
                            1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
            self.beta[1][1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                            1.0 / 4.0 * (u[0] - u[2]) ** 2
            self.beta[2][1] = 13.0 / 12.0 * (u[-2] - 2 * u[0] + u[1]) ** 2 + \
                            1.0 / 4.0 * (u[-2] - 4.0 * u[0] + 3 * u[1]) ** 2

            self.beta[0][-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[1]) ** 2 + \
                            1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[1]) ** 2
            self.beta[1][-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (u[-3] - u[-1]) ** 2
            self.beta[2][-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
                            1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2

            self.beta[0][-1] = 13.0 / 12.0 * (u[-1] - 2 * u[1] + u[2]) ** 2 + \
                            1.0 / 4.0 * (3*u[-1] - 4.0 * u[1] + u[2]) ** 2
            self.beta[1][-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[1]) ** 2 + \
                            1.0 / 4.0 * (u[-2] - u[1]) ** 2
            self.beta[2][-1] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[-1]) ** 2

        elif self.boundary == 'outflow':
            print("outflow!")
            self.beta[0][0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                            1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
            self.beta[1][0] = 13.0 / 12.0 * (u[0] - 2 * u[0] + u[1]) ** 2 + \
                            1.0 / 4.0 * (u[0] - u[1]) ** 2
            self.beta[2][0] = 13.0 / 12.0 * (u[0] - 2 * u[0] + u[0]) ** 2 + \
                            1.0 / 4.0 * (u[0] - 4.0 * u[0] + 3 * u[0]) ** 2

            self.beta[0][1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
                            1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
            self.beta[1][1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                            1.0 / 4.0 * (u[0] - u[2]) ** 2
            self.beta[2][1] = 13.0 / 12.0 * (u[0] - 2 * u[0] + u[1]) ** 2 + \
                            1.0 / 4.0 * (u[0] - 4.0 * u[0] + 3 * u[1]) ** 2

            self.beta[0][-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[-1]) ** 2
            self.beta[1][-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (u[-3] - u[-1]) ** 2
            self.beta[2][-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
                            1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2

            self.beta[0][-1] = 13.0 / 12.0 * (u[-1] - 2 * u[-1] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (3*u[-1] - 4.0 * u[-1] + u[-1]) ** 2
            self.beta[1][-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (u[-2] - u[-1]) ** 2
            self.beta[2][-1] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                            1.0 / 4.0 * (u[-3] - 4.0 * u[-2] + 3 * u[-1]) ** 2

        self.alpha_right = self.iw_right / ((self.EPS + self.beta) ** 2)
        self.alpha_left = self.iw_left / ((self.EPS + self.beta) ** 2)
        self.sum_alpha_right = self.alpha_right[0] + self.alpha_right[1] + self.alpha_right[2]
        self.sum_alpha_left = self.alpha_left[0] + self.alpha_left[1] + self.alpha_left[2]
        self.omega_right = self.alpha_right / self.sum_alpha_right
        self.omega_left = self.alpha_left / self.sum_alpha_left
        self.u_right_boundary = self.omega_right[0] * self.u_right_boundary_approx[0] + \
                           self.omega_right[1] * self.u_right_boundary_approx[1] + \
                           self.omega_right[2] * self.u_right_boundary_approx[2]
        self.u_left_boundary = self.omega_left[0] * self.u_left_boundary_approx[0] + \
                          self.omega_left[1] * self.u_left_boundary_approx[1] + \
                          self.omega_left[2] * self.u_left_boundary_approx[2]


    def identity_numflux(self, a, b):
        """
        flux is identity function.
        """
        maxval = 1
        return 0.5 * (a + b - maxval*(b - a))


    def numflux(self, a, b):
        """
        Return Lax-Friedrichs numerical flux.
        """
        flux = self.flux
        max_flux_deriv = self.max_flux_deriv

        maxval = max_flux_deriv(a, b)

        return 0.5 * (flux(a) + flux(b) - maxval * (b - a))


    def get_x_center(self):
        return self.x_center

    def get_x_boundary(self):
        return self.x_boundary

    def get_dx(self):
        return self.dx


class weno3_fv():
    '''
    This class wraps the finite volume 5-th order WENO.
    '''
    def __init__(self, flux_name):
        '''
        Arg flux (str): specifies the flux function.
        '''
        self.flux_name = flux_name

    ### the burgers flux function. Can be changed to any other functions if needed for future useage.
    def flux(self, val):
        if self.flux_name == 'u2':
            return val ** 2 / 2.
        elif self.flux_name == 'u4': 
            return val ** 4 / 16.
        elif self.flux_name == 'u3':
            return val ** 3 / 9.

    def flux_deriv(self, val):
        if self.flux_name == 'u2':
            return val
        elif self.flux_name == 'u4':
            return val ** 3 / 4.
        elif self.flux_name == 'u3':
            return val ** 2 / 3

    def max_flux_deriv(self, a, b):
        if self.flux_name == 'u2':
            return np.maximum(np.abs(a), np.abs(b))
        elif self.flux_name == 'u4':
            return np.maximum(np.abs(a ** 3 / 4.), np.abs(b ** 3 / 4.))
        elif self.flux_name == 'u3':
            return np.maximum(np.abs(a ** 2 / 3.), np.abs(b ** 2 / 3.))

    def solve(self, x_center, t, u0, cfl, eta=0):
        '''
        This function wraps the process of using finite volume WENO to evolve a grid.  
        Arg x_center (1d np array): the initial x grid  
        Arg t (float): evolving time  
        Arg u0 (1d np array, same shape as x_center): inital values of u on the initial x grid x_center.  
        Arg cfl (float): the cfl number used to determine temporal step size dt.  
        Return: the whole evolving grid from time 0 to t.  
        '''
        dx = x_center[1] - x_center[0]
        left_boundary = x_center[0] - dx * 0.5
        right_boundary = x_center[-1] + dx * 0.5
        ncells = len(x_center)
        dt = dx * cfl
        num_t = int(t / dt) + 1

        w = Weno3(left_boundary, right_boundary, ncells, self.flux, self.flux_deriv, self.max_flux_deriv, 
                dx = dx, dt = dt, cfl_number = cfl, num_t=num_t, eta=eta)
        
        w.integrate(u0, t)
        return w.u_grid[:num_t,:]


if __name__ == '__main__':
    """
    sample usage.
    """
    flux_func = 'u2' 

    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--x_low', type = float, default = 0)
    args.add_argument('--x_high', type = float, default = 1)
    args.add_argument('--dx', type = float, default = 0.02)
    args.add_argument('--cfl', type = float, default = 0.1)
    args.add_argument('--T', type = float, default = 0.5)
    args.add_argument('--save_path', type = str, default =None)
    args = args.parse_args()

    def get_weno_grid(init_condition, dx = 0.001, dt = 0.0001,  T = 0.8, x_low = -1, x_high = 1,
            boundary='periodic', eta=0.04):
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
        ncells = int((x_high - x_low) / dx + 1.) # need to be very small
        num_t = int(T/dt + 1.)
        w = Weno3(left_boundary, right_boundary, ncells, flux, flux_deriv, max_flux_deriv, 
            dx = dx, dt = dt, num_t = num_t + 100, boundary=boundary, eta=eta)
        x_center = w.get_x_center()
        u0 = init_condition(x_center)
        w.integrate(u0, T)
        solutions = w.u_grid[:num_t,:]
        return solutions

    ### the burgers flux function. Can be changed to any other functions if needed for future useage.
    def flux(val, flux = flux_func):
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

    def flux_deriv(val, flux = flux_func):
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

    def max_flux_deriv(a, b, flux = flux_func):
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
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        x_low = x_grid[0]
        x_high = x_grid[-1]
        num_x = len(x_grid)
        num_t = len(solutions)
        print(x_low, x_high)

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

        anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=num_t, interval=50)
        plt.legend()
        plt.title('Solutions')
        plt.tight_layout()

        if save_path is not None:
            save_name = save_path
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            anim.save("./data/video/" + save_name + ".mp4", writer=writer)
        plt.show()
        plt.close()

    # import time

    # test_time = 20
    # time_costs = []
    # for i in range(test_time):
    #     start = time.time()
    #     solutions = get_weno_grid(dx = 0.002, dt = 0.0004, init_condition = init)
    #     tmp = time.time() - start
    #     print('test {}, use time {}'.format(i, tmp))
    #     time_costs.append(tmp)
    
    # print("good weno average time: ", np.mean(time_costs))

    def init_simple(x, t=0):
        return -0.8412740798850631 + 1.6933719575704096 * np.sin(6 * np.pi * x) + 1.47 * np.cos(6 * np.pi * x)

    init = init_simple
    dx = args.dx
    dt = dx * args.cfl
    x_low = args.x_low
    x_high = args.x_high
    T = args.T
    boundary = 'periodic'
    solutions = get_weno_grid(x_low = x_low, x_high  = x_high, dx = dx, dt = dt, init_condition = init, T = T, boundary=boundary, eta=0.04)
    # solutions = get_weno_grid(init_condition = init)
    print(solutions)
    num_x = int((x_high - x_low) / dx) + 1
    x_grid = np.linspace(x_low, x_high, num_x)
    # print(x_grid)
    show(solutions, x_grid, args.save_path)


