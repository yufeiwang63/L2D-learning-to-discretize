import numpy as np
import scipy.optimize as sco

class Weno3:
    def __init__(self, left_boundary, right_boundary, ncells, flux_callback, flux_deriv_callback, 
            max_flux_deriv_callback,  dx, dt, char_speed = 1, cfl_number=0.8, eps=1.0e-6, record = True, num_t = 6000):
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
        # self.u_grid = []


    def integrate(self, u0, time_final):
        # self.dt = self.CFL_NUMBER * self.dx / self.CHAR_SPEED
        self.T = time_final
        self.u_multistage[0] = u0
        if self.record:
            self.u_grid[0] = u0
        # self.u_grid.append(u0)
        index = 1

        evolve_time = int(time_final / self.dt)
        # print('T is {0}, dt is {1}, evolve time is {2}'.format(time_final, self.dt, evolve_time))

        # while self.t < self.T:
        #     if self.t + self.dt > self.T:
        #         self.dt = self.T - self.t
        for _ in range(evolve_time):

            self.t += self.dt
            self.u_multistage[1] = self.u_multistage[0] + self.dt * self.rhs(self.u_multistage[0])
            self.u_multistage[2] = (3 * self.u_multistage[0] + self.u_multistage[1] + self.dt * self.rhs(self.u_multistage[1])) / 4.0
            self.u_multistage[0] = (self.u_multistage[0] + 2.0 * self.u_multistage[2] + 2.0 * self.dt * self.rhs(self.u_multistage[2])) / 3.0
            if self.record:
                self.u_grid[index] = self.u_multistage[0]
            # self.u_grid.append(self.u_multistage[0])
            index += 1


        return self.u_multistage[0]


    def rhs(self, u):
        # WENO Reconstruction
        # Approximations for inner cells 0 < i < N-1.
        self.u_right_boundary_approx[0][2:-2] = 1.0 / 3.0 * u[2:-2] + 5.0 / 6.0 * u[3:-1] - 1.0 / 6.0 * u[4:]
        self.u_right_boundary_approx[1][2:-2] = -1.0 / 6.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] + 1.0 / 3.0 * u[3:-1]
        self.u_right_boundary_approx[2][2:-2] = 1.0 / 3.0 * u[0:-4] - 7.0 / 6.0 * u[1:-3] + 11.0 / 6.0 * u[2:-2]
        self.u_left_boundary_approx[0][2:-2] = 11.0 / 6.0 * u[2:-2] - 7.0 / 6.0 * u[3:-1] + 1.0 / 3.0 * u[4:]
        self.u_left_boundary_approx[1][2:-2] = 1.0 / 3.0 * u[1:-3] + 5.0 / 6.0 * u[2:-2] - 1.0 / 6.0 * u[3:-1]
        self.u_left_boundary_approx[2][2:-2] = -1.0 / 6.0 * u[0:-4] + 5.0 / 6.0 * u[1:-3] + 1.0 / 3.0 * u[2:-2]

        # Approximations for cell i = 0 (the leftmost cell).
        self.u_right_boundary_approx[0][0] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.u_right_boundary_approx[1][0] = -1.0 / 6.0 * u[-1] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.u_right_boundary_approx[2][0] = 1.0 / 3.0 * u[-2] - 7.0 / 6.0 * u[-1] + 11.0 / 6.0 * u[0]
        self.u_left_boundary_approx[0][0] = 11.0 / 6.0 * u[0] - 7.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.u_left_boundary_approx[1][0] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.u_left_boundary_approx[2][0] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]

        # Approximations for cell i = 1.
        self.u_right_boundary_approx[0][1] = 1.0 / 3.0 * u[1] + 5.0 / 6.0 * u[2] - 1.0 / 6.0 * u[3]
        self.u_right_boundary_approx[1][1] = -1.0 / 6.0 * u[0] + 5.0 / 6.0 * u[1] + 1.0 / 3.0 * u[2]
        self.u_right_boundary_approx[2][1] = 1.0 / 3.0 * u[-1] - 7.0 / 6.0 * u[0] + 11.0 / 6.0 * u[1]
        self.u_left_boundary_approx[0][1] = 11.0 / 6.0 * u[1] - 7.0 / 6.0 * u[2] + 1.0 / 3.0 * u[3]
        self.u_left_boundary_approx[1][1] = 1.0 / 3.0 * u[0] + 5.0 / 6.0 * u[1] - 1.0 / 6.0 * u[2]
        self.u_left_boundary_approx[2][1] = -1.0 / 6.0 * u[-1] + 5.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]

        # Approximations for cell i = N-2.
        self.u_right_boundary_approx[0][-2] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.u_right_boundary_approx[1][-2] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]
        self.u_right_boundary_approx[2][-2] = 1.0 / 3.0 * u[-4] - 7.0 / 6.0 * u[-3] + 11.0 / 6.0 * u[-2]
        self.u_left_boundary_approx[0][-2] = 11.0 / 6.0 * u[-2] - 7.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.u_left_boundary_approx[1][-2] = 1.0 / 3.0 * u[-3] + 5.0 / 6.0 * u[-2] - 1.0 / 6.0 * u[-1]
        self.u_left_boundary_approx[2][-2] = -1.0 / 6.0 * u[-4] + 5.0 / 6.0 * u[-3] + 1.0 / 3.0 * u[-2]

        # Approximations for cell i = N-1 (the rightmost cell).
        self.u_right_boundary_approx[0][-1] = 1.0 / 3.0 * u[-1] + 5.0 / 6.0 * u[0] - 1.0 / 6.0 * u[1]
        self.u_right_boundary_approx[1][-1] = -1.0 / 6.0 * u[-2] + 5.0 / 6.0 * u[-1] + 1.0 / 3.0 * u[0]
        self.u_right_boundary_approx[2][-1] = 1.0 / 3.0 * u[-3] - 7.0 / 6.0 * u[-2] + 11.0 / 6.0 * u[-1]
        self.u_left_boundary_approx[0][-1] = 11.0 / 6.0 * u[-1] - 7.0 / 6.0 * u[0] + 1.0 / 3.0 * u[1]
        self.u_left_boundary_approx[1][-1] = 1.0 / 3.0 * u[-2] + 5.0 / 6.0 * u[-1] - 1.0 / 6.0 * u[0]
        self.u_left_boundary_approx[2][-1] = -1.0 / 6.0 * u[-3] + 5.0 / 6.0 * u[-2] + 1.0 / 3.0 * u[-1]

        self.beta[0][2:-2] = 13.0 / 12.0 * (u[2:-2] - 2 * u[3:-1] + u[4:]) ** 2 + \
                             1.0 / 4.0 * (3*u[2:-2] - 4.0 * u[3:-1] + u[4:]) ** 2
        self.beta[1][2:-2] = 13.0 / 12.0 * (u[1:-3] - 2 * u[2:-2] + u[3:-1]) ** 2 + \
                             1.0 / 4.0 * (u[1:-3] - u[3:-1]) ** 2
        self.beta[2][2:-2] = 13.0 / 12.0 * (u[0:-4] - 2 * u[1:-3] + u[2:-2]) ** 2 + \
                           1.0 / 4.0 * (u[0:-4] - 4.0 * u[1:-3] + 3 * u[2:-2]) ** 2

        self.beta[0][0] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                           1.0 / 4.0 * (3*u[0] - 4.0 * u[1] + u[2]) ** 2
        self.beta[1][0] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                           1.0 / 4.0 * (u[-1] - u[1]) ** 2
        self.beta[2][0] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                           1.0 / 4.0 * (u[-2] - 4.0 * u[-1] + 3 * u[0]) ** 2

        self.beta[0][1] = 13.0 / 12.0 * (u[1] - 2 * u[2] + u[3]) ** 2 + \
                        1.0 / 4.0 * (3*u[1] - 4.0 * u[2] + u[3]) ** 2
        self.beta[1][1] = 13.0 / 12.0 * (u[0] - 2 * u[1] + u[2]) ** 2 + \
                        1.0 / 4.0 * (u[0] - u[2]) ** 2
        self.beta[2][1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                        1.0 / 4.0 * (u[-1] - 4.0 * u[0] + 3 * u[1]) ** 2

        self.beta[0][-2] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                        1.0 / 4.0 * (3*u[-2] - 4.0 * u[-1] + u[0]) ** 2
        self.beta[1][-2] = 13.0 / 12.0 * (u[-3] - 2 * u[-2] + u[-1]) ** 2 + \
                        1.0 / 4.0 * (u[-3] - u[-1]) ** 2
        self.beta[2][-2] = 13.0 / 12.0 * (u[-4] - 2 * u[-3] + u[-2]) ** 2 + \
                        1.0 / 4.0 * (u[-4] - 4.0 * u[-3] + 3 * u[-2]) ** 2

        self.beta[0][-1] = 13.0 / 12.0 * (u[-1] - 2 * u[0] + u[1]) ** 2 + \
                        1.0 / 4.0 * (3*u[-1] - 4.0 * u[0] + u[1]) ** 2
        self.beta[1][-1] = 13.0 / 12.0 * (u[-2] - 2 * u[-1] + u[0]) ** 2 + \
                        1.0 / 4.0 * (u[-2] - u[0]) ** 2
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

        # Numerical flux calculation.
        self.fFlux[1:-1] = self.numflux(self.u_right_boundary[0:-1], self.u_left_boundary[1:])
        self.fFlux[0] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])
        self.fFlux[self.N] = self.numflux(self.u_right_boundary[self.N - 1], self.u_left_boundary[0])

        # Right hand side calculation.
        rhsValues = self.fFlux[1:] - self.fFlux[0:-1]
        rhsValues = -rhsValues / self.dx

        return rhsValues


    def numflux(self, a, b):
        """
        Return Lax-Friedrichs numerical flux.
        """
        maxval = self.max_flux_deriv(a, b)

        return 0.5 * (self.flux(a) + self.flux(b) - maxval * (b - a))


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

    def solve(self, x_center, t, u0, cfl):
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
        w = Weno3(left_boundary, right_boundary, ncells, self.flux, self.flux_deriv, self.max_flux_deriv, 
                dx = dx, dt = dt, cfl_number = cfl)
        
        num_t = int(t / dt) + 1
        w.integrate(u0, t)
        return w.u_grid[:num_t,:]


if __name__ == '__main__':
    import scipy.optimize as sco
    def exact_solution(x, time, mesh_size):
        u_exact = np.zeros(mesh_size)

        for i in (range(mesh_size)):
            u_exact[i] = sco.newton(exact_func, 0.5, fprime=exact_func_prime, args=(x[i], time), maxiter=5000)

        return u_exact

    def exact_func(u, x, t):
        return 0.5 + np.sin(np.pi * (x - u * t)) - u

    def exact_func_prime(u, x, t):
        return -np.pi * t * np.cos(np.pi * (x - u * t)) - 1
    pass