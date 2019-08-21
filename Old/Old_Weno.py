import numpy
import Weno.Old_Weno_util as Weno_util
# from matplotlib import pyplot as plt
# from matplotlib import animation


def weno(order, q):
    """
    Do WENO reconstruction
    
    Parameters
    ----------
    
    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct
        
    Returns
    -------
    
    qL : numpy array
        Reconstructed data - boundary points are zero
    """
    C = Weno_util.C_all[order]
    a = Weno_util.a_all[order]
    sigma = Weno_util.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order, len(q)))
    w = numpy.zeros_like(beta)
    np = len(q) - 2 * order
    epsilon = 1e-16
    for i in range(order, np+order):
        q_stencils = numpy.zeros(order)
        alpha = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l+1):
                    beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
            print(beta[k, i])
            if beta[k, i] != beta[k, i]:
                exit()
            alpha[k] = C[k] / (epsilon + beta[k, i]**2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[i+k-l]
        w[:, i] = alpha / numpy.sum(alpha)
        qL[i] = numpy.dot(w[:, i], q_stencils)
    
    return qL


class WENOSimulation(Weno_util.Simulation):
    
    def __init__(self, grid, C=0.5, weno_order=3, dt = None):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.dt = dt
        self.weno_order = weno_order

    def init_cond(self, init_func):
        if init_func == "smooth_sine":
            self.grid.u = numpy.sin(2 * numpy.pi * self.grid.x)
        elif init_func == "gaussian":
            self.grid.u = 1.0 + numpy.exp(-60.0*(self.grid.x - 0.5)**2)
        elif init_func == 'sin':
            self.grid.u = 0.5 + numpy.sin(numpy.pi * self.grid.x)
        elif init_func == 'Riemman':
            self.grid.u[self.grid.x < 0] = 1
            self.grid.u[self.grid.x >= 0] = 0
        elif init_func == 'cos':
            self.grid.u = 1 + 2 * numpy.cos(numpy.pi * self.grid.x)
        elif init_func == 'test':
            self.grid.u = 3.5 - 1.5 * numpy.cos(2 * numpy.pi * self.grid.x)
        elif init_func == "tophat":
            self.grid.u[numpy.logical_and(self.grid.x >= 0.333,
                                          self.grid.x <= 0.666)] = 1.0
        elif init_func == "sine":
            self.grid.u[:] = 2.0

            index = numpy.logical_and(self.grid.x >= 0.333,
                                      self.grid.x <= 0.666)
            self.grid.u[index] += \
                0.5*numpy.sin(2.0*numpy.pi*(self.grid.x[index]-0.333)/0.333)

        elif init_func == "rarefaction":
            self.grid.u[:] = 1.0
            self.grid.u[self.grid.x > 0.5] = 2.0
        else:
            self.grid.u = init_func(self.grid.x)



    def burgers_flux(self, q):
        return 0.5*q**2


    def rk_substep(self):
        
        g = self.grid
        g.fill_BCs()
        f = self.burgers_flux(g.u)
        alpha = numpy.max(abs(g.u))
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[1:] = weno(self.weno_order, fp[:-1])
        fml[-1::-1] = weno(self.weno_order, fm[-1::-1])
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs


    def evolve(self, tmax):
        """ evolve the linear advection equation using RK4 """
        self.t = 0.0
        g = self.grid

        idx = 0

        # main evolution loop
        while self.t < tmax:


            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            if self.dt is None:
                dt = self.timestep(self.C)
            else:
                dt = self.dt # use fixed time step

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK4
            # Store the data at the start of the step
            u_start = g.u.copy() 
            g.u_grid[idx] = u_start # record all the values computed along the way
            k1 = dt * self.rk_substep() 
            g.u = u_start + k1 / 2 
            k2 = dt * self.rk_substep()
            g.u = u_start + k2 / 2 
            k3 = dt * self.rk_substep()
            g.u = u_start + k3 
            k4 = dt * self.rk_substep()
            g.u = u_start + (k1 + 2 * (k2 + k3) + k4) / 6

            self.t += dt
            idx += 1

if __name__ == "__main__":

    #-----------------------------------------------------------------------------
    # sine

    import argparse
    import sys
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--dt', type=float, default=0.01)
    argparser.add_argument('--nx', type=int, default=50)
    argparser.add_argument('--x_low', type=float, default=0)
    argparser.add_argument('--x_high', type=float, default=1)
    argparser.add_argument('--order', type=int, default=5)
    argparser.add_argument('--T', type=float, default=0.5)
    argparser.add_argument('--boundary_condition', type=str, default='periodic')
    argparser.add_argument('--init', type=str, default='rarefaction')
    args = argparser.parse_args()

    xmin = args.x_low
    xmax = args.x_high
    nx = args.nx
    order = args.order
    ng = order+1 # periodic
    g = Weno_util.Grid1d(nx = nx, ng = ng, xmin = xmin, xmax = xmax, bc=args.boundary_condition)

    
    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin)/1.0
    
    C = args.dt / ((xmax - xmin) / nx)
    dt = args.dt
    
    s = WENOSimulation(g, C, order, dt = dt)
    tend = args.T
    t_num = int(tend / dt) + 1

    s.init_cond(args.init)    
    s.evolve(tend)
    solutions = g.u_grid[:t_num, g.ilo:g.ihi+1]

    solutions_ = []
    for t in range(t_num + 1):
        g_ = Weno_util.Grid1d(nx = nx, ng = ng, xmin = xmin, xmax = xmax, bc=args.boundary_condition)
        s_ = WENOSimulation(g_, C, order)
        s_.init_cond(args.init)
        s_.evolve(t * dt)
        solutions_.append(g_.u[g_.ilo:g_.ihi+1])

    fig = plt.figure()
    ax = plt.axes(xlim=(xmin,xmax),ylim=(-2,3.5))
    line, = ax.plot([],[],lw=8, label = 'uniform time')
    line_, = ax.plot([],[],lw = 2, label = 'nonuniform time')

    def init():    
        line.set_data([], [])
        line_.set_data([],[])
        line.set_label('uniform time')
        line_.set_label('non-uniform time')
        return line, line_

    def func(t_num):
        x = numpy.linspace(xmin,xmax,nx)
        y = solutions[t_num]
        line.set_data(x,y)
        y_ = solutions_[t_num]
        line_.set_data(x, y_)
        return line, line_

    anim = animation.FuncAnimation(fig, func, frames=t_num, init_func=init, interval=50)
    plt.tight_layout()
    plt.title('new weno')
    plt.legend()
    plt.show()

        # c = 1.0 - (0.1 + i*0.1)
        # g = s.grid
        # pyplot.ylim(ymin = -2, ymax = 3.5)
        # pyplot.vlines(x=0.5, ymin = 0, ymax = 1)
        # pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], 'o',  color=str(c), markersize = 2)
        # pyplot.title('T is {0}'.format(tend))
        # # pyplot.show()

