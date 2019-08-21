
    def compute_speed(self):
        '''
        rougly compute the discontinuty propagation speed of the solutions
        '''
        def find_discontinous(line):
            for idx in range(self.num_x):
                if np.abs(line[(idx + 1) % self.num_x] - line[idx]) > 1.5:
                    for idx2 in range(idx + 1, self.num_x):
                        if np.abs(line[(idx2 + 1) % self.num_x] - line[idx2]) > 1.5:
                            continue
                        
                        return idx, idx2

            return -1, -1

        weno_fine_grid = np.zeros((self.num_t, self.num_x))
        for t in range(self.num_t):
            weno_fine_grid[t] = self.get_precise_value(t * self.dt)

        grids = [self.RLgrid,  weno_fine_grid]
        names = ['RL', 'weno_fine']
        for idx in range(len(grids)):
            print(names[idx])
            for t in range(self.num_t - 1):
                tleft, tright = find_discontinous(grids[idx][t])
                if tleft != -1:
                    next_t = min(self.num_t - 1, t + 15)
                    tplusleft, tplusright = find_discontinous(grids[idx][next_t])
                    # print(tleft, tright, tplusleft, tplusright)
                    but = (self.x_grid[tleft] + self.x_grid[tright]) / 2
                    butplus = (self.x_grid[tplusleft] + self.x_grid[tplusright]) / 2
                    actualspeed = (butplus - but) / (self.dt * (next_t - t))
                    theoryspeed = (self.flux(grids[idx][t][tleft]) - self.flux(grids[idx][t][tright])) \
                        / (grids[idx][t][tleft] - grids[idx][t][tright])

                    print('t {0}; actual speed: {1}; theory speed {2}; difference: {3}'.format(
                        t, actualspeed, theoryspeed, np.abs(actualspeed - theoryspeed)))


    def compute_exact_solution_grid(self, init_func, init_func_prime, b, func, c):
        self.exact_grid = np.zeros((self.num_t, self.num_x)) # record the value at each (x,t) point
        for t in range(1, self.num_t - 1):
            for i in range(self.num_x):
                time, x = t * self.dt, self.x_grid[i]
                right = 0
                if i > 0:
                    if np.abs(self.exact_grid[t-1, i] - self.exact_grid[t-1, i - 1]) > 2: ### this indicates a breaking point
                       right = 1
                        # self.show_breaking_point_implicit_func(init_func, t, i)
                        # exit()
                self.exact_grid[t, i] = self.exact_solution_point(init_func, init_func_prime, b,func, c, x, time, self.exact_grid[t-1, i], right, t)
                print('t {0}, i {1} point compute over'.format(t, i))

    def show_breaking_point_implicit_func(self, init_func, t, i):
        def exact_func(u, x, t):
            return init_func(x - u * t) - u

        u = np.linspace(-4, 4, 100)
        
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        x1, t1 = self.x_grid[i], self.dt * t
        ax1.plot(u, exact_func(u, x1, t1))
        ax1.vlines(x = self.exact_grid[t-1, i], ymin = -4, ymax = 4)
        x2, t2 = self.x_grid[i+1], self.dt * t
        ax2.plot(u, exact_func(u, x2, t2))
        ax2.vlines(x = self.exact_grid[t-1, i + 1], ymin = -4, ymax = 4)

        plt.show()
        plt.close()

    def exact_solution_point(self, init_func, init_func_prime, b, func, c, x, t, u0, right, t_idx):
        # def exact_func(x0, x, t):
        #     return init_func(x0) * t - x + x0
        # def exact_func_prime(x0, x, t):
        #     return t * init_func_prime(x0) + 1
        # try:
        #     x0 = sco.newton(exact_func, x, fprime=exact_func_prime, args=(x, t), maxiter=5000)
        # except:
        #     plt.figure()
        #     x = np.linspace(-2, 2, 100)
        #     plt.plot(x, init_func(x))
        #     plt.show()
        #     plt.close()
        #     exit()
        # return init_func(x0)

        def exact_func(u, x, t):
            return init_func(x - u * t) - u
        def exact_func_prime(u, x, t):
            return -t * init_func_prime(x - u * t) - 1
        # def exact_func_prime_prime(u, x, t):
        #     return t * t * init_func_prime_prime(x - u* t)

        # try:
            # u = sco.newton(exact_func, u0, fprime=exact_func_prime, args=(x, t), maxiter=5000)
        if right and t_idx > 40:
            try:
                # uleft = sco.newton(exact_func_prime, u0 , fprime=exact_func_prime_prime, args=(x, t), maxiter=5000)
                # uright = sco.newton(exact_func_prime, u0 + 2, fprime=exact_func_prime_prime, args=(x, t), maxiter=5000)
                if func == np.sin:
                    arc_cos = np.arccos(-1 / (t * b * c * np.pi))
                    res = []
                    for k in [-2,-1,0,1,2]:
                        tmp = arc_cos + 2 * k * np.pi
                        u0 = (x - tmp / (c * np.pi)) / t
                        if u0 > 0:
                            res.append((u0, np.abs(u0), np.abs(exact_func(u0, x, t))))
                        tmp = -arc_cos + 2 * k * np.pi
                        u0 = (x - tmp / (c * np.pi)) / t
                        if u0 > 0:
                            res.append((u0, np.abs(u0), np.abs(exact_func(u0, x, t))))
                    res.sort(key = lambda x: x[0])
                    print(res)
                    # exit()
                    uleft = res[0][0]
                    uright = res[1][0]
                else:
                    uleft = (x - 1 / c) / t
                    uright = x / t
                u = sco.bisect(exact_func, uleft, uright, args = (x,t), maxiter=5000)
            except:
                plt.figure()
                u = np.linspace(-10, 10, 100)
                plt.plot(self.x_grid, self.exact_grid[t_idx - 1])
                plt.plot(u, exact_func(u, x, t))
                plt.vlines(x = u0, ymin = -4, ymax = 4, colors='k',label = 'u0')
                plt.vlines(x = uleft, ymin = -4, ymax = 4, colors='b',label = 'uleft')
                plt.vlines(x = uright, ymin = -4, ymax = 4, colors='r' ,label = 'uright')
                plt.legend()
                plt.show()
                plt.close()
        else:
            u = sco.newton(exact_func, u0, fprime=exact_func_prime, args=(x, t), maxiter=5000)
        # except:
        #     plt.figure()
        #     u = np.linspace(-4, 4, 100)
        #     plt.plot(u, exact_func(u, x, t))
        #     plt.vlines(x = u0, ymin = -4, ymax = 4)
        #     plt.show()
        #     plt.close()
        
        return u