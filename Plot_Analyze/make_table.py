import numpy as np
def load_data(path):
    f1=open(path,'r')
    RL_mean=[]
    coarse_mean=[]
    RL_std=[]
    coarse_std=[]
    dx=[]
    dt=[]
    lines = f1.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        data=line.split(' ')
        dx.append(float(data[1]))
        # print(float(data[1]))
        for idx2 in range(idx, idx + 7):
            data = lines[idx2].split(' ')
            # dt.append(float(data[2]))
            # print(float(data[2]), end = '& ')
            if data[3]=='no':
                RL_mean.append(0)
                RL_std.append(0)
                coarse_mean.append(0)
                coarse_std.append(0)
                # print(' & & ', end ='')
            else:
                if data[3]=='out':
                    RL_mean.append(np.nan)
                    RL_std.append(np.nan)
                    # print('NaN (NaN) & ', end = '')
                else:
                    RL_mean.append(float(data[3]))
                    RL_std.append(float(data[4]))
                    # print(' %.3f (%.3f) & ' % (float(data[3]), float(data[4])), end = '')

                if data[8]=='out':
                    coarse_mean.append(np.nan)
                    coarse_std.append(np.nan)
                    # print('NaN (NaN) ', end = '')
                else:
                    coarse_mean.append(float(data[8]))
                    coarse_std.append(float(data[9]))
                    # print(' %.3f (%.3f) &' % (float(data[8]), float(data[9])), end = '')

        # print('\\\\')
        idx += 7
    
    RL_mean = np.array(RL_mean).reshape(4, 7)
    RL_std = np.array(RL_std).reshape(4, 7)
    coarse_mean = np.array(coarse_mean).reshape(4, 7)
    coarse_std = np.array(coarse_std).reshape(4, 7)

    # print(RL_mean)
    RL_mean = RL_mean.T
    RL_std = RL_std.T
    coarse_mean = coarse_mean.T
    coarse_std = coarse_std.T
    # print(RL_mean)
    dts = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    for rowidx in range(len(RL_mean)):
        print(dts[rowidx], end = '& ')
        for colidx in range(len(RL_mean[0]) - 1):
            if colidx < len(RL_mean[0]) - 2:
                print(' %.2f (%.2f) & %.2f (%.2f) & '% (RL_mean[rowidx][colidx]*1e2, RL_std[rowidx][colidx]*1e2, \
                    coarse_mean[rowidx][colidx]*1e2, coarse_std[rowidx][colidx]*1e2), end = '')
            else:
                print(' %.2f (%.2f) & %.2f (%.2f)  '% (RL_mean[rowidx][colidx]*1e2, RL_std[rowidx][colidx]*1e2, \
                    coarse_mean[rowidx][colidx]*1e2, coarse_std[rowidx][colidx]*1e2), end = '')
        print('\\\\ \\hline')
    # return dx,dt,RL_mean,RL_std,coarse_mean,coarse_std


load_data('../NipsExpResult/rk4_u4.txt')