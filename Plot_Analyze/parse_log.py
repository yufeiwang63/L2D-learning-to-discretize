import numpy as np
from matplotlib import pyplot as plt
import sys, argparse
import os

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument('--dir', type = str, default = '')

args = args.parse_args()

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]
if 'record.txt' in onlyfiles:
    onlyfiles.remove('record.txt')

for f in onlyfiles:
    file = open(args.dir + f, 'r')
    lines = file.readlines()
    lines = lines[1:]

    # first add all init keys
    euler_errors = {}
    rk4_errors = {}
    weno_euler_errors = {}
    weno_rk4_errors = {}
    for idx, line in enumerate(lines):
        if "Test Env " in line and 'euler final error' in line:
            idxbeg = len("Test Env ")
            idxend = line.find(" euler final error")
            init = line[idxbeg:idxend]
            if init in euler_errors.keys():
                break
            euler_errors[init] = []
            rk4_errors[init] = []
            
            weno_error_beg = line.find("weno-error ") + len("weno-error ")
            weno_euler_errors[init] = float(line[weno_error_beg:])

        if "Test Env " in line and 'rk4 final error' in line:
            idxbeg = len("Test Env ")
            idxend = line.find(" rk4 final error")
            init = line[idxbeg:idxend]
            
            weno_error_beg = line.find("weno-error ") + len("weno-error ")
            weno_rk4_errors[init] = float(line[weno_error_beg:])


    num = len(euler_errors)
    print("--------------------------------------------------")
    print(f)
    print("there are in total {} init conditions:".format(num))
    for k in euler_errors:
        print(k)

    for idx, line in enumerate(lines):
        if "Test Env " in line:
            idxbeg = len("Test Env ")
            

            if "euler final" in line:
                idxend = line.find(" euler final error")
                init = line[idxbeg:idxend]
                errorbeg = line.find('euler final error ') + len('euler final error ')
                errorend = line.find(' weno-error ')
                error = float(line[errorbeg:errorend])
                euler_errors[init].append(error)

            if "rk4 final" in line:
                idxend = line.find(" rk4 final error")
                init = line[idxbeg:idxend]
                errorbeg = line.find('rk4 final error ') + len('rk4 final error ')
                errorend = line.find(' weno-error ')
                error = float(line[errorbeg:errorend])
                rk4_errors[init].append(error)

    fig = plt.figure(figsize=(num * 5, 8))
    cnt = 1
    l = 0
    for k in euler_errors:
        l = len(euler_errors[k])
        ax = fig.add_subplot(2, num, cnt)
        ax.plot([i * 25 for i in range(len(euler_errors[k]))], np.log(euler_errors[k]), label = 'euler_{}'.format(k))
        ax.set_xlabel("Trained epochs")
        ax.set_ylabel("Log relative L2 error")
        ax.axhline(xmin = 0, xmax = len(euler_errors[k]), y = np.log(weno_euler_errors[k]), linestyle = 'dashed')
        ax.legend()

        ax2 = fig.add_subplot(2, num, cnt + num)
        ax2.plot([i * 25 for i in range(len(rk4_errors[k]))], np.log(rk4_errors[k]), label = 'rk4_{}'.format(k))
        ax2.set_xlabel("Trained epochs")
        ax2.set_ylabel("Log relative L2 error")
        ax2.axhline(xmin = 0, xmax = len(rk4_errors[k]), y = np.log(weno_rk4_errors[k]), linestyle = 'dashed')
        ax2.legend()

        cnt += 1

    eulers = np.zeros(l)
    for k in euler_errors:
        eulers += euler_errors[k]

    rk4s = np.zeros(l)
    for k in rk4_errors:
        rk4s += rk4_errors[k]

    euler_idx = np.argsort(eulers)
    rk4_idx = np.argsort(rk4s)

    min = 10
    print("euler mins:")
    for i in range(min):
        print("{} {}".format((euler_idx[i] + 1) * 25, eulers[euler_idx[i]]))
    
    print("rk4 mins:")
    for i in range(min):
        print("{} {}".format((rk4_idx[i] + 1) * 25, rk4s[rk4_idx[i]]))
    print("--------------------------------------------------")

    # plt.legend()
    plt.tight_layout()
    save_dir = 'figs/' + args.dir[len('output/'):]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = f[:-4]
    plt.savefig("{}{}.png".format(save_dir, save_name))




# error = sorted(errors, key = lambda x: x[1])
# print(error)