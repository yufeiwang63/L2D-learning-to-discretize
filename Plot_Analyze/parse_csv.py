from numpy import genfromtxt
import numpy as np
import argparse
import os
from os import path as osp
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None)


args = parser.parse_args()

data_paths = [
    # './data/4-13-1',
    # './data/4-13-2',
    # './data/4-13-3-longer-noise',
    # './data/4-13-4-load-model-large-noise',
    # './data/4-13-5-load-model-small-noise',
    # './data/4-13-6-larger-memory',
    # './data/4-13-7-larger-memory',
    # './data/4-13-8-load-model-small-noise-large-memory',
    # './data/4-13-9-load-model-small-noise-large-memory',
    # './data/4-13-10-larger-mem',
    # './data/4-13-11-larger-mem',
    # './data/4-13-12-load-model-0.1-noise-4w-0.02',
    # './data/4-13-12-load-model-0.1-noise-4w-0.05',

    './data/4-16-1-10700',
    './data/4-16-2-10700',
    './data/4-16-3-10700-mem-6w',
    './data/4-16-4-10700-mem-6w',
    './data/4-16-5-6150',
    './data/4-16-6-6150',
    './data/4-16-7-6150-mem-6w',
    './data/4-16-8-6150-mem-6w',

]

def parse_init(line):
    beg = line.find("Train Env ") 
    if beg >= 0:
        beg += len("Train Env ")
        end = line.find(" flux")
        return line[beg:end] + 'test_train'

    beg = line.find("Test Env ") 
    if beg >= 0:
        beg += len("Test Env ")
        end = line.find(" flux")
        return line[beg:end] + 'test_test'

    return False

def parse_scheme(line):
    beg = line.find("u2 ") + len("u2 ")
    end = line.find(" final error ")
    return line[beg:end]

def parse_weno_error(line):
    # print(line)
    beg = line.find("weno-error ") + len("weno-error ")
    # print(line[beg:])
    return float(line[beg:])

def parse_std_out(path):
    stdout = osp.join(path, 'stdout.txt')
    stdout = open(stdout, 'r')
    lines = stdout.readlines()[:1000]
    # print(lines)
    test_idx = 0
    for idx in range(len(lines)):
        if 'test epoch: 100' in lines[idx]:
            test_idx = idx + 1
            break

    weno_errors = {}
    for idx in range(test_idx, test_idx + 1000):
        # print(lines[idx])
        init = parse_init(lines[idx])
        # print("init is:", init)
        if not init:
            break
        scheme = parse_scheme(lines[idx])
        weno_error = parse_weno_error(lines[idx])
        weno_errors[init + '_' +  scheme] = weno_error
    
    return weno_errors
    

for path in data_paths:
    print("=" * 50, path,  "=" * 50)
    csv_file = osp.join(path, 'progress.csv')
    var = osp.join(path, 'variant.json')
    script = osp.join(path, 'script.txt')

    script = open(script, 'r')
    script = script.readlines()
    script = script[0]
    print(script)

    weno_errors = parse_std_out(path)
    inits = weno_errors.keys()
    # print(inits)

    with open(var, 'r') as f:
        variant = json.load(f)
    test_every = variant['test_every']

    f = open(csv_file)
    lines = f.readlines()
    header = lines[0]
    header = header.split(',')

    errors = ['test_test_euler', 'test_test_rk4', 'test_train_euler', 'test_train_rk4']
    error_header_idxes = {k:[] for k in errors}
    # init_idxes = {}

    for idx, h in enumerate(header):
        for k in errors:
            if k in h and  not ('ret' in h):
                # print(k, h)
                error_header_idxes[k].append(idx)

        # for init in inits:
        #     for scheme in ['_euler', '_rk4']:
        #         if h == init + scheme:
        #             init_idxes[init + scheme] = idx

    # print(error_header_idxes)
    # print(init_idxes)
    # exit()

    data = genfromtxt(csv_file, delimiter=',')
    data = data[1:]

    num = len(data)
    sum_error = {k:np.zeros(num) for k in errors}
    max_error = {k:np.zeros(num) for k in errors}

    for idx in range(len(data)):
        for k in errors:
            sum_error[k][idx] = np.sum(data[idx][error_header_idxes[k]])

        # max_e = -1000
        # for init in inits:
        #     for scheme in ['_euler', '_rk4']:
        #         max_e = max(max_e, data[idx][init_idxes[init + scheme]] - weno_errors[init][scheme[1:]])

        for k in errors:
            max_e = -1000
            for k_idx in error_header_idxes[k]:
                key = header[k_idx].replace('\n', '')
                RL_minus_weno = data[idx][k_idx] - weno_errors[key]
                max_e = max(max_e, RL_minus_weno) 
            
            max_error[k][idx] = max_e   

    # for k in errors:
    #     # print(k)
    #     # for idx, x in enumerate(sum_error[k]):
    #     #     print("{} {}".format(idx * test_every, x))
    #     best = np.argsort(sum_error[k])
    #     print("=" * 20, k)
    #     print((best[:10] + 1) * test_every)
    #     print(sum_error[k][best[:5]])

    for k in errors:
        best = np.argsort(max_error[k])
        print("=" * 20, k)
        print((best[:10] + 1) * test_every)
        print(max_error[k][best[:5]])

    print()

