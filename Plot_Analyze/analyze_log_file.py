import numpy as np
from matplotlib import pyplot as plt
import sys, argparse

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument('--dir', type = str, default = '')

args = args.parse_args()

file = open(args.dir, 'r')
lines = file.readlines()
lines = lines[1:]

errors = []
for idx, line in enumerate(lines):
    sum_error_euler = 0
    sum_error_rk4 = 0
    if line.startswith('test epoch:'):
        epoch = int(line[len('test epoch: '):])
        for idx2 in range(idx + 13, idx + 25):
            if lines[idx2].find('euler final') != -1:
                errorbeg = lines[idx2].find('final error ') + len('final error ')
                errorend = lines[idx2].find(' weno-self-error ')
                error = float(lines[idx2][errorbeg:errorend])
                sum_error_euler += error
            elif lines[idx2].find('rk4 final') != -1:
                errorbeg = lines[idx2].find('final error ') + len('final error ')
                errorend = lines[idx2].find(' weno-self-error ')
                error = float(lines[idx2][errorbeg:errorend])
                sum_error_rk4 += error
        
        errors.append((epoch, sum_error_euler, sum_error_rk4))

error = sorted(errors, key = lambda x: x[1])
print(error)