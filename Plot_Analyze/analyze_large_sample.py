'''
File function: 
load and parse the log file of a pre-trained model on large-scale init functions.
Plot the figure comparing the order of accuracy of the trained RL model and weno itself.
The x-axis is the index of different environments.
The y-axis is the ratio of the order between RL methods and Weno.
The ratio is sorted from high to low.
'''



import numpy as np
from matplotlib import pyplot as plt

log_file = open('../Burgers-2019-03-30-13-22-06/log.txt', 'r')

lines = log_file.readlines()
for lidx, line in enumerate(lines):
    if line.find('test epoch: 2800') != -1:
        usefullines = lines[lidx + 1:lidx + 45]

record = []
for line in usefullines:
    if line == '\n':
        break
    init_begin = 10
    init_end = line.find('init_t') - 1
    init_cond = line[init_begin: init_end]

    rl_error_beg = line.find('final error') + 12
    rl_error_end = line.find('weno-self-error') - 1

    w_error_beg = line.find('weno-self-error') + 16

    rl_error = float(line[rl_error_beg: rl_error_end])
    weno_error = float(line[w_error_beg: ])

    print(init_cond)
    print(rl_error)
    print(weno_error)

    log_rl_error = np.log(rl_error)
    log_weno_error = np.log(weno_error)

    record.append((init_cond, -log_rl_error, -log_weno_error, -log_rl_error / -log_weno_error))
    
record = sorted(record, key = lambda x:(-x[3]))

for idx, x in enumerate(record):
    print('idx {0} init {1}'.format(idx, x[0]))


log_ratios = [x[3] for x in record]
plt.figure()
plt.hlines(y = 1.0, xmin = 0, xmax = len(log_ratios), linestyles='--', colors='r')
plt.plot(range(len(log_ratios)), log_ratios)
plt.ylabel('RL order / weno order')
plt.xlabel('Different initial conditions')
plt.show()
