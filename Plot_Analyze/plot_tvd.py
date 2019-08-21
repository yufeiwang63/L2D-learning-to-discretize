'''
file function:
load the tvd numpy file of two different models, and plot the figures for comparison.
'''

import numpy as np
from matplotlib import pyplot as plt
import argparse, sys

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument('--dir_0', type = str, default = None)
args.add_argument('--dir_1', type = str, default = None)
args = args.parse_args()

tvd_0 = np.load(args.dir_0 + '/tvd.npy')
tvd_1 = np.load(args.dir_1 + '/tvd.npy')

num = len(tvd_0)
plt.plot(range(num), tvd_0, label = args.dir_0)
plt.plot(range(num), tvd_1, label = args.dir_1)
plt.xlabel('Different Initial Conditions')
plt.ylabel('Total Positive TV Difference')
plt.legend()
plt.show()
figname = args.dir_0[11:] + '-' + args.dir_1[11:]
plt.savefig('../TVD_Figs/{0}.png'.format(figname))


