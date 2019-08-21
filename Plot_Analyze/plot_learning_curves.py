'''
Old file. No longer used. --yfw, 04/18/2019
'''

from matplotlib import pyplot as plt
import numpy as np
import argparse
import sys

ArgumentParser = argparse.ArgumentParser(sys.argv[0])

ArgumentParser.add_argument('--load_path', type = str)
ArgumentParser.add_argument('--train_num', type = int, default = 0)
ArgumentParser.add_argument('--test_num', type = int, default = 0)

args = ArgumentParser.parse_args()

## for gym test plot figures
if args.train_num == 0:
    train_rewards = np.load(args.load_path + 'train_plot_rewards.npy')
    train_idxes = np.load(args.load_path + 'train_plot_idxes.npy')
    test_rewards = np.load(args.load_path + 'test_plot_rewards.npy')
    test_idxes = np.load(args.load_path + 'test_plot_idxes.npy')
    param_log = open(args.load_path + 'Train_Parameters.txt', 'r')
    figure_title = ''
    for line in param_log.readlines():
        figure_title += line

    # print(train_rewards)
    # print(test_rewards)

    plt.figure()
    plt.plot(train_idxes, train_rewards, label = 'Training Curve')
    plt.plot(test_idxes, test_rewards, label = 'Testing Curve')
    plt.title(figure_title)
    plt.show()
    plt.savefig(args.load_path + 'learning curves.png')
    plt.close()

## for RL PDE project plot figures
else:
    train_errors = np.load(args.load_path + 'train_plot_errors.npy')
    train_idxes = np.load(args.load_path + 'train_plot_idxes.npy')
    test_errors = np.load(args.load_path + 'test_plot_errors.npy')
    test_idxes = np.load(args.load_path + 'test_plot_idxes.npy')
    
    param_log = open(args.load_path + 'Train_Parameters.txt', 'r')
    figure_title = ''
    for line in param_log.readlines():
        figure_title += line

    name_log = open(args.load_path + 'Burgers Initial Conditions.txt', 'r')
    initials = []
    for line in name_log.readlines():
        initials.append(line)


    train_errors = [np.log(x) for x in train_errors]
    test_errors = np.log(test_errors)
    # print('train_errors: ', train_errors)
    # print('test errors: ', test_errors)


    for i in range(args.train_num):
        plt.subplot(1, args.train_num, i + 1)
        plt.plot(train_idxes[i], train_errors[i], label = initials[i] + '-Training Curve')
        plt.plot(test_idxes, test_errors[i], label = initials[i] + '-Testing Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Log Error')
        plt.legend()

    plt.title(figure_title[i])
    plt.show()
    plt.savefig(args.load_path + 'learning curves.png')
    plt.close()

