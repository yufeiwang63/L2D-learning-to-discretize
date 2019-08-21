from matplotlib import pyplot as plt
import argparse, sys
import numpy as np
from Test_weno_error import get_weno_error
from self_weno_errors import self_weno_errors

args = argparse.ArgumentParser(sys.argv[0])


'''
file function: plot a initial condition test error Vs train epochs
'''

args.add_argument('--init_condition_0', type = str, default = '')
args.add_argument('--init_condition_1', type = str, default = '')
args.add_argument('--dir', type = str, default = '', help = 'The path of the model to be loaded')
args.add_argument('--mode', type = str, default = 'train')

args = args.parse_args()

log_file = args.dir + '/log.txt'
file = open(log_file, 'r')

plt.figure()
content = file.readlines()[1:]

def init_condition(x_center): # 0
    u0 = 0.5 + np.cos(2 * np.pi * x_center)
    # u0 = -1 + np.cos(2 * np.pi * x_center)
    return u0 

# weno_error = get_weno_error(0.02, 0.001, 1.0, init_condition)
weno_error = self_weno_errors['0.5_cos2']


if args.mode == 'test':
    args.init_condition_0 = 'twobreak'#'-1.5_p2sin2'#'0.5_cos2'
    args.init_condition_1 = ''#'0.5_m2cos'#'1.5_m1.5sin2'#'-1_cos2'
    init_conditions = [args.init_condition_0, args.init_condition_1]

    for init_condition in init_conditions:
        if init_condition == '':
            continue
        x = []
        y = []
        for idx, line in enumerate(content):
            if line.find('test epoch') != -1:
                # print(line)
                test_epoch = int(line[12:])
                x.append(test_epoch)
                for j in range(idx + 1, idx + 100):
                    if content[j][:4] == 'Test' and content[j].find(init_condition) != -1:
                        # print(content)
                        idx_ = content[j].index('error')
                        error = float(content[j][idx_ + 6:])
                        print(error)
                        y.append(np.log(error))
                        break


        plt.plot(x, y, label = init_condition)
        plt.xlabel('Training Epochs')
        plt.ylabel('Log Error')


    plt.legend()
    plt.title('Test Error Figure')
    plt.show()

elif args.mode == 'train':
    # args.init_condition_0 = '-1_cos2'
    # args.init_condition_0 = '0.5_cos2'
    # args.init_condition_0 = '-1.5_p2sin2'
    args.init_condition_0 = '1.5_m1.5sin2'
    init_condition = args.init_condition_0
    x = []
    ytest = []
    ytrain1 = []
    ytrain2 = []
    ytrain3 = []
    ytrain4 = []
    for idx, line in enumerate(content):
        if line.find('test epoch') != -1:
            # print(line)
            test_epoch = int(line[12:])
            x.append(test_epoch)
            for j in range(idx + 1, idx + 20):
                # print(content[j])
                if content[j][0] != 'T':
                    break
                if content[j][:4] == 'Test' and content[j].find(init_condition) != -1:
                    # print(content)
                    idx_ = content[j].index('error')
                    # idx2 = content[j].index('self-weno')
                    error = float(content[j][idx_ + 6: ])
                    # print(error)
                    ytest.append(np.log(error))
                if content[j][:5] == 'Train' and content[j].find(init_condition) != -1:
                    # print('Enter Training Branch')
                    train_idx_begin = content[j].index('init_t') 
                    train_idx_end = content[j].index('final') 
                    train_idx = content[j][train_idx_begin + 7:train_idx_end - 1]
                    # print(train_idx)
                    idx_ = content[j].index('error')
                    # idx2 = content[j].index('weno-self')
                    error = float(content[j][idx_ + 6: ])
                    # print(error)
                    if train_idx == '0':
                        ytrain1.append(np.log(error))
                    elif train_idx == '0.4':
                        ytrain2.append(np.log(error))
                    elif train_idx == '0.8':
                        ytrain3.append(np.log(error))
                    # elif train_idx == '0.7':
                    #     ytrain4.append(np.log(error))    

    plt.plot(x, ytest, label = 'Test')
    plt.plot(x, ytrain1, label = 'Train, 0')
    plt.hlines(xmin = 0, xmax = (len(x) - 1) * 25, y = np.log(weno_error), label='log weno error \n 0.02 and 0.001', linestyles='dashed')
    # plt.plot(x, ytrain2, label = 'Train, 0.4')
    # plt.plot(x, ytrain3, label = 'Train, 0.8')
    # plt.plot(x, ytrain4, label = 'Train, 0.7')
    plt.xlabel('Training Epochs')
    plt.ylabel('Log Error')


    plt.legend()
    plt.title(args.init_condition_0 + ' Error Figure')
    plt.show()


