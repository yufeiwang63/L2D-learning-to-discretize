'''
from ziju shen. ask him for doc.
'''

from DDPG.DDPG_new import DDPG
from BurgersEnv.Burgers import Burgers
from Weno.weno3_2 import Weno3
import copy
import torch
import numpy as np
import copy
import os
import time
import math
import random
import init_util_2 as init_util
import matplotlib
from matplotlib import pyplot as plt

from get_args import get_args
from PPO.train_util import PPO_train, ppo_test, PPO_FCONV_train
from DDPG.train_util import DDPG_train, DDPG_test
from PPO.PPO_new import PPO
from DDPG.DDPG_new import DDPG
from DDPG.util import GaussNoise

import pickle

args, tb_writter = get_args()


### Important: fix numpy and torch seed! ####
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

print('State dimension: {0}'.format(args.state_dim))
print('Action dimension: {0}'.format(args.action_dim))

### RL agents 
agent = DDPG(copy.copy(args), GaussNoise(initial_sig=args.noise_beg), tb_logger=tb_writter)
#agent_dic = {'ppo': ppo, 'ddpg': ddpg}
#agent = agent_dic[args.agent]

### init training and testing encironments
if not args.large_scale_train:
    from init_util import init_env
else:  ### if we want train on large scales
    from init_util_loadprecomputed import init_env
#argss, train_env, test_env = init_env(args, agent)

#args.num_train = len(train_env)
#args.num_test = len(test_env)
#num_train, num_test = len(train_env), len(test_env)

dx_set=[0.01,0.02,0.04]
dt_max=0.02*args.cfl
dt_set=[dt_max,dt_max/2,dt_max/4]
Tscheme_set=['euler','rk4']
if not args.large_scale_train:
    init_funcs = [init_util.init_condition_a, init_util.init_condition_b, init_util.init_condition_c, init_util.init_condition_d, 
    init_util.init_condition_e, init_util.init_condition_e1, init_util.init_condition_e2, init_util.init_condition_e3, init_util.init_condition_e4, init_util.init_condition_e5, 
    init_util.init_condition_k, init_util.init_condition_l]
    init_name=['1_cos4','-1_cos4','-1_p2sin4','-1.5_p2sin4','1.5_m1.5sin4',
                '1_cos2','-1_cos2','-1_p2sin2','-1.5_p2sin2','1.5_m1.5sin2','-1.5_p2cos2','1.5_m1.5cos2']
else:
    pass

if args.load_path is not None:
    print('load models!')
    agent.load(args.load_path)

if args.picture_param_path:
    param_path=args.picture_param_path
    param=open(param_path,'r')
else:
    exit(0)


def save_figure(env, t_iter):
    rl_values = env.RLgrid[t_iter]
    true_values = env.get_precise_value(t_iter * env.dt)
    coarse_value=env.weno_coarse_grid[t_iter]
    T = env.dt * t_iter
    plt.figure()
    plt.plot(env.x_grid, true_values, lw=1, label = 'reference value',color='g')
    plt.plot(env.x_grid, coarse_value,alpha=0.5, lw=2, label = 'weno value',color='b')
    plt.plot(env.x_grid, rl_values, lw=2,color='r',alpha=0.5,label = 'RL value')
    ymax=np.max(true_values)+0.5
    ymin=np.min(true_values)-0.5
    print(ymax,ymin)
    plt.axis([-1.0,1.0,ymin,ymax])
    plt.legend(fontsize = 13,loc='upper right')
    plt.title('dx: {}  dt: {}  T {:.2f}'.format(env.dx, env.dt, T), fontsize = 13)
    plt.tight_layout()
        #plot error figure
    plt.savefig(env.args.save_RL_weno_animation_path + env.args.init + '{:.2f}'.format(env.T) + str(env.dx) + str(env.dt)+env.args.Tscheme+env.args.flux +'.png')
    plt.close()
    print("pic1 finish")

####init_idx, dx, dt ,Tscheme, flux, frame
for lines in param:
    print(lines)
    data=lines.split(' ')
    init=init_funcs[int(data[0].replace('\n',''))]
    init_name1=init_name[int(data[0])]
    exp_args=copy.copy(args)
    exp_args.T=args.T
    exp_args.dx=float(data[1])
    exp_args.init=init_name1
    exp_args.dt=float(data[2])
    exp_args.Tscheme=data[3]
    exp_args.flux=data[4]
    exp_args.save_RL_weno_animation_path=args.save_figure_path+'figure/'
    print(exp_args.save_RL_weno_animation_path)
    env= Burgers(exp_args,init,agent=agent)
    env.get_weno_precise()
    env.get_weno_corase()
    errors=DDPG_test(agent,[env],exp_args)
    save_figure(env,int(data[5].replace('\n','')))

'''
#for a sure dx dt Tscheme give error
dx=0.02
dt=dx*args.cfl
Tscheme=args.Tscheme
rl_error_set=[]
coarse_error_set=[]
rl_error_set_break=[]
coarse_error_set_break=[]
for i in range(10):
    exp_args=copy.copy(args)
    exp_args.T=args.T
    exp_args.dx=dx
    exp_args.dt=dt
    exp_args.init=init_name[i]
    exp_args.Tscheme=Tscheme
    env= Burgers(exp_args,init_funcs[i],agent=agent)
    env.get_weno_precise()
    env.get_weno_corase()
    errors=DDPG_test(agent,[env],exp_args)
    coarse_error = np.zeros(env.num_t)
    RL_error = np.zeros(env.num_t)
    coarse_error_break = np.zeros(env.num_t)
    RL_error_break = np.zeros(env.num_t)
    for i in range(env.num_t):
        breakidx, smoothidx = env.judge_break(i * env.dt)
        coarse_error[i], coarse_error_break[i] = env.relative_error(env.get_precise_value(i * env.dt),env.weno_coarse_grid[i], breakidx, smoothidx)
        RL_error[i], RL_error_break[i] = env.relative_error(env.get_precise_value(i * env.dt),env.RLgrid[i], breakidx, smoothidx)
    rl_error_set.append(RL_error)
    coarse_error_set.append(coarse_error)
    rl_error_set_break.append(RL_error_break)
    coarse_error_set_break.append(coarse_error_break)
  

def draw_pic(plt,x,data_mean,data_std,label=''):
    x_total=np.concatenate([x,x[-1::-1]],0)
    data_up=data_mean+data_std
    data_low=data_mean-data_std
    data_area=np.concatenate([data_up,data_low[-1::-1]],axis=0)
    plt.plot(x,data_mean,label=label)
    plt.fill(x_total,data_area,alpha=0.1)


rl_error_set=np.array(rl_error_set)
coarse_error_set=np.array(coarse_error_set)

# for idx in range(len(coarse_error_break)):
#     if np.isnan(coarse_error_break[idx]):
#         coarse_error_break[idx] = 0
coarse_error_set_break=np.clip(coarse_error_set_break,0,1)
# for idx in range(len(rl_error_set_break)):
#     if np.isnan(rl_error_set_break[idx]):
#         rl_error_set_break[idx] = 0
# rl_error_set_break[np.isnan(rl_error_set_break).astype(int)]=1
rl_error_set_break=np.clip(rl_error_set_break,0,1)


rl_error_mean=np.mean(rl_error_set,0)
rl_error_std=np.std(rl_error_set,0)

rl_error_set_break=np.array(rl_error_set_break)
coarse_error_set_break=np.array(coarse_error_set_break)
rl_error_mean_break=np.mean(rl_error_set_break,0)
rl_error_std_break=np.std(rl_error_set_break,0)



coarse_error_mean=np.mean(coarse_error_set,0)
coarse_error_std=np.std(coarse_error_set,0)
coarse_error_mean_break=np.mean(coarse_error_set_break,0)
coarse_error_std_break=np.std(coarse_error_set_break,0)

np.save(args.save_figure_path+"coarse_mean"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',coarse_error_mean)
np.save(args.save_figure_path+"coarse_std"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',coarse_error_std)
np.save(args.save_figure_path+"rl_mean"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',rl_error_mean)
np.save(args.save_figure_path+"rl_std"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',rl_error_std)
np.save(args.save_figure_path+"coarse_mean_break"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',coarse_error_mean_break)
np.save(args.save_figure_path+"coarse_std_break"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',coarse_error_std_break)
np.save(args.save_figure_path+"rl_mean_break"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',rl_error_mean_break)
np.save(args.save_figure_path+"rl_std_break"+ str(dx) + str(dt) + Tscheme+ args.flux+'.npy',rl_error_std_break)
plt.figure()
interval = 1
draw_pic(plt,np.arange(0,env.num_t,interval)*env.dt,rl_error_mean[0::interval],rl_error_std[0::interval], label='RL error')
draw_pic(plt,np.arange(0,env.num_t,interval)*env.dt,coarse_error_mean[0::interval],coarse_error_std[0::interval], label='weno error')

# draw_pic(plt,np.arange(0,env.num_t,interval)*env.dt,rl_error_mean_break[0::interval],rl_error_std_break[0::interval],label='RL error break')
# draw_pic(plt,np.arange(0,env.num_t,interval)*env.dt,coarse_error_mean_break[0::interval],coarse_error_std_break[0::interval],label='weno error break')

#plt.errorbar(np.arange(env.num_t)*env.dt,rl_error_mean,yerr=rl_error_std)
plt.legend(fontsize = 13,loc='upper right')
plt.title('dx: {0}  dt: {1} Tscheme: {2}'.format(dx, dt,Tscheme), fontsize = 13)
plt.tight_layout()
#plot error figure
plt.savefig(args.save_figure_path+ str(dx) + str(dt) + Tscheme+ args.flux+'error.png')
plt.close()
'''