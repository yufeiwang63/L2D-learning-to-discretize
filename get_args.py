'''
File Description:
This file defines all the command line arguments.
'''

import argparse
import sys
import numpy as np
# from tensorboardX import SummaryWriter

def get_args():
    '''
    This function defines and parses the command line arguments, returns a namespace variable that
    holds all the necessary arguments for the entire training procedure.
    '''
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--load_path', type=str, default=None)
    argparser.add_argument('--save_path', type=str, default=None)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--exp_id', type=str, default='debug')
    argparser.add_argument('--residual', type=int, default=0)
    argparser.add_argument('--supervise_weno_coef', type=float, default=0)

    ### PDE settings 
    argparser.add_argument('--dt', type=float, default=0.002)
    argparser.add_argument('--dx', type=float, default=0.02)
    argparser.add_argument('--precise_dx', type=float, default=0.001)
    argparser.add_argument('--precise_dt', type=float, default=0.00005)
    argparser.add_argument('--x_low', type=float, default=0)
    argparser.add_argument('--x_high', type=float, default=1)
    argparser.add_argument('--initial_t', type=float, default=0.)
    argparser.add_argument('--T', type=float, default=0.8)
    argparser.add_argument('--trainT', type=float, default=0.5)
    argparser.add_argument('--boundary_condition', type=str, default='periodic')
    argparser.add_argument('--Tscheme', type=str, default='euler')
    argparser.add_argument('--init', type=str, default='sinoffset')
    argparser.add_argument('--cfl', type=float, default=0.1)
    argparser.add_argument('--flux', type=str, default='u2')

    ### Training Env Settings
    argparser.add_argument('--state_mode', type=str, default='normalize', help = 'normalize, unnormalize, mix; see torch_network.py \
        in DDPG for more details.')
    argparser.add_argument('--input_normalize', type=int, default=0)
    argparser.add_argument('--num_train', type=int, default=6)
    argparser.add_argument('--num_test', type=int, default=5)
    argparser.add_argument('--state_window_size', type=int, default=2) ### v4 TODO: check
    argparser.add_argument('--action_window_size', type=int, default=2) ### v4 TODO: check
    argparser.add_argument('--state_dim', type=int, default=11)
    argparser.add_argument('--action_dim', type=int, default=5)
    argparser.add_argument('--mode', type=str, default='weno_coef_four')
    argparser.add_argument('--agent', type=str, default='ddpg')
    argparser.add_argument('--formulation', type=str, default='MLP') ### v4 TODO: remove

    ### Training logics settings 
    argparser.add_argument('--record_every', type=int, default=1)
    argparser.add_argument('--debug', type=bool, default=False)
    argparser.add_argument('--test_every', type=int, default=100)
    argparser.add_argument('--save_every', type=int, default=100)
    argparser.add_argument('--train_epoch', type=int, default=30000)
    argparser.add_argument('--num_steps', type=int, default=1000000)
    argparser.add_argument('--large_scale_train', type=int, default=0, help = '\
        whether to perform large-scale training. If set 1, random sample 40 initial conditions, set smaller learning rate, \
        and deeper networks')
    
    ### Reward Settings 
    argparser.add_argument('--reward_type', type = str, default = 'neighbor', help = 'rewarding mechanism, \
        {neighbor, single, all}')
    argparser.add_argument('--reward_scale', type = int, default = 0)


    ### about visulization
    argparser.add_argument('--show_RL_weno_animation', type=int, default=0, help = 'whether to show the evolving animation')
    argparser.add_argument('--save_RL_weno_animation_path', type=str, default='./data/video/', 
        help = 'save path of the evovling animation')
    argparser.add_argument('--test', type=bool, default=False)
    argparser.add_argument('--animation', type=int, default=0, help = 'whether to plot the evolving animation')
    argparser.add_argument('--video_interval', type=int, default=2000)


    ### General RL Algorithm Parameters
    argparser.add_argument('--gamma', type=float, default=0.99)
    argparser.add_argument('--a_lr', type=float, default=1e-4)
    argparser.add_argument('--final_a_lr', type=float, default=1e-7)
    argparser.add_argument('--c_lr', type=float, default=1e-3)
    argparser.add_argument('--final_c_lr', type=float, default=1e-7)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--optimizer', type=str, default='adam')
    argparser.add_argument('--hidden_layer_num', type=int, default=6)
    argparser.add_argument('--max_grad_norm', type=float, default=1)
    argparser.add_argument('--clip_gradient', type=int, default=0)
    argparser.add_argument('--sl_lr', type=float, default=1e-4)
    argparser.add_argument('--sl_train_step', type=float, default=1)

    ### DQN and DDPG and SAC parameter
    argparser.add_argument('--tau', type=float, default=0.02, help = 'soft update target network param')
    argparser.add_argument('--update_every', type=int, default=1, help = 'interval of hard copy params to the target network')
    argparser.add_argument('--replay_traj_num', type=int, default=10000) 
    argparser.add_argument('--replay_size', type=int, default=500000) 
    argparser.add_argument('--noise', type=str, default='action')
    argparser.add_argument('--noise_beg', type = float, default = 0.5, help = 'initial noise scale')
    argparser.add_argument('--noise_end', type = float, default = 0.001, help = 'final noise scale')
    argparser.add_argument('--noise_dec', type = float, default = 0.02, help = 'how much noise to decrease at a time')
    argparser.add_argument('--noise_dec_every', type = int, default = 400, help = 'interval of each noise decrease')
    argparser.add_argument('--num_process', type = int, default =4, help = 'num of paraller threadings.')
    argparser.add_argument('--ddpg_train_iter', type = int, default = 20)
    argparser.add_argument('--ddpg_value_train_iter', type = int, default = 5)
    argparser.add_argument('--update_mode', type = str, default = 'soft')
    argparser.add_argument('--multistep_return', type = int, default = 5, help = 'bellman equation unroll steps when updating the Q-network')
    argparser.add_argument('--ddpg_net', type = str, default = 'roe')
    argparser.add_argument('--automatic_entropy_tuning', type = int, default = 1, help='\
        whether to use sac v2')
    argparser.add_argument('--alpha', type = int, default = 0.2, help='sac alpha')
    argparser.add_argument('--hidden_size', type = int, default = 64, help='sac network hidden_size')
    argparser.add_argument('--updates_per_step', type = int, default = 1, help='sac gradient step')
    
    ### random seeds
    argparser.add_argument('--seed', type=int, default=6)
    args = argparser.parse_args()

    ### just support weno-coef 4 now
    # args.state_dim =  7 + 8 ### add weno weight as input: trying to learn a residual
    args.state_dim =  7 ### nips model
    args.action_dim = 8

    assert np.abs(args.dx * args.cfl - args.dt) < 1e-10, "dx, dt and cfl should match!"
    ### for DDPG, have to carefully set the replay memory size. It should at least hold several trajectories.
    if args.agent == 'ddpg':
        args.batch_size = args.batch_size * args.num_process
        args.replay_size = args.replay_traj_num
        # evolve_step = int(args.trainT / (args.dt))
        # num_x = int((args.x_high - args.x_low) / args.dx)
        # pair_per_train = evolve_step * num_x * args.num_process
        # args.batch_size = pair_per_train // args.ddpg_train_iter * 10
        # args.replay_size = 100 * pair_per_train
 

    return args
