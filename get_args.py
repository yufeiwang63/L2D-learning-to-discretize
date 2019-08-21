'''
File Description:
This file defines all the command line arguments.
'''

import argparse
import sys
import numpy as np
from tensorboardX import SummaryWriter

def get_args():
    '''
    This function defines and parses the command line arguments, returns a namespace variable that
    holds all the necessary arguments for the entire training procedure.

    ### About the .mode argument:
    During this project, we've developed several ways of formulating the problem (of solving numerial PDEs) into a MDP.
    These different formulations correspond to different 'mode' in this code implementation.
    
    # ENO:
        In this mode, the state of the agent is a neighborhood of the current grid point. E.g, if we want to compute 
        u_j^{n+1} and the agent now is at u_j^{n}, then the state is a vector consisting of (u_{j-s}, u_{j-s+1}, ..., u_{j}, ..., u_{j+s}).
        The neighborhood size s is controled by the .state_window_size argument. The action of the agent is a probability vector
        that chooses a filter/kernel from a pre-defined dictionary, which will be applied on the neighborhood to approximate 
        the spatial derivative. In this version, the pre-defined filter dictionary contains five filters: the central difference,
        the first-order and second-order upwind schemes (both directions). The agent will always choose the filter that is assigned
        the largers probability in the action.
    # WENO:
        The state of this mode is identical to that of the ENO mode. However, in this mode the agent uses all (combines) the filters in the 
        pre-defined dictionary to compute the spatial derivative instead only using the one with the highest probability (or, weight).
        The action of the agent is still a probability vector (or, weight vector), and it is used to weight average the approxiamations
        obtained by different filters in the pre-defined dictionary. E.g. if the action is (0.2, 0.1, 0.5, 0.1, 0.2), then the spatial
        approximation would be 0.2*central + 0.1*first-upwind-left + 0.5*frist-upwind-right + 0.1*second-upwind-left + 0.2*second-upwind-right,
        while in the mode ENO the approximation would just be first-upwind-right, as it has the highest weight(probability).
    # 
    '''
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--load_path', type=str, default=None)
    argparser.add_argument('--save_path', type=str, default=None)

    ### PDE settings 
    argparser.add_argument('--dt', type=float, default=0.004)
    argparser.add_argument('--dx', type=float, default=0.02)
    argparser.add_argument('--precise_dx', type=float, default=0.001)
    argparser.add_argument('--x_low', type=float, default=-1)
    argparser.add_argument('--x_high', type=float, default=1)
    argparser.add_argument('--initial_t', type=float, default=0.)
    argparser.add_argument('--T', type=float, default=1.0)
    argparser.add_argument('--boundary_condition', type=str, default='periodic')
    argparser.add_argument('--Tscheme', type=str, default='euler')
    argparser.add_argument('--init', type=str, default='sinoffset')
    argparser.add_argument('--cfl', type=float, default=0.2)
    argparser.add_argument('--flux', type=str, default='u2')

    ### Training Env Settings
    argparser.add_argument('--input_normalize', type=int, default=0) ### v4 TODO: remove
    argparser.add_argument('--num_train', type=int, default=7)
    argparser.add_argument('--num_test', type=int, default=6)
    argparser.add_argument('--state_window_size', type=int, default=2) ### v4 TODO: check
    argparser.add_argument('--action_window_size', type=int, default=2) ### v4 TODO: check
    argparser.add_argument('--state_dim', type=int, default=11)
    argparser.add_argument('--action_dim', type=int, default=5)
    argparser.add_argument('--mode', type=str, default='weno_coef_four')
    argparser.add_argument('--agent', type=str, default='ddpg')
    argparser.add_argument('--formulation', type=str, default='MLP') ### v4 TODO: remove
    argparser.add_argument('--prolong_every_epoch', type=int, default=100000000) ### v4 TODO: remove

    ### Training logics settings 
    argparser.add_argument('--record_every', type=int, default=1)
    argparser.add_argument('--debug', type=bool, default=False)
    argparser.add_argument('--test_every', type=int, default=25)
    argparser.add_argument('--save_every', type=int, default=100)
    argparser.add_argument('--train_epoch', type=int, default=15000)
    argparser.add_argument('--large_scale_train', type=int, default=0, help = '\
        whether to perform large-scale training. If set 1, random sample 40 initial conditions, set smaller learning rate, \
        and deeper networks')
    
    ### Reward Settings 
    argparser.add_argument('--reward_type', type = str, default = 'neighbor', help = 'rewarding mechanism, \
        {neighbor, single, all}')
    argparser.add_argument('--reward_time', type=int, default=10, help = 'how many times reward are given in the training \
        evolve process') ### v4 TODO: check
    argparser.add_argument('--reward_every_step', type=int, default=1, help = 'whether to give reward at every timestep, \
        will override the --reward_time argument') ### v4 TODO: could set reward_every_step as default, and remove the reward_time flag.
    argparser.add_argument('--reward_scale', type=float, default=1.) ### v4 TODO: could remove
    argparser.add_argument('--log_reward', type=int, default=1, help = 'whether to use the log scale of the error as reward')
    argparser.add_argument('--log_reward_clip', type=float, default=1e-50)
    argparser.add_argument('--log_reward_type', type=str, default='max', help = '{max: infinite norm of the error vec; \
        sum: one norm of the error vec}')
    argparser.add_argument('--tv_reward_coef', type=float, default=0, help = 'the coeficient of the tv different reward')
    
    ### about visulization
    argparser.add_argument('--save_RL_weno_animation', type=int, default=0, help = 'whether to save the evolving animation')
    argparser.add_argument('--show_RL_weno_animation', type=int, default=0, help = 'whether to show the evolving animation')
    argparser.add_argument('--save_RL_weno_animation_path', type=str, default='../BurgersVedios/', 
        help = 'save path of the evovling animation')
    argparser.add_argument('--test', type=bool, default=False)
    argparser.add_argument('--animation', type=int, default=0, help = 'whether to plot the evolving animation')
    argparser.add_argument('--plot_upwind', type=int, default=0, help = 'whether to compute and plot the upwind solution') ### v4 TODO: could remove

    ### General RL Algorithm Parameters
    argparser.add_argument('--gamma', type=float, default=0.99)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--a_lr', type=float, default=1e-4)
    argparser.add_argument('--final_a_lr', type=float, default=1e-7)
    argparser.add_argument('--c_lr', type=float, default=1e-3)
    argparser.add_argument('--final_c_lr', type=float, default=1e-7)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--optimizer', type=str, default='rmsprop')
    argparser.add_argument('--hidden_layer_num', type=int, default=3)
    argparser.add_argument('--max_grad_norm', type=float, default=1)
    
    ### DQN and DDPG parameter
    argparser.add_argument('--tau', type=float, default=0.01, help = 'soft update target network param')
    argparser.add_argument('--update_every', type=int, default=100, help = 'interval of hard copy params to the target network')
    argparser.add_argument('--replay_size', type=int, default=150000) 
    argparser.add_argument('--action_low', type=float, default=-100)
    argparser.add_argument('--action_high', type=float, default=100)
    argparser.add_argument('--noise', type=str, default='gauss')
    argparser.add_argument('--noise_beg', type = float, default = 0.1, help = 'initial noise scale')
    argparser.add_argument('--noise_end', type = float, default = 0.001, help = 'final noise scale')
    argparser.add_argument('--noise_dec', type = float, default = 0.005, help = 'how much noise to decrease at a time')
    argparser.add_argument('--upwindnoisebeg', type = float, default = 0.3, help = 'initial noise scale') ### v4 TODO: could remove, as we no longer need the special upwind mode
    argparser.add_argument('--upwindnoiseend', type = float, default = 0.01, help = 'final noise scale') ### v4 TODO: could remove, as we no longer need the special upwind mode
    argparser.add_argument('--upwindnoisedec', type = float, default = 0.015, help = 'how much noise to decrease at a time') ### v4 TODO: could remove, as we no longer need the special upwind mode
    argparser.add_argument('--noise_dec_every', type = int, default = 200, help = 'interval of each noise decrease')
    argparser.add_argument('--num_process', type = int, default = 4, help = 'num of paraller threadings.')
    argparser.add_argument('--ddpg_train_iter', type = int, default = 100)
    argparser.add_argument('--ddpg_value_train_iter', type = int, default = 5)
    argparser.add_argument('--update_mode', type = str, default = 'soft')
    argparser.add_argument('--multistep_return', type = int, default = 5, help = 'bellman equation unroll steps when updating the Q-network')
    argparser.add_argument('--handjudge_upwind', type = int, default = 1) ### v4 TODO: could remove, old mode flags.
    argparser.add_argument('--mem_type', type = str, default = 'multistep_return') ### v4 TODO: could remove, and set the multistep_return mem type as default.
    argparser.add_argument('--supervise_weno', type = int, default = 0) ### v4 TODO: could remove, actually never used in the experiments
    argparser.add_argument('--ddpg_net', type = str, default = 'roe')

    ### AC and PPO parameter
    ### v4 TODO: could remove these AC and PPO parameters since the final agent is DDPG.
    argparser.add_argument('--ac_train_epoch', type = int, default = 200)
    argparser.add_argument('--ppo_train_epoch', type = int, default = 100)
    argparser.add_argument('--ppo_agent', type = int, default = 6)
    argparser.add_argument('--ppo_eps', type = float, default = 0.2)
    argparser.add_argument('--ppo_lambda', type = float, default = 1)
    argparser.add_argument('--ppo_obj', type=str, default='ppo')
    argparser.add_argument('--ppo_value_train_iter', type=int, default=3, help = 'number of updates for the value net \
        per policy net update')
    argparser.add_argument('--entropy_coef', type=float, default=0.0)
    argparser.add_argument('--sample_all', type=int, default=1, help = 'whether to train on all collected tuples')

    ### random seeds
    argparser.add_argument('--py_rng_seed', type=int, default=6)
    argparser.add_argument('--np_rng_seed', type=int, default=66)
    argparser.add_argument('--torch_rng_seed', type=int, default=666)

    ### for ploting figures
    argparser.add_argument('--plot_y_low', type=float, default=-4)
    argparser.add_argument('--plot_y_high', type=float, default=4)
    argparser.add_argument('--compute_tvd', type=int, default=0, help = 'whether to compute the total positive tv difference of a\
            trained model')
    argparser.add_argument('--picture_param_path', type=str, default=None)
    argparser.add_argument('--save_figure_path', type=str, default=None)

    ### constrained flux args
    ### v4 TODO: could remove, as we do not use this specially designed constrained flux network
    argparser.add_argument('--p', type=int, default=2)
    argparser.add_argument('--q', type=int, default=3)
    argparser.add_argument('--scipy_verbose', type=int, default=1)
    argparser.add_argument('--flux_net_hidden_size', type=int, default=128)
    argparser.add_argument('--sigma', type=float, default=3.)
    argparser.add_argument('--value_pre_train_time', type=int, default=1000)
    argparser.add_argument('--constrained_ppo_value_train_iter', type=int, default=100)
    argparser.add_argument('--constrain', type=int, default=1)

    args = argparser.parse_args()

    ### set state and action dimensions for different modes.
    ### if not specified, then its eno, action dim 1, state dim 11 (2 * 5 + 1)
    if args.mode == 'compute_flux':
        args.state_window_size = 3
        args.action_dim = 2 
        args.state_dim = 2 * (1 + 2 * args.state_window_size)
    elif args.mode == 'continuous_filter':
        args.state_window_size = 3
        args.action_dim = 3
        args.state_dim = 2 * (1 + 2 * args.state_window_size)
    elif args.mode == 'constrained_flux':
        args.state_dim = args.p + args.q + 1
    elif args.mode == 'weno_coef':
        args.state_dim = 7
        if args.handjudge_upwind:
            args.action_dim = 6
        else:
            args.action_dim = 8
    elif args.mode == 'weno_coef_four':
        args.state_dim =  7
        args.action_dim = 8
    elif args.mode == 'nonlinear_weno_coef':
        args.state_dim =  7
        args.action_dim = 10

    ### for DDPG, have to carefully set the replay memory size. It should at least hold several trajectories.
    if args.agent == 'ddpg':
        if args.mem_type == 'batch_mem':
            args.batch_size = 75 * args.num_process
            args.replay_size = 50 * args.num_process * 75
        elif args.mem_type == 'multistep_return':
            evolve_step = int(args.T / (args.dx * args.cfl))
            num_x = int((args.x_high - args.x_low) / args.dx)
            pair_per_train = evolve_step * num_x * args.num_process
            args.batch_size = pair_per_train // args.ddpg_train_iter * 10
            args.replay_size = 100 * pair_per_train
 
    ### make save paths
    if args.save_path is None:
        import time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        args.save_path = '../Burgers-' + current_time
        if args.debug:
            args.save_path += '-debug/'
        else:
            args.save_path += '/'

        # args.weno_vedio_save_path =  args.weno_vedio_save_path + current_time + '/'
        # args.RL_weno_vedio_save_path = args.RL_weno_vedio_save_path + current_time + '/'
    
    tb_writter = None
    import os
    if not os.path.exists(args.save_path):
        if not args.test and not args.animation:
            os.makedirs(args.save_path)
            tb_writter = SummaryWriter(log_dir=args.save_path + 'tb/')
    if args.save_RL_weno_animation and not os.path.exists(args.save_RL_weno_animation_path):
        os.makedirs(args.save_RL_weno_animation_path)

    if args.reward_type == 'difference':
        args.reward_every_step = 1

    return args, tb_writter
