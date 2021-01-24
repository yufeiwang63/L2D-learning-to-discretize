from launchers.train_td3 import run_task
from chester.run_exp import run_experiment_lite, VariantGenerator
import time
import click
import numpy as np

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    vg = VariantGenerator()
    vg.add('load_path', [None])
    # vg.add('load_path', [None])
    vg.add('load_epoch', [2550])

    ### PDE settings 
    vg.add('boundary_condition', ['periodic'])
    vg.add('Tscheme', ['euler'])
    vg.add('solution_data_path', ['data/local/solutions/9-24-50-eta-0.01-forcing-1'])
    vg.add('flux', ['u2'])
    vg.add('dx', [[0.02 * np.pi, 0.04 * np.pi]])
    vg.add('eta', [0.01])

    ### Training Env Settings
    vg.add('state_mode', ['normalize']) # 'normalize', 'unnormalize', 'mix'
    vg.add('state_dim', [7])
    vg.add('action_dim', [4])
    vg.add('weno_freq', [0.5])
    vg.add('no_done', [True])
    vg.add('same_time', [True, False])

    ### Training logics settings 
    vg.add('test_interval', [100])
    vg.add('save_interval', [50])
    vg.add('train_epoch', [30000])
    
    ### Reward Settings 
    vg.add('reward_width', [0, 3])
    vg.add('reward_first_deriv_error_weight', [0])

    ### General RL Algorithm Parameters
    vg.add('gamma', [0.99])
    vg.add('actor_lr', [1e-4])
    vg.add('final_actor_lr', [1e-7])
    vg.add('critic_lr', [1e-3])
    vg.add('final_critic_lr', [1e-7])
    vg.add('batch_size', [64])
    vg.add('policy_hidden_layers', [[64, 64, 64, 64, 64, 64]])
    vg.add('critic_hidden_layers', [[64, 64, 64, 64, 64, 64, 64]])
    vg.add('max_grad_norm', [0.5])
    vg.add('clip_gradient', [0])
    vg.add('lr_decay_interval', [0, 2000])

    ### DDPG parameter
    vg.add('tau', [0.02])
    vg.add('replay_buffer_size', [1000000]) 
    vg.add('noise_beg', [0.2])
    vg.add('noise_end', [0.01])
    vg.add('noise_dec', [0.04])
    vg.add('noise_dec_every', [500])
    vg.add('ddpg_value_train_iter', [2])
    vg.add('batch_norm', [False])
    
    if not debug:
        vg.add('seed', [100])
    else:
        vg.add('seed', [100])

    exp_prefix = '9-25-many-64-multiple-dx-forcing-eta-0.01'

    print("there are {} variants to run".format(len(vg.variants())))
    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None

        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break

if __name__ == '__main__':
    main()
