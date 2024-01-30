"""yeeun. for offpolicy """

import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Off-Policy-Benchmarks')

    # set up the algorithms.
    off_policy = ['DDPG', 'SAC', 'TD3', 'DDPGLag', 'TD3Lag', 'SACLag', 'DDPGPID', 'TD3PID', 'SACPID']
    eg.add('algo', off_policy)

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [2000])
    # eg.add('train_cfgs:total_steps', [2000 * 500])
    # which can reproduce results of 1e6 steps.

    # if you want to reproduce results of 3e6 steps, using
    # eg.add('algo_cfgs:steps_per_epoch', [2000])
    # eg.add('train_cfgs:total_steps', [2000 * 1500])

    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    gpu_id = [0, 1, 2, 3]
    # if you want to use CPU, please set gpu_id = None
    # gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # set up the environments.
    eg.add('env_id', [
        'SafetyHopperVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetySwimmerVelocity-v1',
        'SafetyAntVelocity-v1',
        'SafetyHalfCheetahVelocity-v1',
        'SafetyHumanoidVelocity-v1'
        ])
    eg.add('seed', [0, 5, 10, 15, 20])
    eg.run(train, num_pool=5, gpu_id=gpu_id)