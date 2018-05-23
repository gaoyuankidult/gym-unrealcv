import argparse
import time
import os
import logging
from baselines import logger_py2 as logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import training as training
import models_conv
import models
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import gym_unrealcv
import tensorflow as tf
from mpi4py import MPI

import img_preprocessing as I

MEM_SIZE = 1e4



def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--policy-learning-delay', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'test_mode', default=False)
    parser.add_argument('--name', type=str, default="ddpg")
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

#MAIN
args = parse_args()
if MPI.COMM_WORLD.Get_rank() == 0:
    logger.configure()

''' UNPACK SOME ARGS '''
env_id = args.pop('env_id')
seed = args.pop('seed')
noise_type = args.pop('noise_type')
layer_norm = args.pop('layer_norm')
evaluation = args.pop('evaluation')
render = args.pop('render')
#

#RUN
rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)

# Create envs.
env = gym.make(env_id)
env.rendering = render
env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

if evaluation and rank==0:
    eval_env = gym.make(env_id)
    eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    env = bench.Monitor(env, None)
else:
    eval_env = None

# Parse noise_type
action_noise = None
param_noise = None
nb_actions = env.action_space.shape[-1]
for current_noise_type in noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
        pass
    elif 'adaptive-param' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    elif 'ou' in current_noise_type:
        _, stddev = current_noise_type.split('_')
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
    else:
        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


##############
# Configure components.
''' Images or not? '''
obs_shape = env.observation_space.shape
def identity(x):
    return x
if len(obs_shape) == 3:
    print("< Image preprocessing enabled! >")
    print("< Using Conv-Nets! >")
    preprocessor = I.preprocess_img
    critic = models_conv.Critic(layer_norm=layer_norm)
    actor = models_conv.Actor(nb_actions, layer_norm=layer_norm)
    render = False #unrealcv renders without us calling the render() from the training loop, so we disable that
else:
    preprocessor = identity
    critic = models.Critic(layer_norm=layer_norm)
    actor = models.Actor(nb_actions, layer_norm=layer_norm)
observation_shape = preprocessor( np.ones(env.observation_space.shape) ).shape
memory = Memory(limit=int(MEM_SIZE), action_shape=env.action_space.shape, observation_shape=observation_shape )
##############

# Seed everything to make things reproducible.
seed = seed + 1000000 * rank
logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
tf.reset_default_graph()
set_global_seeds(seed)
env.seed(seed)
if eval_env is not None:
    eval_env.seed(seed)


''' START OF TRAINING '''
# Disable logging for rank != 0 to avoid noise.
if rank == 0:
    start_time = time.time()

training.train(env=env, preprocessor=preprocessor, obs_shape=obs_shape, eval_env=eval_env, render=render, param_noise=param_noise,
    action_noise=action_noise, actor=actor, critic=critic, memory=memory, **args)




''' END OF TRAINING '''
env.close()
if eval_env is not None:
    eval_env.close()
if rank == 0:
    logger.info('total runtime: {}s'.format(time.time() - start_time))
