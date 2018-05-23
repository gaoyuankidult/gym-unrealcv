import os
import time
from collections import deque
import pickle

from ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger_py2 as logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import sys

save_dir = "/NNstash/ddpg/"

def train(env, preprocessor, obs_shape, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    test_mode, name, tau=0.01, eval_env=None, param_noise_adaption_interval=50, policy_learning_delay=0):
    rank = MPI.COMM_WORLD.Get_rank()
    inputs = {
              "env" : env,
              "eval_env" : eval_env,
              "gamma" : gamma,
              "tau" : tau,

              "memory" : memory,
              "actor" : actor,
              "actor_lr" : actor_lr,
              "critic" : critic,
              "critic_lr" : critic_lr,
              "critic_l2_reg" : critic_l2_reg,
              "policy_learning_delay" : policy_learning_delay,

              "nb_eval_steps" : nb_eval_steps,
              "nb_epochs" : nb_epochs,
              "nb_epoch_cycles" : nb_epoch_cycles,
              "nb_train_steps" : nb_train_steps,
              "nb_rollout_steps" : nb_rollout_steps,

              "render" : render,
              "render_eval" : render_eval,

              "param_noise_adaption_interval" : param_noise_adaption_interval,
              "action_noise" : action_noise,
              "param_noise" : param_noise,

              "popart" : popart,
              "normalize_returns" : normalize_returns,
              "normalize_observations" : normalize_observations,
              "reward_scale" : reward_scale,
              "clip_norm" : clip_norm,
              "batch_size" : batch_size,

              "preprocessor" : preprocessor,
              "obs_shape" : obs_shape,
              "test_mode" : test_mode,
              "name" : name,
              }
    print("---trainSettings---")
    for x in inputs:
        print (x, inputs[x])
    print("-------------------")

    ''' Make the actions scale over the actions so the agent can stick to [-1, 1] '''
    max_action = env.action_space.high
    min_action = env.action_space.low
    action_mask = np.array([1,1,0])
    middle_action = (0.5*min_action+0.5*max_action) * action_mask
    scale_action = (max_action-middle_action) * action_mask

    logger.info('scaling actions by {} before executing in env'.format(max_action))
    env_un = env.unwrapped
    observation_shape = preprocessor( np.ones(obs_shape) ).shape
    agent = DDPG(actor, critic, memory, preprocessor, observation_shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale, policy_learning_delay=policy_learning_delay)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        saver = tf.train.Saver()
        if test_mode:
            print ("Loading {}".format(name))
            meta_file = name.split("/")[-1]
            meta_folder = name.split(meta_file)[0]
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, tf.restore_latest_checkpoint(meta_folder))
        else:
            saver.save(sess, save_dir+name, global_step=0,write_meta_graph=True)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            print ("======================================")
            print (" Epoch {}:".format(epoch))
            print ("======================================")
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q, q_target, epsilon = agent.pi(obs, epsilon_mode=True, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()

                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(scale_action * action + middle_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    print ( "{}Q:{}({}) \t A:{} \t-> R:{}".format("!"*int(epsilon),q,q_target,action,r) )
                    t += 1

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()
                        print("---<episode_end>---")

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []


                #############################
                #############################
                #############################
                print ("Training({})".format(agent.policy_learning_delay)),
                for t_train in range(nb_train_steps):
                    print("."),
                    sys.stdout.flush()
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()
                print("!")
                print ("epsilon={}".format(agent.epsilon()))

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)



            ########################
            ########################
            ########################
            ########################
            if not test_mode and epoch%100 == 0:
                print ("Saving net..."),
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                sys.stdout.flush()
                saver.save(sess, save_dir+name, global_step=t,write_meta_graph=False)
                print("[x]")
