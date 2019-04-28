from copy import copy
from functools import reduce
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import cloudpickle

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import os
import time
from collections import deque
import pickle
import argparse
import time
import sys
import os
import queue
import yaml
import random
from copy import deepcopy
sys.path.append('/home/harshit/work/baselines/')

import MADRaS
from recovery_utils import *
#python libraries
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env

from baselines import logger
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None



#print(logger.get_dir())
logger.configure('logging/')


def _save_to_file(save_path, data=None, params=None):
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".pkl"

        with open(save_path, "wb") as file_:
            cloudpickle.dump((data, params), file_)
    else:
        # Here save_path is a file-like object, not a path
        cloudpickle.dump((data, params), save_path)


def save(sess,save_path):
    data = {
    }

    params = find_trainable_variables("model")

    params = sess.run(params)

    _save_to_file(save_path, data=data, params=params)



def _load_from_file(self,load_path):
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".pkl"):
                load_path += ".pkl"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

        with open(load_path, "rb") as file:
            data, params = cloudpickle.load(file)
    else:
        # Here load_path is a file-like object, not a path
        data, params = cloudpickle.load(load_path)

    return data, params

def load(sess, load_path, env=None, **kwargs):
    data, params = _load_from_file(load_path)

    # model = cls(None, env, _init_setup_model=False)
    # model.__dict__.update(data)
    # model.__dict__.update(kwargs)
    # model.set_env(env)
    # model.setup_model()
    params1 = find_trainable_variables("model")
    restores = []
    print(np.asarray(params1).shape)
    print(np.asarray(params).shape)
    for param, loaded_p in zip(params1, params):
        restores.append(param.assign(loaded_p))
    sess.run(restores)
    #return model


def find_trainable_variables(key):
    """
    Returns the trainable variables within a given scope

    :param key: (str) The variable scope
    :return: ([TensorFlow Tensor]) the trainable variables
    """
    
    return tf.trainable_variables()


def learn(network,
          env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=1,
          nb_rollout_steps=5000,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=False,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    set_global_seeds(seed)

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    nb_actions = env.action_space.shape[-1]
    #assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(network=network,name='main_critic', **network_kwargs)
    actor = Actor(nb_actions, network=network,name='main_actor', **network_kwargs)


    # Define recovery policy
    # with tf.variable_scope("recovery"):
    disaster_memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    recovery_critic = Critic(network=network,name='rec_critic', **network_kwargs)
    recovery_actor = Actor(nb_actions, network=network,name='rec_actor', **network_kwargs)


    action_noise = None
    param_noise = None

    recovery_action_noise = None
    recovery_param_noise = None

    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
                recovery_param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                recovery_action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                recovery_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    # agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
    #     gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    #     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    #     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #     reward_scale=reward_scale,target_actor_name="main_target_actor",target_critic_name="main_target_critic")

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    # recovery_agent = DDPG(recovery_actor, recovery_critic, disaster_memory, env.observation_space.shape, env.action_space.shape,
    #     gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    #     batch_size=batch_size, action_noise=recovery_action_noise, param_noise=recovery_param_noise, critic_l2_reg=critic_l2_reg,
    #     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #     reward_scale=reward_scale,name="_rec",target_actor_name="rec_target_actor",target_critic_name="rec_target_critic")

    recovery_agent = DDPG(recovery_actor, recovery_critic, disaster_memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=recovery_action_noise, param_noise=recovery_param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)


    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()
    # Prepare everything.
    saver = tf.train.Saver(max_to_keep=100)
    agent.initialize(sess)
    recovery_agent.initialize(sess)
    
    sess.graph.finalize()

    agent.reset()
    recovery_agent.reset()

    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar

    epoch = 0


    # setup tensorboard
    writer = tf.summary.FileWriter("output", sess.graph)
    writer.close()


    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        # Save model
        # if epoch%10==0:
        #     logger.info('saving model...')
        #     saver.save(sess, 'saved_models/my_model', global_step=epoch)
        #     agent.save("normal_agent")
        #     save(sess,"complete_agent")
        #     recovery_agent.save("recovery_agent")
        #     logger.info('done saving model!')

        for cycle in range(nb_epoch_cycles):

            

            # Perform rollouts.
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
                recovery_agent.reset()
            
            count = 0
            for t_rollout in range(nb_rollout_steps):

                count+=1
                # If you are in diaster free zone

                #if(1):
                if(obs[0,20]<=0.8 and obs[0,20]>=-0.8 ):


                    # Predict next action.

                    action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    
                    # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                    new_obs, r, done, info = env.step((max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    # note these outputs are batched from vecenv
                    episode_reward+=r

                    t += 1
                    if rank == 0 and render:
                        env.render()
                    #episode_reward += r
                    episode_step += 1
                    # if (count%1==0):
                    #     print("Main policy | Reward: {}, Cum Reward: {}".format(r,episode_reward))
                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    done = np.asarray(done)
                    r = np.asarray(r)
                    # print(obs.shape)
                    # print(action.shape)
                    # #print(r.shape)
                    # print(new_obs.shape)
                    # print(done.shape)

                    # obs = obs.reshape(1,-1)
                    # action = action.reshape(1,-1)
                    # new_obs = obs.reshape(1,-1)
                    # done = np.asarray(done).reshape(1,-1)
                    # r = np.asarray(r).reshape(1,-1)

                    agent.store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.

                # If you are in disaster zone deploy recovery policy
                else:
                    # Predict next action.

                    action, q, _, _ = recovery_agent.step(obs, apply_noise=True, compute_Q=True)
                    #episode_reward+=r
                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    
                    # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                    new_obs, r, done, info = env.step((max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    # note these outputs are batched from vecenv
                    
                    if(new_obs[0,20]<=0.8 or new_obs[0,20]>=-0.8):
                        r=[1.0]
                    else:
                        r=[-1.0]  


                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1
                    # if (count%1==0):
                    #     print(">>>>>>>Recovery policy | Reward: {}, Cum Reward: {}".format(r,episode_reward))


                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    done = np.asarray(done)
                    r = np.asarray(r)
                    # print(obs.shape)
                    # print(action.shape)
                    # #print(r.shape)
                    # print(new_obs.shape)
                    # print(done.shape)

                    # obs = obs.reshape(1,-1)
                    # action = action.reshape(1,-1)
                    # new_obs = obs.reshape(1,-1)
                    # done = np.asarray(done).reshape(1,-1)
                    # r = np.asarray(r).reshape(1,-1)

                    recovery_agent.store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.


                obs = new_obs



                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        print("Episode reward: {}".format(np.sum(episode_reward)))
                        logger.info("Episode reward: {}".format(np.sum(episode_reward)))
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if epoch%10==0:
                            logger.info('saving model...')
                            saver.save(sess, 'saved_models/my_model', global_step=epoch)

                        if nenvs == 1:
                            agent.reset()
                            recovery_agent.reset()



            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []

            epoch_recovery_actor_losses = []
            epoch_recovery_critic_losses = []
            epoch_recovery_adaptive_distances = []




            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                if disaster_memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = recovery_agent.adapt_param_noise()
                    epoch_recovery_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()
                if(disaster_memory.nb_entries>0):
                    cl_r, al_r = recovery_agent.train()
                    epoch_recovery_critic_losses.append(cl_r)
                    epoch_recovery_actor_losses.append(al_r)
                    recovery_agent.update_target_net()



            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):

                    if eval_obs[0,20]<=0.8 and eval_obs[0,20]>=-0.8:
                        eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        for d in range(len(eval_done)):
                            if eval_done[d]:
                                eval_episode_rewards.append(eval_episode_reward[d])
                                eval_episode_rewards_history.append(eval_episode_reward[d])
                                eval_episode_reward[d] = 0.0

                    else:
                        eval_action, eval_q, _, _ = recovery_agent.step(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        for d in range(len(eval_done)):
                            if eval_done[d]:
                                eval_episode_rewards.append(eval_episode_reward[d])
                                eval_episode_rewards_history.append(eval_episode_reward[d])
                                eval_episode_reward[d] = 0.0


        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

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

        combined_stats['train/loss_ra'] = np.mean(epoch_recovery_actor_losses)
        combined_stats['train/param_recovery_noise_distance'] = np.mean(epoch_recovery_adaptive_distances)
        combined_stats['train/loss_rc'] = np.mean(epoch_recovery_critic_losses)

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

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
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


    return agent


if __name__ == "__main__":

    #env = gym.make('MountainCarContinuous-v0')
    env = make_vec_env('Madras-v0', 'madras', 1, 7)
    model = learn('mlp',env,
        seed=7,
        nb_epochs=500
        
    )