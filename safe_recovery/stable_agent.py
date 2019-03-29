from functools import reduce
import os
import time
from collections import deque
import pickle

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI
import gym
import numpy as np
import sys
sys.path.append('/home/harshit/work/stable-baselines/')

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger
from stable_baselines.ddpg.memory import Memory


from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

def learn(agent,recovery_agent,total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="DDPG"):



    with SetVerbosity(agent.verbose), TensorboardWriter(agent.graph, agent.tensorboard_log, tb_log_name) as writer:
        agent._setup_learn(seed)
        recovery_agent._setup_learn(seed)

        # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
        agent.tb_seen_steps = []
        recovery_agent.tb_seen_steps = []

        rank = MPI.COMM_WORLD.Get_rank()
        # we assume symmetric actions.
        # assert np.all(np.abs(senv.action_space.low) == self.env.action_space.high)
        if agent.verbose >= 2:
            logger.log('Using agent with the following configuration:')
            logger.log(str(agent.__dict__.items()))

        eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        agent.episode_reward = np.zeros((1,))
        recovery_agent.episode_reward = np.zeros((1,))

        with agent.sess.as_default(), agent.graph.as_default():
            # Prepare everything.
            agent._reset()
            recovery_agent._reset()

            obs = env.reset()
            eval_obs = None
            if eval_env is not None:
                eval_obs = eval_env.reset()

            episode_reward = 0.
            episode_step = 0
            episodes = 0
            step = 0
            total_steps = 0

            start_time = time.time()

            epoch_episode_rewards = []
            epoch_episode_steps = []
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            eval_episode_rewards = []
            eval_qs = []
            epoch_actions = []
            epoch_qs = []
            epoch_episodes = 0
            epoch = 0
            while True:
                for _ in range(log_interval):
                    # Perform rollouts.
                    for _ in range(agent.nb_rollout_steps):
                        if total_steps >= total_timesteps:
                            return agent


                        if(obs[0,20]<0.8 and obs[0,20]>-0.8 ):

                            # Predict next action.
                            action, q_value = agent._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == env.action_space.shape

                            # Execute next action.
                            if rank == 0 and agent.render:
                                env.render()
                            new_obs, reward, done, _ = env.step(action * np.abs(agent.action_space.low))

                            if writer is not None:
                                ep_rew = np.array([reward]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                agent.episode_reward = total_episode_reward_logger(agent.episode_reward, ep_rew, ep_done,
                                                                                    writer, total_steps)
                            step += 1
                            total_steps += 1
                            if rank == 0 and agent.render:
                                env.render()
                            episode_reward += reward
                            episode_step += 1

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)

                            agent._store_transition(obs, action, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                # Only stop training if return value is False, not when it is None. This is for backwards
                                # compatibility with callbacks that have no return statement.
                                if callback(locals(), globals()) == False:
                                    return agent
                        else:
                            # Predict next action.
                            action, q_value = recovery_agent._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == env.action_space.shape

                            # Execute next action.
                            if rank == 0 and recovery_agent.render:
                                env.render()
                            new_obs, reward, done, _ = recovery_agent.env.step(action * np.abs(self.action_space.low))

                            if writer is not None:
                                ep_rew = np.array([reward]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                recovery_agent.episode_reward = total_episode_reward_logger(recovery_agent.episode_reward, ep_rew, ep_done,
                                                                                    writer, total_steps)
                            step += 1
                            total_steps += 1
                            if rank == 0 and recovery_agent.render:
                                env.render()
                            episode_reward += reward
                            episode_step += 1

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)

                            recovery_agent._store_transition(obs, action, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                # Only stop training if return value is False, not when it is None. This is for backwards
                                # compatibility with callbacks that have no return statement.
                                if callback(locals(), globals()) == False:
                                    return recovery_agent



                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            episode_rewards_history.append(episode_reward)
                            epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            epoch_episodes += 1
                            episodes += 1
                            break


                    recovery_agent._reset()
                    agent._reset()
                    if not isinstance(env, VecEnv):
                        obs = env.reset()

                    # Train.
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []

                    epoch_rec_actor_losses = []
                    epoch_rec_critic_losses = []
                    epoch_rec_adaptive_distances = []


                    for t_train in range(agent.nb_train_steps):
                        # Adapt param noise, if necessary.
                        if agent.memory.nb_entries >= agent.batch_size and \
                                t_train % agent.param_noise_adaption_interval == 0:
                            distance = agent._adapt_param_noise()
                            epoch_adaptive_distances.append(distance)

                        # weird equation to deal with the fact the nb_train_steps will be different
                        # to nb_rollout_steps
                        step = (int(t_train * (agent.nb_rollout_steps / agent.nb_train_steps)) +
                                total_steps - agent.nb_rollout_steps)

                        critic_loss, actor_loss = agent._train_step(step, writer, log=t_train == 0)
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)
                        agent._update_target_net()
                        ## Update recovery agent
                        # Adapt param noise, if necessary.
                        if recovery_agent.memory.nb_entries >= recovery_agent.batch_size and \
                                t_train % recovery_agent.param_noise_adaption_interval == 0:
                            distance = recovery_agent._adapt_param_noise()
                            epoch_rec_adaptive_distances.append(distance)

                        # weird equation to deal with the fact the nb_train_steps will be different
                        # to nb_rollout_steps
                        step = (int(t_train * (recovery_agent.nb_rollout_steps / recovery_agent.nb_train_steps)) +
                                total_steps - recovery_agent.nb_rollout_steps)

                        critic_loss, actor_loss = recovery_agent._train_step(step, writer, log=t_train == 0)
                        epoch_rec_critic_losses.append(critic_loss)
                        epoch_rec_actor_losses.append(actor_loss)
                        recovery_agent._update_target_net()

                    # # Evaluate.
                    # eval_episode_rewards = []
                    # eval_qs = []
                    # if self.eval_env is not None:
                    #     eval_episode_reward = 0.
                    #     for _ in range(self.nb_eval_steps):
                    #         if total_steps >= total_timesteps:
                    #             return self

                    #         eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                    #         eval_obs, eval_r, eval_done, _ = self.eval_env.step(eval_action *
                    #                                                             np.abs(self.action_space.low))
                    #         if self.render_eval:
                    #             self.eval_env.render()
                    #         eval_episode_reward += eval_r

                    #         eval_qs.append(eval_q)
                    #         if eval_done:
                    #             if not isinstance(self.env, VecEnv):
                    #                 eval_obs = self.eval_env.reset()
                    #             eval_episode_rewards.append(eval_episode_reward)
                    #             eval_episode_rewards_history.append(eval_episode_reward)
                    #             eval_episode_reward = 0.

                mpi_size = MPI.COMM_WORLD.Get_size()
                # Log stats.
                # XXX shouldn't call np.mean on variable length lists
                duration = time.time() - start_time
                stats = agent._get_stats()
                combined_stats = stats.copy()
                combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                if len(epoch_adaptive_distances) != 0:
                    combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                combined_stats['total/duration'] = duration
                combined_stats['total/steps_per_second'] = float(step) / float(duration)
                combined_stats['total/episodes'] = episodes
                combined_stats['rollout/episodes'] = epoch_episodes
                combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                combined_stats['train/loss_ractor'] = np.mean(epoch_rec_actor_losses)
                combined_stats['train/loss_rcritic'] = np.mean(epoch_rec_critic_losses)


                # # Evaluation statistics.
                # if self.eval_env is not None:
                #     combined_stats['eval/return'] = eval_episode_rewards
                #     combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                #     combined_stats['eval/Q'] = eval_qs
                #     combined_stats['eval/episodes'] = len(eval_episode_rewards)






if __name__=="__main__":
    env = make_vec_env('Madras-v0', 'madras', 1, 7)
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    agent = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    recovery_agent = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

    model = learn(agent,recovery_agent,'mlp',env,
        seed=7,
        nb_epochs=500
        
    )
    


























env = gym.make('MountainCarContinuous-v0')
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=400000)
model.save("ddpg_mountain")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_mountain")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()