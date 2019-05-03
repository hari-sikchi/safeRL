import os
import yaml
import tensorflow as tf
import numpy as np
from collections import deque
import pickle

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines import logger

import time
from model_base import Agent

CONFIG_FILE = 'agent_config.yml'
LOGDIR = 'logging/'

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def yaml_loader(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f)
        return data


class Trainer(Agent):
    def __init__(self):
        self.env = make_vec_env('Madras-v0', 'madras', 1, 7)
        self.config = yaml_loader(CONFIG_FILE)
        self.create_agent(self.env, self.config, training=True)

    def save_model(self, epoch):
        logger.info('saving model...')
        self.saver.save(self.sess, 'saved_models/my_model', global_step=epoch)
        logger.info('done saving model!')

    def is_safe(self, obs):
        return obs[0,20] >= -0.8 and obs[0,20] <= 0.8

    def do_rollout(self, init_obs, epoch):
        max_action = self.env.action_space.high
        num_steps_in_episode = []
        episode_rewards_history = []
        episode_reward = np.zeros(self.nenvs, dtype = np.float32) #vector
        episode_step = np.zeros(self.nenvs, dtype = int) # vector

        obs = init_obs
        for t in range(self.config.num_rollout_steps):

            if self.is_safe(obs):
                action, q, _, _ = self.main_agent().step(obs, apply_noise=True, compute_Q=True)
                if self.MPI_rank == 0 and self.config.render:
                    self.env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, done, _ = self.env.step((max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                if not self.is_safe(new_obs):
                    r -= 1000

                episode_reward+=r
                episode_step += 1
                if t%self.config.logtostdout_freq==0:
                    print("\nMain policy | Reward: {}, Cumulative Reward: {}".format(r,episode_reward))

                # Book-keeping.
                done = np.asarray(done)
                r = np.asarray(r)

                self.main_agent().store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.

            else: # If you are in disaster zone deploy recovery policy
                action, q, _, _ = self.recovery_agent().step(obs, apply_noise=True, compute_Q=True)
                if self.MPI_rank == 0 and self.config.render:
                    self.env.render()
                
                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, done, _ = self.env.step((max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                if self.is_safe(obs):
                    r=[1.0]
                else:
                    r=[-1.0]  

                episode_reward += r
                episode_step += 1

                if t%self.config.logtostdout_freq==0:
                    print("\nRecovery policy | Reward: {}, Cum Reward: {}".format(r,episode_reward))

                # Book-keeping.
                done = np.asarray(done)
                r = np.asarray(r)

                self.recovery_agent().store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.

            obs = new_obs
            for d in range(len(done)):
                if done[d]:
                    # Episode done.
                    print("\n\nEpisode reward: {}\n\n".format(np.sum(episode_reward)))
                    logger.info("Episode reward: {}".format(np.sum(episode_reward)))
                    episode_rewards_history.append(episode_reward[d])
                    num_steps_in_episode.append(episode_step[d])
                    if epoch%self.config.save_freq==0:
                        logger.info("Saving model")
                        self.save_model(epoch)
                    if self.nenvs == 1:
                        self.reset()
        return episode_rewards_history, num_steps_in_episode

    def train(self):
        # initialize the environment
        obs = self.env.reset()
        self.nenvs = obs.shape[0]

        episode_rewards_history = deque(maxlen=100)
        episode_steps_history = deque(maxlen=100)
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            for _ in range(self.config.num_cycles_per_epoch):
                if self.nenvs > 1:
                    # if simulating multiple envs in parallel,
                    # impossible to reset agent at the end of the episode in each
                    # of the environments, so resetting here instead
                    self.reset()
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                epoch_recovery_actor_losses = []
                epoch_recovery_critic_losses = []
                epoch_recovery_adaptive_distances = []

                rewards, num_steps = self.do_rollout(obs, epoch)
                episode_rewards_history.extend(rewards)
                episode_steps_history.extend(num_steps)

                # Train
                for t_train in range(self.config.num_train_steps_per_cycle):
                    # Adapt param noise, if necessary.
                    if self.main_agent.Memory.nb_entries >= self.config.batch_size and t_train % self.config.param_noise_adaption_interval == 0:
                        distance = self.main_agent().adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    if self.recovery_agent.Memory.nb_entries >= self.config.batch_size and t_train % self.config.param_noise_adaption_interval == 0:
                        distance = self.recovery_agent().adapt_param_noise()
                        epoch_recovery_adaptive_distances.append(distance)

                    cl, al = self.main_agent().train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    self.main_agent().update_target_net()

                    if self.recovery_agent.Memory.nb_entries>0:
                        cl_r, al_r = self.recovery_agent().train()
                        epoch_recovery_critic_losses.append(cl_r)
                        epoch_recovery_actor_losses.append(al_r)
                        self.recovery_agent().update_target_net()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Eval
            if self.config.do_eval and epoch%self.config.eval_freq==0:
                self.do_eval()
            
            self.log_stats(
                epoch,
                epoch_time,
                episode_rewards_history,
                episode_steps_history,
                epoch_actor_losses,
                epoch_critic_losses,
                epoch_adaptive_distances,
                epoch_recovery_actor_losses,
                epoch_recovery_critic_losses,
                epoch_recovery_adaptive_distances)

    def do_eval(self):
        raise NotImplementedError('Online eval not implemented yet')

    def log_stats(self,
                epoch,
                epoch_time,
                episode_rewards_history,
                episode_steps_history,
                epoch_actor_losses,
                epoch_critic_losses,
                epoch_adaptive_distances,
                epoch_recovery_actor_losses,
                epoch_recovery_critic_losses,
                epoch_recovery_adaptive_distances):
        stats = self.main_agent().get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(episode_steps_history)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['train/loss_ra'] = np.mean(epoch_recovery_actor_losses)
        combined_stats['train/param_recovery_noise_distance'] = np.mean(epoch_recovery_adaptive_distances)
        combined_stats['train/loss_rc'] = np.mean(epoch_recovery_critic_losses)
        combined_stats['total/duration'] = epoch_time
  
        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / self.MPI_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if self.MPI_rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if self.MPI_rank == 0 and logdir:
            if hasattr(self.env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(self.env.get_state(), f)

    def run_training(self):
        set_global_seeds(self.config.random_seed)

        # initialize logger
        logger.configure(LOGDIR)
        logger.info('Using MAIN agent with the following configuration:')
        logger.info(str(self.main_agent.__dict__.items()))
        logger.info('Using RECOVERY agent with the following configuration:')
        logger.info(str(self.recovery_agent.__dict__.items()))

        # set up tensorflow session, saver and tensorboard
        self.sess = U.get_session()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_checkpoints_to_keep)
        self.writer = tf.summary.FileWriter("output", self.sess.graph)

        # initialize the agents
        self.initialize(self.sess)
        self.sess.graph.finalize()
        self.reset()

        self.train()

    def set_up_MPI(self):
        if MPI is not None:
            self.MPI_rank = MPI.COMM_WORLD.Get_rank()
            self.MPI_size = MPI.COMM_WORLD.Get_size()
        else:
            self.MPI_rank = 0
            self.MPI_size = 1

