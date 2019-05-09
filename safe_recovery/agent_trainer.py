import os
import sys
import yaml
import tensorflow as tf
import numpy as np
from collections import deque
import pickle
from munch import Munch
from colorama import Fore, Style

import MADRaS

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines import logger

import time
from model_base import Agent

import gflags
from gflags import FLAGS



gflags.DEFINE_string('CONFIG_FILE', 'agent_config.yml', 'Path to configuration file')
gflags.DEFINE_string('LOGDIR', 'logging/', 'Path to logdir')
gflags.DEFINE_string('SAVEDIR', 'saved_models/', 'Path to checkpoints')
FLAGS(sys.argv)


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def yaml_loader(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f)
        return Munch(data)


class Trainer(Agent):
    def __init__(self):
        self.env = make_vec_env('Madras-v0', 'madras', 1, 7)
        self.config = yaml_loader(FLAGS.CONFIG_FILE)
        self.set_up_MPI()
        self.create_agent(self.env, self.config, training=True)

    def save_model(self, epoch):
        logger.info('saving model...')
        self.saver.save(self.sess, os.path.join(FLAGS.SAVEDIR, 'my_model'), global_step=epoch)
        logger.info('done saving model!')

    def train(self):
        # initialize the environment
        obs = self.env.reset()
        self.nenvs = obs.shape[0]
        if self.nenvs != 1:
            raise ValueError("Only single environment training is supported.")

        episode_rewards_history = deque(maxlen=100)
        episode_steps_history = deque(maxlen=100)
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            rewards_over_cycles = []
            num_constraint_violations_over_cycles = []
            num_recoveries_over_cycles = []
            for _ in range(self.config.num_cycles_per_epoch):
                # pdb.set_trace()
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

                rewards, num_steps, num_constraint_violations, num_recoveries = self.do_rollout(obs, epoch)
                rewards_over_cycles.append(rewards)
                num_constraint_violations_over_cycles.append(num_constraint_violations)
                num_recoveries_over_cycles.append(num_recoveries)
                episode_rewards_history.append(rewards)
                episode_steps_history.append(num_steps)

                # Train
            for t_train in range(self.config.num_train_steps_per_cycle):
                # Adapt param noise, if necessary.
                if self.main_agent.Memory.nb_entries >= self.config.learning_params['main_agent']['batch_size'] and t_train % self.config.param_noise_adaption_interval == 0 and self.config.noise_type == 'adaptive-param':
                    distance = self.main_agent().adapt_param_noise()
                    epoch_adaptive_distances.append(distance)
                if self.config.recovery_mode_training:
                    if self.recovery_agent.Memory.nb_entries >= self.config.learning_params['recovery_agent']['batch_size'] and t_train % self.config.param_noise_adaption_interval == 0 and self.config.noise_type == 'adaptive-param':
                        distance = self.recovery_agent().adapt_param_noise()
                        epoch_recovery_adaptive_distances.append(distance)

                # Train main policy
                cl, al = self.main_agent().train()

                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                self.main_agent().update_target_net()

                if self.config.recovery_mode_training:
                    if self.recovery_agent.Memory.nb_entries>0:
                        # Train recovery policy
                        cl_r, al_r = self.recovery_agent().train()

                        epoch_recovery_critic_losses.append(cl_r)
                        epoch_recovery_actor_losses.append(al_r)
                        self.recovery_agent().update_target_net()

            end_time = time.time()
            epoch_time = end_time - start_time


            if epoch%self.config.save_freq==0:
                logger.info("Saving model")
                self.save_model(epoch)

            # Eval
            if self.config.do_eval and epoch%self.config.eval_freq==0:
                self.do_eval()
            
            self.log_stats(
                epoch,
                epoch_time,
                rewards_over_cycles,
                num_constraint_violations_over_cycles,
                num_recoveries_over_cycles,
                episode_rewards_history,
                episode_steps_history,
                epoch_actor_losses,
                epoch_critic_losses,
                epoch_adaptive_distances,
                epoch_recovery_actor_losses,
                epoch_recovery_critic_losses,
                epoch_recovery_adaptive_distances
                )

    def do_eval(self):
        raise NotImplementedError('Online eval not implemented yet')

    def log_stats(self,
                epoch,
                epoch_time,
                rewards_over_cycles,
                num_constraint_violations_over_cycles,
                num_recoveries_over_cycles,
                episode_rewards_history,
                episode_steps_history,
                epoch_actor_losses,
                epoch_critic_losses,
                epoch_adaptive_distances,
                epoch_recovery_actor_losses,
                epoch_recovery_critic_losses,
                epoch_recovery_adaptive_distances
                ):
        stats = self.main_agent().get_stats()
        combined_stats = stats.copy()

        # Total statistics.
        combined_stats['total/num_epochs'] = epoch + 1
        combined_stats['rollout/REWARD'] = np.mean(rewards_over_cycles)
        combined_stats['rollout/run_avg_eps_reward'] = np.mean(episode_rewards_history)
        combined_stats['rollout/run_avg_eps_steps'] = np.mean(episode_steps_history)
        combined_stats['train/main_actor_loss'] = np.mean(epoch_actor_losses)
        combined_stats['train/main_critic_loss'] = np.mean(epoch_critic_losses)
        if self.config.recovery_mode_training:
            if self.config.noise_type=='adaptive-param':
                combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                combined_stats['train/param_recovery_noise_distance'] = np.mean(epoch_recovery_adaptive_distances)
            combined_stats['rollout/num_violations'] = np.mean(num_constraint_violations_over_cycles)
            combined_stats['rollout/num_recoveries'] = np.mean(num_recoveries_over_cycles)
            combined_stats['train/recovery_actor_loss'] = np.mean(epoch_recovery_actor_losses)
            combined_stats['train/recovery_critic_loss'] = np.mean(epoch_recovery_critic_losses)
        combined_stats['total/epoch_duration'] = epoch_time
  
        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / self.MPI_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

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
        logger.configure(dir=FLAGS.LOGDIR, format_strs=['tensorboard'])
        logger.info('Using MAIN agent with the following configuration:')
        logger.info(str(self.main_agent.__dict__.items()))
        logger.info('Using RECOVERY agent with the following configuration:')
        logger.info(str(self.recovery_agent.__dict__.items()))

        # make a copy of the config file in the savedir
        os.system("cp %s %s"%(FLAGS.CONFIG_FILE, FLAGS.SAVEDIR))

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

def main():
    trainer = Trainer()
    trainer.run_training()

if __name__=='__main__':
    main()
