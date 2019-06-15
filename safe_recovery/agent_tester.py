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
gflags.DEFINE_string('CKPT_FILE', None, 'Checkpoint file')
FLAGS(sys.argv)


def yaml_loader(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f)
        return Munch(data)


class Tester(Agent):
    def __init__(self):
        self.env = make_vec_env('Madras-v0', 'madras', 1, 7)
        self.config = yaml_loader(FLAGS.CONFIG_FILE)
        self.create_agent(self.env, self.config, training=False)

    def restore_model(self):
        if FLAGS.CKPT_FILE:
            self.saver.restore(self.sess, os.path.join(FLAGS.SAVEDIR, FLAGS.CKPT_FILE))
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.SAVEDIR))

    def test(self):
        obs = self.env.reset()
        episode_rewards_history = deque(maxlen=self.config.num_eval_rollouts)
        episode_steps_history = deque(maxlen=self.config.num_eval_rollouts)
        episode_violations_history = deque(maxlen=self.config.num_eval_rollouts)
        episode_recoveries_history = deque(maxlen=self.config.num_eval_rollouts)
        for _ in range(self.config.num_eval_rollouts):
            reward, num_steps, num_constraint_violations, num_recoveries = self.do_rollout(obs, 0)  # supplying 0 for epoch because we want to print results after each episode
            episode_rewards_history.append(reward)
            episode_steps_history.append(num_steps)
            episode_violations_history.append(num_constraint_violations)
            episode_recoveries_history.append(num_recoveries)
        print("Mean reward: ", np.mean(episode_rewards_history))
        print("Mean num steps: ", np.mean(episode_steps_history))
        print("Mean num violations: ", np.mean(episode_violations_history))
        print("Mean num recoveries: ", np.mean(episode_recoveries_history))

    def run_test(self, checkpoint=None):
        set_global_seeds(self.config.random_seed)
        self.sess = U.get_session()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_checkpoints_to_keep)

        # initialize the agents
        self.initialize(self.sess)
        self.sess.graph.finalize()
        self.restore_model()
        self.reset()

        self.test()


def main():
    tester = Tester()
    tester.run_test()


if __name__=='__main__':
    main()