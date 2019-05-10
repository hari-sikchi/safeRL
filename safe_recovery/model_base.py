import numpy as np
import tensorflow as tf
from copy import deepcopy

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory

from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines import logger
from colorama import Fore, Style


class ExplorationNoise:
    def __init__(self, config, nb_actions):
        self.noise_type = config.noise_type
        self.noise_std = config.noise_std
        self.nb_actions = nb_actions

        self.set_noise_vars()

    def set_noise_vars(self):
        if not self.noise_type or self.noise_type == 'none':
            self.action_noise = None
            self.param_noise = None
        elif self.noise_type == 'adaptive-param':
            self.action_noise = None
            self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(self.noise_std), desired_action_stddev=float(self.noise_std))
        elif self.noise_type == 'normal':
            self.action_noise = NormalActionNoise(mu=np.zeros(self.nb_actions), sigma=float(self.noise_std) * np.ones(self.nb_actions))
            self.param_noise = None
        elif self.noise_type == 'ou':
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.nb_actions), sigma=float(self.noise_std) * np.ones(self.nb_actions))
            self.param_noise = None
        else:
            raise RuntimeError('unknown noise type "{}"'.format(self.noise_type))



class ActorCriticAgent:
    def __init__(self, env, config, name='', training=True, tag='main_agent', **kwargs):
        self.name = name
        self.env = env
        self.config = config
        self.nb_actions = self.env.action_space.shape[-1]
        self._training = training
        self.tag = tag

        self.create_agent(kwargs)

    def __call__(self):
        return self.agent

    @property
    def IsTraining(self):
        return self._training

    def build_model(self, network_kwargs):
        self.Actor = Actor(self.nb_actions, network=self.config.network, name=self.name+'_actor', **network_kwargs)
        self.Critic = Critic(network=self.config.network, name=self.name+'_critic', **network_kwargs)

    def create_replay_buffer(self):
        self.Memory = Memory(limit=self.config.replay_memory_size, action_shape=self.env.action_space.shape, observation_shape=self.env.observation_space.shape)

    def create_agent(self, kwargs):
        self.build_model(kwargs)
        self.create_replay_buffer()
        self.noise = ExplorationNoise(self.config, self.nb_actions)
        self.agent = DDPG(self.Actor,
                        self.Critic, 
                        self.Memory, 
                        self.env.observation_space.shape, 
                        self.env.action_space.shape,
                        gamma=self.config.learning_params[self.tag]['gamma'], 
                        tau=self.config.learning_params[self.tag]['tau'],
                        normalize_returns=self.config.learning_params[self.tag]['normalize_returns'],
                        normalize_observations=self.config.learning_params[self.tag]['normalize_observations'],
                        batch_size=self.config.learning_params[self.tag]['batch_size'],
                        action_noise=self.noise.action_noise,
                        param_noise=self.noise.param_noise,
                        critic_l2_reg=self.config.learning_params[self.tag]['critic_l2_reg'],
                        actor_lr=self.config.learning_params[self.tag]['actor_lr'],
                        critic_lr=self.config.learning_params[self.tag]['critic_lr'],
                        enable_popart=self.config.learning_params[self.tag]['popart'],
                        clip_norm=self.config.learning_params[self.tag]['clip_norm'],
                        reward_scale=self.config.learning_params[self.tag]['reward_scale']
                        )

    def initialize(self, sess):
        self.agent.initialize(sess)

    def reset(self):
        self.agent.reset()


class Agent:
    def create_agent(self, env, config, training=True):
        self.env = env
        self.config = config
        self.max_action = self.env.action_space.high
        self._training = training
        self.main_agent = ActorCriticAgent(self.env, self.config, name='main', training=self._training, tag='main_agent')
        self.recovery_agent = ActorCriticAgent(self.env, self.config, name='recovery', training=self._training, tag='recovery_agent')

    def initialize(self, sess):
        self.main_agent.initialize(sess)
        self.recovery_agent.initialize(sess)

    def reset(self):
        self.main_agent.reset()
        self.recovery_agent.reset()

    def is_safe(self, obs):
        return obs[0,20] >= -0.8 and obs[0,20] <= 0.8

    def get_violation_margin(self, obs):
        if obs[0,20] < -0.8:
            return np.abs(obs[0,20] + 0.8)
        elif obs[0,20] > 0.8:
            return np.abs(obs[0,20] - 0.8)
        else:
            return 0.0

    def normal_step(self, obs):
        action, _, _, _ = self.main_agent().step(obs, apply_noise=True, compute_Q=True)
        if self.config.render:
            self.env.render()

        # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
        new_obs, r, done, _ = self.env.step((self.max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
        # note these outputs are batched from vecenv

        r *= 2
        if self.config.recovery_mode_training:
            if not self.is_safe(new_obs):
                r -= 100

        # Book-keeping.
        done = np.asarray(done)
        r = np.asarray(r)

        self.main_agent().store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.
        return new_obs, r, done, {}

    def recovery_step(self, obs):
        action, _, _, _ = self.recovery_agent().step(obs, apply_noise=True, compute_Q=True)
        if self.config.render:
            self.env.render()
        
        # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
        new_obs, r, done, _ = self.env.step((self.max_action * action))  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
        # note these outputs are batched from vecenv

        if self.is_safe(obs):
            r=[1000.0]
        else:
            # r=[-1.0]
            r = [-10*self.get_violation_margin(obs)]

        # Book-keeping.
        done = np.asarray(done)
        r = np.asarray(r)

        self.recovery_agent().store_transition(obs, action, r, new_obs, done*np.ones(1)) #the batched data will be unrolled in memory.py's append.
        return new_obs, r, done, {}

    def do_rollout(self, init_obs, epoch):
        episode_reward = 0  # np.zeros(self.nenvs, dtype = np.float32) #vector
        reward_buffer_for_log = []
        num_constraint_violations = 0
        num_recoveries = 0
        danger_zone_flag = 0
        last_logged_episode_step = 0
        obs = init_obs
        for episode_step in range(self.config.max_rollout_steps):  #TODO(santara) implement multiple rollouts per policy
            if self.config.recovery_mode_training:
                if self.is_safe(obs):
                    if danger_zone_flag:
                        num_recoveries += 1
                        danger_zone_flag = 0
                    obs, r, done, _ = self.normal_step(obs)
                    episode_reward += r[0]
                    episode_step += 1
                    reward_buffer_for_log.append(r[0])
                    if episode_step%self.config.logtostdout_freq == 0:
                        print("\rMain policy | Mean Reward during steps {}-{}: {}, Cumulative Reward: {}".format(last_logged_episode_step+1, episode_step, np.mean(reward_buffer_for_log), episode_reward))
                        reward_buffer_for_log = []
                        last_logged_episode_step = episode_step

                else: # If you are in disaster zone deploy recovery policy
                    if not danger_zone_flag:
                        num_constraint_violations += 1
                        danger_zone_flag = 1
                    obs, r, done, _ = self.recovery_step(obs)
                    episode_reward += r
                    episode_step += 1
                    reward_buffer_for_log.append(r)
                    if episode_step%self.config.logtostdout_freq == 0:
                        print(f"\r{Fore.RED}Recovery policy | Mean Reward during steps {last_logged_episode_step+1}-{episode_step}: {np.mean(reward_buffer_for_log)}, Cum Reward: {episode_reward}, Lane pos: {obs[0, 20]}{Style.RESET_ALL}")
                        reward_buffer_for_log = []
                        last_logged_episode_step = episode_step
            else:
                obs, r, done, _ = self.normal_step(obs)
                episode_reward += r
                episode_step += 1
                reward_buffer_for_log.append(r)
                if episode_step%self.config.logtostdout_freq == 0:
                    print("\rMain policy | Mean Reward during steps {}-{}: {}, Cumulative Reward: {}".format(last_logged_episode_step+1, episode_step, np.mean(reward_buffer_for_log), episode_reward))
                    reward_buffer_for_log = []
                    last_logged_episode_step = episode_step

            if done[0]:
                break

        # Episode done.
        print("\n\nEpisode reward: {}\n\n".format(episode_reward))
        logger.info("Episode reward: {}".format(episode_reward))

        self.reset()
        return episode_reward, episode_step, num_constraint_violations, num_recoveries
