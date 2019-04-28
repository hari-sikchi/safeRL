import numpy as np
import tensorflow as tf
from copy import deepcopy

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory

from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines import logger


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
    def __init__(self, env, config, name='', training=True, **kwargs):
        self.name = name
        self.env = env
        self.config = config
        self.nb_actions = self.env.action_space.shape[-1]
        self._training = training

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
                        gamma=self.config.gamma, 
                        tau=self.config.tau,
                        normalize_returns=self.config.normalize_returns,
                        normalize_observations=self.config.normalize_observations,
                        batch_size=self.config.batch_size,
                        action_noise=self.noise.action_noise,
                        param_noise=self.noise.param_noise,
                        critic_l2_reg=self.config.critic_l2_reg,
                        actor_lr=self.config.actor_lr,
                        critic_lr=self.config.critic_lr,
                        enable_popart=self.config.popart,
                        clip_norm=self.config.clip_norm,
                        reward_scale=self.config.reward_scale
                        )

    def initialize(self, sess):
        self.agent.initialize(sess)

    def reset(self):
        self.agent.reset()


class Agent:
    def create_agent(self, env, config, training=True):
        self.env = env
        self.config = config
        self._training = training
        self.main_agent = ActorCriticAgent(self.env, self.config, name='main', training=self._training)
        self.recovery_agent = ActorCriticAgent(self.env, self.config, name='recovery', training=self._training)

    def initialize(self, sess):
        self.main_agent.initialize(sess)
        self.recovery_agent.initialize(sess)

    def reset(self):
        self.main_agent.reset()
        self.recovery_agent.reset()
