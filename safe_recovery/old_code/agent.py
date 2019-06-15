#general libraries
import argparse
import time
import sys
import os
import queue
import yaml
import random
from copy import deepcopy
sys.path.append('/home/harshit/work/baselines/')
#baseline_common
from baselines import logger
#baseline_ddpg
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
#madras_libraries
import MADRaS
from utils import *
#python libraries
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

figure = plt.figure()
tf.reset_default_graph()
sess = tf.InteractiveSession()
stddev = 0.2

with open("./configurations.yml","r") as ymlfile:
    cfg = yaml.load(ymlfile)

logger.set_level(logger.DEBUG)
dir = cfg['configs']['f_diagnostics']
logger.configure(dir=dir)

def visualize_action_value(list_Q,fig,ep_no):
    actions_lane = []
    actions_vel = []
    Q_values = []
    for act,q in list_Q:
        actions_lane.append(act[0])
        actions_vel.append(act[1])
        Q_values.append(q[0])
    actions_vel = np.asarray(actions_vel)
    actions_lane = np.asarray(actions_lane)
    Q_values = np.asarray(Q_values)
    plot_heatmap(actions_lane, actions_vel, Q_values)
    plt.savefig(os.path.join(cfg['configs']['fig_save_dir'],"plot_"+str(ep_no)+".png"))
    plt.clf()


def playGame(train_indicator):
    #setting up
    env = gym.make("Madras-v0")
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    actor = Actor(env.action_space.shape,layer_norm=True)
    critic = Critic(layer_norm=False)
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]), sigma=float(stddev) * np.ones(env.action_space.shape))
    agent = DDPG(actor, critic, memory, normalize_returns=False, normalize_observations=True, batch_size=256,
            action_noise=action_noise, param_noise=None, critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3,
            enable_popart=False, clip_norm=None, reward_scale=1., action_shape= env.action_space.shape, observation_shape=env.observation_space.shape)
    ###########SAVER
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(cfg['configs']['save_location'])
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    ####################
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))
    ########EPISODE_PARAMS
    nb_epochs = cfg['agent']['max_eps']
    nb_rollout_steps = cfg['agent']['max_steps_eps']
    nb_train_steps = cfg['agent']['max_steps_train']
    batch_size = cfg['agent']['batch_size']
    ##########PRINTER_CLASS
    episode_printer = StructuredPrinter(mode="episode")
    step_printer = StructuredPrinter(mode="step")
    ##########INITIALIZATION
    max_distance = 0.
    episode = 0
    episodes = 0
    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    state_action_pair = queue.Queue(maxsize=1e4)
    epoch_episodes = 0
    episode_running_avg_reward = 0.
    episode_running_avg_distance = 0.
    #########BEGIN
    done = False
    agent.initialize(sess)
    sess.graph.finalize()
    agent.reset()
    s_t = env.reset()

    for i in range(nb_epochs):######NUM_EPISODES

        info = {'termination_cause':0}
        episode_total_reward = 0.
        #episode_distance = 0.
        episode_step = 0.

        for step in range(nb_rollout_steps):

            if train_indicator:
                desire, q,_, _ = agent.step(s_t, apply_noise=True, compute_Q=True)
            else:
                desire,_, _, _ = agent.step(s_t, apply_noise=False, compute_Q=False)

            print("DESIRE                             ",desire.shape)
            print("ENV SHAPE ACT", env.action_space.shape[0])
            desire= desire.reshape(env.action_space.shape[0])
            assert desire.shape == (env.action_space.shape[0],)

            s_t1, r_t, done, info = env.step(desire)
            episode_total_reward += r_t
            episode_step += 1
            #print("STEP:%s, DESIRE:%s" %(step,desire))
            step_printer.data["Dist_Raced"] = 0.0
            step_printer.data["Desired_Trackpos"] = desire[0]
            step_printer.data["Desired_Velocity"] = desire[1]
            step_printer.data["Reward"] = r_t
            step_printer.print_step(step)
            #logging
            epoch_actions.append(desire)
            if train_indicator:
                epoch_qs.append(q)
                epoch_actions.append(desire)
                if state_action_pair.full():
                    state_action_pair.get()
                state_action_pair.put((desire,q))

                agent.store_transition(s_t, desire, r_t, s_t1, done)
            s_t = s_t1
            if done:
                break
        episode_distance = env.distance_traversed
        max_distance = deepcopy(env.distance_traversed) if episode_distance > max_distance else deepcopy(max_distance)
        epoch_episode_rewards.append(episode_total_reward)
        episode_running_avg_reward = running_average(episode_running_avg_reward,i+1,episode_total_reward)
        episode_running_avg_distance = running_average(episode_running_avg_distance,i+1,episode_distance)
        epoch_episode_steps.append(episode_step)
        total_reward = deepcopy(episode_total_reward)
        episode_total_reward = 0
        episode_step = 0
        epoch_episodes += 1
        ###########################TRAINING
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_adaptive_distances = []
        if train_indicator:
            for train in range(nb_train_steps):
                if memory.nb_entries > batch_size:
                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()
        ############EPISODE_PRINTER
        episode_printer.data["Total_Steps"] = step
        episode_printer.data["Dist_Traversed"] = episode_distance
        episode_printer.data["Traj_Reward"] = total_reward
        episode_printer.data["Run_Avg_Traj_Reward"] = episode_running_avg_reward
        episode_printer.data["Run_Avg_Dist_Trav"] = episode_running_avg_distance
        episode_printer.data["Max_Dist_Trav"] = max_distance
        episode_printer.data["Replay_Buffer_Size"] = len(agent.memory.actions)
        episode_printer.print_episode(i)
        ##################LOGGING
        if train_indicator:
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['episode/return'] = np.mean(epoch_episode_rewards)
            combined_stats['episode/running_average_reward'] = episode_running_avg_reward
            combined_stats['episode/running_average_dist'] = episode_running_avg_distance
            combined_stats['episode/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['episode/desire_mean'] = np.mean(epoch_actions)
            combined_stats['episode/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')

            if train_indicator and len(agent.memory.actions) >= cfg['configs']['batch_size']:
                visualize_action_value(random.sample(list(state_action_pair.queue),cfg['configs']['batch_size']),figure,i)



        if i%20==0 and train_indicator:
            saver.save(sess, cfg['configs']['save_location'] + env.env_name + 'network' + '-ddpg_baseline', global_step = i)

        ##############RESETING_THE ENV
        agent.reset()
        s_t = env.reset()

    env.end()
    print("Finished.")

def running_average(prev_avg, num_episodes, new_val):
    total = prev_avg*(num_episodes-1)
    total += new_val
    return float(total/num_episodes)

if __name__ == "__main__":

    print('config_file : ' + cfg['configs']['configFile'])
    playGame(train_indicator=cfg['configs']['is_training'])





