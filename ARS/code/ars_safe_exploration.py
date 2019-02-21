'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import parser
import time
import os
import numpy as np
import gym
import logz
import ray
import utils
import optimizers
from policies_safe import *
import socket
from shared_noise import *
import MADRaS
import sys
import copy
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
from torch.nn import Linear, Module, MSELoss

import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import random

#f = open('logs_1.txt', 'w')
#sys.stdout = f
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

 
        #sys.path.append('/home/harshit/work/')
        #import MADRaS
 
        # initialize OpenAI environment for each worker
        
        # logging.warning('Env start')
        sys.path.append('/home/harshit/work')
        import MADRaS
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(env_seed)
        self.threshold=-200
        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)

        elif policy_params['type'] == 'bilayer':
            self.policy = BilayerPolicy(policy_params)
        elif policy_params['type'] == 'bilayer_safe':
            self.policy = SafeBilayerPolicy(policy_params)
        elif policy_params['type'] == 'bilayer_safe_discrete':
            self.policy = SafeBilayerDiscretePolicy(policy_params)
        
        elif policy_params['type'] == 'bilayer_safe_explorer':
            self.policy = SafeBilayerExplorerPolicy(policy_params)

        
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length

    def __str__(self):
        return "Env_NAME:{} policy_params:{}".format(self.env_name,self.policy_params)
    def __repr__(self):
        return "Env_NAME:{} policy_params:{}".format(self.env_name,self.policy_params)
       

    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        assert (self.policy_params['type'] == 'bilayer' or self.policy_params['type'] == 'bilayer_safe' or self.policy_params['type'] == 'bilayer_safe_discrete' or self.policy_params['type'] == 'bilayer_safe_explorer' or self.policy_params['type'] == 'linear')
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0
        true_violations = 0
        q_violations = 0

        ob = self.env.reset()

        transitions = []
        record_transitions = True
        my_f= open('hello.txt','a')
        for i in range(rollout_length):
            action = self.policy.act(ob)
            

            #Add the constraint for Madras-v0 for now
            my_f.write("{} \n".format(ob[2]))
            next_ob, reward, done, _ = self.env.step(action)
            if record_transitions==True:
                transitions.append([ob,action,reward,next_ob])

            # Constraints for linear safety layer
            # if(next_ob[0]<-0.7 or next_ob[0]>0.7 ):
            #     record_transitions=False                
            steps += 1
            total_reward += (reward - shift)
            ob = next_ob
            if done:
                break

            
        return total_reward, steps,transitions,true_violations,q_violations

    def linesearch(self, delta, backtrack_ratio=0.2, num_backtracks=5):
        deltas = [delta]
        # for i in range(int(num_backtracks)):
        #     deltas.append(delta/((backtrack_ratio)**i))
        #     deltas.insert(0,delta*((backtrack_ratio)**i))
        return deltas

    def print_threshold(self):
        my_f = open("Thresholds.txt",'a')
        my_f.write("{}".format(self.threshold))

    def set_threshold(self, threshold):
        self.threshold=threshold
    
    def get_threshold(self):
        return self.threshold

    def increase_threshold(self):
        self.threshold+= 0.2*abs(self.threshold)

    def decrease_threshold(self):
        self.threshold-= 0.2*abs(self.threshold)


    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx,deltas_ratio,num_episodes = [], [], [],[]
        steps = 0
        test_rewards = []
        all_transitions = []
        true_violations = 0
        q_violations = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, transitions,true_violations,q_violations = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # # set to true so that state statistics are updated 
                # self.policy.update_filter = True

                # # compute reward and number of timesteps used for positive perturbation rollout
                # self.policy.update_weights(w_policy + delta)
                # pos_reward, pos_steps  = self.rollout(shift = shift)

                # # compute reward and number of timesteps used for negative pertubation rollout
                # self.policy.update_weights(w_policy - delta)
                # neg_reward, neg_steps = self.rollout(shift = shift) 
                # steps += pos_steps + neg_steps

                # rollout_rewards.append([pos_reward, neg_reward])
                deltas = self.linesearch(delta)
                max_rollout_reward_diff=-np.inf
                best_delta=delta
                best_pos=0
                best_neg=0
                delta_ratio=1
                
                for i, delta_ in enumerate(deltas):
                    self.policy.update_filter = True

                    # compute reward and number of timesteps used for positive perturbation rollout
                    self.policy.update_weights(w_policy + delta_)
                    pos_reward, pos_steps,transitions,true_violations_,q_violations_  = self.rollout(shift = shift)
                    all_transitions = all_transitions+transitions
                    true_violations+=true_violations_
                    q_violations+=q_violations_
                    # compute reward and number of timesteps used for negative pertubation rollout
                    self.policy.update_weights(w_policy - delta_)

                    neg_reward, neg_steps,transitions,true_violations_,q_violations_ = self.rollout(shift = shift) 
                    all_transitions=all_transitions+transitions
                    true_violations+=true_violations_
                    q_violations+=q_violations_

                    steps += pos_steps + neg_steps
                    test_rewards.append(max(pos_reward,neg_reward))
                    if max_rollout_reward_diff<max(pos_reward,neg_reward):
                        max_rollout_reward_diff=max(pos_reward,neg_reward)
                        best_delta = delta_
                        best_pos= pos_reward
                        best_neg=neg_reward
                        delta_ratio = best_delta/delta 

                
                # try weighted line search--------------------- works very bad
                # test_rewards=np.asarray(test_rewards)
                # best_delta = np.sum(np.multiply(np.asarray(deltas),test_rewards.reshape(-1,1)),axis=0)/np.sum(test_rewards)
                # delta_ratio = best_delta/delta
                # self.policy.update_filter = True

                # # compute reward and number of timesteps used for positive perturbation rollout
                # self.policy.update_weights(w_policy + best_delta)
                # best_pos, pos_steps  = self.rollout(shift = shift)

                # # compute reward and number of timesteps used for negative pertubation rollout
                # self.policy.update_weights(w_policy - best_delta)
                # best_neg, neg_steps = self.rollout(shift = shift) 
                # steps += pos_steps + neg_steps 
                #----------------------------------------------------

                num_episodes.append([2*len(deltas)])
                rollout_rewards.append([best_pos, best_neg])
                deltas_ratio.append(delta_ratio)


        return {'deltas_idx': deltas_idx,'deltas_ratio':deltas_ratio, 'rollout_rewards': rollout_rewards, "steps" : steps,"num_episodes":num_episodes,"test_rewards":test_rewards, "transitions":all_transitions,"true_violations":true_violations,"q_violations":q_violations}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=1000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):
        
        logz.configure_output_dir(logdir)
        logz.save_params(params)
 
        env = gym.make(env_name)
       
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = 0
        self.replay_buffer=np.asarray([])
        self.MEMORY = 1000000

        # Parameters for Q Learner
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.TARGET_UPDATE = 5

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]

        # initialize policy 
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'bilayer':
            self.policy = BilayerPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'bilayer_safe':
            self.policy = SafeBilayerPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'bilayer_safe_discrete':
            self.policy = SafeBilayerDiscretePolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'bilayer_safe_explorer':
            self.policy = SafeBilayerExplorerPolicy(policy_params)
            self.w_policy = self.policy.get_weights()

        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    
    def update_replay_buffer(self,transitions):
        if(self.MEMORY-self.replay_buffer.shape[0]>transitions.shape[0]):
            np.append(self.replay_buffer,transitions)
        elif(transitions.shape[0] - (self.MEMORY-self.replay_buffer.shape[0])):
            self.replay_buffer = self.replay_buffer[transitions.shape[0] - (self.MEMORY-self.replay_buffer.shape[0]):,:]
            np.append(self.replay_buffer,transitions)

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
            #print("TRAIN")
        else:
            num_deltas = num_rollouts
            #print("TEST")
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
        #print("NUM_ROLLOUTS {}".format(num_rollouts))
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx, deltas_ratio,num_episodes = [], [], [],[] 

        test_rewards=[]
        all_transitions = []
        true_violations=0
        q_violations=0
        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            deltas_ratio += result['deltas_ratio']
            rollout_rewards += result['rollout_rewards']
            num_episodes+=result['num_episodes']
            test_rewards.append([result['test_rewards']])
            all_transitions+=result['transitions']
            q_violations+=result['q_violations']
            true_violations+=result['true_violations']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            deltas_ratio += result['deltas_ratio']
            rollout_rewards += result['rollout_rewards']
            num_episodes+=result['num_episodes']
            test_rewards.append([result['test_rewards']])
            all_transitions+=result['transitions']
            q_violations+=result['q_violations']
            true_violations+=result['true_violations']

        # self.update_replay_buffer(all_transitions)
        #print(test_rewards)
        for tran in all_transitions:
            self.memory.push(torch.from_numpy(tran[0]).unsqueeze(0).to(device).float(),torch.tensor([[tran[1]]],device=device, dtype=torch.long),torch.from_numpy(tran[3]).unsqueeze(0).float().to(device),torch.tensor([tran[2]],device=device))

        deltas_idx = np.array(deltas_idx)
        deltas_ratio = np.array(deltas_ratio)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards
        #print(deltas_ratio)
        print('-------------')
        #print(rollout_rewards)
        self.num_episodes_used+=np.sum(np.asarray(num_episodes))



        print(rollout_rewards)

        # Curriculum for threshold so that initialization bias in removed from the Q function
        # if(true_violations<q_violations):
        #     print("q_violations exceed true_violations")
        #     rollout_ids_one_ = [worker.decrease_threshold.remote() for worker in self.workers]
        #     rollout_ids_two_ = [worker.decrease_threshold.remote() for worker in self.workers[:(num_deltas % self.num_workers)]]

        # elif(q_violations<true_violations):
        #     print("True violations exceed q_violations")

        #     rollout_ids_one_ = [worker.increase_threshold.remote() for worker in self.workers]

        #     rollout_ids_two_ = [worker.increase_threshold.remote()  for worker in self.workers[:(num_deltas % self.num_workers)]]




        # self.workers[0].print_threshold.remote() 




        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        deltas_ratio=deltas_ratio[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        if np.std(rollout_rewards)!=0:
            rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get_mod(idx, self.w_policy.size, ratio)
                                                   for idx, ratio in zip(deltas_idx, deltas_ratio)),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat = self.aggregate_rollouts()                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        self.policy.update_weights(self.w_policy)
        return

    def update_safety_net(self):

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)


        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print(state_batch.size())
        # print(action_batch.size())
        #print(self.policy.safeQ(state_batch))
        state_action_values = self.policy.safeQ(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.policy.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.policy.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.safeQ.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()


    def update_explorer_net(self):
        if len(self.memory) < self.BATCH_SIZE:
            return


        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)


        # Convert to numpy arrays
        state_np = np.asarray([i.cpu().numpy() for i in batch.state])
        action_np = np.asarray([i.cpu().numpy().astype(np.float64) for i in batch.action])
        next_state_np = np.asarray([i.cpu().numpy() for i in batch.next_state])


   
        next_state_np = next_state_np.reshape(next_state_np.shape[0],-1)
        cost_next_state = np.asarray([100 if i[0]<=-0.7 or i[0]>=0.7  else 0 for i in next_state_np])
        state_np = state_np.reshape(state_np.shape[0],-1)
        action_np = action_np.reshape(action_np.shape[0],-1)
        cost_state = np.asarray([100 if i[0]<=-0.7 else 0 for i in state_np])

        weights = self.policy.safeQ(state_batch)

        mul = torch.mul(weights,torch.from_numpy(action_np).to(device).float())
        mul = torch.sum(mul,dim=1)
        # print(mul.size())
        # print(torch.from_numpy(cost_state).to(device).float().size())
        target = torch.from_numpy(cost_state).to(device).float()+mul
        # print(target.size())
        # print(target.view(1,-1).size())
        # print(torch.from_numpy(cost_next_state).to(device).float().size())
        loss = F.mse_loss(torch.from_numpy(cost_next_state).to(device).float(), target.view(1,-1))
        # Optimize the model
        self.policy.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy.safeQ.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()





    def train(self, num_iter):
        max_reward_ever=-1
        start = time.time()
        actual_reward_list = []
        num_episodes_till_now =[]
        for i in range(num_iter):
            
            t1 = time.time()
            self.train_step()
            for iter_ in range(10):
                self.update_explorer_net()
            # if i % self.TARGET_UPDATE == 0:
            #     self.policy.target_net.load_state_dict(self.policy.safeQ.state_dict())
           
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')
            # if i == num_iter-1:
            #     np.savez(self.logdir + "/lin_policy_plus" + str(i), w) 
            # record statistics every 10 iterations
            if ((i + 1) % 20 == 0):
                
                rewards = self.aggregate_rollouts(num_rollouts = 30, evaluate = True)
                actual_reward_list.append(np.mean(rewards))
                num_episodes_till_now.append(copy.copy(self.num_episodes_used))
                print("SHAPE",rewards.shape)
                if(np.mean(rewards)>max_reward_ever):
                    max_reward_ever=np.mean(rewards)
                #     np.savez(self.logdir + "/lin_policy_plus", w)

                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/bi_policy_num_plus" + str(i), w)
                torch.save(self.policy.net.state_dict(),self.logdir + "/bi_policy_num_plus_torch" + str(i)+ ".pt")
                torch.save(self.policy.safeQ.state_dict(),self.logdir + "/safeQ_torch" + str(i)+ ".pt")

                print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("BestRewardEver", max_reward_ever)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()
                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)

        np.savez(self.logdir + "/final_reward_array",np.asarray(actual_reward_list))
        np.savez(self.logdir + "/final_episode_array",np.asarray(num_episodes_till_now))
        # utils.plot_info({'rewards':[num_episodes_till_now,actual_reward_list,'Episodes','Average Rewards']},self.logdir)
                        
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make(params['env_name'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'bilayer_safe_explorer',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    

    ARS = ARSLearner(env_name=params['env_name'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    ARS.train(params['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Madras-v0')
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=12)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=1)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='bilayer_safe_explorer')
    parser.add_argument('--dir_path', type=str, default='trained_policies/Madras-explore')
    parser.add_argument('--logdir', type=str, default='trained_policies/Madras-explore')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')



    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(redis_address="10.32.6.37:6382")
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)

