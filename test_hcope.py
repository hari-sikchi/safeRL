import numpy as np
import tensorflow as tf
import gym
from policies import *
import math
from scipy.optimize import minimize
from scipy.special import j1
from scipy.optimize import minimize_scalar


class HCOPE(object):

    def __init__(self,env,policy,eval_policy,rollout_length):
        self.env = env
        self.policy= policy
        self.eval_policy=eval_policy
        self.rollout_length = rollout_length
        self.w_policy = self.policy.get_weights()
        self.e_policy = None


    def setup_e_policy(self):
        noise =  np.random.standard_normal(self.w_policy.shape)
        self.e_policy = self.w_policy + noise

    def rollout(self,shift = 0.,policy = None, rollout_length = None,render = False):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        

        total_reward = 0.
        steps = 0

        if(rollout_length==None):
            rollout_length=self.rollout_length

        ob = self.env.reset()
        for i in range(rollout_length):
            action,prob = policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if(render):
                env.render()
            if done:
                break
            
        return total_reward, steps

    def mod_rollout(self,shift = 0., rollout_length = None,render = False):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        

        total_reward = 0.
        steps = 0
        rewards = []
        probs = []
        eval_probs =[]
        if(rollout_length==None):
            rollout_length=self.rollout_length

        ob = self.env.reset()
        for i in range(rollout_length):
            action,prob = self.policy.act(ob)
            eval_action,eval_prob = self.eval_policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            probs.append(prob)
            eval_probs.append(eval_prob)
            steps += 1
            total_reward += (reward - shift)
            if(render):
                env.render()
            if done:
                break
            
        return total_reward, steps,rewards,probs,eval_probs


    def evaluate(self,n_rollouts=100,render = False):
        self.policy.update_weights(self.w_policy)
        self.policy.update_filter = False
        rewards = []
        for i in  range(n_rollouts):
            total_reward,steps = self.rollout(render=render,policy = self.policy)
            rewards.append(total_reward)            

        rewards = np.asarray(rewards)
        print("Mean Reward: {}",np.mean(rewards))


    def generate_dataset(self,dataset_size = 100,render=False):
        self.policy.update_weights(self.w_policy)
        self.policy.update_filter = False
        self.eval_policy.update_weights(self.e_policy)
        self.eval_policy.update_filter = False
        rewards = []
        probs = []
        eval_probs = []
        for i in  range(dataset_size):
            total_reward,steps,rewards_list,probs_list,eval_probs_list = self.mod_rollout(render=render)
            rewards.append(rewards_list)
            probs.append(probs_list)
            eval_probs.append(eval_probs_list)            



        

        rewards = np.asarray(rewards)
        probs = np.asarray(probs)
        eval_probs = np.asarray(eval_probs)


        permutation = np.random.permutation(probs.shape[0])
        
        rewards = rewards[permutation,:]
        probs = probs[permutation,:]
        eval_probs =eval_probs[permutation,:]

        d_pre = rewards[:int(0.05*dataset_size),:]
        d_post = rewards[int(0.05*dataset_size):,:]
        
        pi_b_pre = probs[:int(0.05*dataset_size),:]
        pi_b_post = probs[int(0.05*dataset_size):,:]

        pi_e_pre = eval_probs[:int(0.05*dataset_size),:]
        pi_e_post = eval_probs[int(0.05*dataset_size):,:]

        eval_estimate = self.hcope_estimator(d_pre, d_post, pi_b_pre,pi_b_post,pi_e_pre,pi_e_post,0.1)
        #print(probs.shape)
        print("Estimate of evaluation policy: {}".format(eval_estimate))
        print("Mean Reward: {}",np.mean(rewards))


        
    def true_eval_estimate(self,n_rollouts=100,render = False):
        self.eval_policy.update_weights(self.e_policy)
        self.eval_policy.update_filter = False
        rewards = []
        for i in  range(n_rollouts):
            total_reward,steps = self.rollout(render=render,policy = self.eval_policy)
            rewards.append(total_reward)            

        rewards = np.asarray(rewards)
        print("True Mean Reward of evaluation policy: {}",np.mean(rewards))



    def estimate_behavior_policy(self):


        pass

    
    def hcope_estimator(self,d_pre, d_post, pi_b_pre,pi_b_post,pi_e_pre,pi_e_post,delta):
        """
        d_pre : float, size = (dataset_split,)
            Trajectory rewards from the behavior policy 

        d_post : float, size = (dataset_size - dataset_split, )
            Trajectory rewards from the behavior policy 

        delta : float, size = scalar
            1-delta is the confidence of the estimator
        
        pi_b : Probabilities for respective trajectories in behaviour policy

        pi_e : Probabilities for respective trajectories in evaluation policy

        RETURNS: lower bound for the mean, mu as per Theorem 1 of Thomas et al. High Confidence Off-Policy Evaluation
        """
        d_pre = np.asarray(d_pre)
        d_post = np.asarray(d_post)
        n_post = len(d_post)
        n_pre = len(d_pre)

        # Estimate c which maximizes the lower bound using estimates from d_pre

        c_estimate = 4.0

        def f(x):
            n_pre = len(d_pre)
            Y = np.asarray([min(np.sum((d_pre[i] * pi_e_pre[i])/pi_b_pre[i].astype(np.float64)), x) for i in range(n_pre)], dtype=float)

            # Empirical mean
            EM = np.sum(Y)/n_pre

            # Second term
            term2 = (7.*x*np.log(2/delta)) / (3*(len(d_post)-1))

            # Third term
            term3 = np.sqrt( ((2*np.log(2/delta))/(n_post*n_pre*(n_pre-1)) * (n_pre*np.sum(np.square(Y)) - np.square(np.sum(Y))) ))
            
            return (-EM+term2+term3) 

        c_estimate = minimize(f,np.array([c_estimate]),method='BFGS').x

        # Use the estimated c for computing the maximum lower bound
        c = c_estimate

        if ~isinstance(c, list):
            c = np.full((n_post,), c, dtype=float)

        
        if n_post<=1:
            raise(ValueError("The value of 'n' must be greater than 1"))


        Y = np.asarray([min(np.sum((d_post[i] * pi_e_post[i])/pi_b_post[i].astype(np.float64)), c[i]) for i in range(len(d_post))], dtype=float)

        # Empirical mean
        EM = np.sum(Y/c)/np.sum(1/c)

        # Second term
        term2 = (7.*n_post*np.log(2/delta)) / (3*(n_post-1)*np.sum(1/c))

        # Third term
        term3 = np.sqrt( ((2*np.log(2/delta))/(n_post-1)) * (n_post*np.sum(np.square(Y/c)) - np.square(np.sum(Y/c))) ) / np.sum(1/c)


        # Final estimate
        return EM - term2 - term3











if __name__=="__main__":
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    action_size = env.action_space.n
    ob_size = env.observation_space.shape[0]

    policy_params={'type':'bilayer',
                   'ob_filter':'MeanStdFilter',
                   'ob_dim':ob_size,
                   'ac_dim':action_size} 
    policy = BilayerPolicy_softmax(policy_params)

    eval_policy = BilayerPolicy_softmax(policy_params)
    my_hcope = HCOPE(env,policy,eval_policy,rollout_length = 1000)
    my_hcope.setup_e_policy()
    #my_hcope.evaluate(n_rollouts=100,render =True)

    my_hcope.generate_dataset()
    my_hcope.true_eval_estimate()

