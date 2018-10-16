'''
Importance sampling estimators

Harshit Sikchi
'''
import numpy as np
import math

'''
Simple importance sampling

'''


def simple_is(pi_b,pi_e,reward):
    estimated_reward = reward
    for i,action_hist_prob in enumerate(pi_b):
        estimated_reward*= pi_e[i]/pi_b[i]
    return estimated_reward



'''
Per Decision Importance Sampling
reward: list of reward obtained per time step

gamma: discount factor
returns normalized estimate of reward under evaluation policy
'''

def per_decision_is(pi_b,pi_e,gamma,reward,reward_high,reward_low):
    horizon = len(reward)
    expected_reward = 0
    gamma_t = 1
    importance_weight = 1
    for t in range(1,horizon+1):
        importance_weight *= pi_e[t-1]/pi_b[t-1] 
        expected_reward+= gamma_t * reward[t-1] *importance_weight  
        gamma_t *= gamma

    return (expected_reward - reward_low)/(reward_high-reward_low)



