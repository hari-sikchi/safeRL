import sys
import numpy as np

sys.path.append('/home/harshit/work/ai-safety-gridworlds')
sys.path.append('/home/harshit/work/safe-grid-gym')

import safe_grid_gym
import gym

# Beta is the tradeoff between real reward and the reachability reward
beta = 3 
env = gym.make("SideEffectsSokoban-v0")

action_space = env.action_space

print(action_space)



def insert_transitions(reachability_list,transitions):
    
    for transition in transitions:
        already_present= False
        for i in reachability_list:
            if(np.array_equal(transition[0],np.array(i[0])) and np.array_equal(transition[3],(i[1]))):
                already_present= True
        if not already_present:
            reachability_list.append([transition[0],transition[3],1])


def insert_rl(reachability_list,reachable_states):
    already_present = False
    for i in reachability_list:
        # print(reachable_states[0])

        if(np.array_equal(reachable_states[0],i[0]) and np.array_equal(reachable_states[1],i[1])):
            already_present= True
            i[2] = min(i[2],reachable_states[2])

    if not already_present:
        reachability_list.append([reachable_states[0],reachable_states[1],1])





def update_reachability(reachability_list):
    reachability_limit = 10
    for iter in range(reachability_limit):
        for i in range(len(reachability_list)):
            for j in range(len(reachability_list)):
                if np.array_equal(reachability_list[i][1],reachability_list[j][0]):
                    insert_rl(reachability_list,[reachability_list[i][0],reachability_list[j][1],reachability_list[i][2]+reachability_list[j][2]])


def compute_relative_reachability(state1,state2,reachability_list):

    sum1 = 0
    sum2 = 0
    gamma = 0.9
    count1=0
    count2=0
    for i in reachability_list:
        if(np.array_equal(i[0],state1)):
            sum1+=gamma**i[2]
            count1+=1
        if(np.array_equal(i[0],state2)):
            sum2+=gamma**i[2]
            count2+=1
        
    if(count1!=0):
        reach1 = sum1/count1
    else:
        reach1=0
    if(count2!=0):
        reach2 = sum2/count2
    else:
        reach2=0

    rel_reach = max(reach1-reach2,0)


    return (rel_reach)



def sample_episodes(q_table,reachability_list,horizon = 10,epsilon=0.8):
    transitions = []
    obs = env.reset()
    for timestep in range(horizon):
        if np.random.random()<epsilon:
            action = np.argmax(q_table[int(enumerate_state(obs)),:])
        else:
            action = action_space.sample()

        next_state,reward,done,info = env.step(action)


        surrogate_reward = reward + beta * compute_relative_reachability(obs,next_state,reachability_list)

        transitions.append([obs,action,reward,next_state])
        obs = next_state

    return transitions


def sample_episodes_evaluate(q_table,reachability_list,horizon = 10,epsilon=0.8):
    transitions = []
    obs = env.reset()
    for timestep in range(horizon):
        action = np.argmax(q_table[int(enumerate_state(obs)),:])
        print(action)
        next_state,reward,done,info = env.step(action)


        surrogate_reward = reward + beta * compute_relative_reachability(obs,next_state,reachability_list)

        transitions.append([obs,action,reward,next_state])
        obs = next_state

    return transitions




def enumerate_state(state):
    count = 0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            count+=1
            if (state[0,i,j]==2):
                break

    count*=10
    count2=0
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            count2+=1
            if (state[0,i,j]==4):
                break

    return count2+count



def q_learning():
    n_states = 200
    n_actions = 4
    q_table = np.zeros((n_states,n_actions))
    alpha = 0.8
    n_episodes = 200
    gamma = 0.9
    reachability_list= []

    for i in range(n_episodes):
        transitions = sample_episodes(q_table,reachability_list)
        insert_transitions(reachability_list,transitions)
        update_reachability(reachability_list)

        for transition in transitions:
            q_table[enumerate_state(transition[0])][transition[1]] = alpha*q_table[enumerate_state(transition[0]),transition[1]] + (1-alpha)*(transition[2]+gamma*np.max(q_table[int(enumerate_state(transition[3])),:]))

        print('Episodes: {}, Reachability Length: {}'.format(i,len(reachability_list)))

    print("Learnt Q table")

    print(q_table)

    # Evaluate
    sample_episodes_evaluate(q_table,reachability_list)





if __name__=="__main__":
    q_learning()

    # for i in range(100):
    #     transitions = sample_episodes(reachability_list)
    #     insert_transitions(reachability_list,transitions)
    #     update_reachability(reachability_list)
    #     print("Iter: {}, Length: {}".format(i,len(reachability_list)))
















# print(episode_return)

