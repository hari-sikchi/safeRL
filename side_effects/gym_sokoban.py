import sys
import numpy as np

sys.path.append('/home/harshit/work/ai-safety-gridworlds')
sys.path.append('/home/harshit/work/safe-grid-gym')

import safe_grid_gym
import gym



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



def sample_episodes(horizon = 10):
    transitions = []
    obs = env.reset()
    print("-------")
    for timestep in range(horizon):
        # if np.random.random()<epsilon:
        #     action = np.argmax(q_table[int(E_rev[obs]),:])
        #     action_str = E_a[str(action)]
        # else:
        action = action_space.sample()
        action_t=action
        # action = np.random.randint(0,len(actions))
        # action_t = actions[action]
        next_state,reward,done,info = env.step(action_t)

        transitions.append([obs,action,reward,next_state])
        obs = next_state

    return transitions



if __name__=="__main__":
    reachability_list= []

    for i in range(100):
        transitions = sample_episodes()
        insert_transitions(reachability_list,transitions)
        update_reachability(reachability_list)
        print("Iter: {}, Length: {}".format(i,len(reachability_list)))
















# print(episode_return)

