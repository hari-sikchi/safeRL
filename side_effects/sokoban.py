import sys
sys.path.append('/home/harshit/work/ai-safety-gridworlds')
import ai_safety_gridworlds
import ai_safety_gridworlds.demonstrations.demonstrations as demonstrations
import numpy as np
import ai_safety_gridworlds.helpers.factory as factory
from ai_safety_gridworlds.environments.shared.safety_game import Actions


environment_name = "side_effects_sokoban"
# demo = demonstrations.get_demonstrations(environment_name)[0]
# np.random.seed(demo.seed)
env = factory.get_environment_obj(environment_name)
#print(env.reset())
episode_return = 0
actions = [Actions.LEFT,Actions.RIGHT,Actions.UP,Actions.DOWN]


# def sample_transitions(q_table,horizon = 4,epsilon = 0.8):
    
#     transitions = []
#     obs = 's1'

#     for timestep in range(horizon):
#         if np.random.random()<epsilon:
#             action = np.argmax(q_table[int(E_rev[obs]),:])
#             action_str = E_a[str(action)]
#         else:
#             action = np.random.randint(0,3)
#             action_str = E_a[str(action)]

#         next_obs = T[obs][action_str]
#         reward = - compute_relative_reachablity(next_obs,obs)
#         transitions.append([obs,action,reward,next_obs])
#         obs = next_obs

#     return transitions




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
        reachability_list.append([transition[0],transition[1],1])





def update_reachability(reachability_list):
    reachability_limit = 10
    for iter in range(reachability_limit):
        for i in range(len(reachability_list)):
            for j in range(len(reachability_list)):
                # print(reachability_list[i][1])
                # print(reachability_list[j][0])
                if np.array_equal(reachability_list[i][1],reachability_list[j][0]):
                    insert_rl(reachability_list,[reachability_list[i][0],reachability_list[j][1],reachability_list[i][2]+reachability_list[j][2]])



def sample_episodes(horizon = 10):
    transitions = []
    obs = env.reset().observation['board']
    print("-------")
    for timestep in range(horizon):
        # if np.random.random()<epsilon:
        #     action = np.argmax(q_table[int(E_rev[obs]),:])
        #     action_str = E_a[str(action)]
        # else:

        action = np.random.randint(0,len(actions))
        action_t = actions[action]
        next_timestep = env.step(action_t)

        # next_obs = env._current_observations
        print('action')
        print(action_t)

        print("Observation")
        print(obs)
        print('Next observation')
        print(next_obs)
        if not np.array_equal(obs,next_obs):
            print("hello")
        # print(next_obs)
        reward = next_timestep.reward
        transitions.append([obs,action,reward,next_obs])
        obs = next_obs

    return transitions



if __name__=="__main__":
    reachability_list= []

    for i in range(100):
        transitions = sample_episodes()
        #print(transitions[0])
        insert_transitions(reachability_list,transitions)
        #print(reachability_list)
        update_reachability(reachability_list)
        print("Iter: {}, Length: {}".format(i,len(reachability_list)))


# for action in actions:
#     timestep = env.step(action)
#     print(action)
#     print(timestep.reward)
#     episode_return += timestep.reward if timestep.reward else 0














# print(episode_return)

