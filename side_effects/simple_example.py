import numpy as np 





# Define environment

T = {}
T['s1'] = {'b1':'s2','b2':'s3','noop':'s1'}
T['s2'] = {'b1':'s2','b2':'s4','noop':'s2'}
T['s3'] = {'b1':'s4','b2':'s3','noop':'s3'}
T['s4'] = {'b1':'s4','b2':'s4','noop':'s4'}

E ={'0':'s1','1':'s2','2':'s3','3':'s4'}
E_a = {'0':'b1','1':'b2','2':'noop'}
E_rev = {'s1':'0','s2':'1','s3':'2','s4':'3'}

def sample_transitions():
    pass




def compute_relative_reachablity(state1,state2):
    global T,E
    gamma = 0.99
    reachability = np.full((4,4),-np.inf)
    for i in range(4):
        reachability[i,i]=1

    for iter_ in range(10):
        for i in range(4):
            for j in range(4):
                temp = 0
                if i!=j:
                    for a in range(3):
                            temp = max(reachability[int(E_rev[T[str(E[str(i)])][E_a[str(a)]]]),j],temp)  
                    
                    reachability[i,j]= gamma * temp


    rel_reach = 0
    count = 0

    #print(reachability)

    for i in range(4):
        rel_reach+= max(reachability[int(E_rev[state1]),i]-reachability[int(E_rev[state2]),i],0)
        # print(rel_reach)
        count+=1

    return (rel_reach/count)




if __name__=="__main__":
    print(compute_relative_reachablity('s3','s2'))



