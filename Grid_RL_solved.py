import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib as mpl
import random


VISUALISE = True
DISCOUNT_FACTOR=0.8
CRITERION = 1e-6
N_ROWS = 4
N_COLS = 4
REWARD_STATE=(0,0)
NEGATIVE_STATES = [(1,0),(1,1)]
TERMINAL_STATES=NEGATIVE_STATES+[REWARD_STATE]

STATES = [(i,j) for i in range(N_ROWS) for j in range(N_COLS)] 
NON_TERMINAL_STATES=list(set(STATES).difference(set(TERMINAL_STATES)))

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]

def functional_reward(s,a):
    if tuple(np.add(s,a))==REWARD_STATE:
        return 5
    if tuple(np.add(s,a))==NEGATIVE_STATES:
        return -5
    else:
        return 0

def get_valid_actions(s):
    if s not in TERMINAL_STATES:
        return list(filter(lambda a: np.add(s,a)[0]>=0 and np.add(s,a)[1]>=0 and np.add(s,a)[0]<N_ROWS and np.add(s,a)[1]<N_COLS, ACTIONS))
    else:
        return None

V = {s:0 for s in STATES}

k=0
while True:
    k+=1
    V_new = V.copy()
    for state in STATES:
        valid_actions = get_valid_actions(state)
        if valid_actions == None:
            max_value = 0
        else:
            max_value=max([functional_reward(state,a) + DISCOUNT_FACTOR*V[tuple(np.add(state,a))] for a in valid_actions])
        V_new[state]=max_value
    
    diff = sum([abs(V_new[state]-V[state]) for state in STATES])
    V=V_new

    if diff<CRITERION:
        print(f"Converged to within {CRITERION}")
        break

# for s in NON_TERMINAL_STATES:
#     print(s)
#     print(V[s])


#Extract Policy
policy={}
for state in STATES:
    valid_actions=get_valid_actions(state)
    if valid_actions==None:
        best_action=(0,0)
    else:
        best_action = max(valid_actions,key=lambda a: functional_reward(state,a) 
                    + DISCOUNT_FACTOR*V[tuple(np.add(state,a))])
    policy[state]=best_action
    # print("State is: ")
    # print(state)
    # print(policy[state])


if VISUALISE:


    cmap = plt.get_cmap("jet")
    #print(cmap)
    # s=(1,0)


    for s in random.sample(NON_TERMINAL_STATES, 5):
        count=0
        fig, ax = plt.subplots()
        ims=[]
        starting_s=s
        print(f"Sample of agent behaviour starting at: {starting_s}")
        while True:
            
            frame = np.zeros((N_COLS,N_ROWS))
            frame[REWARD_STATE[0]][REWARD_STATE[1]]=0.5
            for negative_state in NEGATIVE_STATES:
                frame[negative_state[0]][negative_state[1]]=1.0
            frame[s[0]][s[1]]=0.4
            frame=cmap(frame)
            #print(frame)
            im = ax.imshow(frame,animated=True)
            ims.append([im])
            if s==REWARD_STATE or s in NEGATIVE_STATES:
                # ims.append([im])
                break
            s=tuple(np.add(s,policy[s]))
            if count==0:
                ax.imshow(frame)#Show initial frame
            count+=1
            if count>N_COLS*N_COLS:
                print("Agent repeating states")
                break

        ani = animation.ArtistAnimation(fig, ims, interval=800, blit=True, repeat_delay=1000)
        ani.save(f"{starting_s}.gif",dpi=300, writer=PillowWriter(fps=1))
        plt.show()