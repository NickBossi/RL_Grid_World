import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib as mpl
import random 
import itertools

VISUALISE = True

NROWS = 4
NCOLS = 4
NUM_EPISODES = 100000
EPISODE_NUMBER = 1
EPISODES = []


DISCOUNT_FACTOR = 0.8


REWARD_STATE = (0,0)
STATES = [(i,j) for i in range(NROWS) for j in range(NCOLS)]
NEGATIVE_STATES = [(1,0),(1,1)]
TERMINAL_STATES = NEGATIVE_STATES + [REWARD_STATE]
NON_TERMINAL_STATES = list(set(STATES).difference(set(TERMINAL_STATES)))


def functional_reward(s,a):
    if tuple(np.add(s,a))==REWARD_STATE:
        return 5
    if tuple(np.add(s,a)) in NEGATIVE_STATES:
        return -5
    else:
        return 0
    
def get_valid_actions(s):
    if s not in TERMINAL_STATES:
        return list(filter(lambda a: np.add(s,a)[0]>=0 and np.add(s,a)[1]>=0 and np.add(s,a)[0]<NROWS and np.add(s,a)[1]<NCOLS, ACTIONS))
    else:
        return None

ACTIONS = [(0,1), (1,0), (0,-1), (-1,0)]

STATE_ACTION_PRODUCT = list(itertools.product(STATES,ACTIONS))

#initialises action values and counts to zero
ACTION_VALUES = {pair:0 for pair in STATE_ACTION_PRODUCT}
ACTION_COUNTS = {pair:0 for pair in STATE_ACTION_PRODUCT}

POLICY = {(state, action): 1/len(ACTIONS) for (state, action) in STATE_ACTION_PRODUCT}

#print(POLICY)



for k in range(NUM_EPISODES):
    #________________________________________________________________________#
    #SIMULATES AN EPISODE
    #initialises policy to be random

    episode = []
    t=0

    #Choosing first state and action
    state = random.choice(NON_TERMINAL_STATES)

    while state not in TERMINAL_STATES:

        #gets valid actions for the current state
        VALID_ACTIONS = get_valid_actions(state)

        #gets transition probabilities for the valid actions given current policy
        PROBABILITIES = [POLICY[(state,action)] for  action in VALID_ACTIONS]

        #selects an action given the transition probabilities
        selected_action = random.choices(VALID_ACTIONS, PROBABILITIES)[0]

        #print(f"Selected action for state {state} is {selected_action}")
        #Calculates reward for using action in state
        reward = functional_reward(state, selected_action)

        #adds state, action and reward to the episode
        episode.append((state, selected_action, reward))

        #print(f"State: {state}, Action: {selected_action}, Reward: {reward}")

        #updates the state
        state = tuple(np.add(state, selected_action))
        
        #print(f"New state: {state}")
    #________________________________________________________________________#
        
    #We now update the action-values function given the episode
    Gt = 0    
    
    for i, (state, action, Rt) in enumerate(list(reversed(episode))):
        ACTION_COUNTS[(state,action)] +=1
        Gt = Rt + DISCOUNT_FACTOR*Gt
        ACTION_VALUES[(state, action)] = ACTION_VALUES[(state, action)] + (Gt - ACTION_VALUES[(state, action)])/ACTION_COUNTS[(state,action)]

    #________________________________________________________________________#
        
    # We now update the policy
    epsilon = 0.05      #1/(EPISODE_NUMBER*0.1)
    for state in NON_TERMINAL_STATES:
        best_action = None
        best_action_value = float('-inf')
        for action in get_valid_actions(state):
            if ACTION_VALUES[(state,action)] > best_action_value:
                best_action = action
                best_action_value = ACTION_VALUES[(state,action)]
            
        for action in get_valid_actions(state):
            if action == best_action:
                POLICY[(state, action)] = 1-epsilon + epsilon/len(get_valid_actions(state))
            else:
                POLICY[(state, action)] = epsilon/len(get_valid_actions(state))

    EPISODES.append(episode)
    print(f"Episode {EPISODE_NUMBER} completed")
    EPISODE_NUMBER+=1

    #print(epsilon)
    #print(episode)

OPTIMAL_POLICY = {}

for state in NON_TERMINAL_STATES:
    best_action = None
    best_action_value = float('-inf')
    for action in get_valid_actions(state):
        if ACTION_VALUES[(state,action)] >best_action_value:
            best_action = action
            best_action_value = ACTION_VALUES[(state,action)]
    OPTIMAL_POLICY.update({state: best_action})


if VISUALISE:

    cmap = plt.get_cmap("jet")
    #print(cmap)
    # s=(1,0)

    for s in random.sample(NON_TERMINAL_STATES, 10):
        count=0
        fig, ax = plt.subplots()
        ims=[]
        starting_s=s
        print(f"Sample of agent behaviour starting at: {starting_s}")
        while True:
            
            frame = np.zeros((NCOLS,NROWS))
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
            s=tuple(np.add(s,OPTIMAL_POLICY[s]))
            if count==0:
                ax.imshow(frame)#Show initial frame
            count+=1
            if count>NCOLS*NCOLS:
                print("Agent repeating states")
                break

        ani = animation.ArtistAnimation(fig, ims, interval=800, blit=True, repeat_delay=1000)
        ani.save(f"{starting_s}.gif",dpi=300, writer=PillowWriter(fps=1))
        plt.show()




'''
# Initialize a grid to store the values
grid = [[None for _ in range(NCOLS)] for _ in range(NROWS)]

# Fill the grid with the state values
for i, j, value in states:
    grid[i][j] = value

# Print the grid
for row in grid:
    print(' '.join(f'{value:2}' for value in row))


#Action dictionary:
'''





