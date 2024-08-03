import itertools
import random
import numpy as np
##Programme to randomly sample from an array 
# import random 
# k = [i for i in range(11)]
# print(k)

# rs = random.sample(k, 4)
# print(rs)


# MY_LIST = [i for i in range (7)]
# print(MY_LIST)

# NEW_LIST = list(filter(lambda number: 1+number>4, MY_LIST))
# print(NEW_LIST)
NROWS = 4
NCOLS = 4

terminal_states = {(0,0)}
states = {(i,j) for i in range(NROWS) for j in range(NCOLS)}
non_terminal_states = states - terminal_states
print(non_terminal_states) 

product = set(itertools.product(terminal_states, states))
print(product)

actions = ["U", "D", "L", "R"]
probabilities = [0.7,0.1,0.1,0.1]

for k in range(20):
    print(random.choices(actions, probabilities)[0])

A = [1,1]
B = [2,2]
C = np.add(A,B)
print(C)

D = [(1,2), (3,4), (5,6)]
print(list(reversed(D)))

def func(n):
    return 2*n

nums_greater_than_2_point_5 = list(filter(lambda x: func(x)>7, [1,2,3,4,5,6,7,8,9]))
print(nums_greater_than_2_point_5)

dict = {}
dict.update({"Update":3})
dict.update({"New_thing":4})
print(dict['New_thing'])

print(max(dict.values()))