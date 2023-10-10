import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 1000                # number of training epizodes (multi-stage processes)
gamma = 0.95                                # discount factor

#file_name = 'map_small.txt'
#file_name = 'map_easy.txt'
#file_name = 'map_middle.txt'
#file_name = 'map_big.txt'
file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................
epochs = 50
for i in range(50):
    for raw in range(num_of_rows):
        for col in range(num_of_columns):
            if raw == 0 and col == 0:
                Q = sf.sailor_train(reward_map, Q, number_of_episodes, gamma, [raw, col], True)
            else:
                Q = sf.sailor_train(reward_map, Q, number_of_episodes, gamma, [raw, col], False)

sf.sailor_test(reward_map, Q, number_of_episodes)
sf.draw(reward_map,Q)
