import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 100                # number of training epizodes (multi-stage processes)
gamma = 0.95                           # discount factor
num_of_actions = 4

file_name = 'map_small.txt'
# file_name = 'map_easy.txt'
#file_name = 'map_middle.txt'
#file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, num_of_actions], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................

# ITERACJA STRATEGII
# epochs = 30
# for i in range(epochs):
#     print(f'epoch {i + 1}')
#     for row in range(num_of_rows):
#         for col in range(num_of_columns):
#             for action in range(num_of_actions):
#                 if i == 0:
#                     Q = sf.sailor_train_strategy_iteration(reward_map, Q, number_of_episodes, gamma, [row, col], action, True)
#                 else:
#                     Q = sf.sailor_train_strategy_iteration(reward_map, Q, number_of_episodes, gamma, [row, col], action, False)
#     sf.sailor_test(reward_map, Q, number_of_episodes)
#     sf.draw(reward_map, Q)

# # ITERACJA WARTOŚCI
# epochs = 2000
# for i in range(epochs):
#     for row in range(num_of_rows):
#         for col in range(num_of_columns):
#             for action in range(num_of_actions):
#                 sf.sailor_value(reward_map, Q, i + 1, [row, col], action, gamma)
#     if i % 50 == 0:
#         print(f'epoch {i + 1}')
#         sf.sailor_test(reward_map, Q, number_of_episodes)
#         sf.draw(reward_map, Q)
# sf.sailor_test(reward_map, Q, number_of_episodes)
# sf.draw(reward_map, Q)

# # DYNAMICZNA ITERACJA WARTOŚCI
V = np.zeros([num_of_rows, num_of_columns], dtype=float)
delta = 1000
while delta >= 0.00001:
    Vpom = np.copy(V)
    delta = 0
    for row in range(num_of_rows):
        for col in range(num_of_columns):
            V, delta, Q = sf.dynamic_value_iteration(reward_map, V, Vpom, [row, col], gamma, delta, Q)

    sf.sailor_test(reward_map, Q, number_of_episodes)
    sf.draw(reward_map, Q)

sf.sailor_test(reward_map, Q, number_of_episodes)
sf.draw(reward_map, Q)

