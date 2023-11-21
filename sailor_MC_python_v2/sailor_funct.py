# sailor functions

import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

WALL_COLLID_REWARD = -0.1

def load_data(file_name):
    file_ptr = open(file_name, 'r').read()
    lines = file_ptr.split('\n')
    number_of_lines = lines.__len__() - 1
    row_values = lines[0].split()
    number_of_values = row_values.__len__()

    number_of_rows = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            number_of_rows += 1
            num_of_columns = number_of_values

    map_of_rewards = np.zeros([number_of_rows, num_of_columns], dtype=float)
    print("examples shape = " + str(map_of_rewards.shape))
    
    index = 0
    for i in range(number_of_lines):
        row_values = lines[i].split()
        number_of_values = row_values.__len__()
        if (number_of_values > 0):
            for j in range(number_of_values):
                map_of_rewards[index][j] = float(row_values[j])
            index = index + 1

    return map_of_rewards

def sailor_value(reward_map, Q, epoch, init_state, init_action, gamma):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5 * (num_of_rows + num_of_columns))
    final_reward = 0.0

    state = init_state
    the_end = False
    nr_pos = 0
    while not the_end:
        nr_pos = nr_pos + 1
        action = (init_action if (state[0] == init_state[0] and state[1] == init_state[1]) else np.argmax(Q[state[0], state[1]])) + 1
        state_next, reward = environment(state, action, reward_map)
        state = state_next

        if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
            the_end = True

        final_reward += (gamma ** nr_pos) * reward

    value = Q[init_state[0], init_state[1], init_action]
    alpha = 1 / epoch
    Q[init_state[0], init_state[1], init_action] = (1-alpha) * value + alpha * final_reward

def environment(state, action, reward_map):
    num_of_rows, num_of_columns = reward_map.shape
    prob_side = 0.16
    prob_back = 0.04
    wall_colid_reward = -0.1

    state_new = np.copy(state)
    reward = 0

    los = np.random.random()    # Random number from uniform distr. from range (0,1)

    # Action values (1 - right, 2 - up, 3 - left, 4 - bottom):
    choosen_action = -1
    if action == 1:
        if los < prob_back:
            choosen_action = 3
        elif los < prob_back + prob_side:
            choosen_action = 2
        elif  los < prob_back + 2*prob_side:
            choosen_action = 4
        else:
            choosen_action = 1
    elif action == 2:
        if los < prob_back:
            choosen_action = 4
        elif los < prob_back + prob_side:
            choosen_action = 1
        elif  los < prob_back + 2*prob_side:
            choosen_action = 3
        else:
            choosen_action = 2
    elif action == 3:
        if los < prob_back:
            choosen_action = 1
        elif los < prob_back + prob_side:
            choosen_action = 2
        elif  los < prob_back + 2*prob_side:
            choosen_action = 4
        else:
            choosen_action = 3
    elif action == 4:
        if los < prob_back:
            choosen_action = 2
        elif los < prob_back + prob_side:
            choosen_action = 1
        elif  los < prob_back + 2*prob_side:
            choosen_action = 3
        else:
            choosen_action = 4

    # Action identifiers (1 - right, 2 - up, 3 - left, 4 - bottom): 
    if choosen_action == 1:
        if state[1] < num_of_columns - 1:
            state_new[1] += 1
            reward += reward_map[state_new[0],state_new[1]]
        else:
            reward += wall_colid_reward
    elif  choosen_action == 2:
        if state[0] > 0:
            state_new[0] -= 1
            reward += reward_map[state_new[0],state_new[1]]
        else:
            reward += wall_colid_reward
    if choosen_action == 3:
        if state[1] > 0:
            state_new[1] -= 1
            reward += reward_map[state_new[0],state_new[1]]
        else:
            reward += wall_colid_reward
    elif  choosen_action == 4:
        if state[0] < num_of_rows - 1:
            state_new[0] += 1
            reward += reward_map[state_new[0],state_new[1]]
        else:
            reward += wall_colid_reward

    return state_new, reward

# test for given number of episodes - pure exploration
# higher number of episodes gives higher precision
def sailor_test(reward_map, Q, num_of_episodes):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
    sum_of_rewards = np.zeros([num_of_episodes], dtype=float)

    for episode in range(num_of_episodes):
        state = np.zeros([2],dtype=int)                            # initial state here [1 1] but rather random due to exploration
        state[0] = np.random.randint(0,num_of_rows)
        the_end = False
        nr_pos = 0
        while the_end == False:
            nr_pos = nr_pos + 1                            # move number
        
            # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom): 
            action = 1 + np.argmax(Q[state[0],state[1], :])
            state_next, reward  = environment(state, action, reward_map)
            state = state_next       # going to the next state
        
            # end of episode if maximum number of steps is reached or last column
            # is reached
            if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns-1):
                the_end = True
        
            sum_of_rewards[episode] += reward
    print('test-'+str(num_of_episodes)+' mean sum of rewards = ' + str(np.mean(sum_of_rewards)))

def sailor_train_strategy_iteration(reward_map, Q, num_of_episodes, gamma, init_state, init_action, first_training):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
    sum_of_rewards = np.zeros([num_of_episodes], dtype=float)

    for episode in range(num_of_episodes):
        state = init_state #np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
        the_end = False
        nr_pos = 0
        while the_end == False:
            nr_pos = nr_pos + 1  # move number

            # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
            if nr_pos == 1:
                action = 1 + init_action
            else:
                if first_training:
                    action = 1 + np.random.randint(0, 4)
                else:
                    action = 1 + np.argmax(Q[state[0], state[1], :])
            state_next, reward = environment(state, action, reward_map)
            state = state_next  # going to the next state

            # end of episode if maximum number of steps is reached or last column
            # is reached
            if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
                the_end = True

            sum_of_rewards[episode] += (reward * np.power(gamma, nr_pos))

    for episode in range(num_of_episodes):
        Q[init_state[0]][init_state[1]][init_action] += np.mean(sum_of_rewards)
    return Q

def sailor_train_value_iteration(reward_map, Q, gamma, init_state, init_action, first_training, alpha):
    num_of_rows, num_of_columns = reward_map.shape
    num_of_steps_max = int(5 * (num_of_rows + num_of_columns))  # maximum number of steps in an episode
    sum_of_rewards = 0

    state = init_state #np.zeros([2], dtype=int)  # initial state here [1 1] but rather random due to exploration
    the_end = False
    nr_pos = 0
    while the_end == False:
        nr_pos = nr_pos + 1  # move number

        # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
        action = (init_action if (state[0] == init_state[0] and state[1] == init_state[1]) else np.argmax(Q[state[0], state[1]])) + 1
        state_next, reward = environment(state, action, reward_map)

        sum_of_rewards += np.power(gamma, nr_pos - 1) * reward
        state = state_next  # going to the next state

        # end of episode if maximum number of steps is reached or last column
        # is reached
        if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
            the_end = True

    Q[init_state[0]][init_state[1]][init_action] = (1 - alpha) * Q[init_state[0]][init_state[1]][init_action] + (alpha * sum_of_rewards)
    return Q

# drawing map of rewards and strategy using arrows
def draw(reward_map, Q):
    num_of_rows, num_of_columns = reward_map.shape
    image_map = np.zeros([num_of_rows, num_of_columns, 3], dtype=int)
    for i in range(num_of_rows):
        for j in range(num_of_columns):
            if reward_map[i,j] > 0:
                image_map[i,j,0] = 210
                image_map[i,j,1] = 210
            elif reward_map[i,j] == 0:
                image_map[i,j,2] = 255
            elif reward_map[i,j] >= -1:
                image_map[i,j,2] = 255-32
            elif reward_map[i,j] >= -5:
                image_map[i,j,2] = 255-64
            elif reward_map[i,j] >= -10:
                image_map[i,j,2] = 255-128
    f = plt.figure()

    # Action identifiers (1 - right, 2 - up, 3 - left, 4 - bottom): 
    for i in range(num_of_rows):
        for j in range(num_of_columns-1):
            action = 1 + np.argmax(Q[i,j,:])
            if action == 1:
                # xytext - starting point, xy - end point
                plt.annotate('', xytext=(j-0.4, i), xy=(j+0.4, i),
                    arrowprops=dict(facecolor='red', shrink=0.09),
                    )
            elif action == 2:
                plt.annotate('', xytext=(j, i+0.4), xy=(j, i-0.4),
                    arrowprops=dict(facecolor='red', shrink=0.09),
                    )
            elif action == 3:
                plt.annotate('',  xytext=(j+0.4, i),xy=(j-0.4, i),
                    arrowprops=dict(facecolor='red', shrink=0.09),
                    )
            elif action == 4:
                plt.annotate('',  xytext=(j, i-0.4),xy=(j, i+0.4),
                    arrowprops=dict(facecolor='red', shrink=0.09),
                    )


    im = plt.imshow(image_map)
    plt.show()
    f.savefig('image_best_strategy.svg')

def get_move_probability(action, direction):
    if action == 1:
        if direction == 1:
            return 0.4
        elif direction == 2:
            return 0.2
        elif direction == 3:
            return 0.04
        else:
            return 0.36
    elif action == 2:
        if direction == 1:
            return 0.2
        elif direction == 2:
            return 0.4
        elif direction == 3:
            return 0.36
        else:
            return 0.04
    elif action == 3:
        if direction == 1:
            return 0.04
        elif direction == 2:
            return 0.2
        elif direction == 3:
            return 0.4
        else:
            return 0.36
    elif action == 4:
        if direction == 1:
            return 0.2
        elif direction == 2:
            return 0.04
        elif direction == 3:
            return 0.36
        else:
            return 0.4

def get_reward(init_state, action, reward_map):
    num_of_rows, num_of_columns = reward_map.shape
    if action == 1:
        return reward_map[init_state[0]][init_state[1] + 1] if (init_state[1] + 1) < num_of_columns else WALL_COLLID_REWARD
    elif action == 2:
        return reward_map[init_state[0] - 1][init_state[1]] if (init_state[0] - 1) >= 0 else WALL_COLLID_REWARD
    elif action == 3:
        return reward_map[init_state[0]][init_state[1] - 1] if (init_state[1] - 1) >= 0 else WALL_COLLID_REWARD
    elif action == 4:
        return reward_map[init_state[0] + 1][init_state[1]] if (init_state[0] + 1) < num_of_rows else WALL_COLLID_REWARD

def dynamic_value_iteration(reward_map, V, Vpom, init_state, gamma, delta, Q):
    num_of_rows, num_of_columns = reward_map.shape
    max_reward = -1000
    for action in range(1, 5):
        # r(s,a)
        reward = get_reward(init_state, action, reward_map)
        # Vpom(s')
        right_value = V[init_state[0]][init_state[1] + 1] if (init_state[1] + 1) < num_of_columns else WALL_COLLID_REWARD
        up_value = V[init_state[0] - 1][init_state[1]] if (init_state[0] - 1) >= 0 else WALL_COLLID_REWARD
        left_value = V[init_state[0]][init_state[1] - 1] if (init_state[1] - 1) >= 0 else WALL_COLLID_REWARD
        down_value = V[init_state[0] + 1][init_state[1]] if (init_state[0] + 1) < num_of_rows else WALL_COLLID_REWARD
        # sum(p(s'|s, a) * Vpom(s')
        right_probability = get_move_probability(action, 1)
        up_probability = get_move_probability(action, 2)
        left_probability = get_move_probability(action, 3)
        down_probability = get_move_probability(action, 4)
        # V(s)
        reward += (gamma * ((right_value * right_probability) + (up_value * up_probability) + (left_value * left_probability) + (down_value * down_probability)))
        Q[init_state[0]][init_state[1]][action - 1] = reward

        if reward > max_reward:
            max_reward = reward

    V[init_state[0]][init_state[1]] = max_reward
    delta = max(delta, abs(V[init_state[0]][init_state[1]] - Vpom[init_state[0]][init_state[1]]))
    return V, delta, Q


   
