import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
import board_games_fun as bfun

# train function using Q table - more general approach: for each agent his opponent is 
# treated as an envoronment. The Q-value update is delayed up to opponent's move due to 
# the potentially random opponent strategy.
# inputs:
#    game_object - Tictactoe, Connect4, Chess, etc ...
#    players_to_train - name of player for strategy training: 
#        [1] for 'x', [2] for "o" or [1,2] for both players
#    strategy_x - strategy for player x - probability of each possible move in each state
#    strategy_o - strategy for player o
#    x is the player who starts game (in chess white pieces player)
#    choose_random - numbers of moves with puting x or o into random empty cell 
#    (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
# output:??each possible move in each state)
def board_game_train_Q(game_object, players_to_train, strategy_x=[], strategy_o=[], number_of_games = 1000, choose_random = []):
    
    #number_of_games = 2000   # e.g. number of episodes
    epsylon = 0.3              # exploration factor
    T = 3
    Tmin = 0.3
    if_softmax = True          # softmax (True) with T or epsilon-greedy (False) with epsilon random factor
    alpha = 0.1                # learning speed factor
    alpha_min = 0.01
    # alpha and epsylon can be changed in time
    gamma = 0.9                # discount factor (if < 1 near rewards are better than far, far punishments are better than near)
    lambda_ = 0               # fresh factor (if 0 then without eligibility traces)
    if_sarsa = 0             # 1 - SARSA
                            # 0 - Q-learning

    t1 = time.time()

    dT  = (T - Tmin)/number_of_games                # change of temperature - softmax randomness parameter      
    walpha = np.power(alpha_min/alpha,1/number_of_games)
    Q_x = {}                       # state action valueas tabular version represented by dictionary (board string as a key)
    Q_o = {}
    print("... start training by "+str(number_of_games) + " games")

    for game_nr in range(number_of_games):                # episodes loop
        #print("game = " + str(game_nr))
        State = game_object.initial_state()               # initial state - empty board in tic-tac
        #num_rows, num_col = np.shape(State)

        player = 1                                        # first movement by player 1(cross)

        if_end_of_loop = False    # the one more step is used for update the last action value 
        if_end_of_game = False 
        step_number = 0

        while (if_end_of_loop == False):                           # episode steps loop
            step_number += 1
            Reward = 0

            if not if_end_of_game:
                # Possible next states (2D arrays) for oponent (== current player actions) + rewards for each action 
                #_, next_states, rewards = game_object.whoplay_next_states_rewards(State)
                actions = game_object.actions(State, player)

                num_of_actions = len(actions) 
                #pdb.set_trace()         # traceing (c for continue)
                    
                if player in players_to_train:
                    state_key = str(State)
                    if player == 1:
                        Q = Q_x
                    else:
                        Q = Q_o

                    if state_key in Q:
                        Q_state_actions = Q[state_key]
                    else:
                        Q_state_actions = np.zeros([num_of_actions], dtype = float)
                        Q[state_key] = Q_state_actions
                    
                    if player == 1:
                        Q_best, ind_Q_best = np.max(Q_state_actions), np.argmax(Q_state_actions)
                    else:
                        Q_best, ind_Q_best = np.min(Q_state_actions), np.argmin(Q_state_actions)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:    # using fixed strategy
                    s = str(State)
                    if player == 1:
                        if s in strategy_x:
                            distr = strategy_x[s] # action probability distribution pi(a|s)
                            action_nr = bfun.choose_action(distr) 
                        else:   # if lack of this state in strategy 
                            action_nr = np.random.randint(num_of_actions)
                    elif player == 2:
                        if s in strategy_o:
                            distr = strategy_o[s]
                            action_nr = bfun.choose_action(distr) 
                        else:
                            action_nr = np.random.randint(num_of_actions)
                else:                                   # using learned strategy with exploration
                    if if_softmax:
                        distrib = bfun.softmax((1/T)*Q_state_actions)
                        action_nr = bfun.choose_action(distrib)   
                    elif np.random.random() < epsylon:                                       # epsilon-greedy exploration
                        action_nr = np.random.randint(num_of_actions)  # next state random choose
                    else: 
                        action_nr = ind_Q_best                         # next state optimal-known choose
                    
                            
                NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])


            if player in players_to_train:
                # Informations for future update (opponent move is unknown in this moment)
                if player == 1:
                    ToUpdate1 = [state_key, action_nr, Reward]
                else:
                    ToUpdate2 = [state_key, action_nr, Reward]

                # state value correction evaluation:
                # Calculate the reward for the next state
                R = game_object.reward(NextState)
                # Update Q-value for the current state and action
                # Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state, a)) - Q(state, action))
                # Assuming alpha, gamma, and Q-table are defined and accessible here

                Q[State, action_nr] += alpha * (R + gamma * np.max(Q[NextState]) - Q[State, action_nr])

                # Update information from previous player step:
                if step_number > 2:
                    if player == 1:
                        state_key_prev, action_nr_prev, Reward_prev = ToUpdate1
                    else:
                        state_key_prev, action_nr_prev, Reward_prev = ToUpdate2

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    if player == 1:
                        Q_up = Q_o
                    else:
                        Q_up = Q_x
                    Q_up[state_key_prev][action_nr_prev] += \
                        alpha*(Reward_prev + Reward + gamma*Q_next- Q_up[state_key_prev][action_nr_prev])
            

            State = NextState                               # move to next state        
            player = 3 - player                             # player changing

            if if_end_of_game:                              
                if_end_of_loop = True                       # if all updates are done

            if game_object.end_of_game(Reward,step_number,State,action_nr):      # win or draw
                if_end_of_game = True
                 
        T -= dT
        alpha *= walpha
 

    # transformation Q into strategy (generally non-detrministic: pi(a|s)):
    strategy_x = {}               # strategy is also a dictionary but with prob.distrib. as values
    for key in Q_x.keys():
        Board = bfun.string_to_2Darray(key)
        actions = game_object.actions(Board, player = 1)
        if (len(actions) > 0):
            num_of_actions = len(actions)
            distrib = np.zeros([num_of_actions],dtype=float)
            distrib[np.argmax(Q_x[key])] = 1.0
            strategy_x[key] = distrib

    strategy_o = {}               # strategy is also a dictionary but with prob.distrib. as values
    for key in Q_o.keys():
        Board = bfun.string_to_2Darray(key)
        actions = game_object.actions(Board, player = 2)
        if (len(actions) > 0):
            num_of_actions = len(actions)
            distrib = np.zeros([num_of_actions],dtype=float)
            distrib[np.argmin(Q_o[key])] = 1.0
            strategy_o[key] = distrib

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt*1000/number_of_games) )
    print("number of states for x = " + str(len(strategy_x)) + " for o = "+ str(len(strategy_o)))
    return strategy_x, strategy_o

# Q-learning, but update for each player based on opponent's action in next step 
# (not based on player actions after 2 steps as above)
def board_game_train_Q2(game_object, players_to_train, strategy_x=[], strategy_o=[], number_of_games = 2000):
    
    #number_of_games = 2000   # e.g. number of episodes
    epsylon = 0.3              # exploration factor
    T = 3
    Tmin = 0.3
    if_softmax = False          # softmax (True) with T or epsilon-greedy (False) with epsilon random factor
    alpha = 0.1                # learning speed factor
    alpha_min = 0.01
    # alpha and epsylon can be changed in time
    gamma = 0.9                # discount factor (if < 1 near rewards are better than far, far punishments are better than near)
    lambda_ = 0.3               # fresh factor (if 0 then without eligibility traces)
    if_sarsa = 0             # 1 - SARSA
                            # 0 - Q-learning


    t1 = time.time()

    # numbers of moves with puting x or o into random empty cell (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
    choose_random = []

    dT  = (T - Tmin)/number_of_games                # change of temperature - softmax randomness parameter      
    walpha = np.power(alpha_min/alpha,1/number_of_games)
    Q_x = {}                       # state action valueas tabular version represented by dictionary (board string as a key)
    Q_o = {}
    print("... start training by "+str(number_of_games) + " games")

    for game_nr in range(number_of_games):                # episodes loop
        #print("game = " + str(game_nr))
        State = game_object.initial_state()               # initial state - empty board in tic-tac
        #num_rows, num_col = np.shape(State)

        player = 1                                        # first movement by player 1(cross)

        if_end_of_loop = False    # the one more step is used for update the last action value 
        if_end_of_game = False 
        step_number = 0

        while (if_end_of_loop == False):                           # episode steps loop
            step_number += 1
            Reward = 0

            if not if_end_of_game:
                # Possible next states (2D arrays) for oponent (== current player actions) + rewards for each action 
                #_, next_states, rewards = game_object.whoplay_next_states_rewards(State)
                actions = game_object.actions(State, player)

                num_of_actions = len(actions) 
                #pdb.set_trace()         # traceing (c for continue)
                state_key = str(State)                    

                if player in players_to_train:       
                    if player == 1:
                        Q = Q_x
                    else:
                        Q = Q_o       
                    if state_key in Q:
                        Q_state_actions = Q[state_key]
                    else:
                        Q_state_actions = np.zeros([num_of_actions], dtype = float)
                        Q[state_key] = Q_state_actions
                    
                    if player == 1:
                        Q_best, ind_Q_best = np.max(Q_state_actions), np.argmax(Q_state_actions)
                    else:
                        Q_best, ind_Q_best = np.min(Q_state_actions), np.argmin(Q_state_actions)

                if step_number in choose_random:
                    action_nr = np.random.randint(num_of_actions)
                elif player not in players_to_train:    # using fixed strategy
                    s = str(State)
                    if player == 1:
                        if s in strategy_x:
                            distr = strategy_x[s] # action probability distribution pi(a|s)
                            action_nr = bfun.choose_action(distr) 
                        else:   # if lack of this state in strategy 
                            action_nr = np.random.randint(num_of_actions)
                    elif player == 2:
                        if s in strategy_o:
                            distr = strategy_o[s]
                            action_nr = bfun.choose_action(distr) 
                        else:
                            action_nr = np.random.randint(num_of_actions)
                else:                                   # using learned strategy with exploration
                    if if_softmax:
                        distrib = bfun.softmax((1/T)*Q_state_actions)
                        action_nr = bfun.choose_action(distrib)   
                    elif np.random.random() < epsylon:                                       # epsilon-greedy exploration
                        action_nr = np.random.randint(num_of_actions)  # next state random choose
                    else: 
                        action_nr = ind_Q_best                         # next state optimal-known choose
                    
                            
                NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])


             # Informations for future update (opponent move is unknown in this moment)
            if player == 1:
                ToUpdate1 = [state_key, action_nr, Reward]
            else:
                ToUpdate2 = [state_key, action_nr, Reward]

            if 3 - player in players_to_train:
                # state value correction evaluation:
                # ..................................
                # .......do it by yourself..........
                # ..................................
                #R = game.reward(NextState)

                # Update information from previous player step:
                if step_number > 1:
                    if player == 1:
                        #state_key_prev, action_nr_prev, Reward_prev = ToUpdate1
                        state_key_prev, action_nr_prev, Reward_prev = ToUpdate2
                    else:
                        #state_key_prev, action_nr_prev, Reward_prev = ToUpdate2
                        state_key_prev, action_nr_prev, Reward_prev = ToUpdate1

                    if if_end_of_game:
                        Q_next = 0
                    else:
                        if if_sarsa:
                            Q_next = Q_state_actions[action_nr]
                        else:
                            Q_next = Q_best

                    if player == 1:
                        Q_up = Q_o
                    else:
                        Q_up = Q_x

                    Q_up[state_key_prev][action_nr_prev] += \
                        alpha*(Reward_prev + gamma*Q_next- Q_up[state_key_prev][action_nr_prev])
            

            State = NextState                               # move to next state        
            player = 3 - player                             # player changing

            if if_end_of_game:                              
                if_end_of_loop = True                       # if all updates are done

            if game_object.end_of_game(Reward,step_number,State,action_nr):      # win or draw
                if_end_of_game = True
                 
        T -= dT
        alpha *= walpha
 

    # transformation Q into strategy (generally non-detrministic: pi(a|s)):
    strategy_x = {}               # strategy is also a dictionary but with prob.distrib. as values
    for key in Q_x.keys():
        Board = bfun.string_to_2Darray(key)
        actions = game_object.actions(Board, player = 1)
        if (len(actions) > 0):
            num_of_actions = len(actions)
            distrib = np.zeros([num_of_actions],dtype=float)
            distrib[np.argmax(Q_x[key])] = 1.0
            strategy_x[key] = distrib

    strategy_o = {}               # strategy is also a dictionary but with prob.distrib. as values
    for key in Q_o.keys():
        Board = bfun.string_to_2Darray(key)
        actions = game_object.actions(Board, player = 2)
        if (len(actions) > 0):
            num_of_actions = len(actions)
            distrib = np.zeros([num_of_actions],dtype=float)
            distrib[np.argmin(Q_o[key])] = 1.0
            strategy_o[key] = distrib

    dt = time.time() - t1
    print("training finished after %.3f sec. (%.3f sec./1000 games)" % (dt, dt*1000/number_of_games) )
    print("number of states for x = " + str(len(strategy_x)) + " for o = "+ str(len(strategy_o)))
    return strategy_x, strategy_o

# Test of given game (game_object) and strategies for player x and o
# choose_random - numbers of moves with puting x or o into random empty cell 
# (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
def board_game_test(game_object, strategy_x, strategy_o, number_of_games = 100, choose_random = []):
    #number_of_games = 100   # e.g. number of episodes

   
    num_win_x = 0
    num_win_o = 0
    num_draws = 0
    
    Games = []
    Rewards = []

    for game in range(number_of_games):                   # episodes loop
        #print("game = " + str(game))
        State = game_object.initial_state()               # initial state - empty board in tictac
        #num_rows, num_col = np.shape(State)
        player = 1                                        # first movement by player 1(cross)
        if_end = False 
        step_number = 0
        Boards = []

        Boards.append(State)

        while (if_end == False):                           # episode steps loop
            step_number += 1

            #_, next_states, rewards = game_object.whoplay_next_states_rewards(State)       # next states from board A 
            actions = game_object.actions(State, player)   

            s = str(State)
            if player == 1:
                if s in strategy_x:
                    distr = strategy_x[s] # action probability distribution pi(a|s)
                    action_nr = bfun.choose_action(distr) 
                else:
                    action_nr = np.random.randint(len(actions))
            elif player == 2:
                if s in strategy_o:
                    distr = strategy_o[s]
                    action_nr = bfun.choose_action(distr) 
                else:
                    action_nr = np.random.randint(len(actions))
            #pdb.set_trace()

        
            if step_number in choose_random:
                action_nr = np.random.randint(len(actions))
      
            NextState, Reward =  game_object.next_state_and_reward(player,State, actions[action_nr])


            State = NextState                                        # move to next state
            Boards.append(State)                                     # board for game description

            player = 3 - player                                      # player changing

            if game_object.end_of_game(Reward, step_number,State,action_nr):      # win or draw
                if_end = True
                if Reward == 1:
                    num_win_x += 1
                elif Reward == -1:
                    num_win_o += 1
                elif Reward == 0:
                    num_draws += 1
                Rewards.append(Reward)
        Games.append(Boards)
    return num_win_x, num_win_o, num_draws, Games, Rewards






def experiment_par_train():
    print("\nUCZENIE DWOCH STRATEGII JEDNOCZESNIE\n")
    game = bfun.Tictactoe()                      # game class object
    #game = bfun.Tictac_general(4,3,3,True)
    #game = bfun.Connect4()
    strategy_x, strategy_o = board_game_train_Q2(game,players_to_train = [1,2], number_of_games = 10000)
    bfun.save_strategy("strategy_x.txt",strategy_x)
    bfun.save_strategy("strategy_o.txt",strategy_o)
    print("test stategii uczonych jednocześnie:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game,strategy_x,strategy_o,choose_random=[])
    
    print("test stategii x na losowej o:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x,bfun.random_strategy())
    game.print_test_to_file("gry_uczonej_X_z_losowa_O.txt",num_win_x, num_win_o, num_draws, Games, Rewards)
    print("test stategii o na losowej x:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, bfun.random_strategy(),strategy_o)
    game.print_test_to_file("gry_losowej_X_z_uczona_O.txt",num_win_x, num_win_o, num_draws, Games, Rewards)
    
    print("test stategii x na częściowo losowej o:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        random_factor = i/10
        t.append(random_factor)       
        num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x, bfun.strategy_nondeterm(strategy_o,random_factor))
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    plt.plot(t,nwin_x,"x",t,nwin_o,"o",t, ndraws,"-")
    plt.title("tictac test results with Nash x strategy and partially random o strategy")
    plt.xlabel("randomness od o strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins","num.of o wins","num of draws"])
    plt.savefig("test_Nash_x_strategy_random_o_strategy.png")
    fig1 = plt
    plt.show()

    print("test stategii o na częściowo losowej x:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        random_factor = i/10
        t.append(random_factor)       
        num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, bfun.strategy_nondeterm(strategy_x,random_factor),strategy_o)
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    plt.plot(t,nwin_x,"x",t,nwin_o,"o",t, ndraws,"-")
    plt.title("tictac test results with Nash o strategy and partially random x strategy")
    plt.xlabel("randomness od x strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins","num.of o wins","num of draws"])
    plt.savefig("test_Nash_o_strategy_random_x_strategy.png")
    plt.show()
    fig2 = plt


def self_play():
    print("\nUCZENIE DWOCH STRATEGII METODĄ SELF-PLAY\n")
    game = bfun.Tictactoe()
    epochs = 100
    algorithms = ['Q2']
    #algorithms = ['Q', 'Q2']

    for algorithm in algorithms:
        # inicjalizacja losowych strategii x oraz o
        strategy_x = bfun.random_strategy()
        strategy_o = bfun.random_strategy()

        for epoch in range(epochs):
            if epoch % 2:
                print("\nDouczanie strategii o na ustalonej strategii x, by sprawdzić na ile uczenie równoczesne było skuteczne.\n")
                if algorithm == 'Q2':
                    _, strategy_o = board_game_train_Q2(game, players_to_train=[2], strategy_x=strategy_x, strategy_o=strategy_o, number_of_games=1000)
                else:
                    _, strategy_o = board_game_train_Q(game, players_to_train=[2], strategy_x=strategy_x, strategy_o=strategy_o, number_of_games=1000)
            else:
                print(
                    "\nDouczanie strategii x na ustalonej strategii o, by sprawdzić na ile uczenie równoczesne było skuteczne.\n")
                if algorithm == 'Q2':
                    strategy_x, _ = board_game_train_Q2(game, players_to_train=[2],strategy_x=strategy_x, strategy_o=strategy_o, number_of_games=1000)
                else:
                    strategy_x, _ = board_game_train_Q(game, players_to_train=[2], strategy_x=strategy_x, strategy_o=strategy_o, number_of_games=1000)

        bfun.save_strategy("strategy_o_self-play.txt", strategy_o)
        print("test stategii self-play o oraz x:")
        num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x, strategy_o)
        game.print_test_to_file("test stategii self-play o oraz x.txt", num_win_x, num_win_o, num_draws, Games, Rewards)

        print("test stategii x na częściowo losowej o:")
        t = []
        nwin_x = []
        nwin_o = []
        ndraws = []
        for i in range(10):
            random_factor = i / 10
            t.append(random_factor)
            num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x,
                                                                              bfun.strategy_nondeterm(strategy_o,
                                                                                                      random_factor))
            nwin_x.append(num_win_x)
            nwin_o.append(num_win_o)
            ndraws.append(num_draws)
        plt.plot(t, nwin_x, "x", t, nwin_o, "o", t, ndraws, "-")
        plt.title(f"tictac test results with self-play {algorithm} x strategy and partially random o strategy")
        plt.xlabel("randomness of o strategy (1 - full random)")
        plt.ylabel("number of games")
        plt.legend(["num.of x wins", "num.of o wins", "num of draws"])
        plt.savefig(f"test_self-play_{algorithm}_x_strategy_random_o_strategy.png")
        fig1 = plt
        plt.show()

        print("test stategii o na częściowo losowej x:")
        t = []
        nwin_x = []
        nwin_o = []
        ndraws = []
        for i in range(10):
            random_factor = i / 10
            t.append(random_factor)
            num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, bfun.strategy_nondeterm(strategy_x,
                                                                                                            random_factor),
                                                                              strategy_o)
            nwin_x.append(num_win_x)
            nwin_o.append(num_win_o)
            ndraws.append(num_draws)
        plt.plot(t, nwin_x, "x", t, nwin_o, "o", t, ndraws, "-")
        plt.title(f"tictac test results with self-play {algorithm} o strategy and partially random x strategy")
        plt.xlabel("randomness of x strategy (1 - full random)")
        plt.ylabel("number of games")
        plt.legend(["num.of x wins", "num.of o wins", "num of draws"])
        plt.savefig(f"test_self-play_{algorithm}_o_strategy_random_x_strategy.png")
        plt.show()


#experiment_par_train()
self_play()