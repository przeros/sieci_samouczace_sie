# functions supporting tictac project 
import numpy as np

# -----------------------------------------------------------
# Board games classes (tic-tac-toe, gomoku, renju, connect4, checkers, chess, ...)

# tic-tac-toe  first movement by player 1 - cross, second by player 2 - circle
# each board with 3x3 cells represents state , each cell can be empty (0), contain x (1) or contain o (2)
# e.g. for board A = [[0,2,0],[0,1,0], [0,0,0]]:
#          _ o _
#     A =  _ x _
#          _ _ _
class Tictactoe:
    def __init__(self):
        pass

    def initial_state(self):
        return np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=int)

    # reward value from 'x' player point of view only in terminal states (-1: lose, 0: draw, 1: win)
    def reward(self, A):
        R = 0
        if (np.max(np.sum(A,0)) == 6) | (np.max(np.sum(A,1)) == 6) | \
            (A[0,0] + A[1,1] + A[2,2] == 6) | (A[0,2] + A[1,1] + A[2,0] == 6):
            R = -1
        if (R == 0):
            B = A % 2
            if (np.max(np.sum(B,0)) == 3) | (np.max(np.sum(B,1)) == 3) | \
            (B[0,0] + B[1,1] + B[2,2] == 3) | (B[0,2] + B[1,1] + B[2,0] == 3):
                R = 1
        return R

    def end_of_game(self, _R = 0, _number_of_moves = 0, _Board = [], _action_nr = 0):
        if (np.abs(_R) > 0)|(_number_of_moves >= 9):
            return True
        else: return False

    # output: player, list of states after possible moves, rewards for moves
    def actions(self, A, player = 0):
        actions = []

        empty_cells = np.where(A == 0)
        empty_cells_number = len(empty_cells[0])
                        
        for i in range(empty_cells_number):
            row = empty_cells[0][i]
            column = empty_cells[1][i]
            actions.append([row, column])

        return actions

    def next_state_and_reward(self, player, State, action):
        row, col = action
        NextState = np.copy(State)
        NextState[row, col] = player
        reward = self.reward(NextState)
        return NextState, reward
        
    # printing to text file info about test results and particular games (each game in a row)    
    def print_test_to_file(self, filename,num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename,"w")
        number_of_games = len(Games)
        """ for g in range(number_of_games):
            Boards = Games[g]
            f.write("game " + str(g) + ":\n")
            for i in range(len(Boards)):
                f.write(str(Boards[i]) + "\n")
            f.write("\n\n") """
        num_rows, num_col = np.shape(Games[0][0])
        for g in range(number_of_games):
            Boards = Games[g]
            num_of_boards = len(Boards)
            result = " draw"
            if Rewards[g] == 1:
                result = " x win"
            elif Rewards[g] == -1:
                result = " o win"
            f.write("game " + str(g) + result + ":\n")
            for r in range(num_rows):
                row = ""
                for b in range(num_of_boards):
                    A = Boards[b]
                    for c in range(num_col):
                        if A[r,c] == 0:
                            row += "_"
                        elif A[r,c] == 1:
                            row += "x"
                        elif A[r,c] == 2:
                            row += "o"
                    row += "  "
                f.write(row + "\n")
            f.write("\n") 

        print("results after %d games: " % (number_of_games))
        print("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.write("results after %d games: " % (number_of_games))
        f.write("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.close()

    # end of class Tictactoe

# general cross and circles game: board with given size + number of stones in a row
# as w win/loss condition
# (vertical, horizontal, diagonally) + information if new stones need to be adjecent
# to existing ones (if adjacent - also first move in the center of a board to simplify training of
# an approximator)
class Tictac_general:
    
    def __init__(self, num_of_rows, num_of_columns, num_of_stones, if_adjacent):
        self.num_of_rows = num_of_rows
        self.num_of_columns = num_of_columns
        self.num_of_stones = num_of_stones
        self.if_adjacent = if_adjacent

    def initial_state(self):
        return np.zeros([self.num_of_rows,self.num_of_columns],dtype=int)

    # quicker version of reward which takes into account only
    # pieses in a vertical, horizontel and diagonal / \ sequences
    # contained piece after current move 
    def reward_after_move(self, A, row, column, player):
        R = 0
        # horizontal direction:
        i = 0
        while True:
            if column + i + 1 < self.num_of_columns:
                if A[row,column + i + 1] == player:
                    i += 1
                else: break
            else: break
        num_of_pieces = i + 1 
        i = 0
        while True:
            if column - i - 1 >= 0:
                if A[row,column - i - 1] == player:
                    i += 1
                else: break
            else: break
        num_of_pieces += i
        if num_of_pieces >= self.num_of_stones:
            R = (player == 1) - (player == 2)

        if R == 0:
            # vertical direction:
            i = 0
            while True:
                if row + i + 1 < self.num_of_rows:
                    if A[row + i + 1,column] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if row - i - 1 >= 0:
                    if A[row - i - 1,column] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal \ direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows)&(column + i + 1 < self.num_of_columns):
                    if A[row + i + 1,column + i + 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if (row - i - 1 >= 0)&(column - i - 1 >= 0):
                    if A[row - i - 1,column - i - 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)

        if R == 0:
            # diagonal / direction:
            i = 0
            while True:
                if (row + i + 1 < self.num_of_rows)&(column - i - 1 >= 0):
                    if A[row + i + 1,column - i - 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces = i + 1 
            i = 0
            while True:
                if (row - i - 1 >= 0)&(column + i + 1 < self.num_of_columns):
                    if A[row - i - 1,column + i + 1] == player:
                        i += 1
                    else: break
                else: break
            num_of_pieces += i
            if num_of_pieces >= self.num_of_stones:
                R = (player == 1) - (player == 2)
        return R

    # checking if end of game (draw)
    def end_of_game(self, _R = 0, _number_of_moves = 0, _Board = [], _action_nr = 0):
        if (np.abs(_R) > 0)|(_number_of_moves >= self.num_of_rows*self.num_of_columns):
            return True
        else: return False


    # output: player, list of states after possible moves, rewards for moves
    def actions(self, A, player = 0):
        actions = []

        empty_cells = np.where(A == 0)
        empty_cells_number = len(empty_cells[0])
        number_of_pieces = self.num_of_rows*self.num_of_columns - empty_cells_number
                        
        if (self.if_adjacent)&(number_of_pieces == 0):
            actions.append([self.num_of_rows//2, self.num_of_columns//2])
        elif (self.if_adjacent)&(number_of_pieces == 1):
            actions.append([self.num_of_rows//2-1, self.num_of_columns//2])
            actions.append([self.num_of_rows//2-1, self.num_of_columns//2-1])
        else:
            for i in range(empty_cells_number):
                row = empty_cells[0][i]
                column = empty_cells[1][i]
                if self.if_adjacent:
                    num_of_neibours = 0
                    for r in range(3):
                        for c in range(3):
                            rr = row + r - 1
                            cc = column + c - 1
                            if (rr >= 0)&(rr < self.num_of_rows)&(cc >= 0)&(cc < self.num_of_columns):
                                num_of_neibours += (A[rr,cc] != 0)
                if empty_cells_number == self.num_of_rows*self.num_of_columns:
                    num_of_neibours = 1
                if (self.if_adjacent == False)|(num_of_neibours > 0):
                    actions.append([row, column])

        return actions
        
    def next_state_and_reward(self, player, State, action):
        row, col = action
        NextState = np.copy(State)
        NextState[row, col] = player
        reward = self.reward_after_move(NextState, row, col, player)
        return NextState, reward

    # printing to text file info about test results and particular games (each game in a row)    
    def print_test_to_file(self, filename,num_win_x, num_win_o, num_draws, Games, Rewards):
        f = open(filename,"w")
        number_of_games = len(Games)
        """ for g in range(number_of_games):
            Boards = Games[g]
            f.write("game " + str(g) + ":\n")
            for i in range(len(Boards)):
                f.write(str(Boards[i]) + "\n")
            f.write("\n\n") """
        num_rows, num_col = np.shape(Games[0][0])
        for g in range(number_of_games):
            Boards = Games[g]
            num_of_boards = len(Boards)
            result = " draw"
            if Rewards[g] == 1:
                result = " x win"
            elif Rewards[g] == -1:
                result = " o win"
            f.write("game " + str(g) + result + ":\n")
            for r in range(num_rows):
                row = ""
                for b in range(num_of_boards):
                    A = Boards[b]
                    for c in range(num_col):
                        if A[r,c] == 0:
                            row += "_"
                        elif A[r,c] == 1:
                            row += "x"
                        elif A[r,c] == 2:
                            row += "o"
                    row += "  "
                f.write(row + "\n")
            f.write("\n") 

        print("results after %d games: " % (number_of_games))
        print("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.write("results after %d games: " % (number_of_games))
        f.write("x win = %d, o win = %d, draws = %d" % (num_win_x, num_win_o, num_draws))
        f.close()

    # end of class Tictac_general

class Connect4(Tictac_general):
    def __init__(self):
        super().__init__(6,7,4,False)
    def whoplay_next_states_rewards(self, A):
        number_x = np.sum(A == 1)     
        number_o = np.sum(A == 2)

        next_states = []    # states numbers reachable from current state
        rewards = []

        player = -1
        if number_x == number_o:
            player = 1
        elif number_x == number_o + 1:
            player = 2 

        if player > -1:
            movable_cells = []
            for c in range(self.num_of_columns):
                r = self.num_of_rows-1
                while (r >= 0):
                    if A[r,c] == 0:
                        movable_cells.append([r,c])
                        break
                    r -= 1

            movable_cells_number = len(movable_cells)
                            
            for i in range(movable_cells_number):
                row,column = movable_cells[i]
                
                A[row][column] = player                  # put player symbol (1-cross,2-circle) into free cell
                next_states.append(np.copy(A)) # next state assignment + adding it to list of visited states
                #rewards.append(self.reward(A))
                rewards.append(self.reward_after_move(A,row,column,player))
                A[row][column] = 0

        return player, next_states, rewards

    def actions(self, A, player = 0):
        actions = []
        for c in range(self.num_of_columns):
            r = self.num_of_rows-1
            while (r >= 0):
                if A[r,c] == 0:
                    actions.append([r,c])
                    break
                r -= 1

        return actions


# -----------------------------------------------------------
# General functions

# x - any vector
# returns probability distribution  
def softmax(x):
    max_val = np.max(x)
    return(np.exp(x-max_val)/np.exp(x-max_val).sum())

#  choose action from probability distribution
#  the sum of probabilities must be equal 1
#  T - temperature (1 - neutral, 0 - choosing the best option, high - equal probability of each option)
def choose_action(distribution):
    cumdistr = np.cumsum(distribution)
    x = np.random.random()
    index = len(distribution)-1
    for i in range(len(cumdistr)):
        if x < cumdistr[i]:
            index = i
            break
    #print("cum = " + str(cumdistr) + " x = " + str(x) + " indeks = " + str(index) +"\n")
    return index

# random strategy for tic-tac-toe:
def random_strategy(strategy = {}):
    return strategy_nondeterm(strategy, 1)
    
# flattening the actions probability distribution 
# randomness form 0 to 1. If == 0 -> without flattening, If == 1 -> full random distribution 
# with equal probabilities  
def strategy_nondeterm(strategy, randomness):
    strategy_nond = strategy.copy()
    for key,distrib in strategy_nond.items():
        number_of_actions = len(distrib)
        distrib_new = np.zeros([len(distrib)], dtype = float)
        number_of_actions
        if (number_of_actions > 1):
            for i in range(number_of_actions):
                distrib_new[i] = distrib[i]*(1-randomness) + randomness/number_of_actions
        strategy_nond[key] = distrib_new
    return strategy_nond
            
# strategy to text file
def save_strategy(filename, strategy):
    f = open(filename,"w")
    for key, distrib in strategy.items():
        f.write(str(key) + "  " + str(distrib) + "\n")
    f.close()

# string from np.array2string() or simply str() back to array:
def string_to_2Darray(s):
    snum = ""
    li = []  
    row = []
    level = -1
    for i in range(len(s)):
        if ((s[i] >= '0')&(s[i] <= '9'))|(s[i]=='.')|(s[i]=='-'):
            snum += s[i]
        else:
            if len(snum) > 0:
                number = int(snum)
                if level == 1:
                    row.append(number)                   
            snum = ""
            if s[i] == '[':
                level += 1
            elif s[i] == ']':
                if level == 1:
                    li.append(row)
                    #print("row = " + str(row) + " li = " + str(li))
                    row = []
                level -= 1
    # converting into np.array:
    num_of_rows = len(li)
    num_of_columns = 0
    if num_of_rows > 0:
        num_of_columns = len(li[0])
    #print("num_of_rows = " + str(num_of_rows) + " num_of_columns = " + str(num_of_columns))
    if num_of_columns > 0:
        A = np.zeros([num_of_rows, num_of_columns], dtype=int)
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                A[i,j] = li[i][j]
        return A
    else:
        return []

    # end of general functions
    # --------------------------------------------------------------------