'''
 AUTHOR: Mugdha Ektare
 R Number: R11841641
 FILENAME: phase1.py
 SPECIFICATION: Our problem statement of the game has been divided in 2 phases.
                This code deals with the phase 1 of the problem statement.
                Phase 1 scenario is as follows:
                    At the start of the game, the agent has to choose one of the 4 suits that is Spades, Hearts, Clubs and Diamond.
                    This code deals with training the agent for choosing the optimal suit for a random board.
                    The approach used in this is Q-learning.
 FOR: CS 5392 Reinforcement Learning Section 001
'''

import numpy
import random

# Global Variables

# Winning Positions has all the positions in which if the cards are positioned, the agent will be considered as won.
# There are total 10 positions if we do not consider the direction of the sequence.
winning_positions = [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                         [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                         [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
                         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]],
                         [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                         [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                         [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                         [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                         [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
                         ]

positions_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # Position indices of all the positions on the keyboard.
board = [[12, 4, 3, 7], [3, 14, 11, 15], [6, 8, 1, 9], [2, 10, 5, 0]] # Some temporary board containing random positions of the cards.
suits = [0, 1, 2, 3] # Suits are Spades - 0, Hearts - 1, Clubs - 2, Diamonds - 3
# 0 to 3 - Spades
# 4 to 7 - Hearts
# 8 to 11 - Clubs
# 12 to 15 - Diamonds
suit_to_cards_mapping = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] # Each suit is mapped to the cards menitioned above.
suit_names = ["Spades", "Hearts", "Clubs", "Diamonds"] # Names of all the suits.

'''
    NAME: epsilon_greedy
    PARAMETERS: q_table - List; The Q table for the Q learning. Updated one for each iteration.
                epsilon - Float value; Range = [0, 1]; Needed for epsilon greedy policy to pick a random action or the max action.
    PURPOSE: This function determines which action to be picked according to the random variable policy which has the value between [0, 1], 
             the epsilon value and q_table that is passed.
    PRECONDITION: This function is called by the main function. Before this function is called, the global and local variables are set.
                  The main function has been called. 
    POSTCONDITION:  This function returns the action to be taken according to the epsilon greedy policy.
'''

def epsilon_greedy(q_table, epsilon=0.5):

    # Epsilon greedy policy to pick the action according to the random function and the epsilon...
    policy = random.uniform(0, 1) # Random number generator in the range of 0 to 1...
    if policy < epsilon:
        return random.randint(0, 3) # Pick random action if epsilon > policy
    else:
        return numpy.argmax(q_table) # Else pick the max action from the q_table

'''
    NAME: calculate_sparse
    PARAMETERS: card_ID - These are the cards chosen by the agent or by the opponent.
                board - Nested List; This is the current positions of the cards on the board.
                agent_ID - Binary variable; This indicates if agent is making the choice or if the opponent is making the choice.
    PURPOSE: This will calculate a sparse matrix according to the positions of the cards chosen by teh agent. 
             If the agent has already chosen and we are considering the suit chosen by the opponent, then it indicates the values of the
             agent with 100 and the values of the opponent as 1s. 
    PRECONDITION: This function is called by the new_reward. Before this function is called, the global and local variables are set.
                  The main function has been called and the q_function and new_reward function has also been called. 
    POSTCONDITION:  This function returns the sparse matrix according to the positions of the cards chosen by the opponent and the agent. 
'''

def calculate_sparse(card_ID, board, agent_ID):

    # This function will calculate the sparse matrix as per the current position of the chosen cards...
    current_sparse = [] # Empty matrix...
    flag = 0 # Flag to check if the chosen cards contain 0...
    if 0 in card_ID:
        flag = 1

    # For each element in the board, put 1 in the sparse matrix in it's place...
    for i in range(0, 4):
        row = [0, 0, 0, 0]
        for j in card_ID:
            if j in board[i]:
                board_row = board[i]
                row[board_row.index(j)] = 1
        current_sparse.append(row)
    # print("Sparse: ", current_sparse)
    # print("Board: ", board)
    # If the agent has already played, then put the cards chosen by the agent as 100 in the sparse matrix...
    if agent_ID == 1:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] < 0:
                    current_sparse[i][j] = 100
    # print("Sparse after 100: ", current_sparse)

    # Returning the sparse matrix...
    return current_sparse

'''
    NAME: new_reward
    PARAMETERS: agent_ID - Binary variable; indicates if agent is making the choice or if the opponent is making the choice.
                state - Nested List; Contains the current status of the board.
                action - Integer; Contains the action chosen (suit chosen) by the agent.
                card_IDs - List; The cards that agent has picked as its suit. 
                card_IDs_prev - List; This is an optional parameter, used in the case that agent has picked the suit and now it is the 
                                opponent's turn to pick the suit. It is a list containing the cards agent has already chosen.
    PURPOSE: If it is the agent's turn, it calculates the reward the agent would get after choosing the suit according to the suit's 
             position to the nearest winning position. 
             If the agent has already chosen, and it is the opponent's turn it calculates the maximum reward the opponent could get after
             the agent has chosen a suit depending on the winning position, agent's cards position and opponent's cards position.
             This is being calculated using the sparse matrix and the optimal winning position. 
    PRECONDITION: This function is called by the q_function. Before this function is called, the global and local variables are set.
                  The main function has been called and the q_function has also been called. 
    POSTCONDITION:  This function returns the reward an agent might get depending on whose turn it is. It also returns the latest state 
                    of the board after the agent and/or the opponent has chosen the suit.
'''

def new_reward(agent_ID, state, action, card_IDs, card_IDs_prev=[-1, -1, -1, -1]):

    # Execute the following code if it is the agent's turn to pick the action...
    if agent_ID == 0:
        # For the suit chosen by the agent, convert those card IDs to negative numbers for identification...
        for i in card_IDs:
            for j in range(len(state)):
                if i in state[j]:
                    index = state[j].index(i)
                    state[j][index] = state[j][index] * -1

        # Pass this newly modified board or state to the calculate_sparse function for calculating the sparse matrix for the chosen cards...
        action_sparse = calculate_sparse(card_IDs, state, agent_ID)
        # print("Agent Sparse: ", action_sparse)

        # Here we are comparing the generated sparse matrix with the possible winning positions defined in the global variables section...
        # The sparse matrix is compared with each winning position matrix and the winning matrix which is the most close to the
        # current sparse matrix is chosen for evaluation...
        # For evaluation, we are seeing how many cards are out of place if we compare the current sparse matrix with the chosen winning
        # position matrix...

        # resultant has the matrices generated after overlapping the winning positions with the current sparse matrix
        resultant = []
        # nonzero_count has the count of cards that are out of place in the given resultant...
        nonzero_count = []

        # For each winning position...
        for i in winning_positions:
            resultant_1 = []
            count = 0
            # For each row in the winning position...
            for j in range(len(i)):
                resultant_2 = []
                # For each item in that row...
                for k in range(len(i[j])):
                    # set the resultant as the subtraction of winning position element and the sparse matrix element...
                    resultant_2.append(i[j][k] - action_sparse[j][k])
                resultant_1.append(resultant_2)
                # Increase the count variable if a non-zero value is encountered...
                count += numpy.count_nonzero(resultant_2)
            resultant.append(resultant_1)
            nonzero_count.append(count)
            # print("Resultant_1: ", resultant_1)
        # print("These are non_zero counts for the agent's choice: ", nonzero_count)
        best_winning_position = winning_positions[nonzero_count.index(min(nonzero_count))]

        # Reward returned is the reciprocal of the total number of cards that are out of place, along with the state of the board...
        if min(nonzero_count) != 0:
            return 1 / min(nonzero_count), state
        else:
            return 1, state

    else:
        # Execute the following code if it is the agent's turn to pick the action...
        card_IDs = suit_to_cards_mapping[action]
        # print("State going from the opponent: ", state)

        # For the suit chosen by the opponent, convert those card IDs to negative numbers for identification...
        for i in card_IDs_prev:
            for j in range(len(state)):
                if i in state[j]:
                    index = state[j].index(i)
                    state[j][index] = state[j][index] * -1

        # Send this state to the calculate_sparse function to calculate the sparse matrix...
        action_sparse = calculate_sparse(card_IDs, state, agent_ID)
        # print("Board: ", state)
        # print("Our Cards: ", card_IDs)
        # print("Action Sparse: ", action_sparse)

        # Calculate the resultant exactly as mentioned before in the if condition above...
        # Only this one will have the actions chosen by the agent will have values as 100 and the ones chosen by the opponent will have
        # value 1
        resultant = []
        obstacle_count = []
        for i in winning_positions:
            resultant_1 = []
            count = 0
            for j in range(len(i)):
                resultant_2 = []
                for k in range(len(i[j])):
                    resultant_2.append(i[j][k] - action_sparse[j][k])
                resultant_1.append(resultant_2)
                for k in resultant_2:
                    if k == -1 or k == -99:
                        count += 1
            resultant.append(resultant_1)
            obstacle_count.append(count)
            # print("Resultant_1: ", resultant_1)
        # print("Obstacle Count: ", obstacle_count)
        # print("Final resultant: ", resultant[obstacle_count.index(min(obstacle_count))])
        best_winning_position = winning_positions[obstacle_count.index(min(obstacle_count))]

        # Reward returned is the reciprocal of the total number of cards that are out of place, along with the state of the board...
        if min(obstacle_count) != 0:
            return 1 / min(obstacle_count), state
        else:
            return 1, state

'''
    NAME: q_function
    PARAMETERS: agent_ID - Binary variable; indicates if agent is making the choice or if the opponent is making the choice.
                state - Nested List; Contains the current status of the board.
                action - Integer; Contains the action chosen (suit chosen) by the agent.
                q_table - List; This is the list of q-values in the q learning process.
                gamma - Float; This is the discount factor required in the Q learning.
                cards_prev = List; This is the optional parameter. Contains the cards chosen by the agent to calculate the reward agent 
                             may get after picking that suit. Default is -1, -1, -1, -1.
    PURPOSE: This function calculates the q values for each iteration in the Q learning process and updates the Q table accordingly.
    PRECONDITION: This function is called in the main function, after the global and local variables are initialized and after the
                  random board has been generated as a test case.
    POSTCONDITION:  This function returns the value of the agent's reward - it's opponent's maximum reward. After returning this, the 
                    control goes to the main function and the main function is able to update the q table and print the optimal action the
                    agent would take after specified number of iterations.
'''
def q_function(agent_ID, state, action, q_table, gamma, cards_prev=[-1, -1, -1, -1]):
    # Execute the following code if the agent is picking the suit...
    if agent_ID == 0:
        # Calculating the IDs of the cards that agent has picked...
        agent_suit_cards = [action * 4, action * 4 + 1, action * 4 + 2, action * 4 + 3]

        # Temporary variables that contain the state of the board for consistency of function parameters...
        state_revived = state
        state_copy_1 = state_revived
        state_copy_2 = state_revived
        state_copy_3 = state_revived

        # After the agent has picked one suit, the opponent has 3 choices...
        # Computing these 3 choices here...
        next_action_set = []
        for i in range(0, 4):
            if i != action:
                next_action_set.append(i)
        # q_table[action] = -1

        # Computing the reward for the suit agent has picked...
        value_opponent = []
        reward, state = new_reward(agent_ID, state_revived, action, agent_suit_cards)
        # print("State: ", state)

        agent_ID = 1
        # Setting agent ID as 1 which is agent has picked the suit, but now we will consider what happens after the agent has picked this
        # suit... The opponent will have 3 choices... Computing the reward for each possible choice the opponent has, and appending those
        # to the list called value_opponent declared above...
        # The value appended in the list is probability * (reward - gamma * reward of the next action)
        reward_previous, reward_index = q_function(agent_ID, state_copy_1, next_action_set[0], q_table, gamma,
                                                   agent_suit_cards)
        value_opponent.append(0.33*(reward - gamma * reward_previous))
        reward_previous, reward_index = q_function(agent_ID, state_copy_2, next_action_set[1], q_table, gamma,
                                                   agent_suit_cards)
        value_opponent.append(0.33*(reward - gamma * reward_previous))
        reward_previous, reward_index = q_function(agent_ID, state_copy_3, next_action_set[2], q_table, gamma,
                                                   agent_suit_cards)
        value_opponent.append(0.33*(reward - gamma * reward_previous))

        # In the Q learning equations mentioned above, subtracting the rewards gained from the opponent since, the opponents rewards should
        # minimize the chances of the agent to win...
        # print("Max_Q: ", max(value_opponent), "Max_Q_index: ", numpy.argmax(value_opponent))
        # print("This is what your opponent would choose: ", numpy.argmax(value_opponent))

        # Returning the maximum value the agent can get considering the worst action the opponent could have chosen, along with the action
        # of the agent...
        return max(value_opponent), action

    # Execute this code if the agent has already chosen a suit and now we want to calculate the reward for each suit the opponent could
    # pick after this...
    else:
        # print("State: ", state)

        # Calculating the card_IDs for the action the opponent could pick...
        current_suit = [action * 4, action * 4 + 1, action * 4 + 2, action * 4 + 3]

        # Calling the reward function and calculating the reward for the action opponent could pick after an action picked by the agent...
        reward = new_reward(agent_ID, state, action, current_suit, cards_prev)

        # Returning the reward...
        return reward

'''
    NAME: main
    PARAMETERS: None
    PURPOSE: Launching the function calls and executing the flow of the code.
    PRECONDITION: None
    POSTCONDITION:  After executing the whole code in the main function, the agent will pick a suit according to its expertise till now.
                    As this is executing the Q-learning algorithm, it will have updated Q-table values from time to time.
                    It displays the Suit agent has picked.
'''
def main():
    # Running this agent for 10 iterations. Each iteration works on 10 randomly generated boards.
    for i in range(10):
        random.shuffle(positions_array) # Randomly shuffles the 16 cards.
        board = [] # Empty board
        j = 0
        while j < 13:
            board.append([positions_array[j], positions_array[j + 1], positions_array[j + 2], positions_array[j + 3]])
            j += 4
        # In the above code, the randomly shuffled cards are being placed on the board.
        # Ex. If the shuffled cards are like this: [4, 3, 7, 5, 15, 8, 13, 9, 10, 2, 1, 6, 11, 0, 14, 12]
        # then the board will look like below...
        # [4, 3, 7, 5]
        # [15, 8, 13, 9]
        # [10, 2, 1, 6]
        # [11, 0, 14, 12]
        # board = [[4, 3, 7, 5], [15, 8, 13, 9], [10, 2, 1, 6], [11, 0, 14, 12]]
        print()
        print("This is the new board: ") # Printing the board...
        for row in board:
            print(row)
        q_table = [0, 0, 0, 0] # Initializing the q_table for 4 actions...
        policy_converged = False # Boolean variable to check if the policy has converged or not...
        previous_optimal_action = 0 # Default optimal action of the agent set to select Spades...
        gamma = 0.5 # Discount factor set to 0.5 (Can be changed for training the model...)
        count = 0 # Counting variable
        updates_count = 0 # Counting variable
        while not policy_converged: # Repeat this process while the policy is not converged...
            action = epsilon_greedy(q_table, 0.5) # Select an action according to the epsilon greedy policy...
            agent_ID = 0 # This is the agent playing...
            state = board
            # In the code below we are indicating the cards picked by the agent by negative numbers...
            for k in state:
                for j in range(len(k)):
                    if k[j] < 0:
                        k[j] = k[j] * -1
            # print("State in the main function for loop: ", state)

            # Calling the q_function function for Q learning process...
            q_value, q_table[action] = q_function(agent_ID, state, action, q_table, gamma)
            # print(q_table)
            # print(type(q_table))

            # After the q_function returns a value, the optimal action is chosen according to the maximum value from the q_table
            optimal_action = q_table[q_table.index(max(q_table))]

            # Checking if since previous 5 iterations, is the action same?
            if optimal_action == previous_optimal_action:
                count += 1
            else:
                count = 0
            # print("Updated Q table: ", q_table)
            previous_optimal_action = optimal_action

            # If since past 5 iterations, the action is to pick the same suit on the same board, consider the policy to be converged!
            if count == 5:
                policy_converged = True
            updates_count += 1 # Count how many times do we need to update the q table...

        # Set the predicted action as the action with the maximum value from the q_table
        predicted_action = numpy.argmax(q_table)
        print("The optimal action would be to pick", suit_names[predicted_action]) # Print the picked action...
        print("Card numbers: ", suit_to_cards_mapping[predicted_action])
        # print("Number of Q-table updates: ", updates_count)

# Launcher of the main function.
if __name__=="__main__":
    main()
