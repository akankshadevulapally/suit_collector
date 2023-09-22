"""
AUTHOR:         James, Tishya
FILENAME:       game.py
SPECIFICATION:  Game File for Training in Phase II, Iteration 2.
                This file contains the game class which is an gymnasium environment for the Suit Collector.
                For every action performed, The Game in response makes a action from 'Iron' Neural Network Agent or 
                a valid random action for the opponent.
FOR:            CS 5392 Reinforcement Learning Section 001
"""

# Standard Imports.
import random as rd;
import numpy as np;
import gymnasium as gym;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os.path

from gymnasium import Env, spaces 

"""
NAME:       game
PURPOSE:    This class is used to create a gymnasium environment for the Suit Collector game.
"""
# Suite Collector Game Class File Inherits from Gymnasium Environment.
class game(gym.Env):

    """
    NAME:           init
    PARAMETERS:     self
    PURPOSE:        This constructor sets up the Game Object with all necessary 
                    properties and initializes the board.
    PRECONDITION:   This function should be called to initialize the Game object 
                    before starting any game.
    POSTCONDITION:  Game object is initialized.
    """
    def __init__(self, model_path):
        """This constructor sets up the Game Object.
            Arguments:
                model_path: refers to the path to the weights of Agent Iron.
        """
        # pieces are represented in phase II format where 
        # positive numbers are of the Agents, negative numbers are the opponent.
        # zero's are indifferent card.
        self.pieces = np.array([1,2,3,4,0,0,0,0,0,0,0,0,-1,-2,-3,-4])

        # board is the 4x4 Grid.
        self.board = np.copy(self.pieces)

        # Initializing actions Map.
        # actions[i] = [x,y]
        # action i swaps x , y Position on the 4x4 Grid as shown below.
        # Positions:
        #   0  1  2  3 
        #   4  5  6  7
        #   8  9  10 11
        #   12 13 14 15
        #populating actions
        # swap x and y  
        self.actions = np.full([16*8,2], -1)
        #actionsXY[x,y] = i , Swapping X and Y position if action is i
        self.actionsXY = np.full([16,16],-1)
        # Total number of valid actions.
        self.actionsCount = 0

        # The suites in the Game in Phase II.
        self.suites = np.array([1,-1])
        
        # populate actions data.
        self.generateActions()

        
        # Zero computer goes first
        # 1 is Agent/Player goes first
        #self.turn = -1

        # Max Turns each game before its a draw
        # Changed to 50 to decrease the noise and too many draw/repeating moves in replay buffer.
        self.maxTurnsEachGame = 50
        self.action_space = spaces.Discrete(self.actionsCount)
        self.observation_space = spaces.Box(low=-4, high=4, shape =(16,), dtype=np.float16)

        # create Assistant model/ Iron
        if os.path.exists(model_path):
            self.model = self.create_q_model(self.observation_space.shape , self.action_space.n)
            self.model.load_weights(model_path)
            print('Assistance model is all set.')
        else:
            print('Path ', model_path, 'does not exist.')
            return
        
        # Initializes the board and everything else.
        self.startGame()

    """
    NAME:           generateActions
    PARAMETERS:     self
    PURPOSE:        This function generates the actions and action count.
    PRECONDITION:   This function should be called while initializing the Game object.
    POSTCONDITION:  The action space is populated with valid actions
    """
    def generateActions(self):
        # For each cell populate the unique valid actions.
        for i in range(0,4):
            for j in range(0,4):
                #current position
                a,b = i,j

                # possible swappable positions
                # Top Left
                c,d = i-1,j-1
                self.processAction(a,b,c,d)

                # Top
                c,d = i,j-1
                self.processAction(a,b,c,d)

                # Top Right
                c,d = i+1,j-1
                self.processAction(a,b,c,d)

                # Right
                c,d = i+1,j
                self.processAction(a,b,c,d)

                # Bottom Right
                c,d = i+1,j+1
                self.processAction(a,b,c,d)

                # Bottom 
                c,d = i,j+1
                self.processAction(a,b,c,d)

                # Bottom Left 
                c,d = i-1,j+1
                self.processAction(a,b,c,d)

                # Left 
                c,d = i-1,j-1
                self.processAction(a,b,c,d)
    
    """
    NAME:           startGame
    PARAMETERS:     self
    PURPOSE:        The function initializes a new board for the game.
    PRECONDITION:   This function should be called after initializing the Game object.
    POSTCONDITION:  The function initializes a new random board by shuffling the pieces 
                    array and copying it to the board array. It tracks the positions of 
                    Aces in the new board and updates the Aces property. The time 
                    property is also initialized to 0.
    """
    def startGame(self):
        # Initialize a random board
        self.board = np.copy(self.pieces)
        np.random.shuffle(self.board)

        # Track Aces in the board.
        self.aces = np.full([3],-1)
        for i in range(0,len(self.board)):
            x = self.board[i] 
            if self.board[i] == 1 or self.board[i] == -1:
                self.aces[self.board[i]] = i
        #print(self.aces)

        # time.
        self.time = 0
   
    """
    NAME:           isValid
    PARAMETERS:     Self; x - an integer representing an X or Y position on the grid
    PURPOSE:        Checks if a coordinate is valid or not within the 4x4 grid.
    PRECONDITION:   The input parameter x must be an integer.
    POSTCONDITION:  Returns True if the input x is within the range of 0 to 3, 
                    False otherwise. No variables are changed. The function only 
                    returns a boolean value.
    """
    def isValid(self, x):
        if x >=0 and x < 4:
            return True
        return False

    """
    NAME:           normalizeBoard
    PARAMETERS:     self
    PURPOSE:        The function normalizes the data on the board to values between [-1,1]
    PRECONDITION:   The function can be called after the board has been initialized 
                    with values between 0 and 15.
    POSTCONDITION:  The function returns a normalized board where all values have been 
                    divided by 4 to be within the range of [-1,1].
    """
    def normalizeBoard(self):
        return (self.board / 4)

    """
    NAME:           normalizeBoardForAssistanceModel
    PARAMETERS:     self
    PURPOSE:        The function normalizes the values on the game board to a range of [-1,1] to make it suitable 
                    for processing by the Agent Iron.
    PRECONDITION:   The function must be called by an instance of the class containing this method.
    POSTCONDITION:  The values on the game board are divided by -4 to normalize them to a range of [-1,1]. The normalized 
                    board is then returned as output.
    """
    def normalizeBoardForAssistanceModel(self):
        return (self.board / -4)

    """
    NAME:           populateAction
    PARAMETERS:     self; x, y - Co ordinates of a position on the 4x4 Grid.
    PURPOSE:        The function populates two action maps with a given actionXY map, representing the position on a 4x4 grid.
    PRECONDITION:   The function must be called by an instance of the class containing this method. The x and y arguments must 
                    be integers representing valid coordinates on a 4x4 grid.
    POSTCONDITION:  The actionXY map is updated with the given x,y coordinates, and the actions map is updated with a new entry 
                    at the index of actionsCount, containing the x and y coordinates. The actionsCount variable is also incremented by 1.
    """
    def populateAction(self,x,y):
        """ Populate both action Maps with an actionXY Map. 
        Arguments:
        x,y     -> Co ordinates of a position on the 4x4 Grid.
        """
        self.actionsXY[x,y] = self.actionsCount
        self.actions[self.actionsCount, 0] = x
        self.actions[self.actionsCount, 1] = y
        self.actionsCount += 1
    
    """
    NAME:           isValidAction
    PARAMETERS:     self, action
    PURPOSE:        The function checks if a given action is valid for the current game board.
    PRECONDITION:   The function must be called by an instance of the class containing this method, and an action must be provided as an argument.
    POSTCONDITION:  The function returns True if the action is valid for the current game board, False otherwise. An action is 
                    considered valid if it does not swap opponent's cards and if the cards being swapped are not already flipped (have a negative value on the board). 
    """
    def isValidAction(self, action):
        if(action >=0 and action <  42):
                #player makes an action
                #check if the action is valid or not
                # an action is valid if it does not swap opponents cards.
                card1Pos = self.actions[action][0]
                card2Pos = self.actions[action][1]

                card1 = self.board[card1Pos]
                card2 = self.board[card2Pos]

                if (
                    card1 < 0 or \
                    card2 < 0
                    ):
                    return False
        else:
            return False 
        return True

    """
    NAME:           processAction
    PARAMETERS:     self, a, b, c, d
    PURPOSE:        The function processes an action by checking if the destination coordinates are valid and if the action already 
                    exists in the actions map. It also populates the actions map with a new action if it doesn't already exist.
    PRECONDITION:   The function must be called by an instance of the class containing this method. The a, b, c, and d arguments must 
                    be integers representing valid coordinates on a 4x4 grid.
    POSTCONDITION:  If the destination coordinates are valid and the action does not already exist in the actions map, then a new 
                    entry is added to the actions map using the populateAction method. If the action already exists in the actions map, 
                    then no changes are made to the maps. 
    """
    def processAction(self,a,b,c,d):
        if self.isValid(c) and self.isValid(d):
            x, y = 4 * a + b, 4 * c + d
            if( x > y):
                x,y = y,x
            if(self.actionsXY[x,y] == -1):
                self.populateAction(x,y)

    """
    NAME:           reset
    PARAMETERS:     self
    PURPOSE:        The function resets the game board to a fresh new game state.
    PRECONDITION:   The function must be called by an instance of the class containing this method.
    POSTCONDITION:  The game board is reset to a fresh new game state using the startGame method. 
                    The board is then normalized using the normalizeBoard method and returned as output. 
    """
    def reset(self):
        self.startGame()
        return self.normalizeBoard()

    """
    NAME:           step
    PARAMETERS:     self, action
    PURPOSE:        This function simulates one step of the game, given an action from the User/Agent, 
                    responds with a random valid action from the opponent, and returns the next state, 
                    reward, done?, and additional information.
    PRECONDITION:   The function should be called after initializing the game environment with all 
                    necessary variables and parameters. The parameter 'action' should be an integer 
                    between 0 and 41, representing the action of swapping X and Y position.
    POSTCONDITION:  The function updates the game board with the performed actions and returns the next 
                    state, reward, done?, and additional information. The 'reward' and 'done' variables are 
                    updated based on the outcome of the game, and 'info' is a dictionary containing additional 
                    information about the current state of the game.
    """
    def step(self, action):
        # Additional Information        
        info = {}
        info['random_action'] = -1
        # reward
        reward = 0
        # done?
        done = False

        # If more than 'maxTurnsEachGame' turns are played
        # game is considered to be draw
        self.time+=1
        if(self.time >= self.maxTurnsEachGame):
            done = True
  
  
        if(action >=0 and action <  42):
            #player makes an action
            #check if the action is valid or not
            # an action is valid if it does not swap opponents cards.
            card1Pos = self.actions[action][0]
            card2Pos = self.actions[action][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            if ( card1 < 0 or card2 < 0):
                return self.normalizeBoard(), -1, True, info

            # else action is valid
            # perform the action if valid
            self.board[card1Pos] = card2
            self.board[card2Pos] = card1

            # Track aces if required.
            if self.board[card1Pos] == 1:
                self.aces[1] = card1Pos

            if self.board[card2Pos] == 1:
                self.aces[1] = card2Pos

            # Check if you have won.
            # set reward that we need to return.
            won = self.checkIfSuiteWon(1)
            if won:
                return self.normalizeBoard(), 1, True, info
            # if not won, but played a move which didn't change the game
            # board, set a reward of -0.05. Do not Encourage Draws!
            if self.board[card1Pos] == 0 and self.board[card2Pos] == 0:
                reward = -0.05
        else:
            print("Unknown Error, action was ", action)
            info['error'] = 'Unknown Error!!! action was , '+ str(action);
            self.render()
            return self.normalizeBoard(), 0, True, info

        info['your_action'] = action

        # The Game makes a move in return.
        # takes the action from Iron assistance agent,
        # if its not legal plays a random move. 

        # Predict action Q-values
        # From environment state
        state_tensor = tf.convert_to_tensor(self.normalizeBoardForAssistanceModel())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        # Take best action
        myaction = tf.argmax(action_probs[0]).numpy() 
        #myaction = 0
        info['assisted_action'] = myaction
        card1Pos = None
        card2Pos = None
        while True:
            
            card1Pos = self.actions[myaction][0]
            card2Pos = self.actions[myaction][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            if ( card1 > 0 or  card2 > 0):
                myaction = rd.randint(0,41)
                continue
            else:
                break
            
        # Perform the action
        self.board[card1Pos] = card2
        self.board[card2Pos] = card1
        info['random_action'] = myaction

        # Track aces if required.
        if self.board[card1Pos] == -1:
            self.aces[-1] = card1Pos

        if self.board[card2Pos] == -1:
            self.aces[-1] = card2Pos

        # Check if Agent Iron won.
        # set reward accordingly and return.
        won = self.checkIfSuiteWon(-1)
        if won:
            return self.normalizeBoard(), -1, True, info

        # finally
        return self.normalizeBoard(), reward, done, info 
        
        
    def checkIfSuiteWon(self, suite):
        """ Check if a Suite won the game or not. """
        # get position of ace in suite
        acePos = self.aces[suite]
        x = [0,-1,-2,-3]
        if(suite == 1):
            x = [0,1,2,3]
            

        # Diagonals
        if acePos == 0:
            if self.board[5]  == self.board[0]+x[1] and \
               self.board[10] == self.board[0]+x[2] and \
               self.board[15] == self.board[0]+x[3] :
                return True
        if acePos == 3:
            if self.board[6]  == self.board[3]+x[1] and \
               self.board[9]  == self.board[3]+x[2] and \
               self.board[12] == self.board[3]+x[3] :
                return True
        if acePos == 12:
            if self.board[9]  == self.board[12]+x[1] and \
               self.board[6]  == self.board[12]+x[2] and \
               self.board[3] == self.board[12]+x[3] :
                return True
        if acePos == 15:
            if self.board[10]  == self.board[15]+x[1] and \
               self.board[5]  == self.board[15]+x[2] and \
               self.board[0] == self.board[15]+x[3] :
                return True

        acePosX , acePosY = int(acePos % 4) , int(acePos / 4) 
        # Top Row
        if acePosY == 0:
            if self.board[acePos + 4] == self.board[acePos] + x[1] and \
               self.board[acePos + 8] == self.board[acePos] + x[2] and \
               self.board[acePos + 12] == self.board[acePos] + x[3]  :
                return True

        # Bottom Row
        if acePosY == 3:
            if self.board[acePos - 4] == self.board[acePos] + x[1] and \
               self.board[acePos - 8] == self.board[acePos] + x[2] and \
               self.board[acePos - 12] == self.board[acePos] + x[3]  :
                return True

        # Left Column
        if acePosX == 0:
            if self.board[acePos + 1] == self.board[acePos] + x[1] and \
               self.board[acePos + 2] == self.board[acePos] + x[2] and \
               self.board[acePos + 3] == self.board[acePos] + x[3]  :
                return True

        # Right Column
        if acePosX == 3:
            if self.board[acePos - 1] == self.board[acePos] + x[1] and \
               self.board[acePos - 2] == self.board[acePos] + x[2] and \
               self.board[acePos - 3] == self.board[acePos] + x[3]  :
                return True

        return False
        
    """
    NAME:           TestcheckIfSuiteWon
    PARAMETERS:     board, list of integers; aces, dictionary; suite, integer
    PURPOSE:        Helper method to test if a suite won the game or not by calling the checkIfSuiteWon function
    PRECONDITION:   The board should be a list of integers with length 16, aces should be a dictionary with 
                    keys 1-4 and values in the range 0-15, and suite should be an integer in the range 1-4. The 
                    checkIfSuiteWon function should also be defined.
    POSTCONDITION:  Prints True if the suite won the game, False otherwise. The board and aces instance variables of 
                    the class may be changed, but are reset after the function call.
    """
    def TestcheckIfSuiteWon(self, board, aces, suite):
        """ Helper method to test if a suite won the game or not """
        self.board = board
        self.aces = aces
        print(self.checkIfSuiteWon(suite))
    
    """
    NAME:           render
    PARAMETERS:     self
    PURPOSE:        The function prints the current game state to the console
    PRECONDITION:   The game state should be initialized and valid
    POSTCONDITION:  The game state is printed to the console in a formatted manner
    """
    def render(self):
        """ Renders the game on to the Standard Output console """
        print()
        print("Board: ")
        for i in range(0,4):
            for j in range(0,4):
                print("{:>4}".format(self.board[(i*4)+j]), end="")
            print()
        print("\nSuite: Agent =" , 1, " , ME = ", -1 )
        print("aces position = ", self.aces)
        #print("board = " , self.board)
        #print("normalize board = ", self.normalizeBoard())
        print("time = ", self.time)

    # Network for Agent Iron.
    """
    NAME:           create_q_model
    PARAMETERS:     state_shape - tuple representing the shape of the input state to the neural network
                    total_actions - integer representing the total number of actions available to the agent
    PURPOSE:        Creates a deep neural network model for the Q-learning algorithm used by Agent Iron to select actions.
    PRECONDITION:   state_shape must be a tuple representing the shape of the input state to the neural network.
                    total_actions must be an integer representing the total number of actions available to the agent.
    POSTCONDITION:  Returns a Keras model object representing the Q-network with input shape state_shape and output shape (total_actions,).
                    The model architecture consists of an input layer, two hidden layers of 40 neurons each with ReLU activation, and an output 
                    layer of total_actions neurons with linear activation.
    """
    def create_q_model(self, state_shape, total_actions):
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(40, activation="relu")(inputs)
        layer2 = layers.Dense(40, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        return keras.Model(inputs=inputs, outputs=action)

"""
NAME:           main
PARAMETERS:     None
PURPOSE:        The function serves as the main entry point of the program and runs the game loop where 
                the user interacts with the game and receives feedback
POSTCONDITION:  The function runs the game loop and allows the user to interact with the game by entering an 
                action, rendering the game, and displaying the reward and computer's action. The loop continues 
                until the game is finished or the user chooses to exit.
"""
def main():
    env = game('./assistance_model/model.h5')
    print()
    #env.render()
    print('Number of states: {}'.format(env.observation_space))
    print('Number of actions: {}'.format(env.action_space))   
    #env.TestcheckIfSuiteWon(np.array([0,-4,2,0, 0,-2,1,0, -3,4,0,0, 3,0,-1,0]), [-1 ,6 ,2], -1)
    #env.TestcheckIfSuiteWon(np.array([0,-4,2,0, 0,-2,1,0, -3,4,0,0, 3,0,-1,0]), [-1 ,6 ,2], 1)
    for i in range(100):
        env.reset()
        done = False
        reward = 0
        action = 0
        while done != True:
            env.render()
            print("Enter action: ",end="")
            x = int(input())
            board, reward , done , info = env.step(x)
            print("reward : " , reward, "Your action : ", action , "Computer action : ", info["random_action"])
            print('---------------------------------------------------')
        


if __name__ == "__main__":
    main()

"""
#   Position:
#   0  1  2  3 
#   4  5  6  7
#   8  9  10 11
#   12 13 14 15

#   action [X,Y] // Swap position X and Y, see above
#   0 [0 4]	        #   21 [ 6 10]		#   41 [14 15]
#   1 [0 5]		    #   22 [ 6 11]		#   42 Pick Suite 1
#   2 [0 1]		    #   23 [6 7]		#   43 Pick Suite 2
#   3 [1 4]		    #   24 [ 7 10]		#   44 Pick Suite 3
#   4 [1 5]		    #   25 [ 7 11]		#   45 Pick Suite 4
#   5 [1 6]		    #   26 [ 8 12]		
#   6 [1 2]		    #   27 [ 8 13]		
#   7 [2 5]		    #   28 [8 9]		
#   8 [2 6]		    #   29 [ 9 12]		
#   9 [2 7]		    #   30 [ 9 13]		
#   10 [2 3]		#   31 [ 9 14]		
#   11 [3 6]		#   32 [ 9 10]		
#   12 [3 7]		#   33 [10 13]		
#   13 [4 8]		#   34 [10 14]		
#   14 [4 9]		#   35 [10 15]		
#   15 [4 5]		#   36 [10 11]		
#   16 [5 8]		#   37 [11 14]		
#   17 [5 9]		#   38 [11 15]		
#   18 [ 5 10]		#   39 [12 13]		
#   19 [5 6]		#   40 [13 14]		
#   20 [6 9]				

###

"""