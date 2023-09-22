"""
A Gymnasium Game file for Suit Collector. 
Game File for Training in Phase II, Iteration 2. 
For every action performed, The Game in response makes a action from 'Gold' Neural Network Agent or 
a valid random action for the opponent.
"""

# Standard Imports.
import random as rd;
import numpy as np;
import gymnasium as gym;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K


from gymnasium import Env, spaces 


# Suite Collector Game Class File Inherits from Gymnasium Environment.
class game(gym.Env):

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

    def generateActions(self):
        """ Generate the actions and action count. """
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
    
    def startGame(self):
        """ Initialize a new board """
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
   
    def isValid(self, x):
        """Checks if a co-ordinate is valid or not
        Arguments:
        x       -> Integer representing X or Y representing an (X,Y)
                   Position on the Grid 
        """
        if x >=0 and x < 4:
            return True
        return False

    def normalizeBoard(self):
        """ Normalizes the data on the board to values between [-1,1]. """
        return (self.board / 4)

    def normalizeBoardForAssistanceModel(self):
        """ Normalizes the data on the board to values between [-1,1] for Agent Iron to process. """
        return (self.board / -4)

    def populateAction(self,x,y):
        """ Populate both action Maps with an actionXY Map. 
        Arguments:
        x,y     -> Co ordinates of a position on the 4x4 Grid.
        """
        self.actionsXY[x,y] = self.actionsCount
        self.actions[self.actionsCount, 0] = x
        self.actions[self.actionsCount, 1] = y
        self.actionsCount += 1
    
    def isValidAction(self, action):
        """Checks if a given action is valid for the current board."""
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

    def processAction(self,a,b,c,d):
        """ Processes an action
            1. checks if the destination coordinates are correct.
            2. checks if the action already exists. Since swap(X,Y) = swap(Y,X)
        Arguments:
            a,b     -> Co ordinates of a position on the 4x4 Grid.
            c,d     -> Co ordinates of a position on the 4x4 Grid.
        """
        if self.isValid(c) and self.isValid(d):
            x, y = 4 * a + b, 4 * c + d
            if( x > y):
                x,y = y,x
            if(self.actionsXY[x,y] == -1):
                self.populateAction(x,y)

    def reset(self):
        """ Rests the board = creates a fresh new game
        Returns:
            the normalized Board Position.
        """
        self.startGame()
        return self.normalizeBoard()

    def step(self, action):
        """
        Performs an action from the User/Agent and 
        responds with a random action valid action from the opponent 
        Arguments:
            action -> An Integer representing an action of swapping X and Y position.
                      such that actions[action] = (X,Y)
        Return: 
            next_state, reward, done?, additional_info
        """
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
        
    def TestcheckIfSuiteWon(self, board, aces, suite):
        """ Helper method to test if a suite won the game or not """
        self.board = board
        self.aces = aces
        print(self.checkIfSuiteWon(suite))
    
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

    # Network for Agent Gold.
    def create_q_model(self,state_shape, total_actions):
        # input layer
        inputs = layers.Input(shape=state_shape)
        #initializer1 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)
        #initializer2 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.4)
        #layer1 = layers.Dense(144, activation="relu", kernel_initializer=initializer2 )(inputs)
        #layer2 = layers.Dense(840, activation="relu", kernel_initializer=initializer2)(layer1)
        # (n_samples, height, width, channels)
        layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
        # 24 filters , 2x2 size.
        #initializer1 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.4)
        layer1 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer0)
        layer2 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer1)
        layer3 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
        #layer12 = layers.Conv2D(16, 2, strides=1, activation="relu")(layer1)
        #layer13 = layers.Conv2D(16, 2, strides=1, activation="relu")(layerl2) 
        #layer13 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer12)
        #layer12 = layers.Conv2D(32, 2, strides=1, activation="relu")(layer1) 

        #layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
        #layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
        #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)

        #layer2 = layers.Flatten()(layer1)
        # Hidden layers
        #layer5 = layers.Dense(333, activation="relu" )(layer4)
        #layer6 = layers.Dense(123, activation="relu")(layer5)
        #layer7 = layers.Dense(77, activation="relu")(layer6)
        #layer7 = layers.Dense(77, activation="relu", kernel_initializer=initializer2)(layer6)
        #layer6 = layers.Dense(300, activation="relu")(layer5)
        #layer6 = layers.Dense(207, activation="relu")(layer5)
        layer31 = layers.Flatten()(layer3)
        layer4 = layers.Dense(77, activation="relu")(layer31)
        #layer4 = layers.Dense(45, activation="relu")(layer3)   
        #layer5 = layers.Dense(10, activation="relu")(layer4)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer4)

        return keras.Model(inputs=inputs, outputs=action)
    
    def model_test(self):
        #plt.tick_params(left=False,
        #            bottom=False,
        #            labelleft=False,
        #            labelbottom=False)
        print()
        #self.board = np.array([-2,0,-3,0,1,2,0,4,0,0,3,0,0,0,-4,-1])
        #self.board = np.array([-1,0,0,0,1,2,0,4,-2,-3,3,-4,0,0,0,0])
        self.board = np.array([-1,1,-2,0,    0,2,-3,0,    0,0,3,0,     0,4,-4,0])
        print(self.normalizeBoard().reshape(4,4))
        print()
        #sns.heatmap
        
        sns.heatmap(self.normalizeBoard().reshape(4,4), cmap='viridis')
        plt.show()
        
        self.model.summary()
        state_tensor = tf.convert_to_tensor(self.normalizeBoard())
        state_tensor = tf.expand_dims(state_tensor, 0)
        #action_probs = self.model(state_tensor, training=False)
        #inp = self.model.input  
        #outputs = [layer.output for layer in self.model.layers]          # all layer outputs
        #functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
        #for output in outputs:
        #    print(self.model(input=state_tensor, outputs=output))
        #print(inp, outputs)
        #print(functor)
        #self.get_all_outputs(self.model, state_tensor, 1 )
        
        outputs = [self.model.layers[i].output for i in [2]]
        model_short = keras.Model(inputs =self.model.inputs, outputs = outputs)
        print(model_short.summary())
        out = model_short.predict(state_tensor)
        plt.imshow(out[0].flatten().reshape(24,24), cmap='viridis')

        """
        res = np.full((12,12), np.nan)    
        res.flat[:len(out.flatten())]=out.flatten()
        #print(res)
        #print(out[0])
        sns.heatmap(res, cmap='viridis')
        plt.show()
        """

        c = 8
        r = 16
        plot_total = 128
        plot_c = 12
        plot_r = 12
        for a in out:
            fig = plt.figure(figsize=(12,12))
            for i in range(1, plot_total+1):
                fig = plt.subplot(plot_r,plot_c,i)
                fig.set_xticks([])
                fig.set_yticks([])
                plt.imshow(a[0,:,:,i-1], cmap="viridis")
            plt.show()
        
def main():
    env = game('./model.h5')
    env.model_test()
    """
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
        
    """

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