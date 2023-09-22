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


# Suite Collector Game Class File
class game(gym.Env):

    def __init__(self, model_path):
        # Setting up the Game
        # Pieces - Format AB, A-> Suite , B-> Number
        self.pieces = np.array([1,2,3,4,0,0,0,0,0,0,0,0,-1,-2,-3,-4])
        self.board = np.copy(self.pieces)

        #populating actions
        # swap x and y  
        # actions[i] = [x,y]
        self.actions = np.full([16*8,2], -1)
        #actionsXY[x,y] = i , reverse Map of above map
        self.actionsXY = np.full([16,16],-1)
        self.actionsCount = 0
        # The suites in the Game.
        self.suites = np.array([1,-1])
        
        
        # For each cell populating the valid actions.
        for i in range(0,4):
            for j in range(0,4):
                #8 things
                a,b = i,j

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

        
        # special actions [42=1, 43=2, 44=3, 45=4]
        
        # Zero computer goes first
        # 1 is Agent/Player goes first
        self.turn = -1

        # Max Turns each game before its a draw
        self.maxTurnsEachGame = 50
        self.action_space = spaces.Discrete(self.actionsCount)
        self.observation_space = spaces.Box(low=-4, high=4, shape =(16,), dtype=np.float16)

        # create model
        if os.path.exists(model_path):
            self.model = self.create_q_model(self.observation_space.shape , self.action_space.n)
            self.model.load_weights(model_path)
            print('Assistance model is all set.')
        else:
            print('Path ', model_path, 'does not exist.')
            return
        
        # Initializes the board and everything else.
        self.startGame()

    
    # Initialize a new board.
    def startGame(self):
        # Initialize a random board
        self.board = np.copy(self.pieces)
        np.random.shuffle(self.board)
        #self.board = np.append(self.board, [-1,-1])

        # turn , 0 => Computer , 1 => Player/Agent 
        self.turn = int(rd.getrandbits(1))

        # Track Aces in the board, Format - A1 | A = {1,2,3,4}
        self.aces = np.full([3],-1)
        for i in range(0,len(self.board)):
            x = self.board[i] 
            if self.board[i] == 1 or self.board[i] == -1:
                self.aces[self.board[i]] = i
        #print(self.aces)

        # first play pick a suite
        #if self.turn == 0:
        #    self.board[-1] = np.random.choice(self.suites, None)
        #    self.turn = 1
        self.time = 0

        #print(self.actionsCount)
        #print(self.actions)
        #for i in range(44):
            #print(i , self.actions[i])
        #print(self.actionsXY)
   
    # Checks if a co-ordinate is valid or not
    def isValid(self, x):
        if x >=0 and x < 4:
            return True
        return False

    def normalizeBoard(self):
        return (self.board / 4)

    def normalizeBoardForAssistanceModel(self):
        return (self.board / -4)

    # Populate both action Maps with an action.
    def populateAction(self,x,y):
        self.actionsXY[x,y] = self.actionsCount
        self.actions[self.actionsCount, 0] = x
        self.actions[self.actionsCount, 1] = y
        self.actionsCount += 1

    # Processes a action
    #   checks if the destination coordinates are correct.
    #   checks if the action already exists. swap(X,Y) = swap(Y,X)
    def processAction(self,a,b,c,d):
        if self.isValid(c) and self.isValid(d):
            x, y = 4 * a + b, 4 * c + d
            if( x > y):
                x,y = y,x
            if(self.actionsXY[x,y] == -1):
                self.populateAction(x,y)

    def getActionFromCoOrds(self, x, y):
        if x>y:
            x,y = y,x
        return self.actionsXY[x,y]

    # rests the board = creates a new game 
    def reset(self):
        #self.board = np.copy(self.pieces)
        #np.random.shuffle(self.board)
        #self.board = self.board.reshape([4,4])
        #self.Players = [-1,-1]
        #self.turn = 0
        #self.gameStarted = False
        self.startGame()
        return self.normalizeBoard()

    # Performs an action from the Agent and 
    # responds with a random action from the agent 
    # with nextState, reward, GameOver?, additional info
    def step(self, action):
        # Additional Information        
        info = {}
        reward = 0
        done = False

        # If more than 1500 turns are played
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

            if (
                card1 < 0 or \
                card2 < 0
                ):
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
                #print('Yay!! I won!')
                return self.normalizeBoard(), 1, True, info
            # if not won, but played a correct move set a reward of 1
            if self.board[card1Pos] == 0 and self.board[card2Pos] == 0:
                reward = -0.05
        else:
            print("Unknown Error, action was ", action)
            info['error'] = 'Unknown Error!!! action was , '+ str(action);
            self.render()
            return self.normalizeBoard(), 0, True, info

        #print('You made a move ', action)
        info['your_action'] = action

        # Computer makes a move.
        # takes the action from assistance agent,
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

            if (
                card1 > 0 or \
                card2 > 0
                ):
                myaction = rd.randint(0,41)
                continue
            else:
                break
            
        
        # Perform the action if valid
        self.board[card1Pos] = card2
        self.board[card2Pos] = card1
        #print("Computer playing action ",myaction)
        info['random_action'] = myaction
        # Track aces if required.
        if self.board[card1Pos] == -1:
            self.aces[-1] = card1Pos

        if self.board[card2Pos] == -1:
            self.aces[-1] = card2Pos

        # Check if the computer won.
        # set reward accordingly and return.
        won = self.checkIfSuiteWon(-1)
        if won:
            return self.normalizeBoard(), -1, True, info

            

        
        # finally
        #self.render()
        return self.normalizeBoard(), reward, done, info 
    
    
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
        
    
    #Helper method, used to start a game between two
    #Random players
    def randoVsRando(self):
            #Computer makes your random valid move.
            # repeat - pick a random move and check until its valid
            action = 0
            card1Pos = None
            card2Pos = None
            while True:
                action = rd.randint(0,41)
                card1Pos = self.actions[action][0]
                card2Pos = self.actions[action][1]

                card1 = self.board[card1Pos]
                card2 = self.board[card2Pos]

                if (
                    card1 < 0 or \
                    card2 < 0
                    ):
                    continue
                else:
                    break

            board, reward, done, info = self.step(action)
            return board, reward, done, info, action
        
        
    # Check if a Suite won the game or not.
    def checkIfSuiteWon(self, suite):
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
        
    # Helper method to test if a suite won the game or not
    def TestcheckIfSuiteWon(self, board, aces, suite):
        self.board = board
        self.aces = aces
        print(self.checkIfSuiteWon(suite))
    
    # Renders the game on to the console.
    def render(self):
        print()
        for i in range(0,4):
            for j in range(0,4):
                print(self.board[(i*4)+j], ",  ", end="")
            print()
        print("\nSuite: YOU =" , 1, " , ME = ", -1 )
        print("aces position = ", self.aces)
        print("board = " , self.board)
        print("normalize board = ", self.normalizeBoard())
        print("time = ", self.time)
    
    # Networks
    # def create_q_model(self, state_shape, total_actions):
    #     # input layer
    #     inputs = layers.Input(shape=state_shape)

    #     # Hidden layers
    #     layer1 = layers.Dense(40, activation="relu")(inputs)
    #     layer2 = layers.Dense(40, activation="relu")(layer1)
    
    #     # output layer    
    #     action = layers.Dense(total_actions, activation="linear")(layer2)

    #     return keras.Model(inputs=inputs, outputs=action)

    # Networks
    def create_q_model(self, state_shape, total_actions):
        # input layer
        inputs = layers.Input(shape=state_shape)
        
        layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
        # 24 filters , 2x2 size.
        layer1 =  layers.Conv2D(64, 2, strides=1, activation="relu")(layer0)
        layer2 = layers.Flatten()(layer1)
        # Hidden layers
        #layer5 = layers.Dense(333, activation="relu" )(layer4)
        #layer6 = layers.Dense(123, activation="relu")(layer5)
        #layer7 = layers.Dense(77, activation="relu")(layer6)
        #layer7 = layers.Dense(77, activation="relu", kernel_initializer=initializer2)(layer6)
        #layer6 = layers.Dense(300, activation="relu")(layer5)
        #layer6 = layers.Dense(207, activation="relu")(layer5)
        layer3 = layers.Dense(200, activation="relu")(layer2)
        #layer4 = layers.Dense(45, activation="relu")(layer3)   
        #layer5 = layers.Dense(10, activation="relu")(layer4)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer3)

        return keras.Model(inputs=inputs, outputs=action)


    def model_test(self):
        plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
        print()
        #self.board = np.array([-1,0,0,0,1,2,0,4,-2,-3,3,-4,0,0,0,0])
        print(self.normalizeBoard().reshape(4,4))
        print()
        #sns.heatmap
        sns.heatmap(self.normalizeBoard().reshape(4,4), cmap='viridis')
        #self.model.summary()
        state_tensor = tf.convert_to_tensor(self.normalizeBoard())
        state_tensor = tf.expand_dims(state_tensor, 0)
        #action_probs = self.model(state_tensor, training=False)
        plt.show()
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
        plt.show()
        c = 8
        r = 8
        for a in out:
            fig = plt.figure(figsize=(8,8))
            for i in range(1, c*r+1):
                fig = plt.subplot(r,c,i)
                fig.set_xticks([])
                fig.set_yticks([])
                plt.imshow(a[0,:,:,i-1], cmap="viridis")
            plt.show()
            
        
        
        

def main():
    env = game('./model/model.h5')
    env.model_test()
    #env.render()
    #print('Number of states: {}'.format(env.observation_space))
    #print('Number of actions: {}'.format(env.action_space))   
    #env.TestcheckIfSuiteWon(np.array([0,-4,2,0, 0,-2,1,0, -3,4,0,0, 3,0,-1,0]), [-1 ,6 ,2], -1)
    #env.TestcheckIfSuiteWon(np.array([0,-4,2,0, 0,-2,1,0, -3,4,0,0, 3,0,-1,0]), [-1 ,6 ,2], 1)
    """
    for i in range(100):
        env.reset()
        done = False
        reward = 0
        action = 0
        while done != True:
            #env.render()
            board, reward , done , info, action = env.randoVsRando()
        if(reward != 0):
            env.render()
            print("reward : " , reward, "action : ", action)
            print(board)
            print('---------------------------------------------------')

    """





if __name__ == "__main__":
    main()

###
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