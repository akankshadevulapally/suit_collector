'''
AUTHOR: James Ballari, Mugdha Ektare
FILENAME: game.py
SPECIFICATION: This file is the game model for the Gold agent with modifications to use this for Phase 1 output which is picking a suit. This modified code includes
               using the model for Gold agent to predict which suit would be the best one to pick according to the random board generated on the UI. 
FOR: CS 5392 Reinforcement Learning Section 001
'''

import random as rd;

import numpy
import numpy as np;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os.path

import phase1
from phase1 import *


# Suite Collector Game Class File
class game():

    def __init__(self):
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
        
        self.models = [self.create_q_model_iron((16,),42), self.create_q_model_gold((16,),42), self.create_q_model_diamond1((16,),42), self.create_q_model_diamond2((16,),42)]
        
    def convertToPhase2Board(self, board, suite):
        #Converts a board to phase2 Board
        userSuit = suite[0]
        agentSuit = suite[1]
        self.board = np.full([16], 0)
        for i in range(16):
            card = board[i]
            suit = int(card/10)
            value = int(card%10)
            if suit == userSuit:
                self.board[i] = -value
            if suit == agentSuit:
                self.board[i] = value

    
    def trackAces(self):
        # Track Aces in the board, Format - A1 | A = {1,2,3,4}
        self.aces = np.full([3],-1)
        for i in range(0,len(self.board)):
            x = self.board[i] 
            if self.board[i] == 1 or self.board[i] == -1:
                self.aces[self.board[i]] = i
    
   
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


    # Performs an action from the Agent and 
    # responds with a random action from the agent 
    # with nextState, reward, GameOver?, additional info
    def step(self, action):
        # Additional Information        
        info = {}
        reward = -1
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
                return self.normalizeBoard(), -100, True, info
                info["loss_reason"] = "0" # moved opponents cards lol.. 
            if( card1 == 0 and card2 == 0):
               info["loss_reason"] = "1"  # did not move your card, rofl...
               return self.normalizeBoard(), -100, True, info 

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
                return self.normalizeBoard(), 110, True, info
            # if not won, but played a correct move set a reward of 1
            #if self.board[card1Pos] == 0 and self.board[card2Pos] == 0:
            #    reward = -0.5
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
                (card1 > 0 or \
                card2 > 0 ) or ( card1 ==0 and card2 ==0)
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
            return self.normalizeBoard(), -100, True, info

            

        
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

                if ( card1 < 0 or card2 < 0 ):
                    return -100
                if (card1 == 0 and card2 ==0 ):
                    return -100
        else:
            return -100 
        return 0
        
    
    #Helper method, used to start a game between two
    #Random players

    def agentRock(self, state_tensor, training=False):
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

            if ( card1 < 0 or card2 < 0 ):
                continue
            else:
                break

        return_obj = [0] * 41
        return_obj[action] = 1.0
        return tf.convert_to_tensor([return_obj])
    


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
    def create_q_model_iron(self, state_shape, total_actions):
        """ Create a Q model for Agent Iron"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(40, activation="relu")(inputs)
        layer2 = layers.Dense(40, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-iron.h5')
        return model
    
    def create_q_model_gold(self, state_shape, total_actions):
        """ Create a Q model for Agent Gold"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(500, activation="relu")(inputs)
        layer2 = layers.Dense(250, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-gold.h5')
        return model

    def create_q_model_diamond1(self, state_shape, total_actions):
        """ Create a Q model for Agent Diamond 1"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(500, activation="relu")(inputs)
        layer2 = layers.Dense(250, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-gold.h5')
        return model


    def create_q_model_diamond2(self, state_shape, total_actions):
        """ Create a Q model for Agent Diamond 2"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(500, activation="relu")(inputs)
        layer2 = layers.Dense(250, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-gold.h5')
        return model
    
    def process_request(self, message):
        print(message)
        if message['suits'][0] == 0 or message['suits'][1] == 0:
            return self.processSuits(message)
        else:
            return self.processPlay(message)

        
    '''
    NAME: processSuits
    PARAMETERS: self, message (Message has information such as the positions of cards on the current board, suits picked at the moment, selected cards, and the agent ID)
    PURPOSE: This function takes in the current status of the board on the UI and depending on who is going to pick the suit first, (agent or the user) it uses the 
             Gold Model from phase 2 for picking the suit accordingly.
    PRECONDITION: The server.py file should be running. User should go to the localhost port on the browser and select new game and appropriate dropdown options
                  for User's suit and Agent's suit and hit "Play Turn". After that, the server.py will call the method process_request from game.py which will 
                  call this method if atleast one of the suit is not picked. 
    POSTCONDITION: This method will return a data object which has attributes such as board, processSuits, and suit. This returns the suit that is picked by the 
                   Gold agent model. Return value is sent back to process_request method.
    '''
    def processSuits(self, message):
        # Defining the data object here with board attribute set to None and processSuits attribute set to True.
        data = {}
        data['board'] = None
        data['processSuits'] = True
        
        # Check if the agent playing against us is a random agent (Rock agent), if it is Rock agent, then choose a random suit and return the data object.
        if message['playAgainst'] == 0:
            data['suit'] = str(random.randint(1, 4))
            return data
        
        # If the agent playing against us is not the random agent (Rock agent), follow this flow of code
        else:
            # Check if both the suits are to be picked
            if message['suits'][0] == 0 and message['suits'][1] == 0:
                # Agent has to pick the suit first.
                # We have two choices:
                    # 1. Q Learning
                    # 2. Phase 2 Gold model
                print("Choose which method to use:")
                print("1. Q Learning")
                print("2. Phase 2 Gold Model")
                # Choice is stored in variable m
                m = 2
                # IF we want to use the Q learning model from phase.py file, follow this branch of code...
                if m == 1:
                    # Suits are coded as 1, 2, 3, 4 for Clubs, Hearts, Spades, and Diamonds
                    suits = [1, 2, 3, 4]
                    suits_to_names = ['Clubs', 'Hearts', 'Spades', 'Diamonds']
                    # Call the main procedure from phase1.py file and send the current board information over there with a flag variable which indicates that we 
                    # already have a board present. The return variable answer will have the index of the suit to be picked in the phase.py terminology.
                    answer = phase1.main(True, message['board'])
                    # Converting the suit ID from phase1.py terminology to the UI terminology. Storing the suit ID in suit attribute of data object and returning 
                    # the data object.
                    if answer == 0:
                        data['suit'] = '3'
                    elif answer == 1:
                        data['suit'] = '2'
                    elif answer == 2:
                        data['suit'] = '1'
                    else:
                        data['suit'] = '4'
                    return data

                else:
                    # print(message['board'])
                    # Suits are coded as 1, 2, 3, 4 for Clubs, Hearts, Spades, and Diamonds
                    suits = [1, 2, 3, 4]
                    suits_to_names = ['Clubs', 'Hearts', 'Spades', 'Diamonds']
                    # This is an empty list which will contain the average maximum rewards for each choice of the suit in the future.
                    all_suit_max_rewards = []
                    # There are total 12 combination of choices for the user and agent's suit choices given that both the suits are not chosen yet.
                    # The following for loops generate teh board suitable for phase 2 Gold agent for all the 12 permutations of the suits. 
                    # After generating the board suitable for phase 2 gold agent, we get the initial Q values generated by the Gold agent as a measure to see 
                    # the best reward we can get in the future if that permutation of suits is chosen...
                    # These rewards are stored for each choice of suit made by the agent and the ideal suit would be the one that gives the maximum reward. 
                    for i in suits:
                        rewards = []
                        for j in suits:
                            if j != i:
                                # j is user suit
                                # i is agent suit
                                # print(j, i)
                                self.convertToPhase2Board(message['board'], [j, i])
                                # print(self.board)
                                rewards.append(self.getQinitial(message['playAgainst'], self.board))
                        all_suit_max_rewards.append(max(rewards))
                    print(all_suit_max_rewards)
                    ideal_suit = numpy.argmax(all_suit_max_rewards)
                    print("The ideal suit is: ", suits_to_names[ideal_suit])
                # The ideal suit ID should be stored in the range 1 to 4; so adding 1 to the index value of the maximum reward...
                # Storing the ideal suit in the suit attribute of the data object and returning the data object.
                data['suit'] = str(ideal_suit + 1)
                return data

            # Check if the User has chosen the suit and now its agent's turn to pick the suit...
            elif message['suits'][0] == 0 and message['suits'][1] != 0:
                # Storing the suit of the user in a variable.
                user_suit = message['suits'][1]
                # Here as well we could have two methods for choosing the suit for the agent after the user has chosen the suit...
                # For now, we are implementing this using only the Phase 2 Gold agent model...
                print("Choose which method to use:")
                print("1. Q Learning")
                print("2. Phase 2 Gold Model")
                m = 2
                if m == 1:
                    # Some code
                    return None
                else:
                    # Suits are coded as 1, 2, 3, 4 for Clubs, Hearts, Spades, Diamonds
                    suits = [1, 2, 3, 4]
                    suits_to_names = ['Clubs', 'Hearts', 'Spades', 'Diamonds']
                    # Rewards list will have the rewards of choosing each remaining suit given that one suit is already chosen by the user...
                    rewards = []
                    for i in suits:
                        if i != user_suit:
                            # For each suit that is open for choice, generate the board suitable for the phase 2 gold model terminology
                            self.convertToPhase2Board(message['board'], [user_suit, i])
                            # get the initial Q values generated by the Gold agent as a measure to see the best reward we can get in the future if that 
                            # permutation of suits is chosen...
                            rewards.append(self.getQinitial(message['playAgainst'], self.board))
                    # These rewards are stored for each choice of suit made by the agent and the ideal suit would be the one that gives the maximum reward. 
                    # Ideal chosen suit is stored in the ideal_suit variable
                    ideal_suit = numpy.argmax(rewards)
                    print("The ideal suit is: ", suits_to_names[ideal_suit])
                    # Ideal suit index is incremented by 1 and stored in the suit attribute of the date object. Data object is returned.
                    data['suit'] = str(ideal_suit + 1)
                    return data
            
            # See if the agent wants to proceed playing the game without choosing the suit...
            # If so, generate an error messsage to prompt user to pick a suit.
            # Return the data object along with the error message...
            elif message['suits'][0] != 0 and message['suits'][1] == 0:
                data['msg_e'] = 'Please pick your suit first!!!';
                return data;


    def processPlay(self, message):
        data = {};
        data['board'] = None
        data['processPlay'] = True
        self.convertToPhase2Board(message['board'], message['suits']);
        self.trackAces()
        if self.checkIfSuiteWon(1) or self.checkIfSuiteWon(-1) :
            data['msg_i'] = 'Game Has Ended Already';
            return data;
    
        if(len(message['selectedCards']) == 2):
            # check if the selected cards make a valid aciton or not.
            x = message['selectedCards'][0]
            y = message['selectedCards'][1]
            if(self.checkUserValidAction(x,y)):
                # perform the move and track aces
                self.board[x] , self.board[y] = self.board[y] , self.board[x]
                message['board'][x] , message['board'][y] = message['board'][y] , message['board'][x]
                if self.board[x] == -1:
                    self.aces[-1] = x
                if self.board[y] == -1:
                    self.aces[-1] = y

                # check if User won ?
                if self.checkIfSuiteWon(-1):
                    data['msg_s'] = 'You Won!!!'
                    data['board'] = message['board']
                    return data;
            else:
                data['msg_e'] = 'Invalid Action';
                return data;
    
        # your move next!!
        # pick an action using the agent. 
        agent_id = message['playAgainst']
        action = -1; x = -1; y = -1; isRandom = True
        if(agent_id == 0):
            #use random agent to pick an action.
            action,x,y,isRandom = self.getValidRandomAction()
        else:
            action,x,y,isRandom = self.getAgentAction(agent_id-1)

        data['isRandomAction'] = isRandom
        print(action,x,y, isRandom)

        #perform action and track aces
        self.board[x] , self.board[y] = self.board[y] , self.board[x]
        message['board'][x] , message['board'][y] = message['board'][y] , message['board'][x]
        if self.board[x] == 1:
            self.aces[1] = x
        if self.board[y] == 1:
            self.aces[1] = y

        data['board'] =  message['board']
        data['aiSelectedCards'] = [x,y]
        
        if self.checkIfSuiteWon(1):
            data['msg_e'] = 'You Lost!!!'
            return data;
    
        return data

    def checkUserValidAction(self, x, y):
        if( x > y):
            x,y = y,x
        if(self.actionsXY[x,y] == -1):
            return False
        
        card1 = self.board[x]
        card2 = self.board[y]

        if ( card1 > 0 or card2 > 0 ):
            return False
        else:
            return True
        
    def getValidRandomAction(self):
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

            if ( card1 < 0 or card2 < 0 or (card1 ==0 and card2 ==0)):
                continue
            else:
                break
        return action, card1Pos, card2Pos, True

    def getAgentAction(self, agent_id):
        state_tensor = tf.convert_to_tensor(self.normalizeBoard())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_qvalues = self.models[agent_id](state_tensor, training=False)
        print("action_qvalues: ", action_qvalues)
        # Take best action
        action = tf.argmax(action_qvalues[0]).numpy() 
        card1Pos = self.actions[action][0]
        card2Pos = self.actions[action][1]
        card1 = self.board[card1Pos]
        card2 = self.board[card2Pos]
        if(card1< 0 or card2 <0 ):
            action , x, y = self.performValidRandomAction()
            return action , x, y , True
        return action, card1Pos, card2Pos, False

    '''
    NAME: getQinitial
    PARAMETERS: self, agent_id (which agent the user is playing against), board
    PURPOSE: This function takes in the current status of the board and given that the the suits are already chosen by the user and the agent. 
             It then generates the initial Q values for each move that could be made and returns the larges Q value as the reward.
    PRECONDITION: The server.py file should be running. User should go to the localhost port on the browser and select new game and appropriate dropdown options
                  for User's suit and Agent's suit and hit "Play Turn". After that, the server.py will call the method process_request from game.py which will 
                  call processSuits method if atleast one of the suit is not picked. From there, this method is called in order to fetch the rewards.
    POSTCONDITION: This method will return the maximum reward that could be obtained after both the suits are picked from the given UI board. This reward will be 
                   returned to the processSuits method from where the ideal suit is then picked. 
    '''
    
    def getQinitial(self, agent_id, board):
        # Here the board is normalized
        state_tensor = tf.convert_to_tensor(board/4)
        # Expanding the dimensions of the tensor
        state_tensor = tf.expand_dims(state_tensor, 0)
        # Genrating the Q values from the initial state of the board after picking the suits...
        action_qvalues = self.models[agent_id](state_tensor, training=False)
        # print("action_qvalues: ", action_qvalues[0])
        # print("Max Value: ", max(action_qvalues[0].numpy()))
        # Converting the tensor to a numpy array and taking the max value (reward) out of it and returning it to the processSuits method for picking the suit.
        return max(action_qvalues[0].numpy())




def main():
    return 
    """
    env = game('./model_assisted/model.h5')
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
