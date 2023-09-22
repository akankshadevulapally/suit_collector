"""
This Program is used to find optimal learning rate.
"""

#imports
import gymnasium as gym
import random as rd
from game import game 
from gymnasium import Env, spaces 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging 
import sys
import os.path
from datetime import datetime
import json
from keras import backend as K
DTTM_FORMAT = '%Y%m%d%H%M%S'



### Observations
# ----------------
# Adding to the Observations from train.py from Iteration 1. The Gold Agent playing better moves
# and the probability of making random good moves being less.  Below Changes are made.
# A convoluted Network is included.
#  - Epsilon-Greedy method is changed to be active during the first half of the games in the Training Iterations.
#  - Since the Probability of getting rewards are pretty less, inverting the last move of every loss from Iron Agent to 
#       the training data.
#  - If random move is invalid during epsilon Phase, A negative reward is added and also, another random move is made, until
#    The game moves forward, until draw or loss.  
#    



# Configurations for Training
#-----------------------------------
# Global Variables - Hyper Parameters - Constants

#Date Time Format
DTTM_FORMAT = '%Y%m%d%H%M%S'
#max_steps_per_episode = Controlled within the game.

# Discount factor gamma, High discount factor since rewards
# appear only in the end.  
gamma = 0.99

#log file name
logjsonname = 'log.json'


# Epsilon greedy parameter
# min and max are very low because of very high illegal rate of the game.
min_epsilon = 0.05
#min_epsilon = 0.005
max_epsilon = 0.25
#max_epsilon = 0.015
epsilon = max_epsilon  # slowly goes down to min_epsilon
#pre training
epsilon_decay_factor = 0.000000333
epsilon_reset_after = 1
epsilon_check_after_games = 300 
consider_bad_random_moves = 0.1 #10%

# batch size for training
batch_size = int(32*200) # Good initial choice
timesToSample =  5   # sampling 4 times 
min_memory_size = 32*200

#max_steps_per_episode = Controlled within the game.

# Train the model after X games
update_after_games = 100

# How often to update the target network
update_target_network = 500

# Good enough
max_memory_length = 100000

# Using huber loss for stability
loss_function = keras.losses.Huber()

# Optimizer - Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
#optimizer = keras.optimizers.Adam(learning_rate=0.000125, clipnorm=1.0)
#optimizer = keras.optimizers.Adam(learning_rate=0.0000625, clipnorm=1.0)
#optimizer = keras.optimizers.Adam(learning_rate=0.00003125, clipnorm=1.0) 0.000015625
#optimizer = keras.optimizers.Adam(learning_rate=0.000015625, clipnorm=1.0)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
#optimizer = tf.keras.optimizers.RMSprop()

# Memory
# [state, action, reward, next_state, done]
replay_buffer_state = [];
replay_buffer_action = [];
replay_buffer_reward = [];
replay_buffer_next_state = [];
replay_buffer_done = [];

buffer_state_local = [];
buffer_action_local = [];
buffer_reward_local = [];
buffer_next_state_local = [];
buffer_done_local = [];


buffer_state_ts = [];
buffer_action_ts = [];
buffer_reward_ts = [];
buffer_next_state_ts = [];
buffer_done_ts = [];


running_reward = 0
episode_count = 0

# save weights location 
saved_weights_dir = './model/';
file_name = 'model'
file_extension = '.h5'
saved_weights_location = saved_weights_dir + file_name + file_extension
save_weights_after = 20000

def main():
    """ Trains the Neural Network Agent """
    # Initialize the Game Environment with Gold Agent.
    env = game('./assistance_model/model.h5')
    #log
    total_games = 0
    total_draws = 0
    total_wins = 0
    total_loss = 0
    total_draws_bp = total_draws
    total_wins_bp = total_wins
    total_loss_bp = total_loss 



    # Fetch the Observation space shape 
    # and num of actions from env
    state_shape = env.observation_space.shape
    total_actions = env.action_space.n
    print(' state_shape = ',state_shape, ' action_space = ',total_actions)

    # Create model and target model
    model = create_q_model(state_shape, total_actions)
    # for the sake of consistency 
    model_target = create_q_model(state_shape, total_actions)
    model.summary()

    #first Run values:
    #inital_learning_rate = 0.0000000000001
    #final_learning_rate = 20
    #step_learning_rate = 10
    #second run values:
    inital_learning_rate = 1e-13
    final_learning_rate = 2
    step_learning_rate = 10
    initial_run = True
    lr_index = 0
    lr_array_val = [1e-05,1e-04,1e-03,1e-02,1e-01,1e-00]
    lr_mul_arr = [1,2,3,4,5,6,7,8,9]
    lr_arr = [1e-07,1e-06]
    for i in lr_array_val:
        for j in lr_mul_arr:
            lr_arr.append(i*j)
    print(lr_arr)
    while lr_index < len(lr_arr): 
        
        # set running reward = 0
        running_reward = 0
        
        min_epsilon = 0
        # update target model
        tn = 25
        K.set_value(optimizer.learning_rate, lr_arr[lr_index])
        for i in range(tn):
            epsilon = 0
            good_games = 0
            running_reward = 0
            actions_taken = 0
            buffer_state_ts = [];
            buffer_action_ts = [];
            buffer_reward_ts = [];
            buffer_next_state_ts = [];
            buffer_done_ts = [];
            #print(len(replay_buffer_action))
            # Play 'update_after_games' number of games and Store the S,A,R,S'
            while(len(replay_buffer_action) < (min_memory_size*2) and initial_run):
                # Start a new Game.
                state = env.reset()
                episode_reward = 0
                done = False
                local_actions_taken = 0
                # Complete the new game that is started.
                buffer_state_local = [];
                buffer_action_local = [];
                buffer_reward_local = [];
                buffer_next_state_local = [];
                buffer_done_local = [];
                while done == False:
                    action = -1
                    if epsilon > np.random.rand(1)[0]:
                        # Take random action
                        isValidActionTaken = False
                        times_to_try = 0
                        while isValidActionTaken == False and times_to_try < 120:
                            action = np.random.choice(total_actions)
                            isValidActionResult = env.isValidAction(action)
                            if isValidActionResult != 0:
                                isValidActionTaken = False
                            else:
                                isValidActionTaken = True
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(isValidActionResult);
                                buffer_next_state_local.append(state);
                                buffer_done_local.append(True); 
                                times_to_try = times_to_try + 1

                            
                    else:
                        # Predict action Q-values
                        # From environment state
                        state_tensor = tf.convert_to_tensor(state)
                        state_tensor = tf.expand_dims(state_tensor, 0)
                        action_probs = model(state_tensor, training=False)
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()
                        isValidActionTaken = False
                        times_to_try = 0
                        while isValidActionTaken == False and times_to_try < 120:
                            isValidActionResult = env.isValidAction(action)
                            if isValidActionResult != 0:
                                isValidActionTaken = False
                            else:
                                isValidActionTaken = True
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(isValidActionResult);
                                buffer_next_state_local.append(state);
                                buffer_done_local.append(True); 
                                # Take a random action
                                action = np.random.choice(total_actions)
                                times_to_try = times_to_try + 1
                    
                    # epsilon greedy - decay
                    epsilon -= epsilon_decay_factor
                    epsilon = max(min_epsilon , epsilon)

                    state_next, reward, done, info = env.step(action)
                    

                    # Save actions and states in replay buffer
                    buffer_state_local.append(state);
                    buffer_action_local.append(action);
                    buffer_reward_local.append(reward);
                    buffer_next_state_local.append(state_next);
                    buffer_done_local.append(done);
                    episode_reward = reward
                    # #Learn from loss for positive rewards.
                    if done == True and reward == -100 :
                        zstate = np.copy(state)
                        zstate_next = np.copy(state_next)
                        buffer_state_local.append(zstate*-1);
                        buffer_action_local.append(int(info['random_action']));
                        buffer_reward_local.append(110);
                        buffer_next_state_local.append(zstate_next*-1);
                        buffer_done_local.append(done);
                        
                        

                    state = state_next
                    local_actions_taken += 1
                
                # Game is now completed.                   
                # Check if its a good game.
                isGoodGame = False
                if episode_reward == -1:
                    #game is a draw and a bad game, we need only 10% of these so lets do this.
                    if np.random.rand(1)[0] <= 2.0:
                        # lucky game will be added.
                        isGoodGame = True
                else:
                    isGoodGame = True
                       
                if isGoodGame:
                    total_games += 1
                    good_games += 1
                    actions_taken += local_actions_taken
                    # moving average , pretty cool math
                    running_reward = (episode_reward + (good_games-1) * running_reward ) / good_games
                    
                    # logging more data
                    if(episode_reward == -1 ):
                        total_draws += 1
                    elif(episode_reward == 110):
                        total_wins += 1
                    elif(episode_reward == -100):
                        total_loss += 1
                        
                    
                    for lrbl in range(len(buffer_action_local)):
                        # update the training set
                        buffer_state_ts.append(buffer_state_local[lrbl])
                        buffer_action_ts.append(buffer_action_local[lrbl])
                        buffer_reward_ts.append(buffer_reward_local[lrbl])
                        buffer_next_state_ts.append(buffer_next_state_local[lrbl])
                        buffer_done_ts.append(buffer_done_local[lrbl])
                        # update replay buffer
                        replay_buffer_state.append(buffer_state_local[lrbl])
                        replay_buffer_action.append(buffer_action_local[lrbl])
                        replay_buffer_reward.append(buffer_reward_local[lrbl])
                        replay_buffer_next_state.append(buffer_next_state_local[lrbl])
                        replay_buffer_done.append(buffer_done_local[lrbl])

            initial_run = False
            loss_val_sum = 0
            loss_avg = 0
            loss_count = 0
            # Training
            for _ in range(timesToSample):
                #break;
                # Get indices of samples for replay buffers
                if(len(replay_buffer_action) > min_memory_size):

                    # get random moves from replay buffer so we have 'batch_size' S,A,R,S' for training
                    indices_replay_buffer = np.random.choice(range(len(replay_buffer_action)), size=(batch_size))
                    
                    #  Get S,A,R,S' for training from replay_buffer_state
                    state_sample = np.array([replay_buffer_state[td] for td in indices_replay_buffer])
                    action_sample = np.array([replay_buffer_action[td] for td in indices_replay_buffer])
                    reward_sample = np.array([replay_buffer_reward[td] for td in indices_replay_buffer])
                    state_next_sample = np.array([replay_buffer_next_state[td] for td in indices_replay_buffer])
                    done_sample = np.array([replay_buffer_done[td] for td in indices_replay_buffer])
                    


                    # Build the Q-values for the sampled future states
                    # Use the target model for stability
                    # Get Future rewards from Target Model.
                    future_rewards = model_target.predict(state_next_sample)

                    # Q(s,a) = r + gamma * max(Q(S',a'))
                    updated_q_values = reward_sample + gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # Correct the Q(s,a) for s when s is the final state.                    
                    new_update_q_values = updated_q_values.numpy()
                    for ildse in range(len(done_sample)):
                            if(done_sample[ildse]):
                                if(int(reward_sample[ildse]) != 0):
                                    new_update_q_values[ildse] = reward_sample[ildse]
                    updated_q_values = tf.convert_to_tensor(new_update_q_values)
                    

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, total_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample)
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)
                    
                    # Backpropagation
                    #print(loss)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    loss_count = loss_count + 1
                    loss_val_sum = loss_val_sum  + loss.numpy()
                
            
            # Log details
            if loss_count > 0:
                loss_avg = loss_val_sum / loss_count

            # delim = "\n"
            # loggValues = {};
            # loggValues["r"] = running_reward;
            # loggValues["g"] = total_games;
            # loggValues["d"] = total_draws - total_draws_bp ;
            # loggValues["w"] = total_wins - total_wins_bp ;
            # loggValues["l"] = total_loss - total_loss_bp;
            # loggValues["td"] = total_draws;
            # loggValues["tw"] = total_wins;
            # loggValues["tl"] = total_loss;
            # loggValues["l_"] = loss_avg*10000000;
            # loggValues["e"] = epsilon;
            # loggValues["a"] = len(buffer_action_ts);
            # loggValues["s"] = len(replay_buffer_action);
            # loggValues["t"] = datetime.now().isoformat();
            # loggValues["p"] = agent_training_completed*100/win_all_time;
            # with open(logjsonname, 'a') as f:
            #     json.dump(loggValues, f)
            #     f.write(delim)

            #saving the model
            #model.save_weights(saved_weights_location)
            total_draws_bp = total_draws
            total_wins_bp = total_wins
            total_loss_bp = total_loss
        # Time to Update the Target Model
        # update the the target network with new weights
        #if(total_games%update_target_network == 0):
        #    model_target.set_weights(model.get_weights())


        #saving the model
        #if total_games%save_weights_after == 0:
        #    model.save_weights(saved_weights_dir + file_name + '_' +str(total_games)+ '_' + str(datetime.now().strftime(DTTM_FORMAT)) + file_extension)
        mydata = {}
        mydata["learning_rate"] = lr_arr[lr_index]
        mydata["loss"] = loss_avg*10000000
        lr_index+=1
        p_mydata = json.dumps(mydata)
        print(p_mydata)
        #first run
        #inital_learning_rate = inital_learning_rate * step_learning_rate
        #Second run
        #inital_learning_rate = inital_learning_rate * step_learning_rate
        model = create_q_model(state_shape, total_actions)
        model.set_weights(model_target.get_weights())
        # clear up memory
        #del replay_buffer_action[:]
        #del replay_buffer_done[:]
        #del replay_buffer_next_state[:]
        #del replay_buffer_reward[:]
        #del replay_buffer_state[:]
        initial_run = False
                

# Networks
def create_q_model(state_shape, total_actions):
    """
    Create Deep Convolution Q Network for Glow Stone
    """
    # input layer
    inputs = layers.Input(shape=state_shape)

    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)

    # 20 ways to win for each player , two players so 40 ways. 14 actions approx to win each way therefore 40 * 14 = 560 
    #layer1 =  layers.Conv2D(560, 4, strides=1, activation="relu")(layer0)
    
    # 3x3 Grid,
    # 20 ways to win in a 3x3 world with the same rules each way to win has 3 locations in total.
    # 12 actons 
    # 20 * 12 = 240
    layer2 =  layers.Conv2D(240, 3, strides=1, activation="relu")(layer0)
    
    # 12 ways to swap
    layer3 =  layers.Concatenate()([layers.Flatten()(layer2)])
    

    layer4 = layers.Dense(500, activation="relu")(layer3)
    layer5 = layers.Dense(300, activation="relu")(layer4)
    
    # output layer    
    action = layers.Dense(total_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


if __name__=="__main__":
    main()
