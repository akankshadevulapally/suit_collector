"""
A Trainer Python file to Train a Deep Neural Network with Temporal Difference Q-Learning and DQN
to play Suit Collector Game For Phase II, Iteration 2.The Aim is To Train Deep Neural Network Agent 
that learns to play Suit Collector and wins 1000x10 Games against a Random Playing Agent.
This version is used to Train The Neural Network Agent which will be referred to as 'Gold'. 
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

### Observations
# ----------------
# Adding to the Observations from train.py from Iteration 1. The Iron Agent playing better moves
# and the probability of making random good moves being less.  Below Changes are made.
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
gamma = 0.9


#Epsilon greedy parameter
epsilon = 1
epsilon_decay_factor = 0.000000333
epsilon_check_after_games = 300

# Parameters for Experience Replay.
# ------------------------------
# batch size for training
batch_size = int(32*200) # Good initial choice
timesToSample =  12   # sampling 4 times 
min_memory_size = 32*200
# Good enough
max_memory_length = 100000

# Memory
# [state, action, reward, next_state, done]
# Replay Buffer
replay_buffer_state = [];
replay_buffer_action = [];
replay_buffer_reward = [];
replay_buffer_next_state = [];
replay_buffer_done = [];

# For Training
buffer_state_local = [];
buffer_action_local = [];
buffer_reward_local = [];
buffer_next_state_local = [];
buffer_done_local = [];

# For Current Iterations S,A,R,S'
buffer_state_ts = [];
buffer_action_ts = [];
buffer_reward_ts = [];
buffer_next_state_ts = [];
buffer_done_ts = [];

# Using huber loss for stability
loss_function = keras.losses.Huber()
# https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3

# Optimizer - Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

# Other parameters
# -------------------
# How often to update the target network
update_after_games = 10

# How often to update the target network
update_target_network = 500


running_reward = 0
episode_count = 0

# save weights location 
saved_weights_dir = './model/';
file_name = 'model'
file_extension = '.h5'
saved_weights_location = saved_weights_dir + file_name + file_extension
save_weights_after = 600


def main():
    """ Trains the Neural Network Agent """
    # Initialize the Game Environment with Agent Iron.
    env = game('./assistance_model/model.h5')
    
    # For logs
    total_wins = 0
    total_loss = 0
    total_draws = 0
    min_epsilon = 0.05

    draw_percentage_games = 2.0 
    # Take all Games as good games since only 50 moves per game.
    win_all_time = 1000
    agent_training_completed = 0

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

    # Load weights if available.
    if os.path.exists(saved_weights_location):
        model.load_weights(saved_weights_location)
        model_target.load_weights(saved_weights_location)
        print('weights Loaded!!')

    # Let the Training Begin
    total_games = 0
    while agent_training_completed <= win_all_time: # Will break when trained.
        
        # To gracefully exit and control other parameters in future.
        with open('./taining_control.json', 'r') as file:
            data = file.read()
            controls = json.loads(data)
            if controls["stop_training"] == 1:
                print('Stopping traning Gracefully...')
                return;

        # set running reward = 0
        running_reward = 0
        # minimum epsilon between 0, 0,05 & 0.01 
        # for 40%, 30%, 30% of the games.
        min_epsilon = np.random.choice([0.0,0.05,0.01],1,p=[0.4,0.3,0.3])[0]

        # update target model
        tn = int(epsilon_check_after_games/update_after_games)
        for i in range(tn):
            epsilon_target = int(tn - 17)
            epsilon = (1.0 - (i*1.0/epsilon_target))
            if(epsilon < 0):
                epsilon = 0
            good_games = 0
            running_reward = 0
            actions_taken = 0
            buffer_state_ts = [];
            buffer_action_ts = [];
            buffer_reward_ts = [];
            buffer_next_state_ts = [];
            buffer_done_ts = [];
            # Play 'update_after_games' number of games and Store the S,A,R,S'
            while(update_after_games != good_games):

                # Start a new Game.
                state = env.reset()
                episode_reward = 0
                done = False
                local_actions_taken = 0

                buffer_state_local = [];
                buffer_action_local = [];
                buffer_reward_local = [];
                buffer_next_state_local = [];
                buffer_done_local = [];

                # Complete the new game.
                while done == False:
                    action = -1
                    if epsilon > np.random.rand(1)[0]:
                        # Take random action
                        # if random action is valid update the training data,
                        # with a negative reward and re-take another random action.
                        isValidActionTaken = False
                        times_to_try = 0
                        while isValidActionTaken == False and times_to_try < 120:
                            action = np.random.choice(total_actions)
                            isValidActionTaken = env.isValidAction(action)
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(-1);
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
                        # if predicted action is invalid take a valid random action.
                        while isValidActionTaken == False and times_to_try < 120:
                            isValidActionTaken = env.isValidAction(action)
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(-1);
                                buffer_next_state_local.append(state);
                                buffer_done_local.append(True); 
                                # Take a random action
                                action = np.random.choice(total_actions)
                                times_to_try = times_to_try + 1
                        
                    
                    # epsilon decay 
                    epsilon -= epsilon_decay_factor
                    epsilon = max(min_epsilon , epsilon)

                    state_next, reward, done, info = env.step(action)
                    

                    # Save actions and states in local buffer
                    buffer_state_local.append(state);
                    buffer_action_local.append(action);
                    buffer_reward_local.append(reward);
                    buffer_next_state_local.append(state_next);
                    buffer_done_local.append(done);
                    episode_reward = reward
                    
                    #Learn from loss for positive rewards.
                    if done == True and reward == -1 :
                        zstate = np.copy(state)
                        zstate_next = np.copy(state_next)
                        buffer_state_local.append(zstate*-1);
                        buffer_action_local.append(int(info['random_action']));
                        buffer_reward_local.append(1);
                        buffer_next_state_local.append(zstate_next*-1);
                        buffer_done_local.append(done);

                    state = state_next
                    local_actions_taken += 1
                
                # Game is now completed. Check if its a good game.
                isGoodGame = False
                if episode_reward == 0:
                    #game is a draw and a bad game, we need only 10% of these so lets do this.
                    if np.random.rand(1)[0] <= draw_percentage_games:
                        # lucky game will be added.
                        isGoodGame = True
                else:
                    isGoodGame = True
                       
                if isGoodGame:
                    total_games += 1
                    good_games += 1
                    actions_taken += local_actions_taken
                    # This is called moving average , pretty cool math.
                    running_reward = (episode_reward + (good_games-1) * running_reward ) / good_games
                    
                    # logging more data
                    if(episode_reward == 0 or episode_reward == -0.05 ):
                        total_draws += 1
                    elif(episode_reward == 1):
                        total_wins += 1
                    elif(episode_reward == -1):
                        total_loss += 1               
                    
                    for lrbl in range(len(buffer_action_local)):
                        # update the next training set
                        buffer_state_ts.append(buffer_state_local[lrbl])
                        buffer_action_ts.append(buffer_action_local[lrbl])
                        buffer_reward_ts.append(buffer_reward_local[lrbl])
                        buffer_next_state_ts.append(buffer_next_state_local[lrbl])
                        buffer_done_ts.append(buffer_done_local[lrbl])
                        # update moves in the replay buffer
                        replay_buffer_state.append(buffer_state_local[lrbl])
                        replay_buffer_action.append(buffer_action_local[lrbl])
                        replay_buffer_reward.append(buffer_reward_local[lrbl])
                        replay_buffer_next_state.append(buffer_next_state_local[lrbl])
                        replay_buffer_done.append(buffer_done_local[lrbl])

            # Training
            for _ in range(timesToSample):

                # Get indices of samples for replay buffers
                if(len(replay_buffer_action) > min_memory_size):
                    # get random moves from replay buffer so we have 'batch_size' S,A,R,S' for training
                    indices_replay_buffer = np.random.choice(range(len(replay_buffer_action)), size=(batch_size-len(buffer_action_ts)))
                    
                    #  Get S,A,R,S' for training from replay_buffer_state
                    state_sample = np.array([replay_buffer_state[td] for td in indices_replay_buffer])
                    action_sample = np.array([replay_buffer_action[td] for td in indices_replay_buffer])
                    reward_sample = np.array([replay_buffer_reward[td] for td in indices_replay_buffer])
                    state_next_sample = np.array([replay_buffer_next_state[td] for td in indices_replay_buffer])
                    done_sample = np.array([replay_buffer_done[td] for td in indices_replay_buffer])
                    
                    #  Get S,A,R,S' for training from the current episode training set as well.
                    state_sample = np.append(state_sample, np.array(buffer_state_ts), axis=0)
                    action_sample = np.append(action_sample, np.array(buffer_action_ts), axis=0)
                    reward_sample = np.append(reward_sample, np.array(buffer_reward_ts), axis=0)
                    state_next_sample = np.append(state_next_sample, np.array(buffer_state_ts), axis=0)
                    done_sample = np.append(done_sample, np.array(buffer_done_ts), axis=0)

                    # Randomize the training data
                    indices_sample_order = np.arange(len(state_sample))
                    np.random.shuffle(indices_sample_order)
                    state_sample = state_sample[indices_sample_order]
                    action_sample = action_sample[indices_sample_order] 
                    reward_sample = reward_sample[indices_sample_order]
                    state_next_sample = state_next_sample[indices_sample_order]
                    done_sample = done_sample[indices_sample_order]

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
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # clear up memory if maximum memory length is reached.
            if len(replay_buffer_action) > max_memory_length:
                print('freeing up replay buffer memory')
                del replay_buffer_action[:int(len(replay_buffer_action)/5)]
                del replay_buffer_done[:int(len(replay_buffer_done)/5)]
                del replay_buffer_next_state[:int(len(replay_buffer_next_state)/5)]
                del replay_buffer_reward[:int(len(replay_buffer_reward)/5)]
                del replay_buffer_state[:int(len(replay_buffer_state)/5)]
            
            # if running reward is > 0.9 updated training completed. 
            if running_reward > 0.9:
                agent_training_completed += 1
                
                
            
            # Log details
            print()
            template = "running reward: {:.2f} at game {} ({}) , epsilon = {:.3f} , memory = {}, actions = {} , {}"
            template2 = "total_draws: {}, total_wins: {}, total_loss: {}"
            print(template.format(running_reward, total_games, (i+1), epsilon,len(replay_buffer_action), len(buffer_action_ts), datetime.now()))
            print(template2.format(total_draws,total_wins, total_loss))
            print()

            #saving the model
            model.save_weights(saved_weights_location)

        
        # Time to Update the Target Model
        # update the the target network with new weights
        if(total_games%update_target_network == 0):
            model_target.set_weights(model.get_weights())

        # Log details
        print()
        template = "running reward: {:.2f} at game {} , epsilon = {:.3f} , memory = {}, completed = {:.2f}%"
        print(template.format(running_reward, total_games, epsilon,len(replay_buffer_action), (agent_training_completed*100/win_all_time)), flush=True)
        print()

        #saving the model
        if total_games%save_weights_after == 0:
            model.save_weights(saved_weights_dir + file_name + '_' +str(total_games)+ '_' + str(datetime.now().strftime(DTTM_FORMAT)) + file_extension)


# Network for Agent Gold
def create_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # Hidden layers
    layer1 = layers.Dense(500, activation="relu")(inputs)
    layer2 = layers.Dense(250, activation="relu")(layer1)

    # output layer    
    action = layers.Dense(total_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)


if __name__=="__main__":
    main()
