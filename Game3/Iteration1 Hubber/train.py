"""
A Trainer Python file to Train a Convoluted Deep Neural Network with Temporal Difference Q-Learning and DQN
to play Suit Collector Game For Phase II, Iteration 3.The Aim is To Train Deep Neural Network Agent
that learns to play Suit Collector and wins 1000x10 Games against a Random Playing Agent.
This version is used to Train The Neural Network Agent which will be referred to as 'Diamond1'.
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

# Parameters for Experience Replay.
# ------------------------------
# batch size for training
batch_size = int(32*200) # Good initial choice
timesToSample =  16   # sampling 4 times
min_memory_size = 32*200

#max_steps_per_episode = Controlled within the game.

# Train the model after X games
update_after_games = 10

# How often to update the target network
update_target_network = 500

# Good enough
max_memory_length = 100000

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

# Using huber loss for stability
loss_function = keras.losses.Huber()

# Optimizer - Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
#optimizer = tf.keras.optimizers.RMSprop()


# save weights location
saved_weights_dir = './model/';
file_name = 'model'
file_extension = '.h5'
saved_weights_location = saved_weights_dir + file_name + file_extension
save_weights_after = 600



def main():
    """ Trains the Neural Network Agent """
    # Initialize the Game Environment with Gold Agent.
    env = game('./assistance_model/model.h5')

    #For Logs
    total_wins = 0
    total_loss = 0
    total_draws = 0
    total_draws_bp = 0
    total_wins_bp = 0
    total_loss_bp = 0
    total_games = 0

    min_epsilon = 0.05

    draw_percentage_games = 2.0
    win_all_time = 1000
    agent_training_completed = 49.8

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
    try:
        if os.path.exists(saved_weights_location):
            model.load_weights(saved_weights_location)
            model_target.load_weights(saved_weights_location)
            print('weights Loaded!!')
    except:
        print("!!! Faced an error while Loading weights!!!")

    # Let the Training Begin
    times_epsilon = 1
    epsilon = max_epsilon
    start_epsilon = 0.6
    games_without_epsilon = 12
    while agent_training_completed <= win_all_time: # Will break when trained.

        # # To gracefully exit and control other parameters during Runtime.
        try:
            with open('./taining_control.json', 'r') as file:
                data = file.read()
                controls = json.loads(data)
                if controls["stop_training"] == 1:
                    print('Stopping traning Gracefully...')
                    return;
                start_epsilon = controls["start_epsilon"]
                games_without_epsilon = controls["games_without_epsilon"]
        except:
            pass


        # set running reward = 0
        running_reward = 0

        min_epsilon = np.random.choice([0.0,0.05,0.01],1,p=[0.4,0.3,0.3])[0]
        # update target model
        tn = int(epsilon_check_after_games/update_after_games)
        for i in range(tn):
            epsilon_target = int(tn - games_without_epsilon)
            epsilon = (start_epsilon - (i*start_epsilon/epsilon_target))
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
                # Complete the new game that is started.
                buffer_state_local = [];
                buffer_action_local = [];
                buffer_reward_local = [];
                buffer_next_state_local = [];
                buffer_done_local = [];
                while done == False:
                    action = -1
                    if epsilon > np.random.rand(1)[0]:
                        # Take random Valid action
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

                    #Learn from loss for positive rewards.
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
                    if np.random.rand(1)[0] <= draw_percentage_games:
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

            loss_val_sum = 0
            loss_avg = 0
            loss_count = 0
            # Training
            for _ in range(timesToSample):
                #break;
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
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    loss_count = loss_count + 1
                    loss_val_sum = loss_val_sum  + loss.numpy()


            # clear up memory
            if len(replay_buffer_action) > max_memory_length:
                del replay_buffer_action[:int(len(replay_buffer_action)/5)]
                del replay_buffer_done[:int(len(replay_buffer_done)/5)]
                del replay_buffer_next_state[:int(len(replay_buffer_next_state)/5)]
                del replay_buffer_reward[:int(len(replay_buffer_reward)/5)]
                del replay_buffer_state[:int(len(replay_buffer_state)/5)]

            # if running reward is > 100 updated training completed.
            if running_reward > 100:
                agent_training_completed += 1



            # Log details
            if loss_count > 0:
                loss_avg = loss_val_sum / loss_count
            delim = "\n"
            loggValues = {};
            loggValues["r"] = running_reward;
            loggValues["g"] = total_games;
            loggValues["d"] = total_draws - total_draws_bp ;
            loggValues["w"] = total_wins - total_wins_bp ;
            loggValues["l"] = total_loss - total_loss_bp;
            loggValues["td"] = total_draws;
            loggValues["tw"] = total_wins;
            loggValues["tl"] = total_loss;
            loggValues["l_"] = loss_avg*10000000;
            loggValues["e"] = epsilon;
            loggValues["a"] = len(buffer_action_ts);
            loggValues["s"] = len(replay_buffer_action);
            loggValues["t"] = datetime.now().isoformat();
            loggValues["p"] = agent_training_completed*100/win_all_time;
            with open(logjsonname, 'a') as f:
                json.dump(loggValues, f)
                f.write(delim)

            #saving the model
            model.save_weights(saved_weights_location)
            total_draws_bp = total_draws
            total_wins_bp = total_wins
            total_loss_bp = total_loss
            # Time to Update the Target Model
            # update the the target network with new weights
            if(total_games%update_target_network == 0):
                model_target.set_weights(model.get_weights())



            #saving the model
            if total_games%save_weights_after == 0:
                model.save_weights(saved_weights_dir + file_name + '_' +str(total_games)+ '_' + str(datetime.now().strftime(DTTM_FORMAT)) + file_extension)

# Networks
def create_q_model(state_shape, total_actions):
    """ Create a Q model"""
    # input layer
    inputs = layers.Input(shape=state_shape)

    # Hidden layers
    layer1 = layers.Dense(40, activation="relu")(inputs)
    layer2 = layers.Dense(40, activation="relu")(layer1)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)

"""
# Networks
def create_q_model6(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)
    initializer1 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)
    initializer2 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.4)
    layer1 = layers.Dense(144, activation="relu", kernel_initializer=initializer2 )(inputs)
    layer2 = layers.Dense(840, activation="relu", kernel_initializer=initializer2)(layer1)
    # (n_samples, height, width, channels)
    #layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    #initializer1 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.4)
    #layer1 =  layers.Conv2D(16, 2, strides=1, activation="relu", kernel_initializer=initializer1)(layer0)
    #layer12 = layers.Conv2D(16, 2, strides=1, activation="relu")(layer1)
    #layer13 = layers.Conv2D(16, 2, strides=1, activation="relu")(layerl2)
    #layer13 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer12)
    #layer12 = layers.Conv2D(32, 2, strides=1, activation="relu")(layer1)

    #layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    #layer4 = layers.Concatenate()([layers.Flatten()(layer1),layers.Flatten()(layer12),layers.Flatten()(layer13), layers.Flatten()(layer0)])
    #layer4 = layers.Flatten()(layer1)
    # Hidden layers
    #layer5 = layers.Dense(333, activation="relu" )(layer4)
    #layer6 = layers.Dense(123, activation="relu")(layer5)
    #layer7 = layers.Dense(77, activation="relu")(layer6)
    #layer7 = layers.Dense(77, activation="relu", kernel_initializer=initializer2)(layer6)
    #layer6 = layers.Dense(300, activation="relu")(layer5)
    #layer6 = layers.Dense(207, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)

# Networks
def create_q_model5(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # (n_samples, height, width, channels)
    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    initializer1 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.4)
    layer1 =  layers.Conv2D(16, 2, strides=1, activation="relu", kernel_initializer=initializer1)(layer0)
    #layer12 = layers.Conv2D(16, 2, strides=1, activation="relu")(layer1)
    #layer13 = layers.Conv2D(16, 2, strides=1, activation="relu")(layerl2)
    #layer13 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer12)
    #layer12 = layers.Conv2D(32, 2, strides=1, activation="relu")(layer1)

    #layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    #layer4 = layers.Concatenate()([layers.Flatten()(layer1),layers.Flatten()(layer12),layers.Flatten()(layer13), layers.Flatten()(layer0)])
    layer4 = layers.Flatten()(layer1)
    # Hidden layers
    layer5 = layers.Dense(333, activation="relu" )(layer4)
    layer6 = layers.Dense(123, activation="relu")(layer5)
    layer7 = layers.Dense(77, activation="relu")(layer6)
    #layer7 = layers.Dense(77, activation="relu", kernel_initializer=initializer2)(layer6)
    #layer6 = layers.Dense(300, activation="relu")(layer5)
    #layer6 = layers.Dense(207, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer7)

    return keras.Model(inputs=inputs, outputs=action)

# Networks
def create_q_model4(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # (n_samples, height, width, channels)
    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    initializer1 = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.3)
    layer1 =  layers.Conv2D(16, 2, strides=1, activation="relu", kernel_initializer=initializer1)(layer0)
    #layer12 = layers.Conv2D(16, 2, strides=1, activation="relu")(layer1)
    #layer13 = layers.Conv2D(16, 2, strides=1, activation="relu")(layerl2)
    #layer13 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer12)
    #layer12 = layers.Conv2D(32, 2, strides=1, activation="relu")(layer1)

    #layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    #layer4 = layers.Concatenate()([layers.Flatten()(layer1),layers.Flatten()(layer12),layers.Flatten()(layer13), layers.Flatten()(layer0)])
    layer4 = layers.Flatten()(layer1)
    # Hidden layers
    initializer2 = tf.keras.initializers.RandomNormal(mean=0., stddev=1.0)
    layer5 = layers.Dense(700, activation="relu" , kernel_initializer=initializer2)(layer4)
    layer6 = layers.Dense(300, activation="relu", kernel_initializer=initializer2)(layer5)
    #layer7 = layers.Dense(77, activation="relu", kernel_initializer=initializer2)(layer6)
    #layer6 = layers.Dense(300, activation="relu")(layer5)
    #layer6 = layers.Dense(207, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer6)

    return keras.Model(inputs=inputs, outputs=action)

# Networks
def c3reate_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # (n_samples, height, width, channels)
    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    layer1 =  layers.Conv2D(24, 2, strides=1, activation="relu")(layer0)
    layer12 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer1)
    layer13 = layers.Conv2D(12, 2, strides=1, activation="relu")(layer12)
    #layer12 = layers.Conv2D(32, 2, strides=1, activation="relu")(layer1)

    #layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    layer4 = layers.Concatenate()([layers.Flatten()(layer1),layers.Flatten()(layer12),layers.Flatten()(layer13), layers.Flatten()(layer0)])
    #layer4 = layers.Flatten()(layer1)
    # Hidden layers
    layer5 = layers.Dense(397, activation="relu")(layer4)
    layer6 = layers.Dense(79, activation="relu")(layer5)
    #layer6 = layers.Dense(300, activation="relu")(layer5)
    #layer6 = layers.Dense(207, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer6)

    return keras.Model(inputs=inputs, outputs=action)

# Networks
def cool2_create_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # (n_samples, height, width, channels)
    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    layer1 = layers.Conv2D(24, 2, strides=1, activation="relu")(layer0)
    layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    layer4 = layers.Concatenate()([layers.Flatten()(layer1),layers.Flatten()(layer2),layers.Flatten()(layer3)])
    # Hidden layers
    layer5 = layers.Dense(45, activation="relu")(layer4)
    layer6 = layers.Dense(45, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer6)

    return keras.Model(inputs=inputs, outputs=action)

def cool_create_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # (n_samples, height, width, channels)
    layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)
    # 24 filters , 2x2 size.
    layer1 = layers.Conv2D(24, 2, strides=1, activation="relu")(layer0)
    layer12 = layers.Conv2D(24, 2, strides=1, activation="relu")(layer1)
    layer13 = layers.Conv2D(24, 2, strides=1, activation="relu")(layer12)
    layer2 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer0)
    layer21 = layers.Conv2D(24, 2, strides=1, activation="relu")(layer2)
    layer3 = layers.Conv2D(40, 4, strides=1, activation="relu")(layer0)
    #layer3 = layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
    layer4 = layers.Concatenate()(layer13,layer21,layer3)
    # Hidden layers
    layer5 = layers.Dense(45, activation="relu")(layer4)
    layer6 = layers.Dense(45, activation="relu")(layer5)
    #layer3 = layers.Dense(200, activation="relu")(layer6)
    #layer4 = layers.Dense(45, activation="relu")(layer3)
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer
    action = layers.Dense(total_actions, activation="linear")(layer6)

    return keras.Model(inputs=inputs, outputs=action)
"""

if __name__=="__main__":
    main()
