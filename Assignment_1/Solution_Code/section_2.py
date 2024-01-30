from datetime import datetime
import time
import gym
import numpy as np
import random
from collections import deque

from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import HeUniform
from keras import backend as K

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

LEARNING_RATE = 0.005
MIN_LEARNING_RATE = 1e-10
LEARNING_RATE_DECAY = 0.995
EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
EPISODES = 2000
STEPS = 500
REPLAY_SIZE = 20000
BATCH_SIZE = 64
GAMMA = 0.95
N_HIDDEN_LAYERS = 3
# N_HIDDEN_LAYERS = 5
C = 2

writer = tf.summary.create_file_writer('./logs')


def generate_model(observation_space, action_space, n_hidden_layers):
    init = HeUniform()
    model = Sequential()
    if n_hidden_layers == 3:
        model.add(Dense(64, input_shape=(observation_space,), activation='relu', kernel_initializer=init))
        model.add(Dense(32, activation='relu', kernel_initializer=init))
        model.add(Dense(16, activation='relu', kernel_initializer=init))
    if n_hidden_layers == 5:
        model.add(Dense(32, input_shape=(observation_space,), activation='relu', kernel_initializer=init))
        model.add(Dense(32, activation='relu', kernel_initializer=init))
        model.add(Dense(32, activation='relu', kernel_initializer=init))
        model.add(Dense(32, activation='relu', kernel_initializer=init))
        model.add(Dense(32, activation='relu', kernel_initializer=init))

    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

    return model


# choose an action with decaying ε-greedy method
def sample_action(actions, state, epsilon, model, env):
    if np.random.rand() > epsilon:
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
    else:
        action = env.action_space.sample()

    return action


def add_replay(replay_to_add, replay):
    replay.append(replay_to_add)
    return replay


def experience_replay(replays, model, target_model, observation_space, loss_lst, total_steps):
    if len(replays) < BATCH_SIZE:
        return model, target_model, loss_lst

    sample_batch = random.sample(replays, BATCH_SIZE)

    curr_states = [s[0] for s in sample_batch]
    curr_states = np.reshape(curr_states, [BATCH_SIZE, 4])
    curr_qs_list = model.predict(curr_states, verbose=0)
    new_curr_states = np.array([s[3] for s in sample_batch])
    new_curr_states = np.reshape(new_curr_states, [BATCH_SIZE, 4])
    future_qs_list = target_model.predict(new_curr_states, verbose=0)

    for i, (observation, action, reward, new_observation, terminated) in enumerate(sample_batch):
        curr_qs_list[i][action] = reward if terminated else reward + GAMMA * np.amax(future_qs_list[i])

    model_res = model.fit(np.array(curr_states), np.array(curr_qs_list), batch_size=BATCH_SIZE, verbose=0)
    curr_loss = model_res.history['loss'][0]
    loss_lst.append(curr_loss)

    with writer.as_default():
        tf.summary.scalar('loss', curr_loss, step=total_steps)

    return model, target_model, loss_lst


def train_agent(env, observation_space, action_space, n_hidden_layers):
    replays = deque(maxlen=REPLAY_SIZE)

    epsilon = EPSILON
    learning_rate = LEARNING_RATE

    model = generate_model(observation_space, action_space, n_hidden_layers)
    target_model = generate_model(observation_space, action_space, n_hidden_layers)
    target_model.set_weights(model.get_weights())

    rewards_over_100 = deque(maxlen=100)

    loss_lst = []
    episodes_till_475 = []
    max_phase_count = 0
    max_phase = False
    rewards = []
    overall_steps = 0

    for episode in range(EPISODES):
        if len(rewards_over_100) > 0 and sum(rewards_over_100) / len(rewards_over_100) >= 475.0:
            break

        state = env.reset()
        state = np.reshape(state, [1, observation_space])

        total_rewards = 0
        total_steps_in_episode = 0
        steps_to_update_target_model = 0
        for step in range(STEPS):
            total_steps_in_episode += 1
            steps_to_update_target_model += 1
            overall_steps += 1

            action = sample_action(action_space, state, epsilon, model, env)
            new_state, reward, terminated, info = env.step(action)
            new_state = np.reshape(new_state, [1, observation_space])
            replays = add_replay((state, action, reward, new_state, terminated), replays)

            model, target_model, loss_lst = experience_replay(replays, model, target_model, observation_space, loss_lst,
                                                              overall_steps)

            state = new_state
            total_rewards += reward

            # update for the 5 layers
            # if steps_to_update_target_model == C:
            #     target_model.set_weights(model.get_weights())

            if terminated:
                rewards.append(total_rewards)
                rewards_over_100.append(total_rewards)
                avg_reward_over_100_ep = sum(rewards_over_100) / len(rewards_over_100)
                print("Episode:", episode + 1, "/", EPISODES, "Epsilon:", epsilon, "Learning rate:", learning_rate,
                      "Steps:", step + 1, "Score", total_rewards, "Avg over 100:", avg_reward_over_100_ep)
                if avg_reward_over_100_ep >= 475:
                    episodes_till_475.append(episode + 1)
                    print("######################### got avg reward of " + str(
                        avg_reward_over_100_ep) + " over 100 cosecutive episodes in episode " + str(
                        episode + 1) + " #########################")

                if not max_phase:
                    if total_steps_in_episode == 500:
                        max_phase_count += 1
                    else:
                        max_phase_count = 0
                    if max_phase_count == 2:
                        max_phase = True
                        learning_rate = MIN_LEARNING_RATE
                        K.set_value(model.optimizer.learning_rate, learning_rate)
                    elif learning_rate > MIN_LEARNING_RATE:
                        learning_rate = LEARNING_RATE_DECAY * learning_rate
                        K.set_value(model.optimizer.learning_rate, learning_rate)

                # update for the 3 layers
                if steps_to_update_target_model == C:
                    target_model.set_weights(model.get_weights())

                break

            # update epsilon
            epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)

        with writer.as_default():
            tf.summary.scalar('Reward per episode', total_rewards, step=episode + 1)

    print("agent obtains an average reward of at least 475.0 over 100 consecutive episodes: " + str(episodes_till_475))

    return model


def test_agent(env, observation_space, action_space, model):
    epsilon = 0
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        steps = 0
        terminated = False
        for step in range(STEPS):
            if not terminated:
                action = sample_action(action_space, state, epsilon, model, env)
                new_state, reward, terminated, truncated, info = env.step(action)
                new_state = np.reshape(new_state, [1, observation_space])
                state = new_state
                steps += 1


def main():
    env = gym.make('CartPole-v1')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    model = train_agent(env, observation_space, action_space, N_HIDDEN_LAYERS)
    test_agent(env, observation_space, action_space, model)


if __name__ == '__main__':
    start_time = time.time()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("--- %s start time ---" % (current_time))
    main()
    writer.close()
    print("--- %s seconds ---" % (time.time() - start_time))
