import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import seaborn as sb
import matplotlib.pyplot as plt

EPISODES = 5000
STEPS = 100
EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.1
GAMMA = 0.9995


writer = SummaryWriter()


def get_action(env, q, epsilon, state):
    if np.random.uniform(0, 1) > epsilon:
        action = np.argmax(q[state, :])
    else:
        action = env.action_space.sample()

    return action


def learning(env, q, epsilon):
    state = env.reset()
    state = state[0]
    for step in range(STEPS):
        action = get_action(env, q, epsilon, state)
        new_state, reward, terminated, truncated, info = env.step(action)
        q[state, action] = q[state, action] + LEARNING_RATE * (
                (reward + GAMMA * np.max(q[new_state, :])) - q[state, action])
        state = new_state
        if terminated:
            if reward < 1:
                return q, reward, STEPS
            break

    return q, reward, step + 1


def plot_q_table(q, steps):
    sb.heatmap(q, annot=True)
    plt.title("q table after " + str(steps) + " steps")
    plt.show()


def run_episodes(env, q):
    total_rewards = 0
    total_steps = 0
    epsilon = EPSILON
    reward_per_episode = []
    average_steps = []
    steps_over_100 = []
    for episode in range(1, EPISODES):
        q, reward, steps = learning(env, q, epsilon)
        writer.add_scalar('reward_per_episode', reward, episode)
        total_rewards += reward
        reward_per_episode.append(reward)
        steps_over_100.append(steps)
        total_steps += steps
        if episode % 100 == 0:
            writer.add_scalar('average_steps', total_steps / 100, episode)
            avg_steps = sum(steps_over_100) / 100
            average_steps.append(avg_steps)
            steps_over_100 = []
            total_steps = 0
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
        if episode == 500:
            plot_q_table(q, 500)
        if episode == 2000:
            plot_q_table(q, 2000)
    return q, reward_per_episode, average_steps


def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    q = np.zeros((state_size, action_size))
    q, reward_per_episode, average_steps_to_goal = run_episodes(env, q)
    plot_q_table(q, 5000)
    return reward_per_episode, average_steps_to_goal


if __name__ == '__main__':
    reward_per_episode, average_steps = main()
    writer.close()
