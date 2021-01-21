import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = 0.75
y = 0.99
num_episodes = 10000
# create lists to contain total rewards and steps per episode
steps = []
rewards = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    episode_reward = 0
    done = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state and reward from environment
        next_state, reward, done, _ = env.step(action)
        # Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + lr * (reward + y * np.max(Q[next_state, :]) - Q[state, action])
        episode_reward += reward
        state = next_state
        if done:
            break
    steps.append(j)
    rewards.append(episode_reward)

print("Episodic rewards:\n", rewards)
print("Steps per episode:\n", steps)
print("Score over time: " + str(sum(rewards) / num_episodes))
print("Final Q-Table Values:\n", Q)
print("Final actions:\n", np.argmax(Q, axis=-1))

state = env.reset()
for _ in range(1000):
    time.sleep(2)
    env.render()
    action = np.argmax(Q[state, :], axis=-1)
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()
