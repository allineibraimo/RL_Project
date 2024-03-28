import numpy as np
import matplotlib.pyplot as plt
import gym

    
    
class SarsaAgent:
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon :
            return np.random.choice(self.num_actions)
        else :
            return np.argmax(self.Q[state])
             
    def train(self, env, num_episodes):
        rewards_per_episode = np.zeros(num_episodes)

        for ep in range(num_episodes):
            state, _ = env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0

            while not done:
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.choose_action(next_state)
                total_reward += reward

                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                state = next_state
                action = next_action

            rewards_per_episode[ep] = total_reward

        return self.Q, rewards_per_episode

# Hyperparameters
epsilon = 0.9
alpha = 0.5
gamma = 0.9
num_episodes = 1000

# # Initialize our frozen lake environment from frozen_lake.py
# environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

# # Reset our environment
# environment.reset()


# Train the agent
#Q, rewards_per_ep = sarsa(environment, num_episodes, epsilon, alpha, gamma)

# # Plot rewards per episode
# plt.plot(rewards_per_ep)
# plt.title('Rewards per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.show()

# # After training, you can derive the optimal policy from Q
# optimal_policy = np.argmax(Q, axis=1)

# # Display the optimal policy
# print("Optimal policy:")
# print(optimal_policy)

#environment.close()
