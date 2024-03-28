import numpy as np
import matplotlib.pyplot as plt
import gym


class ExpectedSarsaAgent:
    def __init__(self, num_states, num_actions, epsilon, alpha, gamma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))

    def expected_q(self, next_state):
        probabilities = np.ones(self.num_actions) * self.epsilon / self.num_actions
        best_action = np.argmax(self.Q[next_state])
        #print(best_action)
        probabilities[best_action] += (1.0 - self.epsilon)
        expected_Q = np.sum(probabilities * self.Q[next_state])
        return expected_Q
        
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon :
            return np.random.choice(self.num_actions)
        else :
            return np.argmax(self.Q[state])
            
             
    def train(self, env, num_episodes):
        rewards_per_episode = np.zeros(num_episodes)  # Array to store the total reward per episode
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0  # Variable to keep track of the total reward in this episode
            
            while not done:
                a = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(a)
                
                total_reward += reward  # Increment the total reward by the reward received
                
                expected_Q = self.expected_q(next_state)
                self.Q[state][a] += self.alpha * (reward + self.gamma * expected_Q - self.Q[state][a])
                state = next_state

            rewards_per_episode[ep] = total_reward  # Store the total reward for this episode
            
        return self.Q, rewards_per_episode

import sarsa

# # Hyperparameters
# epsilon = 0.9
# alpha = 0.5
# gamma = 0.9
# num_episodes = 1000

# # Initialize our frozen lake environment from frozen_lake.py
# environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# # Reset our environment
# environment.reset()

# expected_sarsa = ExpectedSarsaAgent(environment.observation_space.n, environment.action_space.n, epsilon, alpha, gamma)
# sarsa = sarsa.SarsaAgent(environment.observation_space.n, environment.action_space.n, epsilon, alpha, gamma)

# plt.figure(figsize=(12, 8))

# # Train the agent
# Q, rewards_per_ep_exsarsa = expected_sarsa.train(environment, num_episodes)
# cumulative_rew_expsar = np.cumsum(rewards_per_ep_exsarsa)
# plt.plot(cumulative_rew_expsar, label='Expected SARSA', color='green')
# Q, rewards_per_ep_sarsa = sarsa.train(environment, num_episodes)
# cumulative_rew_sar = np.cumsum(rewards_per_ep_sarsa)
# plt.plot(cumulative_rew_sar, label='SARSA', color='red')


# # # Plot rewards per episode
# # plt.plot(rewards_per_ep)
# # plt.title('Rewards per Episode')
# # plt.xlabel('Episode')
# # plt.ylabel('Total Reward')
# # plt.show()


# plt.xlabel('Episodes')
# plt.ylabel('Cumulative Reward')
# plt.title('Cumulative Reward Per Episode Comparison')
# plt.legend()
# plt.grid(True)
# plt.savefig('Expected_SARSA_vs_SARSA_results.png')
# plt.show()
# #plt.savefig('Expected_SARSA_vs_SARSA_results.png')

# # After training both agents, calculate mean and variance
# mean_rewards_exsarsa = np.mean(rewards_per_ep_exsarsa)
# variance_rewards_exsarsa = np.var(rewards_per_ep_exsarsa)

# mean_rewards_sarsa = np.mean(rewards_per_ep_sarsa)
# variance_rewards_sarsa = np.var(rewards_per_ep_sarsa)

# # Print out the results
# print(f"Expected SARSA Mean Rewards: {mean_rewards_exsarsa}, Variance: {variance_rewards_exsarsa}")
# print(f"SARSA Mean Rewards: {mean_rewards_sarsa}, Variance: {variance_rewards_sarsa}")


# # After training, you can derive the optimal policy from Q
# optimal_policy = np.argmax(Q, axis=1)

# # Display the optimal policy
# print("Optimal policy:")
# print(optimal_policy)

# environment.close()
