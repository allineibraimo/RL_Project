
import numpy as np

# initialize agente especial

class MCControl:
    #initialize all the parameters of the mc control
    def __init__(self, env, epsilon, gamma):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma 
        self.Q = {}
        self.policy = {}
        self.returns = {}  # Used to keep track of returns for state-action pairs
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        
        for s in range(self.num_states):
            self.Q[s] = {}
            self.returns[s] = {}
            self.policy[s] = np.ones(self.num_actions, dtype=float) / self.num_actions
            for a in range(self.num_actions):
                self.Q[s][a] = 0.0
                self.returns[s][a] = []
    
    # randomly choose actions according to state in policy and
    def choose_action(self, state):
        action_probabilities = self.policy[state]
        action = np.random.choice(self.num_actions, p=action_probabilities)
        return action
        

    def generate_episode(self):
        episode = []
        state, _ = self.env.reset()
        for t in range(50):
            action = self.choose_action(state)
            # print(self.env.step(action))
            next_state, reward, done, _ , _= self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode
    
    
    def run(self, num_episodes, verbose=True):
        rewards_per_episode = np.zeros(num_episodes)
        decay_rate = 0.999
        min_epsilon = 0.1
        for episode_num in range(1, num_episodes + 1):
            # Decay epsilon over episodes
            self.epsilon = max(self.epsilon * decay_rate, min_epsilon)
            episode = self.generate_episode()
            G = 0
            visited_state_action_pairs = set()
            
            for state, action, reward in reversed(episode):
                if (state, action) not in visited_state_action_pairs:
                    visited_state_action_pairs.add((state, action))
                    G = self.gamma * G + reward
                    self.returns[state][action].append(G)
                    # Average the returns for state-action pair
                    self.Q[state][action] = np.mean(self.returns[state][action])
                    # Update the policy to be epsilon-greedy
                    for state in range(self.num_states):
                        A_star = np.argmax([self.Q[state][a] for a in range(self.num_actions)])
                        action_probabilities = np.ones(self.num_actions) * self.epsilon / self.num_actions
                        action_probabilities[A_star] += (1.0 - self.epsilon)
                        self.policy[state] = action_probabilities
                        for a in range(self.num_actions):
                            if a == A_star:
                                self.policy[state][a] = (1 - self.epsilon) + (self.epsilon / self.num_actions)
                            else:
                                self.policy[state][a] = (self.epsilon / self.num_actions)

            # Calculate total reward from the episode
            rewards_per_episode[episode_num - 1] = sum([r for _, _, r in episode])
            
                                    
            if verbose and episode_num % 10 == 0:  # Print every 10 episodes
                print(f"Episode {episode_num}/{num_episodes} complete.")

        # Once training is finished, convert Q to deterministic policy
        final_policy = {s: np.argmax([self.Q[s][a] for a in range(self.num_actions)]) for s in range(self.num_states)}

        return final_policy, self.Q, rewards_per_episode


import numpy as np
import gym
from matplotlib import pyplot as plt
import expectedsarsa

# Initialize our frozen lake environment from frozen_lake.py
environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

# Reset our environment
environment.reset()

epsilon = 1.0
gamma = 0.9
n_episodes = 1000

alpha = 0.5

mc_model = MCControl(environment, epsilon, gamma)
ex_sarsa_model = expectedsarsa.ExpectedSarsaAgent(environment.observation_space.n, environment.action_space.n, epsilon, alpha, gamma)

plt.figure(figsize=(12, 8))

Q, policy, rewards_per_episode_mc = mc_model.run(n_episodes, verbose=True)
cumulative_rew_mc = np.cumsum(rewards_per_episode_mc)
plt.plot(cumulative_rew_mc, label='First Visit Monte Carlo', color='green')
Q, rewards_per_episode_exsars = ex_sarsa_model.train(environment, n_episodes)
cumulative_rew_expsar = np.cumsum(rewards_per_episode_exsars)
plt.plot(cumulative_rew_expsar, label='Expected SARSA', color='blue')

# Plot rewards per episode
# plt.plot(rewards_per_episode)
# plt.title('Rewards per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.grid(True)
# plt.show()

plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Per Episode Comparison')
plt.legend()
plt.grid(True)
plt.savefig('Monte_Carlo_vs_Expected_SARSA.png')
plt.show()

mean_rewards_mc = np.mean(rewards_per_episode_mc)
variance_rewards_mc = np.var(rewards_per_episode_mc)

mean_rewards_exsarsa = np.mean(rewards_per_episode_exsars)
variance_rewards_exsarsa = np.var(rewards_per_episode_exsars)

# Print out the results
print(f"Expected SARSA Mean Rewards: {mean_rewards_exsarsa}, Variance: {variance_rewards_exsarsa}")
print(f"MC Mean Rewards: {mean_rewards_mc}, Variance: {variance_rewards_mc}")


environment.close()



