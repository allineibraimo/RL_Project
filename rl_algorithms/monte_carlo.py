# monte carlo pseudocode
# for all s in S and a in A
# Q(s, a) <- arbitrary
# pi(s) <- arbitrary
# repeat for n_episodes :
# generate episode following e-greedy policy
# Q(s, a) <- evaluate policy using first-visit MC method
# pi <- improve policy greedily

# Q*(s, a) <- Q(s, a)
# pi* <- pi

import gym
import numpy as np
import time

# for testing purposes -
def render_single(env, policy, max_steps=100):
    # Renders policy for an enivronment

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, reward, done, _ = env.step(a)
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

# initialize agente especial

class MCControl:
    def __init__(self, env, num_states, num_actions, epsilon, gamma):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma 
    
    def run_mc_control(self, num_episodes, verbose=True):
        self.init_agent()
        rewards_per_episode = np.array([None] * num_episodes)
        episode_len = np.array([None] * num_episodes)

        for episode in range(num_episodes):
            state_action_reward = self.generate_episode(self.policy)
            G = self.calculate_returns(state_action_reward)
            self.evaluate_policy(G)
            self.improve_policy()

            total_return = 0
            for _, _, reward in state_action_reward:
                total_return += reward
            rewards_per_episode[episode] = total_return
            episode_len = len(state_action_reward)

        # Once training is finished, calculate final policy using argmax approach
        final_policy = self.argmax(self.Q, self.policy)

        if verbose:
            print(f"Finished training agent in {num_episodes} episodes")

        return self.Q, final_policy, rewards_per_episode, episode_len
    
    def init_agent(self):
        self.policy = np.random.choice(self.num_actions, self.num_states)
        self.Q = {}
        self.visit_count = {}

        for state in range(self.num_states):
            self.Q[state] = {}
            self.visit_count[state] = {}
            for action in range(self.num_actions):
                self.Q[state][action] = 0
                self.visit_count[state][action] = 0

    def generate_episode(self, policy):
        G = 0
        s = self.env.reset()
        #print(s)
        a = policy[s[G]]
        #print(a)

        state_action_reward = [(s, a, 0)]
        while True:
            #print(self.env.step(a))
            state, reward, terminated, _, _= self.env.step(a)
            if terminated:
                state_action_reward.append((state, None, reward))
                break
            else:
                a = policy[s[G]]
                state_action_reward.append((state, a, reward))
        
        return state_action_reward
    
    def calculate_returns(self, state_action_reward):
        G = {}
        t = 0
        for state, action, reward in state_action_reward:
            if state not in G:
                G[state] = {action: 0}
            else: 
                if action not in G[state]:
                    G[state][action] = 0
            
            for s in G.keys():
                for a in G[s].keys():
                    G[s][a] += reward * self.gamma ** t

            t += 1
        
        return G
    
    def evaluate_policy(self, G):
        for state in G.keys():
            for action in G[state].keys():
                if action:
                    self.visit_count[state][action] += 1
                    #update Q state and pair - use incremental mean approach to update action value function Q(s,a) after each episode: N(s, a) = N(s, a) + 1
                    # -> Q(s,a) = Q(s,a) + (1 / N(s,a))(G_t - Q(s,a))
                    temp = G[state][action] - self.Q[state][action]
                    self.Q[state][action] += temp / self.visit_count[state][action]


    def improve_policy(self):
        self.policy = self.argmax(self.Q, self.policy)
        for state in range(self.num_states):
            self.policy[state] = self.epsilon_greedy_action(self.policy[state])


    def argmax(self, Q, policy):
        next_policy = policy

        for state in range(self.num_states):
            best_action = None
            best_value = float('-inf')
            for action, value in Q[state].items():
                if value > best_value:
                    best_value = value
                    best_action = action
            next_policy[state] = best_action

        return next_policy
    
    def epsilon_greedy_action(self, greedy_action):
        temp = np.random.random()

        if temp < 1 - self.epsilon:
            return greedy_action
    
        return np.random.randint(0, self.num_actions)

import numpy as np

np.random.seed(1)

epsilon = 0.8
gamma = 1.0
n_episodes = 1000

env = gym.make('FrozenLake-v1', desc=["SFFH", "FHFF", "FFFH", "FHFG"], map_name="4x4", is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n

mc_model = MCControl(env, num_states, num_actions, epsilon, gamma)

Q, policy, _, _ = mc_model.run_mc_control(n_episodes)






