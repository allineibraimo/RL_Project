import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym

# Create the environment with the default "4x4" Frozen Lake map
def create_environment():
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    return env

env = create_environment()

# Policy Evaluation function
def policy_evaluation(policy, env, gamma, theta=1e-9):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            v = 0
            for action, action_prob in enumerate(policy[state]):
                for probability, next_state, reward, done in env.unwrapped.P[state][action]:
                    v += action_prob * probability * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[state] - v))
            V[state] = v
        if delta < theta:
            break
    return V

# Policy Improvement
def policy_improvement(V, env, gamma):
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for state in range(env.observation_space.n):
        Q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for probability, next_state, reward, done in env.unwrapped.P[state][action]:
                Q[action] += probability * (reward + gamma * V[next_state])
        best_action = np.argmax(Q)
        policy[state, :] = 0
        policy[state, best_action] = 1.0
    return policy

# Policy Iteration
def policy_iteration(env, gamma=0.9):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation(policy, env, gamma)
        new_policy = policy_improvement(V, env, gamma)
        if np.all(new_policy == policy):
            break
        policy = new_policy
    return policy, V

# policy iteration
optimal_policy, V = policy_iteration(env)


print("State-Value Function:")
print(V.reshape((4, 4)))
print("\nOptimal Policy :")
print(optimal_policy)

# heatmaps
def plot_policy_and_value(V, policy):
    actions = ['<', 'v', '>', '^']
    policy_arrows = np.array([actions[np.argmax(policy[state])] for state in range(env.observation_space.n)]).reshape(4,4)

    plt.figure(figsize=(8, 8))
    sns.heatmap(V.reshape((4, 4)), annot=True, cmap="YlGnBu", cbar=False, square=True, fmt=".2f")
    plt.title('State-Value Function')
    plt.show()

    plt.figure(figsize=(8, 8))
    sns.heatmap(V.reshape((4, 4)), annot=policy_arrows, cmap="YlGnBu", cbar=False, fmt='', square=True)
    plt.title('Optimal Policy')
    plt.show()

plot_policy_and_value(V, optimal_policy)