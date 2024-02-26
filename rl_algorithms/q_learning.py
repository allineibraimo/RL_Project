import numpy as np
import matplotlib.pyplot as plt
from environment.frozen_lake import create_environment

# Initialize our frozen lake environment from frozen_lake.py
environment = create_environment()

# Reset our environment
environment.reset()

# Initialize the Q-table to zeros
qtable = np.zeros((16, 4))  # Adjust this based on your actual environment's spaces

# Hyperparameters
episodes = 1000
alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.001

# List to store outcomes
outcomes = []

# Training process
for episode in range(episodes):
    state_info = environment.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False
    total_reward = 0

    environment.render()  # Render the initial state of the environment

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(qtable[state, :])

        result = environment.step(action)
        new_state, reward, done = result[0], result[1], result[2]
        # Ensure new_state is correctly extracted if it's part of a complex structure
        new_state = new_state[0] if isinstance(new_state, tuple) else new_state

        environment.render()  # Render the environment after each action

        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state  # Update state for the next iteration
        total_reward += reward

        print(f"Episode {episode + 1} completed with total reward: {total_reward}")
        if done:
            if reward == 1.0:  # Assuming a reward of 1 indicates success
                outcomes.append(1)  # Append 1 for success
                print("Outcome: Success")
            else:
                outcomes.append(0)  # Append 0 for failure
                print("Outcome: Failure")

    # Decay epsilon outside the inner loop
    epsilon = max(epsilon - epsilon_decay, 0)


# Output the Q-table after training
print("Q-table after training:")
print(qtable)

# Plotting the outcomes of episodes
# Convert outcomes to a numeric format for plotting
numeric_outcomes = [1 if outcome == "Success" else 0 for outcome in outcomes]

plt.figure(figsize=(12, 6))
plt.plot(outcomes, '-o', label='Success (1) / Failure (0)')
plt.title('Episode Outcomes Over Time')
plt.xlabel('Episode Number')
plt.ylabel('Outcome (Success=1, Failure=0)')
plt.ylim(-0.1, 1.1)  # Ensure y-axis slightly extends beyond 0 and 1 for visibility
plt.legend()
plt.show()


