import numpy as np
import matplotlib.pyplot as plt
from environment.frozen_lake_v2 import create_environment

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
    # Render the initial environment start
    environment.render()

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(qtable[state, :])

        result = environment.step(action)
        new_state, reward, done = result[0], result[1], result[2]
        # Check data extraction
        new_state = new_state[0] if isinstance(new_state, tuple) else new_state
        # Show environment after each action/step
        environment.render()

        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state

    # Decay epsilon
    epsilon = max(epsilon - epsilon_decay, 0)


# Output the Q-table
print("Q-table after training:")
print(qtable)

# Plot the outcomes
successes = [1 if outcome == "Success" else 0 for outcome in outcomes]
plt.plot(successes)
plt.title('Outcomes of Episodes')
plt.xlabel('Episode')
plt.ylabel('Success (1) or Failure (0)')
plt.show()



