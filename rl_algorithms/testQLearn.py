from environment.frozen_lake_v2 import create_environment

env = create_environment()
state = env.reset()
print(state)

action = env.action_space.sample()
result = env.step(action)
print(result)