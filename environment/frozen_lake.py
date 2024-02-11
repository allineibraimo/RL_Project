import gymnasium as gym

MAX_ITERATIONS = 100    #changeable

env = gym.make('FrozenLake-v1', desc=["SFFH", "FHFF", "FFFH", "FHFG"], map_name="4x4", is_slippery=False)
env.reset()
# render the environment
env.render()

#this accesses the evironment space
env.observation_space

#this accesses action space (down - 0, left - 1, right - 2, up - 3)
env.action_space

# generate random action
randomAction = env.action_space.sample()
returnValue = env.step(randomAction)

env.render()







