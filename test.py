import gym
env = gym.make("MsPacman-v0")
print(env.action_space.n)
env.close()
