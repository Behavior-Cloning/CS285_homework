import gym

env=gym.make("MountainCar-v0")
obs=env.reset()

while True:
    next_obs,reward,done,_=env.step(0)
    print((next_obs,reward,done))
    if done:
        break
