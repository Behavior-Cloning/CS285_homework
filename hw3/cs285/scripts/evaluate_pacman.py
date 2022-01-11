import torch
import gym
import torch.nn as nn
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.infrastructure.atari_wrappers import wrap_deepmind
from gym import wrappers
import time


env = gym.make("MsPacman-v0")
env = wrappers.Monitor(
                env,
                "E:\预习研究生\强化学习\homework_fall2020-master\hw3\cs285\scripts\../data\hw3_q1_MsPacman-v0_03-09-2021_17-23-59",
                force=True,
                video_callable=False,
            )
env = wrap_deepmind(env)
# env.seed(1)

q_net=torch.load("agent_itr_1200000.pt")
replay_buffer=MemoryOptimizedReplayBuffer(10000,4)
replay_buffer_idx=None
ptu.init_gpu()

def qa_values(obs):
    obs = ptu.from_numpy(obs)
    qa_values = q_net(obs)
    return ptu.to_numpy(qa_values)

def get_action(obs):
    if len(obs.shape) > 3:
        observation = obs
    else:
        observation = obs[None]

    qa = ptu.from_numpy(qa_values(observation))
    _, actions = qa.max(dim=1)

    return ptu.to_numpy(actions.squeeze())



last_obs=env.reset()
# print(last_obs)
print(last_obs.shape)

while True:
    replay_buffer_idx = replay_buffer.store_frame(last_obs)

    obs = replay_buffer.encode_recent_observation()
    action=get_action(obs)

    env.render()
    time.sleep(0.15)
    obs, reward, done, info = env.step(action)
    last_obs=obs

    replay_buffer.store_effect(replay_buffer_idx, action, reward, done)

    if done:
        break


env.close()
