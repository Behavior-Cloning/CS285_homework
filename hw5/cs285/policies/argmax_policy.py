import numpy as np
import pdb
from cs285.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        qa_values = ptu.from_numpy(self.critic.qa_values(observation))
        _, actions = qa_values.max(dim=1)

        return ptu.to_numpy(actions.squeeze())
        # TODO: get this from hw3

    ####################################
    ####################################