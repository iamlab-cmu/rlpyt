import numpy as np
from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict


Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
    ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
    ["action", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
    ["observation", "reward", "prev_reward", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.
    """
    __slots__ = ()

    @property
    def size(self):
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self


class MS_VecTrajInfo(AttrDict):

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, n_envs, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = np.zeros(n_envs)
        self.Return = np.zeros(n_envs)
        self.NonzeroRewards = np.zeros(n_envs)
        self.DiscountedReturn = np.zeros(n_envs)
        self._cur_discount = np.ones(n_envs)

    def step(self, observation, action, reward, done, agent_info, env_info):
        done_int = np.asarray(done, dtype=np.int32)
        not_done_int = 1 - done_int
        self.Length += not_done_int
        self.Return += (reward * not_done_int)
        self.NonzeroRewards += (reward * not_done_int) != 0
        self.DiscountedReturn += self._cur_discount * (reward * not_done_int)
        self._cur_discount *= self._discount

    def terminate(self, observation, idx):
        # Create a new traj info.
        info = TrajInfo()
        info.Length = self.Length[idx]
        info.Return = self.Return[idx]
        info.NonzeroRewards = self.NonzeroRewards[idx]
        info.DiscountedReturn = self.DiscountedReturn[idx]
        info._cur_discount = self._cur_discount[idx]

        # Now reset the old one.
        self.Length[idx] = 0
        self.Return[idx] = 0
        self.NonzeroRewards[idx] = 0
        self.DiscountedReturn[idx] = 0
        self._cur_discount[idx] = 0

        return info
