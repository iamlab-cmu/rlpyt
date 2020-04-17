import numpy as np
import os

from autolab_core import YamlConfig

from rlpyt.envs.base import Env
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.utils.quick_args import save__init__args
from rlpyt.envs.gym import info_to_nt
from rlpyt.envs.base import EnvSpaces, EnvStep

class MS_BaseVecEnv(Env):

    def __init__(self, 
                 env_klass=None,
                 cfg=None,
                 act_null_value=0,
                 obs_null_value=0,
                 force_float32=True):
        save__init__args(locals(), underscore=True)
        # super(MS_BaseVecEnv, self).__init__()

        assert cfg is not None and os.path.exists(cfg), "Config does not exist"
        cfg = YamlConfig(cfg)
        assert env_klass is not None
        vec_env = env_klass(cfg)
        self._gym_vec_env = vec_env

        # This allows our policy network to output values beyond the maximum.
        vec_env.set_auto_clip_actions(True)
        vec_env.set_auto_reset_after_done(False)

        # Wrappers around gym action and observation sapces.
        self._action_space = GymSpaceWrapper(
            space=vec_env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=True,
        )
        self._observation_space = GymSpaceWrapper(
            space=vec_env.obs_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=True,
        )
        self.is_vec_env = True
    
    @property
    def n_envs(self):
        return self._gym_vec_env.n_envs

    @property
    def action_space(self):
        # return self._gym_vec_env.action_space
        return self._action_space
    
    def batch_action_space_sample(self):
        n_envs = self._gym_vec_env.n_envs
        a = np.array(
            [self._gym_vec_env.action_space.sample() for _ in range(n_envs)])
        return a

    @property
    def observation_space(self):
        # return self._gym_vec_env.obs_space
        return self._observation_space

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self._observation_space,
            action=self._action_space,
        )
    
    def reset(self):
        return self._observation_space.convert(self._gym_vec_env.reset())
    
    def reset_env_idxs(self, env_idxs):
        # This would return the observations for all environments. This should
        # be fine since the observations for environments that haven't finished
        # yet should not change unless we actually simulate anything.
        o = self._gym_vec_env.reset(env_idxs=env_idxs)
        return self._observation_space.convert(o)

    @property
    def horizon(self):
        return self._gym_vec_env.max_steps

    def step(self, actions):
        assert actions.shape[0] == self._gym_vec_env.n_envs

        # Convert actions data from rlpyt format to gym vec env format?
        a = self._action_space.revert(actions)
        o, r, d, info = self._gym_vec_env.step(a)
        # Convert the above returned values into action env
        obs = self._observation_space.convert(o)

        # TODO(Mohit): Implement timelimit wrappers around env.

        # info = [info_to_nt(info_i) if info_i is not None else None 
        #         for info_i in info]
        assert info[0] is None or len(info[0]) == 0, \
            "Info should not contain anything for now."
        info = [None for _ in range(self._gym_vec_env.n_envs)]
        if isinstance(r, float):
            r = np.dtype("float32").type(r)  # Scalar float32.
        return EnvStep(obs, r, d, info)

    def render(self):
        self._gym_vec_env.render()