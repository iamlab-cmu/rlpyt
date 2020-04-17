import numpy as np

from rlpyt.samplers.collectors import (BaseCollector, BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)


class MS_VecDecorrelatingStartCollector(BaseCollector):
    """Collector which can step all environments through a random number of random
    actions during startup, to decorrelate the states in training batches.
    """

    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        # traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        # Create instance of MS_VecTrajInfo
        traj_infos = self.TrajInfoCls(self.envs.n_envs)

        # TODO: Should this be a list?
        observations = self.envs.reset()

        # We already have the learing dimension hence we don't need it here.
        observation = buffer_from_example(observations, 0)

        observation[:] = observations

        prev_action = np.stack([self.envs.action_space.null_value()
            for _ in range(self.envs.n_envs)])
        prev_reward = np.zeros(self.envs.n_envs, dtype="float32")

        if max_decorrelation_steps != 0:
            raise ValueError("Decorrelation not implemented yet.")

        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward

        return AgentInputs(observation, prev_action, prev_reward), traj_infos

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        # TODO(Mohit): We can reset the env each time we collect a batch of data.
        pass


class MS_VecCpuResetCollector(MS_VecDecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop (i.e.
    use in CPU or serial samplers.)

    It immediately resets any environment which finishes an episode.  This is
    typically indicated by the environment returning ``done=True``.  But this
    collector defers to the ``done`` signal only after looking for
    ``env_info["traj_done"]``, so that RL episodes can end without a call to
    ``env_reset()`` (e.g. used for episodic lives in the Atari env).  The 
    agent gets reset based solely on ``done``.
    """

    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)

            # o is the next observation
            o, r, d, env_info = self.envs.step(action)
            traj_infos.step(observation, action, r, d, agent_info, env_info)

            # TODO(Mohit): If some env in env finishes we should be able to reset
            # it and collect more samples from such an env.
            done_idx = np.where(d)[0]
            if len(done_idx) > 0:
                for traj_idx in done_idx:
                    comp_info = traj_infos.terminate(o, traj_idx)
                    completed_infos.append(comp_info)
                # Observations for envs that have not been reset remains the same. 
                o = self.envs.reset_env_idxs(done_idx.tolist())

                for traj_idx in done_idx:
                    self.agent.reset_one(idx=traj_idx)

            # update next observation, reward and action for next set of data.
            observation = o
            reward = r

            # update buffers
            env_buf.done[t] = d
            env_buf.reward[t] = reward
            if env_info[0] is not None:
                env_buf.env_info[t] = env_info
            
            agent_buf.action[t] = action
            if agent_info:
                agent_buf.agent_info[t] = agent_info
            
            self.envs.render()

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(
                obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos
