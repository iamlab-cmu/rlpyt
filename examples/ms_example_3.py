
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm. Can choose between configurations for use of CPU/GPU for sampling
(serial or parallel) and optimization (serial).

Alternating sampler is another option.  For recurrent agents, a different mixin
is required for alternating sampling (see rlpyt.agents.base.py), feedforward agents
remain unaffected.

"""
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.algos.pg.a2c import A2C
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

from rlpyt.agents.pg.ms_isaac import IsaacFfAgent
from rlpyt.samplers.serial.ms_vec_sampler import MS_VecSerialSampler
from rlpyt.samplers.collections import MS_VecTrajInfo
from rlpyt.envs.isaac_gym import MS_BaseVecEnv

import os
import sys
sys.path.append('/home/mohit/projects/force_control/rl-envs')


def build_and_train(args, 
                    game="pong", 
                    run_ID=0, 
                    cuda_idx=None, 
                    sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    # if sample_mode == "serial":
    #     Sampler = SerialSampler  # (Ignores workers_cpus.)
    #     print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    # elif sample_mode == "cpu":
    #     Sampler = CpuSampler
    #     print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    # elif sample_mode == "gpu":
    #     Sampler = GpuSampler
    #     print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")
    # elif sample_mode == "alternating":
    #     Sampler = AlternatingSampler
    #     affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
    #     affinity["alternating"] = True  # Sampler will check for this.
    #     print(f"Using Alternating GPU parallel sampler, {gpu_cpu} for sampling and optimizing.")


    from envs.block2d import GymBlock2DFittingVecEnv
    env_klass = GymBlock2DFittingVecEnv
    cfg = '/home/mohit/projects/force_control/rl-envs/cfg/run_block2d_fit.yaml'

    sampler = MS_VecSerialSampler(
        EnvCls=MS_BaseVecEnv,
        TrajInfoCls=MS_VecTrajInfo,
        env_kwargs=dict(env_klass=env_klass, cfg=cfg),
        batch_T=600,  # 5 time-steps per sampler iteration.
        batch_B=4,  # 16 parallel environments.
        max_decorrelation_steps=0,
    )
    algo = A2C()  # Run with defaults.
    agent = IsaacFfAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "a2c_" + "2d_fitting"
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    print(f"Will save results to logdir: {args.logdir}")

    with logger_context(args.logdir, run_ID, name, config, 
                        use_summary_writer=True,
                        snapshot_mode="gap"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', type=str, required=True, help='Logdir to save results')
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'cpu', 'gpu', 'alternating'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        args,
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
    )
