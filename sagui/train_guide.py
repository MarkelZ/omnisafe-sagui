import numpy as np
import omnisafe
from omnisafe.envs.sagui_envs import register_sagui_envs
from mpi4py import MPI
from robust.mpi_tools import mpi_fork


if __name__ == '__main__':
    # Experiments
    # envs = ['SafetyPointGuide0-v0', 'SafetyPointGuide1-v0', 'SafetyPointGuide2-v0']
    envs = ['SafetyPointGuide2-v0']
    algos = ['DDPGLagUnfold', 'DDPGRandAct']
    experiments = [(env_id, algo) for env_id in envs for algo in algos]

    # Number of torch threads
    TORCH_THREADS = 8

    # Number of CPUs in the current machine
    NUM_CPUS = 16

    assert NUM_CPUS % TORCH_THREADS == 0, 'The torch threads are not evenly distributed among the CPUs.'

    # Number of processes
    NUM_PROCS = int(NUM_CPUS / TORCH_THREADS)

    assert len(experiments) % NUM_PROCS == 0, 'The experiments are not evenly distributed among the MPI processes.'

    # Fork using mpi
    mpi_fork(NUM_PROCS)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Split the list of experiments into equal chunks
    experiments = np.array(experiments)
    exp_sublists = np.array_split(experiments, NUM_PROCS)

    # Select corresponding chunk of experiments
    exp_chunk = exp_sublists[rank]

    # Register sagui environments
    register_sagui_envs()

    # Create custom configurations dict
    custom_cfgs = {
        'train_cfgs': {
            'torch_threads': TORCH_THREADS,
        },
        'logger_cfgs': {
            'save_model_freq': 25
        },
    }

    # Run the experiments
    for env_id, algo in exp_chunk:
        agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
        agent.learn()

        agent.plot(smooth=1)
        agent.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
        agent.evaluate(num_episodes=1)

    # Gather the processes
    comm.gather([], root=0)

    MPI.Finalize()
