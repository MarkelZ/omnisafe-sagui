from evaluator_transfer import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs
from mpi4py import MPI
from mpi_tools import mpi_fork
import numpy as np


class tcol:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    # Number of processes
    NUM_PROCS = 10

    # Fork using mpi
    mpi_fork(NUM_PROCS)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # LOG_DIR should contain two things:
    # 1. config.json
    # 2. torch_save/{MODEL_FNAME}
    LOG_DIRS = ['./guide2/']
    MODEL_FNAME = 'epoch-500.pt'

    #
    TOTAL_EPS = 100
    NUM_EPS = TOTAL_EPS // NUM_PROCS

    # Dict of coefficients
    coefs = {'body_mass': 1.0, 'dof_damping': 1.0}

    # Register sagui envs
    register_sagui_envs()

    # Calculate the robustness of each agent
    for i, log_dir in enumerate(LOG_DIRS):
        # Print progress
        if rank == 0:
            print(f'{tcol.OKBLUE}{tcol.BOLD}Processing {log_dir} ({i+1} of {len(LOG_DIRS)}).{tcol.ENDC}')

        # Calculate robustness
        evaluator = EvaluatorRobust()
        evaluator.load_saved(save_dir=log_dir, model_name=MODEL_FNAME)
        results = evaluator.evaluate([coefs], num_episodes=NUM_EPS, deterministic=False,
                                     process_name=f'CPU{rank}@{log_dir}')

        # Gather results
        all_results = comm.gather(results, root=0)

        # Save the results
        if rank == 0:
            avg_cost = sum([v[0][1] for v in all_results]) / NUM_PROCS
            print(f'{tcol.OKCYAN}{tcol.BOLD}Average cost: {avg_cost:.2f}{tcol.ENDC}')

    MPI.Finalize()
