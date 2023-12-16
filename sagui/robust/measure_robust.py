import os
import subprocess
import sys
import numpy as np
from evaluator_robust import EvaluatorRobust
from omnisafe.envs.sagui_envs import register_sagui_envs
from mpi4py import MPI


def mpi_fork(n):
    if os.getenv('IN_MPI') is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS='1',
            OMP_NUM_THREADS='1',
            IN_MPI='1'
        )
        args = ['mpirun', '-np', str(n)]

        # Ask to allow running as root
        print('\nAllow running as root? WARNING: Allowing root may break your system. Only enable the feature in virtualized environments.')
        ans = input('Answer yes (y) or no (n): ').lower()
        if ans in ['yes', 'y']:
            args.append('--allow-run-as-root')
            print('Root allowed.')
        elif ans in ['no', 'n']:
            print('Root disallowed.')
        else:
            print('Execution canceled. Please answer "yes" or "no".')
            sys.exit()

        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


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
    LOG_DIRS = ['./save/']
    MODEL_FNAME = 'epoch-500.pt'

    # Create a list of coefficients
    coef_list = [{'body_mass': mass_mult, 'dof_damping': damp_mult}
                 for mass_mult in np.linspace(0.25, 4, 10)
                 for damp_mult in np.linspace(0.5, 1.5, 10)]

    # Split the list of coefficients into equal chunks
    coef_list = np.array(coef_list)
    coef_sublists = np.array_split(coef_list, NUM_PROCS)

    # Select corresponding chunk of data
    coefs_chunk = coef_sublists[rank]

    # Register sagui envs
    register_sagui_envs()

    # Calculate the robustness of each agent
    for log_dir in LOG_DIRS:
        # Calculate robustness
        evaluator = EvaluatorRobust()
        evaluator.load_saved(save_dir=log_dir, model_name=MODEL_FNAME)
        results = evaluator.evaluate(coefs_chunk, num_episodes=100)

        # Gather results
        all_results = comm.gather(results, root=0)

        # Save the results
        if rank == 0:
            # Flatten the results and turn them into a string
            res_flat = [str(x) for r in all_results for x in r]
            res_str = '[\n' + ',\n'.join(res_flat) + '\n]'

            with open(log_dir + 'robust_results.txt', 'w') as f:
                f.write(res_str)

    MPI.Finalize()
