import os
import subprocess
import sys


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
