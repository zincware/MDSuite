import os
import subprocess
import shlex
from os.path import exists


folders = ('Argon','Xenon')#, "CH4", "Krypton")#, 'Xenon_2', 'Argon_2', "CH4_2", "Krypton_2",)
root_folder = os.getcwd()

for folder in folders:
    subfolders = os.listdir(f"{folder}")

    # for each case to run
    for subfolder in subfolders:
        os.chdir(f'{folder}/{subfolder}')
        if not exists('log.lammps'): # if it does not exist, we run the case.
            print(f'Running case: {folder}/{subfolder}')
            p = subprocess.Popen(shlex.split("mpirun -np 6 lmp_mpi -in input.lammps"), shell=False)
            p.wait()
        else:
            print(f'Skipping case: {folder}/{subfolder}')
        os.chdir(root_folder)
