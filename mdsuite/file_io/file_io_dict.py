"""
Module to contain the structured dict for file io.

Summary
-------
"""

from mdsuite.file_io.lammps_trajectory_files import LAMMPSTrajectoryFile
from mdsuite.file_io.lammps_flux_files import LAMMPSFluxFile
from mdsuite.file_io.extxyz_trajectory_reader import EXTXYZFileReader

dict_file_io = {
    'lammps_traj': (LAMMPSTrajectoryFile, 'traj'),
    'lammps_flux': (LAMMPSFluxFile, 'flux'),
    'extxyz': (EXTXYZFileReader, 'traj')
}
