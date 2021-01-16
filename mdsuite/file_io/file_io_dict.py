"""
Module to contain the structured dict for file io.

Summary
-------
"""

from mdsuite.file_io.lammps_trajectory_files import LAMMPSTrajectoryFile
from mdsuite.file_io.lammps_flux_files import LAMMPSFluxFile

dict_file_io = {
    'lammps_traj': (LAMMPSTrajectoryFile, 'traj'),
    'lammps_flux': (LAMMPSFluxFile, 'flux'),
}
