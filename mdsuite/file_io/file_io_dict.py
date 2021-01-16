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


# this is the dict that specifies the properties names in the lammps dump
lammps_traj = {
    "Positions": ['x', 'y', 'z'],
    "Scaled_Positions": ['xs', 'ys', 'zs'],
    "Unwrapped_Positions": ['xu', 'yu', 'zu'],
    "Scaled_Unwrapped_Positions": ['xsu', 'ysu', 'zsu'],
    "Velocities": ['vx', 'vy', 'vz'],
    "Forces": ['fx', 'fy', 'fz'],
    "Box_Images": ['ix', 'iy', 'iz'],
    "Dipole_Orientation_Magnitude": ['mux', 'muy', 'muz'],
    "Angular_Velocity_Spherical": ['omegax', 'omegay', 'omegaz'],
    "Angular_Velocity_Non_Spherical": ['angmomx', 'angmomy', 'angmomz'],
    "Torque": ['tqx', 'tqy', 'tqz'],
    "KE": ["c_KE"],
    "PE": ["c_PE"],
    "Stress": ['c_Stress[1]', 'c_Stress[2]', 'c_Stress[3]', 'c_Stress[4]', 'c_Stress[5]', 'c_Stress[6]']
}

lammps_flux = {
    "Temperature": ["temp"],
    "Time": ["time"],
    "Flux_Thermal": ['c_flux_thermal[1]', 'c_flux_thermal[2]', 'c_flux_thermal[3]']
}