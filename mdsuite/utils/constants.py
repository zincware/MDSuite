"""
The file contains various physical constants used in the calculations. All data is from the NIST, found at:

    https://www.nist.gov/pml/fundamental-physical-constants

Collected by the author of the code.
"""

# this is the dict that specifies the properties names in the lammps dump
lammps_properties_dict = {
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
