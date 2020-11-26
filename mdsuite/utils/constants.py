"""
The file contains various physical constants used in the calculations. All data is from the NIST, found at:

    https://www.nist.gov/pml/fundamental-physical-constants

Collected by the author of the code.
"""

standard_state_pressure = 100000 # Pa -- Standard state pressure
avogadro_constant = 6.02214076e23 # mol^-1 -- Avogadro's constant
elementary_charge = 1.602176634e-19 # C -- Elementary charge
boltzmann_constant = 1.380649e-23 # J K^-1 --  Boltzmann constant
hyperfine_transition_frequency = 9192631770 # Hz -- Hyperfine transition frequency of Cs-133
luminous_efficacy = 683 # lm W^-1 -- Luminous efficacy
planck_constant = 6.62607015e-34 # J Hz^-1 -- Planck constant
reduced_planck_constant = 1.054571817e-34 # J s -- Reduced Planck constant
speed_of_light = 299792458 # m s^-1 -- Speed of light in a vacuum
gravity = 9.80665 # m s^-2 -- Standard acceleration due to gravity on earth
atmosphere = 101325 # Pa -- Standard atmospheric pressure
golden_ratio = 1.618033988749895  # The golden ratio as taken from scipy

lammps_properties_labels = {'x', 'y', 'z',
                                    'xs', 'ys', 'zs',
                                    'xu', 'yu', 'zu',
                                    'xsu', 'ysu', 'zsu',
                                    'ix', 'iy', 'iz',
                                    'vx', 'vy', 'vz',
                                    'fx', 'fy', 'fz',
                                    'mux', 'muy', 'muz', 'mu',
                                    'omegax', 'omegay', 'omegaz',
                                    'angmomx', 'angmomy', 'angmomz',
                                    'tqx', 'tqy', 'tqz'}

lammps_properties = ["Positions", "Scaled_Positions", "Unwrapped_Positions", "Scaled_Unwrapped_Positions",
                             "Velocities", "Forces", "Box_Images", "Dipole_Orientation_Magnitude",
                             "Angular_Velocity_Spherical",
                             "Angular_Velocity_Non_Spherical", "Torque"]