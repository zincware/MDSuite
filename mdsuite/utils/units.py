# we will define each unit experiment as a function and then have a to_si_units which is a dispatch dict.
# to add a new unit experiment, simply add it as a function here, and then in the dictionary in units_to_si.

# we can keep these here, or add them to the corresponding set of units.
# for example, in the metal and real, the boltzman constant is added in that experiment of units.
# This allows to perform all the computations using the experiment of units given, and only perform the transformation of the final result.
standard_state_pressure = 100000  # Pa -- Standard state pressure
avogadro_constant = 6.02214076e23  # mol^-1 -- Avogadro's constant
elementary_charge = 1.602176634e-19  # C -- Elementary charge
boltzmann_constant = 1.380649e-23  # J K^-1 --  Boltzmann constant
hyperfine_transition_frequency = 9192631770  # Hz -- Hyperfine transition frequency of Cs-133
luminous_efficacy = 683  # lm W^-1 -- Luminous efficacy
planck_constant = 6.62607015e-34  # J Hz^-1 -- Planck constant
reduced_planck_constant = 1.054571817e-34  # J s -- Reduced Planck constant
speed_of_light = 299792458  # m s^-1 -- Speed of light in a vacuum
gravity = 9.80665  # m s^-2 -- Standard acceleration due to gravity on earth
atmosphere = 101325  # Pa -- Standard atmospheric pressure
golden_ratio = 1.618033988749895  # The golden ratio as taken from scipy


def units_real():
    units = {'time': 1e-15, 'length': 1e-10, 'energy': 4184 / 6.02214076e23,
             'NkTV2p': 68568.415,
             'boltzman': 0.0019872067, 'temperature':1, 'pressure':101325.0}
    return units


def units_metal():
    units = {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19,
             'NkTV2p': 1.6021765e6,
             'boltzman': 8.617343e-5,'temperature':1, 'pressure':100000}
    return units


def units_SI():
    units = {'time': 1, 'length': 1, 'energy': 1,
             'boltzman': 1.380649e-23,
             'avogadro': 6.02214076e23,
             'elementary_charge': 1.602176634e-19,'temperature':1, 'pressure':1}
    return units


units_dict = {'real': units_real,
              'metal': units_metal,
              'SI': units_SI}
