"""
Class for the calculation of the radial distribution function.

Author: Samuel Tovey

Description: This module contains the code for the radial dsitribution function. This class is called by
the Experiment class and instantiated when the user calls the Experiment.radial_distribution_function method.
The methods in class can then be called by the Experiment.radial_distribution_function method and all necessary
calculations performed.
"""

class RadialDistributionFunction:
    """ Class for the calculation of the radial distribution function """

    def __init__(self):
        """ Standard python constructor """