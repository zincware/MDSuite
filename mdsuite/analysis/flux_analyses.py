"""
Class for the calculation of the einstein diffusion coefficients.

Author: Francisco Torres-Herrador ; Samuel Tovey

Description: This module contains the code for the thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the ... method.
The methods in class can then be called by the ... method and all necessary
calculations performed.
"""
import warnings

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
# Import user packages
from tqdm import tqdm
from mdsuite.convolution_computation.convolution import convolution
from mdsuite.utils.meta_functions import timeit
# MDSuite packages
import mdsuite.utils.constants as constants

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class _GreenKuboThermalConductivityFlux:
    """ Class for the Einstein diffusion coefficient implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, data_range=500):
        self.parent = obj
        self.plot = plot
        self.data_range = data_range
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

    def _autocorrelation_time(self):
        """ Claculate the flux autocorrelation time to ensure correct sampling """
        raise NotImplementedError

    @timeit
    def _compute_thermal_conductivity(self):
        """ Compute the thermal conductivity """

        if self.plot:
            averaged_jacf = np.zeros(self.data_range)

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.parent.temperature ** 2 * self.parent.units['boltzman'] \
                      * self.parent.volume # we use boltzman constant in the units provided.

        # TODO: I had a /2 in data range. I removed it. I think it was wrong.
        prefactor = numerator / denominator

        flux = self.load_flux_matrix()

        loop_range = len(flux) - self.data_range - 1  # Define the loop range

        sigma = convolution(loop_range=loop_range, flux=flux, data_range=self.data_range, time=self.time)

        sigma = prefactor * np.array(sigma)

        # convert to SI units.
        prefactor_units = self.parent.units['energy']/self.parent.units['length']/self.parent.units['time']
        sigma = prefactor_units*sigma

        if self.plot:
            averaged_jacf /= max(averaged_jacf)
            plt.plot(self.time, averaged_jacf)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Current Autocorrelation Function")
            plt.savefig(f"GK_Cond_{self.parent.temperature}.pdf", )
            plt.show()

        print(f"Green-Kubo Thermal Conductivity at {self.parent.temperature}K: {np.mean(sigma)} +- "
              f"{np.std(sigma) / np.sqrt(len(sigma))} W/m/K")

        self.parent.thermal_conductivity["Green-Kubo-flux"] = np.mean(sigma) / 100

    def load_flux_matrix(self):
        """ Load the flux matrix

        returns:
            Matrix of the property flux
        """
        identifiers = [f'c_flux_thermal[{i + 1}]' for i in range(3)]
        matrix_data = []

        for identifier in identifiers:
            column_data = self.parent.load_column(identifier)
            matrix_data.append(column_data)
        matrix_data = np.array(matrix_data).T  # transpose such that [timestep, dimension]
        return matrix_data
