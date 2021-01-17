"""
Class for the calculation of the einstein diffusion coefficients.

Summary
-------
This module contains the code for the thermal conductivity class. This class is called by the
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


class GreenKuboThermalConductivityFlux:
    """
    Class for the Einstein diffusion coefficient implementation

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    time : np.array
            Array of the time.
    """

    def __init__(self, obj, plot=False, data_range=500):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        obj : object
                Experiment class to read and write to
        plot : bool
                If true, a plot of the analysis is saved.
        data_range : int
                Number of configurations to include in each ensemble
        """
        self.parent = obj
        self.plot = plot
        self.data_range = data_range
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

    def _autocorrelation_time(self):
        """
        Calculate the flux autocorrelation time to ensure correct sampling
        """
        pass

    @timeit
    def _calculate_thermal_conductivity(self):
        """
        Compute the thermal conductivity
        """

        if self.plot:
            averaged_jacf = np.zeros(self.data_range)

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.parent.temperature ** 2 * self.parent.units['boltzman'] \
                      * self.parent.volume # we use boltzman constant in the units provided.

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
        """
        Load the flux matrix

        :return: Matrix of the property flux
        """
        # TODO: re-implement
        identifier = 'Flux_Thermal'
        matrix_data = []

        matrix_data = self.parent.load_matrix(identifier)
        matrix_data = np.squeeze(matrix_data)
        return matrix_data

    def run_analysis(self):
        """ Run thermal conductivity calculation analysis

        The thermal conductivity is computed at this step.
        """
        self._autocorrelation_time()  # get the autocorrelation time

        self._calculate_thermal_conductivity()  # calculate the singular diffusion coefficients