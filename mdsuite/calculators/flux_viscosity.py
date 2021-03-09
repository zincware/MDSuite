"""
Class for the calculation of viscosity.

Summary
-------
This module contains the code for the viscosity class. This class is called by the
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
from .calculator import Calculator


# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosityFlux(Calculator):
    """
    Class for the computation of viscosity based on GK formulation for flux files.

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    time : np.array
            Array of the time.
    """

    def __init__(self, obj, plot=False, data_range=500, correlation_time=1):
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

        self.database_group = 'viscosity'  # Which database group to save the data in
        self.analysis_name = 'viscosity_flux'
        self.correlation_time = correlation_time

    def _autocorrelation_time(self):
        """
        Calculate the flux autocorrelation time to ensure correct sampling
        """
        pass

    @timeit
    def _calculate_viscosity(self):
        """
        Compute the thermal conductivity
        """


        # prepare the prefactor for the integral
        numerator = self.parent.volume
        denominator = 3 * (self.data_range - 1) * self.parent.temperature * self.parent.units['boltzman']  # we use boltzman constant in the units provided.

        prefactor = numerator / denominator
        flux = self.load_flux_matrix()
        loop_range = int((len(flux) - self.data_range - 1)/self.correlation_time)  # Define the loop range
        sigma, averaged_jacf  = convolution(loop_range=loop_range,
                                            flux=flux,
                                            data_range=self.data_range,
                                            time=self.time, correlation_time=self.correlation_time)
        sigma = prefactor * np.array(sigma)

        # convert to SI units.
        prefactor_units = self.parent.units['pressure']**2 * self.parent.units['length']**3 * self.parent.units['time']/self.parent.units['energy']
        sigma = prefactor_units * sigma

        if self.plot:
            averaged_jacf /= max(averaged_jacf)
            plt.plot(self.time, averaged_jacf)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Current Autocorrelation Function")
            plt.savefig(f"GK_Cond_{self.parent.temperature}.pdf", )
            plt.show()

        print(f"Green-Kubo Viscosity at {self.parent.temperature}K: {np.mean(sigma)} +- "
              f"{np.std(sigma) / np.sqrt(len(sigma))} Pa.s")

        # self.parent.viscosity["Green-Kubo-flux"] = np.mean(sigma) / 100
        self._update_properties_file(data=[str(np.mean(sigma)), str(np.std(sigma / 100))])

    def load_flux_matrix(self):
        """
        Load the flux matrix

        :return: Matrix of the property flux
        """
        # TODO: re-implement
        identifier = 'Stress_visc/Stress_visc'
        matrix_data = []

        matrix_data = self.parent.load_matrix(path=identifier, select_slice=np.s_[:])
        matrix_data = np.squeeze(matrix_data)
        return matrix_data

    def run_analysis(self):
        """ Run thermal conductivity calculation analysis

        The thermal conductivity is computed at this step.
        """
        self._autocorrelation_time()  # get the autocorrelation time

        self._calculate_viscosity()  # calculate the singular diffusion coefficients
