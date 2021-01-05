"""
Class for the calculation of the Green-Kubo thermal conductivity.

Author: Samuel Tovey; Francisco Torres-Herrador

Description: This module contains the code for the Green-Kubo thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_thermal_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_thermal_conductivity method and all necessary
calculations performed.
"""

# Python standard packages
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm

# Import MDSuite modules
import mdsuite.utils.constants as constants

from mdsuite.analysis.analysis import Analysis

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivity(Analysis):
    """ Class for the Green-Kubo Thermal conductivity implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label='JACF ($C^{2}\cdotm^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_thermal_conductivity'):
        super().__init__(obj,plot, save, data_range, x_label, y_label, analysis_name)
        self.number_of_configurations = self.parent.number_of_configurations - self.parent.number_of_configurations % \
                                        self.parent.batch_size
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.loop_range = self.number_of_configurations - data_range - 1
        self.correlation_time = 1

    def _autocorrelation_time(self):
        """ calculate the current autocorrelation time for correct sampling """
        pass

    def _calculate_system_current(self):
        """ Calculate the thermal current of the system

        :return: system_current (numpy array) -- thermal current of the system as a vector of shape (n_confs, 3)
        """

        ## TODO: re-implement for thermal conductivity.

        velocity_matrix = self.parent.load_matrix("Velocities")  # Load the velocity matrix
        stress_tensor = self.parent.load_matrix("Stress", sym_matrix=True)
        ke = self.parent.load_matrix("KE", scalar=True)
        pe = self.parent.load_matrix("PE", scalar=True)
        energy = ke + pe


        system_current = np.zeros((self.number_of_configurations, 3))  # instantiate the current array
        # Calculate the total system current


        return system_current

    def _calculate_thermal_conductivity(self):
        """ Calculate the thermal conductivity in the system """

        system_current = self._calculate_system_current()  # get the thermal current

        # Calculate the prefactor
        numerator = 1
        denominator = 3 * (self.data_range / 2 - 1) * self.parent.temperature ** 2 * constants.boltzmann_constant \
                      * self.parent.volume * self.parent.units['length'] ** 3

        # not sure why I need the /2 in data range...
        prefactor = numerator / denominator

        sigma = []
        parsed_autocorrelation = np.zeros(self.data_range)  # Define the parsed array
        for i in tqdm(range(0, self.loop_range, self.correlation_time), ncols=100):
            jacf = np.zeros(2 * self.data_range - 1)  # Define the empty jacf array

            # Calculate the current autocorrelation
            jacf += (signal.correlate(system_current[:, 0][i:i + self.data_range],
                                      system_current[:, 0][i:i + self.data_range],
                                      mode='full', method='fft') +
                     signal.correlate(system_current[:, 1][i:i + self.data_range],
                                      system_current[:, 1][i:i + self.data_range],
                                      mode='full', method='fft') +
                     signal.correlate(system_current[:, 2][i:i + self.data_range],
                                      system_current[:, 2][i:i + self.data_range],
                                      mode='full', method='fft'))

            jacf = jacf[int((len(jacf) / 2)):]  # Cut the negative part of the current autocorrelation
            parsed_autocorrelation += jacf
            sigma.append(prefactor * np.trapz(jacf, x=self.time))  # Update the conductivity array

        self.parent.thermal_conductivity["Green-Kubo"] = np.mean(sigma) / 100

        plt.plot(self.time, parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data
        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """ Run a diffusion coefficient analysis

        The thermal conductivity is computed at this step.
        """
        self._autocorrelation_time()  # get the autocorrelation time

        self._calculate_thermal_conductivity()  # calculate the singular diffusion coefficients

