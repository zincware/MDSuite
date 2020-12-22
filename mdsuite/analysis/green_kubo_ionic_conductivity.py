"""
Class for the calculation of the Green-Kubo ionic conductivity.

Author: Samuel Tovey

Description: This module contains the code for the Green-Kubo ionic conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_ionic_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_ionic_conductivity method and all necessary
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


class GreenKuboIonicConductivity(Analysis):
    """ Class for the Green-Kubo ionic conductivity implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'JACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_ionic_conductivity'):
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)

        self.loaded_property = 'Velocities'
        self.batch_loop = None

        self.number_of_configurations = self.parent.number_of_configurations - self.parent.number_of_configurations % \
                                        self.parent.batch_size
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.correlation_time = 100

    def _autocorrelation_time(self):
        """ calculate the current autocorrelation time for correct sampling """

        pass

    def _calculate_system_current(self, velocity_matrix):
        """ Calculate the ionic current of the system

        :return: system_current (numpy array) -- ionic current of the system as a vector of shape (n_confs, 3)
        """

        species_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]

        system_current = np.zeros((self.batch_size['Parallel']*self.data_range, 3))  # instantiate the current array
        # Calculate the total system current
        for i in range(len(velocity_matrix)):
            system_current += np.array(np.sum(velocity_matrix[i][:, 0:], axis=0)) * species_charges[i]

        return system_current

    def _calculate_batch_loop(self):
        """ Calculate the batch loop parameters """
        self.batch_loop = int((self.batch_size['Parallel'] * self.data_range) /
                              (self.data_range + self.correlation_time))

    def _load_batch(self, batch_number, item=None):
        """ Load a batch of data """
        start = batch_number*self.batch_size['Parallel']*self.data_range
        stop = start + self.batch_size['Parallel']*self.data_range

        return self.parent.load_matrix("Velocities", item, select_slice=np.s_[:, start:stop])

    def _calculate_ionic_conductivity(self):
        """ Calculate the ionic conductivity in the system """

        # Calculate the prefactor
        numerator = (constants.elementary_charge ** 2) * (self.parent.units['length'] ** 2)
        denominator = 3 * constants.boltzmann_constant * self.parent.temperature * self.parent.volume * \
                      (self.parent.units['length'] ** 3) * self.data_range * self.parent.units['time']
        prefactor = numerator / denominator

        sigma = []
        parsed_autocorrelation = np.zeros(self.data_range)  # Define the parsed array

        for i in tqdm(range(int(self.n_batches['Parallel'])), ncols=70):
            batch = self._calculate_system_current(velocity_matrix=self._load_batch(i))  # get the ionic current
            for start_index in range(self.batch_loop):
                start = int(start_index*self.data_range + self.correlation_time)
                stop = int(start + self.data_range)

                system_current = np.array(batch)[start:stop]

                jacf = np.zeros(2 * self.data_range - 1)  # Define the empty jacf array

                # Calculate the current autocorrelation
                jacf += (signal.correlate(system_current[:, 0],
                                          system_current[:, 0],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 1],
                                          system_current[:, 1],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 2],
                                          system_current[:, 2],
                                          mode='full', method='auto'))

                jacf = jacf[int((len(jacf) / 2)):]  # Cut the negative part of the current autocorrelation
                parsed_autocorrelation += jacf
                sigma.append(prefactor * np.trapz(jacf, x=self.time))  # Update the conductivity array

        self.parent.ionic_conductivity["Green-Kubo"] = [np.mean(sigma) / 100, (np.std(sigma)/np.sqrt(len(sigma)))/100]

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data
        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """ call relevant methods and run analysis """

        self._autocorrelation_time()          # get the autocorrelation time
        self._collect_machine_properties()    # collect machine properties and determine batch size
        self._calculate_batch_loop()          # Update the batch loop attribute
        self._calculate_ionic_conductivity()  # calculate the ionic conductivity
