"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Author: Samuel Tovey

Description: This module contains the code for the Einstein-Helfand ionic conductivity class. This class is called by
the Experiment class and instantiated when the user calls the Experiment.einstein_helfand_ionic_conductivity method.
The methods in class can then be called by the Experiment.einstein_helfand_ionic_conductivity method and all necessary
calculations performed.
"""

# Python standard packages
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import numpy as np

# Import user packages
from tqdm import tqdm
import tensorflow as tf
# Import MDSuite modules
import mdsuite.utils.meta_functions as meta_functions
from mdsuite.utils.constants import *
from mdsuite.analysis.analysis import Analysis

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandIonicConductivity(Analysis):
    """ Class for the Einstein-Helfand Ionic Conductivity """

    def __init__(self, obj, plot=True, species=None, data_range=500, save=True,
                 x_label='Time (s)', y_label='MSD (m^2/s)', analysis_name='einstein_helfand_ionic_conductivity'):
        """ Python constructor """

        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)  # parse to the parent class

        self.loop_range = self.parent.number_of_configurations - data_range - 1
        self.correlation_time = 10
        self.species = species
        self.time = np.linspace(0.0, self.data_range * self.parent.time_step * self.parent.sample_rate, self.data_range)

    def _autocorrelation_time(self):
        """ Calculate dipole moment autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _calculate_translational_dipole(self, index, data_range):
        """ Calculate the translational dipole of the system """

        # Load the particle positions and sum
        dipole_moment = self.parent.load_matrix("Unwrapped_Positions",
                                                select_slice=np.s_[:, index:index + data_range],
                                                tensor=True)
        dipole_moment = tf.math.reduce_sum(dipole_moment, axis=1)

        # Build the charge tensor for assignment
        system_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]
        charge_tuple = []
        for charge in system_charges:
            charge_tuple.append(tf.ones([data_range, 3]) * charge)
        charge_tensor = tf.stack(charge_tuple)

        dipole_moment *= charge_tensor  # Multiply the dipole moment tensor by the system charges
        dipole_moment = tf.reduce_sum(dipole_moment, axis=0)  # Calculate the final dipole moments

        return dipole_moment

    def _calculate_ionic_conductivity(self):
        """ Calculate the conductivity """

        dipole_msd_array = np.zeros(self.data_range)  # Initialize the msd array

        # Calculate the prefactor
        numerator = (self.parent.units['length'] ** 2) * (elementary_charge ** 2)
        denominator = 6 * self.parent.units['time'] * (self.parent.volume * self.parent.units['length'] ** 3) * \
                      self.parent.temperature * boltzmann_constant
        prefactor = numerator / denominator

        for i in tqdm(range(0, self.loop_range, self.correlation_time), ncols=50):
            window_tensor = self._calculate_translational_dipole(i, self.data_range)

            # Calculate the msd
            msd = (window_tensor - (
                tf.repeat(tf.expand_dims(window_tensor[0], 0), self.data_range, axis=0))) ** 2
            msd = prefactor * tf.reduce_sum(msd, axis=1)

            dipole_msd_array += np.array(msd)

        dipole_msd_array /= int(self.loop_range / self.correlation_time)

        popt, pcov = curve_fit(meta_functions.linear_fitting_function, self.time, dipole_msd_array)
        self.parent.ionic_conductivity["Einstein-Helfand"] = popt[0] / 100

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.parent.units['time'], dipole_msd_array)
            self._plot_data()

        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, dipole_msd_array])

    def run_analysis(self):
        """ Collect methods and run analysis """

        self._autocorrelation_time()  # get the correct correlation time
        self._calculate_ionic_conductivity()  # calculate the ionic conductivity
