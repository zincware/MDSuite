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
from tqdm import tqdm
import tensorflow as tf
from scipy import signal

from mdsuite.calculators.calculator import Calculator
from mdsuite.plot_style.plot_style import apply_style

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivityFlux(Calculator):
    """
    Class for the Thermal conductivity from flux implementation

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    """

    def __init__(self, experiment, plot=False, data_range=500, correlation_time=1, save=True):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        experiment : object
                Experiment class to read and write to
        plot : bool
                If true, a plot of the analysis is saved.
        data_range : int
                Number of configurations to include in each ensemble
        """
        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time)

        self.loaded_property = 'Flux_Thermal'  # Property to be loaded for the analysis
        self.system_property = True

        self.database_group = 'thermal_conductivity'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time (s)'
        self.y_label = 'JACF ($C^{2}\\cdot m^{2}/s^{2}$)'
        self.analysis_name = 'thermal_conductivity_flux'

        self.prefactor: float
        self.jacf = np.zeros(self.data_range)
        self.sigma = []

        apply_style()

    def _update_output_signatures(self):
        """
        Update the output signature for the IC.

        Returns
        -------

        """
        self.batch_output_signature = tf.TensorSpec(shape=(self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64)

    def _calculate_prefactor(self, species: str = None):
        """
        Compute the ionic conductivity prefactor.

        Parameters
        ----------
        species

        Returns
        -------

        """
        # Calculate the prefactor
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.experiment.temperature ** 2 * self.experiment.units['boltzman'] \
                      * self.experiment.volume  # we use boltzmann constant in the units provided.

        prefactor_units = self.experiment.units['energy'] / self.experiment.units['length'] / \
                          self.experiment.units['time']

        self.prefactor = (numerator / denominator) * prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.jacf /= max(self.jacf)

    def _apply_operation(self, ensemble, index):
        """
        Calculate and return the vacf.

        Parameters
        ----------
        ensemble

        Returns
        -------
        updates class vacf with the tensor_values.
        """
        jacf = sum([signal.correlate(ensemble[:, idx], ensemble[:, idx],
                                     mode="full",
                                     method='auto') for idx in range(3)])
        self.jacf += jacf[int(self.data_range - 1):]
        self.sigma.append(np.trapz(jacf[int(self.data_range - 1):], x=self.time))

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)
        self._update_properties_file(data=[np.mean(result), np.std(result) / (np.sqrt(len(result)))])

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.jacf)
            self._plot_data()

        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, self.jacf])
