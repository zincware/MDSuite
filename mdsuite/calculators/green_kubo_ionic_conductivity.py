"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo ionic conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_ionic_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_ionic_conductivity method and all necessary
calculations performed.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
import tensorflow as tf
from scipy import signal
from tqdm import tqdm
from mdsuite.utils.units import boltzmann_constant, elementary_charge
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboIonicConductivity(Calculator):
    """ Class for the Green-Kubo ionic conductivity implementation

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, tensor_values will be saved after the analysis
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, experiment, plot=False, data_range=500, save=True, correlation_time=1):
        """

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        plot : bool
                if true, plot the tensor_values
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        """

        # update experiment class
        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Ionic_Current'  # property to be loaded for the analysis
        self.system_property = True

        self.database_group = 'ionic_conductivity'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time (s)'
        self.y_label = r'JACF ($C^{2}\cdot m^{2}/s^{2}$)'
        self.analysis_name = 'green_kubo_ionic_conductivity'

        self.jacf = np.zeros(self.data_range)
        self.prefactor: float
        self.sigma = []

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
        numerator = (elementary_charge ** 2) * (self.experiment.units['length'] ** 2)
        denominator = 3 * boltzmann_constant * self.experiment.temperature * self.experiment.volume * \
                      (self.experiment.units['length'] ** 3) * self.data_range * self.experiment.units['time']
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.jacf /= max(self.jacf)

    def _apply_operation(self, ensemble, index):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.
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
