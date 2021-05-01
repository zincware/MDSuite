"""
Class for the calculation of the Green-Kubo viscosity.

Summary
This module contains the code for the Green-Kubo viscsity class. This class is called by the
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy import signal

from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import join_path

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosity(Calculator):
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

    def __init__(self, experiment, plot=False, data_range=500, save=True, correlation_time: int = 1,
                 export: bool = False):
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
        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time, export=export)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Momentum_Flux'  # property to be loaded for the analysis
        self.database_group = 'Viscosity'  # Which database_path group to save the tensor_values in
        self.system_property = True

        self.x_label = 'Time (s)'
        self.y_label = r'SACF ($C^{2}\cdot m^{2}/s^{2}$)'
        self.analysis_name = 'Green_Kubo_Viscosity'

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
        # prepare the prefactor for the integral
        numerator = 1  # self.experiment.volume
        denominator = 3 * (self.data_range - 1) * self.experiment.temperature * self.experiment.units[
            'boltzman'] * self.experiment.volume  # we use boltzman constant in the units provided.
        prefactor_units = self.experiment.units['pressure'] ** 2 * self.experiment.units['length'] ** 3 * \
                          self.experiment.units[
                              'time'] / self.experiment.units['energy']

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

        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": ['System'],
                      "data_range": self.data_range,
                      'data': [{'x': np.mean(result), 'uncertainty': np.std(result) / (np.sqrt(len(result)))}]
                      }
        self._update_properties_file(properties)
        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.jacf)
            self._plot_data()

        if self.save:
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": ['System'],
                          "data_range": self.data_range,
                          'data': [{'x': x, 'y': y} for x, y in zip(self.time, self.jacf)],
                          'information': "JACF Array"
                          }
            self._update_properties_file(properties)
        if self.export:
            self._export_data(name=self._build_table_name("System"), data=self._build_pandas_dataframe(self.time,
                                                                                                       self.jacf))
