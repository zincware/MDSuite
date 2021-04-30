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
from tqdm import tqdm
import tensorflow as tf
from scipy import signal

from mdsuite.calculators.calculator import Calculator
from mdsuite.plot_style.plot_style import apply_style

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosityFlux(Calculator):
    """
    Class for the Green Kubo viscosity from flux implementation

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    """

    def __init__(self, experiment, plot=False, data_range=500, correlation_time=1, save=True, export: bool = False):
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
        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time, export=export)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Stress_visc'  # Property to be loaded for the analysis
        self.system_property = True

        self.database_group = 'Viscosity'  # Which database_path group to save the tensor_values in
        self.analysis_name = 'Viscosity_Flux'
        self.x_label = 'Time (s)'
        self.y_label = 'JACF ($C^{2}\\cdot m^{2}/s^{2}$)'

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
        numerator = self.experiment.volume
        denominator = 3 * (self.data_range - 1) * self.experiment.temperature * self.experiment.units['boltzman']

        prefactor_units = self.experiment.units['pressure'] ** 2 * self.experiment.units['length'] ** 3 * \
                          self.experiment.units['time'] / self.experiment.units['energy']

        self.prefactor = (numerator / denominator)*prefactor_units

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
        jacf = (signal.correlate(ensemble[:, 0],
                                 ensemble[:, 0],
                                 mode='full', method='auto') +
                signal.correlate(ensemble[:, 1],
                                 ensemble[:, 1],
                                 mode='full', method='auto') +
                signal.correlate(ensemble[:, 2],
                                 ensemble[:, 2],
                                 mode='full', method='auto'))
        self.jacf += jacf[int(self.data_range - 1):]
        self.sigma.append(np.trapz(jacf[int(self.data_range - 1):], x=self.time))

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self.prefactor*np.array(self.sigma)

        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": ["System"],
                      "data_range": self.data_range,
                      'data': [{'x': np.mean(result), 'uncertainty': np.std(result)/(np.sqrt(len(result)))}]
                      }
        self._update_properties_file(properties)

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.jacf)
            self._plot_data()

        if self.save:
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": ["System"],
                          "data_range": self.data_range,
                          'data': [{'x': x, 'y': y} for x, y in zip(self.time, self.jacf)],
                          'information': "JACF Array"
                          }
            self._update_properties_file(properties)

        if self.export:
            self._export_data(name=self._build_table_name("System"), data=self._build_pandas_dataframe(self.time,
                                                                                                      self.jacf))
