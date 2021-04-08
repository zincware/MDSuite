"""
Class for the calculation of the Green-Kubo thermal conductivity.

Summary
-------
This module contains the code for the Green-Kubo thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_thermal_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_thermal_conductivity method and all necessary
calculations performed.
"""
import warnings

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator
from scipy import signal

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivity(Calculator):
    """
    Class for the Green-Kubo Thermal conductivity implementation

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
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, experiment, data_range=500, correlation_time: int = 1, **kwargs):
        """
        Class for the Green-Kubo Thermal conductivity implementation

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
        correlation_time: int
        """
        super().__init__(experiment, data_range=data_range, correlation_time=correlation_time, **kwargs)

        self.loaded_property = 'Thermal_Flux'  # property to be loaded for the analysis
        self.database_group = 'thermal_conductivity'  # Which database_path group to save the tensor_values in
        self.system_property = True

        self.x_label = 'Time (s)'
        self.y_label = r'JACF ($C^{2}\cdot m^{2}/s^{2}$)'
        self.analysis_name = 'green_kubo_thermal_conductivity'

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
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.experiment.temperature ** 2 * self.experiment.units['boltzman'] \
                      * self.experiment.volume  # we use Boltzmann constant in the units provided.
        prefactor_units = self.experiment.units['energy'] / self.experiment.units['length'] / self.experiment.units[
            'time']

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
        self._update_properties_file(data=[np.mean(result), np.std(result) / (np.sqrt(len(result)))])

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.jacf)
            self._plot_data()
        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, self.jacf])

        if self.export:
            self.export_data_to_csv(f"{species}_{self.analysis_name}", [self.time, self.jacf])