"""
Class for the calculation of the Einstein-Helfand ionic conductivity.

Summary
-------
This class is called by the Experiment class and instantiated when the user calls the
Experiment.einstein_helfand_ionic_conductivity method. The methods in class can then be called by the
Experiment.einstein_helfand_ionic_conductivity method and all necessary calculations performed.
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.units import elementary_charge, boltzmann_constant
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandIonicConductivity(Calculator):
    """
    Class for the Einstein-Helfand Ionic Conductivity

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

    def __init__(self, experiment, plot=True, data_range=500, save=True, correlation_time=1):
        """
        Python constructor

        Parameters
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

        # parse to the experiment class
        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time)

        self.loaded_property = 'Translational_Dipole_Moment'  # Property to be loaded for the analysis
        self.dependency = "Unwrapped_Positions"
        self.system_property = True

        self.database_group = 'ionic_conductivity'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$/s)'
        self.analysis_name = 'einstein_helfand_ionic_conductivity'

        self.msd_array = np.zeros(self.data_range)
        self.prefactor: float

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
        numerator = (self.experiment.units['length'] ** 2) * (elementary_charge ** 2)
        denominator = 6 * self.experiment.units['time'] * (
                    self.experiment.volume * self.experiment.units['length'] ** 3) * \
                      self.experiment.temperature * boltzmann_constant
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

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
        msd = (ensemble - (
            tf.repeat(tf.expand_dims(ensemble[0], 0), self.data_range, axis=0))) ** 2
        msd = self.prefactor*tf.reduce_sum(msd, axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self._fit_einstein_curve([self.time, self.msd_array])
        self._update_properties_file(data=result)

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array)
            self._plot_data()

        # Save the array if required
        if self.save:
            self._save_data(f"{self.analysis_name}", [self.time, self.msd_array])
