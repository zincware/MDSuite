"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the conductivity.

Summary
-------
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator, call

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandThermalKinaci(Calculator):
    """
    Class for the Einstein-Helfand Ionic Conductivity

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis

    See Also
    --------
    mdsuite.calculators.calculator.Calculator

    Examples
    --------
    experiment.run_computation.EinsteinHelfandThermalKinaci(data_range=500, plot=True, correlation_time=10)

    """

    def __init__(self, **kwargs):
        """
        Python constructor

        Parameters
        ----------
        experiment :  object
            Experiment class to call from
        """

        # parse to the experiment class
        super().__init__(**kwargs)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Kinaci_Heat_Current'  # Property to be loaded for the analysis
        self.dependency = "Unwrapped_Positions"
        self.system_property = True

        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$/s)'
        self.analysis_name = 'Einstein_Helfand_Thermal_Conductivity_Kinaci'

        self.database_group = 'Thermal_Conductivity'  # Which database_path group to save the tensor_values in

        self.prefactor: float

    @call
    def __call__(self, plot=True, data_range=500, save=True, correlation_time=1, export: bool = False,
                 gpu: bool = False):
        """
        Python constructor

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        correlation_time : int
                Correlation time to use in the window sampling.
        export : bool
                If true, export the data directly into a csv file.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.
        """
        # parse to the experiment class
        self.update_user_args(plot=plot, data_range=data_range, save=save, correlation_time=correlation_time,
                              export=export, gpu=gpu)
        self.msd_array = np.zeros(self.data_range)  # Initialize the msd array

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
        denominator = 6 * self.experiment.volume * self.experiment.temperature * self.experiment.units['boltzman']
        units_change = self.experiment.units['energy'] / self.experiment.units['length'] / self.experiment.units[
            'time'] / self.experiment.units['temperature']
        self.prefactor = numerator / denominator * units_change

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
        msd = tf.math.squared_difference(ensemble, ensemble[None, 0])

        msd = self.prefactor*tf.reduce_sum(msd, axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self._fit_einstein_curve([self.time, self.msd_array])
        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": ["System"],
                      "data_range": self.data_range,
                      'data': [{'x': result[0], 'uncertainty': result[1]}]
                      }
        self._update_properties_file(properties)

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array)
            self._plot_data()

        if self.save:
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": ["System"],
                          "data_range": self.data_range,
                          'data': [{'x': x, 'y': y} for x, y in zip(self.time, self.msd_array)],
                          'information': "series"
                          }
            self._update_properties_file(properties)

        if self.export:
            self._export_data(name=self._build_table_name("System"), data=self._build_pandas_dataframe(self.time,
                                                                                                       self.msd_array))
