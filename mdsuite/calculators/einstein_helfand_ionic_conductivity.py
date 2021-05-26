"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

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
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.EinsteinHelfandTIonicConductivity(data_range=500, plot=True, correlation_time=10)
    """

    def __init__(self, experiment):
        """
        Python constructor

        Parameters
        ----------
        experiment :  object
            Experiment class to call from
        """

        # parse to the experiment class
        super().__init__(experiment)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Translational_Dipole_Moment'  # Property to be loaded for the analysis
        self.dependency = "Unwrapped_Positions"
        self.system_property = True

        self.database_group = 'Ionic_Conductivity'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$/s)'
        self.analysis_name = 'Einstein_Helfand_Ionic_Conductivity'
        self.prefactor: float

    def __call__(self, plot=True, data_range=500, save=True, correlation_time=1,
                 export: bool = False, gpu: bool = False):
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
        self.update_user_args(plot=plot, data_range=data_range, save=save, correlation_time=correlation_time,
                              export=export, gpu=gpu)
        self.msd_array = np.zeros(self.data_range)

        out = self.run_analysis()

        self.experiment.save_class()
        # need to move save_class() to here, because it can't be done in the experiment any more!

        return out

    def _update_output_signatures(self):
        """
        Update the output signature for the IC.

        Returns
        -------

        """
        self.batch_output_signature = (tf.TensorSpec(shape=(self.batch_size, 3), dtype=tf.float64))
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
        msd = self.prefactor * tf.reduce_sum(msd, axis=1)
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
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array,
                     label=fr'{result[0]:.3E} $\pm$ '
                           f'{result[1]:.3E}')
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
