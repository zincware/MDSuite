"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Class for the calculation of the einstein diffusion coefficients.

Summary
-------
Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
calculations performed.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
from mdsuite.calculators.calculator import Calculator

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDiffusionCoefficients(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
    Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
    The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
    calculations performed.

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    species : list
            Which species to perform the analysis on
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

    def __init__(self, experiment, plot: bool = True, species: list = None, data_range: int = 100, save: bool = True,
                 optimize: bool = False, correlation_time: int = 1, atom_selection=np.s_[:], export: bool = False,
                 molecules: bool = False, gpu: bool = False):
        """

        Parameters
        ----------
        experiment :  object
                Experiment class to call from
        plot : bool
                if true, plot the tensor_values
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        optimize : bool
                If true, the tensor_values range will be optimized
        """

        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time,
                         atom_selection=atom_selection, export=export, gpu=gpu)
        self.scale_function = {'linear': {'scale_factor': 50}}
        self.loaded_property = 'Unwrapped_Positions'
        self.species = species
        self.molecules = molecules
        self.database_group = 'Diffusion_Coefficients'
        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$/s)'
        self.analysis_name = 'Einstein_Self_Diffusion_Coefficients'
        self.loop_condition = False
        self.optimize = optimize
        self.msd_array = np.zeros(self.data_range)  # define empty msd array
        if species is None:
            if molecules:
                self.species = list(self.experiment.molecules)
            else:
                self.species = list(self.experiment.species)
        self.log = logging.getLogger(__name__)
        self.log.info('starting Einstein diffusion computation')

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(shape=(None, self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(None, self.data_range, 3), dtype=tf.float64)

    def _calculate_prefactor(self, species: str = None):
        """
        Compute the prefactor

        Parameters
        ----------
        species : str
                Species being studied.

        Returns
        -------
        Updates the class state.
        """
        if self.molecules:
            # Calculate the prefactor
            numerator = self.experiment.units['length'] ** 2
            denominator = (self.experiment.units['time'] * len(self.experiment.molecules[species]['indices'])) * 6
        else:
            # Calculate the prefactor
            numerator = self.experiment.units['length'] ** 2
            denominator = (self.experiment.units['time'] * len(self.experiment.species[species]['indices'])) * 6

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
            tf.repeat(tf.expand_dims(ensemble[:, 0], 1), self.data_range, axis=1))) ** 2

        # Sum over trajectory and then coordinates and apply averaging and prefactors
        msd = self.prefactor * tf.reduce_sum(tf.reduce_sum(msd, axis=0), axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.
        Returns
        -------

        """

        result = self._fit_einstein_curve([self.time, self.msd_array])
        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": species,
                      "data_range": self.data_range,
                      'data': result[0],
                      'uncertainty': result[1]}
        self._update_properties_file(properties)
        if self.plot:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array,
                     label=fr"{species}: {result[0]: 0.3E} $\pm$ {result[1]: 0.3E}")

        if self.save:
            self._save_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                    self.msd_array))
        if self.export:
            self._export_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                      self.msd_array))

    def _optimized_calculation(self):
        """
        Run an range optimized calculation
        """
        # Optimize the data_range parameter
        # for item in self.species:
        #     while not self.loop_condition:
        #         tensor_values = self._self_diffusion_coefficients(item, parse=True)
        #         self._optimize_einstein_data_range(tensor_values=tensor_values)
        #
        #     self.loop_condition = False
        #     result = self._fit_einstein_curve(tensor_values)  # get the final fits
        #     self._update_properties_file(item='Singular', sub_item=item, tensor_values=result)
        pass
